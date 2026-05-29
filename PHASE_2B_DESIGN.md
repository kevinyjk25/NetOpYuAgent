# Phase 2B — Capability-based A2A Delegation (设计稿，待审核)

> 目标：让一个 agent 在遇到自己 profile 不擅长的任务时，把子任务委派给
> 拥有对应 capability 的 peer agent，流式拿回结果并 merge 进自己的回答。
> 这是 2A(profile 隔离)+ peer-capability-display 修复之后的自然下一步。

---

## 0. 现状盘点(代码尽职调查结论)

| 组件 | 状态 | 位置 |
|------|------|------|
| peer 发现 + capability 注册 | ✅ 已工作(2A 修复后 peer 显示 cap) | `registry/` |
| capability → peer 选择(resolve_all / _candidates_for_skill / _pick) | ✅ **已实现**，含 round-robin / least-loaded 策略 + task-load 计数 | `registry/registry.py:267-345` |
| A2ATaskDispatcher(流式调远端 agent) | ⚠️ 已写但**从未被调用**，且有 URL bug | `task/inter/coordinator.py:33` |
| MultiRoundCoordinator / ResultAggregator | ⚠️ 已写但未接线 | `task/inter/coordinator.py` |
| `build_task_services()` | ⚠️ 存在但 **main.py 没调用** | `task/__init__.py:73` |
| A2A 入站 server（被委派方执行全 agent loop） | ✅ 已工作(`ITOpsAgentExecutor` + memory-aware) | `a2a/agent_executor.py`, `a2a/server.py` |
| `[DELEGATE:agent_id]` 指令解析 | ❌ **不存在** | 需加到 `runtime/directive_parser.py` |
| 中央指令解析器(扩展点) | ✅ `[TOOL:]`/`[SKILL_LOAD:]`/`[TOOL_BATCH:]` 都走它，audit 强制 | `runtime/directive_parser.py` |
| WebUI delegation dropdown | ⚠️ 发 `delegation_mode`(fresh/forked)但**那是 session 上下文继承轴，不是选 agent**；没有"委派给谁"的 UI | `webui/index.html:742` |
| chunk `source_agent` 标记 | ❌ 不存在 | 需加 |

**两个关键发现**：
1. **选择逻辑已经有了** —— 不用重写 capability matching，Phase 2B 主要是"接线"+ "指令入口" + "结果合并"，不是从零造。
2. **`delegation_mode`(fresh/forked)≠ 委派目标**。前者是子 agent 继承多少父上下文(P1 既有概念)；Phase 2B 要新增的是"委派给哪个 peer"这条正交的轴。别混用。

**两个已知 bug(实现时要修)**：
- `A2ATaskDispatcher._stream_request` 打的是 `agent_url + "/stream"`，但 A2A server 实际暴露的是 `POST {agent_url}/`(JSON-RPC `message/stream`)。URL 约定对不上 —— dispatch 一定 404。
- dispatcher body 用 `message/stream` JSON-RPC 格式是对的，但要对齐 `a2a/server.py:116` 的 `@a2a.post("/")` 入口。

---

## 1. 四个待你拍板的产品决策

实现前必须定这四个，否则会返工。我给出**推荐默认值**，你确认或改：

### Q1. 自动委派 vs 显式指令？
- **(A) 显式** `[DELEGATE:dc-agent] <子任务描述>` —— LLM 在推理中主动发出，像 `[TOOL:]` 一样。
- **(B) 自动** —— 框架检测到本地 profile 无对应 capability 时，自动 resolve peer 并委派。
- **推荐：先 A，后 B**。显式指令可控、可审计、可测；自动委派依赖 LLM 判断"我不会做这个"，容易误触发。A 跑通后再加一层"无本地工具命中时建议委派"的 nudge（半自动），最后才考虑全自动。

### Q2. confirmed_facts 是否跨 agent 共享(隐私边界)？
- **(A) 不共享** —— 被委派方只收到 `[DELEGATE:]` 后面那段明确的子任务描述(等价 `delegation_mode=fresh`)。
- **(B) 共享** —— 把父 agent 的 confirmed_facts 一起传过去(等价 `forked`)。
- **推荐：默认 A(fresh)，可显式 forked**。LAN agent 的设备事实不该无脑灌给 DC agent。复用既有 `DelegationMode` 这条轴来控制，不发明新概念。跨 agent 传 facts 时在 audit 里记明"shared N facts to <peer>"。

### Q3. 多个 peer 都能处理时，怎么选？
- registry 的 `_pick` 已支持 `round_robin` / `least_loaded`(看 config）。
- **推荐：沿用 registry 既有策略，不在 Phase 2B 引入新选择器**。显式 `[DELEGATE:dc-agent]` 时直接定位该 agent_id（跳过策略）；只有 `[DELEGATE:*capability]` 这种按能力委派才走 `_pick`。

### Q4. audit log 范围 —— 入口 agent 是否记录被委派方的执行细节？
- **(A) 只记委派边界** —— 入口 agent 记 `DELEGATED`(给谁、什么任务) + `DELEGATION_RESULT`(成功/失败、摘要)；被委派方在自己进程里记自己的完整 audit。
- **(B) 全量回传** —— 被委派方的每个 audit 事件流回入口 agent 落盘。
- **推荐：A**。每个 agent 进程审计自己的执行(已有)；入口只记委派边界 + 关联 id(`task_id` / `context_id`)。要追全链路时用 `context_id` join 两边的 audit。B 会让入口 agent 的 audit 库膨胀且职责不清。

---

## 2. 设计概览(基于推荐默认值)

### 2.1 数据流
```
用户 → LAN agent
  │
  ├─ LLM 推理，发现需要 DC fabric 能力，输出：
  │     [DELEGATE:dc-agent] 查询 spine-1 的 BGP EVPN 邻居状态
  │
  ├─ runtime/loop 解析 [DELEGATE:] 指令(via directive_parser)
  │     ├─ 校验 dc-agent 在 registry 且 healthy(否则降级为普通文本，告知用户)
  │     ├─ 构造 TaskDefinition(description=子任务, context_id=本 session 派生)
  │     ├─ registry.record_task_start(dc-agent)
  │     ├─ audit DELEGATED {peer, task_id, context_id, shared_facts=0}
  │     │
  │     ├─ A2ATaskDispatcher.dispatch(task, assignment) ──HTTP/SSE──▶ DC agent /api/v1/a2a/
  │     │     (DC agent 跑自己的完整 agent loop：dc_bgp_evpn_status 等)
  │     │     ◀──流式 chunk(token / node_result / 可能的 hitl_interrupt)──
  │     │
  │     ├─ 每个回传 chunk 打上 source_agent="dc-agent" 标记，转发给前端
  │     │     (前端显示 "via dc-agent" 来源徽标)
  │     ├─ ResultAggregator 收集，得到子任务最终结果
  │     ├─ registry.record_task_end(dc-agent)
  │     └─ audit DELEGATION_RESULT {peer, task_id, ok, summary}
  │
  └─ 子任务结果作为 observation 注入 LAN agent 的下一轮推理
        (像 tool result 一样)→ LLM 综合成给用户的最终回答
```

### 2.2 指令格式(新增到 directive_parser)
```
[DELEGATE:<agent_id>] <子任务自然语言描述>
[DELEGATE:*<capability>] <子任务>   # 按能力，走 registry._pick 选 peer
```
- 加 `_DELEGATE_RE` + `find_delegate_directives()` 到 `runtime/directive_parser.py`
- 同步更新 `audit_directive_parsing.py` 允许这个新 regex（并禁止它出现在别处）
- system prompt 增加 `[DELEGATE:]` 用法说明 + 何时该用(本 profile 无此能力时)
- 与 `[TOOL:]` 同级：一轮最多一个 delegate，且 delegate 与 tool 互斥(避免歧义)

### 2.3 fresh / forked 复用
- `[DELEGATE:dc-agent]` 默认 fresh —— 只传子任务描述
- `[DELEGATE:dc-agent#forked]` 才传 confirmed_facts(显式 opt-in，且 audit 记 shared count)

### 2.4 失败 / 降级(必须稳)
- peer 不存在 / 不 healthy / capability 不匹配 → **不报错**，返回一段可用文本注入推理:
  "（无法委派给 dc-agent：未注册/不健康；以下基于本地能力回答）"，让 LLM 继续。
  这条沿用 H2 的教训：委派失败不能让 agent 卡死或 LLM 重试循环。
- 远端流中途断 → ResultAggregator 返回已收到的部分 + 标记 incomplete。
- 远端触发 HITL(dc_config_push 等)→ Phase 2B **不做**跨 agent HITL 透传(那是 Phase 3)。
  当前策略：被委派方的 HITL 在它自己的 WebUI 处理；入口 agent 收到 `hitl_interrupt`
  chunk 时，告知用户"dc-agent 需要审批，请到 dc-agent 控制台处理"，不阻塞。

---

## 3. 实现清单(接线为主，非重写)

1. **修 dispatcher URL bug** —— `_stream_request` 对齐 `POST {agent_url}/` + JSON-RPC，删掉 `/stream` 拼接。
2. **directive_parser** —— 加 `[DELEGATE:]` 解析 + 单元测试 + 更新 `audit_directive_parsing`。
3. **runtime/loop 接线** —— 在 tool-directive 处理同层加 delegate 分支：校验 peer → 构造 TaskDefinition → 调 dispatcher → 聚合 → 注入 observation。
4. **registry 集成** —— 显式 agent_id 走 `get_agent`；`*capability` 走 `resolve_all` + `_pick`；包 `record_task_start/end`。
5. **chunk source_agent 标记** —— dispatcher 回传的每个 chunk 注入 `source_agent` 字段；webui SSE 透传；前端加 "via <agent>" 徽标。
6. **audit** —— 新增 `TaskEventKind.DELEGATED`(已有) + `DELEGATION_RESULT`；入口只记边界。
7. **main.py 接线** —— 调 `build_task_services()`，把 dispatcher/coordinator 注入 runtime。
8. **WebUI** —— delegation dropdown 增加目标 agent 选项(或保持 LLM 自主，dropdown 只控 fresh/forked)。先做最小：dropdown 不动，靠 `[DELEGATE:]` 指令；UI 只加来源徽标。
9. **测试** —— (a) directive 解析；(b) peer 选择(mock registry)；(c) dispatch 成功路径(mock SSE)；(d) 降级路径(peer 不存在/不健康)；(e) 端到端双 agent 集成测试(标记 @integration，sandbox 无 httpx 时 skip)。
10. **文档** —— `task/inter/DESIGN.md` 或 ARCHITECTURE §13 写清委派协议 + 失败语义 + 与 H2/HITL 的边界。

**预估**:6 接线 + 4 新增(directive/标记/audit/测试)。比从零造小很多，因为选择器和 dispatcher 主体已存在。

---

## 4. 明确不做(划清 Phase 2B 边界)
- ❌ 跨 agent HITL 透传(委派任务里的审批回传到入口 agent)→ **Phase 3**
- ❌ 多跳委派(A→B→C)→ 先支持单跳，多跳留后
- ❌ 自动委派(LLM 不主动、框架自动判断)→ 先显式，验证后再说
- ❌ 委派结果的跨 agent 记忆写回(被委派方学到的东西回灌入口 agent 记忆)→ 后续
- ❌ 并行多 peer 委派(一次 fan-out 给多个 agent)→ 先单 peer

---

## 5. 决策确认(2026-05)
Q1–Q4 全部按推荐默认:
- Q1 显式 `[DELEGATE:agent_id]`(暂不做自动委派)
- Q2 默认 fresh(不共享 facts),`#forked` 显式 opt-in
- Q3 沿用 registry 既有策略;显式 agent_id 直达,`*capability` 走 `_pick`
- Q4 入口只记委派边界,各 agent 审自己,`context_id` join 全链路
- 额外:`[DELEGATE:]` 与 `[TOOL:]` 互斥(一轮二选一),避免歧义

---

## 6. 与 HITL 设计债的边界声明

委派不依赖 H2 的 per-process 内存态。`_async_registry`(router.py)、
`_async_inject_queue`(loop.py)、`_session_sse_emit`(backend.py)都是 module-level、
per-process 的内存 dict。委派是跨进程操作(入口 agent 通过 HTTP/SSE 调 peer 的 A2A
端点),状态走 A2A task store(sqlite,持久)+ context_id(跨两 agent 的 audit join key)。

因此债清单 #1(状态全内存)/ #2(多 worker SSE)不会被委派放大 —— 委派本来就走持久
task store + HTTP,不碰内存 dict。跨 agent HITL 透传留给 Phase 3。

Phase 2B 顺手纳入两条债(正好在委派路径上):
- #10 action_type 规范:委派新增 `delegate:<agent_id>` 类型,借此加 ProposedAction
  builder + 轻量类型常量,收拢 tool_call/diagnostic/batch/delegate。
- #7 + #12-3 resumption query:做通用 build_resumption_query(),从 agent_memory 取原始
  user query + 上次 final answer 拼进合成 query,H2 follow-up 与委派结果注入共用。

其余债进 TODO.md。

---

## 7. Phase 3 — 跨 agent HITL（mode B，2026-05 已落地）

> Phase 2B 只做"委派 → peer 自主完成 → 返回结果"(case1)。Phase 3 加入 peer
> 侧需要操作员审批的情形(case2),并把委派固化为**有身份、有生命周期、被持久
> 跟踪的任务**,而非 LLM 每轮自由发起的指令。

### 7.1 业务模型（权威规格）

入口 agent A 收到请求 → 处理自己那部分 → 把剩余分解委派给有能力的 peer B。两类终止:
- **case1(同步)**：B 自主完成 → 返回结果 → 委派完成。
- **case2(HITL)**：B 无法自主完成,需操作员审批。HITL 分两阶段(审批动作 +
  工具执行结果),所以委派要**收到两个阶段结果**才算完成。

对 A 的要求:(1) 可并行委派多个 peer,但有依赖的子任务串行;(2) 结果返回后合并
分析是否满足用户请求,满足则总结,不满足则降级(换任务/换 peer/总结失败);
(3) 委派任务带心跳跟踪,**同一分解任务不得重复委派同一 peer**;(4) case2 中 B 若
被拒则按 case1 完成,若批准则等 stage-2 工具结果才算完成。

### 7.2 架构(四层防风暴 + 两阶段回调 + 前端送达)

委派身份 = `(session_id, target_agent)`,状态存在既有 TaskStore(持久、UI 已读)。

- **单一闸门**(`task/delegation.py delegate_fn`):委派前查 TaskStore,该 peer 有
  非终态 `scope==INTER` 出站任务则抑制。取代早期三个 env_ctx 守卫(per-`execute_query`,
  resume 轮重置 → 失效)。
- **park**(`runtime/loop.py`):case2 peer HITL → 发 `cross_agent_parked` marker +
  中间答复 → `return` 结束 stream,等异步 result 回调。不空转(空转会和回调竞速重委派)。
- **per-request 阻断 + 综合轮硬禁**:`_delegated_targets_this_request`(挡 case1 重委派)
  + `_cross_agent_resume`(综合轮禁止 DELEGATE)。
- **两阶段回调**(`integrations/adapters/hitl_executor.py` + `webui/backend.py`):
  approval 阶段推中间状态、**不**转任务状态(闸门保持);result 阶段才把出站任务
  AWAITING_PEER_HITL → COMPLETED 并驱动一次综合轮。`AwaitingPeerRecord.outbound_task_id`
  贯穿 coordinator → bridge → `/hitl_resolved` → driver。
- **前端送达**(`webui/index.html`):park 后原始 SSE 已关,综合答复 buffer 进
  `_pending_resumptions`,前端 `/chat/resumptions` 轮询取。去重 key 必须含 `phase`
  (approval/result 共用一个 `correlation_id`),且只在终态 `phase==result` 停轮询。

### 7.3 关键教训

1. 委派是有状态任务(身份 + 生命周期),不是 per-turn 指令 —— 用守卫事后拦 LLM 重发是打地鼠。
2. env_ctx 是 per-`execute_query`,跨轮/跨 stream 状态必须放 TaskStore。
3. 单一事实来源:闸门读 UI 同一个 store → 状态不一致从构造上不可能。
4. 终态 = {COMPLETED, FAILED, CANCELLED};其余(含 RUNNING / AWAITING_PEER_HITL)= 在途 = 闸门激活,RUNNING→AWAITING 窗口无缝隙。
5. case2 两阶段:approval 推送**不得**转任务状态(保持闸门),只有 result 终态推送完成它。

详细落地与测试见 `runtime/DESIGN.md §4.10` 与 TODO.md Phase 3 段。仍未做:P3-c(correlation
audit join)、P3-d(passthrough mode C、故障硬化、bridge/buffer 持久化、`/hitl_resolved`
鉴权)。
