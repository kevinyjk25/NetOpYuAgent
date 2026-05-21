# hitl_core — 设计与实现说明书

> **Human-in-the-Loop 核心引擎**。中断 agent 执行,请操作员决策,恢复执行。
> **铁律**:这个模块**不依赖** fastapi / langgraph / langchain。是个纯 asyncio + SQLite/Redis pluggable 的 library。HTTP/SSE adapters 隔在 `hitl_core/transport/` 里,主逻辑用 `asyncio.Future` 跟操作员握手。

---

## 1. 职责

| 文件 | 职责 |
|------|------|
| `schema.py` (372) | 数据契约 — `HitlPayload`, `HitlDecision`, `TriggerKind`, `DecisionKind`, `RiskLevel` |
| `pipeline.py` (719) | `HitlPipeline` — 把 agent 工作流划分成 step,中断点支持 `await ctx.request_approval(...)` |
| `router.py` (627) | `HitlRouter` — 路由 operator decision 到 waiting future 或 detached resumer |
| `batch.py` (351) | `BatchCoordinator` — N-target HITL(一次 approve 多个 device)|
| `triggers.py` (357) | 触发条件:low confidence / destructive / severity / policy violation |
| `coreference.py` (308) | 解析 "approve them all" / "the second one" 这种代词 |
| `chunk_queue.py` (401) | resumer 跑 agent 时实时 push 进度 chunks 给 SSE 订阅 |
| `audit.py` (339) | 不可篡改审计日志(memory/file/redis sink) |
| `store.py` (921) | checkpoint 持久化(in-memory / sqlite / redis 三种实现)|
| `transport/` | HTTP/SSE adapter,把核心 future-based API 转成 web 端点 |

### 1.1 Trust mode 不住在这里(2026-05)

Graduated-trust spectrum(`cautious` / `auto_reversible` / `bypass`)的决策点是 `runtime/loop.py:_needs_hitl`,**在 hitl_core 之前**。判断逻辑住在 `runtime.policy_engine.PolicyEngine.should_skip_hitl_for_tool(tool_name)`(因为 runtime 不能 import hitl_core 反向不能,但 PolicyEngine 是 runtime 自有的)。

`hitl_core` 自己**不知道** trust_mode 存在 — 当 runtime 判定 skip,hitl_core 完全不被调用(零侵入)。当 runtime 判定不 skip,流程跟原来一样进 `HitlPipeline.run` / `HitlRouter.deliver`。这是为了:

- 模块独立保持(`audit_module_independence` 通过)
- hitl_core 测试简单(无新概念)
- 真要 hitl 时,**审计仍然走 hitl_core/audit.py 全量记录**

详见 `runtime/DESIGN.md §4.6` 的 trust_mode 设计 rationale。

---

## 2. 公开接口

### 2.1 主入口

```python
from hitl_core import (
    HitlPipeline, PipelineState, PipelineContext, PipelineAborted,
    HitlRouter, BatchCoordinator,
    HitlPayload, HitlDecision, ProposedAction,
    TriggerKind, DecisionKind, RiskLevel,
    InMemoryCheckpointStore, SqliteCheckpointStore, RedisCheckpointStore,
    AuditLogger, AuditEventKind,
)
```

### 2.2 典型用法

```python
# 1. 启动:store + router + pipeline
store    = SqliteCheckpointStore(db_path="data/hitl.sqlite")
audit    = AuditLogger(sinks=[FileAuditSink("data/audit.log")])
router   = HitlRouter(store=store, audit=audit)
pipeline = HitlPipeline(store=store, router=router, audit=audit)

# 2. 注册命名 resumer(detached decision 回调用)
router.register_resumer("agent_loop_resumer", my_agent_resumer)

# 3. 在 agent 流程里:
async def my_plan_step(ctx: PipelineContext):
    if intent_risk_high(ctx.state.user_query):
        decision = await ctx.request_approval(HitlPayload(
            kind=TriggerKind.DESTRUCTIVE,
            risk_level=RiskLevel.HIGH,
            proposed=ProposedAction(tool="restart_service", args={...}),
            resumer_name="agent_loop_resumer",   # 跨 process 用
        ))
        if decision.decision != DecisionKind.APPROVE:
            raise PipelineAborted("Operator rejected")

pipeline.add_step("plan", my_plan_step)

# 4. 跑 pipeline
async for event in pipeline.run(PipelineState(user_query="...")):
    if event["type"] == "interrupt":
        # 推到前端 (SSE / WebSocket / polling),让操作员看 payload
        decision = await operator_decides(event["payload"])
        await pipeline.resume_with(decision)
    elif event["type"] == "batch_interrupt":
        submission = await operator_decides_batch(event["batch"])
        await router.deliver_batch(submission)
```

### 2.3 Detached decision(operator UI 走 POST)

```python
# operator 通过 web POST /hitl/{interrupt_id}/approve 提交
decision = HitlDecision(interrupt_id=..., decision=DecisionKind.APPROVE, ...)
await router.deliver(decision)
# router 找 in-process waiter 或调用 named resumer 恢复 agent
```

### 2.4 Live progress chunks

```python
from hitl_core.chunk_queue import get_chunk_queue_registry

cq = get_chunk_queue_registry()
cq.push(interrupt_id, {"type": "progress", "tool_calls": 5, ...}, session_id=...)

# 订阅端(SSE):
async for chunk in cq.subscribe(interrupt_id, replay_history=True):
    yield f"data: {json.dumps(chunk)}\n\n"
```

---

## 3. 核心数据流

### 3.1 中断与恢复

```
Agent 跑到敏感点:
   ctx.request_approval(HitlPayload(...))
                  │
                  ▼
   pipeline 创建 interrupt_id
                  │
                  ▼
   audit.log(INTERRUPT_RAISED, payload)
                  │
                  ▼
   store.save(interrupt_id, checkpoint)
                  │
                  ▼
   _DecisionWaiter = asyncio.Future()
   router._waiters[interrupt_id] = waiter
                  │
                  ▼
   yield {"type": "interrupt", "payload": ...}  ← 给 caller / SSE
                  │
                  ▼
   await waiter.future  ← agent 在这里挂起

────────────────────── operator 在 UI 上点 approve ──────────────────────

   POST /hitl/{interrupt_id}/approve
                  │
                  ▼
   HitlRouter.deliver(decision)
                  │
                  ▼ 找 waiter
                  │
                  ├─[A]─ 同 process 有 waiter:
                  │       waiter.future.set_result(decision)
                  │       → pipeline 继续(原 task wake up)
                  │
                  └─[B]─ 无 waiter(进程重启 / 跨 replica):
                          resumer_name = payload.resumer_name
                          fn = router._resumers[resumer_name]
                          asyncio.create_task(fn(ResumeHandle, decision))
                          → resumer 从 checkpoint 恢复,重跑 agent
```

### 3.2 Batch HITL

```
Agent 检测到"对 N 个 device 做同一操作":
   ctx.request_batch_approval(HitlBatch(child_ids=[i1, i2, i3], ...))
                  │
                  ▼
   BatchCoordinator.open_batch(batch) → asyncio.Future
                  │
                  ▼
   N 个 child interrupt 各自持久化(store.save 每个)
                  │
                  ▼
   yield {"type": "batch_interrupt", "batch": batch}  ← UI 渲染聚合卡片
                  │
                  ▼
   await batch_future

────────────────────── operator 在 UI 上 approve 各 child ──────────────────────

   每个 child decision 进 BatchCoordinator.on_child_decision:
        accumulate decisions_by_id[interrupt_id] = decision
        if 所有 child 都 decided → resolve future with BatchResolution

   BatchResolution(decisions=[d1,d2,d3], rejected=False, all_approved=True)
                  │
                  ▼ pipeline 继续(producer 拿到 resolution)
                  │
                  ▼ _batch_execute_after_resolution(resolution):
                        for child in resolution.decisions:
                            按 decision 跑 tool / skip / edit args
                        最后一次 SkillEvolver hook (audit_wiring TODO #1)
```

### 3.2.5 Async HITL (H2 — fire-and-forget,2026-05)

**用途**:tool / skill 知道需要操作员决策(如 RADIUS 权限审批、推外部审批工单),但**不能阻塞 agent** — agent 拿默认假设继续推理,等异步 ack 回来时再 merge-back。

**核心 API**:`PipelineContext.request_approval_async(payload, default_value, on_resolved, divergence_check=None, session_id=None)`

```python
# 在一个 skill / tool 里
interrupt_id, default_value = await ctx.request_approval_async(
    payload          = HitlPayload(..., interrupt_mode=InterruptMode.ASYNC_NONBLOCKING),
    default_value    = "permission_ok",
    on_resolved      = my_callback,    # async (iid, decision, default, diverged) -> None
    divergence_check = None,           # 默认: decision != APPROVE 即 diverged
    session_id       = current_session_id,
)
# 立即返回 — 用 default_value 继续推理
```

**生命周期**:

```
tool 调 request_approval_async(default="permission_ok")
  │
  ▼ pipeline:
  │   ├─ payload.interrupt_mode 强制设 ASYNC_NONBLOCKING
  │   ├─ store.save(checkpoint)        ← /hitl/pending 看到
  │   ├─ _async_registry[id] = pending ← router 查找用
  │   ├─ audit ASYNC_DELEGATED
  │   ├─ asyncio.create_task(_sla_timer)
  │   └─ return (id, "permission_ok")  ← agent 继续
  │
  ▼ agent 用 default_value 继续 N 个 turns 直到 answer 发出
  │
  ▼ 时间 t+Xs(X < sla_seconds):
  │   ├─ operator POST /hitl/{id}/decide  (or demo auto-responder)
  │   │       ↓
  │   ├─ HitlRouter.deliver(decision)
  │   ├─ router._dispatch path 0.5 (ASYNC):
  │   │       1. 查 _async_registry[id] → pending
  │   │       2. divergence = check(default, decision) 或 decision != APPROVE
  │   │       3. await pending.on_resolved(id, decision, default, diverged)
  │   │       4. audit ASYNC_RESOLVED
  │   │       5. _async_registry.pop(id)
  │   │       6. return {async_resolved: True, diverged}
  │
  │   ▼ on_resolved callback(caller 自己写,通常 in tool/skill):
  │       a. enqueue_async_inject(session_id, fact_text)
  │           → 下次 turn-start 时 drain 进 state.confirmed_facts
  │           → LLM 在下个 turn 自动看到这条新 fact
  │       b. emit_async_hitl_notify(session_id, chunk)
  │           → 推 SSE chunk type="async_hitl_resolved"
  │           → 前端 dispatchChunk 渲染 🔔 banner + 2 个按钮
  │
  ▼ OR 时间 t+sla_seconds:
      ├─ _sla_timer 触发
      ├─ pending pop from registry
      ├─ store entry → InterruptState.EXPIRED
      ├─ await on_resolved(id, None, default, diverged=True)
      │       ← decision=None 信号 timeout;caller 写 "timed out"
      │         fact + 通常 emit notify
      └─ audit ASYNC_TIMEOUT
```

**Merge-back 策略**(由 caller 在 on_resolved 中决定):
- **Strategy 1 (Fire-and-forget)** — divergence=False 且无需告知:audit-only,不动 confirmed_facts
- **Strategy 2 (Inject-only)** — 总是写 confirmed_facts,但不通知 SSE — answer 已发就只在下次 turn 用
- **Strategy 3 (Inject + Soft-notify)** — 写 fact + 推 SSE → 当前 chat 立即看到 🔔 卡片,**默认推荐**
- **Strategy 4 (Auto re-think)** — 强制启动新 turn(目前没用,留给未来 risk=CRITICAL 场景)

`query_radius_logs` 示例(profiles/lan/tools.py)走 Strategy 3。

**关键设计决定**:
- **`_async_registry` 是 module-level dict**,不放 HitlRouter 实例 — 因为 async pending 状态必须**outlive** 创建它的 PipelineContext(pipeline 已经退出,operator 2 分钟后才决策)
- **merge-back 走 `agent_memory.state.confirmed_facts`** 而非发明新通路 — 跨 turn 持久 + LLM 自动看到 + token budget 已有 L2 Snip 处理
- **inject 在 turn-start 边界**(`runtime/loop.py:drain_async_inject`)— 避免跟 prompt 拼装 race
- **divergence=False 也写 fact** — audit 完整性比 token 节省更重要(每条 fact 大约 100 chars)
- **timeout 走 same on_resolved 路径,decision=None** — 不分两个 callback,caller 一个判断点

**并发安全(2026-05 修复)**:
- **统一注册入口 `register_async_pending(pending, store=)`** — 所有 async-HITL 生产者(tool / skill / `request_approval_async`)必须走它。它在 `_async_registry_lock` 下插入 registry,**并 arm SLA watchdog**。早期 demo 工具直接 `_async_registry[id]=...` 且不带 timer,导致 autoreply 关闭 + 无人决策时 entry 永久泄漏(Bug 2)。
- **`claim_async_pending(id)` 是唯一所有权裁决点** — operator 决策路径(`router.deliver` → `_dispatch` path 0.5)和 SLA 超时路径都用它原子地"认领"记录。谁拿到谁负责 resolve,另一个拿到 None 直接 no-op。修复了之前 `deliver` 先 await `on_resolved`、**最后**才 pop registry 留下的竞态窗口 —— SLA watchdog 能在那个窗口里拿到同一条记录,导致 `on_resolved` 被触发两次(一次真实决定 + 一次超时),inject 两条 fact(Bug 1)。

**多 agent 安全(进程级隔离,by construction)**:
- 每个 agent 是**独立进程**,所以 `_async_registry` / `_async_inject_queue` / `_session_sse_emit` 这些 module-level 全局**天然 per-agent**,不跨 agent 共享。
- `interrupt_id` 是 uuid4(全局唯一),`session_id` 是 uuid4(全局唯一),HITL checkpoint store 走 per-agent 数据目录(`data/agents/<agent_id>/hitl_checkpoints.db`)。三者叠加 ⇒ 两个 agent 的 async-HITL 状态完全隔离,fact 只会 inject 回触发它的那个 agent 的对应 session。
- 这些全局**纯内存、不落盘**(不存在两个进程读同一份 async 状态的可能);跨 agent 的 HITL 透传是 Phase 3 才做,届时走 A2A `INPUT_REQUIRED` + correlation id,不复用本地 registry。
- **inject queue 有容量上限**(`_MAX_INJECT_SESSIONS=512` / `_MAX_INJECT_PER_SESSION=32`)防止"触发了 async HITL 但再也不发消息"的死 session 无限累积。

`query_radius_logs` 示例(profiles/lan/tools.py)走 Strategy 3,并通过 `register_async_pending` 获得 SLA watchdog。

### 3.3 Coreference 处理

```
operator 在 chat 输入: "approve them all"
                  │
                  ▼
Coreferencer.resolve(query, recent_payloads)
                  │
                  ├─ _strip_hitl_templates(query)
                  │   ↓  剥离 "[OPERATOR DISAMBIGUATION] ..." 等模板词
                  │      避免 "proceed using context" 中的 "proceed" 误触发
                  │
                  ▼ _query_looks_like_coreference(stripped)?
                  │   ↓  代词 / 数字序数 / "all" / "继续" 等
                  │
                  ▼ multi-entity gating:
                  │   - bare continuation (≤6 字符) + recall 含 ≥2 entities
                  │     → return source="ambiguous" 不绑(让用户再说一次)
                  │
                  ▼ free_mention 检索 entity → CoreferenceResult(entity, confidence)
```

### 3.4 Chunk queue 生命周期

```
resumer 启动:
   chunk_queue.push(interrupt_id, chunk, session_id="sess-X")
        │
        ▼ _ensure_sync 创建 _InterruptStream(session_id, last_activity_at)
        │
        ▼ stream.history.append(chunk) + stream.new_chunk.set()

订阅:
   async for c in chunk_queue.subscribe(interrupt_id):
        │
        ▼ 私有 cursor 从 history[0] 开始
        ▼ yield 已有 chunks → await new_chunk.wait() → 拿新的
        ▼ stream.done.set() 时退出

新一轮 chat_stream 同 session 启动:
   chunk_queue.close_session_streams("sess-X")
        ↓
   完成所有 session=X 的 streams(stream.done.set())
   防止旧 resumer 的延迟 chunks 漏到新 turn UI

Idle watchdog(后台 task):
   每 30s sweep,stream.last_activity_at > 120s 没动 →
      push synthetic idle_timeout chunk + complete
   防止 hung resumer 让 UI 永远转圈
```

---

## 4. 关键设计决策

### 4.1 为什么 future-based 而不是 callback-based?

操作员决策**异步且阻塞**(可能 30s 也可能 30 min)。直接 `await future` 让 agent 代码看起来是同步的:

```python
decision = await ctx.request_approval(payload)
if decision.decision != APPROVE:
    raise PipelineAborted
```

callback-based 会让流程拆得到处都是,异常处理 nightmare。

### 4.2 为什么 detached resumer 模式?

进程重启 / 跨 replica / 操作员等了 12 小时再 approve —— 原 future 早就没了。`router.register_resumer("agent_loop_resumer", fn)` 让 router 在找不到 in-process waiter 时去查 payload 的 `resumer_name`,起一个新 task 跑 fn,fn 从 checkpoint 恢复 agent。

这等于"future + named function fallback"双层。

### 4.3 BatchCoordinator 为什么独立?

- N 个 child 各自独立 future 太散,UI 渲染聚合卡需要"batch state"概念
- batch 有 policy(`ALL_OR_NOTHING` vs `BEST_EFFORT`),需要专用决策算法
- batch resolution 触发 finalizer hook(执行 tools + SkillEvolver),不属于 router 职责

### 4.4 Store 三态实现

| 实现 | 用途 | 持久化 |
|------|------|--------|
| `InMemoryCheckpointStore` | 测试 / 单 worker dev | ❌ 重启丢 |
| `SqliteCheckpointStore` | 单 replica 生产 | ✅ 文件 |
| `RedisCheckpointStore` | multi-replica(预留)| ✅ 分布式 |

接口统一为 `BaseCheckpointStore`,pipeline/router 不关心后面是谁。换 backend 改 config 即可。

### 4.5 Coreference 跟 HITL 紧耦合的原因

操作员说"approve them"时,需要知道 "them" 指代哪些 interrupts。这个 context 只在 `hitl_core` 持有(`HitlRouter._waiters` + `BatchCoordinator._collected`)。让 NLU 模块跨包 reach in 会破坏边界。

历史 bug:模板词 "[OPERATOR DISAMBIGUATION] ... proceed using ..." 里的 "proceed" 触发了 coref 误绑 ap-02 → 加 `_strip_hitl_templates`。改 coref 时**一定要看 `_strip_hitl_templates`** 的黑名单。

### 4.6 Chunk queue 为什么 session-scoped close?

历史 bug:operator 在第 2 轮 query 还没完成时发第 3 轮,第 2 轮 resumer 的延迟 chunks 进了第 3 轮 UI,显示"上轮的 thinking" 在新 query 卡片里。修复:
- 每个 stream 关联 `session_id`(resumer push 时传)
- 新 chat_stream 开始时 `close_session_streams(session_id)` 关旧的

### 4.7 Audit 三个 sink

不可篡改日志的多 sink 设计:
- **InMemory**:测试 / 临时
- **File**:行式 JSON,适合 logrotate
- **Redis**:Stream 类型,跨 replica 聚合

`AuditLogger` 同时写多 sink,任一失败不影响其他。**生产建议**:File + Redis 双写,Redis 做实时查询,File 做长期归档。

---

## 5. 跨模块依赖

```
hitl_core
   │
   └── (zero external dep — Python stdlib + asyncio)

外部依赖 hitl_core 的:
   - integrations/adapters/hitl_executor.py  (主消费者,通过 HitlRouter + HitlPipeline)
   - webui/routes_hitl.py                    (HTTP 端点:pending/approve/reject/stream)
   - runtime/loop.py                         (raise HitlInterruptRaised)
   - main.py                                 (装配)
```

### 5.1 扩展点

| 任务 | 改哪里 |
|------|--------|
| 加新 trigger(比如 "时间窗口外不允许")| `triggers.py` 新 class 实现 `Trigger` protocol,`TriggerEngine.register` |
| 加新 decision kind(比如 ESCALATE)| `schema.py:DecisionKind` 加 enum,`router._validate_decision_against_payload` 加规则 |
| 改 batch policy | `schema.py:BatchPolicy` + `batch.py:_check_wait_condition` |
| Multi-replica Redis 同步 | 见 `batch.py:BatchCoordinator` docstring 已写好两条路线 |
| 加 audit sink(Kafka / Splunk) | `audit.py` 实现 `AuditSink`,config build_sink_from_config 加 case |
| 改 coref 黑名单 | `coreference.py:_strip_hitl_templates` |

### 5.2 不该在这里加什么

- ❌ HTTP handler 本身 → `webui/routes_hitl.py`(transport/http_adapter 仅做转接)
- ❌ Tool 执行逻辑 → `runtime/loop.py` 或 `integrations/adapters/hitl_executor.py:_batch_execute_after_resolution`
- ❌ LLM prompt → `integrations/clients/llm_engine.py`
- ❌ 业务规则(什么算 destructive)→ `triggers.py` 是合适的位置,但具体阈值通过 `policy_engine` 外部 config 注入

---

## 6. 修改指南

### 6.1 改之前必须知道

- **`HitlRouter._waiters` vs `_resumers` 的区别**:waiters 是 in-process future,resumers 是命名的 callback。一个 interrupt 同时可能有 waiter(原 task)和 resumer(命名 fallback)。`deliver()` 优先 waiter,fallback resumer。
- **Future 不能 `.set_result()` 两次**。Router 提供 `_safe_set_result` 帮助函数。
- **chunk_queue 的 `_ensure_sync` 是无锁 path**(GIL 保护),`ensure` 是有锁 path。push 路径用 sync 是性能优化,正确性靠"last-write-wins is fine because chunk_log is append-only"。
- **coreference 修改**:跑 `tests/test_production_safety` 不够,需要看实际 query log 中 "继续/approve/proceed/all" 的命中频率。改阈值前先 sample 100 条历史。

### 6.2 改完必须跑

```bash
./scripts/precheck.sh --audits
python -m unittest tests.test_production_safety -v

# Chunk queue 单元测试散在文档里的 inline 测试,改 chunk_queue 时按之前
# C2+D fix 的测试模式手测:close_session_streams / idle sweep
```

### 6.3 调试套路

| 症状 | 看哪里 |
|------|-------|
| operator approve 后 agent 没动 | `router.deliver` log,grep `interrupt_id`,看是 waiter 还是 resumer path |
| Batch member 卡住 | `BatchCoordinator._collected[batch_id]` 缺哪个 child id |
| coref 误绑 | `coreference.resolve` 加 debug log,看 `_query_looks_like_coreference` 是否对 modifier 触发 |
| SSE chunk 重复 | 双订阅?subscribe 不可调两次同 interrupt |
| SSE chunk 缺失 | `chunk_queue` subscriber_count 是不是 1?push 时 stream 存在吗? |
| idle_timeout chunk 误触发 | LLM 真的卡 > 120s 还是 push 没更新 last_activity_at?查 push log |

### 6.4 测试覆盖

- **没有 hitl_core 专属单元测试目录**(历史遗留)。覆盖在:
  - `tests/test_production_safety` — auth + memory 间接路径
  - integration 测试在 `webui/` 里用 fastapi TestClient
- **建议**:加 `hitl_core/tests/` —— pipeline/router/batch 都是 asyncio 纯逻辑,适合细粒度单测。

---

## 7. 已知限制 & TODO

- **Multi-replica 不支持**(`batch.py:BatchCoordinator` docstring 详述)。需要 store-polling 或 Redis pubsub。
- **Coreference 只支持简单代词**。复杂句"the third one but not the first" 抓不到。
- **chunk_queue history 无持久化** —— 进程重启丢失。reconnect 的 subscriber 看不到先前 chunks。生产建议:重要 chunks 同时写 audit log。
- **Coref 测试缺乏 corpus**。当前是几个手写 case + ambiguity gating。生产应该接 anonymized query log 跑日常 regression。
