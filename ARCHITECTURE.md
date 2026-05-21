# Architecture — 模块全景

> 每个模块有独立的 `DESIGN.md`(同目录),解释自身职责、接口、数据流、修改指南。
> 本文档负责**串起来**:模块边界、依赖图、startup 装配顺序、跨模块约定。
> **铁律**:任何**跨多个模块**的改动,先读本文档;任何**单模块内部**的改动,直接读该模块的 `DESIGN.md`。

---

## 1. 模块清单

| 模块 | 行数 | 一句话职责 | 详细文档 |
|------|-----:|----------|----------|
| `runtime/` | ~5500 | Agent 主循环、turn iteration、stop policy、directive 解析 | `runtime/DESIGN.md` |
| `agent_memory/` | ~3000 | 5 维记忆(short/mid/long/skill/user model)+ 召回 | `agent_memory/DESIGN.md` |
| `hitl_core/` | ~4500 | Human-in-the-loop 引擎(中断、决策、批量、审计) | `hitl_core/DESIGN.md` |
| `retrieval/` | ~2200 | BM25 + Embedding + Hybrid 检索框架,Meta tools | `retrieval/DESIGN.md` |
| `skills/` | ~2300 | 复用任务模板、SkillEvolver 自动生成、Journal 反馈;`SkillLoader(mode, profile)` 装配 | `skills/DESIGN.md` |
| `tools/` | ~1200 | **通用框架**工具(分页 read_stored_result 等)+ `ToolLoader(mode, profile)`。业务工具已迁出到 profiles/ | `tools/DESIGN.md` |
| `profiles/` | ~1400 | **业务层**(2026-05 新)— default/lan/dc 三个 profile,各自的 tool/skill/capability。和通用框架解耦 | `profiles/DESIGN.md` |
| `integrations/` | ~4500 | 跨模块胶水(HitlExecutor、LLM/Embedder/MCP 客户端、ToolRouter)| `integrations/DESIGN.md` |
| `schema/` | ~800 | Tool arg schema(`ArgSchema`)+ JSON-Schema/Ollama 导出 | — (轻量,本文档 §6) |
| `evaluation/` | ~1500 | Retrieval bench + Tool-compliance bench + CI gate CLI | — (轻量,本文档 §7) |
| `webui/` | ~5500 | FastAPI + SSE 前端(不含 DESIGN.md,改动局限于路由)| — |
| `memory/` | ~250 | `MemoryAdapter` thin wrapper,给 runtime 用 | — |
| `a2a/` | ~1200 | A2A 协议层(inbound JSON-RPC + SSE 流 + AgentCard + EventQueue) | — (轻量,本文档 §11) |
| `registry/` | ~900 | Agent registry — 自注册、peer discovery、health check、load balance | — (轻量,本文档 §11) |
| `task/` | ~1600 | Task graph + 跨 agent dispatcher(`A2ATaskDispatcher`)+ `delegation.py` 委派工厂(Phase 2B 已接线)| — |

---

## 2. 依赖图

```
                    ┌──────────┐
                    │   main   │  (装配 + lifespan)
                    └────┬─────┘
                         │
       ┌─────────────────┼─────────────────────┐
       │                 │                     │
       ▼                 ▼                     ▼
  ┌─────────┐      ┌──────────┐         ┌──────────────┐
  │  webui  │      │integrations│ ───► │ runtime/loop │
  └─────────┘      └──────┬─────┘       └──────┬───────┘
       │                  │                    │
       │                  │   ┌─── runtime ────┤
       │                  │   │                │
       ▼                  ▼   ▼                ▼
  ┌────────────┐    ┌──────────┐         ┌──────────┐
  │ hitl_core  │◄───┤ adapters │         │  tools/  │
  └────────────┘    │  router  │         │ (mock/   │
       ▲            │  clients │         │  prag)   │
       │            └────┬─────┘         └────┬─────┘
       │                 │                    │
       │                 ▼                    ▼
       │           ┌────────────┐       ┌──────────┐
       └───────────┤ retrieval  │       │  skills  │
                   └─────┬──────┘       └────┬─────┘
                         │                   │
                         ▼                   │
                   ┌──────────────┐          │
                   │ agent_memory │◄─────────┘
                   └──────────────┘
                         ▲
                         │
                   ┌──────────┐
                   │ memory/  │  (thin adapter)
                   └──────────┘
```

### 2.1 依赖规则(由 `scripts/audit_module_independence.py` 强制)

| 模块 | 允许 import |
|------|-----------|
| `agent_memory` | (无 — 零外部依赖) |
| `tools` | `schema` |
| `skills` | `retrieval` |
| `memory` | `agent_memory` |
| `retrieval` | `agent_memory.retrieval`(借 EmbeddingBackend 协议) |
| `hitl_core` | (无业务模块,仅 stdlib) |
| `runtime` | `agent_memory`, `skills`, `hitl_core.schema`, `retrieval`(可选注入) |
| `integrations` | 全部上面 + 外部 SDK(httpx, ollama, etc.) |
| `webui` | 全部 |
| `main` | 全部 |

**违反会被 audit 拦下**。例外通过文件头 `# ALLOWED BY DESIGN` 或 `# DEPRECATED SHIM` 标记,审计会跳过。

---

## 3. Startup 装配顺序

`main.py:build_services` 严格按依赖反向装配,**晚依赖的组件用 deferred wiring**:

```
1. Config           (config.py:Config.load) — including `cfg.agent` (Phase-1 identity)
2. Logging          (logging_config.set_*)
3. agent_memory     (MemoryManager + MemoryAdapter)
4. hitl_core        (Store + Router + Pipeline + Audit + ChunkQueue.idle_watchdog)
5. registry         (AgentRegistry)
   ├─► own_card = get_agent_card(a2a_base_url, identity=cfg.agent)
   ├─► merge peer urls = cfg.registry.agent_urls ∪ cfg.agent.peer_urls (deduped)
   └─► register_from_urls(merged) — peer AgentCards fetched + indexed
6. task             (Task module)
7. integrations.adapters.executor = HitlExecutor(...)  ← skill_evolver=None
8. LLM engine       (OllamaEngine 启动 + smoke test)
9. integrations.clients.embedder
10. tools           (ToolLoader → ToolRouter)
11. MCP + OpenAPI   (discover external tools)
12. PolicyEngine    (from config.yaml + cfg.hitl.trust_mode + tool_metadata)
    ├─► PolicyEngine(..., tool_metadata=loader.build_metadata())  ← action_type 索引
    └─► policy_engine.set_trust_mode(cfg.hitl.trust_mode)         ← graduated trust
13. SkillCatalog    (with retriever pending)
14. SkillEvolver    (after LLM ready)
   ├─► executor.set_skill_evolver(_skill_evolver)     ← deferred wiring point
   └─► main.py wires it post-construction
15. retrieval factories
   ├─► build_skill_retriever_async(catalog)
   ├─► build_tool_retriever_async(metadata)
   └─► engine.attach_retrieval(tool_r, skill_r, meta_reg)
16. meta_tool       (list_tools / list_skills / tool_details)
17. memory feature wiring
   ├─► FactConflictDetector → memory_adapter.set_conflict_detector(fcd)
   └─► memory_adapter.set_consolidator(threshold_turns=30)
18. SkillJournalConsumer (background task)
19. webui mount     (FastAPI lifespan: idle_watchdog start, _start_consumer)
```

**关键观察**:步骤 12、14、17 都是 `set_*` deferred wiring。`audit_wiring.py` 验证它们都连上了。

**Reversibility 改动(2026-05)**:步骤 10 → 12 之间的依赖加紧 — PolicyEngine 现在需要 ToolLoader.build_metadata() 才能 index action_type。已经满足(10 在 12 前),但若未来重排,**必须确保 tool_loader 在 PolicyEngine 构造前就绪**。

---

## 4. 跨模块约定(若改动牵涉多模块必看)

### 4.1 Service registry(`services: dict`)

`main.py:build_services` 返回一个 dict。所有跨模块共享的 component 必须放在这里:

```python
services["executor"]          # HitlExecutor
services["llm_engine"]        # OllamaEngine
services["hitl_core_router"]  # HitlRouter
services["skill_evolver"]     # SkillEvolver
services["tool_router"]       # ToolRouter
...
```

**`audit_wiring.py` 强制**:每个 `services[k] = ...` 必须在**别的 file** 有 `services.get(k)` 或 `services[k]` 读。否则就是 ghost service —— 注册了没人用。

**例外**:`_start_consumer` / `_stop_consumer` / `embedder` / `skill_retriever` / `tool_loader` 仅在 main.py 内跨函数共享(因为 main.py 自身分了多个 build phase)。这些会被 audit warn 但 pass。

### 4.2 Deferred wiring 模式

跨模块组件构造时依赖未就绪 → 加 setter:

```python
class HitlExecutor:
    def __init__(self, ..., skill_evolver=None):
        self._skill_evolver = skill_evolver
    
    def set_skill_evolver(self, evolver):
        self._skill_evolver = evolver
```

main.py 在依赖就绪后 wire:

```python
services["skill_evolver"] = SkillEvolver(...)
services["executor"].set_skill_evolver(services["skill_evolver"])
```

**现有 deferred-wired**:
- `HitlExecutor.set_skill_evolver` (batch finalizer)
- `MemoryAdapter.set_conflict_detector` (fact reconcile)
- `MemoryAdapter.set_consolidator` (auto-rollup)

**何时该用 deferred wiring**:
- A 构造时 B 还没建(循环依赖)
- B 是可选功能(未 wire 时降级)
- A 测试时不想造 B(单测友好)

### 4.3 Directive 协议

LLM ↔ Agent 通过 `[TOOL:name] {args}` / `[TOOL_BATCH:name] [{...},...]` / `[SKILL_LOAD:id]` / `[DELEGATE:target[#mode]] <task>`(Phase 2B,跨 agent 委派,与 TOOL 互斥)通信。**所有**解析在 `runtime/directive_parser.py`,`scripts/audit_directive_parsing.py` 强制单一入口。

加新 directive type:
1. `runtime/directive_parser.py:parse_directives` 加 type
2. `runtime/loop.py:_handle_directive` 加 dispatch case
3. **同时更新 `integrations/clients/llm_engine.py` 的 prompt template**,告诉 LLM 新语法

### 4.4 Chunk 流(SSE 通信)

Resumer / runtime 通过 `chunk_queue.push(interrupt_id, chunk_dict, session_id=...)` 把进度 chunks 推到前端。Chunk 是 `dict`,但前端 `dispatchChunk` 按 `chunk.type` / `chunk.node` 路由。**约定的 chunk shape**:

```
{
  "node": "runtime_loop" | "runtime_tool_result" | "skill_load" | ...,
  "node_step": "Turn N: ..." | "TOOL◀ ..." | ...,
  "tool": "<name>", "args": {...}, "result": "...",   # tool 结果
  "type": "progress" | "stop" | "done" | "awaiting_hitl" | ...,
  "message": "...",
  "tool_calls": N, "turns": N, "elapsed_s": T,        # progress
  "interrupt_id": "...",                              # 所有 chunks 应该带
  "session_id": "..."                                 # 用于 close_session_streams
}
```

加新 chunk type:
1. 后端 push 时**必带 `type`** 字段
2. 前端 `webui/index.html:dispatchChunk` 加 `if (c.type === 'new_kind')` **在 `if (c.message)` catch-all 之前**(否则被 Stop Policy 截胡)
3. 前端约定渲染到 thinking-trace 还是 FLOW 面板还是聊天框

### 4.5 错误传播

| 错误类型 | 处理方 |
|---------|--------|
| Tool 调用 bad args | tool 返回 `[Error: ...]` str,LLM 看到自我修正 |
| Tool 真异常(网络断)| raise,runtime catch,记 chunk,**继续 turn 不 abort** |
| LLM 调用失败 | `OllamaEngine.generate` retry,失败 raise → runtime → HITL_ESCALATE |
| HITL 决策失败 | `HitlRouter.deliver` 返回 audit log,resumer 异常被 `try` 包裹不传出 |
| Memory 写失败 | `MemoryAdapter` 容错,**写入失败不阻塞 turn**,记 warn |

---

## 5. 配置流(`config.yaml` → 代码)

```
config.yaml
   │
   ▼ config.py:Config.load() (env override)
   │
   ▼ cfg.memory.auto_consolidate_turns           → MemoryAdapter.set_consolidator
   ▼ cfg.hitl.enabled                            → HitlPipeline + Router 启用
   ▼ cfg.policies[]                              → PolicyEngine.register
   ▼ cfg.retrieval.tool_backend = "hybrid+cache" → build_tool_retriever
   ▼ cfg.skills.evolver_feedback_enabled         → SkillJournalConsumer
   ▼ cfg.mock or pragmatic                       → ToolLoader(mode=...)
```

**改 config 不需要改代码**,但加新 config 字段:
1. `config.yaml` 加默认值
2. `config.py` 加 dataclass 字段 + env override
3. 消费方读 `cfg.<section>.<field>`,**带 fallback**(`getattr(cfg, 'x', default)`)

---

## 6. CI / 质量门禁

`scripts/precheck.sh`(单一入口,本地 + CI 共用):

```bash
./scripts/precheck.sh              # 全部
./scripts/precheck.sh --audits     # 仅静态 audit(< 30s)
./scripts/precheck.sh --eval       # 仅 retrieval eval
```

**6 个 audit**(任何 fail → PR 不可合):

1. **syntax sweep** — 全 .py AST 解析
2. **`audit_module_independence`** — 模块边界
3. **`audit_imports`** — import 路径有效
4. **`audit_prompt_templates`** — prompt brace 转义
5. **`audit_directive_parsing`** — `[TOOL:` 单一入口
6. **`audit_wiring`** — 无 ghost service

**production safety tests**(21 tests):
- Memory per-operator isolation
- JWT auth round-trip / tamper / expiry
- Log redaction(secrets / API keys / SNMP community)
- Tool allow-list(`run_command` 只允许读)

**retrieval eval gate**:
- BM25 阈值 `recall@3 ≥ 0.40, MRR ≥ 0.30`(CI)
- 本地 hybrid `recall@3 ≥ 0.65, MRR ≥ 0.55`(precheck.sh 默认)

**tool-compliance eval**(`evaluation/compliance_cli.py`):
- 与 retrieval 互补:retrieval 测"工具能不能被查到",compliance 测"工具被叫起来时参数填得对不对"
- 3 个独立指标:`parse_ok`(语法)/ `name_ok`(选对工具)/ `args_ok`(参数对)
- CI 只做**结构验证**(golden set JSONL 能解析),实跑 bench 需要 Ollama,放本地 / 夜间
- 用法:`python -m evaluation.compliance_cli --golden data/tool_compliance_set.jsonl --model qwen2.5:7b --native`
- 数据驱动决策:切模型 / 开 native tools 前后跑一次,对比数字而不是凭感觉

**pre-commit hook**(`.pre-commit-config.yaml`)在本地 commit 时跑 `--audits`,catch 早。

---

## 7. 改动决策树

```
我要改 ...

├─ 单个模块内部行为? → 看 该模块 DESIGN.md §6 "修改指南"
│
├─ 跨多个模块? → 本文档 §4 跨模块约定
│   ├─ 加新 service → §4.1 audit_wiring
│   ├─ 加新 setter 注入 → §4.2 deferred wiring
│   ├─ 加新 LLM directive → §4.3
│   └─ 加新 SSE chunk type → §4.4
│
├─ 加新外部依赖(SDK / DB)? → integrations/DESIGN.md(隔离在 clients/)
│
├─ 改 startup 顺序? → main.py:build_services + 本文档 §3
│
├─ 加新 config? → 本文档 §5
│
└─ 加新 audit / 测试? → 本文档 §6 + scripts/precheck.sh
```

---

## 8. 未完成 / 已知设计债

各模块 DESIGN.md §7 各自记录。汇总:

| 优先级 | 项 | 影响范围 |
|--------|-----|---------|
| HIGH | `pragmatic_tools.py` 多 tool 是 stub | tools |
| HIGH | `loop.py` 2634 行,应拆分(下 Sprint) | runtime |
| MED | Multi-replica Redis pubsub 未实现 | hitl_core |
| MED | Vector ANN(hnswlib)未接 | agent_memory + retrieval |
| MED | SkillEvolver 不学失败 case | skills |
| MED | Hooks 机制(PreToolUse/PostToolUse 等)未做 | hitl_core / runtime |
| MED | Subagent sidechain isolation 未做 | runtime / chunk_queue |
| LOW | LLMJudgeRetriever 未上线 | retrieval |
| LOW | `context_budget_v2` 跟 v1 共存 | runtime |
| LOW | hitl_core 缺单元测试目录 | hitl_core |
| LOW | tools 缺单元测试目录 | tools |

### 8.1 Sprint 1 改造日志(2026-05)

完成的 HIGH 优先级项(从 `ARCHITECTURE_REVIEW.md` Sprint 1):

| 项 | 涉及文件 | DESIGN 章节 |
|----|---------|------------|
| ✅ Reversibility 三档分类(`action_type` 元数据) | 33 个 tool registry + `policy_engine.classify_action_type` | tools/§2.4.1 + runtime/§2.3 |
| ✅ Trust mode 配置(`cautious` / `auto_reversible` / `bypass`) | `config.{py,yaml}` + `policy_engine.set_trust_mode` + `runtime/loop.py:_needs_hitl` skip | runtime/§2.3 + hitl_core/§1.1 + integrations/§2.1 |
| ✅ classify_query_intent fast-path(routine query 跳 LLM)| `policy_engine.classify_query_intent` + `runtime/loop.classify_async` | runtime/§2.3, §3.6 |
| ✅ L2 Snip 跨 turn tool_outputs 压缩 | `context_budget.try_snip_tool_outputs` + `loop.py` 两处 turn-start hook | runtime/§3.5, §4.7 |

未做但准备好的(留下个 Sprint):
- ⏳ `loop.py` 拆分(2634 行 → 5 个 file)— 设计已定,机械重构

各项零回归:default trust_mode=cautious 完全保留原行为。`snip_tool_outputs_char_budget=0` 完全关闭 Snip。`action_type` 未声明的 tool 继续走 LLM classify_destructive。**5/5 audits + 21/21 safety tests + retrieval eval gate 全 PASS**。

### 8.2 Sprint 2 改造日志(2026-05)

完成的 MED 优先级项(从 `ARCHITECTURE_REVIEW.md` Sprint 2):

| 项 | 涉及文件 | DESIGN 章节 |
|----|---------|------------|
| ✅ Hooks 机制(6 个核心事件)| `runtime/hooks.py`(新)+ `runtime/loop.py` 5 处 fire 点 + `runtime/__init__.py` export | runtime/§2.6, §3.7, §4.8, §4.9 |
| ✅ Hermes-style structured consolidation | `agent_memory/consolidation.py` template 切换 + 全链 wire(MemoryConsolidator → MemoryManager → MemoryAdapter → main.py)+ `config.{py,yaml}` | agent_memory/§3.4.1 |

未做(评估后**主动跳过**,非 deferred):
- ⏭ `loop.py` 拆分(2634 行 → 5 个 file)— **Sprint 1 + Sprint 2 都评估过两次**,结论:纯机械 refactor 在 chat session 内做风险大于收益(没有 e2e 测试覆盖 1300 行 `stream()`)。留给有 IDE + bisect 工具的环境单独做。

**Hooks 设计要点**:
- 6 个事件:`PRE_TOOL_USE / POST_TOOL_USE / TURN_START / TURN_END(预留)/ SESSION_START / SESSION_END`
- "Observer 而非 gatekeeper" 哲学(`runtime/DESIGN.md §4.8`):异常被 log + swallowed,唯一 block 路径是 `PRE_TOOL_USE` 的 `ctx["blocked"]=True` 显式 flag
- `stream()` 包 try/finally wrapper(`runtime/DESIGN.md §4.9`),保证 `SESSION_END` 在 abort 路径也 fire
- 零 LLM context cost,可对比 Claude Code 4-mechanism extensibility(Hooks 是最便宜的一层)

**Structured rollup 设计要点**:
- 5 节固定格式:`Goal / Progress / Decisions / Devices / NextSteps`
- 跟 Hermes `ContextCompressor` 哲学一致(单层 + 结构化模板,而非 Claude Code 5 层 cascade)
- `MEMORY_CONSOLIDATION_TEMPLATE=legacy` env 一键回滚
- 全链 deferred-wiring:cfg → MemoryAdapter → MemoryManager → MemoryConsolidator

**单元测试**:8/8 hooks(priority / block / exception isolation / unregister)+ 5/5 consolidation(default / explicit / legacy / invalid / no-llm fallback)。

**全 CI PASS**:5/5 audits + 21/21 safety tests + retrieval eval gate。

### 8.3 Phase 1 — 多 agent 身份 + peer discovery(2026-05)

**目标**:让多个 agent 进程互相发现,但**不互相调用**(那是 Phase 2)。这一步是 multi-agent 的地基。

完成的项:

| 项 | 涉及文件 | 备注 |
|----|---------|------|
| ✅ `AgentIdentityConfig` + `AgentSkillSpec` dataclass | `config.py` | YAML loader + 5 个 env override(AGENT_ID/AGENT_DISPLAY_NAME/AGENT_DESCRIPTION/AGENT_PEERS/AGENT_PEER_REFRESH_S) |
| ✅ `config.yaml` 新 `agent:` 段 | `config.yaml` | 含完整示例 + LAN/WAN agent 启动命令注释 |
| ✅ `get_agent_card(base_url, identity=None)` 可选 identity 驱动 | `a2a/agent_card.py` | 向后兼容:identity=None 走 legacy 路径,新加 `agent_id` 顶级字段始终存在 |
| ✅ `main.py` 在装配 registry 时合并 peer urls | `main.py` build_services | `cfg.registry.agent_urls ∪ cfg.agent.peer_urls`,去重保序 |
| ✅ Lifespan 加 peer refresh background task | `main.py` lifespan | 每 `peer_refresh_interval_s` 秒 re-fetch peer cards,失败 log 不中断 |
| ✅ `/system/peers` endpoint | `webui/routes_system.py` | 返回 `{self: {...}, peers: [...]}`,自动排除自己 |
| ✅ 16 个单元测试 | `tests/test_multi_agent_identity.py` | 默认值 / env override / 畸形 yaml 容错 / agent card 双路径 / peer URL 合并 |

**设计要点**:

- **配置优先级**:env → yaml → defaults。defaults 完全还原 legacy 单 agent 行为,**已有部署不改 yaml 也能继续跑**。
- **`agent_id` 字段**:既是 registry key,也是 trace tag,也是 `/system/peers` 用来排除自己的依据。Legacy path 也填上(=`"default-agent"`),所以 peer 视角的 schema 始终一致。
- **`capabilities=[]` 不会清空 skills**:回退到 legacy SKILLS。设计判断:用户改 agent_id 但忘填 capabilities 时,不应该静默失去 retrieval 能力。
- **Peer URL 双源合并**:`cfg.registry.agent_urls`(legacy 单列表)和 `cfg.agent.peer_urls`(Phase 1 per-agent)都能加 peer,在 `main.py` 去重合并,registry 只看到一份。

**Phase 1 不做的事**(留给 Phase 2):

- ❌ 任何形式的 cross-agent dispatch — `A2ATaskDispatcher` 已经在,但**没有任何运行时代码 call 它**
- ❌ PolicyEngine 不知道 peer 存在 — 仍然只分类 read_only/destructive
- ❌ Capability matching(query → 哪个 peer 处理)— 留给 Phase 2 的 `peer_router.py`
- ❌ 跨 agent HITL — Phase 3

**两端启动验证**:

```bash
# Terminal 1 — LAN agent on :8000, sees WAN
AGENT_ID=lan-agent AGENT_DISPLAY_NAME="LAN Agent" \
  AGENT_PEERS="http://localhost:8001/api/v1/a2a" \
  uvicorn main:app --port 8000

# Terminal 2 — WAN agent on :8001, sees LAN
AGENT_ID=wan-agent AGENT_DISPLAY_NAME="WAN Agent" \
  AGENT_PEERS="http://localhost:8000/api/v1/a2a" \
  uvicorn main:app --port 8001

# Both should see each other:
curl http://localhost:8000/webui/system/peers | jq
curl http://localhost:8001/webui/system/peers | jq
```

**全 CI PASS**:6/6 audits + 21/21 safety tests + 16/16 multi-agent identity tests + retrieval eval gate。

**2026-05 续:Phase 1 bug fixes + WebUI**:

| 问题 | 修复 | 文件 |
|------|------|------|
| `/system/peers` 显示 peer 是一串 UUID,不是真实 agent_id | `AgentDiscovery._parse()` 之前不读取 AgentCard JSON 里的顶层 `agent_id` 字段,所以 `AgentEntry.agent_id = Field(default_factory=uuid.uuid4)` 每次 fetch 给新 UUID。修复:`_parse` 显式 `raw.get("agent_id", ...).strip()`,有就用,空白/缺失 fallback 到 UUID(保留对非 Phase-1 peer 的向后兼容)| `registry/discovery.py` |
| **(2026-05 续)peer agent_id 仍显示 `default-agent`** | 上一个修复让 discovery 正确读 card 的 agent_id,但**源头 card 本身就是错的**:`a2a/server.py:create_a2a_app()` 调 `get_agent_card(base_url)` 没传 identity → 发布出去的 `/.well-known/agent-card.json` 永远是 `default-agent`。`/system/peers` 的 self 块对(直接读 cfg.agent),但 peer 互相 fetch 到的 card 错。修复:`create_a2a_app` 加 `identity` 参数,`main.py` 传 `cfg.agent`,加 2 个 source-level regression test | `a2a/server.py` + `main.py` |
| **(2026-05 续)peer refresh loop 崩溃 + 启动竞态导致 `peers: []`** | 两个叠加问题:(1) refresh loop 传 `source=None`,但 `RegistrationSource` 是必填 enum,`AgentEntry` 构造抛 ValidationError,被 `fetch_many` 每-URL 吞掉 → 每 120s 崩一次;(2) 两个 agent 同时启动时,先起的那个 fetch 后起的会失败(对方还没监听)→ 初始注册 0 个 peer → 本该靠 refresh loop 补,但 refresh 被问题(1)弄崩了 → peer 永远不出现。修复:refresh loop 改用 `RegistrationSource.STATIC`;加 fast-bootstrap 阶段(前 30s 每 5s 重试一次,全部发现后提前退出);`create_registry` 加注册数量 vs URL 数量的 mismatch 日志 | `main.py` + `registry/__init__.py` |
| **(2026-05 续)Ctrl+C 退不出来** | Sprint-3-pre 加的 `loop.add_signal_handler(SIGINT, ...)` **抢占了 uvicorn 的 SIGINT handler**,而我的 handler 只 set 一个没人 await 的 event → uvicorn 收不到信号 → lifespan 不退出 → 卡死。修复:**完全移除** 自己的 signal handler,靠 uvicorn 自己的 handler 触发 lifespan shutdown(`yield` 之后的 drain 块)。drain timeout 改为可配 `SHUTDOWN_DRAIN_TIMEOUT_S`(默认 10s,无 in-flight 任务时直接跳过)| `main.py` |
| WebUI 没法看到 peer 邻居 | 加 "Peers" tab(第 7 个),每 15s polling `/system/peers`,显示 self + peers 各自的 health dot / capabilities / URL,带 heartbeat indicator | `webui/index.html` |
| `switchTab` 之前漏了 journal 和 peers | 列表里补齐 `'journal'` + `'peers'`,并加 null-check 防御 | `webui/index.html` |
| 3 个新 discovery 单元测试 + 2 个 server-identity regression 测试 | agent_id 从 card 读取 / 缺失时 UUID fallback / 空白也 fallback;create_a2a_app 接收并传递 identity | `tests/test_multi_agent_identity.py` |

### 8.4 Sprint-3-pre — 生产就绪基础工程(2026-05)

**目标**:把上次生产就绪评估中标为 blocker 的 4 项工程基础设施补上,**不影响 Phase 1 multi-agent**。这一节是面向 SRE / 运维的:多 agent 可以投生产之前,单 agent 必须先达标。

完成的项:

| 项 | 文件 | 解决了什么生产风险 |
|----|------|------------------|
| ✅ OpenTelemetry 可选 tracing 骨架 | `runtime/tracing.py` 新加(242 行)| 此前 prod 排障只能 grep 日志,无法跨进程对齐一次 query 的完整执行链路 |
| ✅ 三处 span 接入 | `llm_engine.py` / `loop.py` / `hitl_executor.py` | `agent.query` / `llm.call` / `tool.dispatch` 三层 span,session_id 作为 trace 锚点 |
| ✅ HITL checkpoint 默认 sqlite | `config.py` + `main.py` | 此前默认 in-memory,SIGTERM 中 pending approval 全部丢失;sqlite 是默认值后,operator 看着卡片时 agent 重启不丢决策 |
| ✅ Pragmatic 模式 in-memory 警告 | `main.py` | 即使有人强行 `HITL_CHECKPOINT_BACKEND=memory`,启动日志会大字提示风险 |
| ✅ Graceful shutdown drain | `main.py` lifespan | SIGTERM 时:1)等 in-flight LLM/tool 调用 30s 内完成,2)flush HITL checkpoint store,3)再退出 |
| ✅ SIGTERM / SIGINT handler + Windows fallback | `main.py` lifespan | 显式 signal handler 而不是依赖 uvicorn 的默认行为;Windows 没有 add_signal_handler 时 fallback 到 lifespan-exit 路径 |
| ✅ SkillEvolver A/B safety net | `skills/evolver.py` + `main.py` wire | LLM 改写 skill prompt 时,先跑 compliance bench 子集对比 baseline/candidate;`args_ok` 下降则**整个 patch 回滚**,旧 skill 保留 |
| ✅ 12 + 1 新 unit tests | `tests/test_sprint3_pre.py` | 4 tracing 行为 + 3 ObservabilityConfig + 2 HITLCheckpointConfig + 4 SkillEvolver 安全网(含正/反两种情况)|

**设计要点**:

- **Tracing 默认 OFF**:`opentelemetry-*` 包是可选依赖。boot 时如果没装,`configure()` 返回 False,所有 `with start_span(...)` 调用全部走 `_NoopSpan` 路径,**零性能影响**。包装函数模式(`_chat → _chat_impl` / `execute_query → _execute_query_inner`)保留了原有异常路径。
- **HITL backend 选择从环境变量改为 `cfg.hitl.checkpoint`**:env var 仍生效(`HITL_CHECKPOINT_BACKEND` 优先级最高),但默认值现在是 yaml/dataclass-driven,可以在 PR 里看到 schema 变化。
- **Drain 顺序**:in-flight 任务先(30s 上限)→ HITL store flush → SkillJournal stop → registry stop → watchdog stop。device-state-affecting 操作的结果有最长 30s 落库窗口,绝不在拥有未持久化 fact 的时候 hard-kill。
- **A/B safety net 的失败模式**:bench runner 抛错 → 视为"无信号",patch 照常应用(避免 flaky bench 困死正常进化);bench 返回 None → 同样视为"无信号"。**只有** bench 返回了有效 `args_rate` 且明确下降时才回滚。
- **新增 service key `hitl_store`** 现在被 `audit_wiring` 识别(read by lifespan flush)。

**Sprint-3-pre 不做的事**(留给真正的 Sprint 3):

- ❌ FastAPI / httpx auto-instrumentation(spans 当前只在 3 个手工标注的地方;runtime/* 其他 hot path 没埋点)
- ❌ session_id → trace_id 确定性派生(操作员现在没办法从 WebUI 点开 Jaeger / Tempo 直接看 trace)
- ❌ `/livez` + `/readyz`(`/health` 是 200-or-bust,k8s probe 拿不到真实依赖状态)
- ❌ Prometheus `/metrics`(`/integrations/metrics` 返回 JSON,不是 OpenMetrics 文本)
- ❌ Docker / compose / k8s 部署交付物
- ❌ Auth 启动强制检查(`auth.enabled=false` + `ENVIRONMENT=production` 仍然能 boot)

**全 CI PASS**:6/6 audits + 21/21 safety tests + 16/16 multi-agent identity tests + 13/13 sprint3_pre tests + retrieval eval gate。

**Bonus bug fix(2026-05 续):`SkillEvolver._parse_json_response`**

合并 A/B safety net 时 grep 出来一个**预先存在的、自项目早期就有的 bug**:`SkillEvolver.apply_feedback()` 和 `evaluate_skill_creation()` 都调用 `self._parse_json_response(raw)`,但这个方法**根本没在类里定义**(只有 `_parse_markdown_to_definition`)。两条路径都默默命中 `except Exception` 然后 `return None`。

**影响**:
- "Hermes Skill 自改进"路径从来没真正生效过 —— 每次 operator 反馈 LLM 改写 prompt 都被静默丢弃
- 自动 skill 创建评估(`evaluate_skill_creation`)也一样,LLM 决策结果从来没被采纳

**修复**:加 `_parse_json_response` 实现,支持 4 种常见 LLM 输出格式:
1. 严格 JSON 直接 `json.loads`
2. 剥 markdown code fence(```json … ``` / ``` … ```)
3. 剥 `<think>...</think>` 块(qwen3 / deepseek-r1 风格)
4. balanced-brace 扫描 提取嵌入在散文里的 first complete object

加 9 个单元测试覆盖每种格式 + 嵌套大括号 + 字符串里 } 的情况。

这个 bug 跟 Sprint-3-pre 没关系,但因为合并时发现 + 修了,记录在这一节末尾。**这个修复实际上"打开"了项目原本设计但从未真正跑过的两条路径**,以后 production 跑起来要观察这两条路径是否带来预期之外的副作用(比如 skill 被 LLM 改得乱七八糟)。

### 8.5 Sprint 3 — C1/C2/D1 observability + 并发保护(2026-05)

从生产就绪清单里选了 3 项做(A/B 暂缓,短期不上生产)。完整待办见根目录 `TODO.md`。

| 项 | 文件 | 解决什么 |
|----|------|---------|
| ✅ C1 — Prometheus `/metrics` | `runtime/metrics.py`(新) + `main.py` `/metrics` 路由 | 此前 `/integrations/metrics` 返回 JSON,Prometheus/Grafana 不识别。新 endpoint 输出标准 OpenMetrics 文本。指标:`netopyu_llm_calls_total{model,outcome}` / `netopyu_llm_call_duration_seconds{model}` histogram / `netopyu_tool_calls_total{tool,outcome}` / `netopyu_hitl_pending` gauge / `netopyu_active_llm_calls` gauge |
| ✅ C2 — FastAPI/httpx auto-instrumentation | `runtime/tracing.py` `instrument_fastapi()` + httpx instrument in `configure()` | Sprint-3-pre 只埋了 3 处手工 span。现在每个 inbound 请求 + 每个 outbound HTTP(peer fetch / OpenAPI / MCP-over-HTTP)自动 span,挂在现有 trace 上 |
| ✅ D1 — LLM 并发信号量 | `config.py` `max_concurrent_calls` + `llm_engine.py` semaphore | 一个 query fan-out 20+ 个内部 LLM 调用,几个并发 query 就打爆 Ollama。`asyncio.Semaphore`(默认 4)限制 in-flight,延迟优雅退化而非雪崩 |

**设计要点**:

- **三项全部可选 + 优雅降级**:`prometheus_client` / `opentelemetry-instrumentation-*` 没装,boot 照常,`/metrics` 返回纯文本提示,instrument 函数返回 False。`requirements.txt` 里标了 OPTIONAL。
- **`runtime/metrics.py` 跟 `tracing.py` 一个套路**:模块级懒加载 collector,record helper 在包缺失时 no-op,zero import cost。模块独立(不 import 任何内部模块)。
- **C1 指标埋点位置**:LLM 调用在 `llm_engine._chat` wrapper(同时含 D1 semaphore + tracing span + metrics,三合一);tool dispatch 在 `loop._dispatch_tool`;HITL pending 在 `/metrics` 被 scrape 时实时读 router。
- **D1 semaphore 懒绑定**:`asyncio.Semaphore` 必须绑定到运行中的 event loop,而 `__init__` 在 uvicorn loop 起来之前就跑了。所以 semaphore 在第一次 `_get_semaphore()` 时才创建,并在 loop 变化时(测试场景)重建。`set_max_concurrent_calls(0)` = 不限制(legacy)。
- **指标 + tracing + semaphore 在同一个 `_chat` wrapper 里叠加**:`async with _sem: with track_active_llm(), time_llm_call(model): await _chat_impl(...)`。三个关切点(并发/指标/追踪)都从业务函数 `_chat_impl` 里抽出来了。

**8 个新单元测试**(`tests/test_sprint3_pre.py`,共 30):metrics helper no-op 安全 / time_llm_call 异常仍记录 / render 返回 bytes / instrument_fastapi disabled 时 no-op / semaphore 默认禁用 / semaphore 真实限并发 / config 有 max_concurrent_calls 字段。

**仍未做(留 `TODO.md`)**:A1(auth 强制)/ A2(CSRF)/ A3(secrets)/ B1(Docker)/ B2(livez+readyz)/ B3(DEPLOYMENT.md)/ C3(backup)/ C4(migrations)。

**全 CI PASS**:6/6 audits + 21/21 safety + 23 multi-agent(4 skip)+ 30 sprint3 tests。

### 8.6 Async HITL (H2 — fire-and-forget,2026-05)

**起因**:用户提出"xxx 用户上不了网"诊断闭环可能触发 3 种性质完全不同的 HITL — 同步追问(查终端)/ 异步推送(查 RADIUS 权限)/ 同步高危(下发配置)。H1/H3 当前已支持,但 H2 完全缺位 — `request_approval()` 永远 `await future` 阻塞整个 turn,跟"agent 拿默认值继续推理"的语义对立。

**解决方案 B**:加 `request_approval_async()` API,merge-back 借用 `agent_memory.state.confirmed_facts` 已有的跨 turn 持久化通道。MFA 推迟单独 sprint。

5 个 Step 实施(2026-05-21 与 Phase 2A profile 重构合并后的最终落点):

| Step | 改动 | 文件 |
|------|------|------|
| 1. Schema | `TriggerKind.EXTERNAL_DELEGATION` / `InterruptMode {SYNC_BLOCKING, ASYNC_NONBLOCKING, MFA_BLOCKING}` / `InterruptState.ACKED/WORKING` / `AuditEventKind.ASYNC_DELEGATED/RESOLVED/TIMEOUT` / `HitlPayload.interrupt_mode` 字段 | `hitl_core/schema.py` |
| 2. Pipeline API | `AsyncPendingHitl` dataclass + `request_approval_async(payload, default_value, on_resolved, divergence_check, session_id)` 返 (id, default) 不 await + SLA timeout task | `hitl_core/pipeline.py` |
| 3. Router 路由 | `_async_registry` module dict + `_dispatch` 加 path 0.5 (ASYNC):查 registry → diverged check → on_resolved cb → audit ASYNC_RESOLVED → pop | `hitl_core/router.py` |
| 4. Runtime 整合 | `HookEvent.ASYNC_HITL_RESOLVED` + `enqueue_async_inject()` / `drain_async_inject()` per-session queue + `stream()` turn-start drain → `state.confirmed_facts` + `webui/backend.py` SSE bridge (`register_session_sse_emit` / `emit_async_hitl_notify`) | `runtime/loop.py`, `runtime/hooks.py`, `webui/backend.py` |
| 5. Demo + 前端 | `query_radius_logs` 工具走完整 H2 路径(在 **`profiles/lan/`** — RADIUS 是 LAN 业务,而不是 framework leaf)+ demo 自动 responder(3-12s 随机 ack)+ 前端 `dispatchChunk` 加 `async_hitl_resolved` case 显 🔔 banner + [让 agent 重新分析] / [我已处理,忽略] 两个按钮 | `profiles/lan/tools.py`, `profiles/lan/tool_meta.py`, `profiles/lan/__init__.py`, `webui/index.html` |
| 6. Follow-up turn | operator approve 后 `_submit_hitl_decision` 检测 `async_resolved` → 跑 `loop.run()` 合成 query → turn-start drain 拿 fact → LLM 给最终答案 → HTTP response 带 `async_followup` → 前端追加渲染 agent message | `webui/backend.py`, `webui/index.html` |

**关键设计决定**(详见 `hitl_core/DESIGN.md §3.2.5`):

- `_async_registry` 是 module-level dict — async pending 必须 outlive 创建它的 PipelineContext
- merge-back 走 `state.confirmed_facts`(跨 turn 持久 + LLM 自动看到 + 已有 L2 Snip 处理 token budget)而非发明新通路
- inject 在 turn-start 边界(`drain_async_inject`)— 避免跟 prompt 拼装 race
- divergence=False 也写 fact — audit 完整性 > token 节省
- timeout 走 same on_resolved 路径,decision=None — 不分两个 callback,caller 一个判断点
- approve 触发 follow-up turn — operator 不用追问,agent 自动用新 fact 给最终答案

**Tool → hitl_core 依赖**:`query_radius_logs` 是 demo 性质的"H2 直接消费者",必须 import `hitl_core`。在 Phase 2A profile 重构后,这条依赖**反而合规** — 工具搬到 `profiles/lan/`,profiles 不在 `audit_module_independence.FUNCTIONAL_MODULES` 中,允许引用 framework(arrow framework→profile 反方向 profile→framework 同样合理)。**生产**用法仍建议通过 PolicyEngine + runtime hook 间接驱动,而非工具直接调 hitl_core。

**全 CI PASS**:7/7 audits(含 profile audit)+ 101 tests(96 baseline + 5 H2 regression,1 skip on pydantic)。

**没做**:MFA(产品决策推迟);H2 真生产路径(operator 真的去 ops 队列审批,要前端 + 队列后端集成,这次只做 demo 自动回复)。

---

### 8.7 Phase 2B — Capability-based delegation(2026-05)

**目标**:agent 遇到自己 profile 不擅长的子任务时,委派给拥有对应能力的 peer agent,流式拿回结果再 merge 进自己的回答。建立在 2A(profile 隔离)+ peer-capability 显示修复之上。

**关键现状发现**(实现前的代码尽职调查):capability→peer 选择逻辑(`registry.resolve` / `_candidates_for_skill` / `_pick`,含 round-robin / least-loaded + task-load 计数)、`A2ATaskDispatcher`(流式调远端)、`create_task_system` 都已存在 —— Phase 2B 主要是**接线**(指令入口 + 注入 runtime + 结果合并),不是从零造。`A2ATaskDispatcher` 的 `/stream` URL 约定经核对是**正确的**(server.py 确有 `POST /stream` 端点)。

**产品决策**(全部按推荐默认):显式 `[DELEGATE:agent_id]`(不做自动委派);默认 fresh 不共享 facts,`#forked` 显式 opt-in;显式 agent_id 直查 / `*capability` 走 `_pick` 排除自己;入口只记委派边界,各 agent 审自己,`context_id` join;`[DELEGATE:]` 与 `[TOOL:]` **互斥**(一轮二选一)。

| 改动 | 文件 |
|------|------|
| `[DELEGATE:target[#mode]] <task>` 解析(显式 id / `*capability` / `#forked`)+ strip/has helper | `runtime/directive_parser.py` |
| audit 把 `[DELEGATE:` 纳入强制(parser 独占该 regex) | `scripts/audit_directive_parsing.py` |
| `AgentRuntimeLoop` 加可选 `delegate_fn` 注入口 + `_handle_delegate()`(fresh/forked、source_agent 标记、peer-HITL 检测降级、`_inject_context` 结果注入)+ DELEGATE/TOOL 互斥分支 | `runtime/loop.py` |
| `build_delegate_fn` 工厂:registry 解析 peer → `TaskDefinition` → dispatcher 流式 → task-load 计数 + 边界 audit;graceful degrade(peer 未知/不健康/无能力匹配 → 注入"本地继续"提示,不抛错) | `task/delegation.py`(新) |
| main.py 构建 delegate_fn 存 services;backend.py 注入 runtime loop | `main.py`, `webui/backend.py` |
| system prompt 加 `[DELEGATE:]` 用法 + 互斥规则 | `integrations/clients/llm_engine.py` |
| WebUI 🤝 "via <agent>" 委派徽标(`node==='delegate'`)| `webui/index.html` |
| coordinator httpx 改 lazy import(只在 dispatch 时需要,sandbox 可加载) | `task/inter/coordinator.py` |

**顺手解决的 HITL 设计债**:
- #10 — `ProposedAction` builder(`tool_call`/`batch`/`diagnostic`/`delegate`)+ `ActionTypePrefix` 常量,收拢散落的 `"tool_call:"+name` 拼接
- #7 + #12-3 — 通用 `build_resumption_query(new_observation, original_query, previous_answer, divergence_note)`,从 message history 取原始 query + 上次答案;H2 follow-up 已接入(不再丢失原始上下文)

**模块独立性**:runtime loop 通过**注入的** `delegate_fn` 调委派,不 import `task/` 或 `registry/` —— `audit_module_independence` 仍 PASS。

**与 H2 / HITL 边界**(详见 PHASE_2B_DESIGN §6):委派是**跨进程**(HTTP/SSE 调 peer A2A 端点),状态走持久 A2A task store + `context_id`,**不依赖** H2 的 per-process 内存态(`_async_registry` 等)。因此 HITL 债 #1/#2(状态全内存 / 多 worker SSE)不被委派放大。

**Phase 2B 不做**(留后续):跨 agent HITL 透传(Phase 3);多跳委派;自动委派;并行 fan-out;委派结果跨 agent 记忆写回。peer 任务里若触发 HITL(如 `dc_config_push`),入口 agent 收到 `hitl_interrupt` chunk 时提示用户到 peer 控制台处理,不阻塞。

**测试**:`test_delegate_directive.py`(12,解析)+ `test_delegation_wiring.py`(11:explicit/capability resolution、fresh/forked facts、4 类降级、task-load 括号、loop-side source_agent 标记 + 注入 + peer-HITL 提示)。

**CI**:7/7 audits + 128 tests(117 baseline + 11 delegation)。

---

## 9. 模块独立维护原则

每个模块的 `DESIGN.md` 是**单文件 onboarding**:新人改一个模块只需读对应一份。

修改边界:
- **改一个模块** → 只读该模块 DESIGN.md + 跑该模块测试
- **改两个模块**(比如 runtime + hitl_core)→ 读两份 + 读本文档 §4
- **改三个或更多** → 设计可能有问题,先讨论拆分

每份 DESIGN.md 必含 6 节:**职责 / 公开接口 / 数据流 / 设计决策 / 跨模块依赖 / 修改指南**。
新加模块照此模板,**作为 PR 必须项**。

---

## 10. 轻量工具模块(无 DESIGN.md)

部分模块代码量小、职责单一、对外接口稳定,不值得维护独立 DESIGN.md。这一节集中说明,以免 ARCHITECTURE.md 之外缺乏文档锚点。

### 10.1 `schema/` — Tool 参数 schema

**职责**:用 dataclass(`ArgSchema` / `ArgField`)统一表达 tool 的输入参数。提供导入器(MCP / OpenAPI)和导出器(prompt 文本 / Ollama 原生 tools / JSON Schema)。

**关键文件**:
- `schema/types.py` — `ArgField` / `ArgSchema` dataclass
- `schema/registry.py` — `SchemaRegistry` 进程级单例,所有 tool 注册时挂上来
- `schema/importers.py` — MCP / OpenAPI spec → `ArgSchema` 转换
- `schema/validator.py` — LLM 给的 args dict 用 `ArgSchema` 校验 + 类型 coercion
- `schema/prompt.py` — `ArgSchema` → "Use [TOOL:name] with {arg1, arg2}..." prompt 文本
- `schema/ollama_export.py` — `ArgSchema` → Ollama 原生 tools API 的 JSON-Schema(2026-05 Tier 1-C 加)

**铁律**:`schema/` 不依赖任何其他业务模块(memory / hitl / runtime)。只依赖 stdlib + dataclasses。

**改动 checklist**:加新字段类型 → 同时更新 validator.py 校验 + ollama_export.py 转换 + prompt.py 文本。

### 10.2 `evaluation/` — 质量度量框架

**职责**:跑 golden-set bench、计算指标、生成阈值报告、给 CI 当 gate。

**两条独立 pipeline**:

**Retrieval bench**(`evaluation/retrieval_bench.py` + `evaluation/cli.py`):
- 输入:`data/golden_set.jsonl`(25 cases)、retrieval backend(BM25/Embedding/Hybrid)
- 输出:`recall@1`/`recall@3`/`recall@5`/`MRR`,按 language / tag / kind 分组
- CI gate:`./scripts/precheck.sh --eval` 跑 BM25,阈值 `recall@3 ≥ 0.40, MRR ≥ 0.30`

**Tool-compliance bench**(`evaluation/tool_compliance_bench.py` + `evaluation/compliance_cli.py`):
- 输入:`data/tool_compliance_set.jsonl`(18 cases)、LLM engine(text 协议或 native tools)
- 输出:`parse_ok`(语法解析率)/`name_ok`(选对工具率)/`args_ok`(参数对率)/`compliance`(全过率)
- 用途:**数据驱动决策切模型 / 开 native tools**。本地或夜间跑,需要 Ollama。CI 只做结构验证(golden set JSONL 能解析)。
- 用法:
  ```bash
  # baseline (text 协议):
  python -m evaluation.compliance_cli --golden data/tool_compliance_set.jsonl --model qwen2.5:7b
  # 对照(native tools):
  python -m evaluation.compliance_cli --golden data/tool_compliance_set.jsonl --model qwen2.5:7b --native
  ```

**关键文件**:
- `evaluation/types.py` — `EvalCase` / `EvalCaseResult` / `BenchReport`(retrieval 用)
- `evaluation/tool_compliance_types.py` — `ToolCallCase` / `ToolCallResult` / `ToolComplianceReport`
- `evaluation/golden_set.py` — JSONL 加载器(retrieval)
- `evaluation/retrieval_bench.py` — retrieval runner
- `evaluation/tool_compliance_bench.py` — compliance runner + 平衡大括号 JSON 提取器
- `evaluation/reporters.py` — text / JSONL 报告
- `evaluation/cli.py` — retrieval CLI
- `evaluation/compliance_cli.py` — compliance CLI

**铁律**:`evaluation/` 只在 bench 时构造 engine,**不参与运行时**。修改 bench 不影响产线行为。

**改动 checklist**:加新 case → JSONL 一行;加新指标 → `BenchReport`/`ToolComplianceReport` 加属性 + CLI 加 `--fail-below-X` flag + ARCHITECTURE.md §6 阈值表更新。


## 11. 多 agent 模块(`a2a/` + `registry/`)

Phase-1 引入 multi-agent foundation 后,这两个模块从"未来预留"升级为活跃组件。**当前还没有跨 agent 调用**——Phase 1 只做 identity + discovery。

### 11.1 `a2a/` — A2A 协议层

**职责**:实现 Google A2A Protocol v0.3.0 inbound 端 + AgentCard 暴露。负责把外部 JSON-RPC 请求翻译成 internal `ITOpsAgentExecutor.execute()` 调用,并把内部 chunk 流转回 SSE。

**关键文件**:
- `a2a/server.py` — FastAPI sub-app factory (`create_a2a_app`),挂载到 `/api/v1/a2a/*`
- `a2a/agent_card.py` — `get_agent_card(base_url, identity=None)`,Phase 1 后 identity 驱动
- `a2a/agent_executor.py` — `ITOpsAgentExecutor` + 6 processor 链(token/batch_token/message/node_step/node_result/extra)
- `a2a/request_handler.py` — JSON-RPC method router(message/send · message/stream · tasks/get · tasks/cancel)
- `a2a/event_queue.py` — 异步事件队列,sealing + 多消费者
- `a2a/schemas.py` — pydantic 模型:Task / Message / Part / Artifact / event types
- `a2a/push_notifications.py` — webhook 回调(指数退避)
- `a2a/task_store.py` — in-memory task state

**铁律**:`a2a/` 不依赖任何 business 模块。它是**纯协议层**,任何 executor 都能 plug in。

### 11.2 `registry/` — Agent 注册表

**职责**:agent 的"电话簿"——自注册、peer discovery、health check、load-balanced resolution。

**关键文件**:
- `registry/registry.py` — `AgentRegistry`,核心 register/resolve/list/health-loop
- `registry/discovery.py` — `AgentDiscovery`,fetch peer AgentCards 通过 HTTP
- `registry/store.py` — `InMemoryRegistryStore` + `RedisRegistryStore`(多副本)
- `registry/router.py` — `/registry/agents/*` HTTP endpoints
- `registry/schemas.py` — `AgentEntry` / `AgentSkill` / `ResolutionResult`

**Phase 1 后的关键 invariant**:每个 agent 实例的 `cfg.agent.agent_id` 是 registry 的 primary key。两个 LAN agent 副本必须 agent_id 不同(否则后注册的覆盖先注册的)。

### 11.3 多 agent roadmap

| Phase | 范围 | 状态 |
|-------|------|------|
| **Phase 1** | Identity + peer discovery + `/system/peers` endpoint | ✅ 完成(2026-05) |
| Phase 2 | Capability-based delegation:LAN agent → WAN agent | ⏳ 设计已定 |
| Phase 3 | Cross-agent HITL transparent passthrough | ⏳ 设计已定,等 Phase 2 稳定 |

Phase 2 +的工作量、风险点、产品决策详见 `task/inter/coordinator.py:A2ATaskDispatcher` 的 docstring + 各模块 DESIGN.md。

---

## 12. 业务 Profile 层(`profiles/`)— Phase 2A(2026-05)

### 12.1 动机

重构前业务工具(`list_devices` 等)直接住在 `tools/mock_tools.py`,通用 agent 循环和**一个**业务领域(企业 LAN 思科)绑死。加第二个领域(数据中心 fabric)、或跑一个纯助手(无业务),都得改框架文件。

Profile 把这个反过来:框架只认"加载当前 profile",不认 LAN/DC。依赖箭头 **框架 → profiles**,绝不反向。

### 12.2 三个 profile

| Profile | tools | skills | 用途 |
|---------|-------|--------|------|
| `default` | 0 | 0 | 纯助手 + 通用 meta 工具。**解耦证明**:框架能在 default 上跑起来,就说明 runtime/a2a/hitl_core 没有偷偷依赖某个业务领域 |
| `lan` | 20 | 7 | 企业 LAN:思科交换机/AP/内部防火墙(从旧 mock_tools.py 迁移)|
| `dc` | 7 | 3 | 数据中心 fabric:spine/leaf VXLAN、BGP EVPN、负载均衡、k8s overlay |

`AGENT_PROFILE` 环境变量(或 config.yaml `agent.profile`)选择,默认 `default`。

### 12.3 角色隔离(这是重点)

`lan` agent 的 tool registry 只有 LAN 工具;`dc` agent 只有 DC 工具。LAN agent 发 `[TOOL:dc_bgp_evpn_status]` 会收到 "tool not found"(runtime 的 fuzzy-match 提示)。跨域唯一办法是**委派**给 peer agent(Phase 2B)。Profile 是前提:没有隔离的工具集,委派毫无意义(每个 agent 本来就啥都有)。

`scripts/audit_profiles.py` 静态强制:(1) 每个 profile 的 callable/metadata key 对齐;(2) 业务 profile 之间工具/skill 名不重叠;(3) default 零业务;(4) 框架不 hard-import `profiles.lan`/`profiles.dc`;(5) **每个 `ToolLoader(`/`SkillLoader(` 调用必须显式传 `profile=`** —— 漏传会静默退化成空的 default profile,这个 bug 反复出现过 5+ 次(webui 重建 ×3、llm_engine fallback ×2、schema registry、retriever corpus),audit 现在从根上挡住。

### 12.4 通用 vs 业务的切分

| 关切 | 住在 | 为什么 |
|------|------|--------|
| `read_stored_result` / `process_stored_chunks` | `tools/common_tools.py` + `tools/builtin/registry.py` | 大结果分页机制,每个 profile 都要 |
| `_ts()` | `tools/common_tools.py` | mock 日志生成器跨 profile 共用 |
| `list_devices` 等 | `profiles/lan/` | LAN 业务 |
| `dc_bgp_evpn_status` 等 | `profiles/dc/` | DC 业务 |
| `ToolLoader`/`SkillLoader` | `tools/`/`skills/` | 框架,profile 无关 |

### 12.5 本地双 agent A2A 验证(角色隔离)

```bash
# Terminal 1 — LAN agent
AGENT_PROFILE=lan AGENT_ID=lan-agent AGENT_DISPLAY_NAME="LAN Agent" \
  AGENT_PEERS="http://localhost:8001/api/v1/a2a" \
  uvicorn main:app --port 8000

# Terminal 2 — DC agent
AGENT_PROFILE=dc AGENT_ID=dc-agent AGENT_DISPLAY_NAME="DC Agent" \
  AGENT_PEERS="http://localhost:8000/api/v1/a2a" \
  uvicorn main:app --port 8001

# 验证工具隔离:
curl -s http://localhost:8000/api/v1/a2a/.well-known/agent-card.json | jq '.skills[].id'
#   → lan_diagnose / lan_config / lan_observability
curl -s http://localhost:8001/api/v1/a2a/.well-known/agent-card.json | jq '.skills[].id'
#   → dc_fabric_diagnose / dc_fabric_config / dc_loadbalancer

# 互相发现(Phase 1):
curl -s http://localhost:8000/webui/system/peers | jq '{self:.self.agent_id, peers:[.peers[].agent_id]}'
#   → {"self":"lan-agent","peers":["dc-agent"]}
```

### 12.6 已知限制(见 TODO.md)

- **pragmatic 模式未按 profile 切分**:真实设备工具(`tools/pragmatic_tools.py`)无视 profile 全量加载。DC agent 在 pragmatic 模式仍会拿到 LAN Netmiko 工具。延后处理 —— A2A 验证用的是 mock 模式。
- **内置 netops MCP/OpenAPI mock 现在按 profile 门控**:`netops` MCP server + `netops_api` OpenAPI mock 是 LAN 业务集成(`get_device_status`/`get_devices` 等),只在 `profile=lan`(或 pragmatic 模式有显式配置)时加载。DC/default profile 不再混入这些 LAN 工具。后续若 DC 需要自己的 MCP/OpenAPI,应做 per-profile 集成声明。
- **Phase 2B(委派)未接线**:profile 给了隔离,利用隔离做跨 agent dispatch 是下一步。

### 12.7 每-agent 数据隔离(2026-05)

profile 隔离了**工具**,但早期所有 agent 共用一个 `data/` 目录 —— DC agent 会读到 LAN agent 的记忆/会话/演化 skill,两个同时跑还会互相覆写数据库。现在每个 `agent_id` 有独立数据子树。

**解析逻辑**(`cfg.agent_data_dir()`):
```
优先级:
  1. AGENT_DATA_DIR 环境变量(显式覆盖,最高)
  2. <memory.data_dir>/agents/<agent_id>/   (默认布局)
```

**每-agent 状态**(全部路由到 `data/agents/<agent_id>/`):

| 状态 | 路径 | 内容 |
|------|------|------|
| Memory | `memory/memory.db` + `memory/tool_cache/` | facts、sessions、user model |
| ToolResultStore | `tool_results.db` | 大工具输出缓存 |
| HITL checkpoint | `hitl_checkpoints.db` | pending 审批(operator 显式配 `hitl.checkpoint.sqlite_path` 仍可覆盖)|
| 演化 skills | `skills/*.md` | SkillEvolver 自动生成 |

**共享只读 fixtures**(留在 `data/`,**不**迁移):`golden_set.jsonl`、`tool_compliance_set.jsonl`(A/B compliance bench 用,跨 agent 共享)。

**两个 agent 的实际布局**:
```
data/
├── golden_set.jsonl              ← 共享
├── tool_compliance_set.jsonl     ← 共享
└── agents/
    ├── lan-agent/
    │   ├── memory/{memory.db, tool_cache/}
    │   ├── tool_results.db
    │   ├── hitl_checkpoints.db
    │   └── skills/
    └── dc-agent/
        ├── memory/{memory.db, tool_cache/}
        ├── tool_results.db
        ├── hitl_checkpoints.db
        └── skills/
```

**从旧单-agent 状态迁移**:`./scripts/migrate_data_to_agent.sh <agent_id>` 把共享 `data/` 里的每-agent 状态移进 `data/agents/<agent_id>/`,保留共享 fixtures。详见下方迁移指导。
