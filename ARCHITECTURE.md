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
| `skills/` | ~2300 | 复用任务模板、SkillEvolver 自动生成、Journal 反馈 | `skills/DESIGN.md` |
| `tools/` | ~2500 | Tool callable 实现(mock + pragmatic)+ metadata | `tools/DESIGN.md` |
| `integrations/` | ~4500 | 跨模块胶水(HitlExecutor、LLM/Embedder/MCP 客户端、ToolRouter)| `integrations/DESIGN.md` |
| `webui/` | ~5500 | FastAPI + SSE 前端(不含 DESIGN.md,改动局限于路由)| — |
| `memory/` | ~250 | `MemoryAdapter` thin wrapper,给 runtime 用 | — |

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
1. Config           (config.py:Config.load)
2. Logging          (logging_config.set_*)
3. agent_memory     (MemoryManager + MemoryAdapter)
4. hitl_core        (Store + Router + Pipeline + Audit + ChunkQueue.idle_watchdog)
5. registry         (AgentRegistry)
6. task             (Task module)
7. integrations.adapters.executor = HitlExecutor(...)  ← skill_evolver=None
8. LLM engine       (OllamaEngine 启动 + smoke test)
9. integrations.clients.embedder
10. tools           (ToolLoader → ToolRouter)
11. MCP + OpenAPI   (discover external tools)
12. PolicyEngine    (from config.yaml)
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

**关键观察**:步骤 14 和 17 都是 `set_*` deferred wiring。`audit_wiring.py` 验证它们都连上了。

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

LLM ↔ Agent 通过 `[TOOL:name] {args}` / `[TOOL_BATCH:name] [{...},...]` / `[SKILL_LOAD:id]` 通信。**所有**解析在 `runtime/directive_parser.py`,`scripts/audit_directive_parsing.py` 强制单一入口。

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
| HIGH | `loop.py` 2634 行,应拆分 | runtime |
| MED | Multi-replica Redis pubsub 未实现 | hitl_core |
| MED | Vector ANN(hnswlib)未接 | agent_memory + retrieval |
| MED | SkillEvolver 不学失败 case | skills |
| LOW | LLMJudgeRetriever 未上线 | retrieval |
| LOW | `context_budget_v2` 跟 v1 共存 | runtime |
| LOW | hitl_core 缺单元测试目录 | hitl_core |
| LOW | tools 缺单元测试目录 | tools |

---

## 9. 模块独立维护原则

每个模块的 `DESIGN.md` 是**单文件 onboarding**:新人改一个模块只需读对应一份。

修改边界:
- **改一个模块** → 只读该模块 DESIGN.md + 跑该模块测试
- **改两个模块**(比如 runtime + hitl_core)→ 读两份 + 读本文档 §4
- **改三个或更多** → 设计可能有问题,先讨论拆分

每份 DESIGN.md 必含 6 节:**职责 / 公开接口 / 数据流 / 设计决策 / 跨模块依赖 / 修改指南**。
新加模块照此模板,**作为 PR 必须项**。
