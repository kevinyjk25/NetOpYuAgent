# integrations — 设计与实现说明书

> 跨模块**胶水层**。所有"把内部能力暴露给外界"或"把外部 SDK/服务接入内部"的代码住这里。
> **铁律**:这个模块写的代码是无 LLM 业务逻辑的纯转接,业务规则属于 `runtime/`, `hitl_core/`, `skills/`。

---

## 1. 职责

| 子目录 | 职责 | 不做什么 |
|--------|------|---------|
| `adapters/` | 把内部能力包装给外部协议(A2A / HITL transport)+ 写入回 `agent_memory` | 不实现业务规则、不直接构造 prompt |
| `clients/` | 调用外部服务的客户端(LLM API、Embedding API、MCP server、OpenAPI server)| 不缓存查询结果(那是 runtime)、不解析 LLM 文本指令(那是 `runtime/directive_parser`)|
| `router/` | `ToolRouter` 把 MCP/OpenAPI/local 工具统一成一个 callable registry | 不决定 *何时* 调工具(那是 runtime loop)|

---

## 2. 公开接口

### 2.1 `integrations.adapters`

```python
from integrations.adapters.hitl_executor import HitlExecutor
from integrations.adapters.fact_conflict_detector import FactConflictDetector
from integrations.adapters.memory_facts_adapter import MemoryFactsAdapter
```

- **`HitlExecutor`**:外部 A2A 协议入口。`execute_query(query, session_id, ...) → result_dict`,内部驱动 runtime loop + `HitlPipeline`,把中断转成 `Multi-mode HITL raised` 异常给外部 catch。
  - **deferred wiring**:`set_skill_evolver(evolver)` — main.py 构造时 evolver 还没建,后调。
- **`FactConflictDetector`**:写 fact 前查相似 fact,LLM 判等价/精化/矛盾/无关,返回 `ReconcileResult`。
- **`MemoryFactsAdapter`**:把 `agent_memory.MemoryFact` 翻成 `hitl_core` 需要的简化 dict。

### 2.2 `integrations.clients`

```python
from integrations.clients.llm_engine import OllamaEngine, patch_runtime_loop
from integrations.clients.embedder import OllamaEmbedder
from integrations.clients.mcp_client import MCPClient
from integrations.clients.openapi_client import OpenAPIClient
```

- **`OllamaEngine`**:LLM 客户端,`generate(...) → str`。支持 think-tag 移除、token 统计、retrieval-aware prompt 拼接。
- **`patch_runtime_loop(executor, llm_engine)`**:monkey-patch `AgentRuntimeLoop._call_llm` 把 stub 替换成真 LLM。**反模式但必要** —— runtime 不能直接 import llm_engine(循环依赖)。
- **`OllamaEmbedder`**:向量化客户端,跟 `agent_memory.EmbeddingBackend` 兼容。
- **`MCPClient`**:Model Context Protocol 客户端,发现远程 tools。
- **`OpenAPIClient`**:从 OpenAPI spec 生成 tool 列表。

### 2.3 `integrations.router`

```python
from integrations.router.tool_router import ToolRouter
```

- **`ToolRouter`**:统一注册 MCP tools + OpenAPI ops + local callables,对外暴露一个 `{name: async_callable}` 的 registry。
- `register_mcp(client)` / `register_openapi(client)` / `register_local(name, fn, schema)`
- `as_callables() -> dict[str, async_fn]` —— 喂给 runtime loop

---

## 3. 核心类与数据流

### 3.1 `HitlExecutor` 数据流(最复杂的 adapter)

```
外部 A2A 协议
      │
      ▼
HitlExecutor.execute_query(query, session_id)
      │
      ▼
runtime/loop.py:AgentRuntimeLoop._run_internal()  ← 通过 self._runtime_loop
      │
      ▼ 检测到需要 HITL approval / clarification
      │
      ▼ raise HitlInterruptRaised(payload, kind, interrupt_id)
      │
      ▼ HitlExecutor catch → 在 hitl_core 注册 interrupt
      │
      ▼ 把 chunk_log + interrupt_id 返回给外部
      │
      ▼ 外部 POST /hitl/{id}/{decision} 触发 resumer
      │
      ▼ HitlRouter dispatch → self._agent_loop_resumer(handle, decision)
      │
      ▼ 恢复 loop,继续 turn,直至完成或再次中断
      │
      ▼ 写 memory (self._writeback) → 返回 final result_dict
```

关键 resumer:
- **`_agent_loop_resumer`** — user_choice / clarification / 整 agent restart
- **`_tool_call_resumer`** — 单 tool approve/edit 后 resume
- **`_batch_execute_after_resolution`** — N-target batch 完成后并发 dispatch + 一次性 SkillEvolver hook

### 3.2 `OllamaEngine` 数据流

```
runtime.loop._call_llm(messages, ...)
       │
       ▼ (patched at startup) OllamaEngine.generate(...)
       │
       ▼ retrieval pre-pass (tool + skill + meta-tools)
       │   ↓  retrieved tools/skills → 注入到 system prompt
       │
       ▼ HTTP POST → http://localhost:11434/api/chat
       │
       ▼ 流式 token decode,累积成 response str
       │
       ▼ strip <think>…</think>(可配置),log token counts
       │
       ▼ return str  → runtime loop 再解析 [TOOL:...] 指令
```

### 3.3 `FactConflictDetector` 数据流(audit_wiring 修复案例)

```
MemoryAdapter.add_fact(...)         ← caller
       │
       ▼ self._conflict_detector.insert_with_reconcile(...)
       │   ↓  (deferred-wired by main.py)
       │
       ▼ 语义检索 top-K 已有 facts
       │
       ▼ LLM 判 verdict ∈ {equivalent, refinement, contradiction, unrelated}
       │
       ▼ 分支执行:
       │   - equivalent     → boost existing.confidence, no insert
       │   - refinement     → boost + insert new
       │   - contradiction  → LLM 综合调和,可能 invalidate 旧 fact
       │   - unrelated      → 直接 insert
       │
       ▼ ReconcileResult(verdict, inserted_fact_id, related_fact_id, action)
       │
       ▼ adapter 拿 fact_id 返回上层
```

---

## 4. 关键设计决策

### 4.1 为什么 adapters 不放 runtime 业务?

A2A 协议改变 / HITL transport 改成 SSE/WebSocket / 写到 Redis 而非 process-local store —— 这些都不该让 runtime loop 改一行代码。adapter 是**协议/存储 boundary**,业务规则上浮到 runtime + hitl_core。

### 4.2 为什么 `patch_runtime_loop` 用 monkey-patch?

`runtime/loop.py` 的 `_call_llm` 是 stub 占位。真正的 LLM 客户端在 `integrations/clients/llm_engine.py`。如果 runtime 直接 import llm_engine:
- 循环依赖:runtime → llm_engine → runtime(为了 retrieval-aware prompt)
- runtime 包失去"无 fastapi/pydantic/httpx 依赖"的承诺(audit 强制)

Monkey-patch 在 startup 一次,把 stub 替换成真实现。**违反 "不要 monkey-patch" 直觉,但是这里是正确解**。

### 4.3 deferred wiring 模式

`HitlExecutor.set_skill_evolver(...)` 这种 setter 是项目级模式(见 `MemoryAdapter.set_conflict_detector`)。理由:
- 构造时依赖未就绪(evolver 需要 LLM smoke test 先过)
- 避免循环依赖(adapter 不该知道 evolver 的存在)
- 让 feature 可关闭(未 wire 时降级到 legacy path)

`scripts/audit_wiring.py` 强制每个 `services[k]` 必须被外部 reader 用,catch "ghost service"(写了不读,导致 feature 装上了但没生效)。

### 4.4 `ToolRouter` 为什么把三种 tool 揉一起?

Runtime loop 只关心"按名字调 callable",不关心 tool 来自 MCP server / OpenAPI / 本地 Python。三种来源在 ToolRouter 内部各自处理 schema 转换 + auth header / endpoint,对外接口完全统一。这让"加新 backend"(grpc / 自定义协议)只需扩 ToolRouter,runtime 不动。

---

## 5. 跨模块依赖与扩展点

### 5.1 这个模块依赖谁

```
integrations
   │
   ├── runtime          (HitlExecutor 调 AgentRuntimeLoop)
   ├── hitl_core        (HitlExecutor 注册/触发 interrupts)
   ├── agent_memory     (FactConflictDetector / MemoryFactsAdapter 读写)
   ├── skills           (HitlExecutor 接收 SkillEvolver injection)
   ├── tools            (ToolRouter 接收 callable registry)
   ├── retrieval        (OllamaEngine 用 tool/skill retriever)
   └── schema           (ToolRouter 用统一 schema 类型)
```

### 5.2 扩展点(改这个模块时常见任务)

| 任务 | 改哪里 |
|------|--------|
| 接入新的 LLM 后端(OpenAI/Anthropic)| 新建 `clients/openai_engine.py`,实现 `LLMEngineProtocol`,main.py 切换 |
| 接入新的 embedding 服务 | 同上,`clients/<x>_embedder.py` |
| 加新的 tool 来源(gRPC server)| `router/tool_router.py` 加 `register_grpc(client)` |
| HITL transport 改成 WebSocket | `hitl_core/transport/` 加新 adapter,executor 不变 |
| 加一个 deferred-wired 新 adapter | 学 `set_skill_evolver`:加 setter,main.py 在依赖就绪后调 |

### 5.3 不该在这里加什么

- ❌ 业务规则(stop policy / context budget / skill matching)→ runtime
- ❌ HITL 决策逻辑(approve/reject 何时触发)→ hitl_core/triggers.py
- ❌ Memory 内部存储格式 → agent_memory/stores
- ❌ Prompt template → integrations/clients/llm_engine.py 是唯一例外(prompt 拼装属 LLM 客户端的协议责任)

---

## 6. 修改指南

### 6.1 改之前必须知道

- **HitlExecutor 是巨型 file(1647 行)**。最近的改动集中在三处:
  - `__init__` 和 `set_skill_evolver`(deferred wiring)
  - `_capture_chunk` + `_heartbeat_loop`(progress chunks for UX)
  - `_batch_execute_after_resolution`(batch finalizer + SkillEvolver hook)
- **LLM engine prompt 模板**:`integrations/clients/llm_engine.py` 是项目里**唯一允许**写 prompt 模板的地方,`scripts/audit_prompt_templates.py` 会扫 brace 正确性,改模板必须保留 `{confirmed_facts_section}` `{extra_tools_section}` `{skill_summary}` 三个白名单 placeholder。
- **ToolRouter 的 schema 协调**:三种 tool 来源 schema 格式各异,在 router 里统一成 JSON schema dict。改的时候**保证 LLM 看到的 schema 形状一致**,不然 LLM 会 hallucinate 错的参数名。

### 6.2 改完必须跑

```bash
./scripts/precheck.sh --audits           # 5 audits 全过
python -m unittest tests.test_production_safety   # 21 tests
```

特别关注:
- `audit_module_independence` — 别把 `agent_memory` 直接 import 到 clients(应该走 adapter)
- `audit_imports` — 改文件名/移动文件后跑
- `audit_wiring` — 加了 service key 但没人读 → 失败

### 6.3 调试套路

| 症状 | 看哪里 |
|------|-------|
| HITL 决策提交后无响应 | `HitlExecutor._agent_loop_resumer` + `hitl_core/router.deliver` log,grep `interrupt_id` |
| LLM 返回空 / 卡住 | `OllamaEngine.generate` 的 token-stream log,看 prompt 是否超长 |
| Tool 调用 NameError | `ToolRouter._registry` 是否包含该 name,grep `register_mcp/openapi/local` |
| Batch 完成但 skill 没 evolve | `_batch_execute_after_resolution` 末尾,`self._skill_evolver` 是否为 None(main.py wire 失败)|

### 6.4 测试覆盖

- **没有专属单元测试**(integrations 是 thin layer,测试在它依赖的模块)
- **production safety 测试**间接覆盖:`tests/test_production_safety.py:TestAgentMemoryIntegration` 测 `MemoryFactsAdapter`,`TestAuthJWT` 测 auth flow(adapter 调用)

---

## 7. 已知限制 & TODO

- `hitl_core/batch.py` Multi-replica Redis pubsub —— **故意不实现**,见该模块 docstring。当前 single-replica only。
- `OllamaEngine` 不支持 native tool calling —— 通过 `[TOOL:name] {args}` 文本协议绕开。换 OpenAI 时要走 native tools(改 `runtime/directive_parser.py` 而不是 engine)。
- `FactConflictDetector` LLM reconcile 慢(8s+/call),写入路径若失败会自动降级到直 insert(behavior 透明)。
