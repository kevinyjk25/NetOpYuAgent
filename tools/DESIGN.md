# tools — 设计与实现说明书

> **可调用的 agent 工具**。tool = "LLM 通过 `[TOOL:name] {args}` 触发的 async function"。每个 tool 既要 callable 实现,也要 metadata(给 LLM 看的 description / args schema)。
> **铁律**:tool 是**无状态 async function** `(args: dict) → str`。状态(cache / pagination)由 `runtime/tool_cache.py:ToolResultStore` 在外部管。

---

## 1. 职责

| 文件 | 职责 |
|------|------|
| `loader.py` (139) | `ToolLoader(mode)` — 按 mode 装配 callables + metadata + skill defs |
| `mock_tools.py` (1361) | mock 模式所有 tool 实现(syslog/prometheus/netflow/device_*/...)|
| `pragmatic_tools.py` | pragmatic 模式实现(调真 device / API)|
| `builtin/registry.py` | 跨 mode 始终注册的 tool metadata(`read_stored_result`, `process_stored_chunks`)|
| `mock/registry.py` | mock 模式专属 metadata |
| `pragmatic/registry.py` | pragmatic 模式专属 metadata |
| `__init__.py` | 仅 export `make_read_stored_result_tool`(需要 ToolResultStore 注入)|

---

## 2. 公开接口

### 2.1 主入口 `ToolLoader`

```python
from tools.loader import ToolLoader

loader = ToolLoader(mode="mock")        # 或 "pragmatic"

callables = loader.build_callables()    # {name: async_fn} — 给 runtime 用
metadata  = loader.build_metadata()     # {name: {description, args_schema, ...}} — 给 LLM prompt 用
skill_defs = loader.skill_definitions() # {skill_id: skill_dict} — 给 SkillCatalog 用
section   = loader.tool_section_for_prompt()  # 直接拼到 system prompt 的字符串
```

### 2.2 ToolResultStore 注入(特例)

```python
from tools import make_read_stored_result_tool
from runtime import ToolResultStore

store = ToolResultStore()
read_fn = make_read_stored_result_tool(store)
# read_fn 是 async (args) → str,通过 ref_id 分页读 spilled 大结果
```

`read_stored_result` 因为需要持有 store 引用,不能纯静态注册,所以需要工厂函数返回 callable。

### 2.3 Tool 协议

每个 callable 必须:

```python
async def my_tool(args: dict[str, Any]) -> str:
    """
    Returns plaintext result. If > N chars, caller (runtime) automatically
    spills to ToolResultStore and replaces with [STORED:tool:ref_id] in prompt.
    """
    # validate args
    if not args.get("device_id"):
        return "[Error: device_id is required]"
    # do work (DB / HTTP / mock data)
    return f"Device {args['device_id']} status: ok\n..."
```

返回 str。错误**也是 str**(以 `[Error: ...]` 起手),不要 raise —— LLM 应该看到错误并修正,raise 会被外层 runtime 当 tool failure。

### 2.4 Metadata schema

```python
# In tools/<mode>/registry.py:
TOOLS = {
    "netflow_dump": {
        "description": "Dump netflow records for a site. Returns large output (use read_stored_result to page).",
        "args_schema": {
            "site":  {"type": "string", "required": True, "example": "site-a"},
            "top_n": {"type": "integer", "required": False, "default": 50},
        },
        "tags": ["network", "diagnostic", "large_output"],
        "returns": "Stored NetFlow records [STORED:] — use read_stored_result to page",
        "hitl": False,                # legacy boolean — kept for backward compat
        "action_type": "read_only",   # reversibility tier (added 2026-05)
    },
    ...
}
```

`description` 是 LLM 看到的;`tags` 给 retrieval;`returns` hint LLM 输出形式;`hitl` 是历史 boolean(`hitl_tool_names` 检查仍用)。

### 2.4.1 `action_type` 字段(reversibility 三档)

每个 tool **必须**声明 `action_type ∈ {read_only, reversible, destructive}`。这是 Claude-Code-inspired graduated trust 的核心数据。

| 值 | 语义 | 例子 |
|----|------|------|
| `read_only` | 无状态变更、无副作用 | `list_devices`, `get_device_status`, `prometheus_query`, `netflow_dump`(只读流量) |
| `reversible` | 创建可撤销 artifact | `rollback_service`, `failover`(可 fail back), `rollback_deploy` |
| `destructive` | 不可撤销变更 | `push_config`, `delete_resource`, `restart_service`, `drain_node` |

**怎么用**:
- `PolicyEngine.set_tool_metadata(...)` 在 startup index 全部 tools 的 action_type(main.py wire)
- `PolicyEngine.classify_action_type(tool_name)` 纯 dict lookup(~1µs vs LLM ~8s)
- `PolicyEngine.should_skip_hitl_for_tool(tool_name)` 结合 trust_mode 决定 HITL skip

**约定**:
- 不声明 `action_type` 的 tool **fall through 到 LLM** classify_destructive(保留向后兼容,零回归)
- 在 `auto_reversible` trust mode 下,**未声明 action_type 默认 NOT 跳 HITL**(安全优先)
- 区分 `reversible` vs `destructive` 看**回滚成本** — 5 分钟操作可恢复 = reversible,需要 30 分钟 incident response = destructive

**`run_command` 的特殊性**:声明为 `read_only`,但 production safety test 强制 allowlist 只允许 `show/display/get` 类命令。若改 allowlist 引入 mutation 命令,action_type 必须改成 `destructive`。

---

## 3. 核心数据流

### 3.1 启动期装配

```
main.py startup:
   loader = ToolLoader(mode="mock")
              │
              ▼ loader._callables() = {
                  ...所有 mock_tools.py 里 async def 的函数...
                  "read_stored_result": make_read_stored_result_tool(store),  ← 注入 store
                  "process_stored_chunks": make_process_stored_chunks_tool(store),
                }
              │
              ▼ loader._metadata() = {
                  从 builtin/registry + mock/registry merge
                }
              │
              ▼ ToolRouter.register_local(name, fn, schema) for each
              │
              ▼ build_tool_retriever_async(metadata) → embedding pre-index
```

### 3.2 LLM 调 tool

```
LLM response: "[TOOL:netflow_dump] {\"site\": \"site-a\", \"top_n\": 20}"
                  │
                  ▼ runtime/directive_parser → ParsedDirective(type=tool, name=..., args=...)
                  │
                  ▼ runtime/loop._dispatch_tool:
                  │   1. 检查 hitl_trigger(destructive?) — 这步在 runtime
                  │   2. _call_key(name, args) → tool_cache lookup
                  │   3. miss → callable = tools_registry[name]
                  │   4. await callable(args) → str result
                  │   5. if len(result) > spill_threshold (8KB):
                  │        ref = ToolResultStore.put(name, result)
                  │        result = f"[STORED:{name}:{ref}]"
                  │   6. cache.set(call_key, result)
                  │   7. return result → 注入下一 turn prompt
```

### 3.3 大结果分页(`read_stored_result`)

```
First call:
   netflow_dump(args) → 50KB
                  │
                  ▼ runtime 自动 spill → ToolResultStore.put(...)
                  │
                  ▼ return "[STORED:netflow_dump:abc123]" 给 LLM

Next turn LLM:
   "[TOOL:read_stored_result] {\"ref_id\": \"abc123\", \"offset\": 0, \"length\": 8000}"
                  │
                  ▼ make_read_stored_result_tool 内置闭包:
                  │
                  ▼ store.read(ref="abc123", offset=0, length=8000)
                  │   ↓ 取 8000 chars + has_more flag
                  │
                  ▼ return:
                  │   "[Page 0:8000 of 50000 chars, has_more=True]\n
                  │    <8000 bytes content>"

下次 LLM offset=8000 继续读;到 has_more=False 停。
```

### 3.4 `process_stored_chunks`(整文件分析)

```
LLM:
   "[TOOL:process_stored_chunks] {\"ref_id\":\"abc123\",\"operation\":\"summarise\"}"
                  │
                  ▼ store 读取**全部** 50KB
                  │
                  ▼ chunk_split (默认 4000 chars/chunk → 13 chunks)
                  │
                  ▼ operation ∈ {summarise / filter / count / extract / passthrough / reject}:
                  │
                  ├ summarise:  每 chunk LLM 摘要 → 合并 → 最终摘要
                  ├ filter:     每 chunk LLM 选 matches → 合并
                  ├ count:      每 chunk LLM 数 hits → sum
                  ├ extract:    每 chunk LLM 抽 patterns → 合并
                  ├ passthrough: 拼全部(适合非 LLM operation)
                  ├ reject:     每 chunk LLM 标 reject_reason → 合并
                  │
                  ▼ return aggregated str(典型 < 4000 chars)
```

**对比**:`read_stored_result` 是 LLM 逐页读再分析(13 turns × 1 read each);`process_stored_chunks` 是工具内部并行 LLM 调用,1 turn 出结果。**慢但少 turn**。

---

## 4. 关键设计决策

### 4.1 mock + pragmatic 双实现

每个 tool 都有 mock 版本(`tools/mock_tools.py`,假数据)和 pragmatic 版本(`tools/pragmatic_tools.py`,调真设备)。**好处**:
- 开发不需要真设备
- CI 跑 mock(确定性,快)
- production 切 pragmatic,接口不变

**契约**:mock 和 pragmatic 实现**必须返回相同 shape 的 str**(同字段、同 format)。LLM 看到的是 str,不知道在跑哪个。

`ToolLoader(mode)` 启动决定。两套 callable **不混用**(避免 LLM 看到 mock 数据后调 pragmatic tool 时上下文不连贯)。

### 4.2 返回 str 不返回 dict

LLM 看到的是文本。如果 tool 返回 dict:
- 要 `json.dumps` 一次(增加 tokens)
- LLM 对 JSON 的 attention 不如自然语言

mock_tools 用 fixed-width 表格 / 自然句段(`Device ap-01 status: ok, BGP neighbors: 3`)。**返回 str 是项目级约定**,改成结构化要全局重构。

### 4.3 错误用 str 不 raise

```python
if not args.get("device_id"):
    return "[Error: device_id is required]"   # ← 不 raise
```

理由:
- LLM 看到 `[Error: ...]` 文本 → 下个 turn 补 args
- raise 会被 runtime catch 当"tool failed",可能进 retry 或 abort
- 用户体验:tool 调错 args 不算系统失败,是 LLM 待修正

但**真实**异常(网络断 / DB unreachable)应该 raise,让 runtime 决定怎么办。

### 4.4 `read_stored_result` / `process_stored_chunks` 的工厂模式

这俩需要 `ToolResultStore` 实例引用。如果直接写成 module-level async function,要全局变量,违反"无状态"原则。

工厂返回闭包:

```python
def make_read_stored_result_tool(store: ToolResultStore):
    async def read_stored_result(args: dict) -> str:
        return store.read(args["ref_id"], args.get("offset", 0), ...)
    return read_stored_result
```

启动期 main.py 注入 store,得到 callable。callable 仍然纯 `(args) → str`,LLM 调用方式跟其他 tool 一样。

### 4.5 metadata vs callable 分离

为什么 `mock/registry.py` 只放 metadata,callable 在 `mock_tools.py`?

- **metadata 是数据**:LLM prompt 装填用,可以序列化、可以 build retrieval index
- **callable 是代码**:依赖 ToolResultStore 等运行时对象
- **分离让 retrieval 可单测**:不 import callable 也能拿到 corpus

`ToolLoader._merge_callables_with_metadata` 在装配时确保 name 一一对应,缺一就报错(metadata 没 callable / callable 没 metadata 都是 bug)。

---

## 5. 跨模块依赖

```
tools
   │
   ├── runtime.ToolResultStore  (read_stored_result / process_stored_chunks 用)
   ├── schema                   (args_schema 共享类型)
   └── (没了)

外部依赖 tools 的:
   - integrations/router/tool_router.py   (register_local)
   - runtime/loop.py                       (callable 调用)
   - retrieval/factory.py                  (tools_to_corpus 转 metadata)
   - main.py                               (装配)
   - skills/                               (skill 引用 tools_required)
```

### 5.1 加新 tool 的 checklist

1. **callable**:在 `mock_tools.py` 或 `pragmatic_tools.py` 写 `async def new_tool(args)`
2. **metadata**:在 `mock/registry.py` 或 `pragmatic/registry.py` 加 `TOOLS["new_tool"] = {...}`
3. **`builtin/registry.py`** 加 metadata(如果跨 mode)
4. **`action_type` 字段(必填,见 §2.4.1)**:
   - 完全无 side effect → `"read_only"`
   - 可撤销 mutation → `"reversible"`(必须能 rollback)
   - 不可撤销 → `"destructive"`
   - 不确定?**保守用 `"destructive"`**(`auto_reversible` 模式不会自动 approve)
5. **callable signature**:必须 `async def fn(args: dict) -> str`,**不要**变 positional args
6. **错误处理**:bad args → return `[Error: ...]`,真异常 → raise
7. **`hitl` 字段(legacy boolean)**:跟 `action_type` 相互独立但要协调 —
   - `action_type=destructive` 通常 `hitl=True`
   - `action_type=reversible` 通常 `hitl=True`(operator 看到再决定要不要 auto-approve)
   - `action_type=read_only` 通常 `hitl=False`
8. **测试**:在 `agent_memory/tests/` 或新建 `tools/tests/` 加 unit test(mock 模式确定性 output 易测)
9. **eval**:如果新 tool 应该被 retrieval 召回,在 `data/golden_set.jsonl` 加 case

### 5.2 不该在这里加什么

- ❌ 缓存逻辑 → `runtime/tool_cache.py`
- ❌ HITL gate(destructive? approve?)→ `hitl_core/triggers.py`
- ❌ Prompt 拼装 → `integrations/clients/llm_engine.py`
- ❌ 调用编排(retry / fallback)→ `runtime/loop.py`

---

## 6. 修改指南

### 6.1 改之前必须知道

- **mock_tools.py 数据需要内部一致**:`list_devices` 返回 `[ap-01, ap-02, ...]`,后续 `get_device_status(device_id="ap-01")` 必须知道 ap-01 存在。共用一份 `_DEVICE_STATE` dict(`mock_tools.py:551 _apply_config_lines_to_state`)。
- **返回 str 的格式**:LLM 对**列结构对齐**敏感(单空格对不齐看不懂表格)。用 fixed-width format 或纯句段,**不**用 ad-hoc 分隔符。
- **read_stored_result 默认 length 8000**(`tools/mock_tools.py:298`),`<= 16000` hard cap。这是为 LLM 节省 turn 数:8000 char / page × 7 page = 全部读完,而不是 25 个 2KB page。改这个要测看 LLM 在不同 page size 下 token 命中。
- **`process_stored_chunks` 慢**:每个 chunk 一次 LLM 调用,50KB 数据 → 13 chunks × 200ms 串行 ≈ 2.6s(并行可加速)。但比起 `read_stored_result` 走 13 个 turn(13 × 60s LLM) 快非常多。

### 6.2 改完必须跑

```bash
./scripts/precheck.sh --audits

# 验证 tool 函数能 import 不冒烟:
python3 -c "
from tools.loader import ToolLoader
loader = ToolLoader(mode='mock')
cbs = loader.build_callables()
md  = loader.build_metadata()
assert set(cbs) == set(md), f'mismatch: {set(cbs) ^ set(md)}'
print(f'{len(cbs)} tools ok')
"
```

### 6.3 调试套路

| 症状 | 看哪里 |
|------|-------|
| Tool 调用 `KeyError` | `ToolRouter._registry` 没有该 name,grep `register_local` 看是否漏了 |
| LLM 不知道有这个 tool | `tool_section_for_prompt()` 输出含吗?retriever index 含吗? |
| Tool 返回乱码 | callable 返回非 str?run mock_tools 函数直接看返回类型 |
| read_stored_result 报 ref_id not found | store TTL 过了?进程重启?(store 是 in-memory)|
| process_stored_chunks 超时 | chunks 数 × LLM latency,改 chunk_size 或 parallel_degree |

### 6.4 测试

- **没有 tools/tests/**(历史遗留)。
- 间接测试:`agent_memory/tests/test_memory.py` 测 ToolResultStore;`tests/test_production_safety.TestRunCommandAllowList` 测 `run_command` 的 allow-list。
- **建议加**:
  - `tools/tests/test_mock_tools.py` — 每个 tool 在多种 args 下返回符合预期(确定性 mock)
  - `tools/tests/test_loader.py` — mode 切换、metadata/callable 配对、builtin merge 行为

---

## 7. 已知限制 & TODO

- **mock_tools.py 1361 行**,但都是单个 tool 实现,没强耦合。拆分成 `mock/{network,device,observability}.py` 会更清晰。
- **`_DEVICE_STATE` 是 module global**,多线程跑 mock 会 race。当前 single-worker 安全;future multi-worker 要 lock 或 per-session state。
- **没有 schema validation**:LLM 调 tool 传错 args 类型(string when expected int)→ Python `TypeError`。当前靠 tool 内部 `int(args.get("top_n", 50))` 兜底。改成 pydantic 自动 validate 会更稳。
- **Tool versioning 缺失**:改 tool 行为(返回 format 变)无版本号,callers(skill / prompt-bound LLM)不知道。生产建议加 `description: "v2 — returns JSON not text"` 或显式 `version` 字段。
- **pragmatic_tools.py 实现度不完整**:很多 tool 的 pragmatic 版本是 TODO / stub。生产部署前必须补齐。
