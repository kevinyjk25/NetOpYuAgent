# runtime — 设计与实现说明书

> Agent 的**大脑**。LLM 调用、tool dispatch、turn 循环、何时停、何时调 HITL,全在这。
> **铁律**:这个模块**不直接**调 fastapi、httpx、ollama SDK。所有外部 I/O 都靠 `integrations/` 注入(monkey-patch `_call_llm` / 通过 callable registry 调 tool)。这样 `runtime/` 是纯逻辑包,可独立测试。

---

## 1. 职责

| 文件 | 职责 |
|------|------|
| `loop.py` (2634) | `AgentRuntimeLoop` 主循环 — turn iteration、tool dispatch、HITL trigger、verify、stop |
| `stop_policy.py` (475) | `StopPolicy` — 何时停。预算/重复/事实充足度多维评分 |
| `policy_engine.py` (326) | `PolicyEngine` — LLM 驱动的二分类(destructive? incident?) |
| `directive_parser.py` (356) | 解析 `[TOOL:name] {args}` / `[TOOL_BATCH:...]` / `[SKILL_LOAD:id]` |
| `tool_cache.py` (332) | 内存 LRU + 大结果 spill 到 `ToolResultStore` 分页读 |
| `context_budget.py` (484) | priority-budget tokenization,什么塞 prompt 什么扔掉 |
| `skill_journal.py` (332) | 每个 turn 记录 skill usage,供 `SkillEvolver` 后续聚合 |

---

## 2. 公开接口

```python
from runtime import (
    AgentRuntimeLoop, RuntimeConfig, LoopResult, VerificationResult,
    ComplexityDecision, QueryComplexity, DelegationMode, ForkContextPolicy,
    StopPolicy, StopPolicyConfig, StopDecision, StopOutcome, LoopState,
    ContextBudgetManager, BudgetConfig, ToolResultStore, DeviceRef,
)
```

### 2.1 `AgentRuntimeLoop` — 主循环

```python
loop = AgentRuntimeLoop(
    tools={"netflow_dump": fn, ...},
    tool_metadata={...},
    config=RuntimeConfig(max_turns=15, ...),
    memory_router=memory_adapter,   # optional
    skill_catalog=catalog,           # optional
)
# 主入口(同步,内部跑 asyncio):
result: LoopResult = await loop.run_async(
    query="..", session_id="..", confirmed_facts=[..],
    on_chunk=callback,  # 每个 turn / tool / event yield 一个 dict
)
```

`run_async` 内部串行:**plan → call_llm → parse directive → dispatch tool / stop / HITL-trigger → verify → store fact → next turn**。

### 2.2 Stop 决策(被 loop 内部调)

```python
sp = StopPolicy(StopPolicyConfig(token_budget=10000, ...))
decision: StopDecision = sp.decide(state)
if decision.should_stop:
    return decision.outcome  # SUFFICIENT / BUDGET / STAGNATION / ...
```

### 2.3 Policy 二分类

```python
pe = get_policy_engine()              # 全局 singleton(main.py 启动设置)
result = await pe.classify("classify_destructive", {"query": "..."})
if result.match:                       # match: bool
    trigger_hitl(...)
```

### 2.4 Directive 解析

```python
from runtime.directive_parser import parse_directives, ParsedDirective
directives: list[ParsedDirective] = parse_directives(llm_response_text)
# 含 type ∈ {tool, tool_batch, skill_load, end}, args, raw
```

### 2.5 Tool 结果 + 分页

```python
store = ToolResultStore()                          # singleton in loop
ref = store.put(tool_name, large_str)             # [STORED:netflow_dump:abc123]
page = store.read(ref, offset=0, length=8000)     # paged read
```

---

## 3. 核心数据流

### 3.1 主循环 turn iteration

```
run_async(query, ...)
   │
   ▼ Turn N start
   │
   ├─[A]─ 拼 system prompt(含 retrieval 检索的 tools/skills + confirmed_facts)
   │
   ├─[B]─ _call_llm(messages) ──→ LLM response str
   │       │
   │       └ (实际通过 integrations.clients.llm_engine.OllamaEngine,monkey-patched)
   │
   ├─[C]─ parse_directives(response) ──→ list[ParsedDirective]
   │
   ├─[D]─ 对每个 directive:
   │       │
   │       ├─ type=tool ──→ dispatch:
   │       │       │
   │       │       ├ HITL trigger 检查(destructive policy / 用户 trigger)
   │       │       │   ↓  若需要 HITL → raise HitlInterruptRaised
   │       │       │
   │       │       ├ tool_cache.lookup(call_key)
   │       │       │   ↓  miss → 调 callable → 存 cache
   │       │       │
   │       │       └ 大结果 → ToolResultStore.put → 返回 [STORED:ref]
   │       │
   │       ├─ type=tool_batch ──→ 同上但收集 N args,raise BatchHitlInterruptRaised
   │       │
   │       ├─ type=skill_load ──→ catalog.load(skill_id) 注入 next prompt
   │       │
   │       └─ type=end ──→ 进入 verification
   │
   ├─[E]─ StopPolicy.decide(state)
   │       │
   │       └ stop=True → break loop
   │
   ├─[F]─ on_chunk(turn_summary) → 外部 SSE 拿到 progress
   │
   └─ Turn N+1
```

### 3.2 Stop policy 决策

```
StopPolicy.decide(state) 多维评分:
  - tokens_used > budget?       → BUDGET
  - 同 (tool, args) 调 2 次?    → DEDUP_REPEAT
  - facts_ledger 满 ≥ 阈值?     → SUFFICIENT  (LLM 答完了)
  - 没新 fact 连续 N turns?     → STAGNATION
  - hard cap max_turns?         → MAX_TURNS
任何一条命中 → StopDecision(should_stop=True, outcome=<X>, reason="...")
```

### 3.3 Context budget(每个 turn 之前)

```
ContextBudgetManager(total_budget) 收到本 turn 候选内容:
  - 必带:user_query, last_tool_result_summary
  - 优先:confirmed_facts(按 freshness × importance 排序)
  - 次优:past turns(回滚到能塞下的最大 N)
  - 可丢:older facts / read_stored_result pages
按 priority 排序,逐项 add_chunk,塞不下就 drop。
返回组装好的 messages list 给 LLM。
```

### 3.4 Directive parser

```
LLM response 文本(可能含 think tag):
  "我想看 ap-01 状态。\n[TOOL:get_device_status] {\"device_id\":\"ap-01\"}"
                       ↓
parse_directives() 用 brace-scan(find_balanced_end)定位 JSON 边界
                       ↓
ParsedDirective(type="tool", name="get_device_status", args={"device_id":"ap-01"})
```

**所有** directive 解析必须通过此文件(`audit_directive_parsing.py` 强制)。

---

## 4. 关键设计决策

### 4.1 `_call_llm` 为什么是 stub + monkey-patch?

见 `integrations/DESIGN.md §4.2`。runtime 不知道 OllamaEngine / OpenAI / Anthropic 存在,只知道 `_call_llm(messages) → str`。这样:
- runtime 可在 mock 测试里塞个 lambda,不需要 LLM
- 换后端不动 runtime
- audit_module_independence 通过(runtime 不 import integrations)

### 4.2 StopPolicy 为什么独立?

最早 stop 逻辑写在 loop 里,变成 200 行 if/elif。独立成 `StopPolicy` 后:
- **可单元测试**:`LoopState` 是纯 dataclass,造一个就能测各 outcome
- **可调优**:`StopPolicyConfig` 集中所有阈值,改 budget 不动 loop
- **可观测**:每个 stop 决策带 `reason` 字段,UI flow trace 看得见

### 4.3 PolicyEngine 为什么独立于 StopPolicy?

两个粒度不同:
- **PolicyEngine**:**LLM-driven** 二分类(是不是 destructive?是不是 P0?),用于 HITL 触发 + skill matching pre-gate。每条 policy 是 prompt template + 期望 JSON 输出。
- **StopPolicy**:**rule-driven** 状态评估(token/turn/dedup),每 turn 末跑一次,纯本地不调 LLM。

混在一起会让快路径(每 turn stop check)被慢路径(LLM classify)拖死。

### 4.4 Tool cache 双层(memory + disk spill)

LLM 经常重复调同一 tool 同一参数(prompt 漂移)。直接做:
- **L1**:`_call_key(tool, args)` 哈希 → memory dict,本 turn 命中 free
- **L2**:超过 N KB 的结果 spill 到 `ToolResultStore`(进程内,带 ref_id),prompt 里塞 `[STORED:ref]` 让 LLM 走 `read_stored_result` 分页读

这样:
- 重复调用免费(LLM 漂移健壮)
- 大结果不污染 prompt budget(50KB netflow_dump 一次 ≈ 12K tokens,塞不下)

### 4.5 Directive parser 强制单一入口

`scripts/audit_directive_parsing.py` 扫整 repo,任何**非** `runtime/directive_parser.py` 出现 `[TOOL:` 正则就报错。原因:
- LLM 生成的 directive 边界很 tricky(嵌套 JSON、引号、转义)
- 历史上有 4 个文件各写一份解析,出过 3 次"a 改了 b 没改"bug
- 集中后用 `find_balanced_end` 一份测试覆盖所有 caller

---

## 5. 跨模块依赖与扩展点

### 5.1 依赖关系

```
runtime
   │
   ├── agent_memory     (可选 — memory_router 注入)
   ├── skills/catalog   (可选 — skill_catalog 注入)
   ├── hitl_core/schema (HitlInterruptRaised 异常类型)
   └── retrieval        (HybridRetriever 注入,用于 system prompt 拼装)

runtime 不依赖:
   ✗ fastapi / pydantic / httpx
   ✗ integrations/* (反向:integrations 调 runtime)
   ✗ webui / main
```

### 5.2 常见扩展任务

| 任务 | 改哪里 |
|------|--------|
| 加新 stop 触发条件(比如"用户在 chat 取消") | `stop_policy.py:StopPolicy.decide`,加 outcome enum |
| 加新 directive 类型(比如 `[ASK_USER:...]`) | `directive_parser.py:parse_directives` + `loop.py:_handle_directive` |
| 改 LLM 后端 | **不要改 runtime**。改 `integrations/clients/llm_engine.py` |
| 改 prompt template | 同上,prompt 在 llm_engine.py |
| 加新 policy(LLM 二分类) | `config.yaml:policies:` 加条目,policy_engine 自动 pick up |
| 改 tool cache 容量 | `tool_cache.py:_DEFAULT_CACHE_SIZE` + `BudgetConfig` |
| 加 skill 自动 trigger | `loop.py` 的 retrieval pre-pass + `skill_catalog.match` |

### 5.3 不该在这里加什么

- ❌ HTTP / FastAPI handler → `webui/`
- ❌ Memory CRUD → `agent_memory/`
- ❌ HITL UI / decision routing → `hitl_core/`
- ❌ 具体 tool 实现 → `tools/`
- ❌ Prompt 模板 → `integrations/clients/llm_engine.py`

---

## 6. 修改指南

### 6.1 改之前必须知道

- **`loop.py` 是项目最大文件(2634 行)**,但里面**只有一个类** `AgentRuntimeLoop`。读之前先扫一遍 `grep "    def " loop.py | head -30` 看方法清单。
- **Nudge 系统**:loop 在检测到 LLM 行为异常时(repeat tool / premature pivot / unread stored)注入 `_NUDGE: ...` 文本到 `confirmed_facts`,下个 turn LLM 看到提示。改 nudge 看 `_check_*` 系列方法。
- **Coreference 处理**:`hitl_core/coreference.py` 跟 loop 紧耦合,改 loop 时确认 HITL template 词不会被 coref 误绑(`_strip_hitl_templates` 防护)。

### 6.2 改完必须跑

```bash
# 编译
find . -name '*.py' -not -path '*/__pycache__/*' -exec python3 -m py_compile {} \;

# Audit(尤其 directive parsing)
./scripts/precheck.sh --audits

# 单元测试
python -m unittest tests.test_production_safety -v
```

如果改了 stop 阈值:跑 `evaluation.cli` 看 recall/MRR 没退化。

### 6.3 调试套路

| 症状 | 看哪里 |
|------|-------|
| Agent 不停 / max_turns 命中 | `StopPolicy.decide` 加 debug log,看每个维度评分 |
| LLM 一直调同一个 tool | `_call_key` dedup 失效?args 微变?`tool_cache._cache` 内容 |
| 大数据分页死循环 | `read_stored_result` length 默认 8000,LLM 选小了 → 看 `unread_stored_nudge` 是否触发 |
| Directive 解析失败 | `directive_parser.parse_directives` 加 print,看 `find_balanced_end` 返回位置 |
| Tool 调了但结果没进 prompt | `ContextBudgetManager` drop 了?priority 排序 + budget 超了? |

### 6.4 测试

- **没有完整 loop e2e 测试**(LLM 依赖,跑得起来要 ollama),用 mock LLM 测各 path。
- **`stop_policy.py` 应该单元测**(纯函数):造 `LoopState`,断言 `decide()` 返回的 outcome。当前未独立测试 — **建议补**。
- `directive_parser` 有 12 个 regression test 在 `scripts/audit_directive_parsing.py` 跑(audit 流程会跑)。

---

## 7. 已知限制 & TODO

- **`loop.py` 太大**。未来拆:turn loop / tool dispatch / verify / stop check 各自一个 file,共享 `_LoopContext` dataclass。
- **`context_budget_v2.py` 跟 `context_budget.py` 共存**(渐进迁移)—— v2 是新 priority-budget 算法,部分 caller 还在用 v1。完工后删 v1。
- **StopPolicy 不感知 user cancel**——前端"停止"按钮还没接进来,需新增 `StopOutcome.USER_CANCELLED`。
- **PolicyEngine 不缓存** —— 每次 classify 都调 LLM。如果同一 query 反复 classify,加 LRU 显著省 LLM call。
