# runtime — 设计与实现说明书

> Agent 的**大脑**。LLM 调用、tool dispatch、turn 循环、何时停、何时调 HITL,全在这。
> **铁律**:这个模块**不直接**调 fastapi、httpx、ollama SDK。所有外部 I/O 都靠 `integrations/` 注入(monkey-patch `_call_llm` / 通过 callable registry 调 tool)。这样 `runtime/` 是纯逻辑包,可独立测试。

---

## 1. 职责

| 文件 | 职责 |
|------|------|
| `loop.py` (~3250) | `AgentRuntimeLoop` 主循环 — turn iteration、tool dispatch、HITL trigger、verify、stop。`_stream_impl` 已拆分为阶段方法(见 §4.5) |
| `loop_types.py` (159) | 循环的公开类型定义：`QueryComplexity` / `DelegationMode` / `ForkContextPolicy` / `VerificationResult` / `ComplexityDecision` / `RuntimeConfig` / `LoopResult`。`loop.py` 重新 import,`runtime/__init__` 照旧 re-export |
| `loop_context.py` (66) | `_LoopContext` — `_stream_impl` 每轮的可变状态容器（state / tool_outputs / called_tools / 澄清标志 / recall+skill 记忆化缓存）。以 `(self, ctx)` 传给各阶段方法 |
| `loop_helpers.py` (216) | 从 loop 抽出的无状态纯函数：`strip_thinking` / `is_complete` / `skill_loads_in` / `format_final` / `query_mentions_concrete_target` / `call_key` / `build_tool_ledger` / `page_default_size_for_ledger`。loop 保留同名薄壳/别名,调用点不变 |
| `stop_policy.py` (475) | `StopPolicy` — 何时停。预算/重复/事实充足度多维评分。`StopOutcome` 含 `USER_CANCELLED` |
| `policy_engine.py` (326) | `PolicyEngine` — LLM 驱动的二分类(destructive? incident?) |
| `directive_parser.py` (356) | 解析 `[TOOL:name] {args}` / `[TOOL_BATCH:...]` / `[SKILL_LOAD:id]` / `[DELEGATE:...]` |
| `tool_cache.py` (332) | 内存 LRU + 大结果 spill 到 `ToolResultStore` 分页读 |
| `context_budget.py` (484) | legacy 上下文组装 + 分页压缩(`compress_paged_outputs`) |
| `context_budget_v2.py` | 可选的 priority-budget 策略(`TokenBudget`,由 `cfg.context_budget.strategy="priority"` 选用,见 §4.6) |
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

### 2.3 Policy 二分类 + fast-path + trust mode

```python
pe = get_policy_engine()              # 全局 singleton(main.py 启动设置)
result = await pe.evaluate("classify_destructive", "...")   # LLM 主路径
if result.match:
    trigger_hitl(...)

# Fast-path (added 2026-05, no LLM):
intent = pe.classify_query_intent("list devices")      # → "read_only"
intent = pe.classify_query_intent("restart prometheus") # → "destructive"
intent = pe.classify_query_intent("ap-01 怎么回事")     # → None (ambiguous, fall through)

# Tool metadata fast-path (after pe.set_tool_metadata(...)):
atype = pe.classify_action_type("list_devices")  # → "read_only"

# Trust mode (graduated trust, added 2026-05):
pe.set_trust_mode("auto_reversible")    # cautious | auto_reversible | bypass
skip, reason = pe.should_skip_hitl_for_tool("list_devices")
# skip=True when (trust_mode=auto_reversible AND action_type ∈ {read_only, reversible})
#               OR trust_mode=bypass
# Falls back to skip=False ("conservative") on any uncertainty
```

`classify_query_intent` 在 `classify_async` 内部自动调用,routine "list/show/check" query 完全跳 LLM,延迟从 ~8s → ~1µs。详见 `runtime/policy_engine.py:classify_query_intent` docstring 的安全论证。

`should_skip_hitl_for_tool` 在 `runtime/loop.py` HITL gate(`_needs_hitl` 决策点)被调用 — 即使 tool 在 `hitl_tool_names` 白名单,如果当前 trust_mode 允许且 action_type 是 read_only/reversible,gate 短路。

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

### 2.6 Hooks(Sprint 2,2026-05)

```python
from runtime import HookEvent, get_hook_registry

reg = get_hook_registry()

# 注册 hook(任何模块都可以,通常在 main.py 启动期)
async def my_audit_hook(ctx):
    audit_log.append({"event": "tool_use", "tool": ctx["tool"], "args": ctx["args"]})
    return ctx

reg.register(HookEvent.PRE_TOOL_USE, my_audit_hook, priority=50)

# Block 一个 tool(只能在 PRE_TOOL_USE)
async def policy_gate(ctx):
    if ctx["tool"] == "delete_resource" and not has_admin_role():
        ctx["blocked"] = True
        ctx["block_reason"] = "operator lacks delete permission"
    return ctx

reg.register(HookEvent.PRE_TOOL_USE, policy_gate, priority=80)  # 后跑
```

6 个核心事件:`PRE_TOOL_USE` / `POST_TOOL_USE` / `TURN_START` / `TURN_END`(预留)/ `SESSION_START` / `SESSION_END`。`runtime/hooks.py` docstring 详细文档。

**关键设计原则**(在 §4.8):
- Hooks 是 observer 而非 gatekeeper:异常被 log + swallowed,不中断 runtime
- 唯一例外 — `PRE_TOOL_USE` 可 block,但**只能**通过 `ctx["blocked"]=True` 显式 flag,不能通过 raise
- Priority order(low → high)— 后跑 hook 可覆盖前跑的 ctx 修改
- Zero LLM context cost(hooks 不出现在 prompt 里)

### 2.7 Tracing(Sprint-3-pre,2026-05)

```python
from runtime.tracing import configure, start_span, is_enabled

# Boot once (main.py 启动期):
configure(
    enabled         = cfg.observability.tracing_enabled,   # 默认 False
    service_name    = "netopyu-agent",
    service_version = "6.0.0",
    otlp_endpoint   = "http://collector:4317",             # 可选
    sample_ratio    = 1.0,                                  # 0.0-1.0
)

# 任何模块的任意热路径:
with start_span("llm.call", **{"llm.model": "qwen3.5:27b"}) as span:
    span.set_attribute("output.chars", len(result))
    # ... 业务代码 ...
```

**关键设计原则**:
- **`opentelemetry-*` 是可选依赖** —— 没装就走 `_NoopSpan`,所有 `start_span()` 调用零成本。
- **默认 OFF** —— `cfg.observability.tracing_enabled=False`。生产开关一行,不动代码。
- **降级失败模式** —— `configure(enabled=True)` 时 OTel SDK 设置失败,自动回退到 no-op,boot 不挂。
- **三处接入**(当前):
  - `hitl_executor.execute_query` → span name `agent.query`,attribute 含 `session_id` + facts count
  - `llm_engine._chat` → span name `llm.call`,attribute 含 model + message count + native_tools flag
  - `runtime.loop._dispatch_tool` → span name `tool.dispatch`,attribute 含 tool name + args count + result chars
- **未来 Sprint 3 扩展**:FastAPI / httpx auto-instrumentation,session_id → trace_id 确定性派生,Jaeger/Tempo dashboard。

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

### 3.5 L2 Snip — graduated context compaction

Claude Code 5 层 cascade 的简化版 — 我们做 **L1 Spill(已有)+ L2 Snip(新增)+ L5 Consolidate(已有 agent_memory)**。L2 在每个 turn 开始时跑,零成本同步。

```
runtime.loop while True:
   state.turns += 1
   │
   ▼ try_snip_tool_outputs(tool_outputs,
                           target_char_budget=cfg.budget.snip_tool_outputs_char_budget,
                           keep_recent=5)
   │
   ├─ total chars < budget → 原 dict 返回,freed=0
   │
   └─ total chars >= budget:
          │
          ▼ keep 最近 N + 任何含 [STORED:] ref 的 entry
          │
          ▼ 其余 entry 替换为单条 placeholder:
          │      "__snip_placeholder__":
          │        "[snip: N earlier tool result(s) omitted — ~M chars freed;
          │         see memory context for full history]"
          │
          ▼ 返回 (new_dict, freed_chars)
          │
          ▼ stream() yield SSE chunk: {node: "context_budget", type: "compaction",
          │                            freed_chars: M}
          ▼ run() 仅 log.info(snip log 内置)
```

关键点:
- **零 LLM 成本**(纯 dict 操作,< 1ms)
- **dropped content 不真丢** — 原始 raw outputs 已存在 `agent_memory` long-term store,下个 turn `recall_memory` 可重新检索
- **`[STORED:]` ref 永远保留**(50 chars 不值得丢,且 LLM 可能还要 `read_stored_result`)
- **零回归保证**:`snip_tool_outputs_char_budget=0` 完全关闭(default 32K chars ≈ 8K tokens)

### 3.6 reversibility fast-path 数据流

```
backend.py POST /chat:
   classify_async(query)
        │
        ▼ get_policy_engine().classify_query_intent(query)
        │
        ├─ "read_only"  → 直接 return ComplexityDecision(SIMPLE), 跳 LLM
        ├─ "destructive" → 直接 return ComplexityDecision(COMPLEX), 跳 LLM
        └─ None         → 走 LLM evaluate_any([...])   ← 原行为

runtime.loop turn HITL gate(_needs_hitl 决策后):
   if _needs_hitl:
        │
        ▼ pe.should_skip_hitl_for_tool(tool_name)
        │
        ├─ skip=True  → _needs_hitl=False, 记 audit log, 直接执行
        │              (例:trust_mode=auto_reversible AND action_type=read_only)
        │
        └─ skip=False → 走原 HITL 路径(raise interrupt)
```

**所有** directive 解析必须通过此文件(`audit_directive_parsing.py` 强制)。

### 3.7 Hook fire 点(Sprint 2)

```
runtime/loop.py:stream() 的 5 个 fire 点:

stream(query, session_id, ...)
   │
   │  ── wrapper(try/finally guarantees SESSION_END) ──
   │
   ▼ get_hook_registry().fire(SESSION_START, {session_id, query, delegation_mode})
   │
   ▼ _stream_impl(...) 进入 while True:
        │
        │  ── 每个 turn 顶 ──
        │
        ▼ L2 Snip(non-hook,turn 内部 prep)
        │
        ▼ get_hook_registry().fire(TURN_START, {session_id, turn, query, facts_count})
        │
        ▼ LLM call → tool dispatch loop:
              │
              ▼ get_hook_registry().fire(PRE_TOOL_USE, {tool, args, session_id, turn})
              │    │
              │    ├─ ctx["blocked"]=True → yield hook_block chunk, skip dispatch
              │    │
              │    └─ ctx["args"] 可能被 hook 修改 → 用修改后的 args
              │
              ▼ raw = await _execute_tool(name, args)
              │
              ▼ get_hook_registry().fire(POST_TOOL_USE, {tool, args, result, session_id, turn})
              │    │
              │    └─ ctx["result"] 可能被 hook 修改(redact / filter)→ 替换 raw
              │
              ▼ store stored output, accumulate tool_outputs
        │
        │  ── stop check → next turn 或 return ──
        │
        ▼ wrapper finally: get_hook_registry().fire(SESSION_END,
                                                     {session_id, total_turns,
                                                      tool_calls, outcome, stop})
            outcome ∈ {completed, consumer_closed, error}
            保证 abort / consumer-close / exception 路径都触发
```

Hook 失败(异常)被 log 并 swallowed,**永不**中断 runtime —— 唯一例外是 `PRE_TOOL_USE` 通过 `ctx["blocked"]=True` 显式标记。`TURN_END` 在 enum 里定义但当前**未 fire**(Sprint 2 — 跟 TURN_START 携带的信息有重叠,SESSION_END 已覆盖最终状态)。

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

### 4.6 Reversibility-weighted fast-path 为什么写在 PolicyEngine 而不是 HitlExecutor?

设计上 trust_mode 跟 HITL gate 紧关联,自然倾向放 `HitlExecutor`。但实际上 **HITL gate 决策点在 `runtime/loop.py`**(line ~1806),而 `runtime/` 不能 import `integrations/`(模块独立 audit 强制)。

解决:`PolicyEngine` 是 `runtime/` 自有,加 `set_trust_mode` + `should_skip_hitl_for_tool`,runtime/loop 通过 `get_policy_engine()` singleton 查询。main.py 是装配点,从 `cfg.hitl.trust_mode` wire 到 `PolicyEngine`。

附带好处:**fast-path 跟 classify_destructive 同处**,缓存 / 阈值 / 实现都一致;HitlExecutor 不需要知道 trust_mode 存在(零侵入)。

### 4.7 L2 Snip 为什么是 dict-based 不是 token-based?

Claude Code 的 5 层 cascade 用 `cache_deleted_input_tokens` 这种 prompt cache 经济学指标。我们用 ollama / OpenAI 直 API 没那个信号。改成:
- **chars**:简单可数,无需 tokenizer
- **dict**:利用 Python 3.7+ 插入序保证,直接按顺序丢老
- **keep `[STORED:]` ref**:50 chars 不值得丢,且 LLM 可能还要分页读

不做的事:
- Microcompact / Context Collapse(需要 prompt cache 信号)
- Auto-Compact fork child agent(已有 `MemoryConsolidator` 做这事)

Sprint roadmap 选 Snip 是因为**单层 + 零成本 + 立即生效**,跟 Hermes 单 `ContextCompressor` 哲学一致而非 Claude Code 多层。

### 4.8 Hooks 为什么是 "observer 而非 gatekeeper"(Sprint 2)

Claude Code 27 个 hook events 都允许 raise 来 abort runtime。我们选**只允许 `PRE_TOOL_USE` 通过 `ctx["blocked"]=True` 显式 block**,其他事件 hook 异常一律 log + swallow,**不**中断 runtime。

理由:
- **生产环境 network ops**:operator 在做 incident response 时,**任何**第三方扩展(metrics / audit / compliance hook)崩了**都不应**让 agent 终止。incident 不能等。
- **可观测性**:hook 失败的 log 是**通知**,让 operator 知道 metrics 可能丢一条,但 agent 仍然完成任务。
- **明确 block 路径**:policy 需要 block 时**显式**通过 ctx flag,不靠副作用(异常)。这让 reviewer 在 PR 时一眼看到"这个 hook 会阻断 tool"。

trade-off:hook 写 bug(忘了 set blocked=True)无声漏检。**所以建议 production hook 走 audit_log + 显式集成测试**,而非单依赖 hook 行为。

### 4.9 为什么 stream() 包 try/finally wrapper(Sprint 2)

原 `stream()` 1300 行 generator 内有 N 处 `return`。SESSION_END hook 如果直接在每处加,容易漏 + 重复 fire。改造:

- `stream()` → `_stream_impl()`(行为字节级未动)
- 新 `stream()` thin wrapper:`try { async for chunk in _stream_impl(): yield chunk } finally { fire SESSION_END }`

`GeneratorExit`(consumer 关 iterator) + exception 路径都进 finally,**保证 SESSION_END 必触发**(且只一次)。outcome 字段标 `completed` / `consumer_closed` / `error`,observer hook 自己分流。

代价:wrapper 多一次 async generator forwarding(每 chunk yield 进 wrapper 再 yield 出)。`pytest` 测过吞吐影响 < 1%,可接受。

### 4.7 `_stream_impl` 拆分(Item 4,2026-05)

`_stream_impl` 曾是 1448 行的单方法,把 turn loop 的所有阶段 + 一堆闭包局部变量挤在一起,定位和修改都困难。重构后(零行为变化,每步全量 audit+test 验证)拆成:

- **`_LoopContext`(`loop_context.py`)** —— 每轮的可变状态容器。原本散在闭包里的 `state` / `tool_outputs` / `called_tools` / 澄清标志 / recall+skill 记忆化缓存 + 刷新节奏簿记,现在集中在一个 dataclass。可变容器(dict/set)在 `_stream_impl` 里以局部别名绑定到 `ctx.*`,in-place mutate 自动同步,所以引用点零改动。
- **阶段方法**(都挂在 `AgentRuntimeLoop` 上,签名 `(self, ctx, ...)`,因为重度依赖 `self._budget`/`self._policy`/`self._memory`/`self._call_llm`,做成方法比独立模块函数干净):
  - `_refresh_recall(ctx, ...)` / `_refresh_skills(ctx, ...)` —— 两个条件性记忆化阶段,纯计算,写回 ctx 缓存。
  - `_assemble_context(...)` —— 统一 legacy / priority 两条上下文组装路径,`run()` 和 `_stream_impl` 共用(见 §4.8)。
  - `_run_clarification_gate(ctx, ...)` —— Type#3 澄清门,async generator。需要终止整个 stream 时 yield `{"_clarification_terminal": True}` 哨兵,调用方转发其余 chunk 并在哨兵处 `return`。
  - `_handle_tools(ctx, new_tool_calls, llm_response)` —— 每轮工具处理(单工具强制 + HITL gate + 执行 + post_verify + 分页 nudge)。async generator,**HITL stop 语义严格不变**:工具需审批时 yield 既有的 `stop_hitl`/`hitl_gate` chunk + `{"_tools_terminal": True}` 哨兵,调用方在哨兵处 `return`(等价于原来的内联 `return`)。

哨兵模式说明:async generator 无法像普通函数那样用 `return value` 把"该终止 stream"信号传出,所以约定用一个特殊 chunk(`_clarification_terminal` / `_tools_terminal`)作为带内信号;`_stream_impl` 的转发循环识别到它就 `return`,其余 chunk 原样 yield 给上游。测试:`test_clarification_gate.py`、`test_handle_tools_phase.py`(含"破坏性工具未经批准不执行"的安全断言)。

剩余:`_stream_impl` 仍约 875 行(主要是 token streaming + DELEGATE + 完成判定 + stop-check/ledger 这些与循环控制紧耦合、不宜外迁的部分)。

### 4.8 Context budget 策略选择(legacy vs priority)

`context_budget.py`(v1)和 `context_budget_v2.py`(v2)**不是新旧替代关系,而是两个并存的可选策略**(v2 自己的 docstring 写明了)。由 `cfg.context_budget.strategy` 选择:

- `legacy`(默认)—— `ContextBudgetManager.assemble`,固定顺序组装,超预算时按固定顺序丢尾部。
- `priority` —— `_assemble_priority` 把各 section 喂进 v2 `TokenBudget`,在硬 `total_chars` 上限下**按优先级裁剪**:P0 confirmed_facts 不可驱逐,P1 tool_results/working_set,P2 skills/retrieved_memory,P3 env 最先被裁。复用 v1 的 section formatter,所以渲染文本一致,只是裁剪策略不同。

两条路径都经 `_assemble_context` 这一个入口(§4.7),`run()` 和 `_stream_impl` 共用 —— 避免了之前 priority 只接在 `run()`、流式路径却一直走 legacy 的隐藏 gap。

### 4.9 业务逻辑注入点(L0/L1 解耦,Stage B,2026-05)

L0 loop 不含任何业务域逻辑;域特定行为靠注入。两个已落地的注入点:

- **`delegate_fn`**(Phase 2B)—— 跨 agent 委派的传输实现,`None` 时优雅降级。
- **`batch_resolver_fn`**(Stage B)—— 决定一个破坏性工具调用是否应该展开成多目标批量 HITL。这是业务逻辑(网络域:"prose 里点名了 sw-core-01 和 sw-core-02");L0 只负责调用注入的 resolver,`None`(如 default profile)→ 单目标 HITL(域无关的安全默认)。契约:`(tool_name, tool_args, llm_response, hitl_tool_names, confirmed_facts, all_parsed) -> Optional[list[(name,args)]]`。网络实现在 `profiles/network_batch_resolver.py`,由 `profiles.get_batch_resolver_for_profile()` 按 profile 选,webui/backend 注入。
- **coreferencer**(`HitlExecutor`)—— L0 默认 `build_neutral_coreferencer()`(空 pattern,永远 no-op);网络 profile 注入 `build_default_device_coreferencer()`。

`runtime/loop.py` **不 import `profiles/`** —— 所有业务 resolver 都是注入进来的,`audit_module_independence` 因此持续绿。后续开发者加新业务域时:实现一个 resolver 函数 → 在 `get_batch_resolver_for_profile` 注册 → 不动 L0 任何代码。这就是 L0(纯能力骨架)/ L1(业务注入)的边界。

### 4.10 跨 agent 委派的循环控制流(A2A Phase 3，2026-05）

`_stream_impl` 里 DELEGATE 处理这一段看似简单(解析 `[DELEGATE:]` → 调 `delegate_fn`),但有四层防重复/防风暴的控制,缺一不可。**委派是有身份、有生命周期的有状态任务,不是 LLM 每轮自由发起的指令** —— 早期版本用 env_ctx 上的计数/标志事后拦截 LLM 重复委派,但 env_ctx 是 per-`execute_query` 的,resume driver 每次起综合轮都重置,守卫形同虚设。现在四层是:

1. **单一委派闸门(跨 turn + 跨 stream,在 `task/delegation.py delegate_fn`)** —— 唯一咽喉点。身份 = `(session_id, target_agent)`;委派前查 TaskStore,若该 peer 有一个 `scope==INTER` 且**非终态**(RUNNING / AWAITING_PEER_HITL / PENDING;终态 = COMPLETED/FAILED/CANCELLED)的出站任务 → 抑制(不建任务、不 dispatch、yield 一个 `_delegation_suppressed` 注入块)。TaskStore 跨 turn 跨 stream 持久,且和 UI 读同一个 store → 闸门与 UI 状态不可能不一致。
2. **case2 peer HITL → park(在 `_stream_impl`)** —— peer 触发操作员审批时,原始 stream 必须**等**异步 stage-2 结果,不能空转 loop(空转会和 result 回调竞速:回调一把任务转 COMPLETED,闸门立刻释放,下一轮就重委派)。检测到 `_peer_hitl_pending` → yield 一个 `cross_agent_parked` marker + 确定性中间答复 → `return` 结束 stream。最终答复由 result 回调驱动的综合轮通过 `/chat/resumptions` 送达前端。
3. **本请求重委派阻断(per-stream,在 `_stream_impl`)** —— `_delegated_targets_this_request` 集合记录本 stream 已委派过的 peer。case1(同步、无 HITL)返回后若 LLM 把"对方只是反问"当成没解决、想重委派同一 peer → 闸门拦不住(任务已终态),由这个集合抑制,逼 LLM 综合/降级(spec 第2点)。per-stream scope 正确:后续新请求是新 stream、新集合。
4. **综合轮硬禁委派(per-turn,在 `_stream_impl`)** —— resume driver 起综合轮时设 `env_context={"_cross_agent_resume": True}`。这一轮只能基于已注入结果综合,任何 DELEGATE 一律丢弃。这是 per-turn 角色声明,不是被删掉的那个跨轮计数守卫。

附带:`[DELEGATE:]` / `[SKILL_LOAD:]` 与 `[TOOL:]` 一样,在可见 token 流里被 strip 掉(`strip_*_directives`),避免纯指令响应(如被抑制的重委派)把裸指令文本泄漏给用户。

回归测试:`test_delegation_gate.py`(6 例,闸门行为)、`test_delegation_park_on_peer_hitl.py`(park + case1 不重委派 + marker + 无指令泄漏)。

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

- **`loop.py` 是项目最大文件(~3250 行)**,主体是 `AgentRuntimeLoop` 这一个类(其公开类型/纯函数/每轮状态已分到 `loop_types.py`/`loop_helpers.py`/`loop_context.py`,见 §4.7)。读之前先扫一遍 `grep "    def \|    async def " loop.py` 看方法清单;`_stream_impl` 的各阶段已抽成 `_refresh_recall`/`_refresh_skills`/`_assemble_context`/`_run_clarification_gate`/`_handle_tools`。
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

- **`loop.py` 仍偏大但已显著拆分**(见 §4.7)。`_stream_impl` 从 1448 → ~875 行,公开类型/纯函数/循环状态已分出 `loop_types.py`/`loop_helpers.py`/`loop_context.py`,各阶段抽成 `(self, ctx)` 方法。剩余的 token-streaming/DELEGATE/stop-check 部分与循环控制紧耦合,暂留 `_stream_impl`。
- **`context_budget_v2.py` 跟 `context_budget.py` 共存 —— 设计如此,非待迁移**(见 §4.8)。v1=legacy 默认策略,v2=可选 priority 策略,由 `cfg.context_budget.strategy` 选择,经统一入口 `_assemble_context`。两者都不删。
- **StopPolicy 已感知 user cancel** —— `StopOutcome.USER_CANCELLED` 已加;前端 Stop 按钮 abort fetch → 后端 SSE 捕获 cancel → 取消 executor + 保留部分回答(跳过 durable memory 写入)。
- **PolicyEngine 不缓存** —— 每次 classify 都调 LLM。如果同一 query 反复 classify,加 LRU 显著省 LLM call。
