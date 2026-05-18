# Phase 7B Refactor — Single-Path HITL via Watch-list

This refactor deletes the `pre_verify` policy gate and the SIMPLE/COMPLEX
backend dispatch, leaving a single clean path:

```
classify (lightweight) → executor.execute_query (wraps loop)
                              ↓
                          loop.stream
                              ↓
                          LLM produces tool_call
                              ↓
                          watch-list (cfg.tools.hitl_tool_names)
                          intercepts BEFORE tool execution
                              ↓
                          HitlExecutor raises hitl_core interrupt
                              ↓
                          /hitl/pending +1
                              ↓
                          operator approves/rejects/edits
                              ↓
                          tool runs (or doesn't)
```

Destructive-action detection happens at exactly **one point**: the LLM
proposes a tool_call whose name is in `cfg.tools.hitl_tool_names`. The LLM
sees full context (recall, tool outputs, confirmed_facts) and produces
concrete `tool_args`. Operators review the proposed parameters, not a
hand-waved "you said 修复" string match.

## What was deleted

### `runtime/loop.py`
- The 71-line `pre_verify` block in `stream()` that ran PolicyEngine
  before letting the LLM work, and emitted a synthesized `stop_hitl`
  chunk on failure with destructive-verb mapping fallback (`修复` →
  `edit_device_config`, `重启` → `restart_service`, etc).
- The `pre_verify` call in `run()`.
- The `_pre_verified` flag in both `run()` and `stream()`.
- The `skip_pre_verify` parameter from `stream()`.
- The `skip_pre_verify` check in the clarification gate.
- The `pre_verify` function itself is **kept** as `DEPRECATED` for
  backwards compatibility (external callers may still reference it).

### `webui/backend.py`
- The `pre_verify` SSE event emit (`{"type":"pre_verify",...}`).
- The whole `if decision.complexity.value == "complex" and executor: ...`
  block (~100 lines of A2A EventQueue plumbing).
- The whole `else: # SIMPLE path` block (~120 lines of `loop.stream`
  consumption + nested HITL fallback path forwarding stop_hitl chunks
  to executor with `force_hitl_tool` metadata).

Replaced by a single ~80-line `executor.execute_query` invocation that
drains chunks via an `on_chunk` callback into an `asyncio.Queue` and
streams them as SSE.

### `integrations/hitl_executor.py`
- The `force_hitl_tool` fast path in `execute()` that bypassed the agent
  loop when backend handed in pre-detected destructive metadata.
- The `skip_pre_verify=True` in `execute_query`'s `loop.stream()` call.
- The `_required_for_action` "missing required" UX hint in
  `_raise_tool_hitl` (no longer needed — LLM produces complete
  `tool_args` so operators see real proposed parameters).

### `runtime/policy_engine.py`
- The `if policy_name == "preverify_safe_to_proceed":` fallback handler.

### `config.yaml`
- The whole `preverify_safe_to_proceed` policy definition (15 lines).

### `config.py`
- `"preverify_safe_to_proceed"` removed from the `_required` and
  `required_policies` validation sets.

## What was added

### `integrations/hitl_executor.py`
- New `on_chunk: Optional[Callable[[dict], Awaitable[None]]]` parameter
  on `execute_query`. Called for every raw chunk from
  `runtime/loop.stream` so `webui/backend` can stream chunks to SSE
  without re-implementing chunk handling.

### `webui/backend.py`
- `_chunk_queue` + `_on_chunk` mechanism that bridges `executor.execute_query`'s
  callbacks into the SSE generator.

## Trade-offs

The refactor accepts these trade-offs:

1. **Two LLM calls become one.** Pre-verify ran a 1-2s policy LLM check
   before the agent could even start thinking. Now classify is the only
   pre-loop LLM call (it's small + cached). Net latency reduction:
   ~25-30 seconds on destructive queries (your logs showed pre_verify
   taking 27s on `qwen3.5:27b`).

2. **No belt-and-suspenders.** Previously, if the LLM ignored the
   watch-list (it shouldn't, since the watch-list is a runtime
   interception, not a prompt instruction), pre_verify offered a second
   line of defence on the raw query. Now there's one line of defence.
   This is fine because the watch-list interception happens in
   `runtime/loop.py:_needs_hitl` AFTER the LLM produces a tool_call —
   the LLM cannot bypass it.

3. **All queries route through `HitlExecutor`.** Read-only diagnostics
   (`查询 ap-01 配置`) now go through the executor too, not just
   destructive ones. The executor adds ~2ms of overhead (coreference +
   recall pre-injection); negligible relative to LLM call costs.

4. **`force_hitl_tool` metadata handshake removed.** Backend used to
   detect `stop_hitl` chunks itself, then re-call `executor.execute()`
   with `force_hitl_tool` metadata. Now the executor itself sees the
   `stop_hitl` chunk inside its own `execute_query` and raises the
   interrupt directly. Cleaner, no double-pass.

## Future work — Phase 7C

When ready:

1. Delete `hitl/` directory (the legacy LangGraph backend).
2. Remove `langgraph`, `langchain-core`, `langchain-openai` from
   `requirements.txt`.
3. Delete the `pre_verify` function entirely from `runtime/loop.py`
   once external callers have migrated.
4. Delete the `enable_pre_verification` config field.

## Future work — Phase 8 (batch waiting list)

The `BatchCoordinator` (in `hitl_core/batch.py`) is wired and tested
but not yet driven by the agent loop. When the LLM produces a
multi-step plan (e.g. "fix RADIUS on ap-01, ap-02, ap-03"), each
destructive step would enter a shared batch waiting list. Operators
would see a single batch-approval card listing all proposed actions
with their dependencies. Approve/reject can be:

- `BEST_EFFORT` — independent steps proceed as approved
- `ALL_OR_NOTHING` — anyone rejected blocks the entire batch

This is the "waiting list 统一审批" idea you suggested. The plumbing
is in place; only the loop integration is missing.
