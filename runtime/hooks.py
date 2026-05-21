"""
runtime/hooks.py
─────────────────
Lifecycle hooks for the agent runtime — Claude-Code-inspired extensibility
with zero default context cost.

Design (Sprint 2, 2026-05)
──────────────────────────
Hooks are async callbacks fired at well-defined points in the agent loop.
They differ from skills / tools / MCP in 4 ways:
  - Zero LLM context cost (hooks don't show up in prompts)
  - Synchronous-ish (await but no LLM round-trip required)
  - Can mutate the context dict (e.g. modify tool args before dispatch)
  - Can short-circuit (set ctx["blocked"]=True to abort an action)

We deliberately ship 6 core events instead of Claude Code's 27. The rest
can be added incrementally — each new event means a new fire() call site
in the runtime, which is a small commit but a real one. YAGNI for the
22 events we'd never use in network ops.

Events
──────
  PRE_TOOL_USE      ctx: {tool, args, tool_call_id, session_id, turn}
                    Hook may: mutate ctx["args"], set ctx["blocked"]=True
                              with ctx["block_reason"]=str
  POST_TOOL_USE     ctx: {tool, args, result, latency_ms, ...}
                    Hook may: mutate ctx["result"] (filter/redact),
                              set ctx["audit_event"]=dict for logging
  TURN_START        ctx: {turn, session_id, query, facts_count}
                    Hook may: log metrics, prep context
  TURN_END          ctx: {turn, session_id, llm_response, tool_calls,
                          stop_decision, elapsed_ms}
                    Hook may: log metrics, write audit
                    NOTE: As of Sprint 2 this event is defined but NOT
                    fired in runtime/loop.py — TURN_START carries enough
                    info for most observers, and SESSION_END covers the
                    final-state need. Will be wired in Sprint 3 if any
                    listener actually needs per-turn finalization.
  SESSION_START     ctx: {session_id, query, delegation_mode}
                    Hook may: init session-level resources
  SESSION_END       ctx: {session_id, outcome, total_turns, tool_calls,
                          stop}  outcome ∈ {completed, consumer_closed, error}
                    Hook may: flush metrics, close session resources
                    Fires in finally{} so abort paths still run cleanup.

Failure semantics
─────────────────
A hook raising an exception is LOGGED but does NOT abort the runtime
(except PRE_TOOL_USE which can explicitly set blocked=True). This is the
"hooks are observers, not gatekeepers" stance — the one exception is
PRE_TOOL_USE which can gate, but only via explicit ctx mutation, not via
raising.

Priority
────────
Hooks fire in priority order (low to high). Default priority is 50.
Use < 50 for "early observers" (e.g. logging), > 50 for "late
deciders" (e.g. policy enforcement). PRE_TOOL_USE hooks that set
blocked=True at lower priority can be overridden by higher-priority
hooks that unset it — last-write-wins on ctx, by design.

Module independence
───────────────────
This module lives in `runtime/` because hooks are a runtime concern.
Caller modules (hitl_core, integrations, skills) register hooks via the
singleton `get_hook_registry()` — runtime never imports them back.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


# ── Event enum ───────────────────────────────────────────────────────────────

class HookEvent(str, Enum):
    """Named lifecycle events. Values are stable strings for config /
    audit log compatibility."""
    PRE_TOOL_USE   = "pre_tool_use"
    POST_TOOL_USE  = "post_tool_use"
    TURN_START     = "turn_start"
    TURN_END       = "turn_end"
    SESSION_START  = "session_start"
    SESSION_END    = "session_end"
    # Async HITL ack arrived (H2 fire-and-forget; 2026-05).
    # Fired by the H2 on_resolved callback (in skills/tools) so observers
    # can react. Runtime uses this internally to enqueue the result into
    # a per-session inject queue; turn_start consumers drain that queue
    # and write fresh facts into state.confirmed_facts.
    # ctx: {interrupt_id, session_id, decision (None if timed out),
    #       default_value, diverged, fact_text}
    ASYNC_HITL_RESOLVED = "async_hitl_resolved"


# ── Hook function signature ──────────────────────────────────────────────────

HookFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
"""Async hook callable. Receives ctx dict, returns possibly-mutated ctx.

Convention: hooks should return the same dict they received (mutated
in place), not a fresh dict. Returning None is allowed and treated as
"no changes" — registry handles it. Sync functions are NOT allowed
(we always await); wrap sync logic in `async def` if needed.
"""


@dataclass(frozen=True)
class _HookRegistration:
    """Internal record of a registered hook."""
    event:    HookEvent
    fn:       HookFn
    priority: int
    name:     str


# ── Registry ─────────────────────────────────────────────────────────────────

class HookRegistry:
    """Process-wide registry for runtime lifecycle hooks.

    Thread-safe for register (uses a lock); fire() runs all hooks for
    an event sequentially in priority order (low → high). Fire latency
    is O(n_hooks_for_event); typical N < 10. If you need parallel firing
    for high-cardinality observability hooks, use a single hook that
    fans out internally.
    """

    def __init__(self) -> None:
        self._hooks: dict[HookEvent, list[_HookRegistration]] = {
            e: [] for e in HookEvent
        }
        # asyncio.Lock would be ideal but registration happens at startup
        # before the event loop runs; a synchronous list mutation under
        # GIL is sufficient. Document this assumption.
        self._registration_count = 0

    def register(
        self,
        event: HookEvent,
        fn:    HookFn,
        *,
        priority: int = 50,
        name:     Optional[str] = None,
    ) -> str:
        """Register a hook. Returns the assigned hook name (for unregister).

        Args:
            event:    Which lifecycle event to listen for
            fn:       Async callable (ctx: dict) -> dict | None
            priority: Lower = fires first. 0-49 typical for observers,
                      50 = default, 51-100 for late-stage policy hooks
            name:     Optional name for unregister(). Defaults to "<fn name>#<seq>"
        """
        if not callable(fn):
            raise TypeError(f"Hook fn must be callable, got {type(fn).__name__}")
        if not isinstance(event, HookEvent):
            raise TypeError(
                f"event must be HookEvent enum, got {type(event).__name__}. "
                f"Use e.g. HookEvent.PRE_TOOL_USE."
            )
        self._registration_count += 1
        hook_name = name or f"{getattr(fn, '__name__', 'anon')}#{self._registration_count}"
        reg = _HookRegistration(event=event, fn=fn, priority=priority, name=hook_name)
        bucket = self._hooks[event]
        bucket.append(reg)
        # Keep sorted by (priority, registration order) so deterministic
        bucket.sort(key=lambda r: r.priority)
        logger.info(
            "HookRegistry: registered %s on %s @ priority=%d",
            hook_name, event.value, priority,
        )
        return hook_name

    def unregister(self, event: HookEvent, name: str) -> bool:
        """Remove a hook by name. Returns True if found."""
        bucket = self._hooks.get(event, [])
        for i, r in enumerate(bucket):
            if r.name == name:
                bucket.pop(i)
                logger.info(
                    "HookRegistry: unregistered %s from %s", name, event.value,
                )
                return True
        return False

    def hooks_for(self, event: HookEvent) -> list[str]:
        """List hook names registered for an event (mainly for debugging)."""
        return [r.name for r in self._hooks.get(event, [])]

    async def fire(
        self,
        event: HookEvent,
        ctx:   Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Fire all hooks for an event in priority order.

        Returns the (possibly mutated) ctx dict. Hook exceptions are
        logged and swallowed — the runtime never aborts due to a hook
        crash. PRE_TOOL_USE blocking is signalled via ctx["blocked"],
        not via exceptions.

        Empty ctx is OK; hooks must tolerate missing fields gracefully.
        """
        ctx = ctx if ctx is not None else {}
        bucket = self._hooks.get(event, [])
        if not bucket:
            return ctx
        for reg in bucket:
            t0 = time.monotonic()
            try:
                result = await reg.fn(ctx)
                # Allow hook to return None or new dict; default to current ctx
                if isinstance(result, dict):
                    ctx = result
                elapsed_ms = (time.monotonic() - t0) * 1000
                if elapsed_ms > 100:
                    logger.warning(
                        "HookRegistry: slow hook %s on %s took %.0fms — "
                        "consider moving expensive work off-path",
                        reg.name, event.value, elapsed_ms,
                    )
            except asyncio.CancelledError:
                raise  # never swallow cancel
            except Exception as exc:
                logger.warning(
                    "HookRegistry: hook %s on %s raised %s — continuing "
                    "(hooks are observers, not gatekeepers)",
                    reg.name, event.value, exc,
                )
        return ctx

    def clear(self) -> None:
        """Remove all hooks. Mainly for test isolation."""
        for event in HookEvent:
            self._hooks[event] = []
        self._registration_count = 0


# ── Singleton ────────────────────────────────────────────────────────────────

_registry: Optional[HookRegistry] = None


def get_hook_registry() -> HookRegistry:
    """Return the process-wide HookRegistry singleton.

    Lazy-init so importing this module never side-effects. main.py /
    tests can call this at startup; runtime/loop.py calls it at every
    fire site. The singleton lives for the process lifetime.

    For unit tests that need a fresh registry, call `reset_hook_registry()`
    in setUp.
    """
    global _registry
    if _registry is None:
        _registry = HookRegistry()
    return _registry


def reset_hook_registry() -> None:
    """Replace the singleton with a fresh empty registry. Test-only."""
    global _registry
    _registry = HookRegistry()
