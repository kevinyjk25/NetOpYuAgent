"""
integrations.hitl_executor — IT-ops domain glue between A2A protocol
and hitl_core.

This module replaces the relevant parts of the old hitl/a2a_integration.py
with a much smaller surface that delegates HITL plumbing to hitl_core.
The executor is responsible for:

  • Translating A2A RequestContext → PipelineState
  • Running the agent loop and exposing its events back to the A2A
    transport (event_queue + status updates)
  • Triggering single + multi-mode + nested HITL through the hitl_core
    primitives (request_approval / request_batch_approval)
  • Resuming after operator decisions, with all the prior bug fixes
    preserved:
      - code-level coreference resolution (_infer_focus_device → coref)
      - editable_param_keys auto-derivation per tool type
      - chat-turn open-ended clarification (no card UI)
      - HITL APPROVED & COMPLETED post-turn writeback
      - in-flight guard against double-clicks (now in hitl_core router)
      - stale placeholder filter (still owned by recall_orchestrator)

This file deliberately stays under ~700 lines. Anything bigger is a
sign that domain logic should move into a separate module (a "skill"
or "policy") and be plugged in here as a callback.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional

from hitl_core import (
    AuditEventKind,
    AuditLogger,
    BatchPolicy,
    BatchSubmission,
    BatchWaitMode,
    CheckpointEntry,
    ChoiceOption,
    ClarificationField,
    Coreferencer,
    DecisionKind,
    HitlDecision,
    HitlPayload,
    HitlPipeline,
    HitlRouter,
    InMemoryAuditSink,
    InMemoryCheckpointStore,
    InterruptState,
    PipelineAborted,
    PipelineContext,
    PipelineState,
    ProposedAction,
    RiskLevel,
    TriggerKind,
    build_default_device_coreferencer,
    build_neutral_coreferencer,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool-type → editable param keys (preserves the auto-derive logic from
# the previous round; lets us avoid forcing every caller to spell out keys)
# ---------------------------------------------------------------------------

_EDITABLE_PARAM_BY_TOOL = {
    "edit_device_config": ["config_lines", "reason", "section"],
    "push_config":         ["config_lines", "device_id", "reason"],
    "restart_service":     ["service", "scope", "reason"],
    "rollback_deploy":     ["target_version", "reason"],
    "drain_node":          ["node_id", "drain_timeout_seconds"],
    "failover":            ["primary", "secondary", "reason"],
    "delete_resource":     ["resource_id", "reason"],
}


def _derive_editable_keys(tool_name: str, tool_args: dict[str, Any]) -> list[str]:
    """Pick which keys from a tool's args are operator-editable.

    First tries the curated table by tool name, filtered to keys the
    LLM actually populated in tool_args (so operators edit real
    fields, not stale placeholders).

    Fallback: when no curated table entry exists, expose any non-trivial
    keys already present in tool_args (excluding device_id, which the
    operator should not change post-coreference).
    """
    table_keys = _EDITABLE_PARAM_BY_TOOL.get(tool_name, [])
    if table_keys:
        return [k for k in table_keys if k in tool_args]
    return [
        k for k, v in (tool_args or {}).items()
        if k != "device_id" and isinstance(v, (str, int, float, list, dict))
    ]


def _extract_delegation_provenance(
    env_context: Optional[dict[str, Any]],
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Pull delegation provenance fields from env_context.

    Returns (source_agent, source_session_id, source_query). All three None
    when this query was not delegated from a peer (the normal case for a
    user-initiated request). When non-None, peer-side HITL cards display
    a "Delegated from <agent>" banner so the operator knows who's upstream
    and what the original user asked.

    The fields originate in task/delegation.py's TaskDefinition.metadata,
    travel over A2A in params.metadata, and are placed in env_context by
    a2a/agent_executor.py before invoking the runtime loop.
    """
    if not env_context:
        return (None, None, None)
    return (
        env_context.get("source_agent")      or None,
        env_context.get("source_session_id") or None,
        env_context.get("source_query")      or None,
    )


# ---------------------------------------------------------------------------
# Destructive verb / clarification heuristics — kept here as IT-ops domain
# concerns. Generic versions live in hitl_core.triggers; this is the
# project-specific list.
# ---------------------------------------------------------------------------

# NOTE: Domain-specific verb lists and heuristic_missing_fields used to live
# here (mirrored from runtime/loop.py's deleted helper). Both are gone now —
# clarification gating is handled by PolicyEngine[assess_query_specificity]
# inside runtime/loop.py. If hitl_executor needs to surface a clarification
# (e.g. when an operator types "fix it" with no recall context), the loop's
# clarification gate fires before any HitlExecutor code runs.


# ---------------------------------------------------------------------------
# HitlExecutor — the public class
# ---------------------------------------------------------------------------

class HitlExecutor:
    """Glue between the A2A protocol layer and hitl_core.

    Replaces hitl/a2a_integration.py's HitlExecutor — same external
    surface (execute / cancel coroutines), much smaller body. All HITL
    plumbing (registration / validation / dispatch / audit) is delegated
    to hitl_core.

    Construction:

        executor = HitlExecutor(
            runtime_loop=patched_runtime_loop,
            llm_engine=llm_engine,
            tool_registry=tool_registry,
            memory_router=memory_router,
            hitl_router=router,    # hitl_core.HitlRouter
            hitl_pipeline=pipe,    # hitl_core.HitlPipeline (for nested HITL)
        )

    The runtime_loop and tool_registry are reused as-is from the rest
    of the project — no rewriting required.
    """

    def __init__(
        self,
        *,
        runtime_loop,
        llm_engine,
        tool_registry: Optional[dict[str, Callable]] = None,
        memory_router=None,
        hitl_router: HitlRouter,
        hitl_pipeline: Optional[HitlPipeline] = None,
        coreferencer: Optional[Coreferencer] = None,
        audit_logger: Optional[AuditLogger] = None,
        skill_evolver=None,
    ):
        self._runtime_loop  = runtime_loop      # patched AgentRuntimeLoop
        self._llm_engine    = llm_engine
        self._tool_registry = dict(tool_registry or {})
        self._memory        = memory_router
        self._router        = hitl_router
        self._pipeline      = hitl_pipeline
        # NOTE (L0/L1 Stage B): default to the DOMAIN-FREE neutral coreferencer.
        # A non-network agent thus gets no spurious device coreference. The
        # active profile (L1) injects build_default_device_coreferencer() for
        # network domains via the `coreferencer` arg (wired in main.py).
        self._coref         = coreferencer or build_neutral_coreferencer()
        self._audit         = audit_logger
        # SkillEvolver may not exist at executor-construction time (it's
        # built later in main.py once the LLM smoke test passes). Use
        # set_skill_evolver() to inject after construction. Used only by
        # the batch finalizer to evolve a skill ONCE per resolved batch
        # — see _batch_execute_after_resolution.
        self._skill_evolver = skill_evolver

        # Task store for recording INBOUND delegations (Phase 2B+, 2026-05).
        # When this executor handles an A2A request whose metadata carries
        # source_agent (set by the dispatching peer's task/delegation.py),
        # we write a TaskDefinition into task_store with metadata.direction
        # = "inbound" so the local Delegations tab can show "I'm processing
        # a request from <source_agent>". Out-of-band (deferred) wiring —
        # main.py calls set_task_store() after services["task_system"] is
        # constructed. Optional: if not wired, inbound recording silently
        # no-ops, so HitlExecutor stays buildable without a task_system.
        self._task_store = None
        # A2A Phase 3 (P3-b): optional peer registry for resolving a source
        # agent's callback URL when calling back after a delegated HITL is
        # resolved here. Injected by main.py via set_peer_registry().
        self._peer_registry = None

        # Register named resumers so the router can invoke us when
        # decisions arrive for interrupts whose pipeline isn't alive.
        # The "agent_loop_resumer" name is what producer code stamps
        # into ResumeHandle when it raises an interrupt.
        hitl_router.register_resumer(
            "agent_loop_resumer", self._agent_loop_resumer,
        )
        hitl_router.register_resumer(
            "tool_call_resumer",  self._tool_call_resumer,
        )

    def set_skill_evolver(self, evolver) -> None:
        """Inject the SkillEvolver after construction.

        Called from main.py once the evolver instance exists (it's built
        after the executor because evolver construction needs an LLM
        engine smoke test). Used by the batch finalizer; safe to leave
        unset if skill evolution is disabled.
        """
        self._skill_evolver = evolver

    def set_task_store(self, task_store) -> None:
        """Inject the project's TaskStore so inbound delegations get logged.

        Deferred-wiring counterpart to ``task_system.store``. Called from
        main.py after services["task_system"] is built. Optional — if
        unset, ``_record_inbound_delegation`` no-ops (existing behavior).
        Decoupled to keep HitlExecutor buildable from tests without a
        live task_system, and to avoid making it import the task module.
        """
        self._task_store = task_store

    def set_peer_registry(self, registry) -> None:
        """Inject the AgentRegistry so cross-agent HITL callbacks can resolve a
        source agent's base URL (A2A Phase 3 P3-b). Optional — if unset, the
        callback is skipped (the delegating side stays in AWAITING_PEER_HITL,
        which P3-d's SLA watchdog will eventually time out)."""
        self._peer_registry = registry

    async def _maybe_callback_source_agent(
        self, *, interrupt_id: str, result_text: str, decision: str,
    ) -> None:
        """If `interrupt_id` was a HITL raised while serving an inbound
        delegation, POST the resolution back to the originating agent's
        /hitl_resolved endpoint so it can resume (A2A Phase 3 mode B).

        No-op when: the interrupt wasn't cross-agent (no bridge record), or no
        peer registry is wired, or the source agent's URL can't be resolved.
        Failures are logged, never raised — a callback hiccup must not break
        the local operator's approval flow.
        """
        if not interrupt_id:
            return
        try:
            from task.inter.cross_agent_hitl import get_cross_agent_hitl_bridge
            rec = get_cross_agent_hitl_bridge().pop_inbound_hitl(interrupt_id)
        except Exception as exc:
            logger.debug("cross-agent callback: bridge lookup failed: %s", exc)
            return
        if rec is None:
            return   # not a cross-agent delegated HITL — nothing to call back
        if self._peer_registry is None:
            logger.warning(
                "cross-agent callback: no peer registry wired — cannot resume "
                "source=%s for interrupt=%s", rec.source_agent, interrupt_id[:12],
            )
            return
        # Resolve the source agent's base URL.
        try:
            agent = await self._peer_registry.get_agent(rec.source_agent)
            base_url = getattr(agent, "base_url", None) if agent else None
        except Exception as exc:
            logger.warning("cross-agent callback: registry resolve failed: %s", exc)
            return
        if not base_url:
            logger.warning(
                "cross-agent callback: no base_url for source=%s", rec.source_agent,
            )
            return
        callback_url = base_url.rstrip("/") + "/hitl_resolved"
        self_agent_id = getattr(self, "_self_agent_id", None) or "this-agent"
        body = {
            "peer_agent":     self_agent_id,   # who is calling back (this agent)
            "interrupt_id":   interrupt_id,
            "result":         result_text,
            "decision":       decision,
            "correlation_id": rec.correlation_id,
            "source_session_id": rec.source_session_id,
        }
        try:
            import httpx
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(callback_url, json=body)
            logger.info(
                "cross-agent callback: POST %s → %s (interrupt=%s decision=%s)",
                callback_url, resp.status_code, interrupt_id[:12], decision,
            )
        except Exception as exc:
            logger.warning(
                "cross-agent callback: POST to %s failed: %s — source agent will "
                "stay AWAITING_PEER_HITL until its SLA watchdog fires (P3-d)",
                callback_url, exc,
            )

    # NOTE: trust_mode lives in PolicyEngine, not in HitlExecutor —
    # runtime/loop.py decides whether to skip HITL gating, and runtime
    # cannot import HitlExecutor (reverse-layer dependency). Both
    # set_trust_mode() and classify_action_type() are on PolicyEngine so
    # runtime can call them without breaking module independence audits.

    # ────────────────────────────────────────────────────────────────
    # Public entry point — called by A2A protocol layer
    # ────────────────────────────────────────────────────────────────

    async def execute_query(
        self,
        *,
        query: str,
        session_id: str,
        confirmed_facts: Optional[list[str]] = None,
        env_context: Optional[dict[str, Any]] = None,
        on_token: Optional[Callable[[str], Awaitable[None]]] = None,
        on_chunk: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> dict[str, Any]:
        """Run the full agent loop with HITL.

        Returns a dict shaped for the protocol layer to forward to
        the operator UI:
          {"text": <agent's final answer>,
           "interrupted": <bool>,           # True if an interrupt is pending
           "interrupt_id": <str|None>,
           "decisions": [...]}              # operator decisions in order

        When `interrupted=True`, the caller must wait for the operator
        to deliver a HitlDecision via HitlRouter.deliver(); the agent
        loop then resumes via the named resumer.

        Hooks:
          on_token(str)  — called for each LLM-generated token (streaming UI)
          on_chunk(dict) — called for EVERY raw chunk from runtime/loop.stream
                           (steps, recall events, tool calls, etc) before
                           the executor inspects it for HITL signals. Lets
                           callers (webui/backend) forward to SSE without
                           re-implementing chunk handling.
        """
        # Sprint-3-pre: top-level span for the whole query execution.
        # Child spans (llm.call, tool.dispatch) hang off this. Use
        # session_id as a high-cardinality attribute for filtering;
        # Sprint 3 will derive a deterministic trace_id from it so
        # the UI can deep-link to traces.
        from runtime.tracing import start_span as _start_span
        confirmed_facts_list = list(confirmed_facts or [])
        self._query_span_cm = _start_span(
            "agent.query",
            **{
                "agent.session_id": session_id,
                "agent.query.chars": len(query or ""),
                "agent.facts.count": len(confirmed_facts_list),
            },
        )
        self._query_span = self._query_span_cm.__enter__()
        try:
            return await self._execute_query_inner(
                query=query, session_id=session_id,
                confirmed_facts=confirmed_facts_list, env_context=env_context,
                on_token=on_token, on_chunk=on_chunk,
            )
        except BaseException as _exc:
            try:
                self._query_span.record_exception(_exc)
            except Exception:
                pass
            raise
        finally:
            try:
                self._query_span_cm.__exit__(None, None, None)
            except Exception:
                pass

    async def _execute_query_inner(
        self,
        *,
        query: str,
        session_id: str,
        confirmed_facts: list[str],
        env_context: Optional[dict[str, Any]] = None,
        on_token: Optional[Callable[[str], Awaitable[None]]] = None,
        on_chunk: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> dict[str, Any]:
        """Internal body of execute_query — wrapped by execute_query() for tracing."""
        env_context = dict(env_context or {})
        confirmed_facts = list(confirmed_facts or [])

        # Pre-recall memory if available — same as the old code path.
        # We attach it to env_context so the agent loop + LLM see it.
        recall_text = ""
        if self._memory and query:
            try:
                rec = await self._memory.recall(query, session_id)
                recall_text = getattr(rec, "prompt_context", "") or ""
                env_context["_fts_context"] = recall_text
            except Exception as exc:
                logger.debug("recall skipped: %s", exc)

        env_context["_confirmed_facts"] = confirmed_facts

        # Code-level coreference (preserves Phase-12 fix). Inject
        # [RESOLVED FROM CONTEXT] line so LLMs that don't infer well
        # still get the right device_id.
        focus = self._coref.infer(
            query=query,
            context_strings=[recall_text] + [str(f) for f in confirmed_facts],
        )
        if focus.entity:
            logger.info(
                "Coreference: query did not name an entity; recall shows %r "
                "as focus. Binding query to that entity (source=%s).",
                focus.entity, focus.source,
            )
            query_for_loop = (
                f"{query}\n\n"
                f"[RESOLVED FROM CONTEXT] target_device = {focus.entity}\n"
                f"(The user did not name a device but the most recent "
                f"tool call in this session was on `{focus.entity}`. "
                f"Use this device unless the operator explicitly says otherwise.)"
            )
            # Stamp env_context so downstream consumers (LLM prompt
            # builder, tool routers) can pull the resolved focus device
            # without re-running coreference.
            env_context["_resolved_focus_device"] = focus.entity
        else:
            query_for_loop = query

        # Run the agent loop, watching for stop_hitl chunks. Each chunk
        # may either:
        #   (a) be a plain token / step event → forward
        #   (b) carry stop_hitl=True → pause and raise HITL via hitl_core
        if self._runtime_loop is None:
            raise RuntimeError(
                "HitlExecutor.execute_query called but _runtime_loop is None. "
                "Make sure main.py's lifespan injects services['runtime_loop'] "
                "before the first request."
            )

        full_text  = ""
        interrupt_id: Optional[str] = None
        decisions:  list[HitlDecision] = []

        async for chunk in self._runtime_loop.stream(
            query=query_for_loop,
            session_id=session_id,
            env_context=env_context,
            confirmed_facts=confirmed_facts,
            tool_registry=self._tool_registry,
        ):
            # Forward EVERY chunk to the backend before our HITL inspection,
            # so callers can stream tokens, step events, recall events, etc.
            # to the UI without us replicating chunk handling here.
            if on_chunk is not None:
                try:
                    await on_chunk(chunk)
                except Exception as exc:
                    logger.debug("on_chunk hook raised: %s", exc)

            tok = chunk.get("token")
            if tok:
                full_text += tok
                if on_token:
                    await on_token(tok)

            # Open-ended clarification surfaced by stream as chat tokens
            # (no card). Pre-existing flow: just stream tokens, return.
            # The runtime loop already does this — we don't need to
            # intercept.

            # USER_CHOICE / CLARIFICATION cards triggered from the loop
            if chunk.get("hitl_kind") in ("user_choice", "clarification"):
                interrupt_id = await self._raise_multi_mode(
                    kind=chunk["hitl_kind"],
                    chunk=chunk, query=query, session_id=session_id,
                    env_context=env_context,
                )
                return {
                    "text":          full_text,
                    "interrupted":   True,
                    "interrupt_id":  interrupt_id,
                    "decisions":     decisions,
                }

            # Single-tool destructive HITL — stream wants approval
            # before invoking a HITL-gated tool. If the chunk carries a
            # `batch_calls` list (LLM emitted multiple same-name
            # destructive calls in one turn — e.g. push_config to N
            # devices), raise a batch interrupt so all N children appear
            # at once instead of sequentially after each approve.
            if chunk.get("stop_hitl") and chunk.get("tool_name"):
                _batch_calls = chunk.get("batch_calls")
                if _batch_calls and len(_batch_calls) > 1:
                    batch_id = await self._raise_tool_hitl_batch(
                        tool_name=chunk["tool_name"],
                        calls=_batch_calls,
                        query=query, session_id=session_id,
                        confirmed_facts=confirmed_facts,
                        chunk=chunk,
                        env_context=env_context,
                    )
                    # The batch is the "interrupt id" surfaced to caller —
                    # GET /hitl/batch/{id} lists children for the UI.
                    return {
                        "text":          full_text,
                        "interrupted":   True,
                        "interrupt_id":  batch_id,
                        "batch":         True,
                        "decisions":     decisions,
                    }

                interrupt_id = await self._raise_tool_hitl(
                    tool_name=chunk["tool_name"],
                    tool_args=chunk.get("tool_args") or {},
                    query=query, session_id=session_id,
                    confirmed_facts=confirmed_facts,
                    chunk=chunk,
                    env_context=env_context,
                )
                return {
                    "text":          full_text,
                    "interrupted":   True,
                    "interrupt_id":  interrupt_id,
                    "decisions":     decisions,
                }

        # Loop completed without interrupt — write back to memory
        await self._writeback(query, full_text, session_id)
        return {
            "text":          full_text,
            "interrupted":   False,
            "interrupt_id":  None,
            "decisions":     decisions,
        }

    # ────────────────────────────────────────────────────────────────
    # Single-tool HITL (destructive operation gating)
    # ────────────────────────────────────────────────────────────────

    async def _raise_tool_hitl(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        query: str,
        session_id: str,
        confirmed_facts: list[str],
        chunk: dict,
        env_context: Optional[dict[str, Any]] = None,
    ) -> str:
        """Raise a destructive-tool-call HITL via hitl_core.

        Builds a HitlPayload with the tool's actual args populated +
        editable_param_keys auto-derived. Registers a tool_call_resumer
        ResumeHandle so the actual tool execution happens on operator
        approval (with parameter_patch merged).

        When env_context carries source_agent / source_session_id /
        source_query (set by a2a/agent_executor.py from inbound A2A
        metadata), the card is tagged as a peer-delegated HITL so the
        operator UI can show "Delegated from <agent>" provenance.
        """
        editable_keys = chunk.get("editable_param_keys") or _derive_editable_keys(
            tool_name, tool_args,
        )

        _target = str(tool_args.get("device_id") or tool_args.get("target") or "-")
        _intent = f"Agent proposes calling `{tool_name}` on `{_target}`"

        _src_agent, _src_session, _src_query = _extract_delegation_provenance(
            env_context,
        )

        payload = HitlPayload(
            thread_id=session_id,
            context_id=session_id,
            user_query=query,
            trigger_kind=TriggerKind.DESTRUCTIVE,
            risk_level=RiskLevel.HIGH,
            intent_summary=_intent,
            confidence_score=float(chunk.get("confidence", 0.9)),
            proposed_action=ProposedAction(
                action_type=f"tool_call:{tool_name}",
                target=_target,
                parameters=dict(tool_args),
                reversible=False,
                risk_level=RiskLevel.HIGH,
            ),
            editable_param_keys=editable_keys,
            source_agent      = _src_agent,
            source_session_id = _src_session,
            source_query      = _src_query,
        )
        from hitl_core.schema import ResumeHandle
        entry = CheckpointEntry(
            interrupt_id=payload.interrupt_id,
            payload=payload,
            resume_handle=ResumeHandle(
                resumer_name="tool_call_resumer",
                state={
                    "tool_name":       tool_name,
                    "tool_args":       tool_args,
                    "session_id":      session_id,
                    "original_query":  query,
                    "confirmed_facts": confirmed_facts,
                },
            ),
        )
        await self._router.register_payload(entry)
        # A2A Phase 3 (P3-b): if this HITL was raised while serving an inbound
        # delegation (provenance present), record where to call back when the
        # operator here resolves it — so dc can POST the result to lan's
        # /hitl_resolved and lan can resume (mode B).
        if _src_agent and _src_session:
            try:
                from task.inter.cross_agent_hitl import get_cross_agent_hitl_bridge
                get_cross_agent_hitl_bridge().record_inbound_hitl(
                    interrupt_id=payload.interrupt_id,
                    source_agent=_src_agent,
                    source_session_id=_src_session,
                    source_query=_src_query or "",
                )
            except Exception as _xa_exc:
                logger.debug("cross-agent inbound record skipped: %s", _xa_exc)
        logger.info(
            "Tool HITL raised: id=%s tool=%s target=%s editable=%s",
            payload.interrupt_id[:12], tool_name, _target, editable_keys,
        )
        return payload.interrupt_id

    # ────────────────────────────────────────────────────────────────
    # Batch tool HITL — same shape as _raise_tool_hitl but for N siblings
    # ────────────────────────────────────────────────────────────────

    async def _raise_tool_hitl_batch(
        self,
        *,
        tool_name: str,
        calls: list[tuple[str, dict[str, Any]]],
        query: str,
        session_id: str,
        confirmed_facts: list[str],
        chunk: dict,
        env_context: Optional[dict[str, Any]] = None,
    ) -> str:
        """Raise a batch of N destructive-tool HITL interrupts that share
        a single batch_id, so the UI shows N pending cards at once and the
        operator can approve/reject each (or use POST /hitl/batch/{id}/...
        to act on the whole group).

        Same per-child shape as _raise_tool_hitl — each child has a
        ResumeHandle pointing at tool_call_resumer with that child's
        specific args. On approval, each child runs independently;
        tool_call_resumer's existing chunk_queue plumbing streams trace
        per child.

        Returns the batch_id (not an interrupt_id) so the caller knows
        to surface the batch endpoints instead of single-interrupt ones.
        """
        from hitl_core.schema import (
            HitlBatch, BatchPolicy, BatchWaitMode, ResumeHandle,
        )

        # First materialise all child CheckpointEntries.
        #
        # editable_keys per child is auto-derived from the tool name + args
        # using the same _derive_editable_keys helper that the singular
        # _raise_tool_hitl uses. Earlier versions of this method referenced
        # a `self._cfg.editable_hitl_tools` attribute that HitlExecutor
        # doesn't actually carry — that path raised AttributeError on every
        # batch HITL ('HitlExecutor' object has no attribute '_cfg'),
        # which silently broke multi-target destructive ops.
        # Provenance is the same for every child in the batch (same parent
        # delegation). Extract once outside the loop.
        _src_agent, _src_session, _src_query = _extract_delegation_provenance(
            env_context,
        )

        child_entries: list[CheckpointEntry] = []
        for (n, a) in calls:
            _editable = _derive_editable_keys(n, a)
            _target = str(a.get("device_id") or a.get("target") or "-")
            _intent = f"Agent proposes calling `{n}` on `{_target}` (batch)"

            child_payload = HitlPayload(
                thread_id=session_id,
                context_id=session_id,
                user_query=query,
                trigger_kind=TriggerKind.DESTRUCTIVE,
                risk_level=RiskLevel.HIGH,
                intent_summary=_intent,
                confidence_score=float(chunk.get("confidence", 0.9)),
                proposed_action=ProposedAction(
                    action_type=f"tool_call:{n}",
                    target=_target,
                    parameters=dict(a),
                    reversible=False,
                    risk_level=RiskLevel.HIGH,
                ),
                editable_param_keys=_editable,
                source_agent      = _src_agent,
                source_session_id = _src_session,
                source_query      = _src_query,
            )
            child_entries.append(CheckpointEntry(
                interrupt_id=child_payload.interrupt_id,
                payload=child_payload,
                resume_handle=ResumeHandle(
                    resumer_name="tool_call_resumer",
                    state={
                        "tool_name":       n,
                        "tool_args":       a,
                        "session_id":      session_id,
                        "original_query":  query,
                        "confirmed_facts": confirmed_facts,
                    },
                ),
            ))

        # Build the batch wrapper. BEST_EFFORT lets individual children
        # succeed/fail independently — better fit for IT-ops where one
        # device may be unreachable but the others should still go.
        # ALL means the producer-side future resolves only after every
        # child has a decision (approve/reject/timeout), which is what
        # the agent loop needs to assemble a complete batch report.
        batch = HitlBatch(
            interrupt_ids=[e.interrupt_id for e in child_entries],
            policy=BatchPolicy.BEST_EFFORT,
            wait_mode=BatchWaitMode.ALL,
        )

        # Stamp batch_id onto each child BEFORE persisting (so the store
        # and the UI can look up siblings via context_snapshot["batch_id"]).
        from hitl_core.batch import BATCH_ID_KEY
        for e in child_entries:
            e.payload.context_snapshot[BATCH_ID_KEY] = batch.batch_id

        # Persist children first
        for e in child_entries:
            await self._router.register_payload(e)

        # Open the batch via the coordinator and CAPTURE the future.
        # Earlier versions threw the future away — the design is that
        # the producer awaits it to receive a BatchResolution once
        # wait_mode is satisfied. The router's batch path correctly
        # short-circuits to `return None` for each child decision (so
        # the singular tool_call_resumer never runs), expecting THIS
        # task to be on the other end of the future executing tools.
        # Without it, batch HITL approve → silent no-op (visible in
        # the UI as "approved action is executing" forever).
        batch_future = await self._router.batch.open_batch(batch)
        # Collect child targets for logging — operators want to see at a
        # glance which devices were batched without spelunking the audit log.
        _targets = [
            str(a.get("device_id") or a.get("target") or "-")
            for (_, a) in calls
        ]
        logger.info(
            "Tool HITL batch raised: batch_id=%s tool=%s children=%d targets=%s",
            batch.batch_id[:12], tool_name, len(child_entries), _targets,
        )

        # Per-child (tool_name, args) snapshot so the background runner
        # can dispatch each tool after batch resolves. We can't read
        # them off the resolution's HitlDecision alone — those carry
        # parameter_patch (operator edits) but not the original args.
        _child_tool_specs: dict[str, tuple[str, dict, str]] = {
            e.interrupt_id: (
                e.resume_handle.state["tool_name"],
                dict(e.resume_handle.state["tool_args"]),
                e.resume_handle.state.get("session_id", ""),
            )
            for e in child_entries
        }

        # Background runner: awaits batch resolution, then dispatches
        # each approved child through the tool registry. Emits chunks
        # via chunk_queue so any /hitl/{id}/stream subscribed before
        # approval sees execution progress.
        #
        # Why a task rather than awaiting inline: matches the singular
        # HITL flow — execute_query returns the interrupt_id immediately
        # and the agent loop is free to terminate; tool execution runs
        # asynchronously after operator decisions arrive (potentially
        # minutes later). Inline await would block the producer for
        # the full HITL lifetime.
        import asyncio as _asyncio
        async def _batch_execute_after_resolution() -> None:
            try:
                resolution = await batch_future
            except Exception as exc:
                logger.exception(
                    "batch_execute: future await failed for %s: %s",
                    batch.batch_id[:12], exc,
                )
                return

            from hitl_core.chunk_queue import get_chunk_queue_registry
            from hitl_core.schema import DecisionKind
            _cq = get_chunk_queue_registry()

            logger.info(
                "batch_execute: batch=%s resolved — all_approved=%s rejected=%s",
                batch.batch_id[:12], resolution.all_approved, resolution.rejected,
            )

            # Memory writebacks run concurrently with subsequent tool
            # dispatches — previously they were awaited inline, which
            # serialized every child's execution behind a 30-60s LLM
            # fact-extraction call (LLM queue contended with HITL
            # classifiers). We collect tasks here and gather at the end.
            _pending_writebacks: list = []

            # Accumulate inputs for the per-batch SkillEvolver call.
            # SkillEvolver runs ONCE after the whole batch finishes
            # (vs. once-per-child inline, which used to wedge each
            # POST behind another 8s+ LLM call). Only successful
            # children contribute — rejects/failures aren't useful as
            # skill exemplars. See backend.py _submit_hitl_decision
            # for the singular-HITL counterpart.
            _evolver_tools: list[str] = []
            _evolver_results: list[str] = []
            _evolver_targets: list[str] = []

            def _schedule_writeback(user_q: str, asst_t: str, sid: str) -> None:
                """Schedule writeback as a background task with low
                importance (0.4) — chunk persisted, no LLM distillation."""
                try:
                    _pending_writebacks.append(
                        _asyncio.create_task(
                            self._writeback(user_q, asst_t, sid, importance=0.4)
                        )
                    )
                except Exception:
                    # If create_task fails (e.g. no running loop), fall
                    # back to silent skip — memory write is best-effort.
                    pass

            # Iterate in submission order (BatchResolution.decisions
            # preserves it via batch.interrupt_ids).
            for child_decision in resolution.decisions:
                iid = child_decision.interrupt_id
                spec = _child_tool_specs.get(iid)
                if spec is None:
                    logger.warning(
                        "batch_execute: no spec for child %s — skipping", iid[:12],
                    )
                    continue
                child_tool, child_args, child_sid = spec

                # Skip rejected children
                if child_decision.decision == DecisionKind.REJECT:
                    msg = f"[BATCH REJECTED — {child_tool} on {child_args.get('device_id','-')}]"
                    logger.info(
                        "batch_execute: child %s rejected — skipping tool",
                        iid[:12],
                    )
                    try:
                        _cq.push(iid, {
                            "node": "hitl_reject",
                            "node_step": f"Operator rejected: {child_tool}",
                        })
                    except Exception:
                        pass
                    _schedule_writeback(query, msg, child_sid)
                    continue

                # Merge operator edit (if any)
                if (child_decision.decision == DecisionKind.EDIT
                        and child_decision.parameter_patch):
                    child_args.update(child_decision.parameter_patch)
                    try:
                        _cq.push(iid, {
                            "node": "hitl_edit",
                            "node_step": (
                                f"Operator edited args: "
                                f"{list(child_decision.parameter_patch.keys())}"
                            ),
                        })
                    except Exception:
                        pass

                # Tool dispatch
                try:
                    _cq.push(iid, {
                        "node": "tool_call",
                        "node_step": f"Calling tool: {child_tool}",
                    })
                except Exception:
                    pass

                if child_tool not in self._tool_registry:
                    logger.warning(
                        "batch_execute: tool %r not in registry — skipping",
                        child_tool,
                    )
                    try:
                        _cq.push(iid, {
                            "node": "tool_error",
                            "node_step": f"Tool {child_tool!r} not registered",
                        })
                    finally:
                        _cq.complete(iid)
                    continue

                try:
                    logger.info(
                        "batch_execute: dispatching %s on %s (iid=%s)",
                        child_tool, child_args.get("device_id", "-"), iid[:12],
                    )
                    result = await self._tool_registry[child_tool](child_args)
                    result_text = str(result)[:1500]
                    logger.info(
                        "batch_execute: %s on %s returned %d chars (iid=%s)",
                        child_tool, child_args.get("device_id", "-"),
                        len(result_text), iid[:12],
                    )
                except Exception as exc:
                    logger.exception(
                        "batch_execute: tool %s on %s failed: %s",
                        child_tool, child_args.get("device_id", "-"), exc,
                    )
                    try:
                        _cq.push(iid, {
                            "node": "tool_error",
                            "node_step": f"Tool {child_tool} FAILED: {exc}",
                        })
                        # Emit batch_member_result for failure so chat UI
                        # gets a visible error bubble (parallel to success
                        # path). Frontend's openHitlStream picks this up.
                        _cq.push(iid, {
                            "type": "batch_member_result",
                            "tool":   child_tool,
                            "target": child_args.get("device_id", "-"),
                            "args":   child_args,
                            "error":  str(exc),
                            "batch_id": batch.batch_id,
                        })
                    finally:
                        _cq.complete(iid)
                    _schedule_writeback(
                        query,
                        f"[BATCH FAILED — {child_tool} on "
                        f"{child_args.get('device_id', '-')}] {exc}",
                        child_sid,
                    )
                    continue

                # Success — emit tool_result + complete the per-child stream.
                # CRITICAL: do NOT include node_step on this chunk. The
                # frontend dispatchChunk checks `if (c.node_step)` FIRST
                # and returns early, so a chunk that carries both fields
                # never reaches the `if (c.node_result)` branch — that's
                # the one that populates the Results tab via addToolResult.
                # Originally we packed them together which silently sent
                # tool results into the step trace and away from Results.
                # Match runtime/loop.py's pattern: emit tool_call (with
                # node_step) and tool_result (with only node_result) as
                # two separate chunks.
                try:
                    _cq.push(iid, {
                        "node_result": {
                            "tool": child_tool,
                            "args": child_args,
                            "result": result_text[:300],
                            "raw":    result_text,
                        },
                    })
                    # Additionally emit a `batch_member_result` chunk so the
                    # frontend's openHitlStream handler can render a chat
                    # bubble with the outcome. Without this, the only visible
                    # change after approve is in the Results tab (which the
                    # operator may not be looking at) and the Flow event log
                    # — chat itself stays empty, so the UX feels broken even
                    # though everything else worked. dispatchChunk doesn't
                    # currently produce chat bubbles for arbitrary chunks
                    # (only via streaming tokens during chat/stream), so we
                    # need a dedicated chunk type the SSE handler can pick up.
                    _cq.push(iid, {
                        "type": "batch_member_result",
                        "tool":   child_tool,
                        "target": child_args.get("device_id", "-"),
                        "args":   child_args,
                        "result": result_text,
                        "batch_id": batch.batch_id,
                    })
                except Exception:
                    pass
                finally:
                    try:
                        _cq.complete(iid)
                    except Exception:
                        pass

                _schedule_writeback(
                    query,
                    f"[BATCH COMPLETED — {child_tool} on "
                    f"{child_args.get('device_id', '-')}] {result_text}",
                    child_sid,
                )

                # Accumulate this child's data for the post-batch
                # SkillEvolver call. We append the tool name once even
                # if multiple children use it — evolver dedups internally
                # via "tools_used" set semantics, but the result snippets
                # are kept distinct for solution-summary diversity.
                _evolver_tools.append(child_tool)
                _evolver_results.append(result_text[:200])
                _evolver_targets.append(child_args.get("device_id", "-"))

            # Wait for any background writebacks to settle so the
            # 'completed' log accurately reflects all-done state. These
            # are wrapped in shield-style return_exceptions=True so a
            # writeback failure can't tank the whole batch finalizer.
            if _pending_writebacks:
                try:
                    await _asyncio.gather(*_pending_writebacks, return_exceptions=True)
                except Exception as _gather_exc:
                    logger.debug(
                        "batch_execute: writeback gather hit %s — ignoring",
                        _gather_exc,
                    )

            # ── SkillEvolver for the batch as a whole ─────────────
            # In the per-child POST path (webui/backend.py:_submit_hitl_decision)
            # batch members are explicitly skipped so we don't run evolver
            # N times for the same logical request. Run it ONCE here
            # against the unioned (tools_used, sample results) so a
            # successful multi-target batch can still mint a skill.
            #
            # Skips:
            #   - rejected/partial batches (no successful children to learn from)
            #   - evolver not wired (e.g. mode=pragmatic startup without LLM)
            #   - batches with zero successful children
            if (
                self._skill_evolver is not None
                and _evolver_tools
                and not resolution.rejected
            ):
                try:
                    # Dedup tools_used; preserve a representative
                    # solution summary by joining the first ~3 result
                    # snippets. Keep complexity moderate (6.0) since
                    # multi-target batches imply some operational
                    # complexity even when each child is mechanically
                    # the same op.
                    _unique_tools = list(dict.fromkeys(_evolver_tools))
                    _solution_summary = " | ".join(
                        f"{t}: {r}"
                        for t, r in zip(_evolver_targets[:3], _evolver_results[:3])
                    )[:400]
                    proposal = await self._skill_evolver.after_task(
                        task_description = query,
                        solution_summary = _solution_summary,
                        tools_used       = _unique_tools,
                        solution_steps   = [
                            f"{t} on {tgt}"
                            for t, tgt in zip(_evolver_tools, _evolver_targets)
                        ][:10],
                        key_observations = [
                            f"batch_size={len(_evolver_tools)}",
                            f"unique_tools={len(_unique_tools)}",
                        ],
                        complexity       = 6.0,
                        session_id       = batch.batch_id,
                    )
                    if proposal is not None:
                        logger.info(
                            "batch_execute: SkillEvolver proposed skill %r "
                            "for batch=%s (reuse_potential=%.2f)",
                            getattr(proposal, "name", "?"),
                            batch.batch_id[:12],
                            getattr(proposal, "reuse_potential", 0.0),
                        )
                    else:
                        logger.debug(
                            "batch_execute: SkillEvolver returned no proposal "
                            "for batch=%s (below threshold)",
                            batch.batch_id[:12],
                        )
                except Exception as _ev_exc:
                    # Evolver failures must never tank the batch finalizer.
                    logger.warning(
                        "batch_execute: SkillEvolver failed for batch=%s: %s",
                        batch.batch_id[:12], _ev_exc,
                    )

            logger.info(
                "batch_execute: batch=%s completed (%d children processed)",
                batch.batch_id[:12], len(resolution.decisions),
            )

        # Fire the background task. We don't keep a reference because
        # the task lifetime is bounded by the batch future — once
        # resolved (or future-failed), the task exits.
        _asyncio.create_task(
            _batch_execute_after_resolution(),
            name=f"batch_exec_{batch.batch_id[:8]}",
        )

        return batch.batch_id

    async def _tool_call_resumer(
        self, decision: HitlDecision, entry: CheckpointEntry,
    ) -> dict[str, Any]:
        """Resumer registered with the router — runs when an operator
        approves / edits a tool-call HITL. Merges parameter_patch into
        tool_args, invokes the tool, persists the result, and returns
        a list of chunks describing what happened so the calling
        endpoint can paint thinking-trace steps in the UI.

        Streams the same chunks through the global ChunkQueueRegistry
        as they are produced, so any frontend that already subscribed
        via GET /hitl/{interrupt_id}/stream sees progress in real time
        (parity with _agent_loop_resumer). Without this, the operator
        clicks approve and the trace freezes until the tool returns —
        which for slow tools (e.g. restart_service over real SSH) can
        be tens of seconds with no UI feedback.
        """
        state = entry.resume_handle.state
        tool_name = state.get("tool_name")
        tool_args = dict(state.get("tool_args") or {})
        session_id = state.get("session_id", "")
        original_query = state.get("original_query", "")
        _chunks: list[dict] = []

        # Live SSE plumbing: identical pattern to _agent_loop_resumer.
        # When the frontend subscribes via /hitl/{id}/stream the queue
        # auto-creates the stream on first push (see chunk_queue.push's
        # _ensure_sync), so we don't need to pre-register here.
        from hitl_core.chunk_queue import get_chunk_queue_registry
        _chunk_queue = get_chunk_queue_registry()
        _interrupt_id = entry.interrupt_id or state.get("interrupt_id", "")

        def _emit(chunk: dict) -> None:
            """Append to the sync result list AND push to the live queue.
            Wrapped in a try so a queue-push hiccup never blocks the tool
            execution path."""
            _chunks.append(chunk)
            if _interrupt_id:
                try:
                    _chunk_queue.push(_interrupt_id, chunk, session_id=session_id)
                except Exception:
                    # Don't let a streaming hiccup block the tool path;
                    # the synchronous return value still carries the
                    # chunk list as a fallback.
                    pass

        # We complete() the stream on every exit path below — package it
        # as a finally so even bare returns / unexpected exceptions still
        # signal end-of-stream to subscribers.
        try:
            if decision.decision == DecisionKind.REJECT:
                _emit({
                    "node":      "hitl_reject",
                    "node_step": f"Operator rejected tool call: {tool_name}",
                })
                await self._writeback(
                    original_query,
                    f"[HITL REJECTED — operator declined tool call `{tool_name}`]",
                    session_id,
                )
                await self._maybe_callback_source_agent(
                    interrupt_id=_interrupt_id,
                    result_text=f"操作员拒绝了工具调用 `{tool_name}`,未执行。",
                    decision="reject",
                )
                return {"tool": tool_name, "decision": "reject", "chunks": _chunks}

            # APPROVE or EDIT — merge patch and invoke
            if decision.decision == DecisionKind.EDIT and decision.parameter_patch:
                tool_args.update(decision.parameter_patch)
                logger.info(
                    "Tool HITL: applying operator edit — patch_keys=%s",
                    list(decision.parameter_patch.keys()),
                )
                _emit({
                    "node":        "hitl_edit",
                    "node_step":   f"Operator edited args: {list(decision.parameter_patch.keys())}",
                })

            if tool_name not in self._tool_registry:
                logger.warning(
                    "tool_call_resumer: tool %r not in registry", tool_name,
                )
                _emit({
                    "node":      "tool_error",
                    "node_step": f"Tool {tool_name!r} not registered — cannot dispatch",
                })
                return {"error": f"tool {tool_name!r} not registered", "chunks": _chunks}

            # Step: tool dispatch
            _emit({
                "node":      "tool_call",
                "node_step": f"Calling tool: {tool_name}",
            })
            try:
                result = await self._tool_registry[tool_name](tool_args)
            except Exception as exc:
                logger.exception("Tool %s execution failed: %s", tool_name, exc)
                await self._writeback(
                    original_query,
                    f"[HITL APPROVED — tool {tool_name} FAILED] {exc}",
                    session_id,
                )
                _emit({
                    "node":      "tool_error",
                    "node_step": f"Tool {tool_name} FAILED: {exc}",
                })
                return {"tool": tool_name, "error": str(exc), "chunks": _chunks}

            result_text = str(result)[:1500]
            await self._writeback(
                original_query,
                f"[HITL APPROVED & COMPLETED — {tool_name}] {result_text}",
                session_id,
            )
            # A2A Phase 3 (P3-b): if this HITL was raised serving an inbound
            # delegation, call the originator back so it can resume (mode B).
            await self._maybe_callback_source_agent(
                interrupt_id=_interrupt_id, result_text=result_text,
                decision="approve",
            )
            # Step: tool returned
            _emit({
                "node":        "tool_result",
                "node_step":   f"Inline: {tool_name}",
                "node_result": {
                    "tool":   tool_name,
                    "args":   tool_args,
                    "result": result_text[:300],
                },
            })
            return {
                "tool":        tool_name,
                "result":      result_text,
                "tool_result": result_text,
                "chunks":      _chunks,
            }
        finally:
            # Signal end-of-stream so any /hitl/{id}/stream subscriber
            # exits the async iterator cleanly. Idempotent if no one
            # subscribed.
            if _interrupt_id:
                try:
                    _chunk_queue.complete(_interrupt_id)
                except Exception:
                    pass

    # ────────────────────────────────────────────────────────────────
    # Multi-mode HITL (USER_CHOICE / CLARIFICATION)
    # ────────────────────────────────────────────────────────────────

    async def _raise_multi_mode(
        self,
        *,
        kind: str,
        chunk: dict,
        query: str,
        session_id: str,
        env_context: Optional[dict[str, Any]] = None,
    ) -> str:
        """Raise USER_CHOICE or CLARIFICATION HITL via hitl_core."""
        from hitl_core.schema import ResumeHandle

        choices_data = chunk.get("choices") or []
        clar_data    = chunk.get("clarification_fields") or []

        # Coerce raw dicts → ChoiceOption / ClarificationField
        choices = [
            ChoiceOption(**c) if isinstance(c, dict) else c
            for c in choices_data
        ]
        clar_fields = [
            ClarificationField(**f) if isinstance(f, dict) else f
            for f in clar_data
        ]

        trigger_kind = (
            TriggerKind.USER_CHOICE if kind == "user_choice"
            else TriggerKind.CLARIFICATION
        )

        _src_agent, _src_session, _src_query = _extract_delegation_provenance(
            env_context,
        )

        payload = HitlPayload(
            thread_id=session_id,
            context_id=session_id,
            user_query=query,
            trigger_kind=trigger_kind,
            risk_level=RiskLevel.LOW,
            intent_summary=chunk.get("summary", ""),
            confidence_score=float(chunk.get("confidence", 0.5)),
            proposed_action=ProposedAction(
                action_type=kind,
                target=session_id,
                parameters={},
            ),
            choices=choices,
            clarification_fields=clar_fields,
            source_agent      = _src_agent,
            source_session_id = _src_session,
            source_query      = _src_query,
        )
        entry = CheckpointEntry(
            interrupt_id=payload.interrupt_id,
            payload=payload,
            resume_handle=ResumeHandle(
                resumer_name="agent_loop_resumer",
                state={
                    "kind":           kind,
                    "session_id":     session_id,
                    "original_query": query,
                },
            ),
        )
        await self._router.register_payload(entry)
        logger.info(
            "Multi-mode HITL raised: id=%s kind=%s",
            payload.interrupt_id[:12], kind,
        )
        return payload.interrupt_id

    async def _agent_loop_resumer(
        self, decision: HitlDecision, entry: CheckpointEntry,
    ) -> dict[str, Any]:
        """Resumer for USER_CHOICE / CLARIFICATION. Folds the operator's
        input into the query and re-runs the agent loop.
        """
        state          = entry.resume_handle.state
        original_query = state.get("original_query", "")
        session_id     = state.get("session_id", "")
        kind           = state.get("kind", "")

        if decision.decision == DecisionKind.REJECT:
            await self._writeback(
                original_query,
                "[HITL REJECTED — operator declined to provide input]",
                session_id,
            )
            return {"decision": "reject"}

        # Build augmented query + env_context based on what kind of
        # operator input we got back.
        _resume_env: dict = {}
        if kind == "user_choice":
            chosen = decision.selected_choice_id or ""
            if chosen and chosen != "__none__":
                # Operator picked a specific skill — tell the loop which
                # one to favour, and mark ambiguity as resolved so the
                # gate doesn't fire again.
                augmented = (
                    f"{original_query}\n\n"
                    f"[OPERATOR DISAMBIGUATION] Use specifically the skill: `{chosen}`. "
                    f"Apply its recipe and tools when relevant to the request."
                )
                _resume_env["_skill_choice_resolved"] = True
                _resume_env["_forced_skill_id"]      = chosen
            else:
                # Operator picked "none" — ambiguity is acknowledged,
                # let the LLM decide unaided. Same flag prevents the
                # gate from re-firing.
                augmented = (
                    f"{original_query}\n\n"
                    f"[OPERATOR DISAMBIGUATION] No specific skill selected — "
                    f"proceed using context and tools without binding to a skill recipe."
                )
                _resume_env["_skill_choice_resolved"] = True
        elif kind == "clarification":
            answers = decision.clarification_answers or {}
            ans_lines = "\n".join(f"  - {k}: {v}" for k, v in answers.items() if v)
            augmented = (
                f"{original_query}\n\n"
                f"[OPERATOR-PROVIDED CLARIFICATIONS]\n{ans_lines}"
            )
            _resume_env["_clarification_resolved"] = True
        else:
            augmented = original_query

        # Re-run agent loop. The env_ctx flags above tell the loop to
        # skip whichever gate produced this interrupt, so we don't loop.
        # We capture every chunk emitted by the sub-stream so the calling
        # /hitl/{id}/approve endpoint can forward them to the UI's
        # thinking trace — otherwise the UI loses sight of everything
        # that happens after operator approval (TURN N, Tool Call,
        # nested HITL, post-action steps).
        _chunk_log: list[dict] = []

        # Push each chunk into the live SSE queue so a subscribed frontend
        # sees progress in real time (in addition to the final JSON response).
        from hitl_core.chunk_queue import get_chunk_queue_registry
        _chunk_queue = get_chunk_queue_registry()
        _interrupt_id = getattr(entry, "interrupt_id", None) or state.get("interrupt_id", "")

        # Progress heartbeat counters: track tool calls + turns so we can
        # inject explicit `progress` chunks every few tool calls. Without
        # this the UI sits silent during a long agent_loop_resumer run
        # (e.g. 24 chained read_stored_result pages) — chunks DO arrive
        # (tool_call, runtime_tool_result) but they look like internal
        # plumbing to operators, not "we're at step 12 of N".
        _progress_state = {
            "tool_calls":          0,
            "turns":               0,
            "last_progress_emit":  0,   # tool_calls at last emit
            "started_at":          __import__("time").monotonic(),
        }
        # Emit a progress chunk on EVERY tool call AND every LLM turn so
        # the UI stays alive even when a single tool call takes 60-90s
        # (typical for slow local LLMs). Setting INTERVAL=1 ensures the
        # operator sees motion no matter what the resumer does next.
        _PROGRESS_INTERVAL = 1

        async def _capture_chunk(ch: dict) -> None:
            try:
                ch_dict = dict(ch)
                _chunk_log.append(ch_dict)

                # Heartbeat: count tool calls (chunks with kind="tool_call"
                # or node=="runtime_tool_result" indicate a tool just ran)
                _kind = (
                    ch_dict.get("kind")
                    or ch_dict.get("type")
                    or ch_dict.get("node")
                    or ""
                )
                _step = ch_dict.get("node_step", "") or ""
                _tool_just_ran = (
                    _kind == "runtime_tool_result" or _step.startswith("TOOL◀")
                )
                _turn_started = _step.startswith("Turn ")
                if _tool_just_ran:
                    _progress_state["tool_calls"] += 1
                if _turn_started:
                    _progress_state["turns"] += 1

                # Emit a synthetic `progress` chunk on EITHER a tool
                # completion OR a new LLM turn. With INTERVAL=1 (every
                # tool call) plus turn-start triggers, the UI gets a
                # heartbeat at every meaningful agent state change:
                #   - "Turn 3 — calling LLM..." (turn-start)
                #   - "Tool call 5 complete — read_stored_result" (tool-end)
                # Operators see continuous motion instead of long silences.
                _should_emit = False
                _tcalls = _progress_state["tool_calls"]
                if _tool_just_ran and _tcalls - _progress_state["last_progress_emit"] >= _PROGRESS_INTERVAL:
                    _progress_state["last_progress_emit"] = _tcalls
                    _should_emit = True
                elif _turn_started:
                    _should_emit = True

                if _should_emit:
                    import time as _t
                    _elapsed = _t.monotonic() - _progress_state["started_at"]
                    # Try to surface the actual tool name in the message
                    # so operators see "Tool call 5 — read_stored_result"
                    # not just "5 tool calls"
                    _last_tool = (
                        ch_dict.get("tool")
                        or ch_dict.get("tool_name")
                        or ""
                    )
                    if _turn_started:
                        _msg = (
                            f"Turn {_progress_state['turns']} — LLM thinking "
                            f"({_tcalls} tools used, {_elapsed:.0f}s elapsed)"
                        )
                    else:
                        _msg = (
                            f"Tool call {_tcalls} complete"
                            + (f" — {_last_tool}" if _last_tool else "")
                            + f" ({_progress_state['turns']} turns, {_elapsed:.0f}s elapsed)"
                        )
                    _progress_chunk = {
                        "type":          "progress",
                        "tool_calls":    _tcalls,
                        "turns":         _progress_state["turns"],
                        "elapsed_s":     round(_elapsed, 1),
                        "last_tool":     _last_tool,
                        "phase":         ("turn_start" if _turn_started else "tool_end"),
                        "message":       _msg,
                        "interrupt_id":  _interrupt_id,
                    }
                    _chunk_log.append(_progress_chunk)
                    if _interrupt_id:
                        _chunk_queue.push(_interrupt_id, _progress_chunk, session_id=session_id)
                    # Stamp heartbeat clock so the wall-clock heartbeat
                    # loop doesn't duplicate-emit right after this.
                    try:
                        _heartbeat_state["last_emit_at"] = __import__("time").monotonic()
                    except Exception:
                        pass

                # Push the original chunk to the live queue for SSE subscribers.
                # session_id is forwarded so a subsequent chat_stream on
                # the same session can close stale streams (audit fix D).
                if _interrupt_id:
                    _chunk_queue.push(_interrupt_id, ch_dict, session_id=session_id)
            except Exception:
                pass

        # Wall-clock heartbeat: if LLM thinking on a slow local model takes
        # 60-120s, the event-driven progress emit (tied to tool completion
        # or turn start) is silent during that whole window. A separate
        # async task emits a "still thinking" progress chunk every
        # HEARTBEAT_INTERVAL seconds so the UI stays alive regardless.
        _HEARTBEAT_INTERVAL = 10.0   # seconds
        _heartbeat_state = {"last_emit_at": _progress_state["started_at"]}

        async def _heartbeat_loop():
            import time as _t
            try:
                while True:
                    await asyncio.sleep(_HEARTBEAT_INTERVAL)
                    _now = _t.monotonic()
                    # Only fire if no event-driven progress fired recently
                    # (otherwise we'd duplicate; check via _capture_chunk's
                    # progress_state.last_progress_emit isn't time-based,
                    # so fall back to our own _heartbeat_state.last_emit_at,
                    # which IS updated whenever ANY progress emits).
                    if _now - _heartbeat_state["last_emit_at"] < _HEARTBEAT_INTERVAL * 0.8:
                        # Something else emitted recently; skip
                        continue
                    _elapsed = _now - _progress_state["started_at"]
                    _hb_chunk = {
                        "type":         "progress",
                        "tool_calls":   _progress_state["tool_calls"],
                        "turns":        _progress_state["turns"],
                        "elapsed_s":    round(_elapsed, 1),
                        "phase":        "heartbeat",
                        "message":      (
                            f"Still working — {_progress_state['tool_calls']} tools, "
                            f"{_progress_state['turns']} turns, {_elapsed:.0f}s elapsed "
                            f"(LLM may be thinking on a long prompt)"
                        ),
                        "interrupt_id": _interrupt_id,
                    }
                    _chunk_log.append(_hb_chunk)
                    if _interrupt_id:
                        _chunk_queue.push(
                            _interrupt_id, _hb_chunk, session_id=session_id,
                        )
                    _heartbeat_state["last_emit_at"] = _now
            except asyncio.CancelledError:
                # Normal — heartbeat is cancelled when execute_query returns
                raise

        _heartbeat_task = asyncio.create_task(
            _heartbeat_loop(), name="hitl_resumer_heartbeat",
        )

        # Wrap execute_query: even on exception, surface captured chunks
        # so the frontend sees progressive trace + a clear error reason.
        try:
            result = await self.execute_query(
                query=augmented, session_id=session_id,
                env_context=_resume_env,
                on_chunk=_capture_chunk,
            )
        except Exception as _eq_exc:
            # Build a synthetic result that still carries the chunks accumulated
            # up to the failure. Frontend renders chunks + an error message
            # instead of going silent.
            logger.warning(
                "agent_loop_resumer: execute_query failed (%s) — surfacing %d "
                "chunks captured before failure",
                _eq_exc, len(_chunk_log),
            )
            _err_text = f"[Resume failed: {_eq_exc}]"
            # Persist the partial trace as a memory write so it isn't lost
            try:
                await self._writeback(original_query, _err_text, session_id)
            except Exception:
                pass
            if _interrupt_id:
                _chunk_queue.complete(_interrupt_id)
            # Cancel the heartbeat task on the error path too
            _heartbeat_task.cancel()
            try:
                await _heartbeat_task
            except (asyncio.CancelledError, Exception):
                pass
            return {
                "text":       _err_text,
                "decision":   decision.decision.value,
                "chunks":     _chunk_log,
                "error":      str(_eq_exc),
                "partial":    True,
            }

        # Signal end-of-stream so SSE subscribers exit
        if _interrupt_id:
            _chunk_queue.complete(_interrupt_id)

        # Cancel the wall-clock heartbeat task on the success path.
        # asyncio.gather wouldn't help here because heartbeat is intended
        # to outlive execute_query if it stalled — we explicitly cancel.
        _heartbeat_task.cancel()
        try:
            await _heartbeat_task
        except (asyncio.CancelledError, Exception):
            pass

        return {
            "text":       result.get("text", ""),
            "decision":   decision.decision.value,
            "chunks":     _chunk_log,
            # Surface any nested HITL the sub-stream raised so the
            # frontend can switch tabs and refresh /hitl/pending.
            "interrupted":  result.get("interrupted", False),
            "interrupt_id": result.get("interrupt_id"),
        }

    # ────────────────────────────────────────────────────────────────
    # Batch HITL — proxy to pipeline.request_batch_approval
    # ────────────────────────────────────────────────────────────────

    async def raise_batch(
        self,
        *,
        payloads: list[HitlPayload],
        title: str = "",
        description: str = "",
        policy: BatchPolicy = BatchPolicy.BEST_EFFORT,
        wait_mode: BatchWaitMode = BatchWaitMode.ALL,
        session_id: str = "",
    ) -> str:
        """Convenience wrapper. The pipeline-driven path is preferred
        (ctx.request_batch_approval inside a step); use this when the
        IT-ops layer wants to fire a batch from outside a pipeline."""
        from hitl_core.schema import HitlBatch, ResumeHandle

        batch = HitlBatch(
            thread_id=session_id, policy=policy, wait_mode=wait_mode,
            title=title, description=description,
        )
        for p in payloads:
            if not p.thread_id:
                p.thread_id = session_id
            from hitl_core.batch import BATCH_ID_KEY
            p.context_snapshot[BATCH_ID_KEY] = batch.batch_id
            batch.interrupt_ids.append(p.interrupt_id)
            entry = CheckpointEntry(
                interrupt_id=p.interrupt_id,
                payload=p,
                resume_handle=ResumeHandle(
                    resumer_name="agent_loop_resumer",
                    state={
                        "kind":           "batch",
                        "session_id":     session_id,
                        "original_query": p.user_query,
                    },
                ),
            )
            await self._router.register_payload(entry)

        # Open via the router's batch coordinator
        await self._router.batch.open_batch(batch)
        return batch.batch_id

    # ────────────────────────────────────────────────────────────────
    # A2A protocol shim — let backend.py call us with the same
    # (RequestContext, EventQueue) signature it uses for the legacy
    # ITOpsHitlAgentExecutor. Internally we extract query/session_id
    # and call our keyword-arg execute().
    # ────────────────────────────────────────────────────────────────

    async def execute(self, context, event_queue) -> None:
        """A2A-protocol-compatible entry point. Mirrors the legacy
        ITOpsHitlAgentExecutor.execute signature so webui/backend.py
        doesn't need to know which backend is wired up.

        Extracts query/session_id/confirmed_facts from RequestContext,
        runs the agent loop, and streams tokens through the EventQueue
        as A2A events. Terminal events (interrupt / done / error) become
        protocol status updates.

        IMPORTANT (2026-05 fix): this MUST use the real EventQueue API —
        `enqueue_event(<A2AEvent>)` — and MUST guarantee the queue is
        finalised (a MessageEvent seals it) on every exit path. The earlier
        version called non-existent helpers (enqueue_event_status /
        enqueue_event_message), so every enqueue silently failed, NO events
        reached the queue, and it was never finalised — the consumer
        (a2a/request_handler._handle_stream) blocked forever on consume(),
        the SSE stream never sent [DONE], and the delegating agent's
        dispatcher hit a ReadTimeout. Root cause of "peer stream failed:
        ReadTimeout" in live two-agent delegation.
        """
        # Lazy import — keep a2a deps optional for non-protocol callers
        from a2a.schemas import (
            Message, TextPart, DataPart, Artifact, TaskStatus, TaskState,
            TaskStatusUpdateEvent, TaskArtifactUpdateEvent, MessageEvent,
        )

        task_id    = getattr(context, "task_id", "")
        context_id = getattr(context, "context_id", "") or task_id

        # 1. Pull inputs from the A2A RequestContext
        try:
            query = (
                context.message.parts[0].text
                if context.message and context.message.parts
                else ""
            )
        except (IndexError, AttributeError):
            query = ""

        meta = getattr(context, "metadata", {}) or {}
        session_id = (
            meta.get("session_id")
            or getattr(context, "context_id", "")
            or ""
        )
        confirmed_facts = list(meta.get("confirmed_facts") or [])
        env_context     = dict(meta.get("env_context") or {})

        # Pull delegation provenance from A2A request metadata into the
        # env_context that flows into the runtime loop. The dispatcher
        # (task/inter/coordinator.py) sets these top-level keys when this
        # agent is processing a [DELEGATE:] from a peer; on a user-initiated
        # request they're absent. _raise_*_hitl methods read them off
        # env_context via _extract_delegation_provenance() and stamp the
        # HitlPayload, so the operator UI can show "Delegated from <agent>"
        # provenance on the approval card. None of these clobber an
        # explicit env_context value if the caller already set one.
        for _k in ("source_agent", "source_session_id", "source_query",
                   "delegated_by", "forked"):
            if _k in meta and _k not in env_context:
                env_context[_k] = meta[_k]

        # Record INBOUND delegation in this agent's task_store so the local
        # Delegations tab can display "received from <source_agent>". This
        # is a different code path from outbound dispatch (task/delegation.py
        # writes outbound TaskDefinitions); both end up in the same store
        # with metadata.direction distinguishing them.
        _inbound_task = None
        if meta.get("source_agent"):
            try:
                _inbound_task = await self._record_inbound_delegation(
                    task_id     = task_id,
                    session_id  = session_id,
                    context_id  = context_id,
                    description = query,
                    meta        = meta,
                )
            except Exception as _ir_exc:
                # Inbound recording is best-effort — never fail the actual
                # delegation just because the audit row couldn't be written.
                logger.debug(
                    "Inbound delegation recording skipped: %s", _ir_exc,
                )

        _finalised = False   # guard so we seal the queue exactly once
        _streamed_any_token = False   # any token went through _emit_token?
                                      # — used by _finalize to avoid double-
                                      # counting (see _finalize body)

        async def _emit_status(state) -> None:
            try:
                await event_queue.enqueue_event(TaskStatusUpdateEvent(
                    task_id=task_id, context_id=context_id,
                    status=TaskStatus(state=state),
                ))
            except Exception as exc:
                logger.debug("a2a status emit failed: %s", exc)

        async def _emit_token(tok: str) -> None:
            # Stream tokens as artifact updates carrying {token,type:token},
            # which the delegating dispatcher's _unwrap_a2a_event maps back to
            # {token: ...} chunks.
            nonlocal _streamed_any_token
            _streamed_any_token = True
            try:
                await event_queue.enqueue_event(TaskArtifactUpdateEvent(
                    task_id=task_id, context_id=context_id,
                    artifact=Artifact(
                        name="llm_token",
                        parts=[DataPart(data={"token": tok, "type": "token"})],
                    ),
                ))
            except Exception as exc:
                logger.debug("a2a token enqueue failed: %s", exc)

        async def _finalize(final_text: str) -> None:
            # A MessageEvent seals the queue (consume() returns). MUST be the
            # last thing we emit on the happy path.
            #
            # CRITICAL (Phase 2B+, 2026-05): when we streamed tokens already,
            # MessageEvent body MUST be empty (or a generic seal token) — not
            # the full final_text. Otherwise the delegating peer receives BOTH
            # the streamed tokens AND the final_text as a single big token,
            # double-counts the content into _result_parts, and the parent LLM
            # composes a final answer that repeats the peer's analysis 2-3x
            # ("Analysis: ...\n leaf-1 Established...\nAnalysis: ...\n leaf-1
            # Established...\n").
            #
            # When no tokens were streamed (e.g. peer-side HITL pending, or
            # non-streaming code paths), keep the legacy behavior and put
            # final_text in the MessageEvent so the peer has SOMETHING to
            # work with.
            nonlocal _finalised
            if _finalised:
                return
            _finalised = True
            _seal_text = "Task completed." if _streamed_any_token else (
                final_text or "Task completed."
            )
            try:
                await event_queue.enqueue_event(MessageEvent(
                    task_id=task_id, context_id=context_id,
                    message=Message(role="assistant",
                                    parts=[TextPart(text=_seal_text)]),
                ))
            except Exception as exc:
                logger.debug("a2a finalize emit failed: %s", exc)

        # 2. Token streaming hook
        async def _on_token(tok: str) -> None:
            await _emit_token(tok)

        # 3. Status: working
        await _emit_status(TaskState.WORKING)

        # 4. Run agent loop.
        try:
            result = await self.execute_query(
                query=query, session_id=session_id,
                confirmed_facts=confirmed_facts,
                env_context=env_context,
                on_token=_on_token,
            )
        except Exception as exc:
            logger.exception("HitlExecutor.execute_a2a failed: %s", exc)
            await _emit_status(TaskState.FAILED)
            # CRITICAL: finalise even on failure so the consumer unblocks and
            # the delegating agent gets a terminal signal instead of timing out.
            await _finalize(f"Task failed on peer: {exc}")
            # Phase 2B+ (2026-05): record the failure on the inbound
            # delegation row too. Without this, the local Delegations tab
            # shows the row stuck in RUNNING forever even though the agent
            # actually errored out (observed: Ollama 300s timeout leaves
            # `← lan-agent ... RUNNING` stuck on dc-agent's UI). Use
            # _fail_inbound_delegation so the row clearly shows the error.
            if _inbound_task is not None:
                try:
                    await self._fail_inbound_delegation(_inbound_task, exc)
                except Exception as _if_exc:
                    logger.debug("Inbound failure update skipped: %s", _if_exc)
            return

        # 5. Terminal handling
        try:
            if result.get("interrupted"):
                # Peer-side HITL — surface input-required + interrupt_id, then
                # finalise so the delegating side doesn't hang. (Cross-agent
                # HITL passthrough is Phase 3; for now the peer's own operator
                # handles it; the delegating agent is told via the message.)
                try:
                    await event_queue.enqueue_event(TaskStatusUpdateEvent(
                        task_id=task_id, context_id=context_id,
                        status=TaskStatus(state=TaskState.INPUT_REQUIRED,
                                          message=str(result.get("interrupt_id", ""))),
                    ))
                except Exception as exc:
                    logger.debug("a2a interrupt-status emit failed: %s", exc)
                await _finalize(
                    "Peer requires operator approval (HITL) to continue; "
                    "handle it on the peer's console."
                )
            else:
                final_text = result.get("text") or ""
                await _emit_status(TaskState.COMPLETED)
                await _finalize(final_text)
            # Update inbound delegation record with terminal state so the
            # local Delegations tab shows completed / input_required / etc.
            if _inbound_task is not None:
                try:
                    await self._complete_inbound_delegation(
                        _inbound_task, result,
                    )
                except Exception as _ic_exc:
                    logger.debug("Inbound completion update skipped: %s", _ic_exc)
        except Exception as exc:
            logger.debug("a2a terminal emit failed: %s", exc)
            # Last-resort finalise so the consumer never hangs.
            await _finalize("")

    async def cancel(self, context, event_queue) -> None:
        """A2A cancel hook. Currently best-effort no-op: the runtime loop
        doesn't expose a cancel primitive, so we acknowledge but continue.
        Hosts that need true cancellation can override by subclassing."""
        from a2a.schemas import TaskState
        try:
            await event_queue.enqueue_event_status(
                task_id=getattr(context, "task_id", ""),
                state=TaskState.cancelled,
            )
        except Exception:
            pass

    async def _record_inbound_delegation(
        self,
        *,
        task_id: str,
        session_id: str,
        context_id: str,
        description: str,
        meta: dict,
    ):
        """Write a TaskDefinition row tagged inbound when handling a peer's
        [DELEGATE:] request, so the local Delegations tab can show
        'received from <source_agent>'. Returns the saved task (so the
        caller can update its state on completion) or None when no
        task_store is wired or write fails.
        """
        if self._task_store is None:
            return None
        from task.schemas import TaskDefinition, TaskState
        try:
            t = TaskDefinition(
                task_id     = task_id,
                session_id  = session_id,
                context_id  = context_id or session_id,
                description = description or "",
                state       = TaskState.RUNNING,
                metadata    = {
                    # Distinguish from outbound dispatches written by
                    # task/delegation.py. The /delegations endpoint reads
                    # this to label rows "→ peer" vs "← peer".
                    "direction":         "inbound",
                    "source_agent":      meta.get("source_agent") or "",
                    "source_session_id": meta.get("source_session_id") or "",
                    "source_query":      meta.get("source_query") or "",
                    "delegated_by":      meta.get("delegated_by") or "",
                    "forked":            bool(meta.get("forked", False)),
                    "shared_facts_count":
                        int(meta.get("shared_facts_count", 0) or 0),
                },
            )
            await self._task_store.save(t)
            return t
        except Exception as exc:
            logger.debug("inbound record build/save failed: %s", exc)
            return None

    async def _complete_inbound_delegation(self, task, result: dict) -> None:
        """Update the inbound TaskDefinition's state + result on terminal."""
        if self._task_store is None or task is None:
            return
        from task.schemas import TaskState
        from datetime import datetime, timezone
        try:
            if result.get("interrupted"):
                # Peer-side HITL pending — runtime loop produced no final
                # text yet. Mark accordingly so the local UI can show
                # "awaiting operator approval".
                task.state = TaskState.PENDING
                task.metadata["awaiting_hitl_id"] = (
                    result.get("interrupt_id") or ""
                )
            else:
                _txt = result.get("text") or ""
                task.state  = TaskState.COMPLETED
                task.result = {"text": _txt}
            task.completed_at = datetime.now(timezone.utc).isoformat()
            await self._task_store.save(task)
        except Exception as exc:
            logger.debug("inbound completion save failed: %s", exc)

    async def _fail_inbound_delegation(self, task, exc: BaseException) -> None:
        """Mark the inbound TaskDefinition as FAILED with the error message.

        Called from execute() when execute_query raises (e.g. Ollama
        timeout, LLM crash, network error to local LLM). Without this,
        the row stays in RUNNING forever — the operator can't tell the
        difference between "still working" and "died 10 minutes ago".
        """
        if self._task_store is None or task is None:
            return
        from task.schemas import TaskState
        from datetime import datetime, timezone
        try:
            task.state        = TaskState.FAILED
            task.error        = str(exc)[:500]   # truncate huge tracebacks
            task.completed_at = datetime.now(timezone.utc).isoformat()
            await self._task_store.save(task)
        except Exception as _exc:
            logger.debug("inbound failure save failed: %s", _exc)

    async def _writeback(
        self, user_text: str, assistant_text: str, session_id: str,
        importance: float = 0.7,
    ) -> None:
        """Persist a turn to memory. Failures are logged but never
        block the protocol layer's response.

        `importance` drives the memory layer's fanout:
          < 0.30 → skipped entirely
          0.30–0.49 → long-term chunk only (no LLM fact distillation)
          0.50–0.74 → chunk + LLM-driven fact distillation (slow)
          ≥ 0.75 → distill + user-profile update (slowest)

        Default is 0.7 to match the original behaviour (full distillation
        on regular HITL flows). Batch executions should pass 0.4 to skip
        the LLM-bound distillation pass, which would otherwise serialize
        N tool executions against an Ollama LLM queue already contended
        for by approval-path classifiers (SkillEvolver, etc.). That's a
        real bug we hit: a 2-child batch took 117s end-to-end because
        each writeback triggered a 30s LLM timeout, blocking the next
        child's tool dispatch.
        """
        if self._memory is None or not session_id:
            return
        try:
            await self._memory.after_turn(
                session_id=session_id,
                user_text=user_text,
                assistant_text=assistant_text,
                tool_calls=[],
                importance=importance,
            )
        except Exception as exc:
            logger.debug("memory writeback skipped: %s", exc)


# ---------------------------------------------------------------------------
# Bootstrap helper — assembles a default executor for the original project
# ---------------------------------------------------------------------------

def build_default_executor(
    *,
    runtime_loop,
    llm_engine,
    tool_registry: dict,
    memory_router=None,
    audit_log_path: Optional[str] = None,
    redis_url: Optional[str] = None,
) -> tuple[HitlExecutor, HitlRouter]:
    """One-call bootstrap for the original project. Picks reasonable
    defaults:
      * In-memory checkpoint store unless redis_url is given
      * In-memory audit sink unless audit_log_path is given
      * Default IT-ops device coreferencer
      * Audit hook wired into the router

    Returns (executor, router) — caller mounts the router's HTTP
    transport on their FastAPI app.
    """
    if redis_url:
        from hitl_core import RedisCheckpointStore
        store = RedisCheckpointStore(redis_url=redis_url)
    else:
        store = InMemoryCheckpointStore()

    if audit_log_path:
        from hitl_core import FileAuditSink
        sink = FileAuditSink(audit_log_path)
    else:
        sink = InMemoryAuditSink()
    audit_logger = AuditLogger(sink=sink)

    router = HitlRouter(store=store, on_audit=audit_logger.as_hook())
    pipeline = HitlPipeline(
        store=store,
        batch_coordinator=router.batch,    # share the same coordinator
        on_audit=audit_logger.as_hook(),
    )

    executor = HitlExecutor(
        runtime_loop=runtime_loop,
        llm_engine=llm_engine,
        tool_registry=tool_registry,
        memory_router=memory_router,
        hitl_router=router,
        hitl_pipeline=pipeline,
        audit_logger=audit_logger,
    )
    return executor, router