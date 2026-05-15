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
    ):
        self._runtime_loop  = runtime_loop      # patched AgentRuntimeLoop
        self._llm_engine    = llm_engine
        self._tool_registry = dict(tool_registry or {})
        self._memory        = memory_router
        self._router        = hitl_router
        self._pipeline      = hitl_pipeline
        self._coref         = coreferencer or build_default_device_coreferencer()
        self._audit         = audit_logger

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
    ) -> str:
        """Raise a destructive-tool-call HITL via hitl_core.

        Builds a HitlPayload with the tool's actual args populated +
        editable_param_keys auto-derived. Registers a tool_call_resumer
        ResumeHandle so the actual tool execution happens on operator
        approval (with parameter_patch merged).
        """
        editable_keys = chunk.get("editable_param_keys") or _derive_editable_keys(
            tool_name, tool_args,
        )

        _target = str(tool_args.get("device_id") or tool_args.get("target") or "-")
        _intent = f"Agent proposes calling `{tool_name}` on `{_target}`"

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
        logger.info(
            "Tool HITL raised: id=%s tool=%s editable=%s",
            payload.interrupt_id[:12], tool_name, editable_keys,
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

        # First materialise all child CheckpointEntries
        child_entries: list[CheckpointEntry] = []
        editable_table = self._cfg.editable_hitl_tools if self._cfg else {}
        for (n, a) in calls:
            _editable = list(editable_table.get(n, []))
            if not _editable:
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

        # Open the batch via the coordinator
        await self._router.batch.open_batch(batch)
        logger.info(
            "Tool HITL batch raised: batch_id=%s tool=%s children=%d",
            batch.batch_id[:12], tool_name, len(child_entries),
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
                    _chunk_queue.push(_interrupt_id, chunk)
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

        async def _capture_chunk(ch: dict) -> None:
            try:
                ch_dict = dict(ch)
                _chunk_log.append(ch_dict)
                # Push to live queue for SSE subscribers (non-blocking)
                if _interrupt_id:
                    _chunk_queue.push(_interrupt_id, ch_dict)
            except Exception:
                pass

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
        as text parts. Terminal events (interrupt / done / error) become
        protocol status updates.
        """
        # Lazy import — keep a2a deps optional for non-protocol callers
        from a2a.schemas import Message, TextPart, TaskState
        from a2a.event_queue import EventQueue  # noqa: F401 — type hint only

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
        task_id         = getattr(context, "task_id", "")

        # 2. Wire token streaming through the event queue
        async def _on_token(tok: str) -> None:
            try:
                await event_queue.enqueue_event_message(
                    message=Message(role="assistant", parts=[TextPart(text=tok)]),
                    task_id=task_id,
                )
            except Exception as exc:
                logger.debug("a2a token enqueue failed: %s", exc)

        # 3. Status: working
        try:
            await event_queue.enqueue_event_status(
                task_id=task_id, state=TaskState.working,
            )
        except Exception:
            pass

        # 4. Run agent loop. The loop emits stop_hitl chunks when its
        #    tool watch-list catches a destructive tool_call; execute_query
        #    raises HITL via hitl_core and returns interrupted=True. The
        #    LLM, not pre_verify, is the source of "destructive" detection.
        try:
            result = await self.execute_query(
                query=query, session_id=session_id,
                confirmed_facts=confirmed_facts,
                env_context=env_context,
                on_token=_on_token,
            )
        except Exception as exc:
            logger.exception("HitlExecutor.execute_a2a failed: %s", exc)
            try:
                await event_queue.enqueue_event_status(
                    task_id=task_id, state=TaskState.failed,
                )
            except Exception:
                pass
            return

        # 5. Terminal status
        try:
            if result.get("interrupted"):
                # Surface the interrupt_id so the UI's HITL tab can pick it up.
                # The protocol convention: emit an "input-required" status with
                # the interrupt_id in metadata; the frontend HITL tab polls
                # /hitl/pending and renders the card.
                await event_queue.enqueue_event_status(
                    task_id=task_id,
                    state=TaskState.input_required,
                    metadata={"interrupt_id": result.get("interrupt_id", "")},
                )
            else:
                # Final answer as one terminal message
                final_text = result.get("text") or ""
                if final_text:
                    await event_queue.enqueue_event_message(
                        message=Message(role="assistant",
                                        parts=[TextPart(text=final_text)]),
                        task_id=task_id, final=True,
                    )
                await event_queue.enqueue_event_status(
                    task_id=task_id, state=TaskState.completed,
                )
        except Exception as exc:
            logger.debug("a2a terminal-status emit failed: %s", exc)

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

    async def _writeback(
        self, user_text: str, assistant_text: str, session_id: str,
    ) -> None:
        """Persist a turn to memory. Failures are logged but never
        block the protocol layer's response."""
        if self._memory is None or not session_id:
            return
        try:
            await self._memory.after_turn(
                session_id=session_id,
                user_text=user_text,
                assistant_text=assistant_text,
                tool_calls=[],
                importance=0.7,
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