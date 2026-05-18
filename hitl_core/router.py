"""
hitl_core.router — Decision validation, callback registry, and dispatch.

The router sits between the transport layer (HTTP / SSE / etc.) and the
pipeline. It owns three responsibilities:

  1. **Validate** decisions against the payload they target (correct
     interrupt_id, choice id from the payload's choice list, required
     clarification fields filled, etc.).

  2. **Dispatch** decisions through one of two paths:

     • In-process resume — when the pipeline that raised the interrupt
       is still alive in this process, hand the decision directly to
       its waiter via HitlPipeline.resume_with. This is the hot path
       for single-process deployments.

     • Resumer registry — when there's no in-process pipeline (the
       request hit a different replica, the original process restarted,
       or the pipeline runs detached), look up the named resumer the
       host registered at startup and call it with the decision +
       checkpoint's resume_handle.state. Resumers know how to rebuild
       the work that needs doing.

  3. **Audit** every transition — interrupt_raised, decision_made,
     graph_resumed, execution_done — through an injected audit hook.

Design notes:

  • The router does NOT know about pipelines, pydantic, or langchain.
    It works against the schema + store interfaces. You can use it
    standalone if you have a different orchestration model.

  • Resumers are async callables matching:
      (decision: HitlDecision, entry: CheckpointEntry) -> Any

  • Validation errors raise DecisionValidationError, which transports
    map to HTTP 400 / a structured error response.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Awaitable, Callable, Optional

from .batch import BATCH_ID_KEY, BatchCoordinator, BatchResolution, get_batch_id
from .schema import (
    AuditEventKind,
    BatchSnapshot,
    BatchSubmission,
    CheckpointEntry,
    DecisionKind,
    HitlBatch,
    HitlDecision,
    HitlPayload,
    InterruptState,
)
from .store import BaseCheckpointStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class DecisionValidationError(ValueError):
    """Raised when a HitlDecision fails to match the constraints of its
    target payload. Transports should surface this as 4xx / structured
    error, not retry."""


class ResumeError(RuntimeError):
    """Raised when a decision is valid but the system can't resume —
    no in-process waiter and no registered resumer."""


# ---------------------------------------------------------------------------
# Resumer protocol
# ---------------------------------------------------------------------------

# A resumer is registered by the host at startup; it knows how to
# resume work after a decision lands. Signature:
#
#     async def my_resumer(decision: HitlDecision,
#                          entry:    CheckpointEntry) -> Any: ...
#
# The router invokes it when no in-process waiter is found. The
# return value flows back to the caller of HitlRouter.deliver().
Resumer = Callable[[HitlDecision, CheckpointEntry], Awaitable[Any]]


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def _validate_decision_against_payload(
    decision: HitlDecision, payload: HitlPayload,
) -> None:
    """Check the operator's decision is internally consistent and matches
    the constraints encoded in the payload. Raises DecisionValidationError
    on mismatch.

    Rules per DecisionKind:

      APPROVE / REJECT / ESCALATE / TIMEOUT
        No payload-specific checks beyond interrupt_id matching.

      EDIT
        - parameter_patch must be a dict
        - every key in parameter_patch must be in payload.editable_param_keys
          (when editable_param_keys is non-empty — empty list = "no
          editing allowed", any patch fails)

      CHOOSE
        - selected_choice_id must be set
        - selected_choice_id must match one of payload.choices[].id

      ANSWER
        - clarification_answers must be a dict
        - every required field in payload.clarification_fields must
          have a non-empty answer
    """
    if decision.interrupt_id != payload.interrupt_id:
        raise DecisionValidationError(
            f"interrupt_id mismatch: decision={decision.interrupt_id!r} "
            f"payload={payload.interrupt_id!r}"
        )

    kind = decision.decision

    if kind == DecisionKind.EDIT:
        patch = decision.parameter_patch
        if not isinstance(patch, dict) or not patch:
            raise DecisionValidationError(
                "EDIT decision requires non-empty parameter_patch"
            )
        editable = set(payload.editable_param_keys or [])
        if not editable:
            raise DecisionValidationError(
                "EDIT not allowed: payload has no editable_param_keys"
            )
        bad_keys = set(patch) - editable
        if bad_keys:
            raise DecisionValidationError(
                f"EDIT contains keys not in editable_param_keys: "
                f"{sorted(bad_keys)} (allowed: {sorted(editable)})"
            )

    elif kind == DecisionKind.CHOOSE:
        chosen = decision.selected_choice_id
        if not chosen:
            raise DecisionValidationError(
                "CHOOSE decision requires selected_choice_id"
            )
        valid_ids = {c.id for c in (payload.choices or [])}
        if not valid_ids:
            raise DecisionValidationError(
                "CHOOSE not allowed: payload has no choices"
            )
        if chosen not in valid_ids:
            raise DecisionValidationError(
                f"selected_choice_id {chosen!r} not in payload.choices "
                f"({sorted(valid_ids)})"
            )

    elif kind == DecisionKind.ANSWER:
        answers = decision.clarification_answers or {}
        if not isinstance(answers, dict):
            raise DecisionValidationError(
                "ANSWER decision requires clarification_answers dict"
            )
        required_keys = [
            f.key for f in (payload.clarification_fields or [])
            if f.required
        ]
        missing = [
            k for k in required_keys
            if not (answers.get(k) or "").strip()
        ]
        if missing:
            raise DecisionValidationError(
                f"ANSWER missing required clarification fields: {missing}"
            )

    # APPROVE / REJECT / ESCALATE / TIMEOUT — no further checks


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class HitlRouter:
    """Single entry point for HITL operations.

    Lifecycle of an interrupt:

      1. Pipeline (or any other producer) creates a HitlPayload and
         hands it to a CheckpointStore via register_payload().

      2. UI fetches it via list_pending() / load_payload().

      3. Operator submits a HitlDecision via deliver(). The router:
         a. Validates structurally
         b. Atomically transitions the entry PENDING → RESOLVED
         c. Hands the decision to the in-process waiter if one exists,
            otherwise to the resume_handle's named resumer.

      4. Audit events fire at every state change.
    """

    def __init__(
        self,
        *,
        store: BaseCheckpointStore,
        batch_coordinator: Optional[BatchCoordinator] = None,
        on_audit: Optional[Callable[..., Awaitable[None]]] = None,
    ):
        self._store = store
        # Batch routing — feed child decisions into the coordinator so
        # the producer's batch future can resolve when wait conditions
        # are met. Construct a default coordinator if the host didn't
        # pass one (matches HitlPipeline default).
        self._batch = (
            batch_coordinator
            if batch_coordinator is not None
            else BatchCoordinator(store=store, on_audit=on_audit)
        )
        self._on_audit = on_audit
        # name → async resumer
        self._resumers: dict[str, Resumer] = {}
        # In-process waiters: interrupt_id → asyncio.Future[HitlDecision]
        # Pipelines register here when they raise an interrupt.
        self._waiters: dict[str, asyncio.Future[HitlDecision]] = {}
        self._waiter_lock = asyncio.Lock()

    @property
    def batch(self) -> BatchCoordinator:
        return self._batch

    # ── Resumer registry ────────────────────────────────────────────

    def register_resumer(self, name: str, fn: Resumer) -> None:
        """Register a named resumer. Hosts call this at startup for
        every kind of detached resume they want to support.

        The name lives in the resume_handle written when the interrupt
        was created. Multiple resumers can coexist — e.g. one for
        agent-loop retries, one for direct tool calls, one for plain
        approve-and-continue."""
        if name in self._resumers:
            logger.warning("register_resumer: overwriting existing %r", name)
        self._resumers[name] = fn

    def unregister_resumer(self, name: str) -> None:
        self._resumers.pop(name, None)

    # ── Waiter registry (for in-process pipelines) ──────────────────

    async def register_waiter(
        self, interrupt_id: str,
    ) -> asyncio.Future[HitlDecision]:
        """Create or return the future a paused pipeline awaits for its
        decision. HitlPipeline calls this internally; external callers
        rarely need it."""
        async with self._waiter_lock:
            existing = self._waiters.get(interrupt_id)
            if existing is not None and not existing.done():
                return existing
            fut: asyncio.Future[HitlDecision] = asyncio.get_running_loop().create_future()
            self._waiters[interrupt_id] = fut
            return fut

    async def unregister_waiter(self, interrupt_id: str) -> None:
        async with self._waiter_lock:
            self._waiters.pop(interrupt_id, None)

    # ── Producer-side: register a new interrupt ─────────────────────

    async def register_payload(
        self,
        entry: CheckpointEntry,
    ) -> None:
        """Persist a new interrupt and emit an INTERRUPT_RAISED audit
        event. Producers call this after building the payload but before
        yielding to wait for a decision."""
        await self._store.save(entry)
        await self._emit_audit(
            AuditEventKind.INTERRUPT_RAISED,
            entry.interrupt_id,
            {
                "trigger_kind": entry.payload.trigger_kind.value,
                "risk_level":   entry.payload.risk_level.value,
                "thread_id":    entry.payload.thread_id,
            },
        )

    # ── Consumer-side: list / load ──────────────────────────────────

    async def list_pending(
        self, *, limit: int = 100, thread_id: Optional[str] = None,
    ) -> list[CheckpointEntry]:
        """List pending interrupts, optionally filtered by thread."""
        return await self._store.list_pending(limit=limit, thread_id=thread_id)

    async def load(self, interrupt_id: str) -> Optional[CheckpointEntry]:
        return await self._store.load(interrupt_id)

    # ── Operator-side: deliver a decision ───────────────────────────

    async def deliver(
        self, decision: HitlDecision,
    ) -> dict[str, Any]:
        """Validate, persist, and dispatch a decision.

        Returns a dict shaped for transport-layer consumption:
          {"interrupt_id": ..., "outcome": "approve" | ...,
           "result": <resumer return value or None>,
           "already_resolved": bool}

        Raises:
          DecisionValidationError — bad input
          ResumeError              — valid but no resumer & no waiter
        """
        entry = await self._store.load(decision.interrupt_id)
        if entry is None:
            raise DecisionValidationError(
                f"No such interrupt: {decision.interrupt_id}"
            )

        # Idempotent guard for double-clicks: if already resolved,
        # return the previous outcome rather than rejecting. Transports
        # surface this with already_resolved=True so the UI can show
        # "already done" instead of an angry error.
        if entry.state == InterruptState.RESOLVED:
            logger.info(
                "deliver: %s already resolved, returning idempotent ack",
                decision.interrupt_id,
            )
            return {
                "interrupt_id":     decision.interrupt_id,
                "outcome":          (entry.decision.decision.value
                                     if entry.decision else "unknown"),
                "result":           None,
                "already_resolved": True,
            }
        if entry.state != InterruptState.PENDING:
            raise DecisionValidationError(
                f"Interrupt {decision.interrupt_id} is in state "
                f"{entry.state.value}, cannot decide"
            )

        # Structural validation — raises DecisionValidationError
        _validate_decision_against_payload(decision, entry.payload)

        # Stamp thread_id from payload if caller didn't include
        if not decision.thread_id and entry.payload.thread_id:
            decision.thread_id = entry.payload.thread_id

        # Atomically flip PENDING → RESOLVED in the store
        updated = await self._store.mark_resolved(
            decision.interrupt_id, decision,
        )
        if updated is None:
            # Lost the race against another deliver()
            return {
                "interrupt_id":     decision.interrupt_id,
                "outcome":          decision.decision.value,
                "result":           None,
                "already_resolved": True,
            }

        await self._emit_audit(
            AuditEventKind.DECISION_MADE,
            decision.interrupt_id,
            {
                "decision":   decision.decision.value,
                "operator":   decision.operator_id,
                "comment":    decision.comment,
                "patch_keys": (
                    list(decision.parameter_patch.keys())
                    if decision.parameter_patch else []
                ),
            },
        )

        # Dispatch
        logger.info(
            "deliver: dispatching decision %s — interrupt=%s, "
            "resumer_name=%r, waiters_count=%d, batch_id=%s",
            decision.decision.value,
            decision.interrupt_id[:12],
            entry.resume_handle.resumer_name,
            len(self._waiters),
            get_batch_id(entry.payload.context_snapshot),
        )
        result = await self._dispatch(decision, updated)
        logger.info(
            "deliver: dispatch returned — interrupt=%s, result_type=%s, "
            "result_keys=%s",
            decision.interrupt_id[:12],
            type(result).__name__,
            (list(result.keys())[:8] if isinstance(result, dict) else None),
        )
        return {
            "interrupt_id":     decision.interrupt_id,
            "outcome":          decision.decision.value,
            "result":           result,
            "already_resolved": False,
        }

    async def _dispatch(
        self, decision: HitlDecision, entry: CheckpointEntry,
    ) -> Any:
        """Hand the decision to whichever resume mechanism applies.

        Decision routing precedence:
          1. Batch member  → BatchCoordinator.record_decision (the
             producer's batch future resolves once wait_mode is met).
          2. In-process waiter → resolve the future directly.
          3. Named resumer → call the host-registered resumer.
          4. None of the above → log + audit; decision recorded but
             nothing acts on it.
        """
        # Path 0 — batch member: route through coordinator. Note we
        # still proceed to also check waiters/resumers below in case
        # a producer registered both (rare; harmless to try).
        batch_id = get_batch_id(entry.payload.context_snapshot)
        if batch_id:
            try:
                resolution = await self._batch.record_decision(batch_id, decision)
                if resolution is not None:
                    await self._emit_audit(
                        AuditEventKind.GRAPH_RESUMED,
                        decision.interrupt_id,
                        {"path": "batch_resolved", "batch_id": batch_id,
                         "all_approved": resolution.all_approved,
                         "rejected": resolution.rejected},
                    )
                # Don't fall through to in-process / resumer paths for
                # batch members — the coordinator owns resumption.
                return None
            except Exception as exc:
                logger.exception(
                    "BatchCoordinator.record_decision failed: %s", exc,
                )
                # Fall through to other paths so the operator's
                # decision isn't silently dropped on a coordinator bug.

        # Path 1 — in-process waiter (pipeline still alive)
        async with self._waiter_lock:
            waiter = self._waiters.pop(decision.interrupt_id, None)
        if waiter is not None and not waiter.done():
            waiter.set_result(decision)
            await self._emit_audit(
                AuditEventKind.GRAPH_RESUMED,
                decision.interrupt_id,
                {"path": "in_process_waiter"},
            )
            # Caller doesn't get a result here — the pipeline keeps
            # running and reports back through whatever channel the
            # host wired up (e.g. SSE).
            return None

        # Path 2 — named resumer (detached / cross-process)
        resumer_name = entry.resume_handle.resumer_name
        if resumer_name in self._resumers:
            try:
                result = await self._resumers[resumer_name](decision, entry)
                await self._emit_audit(
                    AuditEventKind.GRAPH_RESUMED,
                    decision.interrupt_id,
                    {"path": "resumer", "resumer": resumer_name},
                )
                return result
            except Exception as exc:
                logger.exception(
                    "Resumer %r failed for %s: %s",
                    resumer_name, decision.interrupt_id, exc,
                )
                await self._emit_audit(
                    AuditEventKind.EXECUTION_FAILED,
                    decision.interrupt_id,
                    {"resumer": resumer_name, "error": str(exc)},
                )
                # Return the exception info so transports can render
                # an error to the operator, but don't re-raise — the
                # decision itself was valid; the failure was downstream.
                return {"error": str(exc)}

        # Path 3 — no resume mechanism. The decision is recorded but
        # nothing acts on it. Transports may surface this to operators
        # ("decision saved, manual follow-up required").
        if resumer_name in ("inline", "batch_inline"):
            # Default names set by the pipeline when the producer
            # expected to be alive in-process. Hitting Path 3 with
            # one of these means the pipeline died — log a warning
            # but don't raise.
            logger.warning(
                "deliver: interrupt %s used %s resumer but no in-process "
                "context is alive; decision recorded, no further action.",
                decision.interrupt_id, resumer_name,
            )
            return None

        raise ResumeError(
            f"No resumer registered for {resumer_name!r}, and no in-process "
            f"waiter for {decision.interrupt_id}. Decision was recorded but "
            f"cannot be acted on."
        )

    # ── Batch APIs ──────────────────────────────────────────────────

    async def list_pending_batches(
        self, *, limit: int = 50, thread_id: Optional[str] = None,
    ) -> list[HitlBatch]:
        """Pending batches the UI should render. Sorted newest first."""
        return await self._store.list_pending_batches(
            limit=limit, thread_id=thread_id,
        )

    async def load_batch(self, batch_id: str) -> Optional[BatchSnapshot]:
        """Read-model for the batch card UI: envelope + every child
        entry + how many are decided. Returns None if absent."""
        batch = await self._store.load_batch(batch_id)
        if batch is None:
            return None
        children: list[CheckpointEntry] = []
        decided = 0
        pending = 0
        for iid in batch.interrupt_ids:
            entry = await self._store.load(iid)
            if entry is None:
                continue
            children.append(entry)
            if entry.state == InterruptState.RESOLVED:
                decided += 1
            elif entry.state == InterruptState.PENDING:
                pending += 1
        return BatchSnapshot(
            batch=batch,
            children=children,
            decided_count=decided,
            pending_count=pending,
        )

    async def deliver_batch(
        self, submission: BatchSubmission,
    ) -> dict[str, Any]:
        """Process a BatchSubmission — fan out to per-child deliver().

        Each child decision flows through full validation; failures are
        collected and returned alongside successes. Resolution of the
        producer's batch future happens inside deliver() via the
        BatchCoordinator dispatch path; this method just orchestrates
        the fan-out and aggregates results.

        Returns:
          {"batch_id": ..., "results": [{interrupt_id, outcome, ...}, ...],
           "errors":   [{interrupt_id, error}, ...]}
        """
        batch = await self._store.load_batch(submission.batch_id)
        if batch is None:
            raise DecisionValidationError(
                f"No such batch: {submission.batch_id}"
            )
        # Index incoming decisions by interrupt_id for cheap lookup
        by_id = {d.interrupt_id: d for d in submission.decisions}
        results = []
        errors  = []
        for iid in batch.interrupt_ids:
            d = by_id.get(iid)
            if d is None:
                # No decision supplied for this child — skip. Operators
                # may submit partial batches when wait_mode allows.
                continue
            # Stamp operator_id from submission if child's was unspecified
            if d.operator_id == "unknown":
                d.operator_id = submission.operator_id
            if d.comment is None:
                d.comment = submission.comment
            try:
                result = await self.deliver(d)
                results.append(result)
            except DecisionValidationError as exc:
                errors.append({"interrupt_id": iid, "error": str(exc)})
            except Exception as exc:
                logger.exception("deliver_batch child %s failed: %s", iid, exc)
                errors.append({"interrupt_id": iid, "error": str(exc)})

        await self._emit_audit(
            AuditEventKind.DECISION_MADE,
            submission.batch_id,
            {"event": "batch_submission", "operator": submission.operator_id,
             "decided": len(results), "errors": len(errors)},
        )
        return {
            "batch_id": submission.batch_id,
            "results":  results,
            "errors":   errors,
        }

    # ── Internal ────────────────────────────────────────────────────

    async def _emit_audit(
        self, kind: AuditEventKind, interrupt_id: str, payload: dict[str, Any],
    ) -> None:
        if self._on_audit is None:
            return
        try:
            res = self._on_audit(kind, interrupt_id, payload)
            if inspect.isawaitable(res):
                await res
        except Exception as exc:
            # Audit failures must never break decision delivery
            logger.warning("Audit hook raised %s, swallowing", exc)

    # ── Introspection ───────────────────────────────────────────────

    @property
    def registered_resumers(self) -> list[str]:
        return sorted(self._resumers.keys())

    @property
    def in_flight_waiters(self) -> int:
        return len(self._waiters)