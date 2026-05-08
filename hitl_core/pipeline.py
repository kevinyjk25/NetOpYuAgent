"""
hitl_core.pipeline — Linear async pipeline replacing LangGraph StateGraph.

Why this exists:
  LangGraph's interrupt() is built around a state-machine with a single
  pause point per run. Recursive / nested interrupts (operator approves
  intent → agent decides which tool → operator approves the tool's
  parameters) require workarounds. A linear async pipeline trivially
  supports nested interrupts — each `await ctx.request_approval(...)`
  is just a normal coroutine pause. The whole call stack lives in
  process memory, debugging is straightforward, and there's no graph
  state to serialise.

Architecture:

  Caller                 Pipeline                  Store
  ──────                 ────────                  ─────
  run_with_hitl(state) ─►pipeline.start()
                         ├─ classify
                         ├─ preverify
                         ├─ plan_step ────────────►save(checkpoint)
                         │   └─ if HITL: yield ────────────────► UI
                         │       interrupt event
                         │       ↑ (caller awaits decision)
                         │       └─ resume_with(decision) ──────► load + decide
                         ├─ execute_step
                         └─ done

  • Pipeline is an async generator: yields HITL interrupt events and
    streaming tokens; returns a final result dict.
  • `request_approval` is the single primitive — it raises a sentinel
    exception (PipelinePaused) caught by the run loop, which yields the
    interrupt to the caller and suspends. The caller delivers the
    operator's decision via resume_with(); the pipeline picks back up
    where it left off.
  • Nested HITL works because each request_approval is just an await
    on an asyncio.Future.

Caveats:
  • Pause/resume relies on the pipeline's state living in memory between
    events. For multi-replica deployments, the host can flush state to
    the checkpoint store after every yield and reconstitute on resume —
    see ResumeHandle in schema.py.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from .batch import BATCH_ID_KEY, BatchCoordinator, BatchResolution
from .schema import (
    AuditEventKind,
    BatchPolicy,
    BatchState,
    BatchWaitMode,
    CheckpointEntry,
    HitlBatch,
    HitlDecision,
    HitlPayload,
    InterruptState,
    ResumeHandle,
)
from .store import BaseCheckpointStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline state — replaces LangGraph's typed dict state
# ---------------------------------------------------------------------------

@dataclass
class PipelineState:
    """Mutable workspace passed through pipeline steps.

    Steps read/write fields directly; this is intentional — the alternative
    (immutable state with copy-on-write) is much more verbose for a
    procedural pipeline and gives no real safety benefit when each step
    is async-sequential.
    """
    # Identity
    pipeline_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    thread_id: str = ""
    task_id: Optional[str] = None

    # Input
    user_query: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    # Working data — populated as the pipeline progresses
    classification: Optional[dict[str, Any]] = None
    preverify_result: Optional[dict[str, Any]] = None
    plan: Optional[dict[str, Any]] = None
    risk_assessment: Optional[dict[str, Any]] = None
    execution_result: Optional[dict[str, Any]] = None

    # Streaming
    tokens: list[str] = field(default_factory=list)

    # HITL bookkeeping
    decisions: list[HitlDecision] = field(default_factory=list)
    aborted: bool = False
    abort_reason: str = ""


# ---------------------------------------------------------------------------
# Step protocol
# ---------------------------------------------------------------------------

# Each step is an async callable: (PipelineContext) -> None.
# Steps mutate ctx.state directly; raise PipelineAborted to short-circuit.
#
# Steps may call `await ctx.request_approval(payload)` to pause and wait
# for an operator decision. The result is the HitlDecision; the step
# decides what to do with it (proceed / abort / branch).

PipelineStep = Callable[["PipelineContext"], Awaitable[None]]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PipelineAborted(Exception):
    """Raised by a step to terminate the pipeline cleanly with a reason."""
    def __init__(self, reason: str, *, kind: str = "aborted"):
        super().__init__(reason)
        self.reason = reason
        self.kind = kind


class _DecisionWaiter:
    """Internal — holds the future a step is awaiting, so the run loop
    can satisfy it when resume_with arrives."""
    def __init__(self, payload: HitlPayload):
        self.payload = payload
        self.future: asyncio.Future[HitlDecision] = asyncio.get_event_loop().create_future()


class _BatchWaiter:
    """Internal — holds the future a step awaits when it requests batch
    approval. The future resolves once the BatchCoordinator says all
    wait-mode conditions are met (default: all children decided)."""
    def __init__(self, batch: HitlBatch, future: asyncio.Future[BatchResolution]):
        self.batch  = batch
        self.future = future


# ---------------------------------------------------------------------------
# PipelineContext — the API steps see
# ---------------------------------------------------------------------------

class PipelineContext:
    """Runtime context passed to each step. Provides:
      • `state`              — the mutable PipelineState
      • `request_approval`   — pause and wait for an operator decision
      • `emit_token`         — stream a token to the caller
      • `audit`              — log an audit event
    """

    def __init__(
        self,
        state: PipelineState,
        store: BaseCheckpointStore,
        on_audit: Optional[Callable[[AuditEventKind, str, dict[str, Any]], Awaitable[None]]] = None,
        on_token: Optional[Callable[[str], Awaitable[None]]] = None,
        batch_coordinator: Optional[BatchCoordinator] = None,
    ):
        self.state = state
        self._store = store
        self._on_audit = on_audit
        self._on_token = on_token
        self._batch = batch_coordinator
        # Set by the run loop when a step calls request_approval; the
        # loop reads this to know what to yield.
        self._pending_waiter: Optional[_DecisionWaiter] = None
        self._pending_batch: Optional[_BatchWaiter] = None

    # ── Public API ────────────────────────────────────────────────────

    async def request_approval(
        self,
        payload: HitlPayload,
        *,
        resume_handle: Optional[ResumeHandle] = None,
    ) -> HitlDecision:
        """Pause the pipeline, surface an interrupt to the caller, and
        wait for the operator's decision. Returns the decision when
        delivered via the run loop's resume_with().

        Nested approvals are first-class: just await this again later.
        """
        # Stamp identity from current state so payload.thread_id is consistent
        if not payload.thread_id:
            payload.thread_id = self.state.thread_id
        if not payload.context_id:
            payload.context_id = self.state.thread_id
        if payload.task_id is None:
            payload.task_id = self.state.task_id

        # Persist the checkpoint before yielding control
        entry = CheckpointEntry(
            interrupt_id=payload.interrupt_id,
            payload=payload,
            resume_handle=resume_handle or ResumeHandle(
                resumer_name="inline",     # caller resumes via run loop directly
                state={"pipeline_id": self.state.pipeline_id},
            ),
        )
        await self._store.save(entry)
        await self.audit(
            AuditEventKind.INTERRUPT_RAISED,
            payload.interrupt_id,
            {"trigger_kind": payload.trigger_kind.value,
             "risk_level":   payload.risk_level.value},
        )

        # Hand off to the run loop and wait
        waiter = _DecisionWaiter(payload)
        self._pending_waiter = waiter
        try:
            decision = await waiter.future
        finally:
            self._pending_waiter = None

        self.state.decisions.append(decision)
        await self.audit(
            AuditEventKind.DECISION_MADE,
            payload.interrupt_id,
            {"decision": decision.decision.value,
             "operator": decision.operator_id},
        )
        return decision

    # ── Batch approval ───────────────────────────────────────────────

    async def request_batch_approval(
        self,
        payloads: list[HitlPayload],
        *,
        title: str = "",
        description: str = "",
        policy: BatchPolicy = BatchPolicy.BEST_EFFORT,
        wait_mode: BatchWaitMode = BatchWaitMode.ALL,
        threshold_count: Optional[int] = None,
        sla_seconds: int = 1800,
        resume_handle: Optional[ResumeHandle] = None,
    ) -> BatchResolution:
        """Submit N independent HITL interrupts as a single operator
        action. Pauses the pipeline until the batch's wait_mode is
        satisfied; returns the BatchResolution carrying every child
        decision in submission order.

        This is the right primitive for "approve all 5 of these similar
        actions" workflows — the operator sees one card with 5 rows
        rather than 5 sequential cards. Children are independent: a
        child's decision does not affect the others' state.

        For state-dependent sequential approvals (Approve A, then based
        on A pick B, ...) call request_approval one at a time. The
        await pattern handles nested HITLs naturally; batches are for
        parallel-independent flows.

        Args:
          payloads: the children. Each gets an interrupt_id, persisted
                    in the store, and indexed under the new batch.
          title / description: shown to operators in the batch card UI.
          policy: BEST_EFFORT (each child independent) or
                  ALL_OR_NOTHING (any reject fails the batch).
          wait_mode: ALL (every child decided), THRESHOLD (N decided),
                     STREAMING (yield-each — reserved).
          threshold_count: required when wait_mode=THRESHOLD.
          sla_seconds: batch-level SLA. Children inherit this unless
                       they have their own sla_seconds set.
          resume_handle: detached-resume spec; rarely needed for
                         in-process pipelines.

        Returns:
          BatchResolution with .decisions (in submission order),
          .for_interrupt(id), .rejected, .all_approved.
        """
        if self._batch is None:
            raise RuntimeError(
                "PipelineContext has no BatchCoordinator wired up. "
                "Construct HitlPipeline with batch=True (default) or "
                "pass a BatchCoordinator explicitly."
            )
        if not payloads:
            raise ValueError("request_batch_approval requires at least one payload")

        # Stamp identity + cross-link each child to the batch
        batch = HitlBatch(
            thread_id=self.state.thread_id,
            task_id=self.state.task_id,
            policy=policy,
            wait_mode=wait_mode,
            threshold_count=threshold_count,
            title=title,
            description=description,
            sla_seconds=sla_seconds,
        )

        for p in payloads:
            if not p.thread_id:
                p.thread_id = self.state.thread_id
            if not p.context_id:
                p.context_id = self.state.thread_id
            if p.task_id is None:
                p.task_id = self.state.task_id
            # Inherit batch SLA when child didn't set its own
            if p.sla_seconds <= 0:
                p.sla_seconds = sla_seconds
            # Stamp batch id into context_snapshot so the router can
            # detect "this decision is part of a batch" without a
            # separate lookup.
            p.context_snapshot[BATCH_ID_KEY] = batch.batch_id
            batch.interrupt_ids.append(p.interrupt_id)

        # Persist children first, then the envelope. If any child save
        # fails we abort with a clean RuntimeError before the batch is
        # registered (no partial state in the store).
        for p in payloads:
            entry = CheckpointEntry(
                interrupt_id=p.interrupt_id,
                payload=p,
                resume_handle=resume_handle or ResumeHandle(
                    resumer_name="batch_inline",
                    state={"pipeline_id": self.state.pipeline_id,
                           "batch_id":    batch.batch_id},
                ),
            )
            await self._store.save(entry)
            await self.audit(
                AuditEventKind.INTERRUPT_RAISED,
                p.interrupt_id,
                {"trigger_kind": p.trigger_kind.value,
                 "batch_id":     batch.batch_id},
            )

        # Open batch — coordinator stores the envelope and returns a
        # future the pipeline awaits.
        future = await self._batch.open_batch(batch)
        waiter = _BatchWaiter(batch=batch, future=future)
        self._pending_batch = waiter
        try:
            resolution = await future
        finally:
            self._pending_batch = None

        # Append all child decisions to state.decisions in order
        for d in resolution.decisions:
            self.state.decisions.append(d)
        return resolution

    async def emit_token(self, token: str) -> None:
        """Stream a token to the caller (e.g. LLM partial output)."""
        self.state.tokens.append(token)
        if self._on_token:
            await self._on_token(token)

    async def audit(
        self, kind: AuditEventKind, interrupt_id: str, payload: dict[str, Any],
    ) -> None:
        if self._on_audit:
            try:
                await self._on_audit(kind, interrupt_id, payload)
            except Exception as exc:
                # Audit failures must never break the pipeline
                logger.warning("Audit hook raised %s, swallowing", exc)


# ---------------------------------------------------------------------------
# Pipeline — the orchestrator
# ---------------------------------------------------------------------------

class HitlPipeline:
    """Composable async pipeline with first-class HITL.

    Construction:
      pipeline = HitlPipeline(store=store)
      pipeline.add_step("classify",   classify_step)
      pipeline.add_step("preverify",  preverify_step)
      pipeline.add_step("plan",       plan_step)
      pipeline.add_step("execute",    execute_step)

    Execution:
      async for event in pipeline.run(state):
          # event is one of:
          #   {"type": "token", "token": "..."}        — streaming chunk
          #   {"type": "interrupt", "payload": ...}    — HITL pause
          #   {"type": "done", "state": ...}           — pipeline finished
          #   {"type": "aborted", "reason": ...}       — step aborted
          if event["type"] == "interrupt":
              decision = await fetch_operator_decision(event["payload"])
              await pipeline.resume_with(decision)
    """

    def __init__(
        self,
        *,
        store: BaseCheckpointStore,
        batch_coordinator: Optional[BatchCoordinator] = None,
        on_audit: Optional[Callable[[AuditEventKind, str, dict[str, Any]], Awaitable[None]]] = None,
    ):
        self._store = store
        # Auto-construct a BatchCoordinator unless caller passed one or
        # explicitly None (rare — there's almost no reason to disable
        # batching). The coordinator shares the audit hook.
        self._batch = (
            batch_coordinator
            if batch_coordinator is not None
            else BatchCoordinator(store=store, on_audit=on_audit)
        )
        self._on_audit = on_audit
        self._steps: list[tuple[str, PipelineStep]] = []
        # Active runs keyed by pipeline_id, so resume_with can find them
        self._active: dict[str, PipelineContext] = {}

    @property
    def batch(self) -> BatchCoordinator:
        """Expose the coordinator so transports can route batch decisions
        without reaching into private state."""
        return self._batch

    # ── Construction ─────────────────────────────────────────────────

    def add_step(self, name: str, fn: PipelineStep) -> "HitlPipeline":
        """Register a step. Steps run in registration order. `name` is
        used in logging only; not enforced unique."""
        self._steps.append((name, fn))
        return self

    # ── Execution ────────────────────────────────────────────────────

    async def run(self, state: PipelineState) -> AsyncIterator[dict[str, Any]]:
        """Drive the pipeline. Yields events; consumer calls resume_with
        on every "interrupt" event."""
        ctx = PipelineContext(
            state=state, store=self._store,
            on_audit=self._on_audit, batch_coordinator=self._batch,
        )
        self._active[state.pipeline_id] = ctx
        try:
            async for event in self._run_steps(ctx):
                yield event
        finally:
            self._active.pop(state.pipeline_id, None)

    async def _run_steps(self, ctx: PipelineContext) -> AsyncIterator[dict[str, Any]]:
        for name, step in self._steps:
            logger.debug("Pipeline step: %s", name)
            # Run the step in a task so we can intercept request_approval
            # which pauses on an asyncio.Future.
            step_task = asyncio.create_task(step(ctx))
            while not step_task.done():
                # Wait briefly — step is doing work or about to pause.
                # We use a short sleep race with the task so we can detect
                # request_approval calls promptly.
                done, _pending = await asyncio.wait(
                    {step_task}, timeout=0.05,
                )
                if step_task in done:
                    break
                if ctx._pending_waiter is not None:
                    # Step is awaiting an operator decision. Yield the
                    # interrupt to the caller and pause until resume_with
                    # delivers a decision.
                    waiter = ctx._pending_waiter
                    yield {
                        "type":    "interrupt",
                        "payload": waiter.payload,
                    }
                    # Now wait for the future to resolve (resume_with
                    # sets it). The step itself will pick back up because
                    # its `await waiter.future` completes.
                    await waiter.future
                    # Don't break — keep looping until step_task done.
                elif ctx._pending_batch is not None:
                    # Step is awaiting a batch resolution. Yield the
                    # batch envelope so the caller can drive the
                    # batch UI / route batch decisions.
                    bw = ctx._pending_batch
                    yield {
                        "type":  "batch_interrupt",
                        "batch": bw.batch,
                    }
                    await bw.future
                    # Don't break — keep looping until step_task done.

            # Step complete (or raised)
            try:
                await step_task    # re-raise step exception, if any
            except PipelineAborted as exc:
                ctx.state.aborted     = True
                ctx.state.abort_reason = exc.reason
                yield {"type": "aborted", "reason": exc.reason, "kind": exc.kind}
                return
            except Exception as exc:
                logger.exception("Pipeline step %s failed: %s", name, exc)
                yield {
                    "type":   "aborted",
                    "reason": f"{name}: {exc}",
                    "kind":   "step_failure",
                }
                return

            # Drain any tokens the step produced (already streamed via
            # ctx.emit_token; the on_token hook handles real-time delivery
            # so we don't re-yield here).

        # All steps done
        yield {"type": "done", "state": ctx.state}

    # ── Resume ───────────────────────────────────────────────────────

    async def resume_with(
        self, decision: HitlDecision,
    ) -> Optional[CheckpointEntry]:
        """Deliver an operator's decision to the paused pipeline.

        Looks up the pending waiter by interrupt_id, validates state via
        the store, and resolves the future. Returns the updated
        CheckpointEntry, or None if no matching pending interrupt was
        found.

        Idempotent guard: if the entry is already RESOLVED, returns it
        unchanged without raising — useful when the UI double-clicks
        Approve and the second POST arrives a beat later.
        """
        # Ledger update first — atomic state transition lives in the store
        entry = await self._store.mark_resolved(
            decision.interrupt_id, decision,
        )
        if entry is None:
            # Either the interrupt didn't exist or wasn't pending. Look
            # up to figure out which case.
            existing = await self._store.load(decision.interrupt_id)
            if existing is None:
                logger.warning("resume_with: no such interrupt %s",
                               decision.interrupt_id)
                return None
            if existing.state == InterruptState.RESOLVED:
                logger.info("resume_with: interrupt %s already resolved (idempotent)",
                            decision.interrupt_id)
                return existing
            logger.warning(
                "resume_with: interrupt %s in unexpected state %s",
                decision.interrupt_id, existing.state.value,
            )
            return existing

        # Find the active context whose pending waiter matches and
        # resolve its future. Without an active context, the pipeline
        # ran in another process — caller must rebuild via ResumeHandle.
        for pipeline_id, ctx in self._active.items():
            if (
                ctx._pending_waiter is not None
                and ctx._pending_waiter.payload.interrupt_id == decision.interrupt_id
            ):
                if not ctx._pending_waiter.future.done():
                    ctx._pending_waiter.future.set_result(decision)
                return entry
        logger.info(
            "resume_with: interrupt %s resolved in store but no in-process "
            "waiter — caller must reconstitute via ResumeHandle %s",
            decision.interrupt_id, entry.resume_handle.resumer_name,
        )
        return entry

    # ── Introspection ─────────────────────────────────────────────────

    @property
    def active_count(self) -> int:
        """Number of pipelines currently in-flight in this process."""
        return len(self._active)