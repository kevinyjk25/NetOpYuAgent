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
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from .batch import BATCH_ID_KEY, BatchCoordinator, BatchResolution
from .schema import (
    AuditEventKind,
    BatchPolicy,
    BatchState,
    BatchWaitMode,
    CheckpointEntry,
    DecisionKind,
    HitlBatch,
    HitlDecision,
    HitlPayload,
    InterruptMode,
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
    can satisfy it when resume_with arrives.

    BUG-01 fix: use get_running_loop() (not deprecated get_event_loop()).
    BUG-03 fix: notify Event so _run_steps wakes immediately instead of
                relying solely on the 50ms poll interval.
    """
    def __init__(self, payload: HitlPayload):
        self.payload = payload
        loop = asyncio.get_running_loop()
        self.future: asyncio.Future[HitlDecision] = loop.create_future()
        # Event set by request_approval() as soon as the waiter is installed,
        # allowing _run_steps to detect it without waiting out the poll timeout.
        self.ready: asyncio.Event = asyncio.Event()

    def mark_ready(self) -> None:
        """Called by PipelineContext.request_approval() once ctx._pending_waiter
        is assigned, so the run loop wakes up immediately."""
        self.ready.set()


class _BatchWaiter:
    """Internal — holds the future a step awaits when it requests batch
    approval. The future resolves once the BatchCoordinator says all
    wait-mode conditions are met (default: all children decided).

    BUG-03 fix: same Event-based immediate-wake pattern as _DecisionWaiter.
    """
    def __init__(self, batch: HitlBatch, future: asyncio.Future[BatchResolution]):
        self.batch  = batch
        self.future = future
        self.ready: asyncio.Event = asyncio.Event()

    def mark_ready(self) -> None:
        self.ready.set()


# ---------------------------------------------------------------------------
# Async HITL (H2-style fire-and-forget) — 2026-05
# ---------------------------------------------------------------------------

@dataclass
class AsyncPendingHitl:
    """Bookkeeping for a fire-and-forget HITL interrupt.

    Caller (skill / tool) creates one by invoking
    `PipelineContext.request_approval_async(...)`. Pipeline does NOT
    await; the caller's continued execution uses `default_value`.

    When the operator decides (or SLA expires), the HitlRouter looks up
    this record and invokes `on_resolved`. The runtime side merges the
    result back into agent state via the inject queue + turn-start hook
    — see hitl_core/DESIGN.md §async HITL merge-back.

    Fields:
        interrupt_id: matches the payload's interrupt_id; primary key
        payload:      original HitlPayload (kept for audit / inspection)
        default_value: what the caller assumed (e.g. "permission_ok")
        on_resolved:  async callback (interrupt_id, decision, default,
                                       diverged) -> None
                      Caller decides how to write the result into agent
                      memory / SSE / etc.
        divergence_check: optional (default, decision) -> bool; True
                      means the actual decision differs from the
                      assumption (triggers the "soft notify" UX). When
                      None, treats `decision.decision != APPROVE` as
                      divergence — i.e. anything not "yes" is a diverge.
        created_at:   for SLA / audit
        sla_seconds:  inherited from payload.sla_seconds
        session_id:   for SSE notify routing (None when ctx has no
                      session — pipeline may run outside web context)
    """
    interrupt_id:     str
    payload:          HitlPayload
    default_value:    Any
    on_resolved:      Callable[[str, Optional["HitlDecision"], Any, bool], Awaitable[None]]
    divergence_check: Optional[Callable[[Any, "HitlDecision"], bool]] = None
    created_at:       datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sla_seconds:      int = 600
    session_id:       Optional[str] = None


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
        # ── Async HITL pending registry (2026-05) ──────────────────────────
        # Maps interrupt_id → AsyncPendingHitl record. The runtime owns the
        # actual on_resolved invocation via the global HitlRouter; we keep
        # this map per-ctx so cleanup happens automatically when the
        # pipeline exits. See hitl_core/DESIGN.md §async HITL.
        self._async_pending: dict[str, "AsyncPendingHitl"] = {}

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

        # Hand off to the run loop and wait.
        # mark_ready() is called AFTER _pending_waiter is set so the run loop
        # cannot observe a partial state (waiter set but event not fired).
        waiter = _DecisionWaiter(payload)
        self._pending_waiter = waiter
        waiter.mark_ready()   # BUG-03: wake _run_steps immediately
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

    # ── Async approval (H2-style fire-and-forget) ────────────────────────

    async def request_approval_async(
        self,
        payload: HitlPayload,
        *,
        default_value: Any,
        on_resolved: Callable[
            [str, Optional[HitlDecision], Any, bool],
            Awaitable[None],
        ],
        divergence_check: Optional[
            Callable[[Any, HitlDecision], bool]
        ] = None,
        session_id: Optional[str] = None,
        resume_handle: Optional[ResumeHandle] = None,
    ) -> tuple[str, Any]:
        """Fire-and-forget HITL — does NOT block.

        Used for H2-style external delegation. Pipeline persists the
        checkpoint + audits the delegation, then returns immediately
        with the caller-supplied default. The real operator decision
        (if any) arrives later via the HitlRouter, which looks up the
        pending registry and invokes `on_resolved`.

        Returns:
            (interrupt_id, default_value): caller proceeds with default
            and remembers interrupt_id for tracing.

        Args:
            payload:        the HitlPayload. interrupt_mode is forced to
                            ASYNC_NONBLOCKING here; trigger_kind should
                            be EXTERNAL_DELEGATION (or whatever maps to
                            the H2 UX in your front-end).
            default_value:  what the caller assumes (e.g. "permission_ok").
                            Returned immediately AND passed to on_resolved
                            so the callback can compute divergence.
            on_resolved:    coroutine called when operator decides OR
                            SLA expires (with decision=None in the latter
                            case). Signature:
                                async def cb(interrupt_id, decision,
                                             default_value, diverged): ...
                            Caller decides how to merge into agent state.
            divergence_check: optional (default, decision) -> bool. When
                            omitted, divergence = (decision.decision !=
                            APPROVE). The result is passed as the 4th arg
                            of on_resolved.
            session_id:     for SSE notification routing. Pipeline passes
                            this through to AsyncPendingHitl so the
                            HitlRouter / runtime can find the right SSE
                            stream when the operator decides.
            resume_handle:  unused for async flow (kept for symmetry).
                            Async resolution goes through on_resolved.

        See hitl_core/DESIGN.md §async HITL for the merge-back semantics
        and runtime/loop.py for the turn-start drain that surfaces async
        results into LLM context.
        """
        # Force mode regardless of payload field — caller may have passed
        # SYNC_BLOCKING by mistake.
        payload.interrupt_mode = InterruptMode.ASYNC_NONBLOCKING

        # Stamp identity from current state so payload.thread_id is consistent
        if not payload.thread_id:
            payload.thread_id = self.state.thread_id
        if not payload.context_id:
            payload.context_id = self.state.thread_id
        if payload.task_id is None:
            payload.task_id = self.state.task_id

        # Persist checkpoint so /hitl/pending lists it, just like sync interrupts
        entry = CheckpointEntry(
            interrupt_id=payload.interrupt_id,
            payload=payload,
            resume_handle=resume_handle or ResumeHandle(
                resumer_name="async_hitl",   # router looks up AsyncPendingHitl
                state={"pipeline_id": self.state.pipeline_id},
            ),
        )
        await self._store.save(entry)

        # Register in pending so the router can find the on_resolved cb.
        pending = AsyncPendingHitl(
            interrupt_id     = payload.interrupt_id,
            payload          = payload,
            default_value    = default_value,
            on_resolved      = on_resolved,
            divergence_check = divergence_check,
            sla_seconds      = payload.sla_seconds,
            session_id       = session_id,
        )
        self._async_pending[payload.interrupt_id] = pending
        # Register with the global router via the unified helper, which
        # inserts under the registry lock AND arms the SLA watchdog. This
        # replaces the previous direct `_async_registry[...] = pending` +
        # bespoke local timer, so every async-HITL path shares one
        # ownership/timeout mechanism (no double-fire, no leaked entry).
        try:
            from .router import register_async_pending
            register_async_pending(
                pending, store=self._store, on_audit=self._on_audit,
            )
        except Exception as _reg_exc:
            logger.warning(
                "request_approval_async: failed to register %s in "
                "global async registry (%s) — operator decisions won't "
                "route back to on_resolved",
                payload.interrupt_id, _reg_exc,
            )

        # Audit the delegation
        await self.audit(
            AuditEventKind.ASYNC_DELEGATED,
            payload.interrupt_id,
            {
                "trigger_kind":  payload.trigger_kind.value,
                "risk_level":    payload.risk_level.value,
                "sla_seconds":   payload.sla_seconds,
                "session_id":    session_id,
            },
        )
        logger.info(
            "request_approval_async: delegated %s (trigger=%s, sla=%ds)",
            payload.interrupt_id, payload.trigger_kind.value,
            payload.sla_seconds,
        )

        # SLA timeout is now armed by register_async_pending() above (the
        # unified watchdog claims ownership via claim_async_pending so it can
        # never double-fire with an operator decision). No bespoke local
        # timer here anymore — see hitl_core/router.py:register_async_pending.
        return payload.interrupt_id, default_value

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
        waiter.mark_ready()   # BUG-03: wake _run_steps immediately
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

    async def run(
        self, state: PipelineState, *, poll_interval_ms: int = 50
    ) -> AsyncIterator[dict[str, Any]]:
        """Drive the pipeline. Yields events; consumer calls resume_with
        on every "interrupt" event.

        Args:
            poll_interval_ms: Safety-net poll interval for the step waiter loop.
                              Set via config.yaml concurrency.hitl_pipeline_poll_interval_ms.
                              The primary wake mechanism is Event-based (immediate);
                              this only fires when the Event path is unavailable.
        """
        ctx = PipelineContext(
            state=state, store=self._store,
            on_audit=self._on_audit, batch_coordinator=self._batch,
        )
        self._active[state.pipeline_id] = ctx
        try:
            async for event in self._run_steps(ctx, poll_interval_ms=poll_interval_ms):
                yield event
        finally:
            self._active.pop(state.pipeline_id, None)

    async def _run_steps(
        self, ctx: PipelineContext, poll_interval_ms: int = 50
    ) -> AsyncIterator[dict[str, Any]]:
        """Drive all registered steps, surfacing HITL interrupts as they occur.

        BUG-03 fix: Instead of a hard-coded 50ms busy-poll, the loop waits on
        an asyncio.Event that the waiter sets as soon as it is installed.  The
        poll_interval_ms is kept as a safety-net fallback (prevents livelock
        if mark_ready() is somehow missed), but the common path wakes up
        immediately without spinning.

        The poll interval is read from AppConfig at pipeline construction time
        and passed in here so it can be tuned in config.yaml without code changes.
        """
        for name, step in self._steps:
            logger.debug("Pipeline step: %s", name)
            # Run the step in a task so we can intercept request_approval
            # which pauses on an asyncio.Future.
            step_task = asyncio.create_task(step(ctx))
            poll_timeout = poll_interval_ms / 1000.0   # convert ms → seconds
            while not step_task.done():
                # Build a combined "wake me" set: the step task itself, plus
                # any pending waiter's ready Event.  This way we wake as soon
                # as the step installs a waiter — no spin.
                wake_futs: set = {step_task}
                if ctx._pending_waiter is not None:
                    wake_futs.add(
                        asyncio.ensure_future(ctx._pending_waiter.ready.wait())
                    )
                elif ctx._pending_batch is not None:
                    wake_futs.add(
                        asyncio.ensure_future(ctx._pending_batch.ready.wait())
                    )

                done, pending_tasks = await asyncio.wait(
                    wake_futs, timeout=poll_timeout, return_when=asyncio.FIRST_COMPLETED
                )
                # Cancel any helper futures we spawned so they don't leak
                for t in pending_tasks:
                    if t is not step_task:
                        t.cancel()

                if step_task in done:
                    break

                if ctx._pending_waiter is not None:
                    # Step is awaiting an operator decision. Yield the
                    # interrupt to the caller and wait until resume_with
                    # delivers a decision via the future.
                    waiter = ctx._pending_waiter
                    yield {
                        "type":    "interrupt",
                        "payload": waiter.payload,
                    }
                    # Wait for the operator's decision; step resumes automatically
                    # because its `await waiter.future` completes.
                    await waiter.future
                    # Don't break — keep looping until step_task done.

                elif ctx._pending_batch is not None:
                    # Step is awaiting a batch resolution. Yield the batch
                    # envelope so the caller can surface the multi-approval UI.
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


    # ── Restart-recovery (DESIGN-01 fix) ─────────────────────────────────

    async def recover_pending(
        self,
        *,
        thread_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """DESIGN-01 fix: on process restart, load all PENDING interrupt
        checkpoints from the store and return them as interrupt-event dicts
        so the caller (e.g. webui/backend.py on startup) can surface them
        to operators.

        This does NOT resume the original pipeline coroutine (that is gone
        with the old process). Instead it returns the stored payloads so the
        UI can re-display them for re-approval.  When the operator decides,
        the decision is written to the store (mark_resolved), and the caller
        is responsible for re-dispatching via the registered ResumeHandle.

        Usage at startup::

            pipeline = HitlPipeline(store=store)
            for event in await pipeline.recover_pending():
                # surface event["payload"] in the UI
                ...

        Returns a list of interrupt-event dicts, newest first.
        """
        pending = await self._store.list_pending(limit=limit, thread_id=thread_id)
        events = []
        for entry in pending:
            events.append({
                "type":            "interrupt",
                "payload":         entry.payload,
                "recovered":       True,          # flag: from store, not live pipeline
                "interrupt_id":    entry.payload.interrupt_id,
                "registered_at":   entry.registered_at.isoformat(),
                "resumer_name":    entry.resume_handle.resumer_name if entry.resume_handle else None,
            })
            logger.info(
                "recover_pending: surfacing orphaned interrupt %s (thread=%s, registered=%s)",
                entry.payload.interrupt_id,
                entry.payload.thread_id,
                entry.registered_at.isoformat(),
            )
        return events

    async def expire_overdue_interrupts(self) -> int:
        """Sweep PENDING interrupts past their SLA deadline and transition
        them to EXPIRED.  Safe to call periodically (e.g. from a background
        task in main.py) to prevent the pending list growing without bound
        after process restarts.

        Returns the number of entries expired.
        """
        return await self._store.expire_overdue()

    async def list_orphaned_interrupts(
        self, *, thread_id: Optional[str] = None, limit: int = 50
    ) -> list:
        """Return PENDING checkpoints that have no matching in-process waiter.
        Useful for health-check endpoints: if this list is non-empty after
        startup, it means a previous process died with pending approvals.
        """
        pending = await self._store.list_pending(limit=limit, thread_id=thread_id)
        orphans = []
        for entry in pending:
            # Check if any active context owns this interrupt_id
            is_live = any(
                ctx._pending_waiter is not None
                and ctx._pending_waiter.payload.interrupt_id == entry.payload.interrupt_id
                for ctx in self._active.values()
            )
            if not is_live:
                orphans.append(entry)
        return orphans

    @property
    def active_count(self) -> int:
        """Number of pipelines currently in-flight in this process."""
        return len(self._active)