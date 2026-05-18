"""
hitl_core.batch — Batch approval coordinator.

A pipeline can call ctx.request_batch_approval(batch) to submit N
HITL interrupts as a single operator action. The BatchCoordinator
tracks per-child decisions, applies the batch's policy, and resolves
the producer's awaitable when the wait_mode condition is met.

Architecture:

    Pipeline                 BatchCoordinator              Store
    ──────────               ────────────────              ─────
    request_batch_approval ─► open_batch(N children) ─────► save_batch
                              │                          │  save (each child)
                              ▼                          │
                              register N waiters         │
                              │                          │
                              ▼                          │
    [pipeline pauses]         await batch_future
                                                      [UI fetches]
                                                      [operator submits BatchSubmission]
                                                      [transport calls deliver_batch]
                              ◄── deliver child 1 ────── mark_resolved
                              │   ...                     ...
                              ◄── deliver child N ────── mark_resolved
                              ▼
                              policy + wait_mode satisfied
                              ▼
                              resolve batch_future
                              update batch.state = RESOLVED
                                                      [pipeline resumes with
                                                       all decisions]

Wait modes (only ALL implemented in v0; THRESHOLD/STREAMING reserved):

  • ALL — every child decided (any DecisionKind) → resolve
  • THRESHOLD — N approvals or N rejections → resolve (future)
  • STREAMING — yield each decision as it arrives (future)

Policies:

  • BEST_EFFORT — every child decision flows back; pipeline decides
    individually what to do with each result
  • ALL_OR_NOTHING — any REJECT in any child marks the batch as
    rejected; pipeline gets a flag to abort cleanly
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from .schema import (
    AuditEventKind,
    BatchPolicy,
    BatchState,
    BatchWaitMode,
    DecisionKind,
    HitlBatch,
    HitlDecision,
    InterruptState,
)
from .store import BaseCheckpointStore

logger = logging.getLogger(__name__)


class BatchResolution:
    """The result delivered to the pipeline when a batch resolves.

    Carries every child decision in submission order, plus a derived
    summary so the pipeline can branch quickly on common cases without
    iterating.
    """
    def __init__(
        self,
        batch: HitlBatch,
        decisions_by_id: dict[str, HitlDecision],
        rejected: bool,
        all_approved: bool,
    ):
        self.batch = batch
        self.decisions_by_id = decisions_by_id
        self.rejected = rejected             # any child REJECT (or expired without decision)
        self.all_approved = all_approved      # every child APPROVE / EDIT / CHOOSE / ANSWER

    def for_interrupt(self, interrupt_id: str) -> Optional[HitlDecision]:
        return self.decisions_by_id.get(interrupt_id)

    @property
    def decisions(self) -> list[HitlDecision]:
        # Preserve submission order via batch.interrupt_ids
        return [
            self.decisions_by_id[i]
            for i in self.batch.interrupt_ids
            if i in self.decisions_by_id
        ]

    def __repr__(self) -> str:
        return (
            f"BatchResolution(batch={self.batch.batch_id[:8]}, "
            f"n={len(self.decisions_by_id)}, rejected={self.rejected}, "
            f"all_approved={self.all_approved})"
        )


class BatchCoordinator:
    """Single-instance coordinator wiring batches to in-process producers.

    Lifetime: one per HitlRouter / pipeline runtime. The coordinator owns
    the in-memory waiter table — the store owns persistent state.

    Multi-replica behaviour (current state):
      A child decision arriving on a DIFFERENT replica from the one that
      opened the batch falls through to the resumer-by-name path the same
      way unbatched detached decisions do (see HitlRouter._dispatch). The
      child gets persisted via the shared store, but the original
      producer's future on the other replica won't be woken — it will
      time out or hang.

      For now the supported deployment is single-replica. Running
      multiple replicas requires a fan-out layer to deliver child
      decisions back to the producer replica. Two practical options:

        1. Store polling — _check_wait_condition gets called from a
           periodic task that walks pending batches in the store and
           re-evaluates them when new children appear. Simple, no new
           dependencies, ~30s decision-detect latency.

        2. Pubsub (Redis or NATS) — when on_child_decision runs on any
           replica, publish (batch_id, interrupt_id) to a channel. Each
           replica subscribes and dispatches inbound msgs through its
           local on_child_decision. Sub-second latency but adds an infra
           dependency and a deduplication concern (skip self-published).

      Neither is wired today; doing so should remain a separate, focused
      patch behind an `enable_multi_replica_sync: true` config flag.
    """

    def __init__(self, *, store: BaseCheckpointStore,
                 on_audit=None) -> None:
        self._store = store
        self._on_audit = on_audit
        # batch_id → asyncio.Future[BatchResolution]
        self._waiters: dict[str, asyncio.Future[BatchResolution]] = {}
        # batch_id → {interrupt_id: HitlDecision}, accumulating decisions
        self._collected: dict[str, dict[str, HitlDecision]] = {}
        self._lock = asyncio.Lock()

    # ── Producer side ────────────────────────────────────────────────

    async def open_batch(
        self,
        batch: HitlBatch,
    ) -> asyncio.Future[BatchResolution]:
        """Persist the batch envelope and create a future the producer
        awaits. The pipeline's request_batch_approval calls this after
        having saved each child interrupt via the regular store.save.

        Returns the awaitable. When all wait conditions are satisfied,
        the future resolves to a BatchResolution.
        """
        async with self._lock:
            if batch.batch_id in self._waiters:
                # Re-open of an existing batch — return the existing future
                return self._waiters[batch.batch_id]
            fut: asyncio.Future[BatchResolution] = asyncio.get_running_loop().create_future()
            self._waiters[batch.batch_id] = fut
            self._collected[batch.batch_id] = {}

        await self._store.save_batch(batch)
        await self._emit_audit(
            AuditEventKind.INTERRUPT_RAISED,
            batch.batch_id,
            {"event": "batch_opened", "size": len(batch.interrupt_ids),
             "policy": batch.policy.value, "wait_mode": batch.wait_mode.value},
        )
        return fut

    async def cancel_batch(self, batch_id: str, reason: str = "") -> None:
        """Producer-initiated cancellation. Marks state=CANCELLED, resolves
        the future with whatever decisions arrived so far."""
        batch = await self._store.load_batch(batch_id)
        if batch is None:
            return
        async with self._lock:
            collected = self._collected.pop(batch_id, {})
            fut = self._waiters.pop(batch_id, None)
        batch.state = BatchState.CANCELLED
        batch.resolved_at = datetime.now(timezone.utc)
        batch.metadata["cancel_reason"] = reason
        await self._store.save_batch(batch)
        if fut is not None and not fut.done():
            fut.set_result(BatchResolution(
                batch=batch,
                decisions_by_id=collected,
                rejected=True,
                all_approved=False,
            ))

    # ── Consumer (decision) side ─────────────────────────────────────

    async def record_decision(
        self,
        batch_id: str,
        decision: HitlDecision,
    ) -> Optional[BatchResolution]:
        """Called by the router when a child decision is delivered for
        an interrupt that belongs to a batch. Returns a BatchResolution
        if this decision completes the batch; None otherwise.

        Idempotent: re-recording the same child decision is a no-op.
        """
        async with self._lock:
            collected = self._collected.get(batch_id)
            if collected is None:
                # Batch not actively tracked here (likely on a different
                # replica). The store has the decision; coordinator on
                # the right replica will pick it up.
                logger.debug(
                    "record_decision: batch %s not tracked locally", batch_id,
                )
                return None
            if decision.interrupt_id in collected:
                # Already recorded — keep the first one (idempotent)
                return None
            collected[decision.interrupt_id] = decision

        batch = await self._store.load_batch(batch_id)
        if batch is None:
            logger.warning("record_decision: batch %s vanished", batch_id)
            return None

        # Decide whether to resolve based on wait_mode
        ready, resolution = await self._check_wait_condition(batch, collected)
        if not ready:
            return None

        async with self._lock:
            fut = self._waiters.pop(batch_id, None)
            self._collected.pop(batch_id, None)

        # Persist final batch state before resolving the future
        if resolution.rejected and batch.policy == BatchPolicy.ALL_OR_NOTHING:
            batch.state = BatchState.PARTIAL
        else:
            batch.state = BatchState.RESOLVED
        batch.resolved_at = datetime.now(timezone.utc)
        await self._store.save_batch(batch)
        await self._emit_audit(
            AuditEventKind.GRAPH_RESUMED, batch_id,
            {"event": "batch_resolved", "rejected": resolution.rejected,
             "all_approved": resolution.all_approved},
        )

        if fut is not None and not fut.done():
            fut.set_result(resolution)
        return resolution

    async def _check_wait_condition(
        self,
        batch: HitlBatch,
        collected: dict[str, HitlDecision],
    ) -> tuple[bool, BatchResolution]:
        """Return (ready_to_resolve, resolution_object). resolution_object
        is meaningful only when ready=True."""
        n_total = len(batch.interrupt_ids)
        n_done  = len(collected)

        # Compute summary flags incrementally — cheap
        any_reject = any(
            d.decision == DecisionKind.REJECT for d in collected.values()
        )
        all_positive = all(
            d.decision in (
                DecisionKind.APPROVE, DecisionKind.EDIT,
                DecisionKind.CHOOSE,  DecisionKind.ANSWER,
            )
            for d in collected.values()
        )

        if batch.wait_mode == BatchWaitMode.ALL:
            ready = (n_done >= n_total)
        elif batch.wait_mode == BatchWaitMode.THRESHOLD:
            # Reserved for future use — currently behaves like ALL.
            # When implemented, threshold_count specifies "N approvals to
            # short-circuit" for ratio-based decisions like "3 of 5 sites
            # approved is enough".
            ready = (n_done >= n_total)
        else:
            # STREAMING reserved — would yield per-decision events
            # without a single resolution point. v0 falls back to ALL.
            ready = (n_done >= n_total)

        if not ready:
            return False, None  # type: ignore[return-value]

        rejected = any_reject and batch.policy == BatchPolicy.ALL_OR_NOTHING
        # Even under BEST_EFFORT, we report `rejected=True` on the
        # resolution if ANY child was rejected — pipelines that care about
        # "did everyone approve?" can branch on it. But the batch state
        # only flips to PARTIAL under ALL_OR_NOTHING.
        if any_reject:
            rejected = True

        return True, BatchResolution(
            batch=batch,
            decisions_by_id=collected,
            rejected=rejected,
            all_approved=all_positive,
        )

    # ── Lookup ───────────────────────────────────────────────────────

    async def is_pending_locally(self, batch_id: str) -> bool:
        async with self._lock:
            return batch_id in self._waiters

    # ── Internal ─────────────────────────────────────────────────────

    async def _emit_audit(self, kind: AuditEventKind, batch_id: str,
                          payload: dict[str, Any]) -> None:
        if self._on_audit is None:
            return
        try:
            res = self._on_audit(kind, batch_id, payload)
            import inspect
            if inspect.isawaitable(res):
                await res
        except Exception as exc:
            logger.warning("Batch audit hook raised %s, swallowing", exc)


# ---------------------------------------------------------------------------
# Helpers — recovering "is this child part of a batch?" from a payload
# ---------------------------------------------------------------------------

# Each child interrupt's HitlPayload.context_snapshot stores its batch_id
# under this key. Producers don't need to set this manually — the
# pipeline.request_batch_approval helper does it before saving.

BATCH_ID_KEY = "_batch_id"


def get_batch_id(payload_context_snapshot: dict[str, Any]) -> Optional[str]:
    """Return the batch_id this interrupt belongs to, or None for
    unbatched interrupts. Single small helper so the router's batch-aware
    dispatch logic isn't sprinkled with magic-string lookups."""
    val = payload_context_snapshot.get(BATCH_ID_KEY)
    return val if isinstance(val, str) and val else None