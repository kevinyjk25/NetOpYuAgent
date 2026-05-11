"""
skills/journal_consumer.py
--------------------------
Bridge between SkillJournal observability and SkillEvolver self-improvement.

Periodically scans recent journal entries; for any skill that crosses the
dormant-rate threshold (loaded but no tool calls produced), generates a
structured feedback message and calls SkillEvolver.apply_feedback().

This is the "Hermes feedback loop" wired to real usage data instead of
operator-supplied notes. Operators can still inject manual feedback through
the existing apply_feedback API; this just adds an automatic signal source.

Design choices:
  - Pure background task — never blocks the runtime stream loop.
  - Per-skill state tracked across runs so we don't re-feed the same evidence.
  - Configurable thresholds (min_uses, dormant_threshold) prevent over-eager
    rewrites on small samples.
  - All errors logged but never raised — feedback is best-effort observability,
    not a correctness path.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SkillJournalConsumer:
    """Background scanner that converts SkillJournal stats into
    SkillEvolver feedback calls.

    Lifecycle:
        consumer = SkillJournalConsumer(evolver, store, cfg)
        await consumer.start()    # spawns the background task
        ...
        await consumer.stop()     # cancels and awaits clean shutdown

    Single-pass (for tests / one-off runs):
        await consumer.scan_once()
    """

    def __init__(
        self,
        evolver,           # SkillEvolver instance — duck-typed for testability
        journal_store,     # SkillJournalStore — duck-typed
        *,
        interval_s:        int   = 300,
        min_uses:          int   = 3,
        dormant_threshold: float = 0.6,
    ):
        self._evolver  = evolver
        self._store    = journal_store
        self._interval = max(30, int(interval_s))
        self._min_uses = max(1, int(min_uses))
        self._dormant_threshold = float(dormant_threshold)

        # Per-skill last-fed state: skill_id → (last_dormant_count, last_use_count)
        # Used to skip skills whose stats haven't changed since the last feedback,
        # so we never bombard the LLM patcher with stale signals.
        self._last_seen: dict[str, tuple[int, int]] = {}

        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    # ── Lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            logger.debug("SkillJournalConsumer: already started")
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop(), name="skill_journal_consumer")
        logger.info(
            "SkillJournalConsumer: started (interval=%ds, min_uses=%d, dormant_thr=%.2f)",
            self._interval, self._min_uses, self._dormant_threshold,
        )

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop_event.set()
        try:
            await asyncio.wait_for(self._task, timeout=5.0)
        except asyncio.TimeoutError:
            self._task.cancel()
            try: await self._task
            except asyncio.CancelledError: pass
        logger.info("SkillJournalConsumer: stopped")
        self._task = None

    # ── Core loop ────────────────────────────────────────────────────

    async def _run_loop(self) -> None:
        # Delay the first scan so the system has time to accumulate entries
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=self._interval)
            return  # stopped during initial wait
        except asyncio.TimeoutError:
            pass

        while not self._stop_event.is_set():
            try:
                await self.scan_once()
            except Exception as exc:
                logger.warning("SkillJournalConsumer scan failed: %s", exc)

            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._interval)
                return
            except asyncio.TimeoutError:
                pass

    # ── One pass — public for tests + on-demand scans ───────────────

    async def scan_once(self) -> dict[str, Any]:
        """Run a single scan over journal stats; trigger feedback for problem skills.

        Returns a small summary dict so callers can log / report.
        """
        stats = self._store.stats() if hasattr(self._store, "stats") else {}
        if not stats or stats.get("count", 0) == 0:
            return {"fed_back": 0, "candidates": 0, "skipped": 0}

        use_count     = stats.get("skill_use_count")     or {}
        dormant_count = stats.get("skill_dormant_count") or {}

        candidates:    list[str] = []
        fed_back:      list[str] = []
        skipped:       int       = 0

        for skill_id, uses in use_count.items():
            if uses < self._min_uses:
                skipped += 1
                continue
            dormant = dormant_count.get(skill_id, 0)
            rate = dormant / uses if uses else 0.0
            if rate < self._dormant_threshold:
                continue   # healthy skill, leave it alone

            candidates.append(skill_id)

            # De-dupe: skip if nothing new since last feedback
            last = self._last_seen.get(skill_id)
            if last and last == (dormant, uses):
                continue

            # Build feedback narrative — concise, factual, actionable
            recent = self._recent_dormant_queries(skill_id, limit=3)
            feedback = (
                f"Automated feedback from SkillJournal observations: "
                f"this skill was loaded {uses} time(s) but produced no tool calls "
                f"in {dormant}/{uses} cases ({int(rate * 100)}% dormant). "
                f"Operators appear to abandon this skill mid-task. "
                f"Consider whether the purpose statement matches what operators "
                f"actually need, whether the steps are too vague, or whether "
                f"the recommended tools are available. "
                + (f"Recent dormant queries: {recent}" if recent else "")
            )

            try:
                logger.info(
                    "SkillJournalConsumer: feeding back skill=%s (uses=%d dormant=%d rate=%.2f)",
                    skill_id, uses, dormant, rate,
                )
                await self._evolver.apply_feedback(
                    skill_id=skill_id,
                    feedback=feedback,
                    success=False,
                )
                fed_back.append(skill_id)
                self._last_seen[skill_id] = (dormant, uses)
            except Exception as exc:
                logger.warning(
                    "SkillJournalConsumer: apply_feedback for %s failed: %s",
                    skill_id, exc,
                )

        summary = {
            "fed_back":  len(fed_back),
            "candidates": len(candidates),
            "skipped":   skipped,
            "skill_ids": fed_back,
            "ts":        time.time(),
        }
        if fed_back:
            logger.info("SkillJournalConsumer: scan_once summary=%s", summary)
        return summary

    # ── Helpers ──────────────────────────────────────────────────────

    def _recent_dormant_queries(self, skill_id: str, limit: int = 3) -> list[str]:
        """Pull up to N recent journal entries where this skill was dormant.
        Used as concrete evidence in the feedback narrative."""
        try:
            entries = self._store.filter(skill_id=skill_id, limit=20)
        except Exception:
            return []
        out: list[str] = []
        for e in entries:
            for attr in e.get("attribution", []):
                if attr.get("skill_id") == skill_id and attr.get("appeared_dormant"):
                    q = (e.get("query") or "").strip()
                    if q:
                        out.append(q[:80])
                        break
            if len(out) >= limit:
                break
        return out
