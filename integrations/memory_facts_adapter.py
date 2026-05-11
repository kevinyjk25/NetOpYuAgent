"""
integrations/memory_facts_adapter.py
------------------------------------
Cross-module adapter — bridges SkillJournal observations into MemoryFacts.

Framework principle:
  - The skill module (SkillJournal) does NOT import memory module.
  - The memory module (MidTermStore) does NOT know about skills.
  - This adapter is the explicit bridge, opt-in via config.

Why this exists:
  When a skill is consistently dormant for a particular query type, that's
  useful prior knowledge. Storing it as a MemoryFact makes the signal
  available wherever memory is recalled — without coupling the skill
  module to memory.

Behaviour:
  Walk recent journal entries; for each (skill, query-pattern) pair where
  dormancy crosses threshold, emit a fact of fact_type="lesson":
      "Skill 'netflow_analysis' has been observed to be abandoned for
       queries about 'check site netflow' (3/4 attempts dormant). Consider
       alternative skills or solve with tools directly."

  Facts go into mid-term store with metadata.source="journal_adapter" so
  they can be filtered / disabled separately from human-authored facts.

Independence guarantees:
  - This module has ONE dependency on memory: MemoryAdapter (or duck-typed)
  - This module has ONE dependency on skills: SkillJournalStore (or duck-typed)
  - Disabling cfg.cross_module.journal_to_facts.enabled removes the whole feature
  - No other module references this adapter
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol declarations — duck-typed integration points
# ---------------------------------------------------------------------------

class _JournalStoreProtocol(Protocol):
    """The subset of SkillJournalStore methods we need."""
    def stats(self) -> dict[str, Any]: ...
    def filter(self, *, skill_id: Optional[str] = None,
               outcome: Optional[str] = None,
               ambiguous: Optional[bool] = None,
               limit: int = 50) -> list[dict[str, Any]]: ...


class _FactWriterProtocol(Protocol):
    """The subset of memory API we need to write facts."""
    async def add_fact(
        self, session_id: str, user_id: str, fact_text: str,
        fact_type: str = "lesson", confidence: float = 0.8,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str: ...


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class JournalToFactsAdapter:
    """Periodically converts SkillJournal observations into MemoryFacts.

    Config-driven:
      enabled                   — master switch
      interval_s                — scan period (background-task mode)
      min_observations          — minimum journal entries before promoting to fact
      dormant_threshold         — dormant/total ratio to qualify as a "lesson"
      success_threshold         — success ratio to emit positive lesson
      fact_ttl_days             — how long auto-facts live (shorter than user-authored)
      max_facts_per_scan        — rate limit, avoid flooding memory
      target_user_id            — which operator's memory to write to
                                  (use "default" or "_system" for shared facts)
    """

    def __init__(
        self,
        journal_store:    _JournalStoreProtocol,
        fact_writer:      _FactWriterProtocol,
        *,
        interval_s:        int   = 600,
        min_observations:  int   = 3,
        dormant_threshold: float = 0.6,
        success_threshold: float = 0.9,
        fact_ttl_days:     float = 14.0,
        max_facts_per_scan: int  = 10,
        target_user_id:    str   = "_system",
        target_session_id: str   = "_cross_session",
    ):
        self._journal = journal_store
        self._writer  = fact_writer
        self._interval = max(60, int(interval_s))
        self._min_obs  = max(1, int(min_observations))
        self._dormant_thr = float(dormant_threshold)
        self._success_thr = float(success_threshold)
        self._fact_ttl    = float(fact_ttl_days)
        self._max_per_scan = max(1, int(max_facts_per_scan))
        self._target_user_id    = target_user_id
        self._target_session_id = target_session_id

        # Dedup: skill_id → (last_dormant, last_uses) we've already promoted
        self._promoted_state: dict[str, tuple[int, int, str]] = {}  # +last_kind

        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    # ── Lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._loop(), name="journal_to_facts")
        logger.info(
            "JournalToFactsAdapter started (interval=%ds, dormant_thr=%.2f, "
            "success_thr=%.2f, ttl=%.1fd)",
            self._interval, self._dormant_thr, self._success_thr, self._fact_ttl,
        )

    async def stop(self) -> None:
        if not self._task:
            return
        self._stop.set()
        try:
            await asyncio.wait_for(self._task, timeout=5.0)
        except asyncio.TimeoutError:
            self._task.cancel()
            try: await self._task
            except asyncio.CancelledError: pass
        logger.info("JournalToFactsAdapter stopped")
        self._task = None

    async def _loop(self) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
            return
        except asyncio.TimeoutError:
            pass

        while not self._stop.is_set():
            try:
                await self.scan_once()
            except Exception as exc:
                logger.warning("JournalToFactsAdapter scan failed: %s", exc)

            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
                return
            except asyncio.TimeoutError:
                pass

    # ── Public scan-once (for tests + manual ops) ────────────────────

    async def scan_once(self) -> dict[str, Any]:
        """Walk journal stats, emit facts for skills crossing thresholds.

        Returns a summary so callers can log / inspect.
        """
        try:
            stats = self._journal.stats() or {}
        except Exception as exc:
            return {"error": str(exc), "facts_emitted": 0}

        if stats.get("count", 0) < self._min_obs:
            return {"facts_emitted": 0, "reason": "not enough journal entries"}

        use_count     = stats.get("skill_use_count")     or {}
        dormant_count = stats.get("skill_dormant_count") or {}

        emitted: list[str] = []

        for skill_id, uses in use_count.items():
            if len(emitted) >= self._max_per_scan:
                break
            if uses < self._min_obs:
                continue
            dormant = dormant_count.get(skill_id, 0)
            success = uses - dormant
            dormant_rate = dormant / uses if uses else 0.0
            success_rate = success / uses if uses else 0.0

            kind: Optional[str] = None
            fact_text: Optional[str] = None

            if dormant_rate >= self._dormant_thr:
                kind = "dormant"
                queries = self._sample_queries(skill_id, dormant=True)
                fact_text = self._format_dormant_fact(
                    skill_id=skill_id, uses=uses, dormant=dormant,
                    dormant_rate=dormant_rate, sample_queries=queries,
                )
            elif success_rate >= self._success_thr and uses >= self._min_obs * 2:
                kind = "successful"
                queries = self._sample_queries(skill_id, dormant=False)
                fact_text = self._format_success_fact(
                    skill_id=skill_id, uses=uses, success=success,
                    success_rate=success_rate, sample_queries=queries,
                )

            if not kind or not fact_text:
                continue

            # Dedup: skip if we already promoted the same (kind, dormant, uses)
            last = self._promoted_state.get(skill_id)
            if last and last == (dormant, uses, kind):
                continue

            try:
                fact_id = await self._writer.add_fact(
                    session_id = self._target_session_id,
                    user_id    = self._target_user_id,
                    fact_text  = fact_text,
                    fact_type  = "lesson",
                    confidence = min(1.0, dormant_rate if kind == "dormant" else success_rate),
                    metadata   = {
                        "source":       "journal_adapter",
                        "skill_id":     skill_id,
                        "kind":         kind,
                        "uses":         uses,
                        "dormant":      dormant,
                        "ttl_days":     self._fact_ttl,
                        "emitted_at":   time.time(),
                    },
                )
                emitted.append(fact_id)
                self._promoted_state[skill_id] = (dormant, uses, kind)
                logger.info(
                    "JournalToFactsAdapter: emitted %s fact for skill=%s "
                    "(uses=%d dormant=%d rate=%.2f) → fact_id=%s",
                    kind, skill_id, uses, dormant, dormant_rate, fact_id,
                )
            except Exception as exc:
                logger.warning(
                    "JournalToFactsAdapter: writer.add_fact failed for %s: %s",
                    skill_id, exc,
                )

        return {
            "facts_emitted":  len(emitted),
            "fact_ids":       emitted,
            "scanned_skills": len(use_count),
            "ts":             time.time(),
        }

    # ── Fact text formatting (kept local for tunability) ────────────

    def _format_dormant_fact(
        self, *, skill_id: str, uses: int, dormant: int,
        dormant_rate: float, sample_queries: list[str],
    ) -> str:
        line = (
            f"Skill '{skill_id}' observed to be abandoned mid-task "
            f"in {dormant}/{uses} attempts ({int(dormant_rate * 100)}% dormant rate). "
            f"Operators tend to skip this skill or pivot to other tools."
        )
        if sample_queries:
            line += " Recent abandoned queries: " + "; ".join(
                q[:60] for q in sample_queries[:2]
            )
        return line

    def _format_success_fact(
        self, *, skill_id: str, uses: int, success: int,
        success_rate: float, sample_queries: list[str],
    ) -> str:
        line = (
            f"Skill '{skill_id}' has high task-completion rate "
            f"({success}/{uses} attempts, {int(success_rate * 100)}%). "
            f"This skill is a reliable choice for related queries."
        )
        if sample_queries:
            line += " Recent successful queries: " + "; ".join(
                q[:60] for q in sample_queries[:2]
            )
        return line

    def _sample_queries(self, skill_id: str, *, dormant: bool, n: int = 3) -> list[str]:
        try:
            entries = self._journal.filter(skill_id=skill_id, limit=20)
        except Exception:
            return []
        out: list[str] = []
        for e in entries:
            for attr in e.get("attribution", []):
                if attr.get("skill_id") != skill_id:
                    continue
                if attr.get("appeared_dormant") == dormant:
                    q = (e.get("query") or "").strip()
                    if q and q not in out:
                        out.append(q)
                    break
            if len(out) >= n:
                break
        return out
