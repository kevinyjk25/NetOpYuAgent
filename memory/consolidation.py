"""
memory/consolidation.py
------------------------
Consolidation worker: summarise session turns → extract entities →
promote to mid/long-term → decay stale mid-term embeddings.

Run as a background asyncio task at end-of-session or on a timer.
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from .schemas import (
    ConsolidationJob,
    ConsolidationStatus,
    MemoryConfig,
    MemoryRecord,
    MemoryRecordType,
    MemoryTier,
)

if TYPE_CHECKING:
    from hitl.router import MemoryRouter

logger = logging.getLogger(__name__)

# Regex heuristics for entity extraction (replace with NER model in production)
_ENTITY_PATTERNS = {
    "service":    re.compile(r"\b(\w+-service|\w+-api|\w+-worker)\b", re.I),
    "host":       re.compile(r"\b([a-z0-9]+-[a-z0-9]+-\d+)\b"),
    "incident_id":re.compile(r"\b(INC-\d+|JIRA-\d+|PD-\d+)\b", re.I),
    "metric":     re.compile(r"\b(p99|p95|latency|error_rate|cpu|memory|disk)\b", re.I),
}


class ConsolidationWorker:
    """
    End-of-session consolidation pipeline.

    Steps
    -----
    1. Fetch all short-term turns for the session.
    2. Build an LLM summary (stub: concatenate with truncation).
    3. Extract named entities with regex / NER.
    4. Write summary → mid-term; entities → long-term.
    5. Optionally decay mid-term records older than threshold.
    """

    def __init__(
        self,
        router: "MemoryRouter",
        config: Optional[MemoryConfig] = None,
    ) -> None:
        self._router = router
        self._cfg    = config or MemoryConfig()

    async def consolidate(self, session_id: str) -> ConsolidationJob:
        """Run consolidation for one session. Returns a completed ConsolidationJob."""
        job = ConsolidationJob(session_id=session_id, status=ConsolidationStatus.RUNNING)
        logger.info("Consolidation start session=%s job=%s", session_id, job.job_id)

        try:
            turns = await self._router._short.get_by_session(
                session_id, limit=self._cfg.short_term_max_turns
            )
            job.records_processed = len(turns)

            if len(turns) < self._cfg.consolidation_min_turns:
                logger.debug("Consolidation: too few turns (%d), skipping", len(turns))
                job.status = ConsolidationStatus.DONE
                job.completed_at = datetime.now(timezone.utc).isoformat()
                return job

            # Step 1: Summarise
            summary_text = await self._summarise(turns)
            summary_record = MemoryRecord(
                session_id=session_id,
                record_type=MemoryRecordType.SUMMARY,
                tier=MemoryTier.MID_TERM,
                content=summary_text,
                importance=0.65,
            )
            await self._router.write_raw(summary_record)

            # Step 2: Extract entities
            all_text = "\n".join(t.content for t in turns)
            entities = self._extract_entities(all_text)
            job.entities_extracted = len(entities)

            for entity_type, entity_value in entities:
                entity_record = MemoryRecord(
                    session_id=session_id,
                    record_type=MemoryRecordType.ENTITY,
                    tier=MemoryTier.LONG_TERM,
                    content=f"{entity_type}: {entity_value}",
                    metadata={"entity_type": entity_type},
                    importance=0.85,
                )
                await self._router.write_raw(entity_record)

            job.status = ConsolidationStatus.DONE

        except Exception as exc:
            logger.exception("Consolidation failed session=%s: %s", session_id, exc)
            job.status = ConsolidationStatus.FAILED
            job.error  = str(exc)

        job.completed_at = datetime.now(timezone.utc).isoformat()
        logger.info(
            "Consolidation done session=%s entities=%d records=%d",
            session_id, job.entities_extracted, job.records_processed,
        )
        return job

    # ------------------------------------------------------------------

    async def _summarise(self, turns: list[MemoryRecord]) -> str:
        """
        Stub summariser — concatenates up to 1 500 chars.
        Replace with:
            chain = load_summarize_chain(llm, chain_type="map_reduce")
            docs  = [Document(page_content=t.content) for t in turns]
            return await chain.arun(docs)
        """
        combined = "\n".join(t.content for t in turns)
        if len(combined) <= 1_500:
            return combined
        return combined[:1_497] + "..."

    @staticmethod
    def _extract_entities(text: str) -> list[tuple[str, str]]:
        """Heuristic regex entity extraction. Replace with spaCy NER."""
        found = []
        for entity_type, pattern in _ENTITY_PATTERNS.items():
            for match in pattern.finditer(text):
                found.append((entity_type, match.group(0)))
        return list(set(found))   # deduplicate


class LifecycleManager:
    """
    Scheduled background task for TTL eviction and decay.
    Run with: asyncio.create_task(manager.run())
    """

    def __init__(
        self,
        router: "MemoryRouter",
        poll_interval: float = 3_600.0,   # 1 h
    ) -> None:
        self._router  = router
        self._poll    = poll_interval
        self._running = False

    async def run(self) -> None:
        self._running = True
        logger.info("LifecycleManager started (poll=%.0fs)", self._poll)
        while self._running:
            try:
                await self._tick()
            except Exception as exc:
                logger.exception("LifecycleManager tick error: %s", exc)
            await asyncio.sleep(self._poll)

    def stop(self) -> None:
        self._running = False

    async def _tick(self) -> None:
        """One eviction cycle. Override for custom policies."""
        logger.debug("LifecycleManager: tick")
        # In production: query expired records from Postgres and delete
        # Also: bump access_count decay on mid-term, archive cold long-term
