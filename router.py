"""
memory/router.py
----------------
MemoryRouter — single entry-point for all memory reads and writes.

Write fanout policy
-------------------
  importance < 0.4   → real-time + short-term only
  0.4 ≤ imp < 0.75   → real-time + short-term + mid-term
  imp ≥ 0.75         → all four tiers (including long-term)

Read strategy
-------------
  Parallel fetch from all requested tiers, then hand off to
  the RetrievalPipeline for MMR dedup + blending + token trim.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from .pipelines.ingestion import IngestionPipeline
from .pipelines.retrieval import RetrievalPipeline
from memory.schemas import (
    MemoryConfig,
    MemoryRecord,
    MemoryRecordType,
    MemoryTier,
    RetrievalQuery,
    RetrievalResult,
)
from .stores.backends import (
    ChromaMidTermStore,
    ContextWindowStore,
    PostgresLongTermStore,
    RedisShortTermStore,
)

logger = logging.getLogger(__name__)


class MemoryRouter:
    """
    Facade over all four memory backends.

    Usage
    -----
        router = MemoryRouter(config, realtime, short, mid, long)

        # Write a turn
        await router.ingest_turn(session_id, user_text, assistant_text)

        # Retrieve relevant context
        results = await router.retrieve(RetrievalQuery(
            query_text="show P1 alerts",
            session_id="sess-001",
        ))
        context_str = router.format_context(results)
    """

    def __init__(
        self,
        config: MemoryConfig,
        realtime_store: ContextWindowStore,
        short_store: RedisShortTermStore,
        mid_store: ChromaMidTermStore,
        long_store: PostgresLongTermStore,
    ) -> None:
        self._cfg   = config
        self._rt    = realtime_store
        self._short = short_store
        self._mid   = mid_store
        self._long  = long_store

        self._ingestion  = IngestionPipeline(config)
        self._retrieval  = RetrievalPipeline(config)

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    async def ingest_turn(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
        metadata: Optional[dict] = None,
    ) -> MemoryRecord:
        """Ingest a completed conversation turn."""
        content = f"User: {user_text}\nAssistant: {assistant_text}"
        record  = await self._ingestion.process(
            content=content,
            session_id=session_id,
            record_type=MemoryRecordType.TURN,
            metadata=metadata or {},
        )
        await self._fanout(record)
        return record

    async def ingest_task_result(
        self,
        session_id: str,
        task_id: str,
        result_text: str,
        importance: float = 0.6,
    ) -> MemoryRecord:
        """Ingest a completed task result."""
        record = await self._ingestion.process(
            content=result_text,
            session_id=session_id,
            record_type=MemoryRecordType.TASK_RESULT,
            metadata={"task_id": task_id},
            importance_override=importance,
        )
        await self._fanout(record)
        return record

    async def ingest_entity(
        self,
        session_id: str,
        entity_text: str,
        entity_type: str,
    ) -> MemoryRecord:
        """Ingest a named entity or fact (always goes to long-term)."""
        record = await self._ingestion.process(
            content=entity_text,
            session_id=session_id,
            record_type=MemoryRecordType.ENTITY,
            metadata={"entity_type": entity_type},
            importance_override=0.9,   # always promote to long-term
        )
        await self._fanout(record)
        return record

    async def write_raw(self, record: MemoryRecord) -> None:
        """Write a pre-built MemoryRecord (used by consolidation pipeline)."""
        await self._fanout(record)

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        """
        Fetch from all requested tiers in parallel, then blend and trim.
        """
        embedding = await self._ingestion.embed(query.query_text)

        # Parallel fetch
        tasks = []
        store_map = {
            MemoryTier.REALTIME:   self._rt,
            MemoryTier.SHORT_TERM: self._short,
            MemoryTier.MID_TERM:   self._mid,
            MemoryTier.LONG_TERM:  self._long,
        }
        for tier in query.tiers:
            store = store_map[tier]
            tasks.append(
                store.search(
                    query.query_text,
                    query.session_id,
                    embedding,
                    query.top_k,
                )
            )

        tier_results = await asyncio.gather(*tasks, return_exceptions=True)

        raw: list[tuple[MemoryRecord, float]] = []
        for tier, result in zip(query.tiers, tier_results):
            if isinstance(result, Exception):
                logger.warning("Tier %s search failed: %s", tier, result)
                continue
            raw.extend(result)

        return await self._retrieval.process(raw, query)

    def format_context(
        self,
        results: list[RetrievalResult],
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Format retrieval results into a single context string
        for injection into the LLM prompt.
        """
        max_tok = max_tokens or self._cfg.max_context_tokens
        lines   = []
        total   = 0
        for r in results:
            tokens = max(1, len(r.record.content) // 4)
            if total + tokens > max_tok:
                break
            tier_tag = r.tier.value.upper()
            lines.append(f"[{tier_tag}] {r.record.content}")
            total += tokens
        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def delete_session(self, session_id: str) -> None:
        """GDPR deletion cascade across all tiers."""
        await asyncio.gather(
            self._rt.delete_session(session_id),
            self._short.delete_session(session_id),
            self._mid.delete_session(session_id),
            self._long.delete_session(session_id),
        )
        logger.info("Memory: deleted all records for session=%s", session_id)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _fanout(self, record: MemoryRecord) -> None:
        """Write to the appropriate tier(s) based on importance score."""
        stores = [self._rt, self._short]  # always write hot tiers

        if record.importance >= 0.4:
            stores.append(self._mid)
        if record.importance >= self._cfg.long_term_min_importance:
            stores.append(self._long)

        # Embed once before writing to vector store
        if self._mid in stores and record.embedding is None:
            record.embedding = await self._ingestion.embed(record.content)

        results = await asyncio.gather(
            *[s.write(record) for s in stores],
            return_exceptions=True,
        )
        for store, result in zip(stores, results):
            if isinstance(result, Exception):
                logger.error("Write failed to %s: %s", store.tier, result)
