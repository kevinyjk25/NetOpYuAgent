"""
memory/__init__.py
------------------
Public surface + factory function for the Memory module.

Quick start
-----------
    from memory import create_memory_router, MemoryConfig

    router = await create_memory_router(
        redis_url="redis://localhost:6379",
        postgres_dsn="postgresql://user:pass@localhost/db",
        chroma_path="./chroma_db",
    )

    # Write
    await router.ingest_turn(session_id, user_text, assistant_text)

    # Read
    from memory.schemas import RetrievalQuery
    results = await router.retrieve(RetrievalQuery(
        query_text="show P1 alerts",
        session_id="sess-001",
    ))
    context = router.format_context(results)
"""
from __future__ import annotations

from .consolidation import ConsolidationWorker, LifecycleManager
from .pipelines.ingestion import IngestionPipeline
from .pipelines.retrieval import RetrievalPipeline
from memory.router import MemoryRouter
from memory.schemas import (
    ConsolidationJob,
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


async def create_memory_router(
    redis_url: str | None = None,
    postgres_dsn: str | None = None,
    chroma_path: str = "./chroma_db",
    chroma_collection: str = "memory",
    config: MemoryConfig | None = None,
) -> MemoryRouter:
    """
    Factory: build and return a fully-wired MemoryRouter.

    In-memory stubs are used when redis_url / postgres_dsn are None
    (suitable for development and tests).
    """
    cfg = config or MemoryConfig()

    # Tier 1 — always in-process
    realtime = ContextWindowStore(max_tokens=cfg.max_context_tokens)

    # Tier 2 — Redis or in-memory stub
    if redis_url:
        import redis.asyncio as aioredis
        r = aioredis.from_url(redis_url, decode_responses=False)
        short = RedisShortTermStore(r, cfg.short_term_ttl_seconds, cfg.short_term_max_turns)
    else:
        short = _InMemoryShortTermStub()  # type: ignore[assignment]

    # Tier 3 — Chroma or in-memory stub
    try:
        import chromadb
        client     = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_or_create_collection(chroma_collection)
        mid = ChromaMidTermStore(
            collection,
            decay_factor=cfg.mid_term_decay_factor,
            alpha=cfg.relevance_weight,
            beta=cfg.recency_weight,
        )
    except ImportError:
        mid = _InMemoryMidTermStub()   # type: ignore[assignment]

    # Tier 4 — PostgreSQL or in-memory stub
    if postgres_dsn:
        import asyncpg
        pool = await asyncpg.create_pool(postgres_dsn)
        long = PostgresLongTermStore(pool)
    else:
        long = _InMemoryLongTermStub()  # type: ignore[assignment]

    return MemoryRouter(cfg, realtime, short, mid, long)


# ---------------------------------------------------------------------------
# In-memory stubs (dev/test when real backends are unavailable)
# ---------------------------------------------------------------------------

class _InMemoryShortTermStub:
    """Simple list-backed short-term store for dev/test."""
    tier = MemoryTier.SHORT_TERM

    def __init__(self) -> None:
        self._data: list[MemoryRecord] = []

    async def write(self, record: MemoryRecord) -> None:
        self._data.append(record)

    async def search(self, query_text, session_id, embedding, top_k):
        hits = [r for r in self._data if r.session_id == session_id]
        return [(r, 0.8) for r in hits[-top_k:]]

    async def delete(self, record_id):
        self._data = [r for r in self._data if r.record_id != record_id]

    async def delete_session(self, session_id):
        self._data = [r for r in self._data if r.session_id != session_id]

    async def get_by_session(self, session_id, limit=50):
        return [r for r in self._data if r.session_id == session_id][-limit:]


class _InMemoryMidTermStub(_InMemoryShortTermStub):
    tier = MemoryTier.MID_TERM


class _InMemoryLongTermStub(_InMemoryShortTermStub):
    tier = MemoryTier.LONG_TERM


__all__ = [
    "MemoryRouter", "create_memory_router",
    "MemoryConfig", "MemoryRecord", "MemoryTier", "MemoryRecordType",
    "RetrievalQuery", "RetrievalResult", "ConsolidationJob",
    "ConsolidationWorker", "LifecycleManager",
    "IngestionPipeline", "RetrievalPipeline",
    "ContextWindowStore", "RedisShortTermStore",
    "ChromaMidTermStore", "PostgresLongTermStore",
]
