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
import logging

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


def _get_or_recreate_chroma_collection(client, name: str, expected_dim: int):
    """
    Return a ChromaDB collection whose embedding dimension matches expected_dim.

    ChromaDB locks the embedding dimension on the first upsert. If the on-disk
    collection was created with 1536 dims (old OpenAI-stub default) and the live
    embedder now produces 768 dims, every write fails with a dimension mismatch.

    Detection strategy (two-pass):
      1. Check metadata["embedding_dim"] tag (present on collections we created).
      2. If no tag (old collection created without metadata), probe by attempting a
         dummy 1-element query with the expected dim. If Chroma raises InvalidDimensionException
         the collection is locked to a different dim → delete and recreate.

    FTS5 Track-A data (state.db) is unaffected — only the Chroma vector index resets.
    """
    _log = logging.getLogger(__name__)
    try:
        col = client.get_collection(name)
        stored_dim = (col.metadata or {}).get("embedding_dim")

        if stored_dim is not None:
            # We tagged this collection ourselves — trust the metadata
            if int(stored_dim) != expected_dim:
                _log.warning(
                    "ChromaDB %r: metadata dim=%s != expected dim=%d — recreating.",
                    name, stored_dim, expected_dim,
                )
                client.delete_collection(name)
                return client.create_collection(name, metadata={"embedding_dim": expected_dim})
            return col

        # stored_dim is None: collection existed before we started tagging it.
        # Probe with a dummy query to detect the locked dim.
        if col.count() > 0:
            try:
                probe = [0.1] * expected_dim
                col.query(query_embeddings=[probe], n_results=1)
                # Query succeeded → dim is compatible, tag it now
                try:
                    col.modify(metadata={"embedding_dim": expected_dim})
                except Exception:
                    pass
                _log.info("ChromaDB %r: dim probe OK at %d, tagged.", name, expected_dim)
                return col
            except Exception as probe_exc:
                err = str(probe_exc).lower()
                if "dimension" in err or "expected" in err or "invalid" in err:
                    _log.warning(
                        "ChromaDB %r: dim probe failed (%s) — "
                        "collection locked to wrong dim, recreating "
                        "(mid-term vector index reset; FTS5 data preserved).",
                        name, probe_exc,
                    )
                    client.delete_collection(name)
                    return client.create_collection(name, metadata={"embedding_dim": expected_dim})
                # Different error — let it propagate
                raise
        else:
            # Empty collection, no vectors locked yet — tag it and use it
            try:
                col.modify(metadata={"embedding_dim": expected_dim})
            except Exception:
                pass
            return col

    except Exception as exc:
        if "does not exist" in str(exc).lower() or "not found" in str(exc).lower():
            # Collection doesn't exist yet — create fresh
            return client.get_or_create_collection(name, metadata={"embedding_dim": expected_dim})
        raise


async def create_memory_router(
    redis_url: str | None = None,
    postgres_dsn: str | None = None,
    chroma_path: str = "./chroma_db",
    chroma_collection: str = "memory",
    config: MemoryConfig | None = None,
    embedding_dim: int = 768,
) -> MemoryRouter:
    """
    Factory: build and return a fully-wired MemoryRouter.

    In-memory stubs are used when redis_url / postgres_dsn are None
    (suitable for development and tests).
    """
    cfg = config or MemoryConfig(embedding_dim=embedding_dim)

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
        client = chromadb.PersistentClient(path=chroma_path)
        collection = _get_or_recreate_chroma_collection(
            client, chroma_collection, cfg.embedding_dim
        )
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
