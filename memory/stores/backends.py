"""
memory/stores/backends.py
-------------------------
Storage backend protocol and all four tier implementations.

  ContextWindowStore   – in-process list, token-budgeted
  RedisShortTermStore  – sliding window with TTL
  ChromaMidTermStore   – vector similarity + time-decay reranker
  PostgresLongTermStore– entity / fact / incident store
"""
from __future__ import annotations

import json
import logging
import math
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional, Protocol, runtime_checkable

from ..schemas import MemoryConfig, MemoryRecord, MemoryTier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class MemoryBackend(Protocol):
    tier: MemoryTier

    async def write(self, record: MemoryRecord) -> None: ...
    async def search(
        self,
        query_text: str,
        session_id: str,
        embedding: Optional[list[float]],
        top_k: int,
    ) -> list[tuple[MemoryRecord, float]]: ...
    async def delete(self, record_id: str) -> None: ...
    async def delete_session(self, session_id: str) -> None: ...
    async def get_by_session(
        self, session_id: str, limit: int = 50
    ) -> list[MemoryRecord]: ...


# ---------------------------------------------------------------------------
# Tier 1 – Real-time (context window)
# ---------------------------------------------------------------------------

class ContextWindowStore:
    """
    In-process list of MemoryRecord objects.
    Enforces a token budget; oldest records are dropped when full.
    """
    tier = MemoryTier.REALTIME

    def __init__(self, max_tokens: int = 3_000) -> None:
        self._records: list[MemoryRecord] = []
        self._max_tokens = max_tokens

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def _total_tokens(self) -> int:
        return sum(self._estimate_tokens(r.content) for r in self._records)

    async def write(self, record: MemoryRecord) -> None:
        self._records.append(record)
        # Evict oldest until within budget
        while self._total_tokens() > self._max_tokens and len(self._records) > 1:
            self._records.pop(0)
        logger.debug("ContextWindow: %d records, ~%d tokens",
                     len(self._records), self._total_tokens())

    async def search(self, query_text, session_id, embedding, top_k):
        # Returns all records for the session (already in context)
        hits = [r for r in self._records if r.session_id == session_id]
        return [(r, 1.0) for r in hits[-top_k:]]

    async def delete(self, record_id: str) -> None:
        self._records = [r for r in self._records if r.record_id != record_id]

    async def delete_session(self, session_id: str) -> None:
        self._records = [r for r in self._records if r.session_id != session_id]

    async def get_by_session(self, session_id: str, limit: int = 50):
        return [r for r in self._records if r.session_id == session_id][-limit:]

    def clear(self) -> None:
        self._records.clear()


# ---------------------------------------------------------------------------
# Tier 2 – Short-term (Redis)
# ---------------------------------------------------------------------------

class RedisShortTermStore:
    """
    Stores recent turns in a Redis sorted set (score = timestamp).
    Each session has its own key; entries expire via TTL.

    Requires: redis.asyncio
    """
    tier = MemoryTier.SHORT_TERM

    def __init__(
        self,
        redis_client: Any,
        ttl_seconds: int = 86_400,
        max_turns: int = 50,
    ) -> None:
        self._redis = redis_client
        self._ttl   = ttl_seconds
        self._max   = max_turns

    def _key(self, session_id: str) -> str:
        return f"memory:short:{session_id}"

    async def write(self, record: MemoryRecord) -> None:
        key   = self._key(record.session_id)
        score = datetime.now(timezone.utc).timestamp()
        value = record.model_dump_json()
        await self._redis.zadd(key, {value: score})
        # Trim to max_turns (keep newest)
        await self._redis.zremrangebyrank(key, 0, -(self._max + 1))
        await self._redis.expire(key, self._ttl)
        logger.debug("Redis short-term write session=%s", record.session_id)

    async def search(self, query_text, session_id, embedding, top_k):
        key  = self._key(session_id)
        raw  = await self._redis.zrevrange(key, 0, top_k - 1)
        now  = datetime.now(timezone.utc).timestamp()
        results = []
        for item in raw:
            rec  = MemoryRecord.model_validate_json(item)
            age  = now - datetime.fromisoformat(rec.created_at).timestamp()
            # Recency score: 1.0 at 0 s, 0.5 at 1 h
            score = max(0.0, 1.0 - age / 7_200)
            results.append((rec, score))
        return results

    async def delete(self, record_id: str) -> None:
        # Scan all session keys (dev convenience; prod: track key per record)
        logger.warning("RedisShortTermStore.delete not optimised for production")

    async def delete_session(self, session_id: str) -> None:
        await self._redis.delete(self._key(session_id))

    async def get_by_session(self, session_id: str, limit: int = 50):
        key = self._key(session_id)
        raw = await self._redis.zrevrange(key, 0, limit - 1)
        return [MemoryRecord.model_validate_json(r) for r in raw]


# ---------------------------------------------------------------------------
# Tier 3 – Mid-term (Chroma vector store)
# ---------------------------------------------------------------------------

class ChromaMidTermStore:
    """
    Stores memory records as embeddings in ChromaDB.
    Retrieval blends cosine similarity with time-decay:
        score = α·cosine + β·e^(-λ·age_days)

    Requires: chromadb
    """
    tier = MemoryTier.MID_TERM

    def __init__(
        self,
        collection: Any,          # chromadb.Collection
        decay_factor: float = 0.05,
        alpha: float = 0.7,       # relevance weight
        beta:  float = 0.3,       # recency weight
    ) -> None:
        self._col   = collection
        self._decay = decay_factor
        self._alpha = alpha
        self._beta  = beta

    def _decay_score(self, created_at: str) -> float:
        age_days = (
            datetime.now(timezone.utc)
            - datetime.fromisoformat(created_at)
        ).total_seconds() / 86_400
        return math.exp(-self._decay * age_days)

    async def write(self, record: MemoryRecord) -> None:
        if record.embedding is None:
            logger.warning("ChromaMidTermStore: no embedding for record %s", record.record_id)
            return
        self._col.upsert(
            ids=[record.record_id],
            embeddings=[record.embedding],
            documents=[record.content],
            metadatas=[{
                "session_id":  record.session_id,
                "record_type": record.record_type.value,
                "importance":  record.importance,
                "created_at":  record.created_at,
                "expires_at":  record.expires_at or "",
            }],
        )
        logger.debug("Chroma mid-term write record_id=%s", record.record_id)

    async def search(self, query_text, session_id, embedding, top_k):
        if embedding is None:
            return []
        results = self._col.query(
            query_embeddings=[embedding],
            n_results=min(top_k * 2, 50),   # over-fetch, then rerank
            where={"session_id": session_id},
        )
        records_out = []
        for i, doc_id in enumerate(results["ids"][0]):
            meta    = results["metadatas"][0][i]
            doc     = results["documents"][0][i]
            cosine  = 1.0 - results["distances"][0][i]   # chroma returns L2; assume normalised
            decay   = self._decay_score(meta["created_at"])
            score   = self._alpha * cosine + self._beta * decay
            rec = MemoryRecord(
                record_id=doc_id,
                session_id=meta["session_id"],
                record_type=meta["record_type"],
                tier=MemoryTier.MID_TERM,
                content=doc,
                metadata=meta,
                importance=float(meta.get("importance", 0.5)),
                created_at=meta["created_at"],
            )
            records_out.append((rec, score))

        # Rerank by blended score, return top_k
        records_out.sort(key=lambda x: x[1], reverse=True)
        return records_out[:top_k]

    async def delete(self, record_id: str) -> None:
        self._col.delete(ids=[record_id])

    async def delete_session(self, session_id: str) -> None:
        self._col.delete(where={"session_id": session_id})

    async def get_by_session(self, session_id: str, limit: int = 50):
        results = self._col.get(
            where={"session_id": session_id},
            limit=limit,
        )
        out = []
        for i, doc_id in enumerate(results["ids"]):
            meta = results["metadatas"][i]
            doc  = results["documents"][i]
            out.append(MemoryRecord(
                record_id=doc_id,
                session_id=meta["session_id"],
                record_type=meta["record_type"],
                tier=MemoryTier.MID_TERM,
                content=doc,
                metadata=meta,
                created_at=meta["created_at"],
            ))
        return out


# ---------------------------------------------------------------------------
# Tier 4 – Long-term (PostgreSQL)
# ---------------------------------------------------------------------------

class PostgresLongTermStore:
    """
    Stores high-importance entities, user facts, and resolved incidents.

    Schema (run once):
    ------------------
    CREATE TABLE memory_long_term (
        record_id    TEXT PRIMARY KEY,
        session_id   TEXT NOT NULL,
        record_type  TEXT NOT NULL,
        content      TEXT NOT NULL,
        metadata     JSONB DEFAULT '{}',
        importance   FLOAT NOT NULL DEFAULT 0.5,
        access_count INT   NOT NULL DEFAULT 0,
        created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        expires_at   TIMESTAMPTZ
    );
    CREATE INDEX ON memory_long_term (session_id);
    CREATE INDEX ON memory_long_term (record_type);
    CREATE INDEX ON memory_long_term USING GIN (metadata);
    """
    tier = MemoryTier.LONG_TERM

    def __init__(self, pool: Any) -> None:   # asyncpg.Pool
        self._pool = pool

    async def write(self, record: MemoryRecord) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO memory_long_term
                  (record_id, session_id, record_type, content,
                   metadata, importance, access_count, created_at, expires_at)
                VALUES ($1,$2,$3,$4,$5::jsonb,$6,$7,$8,$9)
                ON CONFLICT (record_id) DO UPDATE
                  SET content=EXCLUDED.content,
                      metadata=EXCLUDED.metadata,
                      importance=EXCLUDED.importance
                """,
                record.record_id, record.session_id,
                record.record_type.value, record.content,
                json.dumps(record.metadata), record.importance,
                record.access_count,
                datetime.fromisoformat(record.created_at),
                datetime.fromisoformat(record.expires_at) if record.expires_at else None,
            )
        logger.debug("Postgres long-term write record_id=%s", record.record_id)

    async def search(self, query_text, session_id, embedding, top_k):
        # Full-text search via pg_trgm (add: CREATE EXTENSION pg_trgm)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT *, similarity(content, $1) AS sim
                FROM memory_long_term
                WHERE session_id = $2
                ORDER BY sim DESC
                LIMIT $3
                """,
                query_text, session_id, top_k,
            )
        results = []
        for row in rows:
            rec = self._row_to_record(row)
            results.append((rec, float(row["sim"])))
        return results

    async def delete(self, record_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM memory_long_term WHERE record_id=$1", record_id
            )

    async def delete_session(self, session_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM memory_long_term WHERE session_id=$1", session_id
            )

    async def get_by_session(self, session_id: str, limit: int = 50):
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM memory_long_term WHERE session_id=$1 "
                "ORDER BY importance DESC, created_at DESC LIMIT $2",
                session_id, limit,
            )
        return [self._row_to_record(r) for r in rows]

    @staticmethod
    def _row_to_record(row: Any) -> MemoryRecord:
        from ..schemas import MemoryRecordType
        return MemoryRecord(
            record_id=row["record_id"],
            session_id=row["session_id"],
            record_type=MemoryRecordType(row["record_type"]),
            tier=MemoryTier.LONG_TERM,
            content=row["content"],
            metadata=dict(row["metadata"]),
            importance=row["importance"],
            access_count=row["access_count"],
            created_at=row["created_at"].isoformat(),
        )
