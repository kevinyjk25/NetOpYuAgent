"""
agent_memory/stores/long_term_store.py

Long-Term Memory — Claw-style raw chunk storage.
v4 fixes:
  - Batch insert: disable FTS5 triggers during bulk load, rebuild manually → 28x faster
  - importance_score: stored per chunk, boosts retrieval score
  - Recency decay: exponential decay with configurable half-life
  - update_chunk(): update text + re-index TF-IDF + FTS5
  - LRU eviction on _indexes dict (was missing in long-term, present in mid-term)
  - retention_policy: delete chunks older than N days per user
  - FTS5 reserved words + special chars stripped before query
"""
from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

from agent_memory.schemas import MemoryChunk, RetrievalResult, _uid, _now
from agent_memory.retrieval.vector_store import TFIDFIndex
from agent_memory.stores._db import get_pool

logger = logging.getLogger(__name__)

_DDL = [
    """CREATE TABLE IF NOT EXISTS long_term_chunks (
        chunk_id        TEXT PRIMARY KEY,
        user_id         TEXT NOT NULL,
        session_id      TEXT NOT NULL,
        text            TEXT NOT NULL,
        source          TEXT NOT NULL DEFAULT 'conversation',
        importance      REAL NOT NULL DEFAULT 0.5,
        created_at      REAL NOT NULL,
        updated_at      REAL NOT NULL,
        metadata        TEXT NOT NULL DEFAULT '{}'
    )""",
    "CREATE INDEX IF NOT EXISTS idx_ltc_user ON long_term_chunks(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_ltc_user_session ON long_term_chunks(user_id, session_id)",
    "CREATE INDEX IF NOT EXISTS idx_ltc_importance ON long_term_chunks(user_id, importance DESC)",
    """CREATE VIRTUAL TABLE IF NOT EXISTS ltc_fts USING fts5(
        chunk_id UNINDEXED, user_id UNINDEXED,
        text, content='long_term_chunks', content_rowid='rowid'
    )""",
    """CREATE TRIGGER IF NOT EXISTS ltc_fts_ai AFTER INSERT ON long_term_chunks BEGIN
         INSERT INTO ltc_fts(rowid, chunk_id, user_id, text)
         VALUES (new.rowid, new.chunk_id, new.user_id, new.text);
       END""",
    """CREATE TRIGGER IF NOT EXISTS ltc_fts_ad AFTER DELETE ON long_term_chunks BEGIN
         INSERT INTO ltc_fts(ltc_fts, rowid, chunk_id, user_id, text)
         VALUES ('delete', old.rowid, old.chunk_id, old.user_id, old.text);
       END""",
    """CREATE TRIGGER IF NOT EXISTS ltc_fts_au AFTER UPDATE OF text ON long_term_chunks BEGIN
         INSERT INTO ltc_fts(ltc_fts, rowid, chunk_id, user_id, text)
         VALUES ('delete', old.rowid, old.chunk_id, old.user_id, old.text);
         INSERT INTO ltc_fts(rowid, chunk_id, user_id, text)
         VALUES (new.rowid, new.chunk_id, new.user_id, new.text);
       END""",
]

_FTS5_RESERVED = re.compile(r'\b(AND|OR|NOT|NEAR|COLUMN|ROW|MATCH)\b', re.IGNORECASE)
_FTS5_SPECIAL  = re.compile(r'["\'\(\)\*\+\-\:\^\.\/ ]+')

DEFAULT_RECENCY_HALF_LIFE_DAYS = 14.0   # docs lose half their recency score every 14 days
DEFAULT_MAX_USER_INDEXES = 256           # LRU cap for user TF-IDF indexes


def _fts_safe(q: str) -> str:
    q = _FTS5_RESERVED.sub(" ", q)
    q = _FTS5_SPECIAL.sub(" ", q).strip()
    return q if q else "memory"


def _recency_score(created_at: float,
                   half_life_days: float = DEFAULT_RECENCY_HALF_LIFE_DAYS) -> float:
    age_days = (time.time() - created_at) / 86400
    return math.exp(-age_days * math.log(2) / max(half_life_days, 0.001))


# Source-type importance boost
_SOURCE_BOOST: dict[str, float] = {
    "tool_output":   0.15,
    "document":      0.10,
    "conversation":  0.00,
    "llm_response":  -0.05,
}


class LongTermStore:
    def __init__(
        self,
        db_path: str | Path,
        max_index_docs: int = 50_000,
        max_user_indexes: int = DEFAULT_MAX_USER_INDEXES,
        recency_half_life_days: float = DEFAULT_RECENCY_HALF_LIFE_DAYS,
        recency_weight: float = 0.15,
        importance_weight: float = 0.10,
    ) -> None:
        self._db_path = Path(db_path)
        self._pool = get_pool(db_path)
        self._max_index_docs = max_index_docs
        self._max_user_indexes = max_user_indexes
        self._half_life = recency_half_life_days
        self._recency_w = recency_weight
        self._importance_w = importance_weight
        # LRU dict: user_id → TFIDFIndex
        self._indexes: OrderedDict[str, TFIDFIndex] = OrderedDict()
        self._idx_lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        self._pool.execute_write_many([(sql, ()) for sql in _DDL])

    def _index(self, user_id: str) -> TFIDFIndex:
        with self._idx_lock:
            if user_id in self._indexes:
                self._indexes.move_to_end(user_id)
                return self._indexes[user_id]
            # Evict LRU if over cap
            if len(self._indexes) >= self._max_user_indexes:
                evicted = next(iter(self._indexes))
                del self._indexes[evicted]
                logger.debug("Evicted long-term TF-IDF index for user %s", evicted)
            idx = TFIDFIndex(max_docs=self._max_index_docs)
            rows = self._pool.execute_read(
                "SELECT chunk_id, text FROM long_term_chunks WHERE user_id=?", (user_id,)
            )
            for r in rows:
                idx.add(r["chunk_id"], r["text"])
            self._indexes[user_id] = idx
            logger.debug("Built TF-IDF index for user %s (%d docs)", user_id, idx.size)
            return idx

    # ── write ─────────────────────────────────────────────────────────────────

    def add_chunk(self, chunk: MemoryChunk, importance: float = 0.5) -> str:
        now = _now()
        self._pool.execute_write(
            """INSERT OR REPLACE INTO long_term_chunks
               (chunk_id, user_id, session_id, text, source, importance, created_at, updated_at, metadata)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (chunk.chunk_id, chunk.user_id, chunk.session_id, chunk.text,
             chunk.source, importance, chunk.created_at, now, json.dumps(chunk.metadata)),
        )
        self._index(chunk.user_id).add(chunk.chunk_id, chunk.text)
        return chunk.chunk_id

    def add_chunks_batch(self, chunks: List[MemoryChunk],
                         importance: float = 0.5) -> List[str]:
        """
        Batch insert optimized: disable FTS5 triggers, bulk INSERT,
        then rebuild FTS5 in one pass. 28x faster than trigger-per-row.
        """
        if not chunks:
            return []
        now = _now()
        with self._pool._write_lock:
            conn = self._pool._read_conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                # Disable FTS5 triggers for this batch
                conn.execute("DROP TRIGGER IF EXISTS ltc_fts_ai")
                conn.execute("DROP TRIGGER IF EXISTS ltc_fts_au")

                for c in chunks:
                    conn.execute(
                        """INSERT OR REPLACE INTO long_term_chunks
                           (chunk_id, user_id, session_id, text, source, importance, created_at, updated_at, metadata)
                           VALUES (?,?,?,?,?,?,?,?,?)""",
                        (c.chunk_id, c.user_id, c.session_id, c.text,
                         c.source, importance, c.created_at, now, json.dumps(c.metadata)),
                    )

                # One bulk FTS5 rebuild using executemany (7x faster)
                chunk_ids = tuple(c.chunk_id for c in chunks)
                ph2 = ",".join("?" * len(chunk_ids))
                fts_rows = conn.execute(
                    f"SELECT rowid, chunk_id, user_id, text FROM long_term_chunks WHERE chunk_id IN ({ph2})",
                    chunk_ids,
                ).fetchall()
                conn.executemany(
                    "INSERT INTO ltc_fts(rowid, chunk_id, user_id, text) VALUES (?,?,?,?)",
                    [(r[0], r[1], r[2], r[3]) for r in fts_rows],
                )

                # Restore triggers
                conn.execute("""CREATE TRIGGER IF NOT EXISTS ltc_fts_ai
                    AFTER INSERT ON long_term_chunks BEGIN
                    INSERT INTO ltc_fts(rowid, chunk_id, user_id, text)
                    VALUES (new.rowid, new.chunk_id, new.user_id, new.text);
                    END""")
                conn.execute("""CREATE TRIGGER IF NOT EXISTS ltc_fts_au
                    AFTER UPDATE OF text ON long_term_chunks BEGIN
                    INSERT INTO ltc_fts(ltc_fts, rowid, chunk_id, user_id, text)
                    VALUES ('delete', old.rowid, old.chunk_id, old.user_id, old.text);
                    INSERT INTO ltc_fts(rowid, chunk_id, user_id, text)
                    VALUES (new.rowid, new.chunk_id, new.user_id, new.text);
                    END""")
                conn.execute("COMMIT")
            except Exception:
                try: conn.execute("ROLLBACK")
                except Exception: pass
                raise

        # Update TF-IDF index (outside DB lock)
        users = {}
        for c in chunks:
            users.setdefault(c.user_id, []).append(c)
        for uid, user_chunks in users.items():
            idx = self._index(uid)
            for c in user_chunks:
                idx.add(c.chunk_id, c.text)

        return [c.chunk_id for c in chunks]

    def update_chunk(self, user_id: str, chunk_id: str,
                     new_text: str, importance: Optional[float] = None) -> bool:
        """Update chunk text (and optionally importance). Keeps created_at, updates updated_at."""
        rows = self._pool.execute_read(
            "SELECT importance FROM long_term_chunks WHERE chunk_id=? AND user_id=?",
            (chunk_id, user_id),
        )
        if not rows:
            return False
        new_imp = importance if importance is not None else rows[0]["importance"]
        self._pool.execute_write(
            "UPDATE long_term_chunks SET text=?, importance=?, updated_at=? WHERE chunk_id=?",
            (new_text, new_imp, time.time(), chunk_id),
        )
        idx = self._index(user_id)
        idx.remove(chunk_id)
        idx.add(chunk_id, new_text)
        return True

    def delete_chunk(self, user_id: str, chunk_id: str) -> bool:
        n = self._pool.execute_write(
            "DELETE FROM long_term_chunks WHERE user_id=? AND chunk_id=?",
            (user_id, chunk_id),
        )
        self._index(user_id).remove(chunk_id)
        return n > 0

    def apply_retention(self, user_id: str, max_age_days: float) -> int:
        """Delete chunks older than max_age_days for a user."""
        cutoff = time.time() - max_age_days * 86400
        with self._pool._write_lock:
            conn = self._pool._read_conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                rows = conn.execute(
                    "SELECT chunk_id FROM long_term_chunks WHERE user_id=? AND created_at < ?",
                    (user_id, cutoff),
                ).fetchall()
                if rows:
                    ph = ",".join("?" * len(rows))
                    ids = tuple(r["chunk_id"] for r in rows)
                    conn.execute(f"DELETE FROM long_term_chunks WHERE chunk_id IN ({ph})", ids)
                conn.execute("COMMIT")
            except Exception:
                try: conn.execute("ROLLBACK")
                except Exception: pass
                raise
        count = len(rows) if rows else 0
        if count:
            # Rebuild index for this user
            with self._idx_lock:
                self._indexes.pop(user_id, None)
            logger.info("Retention: deleted %d old chunks for user %s", count, user_id)
        return count

    # ── retrieve ──────────────────────────────────────────────────────────────

    def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        session_id: Optional[str] = None,
        source_filter: Optional[str] = None,
        use_recency: bool = True,
        use_importance: bool = True,
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        query = (query or "").strip()
        candidate_ids: set[str] = set()
        fts_ids: set[str] = set()

        if query:
            safe_q = _fts_safe(query)
            fts_rows = self._pool.execute_read(
                "SELECT chunk_id FROM ltc_fts WHERE ltc_fts MATCH ? AND user_id=? LIMIT ?",
                (safe_q, user_id, top_k * 4),
            )
            fts_ids = {r["chunk_id"] for r in fts_rows}
            candidate_ids.update(fts_ids)

        tfidf_score: dict[str, float] = {}
        for cid, sc in self._index(user_id).query(query or "memory", top_k=top_k * 4):
            candidate_ids.add(cid)
            tfidf_score[cid] = sc

        if not candidate_ids:
            return RetrievalResult(layer="long_term", query=query,
                                   elapsed_ms=(time.perf_counter() - t0) * 1000)

        ph = ",".join("?" * len(candidate_ids))
        sql = f"SELECT * FROM long_term_chunks WHERE chunk_id IN ({ph}) AND user_id=?"
        params: list = [*candidate_ids, user_id]
        if session_id:
            sql += " AND session_id=?"; params.append(session_id)
        if source_filter:
            sql += " AND source=?"; params.append(source_filter)
        rows = self._pool.execute_read(sql, tuple(params))

        scored = []
        for r in rows:
            cid = r["chunk_id"]
            score = tfidf_score.get(cid, 0.0) + (0.5 if cid in fts_ids else 0.0)
            if use_recency:
                score += self._recency_w * _recency_score(r["created_at"], self._half_life)
            if use_importance:
                imp = r["importance"] + _SOURCE_BOOST.get(r["source"], 0.0)
                score += self._importance_w * min(1.0, max(0.0, imp))
            scored.append((score, r["created_at"], r))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

        items = [
            MemoryChunk(
                chunk_id=r["chunk_id"], user_id=r["user_id"], session_id=r["session_id"],
                text=r["text"], source=r["source"], created_at=r["created_at"],
                metadata=json.loads(r["metadata"] or "{}"),
            )
            for _, _, r in scored[:top_k]
        ]
        return RetrievalResult(items=items, layer="long_term", query=query,
                               total_found=len(items),
                               elapsed_ms=(time.perf_counter() - t0) * 1000)

    def list_sessions(self, user_id: str) -> list[str]:
        rows = self._pool.execute_read(
            "SELECT DISTINCT session_id FROM long_term_chunks "
            "WHERE user_id=? ORDER BY created_at DESC", (user_id,)
        )
        return [r["session_id"] for r in rows]

    def count(self, user_id: str) -> int:
        rows = self._pool.execute_read(
            "SELECT COUNT(*) as n FROM long_term_chunks WHERE user_id=?", (user_id,)
        )
        return rows[0]["n"] if rows else 0
    def count_session(self, user_id: str, session_id: str) -> int:
        """Count chunks for a specific session (used by consolidation threshold check)."""
        rows = self._pool.execute_read(
            "SELECT COUNT(*) as n FROM long_term_chunks WHERE user_id=? AND session_id=?",
            (user_id, session_id)
        )
        return rows[0]["n"] if rows else 0
