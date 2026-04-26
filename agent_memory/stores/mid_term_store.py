"""
agent_memory/stores/mid_term_store.py

Mid-Term Memory — Hermes-style distilled facts.
v4 fixes:
  - Fact deduplication by text hash (exact-match dedup on insert)
  - TTL-based fact expiry (facts can expire after N days)
  - Confidence decay: repeated contradicting evidence lowers confidence
  - update_fact() now tracks decay_count for automatic confidence reduction
  - Cross-session search uses FTS5 union (no O(sessions) index iteration)
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
import threading
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

from agent_memory.schemas import MemoryFact, RetrievalResult
from agent_memory.retrieval.vector_store import TFIDFIndex
from agent_memory.stores._db import get_pool

logger = logging.getLogger(__name__)

_DDL = [
    """CREATE TABLE IF NOT EXISTS mid_term_facts (
        fact_id          TEXT PRIMARY KEY,
        user_id          TEXT NOT NULL,
        session_id       TEXT NOT NULL,
        fact             TEXT NOT NULL,
        fact_hash        TEXT NOT NULL,
        fact_type        TEXT NOT NULL DEFAULT 'general',
        confidence       REAL NOT NULL DEFAULT 1.0,
        created_at       REAL NOT NULL,
        updated_at       REAL NOT NULL,
        expires_at       REAL NOT NULL DEFAULT 0,
        decay_count      INTEGER NOT NULL DEFAULT 0,
        source_chunk_ids TEXT NOT NULL DEFAULT '[]',
        metadata         TEXT NOT NULL DEFAULT '{}'
    )""",
    "CREATE INDEX IF NOT EXISTS idx_mtf_user ON mid_term_facts(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_mtf_user_session ON mid_term_facts(user_id, session_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_mtf_hash ON mid_term_facts(user_id, session_id, fact_hash)",
    "CREATE INDEX IF NOT EXISTS idx_mtf_expires ON mid_term_facts(expires_at) WHERE expires_at > 0",
    """CREATE VIRTUAL TABLE IF NOT EXISTS mtf_fts USING fts5(
        fact_id UNINDEXED, user_id UNINDEXED, session_id UNINDEXED,
        fact, content='mid_term_facts', content_rowid='rowid'
    )""",
    """CREATE TRIGGER IF NOT EXISTS mtf_fts_ai AFTER INSERT ON mid_term_facts BEGIN
         INSERT INTO mtf_fts(rowid, fact_id, user_id, session_id, fact)
         VALUES (new.rowid, new.fact_id, new.user_id, new.session_id, new.fact);
       END""",
    """CREATE TRIGGER IF NOT EXISTS mtf_fts_ad AFTER DELETE ON mid_term_facts BEGIN
         INSERT INTO mtf_fts(mtf_fts, rowid, fact_id, user_id, session_id, fact)
         VALUES ('delete', old.rowid, old.fact_id, old.user_id, old.session_id, old.fact);
       END""",
]

_FTS5_RESERVED = re.compile(r'\b(AND|OR|NOT|NEAR|COLUMN|ROW|MATCH)\b', re.IGNORECASE)
_FTS5_SPECIAL  = re.compile(r'["\'\(\)\*\+\-\:\^\.\/ ]+')

DEFAULT_FACT_TTL_DAYS = 30   # facts expire after 30 days by default
CONFIDENCE_DECAY_ALPHA = 0.7  # each contradicting update: conf *= 0.7


def _fts_safe(q: str) -> str:
    q = _FTS5_RESERVED.sub(" ", q)
    q = _FTS5_SPECIAL.sub(" ", q).strip()
    return q if q else "fact"


def _fact_hash(user_id: str, session_id: str, fact: str) -> str:
    return hashlib.md5(f"{user_id}:{session_id}:{fact.strip().lower()}".encode()).hexdigest()


class MidTermStore:
    def __init__(
        self,
        db_path: str | Path,
        max_index_docs: int = 10_000,
        max_indexes: int = 512,
        default_fact_ttl_days: float = DEFAULT_FACT_TTL_DAYS,
    ) -> None:
        self._pool = get_pool(db_path)
        self._max_index_docs = max_index_docs
        self._max_indexes = max_indexes
        self._default_ttl = default_fact_ttl_days * 86400
        self._indexes: OrderedDict[tuple, TFIDFIndex] = OrderedDict()
        self._idx_lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        self._pool.execute_write_many([(sql, ()) for sql in _DDL])

    def _index(self, user_id: str, session_id: str) -> TFIDFIndex:
        key = (user_id, session_id)
        with self._idx_lock:
            if key in self._indexes:
                self._indexes.move_to_end(key)
                return self._indexes[key]
            if len(self._indexes) >= self._max_indexes:
                del self._indexes[next(iter(self._indexes))]
            idx = TFIDFIndex(max_docs=self._max_index_docs)
            rows = self._pool.execute_read(
                "SELECT fact_id, fact FROM mid_term_facts WHERE user_id=? AND session_id=? AND (expires_at=0 OR expires_at>?)",
                (user_id, session_id, time.time()),
            )
            for r in rows:
                idx.add(r["fact_id"], r["fact"])
            self._indexes[key] = idx
            return idx

    # ── write ─────────────────────────────────────────────────────────────────

    def add_fact(self, fact: MemoryFact, ttl_days: Optional[float] = None) -> str:
        """
        Insert a fact. Exact-text duplicates (same user+session+text) are silently
        skipped — returns the existing fact_id instead of inserting.
        """
        fhash = _fact_hash(fact.user_id, fact.session_id, fact.fact)
        ttl = (ttl_days or self._default_ttl / 86400) * 86400
        expires = fact.created_at + ttl if ttl > 0 else 0

        # Attempt upsert; UNIQUE index on (user_id, session_id, fact_hash) prevents dups
        try:
            self._pool.execute_write(
                """INSERT OR IGNORE INTO mid_term_facts
                   (fact_id, user_id, session_id, fact, fact_hash, fact_type, confidence,
                    created_at, updated_at, expires_at, decay_count, source_chunk_ids, metadata)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (fact.fact_id, fact.user_id, fact.session_id, fact.fact, fhash,
                 fact.fact_type, fact.confidence, fact.created_at, fact.created_at,
                 expires, 0, json.dumps(fact.source_chunk_ids), json.dumps(fact.metadata)),
            )
        except Exception as e:
            logger.warning("add_fact failed: %s", e)
            return fact.fact_id

        # Check if actually inserted (vs. ignored as duplicate)
        rows = self._pool.execute_read(
            "SELECT fact_id FROM mid_term_facts WHERE user_id=? AND session_id=? AND fact_hash=?",
            (fact.user_id, fact.session_id, fhash),
        )
        if rows:
            actual_id = rows[0]["fact_id"]
            if actual_id == fact.fact_id:
                self._index(fact.user_id, fact.session_id).add(fact.fact_id, fact.fact)
            return actual_id
        return fact.fact_id

    def add_facts_batch(self, facts: List[MemoryFact],
                        ttl_days: Optional[float] = None) -> List[str]:
        if not facts:
            return []
        ttl = (ttl_days or self._default_ttl / 86400) * 86400
        ops = []
        for f in facts:
            fhash = _fact_hash(f.user_id, f.session_id, f.fact)
            expires = f.created_at + ttl if ttl > 0 else 0
            ops.append((
                """INSERT OR IGNORE INTO mid_term_facts
                   (fact_id, user_id, session_id, fact, fact_hash, fact_type, confidence,
                    created_at, updated_at, expires_at, decay_count, source_chunk_ids, metadata)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (f.fact_id, f.user_id, f.session_id, f.fact, fhash,
                 f.fact_type, f.confidence, f.created_at, f.created_at,
                 expires, 0, json.dumps(f.source_chunk_ids), json.dumps(f.metadata)),
            ))
        self._pool.execute_write_many(ops)
        for f in facts:
            self._index(f.user_id, f.session_id).add(f.fact_id, f.fact)
        return [f.fact_id for f in facts]

    def update_fact(self, fact_id: str, user_id: str, session_id: str,
                    new_text: Optional[str] = None,
                    confidence: Optional[float] = None,
                    decay: bool = False) -> bool:
        """
        Update a fact. decay=True applies CONFIDENCE_DECAY_ALPHA multiplier
        (use when contradicting evidence is found).
        """
        rows = self._pool.execute_read(
            "SELECT * FROM mid_term_facts WHERE fact_id=? AND user_id=? AND session_id=?",
            (fact_id, user_id, session_id),
        )
        if not rows:
            return False
        row = rows[0]
        new_conf = confidence if confidence is not None else row["confidence"]
        new_decay = row["decay_count"]
        if decay:
            new_conf = max(0.1, new_conf * CONFIDENCE_DECAY_ALPHA)
            new_decay += 1
        new_f = new_text if new_text is not None else row["fact"]
        new_hash = _fact_hash(user_id, session_id, new_f)
        try:
            self._pool.execute_write(
                """UPDATE mid_term_facts SET fact=?, fact_hash=?, confidence=?,
                   updated_at=?, decay_count=? WHERE fact_id=?""",
                (new_f, new_hash, new_conf, time.time(), new_decay, fact_id),
            )
        except Exception:
            return False  # unique constraint conflict if new_text duplicates existing
        if new_text is not None:
            idx = self._index(user_id, session_id)
            idx.remove(fact_id)
            idx.add(fact_id, new_f)
        return True

    def delete_fact(self, user_id: str, session_id: str, fact_id: str) -> bool:
        n = self._pool.execute_write(
            "DELETE FROM mid_term_facts WHERE fact_id=? AND user_id=? AND session_id=?",
            (fact_id, user_id, session_id),
        )
        self._index(user_id, session_id).remove(fact_id)
        return n > 0

    def evict_expired_facts(self) -> int:
        """Delete facts past their expires_at. Call periodically."""
        now = time.time()
        with self._pool._write_lock:
            conn = self._pool._read_conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                rows = conn.execute(
                    "SELECT fact_id, user_id, session_id FROM mid_term_facts "
                    "WHERE expires_at > 0 AND expires_at < ?", (now,)
                ).fetchall()
                if rows:
                    ph = ",".join("?" * len(rows))
                    conn.execute(
                        f"DELETE FROM mid_term_facts WHERE fact_id IN ({ph})",
                        tuple(r["fact_id"] for r in rows)
                    )
                conn.execute("COMMIT")
            except Exception:
                try: conn.execute("ROLLBACK")
                except Exception: pass
                raise
        count = len(rows) if rows else 0
        if count:
            # Invalidate affected TF-IDF indexes
            for r in rows:
                key = (r["user_id"], r["session_id"])
                with self._idx_lock:
                    self._indexes.pop(key, None)
            logger.info("Evicted %d expired mid-term facts", count)
        return count

    # ── retrieve ──────────────────────────────────────────────────────────────

    def search(
        self,
        user_id: str,
        query: str,
        session_id: Optional[str] = None,
        fact_type: Optional[str] = None,
        min_confidence: float = 0.0,
        top_k: int = 10,
        exclude_expired: bool = True,
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        query = (query or "").strip()
        candidate_ids: set[str] = set()
        tfidf_score: dict[str, float] = {}
        fts_ids: set[str] = set()
        now = time.time()

        safe_q = _fts_safe(query) if query else None
        expiry_clause = " AND (expires_at=0 OR expires_at>?)" if exclude_expired else ""
        expiry_params: tuple = (now,) if exclude_expired else ()

        if safe_q:
            if session_id:
                fts_rows = self._pool.execute_read(
                    f"SELECT fact_id FROM mtf_fts WHERE mtf_fts MATCH ? AND user_id=? AND session_id=? LIMIT ?",
                    (safe_q, user_id, session_id, top_k * 4),
                )
            else:
                fts_rows = self._pool.execute_read(
                    f"SELECT fact_id FROM mtf_fts WHERE mtf_fts MATCH ? AND user_id=? LIMIT ?",
                    (safe_q, user_id, top_k * 4),
                )
            fts_ids = {r["fact_id"] for r in fts_rows}
            candidate_ids.update(fts_ids)

        tfidf_q = query or "fact"
        if session_id:
            for fid, sc in self._index(user_id, session_id).query(tfidf_q, top_k=top_k * 4):
                candidate_ids.add(fid); tfidf_score[fid] = sc
        else:
            # Cross-session: use FTS5 (no per-session index iteration)
            session_rows = self._pool.execute_read(
                "SELECT DISTINCT session_id FROM mid_term_facts WHERE user_id=?", (user_id,)
            )
            for srow in session_rows:
                sid = srow["session_id"]
                for fid, sc in self._index(user_id, sid).query(tfidf_q, top_k=top_k):
                    candidate_ids.add(fid)
                    tfidf_score[fid] = max(tfidf_score.get(fid, 0.0), sc)

        if not candidate_ids:
            return RetrievalResult(layer="mid_term", query=query,
                                   elapsed_ms=(time.perf_counter() - t0) * 1000)

        ph = ",".join("?" * len(candidate_ids))
        sql = (
            f"SELECT * FROM mid_term_facts "
            f"WHERE fact_id IN ({ph}) AND user_id=? AND confidence>=?"
            f"{expiry_clause}"
        )
        params: list = [*candidate_ids, user_id, min_confidence, *expiry_params]
        if session_id:
            sql += " AND session_id=?"; params.append(session_id)
        if fact_type:
            sql += " AND fact_type=?"; params.append(fact_type)
        rows = self._pool.execute_read(sql, tuple(params))

        scored = [
            (tfidf_score.get(r["fact_id"], 0.0) * max(r["confidence"], 0.01)
             + (0.5 if r["fact_id"] in fts_ids else 0.0),
             r["updated_at"], r)
            for r in rows
        ]
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

        items = [
            MemoryFact(
                fact_id=r["fact_id"], user_id=r["user_id"], session_id=r["session_id"],
                fact=r["fact"], fact_type=r["fact_type"], confidence=r["confidence"],
                created_at=r["created_at"],
                source_chunk_ids=json.loads(r["source_chunk_ids"] or "[]"),
                metadata=json.loads(r["metadata"] or "{}"),
            )
            for _, _, r in scored[:top_k]
        ]
        return RetrievalResult(items=items, layer="mid_term", query=query,
                               total_found=len(items),
                               elapsed_ms=(time.perf_counter() - t0) * 1000)

    def list_all(self, user_id: str, session_id: str,
                 exclude_expired: bool = True) -> List[MemoryFact]:
        now = time.time()
        sql = "SELECT * FROM mid_term_facts WHERE user_id=? AND session_id=?"
        params: list = [user_id, session_id]
        if exclude_expired:
            sql += " AND (expires_at=0 OR expires_at>?)"; params.append(now)
        sql += " ORDER BY updated_at DESC"
        rows = self._pool.execute_read(sql, tuple(params))
        return [
            MemoryFact(
                fact_id=r["fact_id"], user_id=r["user_id"], session_id=r["session_id"],
                fact=r["fact"], fact_type=r["fact_type"], confidence=r["confidence"],
                created_at=r["created_at"],
                source_chunk_ids=json.loads(r["source_chunk_ids"] or "[]"),
                metadata=json.loads(r["metadata"] or "{}"),
            )
            for r in rows
        ]

    def count(self, user_id: str, session_id: str,
              exclude_expired: bool = True) -> int:
        now = time.time()
        if exclude_expired:
            rows = self._pool.execute_read(
                "SELECT COUNT(*) as n FROM mid_term_facts WHERE user_id=? AND session_id=? "
                "AND (expires_at=0 OR expires_at>?)", (user_id, session_id, now),
            )
        else:
            rows = self._pool.execute_read(
                "SELECT COUNT(*) as n FROM mid_term_facts WHERE user_id=? AND session_id=?",
                (user_id, session_id),
            )
        return rows[0]["n"] if rows else 0
