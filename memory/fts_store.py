"""
memory/fts_store.py
--------------------
FTS5SessionStore — SQLite FTS5 full-text search across all past sessions.

Hermes innovation this implements
-----------------------------------
§04 第一层：会话记忆 (Session Memory — Layer 1)
  "每轮对话的内容、工具调用和返回结果，全部写入SQLite数据库，
   同时建FTS5全文搜索索引。关键设计决策是按需检索而不是全量加载。"

§04 FTS5召回
  "新对话开始前，根据当前话题搜索历史记忆，把相关内容加载到上文中。
   不是加载全部历史，是按需搜索。"

Why this is better than your current L1 ContextWindowStore
-----------------------------------------------------------
  Current L1:  in-process list, token-capped, lost on restart, no search
  FTS5 store:  persists across restarts, full-text search in milliseconds,
               returns only relevant excerpts (not full session dumps),
               LLM summarizes retrieved results before injection

Key design decisions (identical to Hermes)
-------------------------------------------
  1. Every turn written to SQLite immediately after completion
  2. FTS5 virtual table for token-cost-free keyword search
  3. Before each new session, FTS5 query finds relevant past context
  4. Retrieved excerpts are passed through an LLM summarizer (auxiliary call)
     so only a compact, useful summary enters the main prompt — not raw text
  5. WAL mode for safe concurrent reads from multiple sessions
  6. Data lives in a single file (state.db) — portable, no external server

Usage
-----
    store = FTS5SessionStore("./data/state.db")
    await store.initialize()

    # After each turn:
    await store.write_turn(session_id, user_text, assistant_text, tool_calls)

    # Before a new session / at start of each turn:
    results = await store.search(query="authentication failure RADIUS", limit=5)
    summary = await store.summarize_results(results, current_query)
    # → inject summary into prompt context
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = """
-- Main session turns table
CREATE TABLE IF NOT EXISTS session_turns (
    turn_id       TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL,
    ts            REAL NOT NULL,
    user_text     TEXT NOT NULL,
    assistant_text TEXT NOT NULL,
    tool_calls    TEXT NOT NULL DEFAULT '[]',
    importance    REAL NOT NULL DEFAULT 0.5,
    tags          TEXT NOT NULL DEFAULT '[]'
);

-- FTS5 virtual table — standalone (no content= to avoid JOIN issues)
-- Stores turn_id and session_id alongside text_blob for direct retrieval
CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
    turn_id,
    session_id,
    text_blob
);

-- Trigger: keep FTS in sync on insert
CREATE TRIGGER IF NOT EXISTS turns_fts_insert
AFTER INSERT ON session_turns BEGIN
    INSERT INTO turns_fts(turn_id, session_id, text_blob)
    VALUES (new.turn_id, new.session_id,
            new.user_text || ' ' || new.assistant_text);
END;

-- Trigger: keep FTS in sync on delete
CREATE TRIGGER IF NOT EXISTS turns_fts_delete
BEFORE DELETE ON session_turns BEGIN
    DELETE FROM turns_fts WHERE turn_id = old.turn_id;
END;

-- Per-session metadata
CREATE TABLE IF NOT EXISTS sessions (
    session_id    TEXT PRIMARY KEY,
    created_at    REAL NOT NULL,
    last_active   REAL NOT NULL,
    topic_summary TEXT,
    turn_count    INTEGER NOT NULL DEFAULT 0,
    platform      TEXT NOT NULL DEFAULT 'cli'
);

-- Nudge log
CREATE TABLE IF NOT EXISTS nudge_log (
    nudge_id      TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL,
    ts            REAL NOT NULL,
    turns_evaluated INTEGER NOT NULL,
    memories_created INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_turns_session    ON session_turns(session_id);
CREATE INDEX IF NOT EXISTS idx_turns_ts         ON session_turns(ts DESC);
CREATE INDEX IF NOT EXISTS idx_turns_importance ON session_turns(importance DESC);
"""


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class TurnRecord:
    turn_id:        str
    session_id:     str
    ts:             float
    user_text:      str
    assistant_text: str
    tool_calls:     list[dict]
    importance:     float
    tags:           list[str]


@dataclass
class FTSSearchResult:
    turn_id:        str
    session_id:     str
    ts:             float
    user_text:      str
    assistant_text: str
    snippet:        str     # FTS5 snippet (highlighted excerpt)
    rank:           float   # BM25 rank (lower = better match in SQLite)
    tool_calls:     list[dict]


@dataclass
class SessionSummary:
    session_id:    str
    created_at:    float
    last_active:   float
    topic_summary: Optional[str]
    turn_count:    int
    platform:      str


# ---------------------------------------------------------------------------
# FTS5SessionStore
# ---------------------------------------------------------------------------

class FTS5SessionStore:
    """
    SQLite FTS5 session store for cross-session full-text recall.

    Thread-safety: all DB ops run in an executor thread (SQLite is not
    asyncio-native) with a single shared connection in WAL mode.
    """

    def __init__(
        self,
        db_path: str | Path = "./data/state.db",
        summarizer: Optional[Callable] = None,
        max_search_results: int = 8,
        snippet_tokens: int = 60,
    ) -> None:
        self._db_path          = Path(db_path)
        self._summarizer        = summarizer   # async fn(results, query) -> str
        self._max_results       = max_search_results
        self._snippet_tokens    = snippet_tokens
        self._conn: Optional[sqlite3.Connection] = None
        self._lock              = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create DB file, apply schema, enable WAL mode."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._sync_initialize)
        logger.info("FTS5SessionStore: initialized at %s", self._db_path)

    def _sync_initialize(self) -> None:
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            isolation_level=None,   # autocommit
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_DDL)

    async def close(self) -> None:
        if self._conn:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._conn.close)

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    async def write_turn(
        self,
        session_id:     str,
        user_text:      str,
        assistant_text: str,
        tool_calls:     Optional[list[dict]] = None,
        importance:     float = 0.5,
        tags:           Optional[list[str]] = None,
    ) -> TurnRecord:
        """
        Write one completed conversation turn to the FTS5 store.
        Called by MemoryRouter after each turn completes.
        """
        turn = TurnRecord(
            turn_id=str(uuid.uuid4()),
            session_id=session_id,
            ts=time.time(),
            user_text=user_text,
            assistant_text=assistant_text,
            tool_calls=tool_calls or [],
            importance=importance,
            tags=tags or [],
        )
        async with self._lock:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._sync_write_turn, turn)
        return turn

    def _sync_write_turn(self, turn: TurnRecord) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO session_turns
               (turn_id, session_id, ts, user_text, assistant_text,
                tool_calls, importance, tags)
               VALUES (?,?,?,?,?,?,?,?)""",
            (turn.turn_id, turn.session_id, turn.ts,
             turn.user_text, turn.assistant_text,
             json.dumps(turn.tool_calls), turn.importance,
             json.dumps(turn.tags)),
        )
        # Upsert session metadata
        self._conn.execute(
            """INSERT INTO sessions (session_id, created_at, last_active, turn_count)
               VALUES (?,?,?,1)
               ON CONFLICT(session_id) DO UPDATE SET
                 last_active=excluded.last_active,
                 turn_count=turn_count+1""",
            (turn.session_id, turn.ts, turn.ts),
        )

    async def update_session_topic(self, session_id: str, topic: str) -> None:
        """Store an LLM-generated topic summary for a session."""
        async with self._lock:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._conn.execute(
                    "UPDATE sessions SET topic_summary=? WHERE session_id=?",
                    (topic, session_id),
                ),
            )

    # ------------------------------------------------------------------
    # FTS5 Search API  (the core Hermes innovation)
    # ------------------------------------------------------------------

    async def search(
        self,
        query:           str,
        limit:           int = 8,
        session_exclude: Optional[str] = None,   # exclude current session
        min_importance:  float = 0.0,
        since_ts:        Optional[float] = None,
    ) -> list[FTSSearchResult]:
        """
        Full-text search across all stored turns.

        Uses SQLite FTS5 BM25 ranking — returns the most relevant past
        conversation excerpts, sorted by relevance score.

        This is the key Hermes recall mechanism:
          "根据当前话题搜索历史记忆，把相关内容加载到上文中"
        """
        # Sanitize query for FTS5 (escape special chars)
        fts_query = self._sanitize_fts_query(query)
        if not fts_query:
            return []

        async with self._lock:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, self._sync_search,
                fts_query, limit, session_exclude,
                min_importance, since_ts,
            )
        return results

    def _sync_search(
        self,
        fts_query:      str,
        limit:          int,
        session_exclude: Optional[str],
        min_importance:  float,
        since_ts:        Optional[float],
    ) -> list[FTSSearchResult]:
        try:
            # First: FTS5 match to get turn_ids and snippets
            fts_sql = """
                SELECT
                    turn_id,
                    session_id,
                    snippet(turns_fts, 2, '<b>', '</b>', '...', ?) AS snippet,
                    rank
                FROM turns_fts
                WHERE turns_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """
            fts_rows = self._conn.execute(
                fts_sql, (self._snippet_tokens, fts_query, limit * 2)
            ).fetchall()

            if not fts_rows:
                return []

            # Second: fetch full turn details from session_turns, applying filters
            results = []
            for row in fts_rows:
                turn_id    = row["turn_id"]
                session_id = row["session_id"]
                snippet    = row["snippet"] or ""
                rank       = row["rank"] or 0.0

                # Apply filters
                if session_exclude and session_id == session_exclude:
                    continue

                detail_sql = """
                    SELECT ts, user_text, assistant_text, tool_calls, importance
                    FROM session_turns WHERE turn_id = ?
                """
                params = [turn_id]
                detail = self._conn.execute(detail_sql, params).fetchone()
                if detail is None:
                    continue
                if detail["importance"] < min_importance:
                    continue
                if since_ts and detail["ts"] < since_ts:
                    continue

                results.append(FTSSearchResult(
                    turn_id=turn_id,
                    session_id=session_id,
                    ts=detail["ts"],
                    user_text=detail["user_text"],
                    assistant_text=detail["assistant_text"],
                    snippet=snippet,
                    rank=rank,
                    tool_calls=json.loads(detail["tool_calls"] or "[]"),
                ))
                if len(results) >= limit:
                    break

            return results
        except sqlite3.OperationalError as exc:
            logger.warning("FTS5 search error (query=%r): %s", fts_query, exc)
            return []

    # ------------------------------------------------------------------
    # LLM Summarization (Hermes §04: retrieved → summarized → injected)
    # ------------------------------------------------------------------

    async def summarize_results(
        self,
        results:       list[FTSSearchResult],
        current_query: str,
        max_chars:     int = 1200,
    ) -> str:
        """
        Compress FTS5 search results into a concise context summary.

        Hermes design: retrieved raw excerpts → auxiliary LLM call →
        compact summary → inject into main prompt.

        If no summarizer is configured, falls back to a structured
        text assembly of the top snippets.
        """
        if not results:
            return ""

        if self._summarizer is not None:
            try:
                return await self._summarizer(results, current_query)
            except Exception as exc:
                logger.warning("FTS5 summarizer failed: %s — using fallback", exc)

        # Fallback: structured text assembly
        import datetime
        lines = [f"[PAST SESSION CONTEXT — relevant to: {current_query}]"]
        for i, r in enumerate(results[:5], 1):
            age = datetime.datetime.fromtimestamp(r.ts).strftime("%Y-%m-%d %H:%M")
            snippet = r.snippet.replace("<b>", "**").replace("</b>", "**")
            lines.append(f"  {i}. [{age}] {snippet[:240]}")
        summary = "\n".join(lines)

        # Trim to max_chars
        if len(summary) > max_chars:
            summary = summary[:max_chars] + "\n... [truncated]"
        return summary

    # ------------------------------------------------------------------
    # Session query API
    # ------------------------------------------------------------------

    async def get_session_turns(
        self,
        session_id: str,
        limit: int = 50,
        min_importance: float = 0.0,
    ) -> list[TurnRecord]:
        """Get turns for a specific session, most recent first."""
        async with self._lock:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self._sync_get_session_turns,
                session_id, limit, min_importance,
            )

    def _sync_get_session_turns(
        self, session_id: str, limit: int, min_importance: float
    ) -> list[TurnRecord]:
        rows = self._conn.execute(
            """SELECT * FROM session_turns
               WHERE session_id=? AND importance>=?
               ORDER BY ts DESC LIMIT ?""",
            (session_id, min_importance, limit),
        ).fetchall()
        return [
            TurnRecord(
                turn_id=r["turn_id"], session_id=r["session_id"],
                ts=r["ts"], user_text=r["user_text"],
                assistant_text=r["assistant_text"],
                tool_calls=json.loads(r["tool_calls"] or "[]"),
                importance=r["importance"],
                tags=json.loads(r["tags"] or "[]"),
            )
            for r in rows
        ]

    async def list_sessions(
        self, limit: int = 20, platform: Optional[str] = None
    ) -> list[SessionSummary]:
        """List recent sessions ordered by last activity."""
        async with self._lock:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self._sync_list_sessions, limit, platform,
            )

    def _sync_list_sessions(
        self, limit: int, platform: Optional[str]
    ) -> list[SessionSummary]:
        where = "WHERE platform=?" if platform else ""
        params: list = [platform] if platform else []
        params.append(limit)
        rows = self._conn.execute(
            f"""SELECT * FROM sessions {where}
                ORDER BY last_active DESC LIMIT ?""",
            params,
        ).fetchall()
        return [
            SessionSummary(
                session_id=r["session_id"],
                created_at=r["created_at"],
                last_active=r["last_active"],
                topic_summary=r["topic_summary"],
                turn_count=r["turn_count"],
                platform=r["platform"],
            )
            for r in rows
        ]

    async def get_stats(self) -> dict[str, Any]:
        """Return store statistics for the WebUI integrations panel."""
        async with self._lock:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._sync_stats)

    def _sync_stats(self) -> dict[str, Any]:
        turn_count    = self._conn.execute("SELECT COUNT(*) FROM session_turns").fetchone()[0]
        session_count = self._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        db_size_kb    = round(self._db_path.stat().st_size / 1024, 1) if self._db_path.exists() else 0
        return {
            "total_turns":    turn_count,
            "total_sessions": session_count,
            "db_size_kb":     db_size_kb,
            "db_path":        str(self._db_path),
        }

    async def write_nudge_log(
        self,
        session_id: str,
        turns_evaluated: int,
        memories_created: int,
    ) -> None:
        async with self._lock:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._conn.execute(
                    """INSERT INTO nudge_log
                       (nudge_id, session_id, ts, turns_evaluated, memories_created)
                       VALUES (?,?,?,?,?)""",
                    (str(uuid.uuid4()), session_id, time.time(),
                     turns_evaluated, memories_created),
                ),
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """
        Convert a natural-language query into a safe FTS5 MATCH expression.

        Strategy:
          1. Extract word tokens (Unicode-aware, min 2 chars)
          2. Remove common English stop words that add noise
          3. Wrap each token in double-quotes for FTS5 (handles any remaining
             special chars and treats each token as a literal phrase)
          4. Join with spaces — FTS5 treats space-separated terms as OR
          5. Cap at 8 tokens to keep queries fast

        FTS5 quoted syntax: "token" matches the literal word.
        This is safe against all FTS5 special characters including
        ?, *, (, ), :, -, ^, NOT, AND, OR.
        """
        import re
        # Extract word tokens — Unicode letters, digits, underscore
        tokens_raw = re.findall(r"\b\w{2,}\b", query, flags=re.UNICODE)

        _STOP = {
            "is", "are", "the", "for", "why", "how", "what", "when",
            "where", "who", "do", "did", "an", "a", "to", "of", "in",
            "on", "at", "be", "it", "its", "and", "or", "not", "was",
            "has", "have", "had", "can", "will", "would", "should",
            "could", "may", "might", "my", "your", "their", "our",
        }

        tokens = [t for t in tokens_raw if t.lower() not in _STOP]
        if not tokens:
            # Fall back to all 2+ char tokens without stop-word filtering
            tokens = [t for t in tokens_raw if len(t) >= 2]
        if not tokens:
            return ""

        # Wrap each token in double-quotes and join with OR for FTS5 safety
        # "token1" OR "token2" — matches any turn containing any of these words.
        # Double-quoting handles all FTS5 special chars; OR gives broad recall.
        quoted = ['"' + t.replace('"', '""') + '"' for t in tokens[:8]]
        return " OR ".join(quoted)

    @property
    def db_path(self) -> Path:
        return self._db_path