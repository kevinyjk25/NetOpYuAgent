"""Durable storage for large tool results returned through harness adapters."""

from __future__ import annotations

import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any


_STORED_REFERENCE = re.compile(r"\[STORED:[^:\]]+:([^\]]+)\]")


def normalize_result_reference(reference: str) -> str:
    """Accept a bare id or a complete ``[STORED:tool:id]`` reference."""
    match = _STORED_REFERENCE.search(reference)
    if match:
        return match.group(1).strip()
    normalized = reference.strip("[]")
    return normalized.rsplit(":", 1)[-1].strip()


class _ResultView:
    """Small mapping-compatible view retained for the paging tool factories."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def __len__(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) FROM results").fetchone()
        return int(row[0]) if row else 0

    def get(self, reference: str, default: str | None = "") -> str | None:
        row = self._connection.execute(
            "SELECT content FROM results WHERE ref_id = ?",
            (normalize_result_reference(reference),),
        ).fetchone()
        return str(row[0]) if row else default


class ToolResultStore:
    """Store oversized tool output in SQLite and return a bounded reference."""

    MAX_INLINE_CHARS = 4_000
    TTL_SECONDS = 86_400

    def __init__(self, db_path: str | None = None) -> None:
        database = Path(db_path or "data/tool_results.sqlite")
        if str(database) != ":memory:":
            database.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(database), check_same_thread=False)
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS results "
            "(ref_id TEXT PRIMARY KEY, content TEXT NOT NULL, created_at REAL NOT NULL)"
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_results_created_at "
            "ON results(created_at)"
        )
        self._connection.commit()
        self._prune()

        # The paging tools use ``get`` and ``len`` on this compatibility view.
        self._store: Any = _ResultView(self._connection)

    def _prune(self) -> None:
        cutoff = time.time() - self.TTL_SECONDS
        self._connection.execute(
            "DELETE FROM results WHERE created_at < ?",
            (cutoff,),
        )
        self._connection.commit()

    def store(self, tool_name: str, raw_output: str) -> str:
        if len(raw_output) <= self.MAX_INLINE_CHARS:
            return raw_output

        reference = uuid.uuid4().hex[:8]
        self._connection.execute(
            "INSERT OR REPLACE INTO results(ref_id, content, created_at) "
            "VALUES (?, ?, ?)",
            (reference, raw_output, time.time()),
        )
        self._connection.commit()
        preview = raw_output[:80].replace("\n", " ")
        return f"[STORED:{tool_name}:{reference}] Preview: {preview}"

    def read(
        self,
        reference: str,
        offset: int = 0,
        length: int = 2_000,
    ) -> str | None:
        row = self._connection.execute(
            "SELECT content FROM results WHERE ref_id = ?",
            (normalize_result_reference(reference),),
        ).fetchone()
        if row is None:
            return None
        return str(row[0])[offset : offset + length]

    def clear_session(self, session_id: str) -> None:
        """Compatibility no-op: stored results are not session-keyed."""

    @property
    def stored_count(self) -> int:
        return len(self._store)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "ToolResultStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
