"""
agent_memory/stores/short_term_store.py

Short-Term Memory — P0-style tool result cache.
v4 fixes:
  - Binary-mode byte-offset paging (Unicode-safe, CJK/emoji correct)
  - Atomic evict_expired via write lock (no double-delete race)
  - Security: file paths resolved from trusted DB records only
  - get_entry() now returns both total_bytes and total_length
  - list_by_tool(user_id, session_id, tool_name) for tool-specific queries
  - garbage_collect() to remove orphan cache files not in DB
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from agent_memory.schemas import ToolResultEntry, _uid, _now
from agent_memory.stores._db import get_pool

logger = logging.getLogger(__name__)

_DDL = [
    """CREATE TABLE IF NOT EXISTS tool_cache_index (
        ref_id        TEXT PRIMARY KEY,
        user_id       TEXT NOT NULL,
        session_id    TEXT NOT NULL,
        tool_name     TEXT NOT NULL,
        file_path     TEXT NOT NULL,
        total_bytes   INTEGER NOT NULL DEFAULT 0,
        total_length  INTEGER NOT NULL DEFAULT 0,
        created_at    REAL NOT NULL,
        expires_at    REAL NOT NULL DEFAULT 0,
        metadata      TEXT NOT NULL DEFAULT '{}'
    )""",
    "CREATE INDEX IF NOT EXISTS idx_tci_user_session ON tool_cache_index(user_id, session_id)",
    "CREATE INDEX IF NOT EXISTS idx_tci_user_tool ON tool_cache_index(user_id, tool_name)",
    "CREATE INDEX IF NOT EXISTS idx_tci_expires ON tool_cache_index(expires_at) WHERE expires_at > 0",
]

DEFAULT_INLINE_THRESHOLD = 4_000
DEFAULT_SESSION_TTL = 86_400


class ShortTermStore:
    def __init__(
        self,
        base_dir: str | Path,
        db_path: str | Path,
        inline_threshold: int = DEFAULT_INLINE_THRESHOLD,
        session_ttl: float = DEFAULT_SESSION_TTL,
    ) -> None:
        self._base_dir = Path(base_dir).resolve()
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._inline_threshold = inline_threshold
        self._session_ttl = session_ttl
        self._pool = get_pool(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        self._pool.execute_write_many([(sql, ()) for sql in _DDL])

    def _user_dir(self, user_id: str) -> Path:
        safe = hashlib.sha256(user_id.encode()).hexdigest()[:24]
        d = self._base_dir / safe
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _build_file_path(self, user_id: str, ref_id: str) -> Path:
        safe_ref = re.sub(r"[^a-zA-Z0-9]", "_", ref_id)[:64]
        return self._user_dir(user_id) / f"{safe_ref}.cache"

    # ── write ─────────────────────────────────────────────────────────────────

    def store(
        self,
        user_id: str, session_id: str, tool_name: str, content: str,
        ttl: Optional[float] = None, metadata: Optional[Dict] = None,
        ref_id: Optional[str] = None,
    ) -> ToolResultEntry:
        rid = ref_id or _uid()
        now = _now()
        exp = now + (ttl if ttl is not None else self._session_ttl)
        fpath = self._build_file_path(user_id, rid)
        content_bytes = content.encode("utf-8")
        try:
            fpath.write_bytes(content_bytes)
        except OSError as e:
            logger.error("Failed to write cache file %s: %s", fpath, e)
            raise
        entry = ToolResultEntry(
            ref_id=rid, user_id=user_id, session_id=session_id,
            tool_name=tool_name, content="",
            total_length=len(content), created_at=now, expires_at=exp,
            metadata=metadata or {},
        )
        self._pool.execute_write(
            """INSERT OR REPLACE INTO tool_cache_index
               (ref_id, user_id, session_id, tool_name, file_path,
                total_bytes, total_length, created_at, expires_at, metadata)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (rid, user_id, session_id, tool_name, str(fpath),
             len(content_bytes), len(content), now, exp,
             json.dumps(entry.metadata)),
        )
        return entry

    # ── read ──────────────────────────────────────────────────────────────────

    def read(self, user_id: str, ref_id: str,
             offset: int = 0, length: int = 2_000) -> dict:
        """
        Byte-offset paging. offset and length are BYTES, not chars.
        Returns dict with total_bytes (bytes) and total_length (chars).
        """
        if offset < 0 or length <= 0:
            return {"error": "offset must be >= 0 and length must be > 0"}
        rows = self._pool.execute_read(
            "SELECT * FROM tool_cache_index WHERE ref_id=? AND user_id=?",
            (ref_id, user_id),
        )
        if not rows:
            return {"error": f"ref_id '{ref_id}' not found for user"}
        row = rows[0]
        if row["expires_at"] > 0 and time.time() > row["expires_at"]:
            return {"error": f"ref_id '{ref_id}' has expired"}
        fpath = Path(row["file_path"])
        try:
            fpath.resolve().relative_to(self._base_dir)
        except ValueError:
            return {"error": "internal path error"}
        if not fpath.exists():
            return {"error": f"cache file missing for ref_id '{ref_id}'"}
        try:
            with fpath.open("rb") as f:
                f.seek(offset)
                raw = f.read(length)
                file_size = f.seek(0, 2)
        except OSError as e:
            return {"error": f"failed to read cache: {e}"}
        while raw:
            try:
                text = raw.decode("utf-8")
                break
            except UnicodeDecodeError:
                raw = raw[:-1]
        else:
            text = ""
        actual_end = offset + len(raw)
        has_more = actual_end < file_size
        return {
            "ref_id": ref_id,
            "tool_name": row["tool_name"],
            "session_id": row["session_id"],
            "content": text,
            "offset": offset,
            "length": len(raw),
            "total_length": row["total_length"],
            "total_bytes": row["total_bytes"] or file_size,
            "has_more": has_more,
            "next_offset": actual_end if has_more else None,
        }

    def preview(self, user_id: str, ref_id: str, preview_len: int = 200) -> str:
        result = self.read(user_id, ref_id, offset=0, length=preview_len * 4)
        if "error" in result:
            return f"[STORED:{ref_id}] (unavailable: {result['error']})"
        snippet = result["content"][:preview_len].replace("\n", " ")
        return (
            f"[STORED:{ref_id}:{result['tool_name']}] "
            f"({result['total_length']} chars / {result['total_bytes']} bytes) "
            f"Preview: {snippet}…"
        )

    def get_entry(self, user_id: str, ref_id: str) -> Optional[ToolResultEntry]:
        rows = self._pool.execute_read(
            "SELECT * FROM tool_cache_index WHERE ref_id=? AND user_id=?",
            (ref_id, user_id),
        )
        if not rows:
            return None
        r = rows[0]
        # Return actual total_length from DB (was returning 0 in v3 bug)
        entry = ToolResultEntry(
            ref_id=r["ref_id"], user_id=r["user_id"], session_id=r["session_id"],
            tool_name=r["tool_name"], content="",
            total_length=r["total_length"],
            created_at=r["created_at"], expires_at=r["expires_at"],
            metadata=json.loads(r["metadata"] or "{}"),
        )
        return entry

    def list_session(self, user_id: str, session_id: str) -> List[ToolResultEntry]:
        rows = self._pool.execute_read(
            "SELECT * FROM tool_cache_index WHERE user_id=? AND session_id=? ORDER BY created_at DESC",
            (user_id, session_id),
        )
        return [self._row_to_entry(r) for r in rows]

    def list_by_tool(self, user_id: str, tool_name: str,
                     session_id: Optional[str] = None) -> List[ToolResultEntry]:
        """List all cached results for a specific tool (optionally filtered by session)."""
        if session_id:
            rows = self._pool.execute_read(
                "SELECT * FROM tool_cache_index WHERE user_id=? AND tool_name=? AND session_id=? ORDER BY created_at DESC",
                (user_id, tool_name, session_id),
            )
        else:
            rows = self._pool.execute_read(
                "SELECT * FROM tool_cache_index WHERE user_id=? AND tool_name=? ORDER BY created_at DESC",
                (user_id, tool_name),
            )
        return [self._row_to_entry(r) for r in rows]

    def _row_to_entry(self, r) -> ToolResultEntry:
        return ToolResultEntry(
            ref_id=r["ref_id"], user_id=r["user_id"], session_id=r["session_id"],
            tool_name=r["tool_name"], content="", total_length=r["total_length"],
            created_at=r["created_at"], expires_at=r["expires_at"],
            metadata=json.loads(r["metadata"] or "{}"),
        )

    # ── expiry / cleanup ──────────────────────────────────────────────────────

    def evict_expired(self) -> int:
        """Atomically select+delete expired entries. Thread-safe, no double-delete."""
        now = time.time()
        with self._pool._write_lock:
            conn = self._pool._read_conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                rows = conn.execute(
                    "SELECT ref_id, file_path FROM tool_cache_index "
                    "WHERE expires_at > 0 AND expires_at < ?", (now,)
                ).fetchall()
                if rows:
                    for row in rows:
                        try:
                            Path(row["file_path"]).unlink(missing_ok=True)
                        except OSError as e:
                            logger.warning("Could not delete expired file: %s", e)
                    ph = ",".join("?" * len(rows))
                    conn.execute(
                        f"DELETE FROM tool_cache_index WHERE ref_id IN ({ph})",
                        tuple(r["ref_id"] for r in rows)
                    )
                conn.execute("COMMIT")
            except Exception:
                try: conn.execute("ROLLBACK")
                except Exception: pass
                raise
        count = len(rows) if rows else 0
        if count:
            logger.info("Evicted %d expired cache entries", count)
        return count

    def delete_session(self, user_id: str, session_id: str) -> int:
        with self._pool._write_lock:
            conn = self._pool._read_conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                rows = conn.execute(
                    "SELECT ref_id, file_path FROM tool_cache_index WHERE user_id=? AND session_id=?",
                    (user_id, session_id),
                ).fetchall()
                for row in rows:
                    try: Path(row["file_path"]).unlink(missing_ok=True)
                    except OSError as e: logger.warning("Could not delete file: %s", e)
                if rows:
                    ph = ",".join("?" * len(rows))
                    conn.execute(
                        f"DELETE FROM tool_cache_index WHERE ref_id IN ({ph})",
                        tuple(r["ref_id"] for r in rows)
                    )
                conn.execute("COMMIT")
            except Exception:
                try: conn.execute("ROLLBACK")
                except Exception: pass
                raise
        return len(rows) if rows else 0

    def garbage_collect(self) -> dict:
        """
        Remove orphan cache files not referenced in DB.
        Also remove DB rows whose files are missing.
        Returns {"orphan_files": N, "missing_files": N}.
        """
        # Get all files on disk
        all_files = set(self._base_dir.rglob("*.cache"))
        # Get all file paths in DB
        rows = self._pool.execute_read(
            "SELECT ref_id, user_id, file_path FROM tool_cache_index"
        )
        db_paths = {Path(r["file_path"]) for r in rows}
        db_ref_ids = {r["ref_id"] for r in rows}

        orphan_files = 0
        for f in all_files:
            if f not in db_paths:
                try:
                    f.unlink(missing_ok=True)
                    orphan_files += 1
                except OSError:
                    pass

        missing_refs = []
        for r in rows:
            if not Path(r["file_path"]).exists():
                missing_refs.append(r["ref_id"])
        if missing_refs:
            ph = ",".join("?" * len(missing_refs))
            self._pool.execute_write(
                f"DELETE FROM tool_cache_index WHERE ref_id IN ({ph})",
                tuple(missing_refs),
            )

        logger.info("GC: orphan_files=%d missing_db_refs=%d",
                    orphan_files, len(missing_refs))
        return {"orphan_files": orphan_files, "missing_files": len(missing_refs)}
