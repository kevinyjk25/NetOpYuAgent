"""
agent_memory/stores/_db.py

Shared SQLite connection pool:
- WAL journal mode (concurrent readers + serialized writer)
- Thread-local connections (one per thread, safe without extra locking)
- Write serialization via threading.Lock on BEGIN IMMEDIATE
- evict_expired is SELECT+DELETE atomic via the same write lock
- Explicit WAL checkpoint on close to prevent unbounded WAL growth
- close() support for clean shutdown and connection release

PRAGMA summary:
  journal_mode=WAL       → concurrent reads never block on writer
  synchronous=NORMAL     → durable enough, faster than FULL
  cache_size=-32000      → 32 MB page cache per connection
  temp_store=MEMORY      → temp tables in memory
  mmap_size=268435456    → 256 MB memory-mapped I/O
  wal_autocheckpoint=500 → checkpoint every 500 pages (~2 MB) to cap WAL size
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_pools: dict[str, "_Pool"] = {}
_pools_lock = threading.Lock()


def get_pool(db_path: str | Path) -> "_Pool":
    key = str(Path(db_path).resolve())
    with _pools_lock:
        if key not in _pools:
            _pools[key] = _Pool(key)
        return _pools[key]


def close_pool(db_path: str | Path) -> None:
    """Close and remove the pool for a given DB path."""
    key = str(Path(db_path).resolve())
    with _pools_lock:
        pool = _pools.pop(key, None)
    if pool:
        pool._shutdown()


class _Pool:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._write_lock = threading.Lock()
        self._local = threading.local()
        self._all_conns: list[sqlite3.Connection] = []
        self._all_conns_lock = threading.Lock()
        self._closed = False
        self._bootstrap()

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self._db_path,
            timeout=30,
            check_same_thread=False,
            isolation_level=None,   # autocommit; we manage transactions manually
        )
        conn.row_factory = sqlite3.Row
        self._apply_pragmas(conn)
        with self._all_conns_lock:
            self._all_conns.append(conn)
        return conn

    @staticmethod
    def _apply_pragmas(conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-32000")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA mmap_size=268435456")
        conn.execute("PRAGMA wal_autocheckpoint=500")   # ~2 MB cap

    def _bootstrap(self) -> None:
        conn = self._open()
        conn.close()

    @property
    def _read_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = self._open()
        return self._local.conn

    # ── read ──────────────────────────────────────────────────────────────────

    def execute_read(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        if self._closed:
            raise RuntimeError("Pool is closed")
        # Use BEGIN DEFERRED for snapshot-consistent reads in WAL mode
        conn = self._read_conn
        conn.execute("BEGIN DEFERRED")
        try:
            rows = conn.execute(sql, params).fetchall()
            conn.execute("COMMIT")
            return rows
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise

    # ── write ─────────────────────────────────────────────────────────────────

    def execute_write(self, sql: str, params: tuple = ()) -> int:
        if self._closed:
            raise RuntimeError("Pool is closed")
        with self._write_lock:
            conn = self._read_conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                cur = conn.execute(sql, params)
                conn.execute("COMMIT")
                return cur.rowcount
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise

    def execute_write_many(self, ops: list[tuple[str, tuple]]) -> None:
        """Batch multiple statements in one transaction."""
        if not ops:
            return
        if self._closed:
            raise RuntimeError("Pool is closed")
        with self._write_lock:
            conn = self._read_conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                for sql, params in ops:
                    conn.execute(sql, params)
                conn.execute("COMMIT")
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise

    def execute_read_write(self, read_sql: str, read_params: tuple,
                           write_sql: str, write_params_fn) -> list[sqlite3.Row]:
        """
        Atomic read-then-write under the write lock.
        write_params_fn(rows) → tuple  (called with read results to build write params).
        Used by evict_expired to prevent double-delete races.
        """
        if self._closed:
            raise RuntimeError("Pool is closed")
        with self._write_lock:
            conn = self._read_conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                rows = conn.execute(read_sql, read_params).fetchall()
                if rows:
                    write_params = write_params_fn(rows)
                    if write_params is not None:
                        conn.execute(write_sql, write_params)
                conn.execute("COMMIT")
                return rows
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def close_thread_conn(self) -> None:
        """Release the current thread's connection. Call when a thread exits."""
        if hasattr(self._local, "conn") and self._local.conn:
            try:
                self._local.conn.close()
            except Exception:
                pass
            self._local.conn = None

    def checkpoint(self) -> None:
        """Manually trigger a WAL checkpoint to reclaim disk space."""
        try:
            conn = self._read_conn
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            logger.debug("WAL checkpoint completed for %s", self._db_path)
        except Exception as e:
            logger.warning("WAL checkpoint failed: %s", e)

    def _shutdown(self) -> None:
        """Close all tracked connections and run final checkpoint."""
        self._closed = True
        self.checkpoint()
        with self._all_conns_lock:
            for conn in self._all_conns:
                try:
                    conn.close()
                except Exception:
                    pass
            self._all_conns.clear()
        logger.debug("Pool shut down: %s", self._db_path)
