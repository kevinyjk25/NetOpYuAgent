"""
hitl_core.store — Pluggable checkpoint store (replaces LangGraph MemorySaver).

Three implementations ship with the module:
  • InMemoryCheckpointStore  — dev / tests; equivalent to MemorySaver
  • RedisCheckpointStore     — production; pickle-free msgpack serialisation
  • SqliteCheckpointStore    — single-host with restart survival

All three implement the same BaseCheckpointStore interface; the router
treats them interchangeably. New backends (Postgres, DynamoDB, etc.)
plug in by subclassing BaseCheckpointStore.

Serialisation contract:
  CheckpointEntry ↔ pure JSON-serialisable dict via Pydantic model_dump().
  We never pickle Python functions — resume_handle.resumer_name is a
  string the host resolves at runtime through HitlRouter.register_resumer.
  This is what makes cross-process / cross-host deployment work.
"""

from __future__ import annotations

import abc
import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

from .schema import (
    BatchState,
    CheckpointEntry,
    HitlBatch,
    HitlDecision,
    HitlPayload,
    InterruptState,
    ResumeHandle,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _entry_to_bytes(entry: CheckpointEntry) -> bytes:
    """Serialise to msgpack if available, else JSON. Both are
    cross-process portable; msgpack is denser/faster."""
    payload = entry.model_dump(mode="json")
    try:
        import msgpack
        return msgpack.packb(payload, use_bin_type=True)
    except ImportError:
        return json.dumps(payload).encode("utf-8")


def _entry_from_bytes(data: bytes) -> CheckpointEntry:
    """Inverse of _entry_to_bytes — auto-detects format by trying msgpack
    first, JSON as fallback."""
    try:
        import msgpack
        try:
            unpacked = msgpack.unpackb(data, raw=False)
        except Exception:
            unpacked = json.loads(data.decode("utf-8"))
    except ImportError:
        unpacked = json.loads(data.decode("utf-8"))
    return CheckpointEntry.model_validate(unpacked)


# ---------------------------------------------------------------------------
# Lua CAS script for atomic mark_resolved in Redis (BUG-02 fix).
# Atomically: load entry → check state == PENDING → mutate → save.
# Returns the serialised entry bytes on success, or nil if the entry
# didn't exist or was already resolved (idempotent guard).
# ---------------------------------------------------------------------------
_LUA_MARK_RESOLVED = """
local raw = redis.call('GET', KEYS[1])
if not raw then return nil end
local ok, entry = pcall(cmsgpack.unpack, raw)
if not ok then
    -- Fallback: try JSON (when msgpack unavailable at write time)
    ok, entry = pcall(cjson.decode, raw)
    if not ok then return nil end
end
if entry['state'] ~= 'pending' then return nil end
entry['state'] = 'resolved'
entry['decision'] = cmsgpack.unpack(ARGV[1])
entry['decided_at'] = tonumber(ARGV[2])
local new_raw = cmsgpack.pack(entry)
redis.call('SET', KEYS[1], new_raw, 'KEEPTTL')
redis.call('ZREM', KEYS[2], KEYS[3])
return new_raw
"""

# Pure-JSON fallback Lua (when msgpack not available on Redis side)
_LUA_MARK_RESOLVED_JSON = """
local raw = redis.call('GET', KEYS[1])
if not raw then return nil end
local entry = cjson.decode(raw)
if entry['state'] ~= 'pending' then return nil end
entry['state'] = 'resolved'
entry['decision'] = cjson.decode(ARGV[1])
entry['decided_at'] = tonumber(ARGV[2])
local new_raw = cjson.encode(entry)
redis.call('SET', KEYS[1], new_raw, 'KEEPTTL')
redis.call('ZREM', KEYS[2], KEYS[3])
return new_raw
"""


# ---------------------------------------------------------------------------
# SQLite connection pool helper (DESIGN-07 fix)
# ---------------------------------------------------------------------------
# Reuses per-thread connections for SqliteCheckpointStore instead of
# opening/closing a new connection on every read/write.

import threading as _threading

class _SqlitePool:
    """Minimal thread-local SQLite connection pool."""
    def __init__(self, db_path: str):
        self._db_path  = db_path
        self._local    = _threading.local()

    def get_conn(self):
        """Return the thread-local connection, creating it if needed."""
        import sqlite3
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    def close_all(self) -> None:
        """Close the current thread's connection (call at thread exit)."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None


_SQLITE_POOLS: dict[str, "_SqlitePool"] = {}
_POOL_LOCK = _threading.Lock()


def _get_sqlite_pool(db_path: str) -> "_SqlitePool":
    """Return (creating if needed) the shared pool for db_path."""
    with _POOL_LOCK:
        if db_path not in _SQLITE_POOLS:
            _SQLITE_POOLS[db_path] = _SqlitePool(db_path)
        return _SQLITE_POOLS[db_path]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseCheckpointStore(abc.ABC):
    """Interface every checkpoint backend implements. All methods async to
    keep the call sites uniform (some backends — InMemory — could be sync,
    but uniform async lets us swap them without changing callers)."""

    @abc.abstractmethod
    async def save(self, entry: CheckpointEntry) -> None:
        """Persist a checkpoint. Overwrites if interrupt_id exists."""

    @abc.abstractmethod
    async def load(self, interrupt_id: str) -> Optional[CheckpointEntry]:
        """Retrieve a checkpoint, or None if absent / expired."""

    @abc.abstractmethod
    async def delete(self, interrupt_id: str) -> bool:
        """Remove a checkpoint. Returns True if it existed."""

    @abc.abstractmethod
    async def list_pending(
        self, *, limit: int = 100, thread_id: Optional[str] = None,
    ) -> list[CheckpointEntry]:
        """Return up to `limit` PENDING checkpoints, optionally filtered
        by thread_id. Used by the UI to render the "pending HITL" list."""

    @abc.abstractmethod
    async def mark_resolved(
        self, interrupt_id: str, decision: HitlDecision,
    ) -> Optional[CheckpointEntry]:
        """Atomic transition PENDING → RESOLVED with the operator's
        decision attached. Returns the updated entry, or None if the
        interrupt didn't exist or wasn't pending."""

    # ── Batch APIs (default impls cover non-redis backends) ─────────────
    # Backends that want to optimise these (e.g. Redis with ZSET indices)
    # override; default implementations build batch state from the per-
    # interrupt entries plus a small per-batch envelope persisted via the
    # generic save_batch / load_batch hooks.

    @abc.abstractmethod
    async def save_batch(self, batch: HitlBatch) -> None:
        """Persist a batch envelope. Children are saved separately via save()."""

    @abc.abstractmethod
    async def load_batch(self, batch_id: str) -> Optional[HitlBatch]:
        """Retrieve a batch envelope by id, or None if absent."""

    @abc.abstractmethod
    async def list_pending_batches(
        self, *, limit: int = 50, thread_id: Optional[str] = None,
    ) -> list[HitlBatch]:
        """Pending batches, newest first, optionally filtered by thread."""

    @abc.abstractmethod
    async def delete_batch(self, batch_id: str) -> bool:
        """Remove a batch envelope. Children are NOT cascaded — the caller
        must delete them separately. Returns True if the envelope existed."""

    # Convenience helpers — concrete subclasses can override for efficiency

    async def expire_overdue(self) -> int:
        """Sweep entries past their SLA; transition PENDING → EXPIRED.
        Returns count expired. Default impl walks list_pending."""
        now = datetime.now(timezone.utc)
        expired = 0
        for entry in await self.list_pending(limit=10_000):
            elapsed = (now - entry.registered_at).total_seconds()
            if elapsed > entry.payload.sla_seconds:
                entry.state = InterruptState.EXPIRED
                await self.save(entry)
                expired += 1
        if expired:
            logger.info("expire_overdue: marked %d interrupts as EXPIRED", expired)
        return expired

    async def close(self) -> None:
        """Release resources. Default no-op — backends with connections override."""


# ---------------------------------------------------------------------------
# In-memory implementation
# ---------------------------------------------------------------------------

class InMemoryCheckpointStore(BaseCheckpointStore):
    """Pure dict + asyncio.Lock. Loses state on restart — fine for dev,
    fine for production single-process where HITLs are short-lived (the
    LangGraph MemorySaver behaves the same way)."""

    def __init__(self) -> None:
        self._entries: dict[str, CheckpointEntry] = {}
        self._batches: dict[str, HitlBatch] = {}
        self._lock = asyncio.Lock()

    async def save(self, entry: CheckpointEntry) -> None:
        async with self._lock:
            self._entries[entry.interrupt_id] = entry

    async def load(self, interrupt_id: str) -> Optional[CheckpointEntry]:
        async with self._lock:
            return self._entries.get(interrupt_id)

    async def delete(self, interrupt_id: str) -> bool:
        async with self._lock:
            return self._entries.pop(interrupt_id, None) is not None

    async def list_pending(
        self, *, limit: int = 100, thread_id: Optional[str] = None,
    ) -> list[CheckpointEntry]:
        async with self._lock:
            results = []
            for entry in self._entries.values():
                if entry.state != InterruptState.PENDING:
                    continue
                if thread_id and entry.payload.thread_id != thread_id:
                    continue
                results.append(entry)
            # Newest first — operators usually want the freshest
            results.sort(key=lambda e: e.registered_at, reverse=True)
            return results[:limit]

    async def mark_resolved(
        self, interrupt_id: str, decision: HitlDecision,
    ) -> Optional[CheckpointEntry]:
        async with self._lock:
            entry = self._entries.get(interrupt_id)
            if entry is None or entry.state != InterruptState.PENDING:
                return None
            entry.state = InterruptState.RESOLVED
            entry.decision = decision
            entry.decided_at = datetime.now(timezone.utc)
            return entry

    async def save_batch(self, batch: HitlBatch) -> None:
        async with self._lock:
            self._batches[batch.batch_id] = batch

    async def load_batch(self, batch_id: str) -> Optional[HitlBatch]:
        async with self._lock:
            return self._batches.get(batch_id)

    async def list_pending_batches(
        self, *, limit: int = 50, thread_id: Optional[str] = None,
    ) -> list[HitlBatch]:
        async with self._lock:
            results = []
            for b in self._batches.values():
                if b.state != BatchState.PENDING:
                    continue
                if thread_id and b.thread_id != thread_id:
                    continue
                results.append(b)
            results.sort(key=lambda b: b.created_at, reverse=True)
            return results[:limit]

    async def delete_batch(self, batch_id: str) -> bool:
        async with self._lock:
            return self._batches.pop(batch_id, None) is not None


# ---------------------------------------------------------------------------
# Redis implementation
# ---------------------------------------------------------------------------

class RedisCheckpointStore(BaseCheckpointStore):
    """Production-grade backend. Uses redis-py>=5 async API (built-in,
    no aioredis dependency).

    Key layout:
      hitl:cp:{interrupt_id}            → msgpack(CheckpointEntry)
      hitl:idx:pending                  → ZSET, score=registered_at_ts, value=interrupt_id
      hitl:idx:thread:{thread_id}       → SET of interrupt_ids

    The pending index is a sorted set so list_pending can return in
    chronological order with O(log N) reads. Per-thread index lets the
    UI filter "my session's" interrupts cheaply.

    Atomicity: save / mark_resolved use Redis pipelines (MULTI/EXEC) so
    a partial write never leaves the index out of sync with the entry.
    """

    PREFIX_ENTRY      = "hitl:cp:"
    KEY_PENDING_IX    = "hitl:idx:pending"
    PREFIX_THREAD     = "hitl:idx:thread:"
    PREFIX_BATCH      = "hitl:batch:"
    KEY_BATCH_PENDING = "hitl:idx:batch:pending"
    PREFIX_BATCH_THREAD = "hitl:idx:batch:thread:"

    def __init__(self, redis_url: str = "redis://localhost:6379/0") -> None:
        try:
            import redis.asyncio as redis_asyncio
        except ImportError as exc:
            raise RuntimeError(
                "RedisCheckpointStore requires redis-py: pip install redis>=5.0"
            ) from exc
        self._redis = redis_asyncio.from_url(redis_url, decode_responses=False)
        self._redis_url = redis_url
        logger.info("RedisCheckpointStore connected: %s", redis_url)

    @property
    def redis(self):
        """Expose underlying client for advanced uses (rarely needed)."""
        return self._redis

    async def save(self, entry: CheckpointEntry) -> None:
        data = _entry_to_bytes(entry)
        ttl = max(entry.payload.sla_seconds * 2, 600)  # generous TTL
        ts = entry.registered_at.timestamp()
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.set(self.PREFIX_ENTRY + entry.interrupt_id, data, ex=ttl)
            if entry.state == InterruptState.PENDING:
                pipe.zadd(self.KEY_PENDING_IX, {entry.interrupt_id: ts})
            else:
                # Resolved/expired/cancelled: drop from pending index
                pipe.zrem(self.KEY_PENDING_IX, entry.interrupt_id)
            if entry.payload.thread_id:
                pipe.sadd(
                    self.PREFIX_THREAD + entry.payload.thread_id,
                    entry.interrupt_id,
                )
                pipe.expire(
                    self.PREFIX_THREAD + entry.payload.thread_id, ttl,
                )
            await pipe.execute()

    async def load(self, interrupt_id: str) -> Optional[CheckpointEntry]:
        data = await self._redis.get(self.PREFIX_ENTRY + interrupt_id)
        if data is None:
            return None
        try:
            return _entry_from_bytes(data)
        except Exception as exc:
            logger.warning("RedisCheckpointStore.load: corrupt entry %s: %s",
                           interrupt_id, exc)
            return None

    async def delete(self, interrupt_id: str) -> bool:
        # Best-effort thread-index cleanup: load entry first to find the
        # thread_id; we don't fail if it's already gone.
        entry = await self.load(interrupt_id)
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.delete(self.PREFIX_ENTRY + interrupt_id)
            pipe.zrem(self.KEY_PENDING_IX, interrupt_id)
            if entry and entry.payload.thread_id:
                pipe.srem(
                    self.PREFIX_THREAD + entry.payload.thread_id,
                    interrupt_id,
                )
            results = await pipe.execute()
        return bool(results[0]) if results else False

    async def list_pending(
        self, *, limit: int = 100, thread_id: Optional[str] = None,
    ) -> list[CheckpointEntry]:
        if thread_id:
            ids = await self._redis.smembers(self.PREFIX_THREAD + thread_id)
            ids = [i.decode() if isinstance(i, bytes) else i for i in ids]
        else:
            # Newest first via ZREVRANGE (high scores first)
            ids = await self._redis.zrevrange(self.KEY_PENDING_IX, 0, limit - 1)
            ids = [i.decode() if isinstance(i, bytes) else i for i in ids]
        if not ids:
            return []
        # Bulk-load entries
        keys = [self.PREFIX_ENTRY + i for i in ids]
        data_list = await self._redis.mget(keys)
        entries = []
        for raw in data_list:
            if raw is None:
                continue
            try:
                entry = _entry_from_bytes(raw)
                if entry.state == InterruptState.PENDING:
                    entries.append(entry)
            except Exception as exc:
                logger.warning("list_pending: corrupt entry skipped: %s", exc)
        # Already newest-first (zrevrange) when no thread filter; sort
        # explicitly when filtering by thread (set has no order).
        entries.sort(key=lambda e: e.registered_at, reverse=True)
        return entries[:limit]

    async def mark_resolved(
        self, interrupt_id: str, decision: HitlDecision,
    ) -> Optional[CheckpointEntry]:
        """Atomic CAS via Lua script — prevents double-approval race (BUG-02).

        The Lua script runs atomically on the Redis server:
          GET entry → check state == PENDING → mutate → SET back
        No window between load and save; concurrent calls on any replica
        will see either the original PENDING state or the already-RESOLVED
        state, never both succeeding.

        Falls back to JSON Lua script if msgpack is unavailable on Redis,
        and finally to a Python-level load-check-save with a logged warning
        if the Lua script itself fails (e.g. old Redis without EVALSHA).
        """
        now_ts = datetime.now(timezone.utc).timestamp()
        now_iso = datetime.now(timezone.utc).isoformat()

        # Serialise decision for Lua consumption
        try:
            import msgpack as _mp
            decision_bytes = _mp.packb(decision.model_dump(mode="json"), use_bin_type=True)
            lua_script = _LUA_MARK_RESOLVED
        except (ImportError, Exception):
            import json as _json
            decision_bytes = _json.dumps(decision.model_dump(mode="json")).encode()
            lua_script = _LUA_MARK_RESOLVED_JSON

        entry_key   = self.PREFIX_ENTRY + interrupt_id
        pending_key = self.KEY_PENDING_IX

        try:
            result = await self._redis.eval(
                lua_script,
                3,                              # numkeys
                entry_key, pending_key, interrupt_id,
                decision_bytes, str(now_ts),
            )
        except Exception as lua_err:
            # Lua eval failed (old Redis, NOSCRIPT, etc.) — fall back to
            # Python-level CAS with a WARNING so ops knows atomicity is degraded.
            logger.warning(
                "mark_resolved: Lua eval failed (%s) — falling back to "
                "non-atomic Python CAS. Concurrent double-approval possible.",
                lua_err,
            )
            entry = await self.load(interrupt_id)
            if entry is None or entry.state != InterruptState.PENDING:
                return None
            entry.state    = InterruptState.RESOLVED
            entry.decision = decision
            entry.decided_at = datetime.now(timezone.utc)
            await self.save(entry)
            return entry

        if result is None:
            # Entry didn't exist or was already resolved — idempotent
            return None

        # Deserialise the mutated entry returned by Lua
        try:
            return _entry_from_bytes(bytes(result))
        except Exception as exc:
            logger.warning("mark_resolved: failed to deserialise Lua result: %s", exc)
            # Best-effort: load from Redis after Lua already resolved it
            return await self.load(interrupt_id)

    async def save_batch(self, batch: HitlBatch) -> None:
        data = self._serialise_batch(batch)
        ttl = max(batch.sla_seconds * 2, 600)
        ts = batch.created_at.timestamp()
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.set(self.PREFIX_BATCH + batch.batch_id, data, ex=ttl)
            if batch.state == BatchState.PENDING:
                pipe.zadd(self.KEY_BATCH_PENDING, {batch.batch_id: ts})
            else:
                pipe.zrem(self.KEY_BATCH_PENDING, batch.batch_id)
            if batch.thread_id:
                pipe.sadd(self.PREFIX_BATCH_THREAD + batch.thread_id, batch.batch_id)
                pipe.expire(self.PREFIX_BATCH_THREAD + batch.thread_id, ttl)
            await pipe.execute()

    async def load_batch(self, batch_id: str) -> Optional[HitlBatch]:
        data = await self._redis.get(self.PREFIX_BATCH + batch_id)
        if data is None:
            return None
        try:
            return self._deserialise_batch(data)
        except Exception as exc:
            logger.warning("load_batch: corrupt batch %s: %s", batch_id, exc)
            return None

    async def list_pending_batches(
        self, *, limit: int = 50, thread_id: Optional[str] = None,
    ) -> list[HitlBatch]:
        if thread_id:
            ids = await self._redis.smembers(self.PREFIX_BATCH_THREAD + thread_id)
            ids = [i.decode() if isinstance(i, bytes) else i for i in ids]
        else:
            ids = await self._redis.zrevrange(self.KEY_BATCH_PENDING, 0, limit - 1)
            ids = [i.decode() if isinstance(i, bytes) else i for i in ids]
        if not ids:
            return []
        keys = [self.PREFIX_BATCH + i for i in ids]
        data_list = await self._redis.mget(keys)
        out = []
        for raw in data_list:
            if raw is None:
                continue
            try:
                b = self._deserialise_batch(raw)
                if b.state == BatchState.PENDING:
                    out.append(b)
            except Exception as exc:
                logger.warning("list_pending_batches: skip corrupt: %s", exc)
        out.sort(key=lambda b: b.created_at, reverse=True)
        return out[:limit]

    async def delete_batch(self, batch_id: str) -> bool:
        batch = await self.load_batch(batch_id)
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.delete(self.PREFIX_BATCH + batch_id)
            pipe.zrem(self.KEY_BATCH_PENDING, batch_id)
            if batch and batch.thread_id:
                pipe.srem(self.PREFIX_BATCH_THREAD + batch.thread_id, batch_id)
            results = await pipe.execute()
        return bool(results[0]) if results else False

    @staticmethod
    def _serialise_batch(batch: HitlBatch) -> bytes:
        payload = batch.model_dump(mode="json")
        try:
            import msgpack
            return msgpack.packb(payload, use_bin_type=True)
        except ImportError:
            return json.dumps(payload).encode("utf-8")

    @staticmethod
    def _deserialise_batch(data: bytes) -> HitlBatch:
        try:
            import msgpack
            try:
                unpacked = msgpack.unpackb(data, raw=False)
            except Exception:
                unpacked = json.loads(data.decode("utf-8"))
        except ImportError:
            unpacked = json.loads(data.decode("utf-8"))
        return HitlBatch.model_validate(unpacked)

    async def close(self) -> None:
        try:
            await self._redis.aclose()
        except Exception as exc:
            logger.debug("RedisCheckpointStore close: %s", exc)


# ---------------------------------------------------------------------------
# SQLite implementation
# ---------------------------------------------------------------------------

class SqliteCheckpointStore(BaseCheckpointStore):
    """File-backed store for single-host deployments that need restart
    survival without standing up a Redis. Uses aiosqlite if available,
    falls back to threadpool-wrapped sqlite3 otherwise.

    Schema:
      checkpoints (
        interrupt_id   TEXT PRIMARY KEY,
        thread_id      TEXT NOT NULL DEFAULT '',
        state          TEXT NOT NULL,
        registered_at  REAL NOT NULL,
        decided_at     REAL,
        data           BLOB NOT NULL    -- msgpack(CheckpointEntry)
      )
      INDEX idx_state ON state, registered_at DESC
      INDEX idx_thread ON thread_id, state
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._init_lock = asyncio.Lock()
        self._initialised = False
        # DESIGN-07 fix: reuse thread-local connections via _SqlitePool
        self._pool: _SqlitePool = _get_sqlite_pool(db_path)
        # BUG-02 fix: serialise concurrent mark_resolved calls.
        # SQLite's per-row locking is at the storage level; we also need
        # application-level serialisation to prevent two coroutines both
        # reading state=PENDING before either writes state=RESOLVED.
        self._resolve_lock = asyncio.Lock()

    async def _ensure_init(self) -> None:
        if self._initialised:
            return
        async with self._init_lock:
            if self._initialised:
                return
            await asyncio.to_thread(self._init_sync)
            self._initialised = True

    def _init_sync(self) -> None:
        # Use pool connection for schema init too
        conn = self._pool.get_conn()
        conn.executescript("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    interrupt_id   TEXT PRIMARY KEY,
                    thread_id      TEXT NOT NULL DEFAULT '',
                    state          TEXT NOT NULL,
                    registered_at  REAL NOT NULL,
                    decided_at     REAL,
                    data           BLOB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_cp_state
                    ON checkpoints(state, registered_at DESC);
                CREATE INDEX IF NOT EXISTS idx_cp_thread
                    ON checkpoints(thread_id, state);

                CREATE TABLE IF NOT EXISTS batches (
                    batch_id       TEXT PRIMARY KEY,
                    thread_id      TEXT NOT NULL DEFAULT '',
                    state          TEXT NOT NULL,
                    created_at     REAL NOT NULL,
                    data           BLOB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_b_state
                    ON batches(state, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_b_thread
                    ON batches(thread_id, state);
            """)
        conn.commit()

    async def save(self, entry: CheckpointEntry) -> None:
        await self._ensure_init()
        await asyncio.to_thread(self._save_sync, entry)

    def _save_sync(self, entry: CheckpointEntry) -> None:
        import sqlite3
        data = _entry_to_bytes(entry)
        decided_at = (
            entry.decided_at.timestamp() if entry.decided_at else None
        )
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                INSERT INTO checkpoints
                    (interrupt_id, thread_id, state, registered_at, decided_at, data)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(interrupt_id) DO UPDATE SET
                    state         = excluded.state,
                    decided_at    = excluded.decided_at,
                    data          = excluded.data
            """, (
                entry.interrupt_id,
                entry.payload.thread_id,
                entry.state.value,
                entry.registered_at.timestamp(),
                decided_at,
                data,
            ))
            conn.commit()

    async def load(self, interrupt_id: str) -> Optional[CheckpointEntry]:
        await self._ensure_init()
        return await asyncio.to_thread(self._load_sync, interrupt_id)

    def _load_sync(self, interrupt_id: str) -> Optional[CheckpointEntry]:
        conn = self._pool.get_conn()
        conn.isolation_level = None  # autocommit off; explicit commit below
        if True:  # pool connection block (DESIGN-07)
            row = conn.execute(
                "SELECT data FROM checkpoints WHERE interrupt_id = ?",
                (interrupt_id,),
            ).fetchone()
        if row is None:
            return None
        try:
            return _entry_from_bytes(row[0])
        except Exception as exc:
            logger.warning("SqliteCheckpointStore.load: corrupt entry %s: %s",
                           interrupt_id, exc)
            return None

    async def delete(self, interrupt_id: str) -> bool:
        await self._ensure_init()
        return await asyncio.to_thread(self._delete_sync, interrupt_id)

    def _delete_sync(self, interrupt_id: str) -> bool:
        conn = self._pool.get_conn()
        conn.isolation_level = None  # autocommit off; explicit commit below
        if True:  # pool connection block (DESIGN-07)
            cur = conn.execute(
                "DELETE FROM checkpoints WHERE interrupt_id = ?",
                (interrupt_id,),
            )
            conn.commit()
            return cur.rowcount > 0

    async def list_pending(
        self, *, limit: int = 100, thread_id: Optional[str] = None,
    ) -> list[CheckpointEntry]:
        await self._ensure_init()
        return await asyncio.to_thread(self._list_pending_sync, limit, thread_id)

    def _list_pending_sync(
        self, limit: int, thread_id: Optional[str],
    ) -> list[CheckpointEntry]:
        conn = self._pool.get_conn()
        conn.isolation_level = None  # autocommit off; explicit commit below
        if True:  # pool connection block (DESIGN-07)
            if thread_id:
                rows = conn.execute("""
                    SELECT data FROM checkpoints
                    WHERE state = ? AND thread_id = ?
                    ORDER BY registered_at DESC LIMIT ?
                """, (InterruptState.PENDING.value, thread_id, limit)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT data FROM checkpoints
                    WHERE state = ?
                    ORDER BY registered_at DESC LIMIT ?
                """, (InterruptState.PENDING.value, limit)).fetchall()
        out = []
        for (raw,) in rows:
            try:
                out.append(_entry_from_bytes(raw))
            except Exception as exc:
                logger.warning("list_pending: skip corrupt: %s", exc)
        return out

    async def mark_resolved(
        self, interrupt_id: str, decision: HitlDecision,
    ) -> Optional[CheckpointEntry]:
        """BUG-02 fix: lock the full load-check-mutate-save sequence so
        concurrent coroutines cannot both observe state=PENDING and both
        proceed to write state=RESOLVED (double-approval).

        The asyncio.Lock() is sufficient for single-host (one process) SQLite
        deployments. For multi-host, use RedisCheckpointStore with Lua CAS.
        """
        async with self._resolve_lock:
            entry = await self.load(interrupt_id)
            if entry is None or entry.state != InterruptState.PENDING:
                return None
            entry.state    = InterruptState.RESOLVED
            entry.decision = decision
            entry.decided_at = datetime.now(timezone.utc)
            await self.save(entry)
            return entry

    async def save_batch(self, batch: HitlBatch) -> None:
        await self._ensure_init()
        await asyncio.to_thread(self._save_batch_sync, batch)

    def _save_batch_sync(self, batch: HitlBatch) -> None:
        import sqlite3
        data = self._serialise_batch(batch)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                INSERT INTO batches (batch_id, thread_id, state, created_at, data)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(batch_id) DO UPDATE SET
                    state = excluded.state, data = excluded.data
            """, (
                batch.batch_id, batch.thread_id, batch.state.value,
                batch.created_at.timestamp(), data,
            ))
            conn.commit()

    async def load_batch(self, batch_id: str) -> Optional[HitlBatch]:
        await self._ensure_init()
        return await asyncio.to_thread(self._load_batch_sync, batch_id)

    def _load_batch_sync(self, batch_id: str) -> Optional[HitlBatch]:
        conn = self._pool.get_conn()
        conn.isolation_level = None  # autocommit off; explicit commit below
        if True:  # pool connection block (DESIGN-07)
            row = conn.execute(
                "SELECT data FROM batches WHERE batch_id = ?", (batch_id,),
            ).fetchone()
        if row is None:
            return None
        try:
            return self._deserialise_batch(row[0])
        except Exception as exc:
            logger.warning("load_batch: corrupt %s: %s", batch_id, exc)
            return None

    async def list_pending_batches(
        self, *, limit: int = 50, thread_id: Optional[str] = None,
    ) -> list[HitlBatch]:
        await self._ensure_init()
        return await asyncio.to_thread(self._list_pending_batches_sync, limit, thread_id)

    def _list_pending_batches_sync(
        self, limit: int, thread_id: Optional[str],
    ) -> list[HitlBatch]:
        conn = self._pool.get_conn()
        conn.isolation_level = None  # autocommit off; explicit commit below
        if True:  # pool connection block (DESIGN-07)
            if thread_id:
                rows = conn.execute("""
                    SELECT data FROM batches
                    WHERE state = ? AND thread_id = ?
                    ORDER BY created_at DESC LIMIT ?
                """, (BatchState.PENDING.value, thread_id, limit)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT data FROM batches
                    WHERE state = ?
                    ORDER BY created_at DESC LIMIT ?
                """, (BatchState.PENDING.value, limit)).fetchall()
        out = []
        for (raw,) in rows:
            try:
                out.append(self._deserialise_batch(raw))
            except Exception as exc:
                logger.warning("list_pending_batches: skip corrupt: %s", exc)
        return out

    async def delete_batch(self, batch_id: str) -> bool:
        await self._ensure_init()
        return await asyncio.to_thread(self._delete_batch_sync, batch_id)

    def _delete_batch_sync(self, batch_id: str) -> bool:
        conn = self._pool.get_conn()
        conn.isolation_level = None  # autocommit off; explicit commit below
        if True:  # pool connection block (DESIGN-07)
            cur = conn.execute("DELETE FROM batches WHERE batch_id = ?", (batch_id,))
            conn.commit()
            return cur.rowcount > 0

    @staticmethod
    def _serialise_batch(batch: HitlBatch) -> bytes:
        payload = batch.model_dump(mode="json")
        try:
            import msgpack
            return msgpack.packb(payload, use_bin_type=True)
        except ImportError:
            return json.dumps(payload).encode("utf-8")

    @staticmethod
    def _deserialise_batch(data: bytes) -> HitlBatch:
        try:
            import msgpack
            try:
                unpacked = msgpack.unpackb(data, raw=False)
            except Exception:
                unpacked = json.loads(data.decode("utf-8"))
        except ImportError:
            unpacked = json.loads(data.decode("utf-8"))
        return HitlBatch.model_validate(unpacked)


    async def close(self) -> None:
        """DESIGN-07: close thread-local connection for this store."""
        await asyncio.to_thread(self._pool.close_all)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_store_from_config(cfg: dict[str, Any]) -> BaseCheckpointStore:
    """Construct a store from a config dict.

    Examples:
      {"backend": "memory"}
      {"backend": "redis", "redis_url": "redis://localhost:6379/0"}
      {"backend": "sqlite", "db_path": "/var/lib/hitl/checkpoints.db"}

    Used at host startup to honour HITL_CHECKPOINT_BACKEND env var without
    leaking concrete imports into the rest of the system.
    """
    backend = (cfg.get("backend") or "memory").lower()
    if backend in ("memory", "inmemory", "in_memory"):
        return InMemoryCheckpointStore()
    if backend == "redis":
        return RedisCheckpointStore(redis_url=cfg.get("redis_url", "redis://localhost:6379/0"))
    if backend == "sqlite":
        return SqliteCheckpointStore(db_path=cfg["db_path"])
    raise ValueError(f"Unknown checkpoint backend: {backend!r}")