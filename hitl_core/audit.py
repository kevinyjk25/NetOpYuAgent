"""
hitl_core.audit — Append-only HITL audit log.

Records every interrupt lifecycle event so operators can reconstruct
"who decided what when" after the fact. Domain-neutral by design: the
log is just a sequence of HitlAuditRecord objects, persisted via a
pluggable AuditSink.

Three sinks ship with the module:

  • InMemoryAuditSink — bounded ring buffer; fine for dev / tests
  • FileAuditSink     — JSONL file; cheap durable option for single-host
  • RedisAuditSink    — Redis stream; production / multi-replica

All implement the same AuditSink protocol; hosts can plug in their own
(SIEM forwarder, Kafka producer, etc.) by subclassing.

Usage:

    sink   = FileAuditSink("/var/log/hitl/audit.jsonl")
    logger = AuditLogger(sink=sink)
    await logger.record(
        interrupt_id="...",
        event_kind=AuditEventKind.DECISION_MADE,
        actor="alice",
        payload={"decision": "approve"},
    )

    # Wire into HitlRouter / HitlPipeline as the on_audit hook:
    router = HitlRouter(store=store, on_audit=logger.as_hook())
"""

from __future__ import annotations

import abc
import asyncio
import json
import logging
import os
from collections import deque
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

from .schema import AuditEventKind, HitlAuditRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sink protocol
# ---------------------------------------------------------------------------

class AuditSink(abc.ABC):
    """Append-only sink for audit records."""

    @abc.abstractmethod
    async def append(self, record: HitlAuditRecord) -> None:
        """Persist one record. Failures should NOT raise — audit
        loss is preferable to blocking decision delivery, since
        decisions are the operator's source of truth."""

    async def query(
        self,
        *,
        interrupt_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[HitlAuditRecord]:
        """Optional read API. Default returns []; subclasses that
        support querying (in-memory, sqlite) override."""
        return []

    async def close(self) -> None:
        """Release resources. Default no-op."""


# ---------------------------------------------------------------------------
# In-memory sink
# ---------------------------------------------------------------------------

class InMemoryAuditSink(AuditSink):
    """Bounded ring buffer. Use for dev / tests where audit doesn't
    need to survive process restarts. Constant memory."""

    def __init__(self, *, max_records: int = 10_000) -> None:
        self._records: deque[HitlAuditRecord] = deque(maxlen=max_records)
        self._lock = asyncio.Lock()

    async def append(self, record: HitlAuditRecord) -> None:
        async with self._lock:
            self._records.append(record)

    async def query(
        self, *,
        interrupt_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[HitlAuditRecord]:
        async with self._lock:
            results: list[HitlAuditRecord] = []
            # Walk newest first (deque is oldest-first by default)
            for rec in reversed(self._records):
                if interrupt_id and rec.interrupt_id != interrupt_id:
                    continue
                if thread_id and rec.thread_id != thread_id:
                    continue
                results.append(rec)
                if len(results) >= limit:
                    break
            return results


# ---------------------------------------------------------------------------
# File sink (JSONL)
# ---------------------------------------------------------------------------

class FileAuditSink(AuditSink):
    """Append-only JSONL file. One record per line. Survives restarts;
    no querying support beyond reading the file directly. Suitable for
    single-host deployments and as a feed for external log processors.

    Failures are logged but not raised — the goal is to never block
    decision delivery on audit infrastructure problems.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = asyncio.Lock()
        # Ensure directory exists; create file lazily on first write
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    async def append(self, record: HitlAuditRecord) -> None:
        async with self._lock:
            try:
                line = record.model_dump_json()
                # asyncio.to_thread keeps disk I/O off the event loop
                await asyncio.to_thread(self._append_sync, line)
            except Exception as exc:
                logger.warning("FileAuditSink.append failed: %s", exc)

    def _append_sync(self, line: str) -> None:
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Redis sink (Redis Streams)
# ---------------------------------------------------------------------------

class RedisAuditSink(AuditSink):
    """Append to a Redis Stream. Use for multi-replica deployments
    where a centralised audit log is needed.

    Stream layout: one stream per host (default "hitl:audit"). Each
    record becomes a stream entry with all HitlAuditRecord fields as
    flat key-value pairs (Redis-streams native shape).
    """

    def __init__(
        self,
        *,
        redis_url: str = "redis://localhost:6379/0",
        stream_key: str = "hitl:audit",
        maxlen: int = 100_000,
    ) -> None:
        try:
            import redis.asyncio as redis_asyncio
        except ImportError as exc:
            raise RuntimeError(
                "RedisAuditSink requires redis-py: pip install redis>=5.0"
            ) from exc
        self._redis = redis_asyncio.from_url(redis_url, decode_responses=True)
        self._stream_key = stream_key
        self._maxlen = maxlen

    async def append(self, record: HitlAuditRecord) -> None:
        # Serialise compound fields as JSON; flat fields go straight in.
        data = record.model_dump(mode="json")
        # Redis stream values must be strings; flatten payload
        flat: dict[str, str] = {}
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                flat[k] = json.dumps(v)
            elif v is None:
                flat[k] = ""
            else:
                flat[k] = str(v)
        try:
            # XADD with MAXLEN ~ for approximate trimming (cheap)
            await self._redis.xadd(
                self._stream_key, flat, maxlen=self._maxlen, approximate=True,
            )
        except Exception as exc:
            logger.warning("RedisAuditSink.append failed: %s", exc)

    async def query(
        self, *,
        interrupt_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[HitlAuditRecord]:
        # XREVRANGE pulls newest first; we filter client-side. This is
        # adequate for moderate volumes; high-volume hosts should index
        # separately (e.g. a per-interrupt index stream).
        try:
            entries = await self._redis.xrevrange(
                self._stream_key, count=limit * 4,  # over-fetch for filtering
            )
        except Exception as exc:
            logger.warning("RedisAuditSink.query failed: %s", exc)
            return []
        results: list[HitlAuditRecord] = []
        for entry_id, fields in entries:
            try:
                record = self._record_from_fields(fields)
            except Exception as exc:
                logger.debug("Skip corrupt audit record: %s", exc)
                continue
            if interrupt_id and record.interrupt_id != interrupt_id:
                continue
            if thread_id and record.thread_id != thread_id:
                continue
            results.append(record)
            if len(results) >= limit:
                break
        return results

    @staticmethod
    def _record_from_fields(fields: dict) -> HitlAuditRecord:
        # Inverse of append's flattening
        rebuilt: dict[str, Any] = {}
        for k, v in fields.items():
            if k in ("payload",):
                try:
                    rebuilt[k] = json.loads(v) if v else {}
                except Exception:
                    rebuilt[k] = {}
            else:
                rebuilt[k] = v
        return HitlAuditRecord.model_validate(rebuilt)

    async def close(self) -> None:
        try:
            await self._redis.aclose()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# AuditLogger — convenience facade
# ---------------------------------------------------------------------------

class AuditLogger:
    """Thin wrapper that builds HitlAuditRecord from raw fields and
    delegates to a sink. The router/pipeline accept a callable for
    on_audit; AuditLogger.as_hook() returns one bound to this logger.
    """

    def __init__(self, *, sink: AuditSink) -> None:
        self._sink = sink

    async def record(
        self,
        *,
        interrupt_id: str,
        event_kind: AuditEventKind,
        actor: str = "system",
        thread_id: str = "",
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        rec = HitlAuditRecord(
            interrupt_id=interrupt_id,
            thread_id=thread_id,
            event_kind=event_kind,
            actor=actor,
            payload=payload or {},
            timestamp=datetime.now(timezone.utc),
        )
        await self._sink.append(rec)

    def as_hook(
        self,
    ) -> Callable[[AuditEventKind, str, dict[str, Any]], Awaitable[None]]:
        """Adapter for HitlRouter / HitlPipeline `on_audit` parameter,
        which expects a 3-arg callable (kind, interrupt_id, payload).
        Inside the hook we extract `actor` from payload if present."""
        async def _hook(
            kind: AuditEventKind, interrupt_id: str, payload: dict[str, Any],
        ) -> None:
            actor = payload.get("operator") or payload.get("actor") or "system"
            thread_id = payload.get("thread_id", "")
            await self.record(
                interrupt_id=interrupt_id,
                event_kind=kind,
                actor=actor,
                thread_id=thread_id,
                payload=payload,
            )
        return _hook

    async def query(
        self, *,
        interrupt_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[HitlAuditRecord]:
        return await self._sink.query(
            interrupt_id=interrupt_id, thread_id=thread_id, limit=limit,
        )

    async def close(self) -> None:
        await self._sink.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_sink_from_config(cfg: dict[str, Any]) -> AuditSink:
    """Construct a sink from a config dict.

    Examples:
      {"backend": "memory"}
      {"backend": "file", "path": "/var/log/hitl/audit.jsonl"}
      {"backend": "redis",
       "redis_url": "redis://prod:6379/0",
       "stream_key": "hitl:audit"}
    """
    backend = (cfg.get("backend") or "memory").lower()
    if backend in ("memory", "inmemory", "in_memory"):
        return InMemoryAuditSink(max_records=cfg.get("max_records", 10_000))
    if backend == "file":
        return FileAuditSink(path=cfg["path"])
    if backend == "redis":
        return RedisAuditSink(
            redis_url=cfg.get("redis_url", "redis://localhost:6379/0"),
            stream_key=cfg.get("stream_key", "hitl:audit"),
            maxlen=cfg.get("maxlen", 100_000),
        )
    raise ValueError(f"Unknown audit backend: {backend!r}")