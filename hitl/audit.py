"""
hitl/audit.py
-------------
Layer 5 — Audit and observability.

HitlAuditService
  - Appends immutable HitlAuditRecord rows to PostgreSQL (production)
    or a local in-memory list (development / testing).
  - Exposes query helpers for the dashboard.
  - Emits Prometheus counters and histograms.

Prometheus metrics exposed
--------------------------
  hitl_interrupts_total{trigger, risk}     Counter
  hitl_decisions_total{decision}           Counter
  hitl_interrupt_latency_seconds{trigger}  Histogram
  hitl_sla_breaches_total{trigger}         Counter

Usage
-----
    # Dev / test
    audit = HitlAuditService.in_memory()

    # Production (PostgreSQL)
    audit = HitlAuditService.postgres(dsn="postgresql://...")

    # Write a record
    await audit.write(HitlAuditRecord(...))

    # Query
    records = await audit.get_by_interrupt(interrupt_id)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Protocol

from .schemas import AuditEventKind, HitlAuditRecord, TriggerKind

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional Prometheus integration
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Histogram

    _INTERRUPTS_TOTAL = Counter(
        "hitl_interrupts_total",
        "Total HITL interrupts fired",
        ["trigger", "risk"],
    )
    _DECISIONS_TOTAL = Counter(
        "hitl_decisions_total",
        "Total HITL decisions received",
        ["decision"],
    )
    _LATENCY = Histogram(
        "hitl_interrupt_latency_seconds",
        "Time between interrupt fired and decision received (seconds)",
        ["trigger"],
        buckets=[30, 60, 120, 300, 600, 900, 1800, 3600],
    )
    _SLA_BREACHES = Counter(
        "hitl_sla_breaches_total",
        "Total HITL SLA breaches (timeouts)",
        ["trigger"],
    )
    _PROMETHEUS_AVAILABLE = True

except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger.debug("prometheus_client not installed; metrics disabled")


def _inc_interrupt(trigger: str, risk: str) -> None:
    if _PROMETHEUS_AVAILABLE:
        _INTERRUPTS_TOTAL.labels(trigger=trigger, risk=risk).inc()


def _inc_decision(decision: str) -> None:
    if _PROMETHEUS_AVAILABLE:
        _DECISIONS_TOTAL.labels(decision=decision).inc()


def _observe_latency(trigger: str, seconds: float) -> None:
    if _PROMETHEUS_AVAILABLE:
        _LATENCY.labels(trigger=trigger).observe(seconds)


def _inc_sla_breach(trigger: str) -> None:
    if _PROMETHEUS_AVAILABLE:
        _SLA_BREACHES.labels(trigger=trigger).inc()


# ---------------------------------------------------------------------------
# Storage backend protocol
# ---------------------------------------------------------------------------

class AuditBackend(Protocol):
    async def append(self, record: HitlAuditRecord) -> None: ...
    async def find_by_interrupt(self, interrupt_id: str) -> list[HitlAuditRecord]: ...
    async def find_by_thread(self, thread_id: str) -> list[HitlAuditRecord]: ...
    async def find_recent(self, limit: int) -> list[HitlAuditRecord]: ...


# ---------------------------------------------------------------------------
# In-memory backend  (dev / testing)
# ---------------------------------------------------------------------------

class InMemoryAuditBackend:
    def __init__(self) -> None:
        self._records: list[HitlAuditRecord] = []
        self._lock = asyncio.Lock()

    async def append(self, record: HitlAuditRecord) -> None:
        async with self._lock:
            self._records.append(record)

    async def find_by_interrupt(self, interrupt_id: str) -> list[HitlAuditRecord]:
        async with self._lock:
            return [r for r in self._records if r.interrupt_id == interrupt_id]

    async def find_by_thread(self, thread_id: str) -> list[HitlAuditRecord]:
        async with self._lock:
            return [r for r in self._records if r.thread_id == thread_id]

    async def find_recent(self, limit: int = 50) -> list[HitlAuditRecord]:
        async with self._lock:
            return list(reversed(self._records[-limit:]))


# ---------------------------------------------------------------------------
# PostgreSQL backend  (production)
# ---------------------------------------------------------------------------


class SqliteAuditBackend:
    """
    SQLite-backed audit log for production use without PostgreSQL.
    Stores immutable HITL decision records locally.
    Schema is compatible with a future migration to PostgreSQL.
    """

    def __init__(self, db_path: str = "data/hitl_audit.db") -> None:
        import sqlite3, pathlib
        pathlib.Path(db_path).parent.mkdir(exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = asyncio.Lock()
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS hitl_audit_log (
                record_id    TEXT PRIMARY KEY,
                interrupt_id TEXT NOT NULL,
                thread_id    TEXT,
                event_kind   TEXT NOT NULL,
                trigger_kind TEXT,
                risk_level   TEXT,
                decision     TEXT,
                query        TEXT,
                operator     TEXT,
                reason       TEXT,
                duration_s   REAL,
                created_at   TEXT NOT NULL
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_interrupt ON hitl_audit_log(interrupt_id)"
        )
        self._conn.commit()

    async def append(self, record: "HitlAuditRecord") -> None:
        async with self._lock:
            self._conn.execute(
                """INSERT OR IGNORE INTO hitl_audit_log
                   (record_id, interrupt_id, thread_id, event_kind, trigger_kind,
                    risk_level, decision, query, operator, reason, duration_s, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    record.record_id,
                    record.interrupt_id,
                    getattr(record, "thread_id", ""),
                    str(record.event_kind.value if hasattr(record.event_kind, "value") else record.event_kind),
                    str(record.trigger_kind.value if hasattr(record, "trigger_kind") and record.trigger_kind else ""),
                    str(record.risk_level.value   if hasattr(record, "risk_level")   and record.risk_level   else ""),
                    str(record.decision.value     if hasattr(record, "decision")     and record.decision     else ""),
                    getattr(record, "query",    "")[:1000],
                    getattr(record, "operator", "system"),
                    getattr(record, "reason",   "")[:500],
                    getattr(record, "duration_s", None),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._conn.commit()

    async def find_by_interrupt(self, interrupt_id: str) -> list:
        async with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM hitl_audit_log WHERE interrupt_id=? ORDER BY created_at",
                (interrupt_id,),
            ).fetchall()
            return rows

    async def recent(self, limit: int = 100) -> list:
        async with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM hitl_audit_log ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return rows


class PostgresAuditBackend:
    """
    Async PostgreSQL backend using asyncpg.

    Schema (run once on first deploy):
    ─────────────────────────────────
    CREATE TABLE hitl_audit_log (
        record_id    TEXT PRIMARY KEY,
        interrupt_id TEXT NOT NULL,
        thread_id    TEXT NOT NULL,
        event_kind   TEXT NOT NULL,
        actor        TEXT NOT NULL,
        payload      JSONB DEFAULT '{}',
        timestamp    TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE INDEX ON hitl_audit_log (interrupt_id);
    CREATE INDEX ON hitl_audit_log (thread_id);
    CREATE INDEX ON hitl_audit_log (timestamp DESC);
    """

    def __init__(self, pool: Any) -> None:
        # pool: asyncpg.Pool
        self._pool = pool

    async def append(self, record: HitlAuditRecord) -> None:
        import json as _json
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO hitl_audit_log
                    (record_id, interrupt_id, thread_id, event_kind, actor, payload, timestamp)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
                """,
                record.record_id,
                record.interrupt_id,
                record.thread_id,
                record.event_kind.value,
                record.actor,
                _json.dumps(record.payload),
                datetime.fromisoformat(record.timestamp),
            )

    async def find_by_interrupt(self, interrupt_id: str) -> list[HitlAuditRecord]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM hitl_audit_log WHERE interrupt_id = $1 ORDER BY timestamp",
                interrupt_id,
            )
        return [self._row_to_record(r) for r in rows]

    async def find_by_thread(self, thread_id: str) -> list[HitlAuditRecord]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM hitl_audit_log WHERE thread_id = $1 ORDER BY timestamp",
                thread_id,
            )
        return [self._row_to_record(r) for r in rows]

    async def find_recent(self, limit: int = 50) -> list[HitlAuditRecord]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM hitl_audit_log ORDER BY timestamp DESC LIMIT $1",
                limit,
            )
        return [self._row_to_record(r) for r in rows]

    @staticmethod
    def _row_to_record(row: Any) -> HitlAuditRecord:
        return HitlAuditRecord(
            record_id=row["record_id"],
            interrupt_id=row["interrupt_id"],
            thread_id=row["thread_id"],
            event_kind=AuditEventKind(row["event_kind"]),
            actor=row["actor"],
            payload=dict(row["payload"]),
            timestamp=row["timestamp"].isoformat(),
        )


# ---------------------------------------------------------------------------
# HitlAuditService  (facade)
# ---------------------------------------------------------------------------

class HitlAuditService:
    """
    Facade over the storage backend.
    Handles Prometheus metric emission automatically on write.
    """

    def __init__(self, backend: AuditBackend) -> None:
        self._backend  = backend
        # interrupt_id → fired timestamp (for latency calculation)
        self._fired_at: dict[str, datetime] = {}

    @classmethod
    def in_memory(cls) -> "HitlAuditService":
        return cls(InMemoryAuditBackend())

    @classmethod
    def sqlite(cls, db_path: str = "data/hitl_audit.db") -> "HitlAuditService":
        """SQLite-backed audit log — production default when PostgreSQL not configured."""
        return cls(SqliteAuditBackend(db_path))

    @classmethod
    async def postgres(cls, dsn: str) -> "HitlAuditService":
        """Create a PostgreSQL-backed audit service."""
        try:
            import asyncpg
            pool = await asyncpg.create_pool(dsn)
            return cls(PostgresAuditBackend(pool))
        except ImportError:
            raise RuntimeError(
                "asyncpg is required for PostgreSQL audit backend. "
                "Install with: pip install asyncpg"
            )

    async def write(self, record: HitlAuditRecord) -> None:
        """Persist an audit record and emit Prometheus metrics."""
        await self._backend.append(record)
        self._emit_metrics(record)
        logger.debug(
            "Audit: interrupt_id=%s event=%s actor=%s",
            record.interrupt_id,
            record.event_kind.value,
            record.actor,
        )

    async def get_by_interrupt(self, interrupt_id: str) -> list[HitlAuditRecord]:
        return await self._backend.find_by_interrupt(interrupt_id)

    async def get_by_thread(self, thread_id: str) -> list[HitlAuditRecord]:
        return await self._backend.find_by_thread(thread_id)

    async def get_recent(self, limit: int = 50) -> list[HitlAuditRecord]:
        return await self._backend.find_recent(limit)

    # ------------------------------------------------------------------

    def _emit_metrics(self, record: HitlAuditRecord) -> None:
        kind = record.event_kind

        if kind == AuditEventKind.INTERRUPT_FIRED:
            trigger = record.payload.get("trigger_kind", "unknown")
            risk    = record.payload.get("risk_level",   "unknown")
            _inc_interrupt(trigger, risk)
            self._fired_at[record.interrupt_id] = datetime.now(timezone.utc)

        elif kind == AuditEventKind.DECISION_RECEIVED:
            decision = record.payload.get("decision", "unknown")
            _inc_decision(decision)

            # Latency from interrupt → decision
            fired = self._fired_at.pop(record.interrupt_id, None)
            if fired:
                latency = (datetime.now(timezone.utc) - fired).total_seconds()
                # Derive trigger from stored interrupt records (best-effort)
                _observe_latency("unknown", latency)

        elif kind == AuditEventKind.TIMEOUT_TRIGGERED:
            trigger = record.payload.get("trigger_kind", "unknown")
            _inc_sla_breach(trigger)
