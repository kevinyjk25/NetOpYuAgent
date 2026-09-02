"""Durable cross-provider Saga coordination above individual L0 plans.

The coordinator never calls a provider directly and never manufactures an
approval.  It binds independently approved Effect Runtime plans into one
immutable business operation, records their terminal outcomes, and exposes the
next forward or reverse compensation action after a restart.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from network_runtime.contracts import canonical_json, sha256_json, utc_now


class SagaError(RuntimeError):
    pass


class SagaState(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPENSATING = "compensating"
    VERIFIED_SUCCESS = "verified_success"
    COMPENSATED = "compensated"
    FAILED = "failed"
    MANUAL_INTERVENTION_REQUIRED = "manual_intervention_required"


class SagaStepState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    VERIFIED = "verified"
    COMPENSATION_REQUIRED = "compensation_required"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    FAILED = "failed"
    MANUAL_INTERVENTION_REQUIRED = "manual_intervention_required"


@dataclass(frozen=True)
class SagaStepSpec:
    step_id: str
    domain: str
    capability_id: str
    depends_on: tuple[str, ...] = ()
    compensation_capability_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["depends_on"] = list(self.depends_on)
        return value


@dataclass(frozen=True)
class SagaDefinition:
    name: str
    version: str
    steps: tuple[SagaStepSpec, ...]
    definition_hash: str

    @classmethod
    def create(
        cls, name: str, version: str, steps: tuple[SagaStepSpec, ...],
    ) -> "SagaDefinition":
        if not name.strip() or not version.strip() or not steps:
            raise ValueError("saga name, version, and at least one step are required")
        positions = {step.step_id: index for index, step in enumerate(steps)}
        if len(positions) != len(steps) or any(not value.strip() for value in positions):
            raise ValueError("saga step ids must be non-empty and unique")
        for index, step in enumerate(steps):
            unknown = set(step.depends_on) - set(positions)
            forward = {
                value for value in step.depends_on
                if value in positions and positions[value] >= index
            }
            if unknown or forward:
                raise ValueError(
                    f"saga step {step.step_id!r} has invalid dependencies: "
                    f"{sorted(unknown | forward)}"
                )
        payload = {
            "name": name,
            "version": version,
            "steps": [step.to_dict() for step in steps],
        }
        return cls(name, version, steps, sha256_json(payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "steps": [step.to_dict() for step in self.steps],
            "definition_hash": self.definition_hash,
        }


class SagaCoordinator:
    """SQLite-backed Saga state machine with a tamper-evident event chain."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path), timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS effect_sagas (
                saga_id TEXT PRIMARY KEY,
                correlation_id TEXT NOT NULL,
                definition_hash TEXT NOT NULL,
                definition_json TEXT NOT NULL,
                state TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS effect_saga_steps (
                saga_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                step_id TEXT NOT NULL,
                domain TEXT NOT NULL,
                capability_id TEXT NOT NULL,
                compensation_capability_id TEXT,
                depends_on_json TEXT NOT NULL,
                state TEXT NOT NULL,
                plan_id TEXT,
                plan_hash TEXT,
                compensation_plan_id TEXT,
                compensation_plan_hash TEXT,
                terminal_state TEXT,
                error TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (saga_id, step_id),
                UNIQUE (saga_id, ordinal),
                FOREIGN KEY(saga_id) REFERENCES effect_sagas(saga_id)
            );
            CREATE TABLE IF NOT EXISTS effect_saga_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                saga_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                previous_hash TEXT,
                event_hash TEXT NOT NULL,
                FOREIGN KEY(saga_id) REFERENCES effect_sagas(saga_id)
            );
            CREATE INDEX IF NOT EXISTS idx_effect_saga_state
                ON effect_sagas(state, updated_at);
        """)
        self.db.commit()
        os.chmod(self.path, 0o600)

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "SagaCoordinator":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def start(
        self, definition: SagaDefinition, *, correlation_id: str,
    ) -> dict[str, Any]:
        if not correlation_id.strip():
            raise SagaError("saga correlation_id is required")
        saga_id = str(uuid.uuid4())
        now = utc_now()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute(
                """INSERT INTO effect_sagas
                   (saga_id, correlation_id, definition_hash, definition_json,
                    state, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    saga_id, correlation_id, definition.definition_hash,
                    canonical_json(definition.to_dict()), SagaState.PLANNED.value, now, now,
                ),
            )
            for ordinal, step in enumerate(definition.steps):
                self.db.execute(
                    """INSERT INTO effect_saga_steps
                       (saga_id, ordinal, step_id, domain, capability_id,
                        compensation_capability_id, depends_on_json, state, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        saga_id, ordinal, step.step_id, step.domain, step.capability_id,
                        step.compensation_capability_id, canonical_json(list(step.depends_on)),
                        SagaStepState.PENDING.value, now,
                    ),
                )
            self._event(saga_id, "saga_started", {
                "definition_hash": definition.definition_hash,
                "correlation_id": correlation_id,
            })
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        return self.inspect(saga_id)

    def bind_plan(
        self, saga_id: str, step_id: str, *, plan_id: str, plan_hash: str,
    ) -> dict[str, Any]:
        if not plan_id or not plan_hash:
            raise SagaError("plan id and hash are required")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            saga, step = self._rows(saga_id, step_id)
            if saga["state"] not in {SagaState.PLANNED.value, SagaState.RUNNING.value}:
                raise SagaError(f"saga {saga_id} cannot bind a forward plan in {saga['state']}")
            if step["state"] == SagaStepState.RUNNING.value:
                if step["plan_id"] == plan_id and step["plan_hash"] == plan_hash:
                    self.db.rollback()
                    return self.inspect(saga_id)
                raise SagaError("saga step already has a different immutable plan")
            if step["state"] != SagaStepState.PENDING.value:
                raise SagaError(f"saga step {step_id} cannot start from {step['state']}")
            self._require_dependencies(saga_id, step)
            now = utc_now()
            self.db.execute(
                """UPDATE effect_saga_steps SET state=?, plan_id=?, plan_hash=?, updated_at=?
                   WHERE saga_id=? AND step_id=?""",
                (SagaStepState.RUNNING.value, plan_id, plan_hash, now, saga_id, step_id),
            )
            self.db.execute(
                "UPDATE effect_sagas SET state=?, updated_at=? WHERE saga_id=?",
                (SagaState.RUNNING.value, now, saga_id),
            )
            self._event(saga_id, "forward_plan_bound", {
                "step_id": step_id, "plan_id": plan_id, "plan_hash": plan_hash,
            })
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        return self.inspect(saga_id)

    def record_outcome(
        self, saga_id: str, step_id: str, *, terminal_state: str,
        error: str | None = None,
    ) -> dict[str, Any]:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            _saga, step = self._rows(saga_id, step_id)
            if step["state"] != SagaStepState.RUNNING.value:
                raise SagaError(f"saga step {step_id} is not running")
            now = utc_now()
            if terminal_state == "verified_success":
                step_state = SagaStepState.VERIFIED
                self.db.execute(
                    """UPDATE effect_saga_steps SET state=?, terminal_state=?, error=?, updated_at=?
                       WHERE saga_id=? AND step_id=?""",
                    (step_state.value, terminal_state, error, now, saga_id, step_id),
                )
                pending = self.db.execute(
                    "SELECT COUNT(*) AS value FROM effect_saga_steps WHERE saga_id=? AND state!=?",
                    (saga_id, SagaStepState.VERIFIED.value),
                ).fetchone()["value"]
                saga_state = SagaState.VERIFIED_SUCCESS if pending == 0 else SagaState.RUNNING
            elif terminal_state == "manual_intervention_required":
                step_state = SagaStepState.MANUAL_INTERVENTION_REQUIRED
                saga_state = SagaState.MANUAL_INTERVENTION_REQUIRED
                self._update_step_failure(saga_id, step_id, step_state, terminal_state, error, now)
            else:
                step_state = SagaStepState.FAILED
                self._update_step_failure(saga_id, step_id, step_state, terminal_state, error, now)
                compensation = self._mark_compensation_required(saga_id, now)
                saga_state = {
                    "required": SagaState.COMPENSATING,
                    "manual": SagaState.MANUAL_INTERVENTION_REQUIRED,
                    "none": SagaState.FAILED,
                }[compensation]
            self.db.execute(
                "UPDATE effect_sagas SET state=?, updated_at=? WHERE saga_id=?",
                (saga_state.value, now, saga_id),
            )
            self._event(saga_id, "forward_outcome_recorded", {
                "step_id": step_id,
                "terminal_state": terminal_state,
                "saga_state": saga_state.value,
                "error": error,
            })
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        return self.inspect(saga_id)

    def bind_compensation_plan(
        self, saga_id: str, step_id: str, *, plan_id: str, plan_hash: str,
    ) -> dict[str, Any]:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            saga, step = self._rows(saga_id, step_id)
            if saga["state"] != SagaState.COMPENSATING.value:
                raise SagaError("saga is not compensating")
            if step["state"] != SagaStepState.COMPENSATION_REQUIRED.value:
                raise SagaError(f"step {step_id} does not require compensation")
            next_action = self._next_compensation_row(saga_id)
            if next_action is None or next_action["step_id"] != step_id:
                raise SagaError("compensation must run in reverse verified order")
            now = utc_now()
            self.db.execute(
                """UPDATE effect_saga_steps SET state=?, compensation_plan_id=?,
                   compensation_plan_hash=?, updated_at=? WHERE saga_id=? AND step_id=?""",
                (
                    SagaStepState.COMPENSATING.value, plan_id, plan_hash, now,
                    saga_id, step_id,
                ),
            )
            self._event(saga_id, "compensation_plan_bound", {
                "step_id": step_id, "plan_id": plan_id, "plan_hash": plan_hash,
            })
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        return self.inspect(saga_id)

    def request_compensation(self, saga_id: str, *, reason: str) -> dict[str, Any]:
        """Enter reverse recovery after a cross-step assertion or read fails."""
        self.db.execute("BEGIN IMMEDIATE")
        try:
            saga = self._saga(saga_id)
            if saga["state"] not in {SagaState.PLANNED.value, SagaState.RUNNING.value}:
                raise SagaError(f"saga {saga_id} cannot request compensation in {saga['state']}")
            running = self.db.execute(
                """SELECT COUNT(*) AS value FROM effect_saga_steps
                   WHERE saga_id=? AND state=?""",
                (saga_id, SagaStepState.RUNNING.value),
            ).fetchone()["value"]
            now = utc_now()
            if running:
                saga_state = SagaState.MANUAL_INTERVENTION_REQUIRED
            else:
                compensation = self._mark_compensation_required(saga_id, now)
                saga_state = {
                    "required": SagaState.COMPENSATING,
                    "manual": SagaState.MANUAL_INTERVENTION_REQUIRED,
                    "none": SagaState.FAILED,
                }[compensation]
            self.db.execute(
                "UPDATE effect_sagas SET state=?, updated_at=? WHERE saga_id=?",
                (saga_state.value, now, saga_id),
            )
            self._event(saga_id, "compensation_requested", {
                "reason": reason, "saga_state": saga_state.value,
                "running_steps": int(running),
            })
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        return self.inspect(saga_id)

    def record_compensation_outcome(
        self, saga_id: str, step_id: str, *, terminal_state: str,
        error: str | None = None,
    ) -> dict[str, Any]:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            _saga, step = self._rows(saga_id, step_id)
            if step["state"] != SagaStepState.COMPENSATING.value:
                raise SagaError(f"step {step_id} is not compensating")
            now = utc_now()
            if terminal_state not in {"verified_success", "rollback_verified"}:
                step_state = SagaStepState.MANUAL_INTERVENTION_REQUIRED
                saga_state = SagaState.MANUAL_INTERVENTION_REQUIRED
            else:
                step_state = SagaStepState.COMPENSATED
                remaining = self.db.execute(
                    """SELECT COUNT(*) AS value FROM effect_saga_steps
                       WHERE saga_id=? AND state=? AND step_id!=?""",
                    (saga_id, SagaStepState.COMPENSATION_REQUIRED.value, step_id),
                ).fetchone()["value"]
                saga_state = SagaState.COMPENSATING if remaining else SagaState.COMPENSATED
            self.db.execute(
                """UPDATE effect_saga_steps SET state=?, terminal_state=?, error=?, updated_at=?
                   WHERE saga_id=? AND step_id=?""",
                (step_state.value, terminal_state, error, now, saga_id, step_id),
            )
            self.db.execute(
                "UPDATE effect_sagas SET state=?, updated_at=? WHERE saga_id=?",
                (saga_state.value, now, saga_id),
            )
            self._event(saga_id, "compensation_outcome_recorded", {
                "step_id": step_id, "terminal_state": terminal_state,
                "saga_state": saga_state.value, "error": error,
            })
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        return self.inspect(saga_id)

    def next_action(self, saga_id: str) -> dict[str, Any] | None:
        saga = self._saga(saga_id)
        if saga["state"] in {SagaState.PLANNED.value, SagaState.RUNNING.value}:
            rows = self.db.execute(
                "SELECT * FROM effect_saga_steps WHERE saga_id=? ORDER BY ordinal",
                (saga_id,),
            ).fetchall()
            for row in rows:
                if row["state"] != SagaStepState.PENDING.value:
                    continue
                try:
                    self._require_dependencies(saga_id, row)
                except SagaError:
                    continue
                return {"action": "execute", **self._step_dict(row)}
            return None
        if saga["state"] == SagaState.COMPENSATING.value:
            row = self._next_compensation_row(saga_id)
            return None if row is None else {"action": "compensate", **self._step_dict(row)}
        return None

    def recoverable(self) -> list[dict[str, Any]]:
        rows = self.db.execute(
            """SELECT saga_id FROM effect_sagas WHERE state IN (?, ?, ?)
               ORDER BY updated_at""",
            (SagaState.PLANNED.value, SagaState.RUNNING.value, SagaState.COMPENSATING.value),
        ).fetchall()
        return [self.inspect(row["saga_id"]) for row in rows]

    def inspect(self, saga_id: str) -> dict[str, Any]:
        saga = self._saga(saga_id)
        steps = self.db.execute(
            "SELECT * FROM effect_saga_steps WHERE saga_id=? ORDER BY ordinal",
            (saga_id,),
        ).fetchall()
        events = self.db.execute(
            """SELECT event_type, payload_json, created_at, previous_hash, event_hash
               FROM effect_saga_events WHERE saga_id=? ORDER BY event_id""",
            (saga_id,),
        ).fetchall()
        saga_value = dict(saga)
        saga_value.pop("definition_json", None)
        rendered_events = []
        for row in events:
            event = dict(row)
            event["payload"] = json.loads(event.pop("payload_json"))
            rendered_events.append(event)
        return {
            **saga_value,
            "definition": json.loads(saga["definition_json"]),
            "steps": [self._step_dict(row) for row in steps],
            "events": rendered_events,
            "audit": self.verify_event_chain(saga_id),
            "next_action": self.next_action(saga_id),
        }

    def verify_event_chain(self, saga_id: str) -> dict[str, Any]:
        rows = self.db.execute(
            """SELECT event_type, payload_json, created_at, previous_hash, event_hash
               FROM effect_saga_events WHERE saga_id=? ORDER BY event_id""",
            (saga_id,),
        ).fetchall()
        previous: str | None = None
        for index, row in enumerate(rows):
            payload = {
                "saga_id": saga_id,
                "event_type": row["event_type"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
                "previous_hash": previous,
            }
            if row["previous_hash"] != previous or row["event_hash"] != sha256_json(payload):
                return {"ok": False, "events": len(rows), "invalid_index": index}
            previous = row["event_hash"]
        return {"ok": bool(rows), "events": len(rows), "head": previous}

    def _saga(self, saga_id: str) -> sqlite3.Row:
        row = self.db.execute(
            "SELECT * FROM effect_sagas WHERE saga_id=?", (saga_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown effect saga {saga_id}")
        definition = json.loads(row["definition_json"])
        if sha256_json({
            "name": definition["name"],
            "version": definition["version"],
            "steps": definition["steps"],
        }) != row["definition_hash"]:
            raise SagaError("saga definition hash mismatch")
        return row

    def _rows(self, saga_id: str, step_id: str) -> tuple[sqlite3.Row, sqlite3.Row]:
        saga = self._saga(saga_id)
        step = self.db.execute(
            "SELECT * FROM effect_saga_steps WHERE saga_id=? AND step_id=?",
            (saga_id, step_id),
        ).fetchone()
        if step is None:
            raise KeyError(f"unknown saga step {step_id!r}")
        return saga, step

    def _require_dependencies(self, saga_id: str, step: sqlite3.Row) -> None:
        dependencies = json.loads(step["depends_on_json"])
        if not dependencies:
            return
        rows = self.db.execute(
            """SELECT step_id, state FROM effect_saga_steps
               WHERE saga_id=? AND step_id IN (%s)""" % ",".join("?" for _ in dependencies),
            (saga_id, *dependencies),
        ).fetchall()
        states = {row["step_id"]: row["state"] for row in rows}
        missing = [
            value for value in dependencies
            if states.get(value) != SagaStepState.VERIFIED.value
        ]
        if missing:
            raise SagaError("saga dependencies are not verified: " + ", ".join(missing))

    def _mark_compensation_required(self, saga_id: str, now: str) -> str:
        rows = self.db.execute(
            """SELECT step_id, compensation_capability_id FROM effect_saga_steps
               WHERE saga_id=? AND state=? ORDER BY ordinal DESC""",
            (saga_id, SagaStepState.VERIFIED.value),
        ).fetchall()
        if any(row["compensation_capability_id"] is None for row in rows):
            self.db.execute(
                """UPDATE effect_saga_steps SET state=?, updated_at=?
                   WHERE saga_id=? AND state=? AND compensation_capability_id IS NULL""",
                (
                    SagaStepState.MANUAL_INTERVENTION_REQUIRED.value, now,
                    saga_id, SagaStepState.VERIFIED.value,
                ),
            )
            return "manual"
        self.db.execute(
            """UPDATE effect_saga_steps SET state=?, updated_at=?
               WHERE saga_id=? AND state=?""",
            (
                SagaStepState.COMPENSATION_REQUIRED.value, now,
                saga_id, SagaStepState.VERIFIED.value,
            ),
        )
        return "required" if rows else "none"

    def _next_compensation_row(self, saga_id: str) -> sqlite3.Row | None:
        return self.db.execute(
            """SELECT * FROM effect_saga_steps WHERE saga_id=? AND state=?
               ORDER BY ordinal DESC LIMIT 1""",
            (saga_id, SagaStepState.COMPENSATION_REQUIRED.value),
        ).fetchone()

    def _update_step_failure(
        self, saga_id: str, step_id: str, state: SagaStepState,
        terminal_state: str, error: str | None, now: str,
    ) -> None:
        self.db.execute(
            """UPDATE effect_saga_steps SET state=?, terminal_state=?, error=?, updated_at=?
               WHERE saga_id=? AND step_id=?""",
            (state.value, terminal_state, error, now, saga_id, step_id),
        )

    def _event(self, saga_id: str, event_type: str, payload: dict[str, Any]) -> None:
        previous = self.db.execute(
            """SELECT event_hash FROM effect_saga_events
               WHERE saga_id=? ORDER BY event_id DESC LIMIT 1""",
            (saga_id,),
        ).fetchone()
        previous_hash = previous["event_hash"] if previous else None
        created_at = utc_now()
        value = {
            "saga_id": saga_id,
            "event_type": event_type,
            "payload": payload,
            "created_at": created_at,
            "previous_hash": previous_hash,
        }
        self.db.execute(
            """INSERT INTO effect_saga_events
               (saga_id, event_type, payload_json, created_at, previous_hash, event_hash)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                saga_id, event_type, canonical_json(payload), created_at,
                previous_hash, sha256_json(value),
            ),
        )

    @staticmethod
    def _step_dict(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["depends_on"] = json.loads(value.pop("depends_on_json"))
        return value
