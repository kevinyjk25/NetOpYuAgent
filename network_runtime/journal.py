"""SQLite append-only execution journal, one-shot leases and resource locks."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import (
    ALLOWED_TRANSITIONS,
    TERMINAL_STATES,
    ApprovalError,
    PlanIntegrityError,
    PlanState,
    PreparedPlan,
    ResourceConflictError,
    StateTransitionError,
    canonical_json,
    sha256_json,
    utc_now,
)


GENESIS_EVENT_HASH = "GENESIS"


def _nonce_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class NetworkJournal:
    def __init__(self, path: str | Path, *, recover_crashed: bool = False):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS plans (
                plan_id TEXT PRIMARY KEY,
                plan_hash TEXT NOT NULL UNIQUE,
                plan_json TEXT NOT NULL,
                state TEXT NOT NULL,
                nonce_hash TEXT NOT NULL,
                nonce_consumed_at TEXT,
                approval_request_id TEXT,
                approval_actor TEXT,
                approved_at TEXT,
                result_json TEXT,
                error_text TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS plan_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_id TEXT NOT NULL,
                from_state TEXT,
                to_state TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                prev_event_hash TEXT,
                event_hash TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(plan_id) REFERENCES plans(plan_id)
            );
            CREATE TABLE IF NOT EXISTS resource_locks (
                resource_key TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL,
                acquired_at TEXT NOT NULL,
                FOREIGN KEY(plan_id) REFERENCES plans(plan_id)
            );
            CREATE INDEX IF NOT EXISTS idx_plan_events_plan
                ON plan_events(plan_id, event_id);
        """)
        self._ensure_column("plan_events", "prev_event_hash", "TEXT")
        self._ensure_column("plan_events", "event_hash", "TEXT")
        self._backfill_event_hashes()
        self._db.commit()
        os.chmod(self.path, 0o600)
        # Recovery is a worker-startup operation, not a connection-open
        # operation. Running it for every concurrent request would falsely
        # classify another request's in-flight plan as a crash.
        if recover_crashed:
            self._recover_crashed_plans()

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "NetworkJournal":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _ensure_column(self, table: str, column: str, declaration: str) -> None:
        existing = {
            str(row["name"])
            for row in self._db.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in existing:
            self._db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    @staticmethod
    def _event_hash(
        *,
        plan_id: str,
        from_state: str | None,
        to_state: str,
        event_type: str,
        payload_json: str,
        created_at: str,
        prev_event_hash: str,
    ) -> str:
        return sha256_json({
            "plan_id": plan_id,
            "from_state": from_state,
            "to_state": to_state,
            "event_type": event_type,
            "payload_json": payload_json,
            "created_at": created_at,
            "prev_event_hash": prev_event_hash,
        })

    def _backfill_event_hashes(self) -> None:
        """Upgrade legacy unhashed rows once without healing later tampering."""
        missing = self._db.execute(
            "SELECT COUNT(*) AS count FROM plan_events "
            "WHERE prev_event_hash IS NULL OR event_hash IS NULL"
        ).fetchone()["count"]
        if not missing:
            return
        previous_by_plan: dict[str, str] = {}
        rows = self._db.execute(
            """SELECT event_id, plan_id, from_state, to_state, event_type,
                      payload_json, prev_event_hash, event_hash, created_at
               FROM plan_events ORDER BY plan_id, event_id"""
        ).fetchall()
        for row in rows:
            plan_id = str(row["plan_id"])
            previous = previous_by_plan.get(plan_id, GENESIS_EVENT_HASH)
            if row["prev_event_hash"] is None and row["event_hash"] is None:
                event_hash = self._event_hash(
                    plan_id=plan_id,
                    from_state=row["from_state"],
                    to_state=str(row["to_state"]),
                    event_type=str(row["event_type"]),
                    payload_json=str(row["payload_json"]),
                    created_at=str(row["created_at"]),
                    prev_event_hash=previous,
                )
                self._db.execute(
                    "UPDATE plan_events SET prev_event_hash=?, event_hash=? WHERE event_id=?",
                    (previous, event_hash, row["event_id"]),
                )
                previous_by_plan[plan_id] = event_hash
            else:
                # A partially populated or invalid row is evidence of damage;
                # preserve it so verify_event_chain reports the inconsistency.
                previous_by_plan[plan_id] = str(row["event_hash"] or "")

    def _recover_crashed_plans(self) -> None:
        now = utc_now()
        rows = self._db.execute(
            "SELECT plan_id, state, plan_json FROM plans WHERE state IN (?, ?, ?)",
            (
                PlanState.EXECUTING.value,
                PlanState.VERIFYING.value,
                PlanState.ROLLING_BACK.value,
            ),
        ).fetchall()
        for row in rows:
            value = json.loads(row["plan_json"])
            value["state"] = PlanState.OUTCOME_INDETERMINATE.value
            self._db.execute(
                "UPDATE plans SET state=?, plan_json=?, error_text=?, updated_at=? WHERE plan_id=?",
                (
                    PlanState.OUTCOME_INDETERMINATE.value,
                    canonical_json(value),
                    "runtime restarted during side-effect processing; reconciliation required",
                    now,
                    row["plan_id"],
                ),
            )
            self._append_event_locked(
                row["plan_id"], row["state"], PlanState.OUTCOME_INDETERMINATE.value,
                "crash_recovery", {"requires_reconciliation": True}, now,
            )
        self._db.commit()

    def create(self, plan: PreparedPlan, execution_nonce: str) -> None:
        plan.verify_integrity()
        now = utc_now()
        with self._lock:
            self._db.execute(
                """INSERT INTO plans
                   (plan_id, plan_hash, plan_json, state, nonce_hash, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    plan.plan_id,
                    plan.plan_hash,
                    canonical_json(plan.to_dict()),
                    plan.state.value,
                    _nonce_hash(execution_nonce),
                    plan.created_at,
                    now,
                ),
            )
            self._append_event_locked(
                plan.plan_id, None, plan.state.value, "plan_created",
                {
                    "plan_hash": plan.plan_hash,
                    "targets": list(plan.targets),
                    "l0_skill_id": plan.l0_skill_id,
                    "intent_hash": plan.intent_hash,
                }, now,
            )
            self._db.commit()

    def append_event(
        self, plan_id: str, event_type: str, payload: dict[str, Any] | None = None,
    ) -> None:
        """Append a hash-chained state-preserving runtime/skill event."""
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                plan = self.get(plan_id)
                self._append_event_locked(
                    plan_id, plan.state.value, plan.state.value,
                    event_type, payload or {}, utc_now(),
                )
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise

    def get(self, plan_id: str) -> PreparedPlan:
        row = self._db.execute(
            "SELECT plan_json, state FROM plans WHERE plan_id=?", (plan_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown network plan {plan_id}")
        value = json.loads(row["plan_json"])
        value["state"] = row["state"]
        return PreparedPlan.from_dict(value)

    def claim_for_execution(
        self,
        *,
        plan_id: str,
        plan_hash: str,
        execution_nonce: str,
        approval_request_id: str,
        approval_actor: str,
    ) -> PreparedPlan:
        if not approval_request_id.strip() or not approval_actor.strip():
            raise ApprovalError("approval request and actor are required")
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute("SELECT * FROM plans WHERE plan_id=?", (plan_id,)).fetchone()
                if row is None:
                    raise KeyError(f"unknown network plan {plan_id}")
                if row["plan_hash"] != plan_hash:
                    raise PlanIntegrityError("approved plan hash does not match stored plan")
                if row["nonce_hash"] != _nonce_hash(execution_nonce) or row["nonce_consumed_at"] is not None:
                    raise ApprovalError("execution nonce is invalid or already consumed")
                plan = self.get(plan_id)
                plan.verify_integrity()
                if plan.state != PlanState.PLAN_READY:
                    raise StateTransitionError(f"plan {plan_id} is {plan.state}, not plan_ready")
                if datetime.fromisoformat(plan.expires_at) <= datetime.now(timezone.utc):
                    self._transition_locked(plan, PlanState.EXPIRED, "plan_expired", {})
                    self._db.commit()
                    raise ApprovalError("network plan expired before execution")
                for target in plan.targets:
                    existing = self._db.execute(
                        "SELECT plan_id FROM resource_locks WHERE resource_key=?", (target,),
                    ).fetchone()
                    if existing is not None and existing["plan_id"] != plan_id:
                        raise ResourceConflictError(
                            f"target {target} is locked by plan {existing['plan_id']}"
                        )
                    self._db.execute(
                        "INSERT OR IGNORE INTO resource_locks(resource_key, plan_id, acquired_at) VALUES (?, ?, ?)",
                        (target, plan_id, utc_now()),
                    )
                now = utc_now()
                self._db.execute(
                    """UPDATE plans SET nonce_consumed_at=?, approval_request_id=?,
                       approval_actor=?, approved_at=?, updated_at=? WHERE plan_id=?""",
                    (now, approval_request_id, approval_actor, now, now, plan_id),
                )
                plan = self._transition_locked(plan, PlanState.APPROVED, "approval_bound", {
                    "approval_request_id": approval_request_id,
                    "approval_actor": approval_actor,
                    "plan_hash": plan_hash,
                })
                plan = self._transition_locked(plan, PlanState.EXECUTING, "execution_started", {})
                self._db.commit()
                return plan
            except Exception:
                self._db.rollback()
                raise

    def transition(
        self,
        plan_id: str,
        to_state: PlanState,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> PreparedPlan:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                plan = self.get(plan_id)
                updated = self._transition_locked(plan, to_state, event_type, payload or {})
                self._db.commit()
                return updated
            except Exception:
                self._db.rollback()
                raise

    def _transition_locked(
        self,
        plan: PreparedPlan,
        to_state: PlanState,
        event_type: str,
        payload: dict[str, Any],
    ) -> PreparedPlan:
        if plan.state in TERMINAL_STATES:
            raise StateTransitionError(f"terminal plan {plan.plan_id} cannot transition from {plan.state}")
        if to_state not in ALLOWED_TRANSITIONS.get(plan.state, set()):
            raise StateTransitionError(f"illegal transition {plan.state} -> {to_state}")
        now = utc_now()
        updated = replace(plan, state=to_state)
        self._db.execute(
            "UPDATE plans SET state=?, plan_json=?, updated_at=? WHERE plan_id=?",
            (to_state.value, canonical_json(updated.to_dict()), now, plan.plan_id),
        )
        self._append_event_locked(
            plan.plan_id, plan.state.value, to_state.value, event_type, payload, now,
        )
        return updated

    def store_outcome(self, plan_id: str, value: dict[str, Any], error: str | None = None) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE plans SET result_json=?, error_text=?, updated_at=? WHERE plan_id=?",
                (canonical_json(value), error, utc_now(), plan_id),
            )
            self._db.commit()

    def release_locks(self, plan_id: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM resource_locks WHERE plan_id=?", (plan_id,))
            self._db.commit()

    def events(self, plan_id: str) -> list[dict[str, Any]]:
        rows = self._db.execute(
            """SELECT event_id, from_state, to_state, event_type, payload_json,
                      prev_event_hash, event_hash, created_at
               FROM plan_events WHERE plan_id=? ORDER BY event_id""",
            (plan_id,),
        ).fetchall()
        return [
            {
                "event_id": row["event_id"],
                "from_state": row["from_state"],
                "to_state": row["to_state"],
                "event_type": row["event_type"],
                "payload": json.loads(row["payload_json"]),
                "prev_event_hash": row["prev_event_hash"],
                "event_hash": row["event_hash"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def record(self, plan_id: str) -> dict[str, Any]:
        row = self._db.execute(
            """SELECT state, approval_request_id, approval_actor, approved_at,
                      nonce_consumed_at, result_json, error_text, created_at, updated_at
               FROM plans WHERE plan_id=?""",
            (plan_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown network plan {plan_id}")
        return {
            "state": row["state"],
            "approval_request_id": row["approval_request_id"],
            "approval_actor": row["approval_actor"],
            "approved_at": row["approved_at"],
            "nonce_consumed": row["nonce_consumed_at"] is not None,
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "error": row["error_text"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def plan_ids_in_state(self, state: PlanState) -> list[str]:
        rows = self._db.execute(
            "SELECT plan_id FROM plans WHERE state=? ORDER BY created_at",
            (state.value,),
        ).fetchall()
        return [str(row["plan_id"]) for row in rows]

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        bounded = max(1, min(int(limit), 200))
        rows = self._db.execute(
            """SELECT plan_id, plan_hash, plan_json, state, created_at, updated_at,
                      approval_request_id, approval_actor, error_text
               FROM plans ORDER BY created_at DESC LIMIT ?""",
            (bounded,),
        ).fetchall()
        values: list[dict[str, Any]] = []
        for row in rows:
            plan = json.loads(row["plan_json"])
            values.append({
                "plan_id": row["plan_id"],
                "plan_hash": row["plan_hash"],
                "profile": plan.get("profile"),
                "tool_name": plan.get("tool_name"),
                "targets": plan.get("targets", []),
                "risk_level": plan.get("risk_level"),
                "state": row["state"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "approval_request_id": row["approval_request_id"],
                "approval_actor": row["approval_actor"],
                "error": row["error_text"],
            })
        return values

    def verify_event_chain(self, plan_id: str) -> dict[str, Any]:
        if self._db.execute(
            "SELECT 1 FROM plans WHERE plan_id=?", (plan_id,),
        ).fetchone() is None:
            raise KeyError(f"unknown network plan {plan_id}")
        rows = self._db.execute(
            """SELECT event_id, plan_id, from_state, to_state, event_type,
                      payload_json, prev_event_hash, event_hash, created_at
               FROM plan_events WHERE plan_id=? ORDER BY event_id""",
            (plan_id,),
        ).fetchall()
        expected_previous = GENESIS_EVENT_HASH
        errors: list[dict[str, Any]] = []
        for row in rows:
            actual_previous = row["prev_event_hash"]
            expected_hash = self._event_hash(
                plan_id=str(row["plan_id"]),
                from_state=row["from_state"],
                to_state=str(row["to_state"]),
                event_type=str(row["event_type"]),
                payload_json=str(row["payload_json"]),
                created_at=str(row["created_at"]),
                prev_event_hash=expected_previous,
            )
            if actual_previous != expected_previous:
                errors.append({
                    "event_id": row["event_id"],
                    "error": "previous_hash_mismatch",
                    "expected": expected_previous,
                    "actual": actual_previous,
                })
            if row["event_hash"] != expected_hash:
                errors.append({
                    "event_id": row["event_id"],
                    "error": "event_hash_mismatch",
                    "expected": expected_hash,
                    "actual": row["event_hash"],
                })
            expected_previous = str(row["event_hash"] or "")
        return {
            "ok": not errors,
            "plan_id": plan_id,
            "events": len(rows),
            "head_hash": expected_previous if rows else GENESIS_EVENT_HASH,
            "errors": errors,
        }

    def _append_event_locked(
        self,
        plan_id: str,
        from_state: str | None,
        to_state: str,
        event_type: str,
        payload: dict[str, Any],
        created_at: str,
    ) -> None:
        previous = self._db.execute(
            "SELECT event_hash FROM plan_events WHERE plan_id=? ORDER BY event_id DESC LIMIT 1",
            (plan_id,),
        ).fetchone()
        prev_event_hash = (
            str(previous["event_hash"])
            if previous is not None and previous["event_hash"]
            else GENESIS_EVENT_HASH
        )
        payload_json = canonical_json(payload)
        event_hash = self._event_hash(
            plan_id=plan_id,
            from_state=from_state,
            to_state=to_state,
            event_type=event_type,
            payload_json=payload_json,
            created_at=created_at,
            prev_event_hash=prev_event_hash,
        )
        self._db.execute(
            """INSERT INTO plan_events
               (plan_id, from_state, to_state, event_type, payload_json,
                prev_event_hash, event_hash, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                plan_id, from_state, to_state, event_type, payload_json,
                prev_event_hash, event_hash, created_at,
            ),
        )
