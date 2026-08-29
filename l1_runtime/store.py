"""Privacy-minimized storage for immutable Decisions and terminal lifecycle state."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from network_runtime.contracts import sha256_json

from .contracts import L1DecisionEnvelope


SCHEMA_VERSION = 3
_CLOSE_REASONS = {
    "no_domain_route", "observation_error", "superseded", "session_end",
}


class DecisionStore:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        migrated_private_values = False
        with self._connect() as database:
            database.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS decisions (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                    decision_id TEXT NOT NULL UNIQUE,
                    session_id TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    harness TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    action TEXT,
                    target TEXT,
                    prompt_digest TEXT NOT NULL,
                    decision_digest TEXT,
                    evidence_digest TEXT NOT NULL,
                    duration_ms REAL NOT NULL,
                    expected_arguments_digest TEXT,
                    expected_argument_keys_json TEXT,
                    lifecycle_status TEXT NOT NULL DEFAULT 'pending',
                    lifecycle_reason TEXT,
                    envelope_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_l1_decisions_session_sequence
                    ON decisions(session_id, sequence DESC);
                CREATE INDEX IF NOT EXISTS idx_l1_decisions_status_sequence
                    ON decisions(status, sequence DESC);
                CREATE TABLE IF NOT EXISTS observations (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                    decision_id TEXT NOT NULL UNIQUE,
                    session_id TEXT NOT NULL,
                    observed_kind TEXT NOT NULL,
                    observed_target TEXT NOT NULL,
                    observed_arguments_digest TEXT NOT NULL,
                    observed_argument_keys_json TEXT NOT NULL,
                    target_match INTEGER NOT NULL,
                    arguments_exact INTEGER,
                    safety_escape INTEGER NOT NULL,
                    outcome TEXT NOT NULL,
                    FOREIGN KEY(decision_id) REFERENCES decisions(decision_id)
                );
                CREATE INDEX IF NOT EXISTS idx_l1_observations_session_sequence
                    ON observations(session_id, sequence DESC);
            """)
            columns = {
                str(row["name"])
                for row in database.execute("PRAGMA table_info(decisions)").fetchall()
            }
            if "expected_arguments_digest" not in columns:
                database.execute(
                    "ALTER TABLE decisions ADD COLUMN expected_arguments_digest TEXT",
                )
            if "expected_argument_keys_json" not in columns:
                database.execute(
                    "ALTER TABLE decisions ADD COLUMN expected_argument_keys_json TEXT",
                )
            if "lifecycle_status" not in columns:
                database.execute(
                    """
                    ALTER TABLE decisions ADD COLUMN lifecycle_status TEXT
                    NOT NULL DEFAULT 'pending'
                    """,
                )
            if "lifecycle_reason" not in columns:
                database.execute(
                    "ALTER TABLE decisions ADD COLUMN lifecycle_reason TEXT",
                )
            legacy_rows = database.execute(
                """
                SELECT sequence, envelope_json
                FROM decisions
                WHERE expected_arguments_digest IS NULL
                   OR expected_argument_keys_json IS NULL
                """,
            ).fetchall()
            for row in legacy_rows:
                payload = json.loads(str(row["envelope_json"]))
                arguments = self._decision_arguments(payload)
                database.execute(
                    """
                    UPDATE decisions
                    SET expected_arguments_digest = ?,
                        expected_argument_keys_json = ?,
                        envelope_json = ?
                    WHERE sequence = ?
                    """,
                    (
                        sha256_json(arguments),
                        json.dumps(sorted(arguments), separators=(",", ":")),
                        json.dumps(
                            self._redacted_record(payload),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        int(row["sequence"]),
                    ),
                )
                migrated_private_values = True
            database.execute(
                """
                INSERT INTO metadata(key, value) VALUES ('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(SCHEMA_VERSION),),
            )
        if migrated_private_values:
            with self._connect() as database:
                database.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                database.execute("VACUUM")

    @staticmethod
    def _decision_arguments(payload: dict[str, Any]) -> dict[str, Any]:
        decision = payload.get("decision")
        if not isinstance(decision, dict):
            return {}
        arguments = decision.get("arguments")
        return arguments if isinstance(arguments, dict) else {}

    @classmethod
    def _redacted_record(cls, payload: dict[str, Any]) -> dict[str, Any]:
        record = json.loads(json.dumps(payload, ensure_ascii=False))
        decision = record.get("decision")
        if isinstance(decision, dict):
            arguments = cls._decision_arguments(payload)
            decision.pop("arguments", None)
            decision["argument_keys"] = sorted(arguments)
        record["privacy"] = "prompt_digest_and_argument_keys_only"
        return record

    def _connect(self) -> sqlite3.Connection:
        database = sqlite3.connect(self.path, timeout=5.0)
        database.row_factory = sqlite3.Row
        database.execute("PRAGMA journal_mode=WAL")
        database.execute("PRAGMA synchronous=FULL")
        database.execute("PRAGMA busy_timeout=5000")
        return database

    def record(self, envelope: L1DecisionEnvelope) -> None:
        payload = envelope.model_dump(by_alias=True, mode="json")
        decision = envelope.decision
        arguments = decision.arguments if decision is not None else {}
        record = self._redacted_record(payload)
        with self._connect() as database:
            database.execute(
                """
                INSERT INTO decisions(
                    decision_id, session_id, profile, harness, mode, status,
                    action, target, prompt_digest, decision_digest,
                    evidence_digest, duration_ms, expected_arguments_digest,
                    expected_argument_keys_json, envelope_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope.decision_id,
                    envelope.session_id,
                    envelope.profile,
                    envelope.harness,
                    envelope.mode,
                    envelope.status,
                    decision.action.value if decision is not None else None,
                    decision.target if decision is not None else None,
                    envelope.evidence.prompt_digest,
                    envelope.decision_digest,
                    envelope.evidence_digest,
                    envelope.evidence.duration_ms,
                    sha256_json(arguments),
                    json.dumps(sorted(arguments), separators=(",", ":")),
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")),
                ),
            )

    def recent(
        self, *, limit: int = 20, session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if not 1 <= limit <= 500:
            raise ValueError("production L1 recent decision limit must be 1..500")
        query = (
            "SELECT sequence, created_at, lifecycle_status, lifecycle_reason, "
            "envelope_json FROM decisions"
        )
        arguments: tuple[Any, ...]
        if session_id is None:
            query += " ORDER BY sequence DESC LIMIT ?"
            arguments = (limit,)
        else:
            query += " WHERE session_id = ? ORDER BY sequence DESC LIMIT ?"
            arguments = (session_id, limit)
        with self._connect() as database:
            rows = database.execute(query, arguments).fetchall()
            decision_ids = [
                json.loads(str(row["envelope_json"]))["decision_id"] for row in rows
            ]
            observations: dict[str, dict[str, Any]] = {}
            if decision_ids:
                placeholders = ",".join("?" for _ in decision_ids)
                observed_rows = database.execute(
                    f"SELECT * FROM observations WHERE decision_id IN ({placeholders})",
                    tuple(decision_ids),
                ).fetchall()
                observations = {
                    str(item["decision_id"]): self._observation_dict(item)
                    for item in observed_rows
                }
        output: list[dict[str, Any]] = []
        for row in rows:
            envelope = json.loads(str(row["envelope_json"]))
            output.append({
                "sequence": int(row["sequence"]),
                "created_at": str(row["created_at"]),
                "lifecycle_status": str(row["lifecycle_status"]),
                "lifecycle_reason": (
                    None if row["lifecycle_reason"] is None
                    else str(row["lifecycle_reason"])
                ),
                "envelope": envelope,
                "observation": observations.get(envelope["decision_id"]),
            })
        return output

    @staticmethod
    def _observation_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "sequence": int(row["sequence"]),
            "created_at": str(row["created_at"]),
            "decision_id": str(row["decision_id"]),
            "session_id": str(row["session_id"]),
            "observed_kind": str(row["observed_kind"]),
            "observed_target": str(row["observed_target"]),
            "observed_arguments_digest": str(row["observed_arguments_digest"]),
            "observed_argument_keys": json.loads(str(row["observed_argument_keys_json"])),
            "target_match": bool(row["target_match"]),
            "arguments_exact": (
                None if row["arguments_exact"] is None else bool(row["arguments_exact"])
            ),
            "safety_escape": bool(row["safety_escape"]),
            "outcome": str(row["outcome"]),
        }

    def observe(
        self,
        *,
        decision_id: str,
        session_id: str,
        observed_kind: str,
        observed_target: str,
        observed_arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if observed_kind not in {"skill", "tool"}:
            raise ValueError("production L1 observed kind must be skill or tool")
        if not observed_target.strip() or len(observed_target) > 128:
            raise ValueError("production L1 observed target is invalid")
        with self._connect() as database:
            row = database.execute(
                """
                SELECT session_id, action, target, expected_arguments_digest,
                       lifecycle_status
                FROM decisions WHERE decision_id = ?
                """,
                (decision_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown L1 decision {decision_id}")
            if str(row["session_id"]) != session_id:
                raise PermissionError("L1 observation session binding mismatch")
            lifecycle_status = str(row["lifecycle_status"])
            if lifecycle_status == "observed":
                observed = database.execute(
                    "SELECT * FROM observations WHERE decision_id = ?", (decision_id,),
                ).fetchone()
                if observed is None:
                    raise RuntimeError("observed L1 decision has no route evidence")
                return self._observation_dict(observed)
            if lifecycle_status != "pending":
                raise RuntimeError("closed L1 decision cannot accept a route observation")
            action = None if row["action"] is None else str(row["action"])
            selection = action in {
                "select_skill", "select_tool",
            }
            expected_kind = (
                action.removeprefix("select_") if selection and action is not None else None
            )
            target_match = bool(
                selection
                and expected_kind == observed_kind
                and str(row["target"]) == observed_target
            )
            safety_escape = not selection
            arguments_exact: bool | None
            if observed_kind == "tool" and selection:
                arguments_exact = (
                    str(row["expected_arguments_digest"])
                    == sha256_json(observed_arguments)
                )
            else:
                arguments_exact = None
            outcome = (
                "safety_escape" if safety_escape
                else "matched" if target_match
                else "mismatched"
            )
            database.execute(
                """
                INSERT INTO observations(
                    decision_id, session_id, observed_kind, observed_target,
                    observed_arguments_digest, observed_argument_keys_json,
                    target_match, arguments_exact, safety_escape, outcome
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_id,
                    session_id,
                    observed_kind,
                    observed_target,
                    sha256_json(observed_arguments),
                    json.dumps(sorted(observed_arguments), separators=(",", ":")),
                    int(target_match),
                    None if arguments_exact is None else int(arguments_exact),
                    int(safety_escape),
                    outcome,
                ),
            )
            lifecycle_update = database.execute(
                """
                UPDATE decisions
                SET lifecycle_status = 'observed', lifecycle_reason = 'domain_route'
                WHERE decision_id = ? AND lifecycle_status = 'pending'
                """,
                (decision_id,),
            )
            if lifecycle_update.rowcount != 1:
                raise RuntimeError("L1 route observation lost its pending lifecycle binding")
            observed = database.execute(
                "SELECT * FROM observations WHERE decision_id = ?", (decision_id,),
            ).fetchone()
        if observed is None:
            raise RuntimeError("production L1 observation was not persisted")
        return self._observation_dict(observed)

    def close(
        self, *, decision_id: str, session_id: str, reason: str,
    ) -> dict[str, Any]:
        if reason not in _CLOSE_REASONS:
            raise ValueError("production L1 close reason is not allowed")
        with self._connect() as database:
            row = database.execute(
                """
                SELECT session_id, lifecycle_status, lifecycle_reason
                FROM decisions WHERE decision_id = ?
                """,
                (decision_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown L1 decision {decision_id}")
            if str(row["session_id"]) != session_id:
                raise PermissionError("L1 close session binding mismatch")
            status = str(row["lifecycle_status"])
            lifecycle_reason = (
                None if row["lifecycle_reason"] is None
                else str(row["lifecycle_reason"])
            )
            if status == "pending":
                update = database.execute(
                    """
                    UPDATE decisions
                    SET lifecycle_status = 'closed', lifecycle_reason = ?
                    WHERE decision_id = ? AND lifecycle_status = 'pending'
                    """,
                    (reason, decision_id),
                )
                if update.rowcount == 1:
                    status = "closed"
                    lifecycle_reason = reason
                else:
                    current = database.execute(
                        """
                        SELECT lifecycle_status, lifecycle_reason
                        FROM decisions WHERE decision_id = ?
                        """,
                        (decision_id,),
                    ).fetchone()
                    if current is None:
                        raise RuntimeError("L1 decision disappeared during close")
                    status = str(current["lifecycle_status"])
                    lifecycle_reason = (
                        None if current["lifecycle_reason"] is None
                        else str(current["lifecycle_reason"])
                    )
            if status == "closed" and lifecycle_reason != reason:
                raise RuntimeError("L1 decision was already closed for another reason")
        return {
            "apiVersion": "netopyu.io/l1-decision-lifecycle/v1",
            "decision_id": decision_id,
            "session_id": session_id,
            "status": status,
            "reason": reason if status == "closed" else "domain_route",
        }


__all__ = ["DecisionStore", "SCHEMA_VERSION"]
