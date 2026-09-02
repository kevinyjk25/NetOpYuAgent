"""Durable operation, snapshot, lease, fencing, and audit state for Network Actor."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

from network_runtime.contracts import canonical_json, sha256_json


GENESIS = "GENESIS"
RECOVERABLE_STATES = {"prepared", "executing", "restoring", "outcome_indeterminate"}
ALLOWED_TRANSITIONS = {
    "prepared": {"executing", "not_applied", "manual_intervention"},
    "executing": {
        "applied", "not_applied", "outcome_indeterminate", "restoring",
        "manual_intervention",
    },
    "outcome_indeterminate": {"applied", "not_applied", "restoring", "manual_intervention"},
    "applied": {"restoring", "committed", "manual_intervention"},
    "not_applied": {"restoring", "restored", "manual_intervention"},
    "restoring": {"restored", "manual_intervention"},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ActorStoreError(RuntimeError):
    """A durable Actor invariant was violated."""


class ActorStore:
    def __init__(self, path: str | Path, *, lease_seconds: float = 120.0) -> None:
        if not 10 <= lease_seconds <= 900:
            raise ValueError("actor lease_seconds must be between 10 and 900")
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        self.lock_directory = self.path.parent / "actor-locks"
        self.lock_directory.mkdir(parents=True, exist_ok=True)
        os.chmod(self.lock_directory, 0o700)
        self.lease_seconds = float(lease_seconds)
        self.owner_id = f"pid-{os.getpid()}-{uuid.uuid4()}"
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS actor_operations (
                operation_id TEXT PRIMARY KEY,
                plan_hash TEXT NOT NULL,
                intent_hash TEXT NOT NULL,
                capability_id TEXT NOT NULL,
                capability_version TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                target_key TEXT NOT NULL,
                arguments_json TEXT NOT NULL,
                arguments_digest TEXT NOT NULL,
                preflight_digest TEXT NOT NULL,
                snapshot_json TEXT NOT NULL,
                snapshot_digest TEXT NOT NULL,
                desired_json TEXT NOT NULL,
                state TEXT NOT NULL,
                fence_token INTEGER NOT NULL,
                result_json TEXT,
                error_text TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS actor_target_fences (
                target_key TEXT PRIMARY KEY,
                last_token INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS actor_target_leases (
                target_key TEXT PRIMARY KEY,
                operation_id TEXT NOT NULL,
                fence_token INTEGER NOT NULL,
                owner_id TEXT NOT NULL,
                lease_expires_at REAL NOT NULL,
                FOREIGN KEY(operation_id) REFERENCES actor_operations(operation_id)
            );
            CREATE TABLE IF NOT EXISTS actor_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                from_state TEXT,
                to_state TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                prev_event_hash TEXT NOT NULL,
                event_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(operation_id) REFERENCES actor_operations(operation_id)
            );
            CREATE INDEX IF NOT EXISTS idx_actor_events_operation
                ON actor_events(operation_id, event_id);
        """)
        self._db.commit()
        os.chmod(self.path, 0o600)

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "ActorStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @asynccontextmanager
    async def target_lock(self, target_key: str) -> AsyncIterator[None]:
        digest = hashlib.sha256(target_key.encode("utf-8")).hexdigest()
        path = self.lock_directory / f"{digest}.lock"
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            import asyncio

            await asyncio.to_thread(fcntl.flock, descriptor, fcntl.LOCK_EX)
            yield
        finally:
            await asyncio.to_thread(fcntl.flock, descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    @staticmethod
    def _immutable(operation: dict[str, Any]) -> dict[str, Any]:
        return {
            key: operation[key]
            for key in (
                "operation_id", "plan_hash", "intent_hash", "capability_id",
                "capability_version", "tool_name", "target_key",
                "arguments_digest", "preflight_digest",
            )
        }

    def claim_operation(
        self,
        *,
        operation_id: str,
        plan_hash: str,
        intent_hash: str,
        capability_id: str,
        capability_version: str,
        tool_name: str,
        target_key: str,
        arguments: dict[str, Any],
        approved_preflight: dict[str, Any],
        snapshot: Any,
        desired: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        now_epoch = time.time()
        now = utc_now()
        candidate = {
            "operation_id": operation_id,
            "plan_hash": plan_hash,
            "intent_hash": intent_hash,
            "capability_id": capability_id,
            "capability_version": capability_version,
            "tool_name": tool_name,
            "target_key": target_key,
            "arguments_digest": sha256_json(arguments),
            "preflight_digest": sha256_json(approved_preflight),
        }
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                existing_row = self._db.execute(
                    "SELECT * FROM actor_operations WHERE operation_id=?", (operation_id,),
                ).fetchone()
                existing = self.get(operation_id) if existing_row else None
                if existing is not None and self._immutable(existing) != candidate:
                    raise ActorStoreError("operation id was reused with different immutable content")
                manual = self._db.execute(
                    "SELECT operation_id FROM actor_operations "
                    "WHERE target_key=? AND state='manual_intervention' "
                    "AND operation_id<>? LIMIT 1",
                    (target_key, operation_id),
                ).fetchone()
                if manual is not None:
                    raise ActorStoreError(
                        f"target is quarantined by manual operation {manual['operation_id']}"
                    )
                lease = self._db.execute(
                    "SELECT * FROM actor_target_leases WHERE target_key=?", (target_key,),
                ).fetchone()
                if (
                    lease is not None
                    and str(lease["operation_id"]) != operation_id
                    and float(lease["lease_expires_at"]) > now_epoch
                ):
                    raise ActorStoreError(
                        f"target is leased by operation {lease['operation_id']}"
                    )
                fence_row = self._db.execute(
                    "SELECT last_token FROM actor_target_fences WHERE target_key=?", (target_key,),
                ).fetchone()
                fence = int(fence_row["last_token"] if fence_row else 0) + 1
                self._db.execute(
                    "INSERT INTO actor_target_fences(target_key,last_token) VALUES(?,?) "
                    "ON CONFLICT(target_key) DO UPDATE SET last_token=excluded.last_token",
                    (target_key, fence),
                )
                if existing is None:
                    self._db.execute(
                        """INSERT INTO actor_operations(
                               operation_id,plan_hash,intent_hash,capability_id,capability_version,
                               tool_name,target_key,arguments_json,arguments_digest,preflight_digest,snapshot_json,
                               snapshot_digest,desired_json,state,fence_token,created_at,updated_at
                           ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            operation_id, plan_hash, intent_hash, capability_id,
                            capability_version, tool_name, target_key,
                            canonical_json(arguments), candidate["arguments_digest"],
                            candidate["preflight_digest"],
                            canonical_json(snapshot), sha256_json(snapshot), canonical_json(desired),
                            "prepared", fence, now, now,
                        ),
                    )
                    self._append_event_locked(
                        operation_id, "operation_prepared", None, "prepared",
                        {"target_key": target_key, "fence_token": fence}, now,
                    )
                    replayed = False
                else:
                    self._db.execute(
                        "UPDATE actor_operations SET fence_token=?, updated_at=? WHERE operation_id=?",
                        (fence, now, operation_id),
                    )
                    replayed = True
                self._db.execute(
                    """INSERT INTO actor_target_leases(
                           target_key,operation_id,fence_token,owner_id,lease_expires_at
                       ) VALUES(?,?,?,?,?)
                       ON CONFLICT(target_key) DO UPDATE SET
                           operation_id=excluded.operation_id,
                           fence_token=excluded.fence_token,
                           owner_id=excluded.owner_id,
                           lease_expires_at=excluded.lease_expires_at""",
                    (
                        target_key, operation_id, fence, self.owner_id,
                        now_epoch + self.lease_seconds,
                    ),
                )
                self._append_event_locked(
                    operation_id, "lease_claimed", existing["state"] if existing else "prepared",
                    existing["state"] if existing else "prepared",
                    {"fence_token": fence, "owner_id": self.owner_id}, now,
                )
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
        return self.get(operation_id), replayed

    def claim_for_restore(
        self, *, operation_id: str, plan_hash: str, target_key: str,
    ) -> dict[str, Any]:
        operation = self.get(operation_id)
        if operation["plan_hash"] != plan_hash or operation["target_key"] != target_key:
            raise ActorStoreError("restore context does not match the durable operation")
        now_epoch = time.time()
        now = utc_now()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                lease = self._db.execute(
                    "SELECT * FROM actor_target_leases WHERE target_key=?", (target_key,),
                ).fetchone()
                if (
                    lease is not None
                    and str(lease["operation_id"]) != operation_id
                    and float(lease["lease_expires_at"]) > now_epoch
                ):
                    raise ActorStoreError(
                        f"target is leased by operation {lease['operation_id']}"
                    )
                fence_row = self._db.execute(
                    "SELECT last_token FROM actor_target_fences WHERE target_key=?", (target_key,),
                ).fetchone()
                fence = int(fence_row["last_token"] if fence_row else 0) + 1
                self._db.execute(
                    "INSERT INTO actor_target_fences(target_key,last_token) VALUES(?,?) "
                    "ON CONFLICT(target_key) DO UPDATE SET last_token=excluded.last_token",
                    (target_key, fence),
                )
                self._db.execute(
                    "UPDATE actor_operations SET fence_token=?, updated_at=? WHERE operation_id=?",
                    (fence, now, operation_id),
                )
                self._db.execute(
                    """INSERT INTO actor_target_leases(
                           target_key,operation_id,fence_token,owner_id,lease_expires_at
                       ) VALUES(?,?,?,?,?)
                       ON CONFLICT(target_key) DO UPDATE SET
                           operation_id=excluded.operation_id,
                           fence_token=excluded.fence_token,
                           owner_id=excluded.owner_id,
                           lease_expires_at=excluded.lease_expires_at""",
                    (
                        target_key, operation_id, fence, self.owner_id,
                        now_epoch + self.lease_seconds,
                    ),
                )
                self._append_event_locked(
                    operation_id, "restore_lease_claimed", operation["state"],
                    operation["state"], {"fence_token": fence}, now,
                )
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
        return self.get(operation_id)

    def assert_fence(self, operation_id: str, fence_token: int) -> None:
        operation = self.get(operation_id)
        row = self._db.execute(
            "SELECT operation_id,fence_token,owner_id,lease_expires_at "
            "FROM actor_target_leases WHERE target_key=?",
            (operation["target_key"],),
        ).fetchone()
        if (
            row is None
            or str(row["operation_id"]) != operation_id
            or int(row["fence_token"]) != int(fence_token)
            or str(row["owner_id"]) != self.owner_id
            or float(row["lease_expires_at"]) <= time.time()
        ):
            raise ActorStoreError("stale or expired Actor fencing token")

    def transition(
        self,
        operation_id: str,
        to_state: str,
        event_type: str,
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                before = self.get(operation_id)
                if before["state"] != to_state:
                    allowed = ALLOWED_TRANSITIONS.get(before["state"], set())
                    if to_state not in allowed:
                        raise ActorStoreError(
                            f"illegal Actor transition {before['state']} -> {to_state}"
                        )
                self._db.execute(
                    """UPDATE actor_operations
                       SET state=?,result_json=COALESCE(?,result_json),error_text=?,updated_at=?
                       WHERE operation_id=?""",
                    (
                        to_state,
                        canonical_json(result) if result is not None else None,
                        error,
                        now,
                        operation_id,
                    ),
                )
                self._append_event_locked(
                    operation_id, event_type, before["state"], to_state,
                    payload or {}, now,
                )
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
        return self.get(operation_id)

    def finalize(self, operation_id: str, plan_hash: str, terminal_state: str) -> dict[str, Any]:
        operation = self.get(operation_id)
        if operation["plan_hash"] != plan_hash:
            raise ActorStoreError("finalize plan hash does not match operation")
        if terminal_state == "verified_success":
            if operation["state"] not in {"applied", "committed"}:
                raise ActorStoreError(
                    f"Runtime cannot commit Actor operation in state={operation['state']}"
                )
            actor_state = "committed"
        elif terminal_state == "rollback_verified":
            if operation["state"] not in {"restored", "not_applied"}:
                raise ActorStoreError(
                    f"Runtime cannot certify rollback for Actor state={operation['state']}"
                )
            actor_state = "restored"
        elif terminal_state == "precondition_changed":
            if operation["state"] not in {"prepared", "not_applied"}:
                raise ActorStoreError(
                    f"Runtime precondition result conflicts with Actor state={operation['state']}"
                )
            actor_state = "not_applied"
        else:
            if operation["state"] in {"committed", "restored"}:
                raise ActorStoreError(
                    f"Runtime terminal state {terminal_state!r} conflicts with "
                    f"Actor state={operation['state']}"
                )
            actor_state = "manual_intervention"
        updated = self.transition(
            operation_id, actor_state, "runtime_finalized",
            payload={"runtime_terminal_state": terminal_state},
        )
        if actor_state != "manual_intervention":
            with self._lock:
                self._db.execute(
                    "DELETE FROM actor_target_leases WHERE operation_id=?", (operation_id,),
                )
                self._db.commit()
        return updated

    def find(self, operation_id: str) -> dict[str, Any] | None:
        row = self._db.execute(
            "SELECT * FROM actor_operations WHERE operation_id=?", (operation_id,),
        ).fetchone()
        if row is None:
            return None
        return self.get(operation_id)

    def recoverable(self) -> list[dict[str, Any]]:
        placeholders = ",".join("?" for _ in RECOVERABLE_STATES)
        rows = self._db.execute(
            f"SELECT * FROM actor_operations WHERE state IN ({placeholders}) ORDER BY created_at",
            tuple(sorted(RECOVERABLE_STATES)),
        ).fetchall()
        return [self.get(str(row["operation_id"])) for row in rows]

    def get(self, operation_id: str) -> dict[str, Any]:
        row = self._db.execute(
            "SELECT * FROM actor_operations WHERE operation_id=?", (operation_id,),
        ).fetchone()
        if row is None:
            raise ActorStoreError(f"unknown Actor operation {operation_id}")
        value = self._decode(row)
        if sha256_json(value["arguments"]) != value["arguments_digest"]:
            raise ActorStoreError("durable Actor arguments digest mismatch")
        if sha256_json(value["snapshot"]) != value["snapshot_digest"]:
            raise ActorStoreError("durable Actor snapshot digest mismatch")
        event = self._db.execute(
            "SELECT payload_json FROM actor_events WHERE operation_id=? "
            "ORDER BY event_id DESC LIMIT 1",
            (operation_id,),
        ).fetchone()
        if event is None:
            raise ActorStoreError("durable Actor operation has no audit event")
        event_payload = json.loads(str(event["payload_json"]))
        if event_payload.get("record_digest") != sha256_json(self._record_integrity(value)):
            raise ActorStoreError("durable Actor operation record digest mismatch")
        return value

    def events(self, operation_id: str) -> list[dict[str, Any]]:
        rows = self._db.execute(
            "SELECT * FROM actor_events WHERE operation_id=? ORDER BY event_id",
            (operation_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def verify_event_chain(self, operation_id: str) -> bool:
        previous = GENESIS
        events = self.events(operation_id)
        for row in events:
            if row["prev_event_hash"] != previous:
                return False
            expected = sha256_json({
                "operation_id": operation_id,
                "event_type": row["event_type"],
                "from_state": row["from_state"],
                "to_state": row["to_state"],
                "payload_json": row["payload_json"],
                "created_at": row["created_at"],
                "prev_event_hash": previous,
            })
            if row["event_hash"] != expected:
                return False
            previous = expected
        if not events:
            return False
        operation_row = self._db.execute(
            "SELECT * FROM actor_operations WHERE operation_id=?", (operation_id,),
        ).fetchone()
        if operation_row is None:
            return False
        payload = json.loads(str(events[-1]["payload_json"]))
        return payload.get("record_digest") == sha256_json(
            self._record_integrity(self._decode(operation_row))
        )

    def _append_event_locked(
        self,
        operation_id: str,
        event_type: str,
        from_state: str | None,
        to_state: str,
        payload: dict[str, Any],
        created_at: str,
    ) -> None:
        row = self._db.execute(
            "SELECT event_hash FROM actor_events WHERE operation_id=? ORDER BY event_id DESC LIMIT 1",
            (operation_id,),
        ).fetchone()
        previous = str(row["event_hash"]) if row else GENESIS
        operation_row = self._db.execute(
            "SELECT * FROM actor_operations WHERE operation_id=?", (operation_id,),
        ).fetchone()
        if operation_row is None:  # pragma: no cover - database invariant
            raise ActorStoreError("cannot audit a missing Actor operation")
        operation = self._decode(operation_row)
        payload_json = canonical_json({
            **payload,
            "record_digest": sha256_json(self._record_integrity(operation)),
        })
        event_hash = sha256_json({
            "operation_id": operation_id,
            "event_type": event_type,
            "from_state": from_state,
            "to_state": to_state,
            "payload_json": payload_json,
            "created_at": created_at,
            "prev_event_hash": previous,
        })
        self._db.execute(
            """INSERT INTO actor_events(
                   operation_id,event_type,from_state,to_state,payload_json,
                   prev_event_hash,event_hash,created_at
               ) VALUES(?,?,?,?,?,?,?,?)""",
            (
                operation_id, event_type, from_state, to_state, payload_json,
                previous, event_hash, created_at,
            ),
        )

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["arguments"] = json.loads(value.pop("arguments_json"))
        value["snapshot"] = json.loads(value.pop("snapshot_json"))
        value["desired"] = json.loads(value.pop("desired_json"))
        value["result"] = json.loads(value["result_json"]) if value.get("result_json") else None
        return value

    @staticmethod
    def _record_integrity(operation: dict[str, Any]) -> dict[str, Any]:
        return {
            "operation_id": operation["operation_id"],
            "plan_hash": operation["plan_hash"],
            "intent_hash": operation["intent_hash"],
            "capability_id": operation["capability_id"],
            "capability_version": operation["capability_version"],
            "tool_name": operation["tool_name"],
            "target_key": operation["target_key"],
            "arguments_digest": operation["arguments_digest"],
            "preflight_digest": operation["preflight_digest"],
            "snapshot_digest": operation["snapshot_digest"],
            "desired_digest": sha256_json(operation["desired"]),
            "state": operation["state"],
            "fence_token": operation["fence_token"],
            "result_digest": sha256_json(operation.get("result")),
            "error_digest": sha256_json(operation.get("error_text")),
        }
