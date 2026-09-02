"""Deterministic, persistent mock implementation of enterprise IT systems.

The store deliberately models business desired state only.  Containerlab owns
observed network state and enforcement.  This separation lets cross-layer
workflows detect drift instead of reading the same mock value twice.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_store_path() -> Path:
    configured = os.environ.get("NETOPYU_SERVICE_STORE")
    return Path(configured).expanduser() if configured else Path("data/service_layer.sqlite")


class ServiceStoreError(RuntimeError):
    """Typed business-system failure returned by MCP as a tool error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ServiceStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or default_store_path()).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # The official MCP server may dispatch synchronous tools concurrently.
        # One process owns one connection, so serialize access locally while
        # SQLite BEGIN IMMEDIATE serializes writes across MCP processes.
        self._lock = threading.RLock()
        self.db = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self._initialize()
        os.chmod(self.path, 0o600)

    def close(self) -> None:
        with self._lock:
            self.db.close()

    def __enter__(self) -> "ServiceStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield self.db
                self.db.commit()
            except Exception:
                self.db.rollback()
                raise

    def _initialize(self) -> None:
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS store_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                department TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('active','suspended','terminated'))
            );
            CREATE TABLE IF NOT EXISTS applications (
                app_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                address TEXT NOT NULL,
                port INTEGER NOT NULL,
                path TEXT NOT NULL,
                owner TEXT NOT NULL,
                tier TEXT NOT NULL,
                valid_roles_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS entitlements (
                user_id TEXT NOT NULL,
                app_id TEXT NOT NULL,
                role TEXT NOT NULL,
                PRIMARY KEY(user_id, app_id, role),
                FOREIGN KEY(user_id) REFERENCES users(user_id),
                FOREIGN KEY(app_id) REFERENCES applications(app_id)
            );
            CREATE TABLE IF NOT EXISTS entitlement_revisions (
                user_id TEXT NOT NULL,
                app_id TEXT NOT NULL,
                revision INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(user_id, app_id),
                FOREIGN KEY(user_id) REFERENCES users(user_id),
                FOREIGN KEY(app_id) REFERENCES applications(app_id)
            );
            CREATE TABLE IF NOT EXISTS changes (
                change_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                window_open INTEGER NOT NULL,
                risk TEXT NOT NULL,
                approved_by TEXT
            );
            CREATE TABLE IF NOT EXISTS endpoint_bindings (
                subject_type TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                endpoint TEXT,
                address TEXT,
                PRIMARY KEY(subject_type, subject_id)
            );
            CREATE TABLE IF NOT EXISTS services (
                service TEXT NOT NULL,
                environment TEXT NOT NULL,
                status TEXT NOT NULL,
                version TEXT NOT NULL,
                replicas_ready INTEGER NOT NULL,
                replicas_desired INTEGER NOT NULL,
                revision INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(service, environment)
            );
            CREATE TABLE IF NOT EXISTS operations (
                operation TEXT NOT NULL,
                target_id TEXT NOT NULL,
                state_json TEXT NOT NULL,
                revision INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(operation, target_id)
            );
            CREATE TABLE IF NOT EXISTS idempotency (
                idempotency_key TEXT PRIMARY KEY,
                operation TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audit_log (
                audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                correlation_id TEXT NOT NULL,
                change_id TEXT,
                operation TEXT NOT NULL,
                target TEXT NOT NULL,
                before_json TEXT NOT NULL,
                after_json TEXT NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
        """)
        users = [
            ("alice", "Alice Chen", "sales", "active"),
            ("bob", "Bob Chen", "support", "active"),
            ("carol", "Carol Li", "engineering", "active"),
            ("dave", "Dave Okafor", "finance", "active"),
            ("erin", "Erin Zhou", "sales", "active"),
            ("guest", "Guest Device", "guest", "active"),
        ]
        apps = [
            ("crm", "Enterprise CRM", "10.20.10.20", 8080, "/health", "sales-platform", "gold", ["sales-rep", "sales-admin"]),
            ("wiki", "Corporate Wiki", "10.20.20.20", 8080, "/health", "collaboration", "silver", ["reader", "editor"]),
            ("monitoring", "Network Monitoring", "10.20.30.20", 8080, "/health", "network-operations", "gold", ["noc-operator"]),
            ("portal", "Public Service Portal", "10.30.10.20", 8080, "/health", "digital-channel", "silver", ["visitor"]),
        ]
        with self.transaction() as db:
            # Seed exactly once per database. Replaying INSERT OR IGNORE on
            # every MCP process start would resurrect deliberately revoked
            # entitlements while leaving their revision counters advanced.
            seeded = db.execute(
                "SELECT value FROM store_metadata WHERE key='seed_version'",
            ).fetchone()
            if seeded is not None:
                return
            db.executemany("INSERT OR IGNORE INTO users VALUES (?,?,?,?)", users)
            db.executemany(
                "INSERT OR IGNORE INTO applications VALUES (?,?,?,?,?,?,?,?)",
                [(*row[:-1], json.dumps(row[-1], sort_keys=True)) for row in apps],
            )
            seed_roles = [
                ("bob", "crm", "sales-rep"),
                ("alice", "wiki", "reader"),
                ("bob", "wiki", "reader"),
                ("carol", "wiki", "reader"),
                ("dave", "monitoring", "noc-operator"),
                ("guest", "portal", "visitor"),
            ]
            db.executemany("INSERT OR IGNORE INTO entitlements VALUES (?,?,?)", seed_roles)
            for user_id, _, _, _ in users:
                for app_id, *_ in apps:
                    db.execute(
                        "INSERT OR IGNORE INTO entitlement_revisions VALUES (?,?,0)",
                        (user_id, app_id),
                    )
            db.executemany(
                "INSERT OR IGNORE INTO changes VALUES (?,?,?,?,?)",
                [
                    ("CHG-1001", "approved", 1, "medium", "local-operator"),
                    ("CHG-1002", "pending", 1, "medium", None),
                    ("CHG-1003", "approved", 0, "high", "local-operator"),
                ],
            )
            bindings = [
                ("user", "bob", "bob-client", "10.10.30.10"),
                ("user", "carol", "carol-client", "10.10.50.10"),
                ("user", "erin", "erin-client", "10.10.20.10"),
                ("user", "guest", "guest-client", "10.10.40.10"),
                ("application", "crm", "crm-server", "10.20.10.20"),
                ("application", "wiki", "wiki-server", "10.20.20.20"),
                ("application", "monitoring", "infra-server", "10.20.30.20"),
                ("application", "portal", "public-web", "10.30.10.20"),
            ]
            db.executemany("INSERT OR IGNORE INTO endpoint_bindings VALUES (?,?,?,?)", bindings)
            for service, version in (("crm", "4.2.0"), ("wiki", "2.8.1"), ("monitoring", "1.9.0")):
                for environment in ("prod", "staging", "dev"):
                    db.execute(
                        "INSERT OR IGNORE INTO services VALUES (?,?,?,?,?,?,0)",
                        (service, environment, "healthy", version, 3, 3),
                    )
            db.execute(
                "INSERT INTO store_metadata(key,value) VALUES ('seed_version','1')",
            )

    @staticmethod
    def correlation_id(value: str | None = None) -> str:
        candidate = str(value or "").strip()
        return candidate or f"corr-{uuid.uuid4().hex}"

    def list_users(self, department: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if department:
                rows = self.db.execute(
                    "SELECT * FROM users WHERE department=? ORDER BY user_id", (department.lower(),),
                ).fetchall()
            else:
                rows = self.db.execute("SELECT * FROM users ORDER BY user_id").fetchall()
            return [dict(row) for row in rows]

    def get_user(self, user_id: str) -> dict[str, Any]:
        with self._lock:
            row = self.db.execute("SELECT * FROM users WHERE user_id=?", (user_id.lower(),)).fetchone()
            if row is None:
                raise ServiceStoreError("unknown_user", f"unknown user {user_id!r}")
            return dict(row)

    def list_applications(self, tier: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if tier:
                rows = self.db.execute(
                    "SELECT * FROM applications WHERE tier=? ORDER BY app_id", (tier.lower(),),
                ).fetchall()
            else:
                rows = self.db.execute("SELECT * FROM applications ORDER BY app_id").fetchall()
            return [self._application(row) for row in rows]

    def get_application(self, app_id: str) -> dict[str, Any]:
        with self._lock:
            row = self.db.execute("SELECT * FROM applications WHERE app_id=?", (app_id.lower(),)).fetchone()
            if row is None:
                raise ServiceStoreError("unknown_application", f"unknown application {app_id!r}")
            return self._application(row)

    @staticmethod
    def _application(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["valid_roles"] = json.loads(value.pop("valid_roles_json"))
        return value

    def entitlement(self, user_id: str, app_id: str) -> dict[str, Any]:
        with self._lock:
            user_id, app_id = user_id.lower(), app_id.lower()
            self.get_user(user_id)
            self.get_application(app_id)
            roles = [
                str(row["role"])
                for row in self.db.execute(
                    "SELECT role FROM entitlements WHERE user_id=? AND app_id=? ORDER BY role",
                    (user_id, app_id),
                ).fetchall()
            ]
            row = self.db.execute(
                "SELECT revision FROM entitlement_revisions WHERE user_id=? AND app_id=?",
                (user_id, app_id),
            ).fetchone()
            return {
                "user_id": user_id,
                "app_id": app_id,
                "roles": roles,
                "allowed": bool(roles),
                "revision": int(row["revision"]),
            }

    def evaluate_policy(self, user_id: str, app_id: str) -> dict[str, Any]:
        user = self.get_user(user_id)
        app = self.get_application(app_id)
        current = self.entitlement(user_id, app_id)
        reasons: list[str] = []
        eligible = user["status"] == "active" and user["department"] != "guest"
        if user["status"] != "active":
            reasons.append(f"user status is {user['status']}")
        if user["department"] == "guest" and app["app_id"] != "portal":
            eligible = False
            reasons.append("guest identities are restricted to the public portal")
        if eligible:
            reasons.append("identity is active and satisfies the deterministic local policy")
        return {
            "user_id": user["user_id"],
            "app_id": app["app_id"],
            "eligible": eligible,
            "reasons": reasons,
            "current_roles": current["roles"],
            "recommended_role": app["valid_roles"][0] if eligible else None,
        }

    def validate_change(self, change_id: str) -> dict[str, Any]:
        with self._lock:
            row = self.db.execute("SELECT * FROM changes WHERE change_id=?", (change_id,)).fetchone()
            if row is None:
                raise ServiceStoreError("unknown_change", f"unknown change {change_id!r}")
            value = dict(row)
            value["window_open"] = bool(value["window_open"])
            reasons: list[str] = []
            if value["status"] != "approved":
                reasons.append(f"change status is {value['status']}")
            if not value["window_open"]:
                reasons.append("change window is closed")
            value["permitted"] = not reasons
            value["reasons"] = reasons or ["approved change and open execution window"]
            return value

    def change(self, change_id: str) -> dict[str, Any]:
        value = self.validate_change(change_id)
        value.pop("permitted")
        value.pop("reasons")
        return value

    def _require_change(self, change_id: str) -> None:
        check = self.validate_change(change_id)
        if not check["permitted"]:
            raise ServiceStoreError("change_not_permitted", "; ".join(check["reasons"]))

    def mutate_entitlement(
        self,
        *,
        operation: str,
        user_id: str,
        app_id: str,
        role: str | None,
        roles: list[str] | None,
        change_id: str,
        reason: str,
        expected_revision: int,
        correlation_id: str,
    ) -> dict[str, Any]:
        if operation not in {"grant", "revoke", "restore"}:
            raise ServiceStoreError("invalid_operation", f"unsupported entitlement operation {operation}")
        user_id, app_id = user_id.lower(), app_id.lower()
        with self.transaction() as db:
            # All decisions that authorize the mutation are inside the same
            # cross-process write transaction as the mutation itself.
            self._require_change(change_id)
            user = self.get_user(user_id)
            app = self.get_application(app_id)
            if operation == "grant":
                if user["status"] != "active":
                    raise ServiceStoreError(
                        "inactive_user", "entitlements can only be granted to active users",
                    )
                if role not in app["valid_roles"]:
                    raise ServiceStoreError(
                        "invalid_role", f"role must be one of {app['valid_roles']}",
                    )
            if operation == "restore":
                desired_roles = sorted(set(roles if roles is not None else []))
            elif operation == "revoke":
                desired_roles = []
            else:
                desired_roles = [str(role)]
            if operation == "restore" and any(
                item not in app["valid_roles"] for item in desired_roles
            ):
                raise ServiceStoreError(
                    "invalid_role", "restore contains a role outside the application catalog",
                )
            idempotency_key = (
                f"{change_id}:{operation}:{user_id}:{app_id}:"
                f"{','.join(desired_roles)}:{expected_revision}"
            )
            existing = db.execute(
                "SELECT response_json FROM idempotency WHERE idempotency_key=?",
                (idempotency_key,),
            ).fetchone()
            if existing:
                value = json.loads(existing["response_json"])
                current = self.entitlement(user_id, app_id)
                if (
                    current["revision"] != value["revision"]
                    or current["roles"] != value["roles"]
                ):
                    raise ServiceStoreError(
                        "idempotency_conflict",
                        "the original operation succeeded but target state changed later; "
                        "refresh state and prepare a new plan",
                    )
                value["replayed"] = True
                return value
            before = self.entitlement(user_id, app_id)
            if before["revision"] != expected_revision:
                raise ServiceStoreError(
                    "revision_conflict",
                    f"expected revision {expected_revision}, observed {before['revision']}",
                )
            if operation == "grant":
                after_roles = sorted(set(before["roles"] + desired_roles))
            elif operation == "revoke":
                after_roles = []
            else:
                after_roles = desired_roles
            db.execute("DELETE FROM entitlements WHERE user_id=? AND app_id=?", (user_id, app_id))
            db.executemany(
                "INSERT INTO entitlements(user_id,app_id,role) VALUES (?,?,?)",
                [(user_id, app_id, item) for item in after_roles],
            )
            next_revision = before["revision"] + 1
            db.execute(
                "UPDATE entitlement_revisions SET revision=? WHERE user_id=? AND app_id=?",
                (next_revision, user_id, app_id),
            )
            after = {
                "user_id": user_id,
                "app_id": app_id,
                "roles": after_roles,
                "allowed": bool(after_roles),
                "revision": next_revision,
                "operation": operation,
                "change_id": change_id,
                "idempotency_key": idempotency_key,
                "replayed": False,
            }
            db.execute(
                "INSERT INTO idempotency VALUES (?,?,?,?)",
                (idempotency_key, f"entitlement.{operation}", json.dumps(after, sort_keys=True), utc_now()),
            )
            db.execute(
                """INSERT INTO audit_log
                   (correlation_id,change_id,operation,target,before_json,after_json,reason,created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    correlation_id, change_id, f"entitlement.{operation}", f"{user_id}:{app_id}",
                    json.dumps(before, sort_keys=True), json.dumps(after, sort_keys=True), reason, utc_now(),
                ),
            )
        return after

    def endpoint_binding(self, subject_type: str, subject_id: str) -> dict[str, Any]:
        if subject_type not in {"user", "application"}:
            raise ServiceStoreError("invalid_subject_type", "subject_type must be user or application")
        with self._lock:
            row = self.db.execute(
                "SELECT * FROM endpoint_bindings WHERE subject_type=? AND subject_id=?",
                (subject_type, subject_id.lower()),
            ).fetchone()
            if row is None:
                return {
                    "subject_type": subject_type,
                    "subject_id": subject_id.lower(),
                    "endpoint": None,
                    "address": None,
                    "network_target_resolved": False,
                }
            value = dict(row)
            value["network_target_resolved"] = bool(value["endpoint"] and value["address"])
            return value

    def service_health(self, service: str, environment: str) -> dict[str, Any]:
        with self._lock:
            row = self.db.execute(
                "SELECT * FROM services WHERE service=? AND environment=?",
                (service.lower(), environment.lower()),
            ).fetchone()
            if row is None:
                raise ServiceStoreError(
                    "unknown_service", f"unknown service/environment {service}/{environment}",
                )
            return dict(row)

    def mutate_service(
        self,
        *,
        operation: str,
        service: str,
        environment: str,
        version: str | None,
        change_id: str,
        reason: str,
        expected_revision: int,
        correlation_id: str,
    ) -> dict[str, Any]:
        if operation not in {"restart", "rollback", "restore"}:
            raise ServiceStoreError("invalid_operation", f"unsupported service operation {operation}")
        with self.transaction() as db:
            self._require_change(change_id)
            before = self.service_health(service, environment)
            target_version = version or before["version"]
            idempotency_key = (
                f"{change_id}:{operation}:{service}:{environment}:"
                f"{target_version}:{expected_revision}"
            )
            existing = db.execute(
                "SELECT response_json FROM idempotency WHERE idempotency_key=?",
                (idempotency_key,),
            ).fetchone()
            if existing:
                value = json.loads(existing["response_json"])
                if (
                    before["revision"] != value["revision"]
                    or before["version"] != value["version"]
                    or before["status"] != value["status"]
                    or before["replicas_ready"] != value["replicas_ready"]
                    or before["replicas_desired"] != value["replicas_desired"]
                ):
                    raise ServiceStoreError(
                        "idempotency_conflict",
                        "the original operation succeeded but target state changed later; "
                        "refresh state and prepare a new plan",
                    )
                value["replayed"] = True
                return value
            if before["revision"] != expected_revision:
                raise ServiceStoreError(
                    "revision_conflict",
                    f"expected revision {expected_revision}, observed {before['revision']}",
                )
            next_revision = before["revision"] + 1
            after = {
                "service": service.lower(), "environment": environment.lower(),
                "status": "healthy", "version": target_version,
                "replicas_ready": 3, "replicas_desired": 3, "revision": next_revision,
                "operation": operation, "change_id": change_id,
                "idempotency_key": idempotency_key, "replayed": False,
            }
            db.execute(
                """UPDATE services SET status='healthy',version=?,replicas_ready=3,
                   replicas_desired=3,revision=? WHERE service=? AND environment=?""",
                (target_version, next_revision, service.lower(), environment.lower()),
            )
            db.execute(
                "INSERT INTO idempotency VALUES (?,?,?,?)",
                (idempotency_key, f"service.{operation}", json.dumps(after, sort_keys=True), utc_now()),
            )
            db.execute(
                """INSERT INTO audit_log
                   (correlation_id,change_id,operation,target,before_json,after_json,reason,created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    correlation_id, change_id, f"service.{operation}",
                    f"{service.lower()}:{environment.lower()}", json.dumps(before, sort_keys=True),
                    json.dumps(after, sort_keys=True), reason, utc_now(),
                ),
            )
        return after

    def operation_status(self, operation: str, target_id: str) -> dict[str, Any]:
        with self._lock:
            row = self.db.execute(
                "SELECT * FROM operations WHERE operation=? AND target_id=?", (operation, target_id),
            ).fetchone()
            if row is None:
                return {"operation": operation, "target_id": target_id, "state": {}, "revision": 0}
            return {
                "operation": row["operation"], "target_id": row["target_id"],
                "state": json.loads(row["state_json"]), "revision": row["revision"],
            }

    def audit_rows(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self.db.execute("SELECT * FROM audit_log ORDER BY audit_id")]
