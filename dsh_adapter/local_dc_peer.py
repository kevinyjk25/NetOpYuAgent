"""Loopback-only DC A2A peer for the local DSH Web + LLM demo.

It runs the reviewed DC workflow, prepares an immutable Network L0 plan,
pauses for a DSH approval, and executes only the exact resumed plan.  It is a
mock test adapter, never a production or pragmatic-mode peer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sqlite3
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from network_runtime.engine import NetworkRuntime, default_journal_path
from network_runtime.l0_skills import REGISTRY as L0_SKILLS
from network_runtime.workflows import WorkflowRuntime


_USER_RE = re.compile(r"\buser_id\s*=\s*([A-Za-z0-9_.-]+)", re.I)
_APP_RE = re.compile(r"\bapp_id\s*=\s*([A-Za-z0-9_.-]+)", re.I)
_KNOWN_APPS = ("crm", "wiki", "payroll", "grafana")


def _message(text: str) -> dict[str, Any]:
    return {"kind": "message", "message": {"parts": [{"kind": "text", "text": text}]}}


def _failed(text: str) -> dict[str, Any]:
    return {"kind": "taskStatusUpdate", "status": {"state": "failed", "message": text}}


def _input_required(interrupt_id: str, approval: dict[str, Any]) -> dict[str, Any]:
    return {"kind": "taskStatusUpdate", "status": {"state": "input-required", "message": {
        "interrupt_id": interrupt_id, "approval": approval,
    }}}


def _targets(prompt: str) -> tuple[str, str]:
    user = _USER_RE.search(prompt)
    app = _APP_RE.search(prompt)
    user_id = user.group(1).lower().rstrip(".,;:!?") if user else ""
    app_id = app.group(1).lower().rstrip(".,;:!?") if app else ""
    if not user_id:
        fallback = re.search(r"\b(?:user|employee)\s+([A-Za-z0-9_.-]+)", prompt, re.I)
        user_id = fallback.group(1).lower() if fallback else ""
    if not app_id:
        lowered = prompt.lower()
        app_id = next((name for name in _KNOWN_APPS if re.search(rf"\b{name}\b", lowered)), "")
    return user_id, app_id


class LocalDcPeer:
    def __init__(self, *, runtime_path: str | Path, state_path: str | Path) -> None:
        if os.environ.get("NETOPYU_DSH_BACKEND", "mock").strip().lower() != "mock":
            raise RuntimeError("local DC A2A peer is mock-only and refuses pragmatic mode")
        self.runtime_path = Path(runtime_path).expanduser().resolve()
        self.state_path = Path(state_path).expanduser().resolve()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS pending_plans (
                    interrupt_id TEXT PRIMARY KEY,
                    source_session_id TEXT NOT NULL,
                    plan_id TEXT NOT NULL UNIQUE,
                    plan_hash TEXT NOT NULL,
                    execution_nonce TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    app_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_text TEXT,
                    error_text TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_pending_target
                    ON pending_plans(source_session_id, user_id, app_id, status);
            """)
        os.chmod(self.state_path, 0o600)

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(str(self.state_path), timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        return db

    async def handle(self, prompt: str, metadata: dict[str, Any]) -> dict[str, Any]:
        interrupt = str(metadata.get("resume_interrupt_id") or "").strip()
        if interrupt:
            return await self._resume(interrupt, str(metadata.get("operator_decision") or "").lower())
        lowered = prompt.lower()
        if "dc-app-access-diagnose" in lowered or (
            "application" in lowered and any(word in lowered for word in ("access", "permission", "role"))
        ):
            return await self._prepare_app_access(prompt, metadata)
        if "dc-path-troubleshoot" in lowered or any(
            phrase in lowered for phrase in ("end-to-end path", "network path", "load-balancer")
        ):
            return await self._verify_path(prompt)
        return _failed("local dc-agent only accepts reviewed app-access or path-verification tasks")

    async def _prepare_app_access(self, prompt: str, metadata: dict[str, Any]) -> dict[str, Any]:
        user_id, app_id = _targets(prompt)
        if not user_id or not app_id:
            return _failed("DC delegation requires explicit user_id and app_id")
        source = str(metadata.get("source_session_id") or metadata.get("session_id") or "unknown")
        with self._db() as db:
            existing = db.execute(
                """SELECT * FROM pending_plans WHERE source_session_id=? AND user_id=?
                   AND app_id=? AND status='waiting' ORDER BY created_at DESC LIMIT 1""",
                (source, user_id, app_id),
            ).fetchone()
        if existing is not None:
            return _input_required(existing["interrupt_id"], self._approval(existing["plan_id"]))

        runtime = NetworkRuntime(self.runtime_path)
        workflow_session = f"dc-a2a:{source}:{user_id}:{app_id}"
        with WorkflowRuntime(self.runtime_path) as workflows:
            workflows.start(session_id=workflow_session, profile="dc", mode="mock",
                            skill_name="dc-app-access-diagnose")
        access = await runtime.invoke_read(
            "dc", "dc_check_user_app_access", {"user_id": user_id, "app_id": app_id},
        )
        with WorkflowRuntime(self.runtime_path) as workflows:
            observed = workflows.observe(
                session_id=workflow_session, tool_name="dc_check_user_app_access",
                arguments={"user_id": user_id, "app_id": app_id}, result=access,
                success=True, mutating=False,
            )
        if observed["facts"].get("allowed") is True:
            return _message(json.dumps({"status": "completed", "skill": "dc-app-access-diagnose",
                "user_id": user_id, "app_id": app_id, "already_allowed": True,
                "verified_result": access}, ensure_ascii=False))

        acl = await runtime.invoke_read("dc", "dc_get_app_acl", {"app_id": app_id})
        with WorkflowRuntime(self.runtime_path) as workflows:
            workflows.observe(session_id=workflow_session, tool_name="dc_get_app_acl",
                arguments={"app_id": app_id}, result=acl, success=True, mutating=False)
        role_match = re.search(r"^\s*role\s+([A-Za-z0-9_.-]+)\s*:", acl, re.MULTILINE)
        if role_match is None:
            return _failed("DC ACL did not yield one reviewed base role")
        role = role_match.group(1)
        reason = f"delegated new-employee onboarding for {app_id}"
        arguments = {"user_id": user_id, "app_id": app_id, "role": role, "reason": reason}
        l0 = L0_SKILLS.for_tool("dc", "dc_grant_app_access")
        if l0 is None:
            return _failed("DC grant has no registered Network L0 Skill")
        prepared = await runtime.prepare("dc", "dc_grant_app_access", arguments,
            session_id=workflow_session, l0_skill_id=l0.skill_id)
        if prepared.get("status") != "plan_ready":
            return _failed("DC Network L0 plan rejected: " + json.dumps(prepared, ensure_ascii=False))
        plan = prepared["plan"]
        interrupt_id = f"dc-l0-{uuid.uuid4()}"
        try:
            with self._db() as db:
                db.execute("""INSERT INTO pending_plans
                    (interrupt_id,source_session_id,plan_id,plan_hash,execution_nonce,
                     user_id,app_id,role,reason,status) VALUES (?,?,?,?,?,?,?,?,?,'waiting')""",
                    (interrupt_id, source, plan["plan_id"], plan["plan_hash"],
                     prepared["execution_nonce"], user_id, app_id, role, reason))
        except Exception:
            runtime.reject(plan_id=plan["plan_id"], plan_hash=plan["plan_hash"],
                reason="local DC peer could not durably record the approval continuation")
            raise
        return _input_required(interrupt_id, self._approval(plan["plan_id"]))

    def _approval(self, plan_id: str) -> dict[str, Any]:
        plan = NetworkRuntime(self.runtime_path).inspect(plan_id)["plan"]
        keys = ("plan_id", "plan_hash", "tool_name", "arguments", "risk_level",
                "l0_skill_id", "l0_skill_version", "l0_contract_hash", "intent_hash",
                "verification_contract", "rollback_contract", "workflow_run_id",
                "workflow_template_hash", "expires_at")
        return {"kind": "network-l0-plan", "profile": "dc",
                **{key: plan[key] for key in keys}}

    async def _resume(self, interrupt_id: str, decision: str) -> dict[str, Any]:
        if decision not in {"approve", "reject"}:
            return _failed("remote continuation decision must be approve or reject")
        with self._db() as db:
            row = db.execute("SELECT * FROM pending_plans WHERE interrupt_id=?", (interrupt_id,)).fetchone()
        if row is None:
            return _failed(f"unknown DC approval interrupt: {interrupt_id}")
        if row["status"] == "completed":
            return _message(row["result_text"] or "DC plan already completed")
        if row["status"] == "rejected":
            return _message(json.dumps({"status": "rejected", "plan_id": row["plan_id"],
                                        "idempotent": True}))
        runtime = NetworkRuntime(self.runtime_path)
        if decision == "reject":
            runtime.reject(plan_id=row["plan_id"], plan_hash=row["plan_hash"],
                reason="rejected by DSH operator through durable A2A continuation")
            result = json.dumps({"status": "rejected", "plan_id": row["plan_id"],
                                 "message": "DC application access was not changed"}, ensure_ascii=False)
            with self._db() as db:
                db.execute("UPDATE pending_plans SET status='rejected',result_text=?,updated_at=CURRENT_TIMESTAMP WHERE interrupt_id=?",
                           (result, interrupt_id))
            return _message(result)
        with self._db() as db:
            claimed = db.execute("""UPDATE pending_plans SET status='executing',updated_at=CURRENT_TIMESTAMP
                WHERE interrupt_id=? AND status='waiting'""", (interrupt_id,)).rowcount
        if claimed != 1:
            return _failed("DC plan is already executing; inspect its audit record before retrying")
        arguments = {"user_id": row["user_id"], "app_id": row["app_id"],
                     "role": row["role"], "reason": row["reason"]}
        try:
            outcome = await runtime.execute(plan_id=row["plan_id"], plan_hash=row["plan_hash"],
                execution_nonce=row["execution_nonce"], approval_request_id=interrupt_id,
                approval_actor="local-dsh-a2a-operator", allow_destructive=True)
            if not outcome.ok:
                raise RuntimeError("DC L0 plan did not verify: " + json.dumps(outcome.to_dict()))
            workflow_session = f"dc-a2a:{row['source_session_id']}:{row['user_id']}:{row['app_id']}"
            with WorkflowRuntime(self.runtime_path) as workflows:
                workflows.observe(session_id=workflow_session, tool_name="dc_grant_app_access",
                    arguments=arguments, result=outcome.result or "", success=True, mutating=True)
            result = json.dumps({"status": "completed", "skill": "dc-app-access-diagnose",
                "plan_id": row["plan_id"], "plan_hash": row["plan_hash"],
                "terminal_state": outcome.state.value, "user_id": row["user_id"],
                "app_id": row["app_id"], "role": row["role"], "verified": True,
                "evidence": [item.to_dict() for item in outcome.evidence]}, ensure_ascii=False)
            with self._db() as db:
                db.execute("UPDATE pending_plans SET status='completed',result_text=?,updated_at=CURRENT_TIMESTAMP WHERE interrupt_id=?",
                           (result, interrupt_id))
            return _message(result)
        except Exception as error:
            with self._db() as db:
                db.execute("UPDATE pending_plans SET status='failed',error_text=?,updated_at=CURRENT_TIMESTAMP WHERE interrupt_id=?",
                           (str(error), interrupt_id))
            return _failed(str(error))

    async def _verify_path(self, prompt: str) -> dict[str, Any]:
        user_id, app_id = _targets(prompt)
        if not user_id or not app_id:
            return _failed("DC path verification requires explicit user_id and app_id")
        runtime = NetworkRuntime(self.runtime_path)
        access = await runtime.invoke_read("dc", "dc_check_user_app_access",
                                           {"user_id": user_id, "app_id": app_id})
        if "✅ ALLOWED" not in access:
            return _failed("DC path verification refused because application access is not granted")
        apps = await runtime.invoke_read("dc", "dc_list_apps", {})
        vip_match = re.search(rf"^\s*{re.escape(app_id)}\s+.*?(\d+(?:\.\d+){{3}})",
                              apps, re.MULTILINE | re.I)
        if vip_match is None:
            return _failed(f"application VIP not found for {app_id}")
        vip = vip_match.group(1)
        path = await runtime.invoke_read("dc", "dc_fabric_path_trace",
                                         {"src": "10.20.0.50", "dst": vip})
        pools = await runtime.invoke_read("dc", "dc_loadbalancer_pools", {})
        return _message(json.dumps({"status": "completed", "skill": "dc-path-troubleshoot",
            "user_id": user_id, "app_id": app_id, "vip": vip,
            "application_access_verified": True, "path_verified": "loss: 0%" in path,
            "path": path, "load_balancer_health": pools}, ensure_ascii=False))


class _Handler(BaseHTTPRequestHandler):
    server_version = "NetOpYuLocalDcPeer/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"dc-peer {self.address_string()} {fmt % args}", flush=True)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/").endswith("agent-card.json"):
            self._json(200, {"agent_id": "dc-agent", "name": "Local Mock DC Agent",
                "description": "Loopback-only reviewed DC application-access and path-verification peer.",
                "url": self.server.base_url, "skills": [
                    {"id": "dc-app-access-diagnose", "name": "DC Application Access",
                     "description": "Grant application RBAC through Network L0 approval.",
                     "tags": ["dc", "rbac", "application-access"]},
                    {"id": "dc-path-troubleshoot", "name": "DC Path Verification",
                     "description": "Verify application VIP path and load-balancer health.",
                     "tags": ["dc", "path", "verification"]}]})
            return
        if self.path == "/health":
            self._json(200, {"ok": True, "agent_id": "dc-agent", "mode": "mock", "pid": os.getpid()})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/stream":
            self._json(404, {"error": "not found"})
            return
        size = int(self.headers.get("Content-Length", "0"))
        if size <= 0 or size > 262_144:
            self._json(400, {"error": "invalid request size"})
            return
        try:
            request = json.loads(self.rfile.read(size))
            params = request["params"]
            parts = params["message"]["parts"]
            if len(parts) != 1 or not isinstance(parts[0].get("text"), str):
                raise ValueError("one text prompt part is required")
            event = asyncio.run(self.server.peer.handle(parts[0]["text"], params.get("metadata") or {}))
        except Exception as error:
            event = _failed(str(error))
        body = f"data: {json.dumps(event, ensure_ascii=False)}\n\ndata: [DONE]\n\n".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the loopback-only local mock DC A2A peer")
    parser.add_argument("--host", default=os.environ.get("NETOPYU_DSH_LOCAL_DC_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("NETOPYU_DSH_LOCAL_DC_PORT", "8765")))
    parser.add_argument("--runtime-store", default=str(default_journal_path()))
    parser.add_argument("--state-store", default=os.environ.get("NETOPYU_DSH_LOCAL_DC_STATE", "data/local_dc_peer.sqlite"))
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("local DC demo peer must bind to a loopback address")
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    server.peer = LocalDcPeer(runtime_path=args.runtime_store, state_path=args.state_store)
    server.base_url = f"http://{args.host}:{server.server_port}"
    print(f"local mock dc-agent: {server.base_url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
