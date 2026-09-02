"""Compiled network-skill templates and durable prerequisite observations.

SKILL.md remains model guidance.  Only the reviewed templates in this module
can authorize an ordered mutating step; arbitrary prose is never interpreted
as executable control flow.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from skills import SkillLoader
from tools.loader import ToolLoader

from .contracts import canonical_json, sha256_json, utc_now


class WorkflowContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class ObservationRequirement:
    tool_name: str
    same_fields: tuple[str, ...]
    expected_facts: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "same_fields": list(self.same_fields),
            "expected_facts": self.expected_facts,
        }


@dataclass(frozen=True)
class WorkflowTemplate:
    skill_id: str
    skill_name: str
    version: str
    allowed_tools: tuple[str, ...]
    write_requirements: dict[str, tuple[ObservationRequirement, ...]]
    terminal_writes: tuple[str, ...]
    template_hash: str

    @classmethod
    def create(
        cls,
        *,
        skill_id: str,
        skill_name: str,
        version: str,
        allowed_tools: tuple[str, ...],
        write_requirements: dict[str, tuple[ObservationRequirement, ...]],
        terminal_writes: tuple[str, ...] | None = None,
    ) -> "WorkflowTemplate":
        payload = {
            "skill_id": skill_id,
            "skill_name": skill_name,
            "version": version,
            "allowed_tools": list(allowed_tools),
            "write_requirements": {
                tool: [item.to_dict() for item in requirements]
                for tool, requirements in sorted(write_requirements.items())
            },
            "terminal_writes": list(terminal_writes or tuple(write_requirements)),
        }
        return cls(template_hash=sha256_json(payload), **{
            **payload,
            "allowed_tools": tuple(allowed_tools),
            "write_requirements": write_requirements,
            "terminal_writes": tuple(terminal_writes or tuple(write_requirements)),
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "skill_name": self.skill_name,
            "version": self.version,
            "allowed_tools": list(self.allowed_tools),
            "write_requirements": {
                tool: [item.to_dict() for item in requirements]
                for tool, requirements in sorted(self.write_requirements.items())
            },
            "terminal_writes": list(self.terminal_writes),
            "template_hash": self.template_hash,
        }


def _require(
    tool_name: str,
    *same_fields: str,
    **expected_facts: Any,
) -> ObservationRequirement:
    return ObservationRequirement(tool_name, tuple(same_fields), expected_facts)


_REVIEWED: dict[str, dict[str, Any]] = {
    "restart_service": {
        "version": "1.0.0", "allowed": ("restart_service",),
        "writes": {"restart_service": ()},
    },
    "rollback_service": {
        "version": "1.0.0", "allowed": ("rollback_service",),
        "writes": {"rollback_service": ()},
    },
    "edit_device_config": {
        "version": "1.0.0", "allowed": (
            "get_device_config", "validate_device_config", "edit_device_config",
        ),
        "writes": {"edit_device_config": ()},
    },
    "lab_ospf_path_remediation": {
        "version": "1.0.0", "allowed": (
            "get_device_config", "get_ospf_neighbors", "lab_probe", "edit_device_config",
        ),
        "writes": {"edit_device_config": (
            _require("get_device_config", "device_id", config_readable=True),
            _require("get_ospf_neighbors", "device_id", full_neighbors=2),
            _require("lab_probe", probe_id="branch-to-dc", probe_ok=True),
        )},
    },
    "lan_new_employee_onboarding_access": {
        "version": "1.0.0", "allowed": (
            "list_users", "get_user_access", "check_nac_policy", "grant_user_access",
        ),
        "writes": {"grant_user_access": (
            _require("get_user_access", "user_id", admitted=False),
            _require("check_nac_policy", "user_id", permit=False),
        )},
    },
    "lan_user_access_diagnose": {
        "version": "1.0.0", "allowed": (
            "list_users", "get_user_access", "check_nac_policy", "grant_user_access",
        ),
        "writes": {"grant_user_access": (
            _require("get_user_access", "user_id", admitted=False),
            _require("check_nac_policy", "user_id", permit=False),
        )},
    },
    "agentized_lan_access_remediation": {
        "version": "1.0.0", "allowed": (
            "get_user_access", "check_nac_policy", "grant_user_access",
        ),
        "writes": {"grant_user_access": (
            _require("get_user_access", "user_id", admitted=False),
            _require("check_nac_policy", "user_id", permit=False),
        )},
    },
    "dc_app_access_diagnose": {
        "version": "1.0.0", "allowed": (
            "dc_list_apps", "dc_check_user_app_access", "dc_get_app_acl", "dc_grant_app_access",
        ),
        "writes": {"dc_grant_app_access": (
            _require("dc_check_user_app_access", "user_id", "app_id", allowed=False),
            _require("dc_get_app_acl", "app_id", acl_loaded=True),
        )},
    },
    "lab_fabric_access_vlan_change": {
        "version": "1.0.0", "allowed": (
            "lab_get_access_vlan", "lab_probe", "fabric_set_access_vlan",
        ),
        "writes": {"fabric_set_access_vlan": (
            _require("lab_get_access_vlan", "device_id", "interface", access_vlan_readable=True),
        )},
    },
    "service_network_access_reconcile": {
        "version": "1.0.0",
        "allowed": (
            "identity_get_user", "application_get", "access_policy_evaluate",
            "access_policy_get_entitlement", "change_validate_window",
            "cmdb_get_endpoint_binding", "network_get_app_enforcement",
            "lab_app_probe", "reconcile_service_network_access",
            "access_policy_grant_entitlement", "access_policy_revoke_entitlement",
            "network_apply_app_enforcement", "network_revoke_app_enforcement",
        ),
        "writes": {
            "access_policy_grant_entitlement": (
                _require("identity_get_user", "user_id", identity_active=True),
                _require("access_policy_evaluate", "user_id", "app_id", eligible=True),
                _require("access_policy_get_entitlement", "user_id", "app_id", allowed=False),
                _require("change_validate_window", "change_id", permitted=True),
            ),
            "access_policy_revoke_entitlement": (
                _require("access_policy_get_entitlement", "user_id", "app_id", allowed=True),
                _require("change_validate_window", "change_id", permitted=True),
            ),
            "network_apply_app_enforcement": (
                _require("access_policy_get_entitlement", "user_id", "app_id", allowed=True),
                _require("network_get_app_enforcement", "user_id", "app_id", allowed=False),
                _require("change_validate_window", "change_id", permitted=True),
            ),
            "network_revoke_app_enforcement": (
                _require("access_policy_get_entitlement", "user_id", "app_id", allowed=False),
                _require("network_get_app_enforcement", "user_id", "app_id", allowed=True),
                _require("change_validate_window", "change_id", permitted=True),
            ),
        },
        "terminal_writes": (
            "network_apply_app_enforcement", "network_revoke_app_enforcement",
        ),
    },
    "enterprise_access_mcp_agent": {
        "version": "1.0.0",
        "allowed": (
            "identity_get_user", "application_get", "access_policy_evaluate",
            "access_policy_get_entitlement", "change_validate_window",
            "access_policy_grant_entitlement",
        ),
        "writes": {"access_policy_grant_entitlement": (
            _require("identity_get_user", "user_id", identity_active=True),
            _require("access_policy_evaluate", "user_id", "app_id", eligible=True),
            _require("access_policy_get_entitlement", "user_id", "app_id", allowed=False),
            _require("change_validate_window", "change_id", permitted=True),
        )},
    },
}


def compile_workflow_templates(profile: str, mode: str) -> dict[str, WorkflowTemplate]:
    skills = SkillLoader(mode=mode, profile=profile).skill_definitions()
    metadata = ToolLoader(mode=mode, profile=profile).build_metadata()
    templates: dict[str, WorkflowTemplate] = {}
    for skill_id, definition in skills.items():
        dependencies = tuple(str(item) for item in definition.get("tool_deps", ()))
        mutating = tuple(
            name for name in dependencies
            if name in metadata and (
                bool(metadata[name].get("hitl"))
                or str(metadata[name].get("action_type", "read_only")) != "read_only"
            )
        )
        if not mutating:
            continue
        reviewed = _REVIEWED.get(skill_id)
        if reviewed is None:
            raise WorkflowContractError(
                f"mutating skill {skill_id} has no reviewed Network Runtime workflow template"
            )
        skill_name = str(definition.get("_std_name") or skill_id.replace("_", "-"))
        allowed = tuple(str(item) for item in reviewed["allowed"])
        unknown = sorted(set(allowed) - set(metadata))
        if unknown:
            raise WorkflowContractError(
                f"workflow {skill_id} references unregistered tools: {', '.join(unknown)}"
            )
        writes = dict(reviewed["writes"])
        if set(mutating) != set(writes):
            raise WorkflowContractError(
                f"workflow {skill_id} write coverage mismatch: skill={sorted(mutating)}, "
                f"template={sorted(writes)}"
            )
        for write_name, requirements in writes.items():
            if write_name not in allowed:
                raise WorkflowContractError(f"workflow {skill_id} write {write_name} is not allowed")
            for requirement in requirements:
                if requirement.tool_name not in allowed:
                    raise WorkflowContractError(
                        f"workflow {skill_id} prerequisite {requirement.tool_name} is not allowed"
                    )
        template = WorkflowTemplate.create(
            skill_id=skill_id,
            skill_name=skill_name,
            version=str(reviewed["version"]),
            allowed_tools=allowed,
            write_requirements=writes,
            terminal_writes=tuple(reviewed.get("terminal_writes") or tuple(writes)),
        )
        templates[skill_name] = template
    return templates


def _extract_facts(tool_name: str, arguments: dict[str, Any], result: str) -> dict[str, Any]:
    if tool_name in {
        "identity_get_user", "access_policy_evaluate", "access_policy_get_entitlement",
        "change_validate_window", "network_get_app_enforcement",
        "reconcile_service_network_access",
    }:
        try:
            payload = json.loads(result)
        except json.JSONDecodeError:
            return {"structured": False}
        if tool_name == "identity_get_user":
            return {"identity_active": payload.get("user", {}).get("status") == "active"}
        if tool_name == "access_policy_evaluate":
            return {"eligible": payload.get("eligible") is True}
        if tool_name == "access_policy_get_entitlement":
            return {
                "allowed": payload.get("allowed") is True,
                "revision": payload.get("revision"),
            }
        if tool_name == "change_validate_window":
            return {"permitted": payload.get("permitted") is True}
        if tool_name == "network_get_app_enforcement":
            return {"allowed": payload.get("allowed") is True}
        return {"consistent": payload.get("consistent") is True}
    if tool_name == "list_users":
        return {"user_ids": sorted(set(re.findall(r"^\s{2}([a-zA-Z0-9_.-]+)\s+", result, re.MULTILINE)))}
    if tool_name == "get_user_access":
        return {"admitted": "✅ ADMITTED" in result}
    if tool_name == "check_nac_policy":
        return {"permit": "result         : PERMIT" in result}
    if tool_name == "dc_check_user_app_access":
        return {"allowed": "✅ ALLOWED" in result}
    if tool_name == "dc_get_app_acl":
        return {"acl_loaded": result.startswith("Access control for application")}
    if tool_name == "get_device_config":
        return {"config_readable": bool(result.strip()) and "[Error]" not in result}
    if tool_name == "get_ospf_neighbors":
        return {"full_neighbors": result.lower().count("full")}
    if tool_name == "lab_probe":
        try:
            payload = json.loads(result)
        except json.JSONDecodeError:
            return {"probe_id": None, "probe_ok": False}
        return {
            "probe_id": payload.get("probe_id"),
            "probe_ok": payload.get("ok") is True,
        }
    if tool_name == "lab_get_access_vlan":
        try:
            payload = json.loads(result)
        except json.JSONDecodeError:
            return {"access_vlan_readable": False}
        return {
            "access_vlan_readable": (
                isinstance(payload, dict)
                and payload.get("ok") is True
                and isinstance(payload.get("current_vlan"), int)
                and bool(payload.get("bridge"))
            ),
            "current_vlan": payload.get("current_vlan") if isinstance(payload, dict) else None,
        }
    return {"completed": True}


class WorkflowRuntime:
    """Durable per-session workflow guard stored beside network plans."""

    def __init__(self, path: str | Path, *, max_age_seconds: int = 1800) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path), timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS workflow_runs (
                run_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                profile TEXT NOT NULL,
                mode TEXT NOT NULL,
                skill_name TEXT NOT NULL,
                template_hash TEXT NOT NULL,
                template_json TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_workflow_session_status
                ON workflow_runs(session_id, status, started_at);
            CREATE TABLE IF NOT EXISTS workflow_observations (
                observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                arguments_json TEXT NOT NULL,
                facts_json TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES workflow_runs(run_id)
            );
        """)
        self.db.commit()
        os.chmod(self.path, 0o600)
        self.max_age_seconds = max_age_seconds

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "WorkflowRuntime":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def start(self, *, session_id: str, profile: str, mode: str, skill_name: str) -> dict[str, Any]:
        if not session_id.strip():
            raise WorkflowContractError("workflow session_id is required")
        template = compile_workflow_templates(profile, mode).get(skill_name)
        if template is None:
            return {"active": False, "reason": "skill has no mutating workflow template"}
        now = utc_now()
        run_id = str(uuid.uuid4())
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute(
                "UPDATE workflow_runs SET status='superseded', updated_at=? WHERE session_id=? AND status='active'",
                (now, session_id),
            )
            self.db.execute(
                """INSERT INTO workflow_runs
                   (run_id, session_id, profile, mode, skill_name, template_hash,
                    template_json, status, started_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)""",
                (
                    run_id, session_id, profile, mode, skill_name, template.template_hash,
                    canonical_json(template.to_dict()), now, now,
                ),
            )
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        return {"active": True, "run_id": run_id, "template": template.to_dict()}

    def active(self, session_id: str) -> dict[str, Any] | None:
        row = self.db.execute(
            """SELECT * FROM workflow_runs
               WHERE session_id=? AND status='active' ORDER BY started_at DESC LIMIT 1""",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        started = datetime.fromisoformat(row["started_at"])
        if started + timedelta(seconds=self.max_age_seconds) <= datetime.now(timezone.utc):
            self.db.execute(
                "UPDATE workflow_runs SET status='expired', updated_at=? WHERE run_id=?",
                (utc_now(), row["run_id"]),
            )
            self.db.commit()
            return None
        return {**dict(row), "template": json.loads(row["template_json"])}

    def authorize(
        self,
        *,
        session_id: str | None,
        tool_name: str,
        arguments: dict[str, Any],
        mutating: bool,
    ) -> dict[str, Any]:
        if not session_id:
            return {"allowed": True, "workflow": None}
        run = self.active(session_id)
        if run is None:
            return {"allowed": True, "workflow": None}
        template = run["template"]
        if not mutating:
            return {
                "allowed": True,
                "workflow": {"run_id": run["run_id"], "template_hash": run["template_hash"]},
            }
        requirements = template["write_requirements"].get(tool_name)
        if tool_name not in template["allowed_tools"] or requirements is None:
            return {
                "allowed": False,
                "reason": f"active skill {run['skill_name']} does not allow write tool {tool_name}",
            }
        observations = self.db.execute(
            """SELECT tool_name, arguments_json, facts_json FROM workflow_observations
               WHERE run_id=? ORDER BY observation_id DESC""",
            (run["run_id"],),
        ).fetchall()
        missing: list[str] = []
        for requirement in requirements:
            matched = False
            for observation in observations:
                if observation["tool_name"] != requirement["tool_name"]:
                    continue
                observed_args = json.loads(observation["arguments_json"])
                if any(observed_args.get(field) != arguments.get(field) for field in requirement["same_fields"]):
                    continue
                facts = json.loads(observation["facts_json"])
                if all(facts.get(key) == expected for key, expected in requirement["expected_facts"].items()):
                    matched = True
                    break
            if not matched:
                missing.append(
                    f"{requirement['tool_name']} facts={requirement['expected_facts']} "
                    f"same_fields={requirement['same_fields']}"
                )
        if missing:
            return {
                "allowed": False,
                "reason": "workflow prerequisites are missing or stale: " + "; ".join(missing),
            }
        return {
            "allowed": True,
            "workflow": {"run_id": run["run_id"], "template_hash": run["template_hash"]},
        }

    def observe(
        self,
        *,
        session_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        result: str,
        success: bool,
        mutating: bool,
    ) -> dict[str, Any]:
        run = self.active(session_id)
        if run is None or not success or tool_name not in run["template"]["allowed_tools"]:
            return {"recorded": False}
        facts = _extract_facts(tool_name, arguments, result)
        now = utc_now()
        self.db.execute(
            """INSERT INTO workflow_observations
               (run_id, tool_name, arguments_json, facts_json, observed_at)
               VALUES (?, ?, ?, ?, ?)""",
            (run["run_id"], tool_name, canonical_json(arguments), canonical_json(facts), now),
        )
        terminal_writes = set(run["template"].get("terminal_writes") or ())
        if mutating and tool_name in terminal_writes:
            self.db.execute(
                "UPDATE workflow_runs SET status='completed', updated_at=? WHERE run_id=?",
                (now, run["run_id"]),
            )
        else:
            self.db.execute(
                "UPDATE workflow_runs SET updated_at=? WHERE run_id=?", (now, run["run_id"]),
            )
        self.db.commit()
        return {"recorded": True, "run_id": run["run_id"], "facts": facts}

    def validate_plan_binding(self, run_id: str, template_hash: str) -> None:
        row = self.db.execute(
            "SELECT template_hash, status FROM workflow_runs WHERE run_id=?", (run_id,),
        ).fetchone()
        if row is None or row["template_hash"] != template_hash or row["status"] != "active":
            raise WorkflowContractError("workflow binding changed or is no longer active")
