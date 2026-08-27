"""Executable local demonstration of L1 reasoning guarded by Network L0 Skills."""

from __future__ import annotations

import copy
import os
import tempfile
from pathlib import Path
from typing import Any

from .engine import NetworkRuntime
from .evidence import typed_evidence
from .l0_skills import REGISTRY as L0_SKILLS
from .workflows import WorkflowRuntime


async def run_l1_l0_access_demo(*, approve_local_simulation: bool) -> dict[str, Any]:
    """Resolve one cross-domain access problem through two L1/L0 pairs.

    The L1 decisions are deliberately represented as reviewed workflow choices
    so the example is reproducible. In DSH those choices are proposed by the
    model from the same SKILL.md instructions; all L0 enforcement is identical.
    """
    if not approve_local_simulation:
        raise PermissionError(
            "the demo executes mock writes; pass --approve-local-simulation explicitly"
        )

    from profiles.dc import tools as dc_tools
    from profiles.lan import tools as lan_tools

    environment_names = (
        "NETOPYU_DSH_BACKEND",
        "NETOPYU_DSH_NETWORK_RUNTIME_STORE",
        "NETOPYU_DSH_TOOL_RESULT_STORE",
    )
    previous_environment = {name: os.environ.get(name) for name in environment_names}
    lan_changes = copy.deepcopy(lan_tools._LAN_ACCESS_CHANGES)
    dc_changes = copy.deepcopy(dc_tools._DC_ACCESS_CHANGES)
    dc_acl = copy.deepcopy(dc_tools._DC_APP_ACL)

    try:
        with tempfile.TemporaryDirectory(prefix="netopyu-l1-l0-demo-") as directory:
            root = Path(directory)
            journal_path = root / "network-runtime.sqlite"
            os.environ["NETOPYU_DSH_BACKEND"] = "mock"
            os.environ["NETOPYU_DSH_NETWORK_RUNTIME_STORE"] = str(journal_path)
            os.environ["NETOPYU_DSH_TOOL_RESULT_STORE"] = str(root / "tool-results.sqlite")

            # Make the use case deterministic even when called inside a test
            # process that previously exercised the same in-memory simulators.
            lan_tools._LAN_ACCESS_CHANGES.clear()
            dc_tools._DC_ACCESS_CHANGES.clear()
            dc_tools._DC_APP_ACL["crm"] = {
                "sales-rep": ["bob", "carol"], "sales-admin": ["dave"],
            }

            runtime = NetworkRuntime(journal_path)
            report: dict[str, Any] = {
                "ok": False,
                "scope": "local-mock-only",
                "problem": {
                    "user_id": "erin",
                    "application": "crm",
                    "request": "Provision verified end-to-end CRM access for new employee erin",
                },
                "stages": [],
                "writes": [],
            }

            lan_session = "demo-l1-lan"
            with WorkflowRuntime(journal_path) as workflows:
                lan_workflow = workflows.start(
                    session_id=lan_session,
                    profile="lan",
                    mode="mock",
                    skill_name="lan-new-employee-onboarding-access",
                )
            report["stages"].append({
                "stage": "l1_skill_selected",
                "domain": "lan",
                "skill": "lan-new-employee-onboarding-access",
                "workflow_run_id": lan_workflow["run_id"],
                "template_hash": lan_workflow["template"]["template_hash"],
                "decision_owner": "L1/LLM candidate; runtime enforces prerequisites",
            })

            users = await runtime.invoke_read("lan", "list_users", {})
            access_before = await runtime.invoke_read("lan", "get_user_access", {"user_id": "erin"})
            nac = await runtime.invoke_read("lan", "check_nac_policy", {"user_id": "erin"})
            with WorkflowRuntime(journal_path) as workflows:
                users_observation = workflows.observe(
                    session_id=lan_session,
                    tool_name="list_users",
                    arguments={},
                    result=users,
                    success=True,
                    mutating=False,
                )
                access_observation = workflows.observe(
                    session_id=lan_session,
                    tool_name="get_user_access",
                    arguments={"user_id": "erin"},
                    result=access_before,
                    success=True,
                    mutating=False,
                )
                nac_observation = workflows.observe(
                    session_id=lan_session,
                    tool_name="check_nac_policy",
                    arguments={"user_id": "erin"},
                    result=nac,
                    success=True,
                    mutating=False,
                )
            report["stages"].append({
                "stage": "l1_prerequisites_observed",
                "domain": "lan",
                "user_exists": "erin" in users_observation["facts"].get("user_ids", []),
                "access": access_observation["facts"],
                "nac": nac_observation["facts"],
                "next_decision": "invoke Network L0 Skill to restore LAN admission",
            })

            lan_l0 = _require_l0("lan", "grant_user_access")
            lan_prepared = await runtime.prepare(
                "lan",
                "grant_user_access",
                {"user_id": "erin", "reason": "new-hire CRM access demo"},
                session_id=lan_session,
                l0_skill_id=lan_l0.skill_id,
            )
            _require_plan(lan_prepared, "LAN admission")
            report["stages"].append(_plan_stage("lan", lan_prepared))
            lan_outcome = await _execute_local_plan(
                runtime, lan_prepared, approval_id="demo-approval-lan",
            )
            if not lan_outcome.ok:
                raise RuntimeError(f"LAN L0 Skill did not verify: {lan_outcome.to_dict()}")
            with WorkflowRuntime(journal_path) as workflows:
                workflows.observe(
                    session_id=lan_session,
                    tool_name="grant_user_access",
                    arguments={"user_id": "erin", "reason": "new-hire CRM access demo"},
                    result=lan_outcome.result or "",
                    success=True,
                    mutating=True,
                )
            lan_inspect = runtime.inspect(lan_prepared["plan"]["plan_id"])
            report["writes"].append(_write_stage("lan", lan_prepared, lan_outcome.to_dict(), lan_inspect))

            dc_session = "demo-l1-dc"
            with WorkflowRuntime(journal_path) as workflows:
                dc_workflow = workflows.start(
                    session_id=dc_session,
                    profile="dc",
                    mode="mock",
                    skill_name="dc-app-access-diagnose",
                )
            report["stages"].append({
                "stage": "l1_skill_delegated",
                "domain": "dc",
                "skill": "dc-app-access-diagnose",
                "workflow_run_id": dc_workflow["run_id"],
                "template_hash": dc_workflow["template"]["template_hash"],
                "routing": "local profile handoff standing in for DSH A2A transport",
            })

            app_before = await runtime.invoke_read(
                "dc", "dc_check_user_app_access", {"user_id": "erin", "app_id": "crm"},
            )
            acl = await runtime.invoke_read("dc", "dc_get_app_acl", {"app_id": "crm"})
            with WorkflowRuntime(journal_path) as workflows:
                app_observation = workflows.observe(
                    session_id=dc_session,
                    tool_name="dc_check_user_app_access",
                    arguments={"user_id": "erin", "app_id": "crm"},
                    result=app_before,
                    success=True,
                    mutating=False,
                )
                acl_observation = workflows.observe(
                    session_id=dc_session,
                    tool_name="dc_get_app_acl",
                    arguments={"app_id": "crm"},
                    result=acl,
                    success=True,
                    mutating=False,
                )
            report["stages"].append({
                "stage": "l1_prerequisites_observed",
                "domain": "dc",
                "application_access": app_observation["facts"],
                "acl": acl_observation["facts"],
                "next_decision": "invoke Network L0 Skill to grant reviewed CRM role",
            })

            dc_l0 = _require_l0("dc", "dc_grant_app_access")
            dc_arguments = {
                "user_id": "erin",
                "app_id": "crm",
                "role": "sales-rep",
                "reason": "new-hire CRM access demo",
            }
            dc_prepared = await runtime.prepare(
                "dc",
                "dc_grant_app_access",
                dc_arguments,
                session_id=dc_session,
                l0_skill_id=dc_l0.skill_id,
            )
            _require_plan(dc_prepared, "DC application access")
            report["stages"].append(_plan_stage("dc", dc_prepared))
            dc_outcome = await _execute_local_plan(
                runtime, dc_prepared, approval_id="demo-approval-dc",
            )
            if not dc_outcome.ok:
                raise RuntimeError(f"DC L0 Skill did not verify: {dc_outcome.to_dict()}")
            with WorkflowRuntime(journal_path) as workflows:
                workflows.observe(
                    session_id=dc_session,
                    tool_name="dc_grant_app_access",
                    arguments=dc_arguments,
                    result=dc_outcome.result or "",
                    success=True,
                    mutating=True,
                )
            dc_inspect = runtime.inspect(dc_prepared["plan"]["plan_id"])
            report["writes"].append(_write_stage("dc", dc_prepared, dc_outcome.to_dict(), dc_inspect))

            lan_final = await runtime.invoke_read("lan", "get_user_access", {"user_id": "erin"})
            dc_final = await runtime.invoke_read(
                "dc", "dc_check_user_app_access", {"user_id": "erin", "app_id": "crm"},
            )
            lan_facts = typed_evidence("get_user_access", lan_final)["facts"]
            dc_facts = typed_evidence("dc_check_user_app_access", dc_final)["facts"]
            resolved = lan_facts.get("admitted") is True and dc_facts.get("allowed") is True
            report["stages"].append({
                "stage": "independent_end_to_end_verification",
                "lan": lan_facts,
                "dc": dc_facts,
                "resolved": resolved,
            })
            report["guarantees_review"] = {
                "l1_scope": "selected workflows, interpreted observations and chose candidate L0 Skills",
                "l0_scope": "compiled intent, fixed steps, approval binding, write, verification and audit",
                "unverified_successes": 0,
                "unbound_writes": 0,
                "event_chains_valid": all(item["audit"]["ok"] for item in report["writes"]),
                "problem_resolved": resolved,
            }
            report["limitations"] = [
                "The demonstration uses local mock adapters and explicit simulated approval.",
                "The LAN-to-DC handoff uses separate local profiles; the A2A network transport is not exercised.",
                "L1 reasoning quality is not declared deterministic; unsafe effects remain bounded by L0 contracts.",
            ]
            report["ok"] = resolved and report["guarantees_review"]["event_chains_valid"]
            return report
    finally:
        lan_tools._LAN_ACCESS_CHANGES.clear()
        lan_tools._LAN_ACCESS_CHANGES.extend(lan_changes)
        dc_tools._DC_ACCESS_CHANGES.clear()
        dc_tools._DC_ACCESS_CHANGES.extend(dc_changes)
        dc_tools._DC_APP_ACL.clear()
        dc_tools._DC_APP_ACL.update(dc_acl)
        for name, value in previous_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _require_l0(profile: str, tool_name: str):
    contract = L0_SKILLS.for_tool(profile, tool_name)
    if contract is None:
        raise RuntimeError(f"missing L0 Skill for {profile}/{tool_name}")
    return contract


def _require_plan(prepared: dict[str, Any], label: str) -> None:
    if prepared.get("status") != "plan_ready":
        raise RuntimeError(f"{label} plan failed: {prepared}")


async def _execute_local_plan(
    runtime: NetworkRuntime, prepared: dict[str, Any], *, approval_id: str,
):
    plan = prepared["plan"]
    return await runtime.execute(
        plan_id=plan["plan_id"],
        plan_hash=plan["plan_hash"],
        execution_nonce=prepared["execution_nonce"],
        approval_request_id=approval_id,
        approval_actor="local-demo-operator",
        allow_destructive=True,
    )


def _plan_stage(domain: str, prepared: dict[str, Any]) -> dict[str, Any]:
    plan = prepared["plan"]
    return {
        "stage": "l0_plan_ready",
        "domain": domain,
        "plan_id": plan["plan_id"],
        "plan_hash": plan["plan_hash"],
        "l0_skill_id": plan["l0_skill_id"],
        "l0_contract_hash": plan["l0_contract_hash"],
        "intent_hash": plan["intent_hash"],
        "intent": plan["intent_spec"],
        "steps": [step["step_id"] for step in plan["step_contract"]],
        "approval_boundary": "explicit local-simulation approval",
    }


def _write_stage(
    domain: str,
    prepared: dict[str, Any],
    outcome: dict[str, Any],
    inspected: dict[str, Any],
) -> dict[str, Any]:
    return {
        "domain": domain,
        "plan_id": prepared["plan"]["plan_id"],
        "l0_skill_id": prepared["plan"]["l0_skill_id"],
        "terminal_state": outcome["state"],
        "evidence": [{
            "type": item["evidence_type"],
            "source": item["source"],
            "passed": item["passed"],
            "predicate": item["predicate"],
        } for item in outcome["evidence"]],
        "l0_events": [{
            "event_type": event["event_type"],
            "step_id": event["payload"].get("step_id"),
        } for event in inspected["events"] if event["event_type"].startswith("l0_step_")],
        "audit": inspected["audit"],
    }
