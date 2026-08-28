#!/usr/bin/env python3
"""Exercise one real Service MCP + Containerlab L1/L0 access saga.

The case starts from an allowed Bob -> CRM baseline, revokes desired-state
entitlement first, reconciles the expected drift, revokes the real simulated
network enforcement, then repeats the reviewed workflow in reverse.  Every
write is prepared, approved, executed, independently verified and audited by
the Effect Runtime.  No provider write is invoked directly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from effect_runtime import EffectRuntime
from network_runtime import PlanState
from network_runtime.l0_skills import REGISTRY as L0_SKILLS
from network_runtime.workflows import WorkflowRuntime


PROFILE = "lan"
SKILL = "service-network-access-reconcile"
CHANGE_ID = "CHG-1001"


async def _read_json(
    runtime: EffectRuntime,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    session_id: str | None = None,
) -> dict[str, Any]:
    rendered = await runtime.invoke_read(PROFILE, tool_name, arguments)
    value = json.loads(rendered)
    if session_id:
        with WorkflowRuntime(runtime.journal_path) as workflows:
            recorded = workflows.observe(
                session_id=session_id,
                tool_name=tool_name,
                arguments=arguments,
                result=rendered,
                success=True,
                mutating=False,
            )
        if recorded.get("recorded") is not True:
            raise RuntimeError(f"workflow rejected observation for {tool_name}")
    return value


async def _execute_l0(
    runtime: EffectRuntime,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    session_id: str | None,
) -> dict[str, Any]:
    contract = L0_SKILLS.for_tool(PROFILE, tool_name)
    if contract is None:
        raise RuntimeError(f"missing L0 Skill for {tool_name}")
    prepared = await runtime.prepare(
        PROFILE,
        tool_name,
        arguments,
        session_id=session_id,
        l0_skill_id=contract.skill_id,
    )
    if prepared.get("status") != "plan_ready":
        raise RuntimeError(
            f"{tool_name} plan rejected: {json.dumps(prepared, ensure_ascii=False)}"
        )
    plan = prepared["plan"]
    outcome = await runtime.execute(
        plan_id=plan["plan_id"],
        plan_hash=plan["plan_hash"],
        execution_nonce=prepared["execution_nonce"],
        approval_request_id=f"local-service-network-e2e-{uuid.uuid4()}",
        approval_actor="local-e2e-operator",
        allow_destructive=True,
    )
    if outcome.state != PlanState.VERIFIED_SUCCESS:
        raise RuntimeError(
            f"{tool_name} ended in {outcome.state.value}: {outcome.error or 'verification failed'}"
        )
    if session_id:
        with WorkflowRuntime(runtime.journal_path) as workflows:
            recorded = workflows.observe(
                session_id=session_id,
                tool_name=tool_name,
                arguments=arguments,
                result=str(outcome.result or ""),
                success=True,
                mutating=True,
            )
        if recorded.get("recorded") is not True:
            raise RuntimeError(f"workflow rejected mutation observation for {tool_name}")
    audit = runtime.audit(plan["plan_id"])
    if audit.get("ok") is not True:
        raise RuntimeError(f"audit chain failed for {plan['plan_id']}")
    return {
        "tool": tool_name,
        "l0_skill_id": contract.skill_id,
        "plan_id": plan["plan_id"],
        "plan_hash": plan["plan_hash"],
        "provider_identity": plan["provider_identity"],
        "state": outcome.state.value,
        "audit_ok": True,
    }


def _start_workflow(runtime: EffectRuntime, suffix: str) -> str:
    session_id = f"service-network-e2e:{suffix}:{uuid.uuid4()}"
    with WorkflowRuntime(runtime.journal_path) as workflows:
        started = workflows.start(
            session_id=session_id,
            profile=PROFILE,
            mode="pragmatic",
            skill_name=SKILL,
        )
    if started.get("active") is not True:
        raise RuntimeError(f"could not start {SKILL}: {started}")
    return session_id


async def _revoke(
    runtime: EffectRuntime,
    user_id: str,
    app_id: str,
) -> list[dict[str, Any]]:
    session_id = _start_workflow(runtime, "revoke")
    entitlement = await _read_json(
        runtime,
        "access_policy_get_entitlement",
        {"user_id": user_id, "app_id": app_id},
        session_id=session_id,
    )
    await _read_json(
        runtime,
        "change_validate_window",
        {"change_id": CHANGE_ID},
        session_id=session_id,
    )
    service = await _execute_l0(
        runtime,
        "access_policy_revoke_entitlement",
        {
            "user_id": user_id,
            "app_id": app_id,
            "change_id": CHANGE_ID,
            "expected_revision": entitlement["revision"],
            "reason": "approved local cross-layer revoke exercise",
        },
        session_id=session_id,
    )
    await _read_json(
        runtime,
        "access_policy_get_entitlement",
        {"user_id": user_id, "app_id": app_id},
        session_id=session_id,
    )
    await _read_json(
        runtime,
        "network_get_app_enforcement",
        {"user_id": user_id, "app_id": app_id},
        session_id=session_id,
    )
    network = await _execute_l0(
        runtime,
        "network_revoke_app_enforcement",
        {
            "user_id": user_id,
            "app_id": app_id,
            "change_id": CHANGE_ID,
            "reason": "enforce approved Service Layer desired-state revoke",
        },
        session_id=session_id,
    )
    return [service, network]


async def _grant(
    runtime: EffectRuntime,
    user_id: str,
    app_id: str,
    role: str,
) -> list[dict[str, Any]]:
    session_id = _start_workflow(runtime, "grant")
    await _read_json(
        runtime, "identity_get_user", {"user_id": user_id}, session_id=session_id,
    )
    await _read_json(
        runtime,
        "access_policy_evaluate",
        {"user_id": user_id, "app_id": app_id},
        session_id=session_id,
    )
    entitlement = await _read_json(
        runtime,
        "access_policy_get_entitlement",
        {"user_id": user_id, "app_id": app_id},
        session_id=session_id,
    )
    await _read_json(
        runtime,
        "change_validate_window",
        {"change_id": CHANGE_ID},
        session_id=session_id,
    )
    service = await _execute_l0(
        runtime,
        "access_policy_grant_entitlement",
        {
            "user_id": user_id,
            "app_id": app_id,
            "role": role,
            "change_id": CHANGE_ID,
            "expected_revision": entitlement["revision"],
            "reason": "approved local cross-layer baseline restoration",
        },
        session_id=session_id,
    )
    await _read_json(
        runtime,
        "access_policy_get_entitlement",
        {"user_id": user_id, "app_id": app_id},
        session_id=session_id,
    )
    await _read_json(
        runtime,
        "network_get_app_enforcement",
        {"user_id": user_id, "app_id": app_id},
        session_id=session_id,
    )
    network = await _execute_l0(
        runtime,
        "network_apply_app_enforcement",
        {
            "user_id": user_id,
            "app_id": app_id,
            "change_id": CHANGE_ID,
            "reason": "enforce approved Service Layer baseline restoration",
        },
        session_id=session_id,
    )
    return [service, network]


async def run_case(args: argparse.Namespace) -> dict[str, Any]:
    if not args.approve_local_lab:
        raise PermissionError(
            "this case changes and restores Service SQLite and one Containerlab policy; "
            "pass --approve-local-lab explicitly"
        )
    os.environ["NETOPYU_CONFIG_PATH"] = str(Path(args.config).expanduser().resolve())
    os.environ["NETOPYU_BACKEND"] = "pragmatic"
    os.environ["NETOPYU_DSH_BACKEND"] = "pragmatic"
    os.environ["NETOPYU_NETWORK_RUNTIME_STORE"] = str(Path(args.journal).expanduser())
    os.environ["NETOPYU_TOOL_RESULT_STORE"] = str(Path(args.results).expanduser())

    runtime = EffectRuntime(args.journal)
    target = {"user_id": args.user, "app_id": args.application}
    baseline = await _read_json(runtime, "reconcile_service_network_access", target)
    desired = baseline["service_desired_state"]
    observed = baseline["network_observed_state"]
    roles = desired.get("roles") or []
    if not (
        baseline.get("consistent") is True
        and desired.get("allowed") is True
        and observed.get("allowed") is True
        and baseline.get("traffic_evidence", {}).get("ok") is True
        and len(roles) == 1
    ):
        return {
            "ok": False,
            "phase": "baseline",
            "error": "expected one-role allowed baseline; no write sent",
            "baseline": baseline,
        }

    plans: list[dict[str, Any]] = []
    restore_attempted = False
    try:
        plans.extend(await _revoke(runtime, args.user, args.application))
        denied = await _read_json(runtime, "reconcile_service_network_access", target)
        if not (
            denied.get("consistent") is True
            and denied["service_desired_state"].get("allowed") is False
            and denied["network_observed_state"].get("allowed") is False
            and denied.get("traffic_evidence", {}).get("ok") is False
        ):
            raise RuntimeError("revoke did not converge across Service and Network layers")
        plans.extend(await _grant(runtime, args.user, args.application, roles[0]))
    except Exception:
        # Only restore through new reviewed L0 plans. Never bypass the Runtime,
        # even in the local test's error path.
        restore_attempted = True
        current = await _read_json(runtime, "access_policy_get_entitlement", target)
        if current.get("roles") != roles:
            plans.extend(await _grant(runtime, args.user, args.application, roles[0]))
        else:
            network = await _read_json(runtime, "network_get_app_enforcement", target)
            if network.get("allowed") is not True:
                plans.append(await _execute_l0(
                    runtime,
                    "network_apply_app_enforcement",
                    {**target, "change_id": CHANGE_ID, "reason": "local E2E failure recovery"},
                    session_id=None,
                ))
        raise

    final = await _read_json(runtime, "reconcile_service_network_access", target)
    restored = (
        final.get("consistent") is True
        and final["service_desired_state"].get("roles") == roles
        and final["network_observed_state"].get("allowed") is True
        and final.get("traffic_evidence", {}).get("ok") is True
    )
    return {
        "ok": restored and all(item["audit_ok"] for item in plans),
        "case": "Service desired state -> Network enforcement -> data plane",
        "target": target,
        "baseline": baseline,
        "denied_checkpoint": denied,
        "plans": plans,
        "final": final,
        "semantic_baseline_restored": restored,
        "restore_attempted_after_error": restore_attempted,
    }


def parser() -> argparse.ArgumentParser:
    temporary = Path(tempfile.gettempdir())
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", default=str(ROOT / "config.small-production-lab.yaml"))
    value.add_argument("--journal", default=str(temporary / "netopyu-service-network-e2e.sqlite"))
    value.add_argument("--results", default=str(temporary / "netopyu-service-network-results.sqlite"))
    value.add_argument("--user", default="bob")
    value.add_argument("--application", default="crm")
    value.add_argument("--approve-local-lab", action="store_true")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        report = asyncio.run(run_case(args))
    except Exception as error:  # report any recovery/manual-intervention failure
        report = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
