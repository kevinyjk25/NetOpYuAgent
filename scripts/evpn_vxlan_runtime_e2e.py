#!/usr/bin/env python3
"""Run the reviewed L1 + L0 access-VLAN rollback case on the real local lab."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dsh_adapter.backend import open_backend
from network_runtime import NetworkRuntime, PlanState
from network_runtime.l0_skills import REGISTRY as L0_SKILLS
from network_runtime.workflows import WorkflowRuntime


async def _call(tool_name: str, arguments: dict[str, object]) -> str:
    backend = await open_backend("dc")
    try:
        return str(await backend.callables[tool_name](arguments))
    finally:
        await backend.close()


async def run_case(args: argparse.Namespace) -> dict[str, object]:
    if not args.approve_local_lab:
        raise PermissionError(
            "this test changes and rolls back one disposable local access port; "
            "pass --approve-local-lab explicitly"
        )
    os.environ["NETOPYU_CONFIG_PATH"] = str(Path(args.config).expanduser().resolve())
    os.environ["NETOPYU_BACKEND"] = "pragmatic"
    os.environ["NETOPYU_DSH_BACKEND"] = "pragmatic"
    os.environ["NETOPYU_NETWORK_RUNTIME_STORE"] = str(Path(args.journal).expanduser())
    os.environ["NETOPYU_TOOL_RESULT_STORE"] = str(Path(args.results).expanduser())

    target = {"device_id": args.device, "interface": args.interface}
    before = json.loads(await _call("lab_get_access_vlan", target))
    baseline_probe = json.loads(await _call("lab_probe", {"probe_id": args.probe}))
    if before.get("ok") is not True or baseline_probe.get("ok") is not True:
        return {
            "ok": False,
            "phase": "baseline",
            "before": before,
            "baseline_probe": baseline_probe,
            "error": "baseline state or protected traffic is not healthy; no write sent",
        }
    if int(before["current_vlan"]) == args.target_vlan:
        return {
            "ok": False,
            "phase": "baseline",
            "before": before,
            "error": "target VLAN already matches current state; no write sent",
        }

    session_id = f"evpn-vxlan-e2e-{uuid.uuid4()}"
    with WorkflowRuntime(args.journal) as workflows:
        workflow = workflows.start(
            session_id=session_id,
            profile="dc",
            mode="pragmatic",
            skill_name="lab-fabric-access-vlan-change",
        )
        if workflow.get("active") is not True:
            return {"ok": False, "phase": "workflow", "workflow": workflow}
        workflows.observe(
            session_id=session_id,
            tool_name="lab_get_access_vlan",
            arguments=target,
            result=json.dumps(before, sort_keys=True),
            success=True,
            mutating=False,
        )
        workflows.observe(
            session_id=session_id,
            tool_name="lab_probe",
            arguments={"probe_id": args.probe},
            result=json.dumps(baseline_probe, sort_keys=True),
            success=True,
            mutating=False,
        )

    runtime = NetworkRuntime(args.journal)
    l0 = L0_SKILLS.for_tool("dc", "fabric_set_access_vlan")
    prepared = await runtime.prepare(
        "dc",
        "fabric_set_access_vlan",
        {
            **target,
            "vlan_id": args.target_vlan,
            "reason": "real local Containerlab L1+L0 rollback verification",
            "verification_probe_id": args.probe,
        },
        session_id=session_id,
        l0_skill_id=l0.skill_id if l0 else None,
    )
    if prepared.get("status") != "plan_ready":
        return {"ok": False, "phase": "prepare", "prepared": prepared}
    plan = prepared["plan"]
    outcome = await runtime.execute(
        plan_id=plan["plan_id"],
        plan_hash=plan["plan_hash"],
        execution_nonce=prepared["execution_nonce"],
        approval_request_id="local-containerlab-e2e-approved",
        approval_actor="local-e2e-operator",
        allow_destructive=True,
    )
    after = json.loads(await _call("lab_get_access_vlan", target))
    recovered_probe = json.loads(await _call("lab_probe", {"probe_id": args.probe}))
    exact = all(before.get(field) == after.get(field) for field in (
        "device_id", "interface", "current_vlan", "bridge", "vlans",
    ))
    passed = (
        outcome.state == PlanState.ROLLBACK_VERIFIED
        and exact
        and recovered_probe.get("ok") is True
        and runtime.audit(plan["plan_id"])["ok"] is True
    )
    return {
        "ok": passed,
        "before": before,
        "baseline_probe": baseline_probe,
        "workflow": {
            "skill": "lab-fabric-access-vlan-change",
            "run_id": workflow["run_id"],
            "template_hash": workflow["template"]["template_hash"],
        },
        "plan": {
            "plan_id": plan["plan_id"],
            "l0_skill_id": plan["l0_skill_id"],
            "risk_level": plan["risk_level"],
            "verification_contract": plan["verification_contract"],
            "rollback_contract": plan["rollback_contract"],
        },
        "outcome": {
            "state": outcome.state.value,
            "error": outcome.error,
            "evidence": [
                {
                    "type": item.evidence_type,
                    "source": item.source,
                    "passed": item.passed,
                }
                for item in outcome.evidence
            ],
        },
        "after": after,
        "exact_state_restored": exact,
        "recovered_probe": recovered_probe,
        "audit_chain_ok": runtime.audit(plan["plan_id"])["ok"],
    }


def parser() -> argparse.ArgumentParser:
    temporary = Path(tempfile.gettempdir())
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--config", default=str(PROJECT_ROOT / "config.evpn-vxlan-lab.yaml"),
    )
    result.add_argument("--journal", default=str(temporary / "netopyu-evpn-e2e.sqlite"))
    result.add_argument("--results", default=str(temporary / "netopyu-evpn-results.sqlite"))
    result.add_argument("--device", default="leaf-1")
    result.add_argument("--interface", default="eth3")
    result.add_argument("--target-vlan", type=int, default=20)
    result.add_argument("--probe", default="tenant-a-l2vpn")
    result.add_argument("--approve-local-lab", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        report = asyncio.run(run_case(args))
    except (OSError, PermissionError, RuntimeError, ValueError) as error:
        report = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
