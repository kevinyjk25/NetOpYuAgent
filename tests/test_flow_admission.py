"""Local branch-to-Effect safety regressions, not language/generalization scores."""

import asyncio
import json
from dataclasses import replace

import pytest

from evaluation.flow_effect_demo import ARGUMENTS, local_gate, run_demo
from evaluation.read_local_demo import DATASET
from network_runtime.contracts import ApprovalError, NetworkRuntimeError, PlanIntegrityError, PlanState, PreparedPlan, sha256_json
from network_runtime.engine import NetworkRuntime


@pytest.fixture
def environment(tmp_path, monkeypatch):
    from profiles.lan import tools as lan_tools
    monkeypatch.setenv("NETOPYU_DSH_BACKEND", "mock")
    monkeypatch.setenv("NETOPYU_DSH_NETWORK_RUNTIME_STORE", str(tmp_path / "journal.sqlite"))
    monkeypatch.setenv("NETOPYU_DSH_TOOL_RESULT_STORE", str(tmp_path / "results.sqlite"))
    monkeypatch.setattr(lan_tools, "_LAN_ACCESS_CHANGES", [])
    monkeypatch.setattr(lan_tools, "_MOCK_OPERATION_STATE", {})
    dataset = tmp_path / "inventory.json"
    dataset.write_bytes(DATASET.read_bytes())
    gate = local_gate(dataset)
    runtime = NetworkRuntime(tmp_path / "journal.sqlite", flow_gates={"local": gate})
    return runtime, dataset, lan_tools


def prepare(runtime, **kwargs):
    target = runtime.flow_gates["local"].effects["grant"]
    return asyncio.run(runtime.prepare("lan", target.tool, dict(ARGUMENTS), l0_skill_id=target.skill_id,
                                       **{"flow_gate_id": "local", **kwargs}))


def execute(runtime, prepared, **kwargs):
    plan = prepared["plan"]
    return asyncio.run(runtime.execute(**{"plan_id": plan["plan_id"], "plan_hash": plan["plan_hash"],
        "execution_nonce": prepared["execution_nonce"], "allow_destructive": True,
        "approval_actor": "unit-test-operator", "approval_request_id": "flow-test", **kwargs}))


def test_bound_plan_roundtrip_and_normal_transaction(environment):
    runtime, _, _ = environment
    prepared = prepare(runtime)
    assert prepared["status"] == "plan_ready"
    plan = PreparedPlan.from_dict(prepared["plan"])
    assert plan.schema_version == 11 and plan.flow_binding
    assert execute(runtime, prepared).state == PlanState.VERIFIED_SUCCESS
    events = runtime.inspect(plan.plan_id)["events"]
    assert any(row["event_type"] == "business_flow_revalidated" for row in events)


@pytest.mark.parametrize("gate_id", [None, "unknown"])
def test_cannot_omit_or_select_unknown_configured_gate(environment, gate_id):
    runtime, _, tools = environment
    assert prepare(runtime, flow_gate_id=gate_id)["status"] == "rejected"
    assert not tools._LAN_ACCESS_CHANGES


@pytest.mark.parametrize("mutation", ["branch", "facts", "missing", "permission", "gate", "context"])
def test_changed_branch_evidence_or_environment_prevents_write(environment, mutation):
    runtime, dataset, tools = environment
    prepared = prepare(runtime)
    gate = runtime.flow_gates["local"]
    if mutation in {"branch", "facts"}:
        raw = json.loads(dataset.read_text())
        raw["campus-sw1"]["site" if mutation == "branch" else "status"] = "changed"
        dataset.write_text(json.dumps(raw))
    elif mutation == "missing":
        dataset.unlink()
    elif mutation == "permission":
        runtime.flow_gates["local"] = replace(gate, context=replace(gate.context, authenticated=False))
    elif mutation == "context":
        runtime.flow_gates["local"] = replace(gate, context=replace(gate.context, subject_id="another-reader"))
    else:
        runtime = NetworkRuntime(runtime.journal_path)
    assert execute(runtime, prepared).state == PlanState.PRECONDITION_CHANGED
    assert not tools._LAN_ACCESS_CHANGES


@pytest.mark.parametrize("mutation", ["remove", "legacy", "decision", "target", "arguments"])
def test_plan_cannot_drop_or_rewrite_flow_evidence(environment, mutation):
    runtime, _, _ = environment
    raw = prepare(runtime)["plan"]
    if mutation == "remove":
        raw.pop("flow_binding")
    elif mutation == "legacy":
        raw["schema_version"] = 10
    elif mutation == "decision":
        raw["flow_binding"]["decision_digest"] = "bad"
    else:
        candidate = raw["flow_binding"]["decision"]["candidate"]
        if mutation == "target":
            candidate["target"]["tool"] = "other"
        else:
            candidate["arguments"]["user_id"] = "bob"
        raw["flow_binding"]["decision_digest"] = sha256_json(raw["flow_binding"]["decision"])
    with pytest.raises(PlanIntegrityError):
        PreparedPlan.from_dict(raw)


def test_flow_gate_does_not_replace_effect_approval(environment):
    runtime, _, tools = environment
    prepared = prepare(runtime)
    with pytest.raises(ApprovalError):
        execute(runtime, prepared, allow_destructive=False)
    assert not tools._LAN_ACCESS_CHANGES


def test_restart_with_same_host_gate_revalidates_and_executes(environment):
    runtime, dataset, _ = environment
    prepared = prepare(runtime)
    restarted = NetworkRuntime(runtime.journal_path, flow_gates={"local": local_gate(dataset)})
    assert execute(restarted, prepared).state == PlanState.VERIFIED_SUCCESS


def test_existing_verification_failure_compensates(environment):
    runtime, _, tools = environment

    def fault(stage, _plan):
        if stage == "before_verify":
            tools._LAN_ACCESS_CHANGES.append({"user_id": "erin", "op": "fault",
                "changes": {"radius": "fail", "dot1x": "rejected", "nac": "quarantine", "vlan": None}, "reason": "test"})

    runtime.fault_hook = fault
    assert execute(runtime, prepare(runtime)).state == PlanState.ROLLBACK_VERIFIED


def test_standalone_plan_remains_schema_10(environment):
    runtime, _, _ = environment
    target = runtime.flow_gates["local"].effects["grant"]
    runtime.flow_gates.clear()
    prepared = asyncio.run(runtime.prepare("lan", target.tool, dict(ARGUMENTS), l0_skill_id=target.skill_id))
    assert prepared["plan"]["schema_version"] == 10 and "flow_binding" not in prepared["plan"]
    PreparedPlan.from_dict(prepared["plan"])


def test_local_demo_success_drift_and_recovery():
    report = asyncio.run(run_demo(approve_local_simulation=True))
    assert [row["state"] for row in report["outcomes"]] == ["verified_success", "precondition_changed", "rollback_verified"]
    assert not report["outcomes"][1]["mockChanges"]
    assert report["modelCalls"] == 0


def test_old_standalone_plan_cannot_bypass_new_gate(environment):
    runtime, _, tools = environment
    target = runtime.flow_gates["local"].effects["grant"]
    standalone = NetworkRuntime(runtime.journal_path)
    prepared = asyncio.run(standalone.prepare("lan", target.tool, dict(ARGUMENTS), l0_skill_id=target.skill_id))
    assert execute(runtime, prepared).state == PlanState.PRECONDITION_CHANGED
    assert not tools._LAN_ACCESS_CHANGES


def test_flow_effect_parameters_cannot_be_substituted(environment):
    runtime, _, tools = environment
    target = runtime.flow_gates["local"].effects["grant"]
    result = asyncio.run(runtime.prepare("lan", target.tool, {**ARGUMENTS, "reason": "different request"},
                                         l0_skill_id=target.skill_id, flow_gate_id="local"))
    assert result["status"] == "rejected" and not tools._LAN_ACCESS_CHANGES


def test_nonce_consumption_still_prevents_second_effect(environment):
    runtime, _, tools = environment
    prepared = prepare(runtime)
    assert execute(runtime, prepared).state == PlanState.VERIFIED_SUCCESS
    before = list(tools._LAN_ACCESS_CHANGES)
    with pytest.raises(NetworkRuntimeError):
        execute(runtime, prepared)
    assert tools._LAN_ACCESS_CHANGES == before


def test_delay_after_revalidation_cannot_dispatch_expired_flow(environment, monkeypatch):
    runtime, _, tools = environment
    prepared = prepare(runtime)
    import time
    original = time.monotonic

    def fault(stage, _plan):
        if stage == "before_send":
            monkeypatch.setattr("network_runtime.engine.time.monotonic", lambda: original() + 400)

    runtime.fault_hook = fault
    outcome = execute(runtime, prepared)
    assert not outcome.ok and not tools._LAN_ACCESS_CHANGES
    assert "write not sent" in outcome.error


def test_final_dispatch_boundary_rechecks_after_other_preflight_io(environment):
    runtime, dataset, tools = environment
    prepared = prepare(runtime)

    def fault(stage, _plan):
        if stage == "before_send":
            data = json.loads(dataset.read_text())
            data["campus-sw1"]["site"] = "idc"
            dataset.write_text(json.dumps(data))

    runtime.fault_hook = fault
    assert execute(runtime, prepared).state == PlanState.PRECONDITION_CHANGED
    assert not tools._LAN_ACCESS_CHANGES


def test_signed_approval_before_drift_does_not_waive_revalidation(environment):
    runtime, dataset, tools = environment
    prepared = prepare(runtime)
    proof = runtime.approval_control_plane.issue_local_compatibility_proof(
        PreparedPlan.from_dict(prepared["plan"]), approval_request_id="flow-test", approval_actor="unit-test-operator")
    data = json.loads(dataset.read_text())
    data["campus-sw1"]["site"] = "idc"
    dataset.write_text(json.dumps(data))
    assert execute(runtime, prepared, approval_proof=proof["approval_proof"]).state == PlanState.PRECONDITION_CHANGED
    assert not tools._LAN_ACCESS_CHANGES
