from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from dsh_adapter.worker import dispatch
from l1_runtime.contracts import (
    L1Decision,
    L1DecisionAction,
    L1DecisionEnvelope,
    L1DecisionEvidence,
)
from network_runtime.contracts import (
    ApprovalError,
    PlanIntegrityError,
    PlanState,
    PreparedPlan,
    sha256_json,
)
from network_runtime.engine import NetworkRuntime
from network_runtime.l0_skills import REGISTRY as L0_SKILLS


def _envelope(
    arguments: dict[str, object],
    *,
    decision_id: str = "l1-canary-test",
    mode: str = "canary",
    session_id: str = "session-canary",
    target: str = "restart_service",
    action: L1DecisionAction = L1DecisionAction.SELECT_TOOL,
    workflow: tuple[str, ...] = (),
) -> dict[str, object]:
    decision = L1Decision(
        action=action,
        target=target,
        arguments=arguments,
        workflow=workflow,
        confidence=0.9,
        reason_code="candidate_schema_00",
    )
    evidence = L1DecisionEvidence(
        prompt_digest=sha256_json({"direct_user_text": "private request"}),
        catalog_digest=sha256_json({"catalog": "test"}),
        candidate_digest=sha256_json({"candidates": [target]}),
        policy_digest=sha256_json({"policy": "test"}),
        model="qualification-model",
        model_attempts=1,
        input_tokens=10,
        output_tokens=4,
        token_usage_complete=True,
        selected_candidate_index=0,
        candidate_ids=((
            "skill" if action == L1DecisionAction.SELECT_SKILL else "tool"
        ) + f":{target}",),
        guard_action="allow",
        guard_reason="allow",
        protocol_valid=True,
        duration_ms=2.0,
    )
    envelope = L1DecisionEnvelope(
        decision_id=decision_id,
        mode=mode,
        profile="lan",
        session_id=session_id,
        harness="dsh",
        status="decided",
        decision=decision,
        evidence=evidence,
        decision_digest=decision.digest,
        evidence_digest=sha256_json(evidence.model_dump(by_alias=True, mode="json")),
    )
    return envelope.model_dump(by_alias=True, mode="json")


@pytest.fixture
def runtime_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[NetworkRuntime, Path]:
    journal = tmp_path / "runtime.sqlite"
    monkeypatch.setenv("NETOPYU_DSH_BACKEND", "mock")
    monkeypatch.setenv("NETOPYU_BACKEND", "mock")
    monkeypatch.setenv("NETOPYU_DSH_NETWORK_RUNTIME_STORE", str(journal))
    monkeypatch.setenv("NETOPYU_NETWORK_RUNTIME_STORE", str(journal))
    monkeypatch.setenv("NETOPYU_DSH_TOOL_RESULT_STORE", str(tmp_path / "results.sqlite"))
    return NetworkRuntime(journal), journal


async def _prepare(
    runtime: NetworkRuntime,
    envelope: dict[str, object],
    *,
    arguments: dict[str, object] | None = None,
    session_id: str = "session-canary",
    route: dict[str, str] | None = None,
) -> dict[str, object]:
    values = arguments or {"service": "crm", "environment": "staging"}
    contract = L0_SKILLS.for_tool("lan", "restart_service")
    assert contract is not None
    return await runtime.prepare(
        "lan",
        "restart_service",
        values,
        session_id=session_id,
        l0_skill_id=contract.skill_id,
        harness="dsh",
        l1_decision_envelope=envelope,
        l1_route_context=route or {"kind": "tool", "target": "restart_service"},
    )


@pytest.mark.asyncio
async def test_canary_decision_is_digest_bound_to_schema_v10_plan(
    runtime_environment: tuple[NetworkRuntime, Path],
) -> None:
    runtime, _journal = runtime_environment
    arguments = {"service": "crm", "environment": "staging"}
    prepared = await _prepare(runtime, _envelope(arguments), arguments=arguments)
    assert prepared["status"] == "plan_ready"
    plan = prepared["plan"]
    binding = plan["l1_decision_binding"]
    assert plan["schema_version"] == 10
    assert binding["authority"] == "proposal_only"
    assert binding["mode"] == "canary"
    assert binding["decision_id"] == "l1-canary-test"
    assert binding["bound_tool_name"] == "restart_service"
    assert binding["bound_l0_skill_id"] == "network.service.restart"
    assert binding["request_arguments_digest"] == sha256_json(arguments)
    assert binding["plan_arguments_digest"] == sha256_json(plan["arguments"])
    assert "private request" not in str(binding)
    assert runtime.audit(plan["plan_id"])["ok"] is True
    with pytest.raises(ApprovalError):
        await runtime.execute(
            plan_id=plan["plan_id"],
            plan_hash=plan["plan_hash"],
            execution_nonce=prepared["execution_nonce"],
            allow_destructive=True,
        )
    outcome = await runtime.execute(
        plan_id=plan["plan_id"],
        plan_hash=plan["plan_hash"],
        execution_nonce=prepared["execution_nonce"],
        approval_request_id="approval-binding-test",
        approval_actor="binding-test-approver",
        allow_destructive=True,
    )
    assert outcome.state == PlanState.VERIFIED_SUCCESS


@pytest.mark.asyncio
async def test_canary_binding_fails_closed_on_mode_session_route_argument_and_digest_drift(
    runtime_environment: tuple[NetworkRuntime, Path],
) -> None:
    runtime, _journal = runtime_environment
    arguments = {"service": "crm", "environment": "staging"}
    cases: list[tuple[str, dict[str, object], dict[str, object]]] = []

    shadow = _envelope(arguments, decision_id="l1-shadow", mode="shadow")
    cases.append(("mode", shadow, {}))
    wrong_session = _envelope(arguments, decision_id="l1-session")
    cases.append(("session", wrong_session, {"session_id": "different-session"}))
    wrong_route = _envelope(arguments, decision_id="l1-route")
    cases.append(("route", wrong_route, {"route": {"kind": "tool", "target": "other"}}))
    wrong_arguments = _envelope(arguments, decision_id="l1-arguments")
    cases.append(("arguments", wrong_arguments, {
        "arguments": {"service": "dns", "environment": "staging"},
    }))
    tampered = _envelope(arguments, decision_id="l1-digest")
    tampered["decision"]["target"] = "delete_resource"  # type: ignore[index]
    cases.append(("digest", tampered, {}))

    for label, envelope, overrides in cases:
        result = await _prepare(runtime, envelope, **overrides)
        assert result["status"] == "rejected", label
        assert result["ok"] is False, label


@pytest.mark.asyncio
async def test_one_decision_cannot_bind_two_runtime_plans(
    runtime_environment: tuple[NetworkRuntime, Path],
) -> None:
    runtime, _journal = runtime_environment
    arguments = {"service": "crm", "environment": "staging"}
    envelope = _envelope(arguments, decision_id="l1-single-use")
    first = await _prepare(runtime, envelope, arguments=arguments)
    second = await _prepare(runtime, envelope, arguments=arguments)
    assert first["status"] == "plan_ready"
    assert second["status"] == "rejected"
    assert "already bound" in " ".join(second["errors"])
    assert len(runtime.recent()) == 1


@pytest.mark.asyncio
async def test_selected_l1_skill_may_bind_only_a_declared_workflow_tool(
    runtime_environment: tuple[NetworkRuntime, Path],
) -> None:
    runtime, _journal = runtime_environment
    arguments = {"service": "crm", "environment": "staging"}
    envelope = _envelope(
        arguments,
        decision_id="l1-skill-route",
        target="restart-service",
        action=L1DecisionAction.SELECT_SKILL,
        workflow=("restart_service",),
    )
    prepared = await _prepare(
        runtime,
        envelope,
        arguments=arguments,
        route={"kind": "skill", "target": "restart-service"},
    )
    assert prepared["status"] == "plan_ready"
    binding = prepared["plan"]["l1_decision_binding"]
    assert binding["action"] == "select_skill"
    assert binding["route_target"] == "restart-service"
    assert binding["workflow"] == ["restart_service"]

    outside = _envelope(
        arguments,
        decision_id="l1-skill-outside",
        target="restart-service",
        action=L1DecisionAction.SELECT_SKILL,
        workflow=("grant_user_access",),
    )
    rejected = await _prepare(
        runtime,
        outside,
        arguments=arguments,
        route={"kind": "skill", "target": "restart-service"},
    )
    assert rejected["status"] == "rejected"
    assert "outside the selected Skill workflow" in " ".join(rejected["errors"])


@pytest.mark.asyncio
async def test_persisted_binding_tamper_fails_even_if_plan_hash_is_recomputed(
    runtime_environment: tuple[NetworkRuntime, Path],
) -> None:
    runtime, _journal = runtime_environment
    arguments = {"service": "crm", "environment": "staging"}
    prepared = await _prepare(runtime, _envelope(arguments), arguments=arguments)
    tampered = dict(prepared["plan"])
    tampered["l1_decision_binding"] = dict(tampered["l1_decision_binding"])
    tampered["l1_decision_binding"]["bound_tool_name"] = "delete_resource"
    immutable = dict(tampered)
    immutable.pop("plan_hash")
    immutable.pop("state")
    tampered["plan_hash"] = sha256_json(immutable)
    with pytest.raises(PlanIntegrityError, match="invalid L1 Decision binding"):
        PreparedPlan.from_dict(tampered)

    legacy = dict(prepared["plan"])
    legacy["schema_version"] = 9
    immutable = dict(legacy)
    immutable.pop("plan_hash")
    immutable.pop("state")
    immutable.pop("l1_decision_binding")
    legacy["plan_hash"] = sha256_json(immutable)
    with pytest.raises(PlanIntegrityError, match="legacy plan.*cannot carry"):
        PreparedPlan.from_dict(legacy)


@pytest.mark.asyncio
async def test_worker_runtime_prepare_carries_optional_binding(
    runtime_environment: tuple[NetworkRuntime, Path],
) -> None:
    _runtime, _journal = runtime_environment
    arguments = {"service": "crm", "environment": "staging"}
    contract = L0_SKILLS.for_tool("lan", "restart_service")
    assert contract is not None
    result = await dispatch({
        "id": "bound-prepare",
        "command": "runtime-prepare",
        "profile": "lan",
        "tool": "restart_service",
        "args": arguments,
        "session_id": "session-canary",
        "l0_skill_id": contract.skill_id,
        "harness": "dsh",
        "l1_decision_envelope": _envelope(arguments, decision_id="l1-worker"),
        "l1_route_context": {"kind": "tool", "target": "restart_service"},
    })
    assert result["status"] == "plan_ready"
    assert result["plan"]["l1_decision_binding"]["decision_id"] == "l1-worker"


def test_canary_remains_disabled_at_harness_configuration_boundary() -> None:
    from hermes_adapter.plugin import HermesAdapterConfig, NetOpYuHermesAdapter
    from hermes_adapter.client import HermesWorkerClient

    config = HermesAdapterConfig(
        profile="lan",
        socket_path=Path("/tmp/nonexistent-netopyu-worker.sock"),
        include_destructive=False,
        operator_id="test",
        own_agent_id="test",
        peer_urls=(),
        timeout_seconds=1,
        decision_mode="canary",
        decision_model="qualification-model",
    )
    with pytest.raises(ValueError, match="off or shadow"):
        NetOpYuHermesAdapter(HermesWorkerClient(config.socket_path), config)
