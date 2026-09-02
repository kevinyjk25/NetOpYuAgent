from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from effect_runtime.reliability import (
    AutonomyDecision,
    EvidenceRecord,
    EvidenceRequirement,
    ExecutionPhase,
    Guard,
    OperationNode,
    Reversibility,
    RiskFactors,
    RiskPolicy,
    TransactionStateMachine,
    TypedExecutionGraph,
    build_transaction_graph,
    contract_from_compiled_l0,
    evaluate_evidence,
    evaluate_guards,
)
from effect_runtime.graph_scheduler import (
    GraphScheduleError,
    NodeOutcome,
    TypedGraphScheduler,
    graph_from_step_contract,
)


def test_no_evidence_no_action_checks_provenance_scope_and_freshness() -> None:
    now = datetime.now(timezone.utc)
    requirement = EvidenceRequirement(
        id="peer-state",
        semantic_type="network.bgp.peer-state",
        source_capability="observer.bgp.peer-state@1",
        phase=ExecutionPhase.PRECHECK,
        max_age_seconds=60,
        scope=("router:r1", "peer:10.0.0.1"),
        associated_action="network.bgp.peer.reset",
    )
    record = EvidenceRecord.create(
        id="e1",
        semantic_type="network.bgp.peer-state",
        source_capability="observer.bgp.peer-state@1",
        collector_identity="observer-a#sha256:abc",
        collected_at=now.isoformat(),
        scope=("router:r1", "peer:10.0.0.1"),
        associated_action="network.bgp.peer.reset",
        payload={"state": "idle"},
    )
    assert evaluate_evidence((requirement,), (record,), now=now).allowed

    stale = EvidenceRecord.create(
        id="e2",
        semantic_type=record.semantic_type,
        source_capability=record.source_capability,
        collector_identity=record.collector_identity,
        collected_at=(now - timedelta(seconds=61)).isoformat(),
        scope=record.scope,
        associated_action=record.associated_action,
        payload=record.payload,
    )
    result = evaluate_evidence((requirement,), (stale,), now=now)
    assert not result.allowed
    assert result.stale == ("peer-state",)


def test_transaction_state_machine_has_explicit_commit_and_recovery_paths() -> None:
    machine = TransactionStateMachine()
    for phase in (
        ExecutionPhase.SNAPSHOT,
        ExecutionPhase.PRECHECK,
        ExecutionPhase.AWAITING_APPROVAL,
        ExecutionPhase.REVALIDATE,
        ExecutionPhase.EXECUTE,
        ExecutionPhase.VERIFY,
        ExecutionPhase.COMMIT,
    ):
        machine.transition(phase, evidence=phase.value)
    assert machine.phase == ExecutionPhase.COMMIT

    recovery = TransactionStateMachine()
    for phase in (
        ExecutionPhase.SNAPSHOT,
        ExecutionPhase.PRECHECK,
        ExecutionPhase.REVALIDATE,
        ExecutionPhase.EXECUTE,
        ExecutionPhase.COMPENSATE,
        ExecutionPhase.VERIFY_RECOVERY,
        ExecutionPhase.ABORT,
    ):
        recovery.transition(phase)
    assert recovery.phase == ExecutionPhase.ABORT
    with pytest.raises(ValueError, match="illegal EnsuredSkill transition"):
        recovery.transition(ExecutionPhase.EXECUTE)


def test_typed_graph_scheduler_is_the_fail_closed_happy_path_gate() -> None:
    graph = build_transaction_graph(compensatable=True)
    scheduler = TypedGraphScheduler(graph)
    for node_id in (
        "snapshot", "precheck", "approval", "revalidate", "execute", "verify", "commit",
    ):
        scheduler.start(node_id)
        scheduler.finish(node_id, NodeOutcome.SUCCEEDED)
    assert scheduler.machine.phase == ExecutionPhase.COMMIT
    with pytest.raises(GraphScheduleError, match="cannot run more than once"):
        scheduler.start("execute")


def test_typed_graph_scheduler_blocks_false_commit_and_controls_recovery() -> None:
    graph = build_transaction_graph(compensatable=True)
    scheduler = TypedGraphScheduler(graph)
    for node_id in ("snapshot", "precheck", "approval", "revalidate", "execute"):
        scheduler.start(node_id)
        scheduler.finish(node_id, NodeOutcome.SUCCEEDED)
    scheduler.start("verify")
    scheduler.finish("verify", NodeOutcome.FAILED, evidence_ids=("evidence:verify",))
    with pytest.raises(GraphScheduleError, match="commit requires"):
        scheduler.start("commit")
    scheduler.start("compensate")
    scheduler.finish("compensate", NodeOutcome.SUCCEEDED)
    scheduler.start("verify_recovery")
    scheduler.finish("verify_recovery", NodeOutcome.SUCCEEDED)
    scheduler.start("abort")
    scheduler.finish("abort", NodeOutcome.SUCCEEDED)
    assert scheduler.machine.phase == ExecutionPhase.ABORT


def test_typed_graph_scheduler_allows_safe_abort_after_successful_revalidation() -> None:
    scheduler = TypedGraphScheduler(build_transaction_graph(compensatable=False))
    for node_id in ("snapshot", "precheck", "approval", "revalidate"):
        scheduler.start(node_id)
        scheduler.finish(node_id, NodeOutcome.SUCCEEDED)
    scheduler.start("abort")
    scheduler.finish("abort", NodeOutcome.SUCCEEDED)
    assert scheduler.machine.phase == ExecutionPhase.ABORT


def test_step_contract_reconstruction_rejects_unknown_dependencies() -> None:
    with pytest.raises(ValueError, match="unknown dependency"):
        graph_from_step_contract(({
            "step_id": "execute",
            "phase": "execute",
            "depends_on": ["not-reviewed"],
            "side_effect": True,
        },))


def test_typed_transaction_graph_binds_side_effect_and_compensation() -> None:
    graph = build_transaction_graph(compensatable=True)
    order = graph.topological_order()
    assert order.index("revalidate") < order.index("execute") < order.index("verify")
    assert order.index("execute") < order.index("compensate") < order.index("verify_recovery")
    assert graph.graph_digest.startswith("sha256:")
    side_effects = {item.id for item in graph.nodes if item.side_effect}
    assert side_effects == {"execute", "compensate"}

    irreversible = build_transaction_graph(compensatable=False)
    assert "abort" in {item.id for item in irreversible.nodes}
    assert "compensate" not in {item.id for item in irreversible.nodes}

    with pytest.raises(ValueError, match="invalid phase"):
        TypedExecutionGraph.create((
            OperationNode("bad", ExecutionPhase.PRECHECK, side_effect=True),
        ))


def test_guards_are_deterministic_and_fail_closed_on_unknown_fields() -> None:
    requirement = EvidenceRequirement(
        id="link-state",
        semantic_type="network.link.state",
        source_capability="network.link.observe",
        phase=ExecutionPhase.PRECHECK,
        max_age_seconds=60,
        associated_action="network.link.disable",
    )
    record = EvidenceRecord.create(
        id="e-link",
        semantic_type=requirement.semantic_type,
        source_capability=requirement.source_capability,
        collector_identity="observer-a",
        collected_at=datetime.now(timezone.utc).isoformat(),
        scope=("device:r1",),
        associated_action=requirement.associated_action,
        payload={"facts": {"affected_prefixes": 10}, "passed": True},
    )
    passing = Guard(
        id="blast-radius",
        field="facts.affected_prefixes",
        operator="less_than",
        expected=100,
        evidence_requirement_id=requirement.id,
    )
    assert evaluate_guards((passing,), (requirement,), (record,)).allowed

    missing = Guard(
        id="unknown-dependency",
        field="facts.critical_service_dependency",
        operator="equals",
        expected=False,
        evidence_requirement_id=requirement.id,
    )
    result = evaluate_guards((missing,), (requirement,), (record,))
    assert not result.allowed
    assert result.failed == ("unknown-dependency",)


def test_risk_policy_returns_execute_ask_or_reject_with_reasons() -> None:
    policy = RiskPolicy()
    low = policy.evaluate(RiskFactors(
        change_scope=1,
        blast_radius=1,
        evidence_confidence=1.0,
        reversibility=Reversibility.STRONG,
        historical_success=1.0,
        service_criticality=0,
    ))
    assert low.decision == AutonomyDecision.EXECUTE

    medium = policy.evaluate(RiskFactors(
        change_scope=2,
        blast_radius=2,
        evidence_confidence=0.9,
        reversibility=Reversibility.CONDITIONAL,
        historical_success=0.8,
        service_criticality=2,
    ))
    assert medium.decision == AutonomyDecision.ASK_HUMAN

    unsafe = policy.evaluate(RiskFactors(
        change_scope=4,
        blast_radius=20,
        evidence_confidence=0.4,
        reversibility=Reversibility.IRREVERSIBLE,
        historical_success=0.2,
        service_criticality=4,
    ))
    assert unsafe.decision == AutonomyDecision.REJECT
    assert any(item.startswith("evidence_uncertainty=") for item in unsafe.reasons)


def test_reviewed_network_l0_projects_into_material_contract() -> None:
    from network_runtime.l0_skills import REGISTRY

    reviewed = next(
        item for item in REGISTRY.contracts()
        if item.compiled_contract is not None
    )
    contract = contract_from_compiled_l0(reviewed.compiled_contract)
    assert contract.operation == reviewed.compiled_contract.spec.effect.capability
    assert contract.evidence
    assert contract.postconditions
    assert contract.resources.writes
    assert contract.contract_digest.startswith("sha256:")


def test_native_agent_mutation_fallback_is_evaluation_only() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        if relative.parts[0] in {"evaluation", "tests"}:
            continue
        if "l1_agent_fallback" in path.read_text(encoding="utf-8"):
            offenders.append(str(relative))
    assert offenders == []
