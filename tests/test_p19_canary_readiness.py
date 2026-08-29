from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import pytest

from l1_runtime.adapter_qualification import ADAPTER_QUALIFICATION_SCHEMA
from l1_runtime.canary_policy import CanaryPolicyResult, CanaryRoute, evaluate_canary_policy
from l1_runtime.canary_readiness import (
    CANARY_OPS_EVIDENCE_SCHEMA,
    CANARY_PRODUCT_EVIDENCE_SCHEMA,
    evaluate_canary_readiness,
)
from l1_runtime.contracts import (
    L1Decision,
    L1DecisionAction,
    L1DecisionEnvelope,
    L1DecisionEvidence,
)
from l1_runtime.qualification import QUALIFICATION_SCHEMA
from network_runtime.contracts import sha256_json


NOW = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_D = {
    "model_artifact_digest": sha256_json({"model": "immutable-7b"}),
    "sealed_manifest_digest": sha256_json({"manifest": "private"}),
    "consensus_labels_digest": sha256_json({"labels": "private"}),
    "catalog_snapshot_digest": sha256_json({"catalog": "reviewed"}),
}
_WORKER_REQUIREMENTS = {
    name: True for name in (
        "sealed_consensus_ready", "catalog_baseline_clean", "immutable_model_artifact",
        "label_coverage", "at_least_two_repetitions", "input_contract_parity",
        "decision_semantic_parity", "dsh_repeatability", "hermes_repeatability",
        "dsh_protocol", "hermes_protocol", "dsh_full_oracle", "hermes_full_oracle",
        "no_unsafe_escape", "all_expected_targets_retrieved",
    )
}
_ADAPTER_REQUIREMENTS = {
    name: True for name in (
        "sealed_consensus_ready", "catalog_baseline_clean", "immutable_model_artifact",
        "exact_case_coverage", "prompt_binding", "input_contract_parity",
        "decision_digest_parity", "protocol_success", "repeatability_when_requested",
    )
}


def _evidence(
    action: L1DecisionAction,
    *,
    target: str | None = "restart_service",
    mode: str = "canary",
    valid: bool = True,
) -> dict[str, object]:
    if action == L1DecisionAction.CLARIFY:
        decision = L1Decision(
            action=action,
            target=target,
            missing_fields=("environment",),
            confidence=0.8,
            reason_code="missing_required_fields",
        )
        guard = "allow"
        status = "policy_terminal"
    elif action in {L1DecisionAction.REFUSE, L1DecisionAction.OUT_OF_SCOPE}:
        decision = L1Decision(
            action=action,
            target=None,
            confidence=1.0,
            reason_code=action.value,
        )
        guard = "refuse" if action == L1DecisionAction.REFUSE else "out_of_scope"
        status = "policy_terminal"
    else:
        decision = L1Decision(
            action=action,
            target=target,
            arguments={"service": "crm"},
            confidence=0.9,
            reason_code="selected",
        )
        guard = "allow"
        status = "decided"
    identity = (
        f"{action.value.removeprefix('select_')}:{target}"
        if action in {L1DecisionAction.SELECT_SKILL, L1DecisionAction.SELECT_TOOL}
        else "tool:restart_service"
    )
    evidence = L1DecisionEvidence(
        prompt_digest=sha256_json({"prompt": "private"}),
        catalog_digest=sha256_json({"catalog": 1}),
        candidate_digest=sha256_json({"candidate": 1}),
        policy_digest=sha256_json({"policy": 1}),
        model="immutable-7b",
        model_attempts=1,
        input_tokens=10,
        output_tokens=3,
        token_usage_complete=True,
        selected_candidate_index=(0 if action.value.startswith("select_") else None),
        candidate_ids=(identity,),
        guard_action=guard,
        guard_reason=guard,
        protocol_valid=valid,
        duration_ms=1.0,
    )
    envelope = L1DecisionEnvelope(
        decision_id="decision-sensitive-id",
        mode=mode,
        profile="lan",
        session_id="session-1",
        harness="dsh",
        status=status,
        decision=decision,
        evidence=evidence,
        decision_digest=decision.digest,
        evidence_digest=sha256_json(evidence.model_dump(by_alias=True, mode="json")),
    )
    return envelope.model_dump(by_alias=True, mode="json")


def _route(*, operation: str = "write", target: str = "restart_service") -> CanaryRoute:
    return CanaryRoute(
        kind="tool",
        target=target,
        operation=operation,
        profile="lan",
        harness="dsh",
        session_id="session-1",
    )


def test_canary_policy_only_preserves_or_blocks_the_original_route() -> None:
    matched = evaluate_canary_policy(
        _evidence(L1DecisionAction.SELECT_TOOL), route=_route(),
    )
    assert matched.status == "continue_original_route"
    assert matched.effect == "unchanged"
    assert matched.authority_granted is False
    assert matched.route_rewritten is False
    assert matched.arguments_rewritten is False
    assert matched.runtime_admission_required is True
    tampered = matched.model_dump(by_alias=True, mode="json")
    tampered["policy_digest"] = sha256_json({"tamper": True})
    with pytest.raises(ValueError, match="digest"):
        CanaryPolicyResult.model_validate(tampered)

    mismatch = evaluate_canary_policy(
        _evidence(L1DecisionAction.SELECT_TOOL),
        route=_route(target="grant_user_access"),
    )
    assert mismatch.status == "blocked"
    assert mismatch.effect == "narrowed"
    assert mismatch.reason_code == "selection_mismatch"


@pytest.mark.parametrize(
    ("action", "reason"),
    (
        (L1DecisionAction.CLARIFY, "decision_requires_clarification"),
        (L1DecisionAction.REFUSE, "decision_refused_route"),
        (L1DecisionAction.OUT_OF_SCOPE, "decision_out_of_scope"),
    ),
)
def test_terminal_canary_decisions_can_only_narrow(
    action: L1DecisionAction,
    reason: str,
) -> None:
    result = evaluate_canary_policy(_evidence(action), route=_route())
    assert result.status == "blocked"
    assert result.effect == "narrowed"
    assert result.reason_code == reason


def test_invalid_canary_material_fails_closed_for_writes_and_is_observed_for_reads() -> None:
    malformed = _evidence(L1DecisionAction.SELECT_TOOL)
    malformed["decision"]["target"] = "tampered"  # type: ignore[index]
    write = evaluate_canary_policy(malformed, route=_route(operation="write"))
    read = evaluate_canary_policy(malformed, route=_route(operation="read"))
    assert write.status == "blocked"
    assert write.reason_code == "invalid_decision_write_blocked"
    assert read.status == "continue_original_route"
    assert read.reason_code == "invalid_decision_read_observed"

    shadow = evaluate_canary_policy(
        _evidence(L1DecisionAction.SELECT_TOOL, mode="shadow"), route=_route(),
    )
    assert shadow.status == "blocked"
    assert shadow.reason_code == "canary_context_mismatch"


def _qualified_reports() -> tuple[dict[str, object], dict[str, object]]:
    worker_body: dict[str, object] = {
        "scope": {
            "level": "shared_worker_decision_contract",
            "full_harness_adapter_loop": False,
            "harnesses": ["dsh", "hermes"],
        },
        "status": "qualified",
        "qualified": True,
        "case_count": 24,
        "repetitions": 2,
        "execution_count": 96,
        "model": "qwen3.6:7b",
        **_D,
        "requirements": _WORKER_REQUIREMENTS,
        "privacy": {
            "raw_prompts_emitted": False,
            "raw_labels_emitted": False,
            "argument_values_emitted": False,
        },
    }
    worker = {
        "apiVersion": QUALIFICATION_SCHEMA,
        **worker_body,
        "report_digest": sha256_json(worker_body),
    }
    adapter_body: dict[str, object] = {
        "scope": {
            "level": "adapter_hook_to_worker",
            "dsh_javascript_agent_pre_step": True,
            "hermes_python_pre_llm_call": True,
            "persistent_owner_only_worker": True,
            "real_dsh_web_process": False,
            "real_hermes_process": False,
        },
        "status": "adapter_parity_passed",
        "passed": True,
        "case_count": 24,
        "repetitions": 1,
        "model": "qwen3.6:7b",
        **_D,
        "requirements": _ADAPTER_REQUIREMENTS,
        "privacy": {
            "raw_prompts_emitted": False,
            "raw_labels_emitted": False,
            "argument_values_emitted": False,
        },
    }
    adapter = {
        "apiVersion": ADAPTER_QUALIFICATION_SCHEMA,
        **adapter_body,
        "report_digest": sha256_json(adapter_body),
    }
    return worker, adapter


def _external_evidence(
    worker: dict[str, object], adapter: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    product_body: dict[str, object] = {
        "apiVersion": CANARY_PRODUCT_EVIDENCE_SCHEMA,
        "issued_at": "2026-08-29T11:00:00Z",
        "expires_at": "2026-08-30T11:00:00Z",
        "bindings": _D,
        "worker_report_digest": worker["report_digest"],
        "adapter_report_digest": adapter["report_digest"],
        "surfaces": [
            {
                "harness": "dsh",
                "entrypoint": "web_ui",
                "real_process_exercised": True,
                "decision_receipt_verified": True,
                "interaction_slo_met": True,
                "distribution_digest": sha256_json({"dsh": "distribution"}),
                "deployment_identity_digest": sha256_json({"dsh": "deployment"}),
                "test_receipt_digest": sha256_json({"dsh": "receipt"}),
            },
            {
                "harness": "hermes",
                "entrypoint": "cli",
                "real_process_exercised": True,
                "decision_receipt_verified": True,
                "interaction_slo_met": True,
                "distribution_digest": sha256_json({"hermes": "distribution"}),
                "deployment_identity_digest": sha256_json({"hermes": "deployment"}),
                "test_receipt_digest": sha256_json({"hermes": "receipt"}),
            },
        ],
        "reviewer_ids": ["secret-product-reviewer-a", "secret-product-reviewer-b"],
    }
    product = {**product_body, "evidence_digest": sha256_json(product_body)}
    def control(name: str) -> dict[str, object]:
        return {
            "configured": True,
            "tested": True,
            "receipt_digest": sha256_json({"control": f"private-{name}-receipt"}),
        }
    ops_body: dict[str, object] = {
        "apiVersion": CANARY_OPS_EVIDENCE_SCHEMA,
        "issued_at": "2026-08-29T11:30:00Z",
        "expires_at": "2026-08-30T11:30:00Z",
        "bindings": _D,
        "product_evidence_digest": product["evidence_digest"],
        "limits": {
            "max_traffic_percent": 1.0,
            "max_duration_minutes": 30,
            "automatic_approval_enabled": False,
            "runtime_bypass_enabled": False,
            "provider_bypass_enabled": False,
        },
        "kill_switch_to_shadow": control("kill"),
        "rollback_to_shadow": control("rollback"),
        "alert_delivery": control("alert"),
        "no_effect_replay": control("replay"),
        "core_controls": {
            "passed": 64,
            "total": 64,
            "report_digest": sha256_json({"core": "64-of-64"}),
        },
        "runtime_trend": {
            "status": "stable",
            "distinct_implementation_versions": 3,
            "p50_ms": 7.6,
            "p95_ms": 8.7,
            "p50_within_threshold": True,
            "p95_within_threshold": True,
            "report_digest": sha256_json({"trend": "stable-three-versions"}),
        },
        "decision_plan_binding": {
            "passed": 12,
            "total": 12,
            "decision_replay_count": 0,
            "authority_escape_count": 0,
            "report_digest": sha256_json({"binding": "12-of-12"}),
        },
        "owner_ids": ["secret-canary-owner", "secret-incident-owner"],
    }
    ops = {**ops_body, "evidence_digest": sha256_json(ops_body)}
    return product, ops


def test_readiness_gate_accepts_only_cross_bound_evidence_and_emits_no_private_values() -> None:
    worker, adapter = _qualified_reports()
    product, ops = _external_evidence(worker, adapter)
    report = evaluate_canary_readiness(
        worker, adapter, product, ops, checked_at=NOW,
    )
    assert report["status"] == "ready_for_review"
    assert report["activation_authorized"] is False
    assert report["configuration_changed"] is False
    assert report["traffic_changed"] is False
    assert all(report["checks"].values())
    rendered = json.dumps(report, ensure_ascii=False)
    for private in (
        "secret-product-reviewer-a", "secret-canary-owner", "private-kill-receipt",
    ):
        assert private not in rendered

    malformed_worker = json.loads(json.dumps(worker))
    malformed_worker["scope"] = None
    malformed_body = dict(malformed_worker)
    malformed_body.pop("apiVersion")
    malformed_body.pop("report_digest")
    malformed_worker["report_digest"] = sha256_json(malformed_body)
    malformed = evaluate_canary_readiness(
        malformed_worker, adapter, product, ops, checked_at=NOW,
    )
    assert malformed["status"] == "not_ready"

    non_finite_worker = dict(worker)
    non_finite_worker["non_finite"] = float("nan")
    non_finite = evaluate_canary_readiness(
        non_finite_worker, adapter, product, ops, checked_at=NOW,
    )
    assert non_finite["status"] == "not_ready"


@pytest.mark.parametrize(
    "mutation",
    ("missing_worker", "tampered_adapter", "expired_product", "failed_drill", "binding_drift"),
)
def test_readiness_gate_fails_closed_for_missing_tampered_stale_or_unbound_evidence(
    mutation: str,
) -> None:
    worker, adapter = _qualified_reports()
    product, ops = _external_evidence(worker, adapter)
    if mutation == "missing_worker":
        worker = None  # type: ignore[assignment]
    elif mutation == "tampered_adapter":
        adapter["passed"] = False
    elif mutation == "expired_product":
        product["issued_at"] = "2026-08-20T00:00:00Z"
        product["expires_at"] = "2026-08-21T00:00:00Z"
        body = dict(product)
        body.pop("evidence_digest")
        product["evidence_digest"] = sha256_json(body)
    elif mutation == "failed_drill":
        ops["kill_switch_to_shadow"]["tested"] = False  # type: ignore[index]
        body = dict(ops)
        body.pop("evidence_digest")
        ops["evidence_digest"] = sha256_json(body)
    elif mutation == "binding_drift":
        ops["bindings"] = {**_D, "catalog_snapshot_digest": sha256_json({"drift": 1})}
        body = dict(ops)
        body.pop("evidence_digest")
        ops["evidence_digest"] = sha256_json(body)
    report = evaluate_canary_readiness(
        worker, adapter, product, ops, checked_at=NOW,
    )
    assert report["status"] == "not_ready"
    assert report["activation_authorized"] is False
    assert report["reason_codes"]


def test_canary_readiness_cli_has_no_configuration_side_effect(tmp_path: Path) -> None:
    worker, adapter = _qualified_reports()
    product, ops = _external_evidence(worker, adapter)
    paths = []
    for name, value in (
        ("worker", worker), ("adapter", adapter), ("product", product), ("ops", ops),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths.append(path)
    protected = (
        PROJECT_ROOT / "dsh-plugin-netopyu" / "src" / "index.js",
        PROJECT_ROOT / "hermes_adapter" / "plugin.py",
    )
    before = {path: path.read_bytes() for path in protected}
    result = subprocess.run(
        [str(PROJECT_ROOT / "scripts" / "netopyu-dsh"), "l1-canary-readiness",
         *(str(path) for path in paths)],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "ready_for_review"
    assert before == {path: path.read_bytes() for path in protected}
