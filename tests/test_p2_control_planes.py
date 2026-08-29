from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import subprocess

import pytest

from effect_runtime.saga import SagaCoordinator, SagaDefinition, SagaStepSpec
from l1_runtime.contracts import (
    L1Decision,
    L1DecisionAction,
    L1DecisionEnvelope,
    L1DecisionEvidence,
)
from l1_runtime.store import DecisionStore
from network_runtime.catalog_control import (
    CatalogGovernanceError,
    bootstrap_runtime_governance_catalog,
    catalog_compatibility_report,
    dump_governance_catalog,
    evaluate_catalog_access,
    load_governance_catalog,
    seal_governance_catalog,
    validate_runtime_catalog_binding,
)
from network_runtime.contracts import Evidence, PlanState, PreparedPlan, RiskLevel, sha256_json
from network_runtime.evidence_plane import (
    EvidencePlaneError,
    analyze_evidence_trend,
    collect_evidence_snapshot,
    export_evidence_html,
    render_evidence_html,
)
from network_runtime.journal import NetworkJournal
from network_runtime.l0.promotion import package_promotion


ROOT = Path(__file__).resolve().parents[1]
PROMOTION = ROOT / "network_runtime" / "l0" / "promotion_examples" / "url1-network-access"
CANDIDATE = ROOT / "network_runtime" / "l0" / "examples" / "s1-network-access-grant.yaml"


def _catalog_body():
    catalog = bootstrap_runtime_governance_catalog()
    value = catalog.model_dump(by_alias=True, mode="json")
    value.pop("catalogHash")
    return value


def test_p21_bootstrap_covers_all_runtime_contracts_and_separates_authority() -> None:
    catalog = bootstrap_runtime_governance_catalog()
    persisted = load_governance_catalog(ROOT / "data" / "capability_governance_catalog.yaml")
    report = validate_runtime_catalog_binding(catalog)
    assert report["ok"] is True
    assert report["runtime_contracts"] == 21
    assert len(catalog.capabilities) == 21
    assert len(catalog.delegations) == 9
    assert persisted.catalog_hash == catalog.catalog_hash
    assert all(item.owner_team != item.steward_team for item in catalog.capabilities)
    assert all(not ({"review", "publish"} <= set(item.actions)) for item in catalog.delegations)
    assert report["authority"] == "catalog_governance_only"
    assert report["runtime_read_authority"] is False
    assert report["runtime_effect_authority"] is False


def test_p21_delegation_allows_only_catalog_workflow_and_expires() -> None:
    catalog = bootstrap_runtime_governance_catalog()
    allowed = evaluate_catalog_access(
        catalog, team_id="lan-operations", action="propose_write",
        capability_id="network.lan.user-access.grant", version="1.0.0",
        tenant="local-lab", environment="local-simulation",
    )
    assert allowed["allowed"] is True
    assert allowed["runtime_read_authority"] is False
    assert allowed["runtime_effect_authority"] is False
    assert allowed["provider_publication_authority"] is False
    assert "lan-operations" not in json.dumps(allowed)

    denied = evaluate_catalog_access(
        catalog, team_id="dc-operations", action="publish",
        capability_id="network.lan.user-access.grant", version="1.0.0",
        tenant="local-lab", environment="local-simulation",
    )
    assert denied["allowed"] is False
    assert "no_governance_grant" in denied["reasons"]

    body = _catalog_body()
    for delegation in body["delegations"]:
        if delegation["id"] == "network-lan-proposal":
            delegation["expiresAt"] = "2020-01-01T00:00:00+00:00"
    expired = seal_governance_catalog(body)
    decision = evaluate_catalog_access(
        expired, team_id="lan-operations", action="propose_write",
        capability_id="network.lan.user-access.grant", version="1.0.0",
        tenant="local-lab", environment="local-simulation",
        at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert decision["allowed"] is False

    read_binding = evaluate_catalog_access(
        catalog, team_id="network-platform", action="bind_read",
        capability_id="network.lan.user-access.grant", version="1.0.0",
        tenant="local-lab", environment="local-simulation",
    )
    assert read_binding["allowed"] is False
    assert "capability_kind_action_mismatch" in read_binding["reasons"]


def test_p21_catalog_integrity_compatibility_and_consumer_impact(tmp_path: Path) -> None:
    catalog = bootstrap_runtime_governance_catalog()
    path = tmp_path / "catalog.yaml"
    path.write_text(dump_governance_catalog(catalog), encoding="utf-8")
    assert load_governance_catalog(path).catalog_hash == catalog.catalog_hash
    tampered = path.read_text(encoding="utf-8").replace("Network Platform", "Other Platform")
    path.write_text(tampered, encoding="utf-8")
    with pytest.raises(CatalogGovernanceError, match="invalid"):
        load_governance_catalog(path)

    body = _catalog_body()
    body["metadata"]["version"] = "2.0.0"
    target = next(
        item for item in body["capabilities"]
        if item["id"] == "network.lan.user-access.grant"
    )
    target["contractHash"] = sha256_json({"changed": True})
    for consumer in target["consumers"]:
        consumer["contractHash"] = target["contractHash"]
    candidate = seal_governance_catalog(body)
    report = catalog_compatibility_report(catalog, candidate)
    assert report["compatible"] is False
    assert any(item["code"] == "IN_PLACE_CONTRACT_OR_SCOPE_CHANGE" for item in report["breaking_changes"])
    assert report["consumer_impacts"]
    assert report["activation_available"] is False


def test_p21_rejects_self_delegation_combined_review_publish_and_symlink(tmp_path: Path) -> None:
    body = _catalog_body()
    delegation = body["delegations"][0]
    delegation["delegateeTeam"] = delegation["delegatorTeam"]
    with pytest.raises(CatalogGovernanceError, match="invalid"):
        seal_governance_catalog(body)

    body = _catalog_body()
    body["delegations"][0]["actions"] = ["review", "publish"]
    with pytest.raises(CatalogGovernanceError, match="invalid"):
        seal_governance_catalog(body)

    real = tmp_path / "real.yaml"
    real.write_text(dump_governance_catalog(bootstrap_runtime_governance_catalog()), encoding="utf-8")
    linked = tmp_path / "linked.yaml"
    linked.symlink_to(real)
    with pytest.raises(CatalogGovernanceError, match="unsafe"):
        load_governance_catalog(linked)


def test_p21_rejects_unknown_drifted_and_cyclic_dependencies() -> None:
    body = _catalog_body()
    first, second = body["capabilities"][:2]
    first["dependencies"] = [{
        "capabilityId": "network.unknown", "version": "1.0.0",
        "contractHash": sha256_json({"unknown": True}),
    }]
    with pytest.raises(CatalogGovernanceError, match="invalid"):
        seal_governance_catalog(body)

    body = _catalog_body()
    first, second = body["capabilities"][:2]
    first["dependencies"] = [{
        "capabilityId": second["id"], "version": second["version"],
        "contractHash": sha256_json({"drift": True}),
    }]
    with pytest.raises(CatalogGovernanceError, match="invalid"):
        seal_governance_catalog(body)

    body = _catalog_body()
    first, second = body["capabilities"][:2]
    first["dependencies"] = [{
        "capabilityId": second["id"], "version": second["version"],
        "contractHash": second["contractHash"],
    }]
    second["dependencies"] = [{
        "capabilityId": first["id"], "version": first["version"],
        "contractHash": first["contractHash"],
    }]
    with pytest.raises(CatalogGovernanceError, match="invalid"):
        seal_governance_catalog(body)


def _plan(plan_id: str) -> PreparedPlan:
    created = datetime.now(timezone.utc)
    digest = sha256_json({"plan": plan_id})
    requester = {"subject_id": "secret-requester"}
    evidence = Evidence(
        evidence_type="preflight", source="test-observer",
        target="secret-target", observed_at=created.isoformat(),
        value={"safe": True}, passed=True,
    )
    return PreparedPlan.create(
        plan_id=plan_id, profile="lan", tool_name="test_effect", tool_version="1.0.0",
        action_type="reversible", provider_identity="test-provider",
        provider_release_digest=digest, provider_manifest_digest=digest,
        provider_qualification_digest=digest, provider_deployment_digest=digest,
        input_schema_digest=digest, output_schema_digest=digest,
        capability_id="network.test.effect", capability_version="1.0.0",
        provider_role="actor", arguments={"secret": "never-project-this"},
        argument_provenance={"secret": "user"}, targets=("secret-target",),
        risk_level=RiskLevel.HIGH, risk_reasons=("test",), preflight=(evidence,),
        verification_contract="test-verifier", rollback_contract="test-rollback",
        l0_skill_id="network.test.effect", l0_skill_version="1.0.0",
        l0_contract_hash=digest, intent_spec={"kind": "test"}, intent_hash=digest,
        step_contract=(), workflow_run_id=None, workflow_template_hash=None,
        requester_identity=requester, requester_digest=sha256_json(requester),
        approval_mode="single", approval_policy_id="test-policy",
        approval_policy_version="1.0.0", approval_policy_hash=digest,
        l1_decision_binding=None, created_at=created.isoformat(),
        expires_at=(created + timedelta(minutes=5)).isoformat(),
    )


def _runtime_store(path: Path) -> None:
    with NetworkJournal(path) as journal:
        first = _plan("secret-success-plan")
        journal.create(first, "secret-nonce-1")
        journal.claim_for_execution(
            plan_id=first.plan_id, plan_hash=first.plan_hash,
            execution_nonce="secret-nonce-1", approval_request_id="secret-approval",
            approval_actor="secret-approver",
        )
        journal.transition(first.plan_id, PlanState.VERIFYING, "verification_started")
        journal.transition(first.plan_id, PlanState.VERIFIED_SUCCESS, "verification_passed")
        journal.release_locks(first.plan_id)

        second = _plan("secret-manual-plan")
        journal.create(second, "secret-nonce-2")
        journal.claim_for_execution(
            plan_id=second.plan_id, plan_hash=second.plan_hash,
            execution_nonce="secret-nonce-2", approval_request_id="secret-approval-2",
            approval_actor="secret-approver-2",
        )
        journal.transition(second.plan_id, PlanState.EXECUTION_FAILED, "write_failed")
        journal.transition(
            second.plan_id, PlanState.MANUAL_INTERVENTION_REQUIRED,
            "runtime_exception_escalated",
        )


def _decision_store(path: Path) -> None:
    decision = L1Decision(
        action=L1DecisionAction.SELECT_SKILL, target="network-test",
        arguments={}, confidence=0.9, reason_code="test",
    )
    evidence = L1DecisionEvidence(
        prompt_digest=sha256_json({"secret-prompt": "never-project-this"}),
        catalog_digest=sha256_json({"catalog": 1}),
        candidate_digest=sha256_json({"candidate": 1}),
        policy_digest=sha256_json({"policy": 1}), model="test-model",
        model_attempts=1, input_tokens=10, output_tokens=2, token_usage_complete=True,
        selected_candidate_index=0, candidate_ids=("skill:network-test",),
        guard_action="allow", guard_reason="test", protocol_valid=True,
        duration_ms=12.5,
    )
    envelope = L1DecisionEnvelope(
        decision_id="secret-decision-id", mode="shadow", profile="lan",
        session_id="secret-session-id", harness="dsh", status="decided",
        decision=decision, evidence=evidence, decision_digest=decision.digest,
        evidence_digest=sha256_json(evidence.model_dump(by_alias=True, mode="json")),
    )
    store = DecisionStore(path)
    store.record(envelope)
    store.observe(
        decision_id=envelope.decision_id, session_id=envelope.session_id,
        observed_kind="skill", observed_target="network-test", observed_arguments={},
    )


def _saga_store(path: Path) -> None:
    definition = SagaDefinition.create(
        "test-saga", "1.0.0",
        (SagaStepSpec("step-one", "network", "network.test.effect"),),
    )
    with SagaCoordinator(path) as coordinator:
        started = coordinator.start(definition, correlation_id="secret-correlation")
        coordinator.bind_plan(
            started["saga_id"], "step-one",
            plan_id="secret-saga-plan", plan_hash=sha256_json({"plan": "saga"}),
        )
        coordinator.record_outcome(
            started["saga_id"], "step-one", terminal_state="verified_success",
        )


def _provider_store(path: Path) -> None:
    with sqlite3.connect(path) as database:
        database.execute(
            "CREATE TABLE release_events(event_id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "event_type TEXT, release_digest TEXT, payload_json TEXT, prev_event_hash TEXT, "
            "event_hash TEXT, created_at TEXT)"
        )
        previous = "GENESIS"
        for event_type in ("release_promoted", "release_rolled_back"):
            created = datetime.now(timezone.utc).isoformat()
            payload = json.dumps({"secret": "never-project-this"}, separators=(",", ":"))
            event_hash = sha256_json({
                "event_type": event_type, "release_digest": "secret-release",
                "payload_json": payload, "prev_event_hash": previous,
                "created_at": created,
            })
            database.execute(
                "INSERT INTO release_events(event_type, release_digest, payload_json, "
                "prev_event_hash, event_hash, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (event_type, "secret-release", payload, previous, event_hash, created),
            )
            previous = event_hash


def _promotion_store(root: Path) -> None:
    package_promotion(
        skill_path=PROMOTION / "SKILL.md", candidate_path=CANDIDATE,
        capability_catalog_path=PROMOTION / "capabilities.yaml",
        l05_path=PROMOTION / "L0.5.yaml", output_directory=root / "secret-proposal-name",
    )


def _snapshot(tmp_path: Path) -> dict:
    runtime = tmp_path / "runtime.sqlite"
    decisions = tmp_path / "decisions.sqlite"
    sagas = tmp_path / "sagas.sqlite"
    providers = tmp_path / "providers.sqlite"
    proposals = tmp_path / "proposals"
    proposals.mkdir()
    _runtime_store(runtime)
    _decision_store(decisions)
    _saga_store(sagas)
    _provider_store(providers)
    _promotion_store(proposals)
    return collect_evidence_snapshot(
        runtime_journals=[runtime], decision_stores=[decisions], saga_stores=[sagas],
        provider_registries=[providers], proposal_roots=[proposals],
        generated_at="2026-08-30T00:00:00+00:00",
    )


def test_p22_collects_five_sources_metrics_incidents_and_no_raw_private_values(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)
    assert snapshot["status"] == "valid"
    assert snapshot["metrics"]["source_count"] == 5
    assert snapshot["metrics"]["event_count"] > 10
    assert snapshot["metrics"]["incident_count"] == 2
    assert snapshot["metrics"]["integrity_counts"] == {"verified": 5}
    assert {item["code"] for item in snapshot["metrics"]["incident_clusters"]} == {
        "PROVIDER_RELEASE_ROLLBACK", "RUNTIME_MANUAL_INTERVENTION_REQUIRED",
    }
    assert snapshot["metrics"]["drift_signals"] == {
        "invalid_promotions": 0,
        "l1_argument_mismatches": 0,
        "l1_safety_escapes": 0,
        "l1_target_mismatches": 0,
        "unverified_or_invalid_sources": 0,
    }
    assert snapshot["authority"] == {
        "observation_only": True, "approval": False, "execution": False,
        "registration": False, "activation": False,
    }
    rendered = json.dumps(snapshot, ensure_ascii=False)
    for secret in (
        "never-project-this", "secret-success-plan", "secret-manual-plan",
        "secret-approval", "secret-approver", "secret-session-id",
        "secret-decision-id", "secret-correlation", "secret-release",
        "secret-proposal-name", str(tmp_path),
    ):
        assert secret not in rendered


def test_p22_html_is_self_contained_filterable_and_digest_bound(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    html = render_evidence_html(snapshot)
    assert "Content-Security-Policy" in html
    assert "fetch(" not in html
    assert "XMLHttpRequest" not in html
    assert "Observation only" in html
    assert 'id="filter"' in html
    assert "<button" not in html.lower()
    assert "activation available" not in html.lower()
    output = tmp_path / "evidence.html"
    result = export_evidence_html(snapshot, output)
    assert result["ok"] is True
    assert result["activation_available"] is False
    tampered = dict(snapshot)
    tampered["status"] = "degraded"
    with pytest.raises(EvidencePlaneError, match="digest"):
        render_evidence_html(tampered)


def test_p22_detects_runtime_chain_tampering_without_mutating_source(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime.sqlite"
    _runtime_store(runtime)
    before_size = runtime.stat().st_size
    with sqlite3.connect(runtime) as database:
        database.execute("UPDATE plan_events SET event_hash='sha256:tampered' WHERE event_id=1")
    tampered_size = runtime.stat().st_size
    snapshot = collect_evidence_snapshot(
        runtime_journals=[runtime], generated_at="2026-08-30T00:00:00+00:00",
    )
    assert snapshot["status"] == "degraded"
    assert any(item["code"] == "RUNTIME_EVENT_CHAIN_INVALID" for item in snapshot["incidents"])
    assert runtime.stat().st_size == tampered_size
    assert tampered_size >= before_size


def test_p22_degrades_legacy_runtime_without_an_event_chain(tmp_path: Path) -> None:
    runtime = tmp_path / "legacy-runtime.sqlite"
    with sqlite3.connect(runtime) as database:
        database.execute(
            "CREATE TABLE plans(plan_id TEXT, plan_hash TEXT, state TEXT, "
            "created_at TEXT, updated_at TEXT)"
        )
        database.execute(
            "CREATE TABLE plan_events(event_id INTEGER PRIMARY KEY, plan_id TEXT, "
            "from_state TEXT, to_state TEXT, event_type TEXT, payload_json TEXT, "
            "created_at TEXT)"
        )
        created = "2026-08-30T00:00:00+00:00"
        database.execute(
            "INSERT INTO plans VALUES (?, ?, ?, ?, ?)",
            ("legacy-plan", sha256_json({"legacy": True}), "verified_success", created, created),
        )
        database.execute(
            "INSERT INTO plan_events VALUES (?, ?, ?, ?, ?, ?, ?)",
            (1, "legacy-plan", None, "prepared", "plan_prepared", "{}", created),
        )

    snapshot = collect_evidence_snapshot(
        runtime_journals=[runtime], generated_at="2026-08-30T00:00:00+00:00",
    )

    assert snapshot["status"] == "degraded"
    assert snapshot["metrics"]["integrity_counts"] == {"unverified": 1}
    assert snapshot["metrics"]["by_source"][snapshot["sources"][0]["source_id"]][
        "event_chain_present"
    ] is False


def test_p22_trend_is_stable_then_flags_integrity_regression(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime.sqlite"
    _runtime_store(runtime)
    first = collect_evidence_snapshot(
        runtime_journals=[runtime], generated_at="2026-08-30T00:00:00+00:00",
    )
    second = collect_evidence_snapshot(
        runtime_journals=[runtime], generated_at="2026-08-30T00:01:00+00:00",
    )
    stable = analyze_evidence_trend([first, second])
    assert stable["status"] == "stable"
    assert stable["production_slo_claim_available"] is False
    assert stable["activation_available"] is False

    with sqlite3.connect(runtime) as database:
        database.execute("UPDATE plan_events SET event_hash='sha256:tampered' WHERE event_id=1")
    degraded = collect_evidence_snapshot(
        runtime_journals=[runtime], generated_at="2026-08-30T00:02:00+00:00",
    )
    regressed = analyze_evidence_trend([first, degraded])
    assert regressed["status"] == "regressed"
    assert "snapshot_status_degraded" in regressed["regression_reasons"]
    assert "degraded_sources_increased" in regressed["regression_reasons"]


def test_p2_cli_bootstrap_validate_evidence_export_and_incident(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.yaml"
    runtime = tmp_path / "runtime.sqlite"
    snapshot = tmp_path / "snapshot.json"
    snapshot_two = tmp_path / "snapshot-two.json"
    html = tmp_path / "evidence.html"
    html_two = tmp_path / "evidence-two.html"
    _runtime_store(runtime)
    commands = [
        ["catalog-bootstrap", "--output", str(catalog)],
        ["catalog-validate", "--catalog", str(catalog)],
        ["evidence-export", "--runtime-journal", str(runtime),
         "--output", str(html), "--snapshot-output", str(snapshot)],
        ["evidence-export", "--runtime-journal", str(runtime),
         "--output", str(html_two), "--snapshot-output", str(snapshot_two)],
    ]
    for command in commands:
        result = subprocess.run(
            [str(ROOT / "scripts" / "netopyu-p2"), *command], cwd=ROOT,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=30, check=False,
        )
        assert result.returncode == 0, result.stderr
    assert html.is_file() and snapshot.is_file()
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    incident_id = payload["incidents"][0]["incident_id"]
    result = subprocess.run(
        [str(ROOT / "scripts" / "netopyu-p2"), "evidence-incident",
         "--snapshot", str(snapshot), "--incident-id", incident_id],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=30, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["incident_id"] == incident_id
    result = subprocess.run(
        [str(ROOT / "scripts" / "netopyu-p2"), "evidence-trend",
         "--snapshot", str(snapshot), "--snapshot", str(snapshot_two)],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=30, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "stable"
