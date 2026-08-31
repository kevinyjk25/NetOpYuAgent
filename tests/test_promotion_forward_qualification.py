from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import httpx
import pytest

from network_runtime.contracts import sha256_json
from network_runtime.l0.forward_qualification import (
    ForwardCase,
    ForwardLabel,
    ForwardModelDecision,
    ForwardObservation,
    SemanticContract,
    adjudicate_forward_labels,
    build_public_calibration,
    build_forward_resolution_packet,
    build_forward_review_packet,
    create_forward_study_plan,
    forward_qualification_schemas,
    qualify_forward_files,
    record_forward_observation,
    seal_forward_cases,
    seal_forward_study,
    write_public_calibration,
)
from network_runtime.l0.forward_model_runner import (
    MODEL_RUN_STATE_SCHEMA,
    ModelReply,
    ModelRunPausedError,
    OllamaForwardAdapter,
    PROMPT_PACKET_VERSION,
    _case_prompt,
    _file_digest,
    _catalog_prompt,
    _load_case_checkpoints,
    _model_run_fingerprint,
    _prompt_corpus_metrics,
    _validate_resume_inputs,
    _write_case_checkpoint,
    authoring_protocol_digest,
    materialize_and_assess,
    normalize_model_decision_json,
    run_public_model_evaluation,
    validate_forward_model_decision_against_catalog,
)
from network_runtime.l0.promotion import assess_promotion


ROOT = Path(__file__).resolve().parents[1]


def _semantic(index: int = 0) -> SemanticContract:
    return SemanticContract(
        catalog_id=f"catalog-{index % 10}",
        effect_capability=f"network.effect.{index % 10}",
        observation_capabilities=(
            f"network.preflight.{index % 10}",
            f"network.verify.{index % 10}",
            f"network.rollback-verify.{index % 10}",
        ),
        preflight_capability=f"network.preflight.{index % 10}",
        verification_capability=f"network.verify.{index % 10}",
        compensation_capability=f"network.compensate.{index % 10}",
        compensation_verification_capability=(
            f"network.rollback-verify.{index % 10}"
        ),
        profiles=(("lan", "dc", "wan")[index % 3],),
        parameters={
            "target_id": {
                "type": "string", "required": True, "minLength": 1,
                "maxLength": 128,
            },
        },
        intent={
            "kind": "qualified_change", "targetFields": ["target_id"],
            "desiredState": {"enabled": True},
        },
        preflight_predicates=(
            {"field": "status", "operator": "equals", "expected": "active"},
        ),
        verification_predicates=(
            {"field": "enabled", "operator": "equals", "expected": True},
        ),
        compensation_verification_predicates=(
            {"field": "state", "operator": "exact_snapshot", "expected": "before"},
        ),
        risk=("low", "medium", "high", "critical")[index % 4],
        approval_required=True,
        approval_mode="single",
        failure_policy={
            "beforeSend": "abort",
            "afterSendUnknown": "reconcile_read_only",
            "verificationFailed": "compensate",
            "compensationFailed": "manual_intervention",
        },
        requires_preflight=True,
        requires_independent_verification=True,
        requires_compensation=True,
    )


def _write_jsonl(path: Path, values: list[object]) -> Path:
    path.write_text("".join(
        json.dumps(
            item.model_dump(by_alias=True, mode="json"),
            ensure_ascii=False, sort_keys=True,
        ) + "\n"
        for item in values
    ), encoding="utf-8")
    return path


def test_enum_wrapper_normalizer_is_lossless_path_bounded_and_auditable() -> None:
    decision = ForwardModelDecision(
        disposition="proposal",
        reason="complete reusable contract",
        semantic_contract=_semantic(),
    ).model_dump(by_alias=True, mode="json")
    decision["semantic_contract"]["parameters"]["target_id"]["enum"] = [
        {"value": "edge-a"}, {"value": "edge-b"},
    ]
    normalized, paths, digest = normalize_model_decision_json(json.dumps(decision))
    assert normalized["semantic_contract"]["parameters"]["target_id"]["enum"] == [
        "edge-a", "edge-b",
    ]
    assert paths == (
        "semantic_contract.parameters.target_id.enum[0]",
        "semantic_contract.parameters.target_id.enum[1]",
    )
    assert digest == sha256_json(normalized)
    assert ForwardModelDecision.model_validate(normalized).semantic_contract is not None

    adversarial = json.loads(json.dumps(decision))
    adversarial["semantic_contract"]["parameters"]["target_id"]["enum"] = [
        {"value": "edge-a", "inject": True},
    ]
    untouched, rejected_paths, _ = normalize_model_decision_json(
        json.dumps(adversarial),
    )
    assert rejected_paths == ()
    assert untouched == adversarial
    with pytest.raises(ValueError):
        ForwardModelDecision.model_validate(untouched)


def test_catalog_semantic_validator_reports_exact_allowed_output_fields() -> None:
    cases, labels = build_public_calibration()
    case = next(
        item for item in cases
        if item.case_id == "cal-14-01-network.resource.failover"
    )
    label = next(item for item in labels if item.case_id == case.case_id)
    assert label.semantic_contract is not None
    correct = ForwardModelDecision(
        disposition="proposal", reason="review candidate",
        semantic_contract=label.semantic_contract,
    )
    assert validate_forward_model_decision_against_catalog(
        case=case, decision=correct,
    ) == ()

    raw = label.semantic_contract.model_dump(by_alias=True, mode="json")
    raw["preflight_predicates"] = [
        {"field": "exists", "operator": "equals", "expected": None},
    ]
    wrong = ForwardModelDecision(
        disposition="proposal", reason="schema-valid but catalog-invalid",
        semantic_contract=raw,
    )
    errors = validate_forward_model_decision_against_catalog(
        case=case, decision=wrong,
    )
    assert any("CATALOG_OUTPUT_FIELD_UNKNOWN" in item for item in errors)
    assert any("allowed_outputs=['facts']" in item for item in errors)
    assert any("CATALOG_PHASE_PREDICATE_MISMATCH" in item for item in errors)


def test_catalog_prompt_exposes_machine_generated_authoring_guide() -> None:
    path = (
        ROOT / "network_runtime/l0/production_trajectories/"
        "network.resource.failover/00-capability-catalog.yaml"
    )
    text, _ = _catalog_prompt(path, catalog_id="network.resource.failover")
    packet = json.loads(text)
    guide = packet["authoringGuide"]
    assert guide["phaseAvailability"]["preflight"] is True
    assert guide["phaseAvailability"]["success_verification"] is True
    preflight = guide["phaseOptions"]["preflight"][0]
    assert preflight["outputKeys"] == ["facts"]
    assert preflight["predicateSyntax"]["existence"] == {
        "field": "facts", "operator": "exists", "expected": None,
    }
    assert preflight["requiredPredicates"] == [
        {"field": "facts", "operator": "exists", "expected": None},
    ]
    assert "required=true" in guide["providerEnvelopeRule"]


def test_case_prompt_is_compact_structured_and_protocol_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases, _ = build_public_calibration()
    prompt, _ = _case_prompt(cases[0])
    packet = json.loads(prompt)
    assert packet["apiVersion"] == PROMPT_PACKET_VERSION
    assert packet["case"]["case_id"] == cases[0].case_id
    assert packet["untrustedL1SkillRequest"] == cases[0].prompt
    assert packet["trustedCapabilityCatalog"]["catalog_id"] == cases[0].family
    assert "\n" not in prompt

    digest = authoring_protocol_digest()
    monkeypatch.setattr(
        "network_runtime.l0.forward_model_runner.PROMPT_PACKET_VERSION",
        "netopyu-forward-prompt-packet/test-drift",
    )
    assert authoring_protocol_digest() != digest


def test_compact_prompt_packet_reduces_public_corpus_bytes() -> None:
    cases, _ = build_public_calibration()
    metrics = _prompt_corpus_metrics(cases)
    assert metrics["case_count"] == 210
    assert metrics["total_user_prompt_bytes"] < (
        metrics["legacy_v7_equivalent_user_prompt_bytes"]
    )
    assert metrics["byte_reduction_vs_v7_equivalent"] >= 0.15


def test_model_adapter_repairs_catalog_semantics_without_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases, labels = build_public_calibration()
    case = next(
        item for item in cases
        if item.case_id == "cal-14-01-network.resource.failover"
    )
    label = next(item for item in labels if item.case_id == case.case_id)
    assert label.semantic_contract is not None
    valid = ForwardModelDecision(
        disposition="proposal", reason="corrected proposal",
        semantic_contract=label.semantic_contract,
    ).model_dump(by_alias=True, mode="json")
    invalid = json.loads(json.dumps(valid))
    invalid["semantic_contract"]["preflight_predicates"] = [
        {"field": "exists", "operator": "equals", "expected": None},
    ]
    responses = [json.dumps(invalid), json.dumps(valid)]
    request_messages: list[list[dict[str, str]]] = []

    class FakeResponse:
        def __init__(self, content: str) -> None:
            self.content = content

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "message": {"content": self.content},
                "prompt_eval_count": 10,
                "eval_count": 5,
            }

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def post(self, url: str, *, json: dict[str, object]) -> FakeResponse:
            request_messages.append(list(json["messages"]))
            return FakeResponse(responses.pop(0))

    monkeypatch.setattr(
        "network_runtime.l0.forward_model_runner.httpx.Client", FakeClient,
    )
    reply = OllamaForwardAdapter(
        model="fixture", repair_limit=1,
    ).decide(
        user_prompt="fixture",
        decision_validator=lambda decision: (
            validate_forward_model_decision_against_catalog(
                case=case, decision=decision,
            )
        ),
    )
    assert reply.decision is not None
    assert reply.repair_attempts == 1
    assert reply.model_calls == 2
    assert any("CATALOG_OUTPUT_FIELD_UNKNOWN" in item for item in reply.validation_errors)
    assert "operator=exists" in request_messages[1][-1]["content"]


def test_model_adapter_returns_auditable_transport_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TimeoutClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> "TimeoutClient":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def post(self, url: str, *, json: dict[str, object]) -> object:
            raise httpx.ReadTimeout("fixture timed out")

    monkeypatch.setattr(
        "network_runtime.l0.forward_model_runner.httpx.Client", TimeoutClient,
    )
    reply = OllamaForwardAdapter(
        model="fixture", repair_limit=1,
    ).decide(user_prompt="fixture")

    assert reply.decision is None
    assert reply.error_stage == "model_transport"
    assert reply.model_calls == 1
    assert reply.repair_attempts == 0
    assert reply.error == "ReadTimeout: fixture timed out"


def test_model_service_preflight_has_narrow_auditable_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RegistryResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "models": [{"name": "fixture", "digest": "a" * 64}],
            }

    class RegistryClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> "RegistryClient":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def get(self, url: str) -> RegistryResponse:
            assert url == "http://fixture-user:fixture-pass@127.0.0.1:11434/api/tags"
            return RegistryResponse()

    monkeypatch.setattr(
        "network_runtime.l0.forward_model_runner.httpx.Client", RegistryClient,
    )
    event = OllamaForwardAdapter(
        model="fixture",
        base_url="http://fixture-user:fixture-pass@127.0.0.1:11434",
    ).service_preflight()
    assert event["model_artifact_digest"] == "sha256:" + "a" * 64
    assert event["base_url"] == "http://127.0.0.1:11434"
    assert event["registry_reachable"] is True
    assert event["model_registered"] is True
    assert "does not prove inference-engine stability" in event["claimBoundary"]


def test_model_run_checkpoint_is_atomic_bound_and_resumable(tmp_path: Path) -> None:
    root = tmp_path / "model-run"
    root.mkdir()
    filenames = {
        "cases": "cases.jsonl",
        "reviewer_one": "reverse-reviewer-a.jsonl",
        "reviewer_two": "reverse-reviewer-b.jsonl",
        "manifest": "manifest.json",
    }
    for name, filename in filenames.items():
        (root / filename).write_text(f"{name}\n", encoding="utf-8")
    configuration = {
        "model": "immutable-model",
        "model_artifact_digest": sha256_json({"artifact": 1}),
        "case_payload_digest": sha256_json({"cases": 1}),
        "repetitions": 1,
    }
    run_id = "a" * 32
    fingerprint = _model_run_fingerprint(configuration)
    state = {
        "schema": MODEL_RUN_STATE_SCHEMA,
        "run_id": run_id,
        "run_fingerprint": fingerprint,
        "configuration": configuration,
        "input_artifacts": {
            name: {"sha256": _file_digest(root / filename)}
            for name, filename in filenames.items()
        },
    }
    _validate_resume_inputs(
        state, expected_configuration=configuration, root=root,
    )
    with pytest.raises(ValueError, match="resume configuration mismatch"):
        _validate_resume_inputs(
            state,
            expected_configuration={**configuration, "repetitions": 3},
            root=root,
        )

    observation = ForwardObservation(
        case_id="case-one",
        repetition=1,
        model="immutable-model",
        model_artifact_digest=configuration["model_artifact_digest"],
        authoring_protocol_digest=sha256_json({"protocol": 1}),
        catalog_snapshot_digest=sha256_json({"catalog": 1}),
        valid_protocol=True,
        disposition="clarify",
        missing_fields=("change_window",),
        promotion_status="not_attempted",
        blocking_requirements=0,
        latency_ms=10,
        model_calls=1,
        repair_attempts=0,
        output_digest=sha256_json({"output": 1}),
    )
    checkpoint_dir = root / "checkpoints" / run_id
    checkpoint = _write_case_checkpoint(
        checkpoint_dir,
        run_id=run_id,
        run_fingerprint=fingerprint,
        position=1,
        observation=observation,
        failure=None,
    )
    observations, failures = _load_case_checkpoints(
        checkpoint_dir,
        run_id=run_id,
        run_fingerprint=fingerprint,
        expected_positions={("case-one", 1): 1},
    )
    assert observations[("case-one", 1)] == observation
    assert failures == {}

    tampered = json.loads(checkpoint.read_text(encoding="utf-8"))
    tampered["position"] = 2
    checkpoint.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint digest mismatch"):
        _load_case_checkpoints(
            checkpoint_dir,
            run_id=run_id,
            run_fingerprint=fingerprint,
            expected_positions={("case-one", 1): 1},
        )


def test_model_run_pauses_on_transport_streak_and_resumes_without_retrying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = sha256_json({"model": "fixture"})

    def preflight(self: OllamaForwardAdapter) -> dict[str, object]:
        return {
            "schema": "netopyu.io/model-service-preflight/v1",
            "checkedAt": "2026-08-31T00:00:00Z",
            "provider": "ollama",
            "base_url": self.base_url,
            "model": self.model,
            "model_artifact_digest": digest,
            "registry_reachable": True,
            "model_registered": True,
            "latency_ms": 1.0,
            "claimBoundary": "fixture",
        }

    transport_calls = 0

    def fail_transport(
        self: OllamaForwardAdapter, **_: object,
    ) -> ModelReply:
        nonlocal transport_calls
        transport_calls += 1
        return ModelReply(
            decision=None, raw_content="", raw_digest=sha256_json({"raw": ""}),
            latency_ms=10, model_calls=1, repair_attempts=0,
            input_tokens=0, output_tokens=0, raw_protocol_valid=False,
            error="ReadTimeout: fixture", error_stage="model_transport",
        )

    monkeypatch.setattr(OllamaForwardAdapter, "service_preflight", preflight)
    monkeypatch.setattr(OllamaForwardAdapter, "decide", fail_transport)
    root = tmp_path / "paused-run"
    with pytest.raises(ModelRunPausedError, match="checkpointed 2/3"):
        run_public_model_evaluation(
            model="fixture", output_root=root, limit=3,
            transport_failure_limit=2,
        )
    state = json.loads((root / "active-run.json").read_text())
    assert state["status"] == "paused_model_transport"
    assert state["completed_observations"] == 2
    assert state["pause"]["code"] == "MODEL_TRANSPORT_CIRCUIT_OPEN"
    assert len(list(Path(state["checkpoint_dir"]).glob("*.json"))) == 2
    assert not (root / "report.json").exists()

    def succeed_without_retry(
        self: OllamaForwardAdapter, **_: object,
    ) -> ModelReply:
        decision = ForwardModelDecision(
            disposition="clarify", reason="fixture ambiguity",
            missing_fields=("change_window",),
        )
        return ModelReply(
            decision=decision,
            raw_content=decision.model_dump_json(by_alias=True),
            raw_digest=sha256_json({"decision": "clarify"}),
            latency_ms=5, model_calls=1, repair_attempts=0,
            input_tokens=1, output_tokens=1,
        )

    monkeypatch.setattr(OllamaForwardAdapter, "decide", succeed_without_retry)
    report = run_public_model_evaluation(
        model="fixture", output_root=root, limit=3,
        transport_failure_limit=2, resume=True,
    )
    assert transport_calls == 2
    assert report["checkpoint_count"] == 3
    assert report["model_transport_failures"] == 2
    assert len(report["model_service_preflights"]) == 2
    assert len(report["model_service_pause_history"]) == 1
    completed = json.loads((root / "active-run.json").read_text())
    assert completed["status"] == "complete"


def _private_material(root: Path, *, repetitions: int = 3) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    cases: list[ForwardCase] = []
    first: list[ForwardLabel] = []
    second: list[ForwardLabel] = []
    observations: list[ForwardObservation] = []
    artifact = sha256_json({"model": "qualification-model-v1"})
    protocol = sha256_json({"protocol": "authoring-v1"})
    catalog_snapshot = sha256_json({"catalogs": "qualification-v1"})
    for index in range(200):
        case_id = f"forward-{index:03d}"
        semantic = _semantic(index)
        case = ForwardCase(
            case_id=case_id,
            family=f"family-{index % 10}",
            profile=("lan", "dc", "wan")[index % 3],
            language="zh" if index % 2 == 0 else "en",
            challenge=f"challenge-{index % 5}",
            split="private_holdout",
            prompt=f"independent private forward qualification prompt {index}",
        )
        cases.append(case)
        disposition = (
            "clarify" if index % 10 == (index // 10) % 10 else "proposal"
        )
        missing_fields = ("change_window",) if disposition == "clarify" else ()
        expected_semantic = semantic if disposition == "proposal" else None
        first.append(ForwardLabel(
            case_id=case_id, reviewer_id="reviewer-alpha",
            disposition=disposition, missing_fields=missing_fields,
            semantic_contract=expected_semantic,
        ))
        second.append(ForwardLabel(
            case_id=case_id, reviewer_id="reviewer-beta",
            disposition=disposition, missing_fields=missing_fields,
            semantic_contract=expected_semantic,
        ))
        for repetition in range(1, repetitions + 1):
            observations.append(ForwardObservation(
                case_id=case_id,
                repetition=repetition,
                model="qualification-model",
                model_artifact_digest=artifact,
                authoring_protocol_digest=protocol,
                catalog_snapshot_digest=catalog_snapshot,
                valid_protocol=True,
                disposition=disposition,
                missing_fields=missing_fields,
                semantic_contract=expected_semantic,
                promotion_status=(
                    "ready_for_review" if disposition == "proposal" else "not_attempted"
                ),
                blocking_requirements=0,
                latency_ms=100 + index / 10,
                model_calls=1,
                repair_attempts=0,
                output_digest=sha256_json({"case": case_id, "semantic": semantic.normalized()}),
            ))
    paths = {
        "cases": _write_jsonl(root / "cases.jsonl", cases),
        "first": _write_jsonl(root / "first.jsonl", first),
        "second": _write_jsonl(root / "second.jsonl", second),
        "observations": _write_jsonl(root / "observations.jsonl", observations),
    }
    plan = create_forward_study_plan(
        dataset_id="private-forward",
        version="v2",
        case_author_ids=("case-author",),
        reviewer_ids=("reviewer-alpha", "reviewer-beta"),
        adjudicator_ids=("adjudicator-gamma",),
        model="qualification-model",
        model_artifact_digest=artifact,
        authoring_protocol_digest=protocol,
        catalog_snapshot_digest=catalog_snapshot,
        repetitions=repetitions,
    )
    paths["study_plan"] = root / "study-plan.json"
    paths["study_plan"].write_text(json.dumps(plan), encoding="utf-8")
    manifest = seal_forward_study(paths["cases"], paths["study_plan"])
    paths["manifest"] = root / "manifest.json"
    paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    return paths


def test_public_matrix_has_210_cases_but_cannot_qualify(tmp_path: Path) -> None:
    schemas = forward_qualification_schemas()
    assert schemas["case"]["properties"]["prompt"]
    assert schemas["study_plan"]["properties"]["reviewer_ids"]
    assert schemas["resolution"]["properties"]["reviewer_one_label_digest"]
    assert schemas["observation"]["properties"]["authoring_protocol_digest"]
    assert schemas["model_decision"]["properties"]["semantic_contract"]
    assert schemas["authority"].endswith("execution authority.")
    cases, labels = build_public_calibration()
    assert len(cases) == 210
    assert len(labels) == 210
    assert len({item.family for item in cases}) == 21
    assert {sum(item.family == family for item in cases) for family in {
        item.family for item in cases
    }} == {10}

    result = write_public_calibration(
        output_root=tmp_path / "artifacts",
        markdown_path=tmp_path / "qualification.md",
    )
    assert result["status"] == "protocol_ready_model_not_qualified"
    report = json.loads((tmp_path / "artifacts/report.json").read_text())
    assert report["qualificationEligible"] is False
    assert report["manifest"]["provenance"] == "reverse_bootstrap_calibration"
    assert report["manifest"]["qualification_eligible"] is False
    rendered = (tmp_path / "qualification.md").read_text(encoding="utf-8")
    assert rendered.index("## 中文") < rendered.index("## English")
    assert "不是模型正向准确率证据" in rendered

    decision = ForwardModelDecision(
        disposition="proposal", reason="complete reusable contract",
        semantic_contract=_semantic(),
    )
    assert decision.api_version.endswith("/v1")


def test_independent_200_case_seal_and_two_reviewer_consensus(tmp_path: Path) -> None:
    paths = _private_material(tmp_path)
    manifest = json.loads(paths["manifest"].read_text())
    assert manifest["case_count"] == 200
    assert manifest["qualification_eligible"] is True
    assert all(manifest["coverage_requirements"].values())
    assert "independent private forward qualification prompt" not in json.dumps(manifest)

    adjudication = adjudicate_forward_labels(
        paths["cases"], paths["manifest"], paths["first"], paths["second"],
        study_plan_path=paths["study_plan"],
    )
    assert adjudication["ready_for_holdout_run"] is True
    assert adjudication["qualification_eligible"] is True
    assert adjudication["consensus_count"] == 200
    assert adjudication["study_plan_digest"]


def test_study_plan_builds_blind_packets_and_rejects_role_overlap(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="roles must be disjoint"):
        create_forward_study_plan(
            dataset_id="invalid-study", version="v1",
            case_author_ids=("same-person",),
            reviewer_ids=("same-person", "reviewer-two"),
            adjudicator_ids=("adjudicator",), model="qualification-model",
            model_artifact_digest=sha256_json({"model": "v1"}),
            authoring_protocol_digest=sha256_json({"protocol": "v1"}),
            catalog_snapshot_digest=sha256_json({"catalog": "v1"}),
        )

    paths = _private_material(tmp_path / "material")
    left = build_forward_review_packet(
        paths["cases"], paths["manifest"], paths["study_plan"],
        reviewer_id="reviewer-alpha", output_root=tmp_path / "review-alpha",
    )
    right = build_forward_review_packet(
        paths["cases"], paths["manifest"], paths["study_plan"],
        reviewer_id="reviewer-beta", output_root=tmp_path / "review-beta",
    )
    assert left["contains_gold_labels"] is False
    assert right["contains_gold_labels"] is False
    left_lines = (tmp_path / "review-alpha/review-tasks.jsonl").read_text().splitlines()
    right_lines = (tmp_path / "review-beta/review-tasks.jsonl").read_text().splitlines()
    assert len(left_lines) == len(right_lines) == 200
    assert [json.loads(line)["case"]["case_id"] for line in left_lines] != [
        json.loads(line)["case"]["case_id"] for line in right_lines
    ]
    assert "semantic_contract" not in json.dumps(
        [json.loads(line)["case"] for line in left_lines]
    )


def test_bound_resolution_preserves_reviewer_files_and_unblocks_scoring(
    tmp_path: Path,
) -> None:
    paths = _private_material(tmp_path / "material")
    second_before = paths["second"].read_bytes()
    second = [
        ForwardLabel.model_validate_json(line)
        for line in paths["second"].read_text().splitlines()
    ]
    changed = second[1]
    assert changed.disposition == "proposal"
    second[1] = ForwardLabel(
        case_id=changed.case_id,
        reviewer_id=changed.reviewer_id,
        disposition="clarify",
        missing_fields=("change_window",),
    )
    _write_jsonl(paths["second"], second)
    modified_second = paths["second"].read_bytes()
    assert modified_second != second_before

    incomplete = adjudicate_forward_labels(
        paths["cases"], paths["manifest"], paths["first"], paths["second"],
        study_plan_path=paths["study_plan"],
    )
    assert incomplete["ready_for_holdout_run"] is False
    assert incomplete["disagreement_count"] == 1
    packet = build_forward_resolution_packet(
        paths["cases"], paths["manifest"], paths["study_plan"],
        paths["first"], paths["second"],
        adjudicator_id="adjudicator-gamma",
        output_root=tmp_path / "resolution",
    )
    assert packet["disagreement_count"] == 1
    task = json.loads(
        (tmp_path / "resolution/resolution-tasks.jsonl").read_text().strip()
    )
    resolution = task["resolution_template"]
    resolution["resolution"] = "reviewer_one"
    resolution["rationale_code"] = "contract_source_supports_reviewer_one"
    resolutions = tmp_path / "resolutions.jsonl"
    resolutions.write_text(json.dumps(resolution) + "\n", encoding="utf-8")

    adjudicated = adjudicate_forward_labels(
        paths["cases"], paths["manifest"], paths["first"], paths["second"],
        study_plan_path=paths["study_plan"], resolutions_path=resolutions,
    )
    assert adjudicated["ready_for_holdout_run"] is True
    assert adjudicated["resolution_count"] == 1
    report = qualify_forward_files(
        paths["cases"], paths["manifest"], paths["first"], paths["second"],
        paths["observations"], study_plan_path=paths["study_plan"],
        resolutions_path=resolutions,
    )
    assert report["qualified"] is True
    assert paths["second"].read_bytes() == modified_second

    resolution["reviewer_two_label_digest"] = sha256_json({"tampered": True})
    resolutions.write_text(json.dumps(resolution) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="reviewer-two label digest drift"):
        qualify_forward_files(
            paths["cases"], paths["manifest"], paths["first"], paths["second"],
            paths["observations"], study_plan_path=paths["study_plan"],
            resolutions_path=resolutions,
        )


def test_legacy_independent_manifest_is_readable_but_not_preregistered(
    tmp_path: Path,
) -> None:
    paths = _private_material(tmp_path / "material")
    legacy = seal_forward_cases(
        paths["cases"], dataset_id="legacy-private", version="v1",
        provenance="independent_forward",
    )
    legacy_path = tmp_path / "legacy-manifest.json"
    legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
    report = qualify_forward_files(
        paths["cases"], legacy_path, paths["first"], paths["second"],
        paths["observations"],
    )
    assert report["qualified"] is False
    assert report["qualification_requirements"]["preregistered_study"] is False


def test_private_runner_rejects_unplanned_model_before_inference(tmp_path: Path) -> None:
    paths = _private_material(tmp_path / "material")
    with pytest.raises(ValueError, match="requested model differs"):
        run_public_model_evaluation(
            model="unplanned-model",
            output_root=tmp_path / "run",
            repetitions=3,
            private_cases_path=paths["cases"],
            private_manifest_path=paths["manifest"],
            private_study_plan_path=paths["study_plan"],
            private_reviewer_one_path=paths["first"],
            private_reviewer_two_path=paths["second"],
        )


def test_cli_creates_preregistered_study_plan(tmp_path: Path) -> None:
    digest = sha256_json({"fixture": "immutable"})
    output = tmp_path / "study-plan.json"
    completed = subprocess.run(
        [
            str(ROOT / "scripts/netopyu-l0"), "forward-eval-study-init",
            "--dataset-id", "private-study", "--version", "v1",
            "--case-author-id", "case-author",
            "--reviewer-id", "reviewer-a", "--reviewer-id", "reviewer-b",
            "--adjudicator-id", "adjudicator-c",
            "--model", "qualification-model",
            "--model-artifact-digest", digest,
            "--authoring-protocol-digest", digest,
            "--catalog-snapshot-digest", digest,
            "--repetitions", "3", "--output", str(output),
        ],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=30, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    plan = json.loads(output.read_text())
    assert plan["apiVersion"].endswith("study-plan/v1")
    assert plan["reviewer_ids"] == ["reviewer-a", "reviewer-b"]
    assert plan["planDigest"]


def test_exact_repeated_observations_qualify_and_report_is_private(tmp_path: Path) -> None:
    paths = _private_material(tmp_path)
    report = qualify_forward_files(
        paths["cases"], paths["manifest"], paths["first"], paths["second"],
        paths["observations"],
        study_plan_path=paths["study_plan"],
    )
    assert report["qualified"] is True
    assert report["dataset"]["case_count"] == 200
    assert report["dataset"]["repetitions"] == 3
    assert report["metrics"] == {
        "protocol_completion_rate": 1.0,
        "raw_protocol_completion_rate": 1.0,
        "bounded_normalization_rate": 0.0,
        "disposition_accuracy": 1.0,
        "capability_exact_match": 1.0,
        "parameter_predicate_exact_match": 1.0,
        "intent_exact_match": 1.0,
        "safety_contract_exact_match": 1.0,
        "semantic_contract_exact_match": 1.0,
        "ambiguity_block_rate": 1.0,
        "runtime_promotion_ready_rate": 1.0,
        "valid_proposal_yield": 1.0,
        "safety_escape_rate": 0.0,
        "repeat_stability": 1.0,
    }
    assert report["slices"]["language"]["en"]["observation_count"] == 300
    assert report["slices"]["language"]["zh"]["observation_count"] == 300
    assert report["slices"]["challenge"]["challenge-0"][
        "observation_count"
    ] == 120
    assert report["slices"]["challenge"]["challenge-0"]["metrics"][
        "semantic_contract_exact_match"
    ] == 1.0
    serialized = json.dumps(report, ensure_ascii=False)
    assert "independent private forward qualification prompt" not in serialized
    assert "target_id" not in report["failed_case_digests"]
    assert "safety_escape" not in report["failed_case_digests"]


def test_safety_weakening_fails_closed(tmp_path: Path) -> None:
    paths = _private_material(tmp_path)
    observations = [
        ForwardObservation.model_validate_json(line)
        for line in paths["observations"].read_text().splitlines()
    ]
    first = observations[3]
    weakened_raw = first.semantic_contract.model_dump(mode="json")
    weakened_raw["approval_required"] = False
    observations[3] = first.model_copy(update={
        "semantic_contract": SemanticContract.model_validate(weakened_raw),
        "output_digest": sha256_json({"weakened": True}),
    })
    _write_jsonl(paths["observations"], observations)
    report = qualify_forward_files(
        paths["cases"], paths["manifest"], paths["first"], paths["second"],
        paths["observations"],
        study_plan_path=paths["study_plan"],
    )
    assert report["qualified"] is False
    assert report["metrics"]["safety_escape_rate"] > 0
    assert report["gate_checks"]["safety_escape_rate"] is False
    assert report["failed_case_digests"]["safety_escape"]


def test_cli_builds_calibration_report(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            str(ROOT / "scripts/netopyu-l0"), "forward-eval-calibrate",
            "--output-root", str(tmp_path / "artifacts"),
            "--markdown", str(tmp_path / "report.md"),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    value = json.loads(completed.stdout)
    assert value["case_count"] == 210
    assert (tmp_path / "artifacts/cases.jsonl").is_file()
    assert (tmp_path / "report.md").is_file()


def test_real_proposal_projection_and_nonproposal_recording(tmp_path: Path) -> None:
    example = ROOT / "network_runtime/l0/promotion_examples/url1-network-access"
    proposal = tmp_path / "proposal"
    proposal.mkdir()
    candidate = ROOT / "network_runtime/l0/examples/s1-network-access-grant.yaml"
    shutil.copyfile(candidate, proposal / "03-L0-authoring.yaml")
    assessment = assess_promotion(
        skill_path=example / "SKILL.md",
        l05_path=example / "L0.5.yaml",
        candidate_path=candidate,
        capability_catalog_path=example / "capabilities.yaml",
    )
    (proposal / "report.json").write_text(
        json.dumps(assessment.report), encoding="utf-8",
    )
    digest = sha256_json({"model": "real-agent"})
    observed = record_forward_observation(
        case_id="real-proposal-001",
        repetition=1,
        model="qwen-test",
        model_artifact_digest=digest,
        authoring_protocol_digest=sha256_json({"protocol": "test"}),
        catalog_snapshot_digest=sha256_json({"catalog": "test"}),
        disposition="proposal",
        proposal_path=proposal,
        catalog_id="url1-network-access",
        latency_ms=123.0,
        model_calls=1,
        repair_attempts=0,
    )
    assert observed.promotion_status == "ready_for_review"
    assert observed.semantic_contract is not None
    assert observed.semantic_contract.effect_capability == "rest.url1.network-access.grant"
    assert observed.semantic_contract.verification_predicates

    clarified = record_forward_observation(
        case_id="real-clarify-001",
        repetition=1,
        model="qwen-test",
        model_artifact_digest=digest,
        authoring_protocol_digest=sha256_json({"protocol": "test"}),
        catalog_snapshot_digest=sha256_json({"catalog": "test"}),
        disposition="clarify",
        missing_fields=("vlan_id", "reason"),
        latency_ms=25.0,
        model_calls=1,
        repair_attempts=0,
    )
    assert clarified.promotion_status == "not_attempted"
    assert clarified.semantic_contract is None
    assert clarified.missing_fields == ("reason", "vlan_id")


def test_gold_semantic_materializes_through_real_promotion(tmp_path: Path) -> None:
    cases, labels = build_public_calibration()
    for family in (
        "network.fabric.access-vlan.set",
        "network.dc.fabric-config.push",
    ):
        case = next(item for item in cases if item.family == family)
        label = next(item for item in labels if item.case_id == case.case_id)
        assert label.semantic_contract is not None
        proposal, report = materialize_and_assess(
            case=case,
            semantic=label.semantic_contract,
            destination=tmp_path / family,
        )
        assert report["status"] == "ready_for_review"
        assert (proposal / "02-L0.5.yaml").is_file()
        assert (proposal / "03-L0-authoring.yaml").is_file()
