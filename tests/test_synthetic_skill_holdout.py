from __future__ import annotations

import json
from pathlib import Path

import pytest

import evaluation.general_effect_model as general_effect_model
from effect_runtime.mcp_lab import DOMAINS
from evaluation.external_synthetic_author import (
    _blueprints,
    _canonical,
    _seal,
)
from evaluation.synthetic_skill_holdout import (
    EVIDENCE_CLASS,
    build_interface_pack,
    export_synthetic_study_workspace,
    inspect_synthetic_packages,
    load_synthetic_dataset,
)
from evaluation.synthetic_evidence_report import build_synthetic_evidence_summary
from network_runtime.contracts import sha256_json


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(_canonical(item) + "\n" for item in values), encoding="utf-8")


def _sealed_study(root: Path) -> dict[str, object]:
    pack = build_interface_pack(requested_cases=240)
    blueprints = _blueprints(pack)
    by_id = {item["case_id"]: item for item in blueprints}
    authored = {}
    for item in blueprints:
        arguments = item["arguments"]
        reason = f"; reason={arguments['reason']}" if "reason" in arguments else ""
        unknown = "; invented_scope=all" if "invented_scope" in arguments else ""
        authored[item["case_id"]] = {
            "case_id": item["case_id"],
            "user_input": (
                f"change={arguments['change_id']}; entity={arguments['entity_id']}; "
                f"desired={arguments['desired_value']}; revision={arguments['expected_revision']}"
                f"{reason}{unknown}"
            ),
            "skill_guidance": (
                "Validate explicit inputs, read and preserve pre-change state, bind immutable "
                "approval, revalidate, issue one effect, independently verify, reconcile an "
                "unknown outcome without blind retry, and restore the exact snapshot with "
                "independent recovery verification before reporting a terminal result."
            ),
            "reference_text": "Only the explicit scoped entity may be changed.",
            "author_id": "model-author",
            "authoring_mode": "model_narrative_with_deterministic_parameter_anchors",
            "model": "offline-test-model",
            "authorPromptDigest": sha256_json({"case": item["case_id"]}),
        }
    reviewer_a = {
        case_id: {
            "case_id": case_id, "reviewer_id": "model-reviewer-a",
            "verdict": "accept", "issue_codes": [], "explanation": "complete",
        }
        for case_id in authored
    }
    reviewer_b = {
        case_id: {
            "case_id": case_id, "reviewer_id": "model-reviewer-b",
            "verdict": "accept", "issue_codes": [], "explanation": "complete",
        }
        for case_id in authored
    }
    _write_jsonl(root / "author/candidates.jsonl", list(authored.values()))
    _write_jsonl(root / "reviewer-a/reviews.jsonl", list(reviewer_a.values()))
    _write_jsonl(root / "reviewer-b/reviews.jsonl", list(reviewer_b.values()))
    _write_jsonl(root / "adjudicator/resolutions.jsonl", [])
    return _seal(
        root, pack,
        {"model": "offline-test-model", "modelArtifactDigest": sha256_json({"model": "offline"})},
        by_id, authored, reviewer_a, reviewer_b, {},
        {
            "author": {"calls": 0, "promptTokens": 0, "outputTokens": 0},
            "reviewerA": {"calls": 0, "promptTokens": 0, "outputTokens": 0},
            "reviewerB": {"calls": 0, "promptTokens": 0, "outputTokens": 0},
            "adjudicator": {"calls": 0, "promptTokens": 0, "outputTokens": 0},
        },
    )


def test_interface_workspace_is_context_isolated(tmp_path: Path) -> None:
    root = tmp_path / "external-study"
    exported = export_synthetic_study_workspace(root, requested_cases=240)
    source = (root / "generate.py").read_text(encoding="utf-8")
    assert exported["officialEsP1QualificationEligible"] is False
    assert "from evaluation" not in source
    assert "import network_runtime" not in source
    assert "production_trajectories" not in (root / "interface-pack.json").read_text(
        encoding="utf-8"
    )


def test_sealed_synthetic_dataset_round_trip_and_claim_boundary(tmp_path: Path) -> None:
    manifest = _sealed_study(tmp_path)
    loaded, cases = load_synthetic_dataset(tmp_path)
    assert manifest["manifestDigest"] == loaded["manifestDigest"]
    assert loaded["evidenceClass"] == EVIDENCE_CLASS
    assert loaded["officialEsP1QualificationEligible"] is False
    assert len(cases) == 240
    assert set(loaded["coverage"]["scenarioPatterns"]) == {
        "success", "missing_required", "unknown_parameter", "approval_denied",
        "revision_conflict", "verification_mismatch", "after_send_unknown",
        "provider_error_before_send", "compensation_failure", "success_alternate",
    }
    inspection = inspect_synthetic_packages(tmp_path)
    assert inspection["packageGates"] == {"passed": 240}
    assert inspection["allPackagesPassed"] is True
    assert inspection["findingCounts"] == {}
    assert loaded["renderer"]["version"] == "synthetic-skill-package-renderer/v3"


def test_synthetic_dataset_cannot_self_declare_esp1_eligibility(tmp_path: Path) -> None:
    _sealed_study(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["officialEsP1QualificationEligible"] = True
    body = {key: value for key, value in manifest.items() if key != "manifestDigest"}
    manifest["manifestDigest"] = sha256_json(body)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be ES-P1 eligible"):
        load_synthetic_dataset(tmp_path)


def test_synthetic_skill_package_tamper_is_detected(tmp_path: Path) -> None:
    _, cases = load_synthetic_dataset(tmp_path) if (tmp_path / "manifest.json").exists() else (
        _sealed_study(tmp_path), (),
    )
    if not cases:
        _, cases = load_synthetic_dataset(tmp_path)
    skill = tmp_path / "skills" / cases[0].skill_id / "SKILL.md"
    skill.write_text(skill.read_text(encoding="utf-8") + "\nchanged\n", encoding="utf-8")
    with pytest.raises(ValueError, match="package digest drift"):
        load_synthetic_dataset(tmp_path)


def test_synthetic_skill_package_rejects_unknown_empty_root(tmp_path: Path) -> None:
    _sealed_study(tmp_path)
    _, cases = load_synthetic_dataset(tmp_path)
    (tmp_path / "skills" / cases[0].skill_id / "unexpected").mkdir()
    with pytest.raises(ValueError, match="unknown root"):
        load_synthetic_dataset(tmp_path)


def test_translation_entrypoint_consumes_sealed_external_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = tmp_path / "study"
    study.mkdir()
    _sealed_study(study)

    class FakeAdapter:
        def __init__(self, model: str) -> None:
            self.model = model

        def preflight(self) -> dict[str, str]:
            return {
                "model": self.model,
                "modelArtifactDigest": sha256_json({"model": self.model}),
            }

        def translate(self, prompt: str):  # type: ignore[no-untyped-def]
            domain = next(
                value for value in DOMAINS
                if f"effect.{value}.state.apply" in prompt
            )
            decision = general_effect_model.TranslationDecision(
                disposition="proposal",
                effect_capability=f"effect.{domain}.state.apply",
                approval_required=True,
                script_execution_allowed=False,
                confidence=1.0,
                explanation="trusted synthetic test proposal",
            )
            return decision, {
                "raw": decision.model_dump_json(),
                "rawProtocolValid": True,
                "modelCalls": 1,
                "inputTokens": 1,
                "outputTokens": 1,
                "latencyMs": 1.0,
                "error": None,
                "rawDigest": sha256_json({"decision": decision.model_dump(mode="json")}),
            }

    monkeypatch.setattr(general_effect_model, "OllamaTranslationAdapter", FakeAdapter)
    report = general_effect_model.run_model_translation(
        output_root=tmp_path / "output",
        model="offline-test-model",
        limit=1,
        resume=False,
        dataset_root=study,
    )
    assert report["dataset"]["syntheticHoldout"] is True
    assert report["dataset"]["officialEsP1QualificationEligible"] is False
    assert report["metrics"]["oraclePassed"] == 1
    assert report["cases"][0]["route"] == "l0_runtime"


def test_synthetic_evidence_summary_requires_and_describes_full_corpus(
    tmp_path: Path,
) -> None:
    study = tmp_path / "study"
    study.mkdir()
    manifest = _sealed_study(study)
    _, cases = load_synthetic_dataset(study)
    rows = [{
        "case_id": case.case_id,
        "feature_family": case.feature_family,
        "domain": case.domain,
        "raw_protocol_valid": True,
        "oracle_passed": True,
        "route": "l0_runtime",
        "semantic_coverage_percent": 100.0,
        "field_results": {"effect_capability": True},
    } for case in cases]
    translation = {
        "dataset": {"digest": manifest["datasetDigest"], "executedCases": 240},
        "model": {"model": "offline"},
        "metrics": {
            "total": 240, "rawProtocolValid": 240,
            "oraclePassed": 240, "fallbacks": 0,
            "latency": {"p50Ms": 1.0, "p95Ms": 1.0},
        },
        "cases": rows,
    }
    translation_path = tmp_path / "translation.json"
    translation_path.write_text(json.dumps(translation), encoding="utf-8")
    report = build_synthetic_evidence_summary(
        dataset_root=study,
        translation_report=translation_path,
        output_root=tmp_path / "summary",
    )
    assert report["officialEsP1QualificationEligible"] is False
    assert report["translation"]["runtimeRouteOracleViolations"] == 0
    assert len(report["translation"]["byFeature"]) == 6
    assert len(report["translation"]["byScenario"]) == 10
    assert (tmp_path / "summary/synthetic-evidence-summary.md").is_file()
