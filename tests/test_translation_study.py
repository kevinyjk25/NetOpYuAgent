from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from evaluation.translation_study import (
    AlignmentReview,
    SCORE_SCHEMA,
    assess_runtime_evaluation_admission,
    create_translation_split_manifest,
    inspect_runtime_evaluation_admission,
    inspect_translation_split_manifest,
    score_translation_v2,
    split_case_ids,
)
from network_runtime.contracts import sha256_json


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_split_is_by_skill_package_and_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paired = tmp_path / "paired"
    cases = [
        {"caseId": f"c{index:02d}", "packageId": f"skill-{index // 2}"}
        for index in range(20)
    ]
    _write_jsonl(paired / "agent/cases.jsonl", cases)
    monkeypatch.setattr(
        "evaluation.translation_study.inspect_public_paired_study_kit",
        lambda _: {"workspaceDigest": "sha256:paired"},
    )
    first = create_translation_split_manifest(paired, tmp_path / "split-a.json")
    second = create_translation_split_manifest(paired, tmp_path / "split-b.json")
    assert first["packages"] == second["packages"]
    assert first["cases"] == second["cases"]
    inspected = inspect_translation_split_manifest(tmp_path / "split-a.json")
    assert inspected["packageCounts"] == {
        "development": 6, "frozen_validation": 2, "sealed_test": 2,
    }
    membership = {
        case_id: split
        for split, case_ids in first["cases"].items()
        for case_id in case_ids
    }
    assert all(membership[f"c{index * 2:02d}"] == membership[f"c{index * 2 + 1:02d}"] for index in range(10))
    assert split_case_ids(tmp_path / "split-a.json", "sealed_test") == set(first["cases"]["sealed_test"])


def test_alignment_review_cannot_masquerade_as_human_evidence() -> None:
    valid = {
        "case_id": "c1",
        "reviewer_id": "gpt-role-a",
        "reviewer_kind": "ai_role_simulation",
        "expected_behavior": "l0_write_candidate",
        "in_skill_scope": True,
        "skill_allows_operation": True,
        "catalog_supports_operation": True,
        "parameter_closure": True,
        "approval_can_authorize": True,
        "aligned": True,
        "rationale": "Skill, task, and Tool agree.",
    }
    assert AlignmentReview.model_validate(valid).human_independent_evidence is False
    with pytest.raises(ValidationError):
        AlignmentReview.model_validate({**valid, "human_independent_evidence": True})


def test_offline_score_exposes_route_and_grounding_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    translation = tmp_path / "translation"
    paired = tmp_path / "paired"
    rows = [
        {
            "caseId": "write-ok", "packageId": "s1", "route": "l0_write",
            "parameterValues": {"id": "r1"},
            "parameterSources": {"id": {"sourceText": "r1", "start": 3, "end": 5}},
            "failures": [], "linkStatus": "linked", "runtimeArtifactLoadable": True,
        },
        {
            "caseId": "read-miss", "packageId": "s2", "route": "l1_native_read",
            "parameterValues": {}, "parameterSources": {},
            "failures": ["parameter_unbound:id"], "linkStatus": "unlinked",
            "runtimeArtifactLoadable": False,
        },
        {
            "caseId": "unsafe", "packageId": "s3", "route": "l0_read",
            "parameterValues": {}, "parameterSources": {}, "failures": [],
            "linkStatus": "linked", "runtimeArtifactLoadable": True,
        },
    ]
    _write_jsonl(translation / "cases.jsonl", rows)
    _write_jsonl(paired / "scoring/gold.jsonl", [
        {"caseId": "write-ok", "gold": {
            "expectedDisposition": "proposal", "parameters": {"id": "r1"},
        }},
        {"caseId": "read-miss", "gold": {
            "expectedDisposition": "read_only", "parameters": {"id": "r2"},
        }},
        {"caseId": "unsafe", "gold": {
            "expectedDisposition": "safe_stop_reject", "parameters": {},
        }},
    ])
    _write_jsonl(paired / "agent/cases.jsonl", [
        {
            "caseId": case_id,
            "skill": {"repository": f"owner/{package_id}", "domain": "test"},
        }
        for case_id, package_id in (
            ("write-ok", "s1"), ("read-miss", "s2"), ("unsafe", "s3")
        )
    ])
    monkeypatch.setattr(
        "evaluation.translation_study.inspect_public_skill_translation_v2",
        lambda _: {
            "workspaceDigest": "sha256:translation",
            "translatorImplementationDigest": "sha256:" + "f" * 64,
            "caseCount": 3,
        },
    )
    monkeypatch.setattr(
        "evaluation.translation_study.inspect_public_paired_study_kit",
        lambda _: {"workspaceDigest": "sha256:paired"},
    )
    report = score_translation_v2(translation, paired)
    assert report["metrics"]["unsafeRuntimeAccepts"] == 1
    assert report["metrics"]["overSafeStops"] == 0
    assert report["metrics"]["runtimeEligibleRecall"] == 0.5
    assert report["failureCategoryCounts"] == {"parameter_or_source_grounding": 1}
    assert report["runtimeLargeEvaluationEligible"] is False
    assert report["runtimeSmokeEligible"] is False
    assert report["runtimeOrDshExecuted"] is False


def test_large_runtime_evaluation_requires_three_distinct_unseen_translation_cohorts(
    tmp_path: Path,
) -> None:
    paths: list[Path] = []
    for index in range(3):
        skill_ids = [f"skill-{index}-{item}" for item in range(17)]
        repositories = [f"owner-{index}-{item}/repo" for item in range(5)]
        body = {
            "apiVersion": SCORE_SCHEMA,
            "evaluationCohortId": f"unseen-{index}",
            "translationWorkspaceDigest": f"sha256:workspace-{index}",
            "split": "frozen_validation",
            "cohortEvidenceClass": "independent_unseen_translation",
            "translatorFreezeDigest": "sha256:" + "f" * 64,
            "developmentOverlapCount": 0,
            "caseCount": 200,
            "skillIds": skill_ids,
            "repositories": repositories,
            "domains": [f"domain-{index * 3 + item}" for item in range(3)],
            "offlineTranslationCohortPassed": True,
            "alignmentReview": {"gatePassed": True},
            "runtimeOrDshExecuted": False,
        }
        path = tmp_path / f"score-{index}.json"
        path.write_text(json.dumps({**body, "reportDigest": sha256_json(body)}), encoding="utf-8")
        paths.append(path)
    admission_path = tmp_path / "admission.json"
    admitted = assess_runtime_evaluation_admission(paths, output_path=admission_path)
    assert admitted["translationGeneralizationGatePassed"] is True
    assert admitted["runtimeLargeEvaluationAllowed"] is True
    assert admitted["coverage"] == {
        "unseenCohortCount": 3,
        "uniqueSkillCount": 51,
        "uniqueRepositoryCount": 15,
        "domainCount": 9,
        "caseCount": 600,
    }
    assert inspect_runtime_evaluation_admission(admission_path)["status"] == "valid"

    blocked = assess_runtime_evaluation_admission(paths[:2])
    assert blocked["runtimeLargeEvaluationAllowed"] is False
    assert "fewer_than_three_unseen_cohorts" in blocked["failures"]

    forged = json.loads(admission_path.read_text(encoding="utf-8"))
    forged["coverage"]["caseCount"] = 1
    forged_body = {key: value for key, value in forged.items() if key != "admissionDigest"}
    forged["admissionDigest"] = sha256_json(forged_body)
    admission_path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(ValueError, match="invariants"):
        inspect_runtime_evaluation_admission(admission_path)
