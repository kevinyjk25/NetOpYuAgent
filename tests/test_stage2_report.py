"""Report denominators and reviews cannot turn preflight stops into model success."""
import copy

import pytest

from evaluation.stage2_report import collect
from evaluation.structured_authoring import seal
from evaluation.structured_binding_probe import write_artifacts


def fixtures(tmp_path, mutate_summary=None, mutate_review=None):
    rows = [{"id": "a", "status": "blocked_before_model_resource_budget"},
            {"id": "b", "status": "l1_only_no_external_read_not_run"}]
    m = seal({"cases": rows, "skillCount": 2, "repositoryCount": 2, "domainCount": 2,
              "protocol": "unit-test", "artifactDigests": {}})
    summary = {"preparationDigest": m["reportDigest"], "rows": copy.deepcopy(rows)}
    review = {"preparationDigest": m["reportDigest"], "reviewKind": "developer_ai_not_independent_gold",
              "reviews": [{"id": r["id"], "reportDigest": None, "verdict": "not_model_evaluated",
                           "findings": ["Static only."], "limitations": ["No model calls."]} for r in rows],
              "limitations": ["Synthetic report plumbing fixture, not semantic evidence."]}
    if mutate_summary:
        mutate_summary(summary)
    if mutate_review:
        mutate_review(review)
    prep, run, decisions = tmp_path / "prep", tmp_path / "run", tmp_path / "review"
    write_artifacts(prep / "freeze", {"manifest.json": m})
    write_artifacts(run / "summary", {"report.json": seal(summary)})
    write_artifacts(decisions, {"review.json": review})
    return prep, run, decisions / "review.json"


def test_static_stops_have_no_semantic_or_cost_success(tmp_path):
    report = collect(*fixtures(tmp_path))
    assert report["skillCount"] == 2 and report["toolBearingSkills"] == 1
    assert report["modelCalls"] == report["modelAttemptedSkills"] == report["acceptedReadRegions"] == 0
    assert report["wholeSkillAccuracy"] is report["parameterOracleAccuracy"] is None
    assert report["requestLatencyMs"] == {"p50": None, "p95": None}
    assert not report["largeRuntimeABUnlocked"]


def test_missing_skill_is_not_removed_from_denominator(tmp_path):
    with pytest.raises(ValueError, match="every selected Skill"):
        collect(*fixtures(tmp_path, mutate_summary=lambda s: s["rows"].pop()))


def test_uncompiled_case_cannot_be_reviewed_as_accepted(tmp_path):
    with pytest.raises(ValueError, match="uncompiled"):
        collect(*fixtures(tmp_path, mutate_review=lambda r: r["reviews"][0].update(verdict="accepted_read_region")))


def test_review_cannot_claim_independence(tmp_path):
    with pytest.raises(ValueError, match="independent Gold"):
        collect(*fixtures(tmp_path, mutate_review=lambda r: r.update(reviewKind="independent_human")))
