import pytest

from evaluation.hybrid_execution_evidence import collect
from evaluation.structured_authoring import seal
from evaluation.structured_binding_probe import write_artifacts


def fixture(tmp_path, *, blocked=False, review_drift=False):
    construction = seal({"records": [{"id": "x", "review": {"scopedPlanAccepted": True}}], "skillCount": 10,
        "repositoryCount": 10, "domainCount": 9, "compiledSkillCount": 1, "reviewedUsefulPlanKinds": {"pure_l1_reason": 1}})
    live = seal({"case": "x", "compilationDigest": "test-compile", "providerCalls": [], "modelCalls": [],
        "execution": {"status": "blocked" if blocked else "governed_graph_completed", "outputs": {}},
        "runtimeWallLatencyMs": 1, "sourceScriptCalls": 0, "effectCalls": 0})
    write_artifacts(tmp_path / "construction", {"report.json": construction})
    write_artifacts(tmp_path / "live/summary", {"report.json": live})
    write_artifacts(tmp_path / "live/freeze", {"inputs.json": seal({"compilationDigest": "test-compile",
        "fixtureIsInProcessNotRealNetwork": True, "fixture": {}})})
    review = seal({"case": "x", "executionDigest": "drift" if review_drift else live["reportDigest"],
        "reviewKind": "developer_ai_not_independent_gold", "findings": ["synthetic report unit test"],
        "limitations": ["no real model or business observations"], "decision": "useful_partial"})
    check = seal({"case": "x", "compilationDigest": "test-compile", "actualModelCalls": 0, "passed": 0, "total": 0})
    write_artifacts(tmp_path / "review", {"report.json": review})
    write_artifacts(tmp_path / "check", {"report.json": check})
    return tmp_path / "construction/report.json", (tmp_path / "live", tmp_path / "review/report.json", tmp_path / "check/report.json")


def test_pure_l1_or_no_actual_execution_never_satisfies_three_mixed_demonstrations(tmp_path):
    path, case = fixture(tmp_path)
    report = collect(path, [case])
    assert report["reviewedUsefulMixedExecutions"] == 0 and not report["threeUsefulMixedDemonstrationsReached"]
    assert report["wholeSkillConversionRate"] is None and report["stageComplete"] is None


@pytest.mark.parametrize("kwargs", [{"blocked": True}, {"review_drift": True}])
def test_unaccepted_execution_or_mismatched_review_cannot_publish_success(tmp_path, kwargs):
    path, case = fixture(tmp_path, **kwargs)
    with pytest.raises(ValueError):
        collect(path, [case])


def test_same_skill_cannot_be_counted_twice(tmp_path):
    path, case = fixture(tmp_path)
    with pytest.raises(ValueError, match="unique"):
        collect(path, [case, case])
