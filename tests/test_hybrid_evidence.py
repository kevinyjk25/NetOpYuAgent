import pytest

from evaluation.hybrid_evidence import summarize
from evaluation.stage2_batch import digest
from evaluation.structured_authoring import seal
from evaluation.structured_binding_probe import write_artifacts


def inputs(tmp_path, *, accept=False, duplicate=False):
    prep, run, review = (tmp_path / name for name in ("prep", "run", "review"))
    write_artifacts(prep / "x", {"input.json": {"source": "inert example"}})
    manifest = seal({"protocol": "synthetic-unit-test", "cases": [{"id": "x", "repository": "example/repo", "domain": "unit"}],
        "artifactDigests": {"x/input.json": digest(prep / "x/input.json")}})
    write_artifacts(prep / "freeze", {"manifest.json": manifest})
    row = {"id": "x", "status": "blocked_before_model_resource_budget", "rounds": []}
    summary = seal({"preparationDigest": manifest["reportDigest"], "rows": [row, row] if duplicate else [row]})
    write_artifacts(run / "summary", {"report.json": summary})
    write_artifacts(review, {"report.json": seal({"runDigest": summary["reportDigest"], "cases": [{"id": "x",
        "scopedPlanAccepted": accept, "reviewKind": "developer_ai_not_independent_gold"}]})})
    return prep, run, review / "report.json"


def test_report_preserves_zero_calls_and_unknown_accuracy(tmp_path):
    result = summarize(*inputs(tmp_path), tmp_path / "out")
    assert result["skillCount"] == 1 and result["calls"] == 0
    assert result["semanticAccuracyOnUnseenSkills"] is None
    assert not result["largeRuntimeABUnlocked"]
    assert result["reviewedUsefulPlanKinds"] == {}


@pytest.mark.parametrize("kwargs", [{"accept": True}, {"duplicate": True}])
def test_no_accepting_uncompiled_or_duplicate_skill_rows(tmp_path, kwargs):
    paths = inputs(tmp_path, **kwargs)
    with pytest.raises(ValueError):
        summarize(*paths, tmp_path / "out")


def test_bound_source_tampering_rejected(tmp_path):
    prep, run, review = inputs(tmp_path)
    (prep / "x/input.json").write_text('{"source":"changed"}')
    with pytest.raises(ValueError, match="artifact drift"):
        summarize(prep, run, review, tmp_path / "out")
