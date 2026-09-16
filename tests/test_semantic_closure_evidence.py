import json

import pytest

from evaluation.flow_tree_authoring import receipt
from evaluation.semantic_closure_evidence import collect, percentile
from evaluation.structured_binding_probe import write_artifacts
from evaluation.structured_authoring import seal


def test_pending_model_attempt_cannot_be_reported_as_complete(tmp_path):
    write_artifacts(tmp_path / "model", {"request.json": {"wireRequest": {"model": "qwen3.5:9b"}}})
    with pytest.raises(ValueError, match="incomplete"):
        collect(tmp_path)


def test_audit_counts_failures_and_unknown_usage_without_grading_semantics(tmp_path):
    folder = tmp_path / "model"
    write_artifacts(folder, {"request.json": {"wireRequest": {}, "model": {"model": "qwen3.5:9b"}},
        "response.json": {"httpStatus": 400}, "result.json": {"status": "failed", "inputTokens": None, "outputTokens": None, "latencyMs": 10}})
    (folder / "receipt.json").write_text(json.dumps(receipt(folder)))
    report = collect(tmp_path)
    assert report["actualChatAttempts"] == report["callsWithUnknownInputUsage"] == 1
    assert report["semanticSuccessRate"] is None and not report["largeRuntimeABUnlocked"]
    (folder / "response.json").write_text('{}')
    with pytest.raises(ValueError, match="changed case"):
        collect(tmp_path)


def test_percentiles_include_small_batches_without_fabricating_empty_data():
    assert percentile([], .95) is None
    assert percentile([100], .95) == 100
    assert percentile([100, 200, 300], .5) == 200


def test_large_audit_manifest_is_bounded_separate_from_execution_reader(tmp_path):
    from evaluation import semantic_closure_evidence as audit
    from evaluation.structured_binding_probe import read_json as business_read
    path = tmp_path / "large.json"
    value = {"unchanged": "x" * (1024 * 1024)}
    path.write_text(json.dumps(value))
    assert audit.read_json(path) == value
    with pytest.raises(ValueError, match="input JSON exceeds"):
        business_read(path)
    path.write_text('{"duplicate": 1, "duplicate": 2}')
    with pytest.raises(ValueError):
        audit.read_json(path)


def test_audit_reader_enforces_its_own_limit(tmp_path, monkeypatch):
    from evaluation import semantic_closure_evidence as audit
    monkeypatch.setattr(audit, "MAX_EVIDENCE_BYTES", 32)
    path = tmp_path / "large.json"
    path.write_text(json.dumps({"text": "x" * 32}))
    with pytest.raises(ValueError, match="audit byte budget"):
        audit.read_json(path)


def collision_fixture(tmp_path):
    linked = seal({"allScheduledNodesBound": False, "failures": [{"node": "a"}], "semanticSuccess": None})
    write_artifacts(tmp_path / "case/summary", {"report.json": linked})
    old = seal({"case": "case", "status": "bounded_diagnostic_completed", "reportDigest": linked["reportDigest"],
                "allScheduledNodesBound": False, "failedNodeCount": 1})
    write_artifacts(tmp_path / "case/checkpoint", {"report.json": old})
    return tmp_path / "case/checkpoint/report.json", old


def test_legacy_collision_requires_explicit_derivative_and_preserves_original(tmp_path):
    path, _ = collision_fixture(tmp_path)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="report digest drift"):
        collect(tmp_path)
    report = collect(tmp_path, reconstruct_duty_checkpoints=True)
    assert path.read_bytes() == original and not report["allOriginalReportSealsValid"]
    assert report["actualChatAttempts"] == 0 and report["semanticSuccessRate"] is None
    reconstructed = report["checkpointReconstructions"][0]
    assert not reconstructed["originalFileChanged"] and not reconstructed["originalSelfSealValid"]
    assert "caseReportDigest" in reconstructed["reconstructed"]


@pytest.mark.parametrize("mutation", ["status", "case", "extra_key", "count", "digest", "linked_digest", "foreign_path"])
def test_reconstruction_never_ignores_arbitrary_report_drift(tmp_path, mutation):
    path, old = collision_fixture(tmp_path)
    if mutation == "linked_digest":
        linked_path = tmp_path / "case/summary/report.json"
        value = json.loads(linked_path.read_text())
        value["semanticSuccess"] = True
        linked_path.write_text(json.dumps(value))
    elif mutation == "foreign_path":
        write_artifacts(tmp_path / "unrelated", {"report.json": old})
    else:
        key, value = {"status": ("status", "success"), "case": ("case", "other"), "extra_key": ("skip", True),
                      "count": ("failedNodeCount", 0), "digest": ("reportDigest", "sha256:bad")}[mutation]
        old[key] = value
        path.write_text(json.dumps(old))
    with pytest.raises(ValueError):
        collect(tmp_path, reconstruct_duty_checkpoints=True)
