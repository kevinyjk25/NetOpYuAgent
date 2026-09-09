"""Aggregation plumbing tests, not model or semantic quality evidence."""

import json

import pytest

from evaluation import flow_batch_evidence as evidence


@pytest.fixture
def sample(tmp_path, monkeypatch):
    structural = {"reportDigest": "structural", "manifestDigest": "frozen", "model": {},
        "flowCount": 2, "structurallyQualified": 1, "rows": [
            {"id": "first", "status": "awaiting_source_review", "latencyMs": 2, "httpStatus": 200,
             "inputTokens": 3, "outputTokens": 4}, {"id": "second", "status": "blocked"}]}
    assessed = {"status": "blocked", "inputDigest": "packet", "reviewDigest": "review", "reportDigest": "result",
        "reviewerId": "test", "reviewerKind": "test_fixture", "assessment": {"verdictCounts": {
            "supported": 0, "contradicted": 1, "insufficient_evidence": 0}},
        "unresolvedQuestions": [], "blockers": [{"claimId": "claim-0001", "l05Pointer": "/purpose",
            "verdict": "contradicted", "rationale": "Deliberate plumbing fixture."}]}
    monkeypatch.setattr(evidence, "report", lambda root: structural)
    monkeypatch.setattr(evidence, "load_run", lambda root: ("source", "draft"))
    monkeypatch.setattr(evidence, "assess", lambda *args: assessed)
    (tmp_path / "first.json").write_text(json.dumps({
        "reviewer_id": "test", "reviewer_kind": "test_fixture", "assessment": {
            "input_digest": "sha256:" + "a" * 64, "scope_note": "Protocol fixture only.", "claims": [{
                "claim_id": "claim-0001", "verdict": "contradicted", "source_span_ids": ["skill-0001"],
                "rationale": "Protocol fixture only.", "suggested_revision": "Inspect source."}]}}))
    (tmp_path / "first-report.json").write_text(json.dumps(assessed))
    return tmp_path, structural


def test_aggregate_keeps_structural_and_semantic_denominators_separate(sample):
    root, _ = sample
    result = evidence.build_evidence(root, root)
    assert result["flowCount"] == 2 and result["sourceReviewedFlows"] == 1
    assert result["reviewSupportedInactiveFlows"] == 0
    assert result["modelResponses"] == 1 and result["modelResponseTotalMs"] == 2
    assert result["claimVerdicts"]["contradicted"] == 1
    assert result["runtimeExecutions"] == 0 and not result["independentHumanEvidence"]


def test_incomplete_batch_cannot_publish_final_evidence(sample):
    root, structural = sample
    structural["rows"][1]["status"] = "not_run"
    with pytest.raises(ValueError, match="not complete"):
        evidence.build_evidence(root, root)


def test_saved_assessment_must_match_replay(sample):
    root, _ = sample
    (root / "first-report.json").write_text("{}")
    with pytest.raises(ValueError, match="differs"):
        evidence.build_evidence(root, root)


def test_missing_review_is_not_a_zero_or_success(sample):
    root, _ = sample
    (root / "first.json").unlink()
    with pytest.raises(FileNotFoundError):
        evidence.build_evidence(root, root)
