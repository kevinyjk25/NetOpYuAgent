from __future__ import annotations

import copy
import json
from pathlib import Path

import httpx
import pytest

from evaluation.translation_case_authoring import run_anchored_case_authoring
from evaluation.translation_source_alignment import (
    ClaimAssessment, SourceAssessment, build_source_input,
    evaluate_source_assessment, load_source_alignment, prepare_source_alignment,
)
from evaluation.translation_source_review import (
    OllamaSourceReviewAdapter, inspect_source_review, run_source_review,
)
from tests.test_translation_case_authoring import FakeAdapter, _corpus


def _input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    corpus = _corpus(tmp_path, monkeypatch)
    authoring = tmp_path / "author"
    run_anchored_case_authoring(corpus, authoring, batch_id="development-01", adapter=FakeAdapter())
    root = tmp_path / "source-input"
    prepare_source_alignment(
        authoring, corpus, root, case_id="development-01-001-nominal", salt="a" * 64,
    )
    return root


def _assessment(payload: dict, verdict: str = "supported") -> SourceAssessment:
    source_id = next(span["source_span_id"] for span in payload["sourceSpans"] if span["kind"] == "skill")
    return SourceAssessment(
        input_digest=payload["inputDigest"],
        claims=tuple(ClaimAssessment(
            claim_id=claim["claimId"], verdict=verdict,
            source_span_ids=tuple(
                source_id if kind == "skill" else "task-0001" for kind in claim["requiredEvidenceKinds"]
            ),
            rationale="Offline plumbing fixture, not a semantic correctness judgement.",
            suggested_revision="Locate and bind a documented source API before continuing.",
        ) for claim in payload["claims"]),
        scope_note="One narrow operation; this is not whole-Skill coverage.",
    )


def test_checklist_covers_all_parameter_facets_and_steps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _input(tmp_path, monkeypatch)
    payload, manifest = load_source_alignment(root)
    assert len(payload["claims"]) == 9  # two task + four operation + three parameter facets
    assert {c["facet"] for c in payload["claims"]} >= {
        "parameter_existence", "parameter_type", "parameter_requiredness",
        "complete_input_shape", "effect_classification", "step_availability",
    }
    serialized = json.dumps(payload)
    assert "development-01" not in serialized
    assert "example_value" not in serialized
    assert "nominal" not in serialized
    assert "goldIncluded" in serialized and payload["goldIncluded"] is False
    assert manifest["sourcePacketDigest"]
    for span in payload["sourceSpans"]:
        assert span["end"] - span["start"] == len(span["exactQuote"])


@pytest.mark.parametrize("change", ["missing", "duplicate", "unknown", "wrong_digest"])
def test_review_cannot_cherry_pick_claims(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str) -> None:
    payload, _ = load_source_alignment(_input(tmp_path, monkeypatch))
    review = _assessment(payload)
    claims = list(review.claims)
    if change == "missing":
        claims.pop()
    elif change == "duplicate":
        claims[-1] = claims[0]
    elif change == "unknown":
        claims[-1] = claims[-1].model_copy(update={"claim_id": "claim-9999"})
    else:
        review = review.model_copy(update={"input_digest": "other"})
    with pytest.raises(ValueError):
        evaluate_source_assessment(payload, review.model_copy(update={"claims": tuple(claims)}))


@pytest.mark.parametrize("citations", [(), ("unknown-span",), ("task-0001",), ("span-0001", "span-0001")])
def test_source_parameter_support_requires_valid_non_circular_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, citations: tuple,
) -> None:
    payload, _ = load_source_alignment(_input(tmp_path, monkeypatch))
    review = _assessment(payload)
    claims = list(review.claims)
    claims[-1] = claims[-1].model_copy(update={"source_span_ids": citations})
    with pytest.raises(ValueError):
        evaluate_source_assessment(payload, review.model_copy(update={"claims": tuple(claims)}))


@pytest.mark.parametrize("verdict,status", [
    ("supported", "ready_for_reference_drafting_review"),
    ("contradicted", "revise_construct"),
    ("insufficient_evidence", "needs_source_evidence"),
])
def test_gate_is_derived_without_claiming_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verdict: str, status: str,
) -> None:
    payload, _ = load_source_alignment(_input(tmp_path, monkeypatch))
    report = evaluate_source_assessment(payload, _assessment(payload, verdict))
    assert report["status"] == status
    assert report["citationBindingVerified"] is True
    assert report["semanticEntailmentProven"] is False
    assert report["sourceApiSchemaVerified"] is False
    assert report["runtimeAuthorityGranted"] is False
    assert report["humanIndependentEvidence"] is False
    assert all(row["pointer"].startswith("/") for row in report["rows"])


def test_unknown_can_abstain_without_fabricating_quotes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload, _ = load_source_alignment(_input(tmp_path, monkeypatch))
    review = _assessment(payload, "insufficient_evidence")
    review = review.model_copy(update={"claims": tuple(c.model_copy(update={"source_span_ids": ()}) for c in review.claims)})
    assert evaluate_source_assessment(payload, review)["status"] == "needs_source_evidence"
    review = review.model_copy(update={"claims": tuple(c.model_copy(update={"suggested_revision": ""}) for c in review.claims)})
    with pytest.raises(ValueError, match="actionable revision"):
        evaluate_source_assessment(payload, review)


def test_tampered_input_or_parameter_claims_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _input(tmp_path, monkeypatch)
    payload, _ = load_source_alignment(root)
    review = _assessment(payload)
    payload["claims"].pop()
    with pytest.raises(ValueError, match="content digest"):
        evaluate_source_assessment(payload, review)
    (root / "model-input.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="file digest"):
        load_source_alignment(root)


class FakeSourceReviewer:
    def __init__(self) -> None:
        self.calls = 0
    def preflight(self) -> dict:
        return {"model": "offline", "modelArtifactDigest": "fixture"}
    def review(self, payload: dict) -> tuple[str, dict]:
        self.calls += 1
        return _assessment(payload).model_dump_json(), {"modelCalls": 1, "latencyMs": 1.0}


def test_completed_run_is_bound_read_only_and_tamper_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _input(tmp_path, monkeypatch)
    output = tmp_path / "review"
    adapter = FakeSourceReviewer()
    report = run_source_review(root, output, adapter=adapter)
    snapshot = {p: p.read_bytes() for p in output.iterdir()}
    assert run_source_review(root, output, adapter=adapter) == report
    assert adapter.calls == 1
    assert {p: p.read_bytes() for p in output.iterdir()} == snapshot
    assert report["verified"] and not report["implementationDrift"]
    response = json.loads((output / "response.json").read_text())
    response["raw"] = "{}"
    (output / "response.json").write_text(json.dumps(response))
    with pytest.raises(ValueError, match="sealed file drift"):
        inspect_source_review(root, output)


def test_incomplete_run_is_not_replayed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _input(tmp_path, monkeypatch)
    output = tmp_path / "review"
    output.mkdir()
    adapter = FakeSourceReviewer()
    with pytest.raises(ValueError, match="incomplete run"):
        run_source_review(root, output, adapter=adapter)
    assert adapter.calls == 0


def test_no_args_api_claim_is_exposed_not_silently_repaired(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _input(tmp_path, monkeypatch)
    blind = json.loads((root / "blind-input.json").read_text())
    blind["untrustedQuotedSkillFiles"] = [{
        "path": "SKILL.md", "content": "Call health({}) with no arguments. Returns service limits.", "truncated": False,
    }]
    payload = build_source_input(blind)
    review = _assessment(payload)
    claims = tuple(c.model_copy(update={"verdict": "contradicted"}) if spec["facet"] in {
        "complete_input_shape", "parameter_requiredness",
    } else c for c, spec in zip(review.claims, payload["claims"], strict=True))
    report = evaluate_source_assessment(payload, review.model_copy(update={"claims": claims}))
    assert report["status"] == "revise_construct"
    assert report["verdictCounts"]["contradicted"] == 2
    assert payload["declaredNonExecutableToolCatalog"]["capabilities"][0]["inputSchema"]["required"] == ["service_id"]


def test_http_payload_never_contains_private_bindings_or_author_answers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload, _ = load_source_alignment(_input(tmp_path, monkeypatch))
    seen = []
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json={"message": {"content": _assessment(payload).model_dump_json()}})
    original = httpx.Client
    monkeypatch.setattr("evaluation.translation_source_review.httpx.Client", lambda **kw: original(transport=httpx.MockTransport(handler), **kw))
    OllamaSourceReviewAdapter().review(copy.deepcopy(payload))
    assert seen[0]["model"] == "qwen3.5:9b"
    assert seen[0]["think"] is False
    sent = json.loads(seen[0]["messages"][1]["content"])
    assert sent == payload
    assert "private-binding" not in json.dumps(seen)
    assert "development-01" not in json.dumps(seen)
