import json

import httpx
import pytest

from evaluation import flow_semantic_pilot as pilot
from evaluation.flow_translation import _write, local_sources
from network_runtime.contracts import sha256_json


def tree():
    raw = json.loads((pilot.ROOT / "examples/read-flow/flow-tree.json").read_text())
    raw["objective_source_ids"] = raw.pop("business_source_ids")
    raw["constraints"] = []
    return raw


def envelope(raw):
    return {"httpStatus": 200, "latencyMs": 1, "body": json.dumps({"message": {"content": raw},
        "prompt_eval_count": 10, "eval_count": 20, "done_reason": "stop"})}


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot.OllamaAnchoredAuthorAdapter, "preflight", lambda self: {"model": "qwen3.5:9b", "digest": "test-only"})
    parent = tmp_path / "parent"
    parent.mkdir()
    body = {"cases": [{"id": "fixture", "sources": local_sources().model_dump(mode="json")}]}
    _write(parent / "manifest.json", {**body, "manifestDigest": sha256_json(body)})
    output = tmp_path / "run"
    pilot.freeze(parent, output)
    return output


def stub(monkeypatch, content):
    calls = []

    def post(self, url, **kwargs):
        calls.append(kwargs["json"])
        return httpx.Response(200, request=httpx.Request("POST", url), content=envelope(content)["body"])
    monkeypatch.setattr(httpx.Client, "post", post)
    return calls


def test_frozen_request_has_no_answer_or_case_label(root):
    manifest = pilot.load(root)
    wire = manifest["cases"][0]["wireRequest"]
    payload = json.loads(wire["messages"][1]["content"])
    assert "expectedObject" not in payload and "fixture" not in payload
    assert payload["outputSchema"] == wire["format"]
    assert "TreeEffect" not in json.dumps(wire["format"])
    assert wire["model"] == "qwen3.5:9b" and wire["options"]["num_predict"] == 2200


def test_no_repeat_no_provider_and_offline_replay(root, monkeypatch):
    calls = stub(monkeypatch, json.dumps(tree()))
    pilot.run(root)
    pilot.run(root)
    assert len(calls) == 1
    monkeypatch.setattr(pilot.OllamaAnchoredAuthorAdapter, "preflight", lambda self: pytest.fail("offline report must not call model"))
    result = pilot.report(root)
    assert result["structurallyQualified"] == 1 and result["sourceReviewed"] == 0
    assert result["reviewSupportedInactiveFlows"] == result["runtimeExecutions"] == 0


@pytest.mark.parametrize("content", ['{"steps":', '{"unexpected": true}', json.dumps({**tree(), "steps": tree()["steps"][:1]})])
def test_failed_derivations_preserve_cost_and_original(root, monkeypatch, content):
    calls = stub(monkeypatch, content)
    pilot.run(root)
    pilot.run(root)
    row = pilot.report(root)["rows"][0]
    assert len(calls) == 1 and row["status"] == "blocked"
    assert row["inputTokens"] == 10 and row["outputTokens"] == 20
    saved = json.loads((root / "fixture/response.json").read_text())
    assert json.loads(saved["body"])["message"]["content"] == content


def test_decoder_cannot_invent_host_effect():
    raw = tree()
    raw["steps"] = [{"kind": "effect_candidate", "source_id": "s0001", "binding_id": "made-up", "arguments": {}}]
    files, status = pilot.derive(local_sources(), envelope(json.dumps(raw)))
    assert "tree.json" in files and "compilation.json" not in files
    assert status["status"] == "blocked"


def test_environment_drift_blocks_before_request(root, monkeypatch):
    monkeypatch.setattr(pilot, "environment", lambda: {})
    calls = stub(monkeypatch, json.dumps(tree()))
    with pytest.raises(ValueError, match="drift"):
        pilot.run(root)
    assert not calls


def test_model_drift_blocks_before_request(root, monkeypatch):
    monkeypatch.setattr(pilot.OllamaAnchoredAuthorAdapter, "preflight", lambda self: {})
    calls = stub(monkeypatch, json.dumps(tree()))
    with pytest.raises(ValueError, match="model identity"):
        pilot.run(root)
    assert not calls


def test_partial_checkpoint_never_retried(root, monkeypatch):
    (root / "fixture").mkdir()
    calls = stub(monkeypatch, json.dumps(tree()))
    with pytest.raises(ValueError, match="partial"):
        pilot.run(root)
    assert not calls


def test_recomputed_receipt_cannot_hide_forged_tree(root, monkeypatch):
    stub(monkeypatch, json.dumps(tree()))
    pilot.run(root)
    folder = root / "fixture"
    (folder / "tree.json").write_text("{}")
    (folder / "receipt.json").write_text(json.dumps(pilot.receipt(folder)))
    with pytest.raises(ValueError, match="derivation"):
        pilot.report(root)


def fixture_review(root):
    packet = json.loads((root / "fixture/review-input.json").read_text())
    # Test-only review exercises binding/coverage, never semantic evidence.
    claims = []
    for claim in packet["claims"]:
        citations = [claim.get("requiredCitationId", "skill-0001")] if "skill" in claim["requiredEvidenceKinds"] else []
        if "host" in claim["requiredEvidenceKinds"]:
            citations.append("host-0001")
        claims.append({"claim_id": claim["claimId"], "verdict": "supported", "source_span_ids": citations,
                       "rationale": "Test fixture only for digest and coverage plumbing, not semantic truth.", "suggested_revision": ""})
    return {"reviewer_id": "unit-test", "reviewer_kind": "test_fixture", "assessment": {
        "input_digest": packet["inputDigest"], "scope_note": "Synthetic review binding fixture only, not Gold or actual source review.", "claims": claims}}


@pytest.mark.parametrize("issue", [False, True])
def test_review_is_bound_inactive_and_cannot_erase_issues(root, monkeypatch, issue):
    raw = tree()
    if issue:
        raw["issues"] = [{"kind": "missing_host_capability", "source_id": "s0007", "question": "Test-only unresolved issue retained for gate verification."}]
    stub(monkeypatch, json.dumps(raw))
    pilot.run(root)
    reviews = root.parent / "reviews"
    reviews.mkdir()
    _write(reviews / "fixture.json", fixture_review(root))
    result = pilot.report(root, reviews)
    assert result["sourceReviewed"] == 1 and result["runtimeExecutions"] == 0
    assert result["reviewSupportedInactiveFlows"] == (0 if issue else 1)
    assert result["rows"][0]["review"]["reviewerKind"] == "test_fixture"


@pytest.mark.parametrize("mutation", ["digest", "missing-claim"])
def test_review_drift_and_missing_claim_rejected(root, monkeypatch, mutation):
    stub(monkeypatch, json.dumps(tree()))
    pilot.run(root)
    reviews = root.parent / "reviews"
    reviews.mkdir()
    raw = fixture_review(root)
    if mutation == "digest":
        raw["assessment"]["input_digest"] = "sha256:" + "0" * 64
    else:
        raw["assessment"]["claims"].pop()
    _write(reviews / "fixture.json", raw)
    with pytest.raises(ValueError):
        pilot.report(root, reviews)
