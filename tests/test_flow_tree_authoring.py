import json
from pathlib import Path

import httpx
import pytest

from evaluation import flow_tree_authoring as author
from evaluation.flow_translation import _write, local_sources
from network_runtime.contracts import sha256_json


def example():
    return json.loads((author.ROOT / "examples/read-flow/flow-tree.json").read_text())


@pytest.fixture
def pilot(tmp_path, monkeypatch):
    model = {"model": "qwen3.5:9b", "modelArtifactDigest": "sha256:test-fixture"}
    monkeypatch.setattr(author.OllamaAnchoredAuthorAdapter, "preflight", lambda self: model)
    parent = tmp_path / "parent"
    parent.mkdir()
    body = {"implementation": {}, "cases": [{"id": "fixture", "sources": local_sources().model_dump(mode="json")}]}
    _write(parent / "manifest.json", {**body, "manifestDigest": sha256_json(body)})
    root = tmp_path / "pilot"
    author.freeze(parent, root)
    return root


def stub_response(monkeypatch, content):
    calls = []

    def post(self, url, **kwargs):
        calls.append(kwargs["json"])
        return httpx.Response(200, request=httpx.Request("POST", url), json={
            "message": {"content": content}, "prompt_eval_count": 1, "eval_count": 2, "done_reason": "stop"})
    monkeypatch.setattr(httpx.Client, "post", post)
    return calls


def test_request_has_tree_schema_without_legacy_graph_instructions():
    wire = author.tree_request(local_sources())
    system = wire["messages"][0]["content"]
    payload = json.loads(wire["messages"][1]["content"])
    assert "nodeSemantics" not in payload
    assert "unresolved_questions" not in system and "zero-based" not in system
    assert set(wire["format"]["properties"]) == {"business_source_ids", "steps", "issues"}
    assert "next" not in wire["format"]["$defs"]["TreeRead"]["properties"]
    assert "when_equal" in wire["format"]["$defs"]["TreeIf"]["properties"]
    assert wire["format"]["$defs"]["TreeEnd"]["properties"]["source_id"]["enum"] == list(author.spans(local_sources()))
    assert wire["model"] == "qwen3.5:9b" and wire["think"] is False
    assert wire["options"]["num_predict"] == 2200
    assert payload["hostReadTools"][0]["sourceDeclarations"]
    assert not any("answer" in key.lower() for key in payload)


def test_one_attempt_and_replay_without_model(pilot, monkeypatch):
    calls = stub_response(monkeypatch, json.dumps(example()))
    author.run(pilot)
    author.run(pilot)
    assert len(calls) == 1
    monkeypatch.setattr(author.OllamaAnchoredAuthorAdapter, "preflight", lambda self: pytest.fail("offline replay"))
    result = author.report(pilot)
    assert result["structurallyQualified"] == 1 and result["sourceReviewed"] == 0
    assert result["reviewSupportedInactiveFlows"] == result["runtimeExecutions"] == 0


@pytest.mark.parametrize("content", ["not json", '{"steps": []}', json.dumps({**example(), "steps": [example()["steps"][0]]})])
def test_failed_outputs_preserved_and_never_retried(pilot, monkeypatch, content):
    calls = stub_response(monkeypatch, content)
    author.run(pilot)
    author.run(pilot)
    assert len(calls) == 1
    assert author.report(pilot)["rows"][0]["status"] == "blocked"
    raw = json.loads((pilot / "fixture/response.json").read_text())
    assert json.loads(raw["body"])["message"]["content"] == content


def test_partial_case_is_not_retried(pilot, monkeypatch):
    (pilot / "fixture").mkdir()
    calls = stub_response(monkeypatch, json.dumps(example()))
    with pytest.raises(ValueError, match="partial"):
        author.run(pilot)
    assert not calls


def test_model_drift_before_request(pilot, monkeypatch):
    monkeypatch.setattr(author.OllamaAnchoredAuthorAdapter, "preflight", lambda self: {"model": "different"})
    calls = stub_response(monkeypatch, json.dumps(example()))
    with pytest.raises(ValueError, match="model identity"):
        author.run(pilot)
    assert not calls and not (pilot / "fixture").exists()


@pytest.mark.parametrize("filename", ["status.json", "compilation.json", "sources.json", "tree.json"])
def test_changed_checkpoint_rejected(pilot, monkeypatch, filename):
    stub_response(monkeypatch, json.dumps(example()))
    author.run(pilot)
    path = pilot / "fixture" / filename
    path.write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        author.report(pilot)


def test_receipt_does_not_replace_semantic_replay(pilot, monkeypatch):
    stub_response(monkeypatch, json.dumps(example()))
    author.run(pilot)
    folder = pilot / "fixture"
    (folder / "tree.json").write_text("{}")
    (folder / "receipt.json").write_text(json.dumps(author.receipt(folder)))
    with pytest.raises(ValueError, match="derivation"):
        author.report(pilot)


def test_frozen_implementation_drift(pilot, monkeypatch):
    original = author.digest_file
    monkeypatch.setattr(author, "digest_file", lambda path: "changed" if Path(path).name == "flow_tree.py" else original(path))
    with pytest.raises(ValueError, match="implementation drift"):
        author.manifest_at(pilot)
