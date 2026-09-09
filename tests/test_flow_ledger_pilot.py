import json

import httpx
import pytest

from evaluation import flow_ledger_pilot as pilot
from evaluation.flow_translation import _write, local_sources
from network_runtime.contracts import sha256_json


@pytest.mark.parametrize("valid", [True, False])
def test_frozen_first_attempt_replay_and_no_duplicates(tmp_path, monkeypatch, valid):
    source = local_sources().model_copy(update={"source_text": "Read input device_id once and finish.\nSnapshot data is not live health.\n"})
    model = {"model": "qwen3.5:9b", "digest": "unit-test"}
    monkeypatch.setattr(pilot.OllamaAnchoredAuthorAdapter, "preflight", lambda self: model)
    parent = tmp_path / "parent"
    parent.mkdir()
    body = {"cases": [{"id": "fixture", "sources": source.model_dump(mode="json")}]}
    _write(parent / "manifest.json", {**body, "manifestDigest": sha256_json(body)})
    root = tmp_path / "run"
    pilot.freeze(parent, root)
    raw = {"objective_source_ids": ["s0001"], "source_dispositions": {
        "s0001": [{"kind": "operation", "node_pointers": ["/steps/0", "/steps/1"], "explanation": "Read the input once then finish this test path."}],
        "s0002": [{"kind": "documentation", "explanation": "Snapshot interpretation is preserved, not live health verification."}]},
        "steps": [{"kind": "read", "source_id": "s0001", "bind": "snapshot", "tool": "read_inventory_device",
            "arguments": {"device_id": {"kind": "reference", "source": "input", "field": "device_id"}}},
            {"kind": "end", "source_id": "s0001", "outcome": "read_path_completed"}], "issues": []}
    if not valid:
        raw["source_dispositions"].pop("s0002")
    calls = []

    def post(self, url, **kwargs):
        calls.append(kwargs)
        return httpx.Response(200, request=httpx.Request("POST", url), json={
            "message": {"content": json.dumps(raw)}, "eval_count": 20, "prompt_eval_count": 10})
    monkeypatch.setattr(httpx.Client, "post", post)
    pilot.run(root)
    pilot.run(root)
    assert len(calls) == 1
    report = pilot.report(root)
    assert report["structurallyQualified"] == int(valid)
    assert report["runtimeExecutions"] == report["sourceReviewed"] == 0
    assert report["rows"][0]["outputTokens"] == 20
    (root / "fixture/status.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        pilot.report(root)
