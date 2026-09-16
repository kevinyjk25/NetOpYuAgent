"""Tool-free compiler transport boundaries; no real model or semantic claims."""
import json

import httpx
import pytest

from evaluation.semantic_closure_transfer import packet_for
from evaluation.structured_flow_demo import fixture
from skill_authoring import compiler, isolated_compiler as isolated
from skill_authoring.artifacts import read_json


@pytest.mark.parametrize("outcome", ["candidate", "duplicate", "truncated", "timeout", "wrong_model", "schema_mismatch"])
def test_isolated_transport_is_one_bounded_symbolic_request_and_accounts_unknowns(tmp_path, monkeypatch, outcome):
    bundle, *_ = fixture()
    packet = packet_for(bundle, {"task": "Describe the supplied source without executing any operation.",
                                "inputSchema": compiler.obj({"userValue": {"type": "string"}}), "tools": []})
    visible = list(compiler.pages_for(packet))
    calls = []
    candidate = {"mode": "read_prefix", "intent_summary": "Retain the complete original task and source for bounded reasoning without any execution authority.",
                 "reads": [], "boundaries": []}
    envelope = {"model": compiler.MODEL, "done": True, "done_reason": "stop",
        "message": {"content": json.dumps(candidate)}, "prompt_eval_count": 20, "eval_count": 5}
    if outcome == "schema_mismatch":
        envelope["message"]["content"] = json.dumps({**candidate, "authority": True})
    if outcome == "duplicate":
        envelope["message"]["content"] = '{"mode":"read_prefix","mode":"proposal"}'
    if outcome == "truncated":
        envelope["done_reason"] = "length"
    if outcome == "wrong_model":
        envelope["model"] = "unexpected-model"
    class Client:
        def __init__(self, **kwargs):
            assert kwargs == {"timeout": 180, "trust_env": False}
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def get(self, url):
            assert url == "http://127.0.0.1:11434/api/tags"
            return httpx.Response(200, json={"models": [{"name": compiler.MODEL, "digest": "local-test"}]}, request=httpx.Request("GET", url))
        def post(self, url, *, json):
            assert url == "http://127.0.0.1:11434/api/chat"
            calls.append(json)
            if outcome == "timeout":
                raise httpx.ReadTimeout("uncertain")
            return httpx.Response(200, json=envelope, request=httpx.Request("POST", url))
    monkeypatch.setattr(isolated.httpx, "Client", Client)
    if outcome == "candidate":
        assert isolated.invoke(packet, visible, tmp_path) == candidate
    else:
        kind = isolated.ResponseRejected if outcome in {"duplicate", "truncated", "schema_mismatch"} else httpx.ReadTimeout if outcome == "timeout" else ValueError
        with pytest.raises(kind):
            isolated.invoke(packet, visible, tmp_path)
    assert len(calls) == 1 and calls[0] == isolated.make_request(packet, visible)
    original = compiler.make_request(packet, visible)
    assert {k: v for k, v in calls[0].items() if k != "format"} == {k: v for k, v in original.items() if k != "format"}
    assert "tools" not in calls[0] and [m["role"] for m in calls[0]["messages"]] == ["system", "user"]
    payload = json.loads(calls[0]["messages"][1]["content"])
    assert calls[0]["format"] == payload["requiredOutputSchema"]
    assert not {"arguments", "resources", "observations", "conversation", "criteria"} & payload.keys()
    cost = read_json(tmp_path / "cost/report.json")
    assert cost["physicalCallAttempted"] and cost["role"] == "isolated_compiler"
    assert cost["inputTokens"] == (None if outcome == "timeout" else 20)
    assert cost["outputTokens"] == (None if outcome == "timeout" else 5)
    if outcome in {"timeout", "wrong_model"}:
        assert cost["status"] == "outcome_unknown"
    with pytest.raises(FileExistsError):
        isolated.invoke(packet, visible, tmp_path)
    assert len(calls) == 1  # Existing attempt never overwrites evidence or retries.
