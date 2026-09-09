"""Offline protocol/transport fixtures, never measured GPT or 9B accuracy."""

import json

import httpx
import pytest

from evaluation import flow_model_comparison as pilot
from evaluation import flow_model_transport as transport
from evaluation.flow_node_evidence import compile_evidence, request
from evaluation.flow_translation import _write
from evaluation.flow_tree_authoring import receipt
from evaluation.flow_tree_capabilities import bounded_request
from tests.test_flow_lean_mapping import review_fixture
from tests.test_flow_node_evidence import fixture


def envelope(arm, value, **updates):
    if arm == "openai":
        raw = dict(model=transport.GPT_MODEL, status="completed", usage=dict(input_tokens=11, output_tokens=22,
            output_tokens_details=dict(reasoning_tokens=5)), output=[dict(type="message", role="assistant",
            content=[dict(type="output_text", text=json.dumps(value))])])
    else:
        raw = dict(model=transport.QWEN_MODEL, done=True, done_reason="stop", prompt_eval_count=11,
            eval_count=22, message=dict(role="assistant", content=json.dumps(value)))
    raw.update(updates)
    return dict(httpStatus=200, body=json.dumps(raw), latencyMs=12)


def prepare(tmp_path, monkeypatch, mode="end-to-end", repeats=1):
    source, tree, mapping = fixture()
    src = tmp_path / "sources.json"
    _write(src, pilot.sealed(dict(cases=[dict(id="fixture", sources=source.model_dump(mode="json"),
                                           previousAnswer="must not leak")]), "manifestDigest"))
    calls = []
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-secret")
    monkeypatch.setattr(pilot.OllamaAnchoredAuthorAdapter, "preflight", lambda self: dict(model=transport.QWEN_MODEL, digest="fixture-only"))

    def send(arm, wire):
        calls.append((arm, wire))
        messages = wire.get("input", wire.get("messages"))
        payload = json.loads(messages[-1]["content"])
        value = mapping if "node_evidence" in payload["requiredOutputSchema"]["properties"] else tree
        return envelope(arm, value.model_dump(mode="json"))

    monkeypatch.setattr(transport, "send", send)
    parent_run = None
    if mode == "mapping-only":
        parent_run = tmp_path / "parent"
        parent = pilot.historical.freeze(src, parent_run)
        folder = parent_run / "fixture" / "flow"
        folder.mkdir(parents=True)
        _write(folder / "request.json", dict(wireRequest=parent["cases"][0]["flowRequest"], model=parent["model"]))
        reply = envelope("ollama", tree.model_dump(mode="json"))
        _write(folder / "response.json", reply)
        files, status = pilot.historical.derive(source, None, reply)
        for name, value in {**files, "status.json": status}.items():
            _write(folder / name, value)
        _write(folder / "receipt.json", receipt(folder))
    root = tmp_path / "probe"
    pilot.freeze(src, root, mode, repeats, parent_run)
    return root, calls, source, tree, mapping


@pytest.mark.parametrize("mode", ["mapping-only", "end-to-end"])
def test_fresh_or_fixed_inputs_identical_between_arms_no_answer_leakage(tmp_path, monkeypatch, mode):
    root, calls, source, tree, mapping = prepare(tmp_path, monkeypatch, mode)
    m = pilot.load(root)
    assert "previousAnswer" not in json.dumps(m)
    assert ("fixedTree" in m["cases"][0]) == (mode == "mapping-only")
    initial = pilot.report(root)
    assert initial["status"] == "pending_model_calls"
    assert all(a["semanticAccuracy"] is None and a["requestP50Ms"] is None for a in initial["arms"].values())
    for arm in pilot.ARMS:
        pilot.run(root, arm, 8)
        pilot.run(root, arm, 8)
    r = pilot.report(root)
    count = 1 if mode == "mapping-only" else 2
    assert len(calls) == count * 2
    assert calls[0][1]["input"] == calls[count][1]["messages"]
    assert r["status"] == "mechanical_results_need_separate_review"
    for arm in pilot.ARMS:
        assert r["arms"][arm]["mappingQualified"] == 1
        assert r["arms"][arm]["sourceReviewed"] == 0
        assert r["arms"][arm]["semanticAccuracy"] is None
        assert r["arms"][arm]["inputTokens"] == count * 11
    assert r["runtimeExecutions"] == r["businessWrites"] == 0
    assert r["decision"] == "not_established"


@pytest.mark.parametrize("phase", ["flow", "mapping"])
def test_original_schema_gate_not_relaxed(tmp_path, monkeypatch, phase):
    root, calls, source, tree, mapping = prepare(tmp_path, monkeypatch)
    original = bounded_request(source) if phase == "flow" else request(source, tree)
    value = tree.model_dump(mode="json") if phase == "flow" else mapping.model_dump(mode="json")
    value["undeclared"] = "not allowed"
    files, status = pilot.derive(source, None if phase == "flow" else tree, "openai", original, envelope("openai", value))
    assert status["status"] == "blocked"
    assert "compilation.json" not in files


def test_resumption_recovers_derived_files_without_reposting(tmp_path, monkeypatch):
    root, calls, *_ = prepare(tmp_path, monkeypatch)
    pilot.run(root, "openai", 1)
    assert len(calls) == 1
    folder = root / "openai/0/fixture/flow"
    for name in ("receipt.json", "status.json", "review-input.json"):
        (folder / name).unlink()
    pilot.run(root, "openai", 1)
    assert len(calls) == 2  # Only the previously unstarted mapping.
    pilot.report(root)


def test_uncertain_response_never_retried(tmp_path, monkeypatch):
    root, calls, *_ = prepare(tmp_path, monkeypatch)
    pilot.run(root, "openai", 1)
    folder = root / "openai/0/fixture/flow"
    (folder / "response.json").unlink()
    assert pilot.report(root)["arms"]["openai"]["rows"][0]["flow"]["status"] == "execution_uncertain"
    with pytest.raises(ValueError, match="uncertain"):
        pilot.run(root, "openai", 1)
    assert len(calls) == 1


@pytest.mark.parametrize("mutation", ["raw", "status", "request", "implementation"])
def test_checkpoint_tampering_or_drift_is_not_scored(tmp_path, monkeypatch, mutation):
    root, _, *_ = prepare(tmp_path, monkeypatch)
    pilot.run(root, "openai", 2)
    if mutation == "implementation":
        monkeypatch.setattr(pilot, "implementation", lambda: {})
    else:
        name = {"raw": "response", "status": "status", "request": "request"}[mutation]
        (root / "openai/0/fixture/mapping" / (name + ".json")).write_text("{}")
    with pytest.raises((ValueError, KeyError)):
        pilot.report(root)


def test_missing_credentials_no_network_or_attempt_checkpoint(tmp_path, monkeypatch):
    root, calls, *_ = prepare(tmp_path, monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(ValueError, match="not configured"):
        pilot.run(root, "openai", 2)
    assert not calls and not (root / "openai").exists()


def test_full_review_is_explicit_digest_and_experiment_bound(tmp_path, monkeypatch):
    root, _, source, tree, mapping = prepare(tmp_path, monkeypatch)
    pilot.run(root, "openai", 2)
    m = pilot.load(root)
    review = review_fixture(compile_evidence(source, tree, mapping)["reviewInput"])
    reviews = tmp_path / "reviews"
    folder = reviews / "openai/0"
    folder.mkdir(parents=True)
    context = dict(manifestDigest=m["manifestDigest"], arm="openai", repeat=0, case="fixture")
    _write(folder / "fixture.json", dict(context=context, review=review.model_dump(mode="json")))
    r = pilot.report(root, reviews)
    assert r["arms"]["openai"]["sourceReviewed"] == 1
    assert r["arms"]["openai"]["semanticAccuracy"] is None  # Fixture is not Gold.
    data = pilot.read_json(folder / "fixture.json")
    data["context"]["arm"] = "ollama"
    (folder / "fixture.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="review experiment"):
        pilot.report(root, reviews)


@pytest.mark.parametrize("mode", ["model", "refusal", "incomplete", "tool", "ambiguous", "bad-json", "usage", "http"])
def test_openai_envelope_errors_are_not_translation_success(mode):
    reply = envelope("openai", {})
    raw = json.loads(reply["body"])
    expected = {"model": "model_identity_mismatch", "refusal": "model_refusal", "incomplete": "incomplete_response",
                "tool": "unexpected_tool_output", "ambiguous": "malformed_response", "bad-json": "malformed_response",
                "usage": "malformed_response", "http": "transport_error"}[mode]
    if mode == "model":
        raw["model"] = "unrequested-model"
    elif mode == "refusal":
        raw["output"][0]["content"] = [dict(type="refusal", refusal="no")]
    elif mode == "incomplete":
        raw["status"] = "incomplete"
    elif mode == "tool":
        raw["output"].append(dict(type="function_call", name="forbidden"))
    elif mode == "ambiguous":
        raw["output"].append(raw["output"][0])
    elif mode == "usage":
        raw["usage"]["input_tokens"] = -1
    reply["body"] = "not JSON" if mode == "bad-json" else json.dumps(raw)
    if mode == "http":
        reply["httpStatus"] = 429
    text, status = transport.decode("openai", reply)
    assert text is None and status["status"] == expected


@pytest.mark.parametrize("error", [False, True])
def test_transport_fixed_endpoint_secret_redaction_and_no_retry(monkeypatch, error):
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-never-save")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://untrusted.invalid")
    calls = []

    def post(self, url, **kwargs):
        calls.append(url)
        assert kwargs["headers"]["Authorization"] == "Bearer test-secret-never-save"
        if error:
            raise httpx.ConnectError("test-secret-never-save")
        return httpx.Response(400, text="test-secret-never-save")

    monkeypatch.setattr(httpx.Client, "post", post)
    result = transport.send("openai", {})
    assert calls == ["https://api.openai.com/v1/responses"]
    assert "test-secret" not in json.dumps(result)


def test_repeat_budget_and_source_only_freeze(tmp_path, monkeypatch):
    root, calls, *_ = prepare(tmp_path, monkeypatch, repeats=3)
    pilot.run(root, "openai", 1)
    assert len(calls) == 1
    assert not pilot.report(root)["arms"]["openai"]["completed"]
    for invalid in (None, 0, 25):
        with pytest.raises(ValueError, match="max-calls"):
            pilot.run(root, "openai", invalid)


@pytest.mark.parametrize("reason", ["http", "identity"])
def test_provider_failure_stops_remaining_budget_without_retry(tmp_path, monkeypatch, reason):
    root, calls, *_ = prepare(tmp_path, monkeypatch, repeats=3)

    def failed(arm, wire):
        calls.append(arm)
        reply = envelope(arm, {}, model="wrong" if reason == "identity" else transport.GPT_MODEL)
        if reason == "http":
            reply["httpStatus"] = 401
        return reply

    monkeypatch.setattr(transport, "send", failed)
    for _ in range(2):
        with pytest.raises(ValueError, match="model run stopped"):
            pilot.run(root, "openai", 24)
    assert len(calls) == 1
    assert pilot.report(root)["arms"]["openai"]["recordedResponses"] == 1


def test_nonfinite_proposal_is_retained_only_in_raw_response(tmp_path, monkeypatch):
    _, _, source, tree, _ = prepare(tmp_path, monkeypatch)
    files, status = pilot.derive(source, None, "openai", bounded_request(source), envelope("openai", {"bad": float("nan")}))
    assert status["status"] == "blocked" and not files


@pytest.mark.parametrize("field,value", [("done", False), ("done_reason", "length"), ("model", "qwen3.5:27b")])
def test_ollama_completion_and_model_identity_required(field, value):
    text, result = transport.decode("ollama", envelope("ollama", {}, **{field: value}))
    assert text is None
    assert result["status"] == ("model_identity_mismatch" if field == "model" else "incomplete_response")


def test_local_model_digest_drift_blocks_new_phase_and_changed_receipt(tmp_path, monkeypatch):
    root, calls, *_ = prepare(tmp_path, monkeypatch)
    pilot.run(root, "ollama", 1)
    monkeypatch.setattr(pilot.OllamaAnchoredAuthorAdapter, "preflight", lambda self: dict(model=transport.QWEN_MODEL, digest="changed"))
    with pytest.raises(ValueError, match="checkpoint derivation"):
        pilot.run(root, "ollama", 1)
    assert len(calls) == 1
    (root / "ollama-identity.json").write_text('{}')
    with pytest.raises(ValueError, match="request binding"):
        pilot.report(root)


@pytest.mark.parametrize("original_builder", ["flow", "mapping"])
def test_common_prompt_contains_full_original_schema_without_inplace_changes(original_builder):
    source, tree, _ = fixture()
    original = bounded_request(source) if original_builder == "flow" else request(source, tree)
    before = json.dumps(original)
    openai = transport.wire_request(original, "openai", 0)
    ollama = transport.wire_request(original, "ollama", 0)
    assert openai["input"] == ollama["messages"]
    assert json.loads(openai["input"][-1]["content"])["requiredOutputSchema"] == original["format"]
    assert before == json.dumps(original)
    assert openai["store"] is False and "tools" not in openai
    assert openai["text"]["format"]["type"] == "json_object" and ollama["format"] == "json"
