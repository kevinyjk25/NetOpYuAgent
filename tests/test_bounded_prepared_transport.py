"""Mechanical unit substitutes, NOT real tokenizer/parser/model evidence."""
import json

import pytest

from evaluation.bounded_budget import BudgetError, BudgetLedger
from evaluation import bounded_preflight as preflight
from evaluation import bounded_runner as runner
from evaluation.bounded_prepared_transport import NoInferenceRunner, PreparedBackend, PreparedBroker
from evaluation.bounded_transport import MODEL


def request():
    return {"model": MODEL, "stream": False, "think": False,
            "messages": [{"role": "user", "content": "fixture only"}],
            "options": {"num_predict": 32, "temperature": 0}}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    pins = {}
    for name in ("renderer", "tokenizer", "model", "codec"):
        path = tmp_path / name
        path.write_bytes(b"not executable - unit fixture")
        pins[name] = {"path": str(path), "sha256": preflight.file_digest(path)}
    assets = preflight.Assets(json.dumps(preflight.seal({
        "schema": preflight.ASSETS_SCHEMA,
        **{name: pins[name] for name in ("renderer", "tokenizer", "model")},
        "dependencies": [], "metadata": {"fixture_only": True}})).encode())

    def helpers(args, payload, timeout):
        if args == [pins["renderer"]["path"]]:
            wire = json.loads(payload)
            return {"rendered_prompt": "fixture prompt", "effective_options": {
                "draft_num_predict": 4, "frequency_penalty": 0, "main_gpu": None,
                "min_p": 0, "num_batch": 512, "num_ctx": 49152, "num_gpu": -1,
                "num_keep": 0, "num_predict": 32, "num_thread": 0, "presence_penalty": 1.5,
                "repeat_last_n": 64, "repeat_penalty": 1, "seed": -1, "stop": None,
                "temperature": 0, "top_k": 20, "top_p": 0.95, "typical_p": 1,
                "use_mmap": None, **wire["options"]}, "identity": {
                    "profile": runner.PROFILE, "ollama_revision": runner.OLLAMA_REVISION,
                    "renderer": "qwen3.5", "parser": "qwen3.5"}, "no_generation": True}
        return {"token_ids": [1, 2, 3], "count": 3, "add_special": True, "parse_special": True}

    monkeypatch.setattr(preflight, "_run", helpers)
    monkeypatch.setattr(runner, "_parse_text", lambda value, content, path, timeout:
                        {"role": "assistant", "content": content})
    ledger = BudgetLedger(tmp_path / "ledger.sqlite")
    ledger.register_study("unit-fixture", "protocol")
    candidate = ledger.register_candidate("unit-fixture", "code")
    arm_id = ledger.start_arm("unit-fixture", candidate, "case", 1, "B", "input")
    receiver = NoInferenceRunner(lambda stage, wire: {"role": "assistant", "content": "fixture"})
    backend = PreparedBackend(assets, pins["codec"], receiver)
    broker = PreparedBroker(ledger, arm_id, backend, tmp_path / "transport")
    yield broker, ledger, receiver, pins
    broker.close()
    receiver.close()


def test_exact_token_payload_and_persistent_reservation_are_one_bound_request(setup):
    broker, ledger, receiver, _ = setup
    result = broker.dispatch(broker.tokens["agent"], "native", request())
    assert result["prompt_eval_count"] == 3
    assert result["eval_count"] == 1  # Explicit synthetic fixture count.
    call = ledger.snapshot("unit-fixture")["calls"][0]
    assert call["status"] == "settled" and call["reserved_input"] == 3
    directory = next(path for path in broker.output.iterdir() if path.is_dir())
    payload = json.loads((directory / "runner-request.json").read_text())
    binding = json.loads((directory / "dispatch-binding.json").read_text())
    assert payload["prompt"] == [1, 2, 3]
    assert payload["n_predict"] == call["reserved_output"] == 32
    assert binding["call_id"] == call["call_id"]
    assert binding["actualModelCalls"] == 0 and binding["liveGenerationEnabled"] is False
    assert receiver.calls[0]["prepared_digest"] == binding["prepared_digest"]
    assert receiver.pending == {}


def test_receiver_claim_cannot_be_replayed_or_replaced(setup):
    broker, _, receiver, _ = setup
    broker._preflight("native", request(), request())
    prepared, raw = broker.pending_prepared
    value, payload = prepared.inspect(), json.loads(raw)
    path = receiver.claim("agent", value, payload)
    with pytest.raises(ValueError, match="drift"):
        receiver.consume(path, {**payload, "prompt": [999]})
    with pytest.raises(PermissionError, match="replayed"):
        receiver.consume(path, payload)
    assert receiver.calls == []


def test_pin_changed_after_reservation_stops_before_send_and_charges_unknown(setup, monkeypatch):
    broker, ledger, receiver, pins = setup
    original = broker._invoke_backend
    def drift(stage, wire, reservation, directory):
        from pathlib import Path
        Path(pins["model"]["path"]).write_text("changed unit fixture")
        return original(stage, wire, reservation, directory)
    monkeypatch.setattr(broker, "_invoke_backend", drift)
    with pytest.raises(ValueError, match="drift"):
        broker.dispatch(broker.tokens["agent"], "native", request())
    assert receiver.calls == []
    assert ledger.snapshot("unit-fixture")["calls"][0]["status"] == "unknown"
    with pytest.raises(BudgetError):
        ledger.check_arm(broker.arm_id)


def test_unknown_runner_usage_halts_without_fallback(setup, monkeypatch):
    broker, ledger, receiver, _ = setup
    original = receiver.consume
    def corrupt(path, payload):
        result = original(path, payload)
        result["timings"].pop("predicted_n")
        return result
    monkeypatch.setattr(receiver, "consume", corrupt)
    with pytest.raises(ValueError, match="usage"):
        broker.dispatch(broker.tokens["compiler"], "native", request())
    assert len(receiver.calls) == 1
    assert ledger.snapshot("unit-fixture")["calls"][0]["status"] == "unknown"
    with pytest.raises(BudgetError):
        broker.set_agent_stage("fallback")


def test_openai_original_and_normalization_remain_bound(setup):
    broker, _, _, _ = setup
    body = {"model": MODEL, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "两个"}, {"type": "text", "text": "片段"}]}],
        "stream": True, "max_tokens": 32, "store": False}
    broker.dispatch(broker.tokens["agent"], "openai", body)
    path = next(broker.output.glob("*/prepared.json"))
    value = json.loads(path.read_text())
    assert value["source_request"] == body
    assert value["wire"]["messages"][0]["content"] == "两个片段"


def test_close_during_preparation_revokes_before_any_reservation(setup, monkeypatch):
    broker, ledger, receiver, _ = setup
    original = broker._preflight
    def close_during(api, source, wire):
        value = original(api, source, wire)
        broker.close()
        return value
    monkeypatch.setattr(broker, "_preflight", close_during)
    with pytest.raises(PermissionError):
        broker.dispatch(broker.tokens["agent"], "native", request())
    assert ledger.snapshot("unit-fixture")["calls"] == []
    assert receiver.calls == []


def test_host_tool_ids_are_bound_to_reservation_and_unique_across_turns(setup, monkeypatch):
    broker, ledger, _, _ = setup
    monkeypatch.setattr(runner, "_parse_text", lambda *_: {"role": "assistant", "content": "",
        "tool_calls": [{"function": {"name": "read", "arguments": {}}}]})
    outputs = [broker.dispatch(broker.tokens["agent"], "native", request()) for _ in range(2)]
    ids = [result["message"]["tool_calls"][0]["id"] for result in outputs]
    request_ids = {c["request_id"] for c in ledger.snapshot("unit-fixture")["calls"]}
    assert len(set(ids)) == 2
    assert set(ids) == {f"call_{request_id}_0" for request_id in request_ids}
