"""Budget broker tests use synthetic counters, not tokenizer or LLM evidence."""
from concurrent.futures import ThreadPoolExecutor
import io
import json
import threading
from types import SimpleNamespace

import pytest

from evaluation.bounded_budget import BudgetError, BudgetLedger
from evaluation.bounded_transport import MODEL, ModelBroker, ScriptedModel, strict_json
import evaluation.bounded_transport as transport
from network_runtime.contracts import sha256_json


def setup(tmp_path, *, arm="B", response=None, clock=None):
    clock = clock or (lambda: 1000.0)
    ledger = BudgetLedger(tmp_path / "ledger.sqlite", clock=clock)
    ledger.register_study("fixture", "protocol")
    candidate = ledger.register_candidate("fixture", "code")
    arm_id = ledger.start_arm("fixture", candidate, "case", 1, arm, "input")
    backend = ScriptedModel(response or (lambda *_: {"content": "Scripted, not an LLM."}))
    return ModelBroker(ledger, arm_id, backend, tmp_path / "transport", clock=clock), ledger


def request():
    return {"model": MODEL, "messages": [{"role": "user", "content": "inert source"}],
            "stream": False, "think": False, "options": {"num_predict": 100}}


def test_roles_share_persisted_budget_without_model_supplied_stage(tmp_path):
    broker, ledger = setup(tmp_path)
    for role in ("agent", "compiler", "runtime"):
        broker.dispatch(broker.tokens[role], "native", request())
    broker.set_agent_stage("fallback")
    broker.dispatch(broker.tokens["agent"], "native", request())
    broker.set_agent_stage("revision")
    broker.dispatch(broker.tokens["agent"], "native", request())
    with pytest.raises(BudgetError):
        broker.set_agent_stage("fallback")
    snapshot = BudgetLedger(ledger.path, clock=ledger.clock).snapshot("fixture")
    assert {row["stage"] for row in snapshot["calls"]} == {"agent", "compiler", "runtime", "fallback", "revision"}
    assert snapshot["usage"]["model_requests"] == 5
    assert len(list(broker.output.glob("*/settlement.json"))) == 5
    for path in broker.output.glob("*/request.json"):
        value = json.loads(path.read_text())
        assert value["actualModelCalls"] == 0
        assert value["preflight"]["counting_method"].startswith("declared_fixture")
    broker.close()
    with pytest.raises(PermissionError):
        broker.dispatch(broker.tokens["agent"], "native", request())


def test_control_cannot_use_intervention_roles_and_body_cannot_change_stage(tmp_path):
    broker, _ = setup(tmp_path, arm="A")
    for role in ("compiler", "runtime"):
        with pytest.raises(PermissionError):
            broker.dispatch(broker.tokens[role], "native", request())
    with pytest.raises(ValueError):
        broker.dispatch(broker.tokens["agent"], "native", {**request(), "stage": "compiler"})
    with pytest.raises(PermissionError):
        broker.dispatch("invalid", "native", request())
    assert broker.backend.calls == []


def test_reservation_precedes_backend_and_failure_halts_all_roles(tmp_path):
    def fail(*_):
        assert ledger.snapshot("fixture")["pending_outcome"]
        raise TimeoutError("scripted unknown")
    broker, ledger = setup(tmp_path, response=fail)
    with pytest.raises(TimeoutError):
        broker.dispatch(broker.tokens["compiler"], "native", request())
    with pytest.raises(BudgetError):
        broker.dispatch(broker.tokens["agent"], "native", request())
    assert len(broker.backend.calls) == 1
    assert ledger.snapshot("fixture")["calls"][0]["status"] == "unknown"


def test_expired_delivery_cannot_be_success_or_allow_next_call(tmp_path):
    clock = [1000.0]
    def delayed(*_):
        clock[0] += 421
        return {"content": "too late"}
    broker, ledger = setup(tmp_path, response=delayed, clock=lambda: clock[0])
    with pytest.raises(TimeoutError):
        broker.dispatch(broker.tokens["agent"], "native", request())
    assert ledger.snapshot("fixture")["study"]["status"] == "halted"
    assert len(broker.backend.calls) == 1


def test_output_preallocation_blocks_call_before_backend(tmp_path):
    broker, _ = setup(tmp_path)
    body = request()
    body["options"]["num_predict"] = 6001
    with pytest.raises(ValueError):
        broker.dispatch(broker.tokens["agent"], "native", body)
    assert broker.backend.calls == []


@pytest.mark.parametrize("body", ['{"x":1,"x":2}', '{"x":NaN}'])
def test_ambiguous_json_rejected(body):
    with pytest.raises(ValueError):
        strict_json(body)


def test_openai_tools_and_source_preserved_before_counting(tmp_path):
    broker, _ = setup(tmp_path)
    wire = {"model": MODEL, "messages": [{"role": "user", "content": "来源❄"}],
            "tools": [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}],
            "max_tokens": 100, "stream": True}
    broker.dispatch(broker.tokens["agent"], "openai", wire)
    saved = json.loads(next(broker.output.glob("*/request.json")).read_text())
    assert saved["wire"]["tools"] == wire["tools"]
    assert saved["wire"]["messages"] == wire["messages"]
    assert saved["wire"]["options"]["num_predict"] == 100


def test_broker_does_not_accept_a_live_forwarder(tmp_path):
    broker, ledger = setup(tmp_path)
    with pytest.raises(ValueError, match="not implemented"):
        ModelBroker(ledger, broker.arm_id, object(), tmp_path / "live")


@pytest.mark.parametrize("api,patch", [
    ("native", {"options": []}),
    ("native", {"options": {"num_predict": 100, "unknown": 1}}),
    ("native", {"options": {"num_predict": 100, "temperature": True}}),
    ("native", {"options": {"num_predict": 100, "top_p": float("nan")}}),
    ("native", {"options": {"num_predict": 100, "num_ctx": -1}}),
    ("native", {"options": {"num_predict": 100, "seed": "seed"}}),
    ("native", {"messages": [42]}),
    ("native", {"messages": [{"role": [], "content": "x"}]}),
    ("native", {"messages": [{"role": "user", "content": ["not text"]}]}),
    ("native", {"messages": [{"role": "user", "content": "x", "images": []}]}),
    ("native", {"format": []}),
    ("native", {"tools": {}}),
    ("native", {"tools": [42]}),
    ("native", {"tools": [{"type": "function", "function": {"name": "read", "parameters": []}}]}),
    ("openai", {"parallel_tool_calls": True}),
    ("openai", {"parallel_tool_calls": None}),
    ("openai", {"stream": "true"}),
    ("openai", {"stream": True, "stream_options": {"include_usage": "true"}}),
    ("openai", {"stream": False, "stream_options": {"include_usage": True}}),
    ("openai", {"messages": [{"role": "assistant", "content": None, "tool_calls": "invalid"}]}),
    ("openai", {"messages": [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": "not JSON"}}]}]}),
    ("openai", {"messages": [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": "[]"}}]}]}),
])
def test_structural_rejections_precede_any_claim_or_backend(tmp_path, api, patch):
    broker, ledger = setup(tmp_path)
    body = request() if api == "native" else {
        "model": MODEL, "messages": request()["messages"], "max_tokens": 100}
    body.update(patch)
    with pytest.raises(ValueError):
        broker.dispatch(broker.tokens["agent"], api, body)
    assert broker.backend.calls == []
    snapshot = ledger.snapshot("fixture")
    assert snapshot["calls"] == [] and snapshot["study"]["status"] == "active"
    assert broker.errors[-1]["code"] == "request_structure_rejected"
    assert broker.errors[-1]["payload_keys"] == list(body)


def test_supported_tool_history_and_controls_remain_supported(tmp_path):
    broker, _ = setup(tmp_path)
    body = {"model": MODEL, "max_tokens": 100, "parallel_tool_calls": False,
        "stream": True, "stream_options": {"include_usage": True},
        "messages": [{"role": "developer", "content": "fixture instruction"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1",
                "type": "function", "function": {"name": "read", "arguments": '{"target":"x"}'}}]},
            {"role": "tool", "tool_call_id": "call_1", "name": "read", "content": "result"}]}
    broker.dispatch(broker.tokens["agent"], "openai", body)
    saved = json.loads(next(broker.output.glob("*/request.json")).read_text())["wire"]
    assert saved["messages"][1]["tool_calls"][0]["function"]["arguments"] == {"target": "x"}
    assert saved["messages"][2]["content"] == "result"
    native = request()
    native["options"].update(num_ctx=49152, seed=20260910, temperature=0)
    broker.dispatch(broker.tokens["compiler"], "native", native)


@pytest.mark.parametrize("fails", [False, True])
def test_close_during_responder_halts_immediately_and_never_returns_result(tmp_path, fails):
    entered, release = threading.Event(), threading.Event()

    def delayed(*_):
        entered.set()
        if not release.wait(5):
            raise AssertionError("test did not release responder")
        if fails:
            raise TimeoutError("fixture failure after close")
        return {"content": "must not escape"}

    broker, ledger = setup(tmp_path, response=delayed)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(broker.dispatch, broker.tokens["agent"], "native", request())
        try:
            assert entered.wait(2)
            closed = broker.close()
            assert not closed["drained"] and closed["workers"] == 1
            snapshot = BudgetLedger(ledger.path, clock=ledger.clock).snapshot("fixture")
            assert snapshot["study"]["status"] == "halted"
            assert snapshot["calls"][0]["status"] == "unknown"
            assert snapshot["usage"]["model_requests"] == 1
        finally:
            release.set()
        with pytest.raises(PermissionError):
            future.result(timeout=3)
    assert broker.close(timeout=1)["drained"]
    snapshot = ledger.snapshot("fixture")
    call = snapshot["calls"][0]
    assert call["status"] == "unknown"
    assert call["charged_output"] == 100  # Late success cannot rewrite unknown usage.
    assert snapshot["study"]["status"] == "halted"
    with pytest.raises(PermissionError):
        broker.dispatch(broker.tokens["agent"], "native", request())
    assert len(broker.backend.calls) == 1


def http_handler(monkeypatch, broker, body, *, api="native"):
    """Run the actual Handler without opening a socket or reaching a model."""
    class Server:
        def __init__(self, _address, handler):
            self.RequestHandlerClass = handler
            self.server_port = 1

        def serve_forever(self):
            pass

        shutdown = server_close = serve_forever

    monkeypatch.setattr(transport, "ThreadingHTTPServer", Server)
    broker.start()
    handler = object.__new__(broker.server.RequestHandlerClass)
    raw = json.dumps(body).encode()
    handler.path = "/r/" + broker.tokens["agent"] + (
        "/api/chat" if api == "native" else "/v1/chat/completions")
    handler.headers = {"Content-Length": str(len(raw))}
    handler.rfile, handler.wfile = io.BytesIO(raw), io.BytesIO()
    handler.connection = SimpleNamespace(settimeout=lambda _: None)
    handler.statuses = []
    handler.send_response = handler.statuses.append
    handler.send_header = lambda *_: None
    handler.end_headers = lambda: None
    return handler


@pytest.mark.parametrize("where,error", [
    ("send_response", BrokenPipeError), ("end_headers", ConnectionResetError),
    ("end_headers", TimeoutError), ("write", TimeoutError), ("flush", OSError),
])
def test_http_delivery_failure_halts_without_second_response_or_lost_charge(tmp_path, monkeypatch, where, error):
    broker, ledger = setup(tmp_path)
    handler = http_handler(monkeypatch, broker, request())

    def fail(*_):
        raise error("scripted socket failure")

    if where == "send_response":
        def send(status):
            handler.statuses.append(status)
            fail()
        handler.send_response = send
    elif where == "end_headers":
        handler.end_headers = fail
    else:
        handler.wfile = SimpleNamespace(write=fail if where == "write" else lambda *_: None,
                                        flush=fail if where == "flush" else lambda: None)
    handler.do_POST()
    assert handler.statuses == [200]  # Never retry with a second HTTP response.
    assert handler.close_connection
    snapshot = ledger.snapshot("fixture")
    assert snapshot["study"]["status"] == "halted"
    assert snapshot["calls"][0]["status"] == "settled"
    assert snapshot["calls"][0]["charged_output"] == 16
    assert snapshot["usage"]["model_requests"] == 1
    with pytest.raises(BudgetError):
        broker.dispatch(broker.tokens["compiler"], "native", request())
    broker.close()


def test_close_after_settlement_before_http_delivery_cannot_send_success(tmp_path, monkeypatch):
    broker, ledger = setup(tmp_path)
    handler = http_handler(monkeypatch, broker, request())
    dispatch = broker.dispatch

    def close_after_dispatch(*args):
        value = dispatch(*args)
        broker.close()
        return value

    monkeypatch.setattr(broker, "dispatch", close_after_dispatch)
    handler.do_POST()
    assert handler.statuses == [] and handler.wfile.getvalue() == b""
    assert handler.close_connection
    assert ledger.snapshot("fixture")["study"]["status"] == "halted"
    assert ledger.snapshot("fixture")["calls"][0]["status"] == "settled"


def test_http_structure_rejection_is_diagnostic_without_payload_or_capability(tmp_path, monkeypatch):
    broker, ledger = setup(tmp_path)
    body = request()
    body["messages"][0]["content"] = "private prompt body"
    body["options"] = []
    handler = http_handler(monkeypatch, broker, body)
    handler.do_POST()
    assert handler.statuses == [409]
    diagnostics = json.dumps(broker.errors)
    assert "private prompt body" not in diagnostics and broker.tokens["agent"] not in diagnostics
    assert broker.errors[0]["message"] == "supported options object required"
    assert broker.errors[0]["model"] == MODEL
    assert broker.errors[0]["message_roles"] == ["user"]
    assert ledger.snapshot("fixture")["calls"] == [] and broker.backend.calls == []
    broker.close()


def test_http_response_conversion_failure_keeps_settlement_and_stops_study(tmp_path, monkeypatch):
    broker, ledger = setup(tmp_path, response=lambda *_: {"tool_calls": [{"function": {}}]})
    body = {"model": MODEL, "max_tokens": 100, "messages": request()["messages"]}
    handler = http_handler(monkeypatch, broker, body, api="openai")
    handler.do_POST()
    assert handler.statuses == [409]
    snapshot = ledger.snapshot("fixture")
    assert snapshot["study"]["status"] == "halted" and snapshot["calls"][0]["status"] == "settled"
    with pytest.raises(BudgetError):
        broker.dispatch(broker.tokens["agent"], "native", request())
    broker.close()


def test_queued_role_cannot_pass_a_failed_http_delivery(tmp_path, monkeypatch):
    broker, ledger = setup(tmp_path)
    handler = http_handler(monkeypatch, broker, request())
    sending, release = threading.Event(), threading.Event()

    def fail_after_wait():
        sending.set()
        if not release.wait(5):
            raise AssertionError("test did not release delivery")
        raise TimeoutError("scripted delivery timeout")

    handler.end_headers = fail_after_wait
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(handler.do_POST)
        try:
            assert sending.wait(2)
            second = pool.submit(broker.dispatch, broker.tokens["runtime"], "native", request())
        finally:
            release.set()
        first.result(timeout=3)
        with pytest.raises(BudgetError):
            second.result(timeout=3)
    assert len(broker.backend.calls) == 1
    assert ledger.snapshot("fixture")["usage"]["model_requests"] == 1
    broker.close()


def test_installed_pi_ai_text_request_normalizes_without_losing_text_or_source(tmp_path, monkeypatch):
    broker, ledger = setup(tmp_path)
    # pi-ai 0.82.1 dist/api/openai-completions.js: buildParams 512-559,
    # convertMessages 803-831, function tools 1044-1053. No real DSH or LLM.
    parts = [{"type": "text", "text": " 第一段\n"}, {"type": "text", "text": ""},
             {"type": "text", "text": "❄e\u0301\t尾部 "}]
    body = {"model": MODEL, "messages": [{"role": "system", "content": "fixture instruction"},
                {"role": "user", "content": parts}],
            "stream": True, "stream_options": {"include_usage": True}, "store": False,
            "max_tokens": 4096, "tools": [{"type": "function", "function": {"name": "read",
                "description": "fixture read", "parameters": {"type": "object", "properties": {}},
                "strict": False}}]}
    original = json.loads(json.dumps(body))
    handler = http_handler(monkeypatch, broker, body, api="openai")
    handler.do_POST()
    assert handler.statuses == [200] and b"data: [DONE]" in handler.wfile.getvalue()
    saved = json.loads(next(broker.output.glob("*/request.json")).read_text())
    assert saved["source_request"] == original == body
    assert saved["source_request_digest"] == sha256_json(original)
    assert saved["normalization"] == "scripted_openai_native_text_v1"
    assert saved["wire"]["messages"][1]["content"] == " 第一段\n❄e\u0301\t尾部 "
    assert saved["wire"]["tools"] == body["tools"]
    assert "store" not in saved["wire"]
    assert saved["preflight"]["request_digest"] == sha256_json(saved["wire"])
    assert saved["preflight"]["counting_method"] == "declared_fixture_count_not_live_tokenization"
    assert ledger.snapshot("fixture")["usage"]["model_requests"] == 1
    broker.close()


@pytest.mark.parametrize("value", [True, None, 0, "false", {}])
def test_store_only_accepts_literal_false_without_claim(tmp_path, value):
    broker, ledger = setup(tmp_path)
    body = {"model": MODEL, "messages": request()["messages"], "max_tokens": 100, "store": value}
    with pytest.raises(ValueError, match="only store=false"):
        broker.dispatch(broker.tokens["agent"], "openai", body)
    assert not broker.backend.calls and ledger.snapshot("fixture")["calls"] == []


@pytest.mark.parametrize("parts", [
    [], ["text"], [{"type": "text", "text": None}],
    [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}],
    [{"type": "text", "text": "preserve"}, {"type": "image_url", "image_url": {"url": "fixture"}}],
    [{"type": "input_audio", "input_audio": {"data": "fixture", "format": "wav"}}],
])
def test_nontext_or_annotated_parts_are_rejected_not_silently_filtered(tmp_path, parts):
    broker, ledger = setup(tmp_path)
    body = {"model": MODEL, "messages": [{"role": "user", "content": parts}], "max_tokens": 100}
    with pytest.raises(ValueError, match="pure text content parts"):
        broker.dispatch(broker.tokens["agent"], "openai", body)
    assert not broker.backend.calls and ledger.snapshot("fixture")["calls"] == []
