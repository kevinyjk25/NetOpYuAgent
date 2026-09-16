"""Probe host wiring checks only; compiler replies are scripted, no inference."""
import copy
import io
import json
import threading
import time
from types import SimpleNamespace

import pytest

from evaluation.bounded_budget import BudgetLedger
from evaluation import bounded_dsh_probe
from evaluation.bounded_dsh_probe import ARGUMENTS, ToolHost, delivered_result, fixture, packet, responder, scripted_proposal
from evaluation.bounded_pilot import write_new
from evaluation.bounded_provider import LocalProviderPool
from evaluation.bounded_transport import MODEL, ModelBroker, ScriptedModel
from skill_authoring import compiler, isolated_compiler
from skill_authoring.model_endpoint import resolve_model_endpoint


def host(tmp_path, treatment):
    ledger = BudgetLedger(tmp_path / "ledger.sqlite")
    ledger.register_study("probe", "protocol")
    candidate = ledger.register_candidate("probe", "code")
    arm = ledger.start_arm("probe", candidate, "one", 1, "B" if treatment else "A", "input")
    provider = LocalProviderPool(tmp_path / "providers").create_arm(arm, "treatment" if treatment else "control", fixture())
    broker = ModelBroker(ledger, arm, ScriptedModel(responder()), tmp_path / "transport")
    routes = tmp_path / "model-routes.json"
    write_new(routes, {"schema": "netopyu.local-model-routes/v1", "model": MODEL,
        "model_digest": broker.backend.digest, "arm_id": arm, "routes": {
            "compile": {"base_url": "http://127.0.0.1:9/r/compile"},
            "runtime": {"base_url": "http://127.0.0.1:9/r/runtime"}}})
    routes.chmod(0o600)
    return ToolHost(provider, broker, tmp_path, treatment=treatment, model_routes=routes)


def test_scripted_candidate_is_admitted_by_existing_compiler_not_handwritten_l0():
    p = packet()
    value = compiler.compile_proposal(p, list(compiler.pages_for(p)), scripted_proposal())
    assert value["status"] == "compiled_mixed_candidate_requires_review"
    assert value["semanticCorrectnessProven"] is False


@pytest.mark.parametrize("treatment", [False, True])
def test_same_public_tool_result_comes_from_independent_database_and_real_runtime(tmp_path, monkeypatch, treatment):
    h = host(tmp_path, treatment)
    called = []
    def scripted(*_):
        called.append(True)
        return scripted_proposal()
    monkeypatch.setattr(isolated_compiler, "invoke", scripted)
    value = h.call({"tool": "read_export", "arguments": ARGUMENTS, "request_id": "read"})
    assert value == {"text": fixture()["state"][ARGUMENTS["path"]]["text"]}
    assert len(h.provider.receipts()) == 1
    assert called == ([True] if treatment else [])
    assert h.route == ("runtime_read_prefix" if treatment else "native")
    if treatment:
        assert h.graph["status"] == "governed_graph_completed"
        assert h.provider.scorer_calls()[0]["origin"] == "runtime"
    with pytest.raises(PermissionError):
        h.call({"tool": "read_export", "arguments": ARGUMENTS, "request_id": "again"})
    assert len(h.provider.receipts()) == 1
    requests = [row for row in h.attempts() if row["event"] == "tool_requested"]
    assert len(requests) == 2 and requests[0]["attempt_id"] != requests[1]["attempt_id"]
    assert h.attempts()[-1]["event"] == "tool_rejected_or_unknown"


def test_deterministic_compiler_rejection_falls_back_before_provider_execution(tmp_path, monkeypatch):
    h = host(tmp_path, True)
    def rejected(*_):
        assert not h.provider.receipts()
        raise isolated_compiler.ResponseRejected("scripted malformed response")
    monkeypatch.setattr(isolated_compiler, "invoke", rejected)
    h.call({"tool": "read_export", "arguments": ARGUMENTS, "request_id": "read"})
    assert h.route == "fallback" and h.graph is None
    assert h.provider.scorer_calls()[0]["origin"] == "agent"
    assert h.broker.agent_stage == "fallback"


def test_unknown_compiler_never_falls_back_or_reads(tmp_path, monkeypatch):
    h = host(tmp_path, True)
    def unknown(*_):
        raise TimeoutError("uncertain fixture")
    monkeypatch.setattr(isolated_compiler, "invoke", unknown)
    with pytest.raises(TimeoutError):
        h.call({"tool": "read_export", "arguments": ARGUMENTS, "request_id": "read"})
    assert h.route is None and not h.provider.receipts()
    assert h.broker.ledger.inspect_arm(h.broker.arm_id)["study_status"] == "halted"
    with pytest.raises(PermissionError):
        h.call({"tool": "read_export", "arguments": ARGUMENTS, "request_id": "after-unknown"})
    assert len([row for row in h.attempts() if row["event"] == "tool_requested"]) == 2
    assert not h.provider.receipts()


def test_wrong_object_is_not_repaired_into_expected_argument(tmp_path, monkeypatch):
    h = host(tmp_path, True)
    monkeypatch.setattr(isolated_compiler, "invoke", lambda *_: scripted_proposal())
    wrong = copy.deepcopy(ARGUMENTS)
    wrong["path"] = "/unknown"
    with pytest.raises(ValueError):
        h.call({"tool": "read_export", "arguments": wrong, "request_id": "wrong"})
    assert h.provider.scorer_calls()[0]["arguments"] == wrong
    assert h.provider.scorer_calls()[0]["outcome"] == "failed"
    assert h.graph["status"] != "governed_graph_completed"


def test_treatment_without_explicit_routes_never_invokes_compiler(tmp_path, monkeypatch):
    h = host(tmp_path, True)
    h.model_routes = None
    monkeypatch.setattr(isolated_compiler, "invoke", lambda *_: pytest.fail("must not resolve a legacy endpoint"))
    with pytest.raises(PermissionError, match="explicit host model routes"):
        h.call({"tool": "read_export", "arguments": ARGUMENTS, "request_id": "read"})
    assert not h.provider.receipts()


def test_invalid_and_closed_attempts_are_durable_without_provider_calls(tmp_path):
    h = host(tmp_path, False)
    with pytest.raises(ValueError):
        h.call({"tool": "not-a-tool", "arguments": ARGUMENTS, "request_id": "invalid"})
    assert h.close()["drained"]
    with pytest.raises(PermissionError):
        h.call({"tool": "read_export", "arguments": ARGUMENTS, "request_id": "closed"})
    requests = [row["details"]["request"]["request_id"] for row in h.attempts() if row["event"] == "tool_requested"]
    assert requests == ["invalid", "closed"] and not h.provider.receipts()
    with pytest.raises(PermissionError):
        h.start()
    with pytest.raises(FileExistsError):
        ToolHost(h.provider, h.broker, tmp_path, treatment=False)


@pytest.mark.parametrize("rejected", [False, True])
def test_close_is_bounded_and_late_compiler_cannot_read_or_fallback(tmp_path, monkeypatch, rejected):
    h = host(tmp_path, True)
    entered, release = threading.Event(), threading.Event()
    outcomes, endpoints = [], []
    def endpoint():
        return resolve_model_endpoint("compile", model=MODEL, default_endpoint="http://127.0.0.1:11434").base_url
    def delayed(*_):
        endpoints.append(endpoint())
        entered.set()
        assert release.wait(3)
        endpoints.append(endpoint())
        if rejected:
            raise isolated_compiler.ResponseRejected("late rejection")
        return scripted_proposal()
    def invoke():
        try:
            h.call({"tool": "read_export", "arguments": ARGUMENTS, "request_id": "late"})
        except BaseException as exc:
            outcomes.append(exc)
    monkeypatch.setattr(isolated_compiler, "invoke", delayed)
    worker = threading.Thread(target=invoke)
    worker.start()
    try:
        assert entered.wait(2)
        before = time.monotonic()
        closed = h.close(timeout=0.05)
        assert time.monotonic() - before < 0.5
        assert closed["closed"] and not closed["drained"]
        assert h.closed and h.halted
    finally:
        release.set()
        worker.join(timeout=3)
    assert not worker.is_alive() and isinstance(outcomes[0], PermissionError)
    assert endpoints == ["http://127.0.0.1:9/r/compile"] * 2
    assert not h.provider.receipts() and h.route is None and h.graph is None
    assert not (tmp_path / "compilation.json").exists()
    assert h.broker.agent_stage == "agent"
    assert h.broker.ledger.inspect_arm(h.broker.arm_id)["study_status"] == "halted"
    assert h.close()["drained"]


def test_late_provider_result_is_not_returned_as_success(tmp_path, monkeypatch):
    h = host(tmp_path, False)
    original = h.provider.invoke
    def late(*args, **kwargs):
        value = original(*args, **kwargs)
        h.close(timeout=0)
        return value
    monkeypatch.setattr(h.provider, "invoke", late)
    with pytest.raises(PermissionError):
        h.call({"tool": "read_export", "arguments": ARGUMENTS, "request_id": "read"})
    assert h.result is None and len(h.provider.receipts()) == 1
    assert not any(row["event"] == "tool_completed" for row in h.attempts())
    h.close()


def fake_handler(h, monkeypatch, *, post=False, failure=None):
    class Server:
        server_port = 12345
        def __init__(self, _address, handler):
            self.RequestHandlerClass = handler
        def serve_forever(self, **_):
            pass
        def shutdown(self):
            pass
        def server_close(self):
            pass
    monkeypatch.setattr(bounded_dsh_probe, "ThreadingHTTPServer", Server)
    h.start()
    handler = object.__new__(h.server.RequestHandlerClass)
    handler.path = "/" + h.token + ("/invoke" if post else "/catalog")
    handler.connection = SimpleNamespace(settimeout=lambda _: None)
    handler.headers = {"Content-Length": "10"}
    handler.rfile = io.BytesIO(b"{}")
    def operation(name):
        def run(*_):
            if failure == name:
                raise BrokenPipeError("scripted uncertain delivery")
        return run
    for name in ("send_response", "send_header", "end_headers"):
        setattr(handler, name, operation(name))
    handler.wfile = SimpleNamespace(write=operation("write"), flush=operation("flush"))
    return handler


@pytest.mark.parametrize("failure", ["send_response", "send_header", "end_headers", "write", "flush"])
def test_http_entire_delivery_failure_halts_and_preserves_attempt(tmp_path, monkeypatch, failure):
    h = host(tmp_path, False)
    handler = fake_handler(h, monkeypatch, failure=failure)
    handler.respond(False)
    assert h.halted and h.broker.ledger.inspect_arm(h.broker.arm_id)["study_status"] == "halted"
    assert h.attempts()[-1]["event"] == "http_delivery_unknown"
    assert handler.close_connection and not h.provider.receipts()
    assert h.close()["drained"]


def test_http_incomplete_body_is_retained_and_never_reaches_provider(tmp_path, monkeypatch):
    h = host(tmp_path, False)
    fake_handler(h, monkeypatch, post=True).respond(True)
    assert h.halted and not h.provider.receipts()
    events = [row["event"] for row in h.attempts()]
    assert "http_body" in events and "http_rejected_or_unknown" in events and "tool_requested" not in events
    assert h.close()["drained"]


@pytest.mark.parametrize("variant,matched", [("success", True), ("wrong_id", False), ("wrong_result", False), ("tool_error", False)])
def test_delivery_requires_actual_result_and_matching_raw_tool_call(tmp_path, variant, matched):
    expected = {"text": "observed independent SQLite text"}
    value = {"error": "PermissionError"} if variant == "tool_error" else {"text": "wrong"} if variant == "wrong_result" else expected
    messages = [{"role": "assistant", "tool_calls": [{"id": "snapshot-call", "type": "function", "function": {
        "name": "read_export", "arguments": json.dumps(ARGUMENTS)}}]},
        {"role": "tool", "tool_call_id": "wrong" if variant == "wrong_id" else "snapshot-call", "content": json.dumps(value)}]
    (tmp_path / "transport" / "one").mkdir(parents=True)
    write_new(tmp_path / "transport" / "one" / "request.json", {"stage": "agent", "source_request": {"messages": messages}})
    assert delivered_result(tmp_path, expected)["matched"] is matched
