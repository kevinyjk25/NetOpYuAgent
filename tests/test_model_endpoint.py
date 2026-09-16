"""Host routes with mocked/loopback HTTP fixtures; no real model requests."""
import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
import os
import threading

import httpx
import pytest

from evaluation.bounded_budget import BudgetLedger
from evaluation.bounded_transport import ModelBroker, ScriptedModel
from evaluation.semantic_closure_transfer import packet_for
from evaluation.structured_flow_demo import fixture
from skill_authoring import compiler, isolated_compiler, local_execution, model_endpoint, reasoning_transport
from skill_authoring.artifacts import read_json


@pytest.fixture(autouse=True)
def clean_routes(monkeypatch):
    monkeypatch.delenv(model_endpoint.ROUTES_ENV, raising=False)
    monkeypatch.delenv(model_endpoint.ARM_ENV, raising=False)


def configure(tmp_path, monkeypatch, transform=None):
    config = {"schema": model_endpoint.SCHEMA, "model": compiler.MODEL,
              "model_digest": "scripted-fixture-not-real-weights", "arm_id": "fixture-arm-B",
              "routes": {"compile": {"base_url": "http://127.0.0.1:18434/r/compile-private"},
                         "runtime": {"base_url": "http://127.0.0.1:18434/r/runtime-private"}}}
    if transform:
        transform(config)
    path = tmp_path / "routes.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    path.chmod(0o600)
    monkeypatch.setenv(model_endpoint.ROUTES_ENV, str(path))
    monkeypatch.setenv(model_endpoint.ARM_ENV, "fixture-arm-B")
    return config, path


def resolve(role="compile"):
    return model_endpoint.resolve_model_endpoint(role, model=compiler.MODEL,
                                                  default_endpoint="http://127.0.0.1:11434")


def test_unconfigured_legacy_endpoint_is_unchanged(monkeypatch):
    monkeypatch.setenv(model_endpoint.ARM_ENV, "irrelevant-without-explicit-routes")
    endpoint = model_endpoint.resolve_model_endpoint("compile", model=compiler.MODEL,
                                                     default_endpoint="legacy-endpoint-for-test")
    assert endpoint.base_url == "legacy-endpoint-for-test"
    assert endpoint.model_digest is None and endpoint.arm_id is None
    endpoint.check_model_digest("legacy-advertised-value")
    with pytest.raises(ValueError):
        resolve("model-selected-role")


@pytest.mark.parametrize("mode", [0o400, 0o600])
def test_explicit_host_roles_and_digest_pin(tmp_path, monkeypatch, mode):
    config, path = configure(tmp_path, monkeypatch)
    path.chmod(mode)
    for role in ("compile", "runtime"):
        endpoint = resolve(role)
        assert endpoint.base_url == config["routes"][role]["base_url"]
        assert endpoint.role == role and endpoint.arm_id == config["arm_id"]
        endpoint.check_model_digest(config["model_digest"])
        with pytest.raises(ValueError, match="digest mismatch"):
            endpoint.check_model_digest("unrelated")


@pytest.mark.parametrize("change", [
    lambda c: c.update(schema="unknown"),
    lambda c: c.update(model="wrong-model"),
    lambda c: c.update(model_digest=""),
    lambda c: c.update(model_digest=True),
    lambda c: c.update(arm_id="another-arm"),
    lambda c: c.update(extra="forbidden"),
    lambda c: c["routes"].pop("runtime"),
    lambda c: c["routes"].update(fallback={"base_url": "http://127.0.0.1:1000"}),
    lambda c: c["routes"]["compile"].update(timeout=10000),
    lambda c: c["routes"].update(runtime=copy.deepcopy(c["routes"]["compile"])),
    lambda c: c["routes"]["runtime"].update(base_url="HTTP://127.0.0.1:018434/r/compile-private/"),
])
def test_invalid_configuration_never_selects_legacy_default(tmp_path, monkeypatch, change):
    configure(tmp_path, monkeypatch, change)
    with pytest.raises(ValueError):
        resolve()


@pytest.mark.parametrize("url", [
    "http://example.com:18434/r/x", "http://192.168.1.2:18434/r/x",
    "http://localhost:18434/r/x", "https://127.0.0.1:18434/r/x",
    "http://127.0.0.1/r/x", "http://127.0.0.1:0/r/x", "http://127.0.0.1:70000/r/x",
    "http://user:secret@127.0.0.1:18434/r/x", "http://127.0.0.1:18434/r/x?",
    "http://127.0.0.1:18434/r/x#", "http://127.0.0.1:18434/r/../x",
    "http://127.0.0.1:18434/r/%2ex", "http://127.0.0.1:18434//r/x",
    "http://127.0.0.1:18434/r/x\n", "http://127.0.0.1:18434/r\\x",
])
def test_nonlocal_or_ambiguous_route_is_rejected(tmp_path, monkeypatch, url):
    configure(tmp_path, monkeypatch, lambda c: c["routes"]["compile"].update(base_url=url))
    with pytest.raises(ValueError):
        resolve()


def test_ipv6_numeric_loopback_is_supported(tmp_path, monkeypatch):
    configure(tmp_path, monkeypatch,
              lambda c: c["routes"]["compile"].update(base_url="http://[0:0:0:0:0:0:0:1]:18434/r/compiler/"))
    assert resolve().base_url == "http://[::1]:18434/r/compiler"


@pytest.mark.parametrize("problem", ["unset_arm", "blank_env", "missing", "relative", "readable_by_group",
                                      "wrong_owner", "symlink", "directory", "fifo", "oversized", "duplicate", "bad_json"])
def test_untrusted_route_file_fails_closed(tmp_path, monkeypatch, problem):
    _, path = configure(tmp_path, monkeypatch)
    if problem == "unset_arm":
        monkeypatch.delenv(model_endpoint.ARM_ENV)
    elif problem == "blank_env":
        monkeypatch.setenv(model_endpoint.ROUTES_ENV, "")
    elif problem == "missing":
        monkeypatch.setenv(model_endpoint.ROUTES_ENV, str(tmp_path / "missing"))
    elif problem == "relative":
        monkeypatch.setenv(model_endpoint.ROUTES_ENV, "routes.json")
    elif problem == "readable_by_group":
        path.chmod(0o640)
    elif problem == "wrong_owner":
        uid = os.getuid()
        monkeypatch.setattr(model_endpoint.os, "getuid", lambda: uid + 1)
    elif problem == "symlink":
        link = tmp_path / "link.json"
        link.symlink_to(path)
        monkeypatch.setenv(model_endpoint.ROUTES_ENV, str(link))
    elif problem == "directory":
        monkeypatch.setenv(model_endpoint.ROUTES_ENV, str(tmp_path))
    elif problem == "fifo":
        fifo = tmp_path / "fifo"
        os.mkfifo(fifo, 0o600)
        monkeypatch.setenv(model_endpoint.ROUTES_ENV, str(fifo))
    elif problem == "oversized":
        path.write_text(" " * 65537)
    elif problem == "duplicate":
        path.write_text('{"schema":"a","schema":"b"}')
    else:
        path.write_text("{")
    with pytest.raises((ValueError, OSError)):
        resolve()


def invocation(role, folder):
    if role == "compile":
        bundle, *_ = fixture()
        packet = packet_for(bundle, {"task": "Describe the supplied source without executing any operation.",
                                    "inputSchema": compiler.obj({}), "tools": []})
        visible = list(compiler.pages_for(packet))
        candidate = {"mode": "read_prefix", "intent_summary": "Retain the complete original task and source for bounded reasoning without any execution authority.",
                     "reads": [], "boundaries": []}
        return (lambda: isolated_compiler.invoke(packet, visible, folder),
                isolated_compiler.make_request(packet, visible), candidate, 180)
    request = {"nodeId": "n7", "instructions": "Return a candidate, never approval.",
               "inputs": {"role": "compile", "base_url": "http://untrusted.invalid"},
               "outputSchema": compiler.obj({"text": {"type": "string"}}),
               "evidencePolicy": {}, "observationAgesAtStartMs": {}, "maxOutputTokens": 2048}
    expected = {"model": compiler.MODEL, "stream": False, "think": False,
                "format": request["outputSchema"],
                "options": {k: v for k, v in compiler.MODEL_CONFIG.items() if k != "think"},
                "messages": reasoning_transport.messages(request)}
    return (lambda: local_execution.invoke_local(request, folder, []), expected, {"text": "unverified"}, 360)


@pytest.mark.parametrize("role", ["compile", "runtime"])
@pytest.mark.parametrize("outcome", ["valid", "wrong_digest", "timeout", "invalid_config"])
def test_transport_uses_fixed_role_and_original_body_without_retry(tmp_path, monkeypatch, role, outcome):
    config, path = configure(tmp_path, monkeypatch)
    invoke, expected, candidate, timeout = invocation(role, tmp_path / "call")
    calls = []
    base = config["routes"][role]["base_url"]
    envelope = {"model": compiler.MODEL, "done": True, "done_reason": "stop",
                "message": {"content": json.dumps(candidate)}, "prompt_eval_count": 7, "eval_count": 3}

    class Client:
        def __init__(self, **kwargs):
            calls.append(("client", kwargs))
            assert kwargs == {"timeout": timeout, "trust_env": False}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def get(self, url):
            calls.append(("get", url))
            assert url == base + "/api/tags"
            advertised = "wrong" if outcome == "wrong_digest" else config["model_digest"]
            return httpx.Response(200, json={"models": [{"name": compiler.MODEL, "digest": advertised}]},
                                  request=httpx.Request("GET", url))

        def post(self, url, *, json):
            calls.append(("post", url))
            assert url == base + "/api/chat" and json == expected
            assert "arm_id" not in json and "routes" not in json
            if outcome == "timeout":
                raise httpx.ReadTimeout("uncertain mocked request")
            return httpx.Response(200, json=envelope, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "Client", Client)
    if outcome == "invalid_config":
        path.chmod(0o644)
    if outcome == "valid":
        invoke()
    else:
        with pytest.raises(httpx.ReadTimeout if outcome == "timeout" else ValueError):
            invoke()
    assert [kind for kind, _ in calls] == ([] if outcome == "invalid_config" else
                                          ["client", "get"] if outcome == "wrong_digest" else
                                          ["client", "get", "post"])
    if role == "compile" and outcome in {"wrong_digest", "invalid_config"}:
        report = read_json(tmp_path / "call/cost/report.json")
        assert report["physicalCallAttempted"] is False and report["status"] == "not_called"


@pytest.mark.parametrize("binding", ["environment", "context"])
def test_actual_compiler_and_runtime_http_share_one_scripted_broker_arm(tmp_path, monkeypatch, binding):
    """Nested transport only: fixture counts do not qualify 9B token preflight."""
    ledger = BudgetLedger(tmp_path / "fixture-ledger.sqlite")
    ledger.register_study("nested-http-fixture", "scripted-not-live-protocol")
    candidate_id = ledger.register_candidate("nested-http-fixture", "scripted-host")
    arm_id = ledger.start_arm("nested-http-fixture", candidate_id, "nested-transport", 1, "B", "same-input")
    compile_call, compile_wire, compile_candidate, _ = invocation("compile", tmp_path / "compiler")
    runtime_call, runtime_wire, runtime_candidate, _ = invocation("runtime", tmp_path / "runtime")
    expected = {"compiler": (compile_wire, compile_candidate), "runtime": (runtime_wire, runtime_candidate)}
    observed = []

    def respond(stage, wire):
        # The real HTTP handler must reserve the same arm before the callback.
        pending = ledger.inspect_arm(arm_id)["calls"]
        assert pending[-1]["stage"] == stage and pending[-1]["status"] == "reserved"
        assert all(row["arm_id"] == arm_id for row in pending)
        assert wire == expected[stage][0]  # Full prompt/schema/options, not just the model name.
        observed.append((stage, copy.deepcopy(wire)))
        return {"content": json.dumps(expected[stage][1])}

    backend = ScriptedModel(respond, input_tokens=64, output_tokens=16)
    broker = ModelBroker(ledger, arm_id, backend, tmp_path / "transport").start()
    try:
        def routes(config):
            config.update(arm_id=arm_id, model_digest=backend.digest, routes={
                "compile": {"base_url": broker.route("compiler")},
                "runtime": {"base_url": broker.route("runtime")}})

        _, route_file = configure(tmp_path, monkeypatch, routes)
        monkeypatch.setenv(model_endpoint.ARM_ENV, arm_id)
        # Neither the HTTP client nor either actual invocation function is mocked.
        scope = model_endpoint.bind_model_routes(route_file, arm_id) if binding == "context" else nullcontext()
        with scope:
            if binding == "context":
                monkeypatch.delenv(model_endpoint.ROUTES_ENV)
                monkeypatch.delenv(model_endpoint.ARM_ENV)
            assert compile_call() == compile_candidate
            runtime_reply = runtime_call()
        assert runtime_reply.candidate == runtime_candidate
        assert (runtime_reply.input_tokens, runtime_reply.output_tokens) == (64, 16)
        assert observed == [("compiler", compile_wire), ("runtime", runtime_wire)]
        assert [call["stage"] for call in backend.calls] == ["compiler", "runtime"]
        assert broker.errors == []
    finally:
        broker.close()

    ledger.finish_arm(arm_id, "completed")
    # Reopen SQLite to check durable shared accounting, not in-memory callbacks.
    recorded = BudgetLedger(ledger.path).inspect_arm(arm_id)
    assert recorded["arm"]["status"] == "completed" and recorded["study_status"] == "active"
    assert [row["stage"] for row in recorded["calls"]] == ["compiler", "runtime"]
    assert all(row["arm_id"] == arm_id and row["status"] == "settled" for row in recorded["calls"])
    assert [(row["actual_input"], row["actual_output"]) for row in recorded["calls"]] == [(64, 16), (64, 16)]
    assert recorded["arm"]["usage"]["model_requests"] == 2  # Identity GETs are not model calls.
    assert recorded["arm"]["usage"]["charged_input_tokens"] == 128
    assert recorded["arm"]["usage"]["charged_output_tokens"] == 32
    receipts = [read_json(path) for path in broker.output.glob("*/request.json")]
    assert len(receipts) == 2
    for receipt in receipts:
        assert receipt["arm_id"] == arm_id and receipt["wire"] == expected[receipt["stage"]][0]
        assert receipt["actualModelCalls"] == 0 and receipt["measurementKind"] == "scripted_transport_fixture"
        assert receipt["preflight"]["counting_method"] == "declared_fixture_count_not_live_tokenization"


def second_context_config(tmp_path, first):
    config = copy.deepcopy(first)
    config["arm_id"] = "fixture-other-arm"
    config["routes"] = {role: {"base_url": f"http://127.0.0.1:28434/r/{role}-other"}
                        for role in ("compile", "runtime")}
    path = tmp_path / "other-routes.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    path.chmod(0o600)
    return config, path


def test_host_context_is_frozen_against_environment_and_file_changes(tmp_path, monkeypatch):
    config, path = configure(tmp_path, monkeypatch)
    expected_environment = {name: os.environ[name] for name in (model_endpoint.ROUTES_ENV, model_endpoint.ARM_ENV)}
    with model_endpoint.bind_model_routes(path, config["arm_id"]):
        assert {name: os.environ[name] for name in expected_environment} == expected_environment
        initial = {role: resolve(role) for role in ("compile", "runtime")}
        monkeypatch.setenv(model_endpoint.ROUTES_ENV, str(tmp_path / "nonexistent-new-arm"))
        monkeypatch.setenv(model_endpoint.ARM_ENV, "unrelated-arm")
        path.write_text("invalid changed configuration")
        assert {role: resolve(role) for role in initial} == initial
        monkeypatch.delenv(model_endpoint.ROUTES_ENV)
        monkeypatch.delenv(model_endpoint.ARM_ENV)
        assert {role: resolve(role) for role in initial} == initial
        with pytest.raises(ValueError, match="model binding"):
            model_endpoint.resolve_model_endpoint("compile", model="another-model", default_endpoint="fallback")
        with pytest.raises(ValueError, match="host model role"):
            resolve("fallback")
    assert resolve().base_url == "http://127.0.0.1:11434"


def test_nested_host_context_restores_outer_binding_even_after_error(tmp_path, monkeypatch):
    first, first_file = configure(tmp_path, monkeypatch)
    second, second_file = second_context_config(tmp_path, first)
    with model_endpoint.bind_model_routes(first_file, first["arm_id"]):
        outer = resolve()
        with pytest.raises(RuntimeError, match="fixture exit"):
            with model_endpoint.bind_model_routes(second_file, second["arm_id"]):
                assert resolve().arm_id == second["arm_id"] and resolve() != outer
                raise RuntimeError("fixture exit")
        assert resolve() == outer
        with pytest.raises(ValueError, match="binding mismatch"):
            with model_endpoint.bind_model_routes(second_file, "incorrect-arm"):
                pytest.fail("invalid nested context must not enter")
        assert resolve() == outer
    assert resolve() == outer  # Original environment remains untouched.


@pytest.mark.parametrize("problem", ["owner_permissions", "missing_role", "schema", "duplicate"])
def test_host_context_uses_the_same_strict_file_validation(tmp_path, monkeypatch, problem):
    config, path = configure(tmp_path, monkeypatch)
    if problem == "owner_permissions":
        path.chmod(0o644)
    elif problem == "missing_role":
        config["routes"].pop("runtime")
        path.write_text(json.dumps(config))
    elif problem == "schema":
        config["schema"] = "unsupported"
        path.write_text(json.dumps(config))
    else:
        path.write_text('{"schema":"one","schema":"two"}')
    with pytest.raises(ValueError):
        with model_endpoint.bind_model_routes(path, config["arm_id"]):
            pytest.fail("invalid explicit binding must not enter or select a default")


def test_host_contexts_are_isolated_between_concurrent_handler_threads(tmp_path, monkeypatch):
    first, first_file = configure(tmp_path, monkeypatch)
    second, second_file = second_context_config(tmp_path, first)
    monkeypatch.delenv(model_endpoint.ROUTES_ENV)
    monkeypatch.delenv(model_endpoint.ARM_ENV)
    barrier = threading.Barrier(3)

    def handler(config, path):
        # A fresh worker does not inherit the parent's context: bind inside it.
        assert resolve().model_digest is None
        with model_endpoint.bind_model_routes(path, config["arm_id"]):
            barrier.wait(timeout=5)
            observed = {role: resolve(role) for role in ("compile", "runtime")}
            barrier.wait(timeout=5)
        assert resolve().model_digest is None
        return observed

    with model_endpoint.bind_model_routes(first_file, first["arm_id"]):
        with ThreadPoolExecutor(max_workers=2) as pool:
            first_result = pool.submit(handler, first, first_file)
            second_result = pool.submit(handler, second, second_file)
            barrier.wait(timeout=5)
            assert resolve().arm_id == first["arm_id"]
            barrier.wait(timeout=5)
            for config, future in ((first, first_result), (second, second_result)):
                observed = future.result(timeout=5)
                assert all(endpoint.arm_id == config["arm_id"] for endpoint in observed.values())
                assert {role: endpoint.base_url for role, endpoint in observed.items()} == {
                    role: route["base_url"] for role, route in config["routes"].items()}
    assert resolve().model_digest is None
