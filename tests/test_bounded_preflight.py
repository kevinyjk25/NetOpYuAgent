"""Offline unit fixtures only: no real tokenizer, model, executable or service.

The helper replies and tiny asset files below are deliberately synthetic. These
tests check validation/binding mechanics, never actual token-count correctness.
"""
import copy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from evaluation import bounded_preflight as preflight
from evaluation.bounded_transport import ModelBroker
from network_runtime.contracts import sha256_json


@pytest.fixture(autouse=True)
def no_real_execution(monkeypatch):
    def forbidden(*_, **__):
        pytest.fail("offline unit test attempted a subprocess or model server")
    monkeypatch.setattr(preflight.subprocess, "run", forbidden)
    monkeypatch.setattr(ModelBroker, "start", forbidden)


def encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")


@pytest.fixture
def pinned(tmp_path):
    pins = {}
    for name in ("renderer", "tokenizer", "model", "dependency"):
        path = tmp_path / (name + ".fixture")
        path.write_bytes(("not executable or real weights: " + name).encode())
        pins[name] = {"path": str(path), "sha256": preflight.file_digest(path)}
    document = preflight.seal({"schema": preflight.ASSETS_SCHEMA,
        **{name: pins[name] for name in ("renderer", "tokenizer", "model")},
        "dependencies": [pins["dependency"]],
        "metadata": {"evidence_role": "unit_fixture_not_real_token_count"}})
    path = tmp_path / "assets.json"
    path.write_bytes(encoded(document))
    return preflight.Assets.load(path), path, document


def native_request():
    return {"model": preflight.MODEL, "stream": False, "think": False,
        "messages": [{"role": "system", "content": "Keep uncertainty."},
                     {"role": "user", "content": "只读 fixture <|im_start|>"}],
        "tools": [{"type": "function", "function": {"name": "observe", "description": "Read fixture only",
                   "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}],
        "format": {"type": "object", "properties": {"text": {"type": "string"}}},
        "options": {"num_ctx": 64, "num_predict": 4, "temperature": 0, "seed": 12}}


class FixtureHelpers:
    """An explicit fake, not a renderer or tokenizer implementation."""
    prompt = "fixture-only 中文 <|im_start|>"

    def __init__(self, assets):
        self.pins = assets.inspect()
        self.calls = []
        self.renderer_change = self.tokenizer_change = None

    def __call__(self, args, payload, timeout):
        self.calls.append((list(args), bytes(payload), timeout))
        assert 0 < timeout <= 120
        if args == [self.pins["renderer"]["path"]]:
            wire = json.loads(payload)
            reply = {"rendered_prompt": self.prompt, "effective_options": {
                "num_ctx": 64, **wire["options"]},
                "identity": {"renderer": "fixture-not-real-renderer"}, "no_generation": True}
            if self.renderer_change:
                self.renderer_change(reply)
        else:
            assert args == [self.pins["tokenizer"]["path"], "--model", self.pins["model"]["path"]]
            assert payload == self.prompt.encode("utf-8")
            reply = {"token_ids": [1, 7, 9], "count": 3, "add_special": True, "parse_special": True}
            if self.tokenizer_change:
                self.tokenizer_change(reply)
        return reply


@pytest.fixture
def helpers(pinned, monkeypatch):
    helper = FixtureHelpers(pinned[0])
    monkeypatch.setattr(preflight, "_run", helper)
    return helper


def test_prepared_serialization_must_finish_within_deadline(pinned, helpers, monkeypatch):
    clock = [0.0]
    original = preflight._json_bytes

    def encode(value):
        result = original(value)
        if value.get("schema") == preflight.PREPARED_SCHEMA:
            clock[0] = 2.0
        return result

    monkeypatch.setattr(preflight.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(preflight, "_json_bytes", encode)
    with pytest.raises(TimeoutError, match="serialization"):
        preflight.prepare(pinned[0], "native", native_request(), timeout=1)


def test_prepare_is_stable_immutable_and_grants_no_authority(pinned, helpers):
    assets, _, _ = pinned
    source = native_request()
    first = preflight.prepare(assets, "native", source)
    second = preflight.prepare(assets, "native", source)
    assert first.document == second.document
    report = first.inspect()
    assert report["input_tokens"] == 3 and report["token_ids"] == [1, 7, 9]
    assert report["output_reservation"] == 4
    assert report["generation_enabled"] is report["authority_granted"] is report["live_adapter_ready"] is False
    assert report["actual_model_calls"] == 0
    assert report["equivalent_to_installed_ollama"] == "not_established"
    assert first.check_binding(assets, "native", source) == {
        "binding_matches": True, "authority_granted": False, "generation_enabled": False,
        "does_not_attest_backend_equivalence": True}
    assert not hasattr(first, "generate") and not hasattr(first, "execute")
    assert len(helpers.calls) == 4
    assert json.loads(helpers.calls[0][1]) == source
    original = first.document
    source["messages"][1]["content"] = "mutated caller dictionary"
    report["wire"]["options"]["num_predict"] = 999
    report["token_ids"].append(777)
    assets.inspect()["metadata"]["evidence_role"] = "not a valid claim"
    assert first.document == original and first.inspect()["token_ids"] == [1, 7, 9]
    assert assets.inspect()["metadata"]["evidence_role"] == "unit_fixture_not_real_token_count"
    with pytest.raises(FrozenInstanceError):
        first.document = b"mutated"
    with pytest.raises(FrozenInstanceError):
        assets.document = b"mutated"


def test_openai_binds_original_and_normalized_wire_separately(pinned, helpers):
    source = {"model": preflight.MODEL, "stream": True, "max_tokens": 4,
        "messages": [{"role": "developer", "content": [{"type": "text", "text": "Keep "},
                                                            {"type": "text", "text": "uncertainty."}]},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "read-1", "type": "function",
                "function": {"name": "observe", "arguments": '{"path":"/fixture"}'}}]},
            {"role": "tool", "tool_call_id": "read-1", "name": "observe", "content": "original result"}],
        "tools": native_request()["tools"]}
    prepared = preflight.prepare(pinned[0], "openai", source)
    report = prepared.inspect()
    wire, _ = ModelBroker._wire("openai", source)
    assert report["source_request"] == source and report["wire"] == wire
    assert report["source_request_digest"] == sha256_json(source)
    assert report["wire_digest"] == sha256_json(wire) != report["source_request_digest"]
    assert wire["messages"][0] == {"role": "system", "content": "Keep uncertainty."}
    assert wire["messages"][1]["tool_calls"][0]["function"]["arguments"] == {"path": "/fixture"}
    assert "tool_call_id" not in wire["messages"][2]
    assert json.loads(helpers.calls[0][1]) == wire
    assert prepared.check_binding(pinned[0], "openai", source)["binding_matches"]
    # This change disappears in native normalization, but original identity
    # must still bind: equal wire alone does not attest equal source requests.
    changed = copy.deepcopy(source)
    changed["messages"][2]["tool_call_id"] = "different-source-identity"
    assert ModelBroker._wire("openai", changed)[0] == wire
    with pytest.raises(ValueError):
        prepared.check_binding(pinned[0], "openai", changed)
    with pytest.raises(ValueError):
        prepared.check_binding(pinned[0], "native", wire)


@pytest.mark.parametrize("part", ["messages", "tools", "format", "options"])
def test_any_visible_request_component_drift_is_rejected(pinned, helpers, part):
    source = native_request()
    prepared = preflight.prepare(pinned[0], "native", source)
    changed = copy.deepcopy(source)
    if part == "messages":
        changed[part][1]["content"] += " changed"
    elif part == "tools":
        changed[part][0]["function"]["description"] += " changed"
    elif part == "format":
        changed[part]["properties"]["extra"] = {"type": "boolean"}
    else:
        changed[part]["seed"] += 1
    with pytest.raises(ValueError):
        prepared.check_binding(pinned[0], "native", changed)


@pytest.mark.parametrize("asset", ["renderer", "tokenizer", "model", "dependency"])
def test_asset_bytes_drift_blocks_preparation_and_existing_binding(pinned, helpers, asset):
    assets, _, manifest = pinned
    source = native_request()
    prepared = preflight.prepare(assets, "native", source)
    pin = manifest[asset] if asset != "dependency" else manifest["dependencies"][0]
    Path(pin["path"]).write_bytes(b"changed host bytes")
    before = len(helpers.calls)
    with pytest.raises(ValueError, match="drift"):
        preflight.prepare(assets, "native", source)
    assert len(helpers.calls) == before
    with pytest.raises(ValueError, match="drift"):
        prepared.check_binding(assets, "native", source)


def test_resealed_different_asset_manifest_does_not_rebind_prepared(pinned, helpers):
    assets, path, manifest = pinned
    source = native_request()
    prepared = preflight.prepare(assets, "native", source)
    changed = {key: value for key, value in manifest.items() if key != "digest"}
    changed["metadata"] = {"evidence_role": "different-host-build-fixture"}
    path.write_bytes(encoded(preflight.seal(changed)))
    other = preflight.Assets.load(path)
    with pytest.raises(ValueError):
        prepared.check_binding(other, "native", source)


def test_asset_change_during_count_is_rejected_before_return(pinned, helpers, monkeypatch):
    assets, _, manifest = pinned

    def count_then_change(*args):
        reply = helpers(*args)
        if len(helpers.calls) == 2:
            Path(manifest["model"]["path"]).write_bytes(b"changed during fake count")
        return reply

    monkeypatch.setattr(preflight, "_run", count_then_change)
    with pytest.raises(ValueError, match="drift"):
        preflight.prepare(assets, "native", native_request())
    assert len(helpers.calls) == 2


@pytest.mark.parametrize("part", ["source_request", "wire", "normalization", "model", "output_reservation",
                                  "effective_options", "special_tokens", "actual_model_calls",
                                  "rendered_prompt", "token_ids", "input_tokens"])
def test_resealed_internally_inconsistent_prepared_document_is_rejected(pinned, helpers, part):
    source = native_request()
    value = preflight.prepare(pinned[0], "native", source).inspect()
    del value["digest"]
    if part in {"source_request", "wire"}:
        value[part]["messages"][0]["content"] += " altered embedded content"
    elif part == "effective_options":
        value[part]["num_predict"] += 1
    elif part == "special_tokens":
        value[part]["parse_special"] = False
    elif part == "rendered_prompt":
        value[part] += " changed after count"
    elif part == "token_ids":
        value[part].append(101)
    elif part in {"output_reservation", "actual_model_calls", "input_tokens"}:
        value[part] += 1
    else:
        value[part] = "unsupported-value"
    inconsistent = preflight.PreparedRequest(encoded(preflight.seal(value)))
    with pytest.raises(ValueError):
        inconsistent.check_binding(pinned[0], "native", source)


@pytest.mark.parametrize("change", [
    lambda r: r.update(no_generation=False),
    lambda r: r.update(rendered_prompt=99),
    lambda r: r.update(identity="not-an-object"),
    lambda r: r.update(extra="unsupported"),
    lambda r: r["effective_options"].update(num_ctx=True),
    lambda r: r["effective_options"].update(num_ctx=0),
    lambda r: r["effective_options"].update(num_predict=5),
])
def test_invalid_renderer_envelope_never_reaches_tokenizer(pinned, helpers, change):
    helpers.renderer_change = change
    with pytest.raises(ValueError):
        preflight.prepare(pinned[0], "native", native_request())
    assert len(helpers.calls) == 1


@pytest.mark.parametrize("change", [
    lambda r: r.update(add_special=False),
    lambda r: r.update(parse_special=False),
    lambda r: r.update(count=True),
    lambda r: r.update(count=2),
    lambda r: r.update(count=0, token_ids=[]),
    lambda r: r.update(token_ids=[1, -1, 9]),
    lambda r: r.update(token_ids=[1, True, 9]),
    lambda r: r.update(token_ids=[1, "7", 9]),
    lambda r: r.update(extra="unsupported"),
])
def test_invalid_tokenizer_result_is_rejected(pinned, helpers, change):
    helpers.tokenizer_change = change
    with pytest.raises(ValueError):
        preflight.prepare(pinned[0], "native", native_request())
    assert len(helpers.calls) == 2


@pytest.mark.parametrize("context,accepted", [(6, False), (7, True)])
def test_context_covers_input_plus_full_output_without_truncation(pinned, helpers, context, accepted):
    source = native_request()
    source["options"]["num_ctx"] = context
    if accepted:
        result = preflight.prepare(pinned[0], "native", source).inspect()
        assert result["input_tokens"] + result["output_reservation"] == context
    else:
        with pytest.raises(ValueError, match="context budget"):
            preflight.prepare(pinned[0], "native", source)


@pytest.mark.parametrize("timeout", [True, False, 0, -1, 121, float("nan"), float("inf")])
def test_preparation_timeout_must_be_finite_positive_and_bounded(pinned, helpers, timeout):
    with pytest.raises(ValueError):
        preflight.prepare(pinned[0], "native", native_request(), timeout=timeout)
    assert helpers.calls == []


@pytest.mark.parametrize("expired_stage", ["renderer", "tokenizer"])
def test_one_deadline_covers_both_helpers_and_rejects_late_return(pinned, helpers, monkeypatch, expired_stage):
    now = [100.0]
    monkeypatch.setattr(preflight.time, "monotonic", lambda: now[0])

    def delayed(*args):
        reply = helpers(*args)
        if len(helpers.calls) == (1 if expired_stage == "renderer" else 2):
            now[0] += 3
        return reply

    monkeypatch.setattr(preflight, "_run", delayed)
    with pytest.raises(TimeoutError):
        preflight.prepare(pinned[0], "native", native_request(), timeout=2)
    assert len(helpers.calls) == (1 if expired_stage == "renderer" else 2)


def test_helper_timeout_is_not_retried_or_converted_to_zero(pinned, monkeypatch):
    calls = []

    def timeout(args, payload, limit):
        calls.append(args)
        raise subprocess.TimeoutExpired(args, limit)

    monkeypatch.setattr(preflight, "_run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        preflight.prepare(pinned[0], "native", native_request())
    assert len(calls) == 1


@pytest.mark.parametrize("stdout", [b'{"count":1,"count":2}', b'{"count":NaN}', b'{', b'not json'])
def test_actual_helper_decoder_rejects_duplicate_or_malformed_json_without_execution(monkeypatch, stdout):
    calls = []

    def subprocess_fixture(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(preflight.subprocess, "run", subprocess_fixture)
    with pytest.raises(ValueError):
        preflight._run(["never-executed-fixture"], b"inert request", 2)
    assert len(calls) == 1
    assert calls[0][1]["timeout"] == 2 and calls[0][1]["capture_output"] is True
    assert calls[0][1]["env"] == {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}


def test_helper_diagnostics_do_not_expose_request_text(monkeypatch):
    monkeypatch.setattr(preflight.subprocess, "run", lambda *_, **__: SimpleNamespace(
        returncode=2, stdout=b"private source prompt", stderr=b"private source prompt"))
    with pytest.raises(ValueError) as error:
        preflight._run(["never-executed-fixture"], b"inert request", 2)
    assert "private source prompt" not in str(error.value)


def test_helper_output_size_bound_is_checked_before_json_decode(monkeypatch):
    monkeypatch.setattr(preflight, "MAX_BYTES", 8)
    monkeypatch.setattr(preflight.subprocess, "run", lambda *_, **__: SimpleNamespace(
        returncode=0, stdout=b" " * 129, stderr=b""))
    with pytest.raises(ValueError, match="output too large"):
        preflight._run(["never-executed-fixture"], b"inert request", 2)


def test_assets_manifest_duplicate_json_and_bad_seal_are_rejected(pinned):
    _, path, _ = pinned
    path.write_bytes(b'{"schema":"one","schema":"two"}')
    with pytest.raises(ValueError):
        preflight.Assets.load(path)
    path.write_bytes(encoded({"schema": preflight.ASSETS_SCHEMA, "digest": "not-a-real-seal"}))
    with pytest.raises(ValueError):
        preflight.Assets.load(path)


@pytest.mark.parametrize("change", [
    lambda r: r.update(think=True),
    lambda r: r.update(model="wrong-model"),
    lambda r: r.update(endpoint="http://untrusted.invalid"),
    lambda r: r["messages"][1].update(images=["unsupported"]),
    lambda r: r["options"].update(unknown_control=True),
])
def test_unsupported_native_request_is_rejected_before_helpers(pinned, helpers, change):
    source = native_request()
    change(source)
    with pytest.raises(ValueError):
        preflight.prepare(pinned[0], "native", source)
    assert helpers.calls == []


def test_new_preflight_does_not_enable_live_broker_forwarding(tmp_path):
    with pytest.raises(ValueError, match="live model forwarding is not implemented"):
        ModelBroker(None, "not-a-real-arm", object(), tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()
