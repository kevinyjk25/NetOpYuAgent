"""Mechanical codecs only; no runner, model, tokenizer, or network is started."""
import copy
import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest

from evaluation import bounded_runner as runner
from evaluation.bounded_pilot import seal
from evaluation.bounded_preflight import PREPARED_SCHEMA
from network_runtime.contracts import sha256_json


def prepared():
    wire = {"model": runner.MODEL, "think": False, "stream": False,
            "messages": [{"role": "user", "content": "fixture only"}],
            "tools": [{"type": "function", "function": {"name": "observe", "parameters": {
                "type": "object", "properties": {"count": {"type": "integer"}, "text": {"type": "string"}}}}}],
            "options": {"num_predict": 5, "temperature": 0}}
    return seal({"schema": PREPARED_SCHEMA, "model": runner.MODEL,
        "generation_enabled": False, "authority_granted": False, "live_adapter_ready": False,
        "actual_model_calls": 0, "equivalent_to_installed_ollama": "not_established",
        "evidence_role": "offline_experimental_preflight_not_generation_parity",
        "special_tokens": {"add_special": True, "parse_special": True},
        "request_api": "native", "normalization": "native_identity_v1", "wire": wire,
        "source_request": copy.deepcopy(wire), "source_request_digest": sha256_json(wire),
        "wire_digest": sha256_json(wire), "assets_digest": "fixture-only",
        "rendered_prompt": "fixture", "rendered_prompt_digest": "sha256:" + hashlib.sha256(b"fixture").hexdigest(),
        "token_ids": [1, 2, 3, 4, 5], "token_ids_digest": sha256_json([1, 2, 3, 4, 5]),
        "input_tokens": 5, "output_reservation": 5,
        "renderer_identity": {"profile": runner.PROFILE, "ollama_revision": runner.OLLAMA_REVISION,
            "renderer": "qwen3.5", "parser": "qwen3.5"},
        "effective_options": {"num_ctx": runner.CONTEXT, "num_predict": 5, "num_keep": 4,
            "temperature": 0, "top_k": 20, "top_p": 0.95, "min_p": 0, "repeat_penalty": 1,
            "repeat_last_n": 64, "frequency_penalty": 0, "presence_penalty": 1.5,
            "typical_p": 1, "seed": -1, "stop": None, **runner._PROCESS_OPTIONS}})


def reseal(value):
    value.pop("digest", None)
    return seal(value)


def reply(value):
    return runner.encode_fixture_completion(value, {"role": "assistant", "content": "fixture answer"})


def test_serializer_preserves_integer_sequence_without_string_prompt_or_fake_controls():
    value = prepared()
    body = runner.serialize_completion(value)
    assert body["prompt"] == value["token_ids"]
    assert all(type(item) is int for item in body["prompt"])
    assert body["n_predict"] == value["output_reservation"]
    assert body["stream"] is body["cache_prompt"] is False
    assert body["preserved_tokens"] == runner.PRESERVED_TOKENS
    assert not set(body) & {"model", "messages", "think", "num_ctx", "shift", "truncate", "stop"}
    body["prompt"].append(6)
    assert len(value["token_ids"]) == 5


@pytest.mark.parametrize("output_format", ["json", {"type": "object", "additionalProperties": False}])
def test_format_is_exact_decoder_constraint(output_format):
    value = prepared()
    value["wire"]["format"] = output_format
    value["source_request"] = copy.deepcopy(value["wire"])
    value["source_request_digest"] = value["wire_digest"] = sha256_json(value["wire"])
    body = runner.serialize_completion(reseal(value))
    if isinstance(output_format, dict):
        assert body["json_schema"] == output_format
        assert body["json_schema"] is not output_format
    else:
        assert body["grammar"].startswith("\nroot   ::= object\n")
        assert '"u" [0-9a-fA-F]' in body["grammar"]


@pytest.mark.parametrize("tokens", [[True], ["1"], [-1], [runner.VOCAB_SIZE], []])
def test_non_native_token_sequence_rejected(tokens):
    value = prepared()
    value["token_ids"], value["input_tokens"] = tokens, len(tokens)
    value["token_ids_digest"] = sha256_json(tokens)
    with pytest.raises(ValueError, match="token array"):
        runner.serialize_completion(reseal(value))


@pytest.mark.parametrize("key,invalid", [("num_ctx", 10), ("num_predict", 6), ("num_gpu", 0),
    ("seed", True), ("temperature", -1), ("top_p", 1.1), ("unknown", 4), ("stop", "x")])
def test_option_drift_is_rejected(key, invalid):
    value = prepared()
    value["effective_options"][key] = invalid
    with pytest.raises(ValueError):
        runner.serialize_completion(reseal(value))


def test_drift_and_context_boundary_rejected():
    value = prepared()
    value["wire"]["messages"][0]["content"] = "changed"
    with pytest.raises(ValueError):
        runner.serialize_completion(reseal(value))
    value = prepared()
    value["token_ids"] = [1] * (runner.CONTEXT - value["output_reservation"])
    value["input_tokens"] = len(value["token_ids"])
    value["token_ids_digest"] = sha256_json(value["token_ids"])
    with pytest.raises(ValueError, match="context"):
        runner.serialize_completion(reseal(value))


@pytest.fixture
def codec(tmp_path, monkeypatch):
    executable = tmp_path / "not-a-real-executable"
    executable.write_text("fixture only")
    calls = []
    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        wire = json.loads(kwargs["input"])
        result = {"message": {"role": "assistant", "content": wire["content"]},
                  "preserved_tokens": runner.PRESERVED_TOKENS, "no_generation": True,
                  "identity": {"ollama_revision": runner.OLLAMA_REVISION, "parser": "qwen3.5",
                    "api": "github.com/ollama/ollama/model/parsers.ParserForName"}}
        return SimpleNamespace(returncode=0, stdout=json.dumps(result).encode())
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    return executable, calls


def test_response_accounting_uses_cache_plus_prompt_and_bounded_official_parser(codec):
    value = prepared()
    response = reply(value)
    response["timings"].update(cache_n=2, prompt_n=3)
    native = runner.parse_completion(value, response, codec[0], timeout=0.2)
    assert native["prompt_eval_count"] == 5 and native["eval_count"] == 1
    assert native["done"] is True and native["done_reason"] == "stop"
    assert codec[1][0][1]["timeout"] == 0.2
    assert codec[1][0][1]["env"] == {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}


@pytest.mark.parametrize("field,invalid", [("stop", False), ("truncated", True), ("truncated", None),
    ("stop_type", "none"), ("tokens_predicted", 2), ("tokens_evaluated", 4), ("content", None),
    ("timings", {}), ("error", None)])
def test_unknown_response_never_reaches_parser(codec, field, invalid):
    value = prepared()
    response = reply(value)
    response[field] = invalid
    with pytest.raises(ValueError):
        runner.parse_completion(value, response, codec[0])
    assert codec[1] == []


@pytest.mark.parametrize("field,invalid", [("cache_n", -1), ("prompt_n", True),
    ("prompt_n", 6), ("predicted_n", 6), ("predicted_n", 1.0)])
def test_missing_invalid_or_over_budget_usage_rejected(codec, field, invalid):
    value = prepared()
    response = reply(value)
    response["timings"][field] = invalid
    with pytest.raises(ValueError):
        runner.parse_completion(value, response, codec[0])
    assert codec[1] == []


def test_timeout_and_helper_failure_are_unusable(codec, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])
    monkeypatch.setattr(runner.subprocess, "run", timeout)
    with pytest.raises(TimeoutError):
        runner.parse_completion(prepared(), reply(prepared()), codec[0], timeout=0.01)
    monkeypatch.setattr(runner.subprocess, "run", lambda *_, **__: SimpleNamespace(returncode=2, stdout=b""))
    with pytest.raises(ValueError, match="rejected"):
        runner.parse_completion(prepared(), reply(prepared()), codec[0])


def test_fixture_encoder_is_explicitly_synthetic_and_declared_tools_only():
    result = runner.encode_fixture_completion(prepared(), {"tool_calls": [{"function": {
        "name": "observe", "arguments": {"count": 2, "text": "中文 & x"}}}]})
    assert result["_fixture_only"] is True
    assert result["tokens_predicted"] == 1
    assert "<function=observe>" in result["content"] and "<parameter=count>2</parameter>" in result["content"]
    with pytest.raises(ValueError, match="declared"):
        runner.encode_fixture_completion(prepared(), {"tool_calls": [{"function": {"name": "other", "arguments": {}}}]})


def test_props_check_is_necessary_but_grants_no_launch_or_generation():
    props = {"total_slots": 1, "model_path": "/pinned/model", "default_generation_settings": {"n_ctx": runner.CONTEXT}}
    assert runner.validate_runner_props(props, "/pinned/model") == {
        "props_match": True, "generation_enabled": False, "launch_flags_attested": False}
    props["total_slots"] = 2
    with pytest.raises(ValueError):
        runner.validate_runner_props(props, "/pinned/model")
    assert runner.RUNNER_STARTUP_CONTRACT["generation_enabled"] is False
