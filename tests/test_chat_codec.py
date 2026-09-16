"""Mechanical conversion compatibility, with no model or HTTP requests."""
import copy
import json
import subprocess
import sys

import pytest

from evaluation import chat_codec as codec


def test_codec_import_does_not_load_http_or_legacy_proxy():
    result = subprocess.run([sys.executable, "-c", """
import sys
import evaluation.chat_codec
assert 'evaluation.ollama_no_think_proxy' not in sys.modules
assert 'http.server' not in sys.modules
assert 'urllib.request' not in sys.modules
"""], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name", ["_arguments", "_native_messages", "_tool_calls", "_openai_response"])
def test_legacy_proxy_reexports_identical_converters(name):
    from evaluation import ollama_no_think_proxy as proxy
    assert getattr(proxy, name) is getattr(codec, name)


@pytest.mark.parametrize("value,expected", [
    ({"id": "one"}, {"id": "one"}), ('{"id":"one"}', {"id": "one"}),
    ("not JSON", {}), ("[]", {}), (None, {}), (1, {}),
])
def test_argument_fallback_semantics_are_not_changed(value, expected):
    assert codec._arguments(value) == expected


def test_native_message_conversion_retains_historical_roles_calls_and_tool_names():
    messages = [{"role": "developer", "content": "说明"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"function": {"name": "read", "arguments": '{"id":"one"}'}}]},
                {"role": "tool", "name": "read", "tool_call_id": "call_0", "content": "结果"}]
    original = copy.deepcopy(messages)
    assert codec._native_messages(messages) == [
        {"role": "system", "content": "说明"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_0", "function": {"index": 0, "name": "read", "arguments": {"id": "one"}}}]},
        {"role": "tool", "tool_name": "read", "content": "结果"}]
    assert messages == original  # Existing omission of tool_call_id is not revised here.


@pytest.mark.parametrize("messages,error", [
    ({}, "must be a list"), ([1], "message is invalid"),
    ([{"role": "unknown", "content": "x"}], "message is invalid"),
    ([{"role": "assistant", "tool_calls": [{}]}], "Tool Call is invalid"),
])
def test_native_message_validation_is_preserved(messages, error):
    with pytest.raises(ValueError, match=error):
        codec._native_messages(messages)


@pytest.mark.parametrize("stream", [False, True])
def test_response_conversion_preserves_stream_shape_usage_and_tool_identity(monkeypatch, stream):
    monkeypatch.setattr(codec.time, "time", lambda: 1234.9)
    native = {"model": "fixture", "message": {"content": "", "tool_calls": [
        {"id": "call-one", "function": {"name": "read", "arguments": {"z": 2, "a": "甲"}}}]},
        "prompt_eval_count": 10, "eval_count": 3}
    result = codec._openai_response(native, stream=stream)
    assert result["created"] == 1234
    assert result["object"] == ("chat.completion.chunk" if stream else "chat.completion")
    assert result["usage"] == {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
    choice = result["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["delta" if stream else "message"]["tool_calls"][0]
    assert call["id"] == "call-one" and call["index"] == 0
    assert call["function"]["arguments"] == '{"a":"甲","z":2}'


def test_missing_usage_is_legacy_zero_not_new_preflight_attestation():
    result = codec._openai_response({"message": {"content": "text"}}, stream=False)
    assert result["usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    assert result["choices"] == [{"index": 0, "message": {"role": "assistant", "content": "text"},
                                   "finish_reason": "stop"}]
    with pytest.raises(ValueError, match="no assistant message"):
        codec._openai_response({}, stream=False)


def test_tool_fallback_ids_remain_stable_and_invalid_calls_reject():
    tools = [{"function": {"name": "read", "arguments": {"x": 1}}}]
    first = codec._tool_calls(tools)
    assert first == codec._tool_calls(tools)
    assert first[0]["id"].startswith("call_")
    assert json.loads(first[0]["function"]["arguments"]) == {"x": 1}
    assert codec._tool_calls(None) == []
    with pytest.raises(ValueError, match="Tool Call is invalid"):
        codec._tool_calls([{}])
