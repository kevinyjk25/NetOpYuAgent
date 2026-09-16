"""Chat-shape conversion without HTTP, benchmark or model transport imports.

Historical conversion semantics are preserved. Callers own validation and
accounting; this codec grants neither authority nor trustworthy token counts.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any


def _arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _native_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        raise ValueError("OpenAI messages must be a list")
    converted: list[dict[str, Any]] = []
    for item in messages:
        if not isinstance(item, dict) or item.get("role") not in {
            "system", "developer", "user", "assistant", "tool",
        }:
            raise ValueError("OpenAI message is invalid")
        role = "system" if item["role"] == "developer" else item["role"]
        message: dict[str, Any] = {"role": role, "content": item.get("content") or ""}
        if role == "assistant" and isinstance(item.get("tool_calls"), list):
            calls = []
            for index, call in enumerate(item["tool_calls"]):
                function = call.get("function") if isinstance(call, dict) else None
                if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                    raise ValueError("OpenAI assistant Tool Call is invalid")
                calls.append({
                    "id": str(call.get("id") or f"call_{index}"),
                    "function": {
                        "index": index,
                        "name": function["name"],
                        "arguments": _arguments(function.get("arguments")),
                    },
                })
            message["tool_calls"] = calls
        if role == "tool" and isinstance(item.get("name"), str):
            message["tool_name"] = item["name"]
        converted.append(message)
    return converted


def _tool_calls(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    calls: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        function = item.get("function") if isinstance(item, dict) else None
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise ValueError("Ollama native Tool Call is invalid")
        arguments = _arguments(function.get("arguments"))
        encoded = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        fallback = hashlib.sha256(
            f"{index}:{function['name']}:{encoded}".encode("utf-8")
        ).hexdigest()[:24]
        calls.append({
            "index": index,
            "id": str(item.get("id") or f"call_{fallback}"),
            "type": "function",
            "function": {"name": function["name"], "arguments": encoded},
        })
    return calls


def _openai_response(native: dict[str, Any], *, stream: bool) -> dict[str, Any]:
    message = native.get("message")
    if not isinstance(message, dict):
        raise ValueError("Ollama native response has no assistant message")
    calls = _tool_calls(message.get("tool_calls"))
    content = str(message.get("content") or "")
    response_id = "chatcmpl-" + hashlib.sha256(
        json.dumps(message, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:24]
    usage = {
        "prompt_tokens": int(native.get("prompt_eval_count") or 0),
        "completion_tokens": int(native.get("eval_count") or 0),
        "total_tokens": int(native.get("prompt_eval_count") or 0)
        + int(native.get("eval_count") or 0),
    }
    body: dict[str, Any] = {
        "id": response_id,
        "object": "chat.completion.chunk" if stream else "chat.completion",
        "created": int(time.time()),
        "model": str(native.get("model") or ""),
        "usage": usage,
    }
    if stream:
        delta: dict[str, Any] = {"role": "assistant", "content": content}
        if calls:
            delta["tool_calls"] = calls
        body["choices"] = [{
            "index": 0, "delta": delta,
            "finish_reason": "tool_calls" if calls else "stop",
        }]
    else:
        output_message: dict[str, Any] = {"role": "assistant", "content": content}
        if calls:
            output_message["tool_calls"] = calls
        body["choices"] = [{
            "index": 0, "message": output_message,
            "finish_reason": "tool_calls" if calls else "stop",
        }]
    return body
