"""Loopback OpenAI-to-Ollama bridge with thinking explicitly disabled.

Ollama's OpenAI-compatible endpoint currently ignores the Qwen ``think``
switch.  This evaluation-only bridge accepts the OpenAI chat-completions shape
used by DSH, invokes Ollama's native ``/api/chat`` endpoint with ``think=false``,
and returns a standards-shaped SSE response.  It never changes the model,
messages, Tools, or Tool arguments.
"""

from __future__ import annotations

import hashlib
import http.server
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


MAX_REQUEST_BYTES = 4 * 1024 * 1024
INVOCATION_PROFILE = "ollama_native_chat_think_false_num_ctx_32768"


def _loopback(value: str) -> str:
    parsed = urllib.parse.urlparse(value.rstrip("/"))
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Ollama no-think bridge requires a loopback HTTP upstream")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Ollama no-think bridge upstream URL is invalid")
    return value.rstrip("/")


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


class _BridgeServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], upstream: str) -> None:
        super().__init__(address, _BridgeHandler)
        self.upstream = _loopback(upstream)


class _BridgeHandler(http.server.BaseHTTPRequestHandler):
    server: _BridgeServer

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _json_error(self, status: int, message: str) -> None:
        body = json.dumps({"error": {"message": message, "type": "bridge_error"}}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - stdlib HTTP handler API
        if urllib.parse.urlparse(self.path).path != "/v1/chat/completions":
            self._json_error(404, "not found")
            return
        try:
            length = int(self.headers.get("Content-Length") or "0")
            if length < 1 or length > MAX_REQUEST_BYTES:
                raise ValueError("OpenAI request size is invalid")
            request = json.loads(self.rfile.read(length).decode("utf-8", "strict"))
            if not isinstance(request, dict) or not isinstance(request.get("model"), str):
                raise ValueError("OpenAI request is invalid")
            options: dict[str, Any] = {
                "num_ctx": 32768,
                "num_predict": max(1, min(int(request.get("max_tokens") or 1400), 4096)),
                "temperature": 0.0,
            }
            for key in ("temperature", "top_p"):
                if isinstance(request.get(key), (int, float)):
                    options[key] = request[key]
            native_request: dict[str, Any] = {
                "model": request["model"],
                "messages": _native_messages(request.get("messages")),
                "stream": False,
                "think": False,
                "options": options,
            }
            if isinstance(request.get("tools"), list):
                native_request["tools"] = request["tools"]
            upstream_request = urllib.request.Request(
                self.server.upstream + "/api/chat",
                data=json.dumps(native_request, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(upstream_request, timeout=300) as response:  # noqa: S310
                native = json.loads(response.read(MAX_REQUEST_BYTES + 1).decode("utf-8", "strict"))
            if not isinstance(native, dict) or native.get("error"):
                raise ValueError(str(native.get("error") or "invalid Ollama response"))
            streaming = bool(request.get("stream"))
            output = _openai_response(native, stream=streaming)
            if streaming:
                body = (
                    "data: " + json.dumps(output, ensure_ascii=False, separators=(",", ":"))
                    + "\n\ndata: [DONE]\n\n"
                ).encode("utf-8")
                content_type = "text/event-stream"
            else:
                body = json.dumps(output, ensure_ascii=False).encode("utf-8")
                content_type = "application/json"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (ValueError, TypeError, json.JSONDecodeError, urllib.error.URLError) as exc:
            self._json_error(502, str(exc)[:500])


class OllamaNoThinkProxy:
    """Own one ephemeral loopback bridge and expose its base URL."""

    def __init__(self, upstream_base_url: str) -> None:
        self.server = _BridgeServer(("127.0.0.1", 0), upstream_base_url)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self) -> "OllamaNoThinkProxy":
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


__all__ = ["INVOCATION_PROFILE", "OllamaNoThinkProxy"]
