"""Loopback OpenAI-to-Ollama bridge with thinking explicitly disabled.

Ollama's OpenAI-compatible endpoint currently ignores the Qwen ``think``
switch.  This evaluation-only bridge accepts the OpenAI chat-completions shape
used by DSH, invokes Ollama's native ``/api/chat`` endpoint with ``think=false``,
and returns a standards-shaped SSE response.  It never changes the model,
messages, Tools, or Tool arguments.
"""

from __future__ import annotations

import http.server
import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .chat_codec import (
    _arguments as _arguments,
    _native_messages as _native_messages,
    _openai_response as _openai_response,
    _tool_calls as _tool_calls,
    hashlib as hashlib,
    time as time,
)

MAX_REQUEST_BYTES = 4 * 1024 * 1024
INVOCATION_PROFILE = "ollama_native_chat_think_false_num_ctx_32768"


def _loopback(value: str) -> str:
    parsed = urllib.parse.urlparse(value.rstrip("/"))
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Ollama no-think bridge requires a loopback HTTP upstream")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Ollama no-think bridge upstream URL is invalid")
    return value.rstrip("/")


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
