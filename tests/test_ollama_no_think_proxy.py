from __future__ import annotations

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from evaluation.ollama_no_think_proxy import OllamaNoThinkProxy


def test_no_think_proxy_translates_openai_tools_to_native_ollama() -> None:
    observed = {}

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, _format, *_args):
            return

        def do_POST(self):  # noqa: N802
            length = int(self.headers["Content-Length"])
            observed.update(json.loads(self.rfile.read(length)))
            body = json.dumps({
                "model": "qwen3.5:9b",
                "message": {
                    "role": "assistant", "content": "",
                    "tool_calls": [{
                        "id": "call-1",
                        "function": {"name": "read_record", "arguments": {"id": "one"}},
                    }],
                },
                "prompt_eval_count": 10, "eval_count": 3,
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    try:
        with OllamaNoThinkProxy(f"http://127.0.0.1:{upstream.server_address[1]}") as proxy:
            payload = {
                "model": "qwen3.5:9b", "stream": True, "max_tokens": 120,
                "messages": [{"role": "user", "content": "read one"}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "read_record", "description": "read",
                        "parameters": {"type": "object"},
                    },
                }],
            }
            request = urllib.request.Request(
                proxy.base_url + "/v1/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310
                result = response.read().decode()
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)
    assert observed["think"] is False
    assert observed["stream"] is False
    assert observed["options"] == {
        "num_ctx": 32768, "num_predict": 120, "temperature": 0.0,
    }
    assert observed["tools"] == payload["tools"]
    assert '"finish_reason":"tool_calls"' in result
    event = json.loads(result.splitlines()[0].removeprefix("data: "))
    arguments = event["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(arguments) == {"id": "one"}
    assert result.endswith("data: [DONE]\n\n")
