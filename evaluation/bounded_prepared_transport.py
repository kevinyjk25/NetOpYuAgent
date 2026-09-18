"""R0 exact-token transport acceptance, with a non-inferencing HTTP receiver.

Uses the production-shaped runner serializer/parser, the real pinned vocabulary
and the persistent arm ledger. The receiver's answers are host-authored fixtures,
not model output. There is intentionally no URL argument or live-run switch.
This establishes a mechanical send contract, not generation parity or benefit.
"""
from __future__ import annotations

import copy
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import socket
import threading
import time

from evaluation.bounded_pilot import seal, write_new
from evaluation.bounded_preflight import Assets, file_digest, prepare
from evaluation.bounded_transport import MAX_BYTES, ModelBroker, strict_json
from network_runtime.contracts import sha256_json


def _encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":")).encode("utf-8")


class NoInferenceRunner:
    """One-use host claims; never receives labels or executes source scripts.

    The host creates this receiver, so callers cannot substitute Ollama or an
    arbitrary network destination. Incoming bodies must match the exact sealed
    token-array payload. A duplicate/unknown claim has no response authority.
    """
    def __init__(self, responder):
        self.responder = responder
        self.pending = {}
        self.calls = []
        self.lock = threading.Lock()
        self.closed = False
        self.connections = set()
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):  # noqa: N802
                with receiver.lock:
                    if receiver.closed:
                        self.close_connection = True
                        return
                    receiver.connections.add(self.connection)
                timer = threading.Timer(10, self.expire)
                timer.daemon = True
                timer.start()
                try:
                    if self.path != "/completion":
                        raise PermissionError("only native completion route is supported")
                    lengths = self.headers.get_all("Content-Length", [])
                    if len(lengths) != 1 or self.headers.get("Transfer-Encoding") is not None:
                        raise ValueError("one bounded request body required")
                    count = int(lengths[0])
                    if not 0 < count <= MAX_BYTES:
                        raise ValueError("runner body too large")
                    raw = self.rfile.read(count)
                    if len(raw) != count:
                        raise ValueError("incomplete runner body")
                    payload = strict_json(raw)
                    result = receiver.consume(self.headers.get("X-R0-Claim", ""), payload)
                    body = _encode(result)
                    with receiver.lock:
                        if receiver.closed:
                            raise PermissionError("receiver closed before delivery")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    self.wfile.flush()
                except (ValueError, PermissionError, KeyError, TypeError, OSError):
                    self.close_connection = True
                finally:
                    timer.cancel()
                    with receiver.lock:
                        receiver.connections.discard(self.connection)

            def expire(self):
                try:
                    self.connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def claim(self, stage, prepared, payload):
        path = secrets.token_hex(24)
        with self.lock:
            if self.closed:
                raise PermissionError("closed non-inferencing receiver")
            self.pending[path] = (stage, copy.deepcopy(prepared), _encode(payload))
        return path

    def consume(self, path, payload):
        from evaluation.bounded_runner import encode_fixture_completion
        with self.lock:
            if self.closed or path not in self.pending:
                raise PermissionError("unknown/replayed receiver claim")
            stage, prepared, expected = self.pending.pop(path)
            if _encode(payload) != expected:
                raise ValueError("serialized runner request drift")
            self.calls.append({"stage": stage, "prepared_digest": prepared["digest"],
                               "payload_digest": sha256_json(payload), "actualModelCalls": 0})
        message = self.responder(stage, copy.deepcopy(prepared["wire"]))
        with self.lock:
            if self.closed:
                raise PermissionError("receiver closed while fixture callback running")
        return encode_fixture_completion(prepared, message)

    def close(self):
        with self.lock:
            self.closed = True
            self.pending.clear()
            connections = tuple(self.connections)
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)
        with self.lock:
            return {"closed": True, "drained": not self.connections}


class PreparedBackend:
    """Host-pinned assets plus a fixture receiver; no inference capability."""
    def __init__(self, assets: Assets, codec_pin, receiver: NoInferenceRunner):
        if type(assets) is not Assets or type(receiver) is not NoInferenceRunner:
            raise ValueError("pinned assets and exact non-inferencing receiver required")
        if not isinstance(codec_pin, dict) or set(codec_pin) != {"path", "sha256"}:
            raise ValueError("parser executable pin required")
        self.assets, self.receiver = assets, receiver
        self.codec = _encode(codec_pin)
        self.verify_codec()
        self.digest = sha256_json({"kind": "exact_token_non_inferencing_receiver",
            "assets": assets.inspect()["digest"], "parser": codec_pin})

    def verify_codec(self):
        pin = strict_json(self.codec)
        if file_digest(pin["path"]) != pin["sha256"]:
            raise ValueError("pinned official parser drift")
        return pin


class PreparedBroker(ModelBroker):
    """Same role, reservation, deadline and delivery lifecycle as ModelBroker.

    Real prompt token counts; fixture output counts. Assets, source, wire,
    token IDs, output limit, stage, arm and reservation are bound before send.
    The pending preparation is consumed once. Uncertain send or parsing halts
    the original ledger with conservative charges, never a fresh fallback.
    """
    def _validate_backend(self, backend):
        if type(backend) is not PreparedBackend:
            raise ValueError("only zero-inference prepared backend is implemented")

    def _measurement(self):
        return {"actualModelCalls": 0, "measurementKind": "exact_token_no_inference_transport",
                "inputAccounting": "pinned_vocab_tokenization",
                "outputAccounting": "fixture_only_not_model_usage",
                "generationParity": "not_tested", "liveGenerationEnabled": False}

    def close(self, timeout=0.5):
        began = time.monotonic()
        result = super().close(timeout=timeout)
        # One eventual receiver cleanup; include it in arm drainage. Never
        # block past the caller's remaining close allowance on a callback.
        with self.shutdown_lock:
            if not hasattr(self, "receiver_shutdown"):
                self.receiver_shutdown = threading.Event()
                self.receiver_shutdown_result = {"closed": False, "drained": False}
                def cleanup():
                    try:
                        self.receiver_shutdown_result = self.backend.receiver.close()
                    finally:
                        self.receiver_shutdown.set()
                threading.Thread(target=cleanup, daemon=True).start()
        self.receiver_shutdown.wait(max(0, timeout - (time.monotonic() - began)))
        receiver_result = dict(self.receiver_shutdown_result)
        return {**result, "receiver_close": receiver_result,
                "drained": bool(result.get("drained") and self.receiver_shutdown.is_set()
                                and receiver_result.get("drained"))}

    def _preflight(self, api, source, wire):
        from evaluation.bounded_runner import serialize_completion
        remaining = self.ledger.check_arm(self.arm_id)["remaining_seconds"]
        prepared = prepare(self.backend.assets, api, source, timeout=min(120, remaining))
        value = prepared.inspect()
        if value["wire"] != wire:
            raise ValueError("prepared normalized request mismatch")
        # Fail unsupported runner controls before reserving or sending anything.
        payload = serialize_completion(value)
        self.pending_prepared = (prepared, _encode(payload))
        return {"request_digest": sha256_json(wire), "input_tokens": value["input_tokens"],
                "prepared_digest": value["digest"], "runner_payload_digest": sha256_json(payload),
                "counting_method": "pinned_renderer_and_vocab_only_tokenizer",
                "output_counting_method": "fixture_not_model_output"}

    def _invoke_backend(self, stage, wire, reservation, directory):
        from evaluation.bounded_runner import parse_completion, serialize_completion
        pending = getattr(self, "pending_prepared", None)
        self.pending_prepared = None
        if pending is None:
            raise PermissionError("missing/already consumed prepared request")
        prepared, serialized = pending
        value = prepared.inspect()
        prepared.check_binding(self.backend.assets, value["request_api"], value["source_request"])
        codec = self.backend.verify_codec()
        payload = serialize_completion(value)
        if (wire != value["wire"] or serialized != _encode(payload)
                or reservation["stage"] != stage
                or reservation["arm_id"] != self.arm_id
                or reservation["reserved_input"] != value["input_tokens"]
                or reservation["reserved_output"] != value["output_reservation"]):
            raise ValueError("reservation/prepared/send binding drift")
        binding = seal({"arm_id": self.arm_id, "stage": stage,
            "request_id": reservation["request_id"], "call_id": reservation["call_id"],
            "prepared_digest": value["digest"], "runner_payload_digest": sha256_json(payload),
            "parser": codec, "deadline": reservation["deadline"], **self._measurement()})
        write_new(directory / "prepared.json", value)
        write_new(directory / "runner-request.json", payload)
        write_new(directory / "dispatch-binding.json", binding)
        # Journaling/hash work must not extend dispatch authority.
        guard = self.ledger.guard_reserved_call(self.arm_id, reservation["request_id"])
        if self.closed or self.clock() >= guard["deadline"]:
            raise PermissionError("dispatch revoked before receiver claim")
        receiver = self.backend.receiver
        path = receiver.claim(stage, value, payload)
        connection = HTTPConnection("127.0.0.1", receiver.server.server_port,
                                    timeout=max(0.001, min(30, guard["remaining_seconds"])))
        try:
            # No redirect following, retries, URL or body supplied by the model.
            guard = self.ledger.guard_reserved_call(self.arm_id, reservation["request_id"])
            if self.closed or self.clock() >= guard["deadline"]:
                raise PermissionError("dispatch revoked before HTTP send")
            connection.request("POST", "/completion", body=serialized,
                               headers={"Content-Type": "application/json", "X-R0-Claim": path})
            reply = connection.getresponse()
            raw = reply.read(MAX_BYTES + 1)
            if reply.status != 200 or len(raw) > MAX_BYTES:
                raise ValueError("invalid bounded runner response")
            response = strict_json(raw)
            write_new(directory / "runner-response.json", response)
        finally:
            connection.close()
        self.backend.verify_codec()
        remaining = self.ledger.guard_reserved_call(self.arm_id, reservation["request_id"])["remaining_seconds"]
        # Parser helper is bounded independently and still inside the arm.
        envelope = parse_completion(value, response, codec["path"], timeout=min(30, remaining))
        self.backend.verify_codec()
        if envelope.get("prompt_eval_count") != value["input_tokens"]:
            raise ValueError("observed runner input count differs from reserved tokens")
        # Official Qwen parsing has no OpenAI call IDs. The host must assign
        # IDs unique across turns, not chat_codec's response-local call_0.
        # Bind to this one-use reservation, never trust a fixture/model ID.
        for index, call in enumerate(envelope["message"].get("tool_calls", [])):
            call["id"] = f"call_{reservation['request_id']}_{index}"
        return envelope


def broker_factory(assets, codec_pin):
    """Factory for the R0 controller; no labels or evaluator passed to backend."""
    receivers = []

    def create(ledger, arm_id, responder, directory):
        receiver = NoInferenceRunner(responder)
        receivers.append(receiver)
        try:
            return PreparedBroker(ledger, arm_id, PreparedBackend(assets, codec_pin, receiver),
                                  Path(directory))
        except BaseException:
            receiver.close()
            raise
    create.receivers = receivers
    return create
