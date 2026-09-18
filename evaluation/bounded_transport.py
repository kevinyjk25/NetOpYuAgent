"""R0 role-bound loopback model broker; only scripted backends are implemented.

No upstream URL is accepted. This exercises actual HTTP clients and persistent
accounting without inference. Fixture preflight counts are declared test values,
NOT estimates for Qwen. A live tokenizer/renderer attestation is still required.
The trusted host owns role capabilities and phase changes, never request JSON.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import secrets
import socket
import threading
import time
import uuid

from evaluation.bounded_budget import BudgetError, BudgetLedger
from evaluation.bounded_pilot import MODEL, seal, write_new
from evaluation.chat_codec import _native_messages, _openai_response
from network_runtime.contracts import sha256_json

MAX_BYTES = 4 * 1024 * 1024
STAGES = {"agent", "compiler", "runtime", "revision", "fallback"}
INGRESS_SECONDS = 10.0
EGRESS_SECONDS = 10.0


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def _function(value, *, call=False, native=False):
    if not isinstance(value, dict):
        raise ValueError("function object required")
    allowed = {"name", "arguments", "index"} if call and native else (
        {"name", "arguments"} if call else {"name", "description", "parameters", "strict"})
    if set(value) - allowed or not isinstance(value.get("name"), str) or not value["name"]:
        raise ValueError("supported named function required")
    if call:
        arguments = value.get("arguments")
        if isinstance(arguments, str) and not native:
            arguments = strict_json(arguments)
        if not isinstance(arguments, dict):
            raise ValueError("function arguments must encode an object")
        if "index" in value and (type(value["index"]) is not int or value["index"] < 0):
            raise ValueError("invalid function index")
    else:
        if not isinstance(value.get("parameters"), dict):
            raise ValueError("function parameter schema object required")
        if "description" in value and not isinstance(value["description"], str):
            raise ValueError("function description must be text")
        if "strict" in value and type(value["strict"]) is not bool:
            raise ValueError("invalid strict function control")


def _messages(messages, *, native):
    if not isinstance(messages, list) or not messages:
        raise ValueError("complete messages required")
    roles = {"system", "user", "assistant", "tool"} | (set() if native else {"developer"})
    allowed = {"role", "content", "tool_calls", "tool_name"} if native else {
        "role", "content", "tool_calls", "name", "tool_call_id"}
    for item in messages:
        if (not isinstance(item, dict) or set(item) - allowed
                or not isinstance(item.get("role"), str) or item["role"] not in roles):
            raise ValueError("supported text message required")
        calls = item.get("tool_calls")
        if not isinstance(item.get("content"), str) and not (
                item.get("content") is None and item["role"] == "assistant" and calls):
            raise ValueError("text-only message content required")
        if "tool_calls" in item:
            if item["role"] != "assistant" or not isinstance(calls, list) or not calls:
                raise ValueError("assistant function calls required")
            for call in calls:
                if (not isinstance(call, dict) or set(call) - {"id", "type", "function"}
                        or call.get("type", "function") != "function"):
                    raise ValueError("invalid function call")
                if "id" in call and (not isinstance(call["id"], str) or not call["id"]):
                    raise ValueError("invalid function call id")
                _function(call.get("function"), call=True, native=native)
        for key in ("name", "tool_name", "tool_call_id"):
            if key in item and (item["role"] != "tool" or not isinstance(item[key], str) or not item[key]):
                raise ValueError("invalid tool result identity")


def _openai_text_messages(messages):
    """Normalize pi-ai's text parts without trimming, filtering or separators.

    The original request is retained in the host audit artifact. This accepts
    only text, not image/audio parts or per-part provider controls.
    """
    normalized = copy.deepcopy(messages)
    if isinstance(normalized, list):
        for item in normalized:
            if isinstance(item, dict) and isinstance(item.get("content"), list):
                parts = item["content"]
                if not parts or any(not isinstance(part, dict) or set(part) != {"type", "text"}
                                    or part["type"] != "text" or not isinstance(part["text"], str)
                                    for part in parts):
                    raise ValueError("nonempty pure text content parts required")
                item["content"] = "".join(part["text"] for part in parts)
    _messages(normalized, native=False)
    return normalized


def _tools(tools):
    if not isinstance(tools, list):
        raise ValueError("function tools must be a list")
    for tool in tools:
        if (not isinstance(tool, dict) or set(tool) != {"type", "function"}
                or tool["type"] != "function"):
            raise ValueError("function tool required")
        _function(tool["function"])


def _options(options):
    if not isinstance(options, dict) or set(options) - {"num_predict", "num_ctx", "temperature", "top_p", "seed"}:
        raise ValueError("supported options object required")
    for key in ("temperature", "top_p"):
        if key in options:
            value = options[key]
            if (type(value) not in (int, float) or (type(value) is float and not math.isfinite(value)) or value < 0
                    or (key == "top_p" and value > 1)):
                raise ValueError("invalid sampling control")
    if "seed" in options and type(options["seed"]) is not int:
        raise ValueError("integer seed required")
    if "num_ctx" in options and (type(options["num_ctx"]) is not int or options["num_ctx"] <= 0):
        raise ValueError("positive context size required")


class ScriptedModel:
    """Host-written deterministic responses, never a source script or an LLM.

    The responder receives the exact normalized request and trusted role. Token
    values are synthetic fixture costs and explicitly labelled in all reports.
    """
    def __init__(self, responder, *, input_tokens=64, output_tokens=16):
        if any(type(v) is not int or v < 0 for v in (input_tokens, output_tokens)):
            raise ValueError("nonnegative integer fixture counts required")
        self.responder = responder
        self.input_tokens, self.output_tokens = input_tokens, output_tokens
        self.digest = sha256_json({"kind": "scripted_transport_not_qwen_weights",
                                   "input_tokens": input_tokens, "output_tokens": output_tokens})
        self.calls = []

    def preflight(self, wire):
        return {"request_digest": sha256_json(wire), "input_tokens": self.input_tokens,
                "counting_method": "declared_fixture_count_not_live_tokenization"}

    def generate(self, stage, wire, *, deadline):
        self.calls.append({"stage": stage, "request_digest": sha256_json(wire), "deadline": deadline})
        message = self.responder(stage, copy.deepcopy(wire))
        return {"model": MODEL, "message": message, "done": True, "done_reason": "stop",
                "prompt_eval_count": self.input_tokens, "eval_count": self.output_tokens}


class ModelBroker:
    """One existing ledger arm, shared by DSH and nested compiler/Runtime calls.

    A path capability pins the caller's role. The model's body may not select a
    role, arm, identity or accounting data. Unknown completion halts the ledger.
    No resume/replay endpoint exists. Host waiting is deadline-bounded; a Python
    callback cannot be forcibly killed. Undrained callbacks remain disclosed,
    their late result cannot settle again or authorize delivery. Fixture only.
    """
    def __init__(self, ledger: BudgetLedger, arm_id, backend: ScriptedModel, output, *, clock=time.time):
        self._validate_backend(backend)
        self.ledger, self.arm_id, self.backend = ledger, arm_id, backend
        self.clock = clock
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        arm = ledger.inspect_arm(arm_id)["arm"]
        self.arm = arm["arm"]
        # Arm deadlines are immutable: lifecycle watchdog setup must not wait
        # for a SQLite write lock. Dispatch authority still reads the ledger.
        self.deadline = arm["deadline"]
        self.tokens = {role: secrets.token_hex(24) for role in ("agent", "compiler", "runtime")}
        self.agent_stage = "agent"
        self.lock = threading.RLock()
        self.dispatch_lock = threading.RLock()
        self.inflight = False
        self.server = None
        self.closed = False
        self.errors = []
        self.workers = set()
        self.connections = set()
        self.shutdown_done = threading.Event()
        self.shutdown_lock = threading.Lock()
        self.shutdown_failed = False

    def _validate_backend(self, backend):
        if type(backend) is not ScriptedModel:
            raise ValueError("live model forwarding is not implemented")

    def _preflight(self, api, source, wire):
        return self.backend.preflight(wire)

    def _invoke_backend(self, stage, wire, reservation, directory):
        return self.backend.generate(stage, wire, deadline=reservation["deadline"])

    def _measurement(self):
        return {"actualModelCalls": 0, "measurementKind": "scripted_transport_fixture"}

    def _remaining(self):
        """Lifecycle time only; unlike a dispatch guard this grants no action."""
        return max(0.0, self.deadline - self.clock())

    def _http_event(self, code, exc=None):
        event = {"code": code, "type": type(exc).__name__ if exc else None, "at": time.time(),
                 "actualModelCalls": 0, "contains_request_text": False}
        self.errors.append(event)
        try:
            write_new(self.output / ("http-" + uuid.uuid4().hex + ".json"), seal(event))
        except (OSError, ValueError):
            self._halt("http_audit_persistence_failure")

    @contextmanager
    def _serial(self):
        # A queued role must not wait indefinitely behind a blocked caller.
        deadline = time.monotonic() + self._remaining()
        while not self.dispatch_lock.acquire(timeout=0.02):
            if self.closed:
                raise PermissionError("broker closed while queued")
            if time.monotonic() >= deadline or self._remaining() <= 0:
                self._halt("transport_queue_deadline")
                raise TimeoutError("queued request deadline reached")
        try:
            if self.closed:
                raise PermissionError("closed broker")
            if time.monotonic() >= deadline or self._remaining() <= 0:
                raise TimeoutError("request deadline reached")
            yield
        finally:
            self.dispatch_lock.release()

    def _generate(self, stage, wire, reservation, directory):
        """One owner settles; the worker only records its actual completion.

        Returning on timeout is not proof of cancellation of an upstream model.
        No real upstream is implemented. A hung fixture is counted as undrained.
        """
        ready, aborted = threading.Event(), threading.Event()
        box = {}
        remaining = self.ledger.guard_reserved_call(self.arm_id, reservation["request_id"])["remaining_seconds"]
        end = time.monotonic() + remaining

        def work():
            try:
                # Recheck inside the actual worker, after any queue/journal delay.
                with self.lock:
                    if self.closed or aborted.is_set():
                        raise PermissionError("broker stopped before backend start")
                    guard = self.ledger.guard_reserved_call(self.arm_id, reservation["request_id"])
                    if self.closed or aborted.is_set():
                        raise PermissionError("broker stopped during backend guard")
                    if time.monotonic() >= end or self.clock() >= guard["deadline"]:
                        raise TimeoutError("backend deadline reached during guard")
                box["response"] = self._invoke_backend(stage, wire, reservation, directory)
                write_new(directory / "backend-response.json", box["response"])
            except BaseException as exc:
                box["error"] = exc
            finally:
                try:
                    write_new(directory / "backend-completion.json", seal({
                        "completed_after_abort": aborted.is_set(),
                        "error_type": type(box["error"]).__name__ if "error" in box else None,
                        "delivery_authorized": False, "actualModelCalls": 0}))
                except BaseException as exc:
                    box["error"] = exc
                ready.set()
                with self.lock:
                    self.workers.discard(threading.current_thread())

        thread = threading.Thread(target=work, daemon=True)
        with self.lock:
            if self.closed:
                raise PermissionError("broker closed before dispatch")
            self.workers.add(thread)
            thread.start()
        try:
            while not ready.wait(min(0.02, max(0, end - time.monotonic()))):
                if self.closed:
                    raise PermissionError("broker closed while backend running")
                if time.monotonic() >= end or self._remaining() <= 0:
                    raise TimeoutError("backend absolute deadline reached")
                self.ledger.guard_reserved_call(self.arm_id, reservation["request_id"])
            if self.closed:
                raise PermissionError("broker closed before backend result delivery")
            if time.monotonic() >= end or self._remaining() <= 0:
                raise TimeoutError("backend completed after deadline")
            if "error" in box:
                raise box["error"]
            return box["response"]
        except BaseException:
            aborted.set()
            raise

    def set_agent_stage(self, stage):
        """Host-only transition after observed workflow state; no HTTP exposure."""
        if self.arm != "B" or stage not in {"fallback", "revision"}:
            raise BudgetError("only treatment host may mark fallback/revision")
        with self.lock:
            if self.closed:
                raise PermissionError("closed broker")
            self.ledger.check_arm(self.arm_id)
            if self.agent_stage == "revision" or (self.agent_stage == "fallback" and stage != "revision"):
                raise BudgetError("agent phase cannot reset")
            self.agent_stage = stage

    def _role(self, token):
        roles = [role for role, expected in self.tokens.items() if secrets.compare_digest(token, expected)]
        if self.closed or len(roles) != 1:
            raise PermissionError("unknown or closed route capability")
        role = roles[0]
        if self.arm == "A" and role != "agent":
            raise PermissionError("control cannot use compiler/runtime routes")
        return self.agent_stage if role == "agent" else role

    def identity(self, token):
        with self.lock:
            self._role(token)
            self.ledger.check_arm(self.arm_id)
            return {"models": [{"name": MODEL, "digest": self.backend.digest}],
                    **self._measurement()}

    def _halt(self, reason):
        arm = self.ledger.inspect_arm(self.arm_id)["arm"]
        self.ledger.pause_study(arm["study_id"], reason)

    def _record_rejection(self, api, payload, exc):
        """Host diagnostics contain structure, never capability or message text."""
        body = payload if isinstance(payload, dict) else {}
        messages = body.get("messages")
        options = body.get("options")

        def scalar(value):
            return value if type(value) in (int, bool) or value is None else type(value).__name__

        self.errors.append({"type": type(exc).__name__, "at": time.time(),
            "code": "request_structure_rejected", "message": str(exc), "api": api,
            "payload_keys": [str(key)[:80] for key in list(body)[:64]],
            "message_roles": [str(item.get("role", "missing"))[:32] if isinstance(item, dict)
                              else type(item).__name__ for item in messages[:64]]
                             if isinstance(messages, list) else [],
            "model": body.get("model", "")[:128] if isinstance(body.get("model"), str) else None,
            "max_tokens": scalar(body.get("max_tokens")),
            "max_completion_tokens": scalar(body.get("max_completion_tokens")),
            "option_keys": [str(key)[:80] for key in list(options)[:64]] if isinstance(options, dict) else [],
            "num_predict": scalar(options.get("num_predict")) if isinstance(options, dict) else None})

    @staticmethod
    def _wire(api, payload):
        if not isinstance(payload, dict) or payload.get("model") != MODEL:
            raise ValueError("host-pinned model identity required")
        if any(k in payload for k in ("stage", "arm_id", "input_tokens", "usage", "endpoint")):
            raise ValueError("request cannot supply host accounting or routing")
        if api == "openai":
            allowed = {"model", "messages", "tools", "tool_choice", "stream", "stream_options",
                       "max_tokens", "max_completion_tokens", "temperature", "top_p", "seed",
                       "parallel_tool_calls", "store"}
            if set(payload) - allowed or payload.get("tool_choice", "auto") != "auto":
                raise ValueError("unsupported OpenAI request field/control")
            # pi-ai 0.82.1 buildParams sends false. No provider storage is
            # implemented; the host's explicit local audit journal still exists.
            if "store" in payload and payload["store"] is not False:
                raise ValueError("only store=false is supported")
            if "parallel_tool_calls" in payload and payload["parallel_tool_calls"] is not False:
                raise ValueError("parallel tool calls are unsupported")
            if type(payload.get("stream", False)) is not bool:
                raise ValueError("boolean stream control required")
            if "stream_options" in payload:
                options = payload["stream_options"]
                if (not isinstance(options, dict) or set(options) - {"include_usage"}
                        or any(type(value) is not bool for value in options.values())
                        or payload.get("stream") is not True):
                    raise ValueError("unsupported stream options")
            messages = _openai_text_messages(payload.get("messages"))
            limit = payload.get("max_tokens", payload.get("max_completion_tokens", 1400))
            if "max_tokens" in payload and "max_completion_tokens" in payload:
                raise ValueError("one output ceiling required")
            wire = {"model": MODEL, "messages": _native_messages(messages),
                    "stream": False, "think": False,
                    "options": {"num_predict": limit, "temperature": payload.get("temperature", 0)}}
            for key in ("top_p", "seed"):
                if key in payload:
                    wire["options"][key] = payload[key]
            if "tools" in payload:
                wire["tools"] = copy.deepcopy(payload["tools"])
        elif api == "native":
            if (set(payload) - {"model", "messages", "tools", "stream", "think", "format", "options"}
                    or payload.get("stream") is not False or payload.get("think") is not False):
                raise ValueError("nonstreaming native no-think request required")
            wire = copy.deepcopy(payload)
            _messages(wire.get("messages"), native=True)
            _options(wire.get("options"))
            limit = wire.get("options", {}).get("num_predict")
            if "format" in wire and not (wire["format"] == "json" or isinstance(wire["format"], dict)):
                raise ValueError("JSON format or schema object required")
        else:
            raise ValueError("unknown API")
        if type(limit) is not int or not 1 <= limit <= 6000:
            raise ValueError("explicit bounded output ceiling required")
        _options(wire.get("options"))
        if "tools" in wire:
            _tools(wire["tools"])
        sha256_json(wire)  # Strict JSON; source/format retained, never truncated.
        return wire, limit

    def dispatch(self, token, api, payload):
        with self._serial():
            with self.lock:
                stage = self._role(token)
                source_request = copy.deepcopy(payload)
                try:
                    wire, limit = self._wire(api, source_request)
                except ValueError as exc:
                    self._record_rejection(api, payload, exc)
                    raise
            # Preparation may read a large pinned vocabulary. Do not hold the
            # lifecycle lock while doing so: close revokes dispatch immediately.
            preflight = self._preflight(api, source_request, wire)
            with self.lock:
                if self._role(token) != stage:
                    raise PermissionError("role changed during preparation")
                request_id = uuid.uuid4().hex
                reservation = self.ledger.reserve_model_call(self.arm_id, request_id, stage,
                    preflight["input_tokens"], limit)
                self.inflight = True
            directory = self.output / request_id
            settled = False
            try:
                directory.mkdir()
                write_new(directory / "request.json", seal({"arm_id": self.arm_id, "stage": stage,
                    "request_api": api, "source_request": source_request,
                    "source_request_digest": sha256_json(source_request),
                    "normalization": "scripted_openai_native_text_v1" if api == "openai" else "native_identity_v1",
                    "wire": wire, "preflight": preflight, "deadline": reservation["deadline"],
                    **self._measurement()}))
                envelope = self._generate(stage, wire, reservation, directory)
                write_new(directory / "response.json", envelope)
                if (envelope.get("model") != MODEL or envelope.get("done") is not True
                        or not isinstance(envelope.get("message"), dict)):
                    raise ValueError("incomplete/wrong model response")
                row = self.ledger.settle_call(self.arm_id, request_id,
                    envelope.get("prompt_eval_count"), envelope.get("eval_count"))
                settled = True
                write_new(directory / "settlement.json", seal(row))
                if row["status"] != "settled":
                    raise BudgetError("unknown/invalid/expired usage; no replay or fallback")
                with self.lock:
                    if self.closed:
                        raise PermissionError("broker closed during request")
                    self.ledger.check_arm(self.arm_id)
                    return copy.deepcopy(envelope)
            except BaseException:
                if not settled:
                    row = self.ledger.settle_call(self.arm_id, request_id)
                    write_new(directory / "settlement.json", seal(row))
                else:
                    self._halt("transport_delivery_or_deadline_failure")
                raise
            finally:
                with self.lock:
                    self.inflight = False

    def start(self):
        if self.server is not None or self.closed:
            raise ValueError("broker cannot restart")
        broker = self
        class Handler(BaseHTTPRequestHandler):
            model_claimed = False

            def handle(self):
                # The watchdog covers the request line, headers AND slow-drip
                # body reads. socket.settimeout alone only bounds idle gaps.
                self.ingress_done = threading.Event()
                self.ingress_expired = threading.Event()
                self.ingress_end = time.monotonic() + min(INGRESS_SECONDS, broker._remaining())

                def expire():
                    if self.ingress_done.is_set():
                        return
                    self.ingress_expired.set()
                    try:
                        self.connection.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    broker._http_event("http_ingress_deadline", TimeoutError())

                with broker.lock:
                    if broker.closed:
                        self.close_connection = True
                        return
                    broker.connections.add(self.connection)
                self.ingress_timer = threading.Timer(max(0, self.ingress_end - time.monotonic()), expire)
                self.ingress_timer.daemon = True
                self.ingress_timer.start()
                try:
                    super().handle()
                finally:
                    self.ingress_done.set()
                    self.ingress_timer.cancel()
                    with broker.lock:
                        broker.connections.discard(self.connection)

            def finish_ingress(self):
                if hasattr(self, "ingress_done"):
                    self.ingress_done.set()
                    self.ingress_timer.cancel()
                    if self.ingress_expired.is_set() or time.monotonic() >= self.ingress_end:
                        raise TimeoutError("absolute HTTP ingress deadline reached")

            def log_message(self, *_):
                pass

            def handle_request(self, post):
                try:
                    pieces = self.path.split("/")
                    if len(pieces) != 5 or pieces[1] != "r":
                        raise PermissionError("unknown route")
                    token, path = pieces[2], "/".join(pieces[3:])
                    if not post and path == "api/tags":
                        self.finish_ingress()
                        output, streaming = broker.identity(token), False
                    elif post and path in {"api/chat", "v1/chat/completions"}:
                        # OpenAI has one extra path segment, handled below.
                        output, streaming = self.model_post(token, path)
                    else:
                        raise PermissionError("unknown route")
                    self.reply(200, output, streaming)
                except (ValueError, PermissionError, KeyError, TypeError, TimeoutError) as exc:
                    broker._http_event("http_request_rejected", exc)
                    self.reply(409, {"error": {"message": type(exc).__name__}}, False)

            def model_post(self, token, path):
                broker._role(token)
                lengths = self.headers.get_all("Content-Length", []) if hasattr(self.headers, "get_all") else [self.headers.get("Content-Length")]
                if len(lengths) != 1 or self.headers.get("Transfer-Encoding") is not None:
                    raise ValueError("one explicit Content-Length and no transfer encoding required")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BYTES:
                    raise ValueError("bounded request body required")
                self.connection.settimeout(min(10.0, broker.ledger.check_arm(broker.arm_id)["remaining_seconds"]))
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("truncated HTTP request body")
                self.finish_ingress()
                payload = strict_json(raw)
                api = "native" if path == "api/chat" else "openai"
                native = broker.dispatch(token, api, payload)
                self.model_claimed = True
                streaming = api == "openai" and payload.get("stream") is True
                try:
                    return (native if api == "native" else _openai_response(native, stream=streaming)), streaming
                except BaseException:
                    broker._halt("transport_response_encoding_failure")
                    raise

            def do_GET(self):  # noqa: N802
                self.model_claimed = False
                self.handle_request(False)

            def do_POST(self):  # noqa: N802
                # A later role cannot obtain a result before this request's
                # delivery outcome is known. dispatch re-enters this same lock.
                try:
                    with broker._serial():
                        self.model_claimed = False
                        self.post_request()
                except (ValueError, PermissionError, TimeoutError) as exc:
                    broker._http_event("http_queue_rejected", exc)
                    self.close_connection = True

            def post_request(self):
                pieces = self.path.split("/")
                if len(pieces) == 6 and pieces[1] == "r" and pieces[3:] == ["v1", "chat", "completions"]:
                    try:
                        self.reply(200, *self.model_post(pieces[2], "v1/chat/completions"))
                    except (ValueError, PermissionError, KeyError, TypeError, TimeoutError) as exc:
                        broker._http_event("http_request_rejected", exc)
                        self.reply(409, {"error": {"message": type(exc).__name__}}, False)
                else:
                    self.handle_request(True)

            def reply(self, status, value, streaming):
                done, expired = threading.Event(), threading.Event()
                end = time.monotonic() + min(EGRESS_SECONDS, broker._remaining())

                def expire():
                    if done.is_set():
                        return
                    expired.set()
                    try:
                        self.connection.shutdown(socket.SHUT_RDWR)
                    except (OSError, AttributeError):
                        pass
                    broker._http_event("http_egress_deadline", TimeoutError())
                    if self.model_claimed:
                        broker._halt("transport_response_delivery_deadline")

                def check():
                    if expired.is_set() or time.monotonic() >= end or broker._remaining() <= 0:
                        raise TimeoutError("absolute HTTP egress deadline reached")
                    if broker.closed:
                        raise PermissionError("broker closed before/during delivery")
                    if self.model_claimed and status == 200:
                        broker.ledger.check_arm(broker.arm_id)

                timer = threading.Timer(max(0, end - time.monotonic()), expire)
                timer.daemon = True
                timer.start()
                try:
                    # The absolute watchdog spans encoding, headers, body and
                    # flush. It closes real sockets even under slow backpressure.
                    # Python encoding itself is cooperative, not forcibly killed.
                    check()
                    self.connection.settimeout(max(0.001, end - time.monotonic()))
                    body = json.dumps(value, ensure_ascii=False, allow_nan=False)
                    raw = ("data: " + body + "\n\ndata: [DONE]\n\n" if streaming else body).encode()
                    check()
                    self.send_response(status)
                    self.send_header("Content-Type", "text/event-stream" if streaming else "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    check()
                    self.end_headers()
                    check()
                    self.wfile.write(raw)
                    check()
                    self.wfile.flush()
                    check()
                except (OSError, ValueError, TypeError) as exc:
                    # No second response: headers or a partial body may already
                    # have been sent. Settled claims stay charged; no next call.
                    self.close_connection = True
                    broker._http_event("http_delivery_failed", exc)
                    if self.model_claimed:
                        broker._halt("transport_response_delivery_failure")
                finally:
                    done.set()
                    timer.cancel()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        return self

    def route(self, role):
        if self.server is None or self.closed:
            raise ValueError("broker is not running")
        return self.base_url + "/r/" + self.tokens[role]

    def close(self, timeout=0.5):
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError("finite nonnegative close timeout required")
        end = time.monotonic() + timeout
        # Revocation precedes any lock wait. No callback is forcibly killed.
        self.closed = True
        # Always schedule eventual cleanup, even when the lifecycle lock is
        # held beyond this caller's timeout. A second close is not required.
        # This separate once-lock never encloses user code, DB or socket I/O.
        with self.shutdown_lock:
            first = not hasattr(self, "shutdown_thread")
            if first:
                def shutdown():
                    try:
                        with self.lock:
                            connections = tuple(self.connections)
                            pending = bool(self.inflight or self.workers)
                        for connection in connections:
                            try:
                                connection.shutdown(socket.SHUT_RDWR)
                            except OSError:
                                pass
                        if self.server is not None:
                            self.server.shutdown()
                            self.server.server_close()
                            self.thread.join()
                        if pending:
                            self._halt("transport_closed_with_inflight_request")
                    except BaseException as exc:
                        self.shutdown_failed = True
                        self._http_event("transport_shutdown_failed", exc)
                    finally:
                        self.shutdown_done.set()
                self.shutdown_thread = threading.Thread(target=shutdown, daemon=True)
                self.shutdown_thread.start()
        while time.monotonic() < end:
            if not self.lock.acquire(timeout=max(0, end - time.monotonic())):
                break
            try:
                if self.shutdown_done.is_set() and not (self.inflight or self.workers or self.connections):
                    break
            finally:
                self.lock.release()
            time.sleep(min(0.01, max(0, end - time.monotonic())))
        if not self.lock.acquire(blocking=False):
            return {"closed": True, "drained": False, "inflight": None, "workers": None, "connections": None}
        try:
            return {"closed": True, "drained": self.shutdown_done.is_set() and not self.shutdown_failed
                    and not (self.inflight or self.workers or self.connections), "cleanup_failed": self.shutdown_failed,
                    "inflight": self.inflight, "workers": len(self.workers), "connections": len(self.connections)}
        finally:
            self.lock.release()
