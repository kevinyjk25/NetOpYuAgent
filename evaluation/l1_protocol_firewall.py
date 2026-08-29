"""Loopback-only P1.8-C2 protocol firewall and complete-attempt meter.

This proxy sits between the immutable C1 Protocol Governor and a local
OpenAI-compatible model endpoint.  It validates the model's typed proposal
against the request's supplied candidate contract, retries malformed protocol
output with bounded generic feedback, and can synthesize only a deterministic
refusal/out-of-scope Tool call required by the reviewed guard policy.  It has
no Runtime, Provider, device, approval, shell, filesystem Tool, or credential
path.
"""

from __future__ import annotations

import http.server
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any

from network_runtime.contracts import sha256_json

from .dsh_controlled_tool import (
    EXPECTED_TOOLS,
    _compile_controlled_decision,
    _typed_envelope,
)
from .dsh_shadow_tool import _loopback_base_url
from .l1_catalog import L1CatalogEntry
from .l1_guard_policy import GuardVerdict, L1GuardPolicy


MAX_FIREWALL_BODY = 8_000_000
FIREWALL_CONTRACT = "netopyu.io/l1-protocol-firewall/v1"


@dataclass(frozen=True)
class ParsedModelAttempt:
    tool_count: int
    tool_name: str | None
    arguments: dict[str, Any] | None
    input_tokens: int
    output_tokens: int
    usage_present: bool
    response_digest: str


@dataclass(frozen=True)
class FirewallAttempt:
    sequence: int
    request_digest: str
    response_digest: str
    policy_action: str
    policy_reason: str
    tool_name: str | None
    candidate_action: str | None
    candidate_target: str | None
    valid: bool
    error_type: str | None
    input_tokens: int
    output_tokens: int
    usage_present: bool
    elapsed_ms: float
    synthetic: bool = False


def _strict_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    return ""


def parse_decision_context(messages: list[Any]) -> tuple[str, tuple[L1CatalogEntry, ...]]:
    marker = "\nUSER_REQUEST="
    for raw in reversed(messages):
        if not isinstance(raw, dict) or raw.get("role") != "user":
            continue
        text = _message_text(raw)
        if not text.startswith("CANDIDATES=") or marker not in text:
            continue
        candidates_raw, prompt_raw = text[len("CANDIDATES="):].split(marker, 1)
        payload = json.loads(candidates_raw)
        prompt = json.loads(prompt_raw)
        if not isinstance(prompt, str) or not isinstance(payload, list) or len(payload) > 32:
            raise ValueError("C2 decision context is invalid")
        candidates: list[L1CatalogEntry] = []
        expected = {
            "target", "kind", "profile", "description", "parameters",
            "required_parameters", "workflow_hint", "risk_level", "requires_approval",
        }
        for item in payload:
            card = _strict_object(item, "candidate")
            if set(card) != expected:
                raise ValueError("C2 public candidate fields are not exact")
            candidates.append(L1CatalogEntry(
                target=str(card["target"]),
                kind=str(card["kind"]),
                profile=str(card["profile"]),
                description=str(card["description"]),
                parameters={str(k): str(v) for k, v in _strict_object(
                    card["parameters"], "candidate parameters",
                ).items()},
                required_parameters=tuple(card["required_parameters"]),
                workflow_hint=tuple(card["workflow_hint"]),
                risk_level=str(card["risk_level"]),
                requires_approval=bool(card["requires_approval"]),
                searchable_text="",
            ))
        return prompt, tuple(candidates)
    raise ValueError("C2 request does not contain one bounded decision context")


def parse_model_attempt(raw: bytes, content_type: str) -> ParsedModelAttempt:
    calls: dict[int, dict[str, str]] = {}
    input_tokens = 0
    output_tokens = 0
    usage_present = False
    values: list[dict[str, Any]] = []
    if "text/event-stream" in content_type:
        for line in raw.splitlines():
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            try:
                value = json.loads(line[6:].decode("utf-8", "strict"))
            except (UnicodeError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                values.append(value)
    else:
        value = json.loads(raw.decode("utf-8", "strict"))
        values = [_strict_object(value, "model response")]

    for value in values:
        usage = value.get("usage")
        if isinstance(usage, dict):
            usage_present = True
            input_tokens = max(input_tokens, int(
                usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            ))
            output_tokens = max(output_tokens, int(
                usage.get("completion_tokens") or usage.get("output_tokens") or 0
            ))
        choices = value.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("delta") if isinstance(choice.get("delta"), dict) else choice.get("message")
            if not isinstance(message, dict):
                continue
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for position, call in enumerate(tool_calls):
                if not isinstance(call, dict):
                    continue
                index = call.get("index") if isinstance(call.get("index"), int) else position
                function = call.get("function")
                if not isinstance(function, dict):
                    continue
                current = calls.setdefault(index, {"name": "", "arguments": ""})
                if isinstance(function.get("name"), str):
                    current["name"] += function["name"]
                if isinstance(function.get("arguments"), str):
                    current["arguments"] += function["arguments"]

    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    if len(calls) == 1:
        only = next(iter(calls.values()))
        tool_name = only["name"] or None
        try:
            decoded = json.loads(only["arguments"])
        except json.JSONDecodeError:
            decoded = None
        arguments = decoded if isinstance(decoded, dict) else None
    return ParsedModelAttempt(
        tool_count=len(calls),
        tool_name=tool_name,
        arguments=arguments,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        usage_present=usage_present,
        response_digest=sha256_json({"raw_sha256": __import__("hashlib").sha256(raw).hexdigest()}),
    )


def _validate_attempt(
    parsed: ParsedModelAttempt,
    candidates: tuple[L1CatalogEntry, ...],
    verdict: GuardVerdict,
) -> tuple[bool, str | None, str | None, str | None]:
    if parsed.tool_count != 1 or parsed.tool_name not in EXPECTED_TOOLS:
        return False, "TypedToolMissingOrMultiple", None, None
    if parsed.arguments is None:
        return False, "TypedArgumentsInvalid", None, None
    try:
        _typed_envelope(parsed.tool_name, parsed.arguments)
        decision = _compile_controlled_decision(parsed.tool_name, parsed.arguments, candidates)
    except (TypeError, ValueError):
        return False, "CandidateContractInvalid", None, None
    expected = {
        "refuse": "refuse_l1_request",
        "out_of_scope": "reject_l1_out_of_scope",
    }.get(verdict.action)
    if expected is not None and parsed.tool_name != expected:
        return False, "GuardPolicyMismatch", decision.action.value, decision.target
    return True, None, decision.action.value, decision.target


def _synthetic_tool_call(model: str, tool_name: str, reason_code: str) -> bytes:
    arguments = json.dumps({
        "confidence": 1.0,
        "reason_code": reason_code[:80],
    }, ensure_ascii=False, separators=(",", ":"))
    first = {
        "id": "netopyu-c2-guard",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant",
                "tool_calls": [{
                    "index": 0,
                    "id": "netopyu-c2-policy-call",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": arguments},
                }],
            },
            "finish_reason": None,
        }],
    }
    final = {
        "id": "netopyu-c2-guard",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "tool_calls",
        }],
    }
    return (
        "data: " + json.dumps(first, separators=(",", ":")) + "\n\n"
        "data: " + json.dumps(final, separators=(",", ":")) + "\n\n"
        "data: [DONE]\n\n"
    ).encode("utf-8")


def _synthetic_invalid(model: str) -> bytes:
    value = {
        "id": "netopyu-c2-invalid",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": "proposal rejected"},
            "finish_reason": "stop",
        }],
    }
    return (
        "data: " + json.dumps(value, separators=(",", ":")) + "\n\n"
        "data: [DONE]\n\n"
    ).encode("utf-8")


class _FirewallServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        upstream_base_url: str,
        model: str,
        policy: L1GuardPolicy,
        repair_limit: int,
    ) -> None:
        super().__init__(("127.0.0.1", 0), _FirewallHandler)
        self.upstream_base_url = _loopback_base_url(upstream_base_url)
        self.model = model
        self.policy = policy
        self.repair_limit = repair_limit
        self.lock = threading.Lock()
        self.attempts: list[FirewallAttempt] = []

    def append(self, value: FirewallAttempt) -> None:
        with self.lock:
            self.attempts.append(value)

    def snapshot(self) -> int:
        with self.lock:
            return len(self.attempts)

    def since(self, index: int) -> tuple[FirewallAttempt, ...]:
        with self.lock:
            return tuple(self.attempts[index:])


class _FirewallHandler(http.server.BaseHTTPRequestHandler):
    server: _FirewallServer
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _send(self, status: int, raw: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)
        self.close_connection = True

    def _reject(self, status: int, message: str) -> None:
        raw = json.dumps({"error": {"message": message}}).encode("utf-8")
        self._send(status, raw, "application/json")

    def _upstream(self, payload: dict[str, Any]) -> tuple[bytes, str, float]:
        request = urllib.request.Request(
            self.server.upstream_base_url + "/v1/chat/completions",
            data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": self.headers.get("Authorization", "Bearer local-loopback"),
            },
            method="POST",
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=900) as response:
            raw = response.read(MAX_FIREWALL_BODY + 1)
            if len(raw) > MAX_FIREWALL_BODY:
                raise ValueError("C2 model response exceeds firewall limit")
            return raw, response.headers.get("Content-Type", "text/event-stream"), (
                time.perf_counter() - started
            ) * 1000

    def do_POST(self) -> None:  # noqa: N802 - stdlib API
        if urllib.parse.urlparse(self.path).path != "/v1/chat/completions":
            self._reject(404, "C2 firewall exposes chat completions only")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._reject(400, "invalid content length")
            return
        if not 1 <= length <= MAX_FIREWALL_BODY:
            self._reject(413, "request body is empty or too large")
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8", "strict"))
            request_object = _strict_object(payload, "request")
            if request_object.get("model") != self.server.model:
                raise ValueError("request model differs from reviewed model")
            messages = request_object.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError("request messages are absent")
            prompt, candidates = parse_decision_context(messages)
            verdict = self.server.policy.classify(prompt)
        except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            self._reject(400, f"C2 request rejected: {type(error).__name__}")
            return

        request_digest = sha256_json({
            "model": self.server.model,
            "messages": messages,
            "tools": request_object.get("tools"),
        })
        last_error = "TypedToolMissingOrMultiple"
        content_type = "text/event-stream"
        try:
            for attempt_index in range(self.server.repair_limit + 1):
                current = dict(request_object)
                if attempt_index:
                    current["messages"] = [
                        *messages,
                        {
                            "role": "system",
                            "content": (
                                "C2_PROTOCOL_REPAIR: the hidden proposal failed the reviewed "
                                f"typed/guard contract ({last_error}). Use exactly one declared "
                                "proposal Tool and only its declared fields. Never add user values."
                            ),
                        },
                    ]
                raw, content_type, elapsed_ms = self._upstream(current)
                parsed = parse_model_attempt(raw, content_type)
                valid, error_type, candidate_action, candidate_target = _validate_attempt(
                    parsed, candidates, verdict,
                )
                self.server.append(FirewallAttempt(
                    sequence=self.server.snapshot() + 1,
                    request_digest=request_digest,
                    response_digest=parsed.response_digest,
                    policy_action=verdict.action,
                    policy_reason=verdict.reason_code,
                    tool_name=parsed.tool_name,
                    candidate_action=candidate_action,
                    candidate_target=candidate_target,
                    valid=valid,
                    error_type=error_type,
                    input_tokens=parsed.input_tokens,
                    output_tokens=parsed.output_tokens,
                    usage_present=parsed.usage_present,
                    elapsed_ms=round(elapsed_ms, 3),
                ))
                if valid:
                    self._send(200, raw, content_type)
                    return
                last_error = error_type or "ProtocolInvalid"

            expected_tool = {
                "refuse": "refuse_l1_request",
                "out_of_scope": "reject_l1_out_of_scope",
            }.get(verdict.action)
            if expected_tool:
                raw = _synthetic_tool_call(
                    self.server.model, expected_tool, verdict.reason_code,
                )
                self.server.append(FirewallAttempt(
                    sequence=self.server.snapshot() + 1,
                    request_digest=request_digest,
                    response_digest=sha256_json({
                        "synthetic": expected_tool, "reason": verdict.reason_code,
                    }),
                    policy_action=verdict.action,
                    policy_reason=verdict.reason_code,
                    tool_name=expected_tool,
                    candidate_action=verdict.action,
                    candidate_target=None,
                    valid=True,
                    error_type=None,
                    input_tokens=0,
                    output_tokens=0,
                    usage_present=True,
                    elapsed_ms=0.0,
                    synthetic=True,
                ))
                self._send(200, raw, "text/event-stream")
                return
            self._send(200, _synthetic_invalid(self.server.model), "text/event-stream")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (OSError, TimeoutError, ValueError, urllib.error.URLError) as error:
            try:
                self._reject(502, f"C2 local model upstream failed: {type(error).__name__}")
            except (BrokenPipeError, ConnectionResetError):
                pass


class L1ProtocolFirewall:
    """Managed loopback firewall with per-attempt cost evidence."""

    def __init__(
        self,
        upstream_base_url: str,
        model: str,
        policy: L1GuardPolicy,
        *,
        repair_limit: int = 2,
    ) -> None:
        if not 0 <= repair_limit <= 4:
            raise ValueError("C2 repair limit must be between 0 and 4")
        self.server = _FirewallServer(upstream_base_url, model, policy, repair_limit)
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            name="netopyu-l1-protocol-firewall",
            daemon=True,
        )
        self.thread.start()
        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}"

    def snapshot(self) -> int:
        return self.server.snapshot()

    def since(self, index: int) -> tuple[FirewallAttempt, ...]:
        return self.server.since(index)

    def metrics(self) -> dict[str, Any]:
        attempts = self.since(0)
        actual = [item for item in attempts if not item.synthetic]
        return {
            "contract": FIREWALL_CONTRACT,
            "model_attempts": len(actual),
            "valid_model_attempts": sum(item.valid for item in actual),
            "repair_attempts": max(0, len(actual) - len({
                item.request_digest for item in actual
            })),
            "synthetic_safe_calls": sum(item.synthetic for item in attempts),
            "usage_complete": all(item.usage_present for item in actual),
            "input_tokens": sum(item.input_tokens for item in actual),
            "output_tokens": sum(item.output_tokens for item in actual),
            "upstream_elapsed_ms": round(sum(item.elapsed_ms for item in actual), 3),
            "failures": {
                error: sum(item.error_type == error for item in actual)
                for error in sorted({item.error_type for item in actual if item.error_type})
            },
        }

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def __enter__(self) -> "L1ProtocolFirewall":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def attempts_as_dict(attempts: tuple[FirewallAttempt, ...]) -> list[dict[str, Any]]:
    return [asdict(item) for item in attempts]


__all__ = [
    "FIREWALL_CONTRACT",
    "FirewallAttempt",
    "L1ProtocolFirewall",
    "ParsedModelAttempt",
    "attempts_as_dict",
    "parse_decision_context",
    "parse_model_attempt",
]
