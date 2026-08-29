"""Loopback-only P1.8-C3 candidate Schema gateway.

The gateway binds each DSH request to one reviewed candidate contract, forces a
single Tool choice, validates the dynamic Tool surface and proposal, meters all
upstream attempts, and may synthesize only a Guard-required refusal or
out-of-scope call.  It cannot select a normal candidate or invent a business
argument.
"""

from __future__ import annotations

import http.server
import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, replace
from typing import Any

from network_runtime.contracts import sha256_json

from .dsh_shadow_tool import _loopback_base_url
from .l1_adapters import AdapterResponse
from .l1_argument_grounding import GroundingResult, L1ArgumentGroundingPolicy
from .l1_catalog import L1CatalogEntry
from .l1_contract import L1Decision
from .l1_guard_policy import GuardVerdict, L1GuardPolicy
from .l1_protocol_firewall import ParsedModelAttempt, parse_model_attempt


SCHEMA_GATEWAY_CONTRACT = "netopyu.io/l1-schema-gateway/v1"
SCHEMA_CONTROLLER_CONTRACT = "netopyu.io/l1-schema-controller/v1"
MAX_SCHEMA_BODY = 8_000_000
_CANDIDATE_TOOL = re.compile(r"select_candidate_(\d{2})\Z")
UNCALIBRATED_CONFIDENCE = 0.55


@dataclass(frozen=True)
class SchemaContext:
    scenario_id: str
    prompt: str
    candidates: tuple[L1CatalogEntry, ...]
    candidate_digest: str


@dataclass(frozen=True)
class SchemaAttempt:
    sequence: int
    scenario_id: str
    request_digest: str
    response_digest: str
    policy_action: str
    policy_reason: str
    tool_name: str | None
    candidate_index: int | None
    candidate_action: str | None
    candidate_target: str | None
    valid: bool
    error_type: str | None
    argument_fields: tuple[str, ...]
    business_argument_fields: tuple[str, ...]
    dropped_argument_fields: tuple[str, ...]
    normalized_argument_fields: tuple[str, ...]
    schema_dropped_argument_fields: tuple[str, ...]
    input_tokens: int
    output_tokens: int
    usage_present: bool
    elapsed_ms: float
    synthetic: bool = False


def candidate_contract(candidates: tuple[L1CatalogEntry, ...]) -> list[dict[str, Any]]:
    if not 1 <= len(candidates) <= 12:
        raise ValueError("C3 requires 1..12 candidate cards")
    cards = [item.public_card() for item in candidates]
    identities = {(item.kind, item.target) for item in candidates}
    if len(identities) != len(candidates):
        raise ValueError("C3 candidate identities must be unique")
    return cards


def candidate_contract_digest(candidates: tuple[L1CatalogEntry, ...]) -> str:
    return sha256_json(candidate_contract(candidates))


def candidate_tool_names(candidates: tuple[L1CatalogEntry, ...]) -> tuple[str, ...]:
    return tuple(sorted((
        *(f"select_candidate_{index:02d}" for index in range(len(candidates))),
        "refuse_l1_request",
        "reject_l1_out_of_scope",
    )))


def schema_decision_prompt(prompt: str, digest: str) -> str:
    return (
        "CANDIDATE_CONTRACT_DIGEST=" + digest
        + "\nUSER_REQUEST=" + json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))
    )


def parse_schema_context(messages: list[Any]) -> tuple[str, str]:
    marker = "\nUSER_REQUEST="
    prefix = "CANDIDATE_CONTRACT_DIGEST="
    for raw in reversed(messages):
        if not isinstance(raw, dict) or raw.get("role") != "user":
            continue
        content = raw.get("content")
        text = content if isinstance(content, str) else ""
        if not text.startswith(prefix) or marker not in text:
            continue
        digest, prompt_raw = text[len(prefix):].split(marker, 1)
        prompt = json.loads(prompt_raw)
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest) or not isinstance(prompt, str):
            raise ValueError("C3 prompt binding is invalid")
        return digest, prompt
    raise ValueError("C3 request lacks its candidate contract binding")


def _strict_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def compile_schema_decision(
    tool_name: str,
    arguments: dict[str, Any],
    candidates: tuple[L1CatalogEntry, ...],
    prompt: str,
    grounding_policy: L1ArgumentGroundingPolicy,
) -> tuple[L1Decision, int | None, GroundingResult]:
    if tool_name in {"refuse_l1_request", "reject_l1_out_of_scope"}:
        if arguments:
            raise ValueError("C3 terminal proposal accepts no business arguments")
        action = "refuse" if tool_name == "refuse_l1_request" else "out_of_scope"
        decision = L1Decision.model_validate({
            "apiVersion": "netopyu.io/l1-decision/v1",
            "action": action,
            "target": None,
            "arguments": {},
            "missing_fields": [],
            "workflow": [],
            "confidence": UNCALIBRATED_CONFIDENCE,
            "reason_code": "model_refusal" if action == "refuse" else "model_out_of_scope",
        })
        return decision, None, GroundingResult({}, (), ())
    match = _CANDIDATE_TOOL.fullmatch(tool_name)
    if match is None:
        raise ValueError("C3 Tool name is outside the dynamic candidate contract")
    index = int(match.group(1))
    if index >= len(candidates):
        raise ValueError("C3 candidate index is outside the supplied contract")
    supplied = arguments
    selected = candidates[index]
    if not set(supplied) <= set(selected.parameters):
        raise ValueError("C3 arguments escape the candidate-specific Schema")
    grounding = grounding_policy.apply(prompt, supplied, set(selected.parameters))
    normalized_arguments = grounding.arguments
    absent = tuple(
        key for key in selected.required_parameters if key not in normalized_arguments
    )
    action = "clarify" if absent else f"select_{selected.kind}"
    decision = L1Decision.model_validate({
        "apiVersion": "netopyu.io/l1-decision/v1",
        "action": action,
        "target": selected.target,
        "arguments": normalized_arguments,
        "missing_fields": absent,
        "workflow": [] if absent else selected.workflow_hint,
        "confidence": UNCALIBRATED_CONFIDENCE,
        "reason_code": f"candidate_schema_{index:02d}",
    })
    return decision, index, grounding


def validate_tool_surface(
    tools: Any,
    candidates: tuple[L1CatalogEntry, ...],
) -> bool:
    if not isinstance(tools, list) or len(tools) != len(candidates) + 2:
        return False
    functions: dict[str, dict[str, Any]] = {}
    for raw in tools:
        if not isinstance(raw, dict) or raw.get("type") != "function":
            return False
        function = raw.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            return False
        functions[function["name"]] = function
    if tuple(sorted(functions)) != candidate_tool_names(candidates):
        return False
    for index, candidate in enumerate(candidates):
        parameters = functions[f"select_candidate_{index:02d}"].get("parameters")
        if not isinstance(parameters, dict) or parameters.get("additionalProperties") is not False:
            return False
        if parameters.get("required") not in (None, []):
            return False
        properties = parameters.get("properties")
        if not isinstance(properties, dict) or set(properties) != set(candidate.parameters):
            return False
    for name in ("refuse_l1_request", "reject_l1_out_of_scope"):
        parameters = functions[name].get("parameters")
        if not isinstance(parameters, dict) or parameters.get("additionalProperties") is not False:
            return False
        if parameters.get("required") not in (None, []) or parameters.get("properties") != {}:
            return False
    return True


def validate_model_attempt(
    parsed: ParsedModelAttempt,
    candidates: tuple[L1CatalogEntry, ...],
    verdict: GuardVerdict,
    prompt: str,
    grounding_policy: L1ArgumentGroundingPolicy,
) -> tuple[bool, str | None, L1Decision | None, int | None, GroundingResult | None]:
    if parsed.tool_count != 1 or parsed.tool_name not in candidate_tool_names(candidates):
        return False, "CandidateToolMissingOrMultiple", None, None, None
    if parsed.arguments is None:
        return False, "CandidateArgumentsInvalid", None, None, None
    try:
        decision, index, grounding = compile_schema_decision(
            parsed.tool_name, parsed.arguments, candidates, prompt, grounding_policy,
        )
    except (TypeError, ValueError):
        return False, "CandidateSchemaInvalid", None, None, None
    required_action = {
        "refuse": "refuse",
        "out_of_scope": "out_of_scope",
    }.get(verdict.action)
    if required_action is not None and decision.action.value != required_action:
        return False, "GuardPolicyMismatch", decision, index, grounding
    return True, None, decision, index, grounding


def constrain_attempt_to_candidate_schema(
    parsed: ParsedModelAttempt,
    candidates: tuple[L1CatalogEntry, ...],
) -> tuple[ParsedModelAttempt, tuple[str, ...]]:
    """Drop only undeclared keys while preserving the model's exact Tool choice."""
    if parsed.tool_count != 1 or parsed.tool_name is None or parsed.arguments is None:
        return parsed, ()
    if parsed.tool_name in {"refuse_l1_request", "reject_l1_out_of_scope"}:
        allowed: set[str] = set()
    else:
        match = _CANDIDATE_TOOL.fullmatch(parsed.tool_name)
        if match is None or int(match.group(1)) >= len(candidates):
            return parsed, ()
        allowed = set(candidates[int(match.group(1))].parameters)
    dropped = tuple(sorted(set(parsed.arguments) - allowed))
    if not dropped:
        return parsed, ()
    constrained = {
        key: value for key, value in parsed.arguments.items() if key in allowed
    }
    return replace(parsed, arguments=constrained), dropped


def _proposal_tool_call(
    model: str,
    tool_name: str,
    arguments: dict[str, Any],
    call_id: str,
) -> bytes:
    encoded = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    values = [
        {
            "id": "netopyu-c3-proposal", "object": "chat.completion.chunk",
            "created": 0, "model": model,
            "choices": [{"index": 0, "delta": {
                "role": "assistant", "tool_calls": [{
                    "index": 0, "id": call_id, "type": "function",
                    "function": {"name": tool_name, "arguments": encoded},
                }],
            }, "finish_reason": None}],
        },
        {
            "id": "netopyu-c3-proposal", "object": "chat.completion.chunk",
            "created": 0, "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        },
    ]
    return ("".join(
        "data: " + json.dumps(value, separators=(",", ":")) + "\n\n"
        for value in values
    ) + "data: [DONE]\n\n").encode()


def _synthetic_tool_call(model: str, tool_name: str, reason_code: str) -> bytes:
    del reason_code
    return _proposal_tool_call(model, tool_name, {}, "netopyu-c3-policy-call")


def _synthetic_text(model: str, text: str) -> bytes:
    values = [
        {
            "id": "netopyu-c3-final", "object": "chat.completion.chunk",
            "created": 0, "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}],
        },
        {
            "id": "netopyu-c3-final", "object": "chat.completion.chunk",
            "created": 0, "model": model,
            "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}],
        },
    ]
    return ("".join(
        "data: " + json.dumps(value, separators=(",", ":")) + "\n\n"
        for value in values
    ) + "data: [DONE]\n\n").encode()


def sanitize_valid_tool_response(raw: bytes, content_type: str) -> tuple[bytes, int]:
    """Remove visible assistant text only after one Tool call passed C3 gates."""
    stripped = 0
    if "text/event-stream" in content_type:
        output: list[bytes] = []
        for line in raw.splitlines():
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                if line:
                    output.append(line + b"\n\n")
                continue
            try:
                value = json.loads(line[6:].decode("utf-8", "strict"))
            except (UnicodeError, json.JSONDecodeError):
                output.append(line + b"\n\n")
                continue
            choices = value.get("choices") if isinstance(value, dict) else None
            if isinstance(choices, list):
                for choice in choices:
                    if not isinstance(choice, dict):
                        continue
                    delta = choice.get("delta")
                    if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                        if delta["content"]:
                            stripped += 1
                        delta["content"] = ""
            output.append((
                "data: " + json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n\n"
            ).encode())
        return b"".join(output), stripped
    try:
        value = json.loads(raw.decode("utf-8", "strict"))
    except (UnicodeError, json.JSONDecodeError):
        return raw, 0
    choices = value.get("choices") if isinstance(value, dict) else None
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                if message["content"]:
                    stripped += 1
                message["content"] = None
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(), stripped


class _SchemaGatewayServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        upstream_base_url: str,
        model: str,
        policy: L1GuardPolicy,
        grounding_policy: L1ArgumentGroundingPolicy,
        repair_limit: int,
    ) -> None:
        super().__init__(("127.0.0.1", 0), _SchemaGatewayHandler)
        self.upstream_base_url = _loopback_base_url(upstream_base_url)
        self.model = model
        self.policy = policy
        self.grounding_policy = grounding_policy
        self.repair_limit = repair_limit
        self.lock = threading.Lock()
        self.contexts: dict[str, SchemaContext] = {}
        self.attempts: list[SchemaAttempt] = []
        self.stripped_visible_text_chunks = 0
        self.schema_constrained_calls = 0

    def bind(self, context: SchemaContext) -> None:
        with self.lock:
            self.contexts[context.candidate_digest] = context

    def resolve(self, digest: str) -> SchemaContext | None:
        with self.lock:
            return self.contexts.get(digest)

    def append(self, attempt: SchemaAttempt) -> None:
        with self.lock:
            self.attempts.append(attempt)

    def snapshot(self) -> int:
        with self.lock:
            return len(self.attempts)

    def since(self, index: int) -> tuple[SchemaAttempt, ...]:
        with self.lock:
            return tuple(self.attempts[index:])

    def record_stripped_text(self, count: int) -> None:
        with self.lock:
            self.stripped_visible_text_chunks += count

    def record_schema_constrained_call(self) -> None:
        with self.lock:
            self.schema_constrained_calls += 1


class _SchemaGatewayHandler(http.server.BaseHTTPRequestHandler):
    server: _SchemaGatewayServer
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
        self._send(status, json.dumps({"error": {"message": message}}).encode(), "application/json")

    def _upstream(self, payload: dict[str, Any]) -> tuple[bytes, str, float]:
        request = urllib.request.Request(
            self.server.upstream_base_url + "/v1/chat/completions",
            data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": self.headers.get("Authorization", "Bearer local-loopback"),
            },
            method="POST",
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=900) as response:
            raw = response.read(MAX_SCHEMA_BODY + 1)
            if len(raw) > MAX_SCHEMA_BODY:
                raise ValueError("C3 model response exceeds gateway limit")
            return raw, response.headers.get("Content-Type", "text/event-stream"), (
                time.perf_counter() - started
            ) * 1000

    def do_POST(self) -> None:  # noqa: N802 - stdlib API
        if urllib.parse.urlparse(self.path).path != "/v1/chat/completions":
            self._reject(404, "C3 gateway exposes chat completions only")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 1 <= length <= MAX_SCHEMA_BODY:
                raise ValueError("request body is empty or too large")
            payload = json.loads(self.rfile.read(length).decode("utf-8", "strict"))
            request_object = _strict_object(payload, "C3 request")
            if request_object.get("model") != self.server.model:
                raise ValueError("request model differs from reviewed model")
            messages = request_object.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError("request messages are absent")
        except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            self._reject(400, f"C3 request rejected: {type(error).__name__}")
            return

        last = next((item for item in reversed(messages) if isinstance(item, dict)), {})
        if last.get("role") == "tool":
            try:
                receipt = json.loads(last.get("content", ""))
            except (TypeError, json.JSONDecodeError):
                receipt = None
            accepted = bool(
                isinstance(receipt, dict)
                and receipt.get("accepted") is True
                and receipt.get("contract") == SCHEMA_CONTROLLER_CONTRACT
            )
            self._send(200, _synthetic_text(
                self.server.model, "proposal captured" if accepted else "capture rejected",
            ), "text/event-stream")
            return

        try:
            digest, prompt = parse_schema_context(messages)
            context = self.server.resolve(digest)
            if context is None or context.prompt != prompt or context.candidate_digest != digest:
                raise ValueError("request is not bound to the active candidate contract")
            if not validate_tool_surface(request_object.get("tools"), context.candidates):
                raise ValueError("dynamic Tool surface differs from the candidate contract")
            verdict = self.server.policy.classify(prompt)
        except (TypeError, ValueError) as error:
            self._reject(400, f"C3 binding rejected: {type(error).__name__}")
            return

        request_digest = sha256_json({
            "model": self.server.model,
            "messages": messages,
            "tools": request_object.get("tools"),
            "candidate_contract": digest,
        })
        last_error = "CandidateToolMissingOrMultiple"
        try:
            for attempt_index in range(self.server.repair_limit + 1):
                current = dict(request_object)
                current["tool_choice"] = "required"
                current["parallel_tool_calls"] = False
                if attempt_index:
                    current["messages"] = [
                        *messages,
                        {"role": "system", "content": (
                            "C3_PROTOCOL_REPAIR: the hidden proposal failed the reviewed "
                            f"candidate Schema/Guard contract ({last_error}). Call exactly one "
                            "declared Tool. For a candidate Tool, keep only explicit values "
                            "under its declared keys and omit missing fields. Emit no text."
                        )},
                    ]
                raw, content_type, elapsed_ms = self._upstream(current)
                raw_parsed = parse_model_attempt(raw, content_type)
                parsed, schema_dropped = constrain_attempt_to_candidate_schema(
                    raw_parsed, context.candidates,
                )
                valid, error_type, decision, candidate_index, grounding = validate_model_attempt(
                    parsed, context.candidates, verdict, context.prompt,
                    self.server.grounding_policy,
                )
                self.server.append(SchemaAttempt(
                    sequence=self.server.snapshot() + 1,
                    scenario_id=context.scenario_id,
                    request_digest=request_digest,
                    response_digest=parsed.response_digest,
                    policy_action=verdict.action,
                    policy_reason=verdict.reason_code,
                    tool_name=parsed.tool_name,
                    candidate_index=candidate_index,
                    candidate_action=decision.action.value if decision else None,
                    candidate_target=decision.target if decision else None,
                    valid=valid,
                    error_type=error_type,
                    argument_fields=tuple(sorted(raw_parsed.arguments)) if raw_parsed.arguments else (),
                    business_argument_fields=tuple(sorted(parsed.arguments)) if parsed.arguments else (),
                    dropped_argument_fields=grounding.dropped_fields if grounding else (),
                    normalized_argument_fields=grounding.normalized_fields if grounding else (),
                    schema_dropped_argument_fields=schema_dropped,
                    input_tokens=parsed.input_tokens,
                    output_tokens=parsed.output_tokens,
                    usage_present=parsed.usage_present,
                    elapsed_ms=round(elapsed_ms, 3),
                ))
                if valid:
                    if schema_dropped:
                        sanitized = _proposal_tool_call(
                            self.server.model,
                            parsed.tool_name or "",
                            parsed.arguments or {},
                            "netopyu-c3-schema-call",
                        )
                        self.server.record_schema_constrained_call()
                    else:
                        sanitized, stripped = sanitize_valid_tool_response(raw, content_type)
                        self.server.record_stripped_text(stripped)
                    self._send(200, sanitized, content_type)
                    return
                last_error = error_type or "CandidateProtocolInvalid"
                if verdict.action != "allow":
                    break

            expected_tool = {
                "refuse": "refuse_l1_request",
                "out_of_scope": "reject_l1_out_of_scope",
            }.get(verdict.action)
            if expected_tool:
                raw = _synthetic_tool_call(self.server.model, expected_tool, verdict.reason_code)
                self.server.append(SchemaAttempt(
                    sequence=self.server.snapshot() + 1,
                    scenario_id=context.scenario_id,
                    request_digest=request_digest,
                    response_digest=sha256_json({"synthetic": expected_tool, "reason": verdict.reason_code}),
                    policy_action=verdict.action,
                    policy_reason=verdict.reason_code,
                    tool_name=expected_tool,
                    candidate_index=None,
                    candidate_action=verdict.action,
                    candidate_target=None,
                    valid=True,
                    error_type=None,
                    argument_fields=(),
                    business_argument_fields=(),
                    dropped_argument_fields=(),
                    normalized_argument_fields=(),
                    schema_dropped_argument_fields=(),
                    input_tokens=0,
                    output_tokens=0,
                    usage_present=True,
                    elapsed_ms=0.0,
                    synthetic=True,
                ))
                self._send(200, raw, "text/event-stream")
                return
            self._send(200, _synthetic_text(self.server.model, "proposal rejected"), "text/event-stream")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (OSError, TimeoutError, ValueError, urllib.error.URLError) as error:
            try:
                self._reject(502, f"C3 local model upstream failed: {type(error).__name__}")
            except (BrokenPipeError, ConnectionResetError):
                pass


class L1SchemaGateway:
    """Managed C3 gateway with dynamic-contract binding and attempt evidence."""

    def __init__(
        self,
        upstream_base_url: str,
        model: str,
        policy: L1GuardPolicy,
        grounding_policy: L1ArgumentGroundingPolicy,
        *,
        repair_limit: int = 1,
    ) -> None:
        if not 0 <= repair_limit <= 3:
            raise ValueError("C3 repair limit must be between 0 and 3")
        self.server = _SchemaGatewayServer(
            upstream_base_url, model, policy, grounding_policy, repair_limit,
        )
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            name="netopyu-l1-schema-gateway",
            daemon=True,
        )
        self.thread.start()
        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}"

    def bind(
        self,
        scenario_id: str,
        prompt: str,
        candidates: tuple[L1CatalogEntry, ...],
    ) -> SchemaContext:
        context = SchemaContext(
            scenario_id=scenario_id,
            prompt=prompt,
            candidates=candidates,
            candidate_digest=candidate_contract_digest(candidates),
        )
        self.server.bind(context)
        return context

    def snapshot(self) -> int:
        return self.server.snapshot()

    def since(self, index: int) -> tuple[SchemaAttempt, ...]:
        return self.server.since(index)

    def metrics(self) -> dict[str, Any]:
        attempts = self.since(0)
        actual = [item for item in attempts if not item.synthetic]
        return {
            "contract": SCHEMA_GATEWAY_CONTRACT,
            "model_attempts": len(actual),
            "valid_model_attempts": sum(item.valid for item in actual),
            "synthetic_safe_calls": sum(item.synthetic for item in attempts),
            "usage_complete": bool(actual) and all(item.usage_present for item in actual),
            "input_tokens": sum(item.input_tokens for item in actual),
            "output_tokens": sum(item.output_tokens for item in actual),
            "dropped_argument_fields": sum(len(item.dropped_argument_fields) for item in actual),
            "normalized_argument_fields": sum(len(item.normalized_argument_fields) for item in actual),
            "schema_dropped_argument_fields": sum(
                len(item.schema_dropped_argument_fields) for item in actual
            ),
            "schema_constrained_calls": self.server.schema_constrained_calls,
            "stripped_visible_text_chunks": self.server.stripped_visible_text_chunks,
            "upstream_elapsed_ms": round(sum(item.elapsed_ms for item in actual), 3),
            "failures": {
                error: sum(item.error_type == error for item in actual)
                for error in sorted({item.error_type for item in actual if item.error_type})
            },
            "failure_shapes": {
                "|".join((
                    item.error_type or "none",
                    item.tool_name or "none",
                    ",".join(item.argument_fields),
                    ",".join(item.business_argument_fields),
                )): sum(
                    other.error_type == item.error_type
                    and other.tool_name == item.tool_name
                    and other.argument_fields == item.argument_fields
                    and other.business_argument_fields == item.business_argument_fields
                    for other in actual
                )
                for item in actual if item.error_type
            },
        }

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def attempts_as_dict(attempts: tuple[SchemaAttempt, ...]) -> list[dict[str, Any]]:
    return [asdict(item) for item in attempts]


__all__ = [
    "L1SchemaGateway",
    "SCHEMA_CONTROLLER_CONTRACT",
    "SCHEMA_GATEWAY_CONTRACT",
    "SchemaAttempt",
    "SchemaContext",
    "attempts_as_dict",
    "candidate_contract",
    "candidate_contract_digest",
    "candidate_tool_names",
    "compile_schema_decision",
    "constrain_attempt_to_candidate_schema",
    "parse_schema_context",
    "schema_decision_prompt",
    "sanitize_valid_tool_response",
    "validate_model_attempt",
    "validate_tool_surface",
]
