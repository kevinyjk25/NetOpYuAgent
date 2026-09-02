"""OpenAI-compatible, candidate-Schema-only model adapter for L1 selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from .catalog import CapabilityCard


@dataclass(frozen=True)
class SelectionAttempt:
    tool_name: str
    arguments: dict[str, Any]
    input_tokens: int
    output_tokens: int


class SelectionProtocolError(ValueError):
    """Sanitized protocol failure carrying only bounded usage metadata."""

    def __init__(
        self,
        code: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        usage_complete: bool = False,
    ) -> None:
        super().__init__(code)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.usage_complete = usage_complete


def _usage(payload: Any) -> tuple[int, int, bool]:
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        return 0, 0, False
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if (
        not isinstance(prompt_tokens, int)
        or isinstance(prompt_tokens, bool)
        or prompt_tokens < 0
        or not isinstance(completion_tokens, int)
        or isinstance(completion_tokens, bool)
        or completion_tokens < 0
    ):
        return 0, 0, False
    return prompt_tokens, completion_tokens, True


class SelectionClient(Protocol):
    model: str

    async def select(
        self,
        prompt: str,
        candidates: tuple[CapabilityCard, ...],
        candidate_contract_digest: str,
        *,
        repair_reason: str | None = None,
    ) -> SelectionAttempt: ...


def candidate_tool_name(index: int) -> str:
    if not 0 <= index <= 11:
        raise ValueError("production L1 candidate index is outside 0..11")
    return f"select_candidate_{index:02d}"


def candidate_tools(candidates: tuple[CapabilityCard, ...]) -> list[dict[str, Any]]:
    if not 1 <= len(candidates) <= 12:
        raise ValueError("production L1 requires 1..12 candidates")
    tools: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        tools.append({
            "type": "function",
            "function": {
                "name": candidate_tool_name(index),
                "description": (
                    f"Select only {candidate.identity}. {candidate.description} "
                    "Supply only values explicitly present in USER_REQUEST; omit missing values."
                ),
                "parameters": {
                    "type": "object",
                    "properties": candidate.parameter_schemas,
                    "additionalProperties": False,
                },
            },
        })
    for name, description in (
        (
            "refuse_l1_request",
            "Select only when the direct request is unsafe or asks to bypass controls.",
        ),
        (
            "reject_l1_out_of_scope",
            "Select only when the request is unrelated to network or service operations.",
        ),
    ):
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object", "properties": {},
                    "additionalProperties": False,
                },
            },
        })
    return tools


class OpenAISelectionClient:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("production L1 selection model is required")
        if not 1 <= timeout_seconds <= 600:
            raise ValueError("production L1 model timeout must be 1..600 seconds")
        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    async def select(
        self,
        prompt: str,
        candidates: tuple[CapabilityCard, ...],
        candidate_contract_digest: str,
        *,
        repair_reason: str | None = None,
    ) -> SelectionAttempt:
        messages: list[dict[str, str]] = [{
            "role": "system",
            "content": (
                "You are the NetOpYu L1 proposal selector. Call exactly one supplied function. "
                "Never invent a capability, field, identifier, reason, or value. Omit any business "
                "argument that is not explicit in USER_REQUEST. This is proposal-only: do not execute "
                "tools and do not claim success."
            ),
        }]
        if repair_reason:
            messages.append({
                "role": "system",
                "content": (
                    "The previous proposal was rejected by the deterministic protocol: "
                    f"{repair_reason}. Produce one corrected function call."
                ),
            })
        messages.append({
            "role": "user",
            "content": (
                f"CANDIDATE_CONTRACT_DIGEST={candidate_contract_digest}\n"
                "USER_REQUEST="
                + json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))
            ),
        })
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        body = {
            "model": self.model,
            "messages": messages,
            "tools": candidate_tools(candidates),
            "tool_choice": "required",
            "temperature": 0,
            "stream": False,
        }
        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            transport=self.transport,
            trust_env=False,
        ) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions", headers=headers, json=body,
            )
        if len(response.content) > 4_000_000:
            raise ValueError("production L1 model response exceeds 4 MB")
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError as error:
            raise SelectionProtocolError("CandidateResponseInvalidJSON") from error
        input_tokens, output_tokens, usage_complete = _usage(payload)

        def protocol_error(code: str) -> SelectionProtocolError:
            return SelectionProtocolError(
                code,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                usage_complete=usage_complete,
            )

        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not isinstance(choices, list) or len(choices) != 1:
            raise protocol_error("CandidateChoiceMissingOrMultiple")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
        if not isinstance(tool_calls, list) or len(tool_calls) != 1:
            raise protocol_error("CandidateToolMissingOrMultiple")
        function = tool_calls[0].get("function") if isinstance(tool_calls[0], dict) else None
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise protocol_error("CandidateToolInvalid")
        raw_arguments = function.get("arguments", "{}")
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as error:
                raise protocol_error("CandidateArgumentsInvalid") from error
        else:
            arguments = raw_arguments
        if not isinstance(arguments, dict):
            raise protocol_error("CandidateArgumentsInvalid")
        return SelectionAttempt(
            tool_name=function["name"],
            arguments=arguments,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


__all__ = [
    "OpenAISelectionClient",
    "SelectionAttempt",
    "SelectionClient",
    "SelectionProtocolError",
    "candidate_tool_name",
    "candidate_tools",
]
