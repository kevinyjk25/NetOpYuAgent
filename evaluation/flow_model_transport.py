"""Explicit, tool-free model transports for a common-JSON diagnostic experiment."""

from __future__ import annotations

import json
import os
import time

import httpx

GPT_MODEL = "gpt-5.5-2026-04-23"
QWEN_MODEL = "qwen3.5:9b"
MAX_OUTPUT = 8192
ENDPOINTS = {
    "openai": "https://api.openai.com/v1/responses",
    "ollama": "http://127.0.0.1:11434/api/chat",
}


def common_messages(original: dict) -> list[dict]:
    """Same full schema in both prompts; do not rewrite the acceptance schema."""
    messages = json.loads(json.dumps(original["messages"]))
    payload = json.loads(messages[-1]["content"])
    payload["requiredOutputSchema"] = original["format"]
    messages[-1]["content"] = json.dumps(payload, ensure_ascii=False)
    return messages


def wire_request(original: dict, arm: str, repeat: int) -> dict:
    messages = common_messages(original)
    if arm == "openai":
        return dict(model=GPT_MODEL, input=messages, store=False,
                    reasoning={"effort": "high"}, max_output_tokens=MAX_OUTPUT,
                    text={"format": {"type": "json_object"}})
    if arm != "ollama":
        raise ValueError("unknown model arm")
    return dict(model=QWEN_MODEL, messages=messages, stream=False, think=False, format="json",
                options=dict(temperature=0, seed=20260908 + repeat, num_ctx=16384, num_predict=MAX_OUTPUT))


def check_credentials(arm: str) -> None:
    if arm not in ENDPOINTS:
        raise ValueError("unknown model arm")
    if arm == "openai" and not os.environ.get("OPENAI_API_KEY", "").strip():
        raise ValueError("OPENAI_API_KEY is not configured; no request sent")


def send(arm: str, wire: dict) -> dict:
    check_credentials(arm)
    headers = {"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]} if arm == "openai" else {}
    started = time.monotonic()
    try:
        # Never follow redirects or send credentials to an environment-selected proxy/URL.
        with httpx.Client(timeout=360, trust_env=False, follow_redirects=False) as client:
            response = client.post(ENDPOINTS[arm], headers=headers, json=wire)
        body = response.text
        if arm == "openai":
            body = body.replace(os.environ["OPENAI_API_KEY"], "[REDACTED]")
        result = dict(httpStatus=response.status_code, body=body)
    except httpx.HTTPError as error:
        # Exception strings can contain request details; do not persist them or retry.
        result = dict(httpStatus=None, body=type(error).__name__)
    return {**result, "latencyMs": (time.monotonic() - started) * 1000}


def decode(arm: str, envelope: dict) -> tuple[str | None, dict]:
    cost = dict(latencyMs=envelope["latencyMs"], inputTokens=None, outputTokens=None, reasoningTokens=None)
    if envelope["httpStatus"] != 200:
        return None, {**cost, "status": "transport_error", "httpStatus": envelope["httpStatus"]}
    try:
        raw = json.loads(envelope["body"])
        if arm == "openai":
            usage = raw.get("usage") or {}
            cost.update(inputTokens=usage.get("input_tokens"), outputTokens=usage.get("output_tokens"),
                        reasoningTokens=(usage.get("output_tokens_details") or {}).get("reasoning_tokens"),
                        reportedModel=raw.get("model"), doneReason=raw.get("status"))
            if raw.get("model") != GPT_MODEL:
                return None, {**cost, "status": "model_identity_mismatch"}
            if raw.get("status") != "completed":
                return None, {**cost, "status": "incomplete_response"}
            output = raw["output"]
            if any(item.get("type") not in {"reasoning", "message"} for item in output):
                return None, {**cost, "status": "unexpected_tool_output"}
            messages = [item for item in output if item.get("type") == "message"]
            content = [part for item in messages for part in item.get("content", [])]
            if any(part.get("type") == "refusal" for part in content):
                return None, {**cost, "status": "model_refusal"}
            if (len(messages) != 1 or messages[0].get("role") != "assistant" or
                    len(content) != 1 or content[0].get("type") != "output_text"):
                raise ValueError("ambiguous output")
            text = content[0]["text"]
        else:
            cost.update(inputTokens=raw.get("prompt_eval_count"), outputTokens=raw.get("eval_count"),
                        reportedModel=raw.get("model"), doneReason=raw.get("done_reason"))
            if raw.get("model") != QWEN_MODEL:
                return None, {**cost, "status": "model_identity_mismatch"}
            if raw.get("done") is not True or raw.get("done_reason") != "stop":
                return None, {**cost, "status": "incomplete_response"}
            if raw["message"].get("tool_calls"):
                return None, {**cost, "status": "unexpected_tool_output"}
            text = raw["message"]["content"]
        if not isinstance(text, str):
            raise ValueError("missing text")
        for name in ("inputTokens", "outputTokens", "reasoningTokens"):
            value = cost[name]
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("invalid token usage")
        return text, {**cost, "status": "text_received"}
    except (ValueError, KeyError, TypeError, AttributeError):
        return None, {**cost, "status": "malformed_response"}
