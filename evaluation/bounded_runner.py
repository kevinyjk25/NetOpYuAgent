"""Pinned token-array runner codec; no transport, runner start, or generation.

These transformations do not grant authority. A mechanical HTTP fixture can
attest serialized bytes, not installed dirty-backend generation parity. The
host must verify assets, process identity, launch policy and budget separately.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

from network_runtime.contracts import sha256_json

OLLAMA_REVISION = "f96e7aa0513b9973a0ccc71be414c2ecb9d65b1a"
LLAMA_REVISION = "d222767c7a6516559a3f49e7721b6c6b1acc87b4"
PROFILE = "qwen35_offline_experimental_49152_v1"
MODEL = "qwen3.5:9b"
CONTEXT = 49152
VOCAB_SIZE = 248320
MAX_BYTES = 8 * 1024 * 1024
GRAMMAR_SHA256 = "fbd6501c635ec361e03529e98c0d18e6d0b7f8ac7d19650ee7f750c317679522"
PRESERVED_TOKENS = ["<think>", "</think>", "<tool_call>", "</tool_call>"]
RUNNER_STARTUP_CONTRACT = {
    "endpoint": "/completion", "host": "127.0.0.1", "n_ctx": CONTEXT,
    "parallel": 1, "context_shift": False, "warmup": False, "fit": "off",
    "cache_prompt": False, "stream": False, "generation_enabled": False,
    "live_generation_parity": "not_established",
    "required_flags": ["--offline", "--no-webui", "--no-context-shift", "--no-warmup",
                       "--fit", "off", "-c", str(CONTEXT), "-np", "1"],
}
_SAMPLING = {"num_keep": "n_keep", "temperature": "temperature", "top_k": "top_k",
    "top_p": "top_p", "min_p": "min_p", "repeat_penalty": "repeat_penalty",
    "repeat_last_n": "repeat_last_n", "frequency_penalty": "frequency_penalty",
    "presence_penalty": "presence_penalty", "typical_p": "typical_p", "seed": "seed"}
_PROCESS_OPTIONS = {"draft_num_predict": 4, "main_gpu": None, "num_batch": 512,
                    "num_gpu": -1, "num_thread": 0, "use_mmap": None}


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _prepared(value):
    # Delayed import avoids a broker/preflight/codec import cycle. The seal is
    # drift detection for a cooperative host, not a signature or permission.
    from evaluation.bounded_preflight import PreparedRequest
    from evaluation.bounded_transport import ModelBroker
    if not isinstance(value, dict):
        raise ValueError("sealed prepared object required")
    value = PreparedRequest(_json_bytes(value)).inspect()
    wire = value.get("wire")
    options = value.get("effective_options")
    identity = value.get("renderer_identity")
    tokens = value.get("token_ids")
    limit = value.get("output_reservation")
    if (not isinstance(wire, dict) or not isinstance(options, dict)
            or not isinstance(identity, dict) or identity.get("profile") != PROFILE
            or identity.get("ollama_revision") != OLLAMA_REVISION
            or identity.get("renderer") != "qwen3.5" or identity.get("parser") != "qwen3.5"):
        raise ValueError("pinned Qwen35 preparation profile required")
    normalized, source_limit = ModelBroker._wire(value["request_api"], value["source_request"])
    if (normalized != wire or source_limit != limit
            or value.get("source_request_digest") != sha256_json(value["source_request"])
            or value.get("wire_digest") != sha256_json(wire)
            or not isinstance(value.get("rendered_prompt"), str)
            or value.get("rendered_prompt_digest") != "sha256:" + hashlib.sha256(
                value["rendered_prompt"].encode("utf-8")).hexdigest()
            or value.get("token_ids_digest") != sha256_json(tokens)):
        raise ValueError("prepared request content binding drift")
    if (not isinstance(tokens, list) or not tokens
            or any(type(token) is not int or not 0 <= token < VOCAB_SIZE for token in tokens)
            or type(value.get("input_tokens")) is not int or value["input_tokens"] != len(tokens)
            or type(limit) is not int or not 1 <= limit <= 6000
            or type(options.get("num_ctx")) is not int or options["num_ctx"] != CONTEXT
            or type(options.get("num_predict")) is not int or options["num_predict"] != limit
            or len(tokens) + limit >= CONTEXT):
        raise ValueError("exact token array and non-truncating context reservation required")
    expected_options = set(_SAMPLING) | set(_PROCESS_OPTIONS) | {"num_ctx", "num_predict", "stop"}
    if set(options) != expected_options:
        raise ValueError("unsupported effective option set")
    for key, expected in _PROCESS_OPTIONS.items():
        if type(options[key]) is not type(expected) or options[key] != expected:
            raise ValueError("runner launch option differs from fixed experimental profile")
    for key in _SAMPLING:
        number = options[key]
        if type(number) not in (int, float) or not math.isfinite(number):
            raise ValueError("finite explicit sampling option required")
    for key in ("num_keep", "top_k", "repeat_last_n", "seed"):
        if type(options[key]) is not int:
            raise ValueError("integer sampling option required")
    if (not 0 <= options["num_keep"] <= len(tokens) or options["top_k"] < 0
            or options["repeat_last_n"] < -1 or options["seed"] < -1
            or options["temperature"] < 0 or options["repeat_penalty"] < 0
            or any(not 0 <= options[key] <= 1 for key in ("top_p", "min_p", "typical_p"))):
        raise ValueError("sampling option outside supported profile range")
    if any(type(options.get(key)) is not type(expected) or options[key] != expected
           for key, expected in wire["options"].items()):
        raise ValueError("explicit request option changed")
    stop = options["stop"]
    if stop is not None and (not isinstance(stop, list) or any(not isinstance(item, str) for item in stop)):
        raise ValueError("stop must be null or string array")
    return value


def serialize_completion(prepared_value):
    """Return the exact native /completion body. Never render or tokenize again."""
    value = _prepared(prepared_value)
    options = value["effective_options"]
    body = {"prompt": list(value["token_ids"]), "stream": False, "cache_prompt": False,
            "n_predict": value["output_reservation"], "preserved_tokens": list(PRESERVED_TOKENS),
            **{destination: options[source] for source, destination in _SAMPLING.items()}}
    if options["stop"] is not None:
        body["stop"] = list(options["stop"])
    if "format" in value["wire"]:
        output_format = value["wire"]["format"]
        if output_format == "json":
            grammar = Path(__file__).with_name("runner_codec").joinpath("grammar_json.gbnf").read_bytes()
            if hashlib.sha256(grammar).hexdigest() != GRAMMAR_SHA256:
                raise ValueError("pinned upstream JSON grammar drift")
            body["grammar"] = grammar.decode("utf-8")
        elif isinstance(output_format, dict):
            body["json_schema"] = copy.deepcopy(output_format)
        else:
            raise ValueError("unsupported decoder format")
    return body


def validate_runner_props(props, model_path):
    """Necessary /props check only; cannot establish process flags or authority."""
    if (not isinstance(props, dict) or type(props.get("total_slots")) is not int
            or props["total_slots"] != 1 or props.get("model_path") != str(model_path)
            or not isinstance(props.get("default_generation_settings"), dict)
            or type(props["default_generation_settings"].get("n_ctx")) is not int
            or props["default_generation_settings"]["n_ctx"] != CONTEXT):
        raise ValueError("runner props differ from pinned single-slot context profile")
    return {"props_match": True, "generation_enabled": False, "launch_flags_attested": False}


def _parse_text(value, content, codec_path, timeout):
    from evaluation.bounded_transport import strict_json
    codec_path = Path(codec_path)
    if not codec_path.is_absolute() or codec_path.is_symlink() or not codec_path.is_file():
        raise ValueError("host-pinned absolute regular parser executable required")
    payload = _json_bytes({"wire": value["wire"], "content": content})
    if len(payload) > MAX_BYTES:
        raise ValueError("bounded parser input required")
    try:
        result = subprocess.run([str(codec_path)], input=payload, capture_output=True, check=False,
                                timeout=timeout, env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"})
    except subprocess.TimeoutExpired as error:
        raise TimeoutError("offline result parser deadline exhausted") from error
    if result.returncode or len(result.stdout) > MAX_BYTES:
        raise ValueError("official result parser rejected completion")
    parsed = strict_json(result.stdout)
    if (not isinstance(parsed, dict)
            or set(parsed) != {"message", "identity", "preserved_tokens", "no_generation"}
            or parsed["no_generation"] is not True or parsed["preserved_tokens"] != PRESERVED_TOKENS
            or parsed["identity"] != {"ollama_revision": OLLAMA_REVISION, "parser": "qwen3.5",
                "api": "github.com/ollama/ollama/model/parsers.ParserForName"}
            or not isinstance(parsed["message"], dict) or parsed["message"].get("role") != "assistant"
            or not isinstance(parsed["message"].get("content"), str)):
        raise ValueError("official parser identity or envelope mismatch")
    return parsed["message"]


def parse_completion(prepared_value, response, codec_path, *, timeout=10):
    """Validate final native runner metrics and use the official Qwen35 parser.

    Missing, inconsistent, over-budget or truncated results are unusable. The
    caller retains/settles its reservation on any exception; this is no refund.
    """
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 120:
        raise ValueError("finite bounded parser timeout required")
    value = _prepared(prepared_value)
    if (not isinstance(response, dict) or "error" in response or response.get("stop") is not True
            or response.get("truncated") is not False or response.get("stop_type") not in {"eos", "word", "limit"}
            or not isinstance(response.get("content"), str) or not isinstance(response.get("timings"), dict)):
        raise ValueError("known successful non-truncated final runner response required")
    timings = response["timings"]
    for key in ("cache_n", "prompt_n", "predicted_n"):
        if type(timings.get(key)) is not int or timings[key] < 0:
            raise ValueError("known exact nonnegative runner usage required")
    input_count = timings["cache_n"] + timings["prompt_n"]
    output_count = timings["predicted_n"]
    if input_count != value["input_tokens"] or output_count > value["output_reservation"]:
        raise ValueError("runner usage differs from exact reserved request")
    for key, expected in (("tokens_evaluated", input_count), ("tokens_predicted", output_count)):
        if type(response.get(key)) is not int or response[key] != expected:
            raise ValueError("runner usage fields disagree")
    message = _parse_text(value, response["content"], codec_path, timeout)
    return {"model": MODEL, "message": message, "done": True,
            "done_reason": "length" if response["stop_type"] == "limit" else "stop",
            "prompt_eval_count": input_count, "eval_count": output_count}


def encode_fixture_completion(prepared_value, message):
    """TEST FIXTURE ONLY: synthetic counts/text, never an actual runner result.

    This is not used by serialize/parse_completion. The HTTP no-inference
    fixture uses it to exercise the same official parser; it grants no evidence
    about token usage, model quality, or generation parity.
    """
    value = _prepared(prepared_value)
    if (not isinstance(message, dict) or message.get("role", "assistant") != "assistant"
            or not isinstance(message.get("content", ""), str) or message.get("thinking")
            or set(message) - {"role", "content", "tool_calls", "thinking"}):
        raise ValueError("fixture assistant text/tool message required")
    content = message.get("content", "")
    tools = {tool["function"]["name"]: tool["function"] for tool in value["wire"].get("tools", [])}
    calls = message.get("tool_calls", [])
    if not isinstance(calls, list):
        raise ValueError("fixture tool calls must be array")
    for call in calls:
        function = call.get("function", {}) if isinstance(call, dict) else {}
        name, arguments = function.get("name"), function.get("arguments")
        if (name not in tools or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", name)
                or not isinstance(arguments, dict)):
            raise ValueError("fixture requires a declared tool and object arguments")
        parts = ["<tool_call>\n<function=" + name + ">"]
        for key, argument in arguments.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", key):
                raise ValueError("unsupported fixture parameter name")
            text = argument if isinstance(argument, str) else json.dumps(argument, ensure_ascii=False, allow_nan=False)
            if any(tag in text for tag in ("<tool_call", "</tool_call", "<function", "</function", "<parameter", "</parameter")):
                raise ValueError("fixture text contains ambiguous Qwen tool delimiters")
            parts.append("<parameter=" + key + ">" + text + "</parameter>")
        content += "\n".join(parts + ["</function>\n</tool_call>"])
    count = 1 if content else 0  # Deliberately synthetic, NOT a token estimator.
    return {"content": content, "stop": True, "stop_type": "eos", "truncated": False,
            "tokens_evaluated": value["input_tokens"], "tokens_predicted": count,
            "timings": {"cache_n": 0, "prompt_n": value["input_tokens"], "predicted_n": count},
            "_fixture_only": True}
