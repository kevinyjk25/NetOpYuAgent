"""Offline experimental request preparation; no server, generation or authority.

Host-pinned Go renderer and vocab-only C tokenizer run as bounded subprocesses.
The installed Ollama process is never contacted. A seal detects drift, not a
malicious operator who controls both artifacts and executables. This does not
prove equivalence to a dirty installed backend or future generation token use.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import time

from evaluation.bounded_pilot import checked_seal, seal, write_new
from evaluation.bounded_transport import MODEL, ModelBroker, strict_json
from network_runtime.contracts import sha256_json

ASSETS_SCHEMA = "ensuredskill.io/offline-preflight-assets/v1"
PREPARED_SCHEMA = "ensuredskill.io/offline-prepared-request/v1"
MAX_BYTES = 4 * 1024 * 1024
MAX_TOKENS = 262144


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _bytes_digest(value):
    return "sha256:" + hashlib.sha256(value).hexdigest()


def file_digest(path):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("absolute regular non-symlink host asset required")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True)
class Assets:
    """Immutable bytes, not a frozen wrapper around mutable dictionaries."""
    document: bytes

    @classmethod
    def load(cls, path):
        raw = Path(path).read_bytes()
        if len(raw) > MAX_BYTES:
            raise ValueError("asset manifest too large")
        value = checked_seal(strict_json(raw))
        assets = cls(_json_bytes(value))
        assets.verify()
        return assets

    def inspect(self):
        value = checked_seal(strict_json(self.document))
        if (set(value) != {"schema", "renderer", "tokenizer", "model", "dependencies", "metadata", "digest"}
                or value["schema"] != ASSETS_SCHEMA or not isinstance(value["metadata"], dict)
                or not isinstance(value["dependencies"], list)):
            raise ValueError("exact offline host asset manifest required")
        return value

    def verify(self):
        value = self.inspect()
        for pin in [value[name] for name in ("renderer", "tokenizer", "model")] + value["dependencies"]:
            if not isinstance(pin, dict) or set(pin) != {"path", "sha256"}:
                raise ValueError("exact host file pin required")
            if file_digest(pin["path"]) != pin["sha256"]:
                raise ValueError("host asset content drift")
        return value


def _run(args, payload, timeout):
    try:
        result = subprocess.run(args, input=payload, capture_output=True, timeout=timeout, check=False,
                                env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"})
    except subprocess.TimeoutExpired as error:
        raise TimeoutError("preflight helper exceeded preparation deadline") from error
    if result.returncode:
        # Do not leak an input prompt from a helper's diagnostic to callers.
        raise ValueError(f"preflight helper rejected input (exit {result.returncode})")
    if len(result.stdout) > 16 * MAX_BYTES:
        raise ValueError("preflight helper output too large")
    return strict_json(result.stdout)


@dataclass(frozen=True)
class PreparedRequest:
    document: bytes

    def inspect(self):
        value = checked_seal(strict_json(self.document))
        fixed = {"schema": PREPARED_SCHEMA, "model": MODEL, "generation_enabled": False,
                 "authority_granted": False, "live_adapter_ready": False, "actual_model_calls": 0,
                 "evidence_role": "offline_experimental_preflight_not_generation_parity",
                 "equivalent_to_installed_ollama": "not_established"}
        if any(type(value.get(key)) is not type(expected) or value[key] != expected
               for key, expected in fixed.items()):
            raise ValueError("offline prepared request grants no execution authority")
        flags = value.get("special_tokens")
        if (not isinstance(flags, dict) or set(flags) != {"add_special", "parse_special"}
                or any(flag is not True for flag in flags.values())):
            raise ValueError("prepared tokenizer flags drift")
        normalization = {"native": "native_identity_v1", "openai": "scripted_openai_native_text_v1"}
        if (value.get("request_api") not in normalization
                or value.get("normalization") != normalization[value["request_api"]]):
            raise ValueError("prepared normalization drift")
        return value

    def check_binding(self, assets, api, source_request):
        """Drift check only. There deliberately is no execute/generate method."""
        value = self.inspect()
        pins = assets.verify()
        wire, limit = ModelBroker._wire(api, source_request)
        if (value["assets_digest"] != pins["digest"] or value["request_api"] != api
                or value["source_request_digest"] != sha256_json(source_request)
                or value["source_request"] != source_request or value["wire"] != wire
                or value["wire_digest"] != sha256_json(wire)
                or value["rendered_prompt_digest"] != _bytes_digest(value["rendered_prompt"].encode("utf-8"))
                or value["token_ids_digest"] != sha256_json(value["token_ids"])
                or value["input_tokens"] != len(value["token_ids"])):
            raise ValueError("prepared request binding drift")
        _check_options(value["effective_options"], wire, limit)
        if (value["output_reservation"] != limit
                or value["input_tokens"] + limit > value["effective_options"]["num_ctx"]):
            raise ValueError("prepared context/output binding drift")
        return {"binding_matches": True, "authority_granted": False, "generation_enabled": False,
                "does_not_attest_backend_equivalence": True}


def _check_options(options, wire, output_limit):
    if (type(options.get("num_ctx")) is not int or options["num_ctx"] <= 0
            or type(options.get("num_predict")) is not int or options["num_predict"] != output_limit
            or any(options.get(key) != value for key, value in wire["options"].items())):
        raise ValueError("effective context/options must preserve explicit request controls")


def prepare(assets: Assets, api, source_request, *, timeout=30):
    if type(timeout) not in (int, float) or not 0 < timeout <= 120:
        raise ValueError("finite bounded preparation timeout required")
    began = time.monotonic()
    raw = _json_bytes(source_request)
    if len(raw) > MAX_BYTES:
        raise ValueError("bounded source request required")
    # Reuse the actual broker's normalization, including pi-ai text parts and
    # tool-call conversion. This does not create a broker or an execution arm.
    source = strict_json(raw)
    wire, output_limit = ModelBroker._wire(api, source)
    pins = assets.verify()
    remaining = timeout - (time.monotonic() - began)
    if remaining <= 0:
        raise TimeoutError("preparation deadline exhausted")
    rendered = _run([pins["renderer"]["path"]], _json_bytes(wire), remaining)
    if (not isinstance(rendered, dict)
            or set(rendered) != {"rendered_prompt", "effective_options", "identity", "no_generation"}
            or rendered["no_generation"] is not True or not isinstance(rendered["rendered_prompt"], str)
            or not isinstance(rendered["identity"], dict) or not isinstance(rendered["effective_options"], dict)):
        raise ValueError("invalid renderer preparation envelope")
    prompt = rendered["rendered_prompt"].encode("utf-8")
    options = rendered["effective_options"]
    if len(prompt) > MAX_BYTES:
        raise ValueError("bounded rendered prompt required")
    _check_options(options, wire, output_limit)
    remaining = timeout - (time.monotonic() - began)
    if remaining <= 0:
        raise TimeoutError("preparation deadline exhausted")
    tokens = _run([pins["tokenizer"]["path"], "--model", pins["model"]["path"]], prompt, remaining)
    if (not isinstance(tokens, dict) or set(tokens) != {"token_ids", "count", "add_special", "parse_special"}
            or tokens["add_special"] is not True or tokens["parse_special"] is not True
            or not isinstance(tokens["token_ids"], list) or type(tokens["count"]) is not int
            or tokens["count"] != len(tokens["token_ids"]) or not 0 < tokens["count"] <= MAX_TOKENS
            or any(type(item) is not int or item < 0 for item in tokens["token_ids"])):
        raise ValueError("invalid vocab-only tokenizer result")
    if tokens["count"] + output_limit > options["num_ctx"]:
        raise ValueError("context budget exceeded; no truncation or generation")
    if time.monotonic() - began >= timeout:
        raise TimeoutError("preparation deadline exhausted")
    assets.verify()
    if time.monotonic() - began >= timeout:
        raise TimeoutError("preparation deadline exhausted")
    prepared = seal({"schema": PREPARED_SCHEMA, "model": MODEL,
        "request_api": api, "source_request": source, "source_request_digest": sha256_json(source),
        "normalization": "scripted_openai_native_text_v1" if api == "openai" else "native_identity_v1",
        "wire": wire, "wire_digest": sha256_json(wire), "assets_digest": pins["digest"],
        "rendered_prompt": rendered["rendered_prompt"], "rendered_prompt_digest": _bytes_digest(prompt),
        "effective_options": options, "renderer_identity": rendered["identity"],
        "token_ids": tokens["token_ids"], "token_ids_digest": sha256_json(tokens["token_ids"]),
        "input_tokens": tokens["count"], "output_reservation": output_limit,
        "special_tokens": {"add_special": True, "parse_special": True},
        "evidence_role": "offline_experimental_preflight_not_generation_parity",
        "generation_enabled": False, "authority_granted": False, "live_adapter_ready": False,
        "actual_model_calls": 0, "equivalent_to_installed_ollama": "not_established"})
    encoded = _json_bytes(prepared)
    if time.monotonic() - began >= timeout:
        raise TimeoutError("preparation deadline exhausted during serialization")
    return PreparedRequest(encoded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path, help="host-created pinned assets, not Skill input")
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path, help="new JSON artifact; never overwritten")
    parser.add_argument("--api", choices=("native", "openai"), default="native")
    args = parser.parse_args()
    raw = args.request.read_bytes()
    if len(raw) > MAX_BYTES:
        raise ValueError("request too large")
    result = prepare(Assets.load(args.assets), args.api, strict_json(raw))
    write_new(args.output, result.inspect())
    print(json.dumps({"prepared": str(args.output), "input_tokens": result.inspect()["input_tokens"],
                      "generation_enabled": False, "actual_model_calls": 0}))


if __name__ == "__main__":
    main()
