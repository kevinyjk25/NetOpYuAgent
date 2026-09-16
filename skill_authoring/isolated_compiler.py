"""One tool-free author request; invocation values and host observations stay out.

This proposes an AST, never executes it or grants authority. Admission is still
the original compiler/runtime. Unknown requests are recorded and never retried.
"""
from __future__ import annotations

import time

import httpx
from jsonschema import Draft202012Validator, ValidationError

from . import compiler, delivery
from .artifacts import write_artifacts
from .contracts import budget, seal

ENDPOINT = "http://127.0.0.1:11434"


class ResponseRejected(ValueError):
    """A completed model response, not an uncertain transport outcome."""


def make_request(packet, visible):
    """Exact author schema at both decoder and admission, without coercion."""
    wire = compiler.make_request(packet, visible)
    wire["format"] = compiler.author_response_schema(packet, visible)
    Draft202012Validator.check_schema(wire["format"])
    return wire


def invoke(packet, visible, folder):
    wire = make_request(packet, visible)
    if not budget(wire)["accepted"]:
        raise ValueError("complete compiler source exceeds budget; no truncation")
    write_artifacts(folder / "input", {"request.json": wire})
    cost = {"role": "isolated_compiler", "model": compiler.MODEL,
            "inputTokens": None, "outputTokens": None, "physicalCallAttempted": False,
            "status": "not_called", "semanticApproval": False}
    began = time.monotonic()
    try:
        with httpx.Client(timeout=180, trust_env=False) as client:
            tags = client.get(ENDPOINT + "/api/tags")
            tags.raise_for_status()
            models = [m for m in tags.json()["models"] if m["name"] == compiler.MODEL]
            if len(models) != 1 or not models[0].get("digest"):
                raise ValueError("exact local compiler model unavailable")
            write_artifacts(folder / "identity", {"model.json": {"name": compiler.MODEL, "digest": models[0]["digest"]}})
            cost.update(physicalCallAttempted=True, status="outcome_unknown")
            response = client.post(ENDPOINT + "/api/chat", json=wire)
            response.raise_for_status()
            envelope = response.json()
            write_artifacts(folder / "response", {"envelope.json": envelope})
            cost.update(inputTokens=envelope.get("prompt_eval_count"), outputTokens=envelope.get("eval_count"))
            if envelope.get("model") != compiler.MODEL or envelope.get("done") is not True:
                raise ValueError("compiler identity/completion not established")
            try:
                if envelope.get("done_reason") != "stop":
                    raise ValueError("incomplete compiler output")
                # Strict JSON object, duplicate keys rejected; no coercion.
                candidate = delivery.decode_response(envelope["message"]["content"])
                Draft202012Validator(wire["format"]).validate(candidate)
            except ValidationError as error:
                cost["status"] = "completed_response_rejected"
                write_artifacts(folder / "rejection", {"report.json": seal({
                    "code": "author_schema_mismatch", "instancePath": error.json_path,
                    "schemaPath": list(error.absolute_schema_path), "semanticApproval": False})})
                raise ResponseRejected("author_schema_mismatch at " + error.json_path) from error
            except (ValueError, TypeError, KeyError) as error:
                cost["status"] = "completed_response_rejected"
                raise ResponseRejected(type(error).__name__) from error
            write_artifacts(folder / "candidate", {"candidate.json": candidate})
            cost["status"] = "candidate_not_admitted"
            return candidate
    except Exception as error:
        cost["errorType"] = type(error).__name__
        raise
    finally:
        cost["latencyMs"] = (time.monotonic() - began) * 1000
        write_artifacts(folder / "cost", {"report.json": seal(cost)})
