"""Run and inspect a one-task, source-cited 9B review. No Skill code is executed."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import time
from pathlib import Path
from typing import Any, Protocol

import httpx

from evaluation.translation_case_authoring import MODEL, OllamaAnchoredAuthorAdapter
from evaluation.translation_source_alignment import (
    AUTHORITY, SYSTEM_PROMPT, SourceAssessment, evaluate_source_assessment,
    implementation_digest, load_source_alignment, prepare_source_alignment,
)
from network_runtime.contracts import sha256_json


class SourceReviewAdapter(Protocol):
    def preflight(self) -> dict[str, str]: ...
    def review(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]: ...


class OllamaSourceReviewAdapter:
    def preflight(self) -> dict[str, str]:
        return OllamaAnchoredAuthorAdapter().preflight()

    def review(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        started = time.monotonic()
        with httpx.Client(timeout=240) as client:
            response = client.post("http://127.0.0.1:11434/api/chat", json={
                "model": MODEL, "stream": False, "think": False,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                "format": SourceAssessment.model_json_schema(),
                "options": {"temperature": 0, "seed": 20260904, "num_ctx": 16384, "num_predict": 5000},
            })
            response.raise_for_status()
            body = response.json()
        return str(body["message"]["content"]), {
            "modelCalls": 1, "latencyMs": (time.monotonic() - started) * 1000,
            "inputTokens": body.get("prompt_eval_count"), "outputTokens": body.get("eval_count"),
        }


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _digest() -> str:
    return sha256_json({
        "alignment": implementation_digest(),
        "runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })


def _assessment_report(payload: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    if response["transportError"] is not None:
        return {"status": "protocol_failed", "error": response["transportError"], "runtimeAuthorityGranted": False}
    try:
        assessment = SourceAssessment.model_validate_json(response["raw"])
        return evaluate_source_assessment(payload, assessment)
    except ValueError as exc:
        return {"status": "protocol_failed", "error": str(exc), "runtimeAuthorityGranted": False}


def inspect_source_review(input_root: Path, output: Path) -> dict[str, Any]:
    payload, manifest = load_source_alignment(input_root)
    expected_files = {"run.json", "response.json", "report.json", "seal.json"}
    if {p.name for p in output.iterdir()} != expected_files:
        raise ValueError("source review output inventory mismatch or incomplete run")
    seal = json.loads((output / "seal.json").read_text())
    if seal.get("sealDigest") != sha256_json({key: value for key, value in seal.items() if key != "sealDigest"}):
        raise ValueError("source review seal mismatch")
    values = {name: json.loads((output / name).read_text()) for name in expected_files - {"seal.json"}}
    if seal["files"] != {name: sha256_json(value) for name, value in values.items()}:
        raise ValueError("source review sealed file drift")
    run, report = values["run.json"], values["report.json"]
    if run["sourceManifestDigest"] != manifest["manifestDigest"] or run["inputDigest"] != payload["inputDigest"]:
        raise ValueError("source review input binding drift")
    core = _assessment_report(payload, values["response.json"])
    if report != _bound_report(core, run, values["response.json"]):
        raise ValueError("source review deterministic report drift")
    return {**report, "verified": True, "implementationDrift": run["implementationDigest"] != _digest()}


def _bound_report(core: dict[str, Any], run: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    body = {
        **core, "sourcePacketDigest": run["sourcePacketDigest"],
        "sourceManifestDigest": run["sourceManifestDigest"],
        "runDigest": sha256_json(run), "responseDigest": sha256_json(response),
        "model": run["model"], "telemetry": response["telemetry"],
        "humanIndependentEvidence": False, "goldAuthored": False,
        "runtimeOrDshExecuted": False, "thirdPartyExecutionAttempted": False,
        "authority": AUTHORITY,
    }
    return {**body, "reportDigest": sha256_json(body)}


def run_source_review(
    input_root: Path, output: Path, *, adapter: SourceReviewAdapter | None = None,
) -> dict[str, Any]:
    input_root, output = input_root.resolve(), output.resolve()
    payload, manifest = load_source_alignment(input_root)
    if output.is_relative_to(input_root):
        raise ValueError("source review must not modify its sealed inputs")
    if output.exists():
        # A completed run is immutable; an interrupted run is not silently replayed.
        return inspect_source_review(input_root, output)
    reviewer = adapter or OllamaSourceReviewAdapter()
    run = {
        "sourceManifestDigest": manifest["manifestDigest"],
        "sourcePacketDigest": manifest["sourcePacketDigest"],
        "inputDigest": payload["inputDigest"], "implementationDigest": _digest(),
        "model": reviewer.preflight(), "authority": AUTHORITY,
    }
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "run.json", run)
    try:
        raw, telemetry = reviewer.review(payload)
        response = {"raw": raw, "telemetry": telemetry, "transportError": None}
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        response = {"raw": "", "telemetry": {"modelCalls": 1}, "transportError": str(exc)}
    _write(output / "response.json", response)
    report = _bound_report(_assessment_report(payload, response), run, response)
    _write(output / "report.json", report)
    body = {"files": {
        name: sha256_json(value) for name, value in (
            ("run.json", run), ("response.json", response), ("report.json", report),
        )
    }}
    _write(output / "seal.json", {**body, "sealDigest": sha256_json(body)})
    return inspect_source_review(input_root, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("authoring_root", type=Path)
    prepare.add_argument("corpus_root", type=Path)
    prepare.add_argument("--case-id", required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    for command in ("run", "inspect"):
        run = sub.add_parser(command)
        run.add_argument("input_root", type=Path)
        run.add_argument("output_root", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_source_alignment(
            args.authoring_root, args.corpus_root, args.output_root,
            case_id=args.case_id, salt=secrets.token_hex(32),
        )
    elif args.command == "run":
        result = run_source_review(args.input_root, args.output_root)
    else:
        result = inspect_source_review(args.input_root, args.output_root)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
