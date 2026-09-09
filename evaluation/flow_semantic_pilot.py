"""Frozen semantic-flow development pilot; archived legacy protocols remain unchanged."""

from __future__ import annotations

import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path

import httpx
from jsonschema import ValidationError as SchemaError

from evaluation.flow_translation import FlowSources, _write
from evaluation.flow_translation_batch import fingerprint
from evaluation.flow_semantics import SemanticFlow, assess_semantic, compile_semantic, request as bounded_request, PROTOCOL, RULES
from evaluation.flow_tree_authoring import ROOT, digest_file, receipt, verify_receipt
from evaluation.read_l05_review import ReadL05Review
from evaluation.translation_case_authoring import OllamaAnchoredAuthorAdapter
from network_runtime.contracts import sha256_json


def environment() -> dict:
    return {name: version(name) for name in ("pydantic", "jsonschema", "httpx")}


def implementation() -> dict:
    paths = set(fingerprint()) | {"evaluation/flow_grounded_translation.py", "evaluation/flow_source_selection.py",
        "evaluation/flow_tree.py", "evaluation/flow_tree_authoring.py", "evaluation/flow_tree_capabilities.py",
        "evaluation/flow_tree_bounded_pilot.py", "evaluation/flow_semantics.py", "evaluation/flow_semantic_pilot.py"} | {row["path"] for row in RULES.values()}
    return {path: digest_file(ROOT / path) for path in sorted(paths)}


def freeze(parent: Path, root: Path) -> dict:
    previous = json.loads((parent / "manifest.json").read_text())
    if previous["manifestDigest"] != sha256_json({k: v for k, v in previous.items() if k != "manifestDigest"}):
        raise ValueError("parent manifest digest mismatch")
    rows = []
    for case in previous["cases"]:
        identifier = case["id"]
        if Path(identifier).name != identifier or identifier in {".", ".."} or identifier in {row["id"] for row in rows}:
            raise ValueError("invalid/duplicate case identifier")
        source = FlowSources.model_validate(case["sources"])
        rows.append({"id": identifier, "sources": source.model_dump(mode="json"), "wireRequest": bounded_request(source)})
    if not rows:
        raise ValueError("empty pilot")
    body = {"protocol": PROTOCOL, "parentManifestDigest": previous["manifestDigest"],
        "evidenceRole": "known_development_semantic_flow_not_holdout", "cases": rows,
        "model": OllamaAnchoredAuthorAdapter().preflight(), "environment": environment(),
        "implementation": implementation(), "attemptsPerCase": 1}
    manifest = {**body, "manifestDigest": sha256_json(body)}
    root.mkdir(parents=True, exist_ok=False)
    _write(root / "manifest.json", manifest)
    return manifest


def load(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest["manifestDigest"] != sha256_json({k: v for k, v in manifest.items() if k != "manifestDigest"})
            or manifest["protocol"] != PROTOCOL or manifest["implementation"] != implementation()
            or manifest["environment"] != environment()):
        raise ValueError("frozen manifest/implementation/environment drift")
    for case in manifest["cases"]:
        if case["wireRequest"] != bounded_request(FlowSources.model_validate(case["sources"])):
            raise ValueError("frozen request drift")
    return manifest


def derive(source: FlowSources, envelope: dict) -> tuple[dict, dict]:
    files, cost = {}, {"latencyMs": envelope["latencyMs"]}
    try:
        if envelope["httpStatus"] != 200:
            raise ValueError("non-200 model response")
        body = json.loads(envelope["body"])
        cost.update(inputTokens=body.get("prompt_eval_count"), outputTokens=body.get("eval_count"), doneReason=body.get("done_reason"))
        raw = json.loads(body["message"]["content"])
        tree = SemanticFlow.model_validate(raw)
        files["tree.json"] = tree.model_dump(mode="json")
        compiled = compile_semantic(source, tree)
        files.update({"compilation.json": compiled, "review-input.json": compiled["reviewInput"]})
        status = {"status": "awaiting_source_review", "inputDigest": compiled["reviewInput"]["inputDigest"]}
    except (ValueError, KeyError, TypeError, SchemaError) as error:
        status = {"status": "blocked", "errorType": type(error).__name__, "error": str(error)}
    return files, {**status, **cost}


def run(root: Path) -> None:
    manifest = load(root)
    for case in manifest["cases"]:
        folder = root / case["id"]
        if folder.exists():
            verify_receipt(folder)
            continue
        model = OllamaAnchoredAuthorAdapter().preflight()
        if model != manifest["model"]:
            raise ValueError("model identity drift")
        folder.mkdir()
        _write(folder / "sources.json", case["sources"])
        _write(folder / "request.json", {"wireRequest": case["wireRequest"], "model": model})
        start = time.monotonic()
        try:
            with httpx.Client(timeout=240, trust_env=False) as client:
                response = client.post("http://127.0.0.1:11434/api/chat", json=case["wireRequest"])
            envelope = {"httpStatus": response.status_code, "body": response.text, "latencyMs": (time.monotonic() - start) * 1000}
            _write(folder / "response.json", envelope)
            files, result = derive(FlowSources.model_validate(case["sources"]), envelope)
            for name, value in files.items():
                _write(folder / name, value)
        except Exception as error:
            result = {"status": "transport_or_storage_error", "errorType": type(error).__name__, "error": str(error)}
        _write(folder / "status.json", result)
        _write(folder / "receipt.json", receipt(folder))
        print(json.dumps({"case": case["id"], **result}), flush=True)


def report(root: Path, reviews: Path | None = None) -> dict:
    manifest = load(root)
    rows = []
    for case in manifest["cases"]:
        folder = root / case["id"]
        verify_receipt(folder)
        if (json.loads((folder / "sources.json").read_text()) != case["sources"] or
                json.loads((folder / "request.json").read_text()) != {"wireRequest": case["wireRequest"], "model": manifest["model"]}):
            raise ValueError("source/request/model drift")
        source = FlowSources.model_validate(case["sources"])
        files, result = derive(source, json.loads((folder / "response.json").read_text()))
        if (set(receipt(folder)) != {"sources.json", "request.json", "response.json", "status.json", *files}
                or json.loads((folder / "status.json").read_text()) != result
                or any(json.loads((folder / name).read_text()) != value for name, value in files.items())):
            raise ValueError("raw response derivation mismatch")
        row = {"case": case["id"], **result, "files": receipt(folder)}
        if result["status"] == "awaiting_source_review" and reviews:
            review = ReadL05Review.model_validate_json((reviews / (case["id"] + ".json")).read_text())
            assessment = assess_semantic(source, SemanticFlow.model_validate(files["tree.json"]), review)
            row["review"] = {key: assessment[key] for key in ("status", "inputDigest", "reviewDigest", "reportDigest", "reviewerId", "reviewerKind")}
            row["review"].update(verdictCounts=assessment["assessment"]["verdictCounts"],
                scopeNote=review.assessment.scope_note, issues=files["tree.json"]["issues"],
                findings=[{key: item[key] for key in ("claimId", "l05Pointer", "l0Pointer", "verdict", "rationale", "suggested_revision")}
                          for item in assessment["assessment"]["rows"] if item["verdict"] != "supported"])
        rows.append(row)
    body = {"protocol": PROTOCOL, "manifestDigest": manifest["manifestDigest"], "parentManifestDigest": manifest["parentManifestDigest"],
        "evidenceRole": manifest["evidenceRole"], "model": manifest["model"], "rows": rows,
        "uniqueKnownFlows": len(rows), "publicSkills": 0,
        "structurallyQualified": sum(row["status"] == "awaiting_source_review" for row in rows),
        "sourceReviewed": sum("review" in row for row in rows),
        "reviewSupportedInactiveFlows": sum(row.get("review", {}).get("status") == "review_supported_inactive_flow" for row in rows),
        "runtimeExecutions": 0, "writes": 0, "independentHumanEvidence": False,
        "boundary": "Answer-free first attempts on known development sources. Source reviews are same-assistant judgments, not independent Gold or calibrated accuracy. Qualified trees and safe missing-capability stops are not business completion or Runtime authority. POST cost includes waiting, excludes preflight/review/tests."}
    return {**body, "reportDigest": sha256_json(body)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "report"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reviews", type=Path)
    args = parser.parse_args()
    if args.command in {"freeze", "report"} and not args.output:
        parser.error("freeze/report require --output")
    if args.command == "freeze":
        print(freeze(args.root, args.output)["manifestDigest"])
    elif args.command == "run":
        run(args.root)
    else:
        _write(args.output, report(args.root, args.reviews))


if __name__ == "__main__":
    main()
