"""Frozen, single-attempt development batch. Generates proposals, never executes tools."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from evaluation.flow_translation import (
    FlowSources, _write, author, author_request, load_run, local_sources,
)
from evaluation.translation_case_authoring import OllamaAnchoredAuthorAdapter
from network_runtime.contracts import sha256_json
from network_runtime.l0.models import ReadObjectSchema

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data/flow_translation_development.json"
# Bind generation, lowering, schema and review implementation before any batch call.
IMPLEMENTATION = (
    "evaluation/flow_translation.py", "evaluation/flow_translation_batch.py",
    "evaluation/read_l05_review.py", "evaluation/translation_source_alignment.py",
    "evaluation/translation_case_authoring.py", "network_runtime/l0/flow.py",
    "network_runtime/l0/models.py", "network_runtime/l0/compiler.py",
)


def fingerprint() -> dict:
    return {name: "sha256:" + hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in IMPLEMENTATION}


def development_sources() -> list[tuple[dict, FlowSources]]:
    fixture = json.loads(CASES.read_text())
    base = local_sources()
    result, seen = [], set()
    for case in fixture["cases"]:
        key = case["id"]
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", key) or key in seen:
            raise ValueError("invalid or duplicate development case ID")
        seen.add(key)
        fields = case.get("inputs", ["device_id"])
        schema = ReadObjectSchema.model_validate({
            "type": "object", "properties": {field: {"type": "string"} for field in fields},
            "required": fields, "additionalProperties": False,
        })
        sources = base.model_copy(update={"source_text": case["source"] + fixture["sharedBoundary"],
            "source_path": "data/flow_translation_development.json#" + key, "input_schema": schema})
        result.append(({"id": key, "feature": case["feature"]}, sources))
    return result


def freeze(output: Path) -> dict:
    # Resolve everything before creating the output, no half manifest on preflight failure.
    rows = [{**case, "sources": sources.model_dump(mode="json"),
             "wireDigest": sha256_json(author_request(sources))} for case, sources in development_sources()]
    body = {"version": "known-flow-batch/v1", "evidenceRole": "known_synthetic_single_domain_development",
            "publicSkillCount": 0, "domainCount": 1, "cases": rows,
            "implementation": fingerprint(), "model": OllamaAnchoredAuthorAdapter().preflight(),
            "attemptsPerCase": 1, "runtimeExecutionAllowed": False,
            "boundary": "No held-out, market Skill, semantic accuracy or production claim. No retries or automatic repair."}
    manifest = {**body, "manifestDigest": sha256_json(body)}
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "manifest.json", manifest)
    return manifest


def load_manifest(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["manifestDigest"] != sha256_json({k: v for k, v in manifest.items() if k != "manifestDigest"}):
        raise ValueError("frozen manifest digest mismatch")
    if manifest["implementation"] != fingerprint():
        raise ValueError("frozen implementation changed; do not resume or tune this batch")
    for case in manifest["cases"]:
        if case["wireDigest"] != sha256_json(author_request(FlowSources.model_validate(case["sources"]))):
            raise ValueError("frozen request changed")
    return manifest


def _artifact_hashes(root: Path) -> dict:
    return {path.name: "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.glob("*.json")) if path.name != "receipt.json"}


def run(root: Path) -> dict:
    manifest = load_manifest(root)
    # A completed receipt is skipped; a directory without a receipt is ambiguous,
    # never re-issued automatically (the model may already have processed a call).
    for case in manifest["cases"]:
        folder = root / case["id"]
        if folder.exists():
            if not (folder / "receipt.json").exists():
                raise ValueError(f"interrupted/active case needs inspection, not retry: {case['id']}")
            verify_case(folder, case, manifest)
            continue
        author(FlowSources.model_validate(case["sources"]), folder, expected_model=manifest["model"])
        _write(folder / "receipt.json", {"manifestDigest": manifest["manifestDigest"], "files": _artifact_hashes(folder)})
        print(json.dumps({"case": case["id"], "status": json.loads((folder / "status.json").read_text())["status"]}), flush=True)
    return report(root)


def verify_case(folder: Path, case: dict, manifest: dict) -> dict:
    receipt = json.loads((folder / "receipt.json").read_text())
    if receipt != {"manifestDigest": manifest["manifestDigest"], "files": _artifact_hashes(folder)}:
        raise ValueError("case checkpoint changed")
    if json.loads((folder / "sources.json").read_text()) != case["sources"]:
        raise ValueError("case sources differ from frozen input")
    if (folder / "request.json").exists():
        request = json.loads((folder / "request.json").read_text())
        if request["model"] != manifest["model"] or sha256_json(request["wireRequest"]) != case["wireDigest"]:
            raise ValueError("actual request/model differs from freeze")
    status = json.loads((folder / "status.json").read_text())
    if status["status"] == "awaiting_source_review":
        load_run(folder)  # Reconstruct raw response and checklist, not just saved status.
    return status


def report(root: Path) -> dict:
    manifest = load_manifest(root)
    rows = []
    for case in manifest["cases"]:
        folder = root / case["id"]
        row = {"id": case["id"], "feature": case["feature"], "status": "not_run"}
        if (folder / "receipt.json").exists():
            row["status"] = verify_case(folder, case, manifest)["status"]
            row["receiptDigest"] = sha256_json(json.loads((folder / "receipt.json").read_text()))
            if (folder / "response.json").exists():
                response = json.loads((folder / "response.json").read_text())
                row["latencyMs"] = response["latencyMs"]
                row["httpStatus"] = response["httpStatus"]
                try:
                    body = json.loads(response["body"])
                    row["inputTokens"] = body.get("prompt_eval_count")
                    row["outputTokens"] = body.get("eval_count")
                except json.JSONDecodeError:
                    pass
            if (folder / "draft.json").exists():
                row["unresolvedQuestions"] = json.loads((folder / "draft.json").read_text())["unresolved_questions"]
            if (folder / "diagnostics.json").exists():
                row["diagnostics"] = json.loads((folder / "diagnostics.json").read_text())
        elif folder.exists():
            row["status"] = "interrupted_or_active_no_retry"
        rows.append(row)
    body = {"manifestDigest": manifest["manifestDigest"], "model": manifest["model"],
            "evidenceRole": manifest["evidenceRole"], "flowCount": len(rows), "publicSkillCount": 0,
            "domainCount": 1, "structurallyQualified": sum(row["status"] == "awaiting_source_review" for row in rows),
            "semanticAccepted": None, "semanticReviewStatus": "pending_bidirectional_source_review",
            "runtimeExecutions": 0, "writes": 0, "scriptExecutions": 0, "rows": rows,
            "boundary": "Structural qualification is not semantic acceptance. Missing environment and unsupported steps may form valid stop proposals, not completed business tasks."}
    return {**body, "reportDigest": sha256_json(body)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "report"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {"freeze": freeze, "run": run, "report": report}[args.command](args.root)
    if args.output:
        _write(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
