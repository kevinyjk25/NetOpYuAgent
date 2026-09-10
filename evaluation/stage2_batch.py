"""Frozen public-development intake and bounded first construction. Never execute tools."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from evaluation import source_ledger as ledger
from evaluation.stage2_cases import author_packet, review_requirements, specification
from evaluation.structured_authoring import seal
from evaluation.structured_binding_probe import read_json, write_artifacts
from evaluation.translation_intake import bundle_from_snapshot

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_selection(selection):
    rows = selection["cases"]
    if (len(rows) != 10 or len({r["id"] for r in rows}) != 10 or len({r["candidateId"] for r in rows}) != 10
            or len({r["domain"] for r in rows}) < 3 or selection["model"] != "qwen3.5:9b"):
        raise ValueError("fixed ten-Skill selection, unique sources and 9B required")
    return rows


def prepare(selection_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("preserve earlier batch preparation")
    selection = read_json(selection_path)
    selected = validate_selection(selection)
    index = {}
    for path in sorted((ROOT / selection["sourceRoot"]).glob("*/snapshot/records.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["candidateId"] in index:
                raise ValueError("ambiguous corpus source")
            index[row["candidateId"]] = (path.parent, row)
    repositories = {index[r["candidateId"]][1]["repository"] for r in selected}
    if len(repositories) < 6:
        raise ValueError("at least six source repositories required")
    rows, files = [], {}
    for case in selected:
        source, record = index[case["candidateId"]]
        bundle = bundle_from_snapshot(source, case["candidateId"])
        folder = output / case["id"]
        inputs = author_packet(bundle, case["id"])
        review = review_requirements(bundle, case["id"])
        body = {"source-bundle.json": bundle, "review-requirements.json": review}
        if inputs is not None:
            body["author-input.json"] = inputs
            state = ledger.initial_state(inputs, "plan_first", semantic_plan=True)
            wire, budget = ledger.make_request(inputs, state)
            body["initial-request.json"] = {"wireRequest": wire, "budget": budget}
            status = "ready_for_first_construction" if budget["accepted"] else "blocked_before_model_resource_budget"
        else:
            budget, status = None, "l1_only_no_external_read_not_run"
        task, schema, _ = specification(case["id"])
        body["task.json"] = {"task": task, "inputSchema": schema, "taskOrigin": "developer_authored_evaluation_request"}
        write_artifacts(folder, body)
        for path in folder.iterdir():
            files[str(path.relative_to(output))] = digest(path)
        rows.append({**case, "repository": record["repository"], "commitSha": record["commitSha"],
            "sourcePath": record["sourcePath"], "bundleDigest": bundle["bundleDigest"],
            "documentCount": len(bundle["documents"]), "quarantinedFiles": len(record.get("quarantinedFiles", [])),
            "status": status, "budget": budget, "semanticVerdict": None})
    manifest = seal({"selection": selection, "selectionDigest": digest(selection_path), "cases": rows,
        "skillCount": len(rows), "repositoryCount": len(repositories), "domainCount": len({r["domain"] for r in rows}),
        "protocol": ledger.PROTOCOL, "implementation": ledger.fingerprint(), "policy": ledger.policy(),
        "preparationCode": {p: digest(ROOT / p) for p in ("evaluation/stage2_cases.py", "evaluation/stage2_batch.py")},
        "artifactDigests": files, "modelCalls": 0, "sourceScriptCalls": 0, "providerCalls": 0,
        "evidenceRole": "known_public_development_with_declared_local_adapters_not_unseen_or_vendor_validation",
        "reviewKind": "developer_ai_not_independent_gold", "wholeSkillAccuracy": None, "largeRuntimeABUnlocked": False})
    write_artifacts(output / "freeze", {"manifest.json": manifest})
    return manifest


def verify_preparation(root):
    root = Path(root)
    m = read_json(root / "freeze/manifest.json")
    if m != seal({k: v for k, v in m.items() if k != "reportDigest"}):
        raise ValueError("preparation manifest drift")
    if m["implementation"] != ledger.fingerprint() or m["protocol"] != ledger.PROTOCOL or m["policy"] != ledger.policy():
        raise ValueError("authoring drift requires separate preparation/revision")
    if any(digest(root / p) != value for p, value in m["artifactDigests"].items()):
        raise ValueError("prepared source/task/reviewer artifact drift")
    if any(digest(ROOT / p) != value for p, value in m["preparationCode"].items()):
        raise ValueError("batch preparation/driver code drift")
    return m


def run(root, output, *, max_new_calls=0):
    """Bound construction, not retries; existing incomplete requests stay unknown."""
    if type(max_new_calls) is not int or not 0 <= max_new_calls <= 140:
        raise ValueError("explicit bounded new-call budget required")
    root, output = Path(root), Path(output)
    m = verify_preparation(root)
    if output.exists():
        raise FileExistsError("preserve completed and interrupted batch directories")
    write_artifacts(output, {"input-manifest.json": m})
    budget, rows = max_new_calls, []
    for case in m["cases"]:
        row = {"id": case["id"], "status": case["status"], "modelCalls": 0, "semanticVerdict": None}
        if case["status"] == "ready_for_first_construction" and budget:
            folder = output / case["id"]
            try:
                ledger.freeze(read_json(root / case["id"] / "author-input.json"), folder,
                              profile="plan_first", semantic_plan=True)
                while True:
                    limit = min(ledger.MAX_ROUNDS, budget)
                    before = len(list(folder.glob("round-*/request.json")))
                    try:
                        result = ledger.run(folder, max_new_calls=limit)
                    finally:
                        after = len(list(folder.glob("round-*/request.json")))
                        budget -= after - before
                    if result["status"] != "new_call_budget_exhausted" or not budget:
                        break
                write_artifacts(folder / "final-report", {"report.json": result})
                row.update(status=result["status"], modelCalls=result["modelCallsRecorded"], compiled=result["compiled"])
            except (ValueError, OSError, RuntimeError) as error:
                # Never retry a failed/unreceipted request. Keep other selected
                # cases visible without treating execution failure as abstention.
                row.update(status="construction_error_preserved", diagnostic=str(error)[:1800])
                requests = len(list(folder.glob("round-*/request.json")))
                receipts = len(list(folder.glob("round-*/receipt.json")))
                row.update(requests=requests, receipts=receipts, unknownRequests=requests - receipts)
                write_artifacts(folder / "driver-error", {"report.json": row})
        elif case["status"] == "ready_for_first_construction":
            row["status"] = "not_run_call_budget"
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = seal({"preparationDigest": m["reportDigest"], "skillCount": m["skillCount"],
        "repositoryCount": m["repositoryCount"], "domainCount": m["domainCount"], "rows": rows,
        "sourceScriptCalls": 0, "providerCalls": 0, "semanticReviewStatus": "pending",
        "wholeSkillAccuracy": None, "largeRuntimeABUnlocked": False})
    write_artifacts(output / "summary", {"report.json": report})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run"])
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--max-new-calls", type=int, default=0)
    args = parser.parse_args()
    result = prepare(args.input, args.output) if args.command == "prepare" else run(
        args.input, args.output, max_new_calls=args.max_new_calls)
    print(result["reportDigest"])


if __name__ == "__main__":
    main()
