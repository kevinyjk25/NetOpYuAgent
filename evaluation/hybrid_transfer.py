"""Frozen, bounded public-development mixed construction; no business execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from evaluation import hybrid_authoring as author
from evaluation.flow_checkpoint import author_once, implementation
from evaluation.flow_model_transport import decode
from evaluation.source_ledger import budget
from evaluation.stage2_batch import digest
from evaluation.stage2_cases import specification
from evaluation.hybrid_task_context import separate
from evaluation.structured_authoring import seal
from evaluation.structured_binding_probe import read_json, write_artifacts

ROOT = Path(__file__).resolve().parents[1]
MAX_CALLS_PER_SKILL = 4


def fingerprint():
    return implementation("evaluation/hybrid_authoring.py", "evaluation/hybrid_parameters.py", "evaluation/hybrid_transfer.py", "evaluation/source_ledger.py",
        "evaluation/hybrid_task_context.py", "evaluation/hybrid_prefix.py", "evaluation/hybrid_reasoning_transport.py",
        "evaluation/structured_authoring.py", "evaluation/translation_intake.py", "evaluation/stage2_cases.py",
        "evaluation/stage2_batch.py", "evaluation/structured_binding_probe.py")


def prepare(original, output):
    original, output = Path(original), Path(output)
    if output.exists():
        raise FileExistsError("new version must preserve prior preparations")
    manifest = read_json(original / "freeze/manifest.json")
    if manifest != seal({k: v for k, v in manifest.items() if k != "reportDigest"}):
        raise ValueError("original intake seal drift")
    if any(digest(original / path) != value for path, value in manifest["artifactDigests"].items()):
        raise ValueError("original input drift")
    rows, files = [], {}
    for case in manifest["cases"]:
        folder = original / case["id"]
        if (folder / "author-input.json").exists():
            packet = read_json(folder / "author-input.json")
        else:
            task, schema, tools = specification(case["id"])
            packet = {"bundle": read_json(folder / "source-bundle.json"), "task": task,
                "taskOrigin": "developer_authored_evaluation_request", "inputSchema": schema,
                "catalog": {"tools": tools, "origin": "declared_local_adapter_contracts_not_vendor_capture"}, "reads": {}}
        original_task = read_json(folder / "task.json")
        packet["task"], task_provenance = separate(packet["task"], original_task["task"])
        task_provenance["frozenBusinessTaskFileDigest"] = digest(folder / "task.json")
        author.validate_packet(packet)
        pages = author.pages_for(packet)
        visible = [key for key, p in pages.items() if p["path"] == packet["bundle"]["entryPath"]]
        wire = author.make_request(packet, visible)
        markdown_pages = [key for key, page in pages.items() if page["path"].lower().endswith(".md")]
        complete_markdown = list(dict.fromkeys([*visible, *markdown_pages]))
        complete_wire = author.make_request(packet, complete_markdown)
        if budget(complete_wire)["accepted"]:
            visible, wire = complete_markdown, complete_wire
        # Keep the full source and every page in the index. This is explicit
        # progressive delivery, not silent source truncation or a larger budget.
        if not budget(wire)["accepted"]:
            visible = visible[:1]
            wire = author.make_request(packet, visible)
        body = {"author-input.json": packet, "task-provenance.json": task_provenance,
            "initial.json": {"visible": visible, "wireRequest": wire, "budget": budget(wire)},
            "review-requirements.json": read_json(folder / "review-requirements.json")}
        write_artifacts(output / case["id"], body)
        files.update({case["id"] + "/" + name: digest(output / case["id"] / name) for name in body})
        rows.append({k: case[k] for k in ("id", "repository", "commitSha", "sourcePath", "domain", "bundleDigest")})
    result = seal({"protocol": author.PROTOCOL, "implementation": fingerprint(), "model": author.MODEL,
        "originalIntakeDigest": manifest["reportDigest"], "cases": rows, "artifactDigests": files,
        "policy": {"maxCallsPerSkill": MAX_CALLS_PER_SKILL, "noRetryOfFailedOrIncompleteCalls": True,
            "authorThinking": False, "runtimeThinking": False, "authorOutputBudget": 4096,
            "exactOutputSchemaVisibleToModel": True,
            "decoder": "json_syntax_only_exact_schema_checked_locally_no_repair",
            "authorSurface": "grounded_read_prefix_then_verbatim_retained_original_L1_task",
            "sourcePackaging": "complete bundled Markdown if it fits unchanged budget; otherwise indexed progressive pages",
            "sameBusinessTasksSchemasHostsAndSourceBundles": True,
            "legacyWrapperSeparatedWithProvenance": True, "tenthCase": "original_tool_free_task_now_representable_as_L1_not_L0",
            "acceptance": "Source-task-host semantic review plus original-engine local behaviors, not schema alone",
            "strictMissingControl": "Reject unsafe downstream operations; preserve partial boundary, not full success",
            "formalGeneralizationGate": "unchanged_closed", "sourceScripts": "inert_never_executed"},
        "evidenceRole": "known_public_development_not_unseen_or_independent_gold", "largeRuntimeABUnlocked": False})
    write_artifacts(output / "freeze", {"manifest.json": result,
        "source-snapshot.json": {path: (ROOT / path).read_text() for path in result["implementation"]}})
    return result


def verify(root):
    root = Path(root)
    manifest = read_json(root / "freeze/manifest.json")
    if manifest != seal({k: v for k, v in manifest.items() if k != "reportDigest"}):
        raise ValueError("mixed preparation seal drift")
    if manifest["implementation"] != fingerprint() or manifest["protocol"] != author.PROTOCOL:
        raise ValueError("implementation drift; use a new version, never reclassify frozen results")
    if any(digest(root / path) != value for path, value in manifest["artifactDigests"].items()):
        raise ValueError("prepared source/task/host/reviewer drift")
    return manifest


def derive(packet, visible, envelope):
    text, cost = decode("ollama", envelope)
    if text is None:
        return {}, {**cost, "status": "construction_failed_no_retry"}
    try:
        choice = json.loads(text)
        Draft202012Validator(author.author_response_schema(packet, visible)).validate(choice)
        files = {"choice.json": choice}
        if choice["mode"] == "request_pages":
            next_visible = list(dict.fromkeys([*visible, *choice["pages"]]))
            if next_visible == visible:
                return files, {**cost, "status": "repeated_source_request_no_retry"}
            return files, {**cost, "status": "source_requested", "nextVisible": next_visible}
        compiled = author.compile_proposal(packet, visible, choice)
        return {**files, "compilation.json": compiled}, {**cost, "status": "compiled_mixed_candidate_requires_review"}
    except (ValueError, KeyError, TypeError, ValidationError) as error:
        return {"raw-choice.json": {"text": text}}, {**cost, "status": "construction_failed_no_retry", "diagnostic": str(error)[:1800]}


def run(preparation, output, selected=None, *, max_new_calls=0):
    preparation, output = Path(preparation), Path(output)
    manifest = verify(preparation)
    ids = [c["id"] for c in manifest["cases"]]
    selected = ids if selected is None else selected
    if len(set(selected)) != len(selected) or set(selected) - set(ids) or type(max_new_calls) is not int or not 0 <= max_new_calls <= 40:
        raise ValueError("unique selected original cases and bounded new-call budget required")
    frozen = seal({"preparationDigest": manifest["reportDigest"], "selected": selected, "implementation": fingerprint()})
    if output.exists():
        if read_json(output / "freeze/selection.json") != frozen:
            raise ValueError("selection drift")
    else:
        write_artifacts(output / "freeze", {"selection.json": frozen})
    rows, remaining = [], max_new_calls
    for key in selected:
        packet = read_json(preparation / key / "author-input.json")
        visible = read_json(preparation / key / "initial.json")["visible"]
        rounds, status = [], "not_run"
        for index in range(MAX_CALLS_PER_SKILL):
            folder = output / key / f"round-{index:02d}"
            wire = author.make_request(packet, visible)
            measured = budget(wire)
            if not measured["accepted"]:
                status = "blocked_before_model_resource_budget"
                break
            if folder.exists() and not (folder / "receipt.json").exists():
                status = "unknown_request_preserved_not_retried"
                break
            if not folder.exists() and not remaining:
                status = "new_call_budget_exhausted"
                break
            new_call = not folder.exists()
            result = author_once(folder, {"protocol": author.PROTOCOL, "wireRequest": wire,
                "authorInputDigest": seal(packet)["reportDigest"], "visible": visible, "budget": measured},
                lambda envelope: derive(packet, visible, envelope), max_new_calls=int(new_call), label="mixed author")
            remaining -= int(new_call)
            row = {"round": index, **result["result"]}
            rounds.append(row)
            status = row["status"]
            print(json.dumps({"case": key, **row}), flush=True)
            if status != "source_requested":
                break
            visible = row["nextVisible"]
        if status == "source_requested":
            status = "source_call_budget_exhausted"
        rows.append({"id": key, "status": status, "rounds": rounds, "lastBudget": measured,
            "semanticVerdict": None, "visible": visible})
        # Checkpoints are authoritative. Summary is only written once at end.
    report = seal({"protocol": author.PROTOCOL, "preparationDigest": manifest["reportDigest"], "rows": rows,
        "sourceScriptCalls": 0, "businessProviderCalls": 0, "model": author.MODEL,
        "semanticReview": "pending", "largeRuntimeABUnlocked": False})
    summary = output / "summary"
    if any(r["status"] == "new_call_budget_exhausted" for r in rows):
        return report  # Per-call receipts are resumable checkpoints; no final summary yet.
    if summary.exists():
        if read_json(summary / "report.json") != report:
            raise ValueError("completed summary drift; preserve old output and inspect checkpoints")
    else:
        write_artifacts(summary, {"report.json": report})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run"])
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--max-new-calls", type=int, default=0)
    args = parser.parse_args()
    result = prepare(args.input, args.output) if args.command == "prepare" else run(
        args.input, args.output, args.cases, max_new_calls=args.max_new_calls)
    print(result["reportDigest"], flush=True)


if __name__ == "__main__":
    main()
