"""Frozen small-transfer workflow with separate inputs, admission and judgment.

Disclosed local 9B and inert in-process fixtures only. No expected answers reach
the author, writer, reviewer or Runtime. First failures are never retried here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path

from evaluation import hybrid_authoring as author
from evaluation.flow_checkpoint import author_once, implementation
from evaluation.hybrid_continuation_run import run as continue_reads
from evaluation.hybrid_draft_loop import REVIEW_CONFIG
from evaluation.hybrid_live_demo import run as run_initial
from evaluation.hybrid_snapshot_review import run as review_draft
from evaluation.hybrid_repair_cells import run as repair_cells
from evaluation.hybrid_transfer import derive
from evaluation.source_ledger import budget
from evaluation.structured_authoring import seal
from evaluation.structured_binding_probe import read_json, write_artifacts
from evaluation.translation_case_authoring import OllamaAnchoredAuthorAdapter
from evaluation.translation_intake import bundle_from_snapshot
from network_runtime.contracts import sha256_json
from network_runtime.l0.structured_reads import StructuredReadManifest, compile_structured_read

ROOT = Path(__file__).resolve().parents[1]


def fingerprint():
    return implementation(*(str(p.relative_to(ROOT)) for p in (ROOT / "evaluation").glob("*.py")))


def freeze(root, *, known_development=False):
    root = Path(root)
    if root.exists():
        raise FileExistsError("new freeze required; no replacement of prior evidence")
    files = fingerprint()
    manifest = seal({"implementation": files, "model": OllamaAnchoredAuthorAdapter().preflight(),
        "writerConfig": author.MODEL_CONFIG, "reviewConfig": REVIEW_CONFIG,
        "sourceSelectionHasOccurred": known_development, "minimum": {"skills": 6, "repositories": 4, "domains": 3, "tasks": 12},
        "sourceSupplyPolicy": "complete_entry_then_whole_inert_reference_documents_within_existing_budget/v2",
        "reviewWireProfile": "host_keyed_check_cells/v2",
        "limitsPerTask": {"authorCalls": 4, "initialWriterCalls": 1, "continuationRounds": 3,
                          "sourceReviewCalls": 1, "isolatedRepairCells": 8, "finalReviewCalls": 1},
        "repairEditor": {"modelConfig": author.MODEL_CONFIG, "decoder": "schema", "repairPasses": 1,
                         "editMode": "grounded_patch", "partition": "complete_sections",
                         "repairTrigger": "located_findings", "unlocatedOmissionsRemainOpen": True,
                         "singleWholeDraftOwnerCanReceiveUnlocatedFindings": True,
                         "transportPropertyOrder": "declared_required_order",
                         "hostOwnsCandidateIdentityAndLocations": True},
        "evidenceRole": "known_development_mechanism_diagnostic_not_unseen_acceptance" if known_development else
                        "small_developer_transfer_not_independent_gold_or_formal_generalization",
        "scriptsExecutable": False, "effectsAuthorized": False, "largeRuntimeABUnlocked": False})
    write_artifacts(root / "freeze", {"manifest.json": manifest})
    with tarfile.open(root / "freeze/source-snapshot.tar.gz", "x:gz") as archive:
        for name in files:
            archive.add(ROOT / name, arcname=name, recursive=False)
    return manifest


def verify(root):
    manifest = read_json(Path(root) / "freeze/manifest.json")
    if manifest != seal({k: v for k, v in manifest.items() if k != "reportDigest"}) or manifest["implementation"] != fingerprint():
        raise ValueError("frozen implementation drift; use a new batch, never reclassify this batch")
    if manifest["writerConfig"] != author.MODEL_CONFIG or manifest["reviewConfig"] != REVIEW_CONFIG:
        raise ValueError("frozen model configuration drift")
    if manifest["model"] != OllamaAnchoredAuthorAdapter().preflight():
        raise ValueError("frozen local model artifact drift")
    return manifest


def case_folder(root, case):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", case):
        raise ValueError("one confined case identifier required")
    return Path(root) / "cases" / case


def packet_for(bundle, specification):
    """Declare primitive adapter contracts, not a graph or expected response."""
    text = next(d["content"] for d in bundle["documents"] if d["path"] == bundle["entryPath"])
    reads = {}
    for index, tool in enumerate(specification["tools"]):
        name = tool["name"]
        if name in reads or tool.get("annotations", {}).get("readOnlyHint") is not True:
            raise ValueError("unique explicitly read-only local tools required")
        adapter = {"tool": name, "capability": f"semantic-transfer.read{index}", "effect": "read_only",
                   "resourceScopes": {}, "access": {"requiredScopes": ["stage2:read"], "dataClassification": "internal"},
                   "limitations": "Disclosed synthetic host; finite resource ACL separately enforced before provider invocation."}
        sources = [{"role": role, "origin": "local-transfer-declaration:" + role, "text": source,
                    "sha256": "sha256:" + hashlib.sha256(source.encode()).hexdigest()}
                   for role, source in (("skill", text), ("tool", json.dumps(tool)), ("adapter", json.dumps(adapter)))]
        contract = StructuredReadManifest.model_validate({"apiVersion": "netopyu.io/l0-structured-read/v1", "kind": "StructuredRead",
            "metadata": {"id": f"semantic-transfer.read{index}", "version": "1.0.0", "owner": "local-development-review"},
            "spec": {**{k: adapter[k] for k in ("tool", "capability", "effect", "resourceScopes", "access")},
                     "inputSchema": tool["inputSchema"], "outputSchema": tool["outputSchema"], "sources": sources}})
        reads[name] = compile_structured_read(contract).model_dump(mode="json", by_alias=True)
    packet = {"bundle": bundle, "task": specification["task"], "taskOrigin": "developer_authored_evaluation_request",
              "inputSchema": specification["inputSchema"], "catalog": {"tools": specification["tools"],
              "origin": "disclosed_local_synthetic_host_not_vendor_capture"}, "reads": reads}
    if "taskScope" in specification:
        packet["taskScope"] = specification["taskScope"]
    author.validate_packet(packet)
    return packet


def prepare(root, case, specification):
    manifest = verify(root)
    folder = case_folder(root, case)
    if folder.exists():
        raise FileExistsError("prior case inputs cannot be replaced")
    bundle = bundle_from_snapshot(specification["snapshot"], specification["candidateId"])
    packet = packet_for(bundle, specification)
    files = {"packet.json": packet, "fixture.json": specification["fixture"],
             "expectations.json": {"criteria": specification["expectations"], "domain": specification["domain"],
                 "reviewKind": "developer_predeclared_not_independent_gold", "neverModelInput": True},
             "selection.json": {"snapshot": specification["snapshot"], "candidateId": specification["candidateId"],
                               "repository": bundle["repository"], "commit": bundle["commitSha"]}}
    write_artifacts(folder / "inputs", files)
    receipt = seal({"freezeDigest": manifest["reportDigest"], "case": case,
                    "inputs": {name: sha256_json(value) for name, value in files.items()}})
    write_artifacts(folder / "intake", {"receipt.json": receipt})
    return receipt


def verify_case(root, case):
    manifest, folder = verify(root), case_folder(root, case)
    receipt = read_json(folder / "intake/receipt.json")
    if (receipt != seal({k: v for k, v in receipt.items() if k != "reportDigest"}) or receipt["freezeDigest"] != manifest["reportDigest"]
            or any(sha256_json(read_json(folder / "inputs" / name)) != value for name, value in receipt["inputs"].items())):
        raise ValueError("case input/expected-outcome drift")
    return folder, read_json(folder / "inputs/packet.json")


def author_case(root, case):
    folder, packet = verify_case(root, case)
    if (folder / "author").exists():
        raise FileExistsError("author attempt already started; inspect retained evidence, do not retry")
    visible = initial_source_pages(packet)
    costs, compilation = [], None
    for index in range(4):
        wire = author.make_request(packet, visible)
        if not budget(wire)["accepted"]:
            costs.append({"newCalls": 0, "status": "context_budget_blocked"})
            break
        response = author_once(folder / "author" / f"call-{index + 1}", {"wireRequest": wire,
            "packetDigest": sha256_json(packet), "visiblePages": visible}, lambda envelope: derive(packet, visible, envelope),
            max_new_calls=1, label="frozen small-transfer authoring")
        costs.append(response["result"])
        if "compilation.json" in response:
            compilation = response["compilation.json"]
            break
        if response["result"]["status"] != "source_requested":
            break
        visible = response["result"]["nextVisible"]
    files = {"report.json": seal({"case": case, "calls": costs, "compilationDigest": compilation["reportDigest"] if compilation else None,
                                 "status": "requires_explicit_local_admission" if compilation else "authoring_incomplete"})}
    if compilation:
        files["compilation.json"] = compilation
    write_artifacts(folder / "author-summary", files)
    return files["report.json"]


def initial_source_pages(packet):
    """Entry-first whole-page budgeting, not all-references-or-first-page.

    Preserve the complete entry whenever it fits. Add whole inert Markdown
    documents in deterministic catalog order afterwards. A too-large entry
    receives a contiguous prefix and keeps the existing explicit paging path;
    omitted pages remain in the source catalog, never declared as read.
    """
    pages = author.pages_for(packet)
    entry = [key for key, page in pages.items() if page["path"] == packet["bundle"]["entryPath"]]
    if not entry:
        raise ValueError("entry has no source pages")
    visible = []
    for key in entry:
        trial = [*visible, key]
        if not budget(author.make_request(packet, trial))["accepted"]:
            # Preserve the first oversize page for the caller's explicit
            # zero-call budget failure; never silently supply an empty source.
            return visible or entry[:1]
        visible = trial
    documents = {}
    for key, page in pages.items():
        if key not in entry and page["path"].lower().endswith(".md"):
            documents.setdefault(page["path"], []).append(key)
    for keys in documents.values():
        trial = [*visible, *keys]
        if budget(author.make_request(packet, trial))["accepted"]:
            visible = trial
    return visible


def _execute(root, case, *, continuation_rounds=3):
    folder, packet = verify_case(root, case)
    if type(continuation_rounds) is not int or not 0 <= continuation_rounds <= 3:
        raise ValueError("zero-to-three continuation rounds required")
    inputs = folder / "inputs"
    # admission.json is explicitly supplied after actual source/graph inspection,
    # never generated from a structural-pass or expected-answer boolean here.
    compilation_path = folder / "author-summary/compilation.json"
    compilation = read_json(compilation_path)
    initial = run_initial(inputs / "packet.json", compilation_path, folder / "admission.json", folder / "initial",
                          case=case, max_model_calls=1, fixture_path=inputs / "fixture.json")
    reports, last, target = {"initial": initial["reportDigest"]}, initial["execution"], folder / "initial"
    needs = any(value["value"].get("remaining_actions") for value in last.get("outputs", {}).values() if isinstance(value["value"], dict))
    if last["status"] == "governed_graph_completed" and needs and packet["reads"] and continuation_rounds:
        fixture = read_json(inputs / "fixture.json")
        continuation = continue_reads(packet, compilation, initial, fixture["arguments"], fixture["resources"],
                                      folder / "continuation", max_rounds=continuation_rounds)
        reports["continuation"] = continuation["reportDigest"]
        last, target = continuation["lastExecution"], folder / "continuation"
    if last["status"] == "governed_graph_completed":
        reviewed = review_draft(target, folder / "review-before", max_model_calls=1)
        reports["review-before"] = reviewed["reportDigest"]
        last = reviewed["execution"]
        if last["status"] == "governed_graph_completed":
            role = read_json(Path(root) / "freeze/manifest.json")["evidenceRole"]
            repaired = repair_cells(folder / "review-before", folder / "repair", decoder="schema", edit_mode="grounded_patch",
                                    evidence_role=role if role.startswith("known_development_") else "frozen_small_developer_transfer_not_independent_gold")
            reports["repair"] = repaired["reportDigest"]
            last = {"status": repaired["finalReviewStatus"] or "repair_incomplete"}
    result = seal({"case": case, "reports": reports, "graphStatus": last["status"],
        "actualDraftReview": "pending_developer_judgment", "semanticSuccess": None,
        "automaticPermission": False, "largeRuntimeABUnlocked": False})
    write_artifacts(folder / "summary", {"report.json": result})
    return result


def execute(root, case, *, continuation_rounds=3):
    folder, _ = verify_case(root, case)
    if (folder / "summary").exists():
        raise FileExistsError("completed or failed case already recorded; never retry")
    try:
        return _execute(root, case, continuation_rounds=continuation_rounds)
    except (ValueError, PermissionError, OSError) as error:
        result = seal({"case": case, "status": "execution_incomplete_preserved", "errorType": type(error).__name__,
            "diagnostic": str(error)[:1200], "semanticSuccess": None, "automaticPermission": False,
            "priorStagesPreserved": True, "largeRuntimeABUnlocked": False})
        write_artifacts(folder / "summary", {"report.json": result})
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "prepare", "author", "execute"))
    parser.add_argument("root")
    parser.add_argument("--case")
    parser.add_argument("--specification")
    parser.add_argument("--known-development", action="store_true", help="Known sources cannot be claimed as a new transfer acceptance")
    args = parser.parse_args()
    if args.action == "freeze":
        result = freeze(args.root, known_development=args.known_development)
    elif args.action == "prepare":
        result = prepare(args.root, args.case, read_json(args.specification))
    elif args.action == "author":
        result = author_case(args.root, args.case)
    else:
        result = execute(args.root, args.case)
    print(json.dumps({k: result.get(k) for k in ("case", "status", "graphStatus", "reportDigest")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
