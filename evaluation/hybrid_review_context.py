"""Lossless task roles and read provenance for candidate review.

Host-created navigation only: no new evidence, freshness, semantic entailment,
permission or inferred resource classification. Historical views without these
fields remain readable; missing provenance is not fabricated from source text.
"""
import json

from evaluation.hybrid_task_context import scope_view
from network_runtime.contracts import sha256_json
from network_runtime.l0.structured_schema import join_pointer


def read_entry(region, node, tool, arguments, payload, receipt_digest):
    return {"region": region, "node": node, "tool": tool, "arguments": arguments,
            "catalogPointer": join_pointer(join_pointer("/observations", region), node),
            "payloadDigest": sha256_json(payload), "receiptDigest": receipt_digest,
            "authorityGranted": False, "freshnessRenewed": False}


def trace_entries(execution, tool_by_hash, regions=None):
    entries = []
    for event in execution["trace"]:
        if event.get("kind") != "strict_region" or event.get("status") != "succeeded":
            continue
        region = (regions or {}).get(event["node"], event["node"])
        for row in event["regionReport"]["trace"]:
            if row["kind"] == "read":
                receipt = row["receipt"]
                entries.append(read_entry(region, row["node"], tool_by_hash[receipt["contractHash"]],
                    row["argumentBinding"]["arguments"], receipt["payload"], receipt["receiptDigest"]))
    return entries


def context_fields(inputs):
    fields = {}
    if "task_scope" in inputs:
        fields["taskScope"] = scope_view(inputs["original_task"], json.loads(inputs["task_scope"]))
    if "read_context" in inputs:
        entries = json.loads(inputs["read_context"])
        if not isinstance(entries, list) or len(entries) > 128:
            raise ValueError("bounded read context list required")
        seen = set()
        for entry in entries:
            region, node = entry["region"], entry["node"]
            payload = inputs["observations"][region]["observations"][node]
            expected = read_entry(region, node, entry["tool"], entry["arguments"], payload, entry["receiptDigest"])
            if entry != expected or (region, node) in seen:
                raise ValueError("read provenance/payload binding drift or duplicate")
            if not isinstance(entry["tool"], str) or not isinstance(entry["arguments"], dict):
                raise ValueError("typed tool/arguments required")
            seen.add((region, node))
        fields["readContext"] = entries
    return fields


def attach(supplied, schema, packet, entries):
    if "taskScope" in packet:
        supplied["task_scope"] = json.dumps(packet["taskScope"], ensure_ascii=False)
    supplied["read_context"] = json.dumps(entries, ensure_ascii=False)
    for key in ("task_scope", "read_context"):
        if key in supplied:
            schema["properties"][key] = {"type": "string"}
            schema["required"].append(key)
    context_fields(supplied)


def import_candidate_context(root, request_path, task_scope=None):
    """Explicit new diagnostic view; historical request/grades remain intact.

    Only import a receipt-bound candidate using exactly the same task, original
    Skill, caller and observation snapshots as the verified execution root.
    """
    from copy import deepcopy
    from pathlib import Path
    from evaluation.hybrid_snapshot_review import prepare
    from evaluation.flow_tree_authoring import verify_receipt
    from evaluation.structured_binding_probe import read_json
    from evaluation.structured_authoring import seal

    packet, previous, actual, schema = prepare(root)
    request_path = Path(request_path).resolve()
    verify_receipt(request_path.parent)
    request = read_json(request_path)
    original = request["governedRequest"]["inputs"]
    for key in ("original_task", "source_material", "caller", "observations"):
        if original[key] != actual[key]:
            raise ValueError(f"historical candidate {key} differs from actual execution; no heuristic merging")
    supplied = deepcopy(original)
    supplied["read_context"] = actual["read_context"]
    schema = deepcopy(schema)
    if task_scope is not None:
        scope_view(original["original_task"], task_scope)
        supplied["task_scope"] = json.dumps(task_scope, ensure_ascii=False)
        schema["properties"]["task_scope"] = {"type": "string"}
        if "task_scope" not in schema["required"]:
            schema["required"].append("task_scope")
    context_fields(supplied)
    provenance = seal({"kind": "lossless_context_candidate_import", "previousExecutionReportDigest": previous["reportDigest"],
        "historicalRequest": str(request_path), "historicalRequestDigest": sha256_json(request),
        "originalInputDigest": sha256_json(original), "newInputDigest": sha256_json(supplied),
        "candidateChanged": False, "sourceTextChanged": False, "taskTextChanged": False,
        "oldJudgmentsImported": False, "historicalGradesChanged": False, "newBusinessReadCalls": 0,
        "knownDevelopmentOnly": True})
    return packet, provenance, supplied, schema
