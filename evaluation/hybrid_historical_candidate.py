"""Import a checked historical edited draft as DATA for a fresh review only.

Old model judgments are never imported, remapped or upgraded. No business read,
action permission, new source credit or retroactive acceptance is produced.
"""
from copy import deepcopy
from pathlib import Path

from evaluation.hybrid_authoring import obj
from evaluation.semantic_closure_evidence import collect, read_json
from evaluation.structured_authoring import seal
from network_runtime.contracts import sha256_json
from network_runtime.l0.structured_schema import validate_data


def checked(path):
    value = read_json(path)
    if value != seal({k: v for k, v in value.items() if k != "reportDigest"}):
        raise ValueError("historical edit report digest drift")
    return value


def prepare(root):
    root = Path(root).resolve()
    summary = checked(root / "summary/report.json")
    frozen = checked(root / "freeze/inputs.json")
    application = checked(root / "materialized/application.json")
    candidate = read_json(root / "materialized/candidate.json")
    original = frozen["sourceInputs"]
    before = original["candidate"]
    if (summary["freezeDigest"] != frozen["reportDigest"]
            or summary["materializedCandidateDigest"] != sha256_json(candidate)
            or application["candidateDigest"] != sha256_json(candidate)
            or application["previousCandidateDigest"] != sha256_json(before)
            or summary["execution"]["status"] != "governed_graph_completed"
            or summary["execution"]["graphDigest"] != frozen["graph"]["graphDigest"]
            or candidate["values"] != before["values"]):
        raise ValueError("historical edited candidate/source binding drift")
    pieces, cursor = [], 0
    if len(application["edits"]) > 8:
        raise ValueError("historical edit range budget exceeded")
    for edit in application["edits"]:
        start, end = edit["start"], edit["end"]
        if (type(start) is not int or type(end) is not int or not cursor <= start < end <= len(before["draft"])
                or before["draft"][start:end] != edit["text"] or edit["action"] != "replace"):
            raise ValueError("historical edit location drift")
        pieces.extend([before["draft"][cursor:start], edit["replacement"]])
        cursor = end
    pieces.append(before["draft"][cursor:])
    if "".join(pieces) != candidate["draft"]:
        raise ValueError("historical edits do not reconstruct the exact delivered draft")
    if "noteEdits" in application:
        from evaluation.hybrid_note_cells import materialize
        if (materialize(before["notes"], application["noteEdits"]) != candidate["notes"]
                or len(application["edits"]) + sum(e["changed"] for e in application["noteEdits"]) > 8):
            raise ValueError("historical note edits or shared edit budget drift")
    # All original receipts/source archives remain checked and linked; no
    # execution of archived Python or third-party Skill material is required.
    evidence = collect(root)
    receipt = seal({"kind": "historical_repair_candidate_data_import", "historicalRoot": str(root),
        "historicalEvidenceDigest": evidence["reportDigest"], "originalSummaryDigest": summary["reportDigest"],
        "originalFreezeDigest": frozen["reportDigest"], "applicationDigest": application["reportDigest"],
        "candidateDigest": sha256_json(candidate), "sourceInputsDigest": sha256_json(original),
        "oldModelJudgmentsImported": False, "newBusinessReadCalls": 0, "newUnseenSources": 0,
        "newActionAuthority": False, "semanticApproval": False})
    supplied = {**deepcopy(original), "candidate": deepcopy(candidate)}
    properties = {k: {"type": "object" if isinstance(v, dict) else "string", "const": deepcopy(v)}
                  for k, v in supplied.items()}
    properties["candidate"] = obj({"values": {"type": "object", "const": candidate["values"]},
        "draft": {"type": "string", "maxLength": 12000},
        "notes": {"type": "array", "maxItems": 48, "items": {"type": "string"}}})
    schema = obj(properties)
    validate_data(schema, supplied)
    packet = {"bundle": {"bundleDigest": frozen["graph"]["proposal"]["source_digest"]},
              "task": supplied["original_task"]}
    return packet, receipt, supplied, schema
