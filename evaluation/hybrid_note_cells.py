"""Owned candidate-note edits sharing the body pass's eight-cell budget.

Original notes remain in the audit. Editing a model-authored limitation cannot
clear a host duty, confer approval, change observations or write the body.
"""
import json

from evaluation.hybrid_review_views import source_view
from network_runtime.l0.structured_schema import validate_data

SYSTEM = """Inspect ONLY the owned unverified candidate note, against the original task and supplied observations.
Return keep, replace, or remove. A note's implied premises need support just as answer prose does: NOT approved
does not imply NOT reviewed; unknown is not false. Correct only an actual unsupported assertion or misleading
limitation. Keep valid uncertainty; do not erase an unresolved host duty, invent an approval or claim completion.
Source Skill examples and prior candidate text are not observations. Read provenance identifies the particular
resource/arguments, not a claim that the payload is true, fresh or globally exhaustive. Execution prohibitions
must be obeyed, but need not be repeated as business findings unless the user asks for them in the deliverable.
Changes need cited sources and a short reason. keep/remove require empty replacement; replace needs one plain
paragraph for this note, not Markdown sections, code fences or the answer document. No body editing, new notes,
tools, script execution, authority or self-approval. One call, <=2048 tokens.
"""


def slots(payload, repair_plan):
    result = []
    for i, note in enumerate(payload["candidate"]["notes"]):
        pointer = f"/candidate/notes/{i}"
        findings = [r for r in repair_plan["unlocatedFindings"] if r.get("candidatePointer") == pointer]
        if findings:
            result.append({"id": f"u{i:03d}", "index": i, "pointer": pointer, "text": note, "findings": findings})
    return result


def schema(payload):
    props = {"operation": {"type": "string", "enum": ["keep", "replace", "remove"]},
        "replacement": {"type": "string", "maxLength": 2000},
        "source_span_ids": {"type": "array", "maxItems": 8, "uniqueItems": True,
            "items": {"type": "string", "enum": [s["source_span_id"] for s in payload["sourceSpans"]]}},
        "rationale": {"type": "string", "minLength": 12, "maxLength": 1200}}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def validate(payload, slot, raw):
    raw = validate_data(schema(payload), raw)
    if payload["candidate"]["notes"][slot["index"]] != slot["text"]:
        raise ValueError("owned note differs from parent candidate")
    if raw["operation"] != "replace" and raw["replacement"]:
        raise ValueError("keep/remove cannot hide a note replacement")
    if raw["operation"] == "replace" and not raw["replacement"].strip():
        raise ValueError("replace requires nonempty note; removal must be explicit")
    if raw["operation"] == "replace" and ("\n" in raw["replacement"] or "\r" in raw["replacement"]
            or raw["replacement"].lstrip().startswith(("#", "```", "~~~"))):
        raise ValueError("owned note replacement must be one paragraph, not a body document")
    if raw["operation"] != "keep" and not raw["source_span_ids"]:
        raise ValueError("note change requires source citations, not authority")
    updated = slot["text"] if raw["operation"] == "keep" else raw["replacement"] if raw["operation"] == "replace" else None
    return {"pointer": slot["pointer"], "index": slot["index"], "before": slot["text"], "after": updated,
            "changed": updated != slot["text"], "proposal": raw, "semanticApproval": False}


def view(supplied):
    return {"originalTask": supplied["originalTask"],
        **({"taskScope": supplied["taskScope"]} if "taskScope" in supplied else {}),
        "sourceSpans": [source_view(s) for s in supplied["sourceSpans"]],
        "ownedNote": supplied["ownedNote"], "readOnlyAnswer": supplied["completePriorDraftReadOnly"],
        "hostOpenDuties": supplied["hostOpenDuties"], "hostDutiesEditable": False}


def materialize(original_notes, edits):
    by_index = {}
    for edit in edits:
        index = edit["index"]
        if (type(index) is not int or not 0 <= index < len(original_notes) or index in by_index
                or edit["pointer"] != f"/candidate/notes/{index}" or original_notes[index] != edit["before"]):
            raise ValueError("duplicate, out-of-range or drifting note edit")
        by_index[index] = edit["after"]
    return [by_index.get(i, note) for i, note in enumerate(original_notes) if by_index.get(i, note) is not None]


PROJECTION_PROFILE = "source-excerpt-note/v1"
PROJECTION_SYSTEM = """Inspect ONLY this owned, unverified note. Choose keep, or quote_observations using
one to four available evidence IDs. Choose a source excerpt only if it usefully grounds or corrects the note;
otherwise keep the unresolved note. Do not substitute an unrelated observation to make the result look complete.
There is NO free-text replacement channel. The host renders exact selected observations with an explicit
unverified-excerpt label. You cannot alter facts, add caveats, invent duties, erase host requirements or edit
the body. The original note stays in the audit; quoting evidence does NOT mean its interpretation or the task
is approved. keep requires empty evidence_ids. No tools, scripts, actions or permission. JSON only.
"""


def projection_input(payload, index, evidence_ids):
    from evaluation.hybrid_predicate_review import evidence_catalog
    if type(index) is not int or not 0 <= index < len(payload["candidate"]["notes"]):
        raise ValueError("owned note index out of range")
    catalog = evidence_catalog(payload)
    if len(set(evidence_ids)) != len(evidence_ids) or any(eid not in catalog["units"] for eid in evidence_ids):
        raise ValueError("note projection requires unique host evidence IDs")
    if any(catalog["units"][eid]["kind"] != "observation" for eid in evidence_ids):
        raise ValueError("note projection cannot promote guidance/task/caller into observations")
    units = {eid: catalog["units"][eid] for eid in evidence_ids}
    parents = {s["source_span_id"]: source_view(s) for s in payload["sourceSpans"]
               if any(u["sourceId"] == s["source_span_id"] for u in units.values())}
    return {"profile": PROJECTION_PROFILE, "candidateDigest": payload["completeCandidateDigest"],
        "catalogDigest": catalog["reportDigest"], "originalTask": payload["originalTask"],
        "ownedNote": {"index": index, "pointer": f"/candidate/notes/{index}", "text": payload["candidate"]["notes"][index]},
        "availableEvidence": units, "completeParents": parents, "hostOpenDuties": payload["hostOpenDuties"],
        "hostDutiesEditable": False, "authorityGranted": False}


def projection_schema(supplied):
    from evaluation.hybrid_semantic_witness import obj
    keys = list(supplied["availableEvidence"])
    return obj({"operation": {"type": "string", "enum": ["keep", *(["quote_observations"] if keys else [])]},
        "evidence_ids": {"type": "array", "maxItems": 4 if keys else 0, "uniqueItems": True,
                         "items": {"type": "string", **({"enum": keys} if keys else {})}}})


def validate_projection(payload, supplied, raw):
    from evaluation.structured_authoring import seal
    if projection_input(payload, supplied["ownedNote"]["index"], list(supplied["availableEvidence"])) != supplied:
        raise ValueError("source projection input drift")
    value = validate_data(projection_schema(supplied), raw)
    ids = value["evidence_ids"]
    if (value["operation"] == "quote_observations") != bool(ids):
        raise ValueError("quote requires evidence; keep cannot hide a projection")
    excerpts = [supplied["availableEvidence"][eid] for eid in ids]
    # JSON string rendering retains exact source bytes after decoding, without
    # allowing a source newline to become a generated heading or code fence.
    after = supplied["ownedNote"]["text"] if not ids else "Unverified observation excerpt(s), not a verified conclusion: " + "; ".join(
        f"[{u['id']}] {json.dumps(u['text'], ensure_ascii=False)}" for u in excerpts)
    if ids and len(after) > 2000:
        raise ValueError("source excerpt note exceeds original 2000-character budget; no truncation")
    return seal({"profile": PROJECTION_PROFILE, "pointer": supplied["ownedNote"]["pointer"],
        "index": supplied["ownedNote"]["index"], "before": supplied["ownedNote"]["text"], "after": after,
        "changed": after != supplied["ownedNote"]["text"], "proposal": value,
        "sourceExcerpts": excerpts, "sourceCatalogDigest": supplied["catalogDigest"],
        "viewKind": "source_excerpt_only" if ids else "retained_unverified_note",
        "modelAuthoredReplacementText": False, "sourceProjectionProven": bool(ids),
        "originalNoteRetainedInAudit": True, "semanticCoverageProven": False, "quoteSelectionUsefulnessProven": False,
        "semanticApproval": False, "originalDutyResolved": False})
