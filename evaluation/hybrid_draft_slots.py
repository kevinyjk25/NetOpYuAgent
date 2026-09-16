"""Host-addressed bounded draft edits. Materialization never proves semantics."""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from evaluation.hybrid_draft_review import obj
from network_runtime.contracts import sha256_json
from network_runtime.l0.structured_schema import snapshot_json, validate_data


def snapshot_output_schema(payload):
    """One coherent draft; locations and edit count are computed by the host."""
    schema = obj({"candidate_digest": {"type": "string", "enum": [payload["completeCandidateDigest"]]},
        "evidence_check": obj({s["draft_span_id"]: {"$ref": "#/$defs/evidence_status"}
            for s in payload["draftSpans"]}),
        "draft": {"type": "string", "maxLength": 12000},
        "notes": {"type": "array", "maxItems": 6, "items": {"type": "string", "minLength": 1, "maxLength": 2000}},
        "source_span_ids": {"type": "array", "minItems": 1, "maxItems": 8, "uniqueItems": True,
                            "items": {"type": "string", "enum": [s["source_span_id"] for s in payload["sourceSpans"]]}},
        "revision_note": {"type": "string", "minLength": 12, "maxLength": 1200}})
    schema["$defs"] = {"evidence_status": {"type": "string", "enum": [
        "observed_or_justified_inference", "nonfactual_heading_or_example", "contains_unsupported_assertion"]}}
    return schema


def apply_snapshot_revision(payload, values, raw):
    proposal = validate_data(snapshot_output_schema(payload), snapshot_json(raw))
    previous = {**payload["candidate"], "values": snapshot_json(values)}
    if sha256_json(previous) != payload["completeCandidateDigest"]:
        raise ValueError("revision candidate binding changed")
    draft, withheld = proposal["draft"], []
    validate_fences(draft)
    for span in payload["draftSpans"]:
        quote = span["exactQuote"].rstrip("\r\n")
        if (proposal["evidence_check"][span["draft_span_id"]] == "contains_unsupported_assertion"
                and quote and quote in draft):
            if draft.count(quote) != 1:
                raise ValueError("unsupported draft block has ambiguous replacement locations")
            start, end = draft.index(quote), draft.index(quote) + len(quote)
            boundaries = [0, *_safe_boundaries(draft), len(draft)]
            left = max(b for b in boundaries if b <= start)
            right = min(b for b in boundaries if b >= end)
            withheld.append({"draftSpanId": span["draft_span_id"], "proposedText": draft[left:right],
                             "reason": "Model marked this block unsupported but retained it; host withholds, not proves it false."})
            draft = draft[:left] + "> [Unverified content withheld by the host; inspect the evidence report before restoring it.]\n\n" + draft[right:]
    validate_fences(draft)
    old, new = previous["draft"].splitlines(keepends=True), draft.splitlines(keepends=True)
    edits = [{"kind": tag, "start": sum(map(len, old[:a])), "end": sum(map(len, old[:b])),
              "replacement": "".join(new[c:d])} for tag, a, b, c, d in SequenceMatcher(None, old, new, autojunk=False).get_opcodes()
             if tag != "equal"]
    if len(edits) > 8:
        raise ValueError("host-computed revision exceeds unchanged eight-change budget")
    materialized = previous["draft"]
    for edit in reversed(edits):
        materialized = materialized[:edit["start"]] + edit["replacement"] + materialized[edit["end"]:]
    if materialized != draft:
        raise ValueError("host-computed revision does not reconstruct the exact candidate")
    notes = ([f"The host withheld {len(withheld)} passage(s) flagged by the model. The flags are not established facts; inspect raw judgments and evidence before restoring content."]
             if withheld else proposal["notes"])
    candidate = {"values": previous["values"], "draft": draft, "notes": notes}
    body = {"status": "host_diff_applied_not_semantic_proof" if edits else "no_material_draft_change",
        "previousCandidateDigest": sha256_json(previous), "candidateDigest": sha256_json(candidate),
        "proposalDigest": sha256_json(proposal), "edits": edits, "editLocationsComputedByHost": True,
        "candidateEvidenceCheck": proposal["evidence_check"], "evidenceCheckIsModelOpinion": True,
        "hostWithholding": withheld, "rawProposalNeededHostWithholding": bool(withheld),
        "rawModelNotes": proposal["notes"], "modelNotesAreNotHostFacts": True,
        "revisionNote": proposal["revision_note"], "mappedValuesPreservedByHost": True,
        "completeAnswerApproved": False, "runtimeAuthorityGranted": False,
        "claimBoundary": "Bounded line-difference accounting, not a semantic edit count or correctness proof."}
    return candidate, {**body, "reportDigest": sha256_json(body)}


def _safe_boundaries(draft):
    """Blank-line boundaries outside code fences; exact original bytes retained."""
    offset, opened = 0, None
    for line in draft.splitlines(keepends=True):
        match = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line.rstrip("\r\n"))
        if match:
            fence, tail = match.groups()
            if opened is None and not (fence[0] == "`" and "`" in tail):
                opened = fence
            elif opened is not None and fence[0] == opened[0] and len(fence) >= len(opened) and not tail.strip():
                opened = None
        offset += len(line)
        if not line.strip() and opened is None:
            yield offset


def section_headings(draft):
    """Located structural labels, not semantic obligations or a Markdown renderer.

    ATX/Setext and standalone non-sentence strong-emphasis labels are protected.
    Ignore fences, quotations, indented code and inline emphasis; retain bytes.
    """
    rows, offset, opened, headings = draft.splitlines(keepends=True), 0, None, []
    consumed_until = 0
    for index, line in enumerate(rows):
        start, offset = offset, offset + len(line)
        plain = line.rstrip("\r\n")
        fence = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", plain)
        if fence:
            marker, tail = fence.groups()
            if opened is None and not (marker[0] == "`" and "`" in tail):
                opened = marker
            elif opened and marker[0] == opened[0] and len(marker) >= len(opened) and not tail.strip():
                opened = None
            continue
        if opened or start < consumed_until or not plain.strip() or plain.startswith(("    ", "\t")):
            continue
        atx = re.match(r"^ {0,3}#{1,6}\s", plain)
        label_line = re.sub(r"^ {0,3}(?:[-+*]|\d+[.)])\s+", "", plain)
        strong = re.fullmatch(r" {0,3}(?:\*\*(.+?)\*\*|__(.+?)__)[ \t]*", label_line)
        label = next((s for s in strong.groups() if s is not None), "") if strong else ""
        strong_label = bool(label and len(label) <= 120 and label[-1] not in ".!?。！？" and "**" not in label and "__" not in label)
        setext = (not re.match(r"^\s*(?:>|(?:\d+[.)]|[-+*])\s)", plain) and index + 1 < len(rows)
                  and re.fullmatch(r" {0,3}(?:=+|-+)[ \t]*", rows[index + 1].rstrip("\r\n")))
        if atx or strong_label or setext:
            end = offset + len(rows[index + 1]) if setext and not atx else offset
            consumed_until = end
            headings.append((start, end, draft[start:end].rstrip("\r\n")))
    return headings


def _section_boundaries(draft):
    """Keep each structural label with its body, never split fenced code."""
    headings = {a: b for a, b, _ in section_headings(draft)}
    offset, heading_end, body_seen = 0, 0, False
    for line in draft.splitlines(keepends=True):
        if offset in headings:
            if body_seen:
                yield offset
            heading_end, body_seen = headings[offset], False
        elif offset >= heading_end and line.strip():
            body_seen = True
        offset += len(line)


def editing_slots(payload, *, complete_sections=False):
    draft = payload["candidate"]["draft"]
    # At most eight whole-block slots. Never cut a fenced code block into an
    # ambiguous fragment the model would need to reconstruct from distant text.
    ends = list(_section_boundaries(draft) if complete_sections else _safe_boundaries(draft))
    ends = sorted(set([*ends, len(draft)]))
    if len(ends) > 8:
        ends = [ends[(i * len(ends) + 7) // 8 - 1] for i in range(1, 9)]
    start, slots = 0, []
    for end in ends:
        slots.append({"id": f"e{len(slots):02d}", "start": start, "end": end, "text": draft[start:end]})
        start = end
    body = {"candidateDigest": payload["completeCandidateDigest"], "slots": slots}
    return {**body, "slotsDigest": sha256_json(body)}


def slot_output_schema(payload, *, complete_sections=False):
    editable = editing_slots(payload, complete_sections=complete_sections)
    edit = obj({
        "action": {"type": "string", "enum": ["keep", "replace"]},
        "replacement": {"type": "string", "maxLength": 12000},
        "source_span_ids": {"type": "array", "maxItems": 8, "uniqueItems": True,
                            "items": {"type": "string", "enum": [s["source_span_id"] for s in payload["sourceSpans"]]}},
        "rationale": {"type": "string", "maxLength": 1200},
    })
    schema = obj({
        "slots_digest": {"type": "string", "enum": [editable["slotsDigest"]]},
        "slots": obj({s["id"]: {"$ref": "#/$defs/edit"} for s in editable["slots"]}),
        "notes": {"type": "array", "maxItems": 6, "items": {"type": "string", "minLength": 1, "maxLength": 2000}},
        "revision_note": {"type": "string", "minLength": 12, "maxLength": 1200},
    })
    schema["$defs"] = {"edit": edit}
    return schema


def validate_fences(draft):
    opened = None
    for line in draft.splitlines():
        match = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if not match:
            continue
        fence, tail = match.groups()
        if opened is None:
            if fence[0] == "`" and "`" in tail:
                continue
            opened = fence
        elif fence[0] == opened[0] and len(fence) >= len(opened) and not tail.strip():
            opened = None
    if opened is not None:
        raise ValueError("revised draft contains an unclosed Markdown code fence")


def apply_slot_revision(payload, values, raw, *, complete_sections=False, preserve_notes=False):
    schema = slot_output_schema(payload, complete_sections=complete_sections)
    if preserve_notes:
        # Immutable host-retained prior notes, not an expanded model-write budget.
        schema["properties"]["notes"] = {"type": "array", "items": {"type": "string"},
                                          "const": payload["candidate"]["notes"]}
    proposal = validate_data(schema, snapshot_json(raw))
    previous = {**payload["candidate"], "values": snapshot_json(values)}
    if sha256_json(previous) != payload["completeCandidateDigest"]:
        raise ValueError("revision candidate binding changed")
    editable = editing_slots(payload, complete_sections=complete_sections)
    pieces, edits = [], []
    for slot in editable["slots"]:
        edit = proposal["slots"][slot["id"]]
        if edit["action"] == "keep":
            if edit["replacement"] or edit["source_span_ids"]:
                raise ValueError("keep slot must not contain a hidden replacement or edit citations")
            pieces.append(slot["text"])
            continue
        if not edit["source_span_ids"] or len(edit["rationale"]) < 12:
            raise ValueError("replace slot requires located sources and explanation")
        if edit["replacement"] == slot["text"]:
            raise ValueError("no-op slot must use keep, not claim a repair")
        pieces.append(edit["replacement"])
        edits.append({**slot, **edit})
    draft = "".join(pieces)
    if len(draft) > 12000:
        raise ValueError("assembled draft exceeds unchanged candidate budget")
    validate_fences(draft)
    candidate = {"values": previous["values"], "draft": draft, "notes": proposal["notes"]}
    body = {"status": "host_slots_applied_not_semantic_proof" if edits else "no_material_draft_change",
            "previousCandidateDigest": sha256_json(previous), "candidateDigest": sha256_json(candidate),
            "proposalDigest": sha256_json(proposal), "slotsDigest": editable["slotsDigest"], "edits": edits,
            "revisionNote": proposal["revision_note"], "mappedValuesPreservedByHost": True,
            "completeAnswerApproved": False, "runtimeAuthorityGranted": False}
    return candidate, {**body, "reportDigest": sha256_json(body)}


def repair_lenses(payload, assessment, raw_review=None, *, complete_sections=False):
    """Derive located edit tasks, not positive reviewer opinions or truth labels.

Source/target quotations and offsets are mechanical evidence. Differences can
be valid paraphrases; relevance/entailment remain an open model/human judgment.
"""
    witnesses = {w["claimId"]: w for w in assessment["draftWitnesses"]}
    units = {c["claimId"]: c for c in payload["claims"]}
    grouped = {}
    for warning in assessment["alignmentWarnings"]:
        grouped.setdefault(warning["claimId"], []).append(warning)
    slots = editing_slots(payload, complete_sections=complete_sections)["slots"]
    lenses = []
    for key, warnings in grouped.items():
        unit, witness = units[key], witnesses[key]
        is_draft = unit["facet"] == "all_claims_in_text_block"
        start, end = (unit["start"], unit["end"]) if is_draft else (witness["start"], witness["end"])
        quote = payload["candidate"]["draft"][start:end] if is_draft else witness["draftQuote"]
        targets = [s["id"] for s in slots if start is not None and s["start"] < end and s["end"] > start]
        lenses.append({"claimId": key, "originalSourceClause": unit["declaredValue"],
            "direction": "draft_to_observations" if is_draft else "observations_to_draft",
            "actualDraftQuote": quote, "editableSlots": targets,
            "unmatchedSourceTerms": sorted({t for w in warnings for t in w.get("terms", w.get("literals", []))}),
            "task": "Check actual draft assertions against observations, not typical defaults. Remove/withhold unsupported facts; preserve legitimate labels, paraphrases and illustrative examples."
                if is_draft else "Inspect the source clause and actual quote. Correct omitted responsibility/condition/value; retain justified paraphrases. Do not replace this check with a reviewer support score."})
    for row in (raw_review or {}).get("claims", []):
        unit = units[row["claim_id"]]
        if row["verdict"] == "supported" or row["claim_id"] in grouped:
            continue
        if unit["facet"] == "all_claims_in_text_block":
            targets = [s["id"] for s in slots if s["start"] < unit["end"] and s["end"] > unit["start"]]
        else:
            targets = []
        lenses.append({"claimId": row["claim_id"], "declaredUnit": unit, "editableSlots": targets,
            "actualDraftQuote": next((s["exactQuote"] for s in payload["draftSpans"]
                                      if s["draft_span_id"] == unit["declaredValue"].get("draftSpanId")), "")
                if isinstance(unit["declaredValue"], dict) else "",
            "sourceSpanIds": row["source_span_ids"], "reportedGapNotEstablishedFact": row["rationale"],
            "untrustedSuggestedRevision": row["suggested_revision"],
            "task": "Verify the reported issue against original evidence before changing text. Reviewer prose is not a fact or permission."})
    return lenses
