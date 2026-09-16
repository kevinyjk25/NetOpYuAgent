"""Addressed negative findings schedule candidate edits; absence is not approval.

Lexical differences and positive AI opinions are inspection aids, not reasons to
regenerate an artifact. Unlocated omissions stay open unless exactly one
host-owned fragment covers the complete draft. That is a unique editing
boundary, not a semantic match. No business authority is created here.
"""


def plan_repairs(payload, slots, review):
    units = {row["claimId"]: row for row in payload["claims"]}
    spans = {row["draft_span_id"]: row for row in payload["draftSpans"]}
    assignments = {slot["id"]: [] for slot in slots}
    whole_owner = (slots[0]["id"] if len(slots) == 1 and slots[0]["start"] == 0
                   and slots[0]["end"] == len(payload["candidate"]["draft"]) else None)
    unresolved = []
    for row in (review or {}).get("claims", []):
        if row["verdict"] == "supported":
            continue
        unit = units[row["claim_id"]]
        if unit["pointer"].startswith("/candidate/notes/"):
            # A note is a different owned field, even when the model echoes a
            # real body span or there is only one complete-draft editing cell.
            # Only a separately owned note editor can consume this concern.
            unresolved.append({"claimId": row["claim_id"], "verdict": row["verdict"],
                "candidatePointer": unit["pointer"], "draftSpanId": "", "locationBasis": "host_note_pointer",
                "rationale": row["rationale"], "sourceSpanIds": row["source_span_ids"],
                "suggestedRevision": row.get("suggested_revision", ""), "semanticProblemProven": False,
                "reason": "No body cell owns a candidate note; only a separately bounded note editor may address it."})
            continue
        # Actual draft-statement addresses belong to the host's claim, never a
        # possibly wrong echoed location in the model's response.
        declared = unit.get("declaredValue")
        key = declared.get("draftSpanId") if isinstance(declared, dict) else None
        if unit["facet"] != "all_claims_in_text_block":
            key = row.get("draft_span_id")
        span = spans.get(key)
        targets = [slot["id"] for slot in slots if span and
                   slot["start"] < span["end"] and slot["end"] > span["start"]]
        finding = {"claimId": row["claim_id"], "verdict": row["verdict"],
                   "rationale": row["rationale"], "sourceSpanIds": row["source_span_ids"],
                   "draftSpanId": key or "", "semanticProblemProven": False,
                   "suggestedRevision": row.get("suggested_revision", ""),
                   "locationBasis": "host_statement_or_model_proposed_span"}
        if len(targets) == 1:
            assignments[targets[0]].append(finding)
        elif not key and whole_owner:
            assignments[whole_owner].append({**finding, "locationBasis": "unique_whole_draft_owner_not_semantic_location"})
        else:
            unresolved.append({**finding, "reason": "No single owned section is located; retain for review, do not broadcast an edit."})
    return {"policy": "located_negative_findings/v3", "assignments": assignments,
            "unlocatedFindings": unresolved, "priorReviewAvailable": review is not None,
            "noFindingMeans": "retain_unverified_candidate_not_semantic_approval",
            "lexicalDifferenceAloneTriggersEdit": False, "authorityGranted": False}
