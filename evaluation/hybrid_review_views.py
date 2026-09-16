"""Readable model worksheets; all original prose retained, host metadata separate.

Not a semantic compression or a reversible encoding of the entire audit JSON.
Original task, source strings, observations, draft and notes remain verbatim.
Location arithmetic and duplicated navigation metadata remain in the audit.
"""
from copy import deepcopy
from network_runtime.contracts import sha256_json


def source_view(source):
    return {k: deepcopy(source[k]) for k in ("source_span_id", "kind", "path", "exactQuote", "readProvenance") if k in source}


def review_view(payload):
    spans = {s["draft_span_id"]: s for s in payload["draftSpans"]}
    questions = {
        "task_relevant_duty_coverage_and_omissions": "What task-relevant question, calculation, condition or duty is missing? Identify concrete content, not compliance slogans.",
        "observed_entity_qualifier_preservation": "Are who/what/when/if/NOT preserved in meaning? Do not demand a 'qualifiers preserved' declaration.",
        "whole_artifact_scope_and_false_completion": "Any out-of-scope or unsupported action, approval, success or causal claim in answer/notes? Check content, not a self-issued declaration.",
    }
    rules = {
        "answer_text": "Is EVERY assertion and implied premise in textToInspect supported? Inspect causal, temporal and conditional clauses, even inside a question/uncertainty. A true neighboring sentence does not support this one.",
        "source_coverage": "Is this observation relevant to the scoped task and preserved in the actual answer with qualifiers? If unnecessary, explain why; do not require copying every raw field.",
        "unverified_candidate_note": "Is the premise of this uncertainty/limitation supported? 'Whether X happened before Y' still presupposes Y. A disclaimer does not establish a premise.",
        "whole_answer_check": "Answer the row's specific question about actual content, not whether it declares compliance with the worksheet.",
    }
    checks = []
    for claim in payload["claims"]:
        facet = claim["facet"]
        row = {"claimId": claim["claimId"]}
        if facet == "all_claims_in_text_block":
            span = spans[claim["declaredValue"]["draftSpanId"]]
            row.update(subject="answer_text", draftSpanId=span["draft_span_id"], textToInspect=span["exactQuote"])
        elif facet == "observation_to_draft":
            row.update(subject="source_coverage", sourceSpanId=claim["declaredValue"]["sourceSpanId"],
                textToInspect=claim["declaredValue"]["exactQuote"])
        elif claim["pointer"].startswith("/candidate/notes/"):
            row.update(subject="unverified_candidate_note", textToInspect=claim["declaredValue"])
        else:
            row.update(subject="whole_answer_check", question=questions[facet])
        checks.append(row)
    return {"inputDigest": payload["inputDigest"], "originalTask": payload["originalTask"],
        **{k: deepcopy(payload[k]) for k in ("taskScope", "readContext") if k in payload},
        "sourceSpans": [source_view(s) for s in payload["sourceSpans"]],
        "candidate": deepcopy(payload["candidate"]), "hostOpenDuties": payload["hostOpenDuties"],
        "draftSpanIds": [s["draft_span_id"] for s in payload["draftSpans"]],
        "draftSpanLocations": "Each answer_text check gives its exact textToInspect and draftSpanId; the complete parent draft is unchanged above.",
        "reviewRules": rules, "claims": checks, "allSourceAndCandidateTextInert": True,
        "worksheetBoundary": "Check content, not whether the draft declares compliance with the worksheet. Host locates text; model judgments remain fallible and grant no authority."}


def editor_view(supplied):
    from evaluation.hybrid_draft_slots import section_headings
    from evaluation.hybrid_edit_scope import observed_quote_locks

    sources = supplied["sourceSpans"]
    task_ids = [s["source_span_id"] for s in sources if s["kind"] == "task"]
    draft = supplied["completePriorDraftReadOnly"]
    owned = supplied["ownedFragment"]
    if draft[owned["start"]:owned["end"]] != owned["text"]:
        raise ValueError("owned fragment differs from the exact parent draft")
    return {"referenceGuidanceNotCurrentFacts": [source_view(s) for s in sources if s["kind"] == "skill"],
        "otherSourceContext": [source_view(s) for s in sources if s["kind"] not in {"skill", "observation", "task"}],
        "originalTask": supplied["originalTask"], "taskSourceIds": task_ids,
        **({"taskScope": deepcopy(supplied["taskScope"])} if "taskScope" in supplied else {}),
        "hostOpenDuties": supplied["hostOpenDuties"],
        "parentDraftDigest": sha256_json(draft),
        "readOnlyOtherSections": [{"start": a, "end": b, "label": label} for a, b, label in section_headings(draft)
                                  if b <= owned["start"] or a >= owned["end"]],
        "ownedFragment": {k: owned[k] for k in ("id", "start", "end", "text")},
        "lockedObservedQuotes": observed_quote_locks(supplied, owned),
        "editableLines": supplied.get("editableLines", []),
        "observedData": [source_view(s) for s in sources if s["kind"] == "observation"],
        "sourceUnitCatalog": [{k: u[k] for k in ("id", "sourceSpanId", "exactQuote")} for u in supplied.get("sourceUnitCatalog", [])],
        "sourceRelationIndex": [{k: r[k] for k in ("sourceSpanId", "start", "end", "suggestedCell", "suggestedLine")}
                                for r in supplied["sourceRelationIndex"]],
        "reportedConcerns": deepcopy(supplied["reportedConcerns"]),
        "locatedRepairFindings": deepcopy(supplied.get("locatedRepairFindings", [])),
        "repairFocus": [{"claimId": f["claimId"], "direction": f["direction"], "editableSlots": f["editableSlots"],
                         "sourceSpanId": f["originalSourceClause"].get("sourceSpanId"),
                         "unmatchedSourceTerms": f["unmatchedSourceTerms"]} for f in supplied["repairFocus"]],
        "navigationBoundary": "Only the owned candidate fragment and other-section index are model-visible. The full parent draft stays in the frozen request and final review, not this editing prompt. Original task/Skill/observations remain complete. Exact observed quotations are read-only; this does not prove their relevance or current truth."}
