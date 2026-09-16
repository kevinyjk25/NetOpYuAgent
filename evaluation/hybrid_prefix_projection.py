"""Recover only a valid read prefix from an untrusted failed proposal.

Separate non-authoritative annotations from executable prefix validation.
Never invent a binding, repair a literal, admit a rejected read or skip over an
invalid predecessor. Retain the original failed response and every dropped row.
This produces a new review candidate, never permission or a success verdict.
"""
from copy import deepcopy

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from evaluation import hybrid_authoring as author
from evaluation.structured_authoring import seal


def project(packet, visible, raw):
    original = deepcopy(raw)
    if not isinstance(raw, dict) or set(raw) != {"mode", "intent_summary", "reads", "boundaries"} or raw["mode"] != "read_prefix":
        raise ValueError("only the exact read-prefix envelope can be projected")
    if not isinstance(raw["reads"], list) or not 1 <= len(raw["reads"]) <= 7 or not isinstance(raw["boundaries"], list) or len(raw["boundaries"]) > 20:
        raise ValueError("bounded original prefix and annotation arrays required")
    schema = author.author_response_schema(packet, visible)
    proposal = next((s for s in schema.get("anyOf", []) if s.get("properties", {}).get("mode", {}).get("const") == "read_prefix"), schema)
    Draft202012Validator(proposal["properties"]["intent_summary"]).validate(raw["intent_summary"])
    check_boundary = Draft202012Validator(proposal["properties"]["boundaries"]["items"])
    kept_annotations, quarantined = [], []
    for index, annotation in enumerate(raw["boundaries"]):
        try:
            check_boundary.validate(annotation)
            kept_annotations.append(deepcopy(annotation))
        except ValidationError as error:
            quarantined.append({"index": index, "value": deepcopy(annotation), "reason": str(error)[:1200]})
    kept, rejected, first_failure = [], [], None
    for index, read in enumerate(raw["reads"]):
        if first_failure is not None:
            rejected.append({"index": index, "value": deepcopy(read), "reason": "after the first invalid prefix operation; not independently skipped over"})
            continue
        probe = {"mode": "read_prefix", "intent_summary": raw["intent_summary"], "reads": [*kept, deepcopy(read)], "boundaries": []}
        try:
            author.compile_proposal(packet, visible, probe)
            kept.append(deepcopy(read))
        except (ValueError, KeyError, TypeError, ValidationError) as error:
            first_failure = index
            rejected.append({"index": index, "value": deepcopy(read), "reason": str(error)[:1200]})
    if not kept or not (quarantined or rejected):
        raise ValueError("projection requires a nonempty valid prefix and a recorded repair need")
    # Existing valid annotations are still model opinions. The host notice adds
    # no business prerequisite, fact, path, argument or execution permission.
    if rejected:
        if len(kept_annotations) == 20:
            quarantined.append({"index": raw["boundaries"].index(kept_annotations[-1]),
                "value": kept_annotations[-1], "reason": "Retained in audit; reserve one bounded annotation slot for the rejected-prefix notice."})
        kept_annotations = kept_annotations[:19] + [{"evidence": ["task"], "kind": "unsupported_control", "explanation":
            "The host rejected part of this proposed read prefix. Rejected operations were not executed and provide no observations. The exact original L1 task remains; unavailable evidence and duties stay unresolved unless independently authorized observations establish them."}]
    projected = {"mode": "read_prefix", "intent_summary": raw["intent_summary"], "reads": kept, "boundaries": kept_annotations}
    compilation = author.compile_proposal(packet, visible, projected)
    return seal({"status": "offline_projected_candidate_requires_review", "originalChoice": original,
        "projectedChoice": projected, "quarantinedAnnotations": quarantined, "rejectedReadSuffix": rejected,
        "compilation": compilation, "newModelCalls": 0, "automaticPermission": False,
        "sourceLiteralOrBindingRepaired": False, "wholeSkillSemanticSuccess": None,
        "evidenceRole": "host_projection_of_untrusted_proposal_not_semantic_acceptance"})
