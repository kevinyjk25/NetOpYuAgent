"""Lossless host-owned task references, not model-written requirement contracts.

Caller roles are navigation only. Exact text survives even when classification
is unavailable/wrong; the host proves provenance, never task interpretation.
"""
from copy import deepcopy

from evaluation import hybrid_duty_contract as legacy
from evaluation.hybrid_task_context import scope_view
from evaluation.structured_authoring import seal

PROFILE = "host-typed-task-contract/v1"
CHECK_SYSTEM = legacy.CHECK_SYSTEM + """
assignedDuty contains the complete exact original task segments, not a paraphrase.
Its role is caller-declared navigation, NOT evidence that every embedded clause has that role.
Review the actual clauses, preserving upper/lower limits, negations and conditions. Do not strengthen them.
Execution prohibitions constrain behavior, not the number/type of answer paragraphs. A draft cannot prove
that an action was/was not executed: absent a relevant execution receipt, that question is unknown.
Do not demand a business answer repeat operational constraints. Source guidance is not current-instance data.
Check ALL assigned clauses; a matching topic or one successful subclause does not establish the whole duty.
"""


def contract(payload):
    source = legacy.source_input(payload)
    task = source["originalTask"]
    scope = source.get("taskScope")
    if scope is None:
        segments = [{"role": "unclassified", "text": task, "start": 0, "end": len(task)}]
    else:
        # Re-derive rather than trust caller-provided offsets or a stale scope seal.
        expected = scope_view(task, [{"role": s["role"], "text": s["text"]} for s in scope["segments"]])
        if expected != scope:
            raise ValueError("typed task scope drift")
        segments = expected["segments"]
    anchors = [{"id": f"task:{i:03d}", "origin": "original_task", **s} for i, s in enumerate(segments)]
    roles = list(dict.fromkeys(a["role"] for a in anchors))
    duties = []
    for i, role in enumerate(roles):
        rows = [deepcopy(a) for a in anchors if a["role"] == role]
        duties.append({"id": f"r{i:03d}", "role": role, "taskAnchors": rows,
                       "interpretation": "Exact original clauses are requirements; role is navigation only."})
    return seal({"profile": PROFILE, "sourceDigest": source["reportDigest"], "taskAnchors": anchors,
        "duties": duties, "overflow": False, "unrepresentedDuties": "",
        "allTaskCharactersRetained": True, "taskAnchorOriginProven": True,
        "semanticClassificationProven": False, "completenessProven": False, "authorityGranted": False})


def context(payload, plan):
    if plan != contract(payload):
        raise ValueError("host task contract drift or model substitution")
    return legacy.context(payload, plan)


def bind_check(ctx, assigned, raw):
    if assigned not in ctx["unverifiedContract"]["duties"]:
        raise ValueError("foreign assigned task anchor")
    return seal({**{k: v for k, v in legacy.bind_check(ctx, assigned, raw).items() if k != "reportDigest"},
        "taskAnchorIds": [a["id"] for a in assigned["taskAnchors"]],
        "exactOriginalTaskIsCriterion": True, "roleIsNotSemanticProof": True})
