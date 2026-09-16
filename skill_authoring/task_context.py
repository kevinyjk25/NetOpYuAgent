"""Separate legacy authoring envelope from the independently frozen task.

No heuristic instruction stripping. Only the exact old stage2_cases wrapper is
recognized; unknown additions remain errors. Original artifacts stay intact.
"""
from __future__ import annotations

from network_runtime.contracts import sha256_json

ROLES = {"business_request", "execution_constraint", "delivery_constraint"}


def scope_view(original, segments):
    """Lossless caller-declared roles, never inferred instruction removal.

    Classification provides navigation, not privilege: every original character
    must be present in order and every constraint remains in the model request.
    """
    if not isinstance(original, str) or not isinstance(segments, list) or not 1 <= len(segments) <= 8:
        raise ValueError("one to eight explicitly declared task segments required")
    result, offset = [], 0
    for item in segments:
        if (not isinstance(item, dict) or set(item) != {"role", "text"}
                or not isinstance(item["role"], str) or item["role"] not in ROLES
                or not isinstance(item["text"], str) or not item["text"].strip()):
            raise ValueError("task segment requires a known role and nonempty exact text")
        end = offset + len(item["text"])
        if original[offset:end] != item["text"]:
            raise ValueError("task scope must preserve all original text in exact order")
        result.append({**item, "start": offset, "end": end})
        offset = end
    if offset != len(original) or not any(s["role"] == "business_request" for s in result):
        raise ValueError("complete task and at least one business request required")
    return {"profile": "lossless_caller_task_scope/v1", "originalTaskDigest": sha256_json(original),
            "segments": result, "classificationIsNotAuthority": True,
            "constraintsRemoved": False, "semanticClassificationProven": False}

PREFIX = "Offline authoring only; do not execute. Future user task: "
SUFFIX = (" Preserve source duties and unresolved boundaries. Future host permission and parameter checks "
          "remain required, not assumed satisfied during construction.")


def separate(original, business_task):
    if not isinstance(business_task, str) or not 12 <= len(business_task) <= 4000:
        raise ValueError("an independently frozen business task is required")
    if original not in {business_task, PREFIX + business_task + SUFFIX}:
        raise ValueError("unknown authoring wrapper; never strip arbitrary instructions")
    return business_task, {"originalAuthorTaskDigest": sha256_json(original), "businessTaskDigest": sha256_json(business_task),
        "normalization": "none" if original == business_task else "exact_legacy_compilation_envelope_separated",
        "legacyAuthoringInstructions": None if original == business_task else {"prefix": PREFIX, "suffix": SUFFIX},
        "sourceRestrictionsRemoved": False, "businessTaskChanged": False}
