"""Separate legacy authoring envelope from the independently frozen task.

No heuristic instruction stripping. Only the exact old stage2_cases wrapper is
recognized; unknown additions remain errors. Original artifacts stay intact.
"""
from __future__ import annotations

from network_runtime.contracts import sha256_json

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
