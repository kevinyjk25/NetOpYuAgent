"""Bounded, static checks for explicit-parameter development fixtures.

These checks parse evidence; they do not understand arbitrary natural language,
prove API fidelity, execute quoted commands, or provide reference labels.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any


CONSTRUCT_VERSION = "translation-construct/v3"
_META_TASK = re.compile(
    r"\b(?:generate|create|produce|construct)\s+(?:a\s+|three\s+)?"
    r"(?:(?:l0\s+)?(?:read|write)|translation[- ]development)\s+candidates?\b"
    r"|\bl0_(?:read|write)_candidate\b",
    re.IGNORECASE,
)
_UNRESOLVED = re.compile(r"\$\(|\$\{|<[^<>\n]+>|\{\{[^{}]+\}\}")
# A deliberately small fixture grammar, not a shell/JSON/program parser.
_VALUE = r'''(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^\s,;!?\)\]\}"'=]+)'''
_END = r"(?=$|[\s,;.!?\)\]\}])"


def _scalar(value: str, value_type: str) -> Any:
    if value_type == "string":
        if not value or any(char in value for char in ("\n", "\r", "\x00")):
            raise ValueError("empty or multiline string")
        return value
    if value_type == "boolean" and value in {"true", "false"}:
        return value == "true"
    if value_type == "integer" and re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if value_type == "number" and re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", value):
        number = float(value)
        if math.isfinite(number):
            return number
    raise ValueError("scalar type mismatch")


def inspect_task_parameters(prompt: str, parameters: list[dict[str, Any]]) -> dict[str, Any]:
    """Find typed name=value and narrowly supported name-is-value evidence.

    Values are case-sensitive and independent of the author's example_value.
    Evidence includes original offsets so a reviewer can locate contradictions.
    Arbitrary prose, negation and corrections still require semantic review.
    """

    evidence: list[dict[str, Any]] = []
    missing: list[str] = []
    conflicting: list[str] = []
    invalid: list[str] = []
    unresolved: list[str] = []
    bound: list[str] = []
    for parameter in parameters:
        name = parameter["name"]
        # `service_id` and `service id` are syntactic aliases only.
        alias = re.escape(name).replace("_", r"[ _]")
        prefix = rf"(?<![\w.-]){alias}(?:\s*=\s*|\s+is\s+)"
        assignment = re.compile(prefix + rf"(?P<value>{_VALUE})" + _END)
        values: list[Any] = []
        starts = {match.start() for match in re.finditer(prefix, prompt)}
        for match in assignment.finditer(prompt):
            starts.discard(match.start())
            raw = match.group("value")
            quoted = raw.startswith(('"', "'"))
            if not quoted:
                raw = raw.rstrip(".")  # sentence terminator; quote a literal trailing dot
            item: dict[str, Any] = {
                "parameter": name, "text": match.group(0),
                "start": match.start(), "end": match.end(), "valid": False,
            }
            try:
                value = json.loads(raw) if raw.startswith('"') else (
                    raw[1:-1].replace("\\'", "'").replace("\\\\", "\\")
                    if raw.startswith("'") else raw
                )
                if _UNRESOLVED.search(value):
                    unresolved.append(name)
                    item["issue"] = "unresolved_expression_or_placeholder"
                else:
                    typed = _scalar(value, parameter["value_type"])
                    values.append(typed)
                    item.update(value=typed, valid=True)
            except (ValueError, TypeError):
                invalid.append(name)
                item["issue"] = "invalid_scalar_literal"
            evidence.append(item)
        if starts:
            invalid.append(name)
        if not values:
            missing.append(name)
        elif len({json.dumps(value, sort_keys=True) for value in values}) > 1:
            conflicting.append(name)
        elif name not in invalid and name not in unresolved:
            bound.append(name)
    return {
        "boundParameters": sorted(bound), "missingParameters": sorted(missing),
        "conflictingParameters": sorted(set(conflicting)),
        "invalidParameters": sorted(set(invalid)),
        "unresolvedParameters": sorted(set(unresolved)), "evidence": evidence,
        "arbitraryProseConsistencyVerified": False,
    }


def inspect_construct(bundle: dict[str, Any], source_text: str) -> dict[str, Any]:
    """Return transparent, bounded fixture findings without changing the input."""

    operation = bundle["operation"]
    findings: list[str] = []
    operation_meta = _META_TASK.search(operation["summary"])
    if operation_meta and not _META_TASK.search(source_text):
        findings.append("unsupported_evaluation_meta_operation")
    tasks = []
    for task in bundle["tasks"]:
        result = inspect_task_parameters(task["user_prompt"], operation["parameters"])
        meta = _META_TASK.search(task["user_prompt"])
        result.update(
            slotId=task["slot_id"],
            unsupportedMetaTask=bool(
                meta and task["challenge"] != "failure_or_adversarial"
                and not _META_TASK.search(source_text)
            ),
        )
        tasks.append(result)
    return {
        "version": CONSTRUCT_VERSION,
        "inputClass": "explicit_parameter_fixture",
        "parameterValuePolicy": "typed_evidence_not_example_value_equality",
        "findings": findings, "tasks": tasks,
        "scope": "one_narrow_operation_family_not_whole_skill",
        "catalogProvenance": "author_proposed_inert_fixture_not_verified_source_api",
        "sourceApiSchemaVerified": False,
        "semanticReviewRequired": [
            "business_task_scope_and_source_step_coverage",
            "natural_language_negation_conflicts_and_parameter_omissions",
            "source_api_parameter_names_types_requiredness",
            "effect_count_verification_and_compensation_availability",
        ],
        "semanticAlignmentProven": False,
    }
