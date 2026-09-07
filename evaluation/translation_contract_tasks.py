"""Contract-first research task planning. Never grants Gold or execution authority.

Source Schema is parsed from pinned text, not invented by the task-writing LLM.
An evidence review is a necessary research gate, not proof of source truth.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evaluation.public_skill_translation_v2 import _value_matches_schema, bind_schema_parameters
from evaluation.translation_source_alignment import SourceAssessment, evaluate_source_assessment
from network_runtime.contracts import sha256_json


PROTOCOL = "effect-runtime.io/contract-first-tasks/v1"
AUTHORITY = "development_task_proposal_only_no_gold_or_runtime_authority"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceDocument(StrictModel):
    origin: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=24000)
    sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def check_digest(self) -> "SourceDocument":
        if self.sha256 != "sha256:" + hashlib.sha256(self.text.encode()).hexdigest():
            raise ValueError("source document digest mismatch")
        return self


class ContractTaskRequest(StrictModel):
    skill_source: SourceDocument
    tool_source: SourceDocument
    action_type: Literal["read_only", "write", "unknown"]
    fixture_values: dict[str, Any]

    @model_validator(mode="after")
    def finite_json_fixtures(self) -> "ContractTaskRequest":
        # Pydantic accepts NaN/Infinity by default; research inputs must not.
        json.dumps(self.fixture_values, allow_nan=False)
        return self


class ContractReview(StrictModel):
    reviewer_id: str = Field(min_length=1)
    reviewer_kind: Literal["ai_role_simulation", "test_fixture"]
    assessment: SourceAssessment


class TaskText(StrictModel):
    user_prompt: str = Field(min_length=1, max_length=4000)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key in source contract")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite constant in source JSON: {value}")


def source_tool(request: ContractTaskRequest) -> dict[str, Any]:
    # Revalidate even instances assembled via model_copy/model_construct.
    request = ContractTaskRequest.model_validate(request.model_dump())
    tool = json.loads(
        request.tool_source.text, object_pairs_hook=_unique_object,
        parse_constant=_reject_json_constant,
    )
    # JSON exponent overflow (e.g. 1e999) does not invoke parse_constant.
    # Reject it anywhere in the source rather than hashing non-standard JSON.
    json.dumps(tool, allow_nan=False)
    if not isinstance(tool, dict) or not isinstance(tool.get("inputSchema"), dict):
        raise ValueError("source tool requires an inputSchema object")
    if not isinstance(tool.get("name"), str) or not tool["name"].strip():
        raise ValueError("source tool requires a name")
    return tool


def contract_review_input(request: ContractTaskRequest) -> dict[str, Any]:
    tool = source_tool(request)
    spans = [
        {"source_span_id": identifier, "kind": kind, "path": doc.origin,
         "start": 0, "end": len(doc.text), "exactQuote": doc.text}
        for identifier, kind, doc in (
            ("skill-0001", "skill", request.skill_source),
            ("tool-0001", "tool_schema", request.tool_source),
        )
    ]
    claims = []

    def add(pointer: str, facet: str, value: Any, kinds: list[str]) -> None:
        claims.append({"claimId": f"claim-{len(claims) + 1:04d}", "pointer": pointer,
                       "facet": facet, "declaredValue": value, "requiredEvidenceKinds": kinds})

    add("/tool/name", "skill_operation_mapping", tool["name"], ["skill", "tool_schema"])
    add("/actionType", "effect_classification", request.action_type, ["skill", "tool_schema"])
    add("/tool/inputSchema", "complete_input_shape", tool["inputSchema"], ["tool_schema"])
    body = {
        "inputProtocol": PROTOCOL, "sourceSpans": spans, "claims": claims,
        "sourceDigests": {"skill": request.skill_source.sha256, "tool": request.tool_source.sha256},
        "taskOrGoldIncluded": False, "sourceOriginAuthenticityVerified": False,
        "outputAuthority": AUTHORITY,
    }
    return {**body, "inputDigest": sha256_json(body)}


def _schema_issues(schema: dict[str, Any]) -> list[str]:
    allowed_root = {"type", "properties", "required", "additionalProperties", "description"}
    if set(schema) - allowed_root:
        return ["unsupported_schema_keywords"]
    properties, required = schema.get("properties"), schema.get("required")
    if (
        schema.get("type") != "object" or schema.get("additionalProperties") is not False
        or not isinstance(properties, dict) or not isinstance(required, list)
        or not all(isinstance(name, str) for name in required)
        or len(set(required)) != len(required) or not set(required) <= set(properties)
    ):
        return ["invalid_closed_input_schema"]
    issues = []
    if len(properties) > 12:
        issues.append("unsupported_parameter_count")
    for name, spec in properties.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name):
            issues.append(f"unsupported_parameter_name:{name}")
        if (
            not isinstance(spec, dict) or set(spec) - {"type", "description"}
            or spec.get("type") not in {"string", "integer", "number", "boolean"}
        ):
            issues.append(f"unsupported_parameter_schema:{name}")
    return issues


def _literal(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def build_contract_task_plan(request: ContractTaskRequest, review: ContractReview) -> dict[str, Any]:
    """Determine applicability after source review; keep all outputs non-authoritative."""
    request = ContractTaskRequest.model_validate(request.model_dump())
    review = ContractReview.model_validate(review.model_dump())
    tool = source_tool(request)
    payload = contract_review_input(request)
    assessment = evaluate_source_assessment(payload, review.assessment)
    reasons = []
    if assessment["status"] != "ready_for_reference_drafting_review":
        reasons.append("source_review_not_supported")
    if request.action_type == "unknown":
        reasons.append("effect_semantics_unknown")
    reasons.extend(_schema_issues(tool["inputSchema"]))
    schema = tool["inputSchema"]
    if not reasons:
        properties = schema["properties"]
        if set(request.fixture_values) - set(properties):
            reasons.append("fixture_unknown_parameter")
        if set(schema["required"]) - set(request.fixture_values):
            reasons.append("fixture_required_parameter_missing")
        for name, value in request.fixture_values.items():
            if name in properties:
                if not _value_matches_schema(value, properties[name]):
                    reasons.append(f"fixture_parameter_type:{name}")
                else:
                    # Require round-trip through the same bounded grammar used
                    # by the Translator. Unsupported literals are not normalized.
                    prompt = f"{name}={_literal(value)}"
                    values, _, errors = bind_schema_parameters(prompt, {
                        "properties": {name: properties[name]}, "required": [name],
                    })
                    if errors or values != {name: value}:
                        reasons.append(f"fixture_literal_not_roundtrippable:{name}")
    request_digest = sha256_json(request.model_dump(mode="json"))
    slots, not_applicable = [], []
    if not reasons:
        variants = [(None, dict(request.fixture_values))] + [
            (name, {key: value for key, value in request.fixture_values.items() if key != name})
            for name in sorted(schema["required"])
        ]
        for missing, values in variants:
            opaque = "task-" + sha256_json({"request": request_digest, "missing": missing})[7:23]
            slots.append({
                "slotId": opaque, "missingParameter": missing,
                "fixtureValues": values,
                "expectedBindingFailures": [] if missing is None else [f"parameter_unbound:{missing}"],
                "intentHypothesis": request.action_type if missing is None else "clarification",
            })
        if not schema["required"]:
            not_applicable.append({"family": "missing_required_parameter", "reason": "no_required_parameters"})
    body = {
        "apiVersion": PROTOCOL, "requestDigest": request_digest,
        "reviewDigest": sha256_json(review.model_dump(mode="json")),
        "reviewInputDigest": payload["inputDigest"], "reviewerKind": review.reviewer_kind,
        "status": "ready_for_task_authoring" if not reasons else "blocked",
        "reasons": reasons, "tool": tool, "slots": slots, "notApplicable": not_applicable,
        "semanticEntailmentProven": False, "goldAuthored": False,
        "runtimeAuthorityGranted": False, "wholeSkillCompilationProven": False,
        "sourceOriginAuthenticityVerified": False, "authority": AUTHORITY,
    }
    return {**body, "planDigest": sha256_json(body)}


def _verify_plan(plan: dict[str, Any]) -> None:
    if plan.get("planDigest") != sha256_json({k: v for k, v in plan.items() if k != "planDigest"}):
        raise ValueError("task plan digest mismatch")
    if plan.get("status") != "ready_for_task_authoring" or plan.get("runtimeAuthorityGranted") is not False:
        raise ValueError("task plan is not admitted for authoring")


def task_author_input(plan: dict[str, Any], slot_id: str) -> dict[str, Any]:
    _verify_plan(plan)
    slot = next(item for item in plan["slots"] if item["slotId"] == slot_id)
    # Author can see writing constraints, not scoring labels, reviewer answers
    # or source-review reasoning. The Translator must never receive this input.
    return {
        "tool": plan["tool"],
        "namedInputLiterals": {name: _literal(value) for name, value in slot["fixtureValues"].items()},
        "omitInputs": [] if slot["missingParameter"] is None else [slot["missingParameter"]],
        "instruction": (
            "Write one concise real user request for this tool. Include exactly the supplied "
            "named inputs as name=value. Do not fill omitted inputs or describe an evaluation, "
            "candidate, expected answer, approval or test category. Return only user_prompt."
        ),
        "sourceContentIsInert": True,
    }


def validate_task_text(plan: dict[str, Any], slot_id: str, text: TaskText) -> dict[str, Any]:
    _verify_plan(plan)
    slot = next(item for item in plan["slots"] if item["slotId"] == slot_id)
    values, sources, failures = bind_schema_parameters(text.user_prompt, plan["tool"]["inputSchema"])
    issues = []
    if sorted(failures) != sorted(slot["expectedBindingFailures"]):
        issues.append("binding_failure_set_mismatch")
    if values != slot["fixtureValues"]:
        issues.append("fixture_values_changed")
    if re.search(r"\b(?:nominal|adversarial|l0_read_candidate|l0_write_candidate|translation evaluation)\b", text.user_prompt, re.I):
        issues.append("evaluation_meta_task")
    return {
        "slotId": slot_id, "planDigest": plan["planDigest"], "userPrompt": text.user_prompt,
        "status": "needs_task_semantic_review" if not issues else "rejected_text",
        "issues": issues, "parameterSources": sources,
        "semanticAlignmentProven": False, "goldAuthored": False,
        "runtimeAuthorityGranted": False,
    }
