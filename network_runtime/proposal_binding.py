"""Strict proposal-only L1 Decision binding for immutable Runtime plans.

This module deliberately does not import :mod:`l1_runtime`.  The effect
Runtime treats a Decision as bounded, untrusted provenance: it verifies the
complete digest chain and mapping, then still applies every normal L0,
identity, approval, preflight, verification, and rollback control.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from .contracts import sha256_json


PLAN_DECISION_BINDING_SCHEMA = "netopyu.io/l1-plan-binding/v1"
_DECISION_SCHEMA = "netopyu.io/l1-decision/v1"
_ENVELOPE_SCHEMA = "netopyu.io/l1-decision-envelope/v1"
_EVIDENCE_SCHEMA = "netopyu.io/l1-decision-evidence/v1"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")

_ENVELOPE_FIELDS = {
    "apiVersion", "decision_id", "mode", "profile", "session_id", "harness",
    "status", "decision", "evidence", "decision_digest", "evidence_digest",
    "authority",
}
_DECISION_FIELDS = {
    "apiVersion", "action", "target", "arguments", "missing_fields", "workflow",
    "confidence", "reason_code",
}
_EVIDENCE_FIELDS = {
    "apiVersion", "prompt_digest", "catalog_digest", "candidate_digest",
    "policy_digest", "model", "model_attempts", "input_tokens", "output_tokens",
    "token_usage_complete", "selected_candidate_index", "candidate_ids",
    "dropped_argument_fields", "normalized_argument_fields", "attempt_error_types",
    "guard_action", "guard_reason", "protocol_valid", "duration_ms",
}
_ROUTE_FIELDS = {"kind", "target"}
_BINDING_FIELDS = {
    "apiVersion", "authority", "mode", "decision_id", "decision_digest",
    "evidence_digest", "profile", "session_id", "harness", "action", "target",
    "route_kind", "route_target", "request_arguments_digest",
    "request_argument_keys", "plan_arguments_digest", "plan_argument_keys",
    "workflow", "prompt_digest", "catalog_digest", "candidate_digest",
    "policy_digest", "bound_tool_name", "bound_l0_skill_id",
    "bound_l0_contract_hash", "created_at", "expires_at", "binding_digest",
}


class ProposalBindingError(ValueError):
    """The supplied L1 proposal cannot be safely bound to this plan."""


def _exact_fields(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ProposalBindingError(f"{label} fields do not match the reviewed contract")


def _require_digest(value: Any, label: str) -> str:
    digest = str(value or "")
    if not _DIGEST.fullmatch(digest):
        raise ProposalBindingError(f"{label} is not a canonical sha256 digest")
    return digest


def _require_identifier(value: Any, label: str) -> str:
    identifier = str(value or "")
    if not _IDENTIFIER.fullmatch(identifier):
        raise ProposalBindingError(f"{label} is not a bounded identifier")
    return identifier


def _require_string(value: Any, label: str, maximum: int) -> str:
    text = str(value or "")
    if not text or len(text) > maximum:
        raise ProposalBindingError(f"{label} is missing or exceeds its bound")
    return text


def compile_plan_decision_binding(
    envelope: dict[str, Any],
    *,
    route_context: dict[str, Any],
    profile: str,
    session_id: str,
    harness: str,
    tool_name: str,
    l0_skill_id: str,
    l0_contract_hash: str,
    request_arguments: dict[str, Any],
    plan_arguments: dict[str, Any],
    created_at: str,
    expires_at: str,
) -> dict[str, Any]:
    """Validate a canary Decision and project a privacy-minimized plan binding."""
    if not isinstance(envelope, dict):
        raise ProposalBindingError("L1 Decision envelope must be an object")
    if not isinstance(route_context, dict):
        raise ProposalBindingError("L1 route context must be an object")
    _exact_fields(envelope, _ENVELOPE_FIELDS, "L1 Decision envelope")
    _exact_fields(route_context, _ROUTE_FIELDS, "L1 route context")
    if envelope.get("apiVersion") != _ENVELOPE_SCHEMA:
        raise ProposalBindingError("unsupported L1 Decision envelope version")
    if envelope.get("authority") != "proposal_only":
        raise ProposalBindingError("L1 Decision cannot carry execution authority")
    if envelope.get("mode") != "canary":
        raise ProposalBindingError("only an explicitly gated canary Decision may bind a plan")
    if envelope.get("status") != "decided":
        raise ProposalBindingError("terminal or failed L1 Decisions cannot bind a plan")
    if envelope.get("profile") != profile:
        raise ProposalBindingError("L1 Decision profile does not match Runtime preparation")
    if envelope.get("session_id") != session_id:
        raise ProposalBindingError("L1 Decision session does not match Runtime preparation")
    if envelope.get("harness") != harness:
        raise ProposalBindingError("L1 Decision harness does not match Runtime preparation")

    decision = envelope.get("decision")
    evidence = envelope.get("evidence")
    if not isinstance(decision, dict) or not isinstance(evidence, dict):
        raise ProposalBindingError("L1 Decision or evidence is missing")
    _exact_fields(decision, _DECISION_FIELDS, "L1 Decision")
    _exact_fields(evidence, _EVIDENCE_FIELDS, "L1 Decision evidence")
    if decision.get("apiVersion") != _DECISION_SCHEMA:
        raise ProposalBindingError("unsupported L1 Decision version")
    if evidence.get("apiVersion") != _EVIDENCE_SCHEMA:
        raise ProposalBindingError("unsupported L1 Decision evidence version")
    decision_digest = _require_digest(envelope.get("decision_digest"), "decision digest")
    evidence_digest = _require_digest(envelope.get("evidence_digest"), "evidence digest")
    if decision_digest != sha256_json(decision):
        raise ProposalBindingError("L1 Decision digest does not bind its payload")
    if evidence_digest != sha256_json(evidence):
        raise ProposalBindingError("L1 evidence digest does not bind its payload")
    if evidence.get("protocol_valid") is not True or evidence.get("guard_action") != "allow":
        raise ProposalBindingError("L1 protocol/Guard did not admit a selectable proposal")

    action = str(decision.get("action") or "")
    if action not in {"select_tool", "select_skill"}:
        raise ProposalBindingError("only a completed selection may bind a plan")
    target = _require_identifier(decision.get("target"), "L1 target")
    missing_fields = decision.get("missing_fields")
    workflow = decision.get("workflow")
    arguments = decision.get("arguments")
    if missing_fields != [] or not isinstance(workflow, list) or not isinstance(arguments, dict):
        raise ProposalBindingError("L1 selection is incomplete or malformed")
    if len(workflow) > 16 or any(not _IDENTIFIER.fullmatch(str(item)) for item in workflow):
        raise ProposalBindingError("L1 workflow exceeds its reviewed identifier bounds")
    if arguments != request_arguments:
        raise ProposalBindingError("Runtime request arguments differ from the grounded L1 proposal")

    route_kind = str(route_context.get("kind") or "")
    route_target = str(route_context.get("target") or "")
    expected_kind = action.removeprefix("select_")
    if route_kind != expected_kind or route_target != target:
        raise ProposalBindingError("observed Harness route differs from the L1 selection")
    if action == "select_tool" and target != tool_name:
        raise ProposalBindingError("selected Tool differs from Runtime preparation")
    if action == "select_skill" and tool_name not in workflow:
        raise ProposalBindingError("Runtime Tool is outside the selected Skill workflow")

    selected_index = evidence.get("selected_candidate_index")
    candidate_ids = evidence.get("candidate_ids")
    if (
        not isinstance(selected_index, int)
        or not isinstance(candidate_ids, list)
        or not 0 <= selected_index < len(candidate_ids) <= 12
        or candidate_ids[selected_index] != f"{expected_kind}:{target}"
    ):
        raise ProposalBindingError("L1 candidate evidence does not bind the selected route")
    prompt_digest = _require_digest(evidence.get("prompt_digest"), "prompt digest")
    catalog_digest = _require_digest(evidence.get("catalog_digest"), "catalog digest")
    candidate_digest = _require_digest(evidence.get("candidate_digest"), "candidate digest")
    policy_digest = _require_digest(evidence.get("policy_digest"), "policy digest")
    decision_id = _require_string(envelope.get("decision_id"), "decision id", 128)
    normalized_session = _require_string(session_id, "session id", 256)
    normalized_harness = _require_string(harness, "harness", 64)
    _require_identifier(tool_name, "bound Tool")
    _require_identifier(l0_skill_id, "bound L0 Skill")
    _require_digest(l0_contract_hash, "bound L0 contract hash")
    try:
        created = datetime.fromisoformat(created_at)
        expires = datetime.fromisoformat(expires_at)
    except ValueError as error:
        raise ProposalBindingError("plan binding timestamps are invalid") from error
    if created.tzinfo is None or expires.tzinfo is None or expires <= created:
        raise ProposalBindingError("plan binding timestamps are not ordered UTC instants")

    body = {
        "apiVersion": PLAN_DECISION_BINDING_SCHEMA,
        "authority": "proposal_only",
        "mode": "canary",
        "decision_id": decision_id,
        "decision_digest": decision_digest,
        "evidence_digest": evidence_digest,
        "profile": profile,
        "session_id": normalized_session,
        "harness": normalized_harness,
        "action": action,
        "target": target,
        "route_kind": route_kind,
        "route_target": route_target,
        "request_arguments_digest": sha256_json(request_arguments),
        "request_argument_keys": sorted(request_arguments),
        "plan_arguments_digest": sha256_json(plan_arguments),
        "plan_argument_keys": sorted(plan_arguments),
        "workflow": list(workflow),
        "prompt_digest": prompt_digest,
        "catalog_digest": catalog_digest,
        "candidate_digest": candidate_digest,
        "policy_digest": policy_digest,
        "bound_tool_name": tool_name,
        "bound_l0_skill_id": l0_skill_id,
        "bound_l0_contract_hash": l0_contract_hash,
        "created_at": created_at,
        "expires_at": expires_at,
    }
    return {**body, "binding_digest": sha256_json(body)}


def verify_persisted_plan_binding(
    binding: dict[str, Any],
    *,
    profile: str,
    tool_name: str,
    l0_skill_id: str,
    l0_contract_hash: str,
    plan_arguments: dict[str, Any],
    requester_identity: dict[str, Any],
    created_at: str,
    expires_at: str,
) -> None:
    """Verify the immutable binding without needing the original L1 package."""
    if not isinstance(binding, dict):
        raise ProposalBindingError("persisted L1 plan binding must be an object")
    _exact_fields(binding, _BINDING_FIELDS, "persisted L1 plan binding")
    digest = _require_digest(binding.get("binding_digest"), "plan binding digest")
    body = dict(binding)
    body.pop("binding_digest")
    if digest != sha256_json(body):
        raise ProposalBindingError("L1 plan binding digest does not match")
    for name in (
        "decision_digest", "evidence_digest", "request_arguments_digest",
        "plan_arguments_digest", "prompt_digest", "catalog_digest",
        "candidate_digest", "policy_digest", "bound_l0_contract_hash",
    ):
        _require_digest(binding.get(name), name)
    if binding.get("apiVersion") != PLAN_DECISION_BINDING_SCHEMA:
        raise ProposalBindingError("unsupported persisted L1 plan binding version")
    if binding.get("authority") != "proposal_only" or binding.get("mode") != "canary":
        raise ProposalBindingError("persisted L1 binding has invalid authority or mode")
    expected = {
        "profile": profile,
        "tool": tool_name,
        "l0_skill": l0_skill_id,
        "l0_hash": l0_contract_hash,
        "session": str(requester_identity.get("session_id") or ""),
        "harness": str(requester_identity.get("harness") or ""),
        "created_at": created_at,
        "expires_at": expires_at,
    }
    actual = {
        "profile": binding.get("profile"),
        "tool": binding.get("bound_tool_name"),
        "l0_skill": binding.get("bound_l0_skill_id"),
        "l0_hash": binding.get("bound_l0_contract_hash"),
        "session": binding.get("session_id"),
        "harness": binding.get("harness"),
        "created_at": binding.get("created_at"),
        "expires_at": binding.get("expires_at"),
    }
    if actual != expected:
        raise ProposalBindingError("persisted L1 binding differs from immutable plan context")
    if binding.get("plan_arguments_digest") != sha256_json(plan_arguments):
        raise ProposalBindingError("persisted L1 binding does not match plan arguments")
    if binding.get("plan_argument_keys") != sorted(plan_arguments):
        raise ProposalBindingError("persisted L1 binding argument keys do not match the plan")


__all__ = [
    "PLAN_DECISION_BINDING_SCHEMA",
    "ProposalBindingError",
    "compile_plan_decision_binding",
    "verify_persisted_plan_binding",
]
