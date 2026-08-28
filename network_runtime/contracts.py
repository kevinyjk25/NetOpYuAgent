"""Versioned JSON contracts and state invariants for network execution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any


SCHEMA_VERSION = 6


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PlanState(StrEnum):
    PLAN_READY = "plan_ready"
    APPROVED = "approved"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    VERIFIED_SUCCESS = "verified_success"
    EXECUTION_FAILED = "execution_failed"
    OUTCOME_INDETERMINATE = "outcome_indeterminate"
    ROLLING_BACK = "rolling_back"
    ROLLBACK_VERIFIED = "rollback_verified"
    PRECONDITION_CHANGED = "precondition_changed"
    MANUAL_INTERVENTION_REQUIRED = "manual_intervention_required"
    REJECTED = "rejected"
    EXPIRED = "expired"


TERMINAL_STATES = {
    PlanState.VERIFIED_SUCCESS,
    PlanState.ROLLBACK_VERIFIED,
    PlanState.PRECONDITION_CHANGED,
    PlanState.MANUAL_INTERVENTION_REQUIRED,
    PlanState.REJECTED,
    PlanState.EXPIRED,
}


ALLOWED_TRANSITIONS: dict[PlanState, set[PlanState]] = {
    PlanState.PLAN_READY: {PlanState.APPROVED, PlanState.REJECTED, PlanState.EXPIRED},
    PlanState.APPROVED: {PlanState.EXECUTING, PlanState.EXPIRED},
    PlanState.EXECUTING: {
        PlanState.VERIFYING,
        PlanState.EXECUTION_FAILED,
        PlanState.OUTCOME_INDETERMINATE,
        PlanState.PRECONDITION_CHANGED,
    },
    PlanState.VERIFYING: {
        PlanState.VERIFIED_SUCCESS,
        PlanState.ROLLING_BACK,
        PlanState.MANUAL_INTERVENTION_REQUIRED,
    },
    PlanState.EXECUTION_FAILED: {
        PlanState.ROLLING_BACK,
        PlanState.MANUAL_INTERVENTION_REQUIRED,
    },
    PlanState.OUTCOME_INDETERMINATE: {
        PlanState.VERIFYING,
        PlanState.ROLLING_BACK,
        PlanState.MANUAL_INTERVENTION_REQUIRED,
    },
    PlanState.ROLLING_BACK: {
        PlanState.ROLLBACK_VERIFIED,
        PlanState.MANUAL_INTERVENTION_REQUIRED,
    },
}


class NetworkRuntimeError(RuntimeError):
    code = "network_runtime_error"


class StateTransitionError(NetworkRuntimeError):
    code = "illegal_state_transition"


class PlanIntegrityError(NetworkRuntimeError):
    code = "plan_integrity_error"


class ApprovalError(NetworkRuntimeError):
    code = "approval_error"


class ResourceConflictError(NetworkRuntimeError):
    code = "resource_conflict"


class OutcomeIndeterminateError(NetworkRuntimeError):
    code = "outcome_indeterminate"


@dataclass(frozen=True)
class Evidence:
    evidence_type: str
    source: str
    target: str
    observed_at: str
    value: Any
    fresh: bool = True
    passed: bool | None = None
    predicate: str | None = None
    expected: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedPlan:
    plan_id: str
    profile: str
    tool_name: str
    tool_version: str
    action_type: str
    provider_identity: str
    input_schema_digest: str
    output_schema_digest: str
    capability_id: str
    capability_version: str
    provider_role: str
    arguments: dict[str, Any]
    argument_provenance: dict[str, str]
    targets: tuple[str, ...]
    risk_level: RiskLevel
    risk_reasons: tuple[str, ...]
    preflight: tuple[Evidence, ...]
    verification_contract: str
    rollback_contract: str | None
    l0_skill_id: str
    l0_skill_version: str
    l0_contract_hash: str
    intent_spec: dict[str, Any]
    intent_hash: str
    step_contract: tuple[dict[str, Any], ...]
    workflow_run_id: str | None
    workflow_template_hash: str | None
    created_at: str
    expires_at: str
    plan_hash: str
    state: PlanState = PlanState.PLAN_READY
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        *,
        plan_id: str,
        profile: str,
        tool_name: str,
        tool_version: str,
        action_type: str,
        provider_identity: str,
        input_schema_digest: str,
        output_schema_digest: str,
        capability_id: str,
        capability_version: str,
        provider_role: str,
        arguments: dict[str, Any],
        argument_provenance: dict[str, str],
        targets: tuple[str, ...],
        risk_level: RiskLevel,
        risk_reasons: tuple[str, ...],
        preflight: tuple[Evidence, ...],
        verification_contract: str,
        rollback_contract: str | None,
        l0_skill_id: str,
        l0_skill_version: str,
        l0_contract_hash: str,
        intent_spec: dict[str, Any],
        intent_hash: str,
        step_contract: tuple[dict[str, Any], ...],
        workflow_run_id: str | None,
        workflow_template_hash: str | None,
        created_at: str,
        expires_at: str,
    ) -> "PreparedPlan":
        stable = {
            "schema_version": SCHEMA_VERSION,
            "plan_id": plan_id,
            "profile": profile,
            "tool_name": tool_name,
            "tool_version": tool_version,
            "action_type": action_type,
            "provider_identity": provider_identity,
            "input_schema_digest": input_schema_digest,
            "output_schema_digest": output_schema_digest,
            "capability_id": capability_id,
            "capability_version": capability_version,
            "provider_role": provider_role,
            "arguments": arguments,
            "argument_provenance": argument_provenance,
            "targets": list(targets),
            "risk_level": risk_level.value,
            "risk_reasons": list(risk_reasons),
            "preflight": [item.to_dict() for item in preflight],
            "verification_contract": verification_contract,
            "rollback_contract": rollback_contract,
            "l0_skill_id": l0_skill_id,
            "l0_skill_version": l0_skill_version,
            "l0_contract_hash": l0_contract_hash,
            "intent_spec": intent_spec,
            "intent_hash": intent_hash,
            "step_contract": list(step_contract),
            "workflow_run_id": workflow_run_id,
            "workflow_template_hash": workflow_template_hash,
            "created_at": created_at,
            "expires_at": expires_at,
        }
        return cls(plan_hash=sha256_json(stable), **{
            **stable,
            "risk_level": risk_level,
            "risk_reasons": tuple(risk_reasons),
            "targets": tuple(targets),
            "preflight": tuple(preflight),
            "step_contract": tuple(step_contract),
        })

    def integrity_payload(self) -> dict[str, Any]:
        value = self.to_dict()
        value.pop("plan_hash", None)
        value.pop("state", None)
        return value

    def verify_integrity(self) -> None:
        payload = self.integrity_payload()
        if sha256_json(payload) == self.plan_hash:
            return
        # Earlier plans predate first-class provider capability binding.
        legacy_payload = dict(payload)
        if self.schema_version < 6:
            for field_name in ("capability_id", "capability_version", "provider_role"):
                legacy_payload.pop(field_name, None)
            if self.schema_version == 5 and sha256_json(legacy_payload) == self.plan_hash:
                return
        # Earlier P0.5 plans predate first-class L0 Skill/Intent contracts.
        # Accept their original immutable shape for inspection and explicit
        # rejection, but execution contract revalidation will fail closed.
        if self.schema_version < 5:
            legacy = dict(legacy_payload)
            for field_name in (
                "provider_identity", "input_schema_digest", "output_schema_digest",
            ):
                legacy.pop(field_name, None)
            if self.schema_version >= 4 and sha256_json(legacy) == self.plan_hash:
                return
        if self.schema_version < 4:
            legacy = dict(legacy_payload)
            for field_name in (
                "provider_identity", "input_schema_digest", "output_schema_digest",
            ):
                legacy.pop(field_name, None)
            for field_name in (
                "l0_skill_id", "l0_skill_version", "l0_contract_hash",
                "intent_spec", "intent_hash", "step_contract",
            ):
                legacy.pop(field_name, None)
            if self.schema_version == 1:
                legacy.pop("workflow_run_id", None)
                legacy.pop("workflow_template_hash", None)
            if sha256_json(legacy) == self.plan_hash:
                return
        raise PlanIntegrityError(f"plan {self.plan_id} hash does not match its immutable payload")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "profile": self.profile,
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "action_type": self.action_type,
            "provider_identity": self.provider_identity,
            "input_schema_digest": self.input_schema_digest,
            "output_schema_digest": self.output_schema_digest,
            "capability_id": self.capability_id,
            "capability_version": self.capability_version,
            "provider_role": self.provider_role,
            "arguments": self.arguments,
            "argument_provenance": self.argument_provenance,
            "targets": list(self.targets),
            "risk_level": self.risk_level.value,
            "risk_reasons": list(self.risk_reasons),
            "preflight": [item.to_dict() for item in self.preflight],
            "verification_contract": self.verification_contract,
            "rollback_contract": self.rollback_contract,
            "l0_skill_id": self.l0_skill_id,
            "l0_skill_version": self.l0_skill_version,
            "l0_contract_hash": self.l0_contract_hash,
            "intent_spec": self.intent_spec,
            "intent_hash": self.intent_hash,
            "step_contract": list(self.step_contract),
            "workflow_run_id": self.workflow_run_id,
            "workflow_template_hash": self.workflow_template_hash,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "plan_hash": self.plan_hash,
            "state": self.state.value,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PreparedPlan":
        plan = cls(
            schema_version=int(value.get("schema_version", 1)),
            plan_id=str(value["plan_id"]),
            profile=str(value["profile"]),
            tool_name=str(value["tool_name"]),
            tool_version=str(value["tool_version"]),
            action_type=str(value["action_type"]),
            provider_identity=str(value.get("provider_identity", "legacy-unbound")),
            input_schema_digest=str(value.get("input_schema_digest", "legacy-unbound")),
            output_schema_digest=str(value.get("output_schema_digest", "legacy-unbound")),
            capability_id=str(value.get("capability_id", "legacy-unbound")),
            capability_version=str(value.get("capability_version", "legacy")),
            provider_role=str(value.get("provider_role", "legacy-unbound")),
            arguments=dict(value["arguments"]),
            argument_provenance=dict(value["argument_provenance"]),
            targets=tuple(value["targets"]),
            risk_level=RiskLevel(value["risk_level"]),
            risk_reasons=tuple(value["risk_reasons"]),
            preflight=tuple(Evidence(**item) for item in value["preflight"]),
            verification_contract=str(value["verification_contract"]),
            rollback_contract=value.get("rollback_contract"),
            l0_skill_id=str(value.get("l0_skill_id", "legacy-unbound")),
            l0_skill_version=str(value.get("l0_skill_version", "legacy")),
            l0_contract_hash=str(value.get("l0_contract_hash", "legacy-unbound")),
            intent_spec=dict(value.get("intent_spec") or {}),
            intent_hash=str(value.get("intent_hash", "legacy-unbound")),
            step_contract=tuple(value.get("step_contract") or ()),
            workflow_run_id=value.get("workflow_run_id"),
            workflow_template_hash=value.get("workflow_template_hash"),
            created_at=str(value["created_at"]),
            expires_at=str(value["expires_at"]),
            plan_hash=str(value["plan_hash"]),
            state=PlanState(value.get("state", PlanState.PLAN_READY.value)),
        )
        plan.verify_integrity()
        return plan


@dataclass(frozen=True)
class ExecutionOutcome:
    plan_id: str
    plan_hash: str
    state: PlanState
    result: str | None
    evidence: tuple[Evidence, ...] = field(default_factory=tuple)
    error: str | None = None
    rollback_result: str | None = None

    @property
    def ok(self) -> bool:
        return self.state == PlanState.VERIFIED_SUCCESS

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "plan_id": self.plan_id,
            "plan_hash": self.plan_hash,
            "state": self.state.value,
            "result": self.result,
            "evidence": [item.to_dict() for item in self.evidence],
            "error": self.error,
            "rollback_result": self.rollback_result,
        }

    def terminal_envelope(self) -> dict[str, Any]:
        """Return the only execution result suitable for an LLM or operator UI.

        Provider responses can contain intermediate states such as ``applied``.
        They are intentionally replaced by a digest; only the Runtime terminal
        state and independently collected evidence cross the harness boundary.
        """
        summaries = {
            PlanState.VERIFIED_SUCCESS: "Runtime independently verified the approved effect.",
            PlanState.ROLLBACK_VERIFIED: "The effect did not verify and Runtime proved restoration.",
            PlanState.PRECONDITION_CHANGED: "Execution was stopped because approved state changed.",
            PlanState.MANUAL_INTERVENTION_REQUIRED: "Runtime cannot prove a safe terminal state; operator action is required.",
            PlanState.REJECTED: "The immutable plan was rejected before execution.",
            PlanState.EXPIRED: "The immutable plan expired before execution.",
        }
        return {
            "contract": "netopyu.effect-runtime-terminal@1.0.0",
            "terminal": self.state in TERMINAL_STATES,
            "ok": self.ok,
            "state": self.state.value,
            "plan_id": self.plan_id,
            "plan_hash": self.plan_hash,
            "summary": summaries.get(
                self.state,
                "Runtime reached a terminal state; inspect evidence and error fields.",
            ),
            "evidence": [item.to_dict() for item in self.evidence],
            "error": self.error,
            "compensation": {
                "performed": self.state == PlanState.ROLLBACK_VERIFIED,
                "verified": self.state == PlanState.ROLLBACK_VERIFIED,
            },
            "provider_result_digest": (
                sha256_json({"result": self.result}) if self.result is not None else None
            ),
        }
