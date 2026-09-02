"""Authority-free, monotonic policy for a future P1.9 canary.

The policy may preserve or block the route already chosen by a Harness.  It
cannot replace that route, rewrite arguments, create a Runtime plan, approve
an effect, or call a Provider.  Invalid Decision material is fail-closed for
writes and observation-only for reads so a canary cannot introduce a new
read-side availability dependency.
"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Literal

from pydantic import Field, model_validator

from .contracts import L1DecisionAction, L1DecisionEnvelope, StrictModel


CANARY_POLICY_RESULT_SCHEMA = "netopyu.io/l1-canary-policy-result/v1"


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{sha256(encoded).hexdigest()}"


class CanaryRoute(StrictModel):
    """The immutable route selected independently by the Harness."""

    kind: Literal["skill", "tool"]
    target: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    operation: Literal["read", "write"]
    profile: Literal["lan", "dc", "wan"]
    harness: str = Field(min_length=1, max_length=64)
    session_id: str = Field(min_length=1, max_length=256)


class CanaryPolicyResult(StrictModel):
    """Privacy-minimized decision; ``authority_granted`` is always false."""

    api_version: Literal[CANARY_POLICY_RESULT_SCHEMA] = Field(
        default=CANARY_POLICY_RESULT_SCHEMA, alias="apiVersion",
    )
    status: Literal["continue_original_route", "blocked"]
    effect: Literal["unchanged", "narrowed"]
    route_digest: str
    decision_digest: str | None
    evidence_digest: str | None
    reason_code: Literal[
        "selection_matches_original_route",
        "decision_requires_clarification",
        "decision_refused_route",
        "decision_out_of_scope",
        "selection_mismatch",
        "canary_context_mismatch",
        "protocol_failure_write_blocked",
        "invalid_decision_write_blocked",
        "protocol_failure_read_observed",
        "invalid_decision_read_observed",
    ]
    authority_granted: Literal[False] = False
    route_rewritten: Literal[False] = False
    arguments_rewritten: Literal[False] = False
    runtime_admission_required: Literal[True] = True
    policy_digest: str

    @model_validator(mode="after")
    def validate_monotonic_result(self) -> "CanaryPolicyResult":
        if self.status == "continue_original_route" and self.effect != "unchanged":
            raise ValueError("a continued route must remain unchanged")
        if self.status == "blocked" and self.effect != "narrowed":
            raise ValueError("a blocked route must be a narrowing")
        body = self.model_dump(
            by_alias=True,
            mode="json",
            exclude={"api_version", "policy_digest"},
        )
        if self.policy_digest != _digest(body):
            raise ValueError("canary policy digest does not bind the result")
        return self


def _result(
    *,
    route: CanaryRoute,
    envelope: L1DecisionEnvelope | None,
    status: Literal["continue_original_route", "blocked"],
    reason_code: str,
) -> CanaryPolicyResult:
    body = {
        "route_digest": _digest(route.model_dump(mode="json")),
        "decision_digest": envelope.decision_digest if envelope else None,
        "evidence_digest": envelope.evidence_digest if envelope else None,
        "status": status,
        "effect": "unchanged" if status == "continue_original_route" else "narrowed",
        "reason_code": reason_code,
        "authority_granted": False,
        "route_rewritten": False,
        "arguments_rewritten": False,
        "runtime_admission_required": True,
    }
    return CanaryPolicyResult(**body, policy_digest=_digest(body))


def evaluate_canary_policy(
    raw_envelope: dict[str, Any],
    *,
    route: CanaryRoute | dict[str, Any],
) -> CanaryPolicyResult:
    """Apply a non-escalating L1 proposal to an existing Harness route.

    This function is deliberately pure and has no Runtime, Provider, adapter,
    environment, filesystem, or network side effects.
    """
    original = route if isinstance(route, CanaryRoute) else CanaryRoute.model_validate(route)
    try:
        envelope = L1DecisionEnvelope.model_validate(raw_envelope)
    except (TypeError, ValueError):
        return _result(
            route=original,
            envelope=None,
            status="blocked" if original.operation == "write" else "continue_original_route",
            reason_code=(
                "invalid_decision_write_blocked"
                if original.operation == "write" else "invalid_decision_read_observed"
            ),
        )

    context_matches = (
        envelope.mode == "canary"
        and envelope.authority == "proposal_only"
        and envelope.profile == original.profile
        and envelope.harness == original.harness
        and envelope.session_id == original.session_id
    )
    if not context_matches:
        return _result(
            route=original,
            envelope=envelope,
            status="blocked" if original.operation == "write" else "continue_original_route",
            reason_code=(
                "canary_context_mismatch"
                if original.operation == "write" else "invalid_decision_read_observed"
            ),
        )

    if (
        envelope.status == "protocol_failure"
        or envelope.decision is None
        or not envelope.evidence.protocol_valid
    ):
        return _result(
            route=original,
            envelope=envelope,
            status="blocked" if original.operation == "write" else "continue_original_route",
            reason_code=(
                "protocol_failure_write_blocked"
                if original.operation == "write" else "protocol_failure_read_observed"
            ),
        )

    decision = envelope.decision
    if decision.action == L1DecisionAction.CLARIFY:
        return _result(
            route=original,
            envelope=envelope,
            status="blocked",
            reason_code="decision_requires_clarification",
        )
    if decision.action == L1DecisionAction.REFUSE or envelope.evidence.guard_action == "refuse":
        return _result(
            route=original,
            envelope=envelope,
            status="blocked",
            reason_code="decision_refused_route",
        )
    if (
        decision.action == L1DecisionAction.OUT_OF_SCOPE
        or envelope.evidence.guard_action == "out_of_scope"
    ):
        return _result(
            route=original,
            envelope=envelope,
            status="blocked",
            reason_code="decision_out_of_scope",
        )

    expected_kind = decision.action.value.removeprefix("select_")
    selected_index = envelope.evidence.selected_candidate_index
    candidate_ids = envelope.evidence.candidate_ids
    selection_is_bound = (
        selected_index is not None
        and selected_index < len(candidate_ids)
        and candidate_ids[selected_index] == f"{expected_kind}:{decision.target}"
    )
    if (
        not selection_is_bound
        or expected_kind != original.kind
        or decision.target != original.target
        or envelope.evidence.guard_action != "allow"
    ):
        return _result(
            route=original,
            envelope=envelope,
            status="blocked",
            reason_code="selection_mismatch",
        )
    return _result(
        route=original,
        envelope=envelope,
        status="continue_original_route",
        reason_code="selection_matches_original_route",
    )


__all__ = [
    "CANARY_POLICY_RESULT_SCHEMA",
    "CanaryPolicyResult",
    "CanaryRoute",
    "evaluate_canary_policy",
]
