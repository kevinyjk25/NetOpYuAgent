"""Fail-closed authorization for observation capabilities.

This is deliberately separate from effect approval.  Read-only operations can
still disclose restricted topology/configuration or overload a provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .capabilities import CapabilityContract, CapabilityKind, DataSensitivity


_SENSITIVITY_RANK = {
    DataSensitivity.PUBLIC: 0,
    DataSensitivity.INTERNAL: 1,
    DataSensitivity.CONFIDENTIAL: 2,
    DataSensitivity.RESTRICTED: 3,
}


@dataclass(frozen=True)
class ObservationAccessContext:
    subject_id: str
    roles: frozenset[str]
    scopes: frozenset[str]
    purpose: str
    clearance: DataSensitivity = DataSensitivity.INTERNAL
    authenticated: bool = True
    session_id: str | None = None
    implicit_local_context: bool = False

    @classmethod
    def local_system(cls, profile: str) -> "ObservationAccessContext":
        """Compatibility principal for owner-only local prototype calls."""
        return cls(
            subject_id="local-effect-runtime",
            roles=frozenset({"system"}),
            scopes=frozenset({"*", f"profile:{profile}"}),
            purpose="local-runtime-compatibility",
            clearance=DataSensitivity.RESTRICTED,
            implicit_local_context=True,
        )

    @classmethod
    def from_value(
        cls, value: "ObservationAccessContext | dict[str, Any] | None", *, profile: str,
    ) -> "ObservationAccessContext":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.local_system(profile)
        raw_clearance = str(value.get("clearance") or "internal")
        try:
            clearance = DataSensitivity(raw_clearance)
        except ValueError as error:
            raise ValueError(f"invalid observation clearance {raw_clearance!r}") from error
        return cls(
            subject_id=str(value.get("subject_id") or "").strip(),
            roles=frozenset(str(item) for item in value.get("roles", ()) if str(item).strip()),
            scopes=frozenset(str(item) for item in value.get("scopes", ()) if str(item).strip()),
            purpose=str(value.get("purpose") or "").strip(),
            clearance=clearance,
            authenticated=bool(value.get("authenticated", True)),
            session_id=(str(value["session_id"]) if value.get("session_id") else None),
        )


@dataclass(frozen=True)
class ObservationAccessDecision:
    allowed: bool
    code: str
    reason: str
    subject_id: str
    capability_id: str
    evaluated_scopes: tuple[str, ...]


class ObservationPolicy:
    """Small policy-enforcement point; an enterprise PDP can replace it later."""

    def authorize(
        self,
        contract: CapabilityContract,
        arguments: dict[str, Any],
        context: ObservationAccessContext,
    ) -> ObservationAccessDecision:
        evaluated = tuple(
            f"{field}:{arguments[field]}"
            for field in contract.scope_fields
            if field in arguments and arguments[field] not in (None, "")
        )
        denied: tuple[str, str] | None = None
        if contract.kind != CapabilityKind.OBSERVATION:
            denied = ("not_observation", "capability is not read-only")
        elif not context.authenticated or not context.subject_id:
            denied = ("unauthenticated", "an authenticated observation subject is required")
        elif contract.required_roles and "system" not in context.roles and not (
            set(contract.required_roles) & set(context.roles)
        ):
            denied = ("role_denied", "subject has none of the required observation roles")
        elif _SENSITIVITY_RANK[context.clearance] < _SENSITIVITY_RANK[contract.sensitivity]:
            denied = ("clearance_denied", "subject clearance is below data sensitivity")
        elif contract.sensitivity in {
            DataSensitivity.CONFIDENTIAL, DataSensitivity.RESTRICTED,
        } and not context.purpose:
            denied = ("purpose_required", "sensitive observation requires a declared purpose")
        elif "system" not in context.roles and "*" not in context.scopes:
            missing = [scope for scope in evaluated if scope not in context.scopes]
            if missing:
                denied = ("scope_denied", "subject scope does not cover: " + ", ".join(missing))
        if denied:
            return ObservationAccessDecision(
                False, denied[0], denied[1], context.subject_id,
                contract.capability_id, evaluated,
            )
        return ObservationAccessDecision(
            True, "allowed", "observation policy allowed the request",
            context.subject_id, contract.capability_id, evaluated,
        )

