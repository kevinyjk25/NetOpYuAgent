"""Trusted subject binding and signed approval proofs for effect execution.

The local verifier deliberately treats the owner-only Harness/Worker boundary
as a simulation trust root.  Production mode never upgrades those local
assertions: it requires an injected enterprise credential verifier and fails
closed when one is not configured.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import stat
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

from .contracts import ApprovalError, RiskLevel, canonical_json, sha256_json


SUBJECT_SCHEMA = "netopyu.subject/v1"
APPROVAL_PROOF_SCHEMA = "netopyu.approval-proof/v1"
POLICY_ID = "netopyu.effect-approval"
POLICY_VERSION = "1.0.0"
LOCAL_MODE = "local-simulation"
ENFORCED_MODE = "enforced"
_MODES = {LOCAL_MODE, ENFORCED_MODE}


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _b64_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


@dataclass(frozen=True)
class SubjectIdentity:
    """A normalized identity after a trusted verifier accepts a credential."""

    subject_id: str
    issuer: str
    harness: str
    session_id: str
    roles: tuple[str, ...]
    scopes: tuple[str, ...]
    purpose: str
    assurance_level: int
    auth_method: str
    authenticated_at: str
    expires_at: str
    credential_id: str
    local_simulation: bool
    gateway_identity: dict[str, Any] = field(default_factory=dict)
    authorization_evidence: dict[str, Any] = field(default_factory=dict)
    subject_attributes: dict[str, Any] = field(default_factory=dict)
    schema: str = SUBJECT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SUBJECT_SCHEMA:
            raise ApprovalError("unsupported subject identity schema")
        if not all((
            self.subject_id.strip(), self.issuer.strip(), self.harness.strip(),
            self.session_id.strip(), self.auth_method.strip(), self.credential_id.strip(),
        )):
            raise ApprovalError("subject identity is incomplete")
        if self.assurance_level < 0 or self.assurance_level > 4:
            raise ApprovalError("subject assurance_level must be between 0 and 4")
        if _parse_time(self.expires_at) <= datetime.now(timezone.utc):
            raise ApprovalError("subject credential expired")
        if _parse_time(self.authenticated_at) > datetime.now(timezone.utc) + timedelta(seconds=30):
            raise ApprovalError("subject authentication time is in the future")
        if not self.roles or not self.scopes:
            raise ApprovalError("subject roles and scopes are required")

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["roles"] = list(self.roles)
        value["scopes"] = list(self.scopes)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SubjectIdentity":
        return cls(
            schema=str(value.get("schema", SUBJECT_SCHEMA)),
            subject_id=str(value.get("subject_id", "")),
            issuer=str(value.get("issuer", "")),
            harness=str(value.get("harness", "")),
            session_id=str(value.get("session_id", "")),
            roles=tuple(sorted({str(item) for item in value.get("roles", []) if str(item)})),
            scopes=tuple(sorted({str(item) for item in value.get("scopes", []) if str(item)})),
            purpose=str(value.get("purpose", "")),
            assurance_level=int(value.get("assurance_level", 0)),
            auth_method=str(value.get("auth_method", "")),
            authenticated_at=str(value.get("authenticated_at", "")),
            expires_at=str(value.get("expires_at", "")),
            credential_id=str(value.get("credential_id", "")),
            local_simulation=bool(value.get("local_simulation", False)),
            gateway_identity=dict(value.get("gateway_identity") or {}),
            authorization_evidence=dict(value.get("authorization_evidence") or {}),
            subject_attributes=dict(value.get("subject_attributes") or {}),
        )


class SubjectVerifier(Protocol):
    def verify(
        self,
        context: dict[str, Any],
        *,
        expected_harness: str,
        expected_session_id: str,
    ) -> SubjectIdentity: ...


class PolicyDecisionPoint(Protocol):
    """External authorization authority; implementations must fail closed."""

    @property
    def binding(self) -> dict[str, Any]: ...

    def decide(self, request: dict[str, Any]) -> dict[str, Any]: ...


class ChangeAuthority(Protocol):
    """Authoritative change-ticket and maintenance-window resolver."""

    @property
    def binding(self) -> dict[str, Any]: ...

    def qualify(self, request: dict[str, Any]) -> dict[str, Any]: ...


class LocalHarnessSubjectVerifier:
    """Normalize an assertion from an owner-only local Harness Adapter."""

    def verify(
        self,
        context: dict[str, Any],
        *,
        expected_harness: str,
        expected_session_id: str,
    ) -> SubjectIdentity:
        now = datetime.now(timezone.utc)
        harness = str(context.get("harness") or expected_harness)
        session_id = str(context.get("session_id") or expected_session_id)
        if harness != expected_harness or session_id != expected_session_id:
            raise ApprovalError("subject Harness/session binding mismatch")
        if context.get("authenticated") is not True:
            raise ApprovalError("subject is not authenticated by the local Harness boundary")
        issuer = str(context.get("issuer") or f"netopyu.local/{harness}")
        if issuer != f"netopyu.local/{harness}":
            raise ApprovalError("local subject issuer does not match the Harness")
        expires_at = str(
            context.get("expires_at")
            or (now + timedelta(minutes=15)).isoformat()
        )
        return SubjectIdentity(
            subject_id=str(context.get("subject_id", "")),
            issuer=issuer,
            harness=harness,
            session_id=session_id,
            roles=tuple(sorted({str(item) for item in context.get("roles", []) if str(item)})),
            scopes=tuple(sorted({str(item) for item in context.get("scopes", []) if str(item)})),
            purpose=str(context.get("purpose") or "interactive-effect-operation"),
            assurance_level=int(context.get("assurance_level", 1)),
            auth_method=str(context.get("auth_method") or "local-harness-process"),
            authenticated_at=str(context.get("authenticated_at") or now.isoformat()),
            expires_at=expires_at,
            credential_id=str(context.get("credential_id") or f"{harness}:{session_id}"),
            local_simulation=True,
        )


class DenyUnconfiguredEnterpriseVerifier:
    """Production-safe default: raw subject dictionaries are never trusted."""

    def verify(
        self,
        context: dict[str, Any],
        *,
        expected_harness: str,
        expected_session_id: str,
    ) -> SubjectIdentity:
        del context, expected_harness, expected_session_id
        raise ApprovalError(
            "enforced identity mode requires a configured enterprise credential verifier"
        )


class DenyUnconfiguredPolicyDecisionPoint:
    @property
    def binding(self) -> dict[str, Any]:
        return {"kind": "unconfigured", "authority": "pdp"}

    def decide(self, request: dict[str, Any]) -> dict[str, Any]:
        del request
        raise ApprovalError("enforced identity mode requires a configured enterprise PDP")


class DenyUnconfiguredChangeAuthority:
    @property
    def binding(self) -> dict[str, Any]:
        return {"kind": "unconfigured", "authority": "change-system"}

    def qualify(self, request: dict[str, Any]) -> dict[str, Any]:
        del request
        raise ApprovalError(
            "an authoritative change ticket requires a configured change-system adapter"
        )


@dataclass(frozen=True)
class ApprovalPolicy:
    policy_id: str = POLICY_ID
    version: str = POLICY_VERSION
    critical_requires_ticket: bool = True
    critical_requires_separation: bool = True

    @property
    def policy_hash(self) -> str:
        return sha256_json(asdict(self))

    @staticmethod
    def _has_scope(subject: SubjectIdentity, *values: str) -> bool:
        scopes = set(subject.scopes)
        return "*" in scopes or any(value in scopes for value in values)

    def authorize_requester(
        self,
        subject: SubjectIdentity,
        *,
        profile: str,
        capability_id: str,
    ) -> None:
        if not set(subject.roles).intersection({
            "network-operator", "service-operator", "change-requester", "system",
        }):
            raise ApprovalError("requester lacks an effect-operation role")
        if not self._has_scope(
            subject, "effect:prepare", f"profile:{profile}", f"capability:{capability_id}",
        ):
            raise ApprovalError("requester lacks effect preparation scope")

    def authorize_approvers(
        self,
        requester: SubjectIdentity,
        approvers: tuple[SubjectIdentity, ...],
        *,
        profile: str,
        capability_id: str,
        risk_level: RiskLevel,
        approval_mode: str,
        enforced: bool,
        change_context: dict[str, Any],
    ) -> dict[str, Any]:
        required = 2 if approval_mode == "dual" else 1
        if len(approvers) != required:
            raise ApprovalError(
                f"approval mode {approval_mode} requires exactly {required} approver(s)"
            )
        ids = [item.subject_id for item in approvers]
        if len(set(ids)) != len(ids):
            raise ApprovalError("approval subjects must be distinct")
        for subject in approvers:
            if not set(subject.roles).intersection({
                "network-approver", "change-approver", "system",
            }):
                raise ApprovalError(f"approver {subject.subject_id} lacks an approval role")
            if not self._has_scope(
                subject, "effect:approve", f"profile:{profile}",
                f"capability:{capability_id}",
            ):
                raise ApprovalError(f"approver {subject.subject_id} lacks approval scope")

        require_separation = approval_mode == "dual" or bool(
            change_context.get("require_separation_of_duties")
        )
        require_ticket = bool(change_context.get("require_change_ticket"))
        if enforced and risk_level == RiskLevel.CRITICAL:
            require_separation = self.critical_requires_separation
            require_ticket = self.critical_requires_ticket
        if require_separation and requester.subject_id in ids:
            raise ApprovalError("separation of duties forbids requester self-approval")
        ticket_id = str(change_context.get("ticket_id") or "").strip()
        if require_ticket and not ticket_id:
            raise ApprovalError("approval policy requires a change ticket")
        window_start = str(change_context.get("window_start") or "").strip()
        window_end = str(change_context.get("window_end") or "").strip()
        now = datetime.now(timezone.utc)
        if bool(window_start) != bool(window_end):
            raise ApprovalError("change window requires both start and end")
        if window_start:
            start = _parse_time(window_start)
            end = _parse_time(window_end)
            if start >= end or not start <= now < end:
                raise ApprovalError("approval is outside the active change window")
        return {
            "required_approvers": required,
            "separation_of_duties": require_separation,
            "ticket_required": require_ticket,
            "ticket_id": ticket_id or None,
            "window_start": window_start or None,
            "window_end": window_end or None,
        }


class ApprovalControlPlane:
    """Bind verified subjects to plans and sign short-lived approval proofs."""

    def __init__(
        self,
        *,
        key_path: str | Path,
        mode: str | None = None,
        verifier: SubjectVerifier | None = None,
        policy: ApprovalPolicy | None = None,
        pdp: PolicyDecisionPoint | None = None,
        change_authority: ChangeAuthority | None = None,
        require_external_controls: bool | None = None,
        signing_key: bytes | None = None,
        proof_ttl_seconds: int = 120,
    ) -> None:
        self.mode = str(mode or os.environ.get("NETOPYU_IDENTITY_MODE", LOCAL_MODE))
        if self.mode not in _MODES:
            raise ValueError(f"unsupported identity mode {self.mode!r}")
        self.verifier = verifier or (
            LocalHarnessSubjectVerifier()
            if self.mode == LOCAL_MODE else DenyUnconfiguredEnterpriseVerifier()
        )
        self.policy = policy or ApprovalPolicy()
        self.require_external_controls = (
            self.enforced if require_external_controls is None
            else bool(require_external_controls)
        )
        self.pdp = pdp or (
            DenyUnconfiguredPolicyDecisionPoint()
            if self.require_external_controls else None
        )
        self.change_authority = change_authority or (
            DenyUnconfiguredChangeAuthority()
            if self.require_external_controls else None
        )
        self.key_path = Path(key_path).expanduser().resolve()
        self._signing_key = signing_key
        if not 30 <= proof_ttl_seconds <= 600:
            raise ValueError("proof_ttl_seconds must be between 30 and 600")
        self.proof_ttl_seconds = proof_ttl_seconds

    @property
    def enforced(self) -> bool:
        return self.mode == ENFORCED_MODE

    @property
    def policy_id(self) -> str:
        return self.policy.policy_id

    @property
    def policy_version(self) -> str:
        return self.policy.version

    @property
    def policy_hash(self) -> str:
        if self.pdp is None and self.change_authority is None:
            return self.policy.policy_hash
        return sha256_json({
            "builtin_policy_hash": self.policy.policy_hash,
            "subject_verifier": getattr(
                self.verifier, "binding",
                {"kind": type(self.verifier).__name__},
            ),
            "pdp": self.pdp.binding if self.pdp is not None else None,
            "change_authority": (
                self.change_authority.binding
                if self.change_authority is not None else None
            ),
            "external_controls_required": self.require_external_controls,
        })

    def _external_decision(
        self,
        *,
        action: str,
        subjects: tuple[SubjectIdentity, ...],
        resource: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.pdp is None:
            return {}
        decision = self.pdp.decide({
            "action": action,
            "subjects": [item.to_dict() for item in subjects],
            "resource": dict(resource),
            "context": dict(context or {}),
        })
        if not isinstance(decision, dict) or decision.get("allow") is not True:
            reason = str(decision.get("reason") or "enterprise PDP denied the request") \
                if isinstance(decision, dict) else "enterprise PDP returned an invalid decision"
            raise ApprovalError(reason)
        required = ("decision_id", "policy_id", "policy_version", "evaluated_at")
        if any(not str(decision.get(name) or "").strip() for name in required):
            raise ApprovalError("enterprise PDP decision evidence is incomplete")
        try:
            evaluated_at = _parse_time(str(decision["evaluated_at"]))
        except (TypeError, ValueError) as error:
            raise ApprovalError("enterprise PDP decision time is invalid") from error
        now = datetime.now(timezone.utc)
        if evaluated_at > now + timedelta(seconds=30) or evaluated_at < now - timedelta(minutes=2):
            raise ApprovalError("enterprise PDP decision is not fresh")
        return dict(decision)

    def _key(self) -> bytes:
        if self._signing_key is not None:
            if len(self._signing_key) < 32:
                raise ApprovalError("approval signing key must contain at least 32 bytes")
            return self._signing_key
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                self.key_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            pass
        else:
            try:
                os.write(descriptor, secrets.token_bytes(32))
            finally:
                os.close(descriptor)
        mode = stat.S_IMODE(self.key_path.stat().st_mode)
        if mode & 0o077:
            raise ApprovalError("approval signing key must be owner-only")
        value = self.key_path.read_bytes()
        if len(value) < 32:
            raise ApprovalError("approval signing key is invalid")
        self._signing_key = value
        return value

    def _verify_subject(
        self,
        context: dict[str, Any],
        *,
        harness: str,
        session_id: str,
    ) -> SubjectIdentity:
        subject = self.verifier.verify(
            dict(context), expected_harness=harness, expected_session_id=session_id,
        )
        if self.enforced and subject.local_simulation:
            raise ApprovalError("local-simulation identity cannot be used in enforced mode")
        if self.enforced and self.require_external_controls and not subject.gateway_identity:
            raise ApprovalError("enforced subject lacks a verified Gateway sender attestation")
        return subject

    def verify_subject_context(
        self,
        context: dict[str, Any],
        *,
        harness: str,
        session_id: str,
    ) -> SubjectIdentity:
        """Verify a subject for adapters and non-effect qualification checks."""
        return self._verify_subject(context, harness=harness, session_id=session_id)

    def authorize_external(
        self,
        *,
        action: str,
        subjects: tuple[SubjectIdentity, ...],
        resource: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Obtain and validate a fresh external decision without executing an effect."""
        return self._external_decision(
            action=action,
            subjects=subjects,
            resource=resource,
            context=context,
        )

    def bind_requester(
        self,
        context: dict[str, Any] | None,
        *,
        harness: str,
        session_id: str,
        profile: str,
        capability_id: str,
    ) -> SubjectIdentity:
        if not session_id.strip():
            if self.enforced:
                raise ApprovalError("enforced identity mode requires a session id")
            session_id = "local-session"
        if context is None:
            if self.enforced:
                raise ApprovalError("enforced identity mode requires a verified requester credential")
            context = {
                "subject_id": "local-effect-runtime",
                "issuer": f"netopyu.local/{harness}",
                "harness": harness,
                "session_id": session_id,
                "roles": ["system"],
                "scopes": ["*"],
                "authenticated": True,
                "auth_method": "implicit-local-compatibility",
                "assurance_level": 0,
                "credential_id": f"implicit:{session_id}",
            }
        subject = self._verify_subject(
            context, harness=harness, session_id=session_id,
        )
        self.policy.authorize_requester(
            subject, profile=profile, capability_id=capability_id,
        )
        decision = self._external_decision(
            action="effect.prepare",
            subjects=(subject,),
            resource={"profile": profile, "capability_id": capability_id},
        )
        if decision:
            subject = replace(subject, authorization_evidence=decision)
        return subject

    def bind_observer(
        self,
        context: dict[str, Any] | None,
        *,
        harness: str,
        session_id: str,
        profile: str,
        capability_id: str,
        arguments: dict[str, Any],
    ) -> SubjectIdentity:
        """Verify and externally authorize one sensitive observation subject."""
        if context is None:
            raise ApprovalError("enforced observation requires a verified subject credential")
        subject = self._verify_subject(
            context, harness=harness, session_id=session_id,
        )
        decision = self._external_decision(
            action="observation.read",
            subjects=(subject,),
            resource={
                "profile": profile,
                "capability_id": capability_id,
                "arguments_hash": sha256_json(arguments),
            },
        )
        if decision:
            subject = replace(subject, authorization_evidence=decision)
        return subject

    def issue_proof(
        self,
        plan: Any,
        *,
        approval_request_id: str,
        approver_contexts: list[dict[str, Any]],
        change_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not approval_request_id.strip():
            raise ApprovalError("approval request id is required")
        requester = SubjectIdentity.from_dict(plan.requester_identity)
        if (
            plan.approval_policy_id != self.policy_id
            or plan.approval_policy_version != self.policy_version
            or plan.approval_policy_hash != self.policy_hash
        ):
            raise ApprovalError("approval policy changed after plan preparation")
        if self.enforced and requester.local_simulation:
            raise ApprovalError("a local-simulation requester cannot enter enforced approval")
        approvers = tuple(
            self._verify_subject(
                item, harness=requester.harness, session_id=requester.session_id,
            )
            for item in approver_contexts
        )
        context = dict(change_context or {})
        builtin_decision = self.policy.authorize_approvers(
            requester,
            approvers,
            profile=plan.profile,
            capability_id=plan.capability_id,
            risk_level=plan.risk_level,
            approval_mode=plan.approval_mode,
            enforced=self.enforced,
            change_context=context,
        )
        pdp_decision = self._external_decision(
            action="effect.approve",
            subjects=approvers,
            resource={
                "profile": plan.profile,
                "capability_id": plan.capability_id,
                "targets": list(plan.targets),
                "risk_level": plan.risk_level.value,
                "plan_id": plan.plan_id,
                "requester": requester.to_dict(),
            },
            context={
                "approval_request_id": approval_request_id,
                "ticket_id": str(context.get("ticket_id") or ""),
            },
        )
        obligations = dict(pdp_decision.get("obligations") or {})
        try:
            required_approvers = int(obligations.get("required_approvers", 0) or 0)
        except (TypeError, ValueError) as error:
            raise ApprovalError("enterprise PDP required_approvers is invalid") from error
        if required_approvers < 0 or required_approvers > 10:
            raise ApprovalError("enterprise PDP required_approvers is out of range")
        for name in ("separation_of_duties", "require_change_ticket"):
            if name in obligations and not isinstance(obligations[name], bool):
                raise ApprovalError(f"enterprise PDP obligation {name!r} must be boolean")
        if required_approvers > len(approvers):
            raise ApprovalError("enterprise PDP requires additional approvers")
        if obligations.get("separation_of_duties") is True and (
            requester.subject_id in {item.subject_id for item in approvers}
        ):
            raise ApprovalError("enterprise PDP requires separation of duties")
        ticket_id = str(context.get("ticket_id") or "").strip()
        ticket_required = bool(
            builtin_decision.get("ticket_required")
            or obligations.get("require_change_ticket")
        )
        if ticket_required and not ticket_id:
            raise ApprovalError("enterprise authorization requires a change ticket")
        change_record: dict[str, Any] = {}
        if ticket_id and self.change_authority is not None:
            change_record = self.change_authority.qualify({
                "ticket_id": ticket_id,
                "requester": requester.to_dict(),
                "approvers": [item.to_dict() for item in approvers],
                "profile": plan.profile,
                "capability_id": plan.capability_id,
                "targets": list(plan.targets),
                "risk_level": plan.risk_level.value,
                "plan_id": plan.plan_id,
            })
        elif ticket_id and self.require_external_controls:
            raise ApprovalError("no authoritative change-system adapter is configured")
        decision: dict[str, Any] = {"builtin": builtin_decision}
        if pdp_decision:
            decision["pdp"] = pdp_decision
        if change_record:
            decision["change_record"] = change_record
        now = datetime.now(timezone.utc)
        expires = min(
            _parse_time(plan.expires_at),
            _parse_time(requester.expires_at),
            now + timedelta(seconds=self.proof_ttl_seconds),
            *(_parse_time(item.expires_at) for item in approvers),
        )
        payload = {
            "schema": APPROVAL_PROOF_SCHEMA,
            "proof_id": str(uuid.uuid4()),
            "approval_request_id": approval_request_id,
            "plan_id": plan.plan_id,
            "plan_hash": plan.plan_hash,
            "requester_digest": requester.digest,
            "requester_subject_id": requester.subject_id,
            "approvers": [item.to_dict() for item in approvers],
            "approval_mode": plan.approval_mode,
            "risk_level": plan.risk_level.value,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_hash": self.policy_hash,
            "policy_decision": decision,
            "issued_at": now.isoformat(),
            "expires_at": expires.isoformat(),
            "issuer": "netopyu.effect-runtime",
            "identity_mode": self.mode,
        }
        encoded = _b64_encode(canonical_json(payload).encode("utf-8"))
        signature = _b64_encode(hmac.new(
            self._key(), encoded.encode("ascii"), hashlib.sha256,
        ).digest())
        token = f"{encoded}.{signature}"
        return {
            "approval_proof": token,
            "evidence": self._public_evidence(payload),
        }

    def issue_local_compatibility_proof(
        self,
        plan: Any,
        *,
        approval_request_id: str,
        approval_actor: str,
    ) -> dict[str, Any]:
        if self.enforced:
            raise ApprovalError("legacy approval actor strings are disabled in enforced mode")
        requester = SubjectIdentity.from_dict(plan.requester_identity)
        return self.issue_proof(
            plan,
            approval_request_id=approval_request_id,
            approver_contexts=[{
                "subject_id": approval_actor,
                "issuer": f"netopyu.local/{requester.harness}",
                "harness": requester.harness,
                "session_id": requester.session_id,
                "roles": ["network-approver", "system"],
                "scopes": ["*"],
                "purpose": "legacy-local-approval-compatibility",
                "authenticated": True,
                "auth_method": "implicit-local-compatibility",
                "assurance_level": 0,
                "credential_id": f"legacy:{approval_request_id}",
            }],
        )

    def verify_proof(self, token: str, plan: Any) -> dict[str, Any]:
        try:
            encoded, supplied_signature = token.split(".", 1)
            expected_signature = _b64_encode(hmac.new(
                self._key(), encoded.encode("ascii"), hashlib.sha256,
            ).digest())
            if not hmac.compare_digest(supplied_signature, expected_signature):
                raise ApprovalError("approval proof signature is invalid")
            payload = json.loads(_b64_decode(encoded))
        except ApprovalError:
            raise
        except Exception as error:
            raise ApprovalError("approval proof is malformed") from error
        if not isinstance(payload, dict) or payload.get("schema") != APPROVAL_PROOF_SCHEMA:
            raise ApprovalError("approval proof schema is invalid")
        if plan.approval_policy_hash != self.policy_hash:
            raise ApprovalError("approval policy changed after proof issuance")
        if payload.get("plan_id") != plan.plan_id or payload.get("plan_hash") != plan.plan_hash:
            raise ApprovalError("approval proof is bound to a different plan")
        requester = SubjectIdentity.from_dict(plan.requester_identity)
        if payload.get("requester_digest") != requester.digest:
            raise ApprovalError("approval proof requester binding is invalid")
        if payload.get("policy_hash") != plan.approval_policy_hash:
            raise ApprovalError("approval proof policy binding is invalid")
        if (
            payload.get("policy_id") != plan.approval_policy_id
            or payload.get("policy_version") != plan.approval_policy_version
        ):
            raise ApprovalError("approval proof policy identity is invalid")
        if payload.get("approval_mode") != plan.approval_mode:
            raise ApprovalError("approval proof mode binding is invalid")
        if payload.get("risk_level") != plan.risk_level.value:
            raise ApprovalError("approval proof risk binding is invalid")
        if payload.get("identity_mode") != self.mode:
            raise ApprovalError("approval proof identity mode is invalid")
        if _parse_time(str(payload.get("expires_at", ""))) <= datetime.now(timezone.utc):
            raise ApprovalError("approval proof expired")
        if _parse_time(str(payload.get("issued_at", ""))) > datetime.now(timezone.utc) + timedelta(seconds=30):
            raise ApprovalError("approval proof issue time is in the future")
        return self._public_evidence(payload)

    @staticmethod
    def _public_evidence(payload: dict[str, Any]) -> dict[str, Any]:
        evidence = dict(payload)
        evidence["proof_hash"] = sha256_json(payload)
        return evidence


def local_subject_context(
    subject_id: str,
    *,
    harness: str,
    session_id: str,
    roles: tuple[str, ...] = ("network-operator", "network-approver"),
    scopes: tuple[str, ...] = ("*",),
) -> dict[str, Any]:
    """Build an explicit local-only context for tests, demos and adapters."""
    now = datetime.now(timezone.utc)
    return {
        "subject_id": subject_id,
        "issuer": f"netopyu.local/{harness}",
        "harness": harness,
        "session_id": session_id,
        "roles": list(roles),
        "scopes": list(scopes),
        "purpose": "interactive-effect-operation",
        "authenticated": True,
        "auth_method": "local-harness-process",
        "assurance_level": 1,
        "authenticated_at": now.isoformat(),
        "expires_at": (now + timedelta(minutes=15)).isoformat(),
        "credential_id": f"{harness}:{session_id}:{subject_id}",
    }
