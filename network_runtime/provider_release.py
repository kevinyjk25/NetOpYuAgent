"""Signed Provider release bundles, trust policy, and durable activation state."""

from __future__ import annotations

import base64
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .capabilities import CapabilityContract
from .contracts import canonical_json, sha256_json
from .l0.models import IDENTIFIER, SEMVER


MANIFEST_SCHEMA = "netopyu.io/provider-manifest/v1"
QUALIFICATION_SCHEMA = "netopyu.io/provider-qualification/v1"
SIGNATURE_SCHEMA = "netopyu.io/provider-signature/v1"
BUNDLE_SCHEMA = "netopyu.io/provider-release-bundle/v1"
TRUST_SCHEMA = "netopyu.io/provider-trust/v1"
DEPLOYMENT_SCHEMA = "netopyu.io/provider-deployment-attestation/v1"
SIGNED_DEPLOYMENT_SCHEMA = "netopyu.io/provider-signed-deployment/v1"
QUALIFICATION_SUITE_VERSION = "1.0.0"
REQUIRED_QUALIFICATION_CHECKS = frozenset({
    "identity_and_schema_binding",
    "timeout_before_send_is_safe",
    "duplicate_operation_is_idempotent",
    "out_of_order_operation_is_rejected",
    "partial_success_is_reconciled",
    "unknown_terminal_is_not_blindly_retried",
    "compensation_restores_baseline",
    "compensation_failure_escalates",
    "restart_recovery_preserves_operation_state",
})
_DIGEST_PREFIX = "sha256:"
_RELEASE_STATES = {"staged", "published", "deprecated"}


class ProviderReleaseError(RuntimeError):
    """A release is untrusted, incompatible, or in an invalid lifecycle state."""


def _parse_time(value: str, *, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _validate_digest(value: str, *, label: str) -> str:
    if (
        not value.startswith(_DIGEST_PREFIX)
        or len(value) != len(_DIGEST_PREFIX) + 64
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _version(value: str) -> tuple[int, int, int]:
    match = SEMVER.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid semantic version {value!r}")
    return tuple(int(item) for item in match.groups())  # type: ignore[return-value]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class ReleasedCapability(StrictModel):
    tool_name: str
    capability_id: str
    capability_version: str
    domain: str
    kind: Literal["observation", "effect"]
    action_type: str
    effect_semantics: Literal["none", "reversible", "destructive", "irreversible"]
    provider_role: str
    provider_identity: str
    provider_kind: str
    input_schema_digest: str
    output_schema_digest: str
    sensitivity: Literal["public", "internal", "confidential", "restricted"]
    required_roles: tuple[str, ...] = ()
    scope_fields: tuple[str, ...] = ()
    freshness_limit_seconds: int = 300
    result_contract: str
    l0_contract_hashes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_contract(self) -> "ReleasedCapability":
        if not self.tool_name.strip() or not IDENTIFIER.fullmatch(self.capability_id):
            raise ValueError("released capability identity is invalid")
        _version(self.capability_version)
        _validate_digest(self.input_schema_digest, label="input_schema_digest")
        _validate_digest(self.output_schema_digest, label="output_schema_digest")
        for value in self.l0_contract_hashes:
            _validate_digest(value, label="l0_contract_hash")
        if self.kind == "effect" and not self.l0_contract_hashes:
            raise ValueError("effect capability requires at least one reviewed L0 contract hash")
        if self.kind == "observation" and self.effect_semantics != "none":
            raise ValueError("observation capability must have effect_semantics=none")
        if self.kind == "effect" and self.provider_role != "actor":
            raise ValueError("effect capability must use provider_role=actor")
        if not self.result_contract.strip():
            raise ValueError("result_contract is required")
        if not 1 <= self.freshness_limit_seconds <= 86_400:
            raise ValueError("freshness_limit_seconds must be between 1 and 86400")
        return self

    @classmethod
    def from_runtime(
        cls,
        contract: CapabilityContract,
        *,
        result_contract: str,
        l0_contract_hashes: tuple[str, ...] = (),
    ) -> "ReleasedCapability":
        value = contract.to_dict()
        return cls(
            **value,
            result_contract=result_contract,
            l0_contract_hashes=l0_contract_hashes,
        )


class ProviderManifest(StrictModel):
    api_version: Literal[MANIFEST_SCHEMA] = Field(alias="apiVersion")
    provider_id: str
    provider_version: str
    provider_identity: str
    runtime_api_version: str = "1.0.0"
    released_at: str
    compatibility: Literal["compatible", "breaking"] = "compatible"
    supersedes: str | None = None
    artifacts: dict[str, str]
    capabilities: tuple[ReleasedCapability, ...]

    @model_validator(mode="after")
    def validate_manifest(self) -> "ProviderManifest":
        if not IDENTIFIER.fullmatch(self.provider_id):
            raise ValueError("provider_id must be a lowercase dotted identifier")
        _version(self.provider_version)
        _version(self.runtime_api_version)
        _parse_time(self.released_at, label="released_at")
        if not self.provider_identity.strip():
            raise ValueError("provider_identity is required")
        if not self.artifacts:
            raise ValueError("at least one immutable artifact digest is required")
        for name, digest in self.artifacts.items():
            if not name.strip():
                raise ValueError("artifact name cannot be empty")
            _validate_digest(digest, label=f"artifact {name}")
        if not self.capabilities:
            raise ValueError("provider manifest must contain capabilities")
        tools = [item.tool_name for item in self.capabilities]
        ids = [item.capability_id for item in self.capabilities]
        if len(set(tools)) != len(tools) or len(set(ids)) != len(ids):
            raise ValueError("provider manifest contains duplicate tool or capability ids")
        if any(item.provider_identity != self.provider_identity for item in self.capabilities):
            raise ValueError("every capability must bind the manifest provider identity")
        if self.supersedes is not None:
            _validate_digest(self.supersedes, label="supersedes")
        if self.compatibility == "breaking" and self.supersedes is None:
            raise ValueError("breaking release must explicitly bind the superseded digest")
        return self

    @property
    def digest(self) -> str:
        return sha256_json(self.model_dump(by_alias=True, mode="json"))


class QualificationReport(StrictModel):
    api_version: Literal[QUALIFICATION_SCHEMA] = Field(alias="apiVersion")
    provider_id: str
    manifest_digest: str
    suite_version: str = QUALIFICATION_SUITE_VERSION
    environment: str
    executed_at: str
    checks: dict[str, bool]
    evidence_digests: dict[str, str]

    @model_validator(mode="after")
    def validate_report(self) -> "QualificationReport":
        if not IDENTIFIER.fullmatch(self.provider_id):
            raise ValueError("qualification provider_id is invalid")
        _validate_digest(self.manifest_digest, label="manifest_digest")
        _version(self.suite_version)
        _parse_time(self.executed_at, label="executed_at")
        missing = sorted(REQUIRED_QUALIFICATION_CHECKS - set(self.checks))
        if missing:
            raise ValueError("qualification checks are missing: " + ", ".join(missing))
        unexpected = sorted(set(self.checks) - REQUIRED_QUALIFICATION_CHECKS)
        if unexpected:
            raise ValueError(
                "qualification contains unsupported checks: " + ", ".join(unexpected)
            )
        failed = sorted(name for name, passed in self.checks.items() if not passed)
        if failed:
            raise ValueError("qualification checks failed: " + ", ".join(failed))
        if set(self.evidence_digests) != set(self.checks):
            raise ValueError("every qualification check requires exactly one evidence digest")
        for name, digest in self.evidence_digests.items():
            _validate_digest(digest, label=f"qualification evidence {name}")
        if not self.environment.strip():
            raise ValueError("qualification environment is required")
        return self

    @property
    def digest(self) -> str:
        return sha256_json(self.model_dump(by_alias=True, mode="json"))


class DetachedSignature(StrictModel):
    api_version: Literal[SIGNATURE_SCHEMA] = Field(alias="apiVersion")
    key_id: str
    role: Literal["publisher", "qualifier", "deployer"]
    algorithm: Literal["Ed25519"] = "Ed25519"
    subject_digest: str
    signed_at: str
    expires_at: str
    signature: str

    @model_validator(mode="after")
    def validate_signature(self) -> "DetachedSignature":
        if not self.key_id.strip():
            raise ValueError("signature key_id is required")
        _validate_digest(self.subject_digest, label="subject_digest")
        signed = _parse_time(self.signed_at, label="signed_at")
        expires = _parse_time(self.expires_at, label="expires_at")
        if expires <= signed:
            raise ValueError("signature expires_at must be after signed_at")
        if expires - signed > timedelta(days=366):
            raise ValueError("signature lifetime cannot exceed 366 days")
        try:
            decoded = base64.b64decode(self.signature, validate=True)
        except ValueError as error:
            raise ValueError("signature is not valid base64") from error
        if len(decoded) != 64:
            raise ValueError("Ed25519 signature must contain 64 bytes")
        return self

    @property
    def signed_payload(self) -> bytes:
        return canonical_json({
            "algorithm": self.algorithm,
            "expires_at": self.expires_at,
            "key_id": self.key_id,
            "role": self.role,
            "signed_at": self.signed_at,
            "subject_digest": self.subject_digest,
        }).encode("utf-8")


class ProviderReleaseBundle(StrictModel):
    api_version: Literal[BUNDLE_SCHEMA] = Field(alias="apiVersion")
    manifest: ProviderManifest
    manifest_signature: DetachedSignature
    qualification: QualificationReport
    qualification_signature: DetachedSignature

    @model_validator(mode="after")
    def validate_links(self) -> "ProviderReleaseBundle":
        if self.manifest_signature.role != "publisher":
            raise ValueError("manifest must carry a publisher signature")
        if self.qualification_signature.role != "qualifier":
            raise ValueError("qualification must carry a qualifier signature")
        if self.manifest_signature.subject_digest != self.manifest.digest:
            raise ValueError("publisher signature is bound to another manifest")
        if self.qualification.manifest_digest != self.manifest.digest:
            raise ValueError("qualification is bound to another manifest")
        if self.qualification.provider_id != self.manifest.provider_id:
            raise ValueError("qualification provider differs from manifest")
        if self.qualification_signature.subject_digest != self.qualification.digest:
            raise ValueError("qualifier signature is bound to another report")
        return self

    @property
    def digest(self) -> str:
        return sha256_json(self.model_dump(by_alias=True, mode="json"))


class ProviderDeploymentAttestation(StrictModel):
    """Deployment-controller evidence for one exact release and environment."""

    api_version: Literal[DEPLOYMENT_SCHEMA] = Field(alias="apiVersion")
    provider_id: str
    provider_version: str
    provider_identity: str
    release_digest: str
    manifest_digest: str
    environment: str
    deployment_id: str
    controller_identity: str
    artifact_digests: dict[str, str]
    deployed_at: str
    expires_at: str

    @model_validator(mode="after")
    def validate_attestation(self) -> "ProviderDeploymentAttestation":
        if not IDENTIFIER.fullmatch(self.provider_id):
            raise ValueError("deployment provider_id is invalid")
        _version(self.provider_version)
        _validate_digest(self.release_digest, label="deployment release_digest")
        _validate_digest(self.manifest_digest, label="deployment manifest_digest")
        if not all((
            self.provider_identity.strip(), self.environment.strip(),
            self.deployment_id.strip(), self.controller_identity.strip(),
        )):
            raise ValueError("deployment identity and environment fields are required")
        if not self.artifact_digests:
            raise ValueError("deployment attestation requires observed artifact digests")
        for name, digest in self.artifact_digests.items():
            if not IDENTIFIER.fullmatch(name):
                raise ValueError("deployment artifact name is invalid")
            _validate_digest(digest, label=f"deployment artifact {name}")
        deployed = _parse_time(self.deployed_at, label="deployed_at")
        expires = _parse_time(self.expires_at, label="deployment expires_at")
        if expires <= deployed:
            raise ValueError("deployment attestation expiry must follow deployment")
        if expires - deployed > timedelta(days=31):
            raise ValueError("deployment attestation lifetime cannot exceed 31 days")
        return self

    @property
    def digest(self) -> str:
        return sha256_json(self.model_dump(by_alias=True, mode="json"))


class SignedProviderDeployment(StrictModel):
    api_version: Literal[SIGNED_DEPLOYMENT_SCHEMA] = Field(alias="apiVersion")
    attestation: ProviderDeploymentAttestation
    signature: DetachedSignature

    @model_validator(mode="after")
    def validate_links(self) -> "SignedProviderDeployment":
        if self.signature.role != "deployer":
            raise ValueError("deployment attestation requires a deployer signature")
        if self.signature.subject_digest != self.attestation.digest:
            raise ValueError("deployer signature is bound to another attestation")
        return self

    @property
    def digest(self) -> str:
        return sha256_json(self.model_dump(by_alias=True, mode="json"))


class TrustedKey(StrictModel):
    key_id: str
    role: Literal["publisher", "qualifier", "deployer"]
    public_key_pem: str
    providers: tuple[str, ...]
    not_before: str
    not_after: str
    revoked: bool = False

    @model_validator(mode="after")
    def validate_key(self) -> "TrustedKey":
        if not self.key_id.strip() or not self.providers:
            raise ValueError("trusted key requires key_id and provider scope")
        if any(value != "*" and not IDENTIFIER.fullmatch(value) for value in self.providers):
            raise ValueError("trusted key provider scope is invalid")
        start = _parse_time(self.not_before, label="not_before")
        end = _parse_time(self.not_after, label="not_after")
        if end <= start:
            raise ValueError("trusted key validity window is invalid")
        try:
            key = serialization.load_pem_public_key(self.public_key_pem.encode("utf-8"))
        except (TypeError, ValueError) as error:
            raise ValueError("trusted key PEM is invalid") from error
        if not isinstance(key, Ed25519PublicKey):
            raise ValueError("trusted key must be Ed25519")
        return self


class ProviderTrustStore(StrictModel):
    api_version: Literal[TRUST_SCHEMA] = Field(alias="apiVersion")
    runtime_api_version: str = "1.0.0"
    max_qualification_age_seconds: int = 2_592_000
    required_artifacts: tuple[str, ...] = ()
    require_deployment_attestation: bool = False
    keys: tuple[TrustedKey, ...]

    @model_validator(mode="after")
    def validate_store(self) -> "ProviderTrustStore":
        _version(self.runtime_api_version)
        if not 300 <= self.max_qualification_age_seconds <= 31_536_000:
            raise ValueError("qualification age must be between five minutes and one year")
        if len(set(self.required_artifacts)) != len(self.required_artifacts):
            raise ValueError("required_artifacts contains duplicates")
        if any(not IDENTIFIER.fullmatch(name) for name in self.required_artifacts):
            raise ValueError("required_artifacts contains an invalid artifact name")
        ids = [item.key_id for item in self.keys]
        if len(set(ids)) != len(ids):
            raise ValueError("trust store contains duplicate key ids")
        if not any(item.role == "publisher" for item in self.keys):
            raise ValueError("trust store has no publisher key")
        if not any(item.role == "qualifier" for item in self.keys):
            raise ValueError("trust store has no independent qualifier key")
        if self.require_deployment_attestation and not any(
            item.role == "deployer" for item in self.keys
        ):
            raise ValueError("deployment attestation policy requires a deployer key")
        material_roles: dict[bytes, set[str]] = {}
        for item in self.keys:
            public_key = serialization.load_pem_public_key(
                item.public_key_pem.encode("utf-8"),
            )
            material = public_key.public_bytes(
                serialization.Encoding.DER,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            material_roles.setdefault(material, set()).add(item.role)
        if any(len(roles) > 1 for roles in material_roles.values()):
            raise ValueError("publisher, qualifier, and deployer must use independent key material")
        return self

    @classmethod
    def from_path(cls, path: str | Path) -> "ProviderTrustStore":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def _key(
        self,
        signature: DetachedSignature,
        *,
        provider_id: str,
        now: datetime,
    ) -> Ed25519PublicKey:
        trusted = next((item for item in self.keys if item.key_id == signature.key_id), None)
        if trusted is None or trusted.revoked:
            raise ProviderReleaseError("release signature key is unknown or revoked")
        if trusted.role != signature.role:
            raise ProviderReleaseError("release signature key has the wrong trust role")
        if "*" not in trusted.providers and provider_id not in trusted.providers:
            raise ProviderReleaseError("release signature key is outside provider scope")
        signed = _parse_time(signature.signed_at, label="signed_at")
        expires = _parse_time(signature.expires_at, label="expires_at")
        if not _parse_time(trusted.not_before, label="not_before") <= signed:
            raise ProviderReleaseError("release was signed before key validity")
        if signed >= _parse_time(trusted.not_after, label="not_after"):
            raise ProviderReleaseError("release was signed after key validity")
        if expires > _parse_time(trusted.not_after, label="not_after"):
            raise ProviderReleaseError("release signature outlives key validity")
        if signed > now + timedelta(seconds=30) or expires <= now:
            raise ProviderReleaseError("release signature is future-dated or expired")
        key = serialization.load_pem_public_key(trusted.public_key_pem.encode("utf-8"))
        if not isinstance(key, Ed25519PublicKey):  # pragma: no cover - model invariant
            raise ProviderReleaseError("trusted release key is not Ed25519")
        return key

    def _verify_signature(
        self,
        signature: DetachedSignature,
        *,
        provider_id: str,
        now: datetime,
    ) -> None:
        key = self._key(signature, provider_id=provider_id, now=now)
        try:
            key.verify(
                base64.b64decode(signature.signature),
                signature.signed_payload,
            )
        except Exception as error:
            raise ProviderReleaseError("release signature verification failed") from error

    def verify_bundle(
        self,
        bundle: ProviderReleaseBundle,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        current = now or datetime.now(timezone.utc)
        missing_artifacts = sorted(
            set(self.required_artifacts) - set(bundle.manifest.artifacts)
        )
        if missing_artifacts:
            raise ProviderReleaseError(
                "Provider Manifest is missing required artifacts: "
                + ", ".join(missing_artifacts)
            )
        if _version(bundle.manifest.runtime_api_version)[0] != _version(self.runtime_api_version)[0]:
            raise ProviderReleaseError("provider and Runtime API major versions are incompatible")
        released = _parse_time(bundle.manifest.released_at, label="released_at")
        executed = _parse_time(bundle.qualification.executed_at, label="executed_at")
        if released > current + timedelta(seconds=30) or executed > current + timedelta(seconds=30):
            raise ProviderReleaseError("release or qualification is future-dated")
        if executed + timedelta(seconds=30) < released:
            raise ProviderReleaseError("qualification predates the Provider manifest")
        if current - executed > timedelta(seconds=self.max_qualification_age_seconds):
            raise ProviderReleaseError("provider qualification evidence is stale")
        self._verify_signature(
            bundle.manifest_signature,
            provider_id=bundle.manifest.provider_id,
            now=current,
        )
        self._verify_signature(
            bundle.qualification_signature,
            provider_id=bundle.manifest.provider_id,
            now=current,
        )
        return {
            "ok": True,
            "release_digest": bundle.digest,
            "manifest_digest": bundle.manifest.digest,
            "qualification_digest": bundle.qualification.digest,
            "provider_id": bundle.manifest.provider_id,
            "provider_version": bundle.manifest.provider_version,
            "publisher_key_id": bundle.manifest_signature.key_id,
            "qualifier_key_id": bundle.qualification_signature.key_id,
        }

    def verify_deployment(
        self,
        bundle: ProviderReleaseBundle,
        deployment: SignedProviderDeployment | None,
        *,
        environment: str,
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        """Verify a deployment-controller signature and exact artifact observation."""
        if deployment is None:
            if self.require_deployment_attestation:
                raise ProviderReleaseError("active release requires a deployment attestation")
            return None
        current = now or datetime.now(timezone.utc)
        attestation = deployment.attestation
        expected = {
            "provider_id": bundle.manifest.provider_id,
            "provider_version": bundle.manifest.provider_version,
            "provider_identity": bundle.manifest.provider_identity,
            "release_digest": bundle.digest,
            "manifest_digest": bundle.manifest.digest,
            "environment": environment,
            "artifact_digests": bundle.manifest.artifacts,
        }
        actual = {
            "provider_id": attestation.provider_id,
            "provider_version": attestation.provider_version,
            "provider_identity": attestation.provider_identity,
            "release_digest": attestation.release_digest,
            "manifest_digest": attestation.manifest_digest,
            "environment": attestation.environment,
            "artifact_digests": attestation.artifact_digests,
        }
        if actual != expected:
            raise ProviderReleaseError(
                "deployment attestation differs from the signed release or environment"
            )
        deployed = _parse_time(attestation.deployed_at, label="deployed_at")
        expires = _parse_time(attestation.expires_at, label="deployment expires_at")
        if deployed > current + timedelta(seconds=30) or expires <= current:
            raise ProviderReleaseError("deployment attestation is future-dated or expired")
        signature_time = _parse_time(deployment.signature.signed_at, label="signed_at")
        signature_expiry = _parse_time(deployment.signature.expires_at, label="expires_at")
        if signature_time + timedelta(seconds=30) < deployed:
            raise ProviderReleaseError("deployment was signed before it was observed")
        if signature_expiry > expires:
            raise ProviderReleaseError(
                "deployment signature cannot outlive the deployment attestation"
            )
        self._verify_signature(
            deployment.signature,
            provider_id=attestation.provider_id,
            now=current,
        )
        return {
            "ok": True,
            "deployment_digest": deployment.digest,
            "attestation_digest": attestation.digest,
            "deployment_id": attestation.deployment_id,
            "controller_identity": attestation.controller_identity,
            "deployer_key_id": deployment.signature.key_id,
        }


def load_private_key(path: str | Path) -> Ed25519PrivateKey:
    key_path = Path(path).expanduser()
    if not key_path.is_file():
        raise ProviderReleaseError("release signing key is not a file")
    if key_path.stat().st_mode & 0o077:
        raise ProviderReleaseError("release signing key must be owner-only")
    try:
        value = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
    except (TypeError, ValueError) as error:
        raise ProviderReleaseError("release signing key is invalid") from error
    if not isinstance(value, Ed25519PrivateKey):
        raise ProviderReleaseError("release signing key must be Ed25519")
    return value


def sign_digest(
    digest: str,
    *,
    private_key: Ed25519PrivateKey,
    key_id: str,
    role: Literal["publisher", "qualifier", "deployer"],
    now: datetime | None = None,
    ttl_seconds: int = 86_400,
) -> DetachedSignature:
    _validate_digest(digest, label="subject_digest")
    if not 300 <= ttl_seconds <= 31_536_000:
        raise ValueError("signature TTL must be between five minutes and one year")
    current = now or datetime.now(timezone.utc)
    unsigned = DetachedSignature(
        apiVersion=SIGNATURE_SCHEMA,
        key_id=key_id,
        role=role,
        subject_digest=digest,
        signed_at=current.isoformat(),
        expires_at=(current + timedelta(seconds=ttl_seconds)).isoformat(),
        signature=base64.b64encode(b"\x00" * 64).decode("ascii"),
    )
    signature = private_key.sign(unsigned.signed_payload)
    return unsigned.model_copy(update={
        "signature": base64.b64encode(signature).decode("ascii"),
    })


def compatibility_report(
    previous: ProviderManifest,
    candidate: ProviderManifest,
) -> dict[str, Any]:
    reasons: list[str] = []
    if previous.provider_id != candidate.provider_id:
        reasons.append("provider_id changed")
    if previous.provider_identity != candidate.provider_identity:
        reasons.append("provider_identity changed")
    if _version(candidate.provider_version) <= _version(previous.provider_version):
        reasons.append("provider_version did not increase")
    old = {item.capability_id: item for item in previous.capabilities}
    new = {item.capability_id: item for item in candidate.capabilities}
    for capability_id in sorted(set(old) - set(new)):
        reasons.append(f"capability removed: {capability_id}")
    guarded_fields = (
        "tool_name", "domain", "kind", "action_type", "effect_semantics",
        "provider_role", "provider_kind", "input_schema_digest",
        "output_schema_digest", "sensitivity", "required_roles", "scope_fields",
        "freshness_limit_seconds", "result_contract", "l0_contract_hashes",
    )
    for capability_id in sorted(set(old) & set(new)):
        left, right = old[capability_id], new[capability_id]
        if _version(left.capability_version)[0] != _version(right.capability_version)[0]:
            reasons.append(f"capability major version changed: {capability_id}")
        if _version(right.capability_version) < _version(left.capability_version):
            reasons.append(f"capability version decreased: {capability_id}")
        for field_name in guarded_fields:
            if getattr(left, field_name) != getattr(right, field_name):
                reasons.append(f"{capability_id} changed {field_name}")
    return {
        "compatible": not reasons,
        "breaking_reasons": reasons,
        "added_capabilities": sorted(set(new) - set(old)),
        "previous_digest": previous.digest,
        "candidate_digest": candidate.digest,
    }


class ProviderReleaseRegistry:
    """SQLite lifecycle registry with an append-only, hash-chained event log."""

    def __init__(self, path: str | Path, trust_store: ProviderTrustStore) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.trust_store = trust_store
        with self._connect() as database:
            database.executescript("""
                CREATE TABLE IF NOT EXISTS releases (
                    release_digest TEXT PRIMARY KEY,
                    provider_id TEXT NOT NULL,
                    provider_version TEXT NOT NULL,
                    provider_identity TEXT NOT NULL,
                    bundle_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(provider_id, provider_version)
                );
                CREATE TABLE IF NOT EXISTS activations (
                    provider_id TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    release_digest TEXT NOT NULL,
                    previous_release_digest TEXT,
                    approval_reference TEXT,
                    activated_at TEXT NOT NULL,
                    deployment_digest TEXT,
                    PRIMARY KEY(provider_id, environment),
                    FOREIGN KEY(release_digest) REFERENCES releases(release_digest)
                );
                CREATE TABLE IF NOT EXISTS deployments (
                    deployment_digest TEXT PRIMARY KEY,
                    provider_id TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    release_digest TEXT NOT NULL,
                    deployment_json TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    FOREIGN KEY(release_digest) REFERENCES releases(release_digest)
                );
                CREATE TABLE IF NOT EXISTS release_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    release_digest TEXT,
                    payload_json TEXT NOT NULL,
                    prev_event_hash TEXT NOT NULL,
                    event_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
            """)
            activation_columns = {
                str(row[1]) for row in database.execute("PRAGMA table_info(activations)")
            }
            if "deployment_digest" not in activation_columns:
                database.execute(
                    "ALTER TABLE activations ADD COLUMN deployment_digest TEXT"
                )
        os.chmod(self.path, 0o600)

    def _connect(self) -> sqlite3.Connection:
        database = sqlite3.connect(str(self.path), timeout=30)
        database.row_factory = sqlite3.Row
        database.execute("PRAGMA journal_mode=WAL")
        database.execute("PRAGMA foreign_keys=ON")
        return database

    @staticmethod
    def _bundle(row: sqlite3.Row) -> ProviderReleaseBundle:
        return ProviderReleaseBundle.model_validate_json(str(row["bundle_json"]))

    def _event(
        self,
        database: sqlite3.Connection,
        event_type: str,
        release_digest: str | None,
        payload: dict[str, Any],
    ) -> None:
        previous = database.execute(
            "SELECT event_hash FROM release_events ORDER BY event_id DESC LIMIT 1"
        ).fetchone()
        prev_hash = str(previous["event_hash"]) if previous else "GENESIS"
        created_at = datetime.now(timezone.utc).isoformat()
        payload_json = canonical_json(payload)
        event_hash = sha256_json({
            "event_type": event_type,
            "release_digest": release_digest,
            "payload_json": payload_json,
            "prev_event_hash": prev_hash,
            "created_at": created_at,
        })
        database.execute(
            "INSERT INTO release_events(event_type, release_digest, payload_json, "
            "prev_event_hash, event_hash, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (event_type, release_digest, payload_json, prev_hash, event_hash, created_at),
        )

    @staticmethod
    def _deployment(row: sqlite3.Row) -> SignedProviderDeployment:
        return SignedProviderDeployment.model_validate_json(str(row["deployment_json"]))

    def _store_deployment(
        self,
        database: sqlite3.Connection,
        deployment: SignedProviderDeployment | None,
    ) -> str | None:
        if deployment is None:
            return None
        attestation = deployment.attestation
        payload = canonical_json(deployment.model_dump(by_alias=True, mode="json"))
        database.execute(
            "INSERT OR IGNORE INTO deployments("
            "deployment_digest, provider_id, environment, release_digest, "
            "deployment_json, recorded_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                deployment.digest, attestation.provider_id, attestation.environment,
                attestation.release_digest, payload,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        row = database.execute(
            "SELECT deployment_json FROM deployments WHERE deployment_digest=?",
            (deployment.digest,),
        ).fetchone()
        if row is None or str(row["deployment_json"]) != payload:
            raise ProviderReleaseError("deployment digest is bound to another record")
        return deployment.digest

    def stage(self, bundle: ProviderReleaseBundle) -> dict[str, Any]:
        evidence = self.trust_store.verify_bundle(bundle)
        now = datetime.now(timezone.utc).isoformat()
        payload = canonical_json(bundle.model_dump(by_alias=True, mode="json"))
        with self._connect() as database:
            database.execute("BEGIN IMMEDIATE")
            existing = database.execute(
                "SELECT release_digest, state FROM releases "
                "WHERE provider_id=? AND provider_version=?",
                (bundle.manifest.provider_id, bundle.manifest.provider_version),
            ).fetchone()
            if existing and str(existing["release_digest"]) != bundle.digest:
                raise ProviderReleaseError("provider version is already bound to another digest")
            if existing:
                return {
                    **evidence, "state": str(existing["state"]), "idempotent": True,
                }
            database.execute(
                "INSERT INTO releases VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    bundle.digest, bundle.manifest.provider_id,
                    bundle.manifest.provider_version, bundle.manifest.provider_identity,
                    payload, "staged", now, now,
                ),
            )
            self._event(database, "release_staged", bundle.digest, evidence)
        return {**evidence, "state": "staged", "idempotent": False}

    def _release_row(
        self, database: sqlite3.Connection, release_digest: str,
    ) -> sqlite3.Row:
        row = database.execute(
            "SELECT * FROM releases WHERE release_digest=?", (release_digest,),
        ).fetchone()
        if row is None:
            raise ProviderReleaseError("provider release is not staged")
        return row

    def publish(self, release_digest: str) -> dict[str, Any]:
        with self._connect() as database:
            database.execute("BEGIN IMMEDIATE")
            row = self._release_row(database, release_digest)
            bundle = self._bundle(row)
            self.trust_store.verify_bundle(bundle)
            state = str(row["state"])
            if state == "published":
                return {"ok": True, "release_digest": release_digest, "state": state}
            if state != "staged":
                raise ProviderReleaseError(f"cannot publish a {state} release")
            now = datetime.now(timezone.utc).isoformat()
            database.execute(
                "UPDATE releases SET state='published', updated_at=? WHERE release_digest=?",
                (now, release_digest),
            )
            self._event(database, "release_published", release_digest, {})
        return {"ok": True, "release_digest": release_digest, "state": "published"}

    def promote(
        self,
        release_digest: str,
        *,
        environment: str,
        approval_reference: str = "",
        deployment: SignedProviderDeployment | None = None,
    ) -> dict[str, Any]:
        if not environment.strip():
            raise ProviderReleaseError("activation environment is required")
        with self._connect() as database:
            database.execute("BEGIN IMMEDIATE")
            row = self._release_row(database, release_digest)
            if str(row["state"]) != "published":
                raise ProviderReleaseError("only a published release can be promoted")
            bundle = self._bundle(row)
            evidence = self.trust_store.verify_bundle(bundle)
            deployment_evidence = self.trust_store.verify_deployment(
                bundle, deployment, environment=environment,
            )
            deployment_digest = self._store_deployment(database, deployment)
            active = database.execute(
                "SELECT * FROM activations WHERE provider_id=? AND environment=?",
                (bundle.manifest.provider_id, environment),
            ).fetchone()
            previous_digest = str(active["release_digest"]) if active else None
            compatibility = None
            active_deployment_digest = (
                str(active["deployment_digest"] or "") if active else ""
            )
            if (
                previous_digest == release_digest
                and active_deployment_digest == str(deployment_digest or "")
            ):
                return {
                    **evidence, "environment": environment,
                    "previous_release_digest": previous_digest, "idempotent": True,
                    "deployment": deployment_evidence,
                }
            if previous_digest and previous_digest != release_digest:
                previous_row = self._release_row(database, previous_digest)
                previous = self._bundle(previous_row)
                compatibility = compatibility_report(previous.manifest, bundle.manifest)
                breaking = not compatibility["compatible"] or bundle.manifest.compatibility == "breaking"
                if breaking and (
                    bundle.manifest.compatibility != "breaking"
                    or bundle.manifest.supersedes != previous.manifest.digest
                    or not approval_reference.strip()
                ):
                    raise ProviderReleaseError(
                        "breaking promotion requires manifest supersedes binding and approval_reference"
                    )
            now = datetime.now(timezone.utc).isoformat()
            database.execute(
                "INSERT INTO activations("
                "provider_id, environment, release_digest, previous_release_digest, "
                "approval_reference, activated_at, deployment_digest"
                ") VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(provider_id, environment) DO UPDATE SET "
                "release_digest=excluded.release_digest, "
                "previous_release_digest=excluded.previous_release_digest, "
                "approval_reference=excluded.approval_reference, "
                "activated_at=excluded.activated_at, "
                "deployment_digest=excluded.deployment_digest",
                (
                    bundle.manifest.provider_id, environment, release_digest,
                    previous_digest, approval_reference or None, now, deployment_digest,
                ),
            )
            self._event(database, "release_promoted", release_digest, {
                "provider_id": bundle.manifest.provider_id,
                "environment": environment,
                "previous_release_digest": previous_digest,
                "approval_reference": approval_reference or None,
                "compatibility": compatibility,
                "deployment": deployment_evidence,
            })
        return {
            **evidence,
            "environment": environment,
            "previous_release_digest": previous_digest,
            "compatibility": compatibility,
            "deployment": deployment_evidence,
            "idempotent": False,
        }

    def rollback(
        self,
        *,
        provider_id: str,
        environment: str,
        approval_reference: str,
        target_release_digest: str | None = None,
        deployment: SignedProviderDeployment | None = None,
    ) -> dict[str, Any]:
        if not approval_reference.strip():
            raise ProviderReleaseError("rollback requires an approval_reference")
        with self._connect() as database:
            database.execute("BEGIN IMMEDIATE")
            active = database.execute(
                "SELECT * FROM activations WHERE provider_id=? AND environment=?",
                (provider_id, environment),
            ).fetchone()
            if active is None:
                raise ProviderReleaseError("provider has no active release in this environment")
            current = str(active["release_digest"])
            target = target_release_digest or str(active["previous_release_digest"] or "")
            if not target or target == current:
                raise ProviderReleaseError("rollback target is unavailable or already active")
            row = self._release_row(database, target)
            if str(row["provider_id"]) != provider_id or str(row["state"]) not in {
                "published", "deprecated",
            }:
                raise ProviderReleaseError("rollback target is not an eligible provider release")
            bundle = self._bundle(row)
            evidence = self.trust_store.verify_bundle(bundle)
            deployment_evidence = self.trust_store.verify_deployment(
                bundle, deployment, environment=environment,
            )
            deployment_digest = self._store_deployment(database, deployment)
            now = datetime.now(timezone.utc).isoformat()
            database.execute(
                "UPDATE activations SET release_digest=?, previous_release_digest=?, "
                "approval_reference=?, activated_at=?, deployment_digest=? "
                "WHERE provider_id=? AND environment=?",
                (
                    target, current, approval_reference, now, deployment_digest,
                    provider_id, environment,
                ),
            )
            self._event(database, "release_rolled_back", target, {
                "provider_id": provider_id,
                "environment": environment,
                "from_release_digest": current,
                "approval_reference": approval_reference,
                "deployment": deployment_evidence,
            })
        return {
            **evidence,
            "environment": environment,
            "rolled_back_from": current,
            "deployment": deployment_evidence,
        }

    def deprecate(self, release_digest: str, *, reason: str) -> dict[str, Any]:
        if not reason.strip():
            raise ProviderReleaseError("deprecation reason is required")
        with self._connect() as database:
            database.execute("BEGIN IMMEDIATE")
            row = self._release_row(database, release_digest)
            active = database.execute(
                "SELECT COUNT(*) AS count FROM activations WHERE release_digest=?",
                (release_digest,),
            ).fetchone()
            if int(active["count"]):
                raise ProviderReleaseError("active release cannot be deprecated")
            if str(row["state"]) not in {"published", "deprecated"}:
                raise ProviderReleaseError("only a published release can be deprecated")
            database.execute(
                "UPDATE releases SET state='deprecated', updated_at=? WHERE release_digest=?",
                (datetime.now(timezone.utc).isoformat(), release_digest),
            )
            self._event(database, "release_deprecated", release_digest, {"reason": reason})
        return {"ok": True, "release_digest": release_digest, "state": "deprecated"}

    def active_release(
        self, provider_id: str, *, environment: str,
    ) -> tuple[ProviderReleaseBundle, SignedProviderDeployment | None]:
        with self._connect() as database:
            row = database.execute(
                "SELECT releases.*, activations.deployment_digest AS activation_deployment_digest "
                "FROM activations JOIN releases USING(release_digest) "
                "WHERE activations.provider_id=? AND activations.environment=?",
                (provider_id, environment),
            ).fetchone()
            if row is None:
                raise ProviderReleaseError("provider has no active release in this environment")
            bundle = self._bundle(row)
            deployment_digest = str(row["activation_deployment_digest"] or "")
            deployment = None
            if deployment_digest:
                deployment_row = database.execute(
                    "SELECT * FROM deployments WHERE deployment_digest=?",
                    (deployment_digest,),
                ).fetchone()
                if deployment_row is None:
                    raise ProviderReleaseError("active deployment attestation is missing")
                deployment = self._deployment(deployment_row)
        self.trust_store.verify_bundle(bundle)
        self.trust_store.verify_deployment(
            bundle, deployment, environment=environment,
        )
        return bundle, deployment

    def active_bundle(self, provider_id: str, *, environment: str) -> ProviderReleaseBundle:
        return self.active_release(provider_id, environment=environment)[0]

    def status(self) -> dict[str, Any]:
        with self._connect() as database:
            releases = [dict(row) for row in database.execute(
                "SELECT release_digest, provider_id, provider_version, provider_identity, "
                "state, created_at, updated_at FROM releases ORDER BY provider_id, provider_version"
            )]
            activations = [dict(row) for row in database.execute(
                "SELECT * FROM activations ORDER BY provider_id, environment"
            )]
            deployments = [dict(row) for row in database.execute(
                "SELECT deployment_digest, provider_id, environment, release_digest, "
                "recorded_at FROM deployments ORDER BY recorded_at"
            )]
        return {
            "ok": True,
            "releases": releases,
            "activations": activations,
            "deployments": deployments,
        }

    def audit(self) -> dict[str, Any]:
        previous = "GENESIS"
        failures: list[int] = []
        with self._connect() as database:
            rows = database.execute("SELECT * FROM release_events ORDER BY event_id").fetchall()
        for row in rows:
            expected = sha256_json({
                "event_type": row["event_type"],
                "release_digest": row["release_digest"],
                "payload_json": row["payload_json"],
                "prev_event_hash": previous,
                "created_at": row["created_at"],
            })
            if row["prev_event_hash"] != previous or row["event_hash"] != expected:
                failures.append(int(row["event_id"]))
            previous = str(row["event_hash"])
        return {
            "ok": not failures,
            "events": len(rows),
            "failures": failures,
            "head": previous,
        }


@dataclass(frozen=True)
class ProviderAdmissionEvidence:
    release_digest: str
    manifest_digest: str
    qualification_digest: str
    deployment_digest: str
    provider_id: str
    provider_version: str
    result_contract: str
    l0_contract_hashes: tuple[str, ...]


class ProviderAdmissionGate:
    def __init__(
        self,
        registry: ProviderReleaseRegistry,
        *,
        environment: str,
    ) -> None:
        if not environment.strip():
            raise ValueError("provider admission environment is required")
        self.registry = registry
        self.environment = environment

    def admit(
        self,
        contract: CapabilityContract,
        *,
        provider_id: str,
        result_contract: str,
    ) -> ProviderAdmissionEvidence:
        if not provider_id.strip():
            raise ProviderReleaseError("external capability has no release_provider_id")
        bundle, deployment = self.registry.active_release(
            provider_id, environment=self.environment,
        )
        manifest = bundle.manifest
        if manifest.provider_identity != contract.provider_identity:
            raise ProviderReleaseError("active release provider identity mismatch")
        released = next(
            (item for item in manifest.capabilities if item.tool_name == contract.tool_name),
            None,
        )
        if released is None:
            raise ProviderReleaseError("active release does not authorize this tool")
        if released.result_contract != result_contract:
            raise ProviderReleaseError("discovered result contract differs from signed release")
        expected = {
            key: value for key, value in released.model_dump(mode="json").items()
            if key not in {"result_contract", "l0_contract_hashes"}
        }
        actual = contract.to_dict()
        if expected != actual:
            raise ProviderReleaseError("discovered capability differs from active signed release")
        return ProviderAdmissionEvidence(
            release_digest=bundle.digest,
            manifest_digest=manifest.digest,
            qualification_digest=bundle.qualification.digest,
            deployment_digest=(deployment.digest if deployment else "not-required"),
            provider_id=provider_id,
            provider_version=manifest.provider_version,
            result_contract=released.result_contract,
            l0_contract_hashes=released.l0_contract_hashes,
        )


def provider_admission_from_environment() -> ProviderAdmissionGate | None:
    """Create the opt-in production admission gate; incomplete config fails closed."""
    mode = os.environ.get("NETOPYU_PROVIDER_ADMISSION", "disabled").strip().lower()
    if mode in {"", "disabled", "off", "0"}:
        return None
    if mode != "enforced":
        raise ProviderReleaseError("NETOPYU_PROVIDER_ADMISSION must be disabled or enforced")
    trust_path = os.environ.get("NETOPYU_PROVIDER_TRUST_STORE", "").strip()
    registry_path = os.environ.get("NETOPYU_PROVIDER_RELEASE_DB", "").strip()
    environment = os.environ.get("NETOPYU_PROVIDER_ENVIRONMENT", "").strip()
    if not trust_path or not registry_path or not environment:
        raise ProviderReleaseError(
            "enforced Provider admission requires trust store, release DB, and environment"
        )
    trust = ProviderTrustStore.from_path(trust_path)
    return ProviderAdmissionGate(
        ProviderReleaseRegistry(registry_path, trust),
        environment=environment,
    )


def load_bundle(path: str | Path) -> ProviderReleaseBundle:
    return ProviderReleaseBundle.model_validate_json(
        Path(path).read_text(encoding="utf-8"),
    )


def load_deployment(path: str | Path) -> SignedProviderDeployment:
    return SignedProviderDeployment.model_validate_json(
        Path(path).read_text(encoding="utf-8"),
    )


__all__ = [
    "BUNDLE_SCHEMA",
    "DEPLOYMENT_SCHEMA",
    "DetachedSignature",
    "MANIFEST_SCHEMA",
    "ProviderAdmissionEvidence",
    "ProviderAdmissionGate",
    "ProviderDeploymentAttestation",
    "ProviderManifest",
    "ProviderReleaseBundle",
    "ProviderReleaseError",
    "ProviderReleaseRegistry",
    "SignedProviderDeployment",
    "ProviderTrustStore",
    "QualificationReport",
    "QUALIFICATION_SCHEMA",
    "QUALIFICATION_SUITE_VERSION",
    "REQUIRED_QUALIFICATION_CHECKS",
    "ReleasedCapability",
    "SIGNATURE_SCHEMA",
    "SIGNED_DEPLOYMENT_SCHEMA",
    "TRUST_SCHEMA",
    "TrustedKey",
    "compatibility_report",
    "load_bundle",
    "load_deployment",
    "load_private_key",
    "provider_admission_from_environment",
    "sign_digest",
]
