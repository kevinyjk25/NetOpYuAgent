"""Strict proposal-only contract for connecting an external system.

The pack describes interfaces and the deterministic controls a write needs.
It cannot register a Provider, activate an L0 Skill, approve a plan, or execute
an effect.  Credentials are referenced by environment-variable name and are
never accepted as document values.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
import yaml

from network_runtime.contracts import sha256_json


INTEGRATION_PACK_SCHEMA = "netopyu.io/integration-pack/v1"
INTEGRATION_ASSESSMENT_SCHEMA = "netopyu.io/integration-assessment/v1"
_MAX_PACK_BYTES = 2_000_000
_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{1,127}$")
_SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
_ENVIRONMENT_NAME = re.compile(r"^[A-Z][A-Z0-9_]{2,127}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class IntegrationPackError(RuntimeError):
    """Raised when an integration proposal cannot be trusted for review."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


def _require_id(value: str, label: str) -> None:
    if not _ID.fullmatch(value):
        raise ValueError(f"{label} is invalid")


def _require_semver(value: str, label: str) -> None:
    if not _SEMVER.fullmatch(value):
        raise ValueError(f"{label} must be semantic version x.y.z")


def _require_digest(value: str, label: str) -> None:
    if not _DIGEST.fullmatch(value):
        raise ValueError(f"{label} must be a complete sha256 digest")


class PackMetadata(_StrictModel):
    id: str
    version: str
    owner: str
    description: str

    @model_validator(mode="after")
    def validate_metadata(self) -> "PackMetadata":
        _require_id(self.id, "pack id")
        _require_semver(self.version, "pack version")
        _require_id(self.owner, "pack owner")
        if not self.description.strip() or len(self.description) > 500:
            raise ValueError("pack description is invalid")
        return self


class AuthenticationReference(_StrictModel):
    mode: Literal["environment", "workload_identity", "mtls"]
    credential_environment: str | None = Field(default=None, alias="credentialEnvironment")
    model_visible: Literal[False] = Field(default=False, alias="modelVisible")

    @model_validator(mode="after")
    def validate_authentication(self) -> "AuthenticationReference":
        if self.mode == "environment":
            if self.credential_environment is None or not _ENVIRONMENT_NAME.fullmatch(
                self.credential_environment
            ):
                raise ValueError("environment auth requires credentialEnvironment name")
        elif self.credential_environment is not None:
            raise ValueError("credentialEnvironment is only valid for environment auth")
        return self


class ProviderProposal(_StrictModel):
    id: str
    protocol: Literal["mcp", "rest", "netconf", "ssh", "controller"]
    endpoint_environment: str = Field(alias="endpointEnvironment")
    authentication: AuthenticationReference
    provider_identity: str = Field(alias="providerIdentity")
    provider_version: str = Field(alias="providerVersion")

    @model_validator(mode="after")
    def validate_provider(self) -> "ProviderProposal":
        _require_id(self.id, "provider id")
        _require_id(self.provider_identity, "provider identity")
        _require_semver(self.provider_version, "provider version")
        if not _ENVIRONMENT_NAME.fullmatch(self.endpoint_environment):
            raise ValueError("endpointEnvironment must name an environment variable")
        return self


class L0BindingProposal(_StrictModel):
    skill_id: str = Field(alias="skillId")
    version: str
    contract_hash: str = Field(alias="contractHash")

    @model_validator(mode="after")
    def validate_binding(self) -> "L0BindingProposal":
        _require_id(self.skill_id, "L0 Skill id")
        _require_semver(self.version, "L0 Skill version")
        _require_digest(self.contract_hash, "L0 contractHash")
        return self


class EffectControls(_StrictModel):
    role: Literal["primary", "compensation"] = "primary"
    risk: Literal["low", "medium", "high", "critical"]
    approval_required: Literal[True] = Field(default=True, alias="approvalRequired")
    idempotency_key_field: str = Field(alias="idempotencyKeyField")
    verifier_ref: str = Field(alias="verifierRef")
    reversible: bool
    compensation_ref: str | None = Field(default=None, alias="compensationRef")
    irreversible_justification: str | None = Field(
        default=None, alias="irreversibleJustification",
    )
    l0_binding: L0BindingProposal | None = Field(default=None, alias="l0Binding")

    @model_validator(mode="after")
    def validate_controls(self) -> "EffectControls":
        _require_id(self.idempotency_key_field, "idempotency key field")
        _require_id(self.verifier_ref, "verifier reference")
        if self.compensation_ref is not None:
            _require_id(self.compensation_ref, "compensation reference")
        if self.reversible and self.compensation_ref is None:
            raise ValueError("reversible writes require compensationRef")
        if not self.reversible:
            if self.compensation_ref is not None:
                raise ValueError("non-reversible writes cannot declare compensationRef")
            if not self.irreversible_justification or not self.irreversible_justification.strip():
                raise ValueError("non-reversible writes require irreversibleJustification")
        elif self.irreversible_justification is not None:
            raise ValueError("reversible writes cannot declare irreversibleJustification")
        return self


class CapabilityProposal(_StrictModel):
    id: str
    version: str
    access: Literal["read", "write"]
    provider_ref: str = Field(alias="providerRef")
    operation: str
    input_schema_digest: str = Field(alias="inputSchemaDigest")
    output_schema_digest: str = Field(alias="outputSchemaDigest")
    sensitive_fields: tuple[str, ...] = Field(default=(), alias="sensitiveFields")
    controls: EffectControls | None = None

    @model_validator(mode="after")
    def validate_capability(self) -> "CapabilityProposal":
        _require_id(self.id, "capability id")
        _require_semver(self.version, "capability version")
        _require_id(self.provider_ref, "provider reference")
        _require_digest(self.input_schema_digest, "inputSchemaDigest")
        _require_digest(self.output_schema_digest, "outputSchemaDigest")
        if not self.operation.strip() or len(self.operation) > 300 or any(
            ord(character) < 32 for character in self.operation
        ):
            raise ValueError("capability operation is invalid")
        if len(set(self.sensitive_fields)) != len(self.sensitive_fields):
            raise ValueError("sensitiveFields must be unique")
        for field_name in self.sensitive_fields:
            _require_id(field_name, "sensitive field")
        if self.access == "read" and self.controls is not None:
            raise ValueError("read capabilities cannot declare effect controls")
        if self.access == "write" and self.controls is None:
            raise ValueError("write capabilities require effect controls")
        return self


class IntegrationPack(_StrictModel):
    api_version: Literal[INTEGRATION_PACK_SCHEMA] = Field(alias="apiVersion")
    purpose: Literal["proposal_only"] = "proposal_only"
    activation_available: Literal[False] = Field(default=False, alias="activationAvailable")
    metadata: PackMetadata
    providers: tuple[ProviderProposal, ...]
    capabilities: tuple[CapabilityProposal, ...]

    @model_validator(mode="after")
    def validate_pack(self) -> "IntegrationPack":
        if not self.providers or not self.capabilities:
            raise ValueError("integration pack requires providers and capabilities")
        providers = {provider.id for provider in self.providers}
        if len(providers) != len(self.providers):
            raise ValueError("provider ids must be unique")
        capabilities = {capability.id: capability for capability in self.capabilities}
        if len(capabilities) != len(self.capabilities):
            raise ValueError("capability ids must be unique")
        for capability in self.capabilities:
            if capability.provider_ref not in providers:
                raise ValueError(f"unknown providerRef for {capability.id}")
            if capability.access != "write" or capability.controls is None:
                continue
            controls = capability.controls
            if controls.verifier_ref == capability.id:
                raise ValueError("a write cannot verify itself")
            verifier = capabilities.get(controls.verifier_ref)
            if verifier is None or verifier.access != "read":
                raise ValueError(f"{capability.id} requires an independent read verifier")
            if controls.compensation_ref is not None:
                if controls.compensation_ref == capability.id:
                    raise ValueError("a write cannot compensate itself")
                compensation = capabilities.get(controls.compensation_ref)
                if compensation is None or compensation.access != "write":
                    raise ValueError(f"{capability.id} compensationRef must be a write")
        return self


def load_integration_pack(path: str | Path) -> IntegrationPack:
    supplied = Path(path).expanduser()
    if supplied.is_symlink() or not supplied.is_file() or supplied.stat().st_size > _MAX_PACK_BYTES:
        raise IntegrationPackError("integration pack is missing, unsafe, or oversized")
    try:
        raw = yaml.safe_load(supplied.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("document root must be an object")
        return IntegrationPack.model_validate(raw)
    except (UnicodeDecodeError, yaml.YAMLError, ValidationError, ValueError) as error:
        raise IntegrationPackError(f"integration pack is invalid: {error}") from error


def assess_integration_pack(pack: IntegrationPack) -> dict[str, Any]:
    """Return review readiness without creating any runtime authority."""
    writes = [capability for capability in pack.capabilities if capability.access == "write"]
    reads = [capability for capability in pack.capabilities if capability.access == "read"]
    unbound = [
        capability.id for capability in writes
        if capability.controls is None or capability.controls.l0_binding is None
    ]
    body: dict[str, Any] = {
        "apiVersion": INTEGRATION_ASSESSMENT_SCHEMA,
        "pack": {
            "id": pack.metadata.id,
            "version": pack.metadata.version,
            "owner": pack.metadata.owner,
        },
        "status": "ready_for_l0_authoring" if unbound else "ready_for_offline_review",
        "interfaceContractValid": True,
        "runtimeBindingComplete": not unbound,
        "activationAvailable": False,
        "counts": {
            "providers": len(pack.providers),
            "readCapabilities": len(reads),
            "writeCapabilities": len(writes),
            "writeCapabilitiesWithIndependentVerifier": len(writes),
            "reversibleWritesWithCompensation": sum(
                1 for item in writes if item.controls and item.controls.reversible
            ),
        },
        "unboundWrites": unbound,
        "nextSteps": ([
            "Author and review L1 prose plus L0.5 for every unbound write.",
            "Compile each write into an immutable L0 contract and add its exact hash.",
            "Qualify the Provider identity/version/schema and failure behavior outside this pack.",
        ] if unbound else [
            "Run L0 trajectory, Provider qualification, Runtime, and retirement gates.",
            "Publish through the separate governed Catalog and release process.",
            "Activation still requires environment authority and is unavailable from this command.",
        ]),
        "boundaries": [
            "This assessment validates a proposal document; it does not connect to the endpoint.",
            "It does not register, publish, approve, activate, or execute a capability.",
            "Credential values are not accepted and modelVisible is fixed to false.",
        ],
    }
    body["assessmentDigest"] = sha256_json(body)
    return body


def integration_pack_json_schema() -> dict[str, Any]:
    return json.loads(json.dumps(IntegrationPack.model_json_schema(by_alias=True)))
