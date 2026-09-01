"""Domain-neutral capability contracts consumed by the Effect Runtime.

Transport names such as MCP, REST, NETCONF, or a local callable are provider
implementation details.  The Runtime reasons only about an observation or an
effect contract and invokes the narrow provider gateway protocol below.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from .contracts import sha256_json


class CapabilityAdmissionError(RuntimeError):
    """An optional provider extension rejected a capability before use.

    This exception lives in the core Capability boundary so the EnsuredSkill
    Runtime never has to import a frozen release/supply-chain implementation.
    """


class CapabilityKind(str, Enum):
    OBSERVATION = "observation"
    EFFECT = "effect"


class DataSensitivity(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class EffectSemantics(str, Enum):
    NONE = "none"
    REVERSIBLE = "reversible"
    DESTRUCTIVE = "destructive"
    IRREVERSIBLE = "irreversible"


_SCOPE_FIELDS = (
    "device_id", "node", "node_id", "resource_id", "service", "environment",
    "user_id", "app_id", "endpoint_id", "source_endpoint", "destination_endpoint",
)


@dataclass(frozen=True)
class CapabilityContract:
    """Canonical provider contract independent of its wire protocol."""

    tool_name: str
    capability_id: str
    capability_version: str
    domain: str
    kind: CapabilityKind
    action_type: str
    effect_semantics: EffectSemantics
    provider_role: str
    provider_identity: str
    provider_kind: str
    input_schema_digest: str
    output_schema_digest: str
    sensitivity: DataSensitivity
    required_roles: tuple[str, ...]
    scope_fields: tuple[str, ...]
    freshness_limit_seconds: int

    @classmethod
    def from_metadata(
        cls,
        tool_name: str,
        metadata: dict[str, Any],
        *,
        source: str,
    ) -> "CapabilityContract":
        action_type = str(metadata.get("action_type") or "read_only")
        kind = (
            CapabilityKind.OBSERVATION
            if action_type == "read_only" and not bool(metadata.get("hitl"))
            else CapabilityKind.EFFECT
        )
        if kind == CapabilityKind.OBSERVATION:
            semantics = EffectSemantics.NONE
        elif action_type == "reversible" or metadata.get("compensator"):
            semantics = EffectSemantics.REVERSIBLE
        elif action_type == "irreversible":
            semantics = EffectSemantics.IRREVERSIBLE
        else:
            semantics = EffectSemantics.DESTRUCTIVE

        capability_id = str(metadata.get("capability_id") or tool_name)
        domain = str(metadata.get("domain") or metadata.get("service_domain") or "network")
        provider_role = str(metadata.get("provider_role") or (
            "observer" if kind == CapabilityKind.OBSERVATION else "actor"
        ))
        raw_sensitivity = str(metadata.get("sensitivity") or "internal")
        try:
            sensitivity = DataSensitivity(raw_sensitivity)
        except ValueError as error:
            raise ValueError(
                f"capability {capability_id!r} has invalid sensitivity {raw_sensitivity!r}"
            ) from error
        required_roles = tuple(sorted({
            str(value) for value in metadata.get("required_roles", ()) if str(value).strip()
        }))
        configured_scope_fields = metadata.get("scope_fields")
        if configured_scope_fields is None:
            configured_scope_fields = tuple(
                name for name in _SCOPE_FIELDS
                if name in (metadata.get("parameters") or {})
            )
        scope_fields = tuple(str(value) for value in configured_scope_fields)
        freshness = int(metadata.get("freshness_limit_seconds") or 300)
        if not 1 <= freshness <= 86_400:
            raise ValueError(
                f"capability {capability_id!r} freshness limit must be 1..86400 seconds"
            )
        return cls(
            tool_name=tool_name,
            capability_id=capability_id,
            capability_version=str(metadata.get("capability_version") or "1.0.0"),
            domain=domain,
            kind=kind,
            action_type=action_type,
            effect_semantics=semantics,
            provider_role=provider_role,
            provider_identity=str(metadata.get("provider_identity") or source),
            provider_kind=str(metadata.get("provider_kind") or source),
            input_schema_digest=str(
                metadata.get("input_schema_digest")
                or sha256_json(metadata.get("parameters") or {})
            ),
            output_schema_digest=str(
                metadata.get("output_schema_digest")
                or sha256_json(metadata.get("output_schema") or {})
            ),
            sensitivity=sensitivity,
            required_roles=required_roles,
            scope_fields=scope_fields,
            freshness_limit_seconds=freshness,
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for field_name in ("kind", "effect_semantics", "sensitivity"):
            value[field_name] = getattr(self, field_name).value
        value["required_roles"] = list(self.required_roles)
        value["scope_fields"] = list(self.scope_fields)
        return value


@runtime_checkable
class CapabilityProviderGateway(Protocol):
    """Structural SPI implemented by every Runtime-facing provider session."""

    def describe_capability(self, tool_name: str) -> CapabilityContract: ...

    async def invoke_observation(
        self, tool_name: str, arguments: dict[str, Any],
    ) -> Any: ...

    async def invoke_effect(
        self, tool_name: str, arguments: dict[str, Any], *, plan: Any, phase: str,
    ) -> Any: ...

    async def finalize_effect(self, plan: Any, terminal_state: str) -> Any | None: ...
