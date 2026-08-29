"""P2.1 multi-team Capability Catalog governance without effect authority.

The catalog describes ownership, tenancy, delegation, consumers, lifecycle,
and compatibility for exact compiled L0 contracts.  Its decisions authorize
catalog workflow only: they never approve a Runtime plan, register a contract,
publish a Provider release, or invoke an effect.
"""

from __future__ import annotations

from datetime import datetime, timezone
from fnmatch import fnmatchcase
import json
from pathlib import Path
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
import yaml

from network_runtime.contracts import sha256_json


GOVERNANCE_CATALOG_SCHEMA = "netopyu.io/capability-governance-catalog/v1"
GOVERNANCE_DECISION_SCHEMA = "netopyu.io/capability-governance-decision/v1"
GOVERNANCE_COMPATIBILITY_SCHEMA = "netopyu.io/capability-governance-compatibility/v1"
_MAX_DOCUMENT_BYTES = 4_000_000
_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{1,127}$")
_PATTERN = re.compile(r"^[a-z0-9*][a-z0-9*_.-]{0,127}$")
_SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")

GovernanceAction = Literal[
    "discover", "bind_read", "propose_write", "review", "publish", "deprecate",
]


class CatalogGovernanceError(RuntimeError):
    pass


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _valid_id(value: str, label: str) -> str:
    if not _ID.fullmatch(value):
        raise ValueError(f"{label} is invalid")
    return value


def _version_key(value: str) -> tuple[int, int, int]:
    match = _SEMVER.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid semantic version {value!r}")
    return tuple(int(item) for item in match.groups())  # type: ignore[return-value]


class GovernanceMetadata(_StrictModel):
    id: str
    version: str

    @model_validator(mode="after")
    def validate_metadata(self) -> "GovernanceMetadata":
        _valid_id(self.id, "catalog id")
        _version_key(self.version)
        return self


class TenantScope(_StrictModel):
    id: str
    environments: tuple[str, ...]

    @model_validator(mode="after")
    def validate_scope(self) -> "TenantScope":
        _valid_id(self.id, "tenant id")
        if not self.environments or len(set(self.environments)) != len(self.environments):
            raise ValueError("tenant environments must be non-empty and unique")
        for value in self.environments:
            _valid_id(value, "environment")
        return self


class Team(_StrictModel):
    id: str
    display_name: str = Field(alias="displayName")

    @model_validator(mode="after")
    def validate_team(self) -> "Team":
        _valid_id(self.id, "team id")
        if not self.display_name.strip() or len(self.display_name) > 160:
            raise ValueError("team displayName is invalid")
        return self


class ConsumerBinding(_StrictModel):
    consumer_id: str = Field(alias="consumerId")
    team_id: str = Field(alias="teamId")
    environments: tuple[str, ...]
    contract_hash: str = Field(alias="contractHash")

    @model_validator(mode="after")
    def validate_consumer(self) -> "ConsumerBinding":
        _valid_id(self.consumer_id, "consumer id")
        _valid_id(self.team_id, "consumer team id")
        if not self.environments:
            raise ValueError("consumer environments cannot be empty")
        if not self.contract_hash.startswith("sha256:"):
            raise ValueError("consumer contractHash must be a sha256 digest")
        return self


class CapabilityDependency(_StrictModel):
    capability_id: str = Field(alias="capabilityId")
    version: str
    contract_hash: str = Field(alias="contractHash")

    @model_validator(mode="after")
    def validate_dependency(self) -> "CapabilityDependency":
        _valid_id(self.capability_id, "dependency capability id")
        _version_key(self.version)
        if not self.contract_hash.startswith("sha256:"):
            raise ValueError("dependency contractHash must be a sha256 digest")
        return self

    @property
    def key(self) -> tuple[str, str]:
        return self.capability_id, self.version


class GovernedCapability(_StrictModel):
    id: str
    version: str
    namespace: str
    owner_team: str = Field(alias="ownerTeam")
    steward_team: str = Field(alias="stewardTeam")
    tenant: str
    environments: tuple[str, ...]
    profiles: tuple[str, ...]
    domain: str
    kind: Literal["observation", "effect"]
    contract_hash: str = Field(alias="contractHash")
    input_schema_digest: str = Field(alias="inputSchemaDigest")
    output_schema_digest: str = Field(alias="outputSchemaDigest")
    lifecycle: Literal["draft", "active", "deprecated", "retired"] = "active"
    supersedes: str | None = None
    dependencies: tuple[CapabilityDependency, ...] = ()
    consumers: tuple[ConsumerBinding, ...] = ()

    @model_validator(mode="after")
    def validate_capability(self) -> "GovernedCapability":
        _valid_id(self.id, "capability id")
        _valid_id(self.namespace, "namespace")
        _valid_id(self.owner_team, "owner team")
        _valid_id(self.steward_team, "steward team")
        _valid_id(self.tenant, "tenant")
        _valid_id(self.domain, "domain")
        _version_key(self.version)
        if self.owner_team == self.steward_team:
            raise ValueError("owner and steward teams must be separated")
        if not self.id.startswith(self.namespace + "."):
            raise ValueError("capability id is outside its namespace")
        if not self.environments or not self.profiles:
            raise ValueError("capability environments and profiles cannot be empty")
        for label, values in (("environment", self.environments), ("profile", self.profiles)):
            if len(set(values)) != len(values):
                raise ValueError(f"capability {label}s must be unique")
            for value in values:
                _valid_id(value, label)
        for digest in (
            self.contract_hash, self.input_schema_digest, self.output_schema_digest,
        ):
            if not digest.startswith("sha256:"):
                raise ValueError("capability digests must use sha256")
        if self.supersedes is not None and not self.supersedes.startswith("sha256:"):
            raise ValueError("supersedes must be a sha256 digest")
        if len({item.key for item in self.dependencies}) != len(self.dependencies):
            raise ValueError("capability dependencies must be unique")
        if self.key in {item.key for item in self.dependencies}:
            raise ValueError("capability cannot depend on itself")
        if len({item.consumer_id for item in self.consumers}) != len(self.consumers):
            raise ValueError("consumer ids must be unique per capability")
        return self

    @property
    def key(self) -> tuple[str, str]:
        return self.id, self.version


class Delegation(_StrictModel):
    id: str
    delegator_team: str = Field(alias="delegatorTeam")
    delegatee_team: str = Field(alias="delegateeTeam")
    actions: tuple[GovernanceAction, ...]
    capability_patterns: tuple[str, ...] = Field(alias="capabilityPatterns")
    tenant: str
    environments: tuple[str, ...]
    expires_at: datetime | None = Field(default=None, alias="expiresAt")

    @model_validator(mode="after")
    def validate_delegation(self) -> "Delegation":
        _valid_id(self.id, "delegation id")
        _valid_id(self.delegator_team, "delegator team")
        _valid_id(self.delegatee_team, "delegatee team")
        _valid_id(self.tenant, "delegation tenant")
        if self.delegator_team == self.delegatee_team:
            raise ValueError("self-delegation is forbidden")
        if not self.actions or len(set(self.actions)) != len(self.actions):
            raise ValueError("delegation actions must be non-empty and unique")
        if "review" in self.actions and "publish" in self.actions:
            raise ValueError("review and publish must be delegated separately")
        if not self.capability_patterns or not self.environments:
            raise ValueError("delegation patterns and environments cannot be empty")
        for pattern in self.capability_patterns:
            if not _PATTERN.fullmatch(pattern):
                raise ValueError("delegation capability pattern is invalid")
        for environment in self.environments:
            _valid_id(environment, "delegation environment")
        if self.expires_at is not None and self.expires_at.tzinfo is None:
            raise ValueError("delegation expiresAt must be timezone-aware")
        return self


class GovernanceCatalog(_StrictModel):
    api_version: Literal[GOVERNANCE_CATALOG_SCHEMA] = Field(alias="apiVersion")
    metadata: GovernanceMetadata
    tenants: tuple[TenantScope, ...]
    teams: tuple[Team, ...]
    capabilities: tuple[GovernedCapability, ...]
    delegations: tuple[Delegation, ...]
    catalog_hash: str = Field(alias="catalogHash")

    @model_validator(mode="after")
    def validate_catalog(self) -> "GovernanceCatalog":
        if not self.tenants or not self.teams or not self.capabilities:
            raise ValueError("catalog tenants, teams, and capabilities cannot be empty")
        tenants = {item.id: set(item.environments) for item in self.tenants}
        teams = {item.id for item in self.teams}
        if len(tenants) != len(self.tenants) or len(teams) != len(self.teams):
            raise ValueError("catalog tenant/team ids must be unique")
        by_key = {item.key: item for item in self.capabilities}
        if len(by_key) != len(self.capabilities):
            raise ValueError("catalog capability id/version pairs must be unique")
        for capability in self.capabilities:
            if capability.owner_team not in teams or capability.steward_team not in teams:
                raise ValueError("capability owner/steward team is unknown")
            if capability.tenant not in tenants:
                raise ValueError("capability tenant is unknown")
            if not set(capability.environments).issubset(tenants[capability.tenant]):
                raise ValueError("capability environment is outside its tenant")
            for consumer in capability.consumers:
                if consumer.team_id not in teams:
                    raise ValueError("capability consumer team is unknown")
                if not set(consumer.environments).issubset(capability.environments):
                    raise ValueError("consumer environment is outside capability scope")
                if consumer.contract_hash != capability.contract_hash:
                    raise ValueError("consumer binding does not match capability contract")
            for dependency in capability.dependencies:
                referenced = by_key.get(dependency.key)
                if referenced is None:
                    raise ValueError("capability dependency is unknown")
                if dependency.contract_hash != referenced.contract_hash:
                    raise ValueError("capability dependency contract has drifted")
            if capability.supersedes is not None and not any(
                item.id == capability.id
                and _version_key(item.version) < _version_key(capability.version)
                and item.contract_hash == capability.supersedes
                for item in self.capabilities
            ):
                raise ValueError("supersedes must bind an older catalog contract")
        dependency_graph = {
            item.key: tuple(dependency.key for dependency in item.dependencies)
            for item in self.capabilities
        }
        visiting: set[tuple[str, str]] = set()
        visited: set[tuple[str, str]] = set()

        def visit(key: tuple[str, str]) -> None:
            if key in visiting:
                raise ValueError("capability dependency cycle is forbidden")
            if key in visited:
                return
            visiting.add(key)
            for dependency in dependency_graph[key]:
                visit(dependency)
            visiting.remove(key)
            visited.add(key)

        for key in dependency_graph:
            visit(key)
        if len({item.id for item in self.delegations}) != len(self.delegations):
            raise ValueError("delegation ids must be unique")
        for delegation in self.delegations:
            if delegation.delegator_team not in teams or delegation.delegatee_team not in teams:
                raise ValueError("delegation team is unknown")
            if delegation.tenant not in tenants:
                raise ValueError("delegation tenant is unknown")
            if not set(delegation.environments).issubset(tenants[delegation.tenant]):
                raise ValueError("delegation environment is outside its tenant")
            matched = [
                capability for capability in self.capabilities
                if capability.tenant == delegation.tenant
                and any(fnmatchcase(capability.id, pattern) for pattern in delegation.capability_patterns)
            ]
            if not matched:
                raise ValueError("delegation does not match any capability")
            if any(item.owner_team != delegation.delegator_team for item in matched):
                raise ValueError("only the capability owner may delegate governance actions")
            if any(not set(delegation.environments).issubset(item.environments) for item in matched):
                raise ValueError("delegation environment exceeds capability scope")
        payload = self.model_dump(by_alias=True, mode="json")
        declared = payload.pop("catalogHash")
        if declared != sha256_json(payload):
            raise ValueError("catalogHash does not bind the governance catalog")
        return self

    def require(self, capability_id: str, version: str) -> GovernedCapability:
        for capability in self.capabilities:
            if capability.key == (capability_id, version):
                return capability
        raise KeyError(f"unknown governed capability {capability_id}@{version}")


def seal_governance_catalog(payload: dict[str, Any]) -> GovernanceCatalog:
    value = json.loads(json.dumps(payload, ensure_ascii=False))
    if "catalogHash" in value:
        raise CatalogGovernanceError("unsealed catalog body must not contain catalogHash")
    try:
        normalized = {
            "apiVersion": value["apiVersion"],
            "metadata": GovernanceMetadata.model_validate(value["metadata"]).model_dump(
                by_alias=True, mode="json",
            ),
            "tenants": [
                TenantScope.model_validate(item).model_dump(by_alias=True, mode="json")
                for item in value["tenants"]
            ],
            "teams": [
                Team.model_validate(item).model_dump(by_alias=True, mode="json")
                for item in value["teams"]
            ],
            "capabilities": [
                GovernedCapability.model_validate(item).model_dump(by_alias=True, mode="json")
                for item in value["capabilities"]
            ],
            "delegations": [
                Delegation.model_validate(item).model_dump(by_alias=True, mode="json")
                for item in value["delegations"]
            ],
        }
        normalized["catalogHash"] = sha256_json(normalized)
        return GovernanceCatalog.model_validate(normalized)
    except (KeyError, TypeError, ValidationError) as error:
        raise CatalogGovernanceError("governance catalog is invalid") from error


def load_governance_catalog(path: str | Path) -> GovernanceCatalog:
    supplied = Path(path).expanduser()
    if supplied.is_symlink() or not supplied.is_file():
        raise CatalogGovernanceError("governance catalog file is missing or unsafe")
    raw = supplied.read_bytes()
    if len(raw) > _MAX_DOCUMENT_BYTES:
        raise CatalogGovernanceError("governance catalog exceeds 4 MB")
    try:
        payload = yaml.safe_load(raw)
        if not isinstance(payload, dict):
            raise TypeError("catalog root is not an object")
        return GovernanceCatalog.model_validate(payload)
    except (UnicodeDecodeError, TypeError, yaml.YAMLError, ValidationError) as error:
        raise CatalogGovernanceError("governance catalog is invalid") from error


def dump_governance_catalog(catalog: GovernanceCatalog) -> str:
    return yaml.safe_dump(
        catalog.model_dump(by_alias=True, mode="json"),
        allow_unicode=True, sort_keys=False,
    )


def bootstrap_runtime_governance_catalog() -> GovernanceCatalog:
    """Project every activated production L0 contract into a governed catalog."""
    from network_runtime.l0.models import CompiledAtomicEffect
    from network_runtime.l0.production import BINDINGS, CATALOG
    from network_runtime.provider_contracts import REGISTRY

    team_names = {
        "network-platform": "Network Platform",
        "service-platform": "Service Platform",
        "risk-governance": "Risk Governance",
        "release-management": "Release Management",
        "lan-operations": "LAN Operations",
        "dc-operations": "DC Operations",
        "wan-operations": "WAN Operations",
    }
    capabilities: list[dict[str, Any]] = []
    for contract in CATALOG.contracts():
        if not isinstance(contract, CompiledAtomicEffect):
            continue
        binding = BINDINGS[(contract.metadata.id, contract.metadata.version)]
        provider = REGISTRY.for_tool(binding.tool_name)
        owner = "service-platform" if contract.metadata.id.startswith("service.") else "network-platform"
        profiles = tuple(contract.spec.profiles)
        consumers = [{
            "consumerId": f"{profile}-runtime",
            "teamId": f"{profile}-operations",
            "environments": ["local-simulation"],
            "contractHash": contract.contract_hash,
        } for profile in profiles]
        capabilities.append({
            "id": contract.metadata.id,
            "version": contract.metadata.version,
            "namespace": contract.metadata.id.split(".", 1)[0],
            "ownerTeam": owner,
            "stewardTeam": "risk-governance",
            "tenant": "local-lab",
            "environments": ["local-simulation"],
            "profiles": list(profiles),
            "domain": contract.metadata.id.split(".", 1)[0],
            "kind": "effect",
            "contractHash": contract.contract_hash,
            "inputSchemaDigest": sha256_json({
                name: value.model_dump(by_alias=True, mode="json")
                for name, value in contract.spec.parameters.items()
            }),
            "outputSchemaDigest": sha256_json({
                "providerCapability": provider.capability_id if provider else None,
                "verification": contract.spec.verification.model_dump(
                    by_alias=True, mode="json",
                ),
            }),
            "lifecycle": "active",
            "consumers": consumers,
        })
    delegations: list[dict[str, Any]] = []
    profiles_by_owner = {
        "network-platform": ("lan", "dc", "wan"),
        "service-platform": ("lan", "dc"),
    }
    for owner, profiles in profiles_by_owner.items():
        namespace = owner.removesuffix("-platform")
        for profile in profiles:
            delegations.append({
                "id": f"{namespace}-{profile}-proposal",
                "delegatorTeam": owner,
                "delegateeTeam": f"{profile}-operations",
                "actions": ["discover", "propose_write"],
                "capabilityPatterns": [f"{namespace}.*"],
                "tenant": "local-lab",
                "environments": ["local-simulation"],
            })
        delegations.extend([
            {
                "id": f"{namespace}-risk-review",
                "delegatorTeam": owner,
                "delegateeTeam": "risk-governance",
                "actions": ["discover", "review"],
                "capabilityPatterns": [f"{namespace}.*"],
                "tenant": "local-lab",
                "environments": ["local-simulation"],
            },
            {
                "id": f"{namespace}-release",
                "delegatorTeam": owner,
                "delegateeTeam": "release-management",
                "actions": ["publish", "deprecate"],
                "capabilityPatterns": [f"{namespace}.*"],
                "tenant": "local-lab",
                "environments": ["local-simulation"],
            },
        ])
    return seal_governance_catalog({
        "apiVersion": GOVERNANCE_CATALOG_SCHEMA,
        "metadata": {"id": "netopyu-local", "version": "1.0.0"},
        "tenants": [{"id": "local-lab", "environments": ["local-simulation"]}],
        "teams": [
            {"id": key, "displayName": value}
            for key, value in sorted(team_names.items())
        ],
        "capabilities": sorted(capabilities, key=lambda item: (item["id"], item["version"])),
        "delegations": sorted(delegations, key=lambda item: item["id"]),
    })


def validate_runtime_catalog_binding(catalog: GovernanceCatalog) -> dict[str, Any]:
    from network_runtime.l0.production import CATALOG

    runtime = {
        (item.metadata.id, item.metadata.version): (
            item.contract_hash if hasattr(item, "contract_hash") else item.definition_hash,
            tuple(item.spec.profiles) if hasattr(item, "spec") else (),
        )
        for item in CATALOG.contracts()
    }
    governed = {
        item.key: (item.contract_hash, tuple(item.profiles))
        for item in catalog.capabilities if item.lifecycle == "active"
    }
    missing = sorted(f"{key[0]}@{key[1]}" for key in set(runtime) - set(governed))
    unknown = sorted(f"{key[0]}@{key[1]}" for key in set(governed) - set(runtime))
    drift = sorted(
        f"{key[0]}@{key[1]}" for key in set(runtime) & set(governed)
        if runtime[key] != governed[key]
    )
    body = {
        "ok": not (missing or unknown or drift),
        "runtime_contracts": len(runtime),
        "active_governed_capabilities": len(governed),
        "missing": missing,
        "unknown": unknown,
        "contract_or_profile_drift": drift,
        "catalog_hash": catalog.catalog_hash,
        "authority": "catalog_governance_only",
        "runtime_read_authority": False,
        "runtime_effect_authority": False,
    }
    return {**body, "report_hash": sha256_json(body)}


def evaluate_catalog_access(
    catalog: GovernanceCatalog,
    *,
    team_id: str,
    action: GovernanceAction,
    capability_id: str,
    version: str,
    tenant: str,
    environment: str,
    at: datetime | None = None,
) -> dict[str, Any]:
    now = at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise CatalogGovernanceError("evaluation time must be timezone-aware")
    try:
        capability = catalog.require(capability_id, version)
    except KeyError as error:
        raise CatalogGovernanceError(str(error)) from error
    teams = {item.id for item in catalog.teams}
    reasons: list[str] = []
    matched_delegations: list[str] = []
    allowed = True
    if team_id not in teams:
        allowed = False
        reasons.append("unknown_team")
    if capability.tenant != tenant or environment not in capability.environments:
        allowed = False
        reasons.append("scope_mismatch")
    if capability.lifecycle == "retired" or (
        capability.lifecycle == "deprecated" and action in {"propose_write", "publish"}
    ):
        allowed = False
        reasons.append("lifecycle_denied")
    if (
        (capability.kind == "observation" and action == "propose_write")
        or (capability.kind == "effect" and action == "bind_read")
    ):
        allowed = False
        reasons.append("capability_kind_action_mismatch")
    inherent = (
        (team_id == capability.owner_team and action in {"discover", "bind_read", "propose_write"})
        or (team_id == capability.steward_team and action in {"discover", "review"})
    )
    for delegation in catalog.delegations:
        if (
            delegation.delegatee_team == team_id
            and action in delegation.actions
            and delegation.tenant == tenant
            and environment in delegation.environments
            and any(fnmatchcase(capability.id, pattern) for pattern in delegation.capability_patterns)
            and (delegation.expires_at is None or delegation.expires_at > now)
        ):
            matched_delegations.append(delegation.id)
    if not inherent and not matched_delegations:
        allowed = False
        reasons.append("no_governance_grant")
    if action == "publish" and (team_id in {capability.owner_team, capability.steward_team}):
        allowed = False
        reasons.append("publish_separation_of_duty")
    body = {
        "apiVersion": GOVERNANCE_DECISION_SCHEMA,
        "allowed": allowed,
        "action": action,
        "capability": f"{capability.id}@{capability.version}",
        "capability_hash": capability.contract_hash,
        "team_digest": sha256_json({"team_id": team_id}),
        "tenant": tenant,
        "environment": environment,
        "matched_delegation_digests": [
            sha256_json({"delegation_id": item}) for item in sorted(matched_delegations)
        ],
        "reasons": sorted(set(reasons)),
        "catalog_hash": catalog.catalog_hash,
        "authority": "catalog_governance_only",
        "runtime_read_authority": False,
        "runtime_effect_authority": False,
        "provider_publication_authority": False,
    }
    return {**body, "decision_hash": sha256_json(body)}


def catalog_compatibility_report(
    previous: GovernanceCatalog,
    candidate: GovernanceCatalog,
) -> dict[str, Any]:
    breaking: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    impacts: list[dict[str, Any]] = []
    if previous.metadata.id != candidate.metadata.id:
        breaking.append({"code": "CATALOG_ID_CHANGED"})
    try:
        if _version_key(candidate.metadata.version) <= _version_key(previous.metadata.version):
            breaking.append({"code": "CATALOG_VERSION_NOT_INCREASED"})
    except ValueError:
        breaking.append({"code": "CATALOG_VERSION_INVALID"})

    old = {item.key: item for item in previous.capabilities}
    new = {item.key: item for item in candidate.capabilities}
    for key in sorted(set(old) - set(new)):
        capability = old[key]
        breaking.append({"code": "CAPABILITY_REMOVED", "capability": f"{key[0]}@{key[1]}"})
        for consumer in capability.consumers:
            impacts.append({
                "consumer_digest": sha256_json({"consumer_id": consumer.consumer_id}),
                "capability": f"{key[0]}@{key[1]}",
                "reason": "capability_removed",
            })
    for key in sorted(set(new) - set(old)):
        review.append({"code": "CAPABILITY_ADDED", "capability": f"{key[0]}@{key[1]}"})
    lifecycle_rank = {"draft": 0, "active": 1, "deprecated": 2, "retired": 3}
    for key in sorted(set(old) & set(new)):
        before, after = old[key], new[key]
        changed_fields = [
            field for field in (
                "namespace", "owner_team", "steward_team", "tenant", "domain", "kind",
                "contract_hash", "input_schema_digest", "output_schema_digest", "profiles",
                "dependencies",
            ) if getattr(before, field) != getattr(after, field)
        ]
        removed_environments = sorted(set(before.environments) - set(after.environments))
        if changed_fields or removed_environments:
            breaking.append({
                "code": "IN_PLACE_CONTRACT_OR_SCOPE_CHANGE",
                "capability": f"{key[0]}@{key[1]}",
                "fields": changed_fields,
                "removed_environments": removed_environments,
            })
            for consumer in before.consumers:
                impacts.append({
                    "consumer_digest": sha256_json({"consumer_id": consumer.consumer_id}),
                    "capability": f"{key[0]}@{key[1]}",
                    "reason": "contract_or_scope_changed",
                })
        added_environments = sorted(set(after.environments) - set(before.environments))
        if added_environments:
            review.append({
                "code": "ENVIRONMENT_SCOPE_WIDENED",
                "capability": f"{key[0]}@{key[1]}",
                "environments": added_environments,
            })
        if lifecycle_rank[after.lifecycle] < lifecycle_rank[before.lifecycle]:
            breaking.append({
                "code": "LIFECYCLE_REGRESSION",
                "capability": f"{key[0]}@{key[1]}",
                "from": before.lifecycle,
                "to": after.lifecycle,
            })
        elif after.lifecycle != before.lifecycle:
            review.append({
                "code": "LIFECYCLE_ADVANCED",
                "capability": f"{key[0]}@{key[1]}",
                "from": before.lifecycle,
                "to": after.lifecycle,
            })
    old_grants = {
        (item.delegator_team, item.delegatee_team, item.actions,
         item.capability_patterns, item.tenant, item.environments)
        for item in previous.delegations
    }
    new_grants = {
        (item.delegator_team, item.delegatee_team, item.actions,
         item.capability_patterns, item.tenant, item.environments)
        for item in candidate.delegations
    }
    if new_grants - old_grants:
        review.append({"code": "DELEGATION_GRANTS_ADDED", "count": len(new_grants - old_grants)})
    if old_grants - new_grants:
        review.append({"code": "DELEGATION_GRANTS_REMOVED", "count": len(old_grants - new_grants)})
    body = {
        "apiVersion": GOVERNANCE_COMPATIBILITY_SCHEMA,
        "compatible": not breaking,
        "requires_review": bool(breaking or review),
        "previous_catalog_hash": previous.catalog_hash,
        "candidate_catalog_hash": candidate.catalog_hash,
        "breaking_changes": breaking,
        "review_changes": review,
        "consumer_impacts": impacts,
        "authority": "analysis_only",
        "activation_available": False,
    }
    return {**body, "report_hash": sha256_json(body)}


__all__ = [
    "CatalogGovernanceError",
    "CapabilityDependency",
    "Delegation",
    "GOVERNANCE_CATALOG_SCHEMA",
    "GOVERNANCE_COMPATIBILITY_SCHEMA",
    "GOVERNANCE_DECISION_SCHEMA",
    "GovernanceCatalog",
    "GovernedCapability",
    "bootstrap_runtime_governance_catalog",
    "catalog_compatibility_report",
    "dump_governance_catalog",
    "evaluate_catalog_access",
    "load_governance_catalog",
    "seal_governance_catalog",
    "validate_runtime_catalog_binding",
]
