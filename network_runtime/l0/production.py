"""Authoritative L0 v2 catalog for every reviewed Runtime mutation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from network_runtime.policies import ToolContract, reviewed_contracts
from network_runtime.provider_contracts import REGISTRY as PROVIDER_CAPABILITIES

from .catalog import L0Catalog
from .compiler import compile_documents, parse_document
from .models import CompiledAtomicEffect, ParameterSpec


@dataclass(frozen=True)
class ProductionDefinition:
    skill_id: str
    version: str
    tool_name: str
    intent_kind: str
    target_fields: tuple[str, ...]
    profiles: tuple[str, ...]
    parameters: tuple[str, ...]
    desired_state: dict[str, Any]


@dataclass(frozen=True)
class RuntimeBinding:
    skill_id: str
    version: str
    tool_name: str
    tool_contract_id: str
    verifier_id: str
    compensator_id: str | None

    @property
    def key(self) -> tuple[str, str]:
        return self.skill_id, self.version


def _a(name: str) -> str:
    return "${arguments." + name + "}"


PRODUCTION_DEFINITIONS: tuple[ProductionDefinition, ...] = (
    ProductionDefinition("network.device.config.edit", "1.0.0", "edit_device_config", "configure_device", ("device_id",), ("lan", "dc"), ("device_id", "section", "changes", "config_lines", "reason", "verification_probe_id"), {"requested_configuration_digest": "${intent.configuration_digest}"}),
    ProductionDefinition("network.device.config.push", "1.0.0", "push_config", "configure_device", ("device_id",), ("lan",), ("device_id", "config_text", "dry_run"), {"requested_configuration_digest": "${intent.configuration_digest}"}),
    ProductionDefinition("network.service.restart", "1.0.0", "restart_service", "restart_service", ("service",), ("lan",), ("service", "environment"), {"service_health": "healthy", "rollout": "complete"}),
    ProductionDefinition("network.service.rollback", "1.0.0", "rollback_service", "rollback_service", ("service",), ("lan",), ("service", "version", "environment"), {"service_health": "healthy", "rollback": "complete"}),
    ProductionDefinition("network.deploy.rollback", "1.0.0", "rollback_deploy", "rollback_deployment", ("deploy_id",), ("lan",), ("deploy_id", "scope"), {"rolled_back": True, "services_healthy": True}),
    ProductionDefinition("network.node.drain", "1.0.0", "drain_node", "drain_node", ("node_id",), ("lan",), ("node_id", "grace_period_s"), {"drained": True, "schedulable": False, "pending": 0, "failed": 0}),
    ProductionDefinition("network.resource.failover", "1.0.0", "failover", "failover_resource", ("resource_id",), ("lan",), ("resource_id", "target"), {"primary": _a("target"), "healthy": True}),
    ProductionDefinition("network.resource.delete", "1.0.0", "delete_resource", "delete_resource", ("resource_id",), ("lan",), ("resource_id", "force"), {"exists": False}),
    ProductionDefinition("network.lan.user-access.grant", "1.0.0", "grant_user_access", "grant_network_access", ("user_id",), ("lan",), ("user_id", "reason"), {"admitted": True}),
    ProductionDefinition("network.lan.user-access.revoke", "1.0.0", "revoke_user_access", "revoke_network_access", ("user_id",), ("lan",), ("user_id", "reason"), {"admitted": False}),
    ProductionDefinition("network.dc.fabric-config.push", "1.0.0", "dc_config_push", "configure_fabric", ("node",), ("dc",), ("node", "config_lines", "reason"), {"requested_configuration_digest": "${intent.configuration_digest}"}),
    ProductionDefinition("network.dc.app-access.grant", "1.0.0", "dc_grant_app_access", "grant_application_access", ("user_id", "app_id"), ("dc",), ("user_id", "app_id", "role", "reason"), {"allowed": True}),
    ProductionDefinition("network.dc.app-access.revoke", "1.0.0", "dc_revoke_app_access", "revoke_application_access", ("user_id", "app_id"), ("dc",), ("user_id", "app_id", "reason"), {"allowed": False}),
    ProductionDefinition("network.wan.path.failover", "1.0.0", "wan_failover_path", "failover_wan_path", ("tunnel",), ("wan",), ("tunnel", "to_transport"), {"transport": _a("to_transport"), "state": "up"}),
    ProductionDefinition("network.fabric.access-vlan.set", "1.0.0", "fabric_set_access_vlan", "set_access_vlan", ("device_id", "interface"), ("dc",), ("device_id", "interface", "vlan_id", "reason", "verification_probe_id"), {"vlan_id": _a("vlan_id")}),
    ProductionDefinition("service.access.entitlement.grant", "1.0.0", "access_policy_grant_entitlement", "grant_service_entitlement", ("user_id", "app_id"), ("lan", "dc"), ("user_id", "app_id", "role", "change_id", "expected_revision", "reason", "correlation_id"), {"allowed": True, "role": _a("role")}),
    ProductionDefinition("service.access.entitlement.revoke", "1.0.0", "access_policy_revoke_entitlement", "revoke_service_entitlement", ("user_id", "app_id"), ("lan", "dc"), ("user_id", "app_id", "change_id", "expected_revision", "reason", "correlation_id"), {"allowed": False, "roles": []}),
    ProductionDefinition("service.platform.restart", "1.0.0", "platform_restart_service", "restart_platform_service", ("service", "environment"), ("lan", "dc"), ("service", "environment", "change_id", "expected_revision", "reason", "correlation_id"), {"status": "healthy", "rollout": "complete"}),
    ProductionDefinition("service.platform.rollback", "1.0.0", "platform_rollback_service", "rollback_platform_service", ("service", "environment"), ("lan", "dc"), ("service", "environment", "version", "change_id", "expected_revision", "reason", "correlation_id"), {"status": "healthy", "version": _a("version")}),
    ProductionDefinition("network.application.enforcement.apply", "1.0.0", "network_apply_app_enforcement", "apply_network_application_enforcement", ("user_id", "app_id"), ("lan", "dc"), ("user_id", "app_id", "change_id", "reason"), {"allowed": True}),
    ProductionDefinition("network.application.enforcement.revoke", "1.0.0", "network_revoke_app_enforcement", "revoke_network_application_enforcement", ("user_id", "app_id"), ("lan", "dc"), ("user_id", "app_id", "change_id", "reason"), {"allowed": False}),
)


_REQUIRED: dict[str, frozenset[str]] = {
    "edit_device_config": frozenset({"device_id", "reason"}),
    "push_config": frozenset({"device_id", "config_text"}),
    "restart_service": frozenset({"service", "environment"}),
    "rollback_service": frozenset({"service", "version", "environment"}),
    "rollback_deploy": frozenset({"deploy_id"}),
    "drain_node": frozenset({"node_id"}),
    "failover": frozenset({"resource_id", "target"}),
    "delete_resource": frozenset({"resource_id"}),
    "grant_user_access": frozenset({"user_id", "reason"}),
    "revoke_user_access": frozenset({"user_id", "reason"}),
    "dc_config_push": frozenset({"node", "config_lines", "reason"}),
    "dc_grant_app_access": frozenset({"user_id", "app_id", "reason"}),
    "dc_revoke_app_access": frozenset({"user_id", "app_id", "reason"}),
    "wan_failover_path": frozenset({"tunnel", "to_transport"}),
    "fabric_set_access_vlan": frozenset({"device_id", "interface", "vlan_id", "reason"}),
    "access_policy_grant_entitlement": frozenset({"user_id", "app_id", "role", "change_id", "expected_revision", "reason"}),
    "access_policy_revoke_entitlement": frozenset({"user_id", "app_id", "change_id", "expected_revision", "reason"}),
    "platform_restart_service": frozenset({"service", "environment", "change_id", "expected_revision", "reason"}),
    "platform_rollback_service": frozenset({"service", "environment", "version", "change_id", "expected_revision", "reason"}),
    "network_apply_app_enforcement": frozenset({"user_id", "app_id", "change_id", "reason"}),
    "network_revoke_app_enforcement": frozenset({"user_id", "app_id", "change_id", "reason"}),
}


def _parameter(name: str, required: bool) -> ParameterSpec:
    values: dict[str, Any] = {"type": "string", "required": required, "maxLength": 4096}
    if name in {"vlan_id", "expected_revision", "grace_period_s"}:
        values = {"type": "integer", "required": required, "minimum": 0}
    if name == "vlan_id":
        values.update({"minimum": 1, "maximum": 4094})
    if name == "grace_period_s":
        values["maximum"] = 3600
    if name in {"force", "dry_run"}:
        values = {"type": "boolean", "required": required}
    if name == "config_lines":
        values = {"type": "array", "required": required, "minLength": 1, "maxLength": 500}
    if name == "changes":
        values = {"type": "object", "required": required}
    if name == "config_text":
        values["maxLength"] = 65536
    if name == "reason":
        values.update({"minLength": 1, "maxLength": 4096})
    if name == "environment":
        values["enum"] = ["prod", "staging", "dev"]
    if name == "to_transport":
        values["enum"] = ["mpls", "broadband", "lte"]
    return ParameterSpec.model_validate(values)


def _capability(tool_name: str, fallback: str) -> str:
    value = PROVIDER_CAPABILITIES.for_tool(tool_name)
    return value.capability_id if value else fallback


def authoring_document(
    definition: ProductionDefinition,
    tool_contract: ToolContract,
) -> dict[str, Any]:
    """Return the authoritative source manifest for one production L0."""
    parameters = {
        name: _parameter(name, name in _REQUIRED[definition.tool_name]).model_dump(
            by_alias=True, mode="json",
        )
        for name in definition.parameters
    }
    preflight_tool = tool_contract.preflight_tool
    preflight_capability = _capability(
        preflight_tool, f"netopyu.tool.{preflight_tool}",
    ) if preflight_tool else "netopyu.runtime.contract.get"
    preflight_arguments = {
        field: _a(field) for field in tool_contract.preflight_fields
    }
    compensation = None
    failure_policy = {
        "beforeSend": "abort", "afterSendUnknown": "reconcile_read_only",
        "verificationFailed": "manual_intervention", "compensationFailed": "manual_intervention",
    }
    if tool_contract.compensator:
        rollback_tool = tool_contract.rollback_tool or tool_contract.compensator
        compensation = {
            "capability": _capability(
                rollback_tool, f"netopyu.compensator.{tool_contract.compensator}",
            ),
            "tool": tool_contract.rollback_tool,
            "arguments": {field: _a(field) for field in tool_contract.rollback_fields},
            "verification": {
                "capability": f"netopyu.rollback-verifier.{tool_contract.compensator}",
                "arguments": {field: _a(field) for field in tool_contract.preflight_fields},
                "predicates": [{"field": "restored", "operator": "equals", "expected": True}],
            },
        }
        failure_policy["verificationFailed"] = "compensate"
    risk = "critical" if definition.tool_name == "delete_resource" else "high"
    return {
        "apiVersion": "netopyu.io/l0-effect/v2",
        "kind": "AtomicEffect",
        "metadata": {
            "id": definition.skill_id, "version": definition.version,
            "owner": "netopyu-runtime",
            "description": f"Production L0 v2 contract for {definition.tool_name}",
            "labels": {
                "runtime-tool-contract": tool_contract.contract_id,
                "runtime-verifier": tool_contract.verifier,
                "runtime-preflight-fields": ",".join(tool_contract.preflight_fields),
                "migration-source": "l0-v1-reviewed",
            },
        },
        "spec": {
            "template": "netopyu-runtime-v2",
            "profiles": list(definition.profiles),
            "effect": {
                "capability": _capability(
                    definition.tool_name,
                    tool_contract.capability_id or f"netopyu.tool.{definition.tool_name}",
                ),
                "tool": definition.tool_name,
                "request": {name: _a(name) for name in definition.parameters},
            },
            "intent": {
                "kind": definition.intent_kind,
                "targetFields": list(definition.target_fields),
                "desiredState": definition.desired_state,
            },
            "parameters": parameters,
            "preflight": [{
                "id": "approved-state", "capability": preflight_capability,
                "arguments": preflight_arguments, "snapshotFields": ["facts"],
                "predicates": [{"field": "facts", "operator": "exists"}],
            }],
            "verification": {
                "capability": f"netopyu.verifier.{tool_contract.verifier}",
                "arguments": {field: _a(field) for field in definition.target_fields},
                "predicates": [{"field": "passed", "operator": "equals", "expected": True}],
            },
            "compensation": compensation,
            "approval": {"required": True, "risk": risk, "mode": "single"},
            "failurePolicy": failure_policy,
        },
    }


def build_production_catalog() -> tuple[L0Catalog, dict[tuple[str, str], RuntimeBinding]]:
    reviewed = reviewed_contracts()
    documents = []
    bindings: dict[tuple[str, str], RuntimeBinding] = {}
    for definition in PRODUCTION_DEFINITIONS:
        contract = reviewed[definition.tool_name]
        documents.append(parse_document(
            authoring_document(definition, contract),
            source=definition.skill_id,
        ))
        binding = RuntimeBinding(
            definition.skill_id, definition.version, definition.tool_name,
            contract.contract_id, contract.verifier, contract.compensator,
        )
        bindings[binding.key] = binding
    catalog = L0Catalog(compile_documents(documents))
    return catalog, bindings


CATALOG, BINDINGS = build_production_catalog()


def contracts() -> tuple[CompiledAtomicEffect, ...]:
    return tuple(
        item for item in CATALOG.contracts() if isinstance(item, CompiledAtomicEffect)
    )
