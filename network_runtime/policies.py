"""Reviewed execution contracts for registered network tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolContract:
    contract_id: str
    preflight_tool: str | None
    preflight_fields: tuple[str, ...]
    verifier: str
    rollback_tool: str | None = None
    rollback_fields: tuple[str, ...] = ()
    compensator: str | None = None
    allowed_sources: tuple[str, ...] = ()
    requires_trusted_mcp: bool = False


_CONTRACTS: dict[str, ToolContract] = {
    "edit_device_config": ToolContract(
        "device-config-v1", "get_device_config", ("device_id", "section"), "device-config",
        "restore_device_config", ("device_id",), "device-config-snapshot-v1",
    ),
    "push_config": ToolContract(
        "mock-config-push-v1", "get_device_config", ("device_id",), "device-config",
    ),
    "restart_service": ToolContract(
        "service-restart-v1", "service_health", ("service", "environment"), "service-health",
    ),
    "rollback_service": ToolContract(
        "service-rollback-v1", "service_health", ("service", "environment"), "service-health",
    ),
    "rollback_deploy": ToolContract(
        "mock-rollback-deploy-v2", "mock_operation_status", ("deploy_id",), "mock-state",
    ),
    "drain_node": ToolContract("mock-drain-v2", "mock_operation_status", ("node_id",), "mock-state"),
    "failover": ToolContract("mock-failover-v2", "mock_operation_status", ("resource_id",), "mock-state"),
    "delete_resource": ToolContract("mock-delete-v2", "mock_operation_status", ("resource_id",), "mock-state"),
    "grant_user_access": ToolContract(
        "lan-access-grant-v1", "get_user_access", ("user_id",), "lan-access-granted",
        "revoke_user_access", ("user_id",), "inverse-tool-v1",
    ),
    "revoke_user_access": ToolContract(
        "lan-access-revoke-v1", "get_user_access", ("user_id",), "lan-access-revoked",
        "grant_user_access", ("user_id",), "inverse-tool-v1",
    ),
    "dc_config_push": ToolContract(
        "dc-config-v2", "dc_get_applied_config", ("node",), "dc-config",
    ),
    "dc_grant_app_access": ToolContract(
        "dc-access-grant-v1", "dc_check_user_app_access", ("user_id", "app_id"), "dc-access-granted",
        "dc_revoke_app_access", ("user_id", "app_id"), "inverse-tool-v1",
    ),
    "dc_revoke_app_access": ToolContract(
        "dc-access-revoke-v1", "dc_check_user_app_access", ("user_id", "app_id"), "dc-access-revoked",
        "dc_grant_app_access", ("user_id", "app_id"), "inverse-tool-v1",
    ),
    "wan_failover_path": ToolContract(
        "wan-failover-v1", "wan_tunnel_status", (), "wan-failover",
    ),
    "fabric_set_access_vlan": ToolContract(
        "fabric-access-vlan-v1",
        "lab_get_access_vlan", ("device_id", "interface"),
        "fabric-access-vlan",
        "fabric_restore_access_vlan", ("device_id", "interface"),
        "fabric-access-vlan-snapshot-v1",
    ),
    "access_policy_grant_entitlement": ToolContract(
        "service-entitlement-grant-v1",
        "access_policy_get_entitlement", ("user_id", "app_id"),
        "service-entitlement-granted",
        "access_policy_restore_entitlement", ("user_id", "app_id"),
        "service-entitlement-snapshot-v1",
        ("mcp:access-policy-service",), True,
    ),
    "access_policy_revoke_entitlement": ToolContract(
        "service-entitlement-revoke-v1",
        "access_policy_get_entitlement", ("user_id", "app_id"),
        "service-entitlement-revoked",
        "access_policy_restore_entitlement", ("user_id", "app_id"),
        "service-entitlement-snapshot-v1",
        ("mcp:access-policy-service",), True,
    ),
    "platform_restart_service": ToolContract(
        "service-platform-restart-v1",
        "platform_get_service_health", ("service", "environment"),
        "service-platform-healthy",
        "platform_restore_service", ("service", "environment"),
        "service-platform-snapshot-v1",
        ("mcp:platform-service",), True,
    ),
    "platform_rollback_service": ToolContract(
        "service-platform-rollback-v1",
        "platform_get_service_health", ("service", "environment"),
        "service-platform-healthy",
        "platform_restore_service", ("service", "environment"),
        "service-platform-snapshot-v1",
        ("mcp:platform-service",), True,
    ),
    "network_apply_app_enforcement": ToolContract(
        "network-app-enforcement-grant-v1",
        "network_get_app_enforcement", ("user_id", "app_id"),
        "network-app-enforcement-granted",
        "network_restore_app_enforcement", ("user_id", "app_id"),
        "network-app-enforcement-snapshot-v1",
        ("network-lab",), False,
    ),
    "network_revoke_app_enforcement": ToolContract(
        "network-app-enforcement-revoke-v1",
        "network_get_app_enforcement", ("user_id", "app_id"),
        "network-app-enforcement-revoked",
        "network_restore_app_enforcement", ("user_id", "app_id"),
        "network-app-enforcement-snapshot-v1",
        ("network-lab",), False,
    ),
}


def reviewed_contracts() -> dict[str, ToolContract]:
    """Return a copy for startup checks, audits and contract-coverage tests."""
    return dict(_CONTRACTS)


def resolve_contract(
    tool_name: str,
    *,
    action_type: str,
    requires_approval: bool,
    mode: str,
    source: str,
    metadata: dict[str, Any] | None = None,
) -> ToolContract | None:
    if not requires_approval and action_type == "read_only":
        return ToolContract("read-only-v1", None, (), "read-result")
    contract = _CONTRACTS.get(tool_name)
    if contract is None:
        return None
    if contract.allowed_sources and source not in contract.allowed_sources:
        return None
    if contract.requires_trusted_mcp:
        metadata = metadata or {}
        if (
            not metadata.get("trusted_for_writes")
            or metadata.get("declared_contract_id") != contract.contract_id
            or metadata.get("result_contract") != "structured-content-required-v1"
            or not str(metadata.get("provider_identity") or "").startswith(f"{source}:")
            or not str(metadata.get("input_schema_digest") or "").startswith("sha256:")
            or not str(metadata.get("output_schema_digest") or "").startswith("sha256:")
        ):
            return None
        return contract
    # External MCP/OpenAPI writes cannot inherit a same-name local contract.
    if source not in {"profile-mock", "pragmatic-device", "network-lab"}:
        return None
    # Pragmatic writes are fail-closed. The local lab may additionally expose
    # manifest-bound access controls whose verifiers and inverse operations are
    # implemented by the same provider. Real-device adapters cannot inherit
    # those simulator contracts merely by reusing a tool name.
    pragmatic_lab_writes = {
        "grant_user_access", "revoke_user_access",
        "dc_grant_app_access", "dc_revoke_app_access",
        "fabric_set_access_vlan",
        "network_apply_app_enforcement", "network_revoke_app_enforcement",
    }
    if mode == "pragmatic" and tool_name != "edit_device_config":
        if source != "network-lab" or tool_name not in pragmatic_lab_writes:
            return None
    return contract


def project_arguments(arguments: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {name: arguments[name] for name in fields if name in arguments}
