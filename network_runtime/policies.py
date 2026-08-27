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


_CONTRACTS: dict[str, ToolContract] = {
    "edit_device_config": ToolContract(
        "device-config-v1", "get_device_config", ("device_id", "section"), "device-config",
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
) -> ToolContract | None:
    if not requires_approval and action_type == "read_only":
        return ToolContract("read-only-v1", None, (), "read-result")
    contract = _CONTRACTS.get(tool_name)
    if contract is None:
        return None
    # External MCP/OpenAPI writes cannot inherit a same-name local contract.
    if source not in {"profile-mock", "pragmatic-device"}:
        return None
    # Only the pragmatic edit tool currently owns verified snapshot/rollback
    # semantics. Other pragmatic writes must add an explicit contract first.
    if mode == "pragmatic" and tool_name != "edit_device_config":
        return None
    return contract


def project_arguments(arguments: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {name: arguments[name] for name in fields if name in arguments}
