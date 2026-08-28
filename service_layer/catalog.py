"""Checked service-tool classification used before dynamic MCP discovery.

Schemas and provider identities still come from the connected MCP server. This
catalog only lets the deterministic L1 workflow compiler identify reads and
writes without starting integration processes.
"""

from __future__ import annotations

from typing import Any


_READS = {
    "identity_list_users", "identity_get_user",
    "application_list", "application_get", "application_check_access",
    "access_policy_evaluate", "access_policy_get_entitlement",
    "change_get", "change_validate_window", "cmdb_get_endpoint_binding",
    "platform_get_service_health",
}
_WRITES = {
    "access_policy_grant_entitlement", "access_policy_revoke_entitlement",
    "platform_restart_service", "platform_rollback_service",
}


def workflow_metadata() -> dict[str, dict[str, Any]]:
    values = {
        name: {
            "description": name.replace("_", " "), "parameters": {},
            "hitl": False, "action_type": "read_only", "tags": ["mcp", "service"],
        }
        for name in _READS
    }
    values.update({
        name: {
            "description": name.replace("_", " "), "parameters": {},
            "hitl": True, "action_type": "reversible", "tags": ["mcp", "service", "write"],
        }
        for name in _WRITES
    })
    return values
