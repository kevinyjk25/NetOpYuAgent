"""Versioned provider capabilities shared by Runtime and MCP adapters.

Tool names remain compatibility aliases for the current DSH/Hermes surfaces.
The stable identity used at the provider boundary is ``capability_id`` plus
``capability_version``.  This lets a future Containerlab, controller, or vendor
adapter implement the same reviewed behavior without inheriting a source-name
special case in the Runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


CAPABILITY_SCHEMA_VERSION = 1
CAPABILITY_VERSION = "1.0.0"
EVIDENCE_CONTRACT = "network-evidence-envelope-v1"


@dataclass(frozen=True)
class ProviderCapability:
    tool_name: str
    capability_id: str
    capability_version: str
    provider_role: str
    action_type: str


class ProviderCapabilityRegistry:
    def __init__(self) -> None:
        self._by_tool: dict[str, ProviderCapability] = {}
        self._by_id: dict[str, ProviderCapability] = {}

    def register(
        self,
        tool_name: str,
        capability_id: str,
        *,
        provider_role: str,
        action_type: str,
        version: str = CAPABILITY_VERSION,
    ) -> None:
        if provider_role not in {"observer", "actor"}:
            raise ValueError(f"invalid provider role {provider_role!r}")
        if action_type not in {"read_only", "reversible", "destructive"}:
            raise ValueError(f"invalid capability action type {action_type!r}")
        if tool_name in self._by_tool:
            raise RuntimeError(f"duplicate provider tool capability {tool_name!r}")
        if capability_id in self._by_id:
            raise RuntimeError(f"duplicate provider capability id {capability_id!r}")
        value = ProviderCapability(
            tool_name=tool_name,
            capability_id=capability_id,
            capability_version=version,
            provider_role=provider_role,
            action_type=action_type,
        )
        self._by_tool[tool_name] = value
        self._by_id[capability_id] = value

    def for_tool(self, tool_name: str) -> ProviderCapability | None:
        return self._by_tool.get(tool_name)

    def get(self, capability_id: str) -> ProviderCapability | None:
        return self._by_id.get(capability_id)

    def observer_tools(self) -> tuple[str, ...]:
        return tuple(sorted(
            item.tool_name for item in self._by_tool.values()
            if item.provider_role == "observer"
        ))

    def contracts(self) -> tuple[ProviderCapability, ...]:
        return tuple(self._by_id[key] for key in sorted(self._by_id))


REGISTRY = ProviderCapabilityRegistry()


def _observer(tool_name: str, capability_id: str) -> None:
    REGISTRY.register(
        tool_name, capability_id,
        provider_role="observer", action_type="read_only",
    )


def _actor(tool_name: str, capability_id: str, action_type: str) -> None:
    REGISTRY.register(
        tool_name, capability_id,
        provider_role="actor", action_type=action_type,
    )


# Inventory, device state and diagnostics.
_observer("list_devices", "network.inventory.devices.list")
_observer("get_device_status", "network.device.status.get")
_observer("get_device_config", "network.device.config.get")
_observer("validate_device_config", "network.device.config.validate")
_observer("get_syslog", "network.device.syslog.get")
_observer("query_interface_metrics", "network.interface.metrics.query")
_observer("get_bgp_summary", "network.routing.bgp.summary.get")
_observer("get_device_facts", "network.device.facts.get")
_observer("run_command", "network.device.command.read")
_observer("multi_device_check", "network.device.check.multi")
_observer("get_ospf_neighbors", "network.routing.ospf.neighbors.get")

# Reviewed topology, path and data-plane evidence.
_observer("lab_topology_status", "network.topology.runtime-status.get")
_observer("lab_get_topology_graph", "network.topology.graph.get")
_observer("lab_get_endpoint", "network.topology.endpoint.get")
_observer("lab_trace_path", "network.path.trace")
_observer("lab_get_enforcement_path", "network.path.enforcement.get")
_observer("lab_probe", "network.dataplane.icmp.probe")
_observer("lab_app_probe", "network.dataplane.http.probe")
_observer("network_get_app_enforcement", "network.application.enforcement.get")

# LAN and DC access views retained as compatibility capabilities.
_observer("list_users", "network.access.users.list")
_observer("get_user_access", "network.access.user.get")
_observer("check_nac_policy", "network.access.policy.check")
_observer("dc_list_apps", "network.dc.applications.list")
_observer("dc_get_app_acl", "network.dc.application-acl.get")
_observer("dc_check_user_app_access", "network.dc.application-access.check")

# EVPN/VXLAN observer capabilities.
_observer("lab_get_fabric_state", "network.fabric.state.get")
_observer("lab_get_access_vlan", "network.fabric.access-vlan.get")
_observer("lab_get_vxlan_state", "network.fabric.vxlan.state.get")
_observer("lab_get_bgp_evpn_summary", "network.fabric.bgp-evpn.summary.get")
_observer("lab_get_evpn_routes", "network.fabric.evpn.routes.get")

# Actor and compensation capabilities.  P0.9 keeps their implementation in
# the trusted local Lab provider; the stable ids are introduced now so an MCP
# actor can replace the implementation without changing L0 contracts later.
_actor("edit_device_config", "network.device.config.edit", "destructive")
_actor("restore_device_config", "network.device.config.restore", "reversible")
_actor("grant_user_access", "network.lan.user-access.grant", "destructive")
_actor("revoke_user_access", "network.lan.user-access.revoke", "destructive")
_actor("dc_grant_app_access", "network.dc.app-access.grant", "destructive")
_actor("dc_revoke_app_access", "network.dc.app-access.revoke", "destructive")
_actor("fabric_set_access_vlan", "network.fabric.access-vlan.set", "destructive")
_actor("fabric_restore_access_vlan", "network.fabric.access-vlan.restore", "reversible")
_actor("network_apply_app_enforcement", "network.application.enforcement.apply", "reversible")
_actor("network_revoke_app_enforcement", "network.application.enforcement.revoke", "reversible")
_actor("network_restore_app_enforcement", "network.application.enforcement.restore", "reversible")


def enrich_metadata(
    tool_name: str,
    metadata: dict[str, Any],
    *,
    provider_kind: str,
) -> dict[str, Any]:
    """Attach stable provider identity fields without changing tool semantics."""
    contract = REGISTRY.for_tool(tool_name)
    if contract is None:
        return dict(metadata)
    return {
        **dict(metadata),
        "capability_schema_version": CAPABILITY_SCHEMA_VERSION,
        "capability_id": contract.capability_id,
        "capability_version": contract.capability_version,
        "provider_role": contract.provider_role,
        "provider_kind": provider_kind,
    }


def validate_declaration(tool_name: str, declared: dict[str, Any]) -> ProviderCapability:
    """Fail closed when an external provider misstates a reviewed capability."""
    contract = REGISTRY.for_tool(tool_name)
    if contract is None:
        raise ValueError(f"unreviewed network provider tool {tool_name!r}")
    actual = {
        "capability_id": declared.get("capability_id"),
        "capability_version": declared.get("capability_version"),
        "provider_role": declared.get("provider_role"),
        "action_type": declared.get("action_type"),
    }
    expected = {
        "capability_id": contract.capability_id,
        "capability_version": contract.capability_version,
        "provider_role": contract.provider_role,
        "action_type": contract.action_type,
    }
    if actual != expected:
        raise ValueError(
            f"network provider capability mismatch for {tool_name}: "
            f"expected={expected!r}, observed={actual!r}"
        )
    return contract
