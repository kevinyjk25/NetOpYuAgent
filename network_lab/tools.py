"""Harness-facing tools backed by a constrained local network lab."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Awaitable, Callable

from .containerlab import ContainerlabProvider


ToolCallable = Callable[[dict[str, Any]], Awaitable[str]]


LAB_TOOL_METADATA: dict[str, dict[str, Any]] = {
    "lab_get_topology_graph": {
        "description": (
            "Return the exact operator-reviewed nodes, interfaces, IPs, zones and links. "
            "Use this instead of inferring topology from device configurations."
        ),
        "parameters": {},
        "returns": "Typed topology graph plus explicit simulation truth boundaries",
        "hitl": False,
        "action_type": "read_only",
        "tags": ["lab", "topology", "deterministic"],
    },
    "lab_get_endpoint": {
        "description": (
            "Resolve one exact lab endpoint, its role, zone, address, link and peer. "
            "Endpoints are not network devices."
        ),
        "parameters": {"endpoint_id": "Exact endpoint identifier from the reviewed manifest"},
        "required": ["endpoint_id"],
        "returns": "Typed endpoint and direct attachment relationship",
        "hitl": False,
        "action_type": "read_only",
        "tags": ["lab", "endpoint", "deterministic"],
    },
    "lab_trace_path": {
        "description": (
            "Run bounded traceroute between two declared endpoints and resolve every hop "
            "to an exact manifest node, ingress interface and link. Fails closed if any hop "
            "or adjacency cannot be proven."
        ),
        "parameters": {
            "source_endpoint": "Exact declared source endpoint identifier",
            "destination_endpoint": "Exact declared destination endpoint identifier",
        },
        "required": ["source_endpoint", "destination_endpoint"],
        "returns": "Observed hop-by-hop path with typed node/link resolution and fail-closed verdict",
        "hitl": False,
        "action_type": "read_only",
        "tags": ["lab", "path", "traceroute", "verification"],
    },
    "lab_get_enforcement_path": {
        "description": (
            "For one declared user/application pair, report the actual simulated admission "
            "and application-policy enforcement points and the observed data-plane path. "
            "Never describes interface-state NAC as RADIUS/802.1X or server routes as ACL/IAM."
        ),
        "parameters": {
            "user_id": "Exact declared lab user identifier",
            "app_id": "Exact declared lab application identifier",
        },
        "required": ["user_id", "app_id"],
        "returns": "Observed enforcement states, exact implementations, and verified path",
        "hitl": False,
        "action_type": "read_only",
        "tags": ["lab", "path", "policy", "verification"],
    },
    "lab_topology_status": {
        "description": "Read the declared local Containerlab topology and node runtime states.",
        "parameters": {},
        "returns": "Structured lab and node status",
        "hitl": False,
        "action_type": "read_only",
        "tags": ["lab", "status"],
    },
    "lab_probe": {
        "description": "Run one predeclared end-to-end ICMP probe; arbitrary destinations are rejected.",
        "parameters": {"probe_id": "Exact probe identifier from the reviewed lab manifest"},
        "required": ["probe_id"],
        "returns": "Typed transmitted/received packet evidence",
        "hitl": False,
        "action_type": "read_only",
        "tags": ["lab", "verification"],
    },
    "get_ospf_neighbors": {
        "description": "Read FRR OSPF neighbor state from a declared lab router.",
        "parameters": {"device_id": "Exact lab device identifier"},
        "required": ["device_id"],
        "returns": "FRR OSPF neighbor table",
        "hitl": False,
        "action_type": "read_only",
        "tags": ["routing", "ospf", "lab"],
    },
}

_ACCESS_COMMON_METADATA: dict[str, dict[str, Any]] = {
    "lab_app_probe": {
        "description": "Run a manifest-bound HTTP probe from one declared lab user to one declared application.",
        "parameters": {"user_id": "Declared lab user", "app_id": "Declared lab application"},
        "required": ["user_id", "app_id"],
        "returns": "Structured policy and HTTP reachability evidence",
        "hitl": False,
        "action_type": "read_only",
        "tags": ["lab", "application", "verification"],
    },
    "network_get_app_enforcement": {
        "description": (
            "Read actual Containerlab application-policy enforcement for one user/application. "
            "This is observed network state, not a business entitlement lookup."
        ),
        "parameters": {"user_id": "Declared lab user", "app_id": "Declared lab application"},
        "required": ["user_id", "app_id"],
        "returns": "Structured enforcement state and implementation boundary",
        "hitl": False,
        "action_type": "read_only",
        "tags": ["network", "lab", "enforcement", "verification"],
    },
    "network_apply_app_enforcement": {
        "description": (
            "Apply manifest-bound network enforcement allowing one user to one application. "
            "Requires Effect Runtime approval, verification and exact compensation."
        ),
        "parameters": {
            "user_id": "Declared lab user", "app_id": "Declared lab application",
            "change_id": "Approved enterprise change identifier", "reason": "Auditable reason",
        },
        "required": ["user_id", "app_id", "change_id", "reason"],
        "returns": "Structured network enforcement mutation",
        "hitl": True,
        "action_type": "reversible",
        "tags": ["network", "lab", "enforcement", "write"],
    },
    "network_revoke_app_enforcement": {
        "description": (
            "Apply manifest-bound network enforcement denying one user to one application. "
            "Requires Effect Runtime approval, verification and exact compensation."
        ),
        "parameters": {
            "user_id": "Declared lab user", "app_id": "Declared lab application",
            "change_id": "Approved enterprise change identifier", "reason": "Auditable reason",
        },
        "required": ["user_id", "app_id", "change_id", "reason"],
        "returns": "Structured network enforcement mutation",
        "hitl": True,
        "action_type": "reversible",
        "tags": ["network", "lab", "enforcement", "write"],
    },
}

_LAN_ACCESS_TOOLS = (
    "list_users", "get_user_access", "check_nac_policy",
    "grant_user_access", "revoke_user_access",
)
_DC_ACCESS_TOOLS = (
    "dc_list_apps", "dc_get_app_acl", "dc_check_user_app_access",
    "dc_grant_app_access", "dc_revoke_app_access",
)
_TOPOLOGY_TOOLS = (
    "lab_get_topology_graph", "lab_get_endpoint", "lab_trace_path",
    "lab_get_enforcement_path",
)

_FABRIC_METADATA: dict[str, dict[str, Any]] = {
    "lab_get_fabric_state": {
        "description": (
            "Read the actual 802.1Q, Linux bridge, VXLAN and FRR BGP EVPN state "
            "and compare it with the reviewed fabric manifest."
        ),
        "parameters": {},
        "returns": "Typed live fabric state with explicit protocol truth boundaries",
        "hitl": False,
        "action_type": "read_only",
        "tags": ["fabric", "evpn", "vxlan", "verification"],
    },
    "lab_get_access_vlan": {
        "description": "Read the actual Linux bridge, PVID and untagged state of one declared access port.",
        "parameters": {
            "device_id": "Exact declared VTEP device identifier",
            "interface": "Exact declared access interface",
        },
        "required": ["device_id", "interface"],
        "returns": "Observed bridge VLAN membership and PVID",
        "hitl": False,
        "action_type": "read_only",
        "tags": ["fabric", "vlan", "bridge", "verification"],
    },
    "lab_get_vxlan_state": {
        "description": "Read actual Linux VXLAN devices and FRR EVPN VNI state from one VTEP.",
        "parameters": {"device_id": "Exact declared VTEP device identifier"},
        "required": ["device_id"],
        "returns": "Linux VXLAN IDs, FRR VNIs and remote VTEPs",
        "hitl": False,
        "action_type": "read_only",
        "tags": ["fabric", "vxlan", "vni"],
    },
    "lab_get_bgp_evpn_summary": {
        "description": "Read the actual FRR L2VPN EVPN BGP neighbor state from one fabric node.",
        "parameters": {"device_id": "Exact declared fabric BGP device identifier"},
        "required": ["device_id"],
        "returns": "Structured EVPN peer state and received/sent prefixes",
        "hitl": False,
        "action_type": "read_only",
        "tags": ["fabric", "bgp", "evpn"],
    },
    "lab_get_evpn_routes": {
        "description": "Read actual FRR EVPN routes, optionally restricted to route type 2, 3, or 5.",
        "parameters": {
            "device_id": "Exact declared fabric BGP device identifier",
            "route_type": {"type": "integer", "description": "Optional EVPN route type: 2, 3, or 5"},
        },
        "required": ["device_id"],
        "returns": "Typed EVPN RIB entries and paths",
        "hitl": False,
        "action_type": "read_only",
        "tags": ["fabric", "bgp", "evpn", "route"],
    },
    "fabric_set_access_vlan": {
        "description": (
            "Move one manifest-declared access port to one declared VLAN using fixed Linux argv. "
            "Requires HITL, live state verification, an optional traffic probe and exact rollback."
        ),
        "parameters": {
            "device_id": "Exact declared VTEP device identifier",
            "interface": "Exact declared access interface",
            "vlan_id": {"type": "integer", "description": "Declared target VLAN ID"},
            "reason": "Operator change reason for the audit journal",
            "verification_probe_id": "Optional exact manifest probe that must pass after the VLAN change",
        },
        "required": ["device_id", "interface", "vlan_id", "reason"],
        "returns": "Observed VLAN mutation; success is decided by Network Runtime verification",
        "hitl": True,
        "action_type": "destructive",
        "tags": ["fabric", "vlan", "write", "destructive"],
    },
}


def lab_access_metadata(profile_id: str) -> dict[str, dict[str, Any]]:
    """Reuse the stable contracts while binding them to the real lab adapter."""
    if profile_id == "lan":
        from profiles.lan.tool_meta import TOOLS

        names = _LAN_ACCESS_TOOLS
    elif profile_id == "dc":
        from profiles.dc.tool_meta import TOOLS

        names = _DC_ACCESS_TOOLS
    else:
        return {}
    return {name: dict(TOOLS[name]) for name in names}


def lab_tool_metadata(
    profile_id: str, *, access_enabled: bool = True, topology_enabled: bool = True,
    fabric_enabled: bool = False,
) -> dict[str, dict[str, Any]]:
    base = dict(LAB_TOOL_METADATA)
    if fabric_enabled and profile_id == "dc":
        base.update(_FABRIC_METADATA)
    if not topology_enabled:
        for name in _TOPOLOGY_TOOLS:
            base.pop(name, None)
    elif not access_enabled:
        base.pop("lab_get_enforcement_path", None)
    if not access_enabled:
        return base
    return {
        **base,
        **_ACCESS_COMMON_METADATA,
        **lab_access_metadata(profile_id),
    }


class LabToolAdapter:
    """Maps existing pragmatic tool contracts to deterministic lab operations."""

    def __init__(self, provider: ContainerlabProvider) -> None:
        self.provider = provider

    def callables(self, profile_id: str | None = None) -> dict[str, ToolCallable]:
        values: dict[str, ToolCallable] = {
            "list_devices": self.list_devices,
            "get_device_status": self.get_device_status,
            "get_device_config": self.get_device_config,
            "edit_device_config": self.edit_device_config,
            "restore_device_config": self.restore_device_config,
            "validate_device_config": self.validate_device_config,
            "get_syslog": self.get_syslog,
            "query_interface_metrics": self.query_interface_metrics,
            "get_bgp_summary": self.get_bgp_summary,
            "get_device_facts": self.get_device_facts,
            "run_command": self.run_command,
            "multi_device_check": self.multi_device_check,
            "lab_topology_status": self.lab_topology_status,
            "lab_probe": self.lab_probe,
            "get_ospf_neighbors": self.get_ospf_neighbors,
        }
        if self.provider.manifest.links:
            values.update({
                "lab_get_topology_graph": self.lab_get_topology_graph,
                "lab_get_endpoint": self.lab_get_endpoint,
                "lab_trace_path": self.lab_trace_path,
            })
            if self.provider.manifest.users and self.provider.manifest.applications:
                values["lab_get_enforcement_path"] = self.lab_get_enforcement_path
        if self.provider.manifest.fabric and profile_id == "dc":
            values.update({
                "lab_get_fabric_state": self.lab_get_fabric_state,
                "lab_get_access_vlan": self.lab_get_access_vlan,
                "lab_get_vxlan_state": self.lab_get_vxlan_state,
                "lab_get_bgp_evpn_summary": self.lab_get_bgp_evpn_summary,
                "lab_get_evpn_routes": self.lab_get_evpn_routes,
                "fabric_set_access_vlan": self.fabric_set_access_vlan,
                "fabric_restore_access_vlan": self.fabric_restore_access_vlan,
            })
        if profile_id == "lan" and self.provider.manifest.users:
            values.update({
                "list_users": self.list_users,
                "get_user_access": self.get_user_access,
                "check_nac_policy": self.check_nac_policy,
                "grant_user_access": self.grant_user_access,
                "revoke_user_access": self.revoke_user_access,
            })
        elif profile_id == "dc" and self.provider.manifest.applications:
            values.update({
                "lab_app_probe": self.lab_app_probe,
                "dc_list_apps": self.dc_list_apps,
                "dc_get_app_acl": self.dc_get_app_acl,
                "dc_check_user_app_access": self.dc_check_user_app_access,
                "dc_grant_app_access": self.dc_grant_app_access,
                "dc_revoke_app_access": self.dc_revoke_app_access,
            })
        if self.provider.manifest.users and self.provider.manifest.applications:
            values.update({
                "lab_app_probe": self.lab_app_probe,
                "network_get_app_enforcement": self.network_get_app_enforcement,
                "network_apply_app_enforcement": self.network_apply_app_enforcement,
                "network_revoke_app_enforcement": self.network_revoke_app_enforcement,
                "network_restore_app_enforcement": self.network_restore_app_enforcement,
            })
        return values

    async def list_devices(self, args: dict[str, Any]) -> str:
        type_filter = str(args.get("type") or "").lower()
        tag_filter = str(args.get("tag") or "").lower()
        devices = [
            item for item in self.provider.inventory()
            if (not type_filter or type_filter in {str(item["platform"]), str(item["role"]).lower()})
            and (not tag_filter or tag_filter in item["tags"])
        ]
        return json.dumps({
            "lab": self.provider.manifest.name,
            "simulated": True,
            "devices": devices,
        }, ensure_ascii=False, sort_keys=True)

    async def get_device_status(self, args: dict[str, Any]) -> str:
        device_id = str(args["device_id"])
        version, interfaces, ospf = await asyncio.gather(
            self.provider.show(device_id, "show version"),
            self.provider.show(device_id, "show interface brief"),
            self.provider.show(device_id, "show ip ospf neighbor"),
        )
        return (
            f"Lab device status: {device_id}\n"
            f"[VERSION]\n{version}\n[INTERFACES]\n{interfaces}\n[OSPF]\n{ospf}"
        )

    async def get_device_config(self, args: dict[str, Any]) -> str:
        return await self.provider.running_config(
            str(args["device_id"]),
            str(args["section"]) if args.get("section") else None,
        )

    async def edit_device_config(self, args: dict[str, Any]) -> str:
        return await self.provider.apply_config(
            str(args["device_id"]),
            tuple(str(item) for item in args["config_lines"]),
        )

    async def restore_device_config(self, args: dict[str, Any]) -> str:
        return await self.provider.restore_last_config(str(args["device_id"]))

    async def validate_device_config(self, args: dict[str, Any]) -> str:
        device_id = str(args["device_id"])
        expected_neighbors = self.provider.manifest.devices[
            device_id
        ].expected_ospf_neighbors
        config, ospf, routes = await asyncio.gather(
            self.provider.running_config(device_id),
            self.provider.show(device_id, "show ip ospf neighbor"),
            self.provider.show(device_id, "show ip route"),
        )
        full_neighbors = ospf.lower().count("full")
        return json.dumps({
            "device_id": device_id,
            "config_readable": bool(config.strip()),
            "ospf_full_neighbors": full_neighbors,
            "routing_table_readable": "Codes:" in routes or "O>*" in routes or "C>*" in routes,
            "expected_ospf_neighbors": expected_neighbors,
            "passed": bool(config.strip()) and full_neighbors >= expected_neighbors,
        }, sort_keys=True)

    async def get_syslog(self, args: dict[str, Any]) -> str:
        return await self.provider.show(str(args["device_id"]), "show logging")

    async def query_interface_metrics(self, args: dict[str, Any]) -> str:
        interface = str(args.get("interface") or "").strip()
        command = "show interface" if not interface else f"show interface {interface}"
        return await self.provider.show(str(args["device_id"]), command)

    async def get_bgp_summary(self, args: dict[str, Any]) -> str:
        return await self.provider.show(str(args["device_id"]), "show bgp summary")

    async def get_device_facts(self, args: dict[str, Any]) -> str:
        return await self.provider.show(str(args["device_id"]), "show version")

    async def run_command(self, args: dict[str, Any]) -> str:
        return await self.provider.show(str(args["device_id"]), str(args["command"]))

    async def multi_device_check(self, args: dict[str, Any]) -> str:
        requested = args.get("device_ids")
        if requested in (None, "all"):
            device_ids = sorted(self.provider.manifest.devices)
        else:
            device_ids = [str(item) for item in requested]
        command = str(args.get("command") or "show version")
        values = await asyncio.gather(*(
            self.provider.show(device_id, command) for device_id in device_ids
        ))
        return json.dumps(dict(zip(device_ids, values, strict=True)), sort_keys=True)

    async def lab_topology_status(self, _args: dict[str, Any]) -> str:
        return json.dumps(await self.provider.topology_status(), sort_keys=True)

    async def lab_get_topology_graph(self, _args: dict[str, Any]) -> str:
        return json.dumps(
            self.provider.topology_graph(), ensure_ascii=False, sort_keys=True,
        )

    async def lab_get_endpoint(self, args: dict[str, Any]) -> str:
        return json.dumps(
            self.provider.endpoint_detail(str(args["endpoint_id"])),
            ensure_ascii=False,
            sort_keys=True,
        )

    async def lab_trace_path(self, args: dict[str, Any]) -> str:
        return json.dumps(await self.provider.trace_path(
            str(args["source_endpoint"]), str(args["destination_endpoint"]),
        ), ensure_ascii=False, sort_keys=True)

    async def lab_get_enforcement_path(self, args: dict[str, Any]) -> str:
        return json.dumps(await self.provider.enforcement_path(
            str(args["user_id"]), str(args["app_id"]),
        ), ensure_ascii=False, sort_keys=True)

    async def lab_get_fabric_state(self, _args: dict[str, Any]) -> str:
        return json.dumps(
            await self.provider.fabric_state(), ensure_ascii=False, sort_keys=True,
        )

    async def lab_get_access_vlan(self, args: dict[str, Any]) -> str:
        return json.dumps(await self.provider.fabric_access_vlan(
            str(args["device_id"]), str(args["interface"]),
        ), ensure_ascii=False, sort_keys=True)

    async def lab_get_vxlan_state(self, args: dict[str, Any]) -> str:
        return json.dumps(await self.provider.fabric_vxlan_state(
            str(args["device_id"]),
        ), ensure_ascii=False, sort_keys=True)

    async def lab_get_bgp_evpn_summary(self, args: dict[str, Any]) -> str:
        return json.dumps(await self.provider.fabric_bgp_evpn_summary(
            str(args["device_id"]),
        ), ensure_ascii=False, sort_keys=True)

    async def lab_get_evpn_routes(self, args: dict[str, Any]) -> str:
        route_type = args.get("route_type")
        return json.dumps(await self.provider.fabric_evpn_routes(
            str(args["device_id"]),
            int(route_type) if route_type is not None else None,
        ), ensure_ascii=False, sort_keys=True)

    async def fabric_set_access_vlan(self, args: dict[str, Any]) -> str:
        return await self.provider.set_fabric_access_vlan(
            str(args["device_id"]), str(args["interface"]), int(args["vlan_id"]),
        )

    async def fabric_restore_access_vlan(self, args: dict[str, Any]) -> str:
        return await self.provider.restore_fabric_access_vlan(
            str(args["device_id"]), str(args["interface"]),
        )

    async def lab_probe(self, args: dict[str, Any]) -> str:
        return json.dumps(await self.provider.probe(str(args["probe_id"])), sort_keys=True)

    async def get_ospf_neighbors(self, args: dict[str, Any]) -> str:
        return await self.provider.show(str(args["device_id"]), "show ip ospf neighbor")

    async def lab_app_probe(self, args: dict[str, Any]) -> str:
        return json.dumps(await self.provider.application_probe(
            str(args["user_id"]), str(args["app_id"]),
        ), ensure_ascii=False, sort_keys=True)

    async def network_get_app_enforcement(self, args: dict[str, Any]) -> str:
        user_id = str(args["user_id"]).strip().lower()
        app_id = str(args["app_id"]).strip().lower()
        user = self.provider._user(user_id)
        app = self.provider._application(app_id)
        blocked = await self.provider.application_access_blocked(user_id, app_id)
        return json.dumps({
            "ok": True,
            "user_id": user_id,
            "app_id": app_id,
            "allowed": not blocked,
            "source_endpoint": user.endpoint,
            "application_endpoint": app.endpoint,
            "implementation": "server-source-blackhole-route",
            "simulation": True,
        }, ensure_ascii=False, sort_keys=True)

    async def network_apply_app_enforcement(self, args: dict[str, Any]) -> str:
        user_id = str(args["user_id"]).strip().lower()
        app_id = str(args["app_id"]).strip().lower()
        await self.provider.set_application_access(user_id, app_id, allowed=True)
        return await self.network_get_app_enforcement(args)

    async def network_revoke_app_enforcement(self, args: dict[str, Any]) -> str:
        user_id = str(args["user_id"]).strip().lower()
        app_id = str(args["app_id"]).strip().lower()
        await self.provider.set_application_access(user_id, app_id, allowed=False)
        return await self.network_get_app_enforcement(args)

    async def network_restore_app_enforcement(self, args: dict[str, Any]) -> str:
        user_id = str(args["user_id"]).strip().lower()
        app_id = str(args["app_id"]).strip().lower()
        allowed = args.get("allowed")
        if not isinstance(allowed, bool):
            raise ValueError("network enforcement restore requires boolean allowed")
        await self.provider.set_application_access(user_id, app_id, allowed=allowed)
        return await self.network_get_app_enforcement(args)

    async def list_users(self, args: dict[str, Any]) -> str:
        department = str(args.get("dept") or "").strip().lower()
        users = [
            user for user in self.provider.manifest.users.values()
            if not department or user.department.lower() == department
        ]
        if not users:
            return f"No user in dept={department!r}."
        lines = ["Enterprise lab users:", "", f"  {'ID':<10}{'NAME':<18}{'DEPT':<12}STATUS"]
        lines.extend(
            f"  {user.user_id:<10}{user.name:<18}{user.department:<12}{user.status}"
            for user in sorted(users, key=lambda item: item.user_id)
        )
        return "\n".join(lines)

    async def get_user_access(self, args: dict[str, Any]) -> str:
        user_id = str(args["user_id"]).strip().lower()
        user = self.provider._user(user_id)
        admitted = await self.provider.user_admitted(user_id)
        lines = [f"Network access for user '{user_id}' ({user.name}):", ""]
        lines.append(f"  RADIUS auth : {'pass' if admitted else 'fail'} (lab identity)")
        lines.append(f"  802.1X      : {'authorized' if admitted else 'rejected'} (endpoint enforcement)")
        lines.append(f"  NAC posture : {'compliant' if admitted else 'quarantine'} (lab policy)")
        lines.append(f"  VLAN        : {user.vlan if admitted else '(none — not admitted)'}")
        lines.append(f"  endpoint    : {user.endpoint}/{user.interface}")
        lines.append("")
        lines.append(f"  network admission: {'✅ ADMITTED' if admitted else '❌ BLOCKED at network layer'}")
        return "\n".join(lines)

    async def check_nac_policy(self, args: dict[str, Any]) -> str:
        user_id = str(args["user_id"]).strip().lower()
        user = self.provider._user(user_id)
        admitted = await self.provider.user_admitted(user_id)
        lines = [f"NAC lab policy evaluation for '{user_id}':", ""]
        lines.append(
            "  matched policy : "
            + ("CORP-COMPLIANT-ENDPOINT" if admitted else "QUARANTINE-ONBOARDING")
        )
        lines.append(f"  identity status: {user.status}")
        lines.append(f"  enforcement    : {user.endpoint}/{user.interface}")
        lines.append(f"  authorization  : {'VLAN ' + str(user.vlan) if admitted else 'quarantine'}")
        lines.append(f"  result         : {'PERMIT' if admitted else 'DENY'}")
        return "\n".join(lines)

    async def grant_user_access(self, args: dict[str, Any]) -> str:
        user_id = str(args["user_id"]).strip().lower()
        reason = str(args.get("reason") or "").strip()
        user = self.provider._user(user_id)
        detail = await self.provider.set_user_admission(user_id, admitted=True)
        return "\n".join([
            "Granted network access:", "", f"  user   : {user_id}",
            f"  changes: endpoint admitted, VLAN={user.vlan}", f"  reason : {reason}",
            f"  target : {detail}", "  status : applied and read back",
        ])

    async def revoke_user_access(self, args: dict[str, Any]) -> str:
        user_id = str(args["user_id"]).strip().lower()
        reason = str(args.get("reason") or "").strip()
        detail = await self.provider.set_user_admission(user_id, admitted=False)
        return "\n".join([
            "Revoked network access:", "", f"  user   : {user_id}",
            "  changes: endpoint quarantined, VLAN=none", f"  reason : {reason}",
            f"  target : {detail}", "  status : applied and read back",
        ])

    async def dc_list_apps(self, args: dict[str, Any]) -> str:
        tier = str(args.get("tier") or "").strip().lower()
        apps = [
            app for app in self.provider.manifest.applications.values()
            if not tier or app.tier.lower() == tier
        ]
        if not apps:
            return f"No application matches tier={tier!r}."
        lines = ["Data-center lab applications:", ""]
        lines.append(f"  {'ID':<10}{'NAME':<22}{'VIP':<16}{'OWNER':<16}TIER")
        lines.extend(
            f"  {app.app_id:<10}{app.name:<22}{app.address:<16}{app.owner:<16}{app.tier}"
            for app in sorted(apps, key=lambda item: item.app_id)
        )
        return "\n".join(lines)

    async def dc_get_app_acl(self, args: dict[str, Any]) -> str:
        app_id = str(args["app_id"]).strip().lower()
        app = self.provider._application(app_id)
        allowed_users = []
        for user_id in sorted(self.provider.manifest.users):
            if not await self.provider.application_access_blocked(user_id, app_id):
                allowed_users.append(user_id)
        members = ", ".join(allowed_users) if allowed_users else "(none)"
        lines = [f"Access control for application '{app_id}':", ""]
        lines.extend(f"  role {role:<12}: {members}" for role in app.roles)
        lines.append(f"  enforcement : exact source /32 policy on {app.endpoint}")
        return "\n".join(lines)

    async def dc_check_user_app_access(self, args: dict[str, Any]) -> str:
        user_id = str(args["user_id"]).strip().lower()
        app_id = str(args["app_id"]).strip().lower()
        app = self.provider._application(app_id)
        probe = await self.provider.application_probe(user_id, app_id)
        allowed = bool(probe["ok"])
        lines = [
            f"Application access check — user '{user_id}' → app '{app_id}' ({app.name}):", "",
            f"  VIP        : {app.address}:{app.port}",
            f"  owner team : {app.owner}",
            f"  result     : {'✅ ALLOWED' if allowed else '❌ DENIED'}",
        ]
        if allowed:
            lines.append(f"  via roles  : {app.roles[0]}")
            lines.append("  proof      : manifest-bound HTTP request succeeded")
        else:
            lines.append("  reason     : application policy or real HTTP probe denied the source")
        return "\n".join(lines)

    async def dc_grant_app_access(self, args: dict[str, Any]) -> str:
        user_id = str(args["user_id"]).strip().lower()
        app_id = str(args["app_id"]).strip().lower()
        app = self.provider._application(app_id)
        role = str(args.get("role") or app.roles[0]).strip()
        if role not in app.roles:
            raise ValueError(f"role {role!r} is not reviewed for application {app_id!r}")
        reason = str(args.get("reason") or "").strip()
        detail = await self.provider.set_application_access(user_id, app_id, allowed=True)
        return "\n".join([
            "Granted application access:", "", f"  user   : {user_id}",
            f"  app    : {app_id} ({app.name})", f"  role   : {role}",
            f"  reason : {reason}", f"  target : {detail}",
            "  status : applied and read back",
        ])

    async def dc_revoke_app_access(self, args: dict[str, Any]) -> str:
        user_id = str(args["user_id"]).strip().lower()
        app_id = str(args["app_id"]).strip().lower()
        app = self.provider._application(app_id)
        reason = str(args.get("reason") or "").strip()
        detail = await self.provider.set_application_access(user_id, app_id, allowed=False)
        return "\n".join([
            "Revoked application access:", "", f"  user   : {user_id}",
            f"  app    : {app_id}", f"  roles removed: {', '.join(app.roles)}",
            f"  reason : {reason}", f"  target : {detail}",
            "  status : applied and read back",
        ])
