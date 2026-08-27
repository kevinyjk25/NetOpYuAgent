"""Strict, versioned manifest for a local network laboratory.

The manifest is operator-owned input.  It is validated before any subprocess
is started so an LLM cannot invent container names, interfaces, probes or
topology paths at execution time.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ManifestError(ValueError):
    """The lab manifest is incomplete, ambiguous or unsafe."""


_ID = re.compile(r"^[a-z][a-z0-9-]{0,62}$")
_INTERFACE = re.compile(r"^eth[1-9][0-9]*$")
_HTTP_PATH = re.compile(r"^/[a-zA-Z0-9._/-]{0,255}$")


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{field} must be an object")
    return {str(key): item for key, item in value.items()}


def _identifier(value: Any, field: str) -> str:
    result = str(value or "").strip()
    if not _ID.fullmatch(result):
        raise ManifestError(f"{field} must match {_ID.pattern}")
    return result


def _validate_topology_links(topology_path: Path, links: dict[str, "LabLink"]) -> None:
    """Prove the typed graph is an exact projection of Containerlab wiring."""
    try:
        raw = yaml.safe_load(topology_path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ManifestError(f"cannot validate topology links: {error}") from error
    topology = _mapping(_mapping(raw, "topology file").get("topology"), "topology")
    raw_links = topology.get("links") or []
    if not isinstance(raw_links, list):
        raise ManifestError("topology.links must be an array")

    wired: set[frozenset[str]] = set()
    for index, value in enumerate(raw_links):
        item = _mapping(value, f"topology.links[{index}]")
        endpoints = item.get("endpoints")
        if (
            not isinstance(endpoints, list) or len(endpoints) != 2
            or any(not isinstance(endpoint, str) for endpoint in endpoints)
        ):
            raise ManifestError(f"topology.links[{index}].endpoints must contain two strings")
        pair = frozenset(endpoint.strip() for endpoint in endpoints)
        if len(pair) != 2 or pair in wired:
            raise ManifestError(f"topology.links[{index}] is duplicate or self-referential")
        wired.add(pair)

    declared = {
        frozenset(f"{member.node}:{member.interface}" for member in link.endpoints)
        for link in links.values()
    }
    if wired != declared:
        missing = sorted(",".join(sorted(pair)) for pair in wired - declared)
        extra = sorted(",".join(sorted(pair)) for pair in declared - wired)
        raise ManifestError(
            "typed links do not exactly match topology wiring; "
            f"missing={missing}, extra={extra}"
        )


@dataclass(frozen=True)
class LabDevice:
    device_id: str
    node: str
    platform: str
    role: str
    zone: str
    tags: tuple[str, ...]
    expected_ospf_neighbors: int
    expected_bgp_neighbors: int


@dataclass(frozen=True)
class LabEndpoint:
    endpoint_id: str
    node: str
    role: str
    zone: str


@dataclass(frozen=True)
class LabZone:
    zone_id: str
    trust: str
    description: str


@dataclass(frozen=True)
class LabLinkMember:
    node: str
    interface: str
    address: str

    @property
    def ip(self) -> str:
        return str(ipaddress.ip_interface(self.address).ip)


@dataclass(frozen=True)
class LabLink:
    link_id: str
    endpoints: tuple[LabLinkMember, LabLinkMember]
    relationship: str
    path_role: str


@dataclass(frozen=True)
class LabSimulationModel:
    forwarding: str
    network_admission: str
    application_policy: str
    security_edge: str


@dataclass(frozen=True)
class LabProbe:
    probe_id: str
    source: str
    destination: str
    count: int
    expected: bool


@dataclass(frozen=True)
class LabFaultTarget:
    fault_id: str
    node: str
    interface: str


@dataclass(frozen=True)
class LabUser:
    user_id: str
    name: str
    department: str
    status: str
    endpoint: str
    interface: str
    address: str
    prefix_length: int
    gateway: str
    application_prefix: str
    route_prefixes: tuple[str, ...]
    vlan: int


@dataclass(frozen=True)
class LabApplication:
    app_id: str
    name: str
    endpoint: str
    address: str
    port: int
    path: str
    owner: str
    tier: str
    roles: tuple[str, ...]


@dataclass(frozen=True)
class LabManifest:
    schema_version: int
    name: str
    source_path: Path
    topology_path: Path
    state_directory: Path
    zones: dict[str, LabZone]
    devices: dict[str, LabDevice]
    endpoints: dict[str, LabEndpoint]
    links: dict[str, LabLink]
    simulation: LabSimulationModel
    probes: dict[str, LabProbe]
    fault_targets: dict[str, LabFaultTarget]
    users: dict[str, LabUser]
    applications: dict[str, LabApplication]

    def container_name(self, node: str) -> str:
        return f"clab-{self.name}-{node}"

    @property
    def nodes(self) -> tuple[str, ...]:
        return tuple(sorted({
            *(item.node for item in self.devices.values()),
            *(item.node for item in self.endpoints.values()),
        }))

    @property
    def address_index(self) -> dict[str, tuple[str, str, str]]:
        """Map a declared interface IP to ``(node, interface, link_id)``."""
        return {
            member.ip: (member.node, member.interface, link.link_id)
            for link in self.links.values()
            for member in link.endpoints
        }

    @property
    def node_index(self) -> dict[str, tuple[str, str, str]]:
        """Map node name to ``(kind, role, zone)`` without model inference."""
        values = {
            item.node: ("device", item.role, item.zone)
            for item in self.devices.values()
        }
        values.update({
            item.node: ("endpoint", item.role, item.zone)
            for item in self.endpoints.values()
        })
        return values


def load_manifest(path: str | Path) -> LabManifest:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ManifestError(f"lab manifest does not exist: {source}")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ManifestError(f"cannot read lab manifest: {error}") from error
    raw = _mapping(raw, "manifest")
    version = raw.get("schema_version")
    if version != 1:
        raise ManifestError("lab manifest schema_version must be 1")
    name = _identifier(raw.get("name"), "name")
    topology_value = str(raw.get("topology") or "").strip()
    if not topology_value:
        raise ManifestError("topology is required")
    topology = (source.parent / topology_value).resolve()
    if source.parent not in topology.parents or not topology.is_file():
        raise ManifestError("topology must be an existing file inside the manifest directory")
    state_directory = (source.parent / ".state").resolve()

    zones: dict[str, LabZone] = {}
    for zone_id, value in _mapping(raw.get("zones") or {}, "zones").items():
        zone_id = _identifier(zone_id, "zones key")
        item = _mapping(value, f"zones.{zone_id}")
        trust = str(item.get("trust") or "unspecified").strip().lower()
        if trust not in {"trusted", "restricted", "untrusted", "external", "unspecified"}:
            raise ManifestError(f"zones.{zone_id}.trust is invalid")
        zones[zone_id] = LabZone(
            zone_id=zone_id,
            trust=trust,
            description=str(item.get("description") or zone_id).strip(),
        )

    def declared_zone(value: Any, field: str) -> str:
        zone = str(value or "unspecified").strip().lower()
        if zone != "unspecified" and zone not in zones:
            raise ManifestError(f"{field} is not a declared zone")
        return zone

    devices: dict[str, LabDevice] = {}
    for device_id, value in _mapping(raw.get("devices"), "devices").items():
        device_id = _identifier(device_id, "devices key")
        item = _mapping(value, f"devices.{device_id}")
        platform = str(item.get("platform") or "").strip().lower()
        if platform != "frr":
            raise ManifestError(f"devices.{device_id}.platform must be frr")
        tags = item.get("tags") or []
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise ManifestError(f"devices.{device_id}.tags must be a string array")
        expected_neighbors = item.get("expected_ospf_neighbors", 1)
        if (
            not isinstance(expected_neighbors, int) or isinstance(expected_neighbors, bool)
            or not 0 <= expected_neighbors <= 16
        ):
            raise ManifestError(
                f"devices.{device_id}.expected_ospf_neighbors must be between 0 and 16"
            )
        expected_bgp_neighbors = item.get("expected_bgp_neighbors", 0)
        if (
            not isinstance(expected_bgp_neighbors, int)
            or isinstance(expected_bgp_neighbors, bool)
            or not 0 <= expected_bgp_neighbors <= 32
        ):
            raise ManifestError(
                f"devices.{device_id}.expected_bgp_neighbors must be between 0 and 32"
            )
        devices[device_id] = LabDevice(
            device_id=device_id,
            node=_identifier(item.get("node"), f"devices.{device_id}.node"),
            platform=platform,
            role=str(item.get("role") or "router").strip(),
            zone=declared_zone(item.get("zone"), f"devices.{device_id}.zone"),
            tags=tuple(sorted(set(tag.strip() for tag in tags if tag.strip()))),
            expected_ospf_neighbors=expected_neighbors,
            expected_bgp_neighbors=expected_bgp_neighbors,
        )
    if not devices:
        raise ManifestError("at least one lab device is required")

    endpoints: dict[str, LabEndpoint] = {}
    for endpoint_id, value in _mapping(raw.get("endpoints"), "endpoints").items():
        endpoint_id = _identifier(endpoint_id, "endpoints key")
        item = _mapping(value, f"endpoints.{endpoint_id}")
        endpoints[endpoint_id] = LabEndpoint(
            endpoint_id=endpoint_id,
            node=_identifier(item.get("node"), f"endpoints.{endpoint_id}.node"),
            role=str(item.get("role") or "endpoint").strip(),
            zone=declared_zone(item.get("zone"), f"endpoints.{endpoint_id}.zone"),
        )

    all_node_ids = {
        *(item.node for item in devices.values()),
        *(item.node for item in endpoints.values()),
    }
    if len(all_node_ids) != len(devices) + len(endpoints):
        raise ManifestError("device and endpoint node names must be unique")

    links: dict[str, LabLink] = {}
    used_interfaces: set[tuple[str, str]] = set()
    used_addresses: set[str] = set()
    for link_id, value in _mapping(raw.get("links") or {}, "links").items():
        link_id = _identifier(link_id, "links key")
        item = _mapping(value, f"links.{link_id}")
        members_raw = item.get("endpoints")
        if not isinstance(members_raw, list) or len(members_raw) != 2:
            raise ManifestError(f"links.{link_id}.endpoints must contain exactly two members")
        members: list[LabLinkMember] = []
        networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
        for index, raw_member in enumerate(members_raw):
            member = _mapping(raw_member, f"links.{link_id}.endpoints[{index}]")
            node = _identifier(member.get("node"), f"links.{link_id}.endpoints[{index}].node")
            if node not in all_node_ids:
                raise ManifestError(f"links.{link_id} references undeclared node {node!r}")
            interface = str(member.get("interface") or "").strip()
            if not _INTERFACE.fullmatch(interface):
                raise ManifestError(
                    f"links.{link_id}.endpoints[{index}].interface must be eth1 or higher"
                )
            try:
                parsed = ipaddress.ip_interface(str(member.get("address") or ""))
            except ValueError as error:
                raise ManifestError(
                    f"links.{link_id}.endpoints[{index}].address must be an IP interface"
                ) from error
            key = (node, interface)
            address = str(parsed)
            if key in used_interfaces:
                raise ManifestError(f"interface {node}:{interface} is declared by multiple links")
            if str(parsed.ip) in used_addresses:
                raise ManifestError(f"interface address {parsed.ip} is declared more than once")
            used_interfaces.add(key)
            used_addresses.add(str(parsed.ip))
            networks.append(parsed.network)
            members.append(LabLinkMember(node=node, interface=interface, address=address))
        if networks[0] != networks[1]:
            raise ManifestError(f"links.{link_id} endpoint addresses are not in the same subnet")
        relationship = str(item.get("relationship") or "routed-link").strip().lower()
        if relationship not in {
            "access", "uplink", "core-peer", "service", "wan", "internet-segment",
        }:
            raise ManifestError(f"links.{link_id}.relationship is invalid")
        path_role = str(item.get("path_role") or "equal").strip().lower()
        if path_role not in {"primary", "backup", "equal", "peer", "edge"}:
            raise ManifestError(f"links.{link_id}.path_role is invalid")
        links[link_id] = LabLink(
            link_id=link_id,
            endpoints=(members[0], members[1]),
            relationship=relationship,
            path_role=path_role,
        )

    simulation_raw = _mapping(raw.get("simulation") or {}, "simulation")
    simulation = LabSimulationModel(
        forwarding=str(simulation_raw.get("forwarding") or "unspecified").strip(),
        network_admission=str(
            simulation_raw.get("network_admission") or "unspecified"
        ).strip(),
        application_policy=str(
            simulation_raw.get("application_policy") or "unspecified"
        ).strip(),
        security_edge=str(simulation_raw.get("security_edge") or "unspecified").strip(),
    )
    if links:
        allowed_simulation = {
            "forwarding": {"linux-l3-frr", "unspecified"},
            "network_admission": {"endpoint-interface-state", "unspecified"},
            "application_policy": {"server-source-blackhole-route", "unspecified"},
            "security_edge": {"routed-wan-edge-no-stateful-firewall", "unspecified"},
        }
        for field, allowed in allowed_simulation.items():
            if getattr(simulation, field) not in allowed:
                raise ManifestError(f"simulation.{field} is invalid")
        _validate_topology_links(topology, links)

    probes: dict[str, LabProbe] = {}
    for probe_id, value in _mapping(raw.get("probes"), "probes").items():
        probe_id = _identifier(probe_id, "probes key")
        item = _mapping(value, f"probes.{probe_id}")
        source_id = _identifier(item.get("source"), f"probes.{probe_id}.source")
        if source_id not in endpoints:
            raise ManifestError(f"probes.{probe_id}.source is not a declared endpoint")
        destination = str(item.get("destination") or "").strip()
        try:
            destination = str(ipaddress.ip_address(destination))
        except ValueError as error:
            raise ManifestError(f"probes.{probe_id}.destination must be an IP address") from error
        count = item.get("count", 3)
        if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 10:
            raise ManifestError(f"probes.{probe_id}.count must be between 1 and 10")
        expected = item.get("expected", True)
        if not isinstance(expected, bool):
            raise ManifestError(f"probes.{probe_id}.expected must be boolean")
        probes[probe_id] = LabProbe(probe_id, source_id, destination, count, expected)

    faults: dict[str, LabFaultTarget] = {}
    for fault_id, value in _mapping(raw.get("fault_targets"), "fault_targets").items():
        fault_id = _identifier(fault_id, "fault_targets key")
        item = _mapping(value, f"fault_targets.{fault_id}")
        node = _identifier(item.get("node"), f"fault_targets.{fault_id}.node")
        if node not in all_node_ids:
            raise ManifestError(f"fault_targets.{fault_id}.node is not declared")
        interface = str(item.get("interface") or "").strip()
        if not _INTERFACE.fullmatch(interface):
            raise ManifestError(f"fault_targets.{fault_id}.interface must be eth1 or higher")
        faults[fault_id] = LabFaultTarget(fault_id, node, interface)

    access = _mapping(raw.get("access") or {}, "access")
    users: dict[str, LabUser] = {}
    for user_id, value in _mapping(access.get("users") or {}, "access.users").items():
        user_id = _identifier(user_id, "access.users key")
        item = _mapping(value, f"access.users.{user_id}")
        endpoint = _identifier(item.get("endpoint"), f"access.users.{user_id}.endpoint")
        if endpoint not in endpoints:
            raise ManifestError(f"access.users.{user_id}.endpoint is not declared")
        interface = str(item.get("interface") or "").strip()
        if not _INTERFACE.fullmatch(interface):
            raise ManifestError(f"access.users.{user_id}.interface must be eth1 or higher")
        try:
            address = str(ipaddress.ip_address(str(item.get("address") or "")))
        except ValueError as error:
            raise ManifestError(f"access.users.{user_id}.address must be an IP address") from error
        prefix_length = item.get("prefix_length", 24)
        if not isinstance(prefix_length, int) or isinstance(prefix_length, bool):
            raise ManifestError(f"access.users.{user_id}.prefix_length must be an integer")
        try:
            user_network = ipaddress.ip_network(f"{address}/{prefix_length}", strict=False)
            gateway = str(ipaddress.ip_address(str(item.get("gateway") or "")))
            application_prefix = str(ipaddress.ip_network(
                str(item.get("application_prefix") or ""), strict=False,
            ))
        except ValueError as error:
            raise ManifestError(
                f"access.users.{user_id} route fields must be valid IP values"
            ) from error
        if ipaddress.ip_address(gateway) not in user_network:
            raise ManifestError(f"access.users.{user_id}.gateway must be in the user subnet")
        route_values = item.get("route_prefixes") or [application_prefix]
        if (
            not isinstance(route_values, list) or not route_values
            or len(route_values) > 16
        ):
            raise ManifestError(
                f"access.users.{user_id}.route_prefixes must contain between 1 and 16 prefixes"
            )
        try:
            route_prefixes = tuple(dict.fromkeys(
                str(ipaddress.ip_network(str(prefix), strict=False))
                for prefix in route_values
            ))
        except ValueError as error:
            raise ManifestError(
                f"access.users.{user_id}.route_prefixes must contain valid networks"
            ) from error
        if application_prefix not in route_prefixes:
            raise ManifestError(
                f"access.users.{user_id}.route_prefixes must include application_prefix"
            )
        vlan = item.get("vlan")
        if not isinstance(vlan, int) or isinstance(vlan, bool) or not 1 <= vlan <= 4094:
            raise ManifestError(f"access.users.{user_id}.vlan must be between 1 and 4094")
        status = str(item.get("status") or "active").strip().lower()
        if status not in {"active", "disabled", "suspended"}:
            raise ManifestError(f"access.users.{user_id}.status is invalid")
        users[user_id] = LabUser(
            user_id=user_id,
            name=str(item.get("name") or user_id).strip(),
            department=str(item.get("department") or "unknown").strip(),
            status=status,
            endpoint=endpoint,
            interface=interface,
            address=address,
            prefix_length=prefix_length,
            gateway=gateway,
            application_prefix=application_prefix,
            route_prefixes=route_prefixes,
            vlan=vlan,
        )

    applications: dict[str, LabApplication] = {}
    for app_id, value in _mapping(access.get("applications") or {}, "access.applications").items():
        app_id = _identifier(app_id, "access.applications key")
        item = _mapping(value, f"access.applications.{app_id}")
        endpoint = _identifier(item.get("endpoint"), f"access.applications.{app_id}.endpoint")
        if endpoint not in endpoints:
            raise ManifestError(f"access.applications.{app_id}.endpoint is not declared")
        try:
            address = str(ipaddress.ip_address(str(item.get("address") or "")))
        except ValueError as error:
            raise ManifestError(
                f"access.applications.{app_id}.address must be an IP address"
            ) from error
        port = item.get("port")
        if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
            raise ManifestError(f"access.applications.{app_id}.port must be between 1 and 65535")
        path_value = str(item.get("path") or "/").strip()
        if not _HTTP_PATH.fullmatch(path_value) or ".." in path_value:
            raise ManifestError(f"access.applications.{app_id}.path is unsafe")
        roles_raw = item.get("roles") or []
        if not isinstance(roles_raw, list) or not roles_raw:
            raise ManifestError(f"access.applications.{app_id}.roles must be a non-empty array")
        roles = tuple(_identifier(role, f"access.applications.{app_id}.roles") for role in roles_raw)
        applications[app_id] = LabApplication(
            app_id=app_id,
            name=str(item.get("name") or app_id).strip(),
            endpoint=endpoint,
            address=address,
            port=port,
            path=path_value,
            owner=str(item.get("owner") or "unknown").strip(),
            tier=str(item.get("tier") or "standard").strip(),
            roles=roles,
        )

    if bool(users) != bool(applications):
        raise ManifestError("access.users and access.applications must be declared together")

    return LabManifest(
        schema_version=version,
        name=name,
        source_path=source,
        topology_path=topology,
        state_directory=state_directory,
        zones=zones,
        devices=devices,
        endpoints=endpoints,
        links=links,
        simulation=simulation,
        probes=probes,
        fault_targets=faults,
        users=users,
        applications=applications,
    )
