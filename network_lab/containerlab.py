"""Containerlab implementation of the reviewed local laboratory contract."""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import os
import platform
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, Sequence

from .manifest import LabApplication, LabDevice, LabEndpoint, LabManifest, LabUser


class LabCommandError(RuntimeError):
    """A bounded lab command failed or produced an unsafe result."""


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class CommandRunner(Protocol):
    async def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        timeout: float = 30.0,
    ) -> CommandResult: ...


class LocalCommandRunner:
    """Run exact argv vectors without a shell or command interpolation."""

    async def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        timeout: float = 30.0,
    ) -> CommandResult:
        values = tuple(str(item) for item in argv)
        try:
            process = await asyncio.create_subprocess_exec(
                *values,
                cwd=str(cwd) if cwd else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=timeout,
            )
        except asyncio.TimeoutError as error:
            process.kill()
            await process.communicate()
            raise LabCommandError(f"lab command timed out after {timeout:.1f}s: {values[0]}") from error
        return CommandResult(
            argv=values,
            returncode=int(process.returncode or 0),
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
        )


_SHOW_COMMAND = re.compile(r"\Ashow(?:[ \t]+[a-zA-Z0-9_.:/-]+)*[ \t]*\Z")
_CONFIG_COMMAND = re.compile(
    r"^(?:"
    r"interface\s+eth[1-9][0-9]*|"
    r"description\s+[a-zA-Z0-9_.:/ -]{1,120}|"
    r"no\s+description|"
    r"ip\s+address\s+[0-9./]+|"
    r"ip\s+ospf\s+(?:area\s+[0-9.]+|cost\s+[1-9][0-9]{0,4}|"
    r"hello-interval\s+[1-9][0-9]*|dead-interval\s+[1-9][0-9]*)|"
    r"no\s+ip\s+ospf\s+cost|"
    r"router\s+ospf|ospf\s+router-id\s+[0-9.]+|"
    r"passive-interface\s+eth[1-9][0-9]*|"
    r"network\s+[0-9./]+\s+area\s+[0-9.]+|"
    r"ip\s+route\s+[0-9./]+\s+[0-9.]+|"
    r"no\s+ip\s+route\s+[0-9./]+\s+[0-9.]+|"
    r"ip\s+prefix-list\s+[a-zA-Z0-9_.-]+(?:\s+seq\s+[0-9]+)?\s+"
    r"(?:permit|deny)\s+[0-9./]+(?:\s+(?:ge|le)\s+[0-9]+)*|"
    r"no\s+ip\s+prefix-list\s+[a-zA-Z0-9_.-]+|"
    r"route-map\s+[a-zA-Z0-9_.-]+\s+(?:permit|deny)\s+[0-9]+|"
    r"match\s+ip\s+address\s+prefix-list\s+[a-zA-Z0-9_.-]+|"
    r"set\s+metric\s+[0-9]+|shutdown"
    r"|no\s+shutdown|exit"
    r")$",
    re.IGNORECASE,
)
_CONTAINERLAB_DOOD_IMAGE = "ghcr.io/srl-labs/containerlab/devcontainer-dood-slim:0.77.0"


def _bounded(value: str, limit: int = 16_384) -> str:
    return value if len(value) <= limit else value[:limit] + "\n...[truncated]"


def normalize_frr_config(value: str) -> str:
    """Remove volatile headers while preserving configuration semantics/order."""
    ignored = (
        "Building configuration...",
        "Current configuration:",
        "Configuration saved to",
    )
    lines = []
    for raw in value.replace("\r\n", "\n").splitlines():
        line = raw.rstrip()
        if not line or any(line.startswith(prefix) for prefix in ignored):
            continue
        lines.append(line)
    return "\n".join(lines).strip() + "\n"


class ContainerlabProvider:
    """Constrained provider; all mutable targets come from a reviewed manifest."""

    def __init__(
        self,
        manifest: LabManifest,
        *,
        runner: CommandRunner | None = None,
        command_timeout: float = 30.0,
    ) -> None:
        if not 1 <= command_timeout <= 300:
            raise ValueError("command_timeout must be between 1 and 300 seconds")
        self.manifest = manifest
        self.runner = runner or LocalCommandRunner()
        self.command_timeout = command_timeout
        self._snapshots: dict[str, str] = {}
        self._fabric_vlan_snapshots: dict[tuple[str, str], dict[str, object]] = {}

    def _device(self, device_id: str) -> LabDevice:
        try:
            return self.manifest.devices[device_id]
        except KeyError as error:
            known = ", ".join(sorted(self.manifest.devices))
            raise LabCommandError(f"unknown lab device {device_id!r}; known: {known}") from error

    def _user(self, user_id: str) -> LabUser:
        try:
            return self.manifest.users[user_id]
        except KeyError as error:
            known = ", ".join(sorted(self.manifest.users))
            raise LabCommandError(f"unknown lab user {user_id!r}; known: {known}") from error

    def _application(self, app_id: str) -> LabApplication:
        try:
            return self.manifest.applications[app_id]
        except KeyError as error:
            known = ", ".join(sorted(self.manifest.applications))
            raise LabCommandError(f"unknown lab application {app_id!r}; known: {known}") from error

    def _endpoint(self, endpoint_id: str) -> LabEndpoint:
        try:
            return self.manifest.endpoints[endpoint_id]
        except KeyError as error:
            known = ", ".join(sorted(self.manifest.endpoints))
            raise LabCommandError(
                f"unknown lab endpoint {endpoint_id!r}; known: {known}"
            ) from error

    async def _run(self, argv: Sequence[str], *, timeout: float | None = None) -> CommandResult:
        result = await self.runner.run(
            argv,
            cwd=self.manifest.source_path.parent,
            timeout=timeout or self.command_timeout,
        )
        return result

    @staticmethod
    def _docker_socket() -> Path | None:
        configured = os.environ.get("NETOPYU_DOCKER_SOCKET", "").strip()
        candidates = [
            Path(configured).expanduser() if configured else None,
            Path.home() / ".docker" / "run" / "docker.sock",
            Path("/var/run/docker.sock"),
        ]
        return next((path for path in candidates if path is not None and path.exists()), None)

    def _containerlab_argv(self, *arguments: str) -> tuple[str, ...]:
        native = shutil.which("containerlab")
        if native:
            return (native, *arguments)
        socket_path = self._docker_socket()
        if shutil.which("docker") is None or socket_path is None:
            raise LabCommandError("neither native containerlab nor Docker Dood is available")
        lab_directory = self.manifest.source_path.parent
        return (
            "docker", "run", "--rm", "--privileged", "--network", "host", "--pid", "host",
            "-v", f"{socket_path}:/var/run/docker.sock",
            "-v", f"{lab_directory}:{lab_directory}",
            "-w", str(lab_directory),
            _CONTAINERLAB_DOOD_IMAGE, "containerlab", *arguments,
        )

    @staticmethod
    def _require_ok(result: CommandResult, operation: str) -> str:
        if result.returncode != 0:
            detail = _bounded((result.stderr or result.stdout).strip(), 2_000)
            raise LabCommandError(f"{operation} failed with exit={result.returncode}: {detail}")
        return _bounded(result.stdout)

    async def preflight(self) -> dict[str, object]:
        checks: list[dict[str, object]] = []
        docker_path = shutil.which("docker")
        checks.append({"check": "binary:docker", "passed": docker_path is not None,
                       "detail": docker_path or "not found"})
        native_containerlab = shutil.which("containerlab")
        docker_ready = False
        if shutil.which("docker"):
            result = await self._run(("docker", "info", "--format", "{{.ServerVersion}}"))
            docker_ready = result.ok and bool(result.stdout.strip())
            checks.append({
                "check": "docker-engine",
                "passed": docker_ready,
                "detail": result.stdout.strip() if docker_ready else _bounded(result.stderr.strip(), 500),
            })
        dood_ready = False
        if not native_containerlab and docker_ready and self._docker_socket() is not None:
            image = await self._run((
                "docker", "image", "inspect", _CONTAINERLAB_DOOD_IMAGE,
                "--format", "{{.Id}}",
            ))
            dood_ready = image.ok and bool(image.stdout.strip())
        checks.append({
            "check": "containerlab-runtime",
            "passed": native_containerlab is not None or dood_ready,
            "detail": native_containerlab or (
                f"Docker Dood image {_CONTAINERLAB_DOOD_IMAGE}" if dood_ready
                else "native binary and pinned local Dood image are unavailable"
            ),
        })
        host_arch = platform.machine().lower()
        checks.append({
            "check": "host-architecture",
            "passed": host_arch in {"x86_64", "amd64", "arm64", "aarch64"},
            "detail": host_arch,
        })
        checks.append({
            "check": "topology-file",
            "passed": self.manifest.topology_path.is_file(),
            "detail": str(self.manifest.topology_path),
        })
        return {
            "ok": all(bool(item["passed"]) for item in checks),
            "provider": "containerlab",
            "lab": self.manifest.name,
            "checks": checks,
        }

    async def deploy(self, *, reconfigure: bool = False) -> str:
        preflight = await self.preflight()
        if not preflight["ok"]:
            failed = [str(item["check"]) for item in preflight["checks"] if not item["passed"]]
            raise LabCommandError("lab preflight failed: " + ", ".join(failed))
        argv = list(self._containerlab_argv(
            "deploy", "--topo", str(self.manifest.topology_path),
        ))
        if reconfigure:
            argv.append("--reconfigure")
        return self._require_ok(await self._run(argv, timeout=300), "containerlab deploy")

    async def destroy(self) -> str:
        argv = self._containerlab_argv(
            "destroy", "--topo", str(self.manifest.topology_path), "--cleanup",
        )
        return self._require_ok(await self._run(argv, timeout=300), "containerlab destroy")

    async def topology_status(self) -> dict[str, object]:
        nodes: list[dict[str, object]] = []
        for node in self.manifest.nodes:
            container = self.manifest.container_name(node)
            result = await self._run((
                "docker", "inspect", "--format", "{{.State.Status}}", container,
            ))
            state = result.stdout.strip() if result.ok else "missing"
            nodes.append({"node": node, "container": container, "state": state})
        return {
            "ok": bool(nodes) and all(item["state"] == "running" for item in nodes),
            "lab": self.manifest.name,
            "nodes": nodes,
        }

    def topology_graph(self) -> dict[str, object]:
        """Return the operator-reviewed graph without deriving links from config text."""
        if not self.manifest.links:
            raise LabCommandError("this lab manifest has no typed topology graph")
        node_index = self.manifest.node_index
        node_addresses: dict[str, list[dict[str, str]]] = {
            node: [] for node in self.manifest.nodes
        }
        for link in self.manifest.links.values():
            for member in link.endpoints:
                node_addresses[member.node].append({
                    "interface": member.interface,
                    "address": member.address,
                    "link_id": link.link_id,
                })
        nodes = []
        for node in self.manifest.nodes:
            kind, role, zone = node_index[node]
            nodes.append({
                "node": node,
                "kind": kind,
                "role": role,
                "zone": zone,
                "interfaces": sorted(
                    node_addresses[node], key=lambda item: str(item["interface"]),
                ),
            })
        links = [{
            "link_id": link.link_id,
            "relationship": link.relationship,
            "path_role": link.path_role,
            "endpoints": [asdict(member) for member in link.endpoints],
        } for link in sorted(self.manifest.links.values(), key=lambda item: item.link_id)]
        return {
            "ok": True,
            "lab": self.manifest.name,
            "source_of_truth": str(self.manifest.source_path),
            "graph_validation": "exact-match-with-containerlab-topology",
            "zones": [
                asdict(self.manifest.zones[key]) for key in sorted(self.manifest.zones)
            ],
            "nodes": nodes,
            "links": links,
            "simulation": asdict(self.manifest.simulation),
            "truth_boundary": {
                "layer2_vlan_switching": False,
                "radius_or_8021x": False,
                "stateful_firewall": False,
                "application_iam": False,
            },
        }

    def endpoint_detail(self, endpoint_id: str) -> dict[str, object]:
        """Resolve an endpoint separately from device inventory."""
        endpoint = self._endpoint(endpoint_id)
        links: list[dict[str, object]] = []
        for link in self.manifest.links.values():
            for member in link.endpoints:
                if member.node != endpoint.node:
                    continue
                peer = next(item for item in link.endpoints if item.node != endpoint.node)
                links.append({
                    "link_id": link.link_id,
                    "local_interface": member.interface,
                    "local_address": member.address,
                    "peer_node": peer.node,
                    "peer_interface": peer.interface,
                    "peer_address": peer.address,
                    "relationship": link.relationship,
                    "path_role": link.path_role,
                })
        users = sorted(
            user.user_id for user in self.manifest.users.values()
            if user.endpoint == endpoint_id
        )
        applications = sorted(
            app.app_id for app in self.manifest.applications.values()
            if app.endpoint == endpoint_id
        )
        return {
            "ok": True,
            "lab": self.manifest.name,
            "endpoint_id": endpoint.endpoint_id,
            "node": endpoint.node,
            "kind": "endpoint",
            "role": endpoint.role,
            "zone": endpoint.zone,
            "links": sorted(links, key=lambda item: str(item["link_id"])),
            "users": users,
            "applications": applications,
        }

    def _single_endpoint_address(self, endpoint_id: str) -> tuple[LabEndpoint, str]:
        endpoint = self._endpoint(endpoint_id)
        addresses = [
            member.ip
            for link in self.manifest.links.values()
            for member in link.endpoints
            if member.node == endpoint.node
        ]
        if len(addresses) != 1:
            raise LabCommandError(
                f"endpoint {endpoint_id!r} must have exactly one declared data-plane address"
            )
        return endpoint, addresses[0]

    async def trace_path(
        self, source_endpoint: str, destination_endpoint: str,
    ) -> dict[str, object]:
        """Run a bounded traceroute and resolve every hop against the typed graph."""
        if not self.manifest.links:
            raise LabCommandError("this lab manifest has no typed topology graph")
        source, source_address = self._single_endpoint_address(source_endpoint)
        destination, destination_address = self._single_endpoint_address(destination_endpoint)
        result = await self._run((
            "docker", "exec", self.manifest.container_name(source.node),
            "traceroute", "-n", "-m", "16", "-w", "2", "-q", "1",
            destination_address,
        ))
        output = _bounded((result.stdout + result.stderr).strip(), 8_000)
        address_index = self.manifest.address_index
        node_index = self.manifest.node_index
        hops: list[dict[str, object]] = []
        unresolved: list[dict[str, object]] = []
        previous_node = source.node
        node_path = [source.node]
        link_path: list[str] = []
        for raw in output.splitlines():
            match = re.match(r"^\s*(\d+)\s+(\*|[0-9a-fA-F:.]+)(?:\s|$)", raw)
            if match is None:
                continue
            hop_number = int(match.group(1))
            address = match.group(2)
            if address == "*":
                item = {"hop": hop_number, "address": None, "resolved": False}
                hops.append(item)
                unresolved.append({**item, "reason": "traceroute-timeout"})
                continue
            try:
                address = str(ipaddress.ip_address(address))
            except ValueError:
                item = {"hop": hop_number, "address": address, "resolved": False}
                hops.append(item)
                unresolved.append({**item, "reason": "invalid-hop-address"})
                continue
            resolved = address_index.get(address)
            if resolved is None:
                item = {"hop": hop_number, "address": address, "resolved": False}
                hops.append(item)
                unresolved.append({**item, "reason": "address-not-in-reviewed-manifest"})
                continue
            node, interface, link_id = resolved
            kind, role, zone = node_index[node]
            link = self.manifest.links[link_id]
            connected_nodes = {member.node for member in link.endpoints}
            adjacency_verified = previous_node in connected_nodes and node in connected_nodes
            item = {
                "hop": hop_number,
                "address": address,
                "resolved": True,
                "node": node,
                "kind": kind,
                "role": role,
                "zone": zone,
                "ingress_interface": interface,
                "link_id": link_id,
                "adjacency_verified": adjacency_verified,
            }
            hops.append(item)
            if not adjacency_verified:
                unresolved.append({
                    "hop": hop_number,
                    "address": address,
                    "reason": f"declared-link-does-not-connect-from-{previous_node}",
                })
            previous_node = node
            node_path.append(node)
            link_path.append(link_id)

        completed = bool(hops) and hops[-1].get("address") == destination_address
        destination_verified = completed and previous_node == destination.node
        ok = result.ok and destination_verified and not unresolved
        return {
            "ok": ok,
            "lab": self.manifest.name,
            "evidence_type": "observed-data-plane-traceroute",
            "source": {
                "endpoint_id": source_endpoint,
                "node": source.node,
                "address": source_address,
            },
            "destination": {
                "endpoint_id": destination_endpoint,
                "node": destination.node,
                "address": destination_address,
            },
            "hops": hops,
            "node_path": node_path,
            "link_path": link_path,
            "destination_verified": destination_verified,
            "all_hops_resolved": not unresolved,
            "unresolved": unresolved,
            "exit_code": result.returncode,
            "raw_output": output,
            "fail_closed": not ok,
        }

    async def enforcement_path(self, user_id: str, app_id: str) -> dict[str, object]:
        """Join actual admission, policy, and traffic evidence for one user/app pair."""
        user = self._user(user_id)
        app = self._application(app_id)
        admitted, blocked = await asyncio.gather(
            self.user_admitted(user_id),
            self.application_access_blocked(user_id, app_id),
        )
        trace: dict[str, object]
        if admitted and not blocked:
            trace = await self.trace_path(user.endpoint, app.endpoint)
        else:
            trace = {
                "ok": False,
                "not_executed": True,
                "reason": (
                    "network-admission-denied" if not admitted
                    else "application-source-policy-denied"
                ),
                "fail_closed": True,
            }
        traffic_allowed = admitted and not blocked and bool(trace.get("ok"))
        return {
            "ok": True,
            "lab": self.manifest.name,
            "user_id": user_id,
            "app_id": app_id,
            "traffic_allowed_and_verified": traffic_allowed,
            "enforcement_points": [
                {
                    "order": 1,
                    "function": "network-admission",
                    "implementation": "simulated-endpoint-interface-state",
                    "node": self.manifest.endpoints[user.endpoint].node,
                    "interface": user.interface,
                    "observed_state": "admitted" if admitted else "quarantined",
                    "is_real_radius_or_8021x": False,
                },
                {
                    "order": 2,
                    "function": "application-source-policy",
                    "implementation": "linux-server-source-blackhole-route",
                    "node": self.manifest.endpoints[app.endpoint].node,
                    "source_prefix": f"{user.address}/32",
                    "observed_state": "blackhole" if blocked else "permit",
                    "is_leaf_acl_or_application_iam": False,
                },
            ],
            "traffic_path": trace,
            "security_boundary": {
                "security_edges_in_observed_internal_path": bool(
                    trace.get("ok") and any(
                        str(node).startswith("security-edge-")
                        for node in trace.get("node_path", [])
                    )
                ),
                "stateful_firewall_simulated": False,
            },
        }

    async def _exec_node(self, node: str, argv: Sequence[str]) -> str:
        if node not in self.manifest.nodes:
            raise LabCommandError(f"node {node!r} is not declared in the manifest")
        container = self.manifest.container_name(node)
        result = await self._run(("docker", "exec", container, *argv))
        return self._require_ok(result, f"docker exec {node}")

    def _fabric(self):
        if self.manifest.fabric is None:
            raise LabCommandError("this lab manifest has no typed fabric contract")
        return self.manifest.fabric

    async def _json_node(self, node: str, argv: Sequence[str]) -> object:
        output = await self._exec_node(node, argv)
        try:
            return json.loads(output)
        except json.JSONDecodeError as error:
            raise LabCommandError(
                f"node {node!r} returned invalid JSON for {tuple(argv)!r}"
            ) from error

    def _fabric_attachment(self, device_id: str, interface: str):
        fabric = self._fabric()
        self._device(device_id)
        for attachment in fabric.attachments.values():
            if attachment.device == device_id and attachment.interface == interface:
                return attachment
        raise LabCommandError(
            f"interface {device_id}:{interface} is not a reviewed fabric attachment"
        )

    async def fabric_access_vlan(self, device_id: str, interface: str) -> dict[str, object]:
        """Read the actual Linux bridge/PVID state for one declared access port."""
        attachment = self._fabric_attachment(device_id, interface)
        if attachment.mode != "access":
            raise LabCommandError("fabric access VLAN operations require an access attachment")
        device = self._device(device_id)
        vlan_payload, link_payload = await asyncio.gather(
            self._json_node(device.node, ("bridge", "-j", "vlan", "show", "dev", interface)),
            self._json_node(device.node, ("ip", "-j", "link", "show", "dev", interface)),
        )
        if not isinstance(vlan_payload, list) or len(vlan_payload) != 1:
            raise LabCommandError("bridge VLAN state is missing or ambiguous")
        if not isinstance(link_payload, list) or len(link_payload) != 1:
            raise LabCommandError("access-port link state is missing or ambiguous")
        raw_vlans = vlan_payload[0].get("vlans") or {}
        if not isinstance(raw_vlans, list):
            raise LabCommandError("bridge VLAN state has an invalid shape")
        vlan_entries = [
            {
                "vlan_id": int(item["vlan"]),
                "pvid": "PVID" in (item.get("flags") or []),
                "untagged": "Egress Untagged" in (item.get("flags") or []),
            }
            for item in raw_vlans
            if isinstance(item, dict) and isinstance(item.get("vlan"), int)
        ]
        pvids = [item["vlan_id"] for item in vlan_entries if item["pvid"]]
        if len(pvids) != 1:
            raise LabCommandError("access port must have exactly one observed PVID")
        current_vlan = pvids[0]
        fabric = self._fabric()
        expected_bridge = (
            fabric.vlans[current_vlan].bridge if current_vlan in fabric.vlans else None
        )
        pvid_entry = next(item for item in vlan_entries if item["pvid"])
        observed_bridge = link_payload[0].get("master")
        return {
            "ok": (
                expected_bridge is not None
                and observed_bridge == expected_bridge
                and pvid_entry["untagged"] is True
            ),
            "evidence_type": "observed-linux-bridge-vlan",
            "device_id": device_id,
            "node": device.node,
            "interface": interface,
            "endpoint": attachment.endpoint,
            "mode": attachment.mode,
            "manifest_vlans": list(attachment.vlans),
            "current_vlan": current_vlan,
            "bridge": observed_bridge,
            "vlans": vlan_entries,
        }

    async def set_fabric_access_vlan(
        self, device_id: str, interface: str, vlan_id: int,
    ) -> str:
        """Move one reviewed access port to one declared VLAN using fixed argv only."""
        fabric = self._fabric()
        attachment = self._fabric_attachment(device_id, interface)
        if attachment.mode != "access":
            raise LabCommandError("only reviewed access ports may be moved between VLANs")
        if vlan_id not in fabric.vlans:
            raise LabCommandError(f"VLAN {vlan_id} is not declared by the fabric manifest")
        before = await self.fabric_access_vlan(device_id, interface)
        if before["ok"] is not True:
            raise LabCommandError("access port is not in a valid declared preflight state")
        current_vlan = int(before["current_vlan"])
        if current_vlan == vlan_id:
            raise LabCommandError("requested access VLAN already matches observed state")
        key = (device_id, interface)
        self._fabric_vlan_snapshots[key] = before
        node = self._device(device_id).node
        target = fabric.vlans[vlan_id]
        commands = (
            ("bridge", "vlan", "del", "dev", interface, "vid", str(current_vlan)),
            ("ip", "link", "set", "dev", interface, "nomaster"),
            ("ip", "link", "set", "dev", interface, "master", target.bridge),
            (
                "bridge", "vlan", "add", "dev", interface, "vid", str(vlan_id),
                "pvid", "untagged",
            ),
            ("ip", "link", "set", "dev", interface, "up"),
        )
        for command in commands:
            await self._exec_node(node, command)
        observed = await self.fabric_access_vlan(device_id, interface)
        if observed["current_vlan"] != vlan_id or observed["bridge"] != target.bridge:
            raise LabCommandError("access VLAN command completed but target state was not observed")
        return json.dumps({
            "ok": True,
            "device_id": device_id,
            "interface": interface,
            "previous_vlan": current_vlan,
            "current_vlan": vlan_id,
            "bridge": target.bridge,
        }, sort_keys=True)

    async def restore_fabric_access_vlan(self, device_id: str, interface: str) -> str:
        """Restore the exact execution-session bridge and PVID snapshot."""
        key = (device_id, interface)
        snapshot = self._fabric_vlan_snapshots.get(key)
        if snapshot is None:
            raise LabCommandError(
                f"no execution-session fabric snapshot exists for {device_id}:{interface}"
            )
        return await self.restore_fabric_access_vlan_snapshot(
            device_id, interface, snapshot,
        )

    async def restore_fabric_access_vlan_snapshot(
        self,
        device_id: str,
        interface: str,
        snapshot: dict[str, object],
    ) -> str:
        """Restore a validated durable bridge/PVID snapshot."""
        if (
            snapshot.get("device_id") != device_id
            or snapshot.get("interface") != interface
            or snapshot.get("mode") != "access"
            or snapshot.get("ok") is not True
        ):
            raise LabCommandError("fabric snapshot does not match the reviewed target")
        previous_vlan = int(snapshot["current_vlan"])
        fabric = self._fabric()
        if previous_vlan not in fabric.vlans:
            raise LabCommandError("fabric snapshot references an undeclared VLAN")
        previous = fabric.vlans[previous_vlan]
        if snapshot.get("bridge") != previous.bridge:
            raise LabCommandError("fabric snapshot bridge does not match the manifest")
        current = await self.fabric_access_vlan(device_id, interface)
        if current["ok"] is not True:
            raise LabCommandError("access port is not in a valid declared state before rollback")
        node = self._device(device_id).node
        current_vlan = int(current["current_vlan"])
        commands = (
            ("bridge", "vlan", "del", "dev", interface, "vid", str(current_vlan)),
            ("ip", "link", "set", "dev", interface, "nomaster"),
            ("ip", "link", "set", "dev", interface, "master", previous.bridge),
            (
                "bridge", "vlan", "add", "dev", interface, "vid", str(previous_vlan),
                "pvid", "untagged",
            ),
            ("ip", "link", "set", "dev", interface, "up"),
        )
        for command in commands:
            await self._exec_node(node, command)
        restored = await self.fabric_access_vlan(device_id, interface)
        comparable = ("current_vlan", "bridge", "vlans")
        if any(restored[field] != snapshot[field] for field in comparable):
            raise LabCommandError("fabric rollback did not restore the exact access-port state")
        return json.dumps({
            "ok": True,
            "device_id": device_id,
            "interface": interface,
            "restored_vlan": previous_vlan,
            "bridge": previous.bridge,
        }, sort_keys=True)

    async def fabric_bgp_evpn_summary(self, device_id: str) -> dict[str, object]:
        fabric = self._fabric()
        if device_id not in {*fabric.route_reflectors, *fabric.vteps}:
            raise LabCommandError(f"device {device_id!r} is not a declared fabric BGP node")
        payload = await self._json_node(
            self._device(device_id).node,
            ("vtysh", "-c", "show bgp l2vpn evpn summary json"),
        )
        if not isinstance(payload, dict) or not isinstance(payload.get("peers"), dict):
            raise LabCommandError("BGP EVPN summary has an invalid shape")
        peers = payload["peers"]
        established = sum(
            1 for peer in peers.values()
            if isinstance(peer, dict) and peer.get("state") == "Established"
        )
        expected = self.manifest.devices[device_id].expected_bgp_neighbors
        return {
            "ok": established >= expected,
            "evidence_type": "observed-frr-bgp-evpn-summary",
            "device_id": device_id,
            "asn": payload.get("as"),
            "router_id": payload.get("routerId"),
            "expected_neighbors": expected,
            "established_neighbors": established,
            "peers": peers,
        }

    async def fabric_vxlan_state(self, device_id: str) -> dict[str, object]:
        fabric = self._fabric()
        if device_id not in fabric.vteps:
            raise LabCommandError(f"device {device_id!r} is not a declared VTEP")
        device = self._device(device_id)
        links, vnis = await asyncio.gather(
            self._json_node(device.node, ("ip", "-j", "-d", "link", "show", "type", "vxlan")),
            self._json_node(device.node, ("vtysh", "-c", "show evpn vni json")),
        )
        if not isinstance(links, list) or not isinstance(vnis, dict):
            raise LabCommandError("VXLAN/VNI state has an invalid shape")
        expected = {vlan.l2vni for vlan in fabric.vlans.values()}
        observed_links = {
            int(item.get("linkinfo", {}).get("info_data", {}).get("id"))
            for item in links if isinstance(item, dict)
            and isinstance(item.get("linkinfo", {}).get("info_data", {}).get("id"), int)
        }
        observed_vnis = {
            int(key) for key, value in vnis.items()
            if str(key).isdigit() and isinstance(value, dict) and value.get("type") == "L2"
        }
        remote_ready = all(
            int(vnis[str(vni)].get("numRemoteVteps", 0)) >= len(fabric.vteps) - 1
            for vni in expected if str(vni) in vnis
        )
        return {
            "ok": expected == observed_links == observed_vnis and remote_ready,
            "evidence_type": "observed-linux-vxlan-and-frr-vni",
            "device_id": device_id,
            "expected_l2vnis": sorted(expected),
            "linux_vxlan_ids": sorted(observed_links),
            "frr_vnis": vnis,
            "remote_vteps_verified": remote_ready,
            "links": links,
        }

    async def fabric_evpn_routes(
        self, device_id: str, route_type: int | None = None,
    ) -> dict[str, object]:
        if route_type is not None and route_type not in {2, 3, 5}:
            raise LabCommandError("EVPN route_type must be 2, 3, or 5")
        payload = await self._json_node(
            self._device(device_id).node,
            ("vtysh", "-c", "show bgp l2vpn evpn route json"),
        )
        if not isinstance(payload, dict):
            raise LabCommandError("EVPN route table has an invalid shape")
        routes: list[dict[str, object]] = []
        for rd, table in payload.items():
            if not isinstance(table, dict):
                continue
            for prefix, value in table.items():
                if prefix == "rd" or not isinstance(value, dict):
                    continue
                paths = value.get("paths") or []
                flattened = [
                    path for group in paths if isinstance(group, list)
                    for path in group if isinstance(path, dict)
                ]
                observed_type = next(
                    (int(path["routeType"]) for path in flattened if "routeType" in path),
                    None,
                )
                if route_type is None or observed_type == route_type:
                    routes.append({
                        "rd": rd,
                        "prefix": prefix,
                        "route_type": observed_type,
                        "paths": flattened,
                    })
        return {
            "ok": bool(routes),
            "evidence_type": "observed-frr-evpn-rib",
            "device_id": device_id,
            "route_type_filter": route_type,
            "route_count": len(routes),
            "routes": routes,
        }

    async def fabric_state(self) -> dict[str, object]:
        fabric = self._fabric()
        bgp_values = await asyncio.gather(*(
            self.fabric_bgp_evpn_summary(device_id)
            for device_id in (*fabric.route_reflectors, *fabric.vteps)
        ))
        vxlan_values = await asyncio.gather(*(
            self.fabric_vxlan_state(device_id) for device_id in fabric.vteps
        ))
        attachments = await asyncio.gather(*(
            self.fabric_access_vlan(item.device, item.interface)
            for item in fabric.attachments.values() if item.mode == "access"
        ))
        return {
            "ok": all(item["ok"] for item in (*bgp_values, *vxlan_values, *attachments)),
            "lab": self.manifest.name,
            "evidence_type": "observed-evpn-vxlan-fabric",
            "contract": {
                "mode": fabric.mode,
                "asn": fabric.asn,
                "route_reflectors": list(fabric.route_reflectors),
                "vteps": list(fabric.vteps),
                "vlans": [asdict(fabric.vlans[key]) for key in sorted(fabric.vlans)],
                "attachments": [
                    asdict(fabric.attachments[key]) for key in sorted(fabric.attachments)
                ],
            },
            "bgp_evpn": list(bgp_values),
            "vxlan": list(vxlan_values),
            "access_ports": list(attachments),
            "truth_boundary": {
                "real_8021q": True,
                "real_linux_bridge": True,
                "real_vxlan_dataplane": True,
                "real_bgp_evpn_control_plane": True,
                "evpn_l2vpn": True,
                "evpn_l3vpn": False,
                "mpls_l2vpn": False,
                "mpls_l3vpn": False,
                "vendor_cli_or_asic": False,
            },
        }

    async def user_admitted(self, user_id: str) -> bool:
        """Read the actual endpoint interface state for a declared lab user."""
        user = self._user(user_id)
        output = await self._exec_node(
            self.manifest.endpoints[user.endpoint].node,
            ("cat", f"/sys/class/net/{user.interface}/operstate"),
        )
        return user.status == "active" and output.strip().lower() == "up"

    async def user_admission_snapshot(self, user_id: str) -> dict[str, object]:
        """Capture the complete manifest-scoped admission effect surface."""
        user = self._user(user_id)
        node = self.manifest.endpoints[user.endpoint].node
        admitted = await self.user_admitted(user_id)
        rendered = await self._exec_node(node, ("ip", "route", "show"))
        lines = tuple(
            " ".join(line.split()) for line in rendered.splitlines() if line.strip()
        )
        routes: dict[str, bool] = {}
        for prefix in user.route_prefixes:
            expected = f"{prefix} via {user.gateway} dev {user.interface}"
            matches = [line for line in lines if line.split(maxsplit=1)[0] == prefix]
            if matches not in ([], [expected]):
                raise LabCommandError(
                    f"user {user_id!r} route {prefix!r} is outside the reviewed admission shape"
                )
            routes[prefix] = matches == [expected]
        if any(present is not admitted for present in routes.values()):
            raise LabCommandError(
                f"user {user_id!r} interface and reviewed routes are not internally consistent"
            )
        return {
            "user_id": user_id,
            "endpoint": user.endpoint,
            "interface": user.interface,
            "admitted": admitted,
            "routes": routes,
        }

    async def restore_user_admission_snapshot(
        self, user_id: str, snapshot: dict[str, object],
    ) -> str:
        """Restore and read back one exact durable admission snapshot."""
        user = self._user(user_id)
        routes = snapshot.get("routes")
        admitted = snapshot.get("admitted")
        if (
            snapshot.get("user_id") != user_id
            or snapshot.get("endpoint") != user.endpoint
            or snapshot.get("interface") != user.interface
            or not isinstance(admitted, bool)
            or not isinstance(routes, dict)
            or set(routes) != set(user.route_prefixes)
            or any(not isinstance(value, bool) for value in routes.values())
            or any(value is not admitted for value in routes.values())
        ):
            raise LabCommandError("durable user-admission snapshot is invalid for this target")
        result = await self.set_user_admission(user_id, admitted=admitted)
        observed = await self.user_admission_snapshot(user_id)
        if observed != snapshot:
            raise LabCommandError("durable user-admission snapshot did not restore exactly")
        return result

    async def set_user_admission(self, user_id: str, *, admitted: bool) -> str:
        """Change only the manifest-bound endpoint interface, never an arbitrary target."""
        user = self._user(user_id)
        if admitted and user.status != "active":
            raise LabCommandError(
                f"user {user_id!r} has identity status={user.status}; network admission is forbidden"
            )
        node = self.manifest.endpoints[user.endpoint].node
        state = "up" if admitted else "down"
        await self._exec_node(node, ("ip", "link", "set", "dev", user.interface, state))
        if admitted:
            # Linux removes dependent routes when an interface is administratively
            # disabled. Reinstall only the exact route reviewed in the manifest.
            for prefix in user.route_prefixes:
                await self._exec_node(node, (
                    "ip", "route", "replace", prefix,
                    "via", user.gateway, "dev", user.interface,
                ))
        observed = await self.user_admitted(user_id)
        if observed is not admitted:
            raise LabCommandError(
                f"endpoint interface state did not converge to admitted={admitted} for {user_id}"
            )
        return f"endpoint={user.endpoint} interface={user.interface} state={state}"

    async def application_access_blocked(self, user_id: str, app_id: str) -> bool:
        """Read the application endpoint's exact per-user blackhole policy."""
        snapshot = await self.application_access_snapshot(user_id, app_id)
        return snapshot["allowed"] is False

    async def application_access_snapshot(
        self, user_id: str, app_id: str,
    ) -> dict[str, object]:
        """Capture the complete manifest-scoped per-user application route."""
        user = self._user(user_id)
        app = self._application(app_id)
        node = self.manifest.endpoints[app.endpoint].node
        output = await self._exec_node(node, ("ip", "route", "show", f"{user.address}/32"))
        routes = [" ".join(line.split()) for line in output.splitlines() if line.strip()]
        expected = f"blackhole {user.address}"
        if routes not in ([], [expected]):
            raise LabCommandError(
                f"application route for {user_id!r}->{app_id!r} is outside the reviewed shape"
            )
        return {
            "user_id": user_id,
            "app_id": app_id,
            "source_prefix": f"{user.address}/32",
            "application_endpoint": app.endpoint,
            "allowed": not routes,
            "route": routes[0] if routes else None,
        }

    async def restore_application_access_snapshot(
        self, user_id: str, app_id: str, snapshot: dict[str, object],
    ) -> str:
        """Restore and read back one exact durable application-policy snapshot."""
        user = self._user(user_id)
        app = self._application(app_id)
        expected_route = None if snapshot.get("allowed") is True else f"blackhole {user.address}"
        if (
            snapshot.get("user_id") != user_id
            or snapshot.get("app_id") != app_id
            or snapshot.get("source_prefix") != f"{user.address}/32"
            or snapshot.get("application_endpoint") != app.endpoint
            or not isinstance(snapshot.get("allowed"), bool)
            or snapshot.get("route") != expected_route
        ):
            raise LabCommandError("durable application-policy snapshot is invalid for this target")
        result = await self.set_application_access(
            user_id, app_id, allowed=bool(snapshot["allowed"]),
        )
        observed = await self.application_access_snapshot(user_id, app_id)
        if observed != snapshot:
            raise LabCommandError("durable application-policy snapshot did not restore exactly")
        return result

    async def set_application_access(
        self, user_id: str, app_id: str, *, allowed: bool,
    ) -> str:
        """Apply an idempotent, manifest-bound application policy to the server namespace."""
        user = self._user(user_id)
        app = self._application(app_id)
        node = self.manifest.endpoints[app.endpoint].node
        currently_blocked = await self.application_access_blocked(user_id, app_id)
        if allowed and currently_blocked:
            await self._exec_node(node, ("ip", "route", "del", "blackhole", f"{user.address}/32"))
        elif not allowed and not currently_blocked:
            await self._exec_node(node, ("ip", "route", "replace", "blackhole", f"{user.address}/32"))
        observed = not await self.application_access_blocked(user_id, app_id)
        if observed is not allowed:
            raise LabCommandError(
                f"application policy did not converge to allowed={allowed} for {user_id}->{app_id}"
            )
        return (
            f"application={app_id} source={user.address}/32 "
            f"policy={'permit' if allowed else 'blackhole'}"
        )

    async def application_probe(self, user_id: str, app_id: str) -> dict[str, object]:
        """Run a real HTTP request from the declared user endpoint to the declared app."""
        user = self._user(user_id)
        app = self._application(app_id)
        node = self.manifest.endpoints[user.endpoint].node
        url = f"http://{app.address}:{app.port}{app.path}"
        result = await self._run((
            "docker", "exec", self.manifest.container_name(node),
            "wget", "-q", "-T", "3", "-O", "-", url,
        ))
        output = _bounded((result.stdout + result.stderr).strip(), 2_000)
        policy_allowed = not await self.application_access_blocked(user_id, app_id)
        return {
            "ok": result.ok and policy_allowed,
            "user_id": user_id,
            "app_id": app_id,
            "source": user.address,
            "destination": app.address,
            "port": app.port,
            "path": app.path,
            "policy_allowed": policy_allowed,
            "http_response": output,
            "exit_code": result.returncode,
        }

    async def show(self, device_id: str, command: str) -> str:
        command = command.strip()
        if not _SHOW_COMMAND.fullmatch(command):
            raise LabCommandError("FRR read commands must be a single bounded 'show ...' command")
        device = self._device(device_id)
        return await self._exec_node(device.node, ("vtysh", "-c", command))

    async def running_config(self, device_id: str, section: str | None = None) -> str:
        output = await self.show(device_id, "show running-config")
        normalized = normalize_frr_config(output)
        if section:
            needle = section.strip().lower()
            if not re.fullmatch(r"[a-zA-Z0-9_.:/-]{1,64}", needle):
                raise LabCommandError("configuration section contains unsupported characters")
            blocks = normalized.split("\n!\n")
            normalized = "\n!\n".join(block for block in blocks if needle in block.lower())
            if normalized and not normalized.endswith("\n"):
                normalized += "\n"
        return f"! Lab configuration for {device_id}\n{normalized}"

    @staticmethod
    def validate_config_lines(lines: Sequence[str]) -> tuple[str, ...]:
        if not lines or len(lines) > 100:
            raise LabCommandError("config_lines must contain between 1 and 100 reviewed commands")
        validated: list[str] = []
        context = "global"
        for raw in lines:
            line = str(raw).strip()
            if "\n" in line or "\r" in line or not _CONFIG_COMMAND.fullmatch(line):
                raise LabCommandError(f"unreviewed FRR configuration command: {line!r}")
            lowered = line.lower()
            if lowered == "exit":
                if context == "global":
                    raise LabCommandError("exit is allowed only to leave a declared subcontext")
                context = "global"
                validated.append(line)
                continue
            if lowered.startswith("interface "):
                context = "interface"
            elif lowered == "router ospf":
                context = "router_ospf"
            elif lowered.startswith("route-map "):
                context = "route_map"
            else:
                allowed_by_context = {
                    "global": (
                        "ip route ", "no ip route ", "ip prefix-list ",
                        "no ip prefix-list ",
                    ),
                    "interface": (
                        "description ", "no description", "ip address ",
                        "ip ospf ", "no ip ospf cost", "shutdown", "no shutdown",
                    ),
                    "router_ospf": ("ospf router-id ", "passive-interface ", "network "),
                    "route_map": ("match ip address prefix-list ", "set metric "),
                }
                if not any(lowered.startswith(prefix) for prefix in allowed_by_context[context]):
                    raise LabCommandError(
                        f"FRR command {line!r} is invalid in reviewed {context} context"
                    )
            try:
                if lowered.startswith("ip address "):
                    ipaddress.ip_interface(line.split(maxsplit=2)[2])
                elif lowered.startswith("ospf router-id "):
                    ipaddress.ip_address(line.split(maxsplit=2)[2])
                elif lowered.startswith("network "):
                    ipaddress.ip_network(line.split()[1], strict=False)
                elif lowered.startswith(("ip route ", "no ip route ")):
                    parts = line.split()
                    offset = 1 if parts[0].lower() == "no" else 0
                    ipaddress.ip_network(parts[offset + 2], strict=False)
                    ipaddress.ip_address(parts[offset + 3])
                elif (
                    " prefix-list " in f" {lowered} "
                    and (" permit " in f" {lowered} " or " deny " in f" {lowered} ")
                ):
                    tokens = line.split()
                    action_index = next(
                        index for index, value in enumerate(tokens)
                        if value.lower() in {"permit", "deny"}
                    )
                    ipaddress.ip_network(tokens[action_index + 1], strict=False)
            except (ValueError, IndexError, StopIteration) as error:
                raise LabCommandError(f"FRR command contains an invalid IP value: {line!r}") from error
            validated.append(line)
        return tuple(validated)

    async def apply_config(self, device_id: str, lines: Sequence[str]) -> str:
        device = self._device(device_id)
        commands = self.validate_config_lines(lines)
        self._snapshots[device_id] = await self.running_config(device_id)
        argv: list[str] = ["vtysh", "-c", "configure terminal"]
        for command in commands:
            argv.extend(("-c", command))
        argv.extend(("-c", "end"))
        output = await self._exec_node(device.node, tuple(argv))
        return (
            f"Config applied to lab device {device_id}.\n"
            f"Commands accepted: {len(commands)}\n{output}"
        )

    async def restore_last_config(self, device_id: str) -> str:
        snapshot = self._snapshots.get(device_id)
        if snapshot is None:
            raise LabCommandError(f"no execution-session snapshot exists for {device_id}")
        return await self.restore_config_snapshot(device_id, snapshot)

    async def restore_config_snapshot(self, device_id: str, snapshot: str) -> str:
        """Restore an explicit durable FRR snapshot after target validation."""
        device = self._device(device_id)
        if not isinstance(snapshot, str) or not snapshot.strip():
            raise LabCommandError("configuration snapshot is empty or invalid")
        payload = "\n".join(
            line for line in snapshot.splitlines()
            if not line.startswith("! Lab configuration for ")
        ).strip() + "\n"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        self.manifest.state_directory.mkdir(parents=True, exist_ok=True)
        os.chmod(self.manifest.state_directory, 0o700)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f"{device_id}-{digest}-",
            suffix=".conf",
            dir=self.manifest.state_directory,
            delete=False,
        ) as handle:
            handle.write(payload)
            snapshot_path = Path(handle.name)
        os.chmod(snapshot_path, 0o600)
        remote_path = f"/tmp/netopyu-{digest}.conf"
        try:
            copy = await self._run((
                "docker", "cp", str(snapshot_path),
                f"{self.manifest.container_name(device.node)}:{remote_path}",
            ))
            self._require_ok(copy, "copy rollback snapshot")
            output = await self._exec_node(device.node, (
                "/usr/lib/frr/frr-reload.py", "--reload", remote_path,
            ))
        finally:
            snapshot_path.unlink(missing_ok=True)
        restored = await self.running_config(device_id)
        if normalize_frr_config(restored) != normalize_frr_config(snapshot):
            raise LabCommandError("FRR reload completed but exact normalized snapshot was not restored")
        return f"Restored exact durable snapshot for {device_id}.\n{output}"

    async def probe(self, probe_id: str) -> dict[str, object]:
        try:
            probe = self.manifest.probes[probe_id]
        except KeyError as error:
            raise LabCommandError(f"unknown probe {probe_id!r}") from error
        destination = str(ipaddress.ip_address(probe.destination))
        endpoint = self.manifest.endpoints[probe.source]
        result = await self._run((
            "docker", "exec", self.manifest.container_name(endpoint.node),
            "ping", "-c", str(probe.count), "-W", "2", destination,
        ))
        output = _bounded((result.stdout + result.stderr).strip(), 4_000)
        received = re.search(r"(\d+) packets transmitted, (\d+) packets received", output)
        if received is None:
            received = re.search(r"(\d+) packets transmitted, (\d+) received", output)
        transmitted = int(received.group(1)) if received else probe.count
        replies = int(received.group(2)) if received else 0
        return {
            "ok": result.ok and transmitted == replies and transmitted > 0,
            "probe_id": probe_id,
            "source": probe.source,
            "destination": destination,
            "transmitted": transmitted,
            "received": replies,
            "output": output,
        }

    async def set_fault(
        self,
        fault_id: str,
        *,
        kind: str,
        value: int | None = None,
    ) -> str:
        try:
            target = self.manifest.fault_targets[fault_id]
        except KeyError as error:
            raise LabCommandError(f"unknown fault target {fault_id!r}") from error
        if kind == "link_down":
            argv = ("ip", "link", "set", "dev", target.interface, "down")
        elif kind == "link_up":
            argv = ("ip", "link", "set", "dev", target.interface, "up")
        elif kind in {"delay_ms", "loss_pct"}:
            if not isinstance(value, int) or isinstance(value, bool):
                raise LabCommandError(f"{kind} requires an integer value")
            if kind == "delay_ms" and not 1 <= value <= 5_000:
                raise LabCommandError("delay_ms must be between 1 and 5000")
            if kind == "loss_pct" and not 1 <= value <= 100:
                raise LabCommandError("loss_pct must be between 1 and 100")
            option = ("delay", f"{value}ms") if kind == "delay_ms" else ("loss", f"{value}%")
            argv = ("tc", "qdisc", "replace", "dev", target.interface, "root", "netem", *option)
        elif kind == "clear_netem":
            argv = ("tc", "qdisc", "del", "dev", target.interface, "root")
        else:
            raise LabCommandError("unsupported fault kind")
        return await self._exec_node(target.node, argv)

    def inventory(self) -> list[dict[str, object]]:
        return [asdict(self.manifest.devices[key]) for key in sorted(self.manifest.devices)]
