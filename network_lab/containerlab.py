"""Containerlab implementation of the reviewed local laboratory contract."""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
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

    async def user_admitted(self, user_id: str) -> bool:
        """Read the actual endpoint interface state for a declared lab user."""
        user = self._user(user_id)
        output = await self._exec_node(
            self.manifest.endpoints[user.endpoint].node,
            ("cat", f"/sys/class/net/{user.interface}/operstate"),
        )
        return user.status == "active" and output.strip().lower() == "up"

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
        user = self._user(user_id)
        app = self._application(app_id)
        node = self.manifest.endpoints[app.endpoint].node
        output = await self._exec_node(node, ("ip", "route", "show", f"{user.address}/32"))
        return output.strip().lower().startswith("blackhole ")

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
        device = self._device(device_id)
        snapshot = self._snapshots.get(device_id)
        if snapshot is None:
            raise LabCommandError(f"no execution-session snapshot exists for {device_id}")
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
        return f"Restored exact execution-session snapshot for {device_id}.\n{output}"

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
