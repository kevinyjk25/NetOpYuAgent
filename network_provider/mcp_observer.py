"""Identity-pinned read-only MCP façade over a reviewed network provider.

The first implementation wraps Containerlab, but its public contract contains
no Containerlab-specific transaction state.  Every result is carried in a
versioned evidence envelope.  The NetOpYu MCP client validates and unwraps the
envelope before existing Runtime verifiers consume the compatibility payload.
"""

from __future__ import annotations

import argparse
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

from mcp.server.mcpserver import MCPServer

from network_lab import ContainerlabProvider, load_manifest
from network_lab.tools import LabToolAdapter, lab_tool_metadata
from network_runtime.contracts import sha256_json
from network_runtime.provider_contracts import EVIDENCE_CONTRACT, REGISTRY
from tools.pragmatic.registry import TOOLS as PRAGMATIC_METADATA

from .models import NetworkEvidenceEnvelope


SERVER_NAME = "netopyu.network-observer"
SERVER_VERSION = "1.0.0"
PROVIDER_IDENTITY = f"{SERVER_NAME}@{SERVER_VERSION}"


def _decode_payload(rendered: str) -> tuple[Any, str]:
    try:
        return json.loads(rendered), "application/json"
    except (TypeError, json.JSONDecodeError):
        return rendered, "text/plain"


class ObserverTools:
    def __init__(self, adapter: LabToolAdapter, profile: str) -> None:
        self.adapter = adapter
        self.profile = profile
        self.callables = adapter.callables(profile)

    async def _observe(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        correlation_id: str | None,
    ) -> NetworkEvidenceEnvelope:
        capability = REGISTRY.for_tool(tool_name)
        if capability is None or capability.provider_role != "observer":
            raise RuntimeError(f"unreviewed observer capability {tool_name!r}")
        callable_ = self.callables.get(tool_name)
        if callable_ is None:
            raise RuntimeError(
                f"capability {capability.capability_id!r} is unavailable for profile {self.profile!r}"
            )
        rendered = await callable_(arguments)
        payload, content_type = _decode_payload(rendered)
        return NetworkEvidenceEnvelope(
            correlation_id=correlation_id or f"corr-{uuid.uuid4().hex}",
            observed_at=datetime.now(timezone.utc).isoformat(),
            simulation=True,
            provider_identity=PROVIDER_IDENTITY,
            capability_id=capability.capability_id,
            capability_version=capability.capability_version,
            payload_digest=sha256_json(payload),
            content_type=content_type,
            payload=payload,
        )

    async def list_devices(
        self, type: str | None = None, tag: str | None = None,
        correlation_id: str | None = None,
    ) -> NetworkEvidenceEnvelope:
        return await self._observe("list_devices", {"type": type, "tag": tag}, correlation_id)

    async def get_device_status(
        self, device_id: str, correlation_id: str | None = None,
    ) -> NetworkEvidenceEnvelope:
        return await self._observe("get_device_status", {"device_id": device_id}, correlation_id)

    async def get_device_config(
        self, device_id: str, section: str | None = None,
        correlation_id: str | None = None,
    ) -> NetworkEvidenceEnvelope:
        return await self._observe(
            "get_device_config", {"device_id": device_id, "section": section}, correlation_id,
        )

    async def validate_device_config(
        self, device_id: str, correlation_id: str | None = None,
    ) -> NetworkEvidenceEnvelope:
        return await self._observe("validate_device_config", {"device_id": device_id}, correlation_id)

    async def get_syslog(
        self, device_id: str, level: str | None = None, lines: int | None = None,
        correlation_id: str | None = None,
    ) -> NetworkEvidenceEnvelope:
        return await self._observe(
            "get_syslog", {"device_id": device_id, "level": level, "lines": lines}, correlation_id,
        )

    async def query_interface_metrics(
        self, device_id: str, interface: str | None = None,
        correlation_id: str | None = None,
    ) -> NetworkEvidenceEnvelope:
        return await self._observe(
            "query_interface_metrics", {"device_id": device_id, "interface": interface}, correlation_id,
        )

    async def get_bgp_summary(
        self, device_id: str, correlation_id: str | None = None,
    ) -> NetworkEvidenceEnvelope:
        return await self._observe("get_bgp_summary", {"device_id": device_id}, correlation_id)

    async def get_device_facts(
        self, device_id: str, correlation_id: str | None = None,
    ) -> NetworkEvidenceEnvelope:
        return await self._observe("get_device_facts", {"device_id": device_id}, correlation_id)

    async def run_command(
        self, device_id: str, command: str, correlation_id: str | None = None,
    ) -> NetworkEvidenceEnvelope:
        return await self._observe(
            "run_command", {"device_id": device_id, "command": command}, correlation_id,
        )

    async def multi_device_check(
        self, device_ids: list[str] | str | None = None, command: str | None = None,
        tag: str | None = None, correlation_id: str | None = None,
    ) -> NetworkEvidenceEnvelope:
        if isinstance(device_ids, str) and device_ids != "all":
            device_ids = [device_ids]
        return await self._observe("multi_device_check", {
            "device_ids": device_ids, "command": command, "tag": tag,
        }, correlation_id)

    async def get_ospf_neighbors(
        self, device_id: str, correlation_id: str | None = None,
    ) -> NetworkEvidenceEnvelope:
        return await self._observe("get_ospf_neighbors", {"device_id": device_id}, correlation_id)

    async def lab_topology_status(
        self, correlation_id: str | None = None,
    ) -> NetworkEvidenceEnvelope:
        return await self._observe("lab_topology_status", {}, correlation_id)

    async def lab_get_topology_graph(
        self, correlation_id: str | None = None,
    ) -> NetworkEvidenceEnvelope:
        return await self._observe("lab_get_topology_graph", {}, correlation_id)

    async def lab_get_endpoint(
        self, endpoint_id: str, correlation_id: str | None = None,
    ) -> NetworkEvidenceEnvelope:
        return await self._observe("lab_get_endpoint", {"endpoint_id": endpoint_id}, correlation_id)

    async def lab_trace_path(
        self, source_endpoint: str, destination_endpoint: str,
        correlation_id: str | None = None,
    ) -> NetworkEvidenceEnvelope:
        return await self._observe("lab_trace_path", {
            "source_endpoint": source_endpoint,
            "destination_endpoint": destination_endpoint,
        }, correlation_id)

    async def lab_get_enforcement_path(
        self, user_id: str, app_id: str, correlation_id: str | None = None,
    ) -> NetworkEvidenceEnvelope:
        return await self._observe(
            "lab_get_enforcement_path", {"user_id": user_id, "app_id": app_id}, correlation_id,
        )

    async def lab_probe(
        self, probe_id: str, correlation_id: str | None = None,
    ) -> NetworkEvidenceEnvelope:
        return await self._observe("lab_probe", {"probe_id": probe_id}, correlation_id)

    async def lab_app_probe(
        self, user_id: str, app_id: str, correlation_id: str | None = None,
    ) -> NetworkEvidenceEnvelope:
        return await self._observe(
            "lab_app_probe", {"user_id": user_id, "app_id": app_id}, correlation_id,
        )

    async def network_get_app_enforcement(
        self, user_id: str, app_id: str, correlation_id: str | None = None,
    ) -> NetworkEvidenceEnvelope:
        return await self._observe(
            "network_get_app_enforcement", {"user_id": user_id, "app_id": app_id}, correlation_id,
        )

    async def list_users(
        self, dept: str | None = None, correlation_id: str | None = None,
    ) -> NetworkEvidenceEnvelope:
        return await self._observe("list_users", {"dept": dept}, correlation_id)

    async def get_user_access(
        self, user_id: str, correlation_id: str | None = None,
    ) -> NetworkEvidenceEnvelope:
        return await self._observe("get_user_access", {"user_id": user_id}, correlation_id)

    async def check_nac_policy(
        self, user_id: str, correlation_id: str | None = None,
    ) -> NetworkEvidenceEnvelope:
        return await self._observe("check_nac_policy", {"user_id": user_id}, correlation_id)

    async def dc_list_apps(
        self, tier: str | None = None, correlation_id: str | None = None,
    ) -> NetworkEvidenceEnvelope:
        return await self._observe("dc_list_apps", {"tier": tier}, correlation_id)

    async def dc_get_app_acl(
        self, app_id: str, correlation_id: str | None = None,
    ) -> NetworkEvidenceEnvelope:
        return await self._observe("dc_get_app_acl", {"app_id": app_id}, correlation_id)

    async def dc_check_user_app_access(
        self, user_id: str, app_id: str, correlation_id: str | None = None,
    ) -> NetworkEvidenceEnvelope:
        return await self._observe(
            "dc_check_user_app_access", {"user_id": user_id, "app_id": app_id}, correlation_id,
        )

    async def lab_get_fabric_state(
        self, correlation_id: str | None = None,
    ) -> NetworkEvidenceEnvelope:
        return await self._observe("lab_get_fabric_state", {}, correlation_id)

    async def lab_get_access_vlan(
        self, device_id: str, interface: str, correlation_id: str | None = None,
    ) -> NetworkEvidenceEnvelope:
        return await self._observe("lab_get_access_vlan", {
            "device_id": device_id, "interface": interface,
        }, correlation_id)

    async def lab_get_vxlan_state(
        self, device_id: str, correlation_id: str | None = None,
    ) -> NetworkEvidenceEnvelope:
        return await self._observe("lab_get_vxlan_state", {"device_id": device_id}, correlation_id)

    async def lab_get_bgp_evpn_summary(
        self, device_id: str, correlation_id: str | None = None,
    ) -> NetworkEvidenceEnvelope:
        return await self._observe(
            "lab_get_bgp_evpn_summary", {"device_id": device_id}, correlation_id,
        )

    async def lab_get_evpn_routes(
        self, device_id: str, route_type: int | None = None,
        correlation_id: str | None = None,
    ) -> NetworkEvidenceEnvelope:
        return await self._observe("lab_get_evpn_routes", {
            "device_id": device_id, "route_type": route_type,
        }, correlation_id)


def _metadata(profile: str, provider: ContainerlabProvider) -> dict[str, dict[str, Any]]:
    callables = LabToolAdapter(provider).callables(profile)
    values = {
        name: dict(PRAGMATIC_METADATA[name])
        for name in callables if name in PRAGMATIC_METADATA
    }
    values.update(lab_tool_metadata(
        profile,
        access_enabled=bool(provider.manifest.users and provider.manifest.applications),
        topology_enabled=bool(provider.manifest.links),
        fabric_enabled=provider.manifest.fabric is not None,
    ))
    return values


def _tool_meta(tool_name: str) -> dict[str, Any]:
    capability = REGISTRY.for_tool(tool_name)
    if capability is None:
        raise RuntimeError(f"missing capability contract for observer tool {tool_name!r}")
    sensitive = {
        "get_device_config", "get_syslog", "run_command", "multi_device_check",
        "list_users", "get_user_access", "check_nac_policy",
        "dc_get_app_acl", "dc_check_user_app_access",
    }
    return {
        "netopyu": {
            "domain": "network",
            "provider_role": capability.provider_role,
            "provider_kind": "network-observer-mcp",
            "action_type": capability.action_type,
            "requires_approval": False,
            "capability_id": capability.capability_id,
            "capability_version": capability.capability_version,
            "result_contract": EVIDENCE_CONTRACT,
            "sensitivity": "restricted" if tool_name in sensitive else "internal",
            "required_roles": ["operations-reader"],
            "freshness_limit_seconds": 300,
        }
    }


def build_server(
    profile: str,
    manifest_path: str | Path,
    *,
    command_timeout: float = 30.0,
    provider: ContainerlabProvider | None = None,
) -> MCPServer:
    if profile not in {"lan", "dc"}:
        raise ValueError("network observer profile must be lan or dc")
    selected = provider or ContainerlabProvider(
        load_manifest(manifest_path), command_timeout=command_timeout,
    )
    tools = ObserverTools(LabToolAdapter(selected), profile)
    metadata = _metadata(profile, selected)

    @asynccontextmanager
    async def lifespan(_server: MCPServer) -> AsyncIterator[ContainerlabProvider]:
        try:
            yield selected
        finally:
            close = getattr(selected, "close", None)
            if close is not None:
                result = close()
                if hasattr(result, "__await__"):
                    await result

    server = MCPServer(
        SERVER_NAME,
        version=SERVER_VERSION,
        description="Read-only, capability-bound network observation provider",
        lifespan=lifespan,
    )
    for tool_name in REGISTRY.observer_tools():
        if tool_name not in tools.callables:
            continue
        method = getattr(tools, tool_name)
        server.add_tool(
            method,
            name=tool_name,
            description=str(metadata.get(tool_name, {}).get("description") or tool_name),
            meta=_tool_meta(tool_name),
            structured_output=True,
        )
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("lan", "dc"), required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--command-timeout", type=float, default=30.0)
    parser.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()
    server = build_server(
        args.profile, args.manifest, command_timeout=args.command_timeout,
    )
    if args.transport == "stdio":
        server.run("stdio")
    else:
        server.run(
            "streamable-http", host=args.host, port=args.port,
            stateless_http=True, json_response=True,
        )


if __name__ == "__main__":
    main()
