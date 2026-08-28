"""Durable, identity-pinned Network Actor MCP over reviewed Containerlab effects."""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from mcp.server.mcpserver import MCPServer

from network_lab import ContainerlabProvider, load_manifest
from network_lab.tools import LabToolAdapter, lab_tool_metadata
from network_runtime.policies import reviewed_contracts
from network_runtime.provider_contracts import REGISTRY
from tools.pragmatic.registry import TOOLS as PRAGMATIC_METADATA

from .actor import DurableNetworkActor, EffectContext
from .actor_store import ActorStore
from .models import ActorEffectResult, ActorFinalizeResult


SERVER_NAME = "netopyu.network-actor"
SERVER_VERSION = "1.0.0"
INTERNAL_CONTEXT = (
    "operation_id", "plan_hash", "intent_hash", "approved_preflight", "effect_phase",
)
INTERNAL_TOOLS = {
    "restore_device_config", "fabric_restore_access_vlan",
    "network_restore_app_enforcement", "network_actor_finalize",
}
_CONTRACT_IDS = {
    name: contract.contract_id for name, contract in reviewed_contracts().items()
}
_CONTRACT_IDS.update({
    "restore_device_config": "device-config-restore-internal-v1",
    "fabric_restore_access_vlan": "fabric-access-vlan-restore-internal-v1",
    "network_restore_app_enforcement": "network-app-enforcement-restore-internal-v1",
    "network_actor_finalize": "network-actor-finalize-internal-v1",
})


def _context(
    operation_id: str,
    plan_hash: str,
    intent_hash: str,
    approved_preflight: dict[str, Any],
    effect_phase: str,
) -> EffectContext:
    return EffectContext.create(
        operation_id=operation_id,
        plan_hash=plan_hash,
        intent_hash=intent_hash,
        approved_preflight=approved_preflight,
        effect_phase=effect_phase,
    )


class ActorTools:
    def __init__(self, actor: DurableNetworkActor) -> None:
        self.actor = actor

    async def _execute(
        self, tool_name: str, arguments: dict[str, Any], context: EffectContext,
    ) -> ActorEffectResult:
        return ActorEffectResult.model_validate(
            await self.actor.execute(tool_name, arguments, context)
        )

    async def edit_device_config(
        self,
        device_id: str,
        config_lines: list[str],
        reason: str,
        operation_id: str,
        plan_hash: str,
        intent_hash: str,
        approved_preflight: dict[str, Any],
        effect_phase: str,
        section: str | None = None,
        verification_probe_id: str | None = None,
    ) -> ActorEffectResult:
        arguments: dict[str, Any] = {
            "device_id": device_id, "config_lines": config_lines, "reason": reason,
        }
        if section is not None:
            arguments["section"] = section
        if verification_probe_id is not None:
            arguments["verification_probe_id"] = verification_probe_id
        return await self._execute(
            "edit_device_config", arguments,
            _context(operation_id, plan_hash, intent_hash, approved_preflight, effect_phase),
        )

    async def restore_device_config(
        self,
        device_id: str,
        operation_id: str,
        plan_hash: str,
        intent_hash: str,
        approved_preflight: dict[str, Any],
        effect_phase: str,
    ) -> ActorEffectResult:
        return await self._execute(
            "restore_device_config", {"device_id": device_id},
            _context(operation_id, plan_hash, intent_hash, approved_preflight, effect_phase),
        )

    async def grant_user_access(
        self,
        user_id: str,
        reason: str,
        operation_id: str,
        plan_hash: str,
        intent_hash: str,
        approved_preflight: dict[str, Any],
        effect_phase: str,
    ) -> ActorEffectResult:
        return await self._execute(
            "grant_user_access", {"user_id": user_id, "reason": reason},
            _context(operation_id, plan_hash, intent_hash, approved_preflight, effect_phase),
        )

    async def revoke_user_access(
        self,
        user_id: str,
        reason: str,
        operation_id: str,
        plan_hash: str,
        intent_hash: str,
        approved_preflight: dict[str, Any],
        effect_phase: str,
    ) -> ActorEffectResult:
        return await self._execute(
            "revoke_user_access", {"user_id": user_id, "reason": reason},
            _context(operation_id, plan_hash, intent_hash, approved_preflight, effect_phase),
        )

    async def dc_grant_app_access(
        self,
        user_id: str,
        app_id: str,
        reason: str,
        operation_id: str,
        plan_hash: str,
        intent_hash: str,
        approved_preflight: dict[str, Any],
        effect_phase: str,
        role: str | None = None,
    ) -> ActorEffectResult:
        arguments: dict[str, Any] = {
            "user_id": user_id, "app_id": app_id, "reason": reason,
        }
        if role is not None:
            arguments["role"] = role
        return await self._execute(
            "dc_grant_app_access", arguments,
            _context(operation_id, plan_hash, intent_hash, approved_preflight, effect_phase),
        )

    async def dc_revoke_app_access(
        self,
        user_id: str,
        app_id: str,
        reason: str,
        operation_id: str,
        plan_hash: str,
        intent_hash: str,
        approved_preflight: dict[str, Any],
        effect_phase: str,
    ) -> ActorEffectResult:
        return await self._execute(
            "dc_revoke_app_access",
            {"user_id": user_id, "app_id": app_id, "reason": reason},
            _context(operation_id, plan_hash, intent_hash, approved_preflight, effect_phase),
        )

    async def fabric_set_access_vlan(
        self,
        device_id: str,
        interface: str,
        vlan_id: int,
        reason: str,
        operation_id: str,
        plan_hash: str,
        intent_hash: str,
        approved_preflight: dict[str, Any],
        effect_phase: str,
        verification_probe_id: str | None = None,
    ) -> ActorEffectResult:
        arguments: dict[str, Any] = {
            "device_id": device_id, "interface": interface,
            "vlan_id": vlan_id, "reason": reason,
        }
        if verification_probe_id is not None:
            arguments["verification_probe_id"] = verification_probe_id
        return await self._execute(
            "fabric_set_access_vlan", arguments,
            _context(operation_id, plan_hash, intent_hash, approved_preflight, effect_phase),
        )

    async def fabric_restore_access_vlan(
        self,
        device_id: str,
        interface: str,
        operation_id: str,
        plan_hash: str,
        intent_hash: str,
        approved_preflight: dict[str, Any],
        effect_phase: str,
    ) -> ActorEffectResult:
        return await self._execute(
            "fabric_restore_access_vlan",
            {"device_id": device_id, "interface": interface},
            _context(operation_id, plan_hash, intent_hash, approved_preflight, effect_phase),
        )

    async def network_apply_app_enforcement(
        self,
        user_id: str,
        app_id: str,
        change_id: str,
        reason: str,
        operation_id: str,
        plan_hash: str,
        intent_hash: str,
        approved_preflight: dict[str, Any],
        effect_phase: str,
    ) -> ActorEffectResult:
        return await self._execute(
            "network_apply_app_enforcement",
            {"user_id": user_id, "app_id": app_id, "change_id": change_id, "reason": reason},
            _context(operation_id, plan_hash, intent_hash, approved_preflight, effect_phase),
        )

    async def network_revoke_app_enforcement(
        self,
        user_id: str,
        app_id: str,
        change_id: str,
        reason: str,
        operation_id: str,
        plan_hash: str,
        intent_hash: str,
        approved_preflight: dict[str, Any],
        effect_phase: str,
    ) -> ActorEffectResult:
        return await self._execute(
            "network_revoke_app_enforcement",
            {"user_id": user_id, "app_id": app_id, "change_id": change_id, "reason": reason},
            _context(operation_id, plan_hash, intent_hash, approved_preflight, effect_phase),
        )

    async def network_restore_app_enforcement(
        self,
        user_id: str,
        app_id: str,
        allowed: bool,
        change_id: str,
        reason: str,
        operation_id: str,
        plan_hash: str,
        intent_hash: str,
        approved_preflight: dict[str, Any],
        effect_phase: str,
    ) -> ActorEffectResult:
        return await self._execute(
            "network_restore_app_enforcement",
            {
                "user_id": user_id, "app_id": app_id, "allowed": allowed,
                "change_id": change_id, "reason": reason,
            },
            _context(operation_id, plan_hash, intent_hash, approved_preflight, effect_phase),
        )

    async def network_actor_finalize(
        self, operation_id: str, plan_hash: str, terminal_state: str,
    ) -> ActorFinalizeResult:
        return ActorFinalizeResult.model_validate(
            await self.actor.finalize(operation_id, plan_hash, terminal_state)
        )


def _metadata(profile: str, provider: ContainerlabProvider) -> dict[str, dict[str, Any]]:
    callables: dict[str, Any] = {}
    for item in (("lan", "dc") if profile == "all" else (profile,)):
        callables.update(LabToolAdapter(provider).callables(item))
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


def _tool_meta(tool_name: str, profiles: tuple[str, ...]) -> dict[str, Any]:
    capability = REGISTRY.for_tool(tool_name)
    if capability is None or capability.provider_role != "actor":
        raise RuntimeError(f"missing Actor capability contract for {tool_name!r}")
    return {
        "netopyu": {
            "domain": "network",
            "provider_role": "actor",
            "provider_kind": "network-actor-mcp",
            "action_type": capability.action_type,
            "requires_approval": True,
            "capability_id": capability.capability_id,
            "capability_version": capability.capability_version,
            "contract_id": _CONTRACT_IDS[tool_name],
            "result_contract": "structured-content-required-v1",
            "profiles": list(profiles),
            "internal_only": tool_name in INTERNAL_TOOLS,
            "internal_parameters": (
                ["operation_id", "plan_hash", "terminal_state"]
                if tool_name == "network_actor_finalize" else list(INTERNAL_CONTEXT)
            ),
        }
    }


def build_server(
    profile: str,
    manifest_path: str | Path,
    *,
    store_path: str | Path | None = None,
    command_timeout: float = 30.0,
    lease_seconds: float = 120.0,
    provider: ContainerlabProvider | None = None,
) -> MCPServer:
    if profile not in {"lan", "dc", "all"}:
        raise ValueError("network actor profile must be lan, dc, or all")
    selected = provider or ContainerlabProvider(
        load_manifest(manifest_path), command_timeout=command_timeout,
    )
    database = Path(store_path) if store_path else selected.manifest.state_directory / "network_actor.sqlite"
    store = ActorStore(database, lease_seconds=lease_seconds)
    actor = DurableNetworkActor(selected, store)
    tools = ActorTools(actor)
    projected = {
        item: set(LabToolAdapter(selected).callables(item))
        for item in (("lan", "dc") if profile == "all" else (profile,))
    }
    available = set().union(*projected.values())
    metadata = _metadata(profile, selected)

    @asynccontextmanager
    async def lifespan(_server: MCPServer) -> AsyncIterator[DurableNetworkActor]:
        try:
            await actor.reconcile_startup()
            yield actor
        finally:
            store.close()
            close = getattr(selected, "close", None)
            if close is not None:
                result = close()
                if hasattr(result, "__await__"):
                    await result

    server = MCPServer(
        SERVER_NAME,
        version=SERVER_VERSION,
        description="Durable capability-bound Network Actor",
        lifespan=lifespan,
    )
    actor_tools = [
        item.tool_name for item in REGISTRY.contracts()
        if item.provider_role == "actor" and item.tool_name in available
    ]
    actor_tools.append("network_actor_finalize")
    for tool_name in actor_tools:
        method = getattr(tools, tool_name)
        server.add_tool(
            method,
            name=tool_name,
            description=str(metadata.get(tool_name, {}).get("description") or tool_name),
            meta=_tool_meta(
                tool_name,
                (
                    ("lan", "dc")
                    if tool_name == "network_actor_finalize"
                    else tuple(item for item, names in projected.items() if tool_name in names)
                ),
            ),
            structured_output=True,
        )
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("lan", "dc", "all"), required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--store")
    parser.add_argument("--command-timeout", type=float, default=30.0)
    parser.add_argument("--lease-seconds", type=float, default=120.0)
    parser.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8092)
    args = parser.parse_args()
    server = build_server(
        args.profile,
        args.manifest,
        store_path=args.store,
        command_timeout=args.command_timeout,
        lease_seconds=args.lease_seconds,
    )
    if args.transport == "stdio":
        server.run("stdio")
    else:
        server.run(
            "streamable-http", host=args.host, port=args.port,
            stateless_http=False, json_response=True,
        )


if __name__ == "__main__":
    main()
