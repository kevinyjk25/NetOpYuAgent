"""Backend lifecycle for harness-exposed NetOpYu tools.

Mock mode keeps the profile-local callables used by the existing demo. Pragmatic
mode builds the real device/MCP/OpenAPI router for each short-lived bridge
process and deliberately never falls back to mock integrations.
"""

from __future__ import annotations

import json
import os
from importlib.util import find_spec
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable


ToolCallable = Callable[[dict[str, Any]], Awaitable[Any]]


@dataclass
class BackendSession:
    mode: str
    profile_id: str
    callables: dict[str, ToolCallable]
    metadata: dict[str, dict[str, Any]]
    sources: dict[str, str]
    report: dict[str, Any]
    _mcp_clients: list[Any] = field(default_factory=list)
    _openapi_clients: list[Any] = field(default_factory=list)
    _lab_providers: list[Any] = field(default_factory=list)
    _tool_store: Any | None = None
    _provider_admission_gate: Any | None = None

    def describe_capability(self, tool_name: str) -> Any:
        """Return the canonical contract without exposing transport details."""
        from network_runtime.capabilities import CapabilityContract

        metadata = self.metadata.get(tool_name)
        if metadata is None or tool_name not in self.callables:
            raise KeyError(f"unknown provider capability {tool_name!r}")
        contract = CapabilityContract.from_metadata(
            tool_name,
            metadata,
            source=self.sources.get(tool_name, "unknown"),
        )
        source = self.sources.get(tool_name, "unknown")
        if self._provider_admission_gate is not None and (
            source.startswith("mcp:") or source.startswith("openapi")
        ):
            if source.startswith("openapi") and contract.provider_identity == "openapi-unpinned":
                from network_runtime.capabilities import CapabilityAdmissionError

                raise CapabilityAdmissionError(
                    "enforced OpenAPI Provider admission requires a deployment-owned provider_identity"
                )
            evidence = self._provider_admission_gate.admit(
                contract,
                provider_id=str(metadata.get("release_provider_id") or ""),
                result_contract=str(metadata.get("result_contract") or ""),
            )
            metadata.update({
                "provider_release_digest": evidence.release_digest,
                "provider_manifest_digest": evidence.manifest_digest,
                "provider_qualification_digest": evidence.qualification_digest,
                "provider_deployment_digest": evidence.deployment_digest,
                "release_provider_id": evidence.provider_id,
                "release_provider_version": evidence.provider_version,
                "provider_l0_contract_hashes": list(evidence.l0_contract_hashes),
            })
        return contract

    async def invoke_observation(
        self, tool_name: str, arguments: dict[str, Any],
    ) -> Any:
        """Invoke one observation through the protocol-neutral gateway."""
        contract = self.describe_capability(tool_name)
        if contract.kind.value != "observation":
            raise RuntimeError(f"capability {contract.capability_id!r} is not an observation")
        return await self.callables[tool_name](dict(arguments))

    async def invoke_effect(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        plan: Any,
        phase: str,
    ) -> Any:
        """Invoke one effect while injecting provider-internal immutable context."""
        tool = self.callables[tool_name]
        metadata = self.metadata.get(tool_name, {})
        internal = tuple(metadata.get("internal_parameters") or ())
        if not internal:
            return await tool(dict(arguments))
        if not plan.preflight:
            raise RuntimeError("provider-internal effect context requires approved preflight")
        context = {
            "operation_id": plan.plan_id,
            "plan_hash": plan.plan_hash,
            "intent_hash": plan.intent_hash,
            "approved_preflight": plan.preflight[0].value,
            "effect_phase": phase,
        }
        missing = sorted(set(internal) - set(context))
        if missing:
            raise RuntimeError(f"unsupported provider-internal effect fields: {missing}")
        overlap = sorted(set(arguments) & set(internal))
        if overlap:
            raise RuntimeError(f"model arguments attempted to set internal fields: {overlap}")
        return await tool({**dict(arguments), **{name: context[name] for name in internal}})

    async def finalize_effect(self, plan: Any, terminal_state: str) -> Any | None:
        metadata = self.metadata.get(plan.tool_name, {})
        if metadata.get("provider_kind") != "network-actor-mcp":
            return None
        finalizer = self.callables.get("network_actor_finalize")
        if finalizer is None:
            raise RuntimeError("Network Actor provider has no durable finalizer")
        return await finalizer({
            "operation_id": plan.plan_id,
            "plan_hash": plan.plan_hash,
            "terminal_state": terminal_state,
        })

    async def close(self) -> None:
        errors: list[str] = []
        for provider in reversed(self._lab_providers):
            try:
                close = getattr(provider, "close", None)
                if close is not None:
                    value = close()
                    if hasattr(value, "__await__"):
                        await value
            except Exception as error:  # pragma: no cover - defensive cleanup
                errors.append(f"lab cleanup: {error}")
        for client in reversed(self._openapi_clients):
            try:
                await client.unload()
            except Exception as error:  # pragma: no cover - defensive cleanup
                errors.append(f"OpenAPI cleanup: {error}")
        for client in reversed(self._mcp_clients):
            try:
                await client.disconnect_all()
            except Exception as error:  # pragma: no cover - defensive cleanup
                errors.append(f"MCP cleanup: {error}")
        if self._tool_store is not None:
            try:
                self._tool_store.close()
            except Exception as error:  # pragma: no cover - defensive cleanup
                errors.append(f"tool-result store cleanup: {error}")
        if errors:
            self.report.setdefault("cleanup_warnings", []).extend(errors)


def _attach_common_tools(
    callables: dict[str, ToolCallable],
    metadata: dict[str, dict[str, Any]],
    sources: dict[str, str],
) -> Any:
    """Bind the shared large-result paging tools to the durable result store."""
    from runtime import ToolResultStore
    from tools import make_read_stored_result_tool
    from tools.builtin.registry import TOOLS as BUILTIN_TOOLS

    configured = (
        os.environ.get("NETOPYU_TOOL_RESULT_STORE")
        or os.environ.get("NETOPYU_DSH_TOOL_RESULT_STORE")
    )
    database = (
        Path(configured).expanduser()
        if configured
        else Path("data/tool_results.sqlite")
    )
    database.parent.mkdir(parents=True, exist_ok=True)
    store = ToolResultStore(db_path=str(database))
    read_result, process_chunks = make_read_stored_result_tool(store)
    callables["read_stored_result"] = read_result
    callables["process_stored_chunks"] = process_chunks
    metadata.update(BUILTIN_TOOLS)
    sources["read_stored_result"] = "netopyu-runtime"
    sources["process_stored_chunks"] = "netopyu-runtime"
    return store


def resolve_backend_mode() -> str:
    """Resolve the shared harness backend without silently accepting a typo."""
    configured = (
        os.environ.get("NETOPYU_BACKEND")
        or os.environ.get("NETOPYU_DSH_BACKEND")
        or os.environ.get("MODE")
    )
    if configured is None:
        from config import load

        configured = load(os.environ.get("NETOPYU_CONFIG_PATH", "config.yaml")).mode
    mode = configured.strip().lower()
    if mode not in {"mock", "pragmatic"}:
        raise ValueError(
            f"unsupported NetOpYu backend {mode!r}; expected mock or pragmatic"
        )
    return mode


def _load_app_config() -> Any:
    from config import load

    path = os.environ.get("NETOPYU_CONFIG_PATH", "config.yaml")
    return load(path)


def _parse_mcp_config(value: str) -> dict[str, dict[str, Any]]:
    if not value.strip():
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        payload = json.loads(Path(value).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or any(not isinstance(v, dict) for v in payload.values()):
        raise ValueError("MCP_CONFIG_JSON must be an object mapping server names to configs")
    return payload


def _validate_auth(label: str, auth: dict[str, Any]) -> None:
    auth_type = str(auth.get("type", "")).lower()
    if not auth_type:
        return
    if auth_type == "bearer":
        env_name = str(auth.get("token_env", ""))
        if not auth.get("token") and (not env_name or not os.environ.get(env_name)):
            raise ValueError(f"{label} bearer credential is missing ({env_name or 'token_env not set'})")
    elif auth_type == "api_key":
        env_name = str(auth.get("key_env", ""))
        if not auth.get("key") and (not env_name or not os.environ.get(env_name)):
            raise ValueError(f"{label} API key is missing ({env_name or 'key_env not set'})")
    elif auth_type == "basic":
        env_name = str(auth.get("password_env", ""))
        if not auth.get("password") and (not env_name or not os.environ.get(env_name)):
            raise ValueError(f"{label} basic-auth password is missing ({env_name or 'password_env not set'})")


def _mcp_metadata(spec: Any) -> dict[str, Any]:
    schema = spec.parameters if isinstance(spec.parameters, dict) else {}
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    declared = (
        spec.meta.get("netopyu", {})
        if isinstance(getattr(spec, "meta", None), dict) else {}
    )
    prefix_action = _external_action_type(spec.name)
    declared_action = str(declared.get("action_type") or "")
    internal_parameters = declared.get("internal_parameters") or []
    if not isinstance(internal_parameters, list) or any(
        not isinstance(name, str) for name in internal_parameters
    ):
        raise ValueError(f"MCP tool {spec.name!r} has invalid internal_parameters metadata")
    declared_profiles = declared.get("profiles") or []
    if (
        not isinstance(declared_profiles, list)
        or any(item not in {"lan", "dc", "wan"} for item in declared_profiles)
    ):
        raise ValueError(f"MCP tool {spec.name!r} has invalid profiles metadata")
    declared_scope_fields = declared.get("scope_fields")
    if declared_scope_fields is not None and (
        not isinstance(declared_scope_fields, list)
        or any(not isinstance(item, str) for item in declared_scope_fields)
    ):
        raise ValueError(f"MCP tool {spec.name!r} has invalid scope_fields metadata")
    properties = {
        name: value for name, value in properties.items()
        if name not in internal_parameters
    }
    required = [name for name in required if name not in internal_parameters]
    configured_domain = str(getattr(spec, "configured_domain", "external") or "external")
    if configured_domain == "network" and declared.get("domain") != "network":
        raise ValueError(
            f"configured network MCP tool {spec.name!r} omitted domain=network metadata"
        )
    if configured_domain == "network" or declared.get("domain") == "network":
        from network_runtime.provider_contracts import validate_declaration

        validate_declaration(spec.name, declared)
    if spec.trusted_for_writes:
        action_type = declared_action or prefix_action
    elif (
        spec.identity_pinned
        and declared.get("domain") == "network"
        and declared.get("provider_role") == "observer"
        and declared_action == "read_only"
    ):
        # Pinned observer identity + reviewed capability metadata is stronger
        # than a tool-name prefix heuristic.  This permits compatibility names
        # such as lab_trace_path without letting an arbitrary MCP self-declare
        # a mutating tool as read-only.
        action_type = "read_only"
    elif spec.identity_pinned and declared_action == "read_only":
        action_type = "read_only"
    else:
        # An untrusted server cannot make a write look read-only by metadata or
        # by choosing a misleading name. Both signals must agree on read-only.
        action_type = (
            "read_only"
            if declared_action == "read_only" and prefix_action == "read_only"
            else "destructive"
        )
    return {
        "description": spec.description or spec.name,
        "parameters": properties if isinstance(properties, dict) else {},
        "required": required if isinstance(required, list) else [],
        "output_schema": spec.output_schema,
        "action_type": action_type,
        "hitl": action_type != "read_only",
        "tags": ["mcp", spec.server_name, str(declared.get("domain") or "external")],
        "provider_identity": (
            f"mcp:{spec.server_name}:{spec.server_identity}@{spec.server_version}"
        ),
        "input_schema_digest": spec.input_schema_digest,
        "output_schema_digest": spec.output_schema_digest,
        "declared_contract_id": declared.get("contract_id"),
        "result_contract": declared.get("result_contract"),
        "trusted_for_writes": bool(spec.trusted_for_writes),
        "service_domain": declared.get("service_domain"),
        "domain": declared.get("domain") or configured_domain,
        "sensitivity": declared.get("sensitivity") or "internal",
        "required_roles": list(declared.get("required_roles") or []),
        "scope_fields": declared_scope_fields,
        "freshness_limit_seconds": int(declared.get("freshness_limit_seconds") or 300),
        "internal_only": bool(declared.get("internal_only", False)),
        "capability_id": declared.get("capability_id"),
        "capability_version": declared.get("capability_version"),
        "provider_role": declared.get("provider_role"),
        "provider_kind": declared.get("provider_kind"),
        "profiles": list(declared_profiles),
        "internal_parameters": list(internal_parameters),
        # Provider-id selection is deployment-owned configuration, never a
        # self-assertion from model-visible MCP tool metadata.
        "release_provider_id": str(getattr(spec, "release_provider_id", "") or ""),
    }


def _openapi_metadata(
    operation: Any,
    *,
    release_provider_id: str = "",
    provider_identity: str = "",
) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    required: list[str] = []
    for parameter in operation.parameters:
        schema = dict(parameter.schema or {})
        schema["description"] = parameter.description or parameter.name
        parameters[parameter.name] = schema
        if parameter.required:
            required.append(parameter.name)
    body = operation.request_body_schema or {}
    for name, schema in (body.get("properties", {}) or {}).items():
        parameters[name] = dict(schema or {})
    required.extend(name for name in (body.get("required", []) or []) if name not in required)
    read_only = operation.method.upper() in {"GET", "HEAD", "OPTIONS"}
    return {
        "description": operation.description or operation.summary or operation.tool_name(),
        "parameters": parameters,
        "required": required,
        "action_type": "read_only" if read_only else "destructive",
        "hitl": not read_only,
        "tags": ["openapi", operation.method.lower(), *operation.tags],
        "domain": "external",
        "sensitivity": "internal",
        "release_provider_id": release_provider_id,
        "provider_identity": provider_identity or "openapi-unpinned",
        "result_contract": "openapi-response-v1",
    }


_READ_ONLY_PREFIXES = (
    "get_", "list_", "show_", "query_", "search_", "check_", "validate_",
    "diff_", "describe_", "read_", "fetch_", "lookup_", "health_",
)


def _external_action_type(name: str) -> str:
    """Conservatively classify MCP tools that carry no standard risk field."""
    return "read_only" if name.lower().startswith(_READ_ONLY_PREFIXES) else "destructive"


async def open_backend(profile_id: str = "lan") -> BackendSession:
    mode = resolve_backend_mode()
    # Provider release/supply-chain qualification is frozen future engineering.
    # Keep it outside the prototype dependency graph unless a caller opts in
    # explicitly. The default EnsuredSkill path has no import-time dependency
    # on that productization surface.
    admission_mode = os.environ.get(
        "NETOPYU_PROVIDER_ADMISSION", "disabled",
    ).strip().lower()
    provider_admission = None
    if admission_mode not in {"", "disabled", "off", "0"}:
        from network_runtime.provider_release import provider_admission_from_environment

        provider_admission = provider_admission_from_environment()
    if mode == "mock":
        from profiles import load_profile

        profile = load_profile(profile_id)
        sources = {name: "profile-mock" for name in profile.tool_callables}
        callables = dict(profile.tool_callables)
        metadata = dict(profile.tool_metadata)
        tool_store = _attach_common_tools(callables, metadata, sources)
        return BackendSession(
            mode=mode,
            profile_id=profile.profile_id,
            callables=callables,
            metadata=metadata,
            sources=sources,
            report={
                "mode": mode,
                "ready": True,
                "profile": profile.profile_id,
                "sources": {
                    "profile-mock": len(profile.tool_callables),
                    "netopyu-runtime": 2,
                },
                "warnings": ["Network results are simulated; MODE=pragmatic is required for real systems."],
            },
            _tool_store=tool_store,
            _provider_admission_gate=provider_admission,
        )

    cfg = _load_app_config()
    from integrations.clients.mcp_client import MCPClient
    from integrations.clients.openapi_client import OpenAPIClient
    from integrations.router.tool_router import ToolRouter
    from tools.loader import ToolLoader
    from tools.pragmatic_tools import register_devices, reset_devices

    valid_devices = [
        device for device in cfg.pragmatic.device_inventory
        if device.id and device.host and device.username and device.password
    ]
    invalid_devices = [
        device.id or "<missing-id>" for device in cfg.pragmatic.device_inventory
        if device not in valid_devices
    ]
    drivers = {
        "netmiko": find_spec("netmiko") is not None,
        "napalm": find_spec("napalm") is not None,
        "nornir": find_spec("nornir") is not None,
    }
    if valid_devices and not drivers["netmiko"]:
        raise RuntimeError("pragmatic device inventory requires the netmiko package")
    if cfg.pragmatic.lab.enabled and cfg.pragmatic.device_inventory:
        raise ValueError(
            "pragmatic.lab and pragmatic.device_inventory cannot be enabled together; "
            "separate lab and real-device runtimes to preserve target identity"
        )
    reset_devices()
    register_devices(valid_devices)

    loader = ToolLoader(mode="pragmatic", profile=profile_id)
    lab_providers: list[Any] = []
    if cfg.pragmatic.lab.enabled:
        if cfg.pragmatic.lab.provider != "containerlab":
            raise ValueError("the local lab supports only pragmatic.lab.provider=containerlab")
        if not 1 <= cfg.pragmatic.lab.command_timeout <= 300:
            raise ValueError("pragmatic.lab.command_timeout must be between 1 and 300")
        from network_lab import ContainerlabProvider, load_manifest
        from network_lab.tools import LabToolAdapter, lab_tool_metadata
        from tools.pragmatic.registry import TOOLS as PRAGMATIC_METADATA

        manifest = load_manifest(cfg.pragmatic.lab.manifest)
        provider = ContainerlabProvider(
            manifest, command_timeout=cfg.pragmatic.lab.command_timeout,
        )
        adapter = LabToolAdapter(provider)
        local_callables = adapter.callables(profile_id)
        metadata = {
            name: dict(PRAGMATIC_METADATA[name])
            for name in local_callables if name in PRAGMATIC_METADATA
        }
        edit_metadata = metadata.get("edit_device_config")
        if edit_metadata is not None:
            edit_metadata["parameters"] = {
                **dict(edit_metadata.get("parameters") or {}),
                "verification_probe_id": (
                    "Optional exact manifest probe that must pass after the configuration write"
                ),
            }
        metadata.update(lab_tool_metadata(
            profile_id,
            access_enabled=bool(manifest.users and manifest.applications),
            topology_enabled=bool(manifest.links),
            fabric_enabled=manifest.fabric is not None,
        ))
        from network_runtime.provider_contracts import enrich_metadata

        metadata = {
            name: enrich_metadata(name, value, provider_kind="network-lab")
            for name, value in metadata.items()
        }
        lab_providers.append(provider)
    else:
        local_callables = loader.build_callables()
        metadata = loader.build_metadata()
    router = ToolRouter(default_timeout=float(
        os.environ.get("NETOPYU_TOOL_TIMEOUT")
        or os.environ.get("NETOPYU_DSH_TOOL_TIMEOUT", "90")
    ))
    mcp_clients: list[Any] = []
    openapi_clients: list[Any] = []

    mcp_config = _parse_mcp_config(cfg.tools.mcp.config_json)
    for server in cfg.pragmatic.mcp_servers:
        mcp_config[server.name] = {
            "transport": server.transport,
            "url": server.url,
            "command": server.command,
            "auth": server.auth,
            "env": server.env,
            "cwd": server.cwd or None,
            "domain": server.domain,
            "trusted_for_writes": server.trusted_for_writes,
            "expected_server_name": server.expected_server_name,
            "expected_server_version": server.expected_server_version,
            "release_provider_id": server.release_provider_id,
            "timeout": server.timeout,
        }
    mock_servers = [
        name for name, server in mcp_config.items()
        if str(server.get("transport", "")).lower() == "mock"
    ]
    if mock_servers:
        raise ValueError(
            "pragmatic backend refuses MCP transport=mock for servers: "
            + ", ".join(sorted(mock_servers))
        )
    for name, server in mcp_config.items():
        transport = str(server.get("transport", "")).lower()
        if transport in {"http", "streamable-http"}:
            if not server.get("url"):
                raise ValueError(f"MCP HTTP server {name!r} has no url")
        if transport == "stdio" and not server.get("command"):
            raise ValueError(f"MCP stdio server {name!r} has no command")
        if transport not in {"stdio", "http", "streamable-http"}:
            raise ValueError(f"MCP server {name!r} has unsupported transport {transport!r}")
        _validate_auth(f"MCP server {name!r}", server.get("auth", {}) or {})
    if mcp_config:
        if find_spec("mcp") is None:
            raise RuntimeError("configured MCP servers require the official 'mcp>=2,<3' SDK")
        client = MCPClient.from_config(mcp_config)
        await client.connect_all()
        mcp_clients.append(client)
        projected_metadata: dict[str, dict[str, Any]] = {}
        for spec in client.list_tools():
            value = _mcp_metadata(spec)
            profiles = value.get("profiles") or []
            if profiles and profile_id not in profiles:
                continue
            projected_metadata[spec.name] = value
        router.register_mcp(client, allowed_tools=set(projected_metadata))
        metadata.update(projected_metadata)

    openapi_configured = bool(cfg.tools.openapi.spec_url and cfg.tools.openapi.base_url)
    if openapi_configured:
        if find_spec("httpx") is None:
            raise RuntimeError("OpenAPI pragmatic backend requires the httpx package")
        _validate_auth("OpenAPI", {
            "type": cfg.tools.openapi.auth_type,
            "token_env": cfg.tools.openapi.token_env,
            "key_env": cfg.tools.openapi.token_env,
        })
        client = OpenAPIClient.from_url(
            name="netops_api",
            spec_url=cfg.tools.openapi.spec_url,
            base_url=cfg.tools.openapi.base_url,
            auth={
                "type": cfg.tools.openapi.auth_type,
                "token_env": cfg.tools.openapi.token_env,
            },
        )
        await client.load()
        openapi_clients.append(client)
        router.register_openapi(client)
        for operation in client.list_operations():
            metadata[operation.tool_name()] = _openapi_metadata(
                operation,
                release_provider_id=cfg.tools.openapi.release_provider_id,
                provider_identity=cfg.tools.openapi.provider_identity,
            )

    router.register_local(local_callables)
    callables = router.registry
    tool_rows = router.get_tool_list()
    sources = {
        row["name"]: (
            "network-lab" if row["source"] == "local" and cfg.pragmatic.lab.enabled
            else "pragmatic-device" if row["source"] == "local"
            else row["source"]
        )
        for row in tool_rows
    }
    from effect_runtime.reconciliation import METADATA as RECONCILIATION_METADATA
    from effect_runtime.reconciliation import build as build_reconciliation_tools

    reconciliation = build_reconciliation_tools(callables)
    callables.update(reconciliation)
    for name in reconciliation:
        metadata[name] = dict(RECONCILIATION_METADATA[name])
        sources[name] = "effect-runtime"
    tool_store = _attach_common_tools(callables, metadata, sources)
    source_counts = router.tool_count()
    source_counts["netopyu-runtime"] = 2
    if reconciliation:
        source_counts["effect-runtime"] = len(reconciliation)
    if "local" in source_counts:
        source_counts[
            "network-lab" if cfg.pragmatic.lab.enabled else "pragmatic-device"
        ] = source_counts.pop("local")
    configured_sources = (
        len(valid_devices)
        + (1 if cfg.pragmatic.lab.enabled else 0)
        + len(mcp_config)
        + (1 if openapi_configured else 0)
    )
    warnings: list[str] = []
    if not cfg.pragmatic.device_inventory and not cfg.pragmatic.lab.enabled:
        warnings.append("pragmatic.device_inventory is empty")
    if invalid_devices:
        warnings.append(
            "Ignoring device entries missing id/host/username/password: "
            + ", ".join(invalid_devices)
        )
    if configured_sources == 0:
        warnings.append("No real device, MCP, or OpenAPI source is configured")
    if cfg.pragmatic.lab.enabled:
        warnings.append(
            "Local network simulation is enabled; results are not evidence of hardware/RF behavior"
        )

    return BackendSession(
        mode=mode,
        profile_id=profile_id,
        callables=callables,
        metadata={name: metadata[name] for name in callables if name in metadata},
        sources=sources,
        report={
            "mode": mode,
            "ready": configured_sources > 0,
            "profile": profile_id,
            "device_count": len(valid_devices),
            "lab_enabled": cfg.pragmatic.lab.enabled,
            "lab_name": (
                lab_providers[0].manifest.name if lab_providers else None
            ),
            "lab_device_count": (
                len(lab_providers[0].manifest.devices) if lab_providers else 0
            ),
            "invalid_device_count": len(invalid_devices),
            "drivers": drivers,
            "mcp_server_count": len(mcp_config),
            "openapi_configured": openapi_configured,
            "sources": source_counts,
            "warnings": warnings,
        },
        _mcp_clients=mcp_clients,
        _openapi_clients=openapi_clients,
        _lab_providers=lab_providers,
        _tool_store=tool_store,
        _provider_admission_gate=provider_admission,
    )
