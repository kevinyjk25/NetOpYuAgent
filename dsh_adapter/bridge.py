"""Expose NetOpYu profile tools through a small JSON-safe DSH bridge."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from .backend import open_backend


_INTEGER_KEYS = {
    "flows", "grace_period_s", "lines", "minutes", "range_minutes", "top_n",
}
_BOOLEAN_KEYS = {"dry_run", "force", "graceful"}
_ARRAY_KEYS = {"config_lines", "device_ids"}
_JSON_KEYS = {"changes"}


def _parameter_schema(name: str, definition: Any, required: bool) -> dict[str, Any]:
    if isinstance(definition, dict):
        schema = dict(definition)
        description = str(schema.get("description", name))
    else:
        schema = {}
        description = str(definition)
    if "type" in schema:
        pass
    elif name in _INTEGER_KEYS:
        schema: dict[str, Any] = {"type": "integer"}
    elif name in _BOOLEAN_KEYS:
        schema = {"type": "boolean"}
    elif name in _ARRAY_KEYS:
        schema = {"type": "array", "items": {"type": "string"}}
    elif name in _JSON_KEYS:
        schema = {}
    else:
        schema = {"type": "string"}
    schema["description"] = description
    if required:
        schema["required"] = True
    return schema


async def _build_manifest(profile_id: str, *, include_destructive: bool) -> dict[str, Any]:
    backend = await open_backend(profile_id)
    from config import load

    app_config = load(os.environ.get("NETOPYU_CONFIG_PATH", "config.yaml"))
    editable_tools = app_config.tools.editable_hitl_tools
    tools: list[dict[str, Any]] = []
    try:
        for name, metadata in sorted(backend.metadata.items()):
            action_type = str(metadata.get("action_type", "read_only"))
            destructive = bool(metadata.get("hitl")) or action_type != "read_only"
            if destructive and not include_destructive:
                continue
            required = set(metadata.get("required", []))
            parameters = {
                key: _parameter_schema(key, definition, key in required)
                for key, definition in metadata.get("parameters", {}).items()
            }
            editable_parameters = [
                key for key in editable_tools.get(name, []) if key in parameters
            ]
            tools.append({
                "name": name,
                "description": str(metadata.get("description", name)),
                "parameters": parameters,
                "action_type": action_type,
                "requires_approval": destructive,
                "editable_parameters": editable_parameters,
                "source": backend.sources.get(name, "unknown"),
                "tags": list(metadata.get("tags", [])),
            })
        return {
            "profile": backend.profile_id,
            "display_name": "Enterprise LAN Agent" if backend.profile_id == "lan" else backend.profile_id,
            "description": f"NetOpYu {backend.profile_id} tools via {backend.mode} backend",
            "backend": backend.report,
            "tools": tools,
        }
    finally:
        await backend.close()


def build_manifest(profile_id: str = "lan", *, include_destructive: bool = False) -> dict[str, Any]:
    """Return DSH-facing declarations, including dynamically discovered tools."""
    return asyncio.run(_build_manifest(profile_id, include_destructive=include_destructive))


async def backend_report(profile_id: str = "lan") -> dict[str, Any]:
    backend = await open_backend(profile_id)
    try:
        return backend.report
    finally:
        await backend.close()


async def invoke_tool(
    profile_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    allow_destructive: bool | None = None,
) -> str:
    """Invoke one profile tool, retaining a hard gate around mutating operations."""
    backend = await open_backend(profile_id)
    try:
        tool = backend.callables.get(tool_name)
        metadata = backend.metadata.get(tool_name)
        if tool is None or metadata is None:
            raise KeyError(f"unknown tool {tool_name!r} in {backend.mode} backend")

        action_type = str(metadata.get("action_type", "read_only"))
        destructive = bool(metadata.get("hitl")) or action_type != "read_only"
        destructive_allowed = (
            os.environ.get("NETOPYU_DSH_ALLOW_DESTRUCTIVE") == "1"
            if allow_destructive is None
            else allow_destructive
        )
        if destructive and not destructive_allowed:
            raise PermissionError(
                f"{tool_name} is {action_type} and remains disabled until the durable HITL plugin is active"
            )
        result = await tool(arguments)
        rendered = result if isinstance(result, str) else str(result)
        # Preserve the legacy context-budget contract under DSH: large tool
        # payloads become durable references that the two common paging tools
        # can read in later bridge processes. Never re-store paging output.
        if tool_name not in {"read_stored_result", "process_stored_chunks"}:
            rendered = backend._tool_store.store(tool_name, rendered)
        return rendered
    finally:
        await backend.close()
