"""Expose NetOpYu profile tools through a small JSON-safe harness bridge."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from .backend import open_backend, resolve_backend_mode
from effect_runtime import EffectRuntime
from network_runtime.engine import default_journal_path
from network_runtime.l0_skills import REGISTRY as L0_SKILLS
from network_runtime.workflows import WorkflowRuntime


_INTEGER_KEYS = {
    "flows", "grace_period_s", "length", "lines", "minutes", "offset", "range_minutes",
    "route_type", "top_n", "vlan_id", "vni",
}
_BOOLEAN_KEYS = {"dry_run", "force", "graceful", "rolling"}
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
        schema = {"type": "object"}
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
    l0_skills: dict[str, dict[str, Any]] = {}
    try:
        for name, metadata in sorted(backend.metadata.items()):
            if metadata.get("internal_only"):
                continue
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
            l0_contract = L0_SKILLS.for_tool(backend.profile_id, name) if destructive else None
            if destructive and l0_contract is None:
                raise RuntimeError(
                    f"mutating tool {name!r} has no Network L0 Skill contract; fail closed"
                )
            declaration = {
                "name": name,
                "description": str(metadata.get("description", name)),
                "parameters": parameters,
                "action_type": action_type,
                "requires_approval": destructive,
                "editable_parameters": editable_parameters,
                "source": backend.sources.get(name, "unknown"),
                "tags": list(metadata.get("tags", [])),
                "capability_contract": backend.describe_capability(name).to_dict(),
            }
            if l0_contract is not None:
                declaration["l0_skill_id"] = l0_contract.skill_id
                declaration["l0_skill_version"] = l0_contract.version
                declaration["l0_contract_hash"] = l0_contract.contract_hash
                declaration["intent_kind"] = l0_contract.intent_kind
                declaration["execution_boundary"] = "domain_effect_runtime"
                declaration["provider_identity"] = str(
                    metadata.get("provider_identity") or backend.sources.get(name, "unknown")
                )
                declaration["input_schema_digest"] = metadata.get("input_schema_digest")
                declaration["output_schema_digest"] = metadata.get("output_schema_digest")
                l0_skills[l0_contract.skill_id] = l0_contract.to_dict()
            tools.append(declaration)
        return {
            "profile": backend.profile_id,
            "display_name": "Enterprise LAN Agent" if backend.profile_id == "lan" else backend.profile_id,
            "description": f"NetOpYu {backend.profile_id} tools via {backend.mode} backend",
            "backend": backend.report,
            "tools": tools,
            "l0_skills": [l0_skills[key] for key in sorted(l0_skills)],
        }
    finally:
        await backend.close()


def build_manifest(profile_id: str = "lan", *, include_destructive: bool = False) -> dict[str, Any]:
    """Return harness-facing declarations, including dynamically discovered tools."""
    return asyncio.run(_build_manifest(profile_id, include_destructive=include_destructive))


def build_l0_skill_catalog(profile_id: str = "lan") -> dict[str, Any]:
    manifest = build_manifest(profile_id, include_destructive=True)
    return {
        "profile": manifest["profile"],
        "backend": manifest["backend"],
        "l0_skills": manifest["l0_skills"],
    }


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
    access_context: dict[str, Any] | None = None,
    session_id: str | None = None,
    harness: str = "local",
) -> str:
    """Invoke a strictly validated read.

    ``allow_destructive`` is retained only for wire compatibility.  It can no
    longer bypass the plan/approval/evidence runtime.
    """
    del allow_destructive
    return await EffectRuntime().invoke_read(
        profile_id, tool_name, arguments, access_context=access_context,
        session_id=session_id, harness=harness,
    )


async def prepare_network_plan(
    profile_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    session_id: str | None = None,
    l0_skill_id: str | None = None,
    subject_context: dict[str, Any] | None = None,
    harness: str = "local",
    l1_decision_envelope: dict[str, Any] | None = None,
    l1_route_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return await EffectRuntime().prepare(
        profile_id, tool_name, arguments,
        session_id=session_id, l0_skill_id=l0_skill_id,
        subject_context=subject_context, harness=harness,
        l1_decision_envelope=l1_decision_envelope,
        l1_route_context=l1_route_context,
    )


def approve_network_plan(arguments: dict[str, Any]) -> dict[str, Any]:
    approvers = arguments.get("approver_contexts")
    if not isinstance(approvers, list) or not all(isinstance(item, dict) for item in approvers):
        raise ValueError("runtime approval requires an approver_contexts array")
    change_context = arguments.get("change_context")
    if change_context is not None and not isinstance(change_context, dict):
        raise ValueError("runtime approval change_context must be an object")
    return EffectRuntime().approve(
        plan_id=str(arguments.get("plan_id", "")),
        plan_hash=str(arguments.get("plan_hash", "")),
        approval_request_id=str(arguments.get("approval_request_id", "")),
        approver_contexts=approvers,
        change_context=change_context,
    )


async def execute_network_plan(arguments: dict[str, Any], *, allow_destructive: bool) -> dict[str, Any]:
    required = {"plan_id", "plan_hash", "execution_nonce"}
    missing = sorted(name for name in required if not str(arguments.get(name, "")).strip())
    if missing:
        raise ValueError("runtime execute missing fields: " + ", ".join(missing))
    if not str(arguments.get("approval_proof", "")).strip() and not (
        str(arguments.get("approval_request_id", "")).strip()
        and str(arguments.get("approval_actor", "")).strip()
    ):
        raise ValueError("runtime execute requires approval_proof or local compatibility approval fields")
    outcome = await EffectRuntime().execute(
        plan_id=str(arguments["plan_id"]),
        plan_hash=str(arguments["plan_hash"]),
        execution_nonce=str(arguments["execution_nonce"]),
        allow_destructive=allow_destructive,
        approval_proof=(
            str(arguments["approval_proof"])
            if arguments.get("approval_proof") else None
        ),
        approval_request_id=(
            str(arguments["approval_request_id"])
            if arguments.get("approval_request_id") else None
        ),
        approval_actor=(
            str(arguments["approval_actor"])
            if arguments.get("approval_actor") else None
        ),
    )
    return {
        **outcome.to_dict(),
        "terminal_envelope": outcome.terminal_envelope(),
    }


def inspect_network_plan(plan_id: str) -> dict[str, Any]:
    return EffectRuntime().inspect(plan_id)


def audit_network_plan(plan_id: str) -> dict[str, Any]:
    return EffectRuntime().audit(plan_id)


def recent_network_plans(limit: int = 20) -> list[dict[str, Any]]:
    return EffectRuntime().recent(limit)


def reject_network_plan(arguments: dict[str, Any]) -> dict[str, Any]:
    return EffectRuntime().reject(
        plan_id=str(arguments.get("plan_id", "")),
        plan_hash=str(arguments.get("plan_hash", "")),
        reason=str(arguments.get("reason", "approval was not granted")),
    )


def start_network_workflow(profile_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    with WorkflowRuntime(default_journal_path()) as runtime:
        return runtime.start(
            session_id=str(arguments.get("session_id", "")),
            profile=profile_id,
            mode=resolve_backend_mode(),
            skill_name=str(arguments.get("skill_name", "")),
        )


async def observe_network_workflow(profile_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    session_id = str(arguments.get("session_id", ""))
    tool_name = str(arguments.get("tool_name", ""))
    tool_arguments = arguments.get("tool_arguments") or {}
    if not session_id or not tool_name or not isinstance(tool_arguments, dict):
        raise ValueError("workflow observation requires session_id, tool_name and object tool_arguments")
    backend = await open_backend(profile_id)
    try:
        metadata = backend.metadata.get(tool_name)
        if metadata is None:
            return {"recorded": False, "reason": "tool is not a profile network tool"}
        action = str(metadata.get("action_type", "read_only"))
        mutating = bool(metadata.get("hitl")) or action != "read_only"
    finally:
        await backend.close()
    with WorkflowRuntime(default_journal_path()) as runtime:
        return runtime.observe(
            session_id=session_id,
            tool_name=tool_name,
            arguments=tool_arguments,
            result=str(arguments.get("result", "")),
            success=bool(arguments.get("success")),
            mutating=mutating,
        )
