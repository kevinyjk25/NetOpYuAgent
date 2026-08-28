"""Official Hermes Plugin surface backed by the NetOpYu harness Worker."""

from __future__ import annotations

import getpass
import json
import os
import shlex
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .client import HermesWorkerClient
from .pending import PendingActions


_TRUE = {"1", "true", "yes", "on"}


def _default_runtime_root() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/NetOpYuAgent/hermes-runtime"
    data_root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    return data_root / "netopyu-agent/hermes-runtime"


@dataclass(frozen=True)
class HermesAdapterConfig:
    profile: str
    socket_path: Path
    include_destructive: bool
    operator_id: str
    own_agent_id: str
    peer_urls: tuple[str, ...]
    timeout_seconds: float
    memory_dir: Path | None = None

    @classmethod
    def from_env(cls) -> "HermesAdapterConfig":
        profile = os.environ.get("NETOPYU_HERMES_PROFILE") or os.environ.get("NETOPYU_PROFILE", "lan")
        if profile not in {"default", "lan", "dc", "wan"}:
            raise ValueError(f"unsupported NetOpYu Hermes profile: {profile}")
        socket_path = Path(
            os.environ.get("NETOPYU_HERMES_WORKER_SOCKET", _default_runtime_root() / "bridge.sock")
        ).expanduser()
        include_destructive = os.environ.get(
            "NETOPYU_HERMES_ENABLE_DESTRUCTIVE", "0"
        ).strip().lower() in _TRUE
        operator_id = os.environ.get(
            "NETOPYU_HERMES_OPERATOR_ID", f"local:{getpass.getuser()}"
        ).strip()
        if include_destructive and not operator_id:
            raise ValueError("NETOPYU_HERMES_OPERATOR_ID is required for mutating tools")
        peer_urls = tuple(
            value.strip().rstrip("/")
            for value in os.environ.get("NETOPYU_HERMES_A2A_PEERS", "").split(",")
            if value.strip()
        )
        return cls(
            profile=profile,
            socket_path=socket_path,
            include_destructive=include_destructive,
            operator_id=operator_id,
            own_agent_id=os.environ.get("NETOPYU_HERMES_AGENT_ID", f"hermes-{profile}"),
            peer_urls=peer_urls,
            timeout_seconds=float(os.environ.get("NETOPYU_HERMES_TIMEOUT", "120")),
            memory_dir=(
                Path(os.environ["NETOPYU_HERMES_MEMORY_DIR"]).expanduser()
                if os.environ.get("NETOPYU_HERMES_MEMORY_DIR") else None
            ),
        )


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _error(error: Exception) -> str:
    return _json({
        "ok": False,
        "status": "error",
        "error_type": type(error).__name__,
        "error": str(error),
    })


def _schema(tool: dict[str, Any]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, definition in tool.get("parameters", {}).items():
        definition = dict(definition)
        if definition.pop("required", False):
            required.append(name)
        properties[name] = definition
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        parameters["required"] = required
    suffix = (
        " This call only prepares an immutable Network L0 plan. It never performs the write; "
        "the authenticated operator must invoke the exact /netopyu-approve command returned."
        if tool.get("requires_approval") else ""
    )
    return {
        "name": tool["name"],
        "description": str(tool.get("description", tool["name"])) + suffix,
        "parameters": parameters,
    }


class NetOpYuHermesAdapter:
    """Project Network Runtime capabilities through Hermes' public plugin API."""

    def __init__(
        self,
        client: HermesWorkerClient,
        config: HermesAdapterConfig,
        *,
        pending: PendingActions | None = None,
    ) -> None:
        self.client = client
        self.config = config
        self.pending = pending or PendingActions()
        self.manifest: dict[str, Any] | None = None
        self.skill_manifest: dict[str, Any] | None = None

    def _observe_read(
        self,
        *,
        session_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        result: str,
    ) -> None:
        """Record reviewed-workflow evidence without changing the read result.

        A failed observation stays fail-closed for any later guarded write:
        Network Runtime will report missing prerequisites.
        """
        try:
            self.client.request(
                "workflow-observe",
                profile=self.config.profile,
                args={
                    "session_id": session_id,
                    "tool_name": tool_name,
                    "tool_arguments": arguments,
                    "result": result,
                    "success": True,
                },
            )
        except Exception:
            pass

    def _read_handler(self, tool: dict[str, Any]) -> Callable[..., str]:
        def handler(args: dict[str, Any], **kwargs: Any) -> str:
            try:
                session_id = str(kwargs.get("task_id") or kwargs.get("session_id") or "hermes")
                result = self.client.request(
                    "invoke", profile=self.config.profile, tool=tool["name"], args=args,
                    allow_destructive=False,
                )
                if isinstance(result, dict) and result.get("ok") is True:
                    self._observe_read(
                        session_id=session_id,
                        tool_name=tool["name"],
                        arguments=args,
                        result=str(result.get("result", "")),
                    )
                return _json(result)
            except Exception as error:
                return _error(error)
        return handler

    def _write_handler(self, tool: dict[str, Any]) -> Callable[..., str]:
        def handler(args: dict[str, Any], **kwargs: Any) -> str:
            try:
                session_id = str(kwargs.get("task_id") or kwargs.get("session_id") or "hermes")
                prepared = self.client.request(
                    "runtime-prepare",
                    profile=self.config.profile,
                    tool=tool["name"],
                    args=args,
                    session_id=session_id,
                    l0_skill_id=tool["l0_skill_id"],
                )
                if not isinstance(prepared, dict) or prepared.get("status") != "plan_ready":
                    return _json(prepared)
                pending = self.pending.add_plan(prepared)
                public = dict(prepared)
                public.pop("execution_nonce", None)
                public.update({
                    "ok": True,
                    "status": "approval_required",
                    "executed": False,
                    "approval": {
                        "actor": self.config.operator_id,
                        "command": f"/netopyu-approve {pending.plan_id} {pending.plan_hash}",
                        "deny_command": f"/netopyu-deny {pending.plan_id} {pending.plan_hash}",
                        "notice": "Only an explicit user slash command authorizes this exact plan.",
                    },
                })
                return _json(public)
            except Exception as error:
                return _error(error)
        return handler

    @staticmethod
    def _two_args(raw: str, command: str) -> tuple[str, str]:
        values = shlex.split(raw)
        if len(values) != 2 or any(not value.strip() for value in values):
            raise ValueError(f"usage: /{command} <id> <full-plan-hash>")
        return values[0], values[1]

    def approve(self, raw: str) -> str:
        try:
            plan_id, plan_hash = self._two_args(raw, "netopyu-approve")
            pending = self.pending.claim_plan(plan_id, plan_hash)
            outcome = self.client.request(
                "runtime-execute",
                profile=pending.profile,
                tool=pending.tool_name,
                args={
                    "plan_id": pending.plan_id,
                    "plan_hash": pending.plan_hash,
                    "execution_nonce": pending.execution_nonce,
                    "approval_request_id": f"hermes-slash:{uuid.uuid4()}",
                    "approval_actor": self.config.operator_id,
                },
                allow_destructive=True,
            )
            return _json(outcome)
        except Exception as error:
            return _error(error)

    def deny(self, raw: str) -> str:
        try:
            plan_id, plan_hash = self._two_args(raw, "netopyu-deny")
            pending = self.pending.claim_plan(plan_id, plan_hash)
            result = self.client.request(
                "runtime-reject",
                profile=pending.profile,
                args={
                    "plan_id": pending.plan_id,
                    "plan_hash": pending.plan_hash,
                    "reason": f"rejected by Hermes operator {self.config.operator_id}",
                },
            )
            return _json(result)
        except Exception as error:
            return _error(error)

    def pending_status(self, raw: str = "") -> str:
        if raw.strip():
            return _error(ValueError("usage: /netopyu-pending"))
        return _json(self.pending.list_public())

    def _peer_list(self, args: dict[str, Any], **kwargs: Any) -> str:
        del args, kwargs
        try:
            fields = {"peer_urls": list(self.config.peer_urls)} if self.config.peer_urls else {}
            return _json(self.client.request("a2a-peers", args=fields))
        except Exception as error:
            return _error(error)

    def _delegate(self, args: dict[str, Any], **kwargs: Any) -> str:
        try:
            if not args.get("target") and not args.get("capability"):
                raise ValueError("netopyu_delegate requires target or capability")
            request = {
                "prompt": str(args.get("prompt", "")),
                "target": str(args.get("target", "")),
                "capability": str(args.get("capability", "")),
                "session_id": str(kwargs.get("task_id") or kwargs.get("session_id") or uuid.uuid4()),
                "own_agent_id": self.config.own_agent_id,
                "delegation_chain": [],
                "timeout_seconds": self.config.timeout_seconds,
                "max_hops": 3,
                **({"peer_urls": list(self.config.peer_urls)} if self.config.peer_urls else {}),
            }
            result = self.client.request("a2a-delegate", args=request)
            if isinstance(result, dict) and result.get("status") == "input-required":
                pending = self.pending.add_remote(result, request)
                result = dict(result)
                result["continuation_id"] = pending.continuation_id
                result["approval_command"] = (
                    f"/netopyu-a2a-approve {pending.continuation_id} {pending.plan_hash}"
                )
                result["deny_command"] = (
                    f"/netopyu-a2a-deny {pending.continuation_id} {pending.plan_hash}"
                )
                result["notice"] = "The model cannot authorize the remote Network L0 plan."
            return _json(result)
        except Exception as error:
            return _error(error)

    def _resume_remote(self, raw: str, decision: str) -> str:
        try:
            continuation_id, plan_hash = self._two_args(raw, f"netopyu-a2a-{decision}")
            pending = self.pending.claim_remote(continuation_id, plan_hash)
            result = self.client.request(
                "a2a-delegate",
                args={
                    **pending.request,
                    "resume_interrupt_id": pending.interrupt_id,
                    "operator_decision": "approve" if decision == "approve" else "reject",
                },
            )
            return _json(result)
        except Exception as error:
            return _error(error)

    def approve_remote(self, raw: str) -> str:
        return self._resume_remote(raw, "approve")

    def deny_remote(self, raw: str) -> str:
        return self._resume_remote(raw, "deny")

    def _inspect_handler(self, args: dict[str, Any], **kwargs: Any) -> str:
        del kwargs
        try:
            return _json(self.client.request(
                "runtime-inspect", profile=self.config.profile,
                args={"plan_id": str(args.get("plan_id", ""))},
            ))
        except Exception as error:
            return _error(error)

    def _audit_handler(self, args: dict[str, Any], **kwargs: Any) -> str:
        del kwargs
        try:
            return _json(self.client.request(
                "runtime-audit", profile=self.config.profile,
                args={"plan_id": str(args.get("plan_id", ""))},
            ))
        except Exception as error:
            return _error(error)

    def _skill(self, name: str) -> dict[str, Any]:
        if not isinstance(self.skill_manifest, dict):
            raise LookupError("NetOpYu Skill manifest is unavailable")
        clean = name.split(":", 1)[-1]
        for skill in self.skill_manifest.get("skills", []):
            if clean in {str(skill.get("name", "")), str(skill.get("id", ""))}:
                return skill
        raise LookupError(f"NetOpYu Skill not found for profile {self.config.profile}: {name}")

    def _start_skill_workflow(self, name: str, session_id: str) -> dict[str, Any]:
        skill = self._skill(name)
        workflow = self.client.request(
            "workflow-start",
            profile=self.config.profile,
            args={"session_id": session_id, "skill_name": str(skill["name"])},
        )
        return {"skill": skill, "workflow": workflow}

    def _skill_view_handler(self, args: dict[str, Any], **kwargs: Any) -> str:
        try:
            session_id = str(kwargs.get("task_id") or kwargs.get("session_id") or "hermes")
            started = self._start_skill_workflow(str(args.get("name", "")), session_id)
            skill = started["skill"]
            return _json({
                "ok": True,
                "name": skill["name"],
                "description": skill["description"],
                "content": skill["content"],
                "resources": skill["resource_base"],
                "network_workflow": skill.get("network_workflow"),
                "workflow": started["workflow"],
                "notice": "Use this Skill in order. Network Runtime enforces reviewed write prerequisites.",
            })
        except Exception as error:
            return _error(error)

    def _skill_catalog_handler(self, args: dict[str, Any], **kwargs: Any) -> str:
        del args, kwargs
        skills = self.skill_manifest.get("skills", []) if isinstance(self.skill_manifest, dict) else []
        return _json({
            "profile": self.config.profile,
            "skills": [
                {
                    "name": skill.get("name"),
                    "description": skill.get("description"),
                    "has_network_workflow": "network_workflow" in skill,
                }
                for skill in skills
            ],
            "next_action": "Call netopyu_skill_view with one exact name before executing the workflow.",
        })

    def _capability_search_handler(self, args: dict[str, Any], **kwargs: Any) -> str:
        del kwargs
        try:
            allowed = [
                str(tool["name"])
                for tool in (self.manifest or {}).get("tools", [])
            ]
            return _json(self.client.request(
                "capability-search",
                profile=self.config.profile,
                args={
                    "query": str(args.get("query", "")),
                    "top_k": args.get("top_k"),
                    "kinds": args.get("kinds"),
                    "allowed_tool_names": allowed,
                },
            ))
        except Exception as error:
            return _error(error)

    def _memory_recall_handler(self, args: dict[str, Any], **kwargs: Any) -> str:
        try:
            if self.config.memory_dir is None:
                return _json({"available": False, "reason": "NETOPYU_HERMES_MEMORY_DIR is not configured"})
            session_id = str(kwargs.get("task_id") or kwargs.get("session_id") or "hermes")
            return _json(self.client.request(
                "memory-recall",
                profile=self.config.profile,
                args={
                    "memory_dir": str(self.config.memory_dir),
                    "operator_id": self.config.operator_id,
                    "session_id": session_id,
                    "query": str(args.get("query", "")),
                    "max_chars": args.get("max_chars"),
                    "recent_turns": args.get("recent_turns"),
                },
            ))
        except Exception as error:
            return _error(error)

    def _skill_pre_hook(self, **kwargs: Any) -> None:
        """Mirror Hermes built-in skill_view into the reviewed workflow guard."""
        if kwargs.get("tool_name") != "skill_view":
            return
        args = kwargs.get("args") or {}
        name = str(args.get("name", "")) if isinstance(args, dict) else ""
        if not name.startswith("netopyu:"):
            return
        session_id = str(kwargs.get("task_id") or kwargs.get("session_id") or "hermes")
        self._start_skill_workflow(name, session_id)

    def register(self, ctx: Any) -> None:
        self.client.ping()
        if self.config.include_destructive and not callable(getattr(ctx, "register_command", None)):
            raise RuntimeError(
                "Hermes register_command API is required before exposing NetOpYu mutation tools"
            )
        self.manifest = self.client.request(
            "manifest", profile=self.config.profile,
            include_destructive=self.config.include_destructive,
        )
        if not isinstance(self.manifest, dict):
            raise RuntimeError("NetOpYu Worker returned an invalid manifest")
        self.skill_manifest = self.client.request(
            "skill-manifest", profile=self.config.profile,
        )
        if not isinstance(self.skill_manifest, dict):
            raise RuntimeError("NetOpYu Worker returned an invalid Skill manifest")

        if self.config.include_destructive:
            ctx.register_command(
                "netopyu-approve", self.approve,
                "Approve and execute one exact pending Network L0 plan.",
            )
            ctx.register_command(
                "netopyu-deny", self.deny,
                "Reject one exact pending Network L0 plan.",
            )
            ctx.register_command(
                "netopyu-a2a-approve", self.approve_remote,
                "Approve one exact remote Network L0 continuation.",
            )
            ctx.register_command(
                "netopyu-a2a-deny", self.deny_remote,
                "Reject one exact remote Network L0 continuation.",
            )
        ctx.register_command(
            "netopyu-pending", self.pending_status,
            "List process-local pending NetOpYu approvals.",
        )
        if callable(getattr(ctx, "register_hook", None)):
            ctx.register_hook("pre_tool_call", self._skill_pre_hook)

        for tool in self.manifest.get("tools", []):
            schema = _schema(tool)
            handler = self._write_handler(tool) if tool.get("requires_approval") else self._read_handler(tool)
            ctx.register_tool(
                name=tool["name"], toolset="netopyu", schema=schema, handler=handler,
            )

        for name, description, handler in (
            ("netopyu_peer_list", "Discover configured NetOpYu A2A peers.", self._peer_list),
            ("netopyu_delegate", "Delegate a self-contained network task to one configured NetOpYu A2A peer.", self._delegate),
            ("netopyu_runtime_inspect", "Inspect one Network Runtime plan and its event journal.", self._inspect_handler),
            ("netopyu_runtime_audit", "Verify one Network Runtime plan event hash chain.", self._audit_handler),
            ("netopyu_skill_catalog", "List canonical NetOpYu Skills for the active profile.", self._skill_catalog_handler),
            ("netopyu_skill_view", "Load one canonical NetOpYu Skill and start its reviewed Network Runtime workflow.", self._skill_view_handler),
            ("netopyu_capability_search", "Search active-profile tools and Skills with CJK-aware retrieval.", self._capability_search_handler),
            ("netopyu_memory_recall", "Explicitly recall operator/session-scoped NetOpYu memory.", self._memory_recall_handler),
        ):
            if name == "netopyu_delegate":
                parameters = {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "prompt": {"type": "string"},
                        "target": {"type": "string"},
                        "capability": {"type": "string"},
                    },
                    "required": ["description", "prompt"],
                    "additionalProperties": False,
                }
            elif name in {"netopyu_runtime_inspect", "netopyu_runtime_audit"}:
                parameters = {
                    "type": "object",
                    "properties": {"plan_id": {"type": "string"}},
                    "required": ["plan_id"],
                    "additionalProperties": False,
                }
            elif name == "netopyu_skill_view":
                parameters = {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                    "additionalProperties": False,
                }
            elif name == "netopyu_capability_search":
                parameters = {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                        "kinds": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["tool", "skill"]},
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                }
            elif name == "netopyu_memory_recall":
                parameters = {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_chars": {"type": "integer", "minimum": 200, "maximum": 4000},
                        "recent_turns": {"type": "integer", "minimum": 0, "maximum": 10},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                }
            else:
                parameters = {"type": "object", "properties": {}, "additionalProperties": False}
            ctx.register_tool(
                name=name,
                toolset="netopyu",
                schema={"name": name, "description": description, "parameters": parameters},
                handler=handler,
            )

        if callable(getattr(ctx, "register_skill", None)):
            for skill in self.skill_manifest.get("skills", []):
                ctx.register_skill(str(skill["name"]), str(skill["path"]))


def register(ctx: Any) -> None:
    """Entry point used by Hermes' official project/user plugin loader."""
    config = HermesAdapterConfig.from_env()
    NetOpYuHermesAdapter(
        HermesWorkerClient(config.socket_path, timeout_seconds=config.timeout_seconds),
        config,
    ).register(ctx)
