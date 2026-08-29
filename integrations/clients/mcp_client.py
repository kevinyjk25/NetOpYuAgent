"""Official-SDK MCP client boundary used by the pragmatic backend.

Only real protocol transports are supported here. Local simulations run as
independent stdio MCP server processes; an in-process ``mock`` transport is
intentionally not available in pragmatic mode.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def schema_digest(value: dict[str, Any] | None) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value or {}).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MCPToolSpec:
    name: str
    description: str
    server_name: str
    server_identity: str
    server_version: str
    configured_domain: str = "external"
    parameters: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    returns_large: bool = False
    trusted_for_writes: bool = False
    identity_pinned: bool = False
    release_provider_id: str = ""

    @property
    def input_schema_digest(self) -> str:
        return schema_digest(self.parameters)

    @property
    def output_schema_digest(self) -> str:
        return schema_digest(self.output_schema)

    def to_skill_summary(self) -> dict[str, Any]:
        netopyu = self.meta.get("netopyu", {}) if isinstance(self.meta, dict) else {}
        action = str(netopyu.get("action_type") or "read_only")
        return {
            "name": self.name,
            "purpose": self.description[:120],
            "risk_level": "low" if action == "read_only" else "high",
            "requires_hitl": action != "read_only",
            "tags": ["mcp", self.server_name, str(netopyu.get("domain") or "external")],
            "description": self.description,
            "parameters": {
                key: value.get("description", key)
                for key, value in self.parameters.get("properties", {}).items()
            },
            "returns": "validated MCP structuredContent",
            "estimated_size": "large" if self.returns_large else "small",
            "returns_large": self.returns_large,
            "examples": [],
        }


@dataclass(frozen=True)
class MCPCallResult:
    tool_name: str
    server_name: str
    content: str
    structured_content: Any = None
    is_error: bool = False
    error_msg: str = ""
    call_ms: int = 0
    evidence_envelope: dict[str, Any] | None = None


def validate_evidence_envelope(
    spec: MCPToolSpec,
    structured: Any,
    *,
    server_identity: str,
    server_version: str,
    now: datetime | None = None,
) -> tuple[str, dict[str, Any]]:
    """Validate and unwrap a capability-bound provider evidence envelope."""
    if not isinstance(structured, dict):
        raise RuntimeError("network evidence result has no structuredContent object")
    declared = spec.meta.get("netopyu", {}) if isinstance(spec.meta, dict) else {}
    required = {
        "ok", "code", "correlation_id", "observed_at", "simulation",
        "provider_identity", "capability_id", "capability_version",
        "payload_digest", "content_type", "payload",
    }
    missing = sorted(required - set(structured))
    if missing:
        raise RuntimeError(f"network evidence envelope is missing fields: {missing}")
    if structured.get("ok") is not True:
        raise RuntimeError("network evidence envelope returned ok=false")
    expected_identity = f"{server_identity}@{server_version}"
    if structured.get("provider_identity") != expected_identity:
        raise RuntimeError(
            "network evidence provider identity mismatch: "
            f"expected {expected_identity!r}, got {structured.get('provider_identity')!r}"
        )
    for field_name in ("capability_id", "capability_version"):
        if structured.get(field_name) != declared.get(field_name):
            raise RuntimeError(
                f"network evidence {field_name} mismatch for {spec.name!r}"
            )
    observed_at = str(structured.get("observed_at") or "")
    try:
        parsed = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeError("network evidence observed_at is not ISO-8601") from error
    if parsed.tzinfo is None:
        raise RuntimeError("network evidence observed_at must include a timezone")
    current = now or datetime.now(timezone.utc)
    age_seconds = (current - parsed.astimezone(timezone.utc)).total_seconds()
    if age_seconds > 300:
        raise RuntimeError("network evidence is older than the 300-second freshness limit")
    if age_seconds < -30:
        raise RuntimeError("network evidence observed_at exceeds the 30-second future-skew limit")
    payload = structured.get("payload")
    expected_digest = "sha256:" + hashlib.sha256(
        _canonical(payload).encode("utf-8")
    ).hexdigest()
    if structured.get("payload_digest") != expected_digest:
        raise RuntimeError("network evidence payload digest mismatch")
    content = payload if isinstance(payload, str) else json.dumps(
        payload, ensure_ascii=False, sort_keys=True,
    )
    return content, dict(structured)


class MCPServer:
    """One identity-pinned official SDK client connection."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name = name
        self.config = dict(config)
        self._client: Any | None = None
        self._http_client: Any | None = None
        self._tools: list[MCPToolSpec] = []
        self.server_identity = ""
        self.server_version = ""

    @property
    def tools(self) -> list[MCPToolSpec]:
        return list(self._tools)

    async def connect(self) -> None:
        try:
            from mcp import Client, StdioServerParameters
        except ImportError as error:  # pragma: no cover - dependency gate
            raise RuntimeError(
                "the official MCP Python SDK is required: pip install 'mcp>=2,<3'"
            ) from error

        transport = str(self.config.get("transport") or "").lower()
        timeout = float(self.config.get("timeout", 30.0))
        if transport == "stdio":
            command = self.config.get("command") or []
            if not isinstance(command, list) or not command:
                raise ValueError(f"MCP stdio server {self.name!r} has no command")
            target: Any = StdioServerParameters(
                command=str(command[0]),
                args=[str(item) for item in command[1:]],
                env={str(k): str(v) for k, v in (self.config.get("env") or {}).items()} or None,
                cwd=self.config.get("cwd"),
            )
        elif transport in {"http", "streamable-http"}:
            from mcp.client.streamable_http import streamable_http_client
            import httpx2

            url = str(self.config.get("url") or "")
            if not url:
                raise ValueError(f"MCP HTTP server {self.name!r} has no url")
            self._http_client = httpx2.AsyncClient(
                headers=_build_auth_headers(self.config.get("auth") or {}),
                timeout=timeout,
                follow_redirects=True,
            )
            target = streamable_http_client(url, http_client=self._http_client)
        else:
            raise ValueError(
                f"unsupported MCP transport {transport!r}; use stdio or streamable-http"
            )

        self._client = Client(target, read_timeout_seconds=timeout)
        try:
            await self._client.__aenter__()
            info = self._client.server_info
            self.server_identity = str(getattr(info, "name", "") or "")
            self.server_version = str(getattr(info, "version", "") or "")
            self._verify_identity()
            await self._discover_tools()
        except BaseException:
            # A server that initialized but failed identity/schema discovery is
            # not appended to MCPClient.connect_all()'s connected list yet.
            # Close it here so a fail-closed trust check never leaks a process
            # or an HTTP connection.
            try:
                await self.disconnect()
            except BaseException:
                # Preserve the trust/discovery failure as the primary error.
                # A later caller may still call disconnect_all defensively.
                pass
            raise

    async def disconnect(self) -> None:
        if self._client is not None:
            await self._client.__aexit__(None, None, None)
            self._client = None
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    def _verify_identity(self) -> None:
        trusted = bool(self.config.get("trusted_for_writes", False))
        configured_domain = str(self.config.get("domain") or "external")
        expected_name = str(self.config.get("expected_server_name") or "")
        expected_version = str(self.config.get("expected_server_version") or "")
        if trusted and (not expected_name or not expected_version):
            raise ValueError(
                f"trusted MCP server {self.name!r} requires expected_server_name and expected_server_version"
            )
        if configured_domain == "network" and (not expected_name or not expected_version):
            raise ValueError(
                f"network MCP server {self.name!r} requires expected_server_name and expected_server_version"
            )
        if expected_name and self.server_identity != expected_name:
            raise RuntimeError(
                f"MCP identity mismatch for {self.name}: expected {expected_name}, got {self.server_identity}"
            )
        if expected_version and self.server_version != expected_version:
            raise RuntimeError(
                f"MCP version mismatch for {self.name}: expected {expected_version}, got {self.server_version}"
            )

    async def _discover_tools(self) -> None:
        self._tools = []
        cursor: str | None = None
        while True:
            result = await self._client.list_tools(cursor=cursor)
            for tool in result.tools:
                meta = dict(getattr(tool, "meta", None) or {})
                self._tools.append(MCPToolSpec(
                    name=str(tool.name),
                    description=str(tool.description or ""),
                    server_name=self.name,
                    server_identity=self.server_identity,
                    server_version=self.server_version,
                    configured_domain=str(self.config.get("domain") or "external"),
                    parameters=dict(tool.input_schema or {}),
                    output_schema=dict(tool.output_schema or {}),
                    meta=meta,
                    returns_large=bool(meta.get("x-returns-large", False)),
                    trusted_for_writes=bool(self.config.get("trusted_for_writes", False)),
                    identity_pinned=bool(
                        self.config.get("expected_server_name")
                        and self.config.get("expected_server_version")
                    ),
                    release_provider_id=str(
                        self.config.get("release_provider_id") or ""
                    ),
                ))
            cursor = getattr(result, "next_cursor", None)
            if not cursor:
                break

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> MCPCallResult:
        import time

        started = time.monotonic()
        try:
            result = await self._client.call_tool(tool_name, arguments)
            stamp = (getattr(result, "meta", None) or {}).get(
                "io.modelcontextprotocol/serverInfo", {}
            )
            if self.config.get("trusted_for_writes") and (
                stamp.get("name") != self.server_identity
                or str(stamp.get("version") or "") != self.server_version
            ):
                raise RuntimeError("trusted MCP result is missing the expected server identity stamp")
            blocks = []
            for block in result.content:
                text = getattr(block, "text", None)
                if text is not None:
                    blocks.append(str(text))
            is_error = bool(result.is_error)
            structured = getattr(result, "structured_content", None)
            evidence_envelope: dict[str, Any] | None = None
            spec = next((item for item in self._tools if item.name == tool_name), None)
            declared = (
                spec.meta.get("netopyu", {})
                if spec is not None and isinstance(spec.meta, dict) else {}
            )
            if not is_error and declared.get("result_contract") == "network-evidence-envelope-v1":
                if spec is None:  # pragma: no cover - discovery invariant
                    raise RuntimeError(f"MCP tool {tool_name!r} was not discovered")
                content, evidence_envelope = validate_evidence_envelope(
                    spec,
                    structured,
                    server_identity=self.server_identity,
                    server_version=self.server_version,
                )
            else:
                content = (
                    json.dumps(structured, ensure_ascii=False, sort_keys=True)
                    if structured is not None else "\n".join(blocks)
                )
            return MCPCallResult(
                tool_name=tool_name,
                server_name=self.name,
                content=content,
                structured_content=structured,
                is_error=is_error,
                error_msg="\n".join(blocks) if is_error else "",
                call_ms=int((time.monotonic() - started) * 1000),
                evidence_envelope=evidence_envelope,
            )
        except Exception as error:
            return MCPCallResult(
                tool_name=tool_name,
                server_name=self.name,
                content="",
                is_error=True,
                error_msg=f"{type(error).__name__}: {error}",
                call_ms=int((time.monotonic() - started) * 1000),
            )


class MCPClient:
    """Manage multiple real MCP server connections with unique tool names."""

    def __init__(self, servers: dict[str, MCPServer]) -> None:
        self._servers = servers

    @classmethod
    def from_config(cls, config: dict[str, dict[str, Any]]) -> "MCPClient":
        return cls({name: MCPServer(name, value) for name, value in config.items()})

    async def connect_all(self) -> None:
        connected: list[MCPServer] = []
        try:
            for server in self._servers.values():
                await server.connect()
                connected.append(server)
            names: dict[str, str] = {}
            for server in self._servers.values():
                for tool in server.tools:
                    previous = names.get(tool.name)
                    if previous:
                        raise RuntimeError(
                            f"duplicate MCP tool {tool.name!r} from {previous!r} and {server.name!r}"
                        )
                    names[tool.name] = server.name
        except Exception:
            for server in reversed(connected):
                await server.disconnect()
            raise

    async def disconnect_all(self) -> None:
        for server in reversed(list(self._servers.values())):
            await server.disconnect()

    def list_tools(self) -> list[MCPToolSpec]:
        return [tool for server in self._servers.values() for tool in server.tools]

    def get_tool_spec(self, tool_name: str) -> MCPToolSpec | None:
        return next((item for item in self.list_tools() if item.name == tool_name), None)

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> MCPCallResult:
        for server in self._servers.values():
            if any(tool.name == tool_name for tool in server.tools):
                return await server.call_tool(tool_name, arguments)
        return MCPCallResult(
            tool_name=tool_name,
            server_name="unknown",
            content="",
            is_error=True,
            error_msg=f"tool {tool_name!r} is not exposed by any connected MCP server",
        )

    @property
    def server_names(self) -> list[str]:
        return list(self._servers)


def _build_auth_headers(auth: dict[str, Any]) -> dict[str, str]:
    if not auth:
        return {}
    auth_type = str(auth.get("type") or "").lower()
    if auth_type == "bearer":
        token = os.getenv(str(auth.get("token_env") or ""), str(auth.get("token") or ""))
        return {"Authorization": f"Bearer {token}"}
    if auth_type == "api_key":
        key = os.getenv(str(auth.get("key_env") or ""), str(auth.get("key") or ""))
        return {str(auth.get("header") or "X-API-Key"): key}
    if auth_type == "basic":
        password = os.getenv(
            str(auth.get("password_env") or ""), str(auth.get("password") or "")
        )
        encoded = base64.b64encode(
            f"{auth.get('username', '')}:{password}".encode("utf-8")
        ).decode("ascii")
        return {"Authorization": f"Basic {encoded}"}
    raise ValueError(f"unsupported MCP authentication type {auth_type!r}")
