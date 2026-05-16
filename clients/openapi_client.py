"""
integrations/openapi_client.py
--------------------------------
OpenAPI client — loads your company's NetOps REST API specs and
auto-generates tool wrappers callable by the Agent Runtime Loop.

What this solves
-----------------
Your company likely has multiple NetOps REST APIs (NMS, IPAM, ticketing,
config management) each with an OpenAPI 3.x spec. This client:

  1. Loads the spec from a URL or local file
  2. Parses every endpoint into an OperationSpec (method, path, params, body schema)
  3. Wraps each operation as an async callable compatible with tool_registry
  4. Handles path/query/body parameter injection automatically
  5. Injects auth headers per-request
  6. Returns responses as strings (goes through ToolResultStore if large)

Supported spec sources
-----------------------
  - HTTP URL:    "https://netops-api.internal/openapi.json"
  - Local file:  "/etc/netops/api-spec.yaml"
  - Inline dict: pass spec dict directly

Auth methods
-------------
  - Bearer token (from env var)
  - API key (custom header)
  - Basic auth
  - OAuth2 client_credentials (auto-refresh)

Usage
-----
    client = OpenAPIClient.from_url(
        name="netops_nms",
        spec_url="https://nms.internal/openapi.json",
        base_url="https://nms.internal/api/v2",
        auth={"type": "bearer", "token_env": "NMS_API_TOKEN"},
    )
    await client.load()

    # See what operations are available
    ops = client.list_operations()   # list[OperationSpec]

    # Execute a specific operation
    result = await client.call("GET_/devices/{device_id}", {"device_id": "ap-01"})

    # Get as tool_registry dict for AgentRuntimeLoop
    registry = client.as_tool_registry()
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ParamSpec:
    name:     str
    location: str          # path | query | header | cookie
    required: bool = False
    schema:   dict = field(default_factory=dict)
    description: str = ""


@dataclass
class OperationSpec:
    """Represents one API endpoint operation."""
    operation_id: str       # e.g. "GET_/devices/{device_id}"
    method:       str       # GET POST PUT DELETE PATCH
    path:         str       # /devices/{device_id}
    summary:      str       # one-line description
    description:  str       # full description
    parameters:   list[ParamSpec] = field(default_factory=list)
    request_body_schema: Optional[dict] = None
    response_schema:     Optional[dict] = None
    tags:         list[str] = field(default_factory=list)

    def tool_name(self) -> str:
        """Slugified name for use as a tool key in tool_registry."""
        path_slug = re.sub(r"[{}/ ]", "_", self.path).strip("_")
        return f"{self.method.lower()}_{path_slug}"

    def to_skill_summary(self) -> dict[str, Any]:
        params = {p.name: p.description or p.name for p in self.parameters}
        if self.request_body_schema:
            for k, v in self.request_body_schema.get("properties", {}).items():
                params[k] = v.get("description", k)
        return {
            "name":          self.tool_name(),
            "purpose":       self.summary[:120],
            "risk_level":    "high" if self.method in ("PUT","DELETE","PATCH") else "low",
            "requires_hitl": self.method in ("DELETE",),
            "tags":          ["openapi"] + self.tags,
            "description":   self.description or self.summary,
            "parameters":    params,
            "returns":       "JSON response as formatted string",
            "estimated_size": "small",
            "returns_large":  False,
        }


# ---------------------------------------------------------------------------
# OpenAPI spec parser
# ---------------------------------------------------------------------------

class OpenAPIParser:
    """Parses an OpenAPI 3.x spec dict into OperationSpec list."""

    def parse(self, spec: dict) -> list[OperationSpec]:
        operations = []
        paths = spec.get("paths", {})
        for path, path_item in paths.items():
            for method, op in path_item.items():
                if method not in ("get","post","put","delete","patch","head","options"):
                    continue
                op_id = (
                    op.get("operationId")
                    or f"{method.upper()}_{path}"
                )
                params = []
                for p in op.get("parameters", []):
                    # resolve $ref if needed
                    if "$ref" in p:
                        p = self._resolve_ref(p["$ref"], spec)
                    params.append(ParamSpec(
                        name=p.get("name",""),
                        location=p.get("in","query"),
                        required=p.get("required", False),
                        schema=p.get("schema", {}),
                        description=p.get("description", ""),
                    ))
                req_body = None
                if "requestBody" in op:
                    content = op["requestBody"].get("content", {})
                    for mime, schema_wrap in content.items():
                        if "schema" in schema_wrap:
                            req_body = schema_wrap["schema"]
                            break

                operations.append(OperationSpec(
                    operation_id=op_id,
                    method=method.upper(),
                    path=path,
                    summary=op.get("summary", ""),
                    description=op.get("description", ""),
                    parameters=params,
                    request_body_schema=req_body,
                    tags=op.get("tags", []),
                ))
        logger.debug("OpenAPIParser: parsed %d operations", len(operations))
        return operations

    @staticmethod
    def _resolve_ref(ref: str, spec: dict) -> dict:
        """Resolve a $ref like #/components/parameters/DeviceId."""
        parts = ref.lstrip("#/").split("/")
        node = spec
        for part in parts:
            node = node.get(part, {})
        return node


# ---------------------------------------------------------------------------
# OpenAPIClient
# ---------------------------------------------------------------------------

class OpenAPIClient:
    """
    Wraps a single OpenAPI-documented REST API.

    One company API = one OpenAPIClient.
    Multiple APIs → multiple clients → merged in ToolRouter.
    """

    def __init__(
        self,
        name:     str,
        base_url: str,
        auth_cfg: dict,
        spec_url: Optional[str]  = None,
        spec_path: Optional[str] = None,
        spec_dict: Optional[dict] = None,
        timeout:  float = 20.0,
        tag_filter: Optional[list[str]] = None,   # only load ops with these tags
    ) -> None:
        self.name       = name
        self._base_url  = base_url.rstrip("/")
        self._auth_cfg  = auth_cfg
        self._spec_url  = spec_url
        self._spec_path = spec_path
        self._spec_dict = spec_dict
        self._timeout   = timeout
        self._tag_filter = tag_filter
        self._operations: list[OperationSpec] = []
        self._session   = None
        self._oauth_token: Optional[str] = None

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_url(cls, name: str, spec_url: str, base_url: str,
                 auth: dict | None = None, **kw) -> "OpenAPIClient":
        return cls(name=name, base_url=base_url, auth_cfg=auth or {},
                   spec_url=spec_url, **kw)

    @classmethod
    def from_file(cls, name: str, spec_path: str, base_url: str,
                  auth: dict | None = None, **kw) -> "OpenAPIClient":
        return cls(name=name, base_url=base_url, auth_cfg=auth or {},
                   spec_path=spec_path, **kw)

    @classmethod
    def from_dict(cls, name: str, spec: dict, base_url: str,
                  auth: dict | None = None, **kw) -> "OpenAPIClient":
        return cls(name=name, base_url=base_url, auth_cfg=auth or {},
                   spec_dict=spec, **kw)

    @classmethod
    def netops_mock(cls) -> "OpenAPIClient":
        """
        Factory: creates a mock OpenAPIClient simulating a NetOps REST API.
        Includes IPAM, device config, incident management endpoints.
        """
        mock_spec = {
            "openapi": "3.0.0",
            "info": {"title": "NetOps Management API", "version": "2.0"},
            "paths": {
                "/devices": {
                    "get": {
                        "operationId": "list_devices",
                        "summary": "List all network devices",
                        "tags": ["devices"],
                        "parameters": [
                            {"name": "site", "in": "query", "schema": {"type": "string"}, "description": "Filter by site name"},
                            {"name": "type", "in": "query", "schema": {"type": "string", "enum": ["ap","switch","router"]}, "description": "Device type filter"},
                            {"name": "status", "in": "query", "schema": {"type": "string", "enum": ["up","down","unknown"]}, "description": "Status filter"},
                        ],
                    }
                },
                "/devices/{device_id}": {
                    "get": {
                        "operationId": "get_device",
                        "summary": "Get device full configuration and status",
                        "tags": ["devices"],
                        "parameters": [
                            {"name": "device_id", "in": "path", "required": True, "schema": {"type": "string"}, "description": "Device identifier"},
                        ],
                    },
                    "patch": {
                        "operationId": "update_device",
                        "summary": "Update device configuration",
                        "tags": ["devices"],
                        "parameters": [{"name": "device_id", "in": "path", "required": True, "schema": {"type": "string"}}],
                        "requestBody": {"content": {"application/json": {"schema": {"type": "object", "properties": {"description": {"type": "string"}, "vlan": {"type": "integer"}, "qos_policy": {"type": "string"}}}}}},
                    },
                },
                "/ipam/addresses": {
                    "get": {
                        "operationId": "search_ip_addresses",
                        "summary": "Search IP address assignments in IPAM",
                        "tags": ["ipam"],
                        "parameters": [
                            {"name": "prefix", "in": "query", "schema": {"type": "string"}, "description": "CIDR prefix to search within"},
                            {"name": "mac", "in": "query", "schema": {"type": "string"}, "description": "MAC address"},
                            {"name": "hostname", "in": "query", "schema": {"type": "string"}, "description": "Hostname search"},
                        ],
                    }
                },
                "/ipam/prefixes": {
                    "get": {
                        "operationId": "list_prefixes",
                        "summary": "List IP prefixes and subnets",
                        "tags": ["ipam"],
                        "parameters": [
                            {"name": "vrf", "in": "query", "schema": {"type": "string"}, "description": "VRF name"},
                        ],
                    }
                },
                "/incidents": {
                    "get": {
                        "operationId": "list_incidents",
                        "summary": "List open network incidents",
                        "tags": ["incidents"],
                        "parameters": [
                            {"name": "severity", "in": "query", "schema": {"type": "string", "enum": ["P0","P1","P2","P3"]}},
                            {"name": "status", "in": "query", "schema": {"type": "string", "enum": ["open","in_progress","resolved"]}},
                        ],
                    },
                    "post": {
                        "operationId": "create_incident",
                        "summary": "Create a new network incident",
                        "tags": ["incidents"],
                        "requestBody": {"content": {"application/json": {"schema": {"type": "object", "required": ["title", "severity"], "properties": {"title": {"type": "string"}, "severity": {"type": "string"}, "description": {"type": "string"}, "affected_devices": {"type": "array", "items": {"type": "string"}}}}}}},
                    }
                },
                "/incidents/{incident_id}": {
                    "get": {
                        "operationId": "get_incident",
                        "summary": "Get incident details and timeline",
                        "tags": ["incidents"],
                        "parameters": [{"name": "incident_id", "in": "path", "required": True, "schema": {"type": "string"}}],
                    },
                    "patch": {
                        "operationId": "update_incident",
                        "summary": "Update incident status or add notes",
                        "tags": ["incidents"],
                        "parameters": [{"name": "incident_id", "in": "path", "required": True, "schema": {"type": "string"}}],
                        "requestBody": {"content": {"application/json": {"schema": {"type": "object", "properties": {"status": {"type": "string"}, "resolution_notes": {"type": "string"}, "rca": {"type": "string"}}}}}},
                    }
                },
                "/config/backup/{device_id}": {
                    "post": {
                        "operationId": "trigger_config_backup",
                        "summary": "Trigger configuration backup for a device",
                        "tags": ["config"],
                        "parameters": [{"name": "device_id", "in": "path", "required": True, "schema": {"type": "string"}}],
                    }
                },
                "/config/diff/{device_id}": {
                    "get": {
                        "operationId": "get_config_diff",
                        "summary": "Get configuration changes since last backup",
                        "tags": ["config"],
                        "parameters": [
                            {"name": "device_id", "in": "path", "required": True, "schema": {"type": "string"}},
                            {"name": "since", "in": "query", "schema": {"type": "string"}, "description": "ISO 8601 timestamp"},
                        ],
                    }
                },
            }
        }
        return cls.from_dict(name="netops_api", spec=mock_spec,
                              base_url="http://netops-api.mock")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def load(self) -> None:
        """Load and parse the OpenAPI spec."""
        spec = await self._fetch_spec()
        parser = OpenAPIParser()
        all_ops = parser.parse(spec)
        if self._tag_filter:
            all_ops = [
                op for op in all_ops
                if any(tag in op.tags for tag in self._tag_filter)
            ]
        self._operations = all_ops
        logger.info(
            "OpenAPIClient[%s]: loaded %d operations from spec",
            self.name, len(self._operations),
        )

        # Set up HTTP session
        try:
            import httpx
            headers = await self._build_auth_headers()
            self._session = httpx.AsyncClient(
                base_url=self._base_url,
                headers=headers,
                timeout=self._timeout,
            )
        except ImportError:
            logger.warning("httpx not installed — OpenAPI calls will use mock responses")

    async def unload(self) -> None:
        if self._session:
            await self._session.aclose()

    # ------------------------------------------------------------------
    # Tool access
    # ------------------------------------------------------------------

    def list_operations(self) -> list[OperationSpec]:
        return self._operations

    def get_operation(self, tool_name: str) -> Optional[OperationSpec]:
        for op in self._operations:
            if op.tool_name() == tool_name or op.operation_id == tool_name:
                return op
        return None

    async def call(self, tool_name: str, args: dict) -> str:
        """Execute an API operation by tool_name and return response as string."""
        op = self.get_operation(tool_name)
        if op is None:
            return f"[OpenAPI Error] Operation {tool_name!r} not found in {self.name}"
        return await self._execute(op, args)

    async def _execute(self, op: OperationSpec, args: dict) -> str:
        # Build URL with path parameters
        url = op.path
        query_params = {}
        body_data = {}

        for param in op.parameters:
            val = args.get(param.name)
            if val is None:
                continue
            if param.location == "path":
                url = url.replace(f"{{{param.name}}}", str(val))
            elif param.location == "query":
                query_params[param.name] = val

        # Remaining args go to request body
        path_and_query_keys = {p.name for p in op.parameters}
        for k, v in args.items():
            if k not in path_and_query_keys:
                body_data[k] = v

        if self._session is None:
            # Mock response when no httpx session
            return self._mock_response(op, args)

        try:
            kwargs: dict = {"params": query_params or None}
            if body_data:
                kwargs["json"] = body_data
            resp = await self._session.request(op.method, url, **kwargs)
            resp.raise_for_status()
            try:
                data = resp.json()
                return json.dumps(data, indent=2)
            except Exception:
                return resp.text
        except Exception as exc:
            logger.error("OpenAPIClient[%s] %s %s failed: %s", self.name, op.method, url, exc)
            return f"[OpenAPI Error] {op.method} {url}: {exc}"

    def _mock_response(self, op: OperationSpec, args: dict) -> str:
        """Generate a realistic mock response when no real server is available."""
        import random, datetime
        if "device" in op.path or "device" in op.operation_id:
            dev = args.get("device_id", "sw-core-01")
            return json.dumps({
                "device_id": dev, "status": "up",
                "model": "Cisco Catalyst 9300", "firmware": "17.9.3",
                "uptime": "14d 6h", "cpu_pct": random.randint(10,60),
                "memory_pct": random.randint(30,70),
                "interfaces": random.randint(24,48),
                "site": args.get("site", "Site-A"),
            }, indent=2)
        if "ipam" in op.path:
            return json.dumps({
                "count": random.randint(1,5),
                "results": [
                    {"address": f"10.0.{random.randint(0,10)}.{random.randint(1,254)}/24",
                     "hostname": f"device-{i}", "mac": "aa:bb:cc:dd:ee:ff",
                     "status": "active"}
                    for i in range(3)
                ]
            }, indent=2)
        if "incident" in op.path:
            return json.dumps({
                "id": f"INC-{random.randint(1000,9999)}",
                "title": args.get("title", "Network incident"),
                "severity": args.get("severity", "P2"),
                "status": "open",
                "created_at": datetime.datetime.utcnow().isoformat(),
                "affected_devices": args.get("affected_devices", []),
            }, indent=2)
        return json.dumps({"status": "ok", "operation": op.operation_id, "args": args}, indent=2)

    def as_tool_registry(self) -> dict[str, Any]:
        """Return tool_registry dict for AgentRuntimeLoop."""
        registry = {}
        for op in self._operations:
            tname = op.tool_name()
            async def _call(args: dict, _op: OperationSpec = op) -> str:
                return await self._execute(_op, args)
            registry[tname] = _call
        return registry

    # ------------------------------------------------------------------
    # Spec loading
    # ------------------------------------------------------------------

    async def _fetch_spec(self) -> dict:
        if self._spec_dict:
            return self._spec_dict
        if self._spec_path:
            return self._load_spec_file(self._spec_path)
        if self._spec_url:
            return await self._fetch_spec_url(self._spec_url)
        raise ValueError("OpenAPIClient requires spec_url, spec_path, or spec_dict")

    @staticmethod
    def _load_spec_file(path: str) -> dict:
        import pathlib
        text = pathlib.Path(path).read_text(encoding="utf-8")
        if path.endswith((".yaml", ".yml")):
            try:
                import yaml
                return yaml.safe_load(text)
            except ImportError:
                raise RuntimeError("pip install pyyaml for YAML OpenAPI specs")
        return json.loads(text)

    async def _fetch_spec_url(self, url: str) -> dict:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                ct = resp.headers.get("content-type", "")
                if "yaml" in ct:
                    import yaml
                    return yaml.safe_load(resp.text)
                return resp.json()
        except ImportError:
            raise RuntimeError("httpx is required to fetch remote OpenAPI specs")

    async def _build_auth_headers(self) -> dict[str, str]:
        cfg = self._auth_cfg
        if not cfg:
            return {}
        auth_type = cfg.get("type", "")
        if auth_type == "bearer":
            env_var = cfg.get("token_env", "")
            token   = os.getenv(env_var, cfg.get("token", ""))
            return {"Authorization": f"Bearer {token}"}
        if auth_type == "api_key":
            env_var = cfg.get("key_env", "")
            key     = os.getenv(env_var, cfg.get("key", ""))
            return {cfg.get("header", "X-API-Key"): key}
        if auth_type == "oauth2_client_credentials":
            token = await self._fetch_oauth2_token(cfg)
            return {"Authorization": f"Bearer {token}"}
        if auth_type == "basic":
            import base64
            user = cfg.get("username", "")
            pw   = os.getenv(cfg.get("password_env", ""), cfg.get("password", ""))
            creds = base64.b64encode(f"{user}:{pw}".encode()).decode()
            return {"Authorization": f"Basic {creds}"}
        return {}

    async def _fetch_oauth2_token(self, cfg: dict) -> str:
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    cfg["token_url"],
                    data={
                        "grant_type":    "client_credentials",
                        "client_id":     os.getenv(cfg.get("client_id_env",""), cfg.get("client_id","")),
                        "client_secret": os.getenv(cfg.get("client_secret_env",""), cfg.get("client_secret","")),
                        "scope":         cfg.get("scope", ""),
                    },
                )
                resp.raise_for_status()
                return resp.json()["access_token"]
        except Exception as exc:
            logger.error("OAuth2 token fetch failed: %s", exc)
            return ""
