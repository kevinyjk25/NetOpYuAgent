"""
integrations/mcp_client.py
---------------------------
MCP (Model Context Protocol) client for connecting to your company's
existing MCP servers.

What MCP is
-----------
MCP is Anthropic's open protocol for connecting LLMs to external tools and
data sources. Your company NetOps MCP server exposes tools like:
  - get_device_status(device_id)
  - query_interface_metrics(host, interface, duration)
  - get_bgp_summary(router)
  - trigger_packet_capture(device, filter, duration)
  - get_syslog(host, severity, lines)

This client:
  1. Connects to one or more MCP servers via JSON-RPC over stdio / HTTP
  2. Discovers available tools (tools/list)
  3. Executes tool calls (tools/call)
  4. Returns results in a format compatible with ToolResultStore

MCP transport modes supported
------------------------------
  stdio   — subprocess with stdin/stdout (local MCP servers)
  http    — HTTP/SSE transport (remote MCP servers)
  mock    — in-process mock for testing (no subprocess needed)

Integration with existing system
----------------------------------
MCPClient feeds into ToolRouter, which is called by AgentRuntimeLoop
and TaskExecutor. No changes needed to any existing module.

Usage
-----
    client = MCPClient.from_config({
        "netops": {
            "transport": "http",
            "url": "http://netops-mcp.internal:8080",
            "auth": {"type": "bearer", "token_env": "NETOPS_MCP_TOKEN"},
        },
        "syslog": {
            "transport": "stdio",
            "command": ["python", "-m", "syslog_mcp_server"],
        },
    })
    await client.connect_all()

    # Discover tools
    tools = await client.list_tools()  # list[MCPToolSpec]

    # Call a tool
    result = await client.call_tool("get_device_status", {"device_id": "ap-01"})
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class MCPToolSpec:
    """Describes one tool exposed by an MCP server."""
    name:        str
    description: str
    server_name: str
    parameters:  dict[str, Any] = field(default_factory=dict)  # JSON Schema
    returns_large: bool = False   # hint for ToolResultStore

    def to_skill_summary(self) -> dict[str, Any]:
        """Convert to the format expected by SkillCatalogService."""
        return {
            "name":          self.name,
            "purpose":       self.description[:120],
            "risk_level":    "medium",
            "requires_hitl": False,
            "tags":          ["mcp", self.server_name],
            "description":   self.description,
            "parameters":    {
                k: v.get("description", str(v))
                for k, v in self.parameters.get("properties", {}).items()
            },
            "returns":       "string (MCP tool output)",
            "estimated_size": "large" if self.returns_large else "small",
            "returns_large":  self.returns_large,
            "examples":      [],
        }


@dataclass
class MCPCallResult:
    tool_name:   str
    server_name: str
    content:     str          # raw text result
    is_error:    bool = False
    error_msg:   str  = ""
    call_ms:     int  = 0     # latency in milliseconds


# ---------------------------------------------------------------------------
# Transport implementations
# ---------------------------------------------------------------------------

class _MCPTransport:
    """Base class for MCP transport implementations."""

    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    async def send_request(self, method: str, params: dict) -> dict: ...


class _HTTPTransport(_MCPTransport):
    """MCP over HTTP (JSON-RPC POST). Used for remote MCP servers."""

    def __init__(self, url: str, headers: dict[str, str] | None = None,
                 timeout: float = 30.0) -> None:
        self._url     = url.rstrip("/")
        self._headers = {"Content-Type": "application/json", **(headers or {})}
        self._timeout = timeout
        self._session = None

    async def connect(self) -> None:
        try:
            import httpx
            self._session = httpx.AsyncClient(
                headers=self._headers, timeout=self._timeout
            )
            logger.info("MCPHTTPTransport: connected to %s", self._url)
        except ImportError:
            raise RuntimeError(
                "httpx is required for HTTP MCP transport: pip install httpx"
            )

    async def disconnect(self) -> None:
        if self._session:
            await self._session.aclose()

    async def send_request(self, method: str, params: dict) -> dict:
        payload = {
            "jsonrpc": "2.0",
            "id":      str(uuid.uuid4()),
            "method":  method,
            "params":  params,
        }
        resp = await self._session.post(self._url + "/", json=payload)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"MCP error {data['error']['code']}: {data['error']['message']}")
        return data.get("result", {})


class _StdioTransport(_MCPTransport):
    """MCP over stdio subprocess. Used for local MCP servers."""

    def __init__(self, command: list[str], env: dict[str, str] | None = None) -> None:
        self._command = command
        self._env     = {**os.environ, **(env or {})}
        self._proc:   Optional[asyncio.subprocess.Process] = None
        self._pending: dict[str, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            *self._command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._env,
        )
        self._reader_task = asyncio.create_task(self._read_loop())
        # MCP initialise handshake
        await self.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "it-ops-agent", "version": "4.0"},
        })
        logger.info("MCPStdioTransport: started %s", self._command[0])

    async def disconnect(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
        if self._proc:
            self._proc.terminate()
            await self._proc.wait()

    async def send_request(self, method: str, params: dict) -> dict:
        req_id  = str(uuid.uuid4())
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = future

        payload = json.dumps({"jsonrpc":"2.0","id":req_id,"method":method,"params":params})
        self._proc.stdin.write((payload + "\n").encode())
        await self._proc.stdin.drain()

        return await asyncio.wait_for(future, timeout=30.0)

    async def _read_loop(self) -> None:
        while True:
            try:
                line = await self._proc.stdout.readline()
                if not line:
                    break
                msg = json.loads(line.decode())
                req_id = msg.get("id")
                if req_id and req_id in self._pending:
                    fut = self._pending.pop(req_id)
                    if "error" in msg:
                        fut.set_exception(RuntimeError(msg["error"]["message"]))
                    else:
                        fut.set_result(msg.get("result", {}))
            except Exception as exc:
                logger.debug("MCPStdioTransport read error: %s", exc)
                break


class _MockTransport(_MCPTransport):
    """In-process mock transport for testing without a real MCP server."""

    def __init__(self, tools: list[dict], handler: Any = None) -> None:
        self._tools   = tools
        self._handler = handler   # optional async callable(tool_name, args) -> str

    async def connect(self) -> None:
        logger.info("MCPMockTransport: connected (in-process mock)")

    async def disconnect(self) -> None: pass

    async def send_request(self, method: str, params: dict) -> dict:
        if method in ("initialize", "notifications/initialized"):
            return {"protocolVersion": "2024-11-05", "capabilities": {}}
        if method == "tools/list":
            return {"tools": self._tools}
        if method == "tools/call":
            name = params.get("name", "")
            args = params.get("arguments", {})
            if self._handler:
                text = await self._handler(name, args)
            else:
                text = f"[Mock MCP] {name}({args}) → ok"
            return {"content": [{"type": "text", "text": text}]}
        return {}


# ---------------------------------------------------------------------------
# Single MCP server connection
# ---------------------------------------------------------------------------

class MCPServer:
    """Manages connection and tool calls for one MCP server."""

    def __init__(self, name: str, transport: _MCPTransport) -> None:
        self.name       = name
        self._transport = transport
        self._tools:    list[MCPToolSpec] = []
        self._connected = False

    async def connect(self) -> None:
        await self._transport.connect()
        self._connected = True
        await self._discover_tools()

    async def disconnect(self) -> None:
        await self._transport.disconnect()
        self._connected = False

    async def _discover_tools(self) -> None:
        try:
            result = await self._transport.send_request("tools/list", {})
            raw_tools = result.get("tools", [])
            self._tools = []
            for t in raw_tools:
                schema = t.get("inputSchema", {})
                self._tools.append(MCPToolSpec(
                    name=t["name"],
                    description=t.get("description", ""),
                    server_name=self.name,
                    parameters=schema,
                    returns_large=t.get("x-returns-large", False),
                ))
            logger.info(
                "MCPServer[%s]: discovered %d tools: %s",
                self.name, len(self._tools),
                [t.name for t in self._tools],
            )
        except Exception as exc:
            logger.warning("MCPServer[%s]: tool discovery failed: %s", self.name, exc)

    @property
    def tools(self) -> list[MCPToolSpec]:
        return self._tools

    async def call_tool(self, tool_name: str, arguments: dict) -> MCPCallResult:
        import time
        start = time.monotonic()
        try:
            result = await self._transport.send_request("tools/call", {
                "name":      tool_name,
                "arguments": arguments,
            })
            # MCP returns content array; extract text
            content_blocks = result.get("content", [])
            text = "\n".join(
                block.get("text", str(block))
                for block in content_blocks
                if isinstance(block, dict)
            ) or str(result)
            ms = int((time.monotonic() - start) * 1000)
            return MCPCallResult(
                tool_name=tool_name, server_name=self.name,
                content=text, call_ms=ms,
            )
        except Exception as exc:
            ms = int((time.monotonic() - start) * 1000)
            logger.error("MCPServer[%s]: call_tool %s failed: %s", self.name, tool_name, exc)
            return MCPCallResult(
                tool_name=tool_name, server_name=self.name,
                content="", is_error=True, error_msg=str(exc), call_ms=ms,
            )


# ---------------------------------------------------------------------------
# MCPClient — manages multiple servers
# ---------------------------------------------------------------------------

class MCPClient:
    """
    Manages connections to one or more MCP servers.

    Config format (dict or JSON file):
    {
        "server_name": {
            "transport": "http" | "stdio" | "mock",
            "url": "http://...",                          // for http
            "command": ["python", "-m", "server"],        // for stdio
            "auth": {
                "type": "bearer" | "api_key" | "basic",
                "token_env": "MY_TOKEN_ENV_VAR",          // env var name
                "header": "Authorization",                // optional
            },
            "tools": [...],      // for mock: list of tool dicts
        }
    }
    """

    def __init__(self) -> None:
        self._servers: dict[str, MCPServer] = {}

    @classmethod
    def from_config(cls, config: dict[str, dict]) -> "MCPClient":
        client = cls()
        for name, cfg in config.items():
            transport_type = cfg.get("transport", "mock")

            if transport_type == "http":
                headers = _build_auth_headers(cfg.get("auth", {}))
                transport = _HTTPTransport(
                    url=cfg["url"],
                    headers=headers,
                    timeout=cfg.get("timeout", 30.0),
                )
            elif transport_type == "stdio":
                transport = _StdioTransport(
                    command=cfg["command"],
                    env=cfg.get("env"),
                )
            elif transport_type == "mock":
                transport = _MockTransport(
                    tools=cfg.get("tools", []),
                    handler=cfg.get("handler"),
                )
            else:
                raise ValueError(f"Unknown MCP transport: {transport_type!r}")

            client._servers[name] = MCPServer(name=name, transport=transport)
        return client

    @classmethod
    def from_netops_mock(cls) -> "MCPClient":
        """
        Factory: creates a mock MCPClient simulating a NetOps MCP server.
        Use this for development / CI without a real MCP server.
        """
        async def netops_handler(tool_name: str, args: dict) -> str:
            import random
            if tool_name == "get_device_status":
                dev = args.get("device_id", "unknown")
                return (
                    f"Device: {dev}\n"
                    f"  Status: {'up' if random.random() > 0.1 else 'down'}\n"
                    f"  Uptime: {random.randint(1, 365)}d\n"
                    f"  CPU: {random.randint(5, 85)}%\n"
                    f"  Memory: {random.randint(30, 90)}%\n"
                    f"  Interfaces up: {random.randint(4, 48)}/{random.randint(48, 48)}"
                )
            if tool_name == "query_interface_metrics":
                host = args.get("host", "sw-01")
                iface = args.get("interface", "GigE0/0")
                lines = [f"Interface metrics: {host} {iface}"]
                for i in range(10):
                    lines.append(
                        f"  t-{10-i}min: in={random.randint(100,900)}Mbps "
                        f"out={random.randint(50,800)}Mbps "
                        f"errors={random.randint(0,5)}"
                    )
                return "\n".join(lines)
            if tool_name == "get_bgp_summary":
                return (
                    f"BGP Summary for {args.get('router','router-01')}:\n"
                    f"  Neighbor  State    Prefixes  Uptime\n"
                    f"  10.0.0.1  Established  12500  14d\n"
                    f"  10.0.0.2  Established   8200  14d\n"
                    f"  10.0.0.3  Active           0   2m"
                )
            if tool_name == "get_syslog":
                import datetime
                host = args.get("host", "sw-01")
                severity = args.get("severity", "error")
                count = args.get("lines", 50)
                msgs = [
                    "Interface GigE0/1 changed state to down",
                    "OSPF adjacency lost with 10.0.0.5",
                    "Authentication failure from 192.168.1.100",
                    "High CPU utilization: 95%",
                    "Memory exhausted in process BGP",
                ]
                now = datetime.datetime.utcnow()
                lines = [f"# syslog {host} severity={severity} lines={count}"]
                import random as _r
                for i in range(min(count, 50)):
                    t = (now - datetime.timedelta(minutes=count-i)).strftime("%b %d %H:%M:%S")
                    lines.append(f"{t} {host} [{severity.upper()}] {_r.choice(msgs)}")
                return "\n".join(lines)
            if tool_name == "trigger_packet_capture":
                return (
                    f"Packet capture started on {args.get('device','sw-01')} "
                    f"filter='{args.get('filter','any')}' "
                    f"duration={args.get('duration',60)}s\n"
                    f"capture_id=pcap-{uuid.uuid4().hex[:8]}\n"
                    f"Status: running (check back in {args.get('duration',60)}s)"
                )
            return f"[NetOps MCP] {tool_name}({args}) → completed"

        mock_tools = [
            {"name": "get_device_status",     "description": "Get live status and resource usage of a network device",    "inputSchema": {"type":"object","properties":{"device_id":{"type":"string","description":"Device ID e.g. ap-01, sw-core"}}}},
            {"name": "query_interface_metrics","description": "Query time-series interface utilisation metrics",           "inputSchema": {"type":"object","properties":{"host":{"type":"string"},"interface":{"type":"string"},"duration":{"type":"string","description":"e.g. 10m 1h"}}}},
            {"name": "get_bgp_summary",        "description": "Show BGP neighbour table and prefix counts for a router",  "inputSchema": {"type":"object","properties":{"router":{"type":"string"}}}},
            {"name": "get_syslog",             "description": "Fetch syslog entries with severity filter",                 "inputSchema": {"type":"object","properties":{"host":{"type":"string"},"severity":{"type":"string","enum":["error","warn","info","debug"]},"lines":{"type":"integer","default":50}}},"x-returns-large":True},
            {"name": "trigger_packet_capture", "description": "Start a packet capture on a device interface",             "inputSchema": {"type":"object","properties":{"device":{"type":"string"},"filter":{"type":"string","description":"tcpdump filter expression"},"duration":{"type":"integer","description":"seconds"}}}},
        ]

        return cls.from_config({
            "netops": {
                "transport": "mock",
                "tools":     mock_tools,
                "handler":   netops_handler,
            }
        })

    async def connect_all(self) -> None:
        tasks = [s.connect() for s in self._servers.values()]
        await asyncio.gather(*tasks)
        logger.info("MCPClient: connected to %d server(s): %s",
                    len(self._servers), list(self._servers.keys()))

    async def disconnect_all(self) -> None:
        tasks = [s.disconnect() for s in self._servers.values()]
        await asyncio.gather(*tasks)

    def list_tools(self) -> list[MCPToolSpec]:
        """Return all tools from all connected servers."""
        tools = []
        for server in self._servers.values():
            tools.extend(server.tools)
        return tools

    def get_tool_spec(self, tool_name: str) -> Optional[MCPToolSpec]:
        for server in self._servers.values():
            for t in server.tools:
                if t.name == tool_name:
                    return t
        return None

    async def call_tool(self, tool_name: str, arguments: dict) -> MCPCallResult:
        """Call a tool by name on whichever server exposes it."""
        for server in self._servers.values():
            if any(t.name == tool_name for t in server.tools):
                return await server.call_tool(tool_name, arguments)
        return MCPCallResult(
            tool_name=tool_name, server_name="unknown",
            content="", is_error=True,
            error_msg=f"Tool {tool_name!r} not found in any connected MCP server",
        )

    def as_tool_registry(self) -> dict[str, Any]:
        """
        Return a dict compatible with AgentRuntimeLoop's tool_registry.
        Each value is an async callable(args: dict) -> str.
        """
        registry = {}
        for spec in self.list_tools():
            tool_name = spec.name   # capture for closure
            async def _call(args: dict, _name: str = tool_name) -> str:
                result = await self.call_tool(_name, args)
                if result.is_error:
                    return f"[MCP Error] {result.error_msg}"
                return result.content
            registry[tool_name] = _call
        return registry

    @property
    def server_names(self) -> list[str]:
        return list(self._servers.keys())


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _build_auth_headers(auth_cfg: dict) -> dict[str, str]:
    """Build HTTP auth headers from a config dict."""
    if not auth_cfg:
        return {}
    auth_type = auth_cfg.get("type", "")
    if auth_type == "bearer":
        env_var = auth_cfg.get("token_env", "")
        token   = os.getenv(env_var, auth_cfg.get("token", ""))
        return {"Authorization": f"Bearer {token}"}
    if auth_type == "api_key":
        env_var = auth_cfg.get("key_env", "")
        key     = os.getenv(env_var, auth_cfg.get("key", ""))
        header  = auth_cfg.get("header", "X-API-Key")
        return {header: key}
    if auth_type == "basic":
        import base64
        user    = auth_cfg.get("username", "")
        pw_env  = auth_cfg.get("password_env", "")
        pw      = os.getenv(pw_env, auth_cfg.get("password", ""))
        creds   = base64.b64encode(f"{user}:{pw}".encode()).decode()
        return {"Authorization": f"Basic {creds}"}
    return {}
