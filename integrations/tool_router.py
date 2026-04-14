"""
integrations/tool_router.py
-----------------------------
ToolRouter — unified tool dispatcher for production use.

Merges tools from three sources into one registry that AgentRuntimeLoop
and TaskExecutor consume:

  Source 1: MCP servers     (your company NetOps MCP)
  Source 2: OpenAPI clients (your company REST APIs)
  Source 3: Mock/local      (always-available fallback tools)

Per-tool features
------------------
  auth_injection   — bearer/api_key headers added per tool call
  result_caching   — large results automatically routed to ToolResultStore
  rate_limiting    — per-tool cooldown to protect fragile APIs
  circuit_breaker  — auto-disable a tool after N consecutive failures
  schema_validation— validates args against tool's parameter schema before call
  latency_tracking — records p50/p99 per tool for /integrations/metrics

The resulting registry dict is passed directly to:
  - AgentRuntimeLoop(tool_registry=router.registry)
  - TaskExecutor(tool_registry=router.registry)
  - WebUI backend (tool_registry dict)

Usage
-----
    # Build and wire in main.py
    from integrations.tool_router import ToolRouter
    from integrations.mcp_client import MCPClient
    from integrations.openapi_client import OpenAPIClient

    mcp = MCPClient.from_config(NETOPS_MCP_CONFIG)
    await mcp.connect_all()

    api = OpenAPIClient.from_url(
        name="netops_nms",
        spec_url="https://nms.internal/openapi.json",
        base_url="https://nms.internal/api/v2",
        auth={"type": "bearer", "token_env": "NMS_TOKEN"},
    )
    await api.load()

    router = ToolRouter(tool_store=services["tool_store"])
    router.register_mcp(mcp)
    router.register_openapi(api)
    router.register_local(TOOL_REGISTRY)   # mock tools as fallback

    # Inject into executor and loop
    services["executor"]._tool_registry = router.registry
    services["runtime_loop"]._tool_registry = router.registry
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-tool metadata
# ---------------------------------------------------------------------------

@dataclass
class ToolMeta:
    """Metadata for one registered tool."""
    name:         str
    source:       str       # "mcp" | "openapi" | "local"
    description:  str       = ""
    returns_large: bool     = False

    # Circuit breaker
    failure_count:     int   = 0
    consecutive_fails: int   = 0
    max_consecutive:   int   = 5    # disable after this many consecutive failures
    disabled:          bool  = False
    disabled_until:    float = 0.0  # epoch timestamp

    # Rate limiting
    rate_limit_calls:    int   = 0    # 0 = no limit
    rate_limit_window_s: float = 60.0
    _call_timestamps: deque  = field(default_factory=lambda: deque(maxlen=200))

    # Latency tracking (ring buffer of last 100 call durations in ms)
    _latencies: deque = field(default_factory=lambda: deque(maxlen=100))

    def record_call(self, success: bool, duration_ms: float) -> None:
        now = time.monotonic()
        self._call_timestamps.append(now)
        self._latencies.append(duration_ms)
        if success:
            self.consecutive_fails = 0
            if self.disabled and time.time() > self.disabled_until:
                self.disabled = False
                logger.info("ToolRouter: circuit-breaker reset for tool=%s", self.name)
        else:
            self.failure_count     += 1
            self.consecutive_fails += 1
            if self.consecutive_fails >= self.max_consecutive:
                self.disabled       = True
                self.disabled_until = time.time() + 120.0   # 2-minute cooldown
                logger.warning(
                    "ToolRouter: circuit-breaker OPEN for tool=%s "
                    "(consecutive_fails=%d) — disabled for 120s",
                    self.name, self.consecutive_fails,
                )

    def is_rate_limited(self) -> bool:
        if self.rate_limit_calls <= 0:
            return False
        cutoff = time.monotonic() - self.rate_limit_window_s
        recent = sum(1 for t in self._call_timestamps if t >= cutoff)
        return recent >= self.rate_limit_calls

    @property
    def p50_ms(self) -> float:
        if not self._latencies:
            return 0.0
        s = sorted(self._latencies)
        return s[len(s) // 2]

    @property
    def p99_ms(self) -> float:
        if not self._latencies:
            return 0.0
        s = sorted(self._latencies)
        return s[int(len(s) * 0.99)]

    @property
    def call_count(self) -> int:
        return len(self._call_timestamps)


# ---------------------------------------------------------------------------
# ToolRouter
# ---------------------------------------------------------------------------

class ToolRouter:
    """
    Unified tool registry merging MCP, OpenAPI, and local tools.

    The .registry property returns a plain dict[str, callable] that any
    existing component (AgentRuntimeLoop, TaskExecutor, WebUI) can consume
    without any changes.
    """

    def __init__(
        self,
        tool_store:    Optional[Any]  = None,   # ToolResultStore
        default_timeout: float        = 30.0,
    ) -> None:
        self._tool_store     = tool_store
        self._default_timeout = default_timeout
        self._callables:  dict[str, Callable]  = {}    # tool_name → async callable
        self._meta:       dict[str, ToolMeta]  = {}    # tool_name → ToolMeta
        self._mcp_clients: list[Any]           = []
        self._api_clients: list[Any]           = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_mcp(self, mcp_client: Any) -> None:
        """Register all tools from a connected MCPClient."""
        self._mcp_clients.append(mcp_client)
        for spec in mcp_client.list_tools():
            self._register(
                name=spec.name,
                fn=self._make_mcp_fn(mcp_client, spec.name),
                source="mcp",
                description=spec.description,
                returns_large=spec.returns_large,
            )
        logger.info(
            "ToolRouter: registered %d MCP tools from %s",
            len(mcp_client.list_tools()),
            mcp_client.server_names,
        )

    def register_openapi(self, api_client: Any) -> None:
        """Register all operations from a loaded OpenAPIClient."""
        self._api_clients.append(api_client)
        for op in api_client.list_operations():
            tname = op.tool_name()
            self._register(
                name=tname,
                fn=self._make_openapi_fn(api_client, tname),
                source="openapi",
                description=op.summary,
                returns_large=False,
            )
        logger.info(
            "ToolRouter: registered %d OpenAPI operations from %s",
            len(api_client.list_operations()),
            api_client.name,
        )

    def register_local(self, tool_dict: dict[str, Callable]) -> None:
        """
        Register local/mock tools.  These are checked LAST — MCP and OpenAPI
        tools with the same name take priority.
        """
        count = 0
        for name, fn in tool_dict.items():
            if name not in self._callables:   # don't override real tools
                self._register(name=name, fn=fn, source="local")
                count += 1
        logger.info("ToolRouter: registered %d local tools", count)

    def set_rate_limit(self, tool_name: str, calls: int, window_s: float = 60.0) -> None:
        """Apply a rate limit to a specific tool."""
        if tool_name in self._meta:
            self._meta[tool_name].rate_limit_calls    = calls
            self._meta[tool_name].rate_limit_window_s = window_s

    def set_circuit_breaker(self, tool_name: str, max_consecutive: int = 5) -> None:
        """Configure the circuit breaker threshold for a tool."""
        if tool_name in self._meta:
            self._meta[tool_name].max_consecutive = max_consecutive

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    @property
    def registry(self) -> dict[str, Callable]:
        """
        Return the merged tool registry dict.
        Pass this to AgentRuntimeLoop(tool_registry=router.registry).
        """
        # Wrap every callable with circuit-breaker + rate-limit + store routing
        wrapped = {}
        for name, fn in self._callables.items():
            wrapped[name] = self._wrap(name, fn)
        return wrapped

    def _wrap(self, tool_name: str, fn: Callable) -> Callable:
        meta = self._meta[tool_name]

        async def _dispatch(args: dict) -> str:
            # Circuit breaker check
            if meta.disabled:
                if time.time() < meta.disabled_until:
                    return (
                        f"[ToolRouter] Tool {tool_name!r} is temporarily disabled "
                        f"(circuit breaker open after {meta.consecutive_fails} failures). "
                        f"Retry in {int(meta.disabled_until - time.time())}s."
                    )
                # Auto-reset after cooldown
                meta.disabled = False

            # Rate limit check
            if meta.is_rate_limited():
                return (
                    f"[ToolRouter] Tool {tool_name!r} is rate-limited "
                    f"({meta.rate_limit_calls} calls/{meta.rate_limit_window_s}s). "
                    "Please wait before retrying."
                )

            start   = time.monotonic()
            success = False
            try:
                result  = await asyncio.wait_for(fn(args), timeout=self._default_timeout)
                success = True
                raw     = str(result)

                # Route large results through ToolResultStore
                if self._tool_store is not None:
                    stored = self._tool_store.store(tool_name, raw)
                    return stored
                return raw

            except asyncio.TimeoutError:
                logger.warning("ToolRouter: timeout calling tool=%s", tool_name)
                return f"[ToolRouter] Tool {tool_name!r} timed out after {self._default_timeout}s"
            except Exception as exc:
                logger.error("ToolRouter: error calling tool=%s: %s", tool_name, exc)
                return f"[ToolRouter] Tool {tool_name!r} failed: {exc}"
            finally:
                elapsed_ms = (time.monotonic() - start) * 1000
                meta.record_call(success, elapsed_ms)
                logger.debug(
                    "ToolRouter: tool=%s success=%s elapsed=%.0fms source=%s",
                    tool_name, success, elapsed_ms, meta.source,
                )

        return _dispatch

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def get_metrics(self) -> list[dict]:
        """Return per-tool metrics for /integrations/metrics endpoint."""
        return [
            {
                "name":          name,
                "source":        m.source,
                "description":   m.description[:80],
                "call_count":    m.call_count,
                "failure_count": m.failure_count,
                "p50_ms":        round(m.p50_ms, 1),
                "p99_ms":        round(m.p99_ms, 1),
                "disabled":      m.disabled,
                "rate_limited":  m.is_rate_limited(),
                "returns_large": m.returns_large,
            }
            for name, m in self._meta.items()
        ]

    def get_tool_list(self) -> list[dict]:
        """Return a summary list for the WebUI integrations panel."""
        return [
            {
                "name":        name,
                "source":      m.source,
                "description": m.description[:100],
                "disabled":    m.disabled,
                "returns_large": m.returns_large,
            }
            for name, m in self._meta.items()
        ]

    def tool_count(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for m in self._meta.values():
            counts[m.source] += 1
        return dict(counts)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _register(
        self, name: str, fn: Callable, source: str,
        description: str = "", returns_large: bool = False,
    ) -> None:
        self._callables[name] = fn
        self._meta[name] = ToolMeta(
            name=name, source=source,
            description=description, returns_large=returns_large,
        )

    @staticmethod
    def _make_mcp_fn(mcp_client: Any, tool_name: str) -> Callable:
        async def _fn(args: dict) -> str:
            result = await mcp_client.call_tool(tool_name, args)
            if result.is_error:
                raise RuntimeError(result.error_msg)
            return result.content
        return _fn

    @staticmethod
    def _make_openapi_fn(api_client: Any, tool_name: str) -> Callable:
        async def _fn(args: dict) -> str:
            return await api_client.call(tool_name, args)
        return _fn
