"""
retrieval/meta_tool.py
----------------------
MetaTool framework — registry for "always-injected" tools that operate
on the agent's own state (tool catalog, skill catalog, memory, ...).

Design:
  - Meta-tools are NORMAL ToolRouter tools at execution time
  - The difference is purely at PROMPT-INJECTION time:
      → meta-tools are ALWAYS listed in the system prompt
      → regular tools are listed only when retrieved as relevant
  - Adding a new meta-tool is one register() call — no runtime/prompt code changes

Built-in meta-tools (provided in this file):
  - list_tools       : query → top-N tools for the current task
  - list_skills      : query → top-N skills (delegates to SkillCatalog)
  - tool_details     : tool_name → full description / parameters
  - skill_details    : skill_id  → full skill markdown

Custom meta-tools can be registered by host code via:
    get_meta_tool_registry().register(MetaTool(name=..., description=..., handler=async_fn))

In production, register() is typically called from main.py after the
tool retriever and skill catalog are wired.
"""
from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MetaTool dataclass
# ---------------------------------------------------------------------------

@dataclass
class MetaTool:
    """A meta-tool registered with the agent runtime.

    Attributes:
        name:        canonical tool name (used in [TOOL:name] calls)
        description: short prose; appears verbatim in the system prompt
        handler:     async callable handling the call. Signature:
                       async def handler(**kwargs) -> str
                     Returning str (raw or JSON-serialised) keeps it
                     compatible with ToolRouter's existing pipe.
        parameters:  {param_name: param_description}; documented in prompt
        tags:        free-form taxonomy; e.g. ["meta", "discovery"]
        always_inject: when True (default), this tool is ALWAYS in prompt.
                       Set False for meta-tools that should themselves be
                       retrieved (rare).
    """
    name:          str
    description:   str
    handler:       Callable[..., Awaitable[str]]
    parameters:    dict[str, str]   = field(default_factory=dict)
    tags:          list[str]        = field(default_factory=lambda: ["meta"])
    always_inject: bool             = True
    examples:      list[str]        = field(default_factory=list)
    """Optional usage examples appended to the prompt block."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class MetaToolRegistry:
    """Runtime registry for meta-tools.

    Thread-safe for concurrent register/lookup (uses a lock).
    Process-global singleton accessed via get_meta_tool_registry();
    host code may also instantiate its own for tests.
    """

    def __init__(self):
        import threading
        self._lock = threading.RLock()
        self._tools: dict[str, MetaTool] = {}

    # ── Registration ──────────────────────────────────────────────────

    def register(self, tool: MetaTool, *, replace: bool = False) -> None:
        """Register (or replace) a meta-tool.

        Args:
            tool:    MetaTool instance.
            replace: if True, overwrite an existing tool with the same name.
                     If False (default), raise on duplicate.
        """
        with self._lock:
            if tool.name in self._tools and not replace:
                raise ValueError(
                    f"MetaTool {tool.name!r} already registered "
                    f"(use replace=True to override)"
                )
            if not inspect.iscoroutinefunction(tool.handler):
                raise TypeError(
                    f"MetaTool {tool.name!r} handler must be an async function"
                )
            self._tools[tool.name] = tool
            logger.info(
                "MetaToolRegistry: registered %r (always_inject=%s)",
                tool.name, tool.always_inject,
            )

    def unregister(self, name: str) -> bool:
        with self._lock:
            return self._tools.pop(name, None) is not None

    # ── Lookup ────────────────────────────────────────────────────────

    def get(self, name: str) -> Optional[MetaTool]:
        with self._lock:
            return self._tools.get(name)

    def list_all(self) -> list[MetaTool]:
        with self._lock:
            return list(self._tools.values())

    def list_always_injected(self) -> list[MetaTool]:
        with self._lock:
            return [t for t in self._tools.values() if t.always_inject]

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._tools

    def __len__(self) -> int:
        with self._lock:
            return len(self._tools)

    # ── Prompt assembly ───────────────────────────────────────────────

    def build_prompt_section(self) -> str:
        """Return the always-injected meta-tool block for the system prompt.

        Compact format to minimise token usage:
            META TOOLS — always available:
              [TOOL:list_tools] {"query": "..."} — top-N tools matching query
                Args: query (search description)
              [TOOL:list_skills] ...
        """
        with self._lock:
            tools = [t for t in self._tools.values() if t.always_inject]
        if not tools:
            return ""

        lines = ["META TOOLS — always available, use to discover capabilities:"]
        for t in sorted(tools, key=lambda x: x.name):
            lines.append(f"  [TOOL:{t.name}] — {t.description}")
            if t.parameters:
                params_doc = ", ".join(f"{p} ({d})" for p, d in t.parameters.items())
                lines.append(f"    Args: {params_doc}")
            for ex in t.examples[:1]:           # at most one example to stay compact
                lines.append(f"    e.g. {ex}")
        return "\n".join(lines)

    # ── ToolRouter integration ────────────────────────────────────────

    def as_local_callables(self) -> dict[str, Callable]:
        """Return {name: handler} dict suitable for ToolRouter.register_local().

        Lets meta-tools be invoked exactly like any other tool — the
        runtime loop's existing [TOOL:name] dispatch picks them up.
        """
        with self._lock:
            return {t.name: t.handler for t in self._tools.values()}


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_GLOBAL_REGISTRY: Optional[MetaToolRegistry] = None


def get_meta_tool_registry() -> MetaToolRegistry:
    """Return the process-wide registry (creating it on first call)."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = MetaToolRegistry()
    return _GLOBAL_REGISTRY


# ---------------------------------------------------------------------------
# Built-in meta-tool factories
# ---------------------------------------------------------------------------
# These are NOT auto-registered — host code (main.py) decides which ones
# to wire after constructing the retrievers. This avoids hidden side effects.

def make_list_tools_meta_tool(
    tool_retriever:  Any,
    *,
    default_top_k:   int = 5,
    name:            str = "list_tools",
) -> MetaTool:
    """Factory: meta-tool that returns top-K tools matching a query.

    Args:
        tool_retriever: any Retriever instance indexed with the agent's tools.
        default_top_k:  fallback when caller doesn't pass top_k.
        name:           override if you want a different verbatim name.
    """
    async def _handler(**kwargs) -> str:
        query = str(kwargs.get("query", "") or "").strip()
        top_k = int(kwargs.get("top_k", default_top_k))
        if not query:
            return "Error: list_tools requires a 'query' parameter."

        # Prefer async retrieval when available
        if hasattr(tool_retriever, "retrieve_async"):
            res = await tool_retriever.retrieve_async(query, top_k=top_k)
        else:
            res = tool_retriever.retrieve(query, top_k=top_k)

        if not res.matches:
            return f"No tools matched '{query}'. Available pool: {res.total_pool} tools."

        lines = [f"Top {len(res.matches)} tools for '{query}':"]
        for m in res.matches:
            it = m.item
            hitl = " ⚠HITL" if it.get("hitl") else ""
            lines.append(
                f"  [TOOL:{m.id}]{hitl} (score={m.score:.2f}) — "
                f"{it.get('description','')[:120]}"
            )
            params = it.get("parameters") or {}
            if params:
                lines.append(f"    Args: " + ", ".join(
                    f"{k}" for k in list(params.keys())[:6]
                ))
        return "\n".join(lines)

    return MetaTool(
        name=name,
        description="Search for tools matching a description; returns top-K with usage args",
        handler=_handler,
        parameters={
            "query": "natural-language description of the capability you need",
            "top_k": "max results to return (default 5)",
        },
        tags=["meta", "discovery"],
        examples=[
            '[TOOL:list_tools] {"query": "check device interface metrics"}',
        ],
    )


def make_list_skills_meta_tool(
    skill_retriever: Any,
    *,
    default_top_k:   int = 3,
    name:            str = "list_skills",
) -> MetaTool:
    """Factory: meta-tool that returns top-K skills matching a query."""
    async def _handler(**kwargs) -> str:
        query = str(kwargs.get("query", "") or "").strip()
        top_k = int(kwargs.get("top_k", default_top_k))
        if not query:
            return "Error: list_skills requires a 'query' parameter."

        if hasattr(skill_retriever, "retrieve_async"):
            res = await skill_retriever.retrieve_async(query, top_k=top_k)
        else:
            res = skill_retriever.retrieve(query, top_k=top_k)

        if not res.matches:
            return f"No skills matched '{query}'. Pool: {res.total_pool} skills."

        lines = [f"Top {len(res.matches)} skills for '{query}':"]
        for m in res.matches:
            it = m.item
            hitl = " ⚠HITL" if it.get("hitl") else ""
            lines.append(
                f"  [{m.id}]{hitl} (score={m.score:.2f}) — "
                f"{it.get('description','')[:120]}"
            )
            lines.append(
                f"    Use [SKILL_LOAD:{m.id}] to read the full procedural guide."
            )
        return "\n".join(lines)

    return MetaTool(
        name=name,
        description="Search for skills (procedural guides) matching a description",
        handler=_handler,
        parameters={
            "query": "natural-language description of the task",
            "top_k": "max results to return (default 3)",
        },
        tags=["meta", "discovery"],
        examples=[
            '[TOOL:list_skills] {"query": "diagnose RADIUS authentication failure"}',
        ],
    )


def make_tool_details_meta_tool(
    tool_metadata_provider: Callable[[], dict[str, dict[str, Any]]],
    *,
    name:                   str = "tool_details",
) -> MetaTool:
    """Factory: meta-tool returning full description + parameters for one tool.

    Args:
        tool_metadata_provider: zero-arg callable returning the canonical
                                {name: {description, parameters, tags, hitl}} dict.
                                Re-evaluated on every call so dynamic registration
                                of new tools is visible immediately.
    """
    async def _handler(**kwargs) -> str:
        tool_name = str(kwargs.get("tool_name", "") or "").strip()
        if not tool_name:
            return "Error: tool_details requires 'tool_name' parameter."
        meta = tool_metadata_provider() or {}
        info = meta.get(tool_name)
        if info is None:
            available = ", ".join(sorted(meta.keys())[:10])
            return (
                f"No tool named '{tool_name}'. "
                f"Did you mean one of: {available}? "
                f"Use [TOOL:list_tools] to search by description."
            )
        lines = [
            f"Tool: {tool_name}",
            f"Description: {info.get('description', '(no description)')}",
        ]
        if info.get("hitl"):
            lines.append("⚠ HITL approval required before execution")
        params = info.get("parameters") or {}
        if params:
            lines.append("Parameters:")
            for p, d in params.items():
                lines.append(f"  - {p}: {d}")
        if info.get("returns"):
            lines.append(f"Returns: {info['returns']}")
        if info.get("tags"):
            lines.append(f"Tags: {', '.join(info['tags'])}")
        return "\n".join(lines)

    return MetaTool(
        name=name,
        description="Get full description, parameters, and HITL requirement for one named tool",
        handler=_handler,
        parameters={"tool_name": "exact tool name (e.g. get_device_config)"},
        tags=["meta", "discovery"],
        examples=['[TOOL:tool_details] {"tool_name": "edit_device_config"}'],
    )
