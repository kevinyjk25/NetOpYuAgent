"""
tools/loader.py
───────────────
ToolLoader: assembles the active tool registry and metadata by mode.

Principles
----------
- Information density : only tools valid for the current mode reach the prompt.
- Minimal tool set    : no mock tools visible in pragmatic mode; no pragmatic
                        tool metadata visible in mock mode.
- Layered memory      : builtin tools are always present; mode tools layer on top;
                        registered (MCP/OpenAPI) tools layer on top of those.

Usage (from main.py)
--------------------
    from tools.loader import ToolLoader
    loader = ToolLoader(mode="pragmatic")          # or "mock"
    callables = loader.build_callables()           # {name: async_fn}
    metadata  = loader.build_metadata()            # {name: {description, parameters, ...}}
    skill_defs = loader.skill_definitions()        # {skill_id: {...}} for SkillCatalogService
"""
from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ToolLoader:
    """
    Assembles the tool registry and skill definitions for a given mode.

    mode: "mock" | "pragmatic"
    """

    def __init__(self, mode: str) -> None:
        self._mode = mode.lower().strip()
        if self._mode not in ("mock", "pragmatic"):
            raise ValueError(f"ToolLoader: unknown mode {mode!r} — expected 'mock' or 'pragmatic'")

    # ── Public API ────────────────────────────────────────────────────────────

    def build_callables(self) -> dict[str, Callable]:
        """
        Return {tool_name: async_callable} for all tools active in this mode.
        Includes: builtin tools + mode-specific tools.
        Does NOT include read_stored_result/process_stored_chunks — these are
        injected later by build_services() after ToolResultStore is initialised.
        """
        callables: dict[str, Callable] = {}

        if self._mode == "mock":
            from tools.mock_tools import TOOL_REGISTRY as MOCK_CALLABLES
            callables.update(MOCK_CALLABLES)
            logger.info("ToolLoader[mock]: loaded %d mock callables", len(MOCK_CALLABLES))
        else:
            from tools.pragmatic_tools import PRAGMATIC_TOOL_REGISTRY as REAL_CALLABLES
            callables.update(REAL_CALLABLES)
            logger.info("ToolLoader[pragmatic]: loaded %d real callables", len(REAL_CALLABLES))

        return callables

    def build_metadata(self) -> dict[str, dict[str, Any]]:
        """
        Return {tool_name: {description, parameters, returns, hitl, tags}}
        for all tools active in this mode.

        This dict is the ONLY source used to build the agent's tool section
        in the system prompt. No tool name is hardcoded in llm_engine.py.
        """
        from tools.builtin.registry import TOOLS as BUILTIN_TOOLS

        meta: dict[str, dict[str, Any]] = {}
        meta.update(BUILTIN_TOOLS)

        if self._mode == "mock":
            from tools.mock.registry import TOOLS as MOCK_TOOLS
            meta.update(MOCK_TOOLS)
        else:
            from tools.pragmatic.registry import TOOLS as PRAGMA_TOOLS
            meta.update(PRAGMA_TOOLS)

        logger.info("ToolLoader[%s]: %d tools in metadata", self._mode, len(meta))
        return meta

    def skill_definitions(self) -> dict[str, dict[str, Any]]:
        """
        Return {skill_id: {...}} for all skills active in this mode.

        Used to populate SkillCatalogService — no filtering needed because
        only mode-appropriate skills are loaded to begin with.
        """
        from skills.builtin.registry import SKILLS as BUILTIN_SKILLS

        defs: dict[str, dict[str, Any]] = {}
        defs.update(BUILTIN_SKILLS)

        if self._mode == "mock":
            from skills.mock.registry import SKILLS as MOCK_SKILLS
            defs.update(MOCK_SKILLS)
        else:
            from skills.pragmatic.registry import SKILLS as PRAGMA_SKILLS
            defs.update(PRAGMA_SKILLS)

        logger.info("ToolLoader[%s]: %d skills in definitions", self._mode, len(defs))
        return defs

    def tool_section_for_prompt(self) -> str:
        """
        Build the AVAILABLE TOOLS section of the system prompt dynamically
        from the active tool metadata. Returns a compact multi-line string.

        Format per tool:
          [TOOL:name] — description
            Parameters: param1 (desc), param2 (desc)
            Returns: <returns>
            ⚠ HITL required    (only when hitl=True)
        """
        meta = self.build_metadata()
        lines: list[str] = ["AVAILABLE TOOLS (use [TOOL:name] {\"arg\": \"value\"} format):"]

        # Group by tags for readability
        grouped: dict[str, list[str]] = {}
        for name, info in sorted(meta.items()):
            primary_tag = info.get("tags", ["other"])[0]
            grouped.setdefault(primary_tag, []).append(name)

        for tag in sorted(grouped):
            lines.append(f"\n  [{tag.upper()}]")
            for name in grouped[tag]:
                info = meta[name]
                hitl_note = " ⚠ HITL" if info.get("hitl") else ""
                lines.append(f"  [TOOL:{name}]{hitl_note} — {info['description']}")
                params = info.get("parameters", {})
                if params:
                    param_str = ", ".join(f"{k}: {v}" for k, v in params.items())
                    lines.append(f"    Args: {param_str}")

        return "\n".join(lines)
