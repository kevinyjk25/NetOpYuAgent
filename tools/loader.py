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

Usage (from the DSH backend)
--------------------
    from tools.loader import ToolLoader
    loader = ToolLoader(mode="pragmatic")          # or "mock"
    callables = loader.build_callables()           # {name: async_fn}
    metadata  = loader.build_metadata()            # {name: {description, parameters, ...}}
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

    def __init__(self, mode: str, profile: str = "default") -> None:
        self._mode = mode.lower().strip()
        if self._mode not in ("mock", "pragmatic"):
            raise ValueError(f"ToolLoader: unknown mode {mode!r} — expected 'mock' or 'pragmatic'")
        # Business profile selects which domain tools load. "default" = none.
        # In pragmatic mode the real device tools come from pragmatic_tools.py
        # regardless of profile (profile only gates the MOCK business tools);
        # this keeps pragmatic mode working as before during the transition.
        self._profile_id = (profile or "default").strip().lower()

    # ── Public API ────────────────────────────────────────────────────────────

    def build_callables(self) -> dict[str, Callable]:
        """
        Return {tool_name: async_callable} for all tools active in this
        mode + profile. Includes: common tools + profile business tools.

        Does NOT include read_stored_result/process_stored_chunks — these are
        injected later by the DSH backend after ToolResultStore is initialised.

        mock mode      → common + profile.tool_callables (business)
        pragmatic mode → common + real pragmatic tools (profile-independent
                         for now; pragmatic device tooling isn't split by
                         profile yet)
        """
        callables: dict[str, Callable] = {}

        if self._mode == "mock":
            from profiles import load_profile
            profile = load_profile(self._profile_id)
            callables.update(profile.tool_callables)
            logger.info(
                "ToolLoader[mock, profile=%s]: loaded %d business callable(s)",
                self._profile_id, len(profile.tool_callables),
            )
        else:
            from tools.pragmatic_tools import PRAGMATIC_TOOL_REGISTRY as REAL_CALLABLES
            callables.update(REAL_CALLABLES)
            logger.info("ToolLoader[pragmatic]: loaded %d real callables", len(REAL_CALLABLES))

        return callables

    def build_metadata(self) -> dict[str, dict[str, Any]]:
        """
        Return {tool_name: {description, parameters, returns, hitl, tags}}
        for all tools active in this mode + profile.

        This dictionary is the canonical source for DSH tool projection.

        Always includes the common builtin tools (read_stored_result etc.);
        adds the active profile's business tool metadata (mock) or the
        pragmatic tool metadata (pragmatic mode).
        """
        from tools.builtin.registry import TOOLS as BUILTIN_TOOLS

        meta: dict[str, dict[str, Any]] = {}
        meta.update(BUILTIN_TOOLS)

        if self._mode == "mock":
            from profiles import load_profile
            profile = load_profile(self._profile_id)
            meta.update(profile.tool_metadata)
        else:
            from tools.pragmatic.registry import TOOLS as PRAGMA_TOOLS
            meta.update(PRAGMA_TOOLS)

        logger.info(
            "ToolLoader[%s, profile=%s]: %d tools in metadata",
            self._mode, self._profile_id, len(meta),
        )
        return meta

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
