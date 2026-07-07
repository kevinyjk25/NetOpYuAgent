"""
profiles/base.py — Business profile abstraction
================================================

A **Profile** packages everything that makes an agent instance domain-specific:
its business tools (callables + prompt-facing metadata), its business skills
(SOPs), and the capabilities it advertises to peers. The common agent
framework (runtime/, a2a/, registry/, hitl_core/, retrieval/, integrations/,
tools/builtin, skills/builtin) knows NOTHING about LAN vs DC vs anything else —
it only knows how to load "the active profile".

Why this split exists
─────────────────────
Before profiles, business tools (`list_devices`, `edit_device_config`, …) lived
directly in `tools/mock_tools.py` and the framework imported them by name. That
coupled the generic agent loop to one specific business domain (enterprise LAN
Cisco gear). Adding a second domain (data-center fabric) — or running a pure
no-business "assistant" agent — meant editing framework files.

With profiles:
  - `AGENT_PROFILE=default`  → no business tools/skills, just common meta tools
  - `AGENT_PROFILE=lan`      → enterprise LAN Cisco tools + LAN SOPs
  - `AGENT_PROFILE=dc`       → data-center fabric tools + DC SOPs
  - `AGENT_PROFILE=wan`      → wide-area SD-WAN / transport tools + WAN SOPs
…all from the SAME process image. Tool isolation between roles is then a
natural consequence: a `lan` agent's registry simply doesn't contain the `dc`
tools, so it physically cannot call them (it must delegate via A2A — Phase 2B).

Contract
────────
A profile module under `profiles/<id>/` must expose a module-level
`PROFILE: Profile`. The ProfileRegistry discovers it by id. Profiles must NOT
import framework internals beyond the lightweight types they need (this file +
config dataclasses); keeping the dependency arrow pointing
framework → profile (never the reverse) is what keeps the business layer
swappable.
"""
from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class Profile:
    """Everything domain-specific about an agent instance.

    Fields
    ------
    profile_id:
        Stable identifier, e.g. "lan" / "dc" / "default". Matches the
        directory name under profiles/ and the AGENT_PROFILE env value.
    display_name / description:
        Human-facing labels. If the operator doesn't override
        agent.display_name / agent.description in config, the framework
        falls back to these so each profile has a sensible identity.
    domain_tags:
        Coarse domain labels (e.g. ["lan","switch","ap"]). Used by Phase-2B
        capability matching to decide which peer should handle a query.
    tool_callables:
        {tool_name: async def tool(args: dict) -> str}. The business tool
        implementations. Merged with the common builtin tools by ToolLoader.
    tool_metadata:
        {tool_name: {description, parameters, returns, hitl, action_type,
        tags, ...}}. Prompt-facing declarations; this is what the LLM sees.
        Keys MUST match tool_callables (validated at registration).
    skill_defs:
        {skill_id: {...skill definition dict...}}. Business SOPs. Merged
        with common builtin skills by SkillLoader.
    capabilities:
        AgentCard skills advertised to peers. If the operator leaves
        agent.capabilities empty in config, the framework publishes these.
        Each entry is a plain dict: {skill_id, name, description, tags}.
    """
    profile_id:     str
    display_name:   str = ""
    description:    str = ""
    domain_tags:    list[str] = field(default_factory=list)
    tool_callables: dict[str, Callable] = field(default_factory=dict)
    tool_metadata:  dict[str, dict[str, Any]] = field(default_factory=dict)
    skill_defs:     dict[str, dict[str, Any]] = field(default_factory=dict)
    capabilities:   list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.display_name:
            self.display_name = f"{self.profile_id} agent"
        # Validate callable/metadata key alignment early — a tool with a
        # callable but no metadata won't appear in the prompt; metadata
        # with no callable will 404 at dispatch. Either is a profile bug.
        cb = set(self.tool_callables)
        md = set(self.tool_metadata)
        missing_meta = cb - md
        missing_cb   = md - cb
        if missing_meta:
            logger.warning(
                "Profile %r: tools with callable but NO metadata (won't show "
                "in prompt): %s", self.profile_id, sorted(missing_meta),
            )
        if missing_cb:
            logger.warning(
                "Profile %r: tools with metadata but NO callable (will 404 "
                "on dispatch): %s", self.profile_id, sorted(missing_cb),
            )

    def tool_names(self) -> list[str]:
        return sorted(self.tool_callables.keys())


# ── Registry ─────────────────────────────────────────────────────────────
# Profiles are discovered lazily by id so importing this module doesn't drag
# in every business domain's dependencies. profiles/<id>/__init__.py must
# expose a module-level `PROFILE: Profile`.

_KNOWN_PROFILE_IDS = ("default", "lan", "dc", "wan")
_cache: dict[str, Profile] = {}


def available_profiles() -> tuple[str, ...]:
    """Profile ids the framework knows how to load."""
    return _KNOWN_PROFILE_IDS


def load_profile(profile_id: str) -> Profile:
    """Load (and cache) the Profile for the given id.

    Falls back to the 'default' profile (no business tools/skills) when the
    id is unknown, so a typo in AGENT_PROFILE degrades to a safe assistant
    rather than crashing the boot.
    """
    pid = (profile_id or "default").strip().lower()
    if pid in _cache:
        return _cache[pid]

    if pid not in _KNOWN_PROFILE_IDS:
        logger.warning(
            "Unknown AGENT_PROFILE %r — falling back to 'default' (no business "
            "tools/skills). Known profiles: %s", pid, _KNOWN_PROFILE_IDS,
        )
        pid = "default"

    try:
        mod = importlib.import_module(f"profiles.{pid}")
        profile = getattr(mod, "PROFILE", None)
        if not isinstance(profile, Profile):
            raise TypeError(
                f"profiles.{pid} must expose a module-level PROFILE: Profile"
            )
    except Exception as exc:
        logger.error(
            "Failed to load profile %r (%s) — falling back to 'default'",
            pid, exc,
        )
        if pid != "default":
            return load_profile("default")
        raise

    _cache[pid] = profile
    logger.info(
        "Loaded profile %r: %d tool(s), %d skill(s), %d capability(ies)",
        profile.profile_id, len(profile.tool_callables),
        len(profile.skill_defs), len(profile.capabilities),
    )
    return profile
