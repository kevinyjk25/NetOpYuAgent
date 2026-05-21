"""
skills/loader.py
----------------
SkillLoader — mode-aware skill registration entry point.

Mirrors tools/loader.py's pattern but for SKILLS. This separation matters:
  - skills/ owns skill definitions (built-in + mode-specific)
  - tools/  owns tool callables and metadata
  - Neither should know about the other

Usage:
    loader = SkillLoader(mode="mock")        # or "pragmatic"
    defs   = loader.skill_definitions()       # {skill_id: {...flat dict...}}

    catalog = SkillCatalogService()
    catalog.register_all(defs)

The mode-dispatch is identical to ToolLoader for consistency.
"""
from __future__ import annotations

import logging
from typing import Any


logger = logging.getLogger(__name__)


class SkillLoader:
    """Mode-aware skill registry loader.

    Independent of ToolLoader — uses the same mode parameter ("mock" |
    "pragmatic") but reads from skills/{builtin,mock,pragmatic}/registry.py.
    """

    def __init__(self, mode: str = "mock", profile: str = "default"):
        self._mode = mode
        self._profile_id = (profile or "default").strip().lower()

    def skill_definitions(self) -> dict[str, dict[str, Any]]:
        """Return {skill_id: {...flat dict...}} for all skills active in this
        mode + profile.

        Always includes the common builtin skills; adds the active profile's
        business skills (mock mode) or the pragmatic skills (pragmatic mode).
        """
        from skills.builtin.registry import SKILLS as BUILTIN_SKILLS

        defs: dict[str, dict[str, Any]] = {}
        defs.update(BUILTIN_SKILLS)

        if self._mode == "mock":
            from profiles import load_profile
            profile = load_profile(self._profile_id)
            defs.update(profile.skill_defs)
            logger.info(
                "SkillLoader[mock, profile=%s]: %d skills loaded",
                self._profile_id, len(defs),
            )
        else:
            from skills.pragmatic.registry import SKILLS as PRAGMA_SKILLS
            defs.update(PRAGMA_SKILLS)
            logger.info("SkillLoader[pragmatic]: %d skills loaded", len(defs))

        return defs
