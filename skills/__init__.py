"""
skills/
────────
Skill definitions for mock and pragmatic modes.

Entry point: skills.loader.SkillLoader.skill_definitions()
  Returns the correct set for the running mode — no filtering needed.

Implementation:
  skills/builtin/<name>/SKILL.md — always-available skills (standard format)
  profiles/<id>/skills/<name>/SKILL.md — business skills (lan, dc, …)
  skills/pragmatic/<name>/SKILL.md — pragmatic-mode skills (standard format)
  skills/catalog.py              — SkillCatalogService (register, load, format)
  skills/evolver.py              — runtime skill creation from LLM
"""
from .catalog import (
    Skill,
    SkillSummary,
    SkillDetail,
    SkillCatalogService,
)

__all__ = [
    "Skill",
    "SkillSummary",
    "SkillDetail",
    "SkillCatalogService",
]

from skills.loader import SkillLoader  # public re-export
