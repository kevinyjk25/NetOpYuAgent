"""
skills/
────────
Skill definitions for mock and pragmatic modes.

Entry point: tools.loader.ToolLoader.skill_definitions()
  Returns the correct set for the running mode — no filtering needed.

Implementation:
  skills/builtin/registry.py     — always-available skills
  skills/mock/registry.py        — mock-mode skills
  skills/pragmatic/registry.py   — pragmatic-mode skills
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
