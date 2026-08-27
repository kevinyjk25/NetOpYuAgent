"""Expose the active NetOpYu profile's canonical SKILL.md files to DSH."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from skills import SkillLoader
from skills.skill_format import parse_skill_md


def build_skill_manifest(profile_id: str, mode: str) -> dict[str, Any]:
    definitions = SkillLoader(mode=mode, profile=profile_id).skill_definitions()
    skills: list[dict[str, Any]] = []
    for skill_id, definition in sorted(definitions.items()):
        directory = Path(definition["skill_dir"]).resolve()
        path = directory / "SKILL.md"
        parsed = parse_skill_md(path.read_text(encoding="utf-8"))
        skills.append({
            "id": skill_id,
            "name": str(parsed.frontmatter["name"]),
            "description": str(parsed.frontmatter["description"]),
            "content": parsed.body,
            "path": str(path),
            "resource_base": str(directory),
            "metadata": parsed.frontmatter.get("metadata", {}),
        })
    return {"profile": profile_id, "mode": mode, "skills": skills}
