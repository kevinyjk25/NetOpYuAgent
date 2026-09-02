"""Expose canonical profile SKILL.md files to DSH and Hermes adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from skills import SkillLoader
from skills.skill_format import parse_skill_md
from network_runtime.workflows import compile_workflow_templates


def build_skill_manifest(profile_id: str, mode: str) -> dict[str, Any]:
    definitions = SkillLoader(mode=mode, profile=profile_id).skill_definitions()
    workflows = compile_workflow_templates(profile_id, mode)
    skills: list[dict[str, Any]] = []
    for skill_id, definition in sorted(definitions.items()):
        directory = Path(definition["skill_dir"]).resolve()
        path = directory / "SKILL.md"
        parsed = parse_skill_md(path.read_text(encoding="utf-8"))
        skill = {
            "id": skill_id,
            "name": str(parsed.frontmatter["name"]),
            "description": str(parsed.frontmatter["description"]),
            "content": parsed.body,
            "path": str(path),
            "resource_base": str(directory),
            "metadata": parsed.frontmatter.get("metadata", {}),
        }
        workflow = workflows.get(skill["name"])
        if workflow is not None:
            skill["network_workflow"] = workflow.to_dict()
        skills.append(skill)
    return {"profile": profile_id, "mode": mode, "skills": skills}
