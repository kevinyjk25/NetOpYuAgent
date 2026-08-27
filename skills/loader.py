"""
skills/loader.py
----------------
SkillLoader — mode-aware skill registration entry point.

As of the SKILL.md standardization (2026-06), skills are loaded from
Anthropic-standard SKILL.md folders rather than in-code Python dicts:

    skills/builtin/<name>/SKILL.md          — always-available
    skills/pragmatic/<name>/SKILL.md        — pragmatic mode
    profiles/<id>/skills/<name>/SKILL.md    — profile business SOPs

Each SKILL.md is parsed by skills.skill_format into the SAME flat-dict shape
the rest of the system already consumes (SkillCatalogService.register_all,
the retriever corpus, golden_set expected_ids, [SKILL_LOAD:id]). The
snake_case skill_id is preserved via metadata.skill_id, so no downstream
consumer changes.

Usage:
    loader = SkillLoader(mode="mock", profile="lan")
    defs   = loader.skill_definitions()       # {skill_id: {...flat dict...}}

    catalog = SkillCatalogService()
    catalog.register_all(defs)

Fail-soft policy:
  - A malformed business/pragmatic SKILL.md is logged and SKIPPED (one bad
    skill must not take down the boot).
  - A malformed builtin SKILL.md is fatal — builtins are core capability.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

from skills.skill_format import SkillFormatError, load_skill_md

logger = logging.getLogger(__name__)

# Repo-root-relative skill folders. Resolved against the package location so
# the loader works regardless of the process CWD.
_SKILLS_PKG_DIR = Path(__file__).resolve().parent          # .../skills
_REPO_ROOT      = _SKILLS_PKG_DIR.parent                    # repo root


class SkillLoader:
    """Mode-aware, profile-aware SKILL.md folder loader.

    Independent of ToolLoader — uses the same mode parameter ("mock" |
    "pragmatic") and the same profile id.
    """

    def __init__(
        self,
        mode: str = "mock",
        profile: str = "default",
        skills_root: Optional[str | Path] = None,
    ):
        self._mode = mode
        self._profile_id = (profile or "default").strip().lower()
        self._root = Path(skills_root).resolve() if skills_root else _REPO_ROOT

    # ── public API (unchanged signature) ──────────────────────────────────

    def skill_definitions(self) -> dict[str, dict[str, Any]]:
        """Return {skill_id: {...flat dict...}} for all skills active in this
        mode + profile.

        Always includes the common builtin skills; adds the active profile's
        business skills (mock mode) or the pragmatic skills (pragmatic mode).
        """
        defs: dict[str, dict[str, Any]] = {}

        # Builtins first — fatal on error (core capability).
        defs.update(self._load_folder(self._root / "skills" / "builtin", fatal=True))

        if self._mode == "mock":
            prof_dir = self._root / "profiles" / self._profile_id / "skills"
            biz = self._load_folder(prof_dir, fatal=False)
            defs.update(biz)
            logger.info(
                "SkillLoader[mock, profile=%s]: %d skill(s) loaded (%d business)",
                self._profile_id, len(defs), len(biz),
            )
        else:
            prag = self._load_folder(self._root / "skills" / "pragmatic", fatal=False)
            defs.update(prag)
            lab: dict[str, dict[str, Any]] = {}
            try:
                from config import load

                if load(os.environ.get("NETOPYU_CONFIG_PATH", "config.yaml")).pragmatic.lab.enabled:
                    from network_lab import load_manifest

                    cfg = load(os.environ.get("NETOPYU_CONFIG_PATH", "config.yaml"))
                    manifest = load_manifest(cfg.pragmatic.lab.manifest)
                    capabilities: set[str] = set()
                    if manifest.users and manifest.applications:
                        capabilities.add("access")
                    if manifest.links:
                        capabilities.add("topology")
                    discovered = self._load_folder(self._root / "skills" / "lab", fatal=False)
                    lab = {
                        skill_id: definition
                        for skill_id, definition in discovered.items()
                        if not definition.get("profiles")
                        or self._profile_id in definition["profiles"]
                        if not definition.get("lab_capability")
                        or definition["lab_capability"] in capabilities
                    }
                    defs.update(lab)
            except (OSError, ValueError):
                # Backend construction performs strict config validation. Skill
                # discovery remains fail-soft as documented for business skills.
                logger.warning("SkillLoader: lab skill discovery skipped due to invalid config")
            logger.info(
                "SkillLoader[pragmatic]: %d skill(s) loaded (%d lab)", len(defs), len(lab),
            )

        return defs

    def profile_skill_definitions(self, profile_id: str) -> dict[str, dict[str, Any]]:
        """Load ONLY a profile's business skills (no builtins).

        Used by profiles.base.load_profile to populate Profile.skill_defs.
        """
        pid = (profile_id or "default").strip().lower()
        prof_dir = self._root / "profiles" / pid / "skills"
        return self._load_folder(prof_dir, fatal=False)

    # ── internals ─────────────────────────────────────────────────────────

    def _load_folder(self, folder: Path, *, fatal: bool) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        if not folder.exists():
            logger.debug("SkillLoader: no skill folder at %s", folder)
            return out

        for skill_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
            md_path = skill_dir / "SKILL.md"
            if not md_path.exists():
                continue
            try:
                text = md_path.read_text(encoding="utf-8")
                # folder name (kebab) → snake_case hint, overridden by
                # metadata.skill_id when present.
                hint = skill_dir.name.replace("-", "_")
                skill_id, defn = load_skill_md(text, skill_id_hint=hint)
                # Anthropic-standard folder layout: capture the skill's source
                # directory + bundled resources so runtime can locate scripts
                # (to execute) and references/assets (to read on demand).
                defn["skill_dir"] = str(skill_dir.resolve())
                defn["scripts"] = sorted(
                    p.name for p in (skill_dir / "scripts").glob("*.py")
                ) if (skill_dir / "scripts").is_dir() else []
                defn["references"] = sorted(
                    p.name for p in (skill_dir / "references").iterdir() if p.is_file()
                ) if (skill_dir / "references").is_dir() else []
                defn["assets"] = sorted(
                    p.name for p in (skill_dir / "assets").iterdir() if p.is_file()
                ) if (skill_dir / "assets").is_dir() else []
                if skill_id in out:
                    logger.warning(
                        "SkillLoader: duplicate skill_id %r (folder %s) — overwriting",
                        skill_id, skill_dir.name,
                    )
                out[skill_id] = defn
            except (SkillFormatError, OSError, UnicodeDecodeError) as exc:
                if fatal:
                    logger.error(
                        "SkillLoader: FATAL — builtin skill %s is malformed: %s",
                        md_path, exc,
                    )
                    raise
                logger.warning(
                    "SkillLoader: skipping malformed skill %s: %s", md_path, exc
                )
        return out
