"""
scripts/migrate_skills_to_standard.py
─────────────────────────────────────
Normalize / re-serialize Anthropic-standard SKILL.md skill folders.

Layout (the authoritative source of truth):
    skills/builtin/<name>/SKILL.md
    skills/pragmatic/<name>/SKILL.md
    profiles/lan/skills/<name>/SKILL.md
    profiles/dc/skills/<name>/SKILL.md

What this does
--------------
The original one-shot migration (from the legacy in-code registries) is
complete — those dicts have been deleted and SKILL.md folders are now the
only source. This script is retained as a maintenance utility that:

  - validates every SKILL.md against the Anthropic standard
    (skills.skill_format.validate_frontmatter), and
  - optionally re-writes each file through the canonical serializer so the
    on-disk form stays consistent after hand-edits (`--normalize`).

Run:
    python scripts/migrate_skills_to_standard.py             # validate only
    python scripts/migrate_skills_to_standard.py --normalize # validate + rewrite
    python scripts/migrate_skills_to_standard.py --dry-run --normalize

Exit code is non-zero if any SKILL.md fails validation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


from skills.skill_format import (
    flat_dict_to_skill_md,
    load_skill_md,
    parse_skill_md,
)

# Folders that hold standard SKILL.md skill directories.
SKILL_ROOTS = [
    Path("skills/builtin"),
    Path("skills/pragmatic"),
    Path("profiles/lan/skills"),
    Path("profiles/dc/skills"),
]


def _iter_skill_md(repo_root: Path):
    for rel in SKILL_ROOTS:
        base = repo_root / rel
        if not base.exists():
            continue
        for md in sorted(base.glob("*/SKILL.md")):
            yield md


def run(repo_root: Path, *, normalize: bool, dry_run: bool) -> int:
    n_ok = n_bad = n_written = 0
    for md in _iter_skill_md(repo_root):
        hint = md.parent.name.replace("-", "_")
        try:
            # Validate (raises SkillFormatError on any standard violation).
            parse_skill_md(md.read_text(encoding="utf-8"))
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"INVALID  {md}: {exc}")
            n_bad += 1
            continue

        if normalize:
            skill_id, defn = load_skill_md(md.read_text(encoding="utf-8"), skill_id_hint=hint)
            canonical = flat_dict_to_skill_md(skill_id, defn)
            if dry_run:
                print(f"[dry-run] would normalize {md}")
            else:
                md.write_text(canonical, encoding="utf-8")
                print(f"normalized {md}")
            n_written += 1

    print(
        f"\n{n_ok} valid, {n_bad} invalid"
        + (f", {n_written} {'would be ' if dry_run else ''}normalized" if normalize else "")
    )
    return 1 if n_bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--normalize", action="store_true",
                    help="rewrite each SKILL.md through the canonical serializer")
    ap.add_argument("--dry-run", action="store_true", help="preview only")
    ap.add_argument("--root", default=".", help="repo root (default: cwd)")
    args = ap.parse_args(argv)
    return run(Path(args.root).resolve(), normalize=args.normalize, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
