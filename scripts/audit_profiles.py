#!/usr/bin/env python3
"""
scripts/audit_profiles.py
-------------------------
Static audit for the business profile layer (profiles/).

Checks, for every known profile (default / lan / dc):

  1. The profile module exposes a module-level PROFILE: Profile.
  2. tool_callables and tool_metadata reference the SAME set of tool names.
     (callable-only → tool never shows in the prompt; metadata-only → tool
      404s at dispatch. Either is a profile bug.)
  3. No two business profiles (lan, dc, …) share a tool name. Role
     isolation is the precondition for meaningful A2A delegation — if two
     profiles both had `list_devices`, "delegate to the other agent" would
     be ambiguous and the isolation guarantee would be false.
  4. The default profile carries NO business tools/skills (the decoupling
     proof — the framework must run with the business layer fully removed).
  5. The framework does NOT hard-import a specific profile package
     (profiles.lan / profiles.dc). All access must go through the
     profiles.load_profile factory, keeping the dependency arrow
     framework → profiles (never framework → a specific business domain).

Run:
    python scripts/audit_profiles.py

Exits 0 on success, 1 on any violation.
"""
from __future__ import annotations

import pathlib
import re
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
from _audit_common import iter_repo_python_files  # noqa: E402


def main() -> int:
    errors: list[str] = []

    # ── Checks 1-4: load each profile and inspect ────────────────────────
    try:
        from profiles import load_profile, available_profiles
    except Exception as exc:
        print(f"  ✗ cannot import profiles package: {exc}", file=sys.stderr)
        return 1

    business_tool_owner: dict[str, str] = {}   # tool_name → profile_id
    business_skill_owner: dict[str, str] = {}

    for pid in available_profiles():
        try:
            p = load_profile(pid)
        except Exception as exc:
            errors.append(f"profile {pid!r} failed to load: {exc}")
            continue

        # Check 2 — callable/metadata alignment
        cb = set(p.tool_callables)
        md = set(p.tool_metadata)
        if cb != md:
            only_cb = cb - md
            only_md = md - cb
            if only_cb:
                errors.append(
                    f"profile {pid!r}: tools with callable but NO metadata "
                    f"(won't appear in prompt): {sorted(only_cb)}"
                )
            if only_md:
                errors.append(
                    f"profile {pid!r}: tools with metadata but NO callable "
                    f"(will 404 at dispatch): {sorted(only_md)}"
                )

        # Check 4 — default profile must be empty of business
        if pid == "default":
            if p.tool_callables:
                errors.append(
                    f"default profile must have NO business tools, found: "
                    f"{sorted(p.tool_callables)}"
                )
            if p.skill_defs:
                errors.append(
                    f"default profile must have NO business skills, found: "
                    f"{sorted(p.skill_defs)}"
                )
            continue   # don't register default's (empty) tools below

        # Check 3 — cross-profile tool isolation
        for tool_name in p.tool_callables:
            if tool_name in business_tool_owner:
                errors.append(
                    f"tool {tool_name!r} appears in BOTH profile "
                    f"{business_tool_owner[tool_name]!r} and {pid!r} — "
                    f"business profiles must have disjoint tools (role isolation)"
                )
            else:
                business_tool_owner[tool_name] = pid
        for skill_id in p.skill_defs:
            if skill_id in business_skill_owner:
                errors.append(
                    f"skill {skill_id!r} appears in BOTH profile "
                    f"{business_skill_owner[skill_id]!r} and {pid!r}"
                )
            else:
                business_skill_owner[skill_id] = pid
            # Skill catalog (SkillCatalogService.register_all) requires a
            # 'name' field; a skill missing it crashes catalog build at boot.
            sk = p.skill_defs[skill_id]
            if not isinstance(sk, dict) or not sk.get("name"):
                errors.append(
                    f"profile {pid!r} skill {skill_id!r} is missing the "
                    f"required 'name' field (SkillCatalogService.register_all "
                    f"would crash at boot)"
                )

    # ── Check 5 — framework must not hard-import a specific profile ──────
    # Allowed: `from profiles import load_profile` / `import profiles`.
    # Forbidden anywhere outside profiles/ itself:
    #   `from profiles.lan ...`, `import profiles.dc`, etc.
    bad_import = re.compile(r"(?:from|import)\s+profiles\.(lan|dc|default)\b")
    for f in iter_repo_python_files(pathlib.Path(".")):
        sf = str(f)
        if "/profiles/" in sf or sf.startswith("profiles/"):
            continue   # profiles may import their own submodules
        if "/tests/" in sf or sf.startswith("tests/"):
            continue   # tests may import a specific profile to assert on it
        if "/scripts/" in sf or sf.startswith("scripts/"):
            continue   # audit scripts mention the patterns as string literals
        try:
            text = open(f).read()
        except Exception:
            continue
        for m in bad_import.finditer(text):
            errors.append(
                f"{sf}: framework hard-imports profiles.{m.group(1)} — "
                f"use profiles.load_profile(profile_id) instead so the "
                f"framework stays business-agnostic"
            )

    # ── Check 6 — every ToolLoader()/SkillLoader() call must pass profile= ──
    # The loaders default profile to "default" (no business tools). A call
    # that forgets profile= silently degrades to the empty default profile —
    # this exact bug shipped twice (webui rebuild, llm_engine fallback) and
    # made a dc/lan agent lose all its business tools at runtime. Enforce
    # statically: any `ToolLoader(` / `SkillLoader(` construction must include
    # `profile=` on the same logical call.
    loader_call = re.compile(
        r"\b(?:ToolLoader|SkillLoader|_TL|_TL2|_TL_be|_SL)\s*\(([^)]*)\)"
    )
    for f in iter_repo_python_files(pathlib.Path(".")):
        sf = str(f)
        if "/scripts/" in sf or sf.startswith("scripts/"):
            continue
        if "/tests/" in sf:
            continue   # tests construct loaders with explicit profile kwargs already
        try:
            text = open(f).read()
        except Exception:
            continue
        for m in loader_call.finditer(text):
            args = m.group(1)
            # Skip zero-arg or self-referential matches like definitions.
            if not args.strip():
                continue
            # Skip matches inside a docstring or comment line — these are
            # usage examples in prose, not real construction calls. We detect
            # this by checking whether the match's line is inside a triple-
            # quoted block or starts with a comment / contains the call only
            # as documentation.
            line_start = text.rfind("\n", 0, m.start()) + 1
            line_end = text.find("\n", m.start())
            line_text = text[line_start: line_end if line_end != -1 else len(text)]
            stripped = line_text.strip()
            if stripped.startswith("#") or stripped.startswith('"') or stripped.startswith("'"):
                continue
            # Count unescaped triple-quotes before the match — odd means we're
            # inside a docstring.
            before = text[: m.start()]
            if (before.count('"""') + before.count("'''")) % 2 == 1:
                continue
            # Must mention profile= in the call args.
            if "profile=" not in args:
                # Compute line number for a useful message.
                line_no = text[: m.start()].count("\n") + 1
                errors.append(
                    f"{sf}:{line_no}: loader call {m.group(0)[:60]!r} is "
                    f"missing profile= — it will silently use the empty "
                    f"'default' profile. Pass profile=cfg.agent.profile."
                )

    # ── Report ───────────────────────────────────────────────────────────
    if errors:
        for e in errors:
            print(f"  ✗ {e}", file=sys.stderr)
        print(f"\naudit_profiles: ✗ FAIL — {len(errors)} issue(s)", file=sys.stderr)
        return 1

    n_tools = len(business_tool_owner)
    n_profiles = len(available_profiles())
    print(
        f"audit_profiles: ✓ PASS — {n_profiles} profiles, "
        f"{n_tools} business tools, all isolated + consistent."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
