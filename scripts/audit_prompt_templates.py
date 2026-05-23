#!/usr/bin/env python3
"""
scripts/audit_prompt_templates.py
---------------------------------
Static audit: every str.format() prompt template must have its literal
braces escaped (`{{` and `}}`). Placeholders must be one of the documented
names whitelisted below.

Catches the class of bug where adding a JSON example like

    [TOOL:read_stored_result] {"ref_id": "abc"}

without escaping the JSON braces causes `str.format()` to raise KeyError
at runtime ('"ref_id"' is not a placeholder name).

Run on every CI build:
    python scripts/audit_prompt_templates.py

Exits 0 if clean, 1 if any unescaped non-placeholder braces found.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path


# Files known to define str.format() templates
TEMPLATE_FILES = [
    "integrations/clients/llm_engine.py",
]

# Whitelist of valid placeholder names (extend when you add new ones)
VALID_PLACEHOLDERS = {
    "extra_tools_section",
    "skill_summary",
    "confirmed_facts_section",
    "peers_section",   # AVAILABLE PEERS list (Phase 2B peer-aware prompt, 2026-05)
}

# Regex for "all caps template var assignments" — TOOL_CALL_SYSTEM,
# TOOL_CALL_SYSTEM_SLIM, etc.
TEMPLATE_VAR_RE = re.compile(r'^\s*([A-Z][A-Z0-9_]*)\s*=\s*"""', re.MULTILINE)


def find_unescaped_braces(template: str, label: str = "") -> list[str]:
    """Walk the template, return list of human-readable warnings for any
    single (unescaped) brace that isn't a known placeholder.
    """
    problems: list[str] = []
    i = 0
    line_no = 1
    while i < len(template):
        c = template[i]
        if c == "\n":
            line_no += 1
            i += 1
            continue

        if c == "{":
            if i + 1 < len(template) and template[i + 1] == "{":
                i += 2
                continue
            m = re.match(r"\{(\w+)\}", template[i:])
            if m and m.group(1) in VALID_PLACEHOLDERS:
                i += m.end()
                continue
            # Unescaped brace
            preview = template[max(0, i - 25):i + 45].replace("\n", " | ")
            problems.append(
                f"{label}line ~{line_no}: unescaped '{{' near `{preview}` "
                f"— escape literal braces as '{{{{' or whitelist a placeholder name"
            )
            i += 1
            continue

        if c == "}":
            if i + 1 < len(template) and template[i + 1] == "}":
                i += 2
                continue
            # If preceded by a placeholder, the placeholder-handling above
            # already consumed both braces. A bare } here means trouble.
            preview = template[max(0, i - 25):i + 45].replace("\n", " | ")
            problems.append(
                f"{label}line ~{line_no}: unescaped '}}' near `{preview}`"
            )
            i += 1
            continue

        i += 1
    return problems




def _is_formatted(src: str, template_name: str) -> bool:
    """Return True if `template_name` is used with `.format(` anywhere in src."""
    # Direct: NAME.format(
    if re.search(rf"\b{re.escape(template_name)}\b\s*\.\s*format\(", src):
        return True
    # Via local var: foo = self.NAME ... foo.format(
    m_assign = re.search(rf"(\w+)\s*=\s*\(?\s*self\.{re.escape(template_name)}\b", src)
    if m_assign:
        local = m_assign.group(1)
        if re.search(rf"\b{re.escape(local)}\s*\.\s*format\(", src):
            return True
    # Via ternary in local var: foo = (X if y else NAME) ... foo.format(
    m_tern = re.search(
        rf"(\w+)\s*=\s*\([^)]*\b{re.escape(template_name)}\b[^)]*\)", src,
    )
    if m_tern:
        local = m_tern.group(1)
        if re.search(rf"\b{re.escape(local)}\s*\.\s*format\(", src):
            return True
    return False


def audit_file(path: Path) -> list[str]:
    src = path.read_text(encoding="utf-8")
    out: list[str] = []
    # Find every template variable assignment with triple-quoted body
    # (we use AST for accuracy)
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as e:
        return [f"{path}: syntax error at line {e.lineno}: {e.msg}"]

    # Walk all class bodies + module-level assigns for `NAME = "..."` where
    # name is ALL_CAPS_WITH_UNDERSCORES (heuristic for prompt templates).
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
            continue
        body = node.value.value
        if not body or "{" not in body:
            continue
        # Only audit if name is ALL CAPS and contains 'SYSTEM' or 'PROMPT' or 'TEMPLATE'
        for t in node.targets:
            if isinstance(t, ast.Name) and re.match(r"^[A-Z][A-Z0-9_]*$", t.id):
                if any(k in t.id for k in ("SYSTEM", "PROMPT", "TEMPLATE", "TPL")):
                    # Only audit templates that are actually used with .format().
                    # Templates passed verbatim to the LLM (e.g. as a system
                    # message) don't need brace escaping.
                    if not _is_formatted(src, t.id):
                        continue
                    problems = find_unescaped_braces(
                        body, label=f"{path.name}:{t.id} ",
                    )
                    out.extend(problems)
    return out


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    print("=" * 70)
    print("Prompt Template Brace Audit")
    print("=" * 70)
    print(f"Whitelisted placeholders: {sorted(VALID_PLACEHOLDERS)}")
    print(f"Files audited: {TEMPLATE_FILES}")
    print()

    all_problems: list[str] = []
    for rel in TEMPLATE_FILES:
        p = repo / rel
        if not p.exists():
            print(f"  SKIP: {rel} not found")
            continue
        problems = audit_file(p)
        all_problems.extend(problems)

    if all_problems:
        print(f"Found {len(all_problems)} issue(s):")
        for p in all_problems:
            print(f"  ✗ {p}")
        print()
        print("Fix options:")
        print("  1. Escape literal braces as '{{' and '}}'")
        print("  2. Add a new placeholder name to VALID_PLACEHOLDERS")
        return 1

    print("✓ All prompt templates have correctly-escaped braces.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
