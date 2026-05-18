#!/usr/bin/env python3
"""
Audit: directive parsing must go through runtime/directive_parser.

Background
==========
The framework defines an in-band text protocol for [TOOL:name],
[SKILL_LOAD:name], and [TOOL_BATCH:name] directives. We learned the
hard way that re-implementing the parse regex in multiple call sites
caused drift — one site tolerated `[TOOL: name]` (space after colon),
another did not, and the agent would silently terminate mid-task when
a small model emitted a slightly off-canonical format.

This script enforces that all directive parsing goes through one
module: runtime/directive_parser.py. Any other file containing a
strict directive regex (`r"\\[TOOL:"`, `r"\\[SKILL_LOAD:"`,
`r"\\[TOOL_BATCH:"`) is a regression.

Exemptions
==========
  - runtime/directive_parser.py — owns the regex, by design.
  - logging_config.py + integrations/clients/llm_engine.py docstrings
    that mention the syntax for documentation — they're string literals
    NOT compiled as regex, so they're filtered out.
  - examples/ and mock_file/ — illustrative, not parsed.

Exit codes
==========
  0  no strict regex found outside the parser module
  1  at least one offending site exists — print path:line for each
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Files allowed to contain the raw regex (they implement the parser
# themselves) OR files where matches are in docstrings/example text,
# not code paths that get compiled and used.
EXEMPT_PATHS = {
    "runtime/directive_parser.py",
    "scripts/audit_directive_parsing.py",
    "examples/uploads/device_config_tool.py",
    "mock_file/device_config_tool.py",
}

# Patterns we forbid in non-parser code. These are the EXACT
# strict-regex starts that fail to tolerate whitespace.
FORBIDDEN_PATTERNS = [
    re.compile(r'r"\\\[TOOL:'),           # raw-string regex \[TOOL:
    re.compile(r'r"\\\[SKILL_LOAD:'),
    re.compile(r'r"\\\[TOOL_BATCH:'),
    re.compile(r"r'\\\[TOOL:"),           # single-quoted variant
    re.compile(r"r'\\\[SKILL_LOAD:"),
    re.compile(r"r'\\\[TOOL_BATCH:"),
]


def is_doc_or_string_only(line: str) -> bool:
    """True if this is in a docstring or comment, not real regex."""
    s = line.lstrip()
    return s.startswith("#") or s.startswith('"""') or s.startswith("'''")


def main() -> int:
    # Shared iterator handles .venv/site-packages/__pycache__/etc so we
    # don't re-discover the "tools/" name collision with third-party libs.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _audit_common import iter_repo_python_files

    offenders: list[str] = []

    for py in iter_repo_python_files(ROOT):
        rel = py.relative_to(ROOT).as_posix()
        if rel in EXEMPT_PATHS:
            continue

        try:
            text = py.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        for line_no, line in enumerate(text.splitlines(), 1):
            if is_doc_or_string_only(line):
                continue
            for pat in FORBIDDEN_PATTERNS:
                if pat.search(line):
                    offenders.append(f"{rel}:{line_no}: {line.strip()}")
                    break

    print("=" * 70)
    print("Directive parser audit")
    print("=" * 70)
    print(f"Files scanned: every .py under {ROOT}")
    print(f"Exempt paths : {sorted(EXEMPT_PATHS)}")
    print()
    if offenders:
        print(f"✗ Found {len(offenders)} site(s) using strict directive regex.")
        print(f"  These must be refactored to use runtime/directive_parser.py:")
        print()
        for o in offenders:
            print(f"  {o}")
        print()
        print("How to fix: replace the inline regex with one of:")
        print("    from runtime.directive_parser import find_tool_names")
        print("    from runtime.directive_parser import find_skill_load_names")
        print("    from runtime.directive_parser import find_tool_batch_directives")
        print("    from runtime.directive_parser import strip_tool_directives")
        print("    from runtime.directive_parser import normalize_directives")
        return 1

    print("✓ All directive parsing goes through runtime/directive_parser.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
