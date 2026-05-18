#!/usr/bin/env python3
"""
scripts/audit_module_independence.py
------------------------------------
Static audit: each functional module must not import from other functional
modules (with documented exceptions for adapters and entry points).

Run on every CI build:
    python scripts/audit_module_independence.py

Exits 0 if clean, 1 if violations.

Add to .pre-commit-config.yaml as:
    - repo: local
      hooks:
        - id: module-independence
          name: Module independence audit
          entry: python scripts/audit_module_independence.py
          language: system
          pass_filenames: false
"""
from __future__ import annotations

import sys
from pathlib import Path


# ── Configuration ───────────────────────────────────────────────────────────

FUNCTIONAL_MODULES = [
    "agent_memory",
    "memory",
    "hitl_core",
    "skills",
    "tools",
    "retrieval",
    "schema",
    "evaluation",
    "registry",
]

# Cross-module imports that ARE allowed (documented exceptions)
ALLOWED_CROSS_DEPS = {
    "memory":  {"agent_memory"},   # adapter wraps the core
    "tools":   {"schema"},          # tools may declare schema metadata
    "skills":  {"retrieval"},       # catalog uses retriever Protocol
}

# Files exempt from the rule (application entry points)
GLUE_FILES = {"cli.py", "__main__.py"}

# Annotations on import lines that mark them as exempt:
EXEMPT_MARKERS = ["DEPRECATED SHIM", "ALLOWED BY DESIGN"]


# ── Audit ──────────────────────────────────────────────────────────────────

def audit(repo_root: Path) -> list[str]:
    violations: list[str] = []
    for mod in FUNCTIONAL_MODULES:
        mod_path = repo_root / mod
        if not mod_path.exists():
            continue
        allowed = ALLOWED_CROSS_DEPS.get(mod, set())
        for pyfile in mod_path.rglob("*.py"):
            if "__pycache__" in str(pyfile):
                continue
            if pyfile.name in GLUE_FILES:
                continue
            with open(pyfile, encoding="utf-8") as f:
                content = f.read()
            for ln, line in enumerate(content.split("\n"), start=1):
                stripped = line.strip()
                if not (stripped.startswith("from ") or stripped.startswith("import ")):
                    continue
                if any(marker in line for marker in EXEMPT_MARKERS):
                    continue
                for other in FUNCTIONAL_MODULES:
                    if other == mod or other in allowed:
                        continue
                    if (stripped.startswith(f"from {other} ") or
                        stripped.startswith(f"from {other}.") or
                        stripped == f"import {other}"):
                        rel = pyfile.relative_to(repo_root)
                        violations.append(
                            f"  {rel}:{ln}: {mod}/ imports from {other}/ "
                            f"  ← VIOLATES MODULE INDEPENDENCE"
                        )
    return violations


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    violations = audit(repo)

    print("=" * 70)
    print("Module Independence Audit")
    print("=" * 70)
    print(f"Functional modules: {FUNCTIONAL_MODULES}")
    print(f"Allowed exceptions: {dict(ALLOWED_CROSS_DEPS)}")
    print(f"Exempt files:       {sorted(GLUE_FILES)}")
    print(f"Exempt markers:     {EXEMPT_MARKERS}")
    print()

    if violations:
        print(f"Found {len(violations)} violation(s):")
        for v in violations:
            print(v)
        print()
        print("Fix options:")
        print("  1. Restructure so module A doesn't need module B")
        print("  2. Add module B to ALLOWED_CROSS_DEPS if architecturally justified")
        print("  3. Annotate the line with '# ALLOWED BY DESIGN: <reason>'")
        return 1
    else:
        print("✓ All functional modules are properly independent.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
