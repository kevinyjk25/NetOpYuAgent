#!/usr/bin/env python3
"""
scripts/audit_imports.py
------------------------
Static audit: ALL `from integrations.X import Y` and `from skills.X import Y`
references actually resolve to existing modules and names.

Catches the class of bug introduced by a sub-package refactor where files
get moved but a few `from old.flat.path` calls aren't updated.

This is complementary to audit_module_independence.py (which checks WHO
imports WHOM); this script checks IF the imports actually work.

Run on every CI build:
    python scripts/audit_imports.py

Exits 0 if clean, 1 if anything won't resolve.

Notes:
  - We only check that the module path exists. We don't verify symbol names
    (that would require importing every file, which needs all deps).
  - For dynamic imports inside try/except, the static check still flags them
    as suspicious if the path doesn't exist as a module.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path


# ── Configuration ──────────────────────────────────────────────────────────

# Packages we audit. Add new top-level packages here as the project grows.
AUDITED_PACKAGES = ["integrations", "skills", "tools", "agent_memory",
                     "memory", "hitl_core", "retrieval", "schema",
                     "evaluation", "runtime", "registry"]

# Files whose imports we IGNORE (test fixtures, examples, etc.).
# Path-fragment exclusions (venv, site-packages, caches) live in
# _audit_common.iter_repo_python_files so every audit shares the same
# blind-spot list — see commit history for why this matters (onnxruntime
# ships a `tools/` submodule that collides with our top-level package).
IGNORE_FILES = {"__main__.py"}


def collect_import_lines(repo: Path) -> list[tuple[str, int, str]]:
    """Walk every .py file, yield (file_relpath, line_no, module_path) for each
    `from X import ...` / `import X` where X starts with one of AUDITED_PACKAGES."""
    from _audit_common import iter_repo_python_files
    results: list[tuple[str, int, str]] = []
    for f in iter_repo_python_files(repo):
        if f.name in IGNORE_FILES: continue
        try:
            with open(f, encoding="utf-8") as fp:
                source = fp.read()
            tree = ast.parse(source, filename=str(f))
        except (SyntaxError, OSError) as e:
            print(f"WARN: failed to parse {f}: {e}", file=sys.stderr)
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                top = mod.split(".")[0] if mod else ""
                if top in AUDITED_PACKAGES:
                    results.append((str(f.relative_to(repo)), node.lineno, mod))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top in AUDITED_PACKAGES:
                        results.append((str(f.relative_to(repo)), node.lineno, alias.name))
    return results


def module_path_exists(repo: Path, module_path: str) -> bool:
    """Check if a dotted module path corresponds to a real file or package."""
    parts = module_path.split(".")
    # Try as a package: foo/bar/__init__.py
    pkg_init = repo.joinpath(*parts) / "__init__.py"
    if pkg_init.exists(): return True
    # Try as a module: foo/bar.py
    mod_file = repo.joinpath(*parts[:-1], parts[-1] + ".py")
    if mod_file.exists(): return True
    return False


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    lines = collect_import_lines(repo)

    print("=" * 70)
    print("Import Path Audit")
    print("=" * 70)
    print(f"Scanned packages: {AUDITED_PACKAGES}")
    print(f"Found {len(lines)} import statements referencing audited packages.")
    print()

    bad: list[tuple[str, int, str]] = []
    for path, ln, mod in lines:
        if not module_path_exists(repo, mod):
            bad.append((path, ln, mod))

    if bad:
        print(f"BROKEN imports ({len(bad)}):")
        for path, ln, mod in bad:
            print(f"  {path}:{ln}: `from {mod} import ...` — module does not exist")
        print()
        print("Fix options:")
        print("  1. Update the import to the new sub-package path")
        print("  2. Add a back-compat shim at the old location")
        return 1

    print("✓ All audited imports resolve to existing modules.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
