"""Fail when retired harness surfaces or imports return to the DSH project."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RETIRED_PATHS = (
    "main.py", "webui", "a2a", "task", "hitl_core", "scheduler", "memory",
    "scripts/netopyu-legacy", "scripts/run_3agents.sh", "scripts/check_peers.sh",
)
RUNTIME_ROOTS = ("dsh_adapter", "profiles", "tools", "integrations", "registry", "runtime")
BANNED_IMPORTS = ("main", "webui", "a2a", "task", "hitl_core", "scheduler", "memory", "runtime.loop")


def main() -> int:
    errors: list[str] = []
    for relative in RETIRED_PATHS:
        if (ROOT / relative).exists():
            errors.append(f"retired path exists: {relative}")
    for root_name in RUNTIME_ROOTS:
        root = ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError as error:
                errors.append(f"syntax error {path.relative_to(ROOT)}:{error.lineno}: {error.msg}")
                continue
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    if any(name == banned or name.startswith(banned + ".") for banned in BANNED_IMPORTS):
                        errors.append(f"retired import {name} in {path.relative_to(ROOT)}:{node.lineno}")
    if errors:
        print("\n".join(errors))
        return 1
    print("DSH-only architecture audit: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
