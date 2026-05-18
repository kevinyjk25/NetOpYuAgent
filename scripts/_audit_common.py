"""
scripts/_audit_common.py
------------------------
Shared helpers for the audit_*.py scripts. The most important thing here
is `iter_repo_python_files()` — it returns every .py file the audits
should consider, with sensible defaults baked in:

  - Skips .venv / venv / env / site-packages / dist-packages — these
    contain third-party libs whose internal module names sometimes
    collide with our top-level packages (e.g. onnxruntime ships a
    `tools/` submodule), producing false-positive audit failures.
  - Skips __pycache__ / .mypy_cache / .pytest_cache / .tox.
  - Skips .git / build / dist / node_modules — defensive.
  - Skips /tests/ and /examples/ by default (audit code is checking
    PRODUCTION code; test fixtures legitimately do weird things).

Each audit can pass `extra_excludes=...` to widen the skip list, or
`include_tests=True` / `include_examples=True` to opt-in to those.

Putting this in ONE place means the next time we discover a directory
that needs skipping, we don't have to remember to update 5 scripts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator


# Path fragments. If any appears anywhere in a file's path, we skip it.
# Why fragments rather than glob/regex: simple, fast, easy to scan visually,
# and works regardless of OS path separator (we normalize to "/" below).
_DEFAULT_EXCLUDES: tuple[str, ...] = (
    "/__pycache__/",
    "/.venv/", "/venv/", "/env/",
    "/site-packages/", "/dist-packages/",
    "/.git/", "/build/", "/dist/", "/node_modules/",
    "/.tox/", "/.mypy_cache/", "/.pytest_cache/",
)

# Additional fragments that audits MAY exclude (most do by default).
_TEST_EXCLUDES: tuple[str, ...] = ("/tests/",)
_EXAMPLE_EXCLUDES: tuple[str, ...] = ("/examples/",)


def iter_repo_python_files(
    repo: Path,
    *,
    include_tests: bool = False,
    include_examples: bool = False,
    extra_excludes: tuple[str, ...] = (),
) -> Iterator[Path]:
    """Yield every .py under `repo` that should be audited.

    Filters out third-party / virtualenv / cache / build dirs. Pass
    `include_tests=True` or `include_examples=True` to opt those in;
    pass `extra_excludes` for audit-specific skips.
    """
    excludes = list(_DEFAULT_EXCLUDES)
    if not include_tests:
        excludes.extend(_TEST_EXCLUDES)
    if not include_examples:
        excludes.extend(_EXAMPLE_EXCLUDES)
    excludes.extend(extra_excludes)

    for f in repo.rglob("*.py"):
        # Normalize to forward slashes so the same fragment matches on
        # Windows runners too — GitHub Actions has Linux runners by
        # default but contributors may run audits locally on Windows.
        s = str(f).replace("\\", "/")
        if any(frag in s for frag in excludes):
            continue
        yield f
