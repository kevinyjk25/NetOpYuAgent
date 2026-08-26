"""skills/script_runner.py — Anthropic-standard skill script execution.

Implements "script-as-tool" (路线 1): a skill's bundled `scripts/*.py` are
loaded, AST-validated against a denylist, and exposed as async tool callables
(``async (args: dict) -> str``) registerable via ToolRouter.register_local.

Why not exec arbitrary files via bash (as Claude Code does):
  This project's execution environment explicitly forbids arbitrary code
  execution (see webui AST denylist + "NOT a sandbox" note). Claude Code runs
  on the user's own trusted machine with a real shell; this is a server-side
  multi-agent service with a different trust model. So skill scripts are
  treated as TRUSTED, project-deployed, validated functions — not sandboxed
  arbitrary code. Each script must expose:

      def run(inputs: dict) -> dict

  The denylist (shared with the webui uploaded-tool path) blocks the obvious
  dangerous imports/calls. It is best-effort, NOT a sandbox — skills are a
  trusted, audited asset (install only from trusted sources, per the Anthropic
  security guidance).

Tool naming: ``<skill_id>__<script_stem>`` (e.g. ``probe_skill__compute``).
These are NOT added to the LLM tool-retrieval corpus — they are internal
computation steps a skill invokes by name, not tools the LLM freely selects.
"""
from __future__ import annotations

import ast
import importlib.util
import logging
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Shared denylist (parity with webui/backend.py uploaded-tool validation).
DENIED_IMPORTS = {
    "os", "subprocess", "sys", "shutil", "socket", "ctypes",
    "multiprocessing", "pty", "popen2", "commands", "_winreg", "winreg",
    "requests", "urllib", "http", "ftplib", "telnetlib", "smtplib",
}
DENIED_CALLS = {"eval", "exec", "compile", "__import__", "open"}


class ScriptValidationError(Exception):
    """A skill script failed AST validation (dangerous import/call)."""


def validate_script_source(source: str, filename: str = "<script>") -> None:
    """Raise ScriptValidationError if the source uses a denied import/call.

    Best-effort, NOT a sandbox. Skills are a trusted/audited asset; this
    catches accidental or obvious misuse, not a determined attacker.
    """
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        raise ScriptValidationError(f"syntax error: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                if n.name.split(".")[0] in DENIED_IMPORTS:
                    raise ScriptValidationError(f"import of {n.name!r} not allowed")
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in DENIED_IMPORTS:
                raise ScriptValidationError(f"import from {node.module!r} not allowed")
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in DENIED_CALLS:
                raise ScriptValidationError(f"call to {fn.id!r} not allowed")


def _load_script_run_fn(script_path: Path) -> Callable[[dict], Any]:
    """Validate + import a script module, return its ``run`` function."""
    source = script_path.read_text(encoding="utf-8")
    validate_script_source(source, filename=str(script_path))

    spec = importlib.util.spec_from_file_location(
        f"skill_script_{script_path.stem}", script_path
    )
    if spec is None or spec.loader is None:
        raise ScriptValidationError(f"cannot import {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # validated above

    run_fn = getattr(module, "run", None)
    if not callable(run_fn):
        raise ScriptValidationError(
            f"{script_path.name} must define a top-level run(inputs: dict) -> dict"
        )
    return run_fn


def build_script_tools(skill_id: str, skill_dir: str | Path) -> dict[str, Callable]:
    """Return {tool_name: async callable(args: dict) -> str} for every
    ``scripts/*.py`` in the skill that exposes a ``run`` function.

    Tool name = ``<skill_id>__<script_stem>``. Validation failures skip that
    one script (logged) without aborting the others.
    """
    scripts_dir = Path(skill_dir) / "scripts"
    if not scripts_dir.is_dir():
        return {}

    tools: dict[str, Callable] = {}
    for script_path in sorted(scripts_dir.glob("*.py")):
        if script_path.name.startswith("_"):
            continue
        tool_name = f"{skill_id}__{script_path.stem}"
        try:
            run_fn = _load_script_run_fn(script_path)
        except ScriptValidationError as exc:
            logger.warning(
                "skill script %s rejected: %s — skipping", script_path, exc
            )
            continue

        def _make_tool(fn: Callable, name: str) -> Callable:
            async def _tool(args: dict) -> str:
                try:
                    result = fn(args or {})
                except Exception as exc:   # script bug → string error, never raise
                    return f"ERROR: script {name} failed: {exc}"
                # Contract is run(inputs)->dict; stringify for the tool channel.
                if isinstance(result, dict):
                    import json
                    return json.dumps(result, ensure_ascii=False)
                return str(result)
            _tool.__name__ = name
            return _tool

        tools[tool_name] = _make_tool(run_fn, tool_name)
        logger.info("skill script registered as tool: %s", tool_name)

    return tools


def build_all_script_tools(skill_defs: dict[str, dict]) -> dict[str, Callable]:
    """Across all loaded skills (defn must carry ``skill_dir`` + ``scripts``),
    build every script tool. Used at startup to register them in one pass."""
    out: dict[str, Callable] = {}
    for skill_id, defn in skill_defs.items():
        if not defn.get("scripts"):
            continue
        skill_dir = defn.get("skill_dir")
        if not skill_dir:
            continue
        out.update(build_script_tools(skill_id, skill_dir))
    return out
