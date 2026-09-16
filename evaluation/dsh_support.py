"""Neutral DSH discovery and config parsing shared by evaluation entry points.

Only stdlib helpers live here. Importing this module never loads a benchmark,
contacts a model, starts DSH, or changes the reviewed plugin allowlists.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil


DSH_TESTED_VERSION = "0.1.1-rc.2"

# This is deliberately exact. A DSH upgrade or a newly activated plugin must
# be reviewed before an evaluator is allowed to contact a model.
SAFE_ACTIVE_IDS = frozenset({
    "timer",
    "llm",
    "session",
    "typert",
    "typert-loader",
    "typert-gateway",
    "session-title",
    "agent",
    "agent-default-model",
    "settings",
    "credentials",
    "llm-pi-ai",
    "session-persistence-jsonl",
    "session-query-sqlite",
    "session-projection",
    "token-meter",
    "compaction-basic",
    "timeout-policy",
    "spill-local",
    "spill-policy",
    "session-checkpoint-policy",
    "repeat-tool-reminder",
    "tools",
    "system-prompt",
    "agent-loop",
    "headless-startup",
    "headless-runner",
})

REQUIRED_DISABLED_IDS = frozenset({
    "session-title-llm",
    "user-questions",
    "llm-retry",
    "attachment-local",
    "session-telemetry-otel",
    "jobs",
    "subprocess",
    "sandbox",
    "sandbox-policy",
    "bash-sandbox",
    "pwsh-sandbox",
    "approval",
    "permission",
    "shell-env",
    "fs-observation-policy",
    "agent-instructions",
    "skill",
    "skill-filesystem",
    "skill-badge",
    "commands",
    "command-feedback",
    "command-compact",
    "goal",
    "goal-round-driver",
    "command-goal",
    "plan-mode",
    "subagent",
    "subagent-spawn-in-process",
    "subagent-fork-in-process",
    "workflow-worker-thread",
    "web",
    "web-search-deepseek",
    "llm-deepseek",
    "code-runtime",
    "fs-sandbox",
    "tool-bash",
    "tool-pwsh",
    "tool-jobs",
    "tool-fs",
    "tool-fs-search",
    "tool-skill",
    "tool-subagent-control",
    "tool-subagent-list-agents",
    "tool-subagent",
    "tool-subagent-fork",
    "tool-subagent-report",
    "tool-workflow",
    "tool-result-pruner",
    "tool-todo",
    "tool-goal",
    "tool-ralph",
    "tool-str-replace-editor",
    "tool-web",
})


@dataclass(frozen=True)
class ConfigEntry:
    entry_id: str
    plugin_name: str
    disabled: bool


def parse_dumped_config(text: str) -> tuple[ConfigEntry, ...]:
    """Parse only entry identity/name/disabled state from DSH's JS-YAML dump."""
    starts = list(re.finditer(r"^- id: ([^\n]+)$", text, re.MULTILINE))
    entries: list[ConfigEntry] = []
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        block = text[match.start():end]
        name = re.search(r"^  name: ['\"]?([^'\"\n]+)['\"]?$", block, re.MULTILINE)
        entries.append(ConfigEntry(
            entry_id=match.group(1).strip(),
            plugin_name=name.group(1).strip() if name else "",
            disabled=bool(re.search(r"^  disabled: true$", block, re.MULTILINE)),
        ))
    if not entries:
        raise ValueError("DSH shadow config contains no entries")
    if len({item.entry_id for item in entries}) != len(entries):
        raise ValueError("DSH shadow config contains duplicate entry ids")
    return tuple(entries)


def _default_dsh_binary() -> Path:
    configured = os.environ.get("NETOPYU_DSH_BIN")
    if configured:
        return Path(configured).expanduser().resolve()
    return (
        Path.home()
        / "Library/Application Support/NetOpYuAgent/dsh-runtime/node_modules/.bin/dsh"
    )


def _node_path() -> str:
    current = os.environ.get("PATH", "")
    if shutil.which("node", path=current):
        return current
    bundled = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies"
    node = bundled / "node/bin/node"
    if node.is_file():
        return f"{bundled / 'node/bin'}:{bundled / 'bin/fallback'}:{current}"
    return current
