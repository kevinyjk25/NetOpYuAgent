"""
tools/builtin/registry.py
─────────────────────────
Mode-agnostic built-in tools present in ALL modes.

These tools operate on the agent's own internal state (paged result store,
chunk processing) and never touch real or mock network devices.

TOOLS dict is the single source of truth.  ToolLoader imports this directly.

action_type field (added for reversibility-weighted policy):
  read_only  — no state change, no side effect, safe to call freely
  reversible — creates artifact that can be undone (e.g. staged config)
  destructive — irreversible mutation (no automatic action_type fallback;
                must be explicitly set)
Fast-path: runtime.policy_engine classify_action_type checks this BEFORE
LLM evaluation. Tools without action_type fall back to LLM classification
to preserve current behaviour (no regression).
"""
from __future__ import annotations
from typing import Any


# ── Registry ─────────────────────────────────────────────────────────────────

TOOLS: dict[str, dict[str, Any]] = {
    "read_stored_result": {
        "description": (
            "Read a page of a previously stored large tool result. "
            "Use when a tool returned a [STORED:name:ref_id] label."
        ),
        "parameters": {
            "ref_id":  "Reference ID from the [STORED:] label (e.g. '6ac5ade7' or 'netflow_dump:6ac5ade7')",
            "offset":  "Character offset to start reading from (default 0)",
            "length":  "Maximum characters to return (default 2000)",
        },
        "returns":     "Page of stored text with metadata: total size, has_more, next offset",
        "hitl":        False,
        "action_type": "read_only",
        "tags":        ["storage", "paging"],
    },
    "process_stored_chunks": {
        "description": "Summarise multiple pages of a stored result into key findings.",
        "parameters": {
            "ref_id":  "Reference ID of the stored result",
            "task":    "What to extract or summarise from the data",
        },
        "returns":     "Structured summary of the stored result",
        "hitl":        False,
        "action_type": "read_only",
        "tags":        ["storage", "analysis"],
    },
}
