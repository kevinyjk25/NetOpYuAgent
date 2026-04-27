"""
tools/builtin/registry.py
─────────────────────────
Mode-agnostic built-in tools present in ALL modes.

These tools operate on the agent's own internal state (paged result store,
chunk processing) and never touch real or mock network devices.

TOOLS dict is the single source of truth.  ToolLoader imports this directly.
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
        "tags":        ["storage", "analysis"],
    },
}
