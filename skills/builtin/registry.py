"""
skills/builtin/registry.py
──────────────────────────
Skills available in ALL modes (mock + pragmatic).

These skills are device-agnostic guides backed by tools present in every mode.
"""
from __future__ import annotations
from typing import Any

SKILLS: dict[str, dict[str, Any]] = {
    "read_stored_result": {
        "name":        "Read Stored Result",
        "purpose":     "Page through a large stored tool result",
        "risk_level":  "low",
        "requires_hitl": False,
        "tags":        ["storage", "paging"],
        "description": (
            "When a tool returns [STORED:name:ref_id], use this skill to read it page by page. "
            "Call [TOOL:read_stored_result] with the ref_id and increasing offsets. "
            "Write 2-3 sentences of findings after each page before calling the next."
        ),
        "parameters":  {"ref_id": "The ref_id from the [STORED:] label"},
        "returns":     "Pages of stored content with metadata",
        "tool_deps":   ["read_stored_result"],
        "examples":    [],
    },
}
