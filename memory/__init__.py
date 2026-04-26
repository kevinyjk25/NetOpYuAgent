"""
memory/__init__.py
──────────────────
Thin facade. The actual memory implementation lives in agent_memory/
(MemoryManager + 6 stores + UserModelEngine + ConsolidationWorker + ReflectionEngine,
311 unit tests). This module exposes the async adapter that the runtime,
backend, and HITL executor call.
"""
from __future__ import annotations

from memory.adapter import (
    MemoryAdapter,
    RecallResult,
    set_current_operator,
    get_current_operator,
)

__all__ = [
    "MemoryAdapter",
    "RecallResult",
    "set_current_operator",
    "get_current_operator",
]
