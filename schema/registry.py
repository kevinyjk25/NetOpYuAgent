"""
schema/registry.py
------------------
SchemaRegistry — process-wide registry mapping tool_name → ArgSchema.

Single source of truth used by:
  - ToolRouter._wrap   : validate+coerce before tool dispatch
  - LLMEngine prompt  : render Args lines for top-K tools
  - Diagnostics        : error messages back to LLM include schema context

The registry is populated at startup time from three sources (in order):

  1. MCP tool specs       — auto-import via from_mcp_input_schema
  2. OpenAPI operations   — auto-import via from_openapi_operation
  3. Local tool metadata  — auto-import via from_dict_metadata

Later registrations win (last-write), giving operators a way to override
auto-imported schemas with hand-tuned ones in main.py.
"""
from __future__ import annotations

import logging
import threading
from typing      import Any, Optional

from .types import ArgSchema

logger = logging.getLogger(__name__)


class SchemaRegistry:
    """Thread-safe registry of tool schemas."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._schemas: dict[str, ArgSchema] = {}

    # ── Registration ──────────────────────────────────────────────────

    def register(self, schema: ArgSchema, *, replace: bool = True) -> None:
        with self._lock:
            if not replace and schema.tool_name in self._schemas:
                return
            self._schemas[schema.tool_name] = schema

    def register_many(self, schemas: list[ArgSchema], *, replace: bool = True) -> None:
        with self._lock:
            for s in schemas:
                self.register(s, replace=replace)

    def unregister(self, tool_name: str) -> bool:
        with self._lock:
            return self._schemas.pop(tool_name, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._schemas.clear()

    # ── Lookup ────────────────────────────────────────────────────────

    def get(self, tool_name: str) -> Optional[ArgSchema]:
        with self._lock:
            return self._schemas.get(tool_name)

    def has(self, tool_name: str) -> bool:
        with self._lock:
            return tool_name in self._schemas

    def list_all(self) -> list[ArgSchema]:
        with self._lock:
            return list(self._schemas.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._schemas)

    def __contains__(self, tool_name: str) -> bool:
        return self.has(tool_name)


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------

_GLOBAL: Optional[SchemaRegistry] = None


def get_schema_registry() -> SchemaRegistry:
    """Return the process-wide schema registry, creating it on first call."""
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = SchemaRegistry()
    return _GLOBAL
