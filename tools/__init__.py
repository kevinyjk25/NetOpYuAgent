"""
tools/
──────
Tool callables and metadata for mock and pragmatic modes.

Entry point: tools.loader.ToolLoader
  ToolLoader(mode="mock" | "pragmatic")
    .build_callables()           -> {name: async_fn}   (used by runtime loop)
    .build_metadata()            -> {name: {...}}       (used by llm_engine prompt)
    .skill_definitions()         -> {skill_id: {...}}   (used by SkillCatalogService)
    .tool_section_for_prompt()   -> str                 (injected into system prompt)

Implementation files:
  tools/mock_tools.py            — callable implementations for mock mode
  tools/pragmatic_tools.py       — callable implementations for pragmatic mode
  tools/builtin/registry.py      — metadata for always-available tools
  tools/mock/registry.py         — metadata for mock-only tools
  tools/pragmatic/registry.py    — metadata for pragmatic-only tools

make_read_stored_result_tool is still used directly by main.py to wire
the ToolResultStore instance into the read_stored_result callable.
"""
from .mock_tools import make_read_stored_result_tool

__all__ = ["make_read_stored_result_tool"]
