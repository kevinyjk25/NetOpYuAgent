"""
tools/
──────
Common profile-independent tool infrastructure for the DSH domain bridge.

Business tools live in the profiles/ package (profiles/lan, profiles/dc, …),
NOT here. This package holds only what every profile shares:

Entry point: tools.loader.ToolLoader
  ToolLoader(mode="mock" | "pragmatic", profile="default" | "lan" | "dc")
    .build_callables()           -> {name: async_fn}
    .build_metadata()            -> {name: {...}}

Common implementation files:
  tools/common_tools.py          — profile-independent tools (read_stored_result,
                                   process_stored_chunks) + shared _ts() helper
  tools/builtin/registry.py      — metadata for the common tools above
  tools/pragmatic_tools.py       — real device callables for pragmatic mode
  tools/pragmatic/registry.py    — metadata for pragmatic-only tools

Business implementation files now live under profiles/<id>/:
  profiles/lan/tools.py, profiles/lan/tool_meta.py, profiles/lan/skills/ (SKILL.md)
  profiles/dc/tools.py,  profiles/dc/tool_meta.py,  profiles/dc/skills/ (SKILL.md)

make_read_stored_result_tool is used by the DSH backend to wire the
ToolResultStore instance into the read_stored_result callable.
"""
from tools.common_tools import make_read_stored_result_tool

__all__ = ["make_read_stored_result_tool"]
