"""
integrations.router — the unified tool dispatch router.

ToolRouter accepts callables from THREE sources:
  - local Python functions (tools/{mock,pragmatic}/registry.py)
  - MCP servers (via integrations.clients.mcp_client)
  - OpenAPI operations (via integrations.clients.openapi_client)

It exposes ONE dispatch interface to runtime/loop.py, so the loop doesn't
care where a tool lives. Schema validation (schema/) runs as an opt-in
middleware step before dispatch.
"""
from .tool_router import ToolRouter, ToolMeta  # noqa: F401
