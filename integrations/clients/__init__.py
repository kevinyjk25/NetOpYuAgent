"""Outbound clients retained for DSH pragmatic MCP/OpenAPI tools."""

from .mcp_client import MCPClient, MCPServer, MCPToolSpec, MCPCallResult
from .openapi_client import OpenAPIClient, OpenAPIParser, OperationSpec, ParamSpec

__all__ = [
    "MCPClient", "MCPServer", "MCPToolSpec", "MCPCallResult",
    "OpenAPIClient", "OpenAPIParser", "OperationSpec", "ParamSpec",
]
