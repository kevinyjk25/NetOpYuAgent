"""
integrations.clients — outbound clients for external protocols.

These modules wrap third-party protocols and present them as Python objects:
  - llm_engine     : OllamaEngine, OpenAIEngine, AnthropicEngine, MockEngine
  - mcp_client     : MCP (Model Context Protocol) server clients
  - openapi_client : OpenAPI 3 spec parser + invoker
  - embedder       : embedding-model adapters (Ollama, OpenAI, stub)

Modules in clients/ DO NOT import from adapters/ or router/.
"""
from .llm_engine import (  # noqa: F401
    LLMEngine, OllamaEngine, OpenAIEngine, AnthropicEngine, MockEngine,
    IntentResult, patch_runtime_loop, patch_hitl_graph,
)
from .mcp_client     import MCPClient, MCPServer, MCPToolSpec, MCPCallResult  # noqa: F401
from .openapi_client import OpenAPIClient, OpenAPIParser, OperationSpec, ParamSpec  # noqa: F401
from .embedder       import build_embedder, OllamaEmbedder, OpenAIEmbedder, StubEmbedder  # noqa: F401
