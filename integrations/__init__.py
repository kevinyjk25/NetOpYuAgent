"""
integrations — Production integration layer
============================================

Connects the IT Ops Agent to your company's existing MCP servers
and OpenAPI-documented REST APIs.

Quick start (in main.py)
-------------------------
    from integrations import (
        MCPClient, OpenAPIClient, LLMEngine,
        ToolRouter, patch_runtime_loop, patch_hitl_graph,
    )

    # 1. Connect to MCP
    mcp = MCPClient.from_config(NETOPS_MCP_CONFIG)
    await mcp.connect_all()

    # 2. Load OpenAPI specs
    api = OpenAPIClient.from_url(
        name="netops_nms",
        spec_url="https://nms.internal/openapi.json",
        base_url="https://nms.internal/api/v2",
        auth={"type": "bearer", "token_env": "NMS_TOKEN"},
    )
    await api.load()

    # 3. Build unified tool router
    router = ToolRouter(tool_store=services["tool_store"])
    router.register_mcp(mcp)
    router.register_openapi(api)
    router.register_local(TOOL_REGISTRY)   # mock tools as fallback

    # 4. Wire real LLM (Ollama)
    engine = LLMEngine.from_config({
        "backend":  "ollama",
        "model":    "mistral",
        "base_url": "http://localhost:11434",
    })
    patch_runtime_loop(services["runtime_loop"], engine)
    patch_hitl_graph(engine)

    # 5. Inject tool registry
    services["runtime_loop"]._tool_registry = router.registry
    services["executor"]._tool_registry = router.registry

For development without real servers, use mock factories:
    mcp    = MCPClient.from_netops_mock()
    api    = OpenAPIClient.netops_mock()
    engine = LLMEngine.from_config({"backend": "mock"})
"""

from .llm_engine import (
    LLMEngine, OllamaEngine, OpenAIEngine, AnthropicEngine, MockEngine,
    IntentResult, patch_runtime_loop, patch_hitl_graph,
)
from .mcp_client import MCPClient, MCPServer, MCPToolSpec, MCPCallResult
from .openapi_client import OpenAPIClient, OpenAPIParser, OperationSpec, ParamSpec
from .tool_router import ToolRouter, ToolMeta

__all__ = [
    # LLM
    "LLMEngine", "OllamaEngine", "OpenAIEngine", "AnthropicEngine", "MockEngine",
    "IntentResult", "patch_runtime_loop", "patch_hitl_graph",
    # MCP
    "MCPClient", "MCPServer", "MCPToolSpec", "MCPCallResult",
    # OpenAPI
    "OpenAPIClient", "OpenAPIParser", "OperationSpec", "ParamSpec",
    # Router
    "ToolRouter", "ToolMeta",
]
