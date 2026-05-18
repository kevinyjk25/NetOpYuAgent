"""
integrations/ — outbound clients, cross-module adapters, and the tool router.

Sub-packages (organised by purpose):
  - clients/   external protocol wrappers (LLM, MCP, OpenAPI, embedder)
  - adapters/  optional cross-module bridges (memory ↔ skill journal, fact reconcile, hitl)
  - router/    unified tool dispatch

Backwards compatibility: all symbols are re-exported at this level so
`from integrations import ToolRouter` still works. New code should prefer
the explicit sub-package path:
  `from integrations.router import ToolRouter`
"""
# ── Clients ──────────────────────────────────────────────────────────────
from .clients.llm_engine import (  # noqa: F401
    LLMEngine, OllamaEngine, OpenAIEngine, AnthropicEngine, MockEngine,
    IntentResult, patch_runtime_loop, patch_hitl_graph,
)
from .clients.mcp_client     import MCPClient, MCPServer, MCPToolSpec, MCPCallResult  # noqa: F401
from .clients.openapi_client import OpenAPIClient, OpenAPIParser, OperationSpec, ParamSpec  # noqa: F401
from .clients.embedder       import build_embedder, OllamaEmbedder, OpenAIEmbedder, StubEmbedder  # noqa: F401

# ── Adapters ─────────────────────────────────────────────────────────────
from .adapters.memory_facts_adapter   import JournalToFactsAdapter  # noqa: F401
from .adapters.fact_conflict_detector import (  # noqa: F401
    FactConflictDetector, ReconcileResult,
    VERDICT_EQUIVALENT, VERDICT_REFINEMENT,
    VERDICT_CONTRADICTION, VERDICT_UNRELATED,
)
from .adapters import hitl_executor  # noqa: F401

# ── Router ───────────────────────────────────────────────────────────────
from .router.tool_router import ToolRouter, ToolMeta  # noqa: F401

__all__ = [
    # Clients
    "LLMEngine", "OllamaEngine", "OpenAIEngine", "AnthropicEngine", "MockEngine",
    "IntentResult", "patch_runtime_loop", "patch_hitl_graph",
    "MCPClient", "MCPServer", "MCPToolSpec", "MCPCallResult",
    "OpenAPIClient", "OpenAPIParser", "OperationSpec", "ParamSpec",
    "build_embedder", "OllamaEmbedder", "OpenAIEmbedder", "StubEmbedder",
    # Adapters
    "JournalToFactsAdapter",
    "FactConflictDetector", "ReconcileResult",
    "VERDICT_EQUIVALENT", "VERDICT_REFINEMENT",
    "VERDICT_CONTRADICTION", "VERDICT_UNRELATED",
    "hitl_executor",
    # Router
    "ToolRouter", "ToolMeta",
]
