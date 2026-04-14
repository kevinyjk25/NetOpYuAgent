"""
main.py  [v4 — with Agent Registry]
-------------------------------------
Changes from v3
---------------
  1. Imports and initialises the Agent Registry module.
  2. Self-registers this agent's own AgentCard at startup.
  3. Fetches and registers peer agents from AGENT_URLS env var.
  4. Passes ``registry`` into create_task_system() so TaskPlanner
     resolves AgentAssignment automatically.
  5. Mounts /registry/* FastAPI router.
  6. /health now includes registry agent counts.

Environment variables (new in v4)
----------------------------------
  AGENT_URLS          comma-separated peer agent base URLs to pre-register
                      e.g. "http://agent-b:8001/api/v1/a2a,http://pred:8002/api/v1/a2a"
  REGISTRY_LB         load-balance strategy: round_robin | random | least_loaded
  REGISTRY_HEALTH_INTERVAL  health-check interval in seconds (default 60)
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────
A2A_BASE_URL  = os.getenv("A2A_BASE_URL",  "http://localhost:8000/api/v1/a2a")
REDIS_URL     = os.getenv("REDIS_URL")
POSTGRES_DSN  = os.getenv("POSTGRES_DSN")
CHROMA_PATH   = os.getenv("CHROMA_PATH", "./chroma_db")
AGENT_URLS    = [u.strip() for u in os.getenv("AGENT_URLS", "").split(",") if u.strip()]
REGISTRY_LB   = os.getenv("REGISTRY_LB", "round_robin")
REGISTRY_HEALTH_INTERVAL = int(os.getenv("REGISTRY_HEALTH_INTERVAL", "60"))

# ── Integration config ────────────────────────────────────────────────────
# LLM backend: "ollama" | "openai" | "anthropic" | "mock"
LLM_BACKEND   = os.getenv("LLM_BACKEND",   "ollama")
LLM_MODEL     = os.getenv("LLM_MODEL",     "qwen3.5:27b")
LLM_BASE_URL  = os.getenv("LLM_BASE_URL",  "http://localhost:11434")

# MCP server config (JSON string or path to JSON file)
# Example env:  MCP_CONFIG='{"netops":{"transport":"http","url":"http://mcp:8080","auth":{"type":"bearer","token_env":"MCP_TOKEN"}}}'
MCP_CONFIG_JSON = os.getenv("MCP_CONFIG_JSON", "")
MCP_USE_MOCK    = os.getenv("MCP_USE_MOCK", "true").lower() == "true"

# OpenAPI spec (URL or local path)
OPENAPI_SPEC_URL  = os.getenv("OPENAPI_SPEC_URL", "")
OPENAPI_BASE_URL  = os.getenv("OPENAPI_BASE_URL", "")
OPENAPI_AUTH_TYPE = os.getenv("OPENAPI_AUTH_TYPE", "bearer")
OPENAPI_TOKEN_ENV = os.getenv("OPENAPI_TOKEN_ENV", "NETOPS_API_TOKEN")
OPENAPI_USE_MOCK  = os.getenv("OPENAPI_USE_MOCK", "true").lower() == "true"


# ---------------------------------------------------------------------------
# Build all services
# ---------------------------------------------------------------------------

async def build_services() -> dict[str, Any]:
    services: dict[str, Any] = {}

    # ── 1. Memory module ─────────────────────────────────────────────────
    from memory import create_memory_router
    memory_router = await create_memory_router(
        redis_url=REDIS_URL,
        postgres_dsn=POSTGRES_DSN,
        chroma_path=CHROMA_PATH,
    )
    services["memory"] = memory_router
    logger.info("Memory module ready")

    # ── 2. HITL module ───────────────────────────────────────────────────
    from hitl import (
        HitlAuditService, HitlDecisionRouter, HitlReviewService,
        HitlTimeoutWatchdog, ReviewChannelConfig, build_hitl_graph,
    )
    from hitl.triggers import HitlConfig

    hitl_config    = HitlConfig(
        confidence_threshold=float(os.getenv("HITL_CONFIDENCE_THRESHOLD", "0.75")),
        max_auto_host_count=int(os.getenv("HITL_MAX_AUTO_HOST_COUNT", "5")),
    )
    review_config  = ReviewChannelConfig(
        slack_webhook_url=os.getenv("HITL_SLACK_WEBHOOK_URL"),
        pagerduty_routing_key=os.getenv("HITL_PAGERDUTY_ROUTING_KEY"),
        enable_sse=True,
    )
    hitl_audit     = HitlAuditService.in_memory()
    review_service = HitlReviewService.from_config(review_config)
    hitl_graph     = build_hitl_graph(hitl_config)
    hitl_router    = HitlDecisionRouter(graph=hitl_graph, audit=hitl_audit)
    hitl_watchdog  = HitlTimeoutWatchdog(router=hitl_router, poll_interval=60.0)
    services.update(dict(
        hitl_audit=hitl_audit, review_service=review_service,
        hitl_router=hitl_router, hitl_watchdog=hitl_watchdog,
        hitl_config=hitl_config,
    ))
    logger.info("HITL module ready")

    # ── 3. Agent Registry ─────────────────────────────────────────────────
    from registry import create_registry, RegistryConfig
    from a2a.agent_card import get_agent_card

    registry_config = RegistryConfig(
        lb_strategy=REGISTRY_LB,
        health_check_interval_seconds=REGISTRY_HEALTH_INTERVAL,
    )
    own_card = get_agent_card(A2A_BASE_URL)
    registry = await create_registry(
        static_urls=AGENT_URLS,
        redis_url=REDIS_URL,
        config=registry_config,
        own_card=own_card,
    )
    await registry.start()
    services["registry"] = registry
    logger.info(
        "Agent Registry ready — %d peer agent(s) pre-registered",
        len(AGENT_URLS),
    )

    # ── 4. Task module ───────────────────────────────────────────────────
    from task import create_task_system
    task_system = await create_task_system(
        hitl_router=hitl_router,
        review_svc=review_service,
        registry=registry,          # ← NEW: registry-aware planner
    )
    services["task_system"] = task_system
    logger.info("Task module ready (registry-aware planner)")

    # ── 5. A2A executor ──────────────────────────────────────────────────
    from hitl import ITOpsHitlAgentExecutor
    executor = ITOpsHitlAgentExecutor(
        hitl_router=hitl_router,
        review_service=review_service,
        audit_service=hitl_audit,
        hitl_config=hitl_config,
        memory_router=memory_router,
        task_system=task_system,
    )
    services["executor"] = executor
    logger.info("A2A executor ready")

    # ── 6. Integrations layer (MCP + OpenAPI + LLM) ───────────────────────
    try:
        from integrations import (
            MCPClient, OpenAPIClient, LLMEngine, ToolRouter,
            patch_runtime_loop, patch_hitl_graph,
        )
        from tools import TOOL_REGISTRY, make_read_stored_result_tool
        from runtime import ToolResultStore

        # Shared tool store (may already exist in webui services)
        tool_store = services.get("tool_store") or ToolResultStore()
        services["tool_store"] = tool_store

        # ── 6a. LLM engine ──────────────────────────────────────────────
        llm_engine = LLMEngine.from_config({
            "backend":  LLM_BACKEND,
            "model":    LLM_MODEL,
            "base_url": LLM_BASE_URL,
        })
        # Patch the runtime loop and HITL graph before graph is compiled
        # (graph is already built above, so we patch intent_classifier only)
        patch_hitl_graph(llm_engine)
        services["llm_engine"] = llm_engine
        logger.info("LLM engine ready: backend=%s model=%s", LLM_BACKEND, LLM_MODEL)

        # ── 6b. MCP client ──────────────────────────────────────────────
        if MCP_CONFIG_JSON:
            import json as _json
            try:
                mcp_config = _json.loads(MCP_CONFIG_JSON)
            except Exception:
                # treat as a file path
                import pathlib
                mcp_config = _json.loads(pathlib.Path(MCP_CONFIG_JSON).read_text())
            mcp_client = MCPClient.from_config(mcp_config)
        else:
            if MCP_USE_MOCK:
                mcp_client = MCPClient.from_netops_mock()
                logger.info("MCP: using built-in NetOps mock (set MCP_CONFIG_JSON to use real server)")
            else:
                mcp_client = MCPClient()   # empty — no tools
        await mcp_client.connect_all()
        services["mcp_client"] = mcp_client

        # ── 6c. OpenAPI client ──────────────────────────────────────────
        if OPENAPI_SPEC_URL and OPENAPI_BASE_URL:
            api_client = OpenAPIClient.from_url(
                name="netops_api",
                spec_url=OPENAPI_SPEC_URL,
                base_url=OPENAPI_BASE_URL,
                auth={"type": OPENAPI_AUTH_TYPE, "token_env": OPENAPI_TOKEN_ENV},
            )
            await api_client.load()
            logger.info("OpenAPI client loaded %d operations", len(api_client.list_operations()))
        else:
            if OPENAPI_USE_MOCK:
                api_client = OpenAPIClient.netops_mock()
                await api_client.load()
                logger.info("OpenAPI: using built-in NetOps mock (set OPENAPI_SPEC_URL to use real API)")
            else:
                api_client = None
        services["api_client"] = api_client

        # ── 6d. Unified ToolRouter ──────────────────────────────────────
        tool_registry_local = dict(TOOL_REGISTRY)
        tool_registry_local["read_stored_result"] = make_read_stored_result_tool(tool_store)

        router = ToolRouter(tool_store=tool_store)
        router.register_mcp(mcp_client)
        if api_client:
            router.register_openapi(api_client)
        router.register_local(tool_registry_local)

        services["tool_router"] = router
        counts = router.tool_count()
        logger.info(
            "ToolRouter ready: mcp=%d openapi=%d local=%d",
            counts.get("mcp", 0),
            counts.get("openapi", 0),
            counts.get("local", 0),
        )

        # ── 6e. Inject real tool registry + LLM into runtime loop ───────
        real_registry = router.registry
        if "runtime_loop" in services:
            services["runtime_loop"]._tool_registry = real_registry
            patch_runtime_loop(services["runtime_loop"], llm_engine)
            logger.info("Runtime loop: patched with real LLM + ToolRouter")

        # Inject into executor
        executor._tool_registry = real_registry

    except Exception as exc:
        logger.warning(
            "Integrations layer failed to initialise (%s). "
            "System will run with mock tools and stub LLM.",
            exc,
        )

    # ── 7. Hermes Learning Loop ───────────────────────────────────────
    try:
        from memory.fts_store import FTS5SessionStore
        from memory.curator   import MemoryCurator
        from memory.user_model import UserModelEngine
        from skills.evolver   import SkillEvolver

        import pathlib as _pl
        _data_dir = _pl.Path(os.getenv("HERMES_DATA_DIR", "./data"))
        _data_dir.mkdir(parents=True, exist_ok=True)

        fts_store = FTS5SessionStore(db_path=str(_data_dir / "state.db"))
        await fts_store.initialize()

        llm_engine = services.get("llm_engine")
        llm_fn = None
        if llm_engine:
            # Hermes modules use async (system: str, user: str) -> str.
            # Clean separation — no prompt splitting needed.
            async def _llm_fn(system: str, user: str, _e=llm_engine) -> str:
                if hasattr(_e, "_chat"):
                    messages = [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ]
                    raw = await _e._chat(messages)
                    if hasattr(_e, "_strip_think"):
                        raw = _e._strip_think(raw)
                    return raw
                # Fallback for engines without _chat
                return await _e.call(user, context=system)
            llm_fn = _llm_fn

        memory_router = services.get("memory")
        skill_catalog = services.get("skill_catalog")

        # Attach FTS summarizer to use the same LLM
        if llm_fn:
            async def _fts_summarizer(results, query, _fn=llm_fn):
                import datetime as _dt
                excerpts = "\n".join(
                    f"[{_dt.datetime.fromtimestamp(r.ts).strftime('%Y-%m-%d')}] "
                    f"{r.user_text[:120]} → {r.assistant_text[:120]}"
                    for r in results[:5]
                )
                system = "You are an IT operations assistant. Summarise past session excerpts in 2-3 concise sentences."
                user   = f"Query: {query}\n\nPast session excerpts:\n{excerpts}"
                return await _fn(system, user)
            fts_store._summarizer = _fts_summarizer

        memory_curator = MemoryCurator(
            fts_store=fts_store,
            memory_router=memory_router or _NullMemoryRouter(),
            llm_fn=llm_fn,
        )
        user_model = UserModelEngine(
            fts_store=fts_store,
            llm_fn=llm_fn,
        )
        skill_evolver = SkillEvolver(
            catalog=skill_catalog or _NullSkillCatalog(),
            llm_fn=llm_fn,
            fts_store=fts_store,
        )

        services.update({
            "fts_store":      fts_store,
            "memory_curator": memory_curator,
            "user_model":     user_model,
            "skill_evolver":  skill_evolver,
            "_llm_backend":   os.getenv("LLM_BACKEND", "mock"),
            "_llm_model":     os.getenv("LLM_MODEL", "stub"),
            "_hermes_data":   str(_data_dir / "state.db"),
        })

        # Inject into executor
        executor = services.get("executor")
        if executor:
            executor._fts_store     = fts_store
            executor._curator       = memory_curator
            executor._user_model    = user_model
            executor._skill_evolver = skill_evolver
            executor._skill_catalog = skill_catalog

        llm_label = "none (stub)" if llm_fn is None else (
            f"ollama/{os.getenv('LLM_MODEL','?')}" if os.getenv("LLM_BACKEND") == "ollama"
            else os.getenv("LLM_BACKEND", "mock")
        )
        logger.info(
            "━━ System wiring ━━\n"
            "  LLM backend : %s\n"
            "  FTS5 store  : %s\n"
            "  Curator     : yes (llm=%s)\n"
            "  User model  : yes (llm=%s)\n"
            "  SkillEvolver: yes (catalog=%s)\n"
            "  Executor    : %s\n"
            "  HITL router : %s",
            llm_label,
            _data_dir / "state.db",
            "real" if llm_fn else "stub",
            "real" if llm_fn else "stub",
            "real" if skill_catalog else "null",
            "wired" if services.get("executor") else "MISSING",
            "wired" if services.get("hitl_router") else "MISSING",
        )

    except Exception as exc:
        logger.warning("Hermes learning loop failed to initialise: %s", exc)

    return services


# ---------------------------------------------------------------------------
# Null-object stubs (used when optional dependencies are absent)
# ---------------------------------------------------------------------------

class _NullMemoryRouter:
    async def ingest_entity(self, *a, **kw): pass
    async def retrieve(self, *a, **kw): return []


class _NullSkillCatalog:
    def register_all(self, d): pass
    def load_detail(self, s): return None
    def get_summary(self, s): return None


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

_services: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _services
    logger.info("Starting IT Ops Agent v4")
    _services = await build_services()

    # ── Mount routers (must happen before yield so routes are live) ───────
    from a2a import create_a2a_app, InMemoryTaskStore
    a2a_app = create_a2a_app(
        base_url=A2A_BASE_URL,
        executor=_services["executor"],
        task_store=InMemoryTaskStore(),
    )
    app.mount("/api/v1/a2a", a2a_app)

    from hitl.router import create_hitl_router
    from hitl.review import get_sse_channel
    hitl_api = create_hitl_router(
        decision_router=_services["hitl_router"],
        audit=_services["hitl_audit"],
        sse_channel=get_sse_channel(),
    )
    app.include_router(hitl_api, prefix="/hitl")

    from registry.router import create_registry_router
    reg_api = create_registry_router(_services["registry"])
    app.include_router(reg_api, prefix="/registry")

    logger.info("Modules mounted: /api/v1/a2a · /hitl · /registry")

    # ── WebUI (browser interface) ─────────────────────────────────────
    from webui.backend import create_webui_app
    webui = create_webui_app(_services)
    app.mount("/webui", webui)
    logger.info("WebUI mounted at /webui")

    # ── Start background tasks ────────────────────────────────────────────
    from memory.consolidation import LifecycleManager
    lifecycle      = LifecycleManager(_services["memory"])
    watchdog_task  = asyncio.create_task(_services["hitl_watchdog"].run())
    lifecycle_task = asyncio.create_task(lifecycle.run())

    logger.info("Background tasks started")
    yield

    # ── Graceful shutdown ─────────────────────────────────────────────────
    _services["hitl_watchdog"].stop()
    await _services["registry"].stop()
    lifecycle.stop()
    for t in (watchdog_task, lifecycle_task):
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
    logger.info("IT Ops Agent v4 shut down cleanly")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="IT Ops Monitoring Agent",
    version="4.0.0",
    description="IT Ops AI Agent — A2A · HITL · Memory · Task · Agent Registry",
    lifespan=lifespan,
)


# Serve the WebUI
from fastapi.responses import FileResponse
import pathlib

@app.get("/", include_in_schema=False)
async def serve_webui():
    html_path = pathlib.Path(__file__).parent / "webui" / "index.html"
    return FileResponse(str(html_path), media_type="text/html")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    reg    = _services.get("registry")
    agents = await reg.list_agents() if reg else []
    from registry.schemas import AgentHealthState
    healthy = sum(1 for a in agents if a.health == AgentHealthState.HEALTHY)

    task_sys = _services.get("task_system")
    pending_tasks = len(await task_sys.store.list_pending()) if task_sys else 0

    hitl_rtr = _services.get("hitl_router")
    pending_hitl = sum(
        1 for p in hitl_rtr._payload_store.values()
        if p.status.value == "pending"
    ) if hitl_rtr else 0

    return {
        "status":  "ok",
        "version": "4.0.0",
        "registry": {
            "total_agents":   len(agents),
            "healthy_agents": healthy,
        },
        "pending_tasks":          pending_tasks,
        "pending_hitl_interrupts": pending_hitl,
    }


# ---------------------------------------------------------------------------
# Entry point — allows `python main.py` in addition to `uvicorn main:app`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8001")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
        log_level="info",
    )