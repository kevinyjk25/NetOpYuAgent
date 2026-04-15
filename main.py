"""
main.py  [v5 — config.yaml driven]
------------------------------------
All runtime configuration is now read from config.yaml (project root).
Environment variables still override any YAML value — same 12-factor
compatibility, but no scattered os.getenv() throughout the code.

Quick-start:
    uvicorn main:app --port 8000 --reload

Change a setting without editing code:
    # In config.yaml:
    llm:
      model: "qwen3.5:14b"

    # Or via env var (always wins over YAML):
    LLM_MODEL=qwen3.5:14b uvicorn main:app

See config.yaml for all available settings and their defaults.
Default LLM: Ollama / qwen3.5:27b
"""
from __future__ import annotations

import asyncio
import logging
import pathlib
from contextlib import asynccontextmanager
from typing import Any

# ── Config must be first — logging_config reads cfg.logging.mode ──────────
from config import cfg
import logging_config as _lc
_lc.configure(mode=cfg.logging.mode)

from fastapi import FastAPI

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Build all services
# ---------------------------------------------------------------------------

async def build_services() -> dict[str, Any]:
    services: dict[str, Any] = {}

    # ── 1. Memory module ─────────────────────────────────────────────────
    from memory import create_memory_router
    memory_router = await create_memory_router(
        redis_url    = cfg.memory.redis_url,
        postgres_dsn = cfg.memory.postgres_dsn,
        chroma_path  = cfg.memory.chroma_path,
    )
    services["memory"] = memory_router
    logger.info("Memory module ready")

    # ── 2. HITL module ───────────────────────────────────────────────────
    from hitl import (
        HitlAuditService, HitlDecisionRouter, HitlReviewService,
        HitlTimeoutWatchdog, ReviewChannelConfig, build_hitl_graph,
    )
    from hitl.triggers import HitlConfig
    import dataclasses as _dc

    _hitl_tool_names = tuple(cfg.tools.hitl_tool_names)
    _hitl_fields     = {f.name for f in _dc.fields(HitlConfig)}
    _hitl_kwargs: dict = {
        "confidence_threshold": cfg.hitl.confidence_threshold,
        "max_auto_host_count":  cfg.hitl.max_auto_host_count,
    }
    if "tool_call_hitl_tools" in _hitl_fields and _hitl_tool_names:
        _hitl_kwargs["tool_call_hitl_tools"] = _hitl_tool_names

    hitl_config    = HitlConfig(**_hitl_kwargs)
    review_config  = ReviewChannelConfig(
        slack_webhook_url     = cfg.hitl.slack_webhook_url,
        pagerduty_routing_key = cfg.hitl.pagerduty_routing_key,
        enable_sse            = True,
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
    from registry import create_registry, RegistryConfig as RegCfg
    from a2a.agent_card import get_agent_card

    registry_config = RegCfg(
        lb_strategy                   = cfg.registry.lb_strategy,
        health_check_interval_seconds = cfg.registry.health_check_interval,
    )
    own_card = get_agent_card(cfg.server.a2a_base_url)
    registry = await create_registry(
        static_urls = cfg.registry.agent_urls,
        redis_url   = cfg.memory.redis_url,
        config      = registry_config,
        own_card    = own_card,
    )
    await registry.start()
    services["registry"] = registry
    logger.info(
        "Agent Registry ready — %d peer agent(s) pre-registered",
        len(cfg.registry.agent_urls),
    )

    # ── 4. Task module ───────────────────────────────────────────────────
    from task import create_task_system
    task_system = await create_task_system(
        hitl_router = hitl_router,
        review_svc  = review_service,
        registry    = registry,
    )
    services["task_system"] = task_system
    logger.info("Task module ready (registry-aware planner)")

    # ── 5. A2A executor ──────────────────────────────────────────────────
    from hitl import ITOpsHitlAgentExecutor
    executor = ITOpsHitlAgentExecutor(
        hitl_router    = hitl_router,
        review_service = review_service,
        audit_service  = hitl_audit,
        hitl_config    = hitl_config,
        memory_router  = memory_router,
        task_system    = task_system,
    )
    services["executor"] = executor
    logger.info("A2A executor ready")

    # ── 6. Integrations (MCP + OpenAPI + LLM + ToolRouter) ───────────────
    try:
        from integrations import (
            MCPClient, OpenAPIClient, LLMEngine, ToolRouter,
            patch_runtime_loop, patch_hitl_graph,
        )
        from tools import TOOL_REGISTRY, make_read_stored_result_tool
        from runtime import ToolResultStore

        tool_store = services.get("tool_store") or ToolResultStore()
        services["tool_store"] = tool_store

        # 6a. LLM engine
        llm_engine = LLMEngine.from_config({
            "backend":     cfg.llm.backend,
            "model":       cfg.llm.model,
            "base_url":    cfg.llm.base_url,
            "temperature": cfg.llm.temperature,
            "max_tokens":  cfg.llm.max_tokens,
        })
        patch_hitl_graph(llm_engine)
        services["llm_engine"] = llm_engine
        logger.info(
            "LLM engine ready: backend=%s model=%s",
            cfg.llm.backend, cfg.llm.model,
        )

        # 6b. MCP client
        if cfg.tools.mcp.config_json:
            import json as _json
            try:
                mcp_config_data = _json.loads(cfg.tools.mcp.config_json)
            except Exception:
                mcp_config_data = _json.loads(
                    pathlib.Path(cfg.tools.mcp.config_json).read_text()
                )
            mcp_client = MCPClient.from_config(mcp_config_data)
        elif cfg.tools.mcp.use_mock:
            mcp_client = MCPClient.from_netops_mock()
            logger.info("MCP: using built-in NetOps mock")
        else:
            mcp_client = MCPClient()
        await mcp_client.connect_all()
        services["mcp_client"] = mcp_client

        # 6c. OpenAPI client
        if cfg.tools.openapi.spec_url and cfg.tools.openapi.base_url:
            api_client = OpenAPIClient.from_url(
                name     = "netops_api",
                spec_url = cfg.tools.openapi.spec_url,
                base_url = cfg.tools.openapi.base_url,
                auth     = {
                    "type":      cfg.tools.openapi.auth_type,
                    "token_env": cfg.tools.openapi.token_env,
                },
            )
            await api_client.load()
            logger.info(
                "OpenAPI client loaded %d operations",
                len(api_client.list_operations()),
            )
        elif cfg.tools.openapi.use_mock:
            api_client = OpenAPIClient.netops_mock()
            await api_client.load()
            logger.info("OpenAPI: using built-in NetOps mock")
        else:
            api_client = None
        services["api_client"] = api_client

        # 6d. ToolRouter
        tool_registry_local = dict(TOOL_REGISTRY)
        read_stored_fn, process_chunks_fn = make_read_stored_result_tool(tool_store)
        tool_registry_local["read_stored_result"]    = read_stored_fn
        tool_registry_local["process_stored_chunks"] = process_chunks_fn

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

        # 6e. Patch runtime loop with real LLM + tools
        real_registry = router.registry
        if "runtime_loop" in services:
            services["runtime_loop"]._tool_registry = real_registry
            patch_runtime_loop(services["runtime_loop"], llm_engine)
            logger.info("Runtime loop: patched with real LLM + ToolRouter")
        executor._tool_registry = real_registry

    except Exception as exc:
        logger.warning(
            "Integrations layer failed to initialise (%s). "
            "Running with mock tools and stub LLM.",
            exc,
        )

    # ── 7. Hermes + Dual-Track Memory ────────────────────────────────────
    try:
        from memory.fts_store  import FTS5SessionStore
        from memory.curator    import MemoryCurator
        from memory.user_model import UserModelEngine
        from memory.dual_track import DualTrackMemory
        from skills.evolver    import SkillEvolver

        _data_dir = pathlib.Path(cfg.memory.data_dir)
        _data_dir.mkdir(parents=True, exist_ok=True)

        fts_store = FTS5SessionStore(db_path=str(_data_dir / "state.db"))
        await fts_store.initialize()

        # Build shared LLM callable for Hermes modules
        llm_engine = services.get("llm_engine")
        llm_fn = None
        if llm_engine:
            async def _llm_fn(system: str, user: str, _e=llm_engine) -> str:
                if hasattr(_e, "_chat"):
                    messages = [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ]
                    raw = await _e._chat(messages)
                    return _e._strip_think(raw) if hasattr(_e, "_strip_think") else raw
                return await _e.call(user, context=system)
            llm_fn = _llm_fn

        # FTS5 summarizer (uses same LLM)
        if llm_fn:
            async def _fts_summarizer(results, query, _fn=llm_fn):
                import datetime as _dt
                excerpts = "\n".join(
                    f"[{_dt.datetime.fromtimestamp(r.ts).strftime('%Y-%m-%d')}] "
                    f"{r.user_text[:120]} → {r.assistant_text[:120]}"
                    for r in results[:5]
                )
                return await _fn(
                    "You are an IT operations assistant. "
                    "Summarise past session excerpts in 2-3 concise sentences.",
                    f"Query: {query}\n\nPast session excerpts:\n{excerpts}",
                )
            fts_store._summarizer = _fts_summarizer

        skill_catalog  = services.get("skill_catalog")
        memory_router2 = services.get("memory")

        memory_curator = MemoryCurator(
            fts_store     = fts_store,
            memory_router = memory_router2 or _NullMemoryRouter(),
            llm_fn        = llm_fn,
        )
        user_model = UserModelEngine(
            fts_store = fts_store,
            llm_fn    = llm_fn,
        )
        skill_evolver = SkillEvolver(
            catalog    = skill_catalog or _NullSkillCatalog(),
            llm_fn     = llm_fn,
            fts_store  = fts_store,
            skills_dir = str(_data_dir / "skills"),
        )

        # Dual-Track Memory — reads all tuning from cfg
        dtm = DualTrackMemory(
            fts_store               = fts_store,
            curator                 = memory_curator,
            user_model              = user_model,
            data_dir                = str(_data_dir),
            llm_fn                  = llm_fn,
            compaction_turns        = cfg.memory.dtm.compaction_turns,
            nudge_turns             = cfg.memory.dtm.nudge_turns,
            track_b_weight          = cfg.memory.dtm.track_b_weight,
            temporal_half_life_days = cfg.memory.dtm.temporal_half_life_days,
        )

        services.update({
            "fts_store":      fts_store,
            "memory_curator": memory_curator,
            "user_model":     user_model,
            "skill_evolver":  skill_evolver,
            "dtm":            dtm,
            "_llm_backend":   cfg.llm.backend,
            "_llm_model":     cfg.llm.model,
            "_hermes_data":   str(_data_dir / "state.db"),
            "_mcp_mock":      cfg.tools.mcp.use_mock,
        })

        # Inject into executor
        executor = services.get("executor")
        if executor:
            executor._fts_store     = fts_store
            executor._curator       = memory_curator
            executor._user_model    = user_model
            executor._skill_evolver = skill_evolver
            executor._skill_catalog = skill_catalog
            executor._dtm           = dtm

        logger.info(cfg.dump_summary())

    except Exception as exc:
        logger.warning("Hermes learning loop failed to initialise: %s", exc)

    return services


# ---------------------------------------------------------------------------
# Null-object stubs
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
    logger.info("Starting IT Ops Agent v5 (config.yaml driven)")
    _services = await build_services()

    # Mount sub-apps (must happen before yield so routes are live)
    from a2a import create_a2a_app, InMemoryTaskStore
    a2a_app = create_a2a_app(
        base_url   = cfg.server.a2a_base_url,
        executor   = _services["executor"],
        task_store = InMemoryTaskStore(),
    )
    app.mount("/api/v1/a2a", a2a_app)

    from hitl.router import create_hitl_router
    from hitl.review import get_sse_channel
    hitl_api = create_hitl_router(
        decision_router = _services["hitl_router"],
        audit           = _services["hitl_audit"],
        sse_channel     = get_sse_channel(),
    )
    app.include_router(hitl_api, prefix="/hitl")

    from registry.router import create_registry_router
    reg_api = create_registry_router(_services["registry"])
    app.include_router(reg_api, prefix="/registry")

    logger.info("Modules mounted: /api/v1/a2a · /hitl · /registry")

    from webui.backend import create_webui_app
    webui = create_webui_app(_services)
    app.mount("/webui", webui)
    logger.info("WebUI mounted at /webui")

    from memory.consolidation import LifecycleManager
    lifecycle      = LifecycleManager(_services["memory"])
    watchdog_task  = asyncio.create_task(_services["hitl_watchdog"].run())
    lifecycle_task = asyncio.create_task(lifecycle.run())
    logger.info("Background tasks started")

    yield

    # Graceful shutdown
    _services["hitl_watchdog"].stop()
    await _services["registry"].stop()
    lifecycle.stop()
    for t in (watchdog_task, lifecycle_task):
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
    logger.info("IT Ops Agent v5 shut down cleanly")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "IT Ops Monitoring Agent",
    version     = "5.0.0",
    description = "IT Ops AI Agent — A2A · HITL · Memory · Task · Agent Registry",
    lifespan    = lifespan,
)

from fastapi.responses import FileResponse

@app.get("/", include_in_schema=False)
async def serve_webui():
    html_path = pathlib.Path(__file__).parent / "webui" / "index.html"
    return FileResponse(str(html_path), media_type="text/html")


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
        "version": "5.0.0",
        "registry": {
            "total_agents":   len(agents),
            "healthy_agents": healthy,
        },
        "pending_tasks":           pending_tasks,
        "pending_hitl_interrupts": pending_hitl,
    }


# ---------------------------------------------------------------------------
# Entry point — `python main.py` or `uvicorn main:app`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host      = cfg.server.host,
        port      = cfg.server.port,
        reload    = cfg.server.reload,
        log_level = "info",
    )