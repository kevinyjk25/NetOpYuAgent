"""
main.py  [v6 — mode-aware: mock | pragmatic]
--------------------------------------------
Both modes: real LLM (Ollama/OpenAI/Anthropic), real embeddings, real Redis.
Mode only controls tools/MCP:

  mock       — Built-in simulated network tools (from_netops_mock + mock_tools.py).
               Zero device credentials needed. Good for dev/demo/CI.

  pragmatic  — Real device access via Netmiko/NAPALM/Nornir.
               Add devices to pragmatic.device_inventory in config.yaml.
               Optional: additional real MCP servers in pragmatic.mcp_servers.

Switch:
    MODE=pragmatic uvicorn main:app --port 8001 --reload
    # or in config.yaml:  mode: "pragmatic"
"""
from __future__ import annotations

import asyncio
import logging
import pathlib
from contextlib import asynccontextmanager
from typing import Any

from config import cfg
import logging_config as _lc
_lc.configure(mode=cfg.logging.mode)

from fastapi import FastAPI

logger = logging.getLogger(__name__)


def _print_banner() -> None:
    if cfg.is_pragmatic:
        n_dev = len(cfg.pragmatic.device_inventory)
        logger.info(
            "\n╔══════════════════════════════════════════════════════╗\n"
            "║  🔧  PRAGMATIC MODE                                  ║\n"
            f"║  Real devices: {n_dev:<36} ║\n"
            "║  Real LLM · Real embeddings · Real Redis (if set)    ║\n"
            "╚══════════════════════════════════════════════════════╝"
        )
    else:
        logger.info(
            "\n╔══════════════════════════════════════════════════════╗\n"
            "║  🎭  MOCK MODE                                       ║\n"
            "║  Simulated tools · Real LLM · Real embeddings        ║\n"
            "║  Set MODE=pragmatic to connect real devices           ║\n"
            "╚══════════════════════════════════════════════════════╝"
        )


class _NullMemoryRouter:
    async def ingest_entity(self, *a, **kw): pass
    async def retrieve(self, *a, **kw): return []

class _NullSkillCatalog:
    def register_all(self, d): pass
    def load_detail(self, s): return None
    def get_summary(self, s): return None
    def format_summary(self): return ""


async def build_services() -> dict[str, Any]:
    services: dict[str, Any] = {}
    _print_banner()

    # ── 1. Memory ────────────────────────────────────────────────────────────
    from memory import create_memory_router
    memory_router = await create_memory_router(
        redis_url     = cfg.memory.redis_url,
        postgres_dsn  = cfg.memory.postgres_dsn,
        chroma_path   = cfg.memory.chroma_path,
        embedding_dim = cfg.embeddings.dim,
    )
    services["memory"] = memory_router
    logger.info("Memory module ready")

    # ── 2. HITL ──────────────────────────────────────────────────────────────
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
    hitl_audit     = HitlAuditService.sqlite("data/hitl_audit.db")
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

    # ── 3. Registry ──────────────────────────────────────────────────────────
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
    logger.info("Agent Registry ready — %d peer(s)", len(cfg.registry.agent_urls))

    # ── 4. Task module ───────────────────────────────────────────────────────
    from task import create_task_system
    task_system = await create_task_system(
        hitl_router = hitl_router,
        review_svc  = review_service,
        registry    = registry,
    )
    services["task_system"] = task_system
    logger.info("Task module ready")

    # ── 5. A2A executor ──────────────────────────────────────────────────────
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

    # ── 6. Integrations ──────────────────────────────────────────────────────
    try:
        from integrations import (
            MCPClient, OpenAPIClient, LLMEngine, ToolRouter,
            patch_runtime_loop, patch_hitl_graph,
        )
        from tools import TOOL_REGISTRY, make_read_stored_result_tool
        from runtime import ToolResultStore

        tool_store = ToolResultStore()
        services["tool_store"] = tool_store

        # 6a. LLM engine — always real (both modes)
        llm_engine = LLMEngine.from_config({
            "backend":     cfg.llm.backend,
            "model":       cfg.llm.model,
            "base_url":    cfg.llm.base_url,
            "temperature": cfg.llm.temperature,
            "max_tokens":  cfg.llm.max_tokens,
        })
        services["llm_engine"] = llm_engine  # patch_hitl_graph called after tool registry is built
        logger.info("LLM engine: %s/%s", cfg.llm.backend, cfg.llm.model)

        # 6b. Real embeddings — always (both modes)
        try:
            from integrations.embedder import build_embedder
            embedder = build_embedder(cfg.embeddings)
            services["embedder"] = embedder
            logger.info("Embedder: %s/%s dim=%d",
                        cfg.embeddings.backend, cfg.embeddings.model, cfg.embeddings.dim)
        except Exception as exc:
            logger.warning("Embedder init failed (%s) — using hash stub", exc)

        # 6c. Build tool registry based on mode
        tool_registry_local = dict(TOOL_REGISTRY)   # always include mock_tools as base
        read_stored_fn, process_chunks_fn = make_read_stored_result_tool(tool_store)
        tool_registry_local["read_stored_result"]    = read_stored_fn
        tool_registry_local["process_stored_chunks"] = process_chunks_fn

        if cfg.is_pragmatic:
            tool_registry_local = await _build_pragmatic_tools(tool_registry_local)
        
        # 6d. MCP client
        mcp_client = await _build_mcp_client(MCPClient)
        await mcp_client.connect_all()
        services["mcp_client"] = mcp_client

        # 6e. OpenAPI client (mock in both modes unless explicitly configured)
        api_client = await _build_openapi_client(OpenAPIClient)
        services["api_client"] = api_client

        # 6f. Pragmatic extra MCP servers
        extra_mcp_clients = []
        if cfg.is_pragmatic and cfg.pragmatic.mcp_servers:
            extra_mcp_clients = await _load_pragmatic_mcp_servers(MCPClient)

        # 6g. ToolRouter
        router = ToolRouter(tool_store=tool_store)
        router.register_mcp(mcp_client)
        for ec in extra_mcp_clients:
            router.register_mcp(ec)
        if api_client:
            router.register_openapi(api_client)
        router.register_local(tool_registry_local)
        services["tool_router"] = router
        counts = router.tool_count()
        logger.info("ToolRouter: mcp=%d openapi=%d local=%d",
                    counts.get("mcp", 0), counts.get("openapi", 0), counts.get("local", 0))

        real_registry = router.registry
        executor._tool_registry = real_registry
        patch_hitl_graph(llm_engine, tool_registry=real_registry)
        patch_runtime_loop(executor, llm_engine)

        # ── Wire prompt-based PolicyEngine ────────────────────────────────────
        # Loads policies from config.yaml and registers the global singleton.
        # classify(), pre_verify(), and HITL triggers use this automatically.
        try:
            from runtime.policy_engine import (
                PolicyEngine, load_policies_from_config, set_policy_engine
            )
            # cfg.policies is populated from config.yaml at import time
            _policy_defs = load_policies_from_config(cfg.policies)

            async def _policy_llm_call(system: str, user: str) -> str:
                """Thin wrapper: call the real LLM for policy evaluation (JSON output)."""
                # Use a minimal context so the policy call is fast and cheap
                return await llm_engine.call(
                    query=user,
                    context=system,
                    state=None,
                    skill_catalog=None,
                )

            _policy_engine = PolicyEngine(
                policies    = _policy_defs,
                llm_call    = _policy_llm_call,
                cache_ttl_s = 120,
            )
            set_policy_engine(_policy_engine)
            logger.info(
                "PolicyEngine: wired with %d policies from config.yaml",
                len(_policy_defs),
            )
        except Exception as _pe_exc:
            logger.warning("PolicyEngine: startup failed (%s) — keyword heuristics active", _pe_exc)
        logger.info("Runtime loop and HITL graph patched with real LLM + tool registry")

    except Exception as exc:
        logger.warning("Integrations layer failed (%s). Running degraded.", exc)

    # ── 7. Embedder injection into memory pipelines ───────────────────────────
    embedder = services.get("embedder")
    if embedder:
        try:
            from memory.pipelines.ingestion import IngestionPipeline
            _inject_embedder(memory_router, embedder, IngestionPipeline)
        except Exception as exc:
            logger.warning("Embedder injection failed: %s", exc)

    # ── 8. Hermes + DTM ──────────────────────────────────────────────────────
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
        user_model = UserModelEngine(fts_store=fts_store, llm_fn=llm_fn)
        skill_evolver = SkillEvolver(
            catalog    = skill_catalog or _NullSkillCatalog(),
            llm_fn     = llm_fn,
            fts_store  = fts_store,
            skills_dir = str(_data_dir / "skills"),
        )
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
        logger.warning("Hermes DTM failed to initialise: %s", exc)

    return services


# ─────────────────────────────────────────────────────────────────────────────
# Mode-specific helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _build_pragmatic_tools(base_registry: dict) -> dict:
    """Load real Netmiko/NAPALM tools and register devices from config."""
    from tools.pragmatic_tools import PRAGMATIC_TOOL_REGISTRY, register_devices
    if not cfg.pragmatic.device_inventory:
        logger.warning(
            "Pragmatic mode: no devices in pragmatic.device_inventory — "
            "real device tools registered but will return 'no devices' until config populated."
        )
    else:
        register_devices(cfg.pragmatic.device_inventory)
    # Pragmatic tools override mock tools with same name
    merged = dict(base_registry)
    merged.update(PRAGMATIC_TOOL_REGISTRY)
    logger.info("Pragmatic tools loaded: %s", list(PRAGMATIC_TOOL_REGISTRY.keys()))
    return merged


async def _build_mcp_client(MCPClient):
    import json as _json
    if cfg.tools.mcp.config_json:
        try:
            try:
                mcp_data = _json.loads(cfg.tools.mcp.config_json)
            except Exception:
                mcp_data = _json.loads(pathlib.Path(cfg.tools.mcp.config_json).read_text())
            client = MCPClient.from_config(mcp_data)
            logger.info("MCP: using config_json")
            return client
        except Exception as exc:
            logger.warning("MCP config_json failed (%s), using mock", exc)
    if cfg.tools.mcp.use_mock:
        logger.info("MCP: using built-in NetOps mock")
        return MCPClient.from_netops_mock()
    return MCPClient()


async def _build_openapi_client(OpenAPIClient):
    if cfg.tools.openapi.spec_url and cfg.tools.openapi.base_url:
        try:
            client = OpenAPIClient.from_url(
                name     = "netops_api",
                spec_url = cfg.tools.openapi.spec_url,
                base_url = cfg.tools.openapi.base_url,
                auth     = {"type": cfg.tools.openapi.auth_type,
                            "token_env": cfg.tools.openapi.token_env},
            )
            await client.load()
            logger.info("OpenAPI: %d operations", len(client.list_operations()))
            return client
        except Exception as exc:
            logger.warning("OpenAPI spec failed (%s), using mock", exc)
    if cfg.tools.openapi.use_mock:
        client = OpenAPIClient.netops_mock()
        await client.load()
        logger.info("OpenAPI: using mock")
        return client
    return None


async def _load_pragmatic_mcp_servers(MCPClient) -> list:
    clients = []
    for srv in cfg.pragmatic.mcp_servers:
        try:
            srv_dict = {srv.name: {
                "transport": srv.transport,
                "url":       srv.url,
                "command":   srv.command,
                "auth":      srv.auth,
            }}
            client = MCPClient.from_config(srv_dict)
            await client.connect_all()
            clients.append(client)
            logger.info("Pragmatic MCP: %s (%s)", srv.name, srv.transport)
        except Exception as exc:
            logger.warning("Pragmatic MCP %s failed: %s", srv.name, exc)
    return clients


def _inject_embedder(memory_router, embedder, IngestionPipeline) -> None:
    injected = 0
    for attr in ("_ingestion", "_pipeline", "pipeline", "ingestion"):
        obj = getattr(memory_router, attr, None)
        if isinstance(obj, IngestionPipeline):
            obj.set_embedder(embedder)
            injected += 1
    if injected == 0:
        logger.debug("_inject_embedder: no IngestionPipeline found in memory_router")
    else:
        logger.info("Embedder injected into %d IngestionPipeline(s)", injected)


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI lifespan
# ─────────────────────────────────────────────────────────────────────────────

_services: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _services
    logger.info("Starting IT Ops Agent v6 (mode=%s)", cfg.mode.upper())
    _services = await build_services()

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

    from webui.backend import create_webui_app
    webui = create_webui_app(_services)
    app.mount("/webui", webui)
    logger.info("All modules mounted")

    from memory.consolidation import LifecycleManager
    lifecycle      = LifecycleManager(_services["memory"])
    watchdog_task  = asyncio.create_task(_services["hitl_watchdog"].run())
    lifecycle_task = asyncio.create_task(lifecycle.run())

    yield

    _services["hitl_watchdog"].stop()
    await _services["registry"].stop()
    lifecycle.stop()
    for t in (watchdog_task, lifecycle_task):
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
    # Flush any pending DTM daily buffer so no turns are lost on shutdown
    dtm = _services.get("dtm")
    if dtm and hasattr(dtm, "_compact_today") and getattr(dtm, "_today_turns", []):
        try:
            await dtm._compact_today()
            logger.info("DTM: flushed %d turn(s) to daily file on shutdown",
                        len(getattr(dtm, "_today_turns", [])))
        except Exception as exc:
            logger.warning("DTM shutdown flush failed: %s", exc)
    logger.info("IT Ops Agent shut down cleanly")


app = FastAPI(
    title       = "IT Ops Monitoring Agent",
    version     = "6.0.0",
    description = "IT Ops AI Agent — A2A · HITL · Memory · Task · Registry",
    lifespan    = lifespan,
)

from fastapi.responses import FileResponse

@app.get("/", include_in_schema=False)
async def serve_webui():
    html_path = pathlib.Path(__file__).parent / "webui" / "index.html"
    return FileResponse(str(html_path), media_type="text/html")


@app.get("/health")
async def health():
    reg     = _services.get("registry")
    agents  = await reg.list_agents() if reg else []
    from registry.schemas import AgentHealthState
    healthy = sum(1 for a in agents if a.health == AgentHealthState.HEALTHY)
    task_sys     = _services.get("task_system")
    pending_tasks = len(await task_sys.store.list_pending()) if task_sys else 0
    hitl_rtr     = _services.get("hitl_router")
    pending_hitl = sum(
        1 for p in hitl_rtr._payload_store.values()
        if p.status.value == "pending"
    ) if hitl_rtr else 0
    return {
        "status":  "ok", "version": "6.0.0", "mode": cfg.mode,
        "registry": {"total": len(agents), "healthy": healthy},
        "pending_tasks": pending_tasks,
        "pending_hitl":  pending_hitl,
    }


@app.get("/mode")
async def get_mode():
    n_dev = len(cfg.pragmatic.device_inventory)
    return {
        "mode":            cfg.mode,
        "llm":             f"{cfg.llm.backend}/{cfg.llm.model}",
        "embeddings":      f"{cfg.embeddings.backend}/{cfg.embeddings.model} dim={cfg.embeddings.dim}",
        "devices_in_cfg":  n_dev,
        "pragmatic_mcps":  len(cfg.pragmatic.mcp_servers),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host      = cfg.server.host,
        port      = cfg.server.port,
        reload    = cfg.server.reload,
        log_level = "info",
    )
