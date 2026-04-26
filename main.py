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

# Install log redaction filter ASAP — before any module logs anything sensitive.
# Scrubs passwords, secrets, community strings, Bearer tokens before they reach
# any handler (console, file, syslog).
from log_redaction import install_log_filter
install_log_filter()

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
    # Production memory backend: agent_memory.MemoryManager wrapped by
    # MemoryAdapter for async + per-operator scoping.
    # Multi-user isolated, SQLite WAL persistent, 311 unit tests.
    from memory import MemoryAdapter
    memory_router = MemoryAdapter(
        data_dir          = cfg.memory.data_dir,
        inline_threshold  = 4_000,
        session_ttl       = 86_400,
        enable_user_model = True,
    )
    services["memory"] = memory_router
    logger.info("Memory module ready (agent_memory backend)")

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
        from tools import make_read_stored_result_tool
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

        # Wire the LLM into the memory module so fact extraction works for
        # any language (default rule-based extractor is English-only regex).
        try:
            import asyncio as _asyncio
            def _sync_llm_for_memory(prompt: str) -> str:
                """
                Sync wrapper for the FactExtractor. Uses the engine's lightweight
                _chat primitive (single-message, no full system prompt) so the
                extractor sees only its own EXTRACT_PROMPT.
                """
                messages = [{"role": "user", "content": prompt}]
                async def _go():
                    if hasattr(llm_engine, "_chat"):
                        return await llm_engine._chat(messages)
                    return await llm_engine.call(prompt, "", state=None)
                try:
                    return _asyncio.run(_go())
                except RuntimeError:
                    new_loop = _asyncio.new_event_loop()
                    try:
                        return new_loop.run_until_complete(_go())
                    finally:
                        new_loop.close()
            memory_router.set_llm_fn(_sync_llm_for_memory)
        except Exception as _exc:
            logger.warning("memory llm_fn wiring failed: %s — facts will use rule-based extraction", _exc)

        # 6b. Real embeddings — always (both modes)
        try:
            from integrations.embedder import build_embedder
            embedder = build_embedder(cfg.embeddings)
            services["embedder"] = embedder
            logger.info("Embedder: %s/%s dim=%d",
                        cfg.embeddings.backend, cfg.embeddings.model, cfg.embeddings.dim)
        except Exception as exc:
            logger.warning("Embedder init failed (%s) — using hash stub", exc)

        # 6c. Build tool registry via ToolLoader — single source of truth by mode.
        # ToolLoader assembles: builtin tools + mode-specific tools (mock XOR pragmatic).
        # No tool name is hardcoded here or in llm_engine — metadata comes from registries.
        from tools.loader import ToolLoader as _ToolLoader
        _loader = _ToolLoader(mode=cfg.mode)
        read_stored_fn, process_chunks_fn = make_read_stored_result_tool(tool_store)
        tool_registry_local = _loader.build_callables()
        tool_registry_local["read_stored_result"]    = read_stored_fn
        tool_registry_local["process_stored_chunks"] = process_chunks_fn
        # Store loader on services so llm_engine can build the dynamic tool section
        services["tool_loader"] = _loader
        logger.info("ToolLoader[%s]: %d tools assembled", cfg.mode, len(tool_registry_local))
        
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

        # ── Skill catalog — built from ToolLoader.skill_definitions() ──────────────
        # Skills are mode-specific: only the correct set is loaded.
        # No cross-mode contamination; no filter_to_registry needed here because
        # the loader only returns skills valid for the current mode.
        try:
            from skills import SkillCatalogService
            _skill_defs = _loader.skill_definitions()
            _skill_catalog = SkillCatalogService()
            _skill_catalog.register_all(_skill_defs)
            services["skill_catalog"] = _skill_catalog
            logger.info(
                "SkillCatalog[%s]: %d skills registered", cfg.mode, len(_skill_catalog._skills)
            )
        except Exception as _sc_exc:
            logger.warning("SkillCatalog: build failed (%s) — catalog unavailable", _sc_exc)

        logger.info("Runtime loop and HITL graph patched with real LLM + tool registry")

    except Exception as exc:
        logger.warning("Integrations layer failed (%s). Running degraded.", exc)

    # MemoryAdapter (set above as services["memory"]) wraps agent_memory.MemoryManager,
    # which handles its own embedding internally — no separate injection step needed.

    return services


# ─────────────────────────────────────────────────────────────────────────────
# Mode-specific helpers
# ─────────────────────────────────────────────────────────────────────────────



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
    watchdog_task  = asyncio.create_task(_services["hitl_watchdog"].run())

    yield

    _services["hitl_watchdog"].stop()
    await _services["registry"].stop()
    for t in (watchdog_task,):
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
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
