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
    # HITL_BACKEND env var picks which implementation to wire up:
    #   "langgraph" (default) — original hitl/* + LangGraph StateGraph
    #   "core"               — new hitl_core/ + integrations/hitl_executor
    # Both backends export the same external API surface (HitlRouter +
    # ITOpsHitlAgentExecutor), so the rest of main.py and webui/backend
    # don't change. Switch via:  HITL_BACKEND=core python main.py
    import os as _os
    _hitl_backend = (_os.getenv("HITL_BACKEND") or "langgraph").lower()
    logger.info("HITL backend selected: %s", _hitl_backend)

    if _hitl_backend == "core":
        # New path — hitl_core
        from hitl_core import (
            HitlPipeline, HitlRouter, InMemoryCheckpointStore,
            AuditLogger, InMemoryAuditSink, FileAuditSink,
            build_default_device_coreferencer,
        )

        # Pick checkpoint backend by env
        _cp_backend = (_os.getenv("HITL_CHECKPOINT_BACKEND") or "memory").lower()
        if _cp_backend == "redis":
            from hitl_core import RedisCheckpointStore
            hitl_store = RedisCheckpointStore(
                redis_url=_os.getenv("HITL_REDIS_URL", cfg.memory.redis_url
                                                       or "redis://localhost:6379/0"),
            )
        elif _cp_backend == "sqlite":
            from hitl_core import SqliteCheckpointStore
            hitl_store = SqliteCheckpointStore(
                db_path=_os.getenv("HITL_SQLITE_PATH", "data/hitl_checkpoints.db"),
            )
        else:
            hitl_store = InMemoryCheckpointStore()
        logger.info("HITL checkpoint backend: %s", _cp_backend)

        # Audit sink: file by default in production, in-memory for dev
        _audit_path = _os.getenv("HITL_AUDIT_LOG_PATH")
        if _audit_path:
            audit_sink = FileAuditSink(_audit_path)
        else:
            audit_sink = InMemoryAuditSink(max_records=10_000)
        audit_logger = AuditLogger(sink=audit_sink)

        # Build router + pipeline (sharing the batch coordinator)
        hitl_core_router = HitlRouter(
            store=hitl_store, on_audit=audit_logger.as_hook(),
        )
        hitl_core_pipeline = HitlPipeline(
            store=hitl_store,
            batch_coordinator=hitl_core_router.batch,  # share batch coord
            on_audit=audit_logger.as_hook(),
        )
        services.update(dict(
            hitl_core_router   = hitl_core_router,
            hitl_core_pipeline = hitl_core_pipeline,
            hitl_core_store    = hitl_store,
            hitl_core_audit    = audit_logger,
        ))

        # Stub the old-path objects with None so any leftover references
        # to services["hitl_router"] / services["hitl_audit"] error
        # loudly instead of silently doing the wrong thing.
        services.update(dict(
            hitl_router    = None,
            hitl_audit     = None,
            review_service = None,
            hitl_watchdog  = None,
            hitl_config    = None,
        ))
        logger.info("HITL module ready (backend=core, %d step(s) wired)",
                    len(hitl_core_pipeline._steps))

    else:
        # Legacy path — hitl/* + LangGraph
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
        # Stub the new-path objects so backend.py doesn't break either way
        services.update(dict(
            hitl_core_router=None, hitl_core_pipeline=None,
            hitl_core_store=None,  hitl_core_audit=None,
        ))
        logger.info("HITL module ready (backend=langgraph)")

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
    # Task system needs a hitl_router. In core mode we don't have the legacy
    # router, but task_system uses it only for register/decide which we can
    # adapter-shim. For now, in core mode we pass None and let the task
    # system gracefully degrade (HITL escalation in tasks goes through the
    # core router via the executor instead).
    from task import create_task_system
    task_system = await create_task_system(
        hitl_router = (hitl_router if _hitl_backend != "core" else None),
        review_svc  = (review_service if _hitl_backend != "core" else None),
        registry    = registry,
    )
    services["task_system"] = task_system
    logger.info("Task module ready")

    # ── 5. A2A executor ──────────────────────────────────────────────────────
    if _hitl_backend == "core":
        # New executor — built later, after llm_engine + tool_registry are
        # available (section 6). Stub the slot for now so any reference
        # to services["executor"] before then errors clearly.
        services["executor"] = None
        logger.info("A2A executor (core) — deferred to after LLM + tool wiring")
    else:
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
        logger.info("A2A executor ready (langgraph)")

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
            import concurrent.futures as _futures
            import threading as _threading

            # Fix: when we're inside an already-running event loop (uvicorn
            # startup), neither asyncio.run() nor a new_event_loop() works.
            # Instead, dispatch the coroutine to a dedicated background thread
            # that owns its own loop. This works in BOTH cases: with or
            # without an active loop on the calling thread.
            _bg_loop = _asyncio.new_event_loop()
            def _bg_runner():
                _asyncio.set_event_loop(_bg_loop)
                _bg_loop.run_forever()
            _bg_thread = _threading.Thread(target=_bg_runner, daemon=True, name="MemoryLLMLoop")
            _bg_thread.start()

            def _sync_llm_for_memory(prompt: str) -> str:
                """
                Sync wrapper for the FactExtractor. Submits the async call to
                a dedicated background event loop (started above) and blocks
                until done. Avoids the "loop already running" RuntimeWarning
                that happens when this is called from inside FastAPI startup.
                """
                messages = [{"role": "user", "content": prompt}]
                async def _go():
                    if hasattr(llm_engine, "_chat"):
                        return await llm_engine._chat(messages)
                    return await llm_engine.call(prompt, "", state=None)
                # Submit to background loop, wait for result
                fut = _asyncio.run_coroutine_threadsafe(_go(), _bg_loop)
                # 60s budget for the smoke test + extraction calls
                return fut.result(timeout=60)
            memory_router.set_llm_fn(_sync_llm_for_memory)
            # Smoke test: invoke the wrapper once with a tiny prompt to confirm
            # the LLM is reachable. If this fails, fact extraction silently
            # falls back to English-only regex and B-track stays at 0 forever.
            try:
                _smoke = _sync_llm_for_memory(
                    "Return ONLY this JSON array, no other text: []"
                )
                logger.info(
                    "Memory LLM smoke test OK — response %d chars: %r",
                    len(_smoke or ""), (_smoke or "")[:120],
                )
            except Exception as _smk:
                logger.warning(
                    "Memory LLM smoke test FAILED: %s — fact extraction will "
                    "fall back to rule-based regex (English-only). Facts (track B) "
                    "will likely stay at 0 for non-English conversations.", _smk,
                )
        except Exception as _exc:
            logger.warning("memory llm_fn wiring failed: %s — facts will use rule-based extraction", _exc)

        # Attach the LLM engine to the HITL executor so that interrupts without
        # a specific tool callback (low_confidence triggers) can produce a real
        # LLM answer when approved, instead of empty no-op execution.
        # In core mode, the executor doesn't exist yet (built in section 6
        # after LLM + tool wiring) — that path constructs it WITH llm_engine
        # already, so there's nothing to attach here.
        _early_executor = services.get("executor")
        if _early_executor is not None:
            try:
                _early_executor._llm_engine = llm_engine
                logger.info("HITL executor: LLM-answer fallback enabled")
            except Exception as _exc:
                logger.warning("HITL executor LLM wiring failed: %s", _exc)

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
        if _hitl_backend == "core":
            # Build the core executor now that we have llm_engine + tool_registry.
            # This replaces the legacy hitl/a2a_integration ITOpsHitlAgentExecutor
            # but presents the same external interface (.execute / .cancel)
            # so webui/backend doesn't change.
            from integrations.hitl_executor import HitlExecutor
            executor = HitlExecutor(
                runtime_loop=None,             # injected later from services["runtime_loop"]
                llm_engine=llm_engine,
                tool_registry=real_registry,
                memory_router=memory_router,
                hitl_router=services["hitl_core_router"],
                hitl_pipeline=services["hitl_core_pipeline"],
                audit_logger=services["hitl_core_audit"],
            )
            services["executor"] = executor
            logger.info("A2A executor (core) constructed with %d tool(s)",
                        len(real_registry))
            # Don't patch the langgraph hitl_graph — it doesn't exist in core mode.
            # Don't patch_runtime_loop(executor, ...) either — that's a legacy
            # path for executors that have their own internal AgentRuntimeLoop;
            # the core executor uses the external (already-patched) services
            # ["runtime_loop"] instance instead, which gets patched separately
            # in webui/backend.py at startup.
        else:
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

        # ── SkillEvolver — wires the LLM so /skills/generate produces real content ─
        # Without this, backend.py auto-creates a fallback evolver with no LLM,
        # and every "Generate skill from text" call returns the hardcoded
        # _stub_llm output (the generic Network Diagnostic Procedure markdown).
        try:
            from skills.evolver import SkillEvolver
            import os as _os, pathlib as _pl

            # Async wrapper matching SkillEvolver._call_llm's signature:
            #   async (system: str, user: str) -> str
            # SkillEvolver expects raw markdown OR a JSON object string back.
            async def _async_llm_for_skills(system: str, user: str) -> str:
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ]
                if hasattr(llm_engine, "_chat"):
                    return await llm_engine._chat(messages)
                # Fallback: call() takes (user, system, state)
                return await llm_engine.call(user, system, state=None)

            _skills_dir = _os.getenv("HERMES_DATA_DIR", "./data")
            _skill_evolver = SkillEvolver(
                catalog    = services["skill_catalog"],
                llm_fn     = _async_llm_for_skills,
                skills_dir = str(_pl.Path(_skills_dir) / "skills"),
            )
            services["skill_evolver"] = _skill_evolver

            # Smoke test: prove the LLM actually responds. We don't care about
            # the content — just that no exception is raised. If this fails,
            # /skills/generate would silently fall back to stubs.
            try:
                _probe = await _async_llm_for_skills(
                    "You are a helper. Reply with a single short word.",
                    "Reply with: ok",
                )
                logger.info(
                    "SkillEvolver: LLM smoke test OK — got %d chars: %r",
                    len(_probe or ""), (_probe or "")[:80],
                )
            except Exception as _se_exc:
                logger.warning(
                    "SkillEvolver: LLM smoke test FAILED (%s) — /skills/generate "
                    "will fall back to stub content. Verify llm_engine is reachable.",
                    _se_exc,
                )
        except Exception as _se_outer:
            logger.warning(
                "SkillEvolver setup failed (%s) — backend will create a no-LLM "
                "fallback evolver and /skills/generate will return stub content.",
                _se_outer,
            )

        logger.info("Runtime loop and HITL graph patched with real LLM + tool registry")

        # ── Schema registry — auto-import from MCP + OpenAPI + dict metadata ──
        # One unified ArgSchema per tool. ToolRouter validates+coerces args
        # against this schema before dispatch — catches LLM shape mistakes
        # before they crash the tool implementation.
        try:
            from schema import (
                get_schema_registry, from_mcp_input_schema,
                from_openapi_operation, from_dict_metadata,
            )
            schema_reg = get_schema_registry()

            n_mcp = n_oa = n_dict = 0

            # 1) MCP tools (already JSON Schema in inputSchema)
            try:
                mcp_clients = services.get("mcp_clients") or []
                if not mcp_clients and services.get("mcp_client"):
                    mcp_clients = [services["mcp_client"]]
                for client in mcp_clients:
                    for srv in getattr(client, "_servers", {}).values():
                        for spec in getattr(srv, "_tools", []) or []:
                            spec_dict = {
                                "name":         getattr(spec, "name", ""),
                                "description":  getattr(spec, "description", ""),
                                "inputSchema":  getattr(spec, "input_schema", {}) or
                                                getattr(spec, "inputSchema", {}),
                            }
                            if spec_dict["name"]:
                                schema_reg.register(
                                    from_mcp_input_schema(spec_dict["name"], spec_dict)
                                )
                                n_mcp += 1
            except Exception as _mcp_exc:
                logger.warning("Schema import from MCP failed: %s", _mcp_exc)

            # 2) OpenAPI operations
            try:
                openapi_clients = services.get("openapi_clients") or []
                if not openapi_clients and services.get("openapi_client"):
                    openapi_clients = [services["openapi_client"]]
                for client in openapi_clients:
                    for op in getattr(client, "_operations", []) or []:
                        op_dict = {
                            "operationId": op.operation_id if hasattr(op, "operation_id") else "",
                            "description": getattr(op, "description", ""),
                            "summary":     getattr(op, "summary", ""),
                            "parameters":  [],
                            "requestBody": getattr(op, "request_body", {}) or {},
                        }
                        # Convert ParamSpec list to OpenAPI shape
                        for p in getattr(op, "parameters", []) or []:
                            op_dict["parameters"].append({
                                "name":        getattr(p, "name", ""),
                                "in":          getattr(p, "location", "query"),
                                "required":    getattr(p, "required", False),
                                "description": getattr(p, "description", ""),
                                "schema":      getattr(p, "schema", {"type": "string"}),
                            })
                        if op_dict["operationId"]:
                            schema_reg.register(
                                from_openapi_operation(op_dict["operationId"], op_dict)
                            )
                            n_oa += 1
            except Exception as _oa_exc:
                logger.warning("Schema import from OpenAPI failed: %s", _oa_exc)

            # 3) Local tool metadata (dict format) — last so overrides auto-imports
            try:
                from tools.loader import ToolLoader as _TL
                _meta = _TL(mode=cfg.mode).build_metadata()
                for tool_name, tool_meta in _meta.items():
                    schema_reg.register(from_dict_metadata(tool_name, tool_meta))
                    n_dict += 1
            except Exception as _dict_exc:
                logger.warning("Schema import from dict metadata failed: %s", _dict_exc)

            services["schema_registry"] = schema_reg

            logger.info(
                "Schema registry: %d total schemas (mcp=%d openapi=%d dict=%d)",
                len(schema_reg), n_mcp, n_oa, n_dict,
            )
        except Exception as _sch_exc:
            logger.warning(
                "Schema registry wiring failed (%s) — ToolRouter will skip validation",
                _sch_exc,
            )

        # ── SkillJournalConsumer — auto-feedback to SkillEvolver ──────
        # Periodically scans SkillJournal stats; for skills that are
        # consistently dormant (loaded but no tool calls), generates
        # structured feedback and calls SkillEvolver.apply_feedback().
        # Closes the loop between observability (Plan A) and learning.
        try:
            _so_cfg = getattr(cfg, "skill_orchestration", None)
            if _so_cfg and getattr(_so_cfg, "evolver_feedback_enabled", True):
                from skills.journal_consumer import SkillJournalConsumer
                from runtime.skill_journal import get_journal_store
                _evolver = services.get("skill_evolver")
                if _evolver is not None:
                    consumer = SkillJournalConsumer(
                        evolver=_evolver,
                        journal_store=get_journal_store(),
                        interval_s=int(getattr(_so_cfg, "evolver_feedback_interval_s", 300)),
                        min_uses=int(getattr(_so_cfg, "evolver_feedback_min_uses", 3)),
                        dormant_threshold=float(getattr(_so_cfg, "evolver_dormant_threshold", 0.6)),
                    )
                    services["skill_journal_consumer"] = consumer
                    async def _start_consumer():
                        await consumer.start()
                    async def _stop_consumer():
                        await consumer.stop()
                    services["_start_consumer"] = _start_consumer
                    services["_stop_consumer"]  = _stop_consumer
                    logger.info(
                        "SkillJournalConsumer ready (interval=%ds, dormant_thr=%.2f) — starts on app startup",
                        consumer._interval, consumer._dormant_threshold,
                    )
                else:
                    logger.info(
                        "SkillJournalConsumer skipped — no SkillEvolver available "
                        "(mode=%s)", cfg.mode,
                    )
            else:
                logger.info("SkillJournalConsumer disabled (skill_orchestration.evolver_feedback_enabled=false)")
        except Exception as _jc_exc:
            logger.warning("SkillJournalConsumer setup failed: %s — auto-feedback off", _jc_exc)

        # ── Retrieval framework + meta-tool wiring ─────────────────────────
        # Replaces the full-catalog dump in _build_system_prompt with top-K
        # semantic retrieval, plus an extensible meta-tool registry.
        # Drops prompt size 30-50% on systems with >10 tools/skills.
        try:
            from retrieval import (
                build_tool_retriever, build_skill_retriever,
                get_meta_tool_registry,
                make_list_tools_meta_tool, make_list_skills_meta_tool,
                make_tool_details_meta_tool,
            )

            _embedder = services.get("embedder")
            _loader   = services.get("tool_loader")  # may be None on early errors

            # Build the corpora.  Tool metadata: prefer the live ToolLoader
            # used at startup; skill defs come from the same source.
            from tools.loader import ToolLoader as _TL
            _tool_meta = _TL(mode=cfg.mode).build_metadata()
            _skill_defs = _TL(mode=cfg.mode).skill_definitions()

            # Choose async or sync indexing path based on backend + embedder presence.
            # Async path: bounded-concurrency batched embed() calls, ~5-10x faster
            # startup vs the per-item thread-spawn fallback.
            _use_async_index = (
                _embedder is not None and
                cfg.retrieval.backend in ("hybrid", "embedding")
            )
            if _use_async_index:
                # Reuse the background event loop spawned for memory LLM smoke test
                # if available; otherwise schedule on a fresh helper thread.
                try:
                    import asyncio as _asyncio
                    import concurrent.futures as _futs
                    from retrieval import (
                        build_tool_retriever_async, build_skill_retriever_async,
                    )

                    # Try to use the bg loop from earlier in build_services
                    _bg = locals().get("_bg_loop")
                    # Build judge_llm_fn for backend=llm_judge.
                    # Re-use _async_llm_for_skills if available (already wires
                    # the active LLM engine via _chat); else build a thin shim.
                    _judge_fn = locals().get("_async_llm_for_skills")
                    if _judge_fn is None and llm_engine is not None:
                        async def _judge_fn(system: str, user: str) -> str:   # noqa: F811
                            messages = [
                                {"role": "system", "content": system},
                                {"role": "user",   "content": user},
                            ]
                            if hasattr(llm_engine, "_chat"):
                                return await llm_engine._chat(messages)
                            return await llm_engine.call(user, system, state=None)

                    if _bg is not None:
                        tool_fut  = _asyncio.run_coroutine_threadsafe(
                            build_tool_retriever_async (
                                cfg, _embedder, _tool_meta, judge_llm_fn=_judge_fn,
                            ), _bg,
                        )
                        skill_fut = _asyncio.run_coroutine_threadsafe(
                            build_skill_retriever_async(
                                cfg, _embedder, _skill_defs, judge_llm_fn=_judge_fn,
                            ), _bg,
                        )
                        # Generous timeout: 17 tools × 30s embed timeout / 8 concurrency
                        tool_retriever  = tool_fut.result(timeout=180)
                        skill_retriever = skill_fut.result(timeout=180)
                    else:
                        # No bg loop — fall back to sync indexing (still works,
                        # just slower at startup).
                        raise RuntimeError("no bg loop available; using sync path")
                except Exception as _async_exc:
                    logger.info(
                        "Retrieval: async indexing path unavailable (%s) — using sync",
                        _async_exc,
                    )
                    _judge_fn_sync = locals().get("_async_llm_for_skills")
                    tool_retriever  = build_tool_retriever (
                        cfg, _embedder, _tool_meta, judge_llm_fn=_judge_fn_sync,
                    )
                    skill_retriever = build_skill_retriever(
                        cfg, _embedder, _skill_defs, judge_llm_fn=_judge_fn_sync,
                    )
            else:
                # Pure BM25/keyword path — sync indexing is fast (millis per item)
                _judge_fn_pure = locals().get("_async_llm_for_skills")
                tool_retriever  = build_tool_retriever (
                    cfg, _embedder, _tool_meta, judge_llm_fn=_judge_fn_pure,
                )
                skill_retriever = build_skill_retriever(
                    cfg, _embedder, _skill_defs, judge_llm_fn=_judge_fn_pure,
                )
            services["tool_retriever"]  = tool_retriever
            services["skill_retriever"] = skill_retriever

            # Attach the retriever to the SkillCatalogService so its
            # select_skills_for_query() uses Hybrid (BM25+embedding) scoring
            # instead of the legacy keyword path. This is the production
            # upgrade for skill scoring accuracy on CJK / paraphrase / rare-word
            # queries.
            try:
                _sc = services.get("skill_catalog")
                if _sc is not None and hasattr(_sc, "attach_retriever"):
                    _sc.attach_retriever(skill_retriever)
                    logger.info(
                        "SkillCatalog: attached retriever for upgraded scoring (backend=%s)",
                        getattr(skill_retriever, "name", "?"),
                    )
            except Exception as _arx:
                logger.warning("SkillCatalog retriever attach failed: %s", _arx)

            # Meta-tool registry — register the built-ins enabled in config.
            mt_reg = get_meta_tool_registry()
            if cfg.meta_tools.builtin.list_tools:
                mt_reg.register(
                    make_list_tools_meta_tool(tool_retriever,
                                              default_top_k=cfg.retrieval.tool_top_k),
                    replace=True,
                )
            if cfg.meta_tools.builtin.list_skills:
                mt_reg.register(
                    make_list_skills_meta_tool(skill_retriever,
                                               default_top_k=cfg.retrieval.skill_top_k),
                    replace=True,
                )
            if cfg.meta_tools.builtin.tool_details:
                mt_reg.register(
                    make_tool_details_meta_tool(lambda: _tool_meta),
                    replace=True,
                )
            services["meta_tool_registry"] = mt_reg

            # ── HITL safety-net validation ─────────────────────────────
            # Warn if cfg.tools.hitl_tool_names lists tools that aren't
            # actually registered. The prompt builder uses those names as
            # the "always-inject" safety net, so a mismatch silently weakens
            # HITL coverage.
            try:
                _registered = set(_tool_meta.keys())
                _hitl_cfg   = set(getattr(cfg.tools, "hitl_tool_names", []) or [])
                _missing    = sorted(_hitl_cfg - _registered)
                _hits       = sorted(_hitl_cfg & _registered)
                if _missing:
                    logger.warning(
                        "HITL safety-net: %d/%d tool names from cfg.tools.hitl_tool_names "
                        "are NOT registered in mode=%s — they cannot fire HITL. "
                        "Missing: %s. Active: %s",
                        len(_missing), len(_hitl_cfg), cfg.mode,
                        _missing, _hits,
                    )
                else:
                    logger.info(
                        "HITL safety-net: all %d configured tool names are registered (%s)",
                        len(_hitl_cfg), _hits,
                    )
            except Exception as _hsv_exc:
                logger.warning("HITL safety-net validation failed: %s", _hsv_exc)

            # Wire the LLM engine to use the retrievers + registry at prompt time
            llm_engine.attach_retrieval(
                tool_retriever     = tool_retriever,
                skill_retriever    = skill_retriever,
                meta_tool_registry = mt_reg,
            )

            # Register meta-tools as ordinary local callables in the ToolRouter
            # so [TOOL:list_tools] dispatches to the meta-tool handler at execution.
            try:
                router = services.get("tool_router")
                if router is not None:
                    router.register_local(mt_reg.as_local_callables())
                    logger.info(
                        "Meta-tools wired into ToolRouter: %d callable(s)", len(mt_reg)
                    )
            except Exception as _mr_exc:
                logger.warning("ToolRouter meta-tool wiring failed: %s", _mr_exc)

            logger.info(
                "Retrieval framework: backend=%s tool_top_k=%d skill_top_k=%d "
                "meta_tools=%d",
                cfg.retrieval.backend, cfg.retrieval.tool_top_k,
                cfg.retrieval.skill_top_k, len(mt_reg),
            )

        except Exception as _ret_exc:
            logger.warning(
                "Retrieval framework wiring failed (%s) — falling back to "
                "full-catalog dumps in prompts", _ret_exc,
            )

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

    # Mount the legacy /hitl/* router only when running on the langgraph
    # backend. In core mode, /hitl/* endpoints are served by webui/backend
    # (which speaks to hitl_core.HitlRouter directly via services).
    if _services.get("hitl_router") is not None:
        from hitl.router import create_hitl_router
        from hitl.review import get_sse_channel
        hitl_api = create_hitl_router(
            decision_router = _services["hitl_router"],
            audit           = _services["hitl_audit"],
            sse_channel     = get_sse_channel(),
        )
        app.include_router(hitl_api, prefix="/hitl")
        logger.info("Legacy /hitl/* router mounted (langgraph backend)")
    else:
        logger.info("Legacy /hitl/* router skipped (core backend; webui/backend handles HITL endpoints)")

    from registry.router import create_registry_router
    reg_api = create_registry_router(_services["registry"])
    app.include_router(reg_api, prefix="/registry")

    from webui.backend import create_webui_app
    webui = create_webui_app(_services)
    app.mount("/webui", webui)

    # Attach the runtime loop to the HITL executor so post-HITL fallback
    # callbacks can run the full agent (with tool registry + memory recall +
    # skills) on approved queries — not just a single-shot LLM call.
    try:
        _executor   = _services.get("executor")
        _agent_loop = _services.get("runtime_loop")
        if _executor is not None and _agent_loop is not None:
            _executor._runtime_loop = _agent_loop
            logger.info("HITL executor: full-agent post-HITL fallback enabled")
    except Exception as _exc:
        logger.warning("HITL executor runtime-loop wiring failed: %s", _exc)

    logger.info("All modules mounted")
    watchdog_task = None
    if _services.get("hitl_watchdog") is not None:
        watchdog_task = asyncio.create_task(_services["hitl_watchdog"].run())

    # Start SkillJournalConsumer background task
    try:
        _start = _services.get("_start_consumer")
        if _start:
            await _start()
    except Exception as _jc_exc:
        logger.warning("SkillJournalConsumer start failed: %s", _jc_exc)

    yield

    # Stop SkillJournalConsumer before other teardown
    try:
        _stop = _services.get("_stop_consumer")
        if _stop:
            await _stop()
    except Exception as _jc_exc:
        logger.warning("SkillJournalConsumer stop failed: %s", _jc_exc)

    if _services.get("hitl_watchdog") is not None:
        _services["hitl_watchdog"].stop()
    await _services["registry"].stop()
    if watchdog_task is not None:
        watchdog_task.cancel()
        try:
            await watchdog_task
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