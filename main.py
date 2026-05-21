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

    # ── Per-agent data isolation (2026-05) ──────────────────────────────
    # Resolve this agent's private data root ONCE, up front, so every
    # component below (memory, HITL checkpoints, tool-result cache, evolved
    # skills, journal) writes under data/agents/<agent_id>/ instead of the
    # shared data/. Two agents (lan / dc) running from the same image then
    # never read each other's facts or overwrite each other's databases.
    _agent_data_dir = cfg.agent_data_dir()
    logger.info("Agent data dir: %s (agent_id=%s)", _agent_data_dir, cfg.agent.agent_id)

    # ── 0. Observability / tracing (Sprint-3-pre, 2026-05) ──────────────
    # Initialize OpenTelemetry BEFORE building anything else so any
    # spans created during component init are captured. Defaults OFF
    # — see runtime/tracing.py for degradation contract.
    try:
        from runtime.tracing import configure as _configure_tracing
        _obs = getattr(cfg, "observability", None)
        _configure_tracing(
            enabled         = bool(getattr(_obs, "tracing_enabled", False)),
            service_name    = str (getattr(_obs, "service_name",    "netopyu-agent")),
            service_version = str (getattr(_obs, "service_version", "6.0.0")),
            otlp_endpoint   =      getattr(_obs, "otlp_endpoint",   None),
            sample_ratio    = float(getattr(_obs, "sample_ratio",   1.0)),
        )
    except Exception as _tr_exc:
        logger.warning("Tracing configure failed: %s — proceeding without", _tr_exc)

    # ── 1. Memory ────────────────────────────────────────────────────────────
    # Production memory backend: agent_memory.MemoryManager wrapped by
    # MemoryAdapter for async + per-operator scoping.
    # Multi-user isolated, SQLite WAL persistent, 311 unit tests.
    from memory import MemoryAdapter
    # Per-agent data isolation: each agent_id gets its own data subtree
    # (resolved at the top of build_services). Shared read-only fixtures
    # stay at cfg.memory.data_dir.
    memory_router = MemoryAdapter(
        data_dir          = _agent_data_dir + "/memory",
        inline_threshold  = 4_000,
        session_ttl       = 86_400,
        enable_user_model = True,
        # Sprint 2 (2026-05): Hermes-style structured rollup vs legacy
        # free-form. Validated downstream in MemoryConsolidator;
        # invalid → falls back to 'structured' with a warning.
        consolidation_template = getattr(cfg.memory, "consolidation_template", "structured"),
    )
    # Auto-consolidate gate — see MemoryAdapter.set_consolidator docstring.
    # Cfg-driven; 0 disables. Background task style, hot path unblocked.
    try:
        _auto_n = int(getattr(cfg.memory, "auto_consolidate_turns", 30))
        memory_router.set_consolidator(threshold_turns=_auto_n)
    except Exception as _cons_exc:
        logger.warning("Memory auto-consolidate setup failed (%s) — disabled", _cons_exc)
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

        # Pick checkpoint backend from cfg.hitl.checkpoint. Env overrides
        # (HITL_CHECKPOINT_BACKEND etc.) already applied in config.py.
        # Default backend is "sqlite" (changed from "memory" in 2026-05)
        # so pending approvals survive agent restart — operators discussing
        # a destructive-action card while the process crashes would
        # otherwise see their producing query hang on a dead asyncio.Future.
        _cp_cfg = getattr(cfg.hitl, "checkpoint", None)
        _cp_backend = (_cp_cfg.backend if _cp_cfg else "sqlite").lower()
        if _cp_backend == "redis":
            from hitl_core import RedisCheckpointStore
            _redis_url = (_cp_cfg.redis_url if _cp_cfg else None) or cfg.memory.redis_url \
                         or "redis://localhost:6379/0"
            hitl_store = RedisCheckpointStore(redis_url=_redis_url)
        elif _cp_backend == "sqlite":
            from hitl_core import SqliteCheckpointStore
            # Default lives under the per-agent data dir so each agent's
            # pending approvals are isolated. An explicit hitl.checkpoint.
            # sqlite_path in config still wins (operator override).
            _default_hitl_db = _agent_data_dir + "/hitl_checkpoints.db"
            _sq_path = (_cp_cfg.sqlite_path if _cp_cfg and _cp_cfg.sqlite_path
                        and _cp_cfg.sqlite_path != "data/hitl_checkpoints.db"
                        else None) or _default_hitl_db
            hitl_store = SqliteCheckpointStore(db_path=_sq_path)
        else:
            hitl_store = InMemoryCheckpointStore()
            # Pragmatic-mode + in-memory checkpoint = data-loss risk.
            # Pragmatic mode runs real device operations, so a pending
            # destructive approval lost on restart means an operator's
            # in-progress decision evaporates with no audit trail of
            # what was being asked. Warn loudly.
            _mode = getattr(cfg, "mode", None) or _os.getenv("MODE", "mock")
            if str(_mode).lower() == "pragmatic":
                logger.warning(
                    "HITL using in-memory checkpoint store in pragmatic mode. "
                    "Pending approvals will be LOST on restart. Set "
                    "HITL_CHECKPOINT_BACKEND=sqlite or redis (or "
                    "hitl.checkpoint.backend in config.yaml) for production."
                )
        logger.info("HITL checkpoint backend: %s", _cp_backend)
        # Stash for graceful shutdown (Sprint-3-pre, 2026-05):
        # lifespan exit will call hitl_store.flush()/close() if available
        # so any pending state is persisted before we kill in-flight tasks.
        services["hitl_store"] = hitl_store

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
        # Legacy LangGraph backend (HITL_BACKEND != "core") has been retired
        # in this build. The `hitl/` package was a thin schema stub and the
        # implementation modules (hitl/graph.py, hitl/router.py, hitl/review.py,
        # hitl/triggers.py, etc.) were never packaged here. Attempting to
        # construct it would explode with an opaque ImportError on
        # `from hitl import HitlAuditService`. Fail explicitly instead so
        # operators understand the choice.
        raise NotImplementedError(
            "HITL_BACKEND=%r requires the legacy LangGraph hitl/* package, "
            "which is not part of this build. Set HITL_BACKEND=core (the "
            "default) to use the in-house HITL pipeline at hitl_core/."
            % _hitl_backend
        )

    # ── 3. Registry ──────────────────────────────────────────────────────────
    from registry import create_registry, RegistryConfig as RegCfg
    from a2a.agent_card import get_agent_card
    registry_config = RegCfg(
        lb_strategy                   = cfg.registry.lb_strategy,
        health_check_interval_seconds = cfg.registry.health_check_interval,
    )
    # AgentCard now carries this agent's identity (agent_id, capabilities)
    # so peers see who we are. Identity defaults reproduce the legacy
    # single-agent behaviour when cfg.agent is unset in yaml.
    #
    # 2A profile enrichment: if the operator didn't hand-write capabilities
    # / display_name / description in config, fill them from the active
    # business profile so each profile advertises a sensible identity to
    # peers without yaml boilerplate. Operator-set values always win.
    try:
        from profiles import load_profile
        from config import AgentSkillSpec as _AgentSkillSpec
        _profile = load_profile(cfg.agent.profile)
        if not cfg.agent.capabilities and _profile.capabilities:
            cfg.agent.capabilities = [
                _AgentSkillSpec(
                    skill_id    = c.get("skill_id", ""),
                    name        = c.get("name", ""),
                    description = c.get("description", ""),
                    tags        = list(c.get("tags", [])),
                )
                for c in _profile.capabilities if c.get("skill_id")
            ]
        # Only override display_name/description if still at the dataclass
        # defaults (operator didn't customise them).
        if cfg.agent.display_name in ("", "IT Ops Agent") and _profile.display_name:
            cfg.agent.display_name = _profile.display_name
        logger.info(
            "Profile %r active: %d business tool(s), %d capability(ies)",
            cfg.agent.profile, len(_profile.tool_callables),
            len(cfg.agent.capabilities),
        )
    except Exception as _prof_exc:
        logger.warning("Profile enrichment failed: %s", _prof_exc)

    own_card = get_agent_card(cfg.server.a2a_base_url, identity=cfg.agent)
    # Peer URLs come from BOTH places for backwards compat:
    #   1. cfg.registry.agent_urls (legacy single-list)
    #   2. cfg.agent.peer_urls (Phase 1, per-agent identity-aware)
    # Deduped, order-preserving union — registry sees one list either way.
    _all_peer_urls: list[str] = []
    _seen: set[str] = set()
    for u in (cfg.registry.agent_urls or []) + (cfg.agent.peer_urls or []):
        u = u.strip()
        if u and u not in _seen:
            _seen.add(u)
            _all_peer_urls.append(u)
    registry = await create_registry(
        static_urls = _all_peer_urls,
        redis_url   = cfg.memory.redis_url,
        config      = registry_config,
        own_card    = own_card,
    )
    await registry.start()
    services["registry"] = registry
    # Stash the merged peer-url list so the lifespan peer-refresh loop
    # picks up the same set without re-merging. Underscore prefix marks
    # it as internal (audit_wiring whitelist already handles this style).
    services["_peer_urls"] = list(_all_peer_urls)
    logger.info(
        "Agent Registry ready — agent_id=%s peers=%d",
        cfg.agent.agent_id, len(_all_peer_urls),
    )

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

    # ── 4b. Delegation hook (Phase 2B) ───────────────────────────────────────
    # Build the delegate_fn the runtime loop calls on [DELEGATE:...] directives.
    # Injected (not imported) into AgentRuntimeLoop so the loop stays
    # registry/task-agnostic. Resolves agent_id / *capability → peer via the
    # registry, streams the subtask through A2ATaskDispatcher.
    try:
        from task.delegation import build_delegate_fn
        services["delegate_fn"] = build_delegate_fn(
            registry   = registry,
            dispatcher = task_system.dispatcher,
            task_store = task_system.store,
            own_agent_id = cfg.agent.agent_id,
        )
        logger.info("Delegation hook ready (agent_id=%s)", cfg.agent.agent_id)
    except Exception as _dlg_exc:
        services["delegate_fn"] = None
        logger.warning("Delegation hook setup failed: %s", _dlg_exc)

    # ── 5. A2A executor ──────────────────────────────────────────────────────
    if _hitl_backend == "core":
        # New executor — built later, after llm_engine + tool_registry are
        # available (section 6). Stub the slot for now so any reference
        # to services["executor"] before then errors clearly.
        services["executor"] = None
        logger.info("A2A executor (core) — deferred to after LLM + tool wiring")
    else:
        # Unreachable: the earlier section-2 else-branch already raises
        # NotImplementedError for non-core backends, so we never get here.
        # Kept as a defense-in-depth guard in case someone refactors the
        # earlier branch without realising this depends on it.
        raise NotImplementedError(
            "Legacy `hitl.ITOpsHitlAgentExecutor` not packaged in this build; "
            "see the section-2 HITL_BACKEND guard above."
        )

    # ── 6. Integrations ──────────────────────────────────────────────────────
    try:
        from integrations import (
            MCPClient, OpenAPIClient, LLMEngine, ToolRouter,
            patch_runtime_loop, patch_hitl_graph,
        )
        from tools import make_read_stored_result_tool
        from runtime import ToolResultStore

        tool_store = ToolResultStore(db_path=_agent_data_dir + "/tool_results.db")
        services["tool_store"] = tool_store

        # 6a. LLM engine — always real (both modes).
        # `capabilities` plumbs through per-model behaviour (thinking_tag,
        # format_compliance, etc.) so the engine doesn't have to guess from
        # the model NAME via string matching.
        llm_engine = LLMEngine.from_config({
            "backend":      cfg.llm.backend,
            "model":        cfg.llm.model,
            "base_url":     cfg.llm.base_url,
            "temperature":  cfg.llm.temperature,
            "max_tokens":   cfg.llm.max_tokens,
            "capabilities": cfg.llm.capabilities,
        })
        services["llm_engine"] = llm_engine  # patch_hitl_graph called after tool registry is built
        # D1 (Sprint 3): apply the concurrency cap. Default 4; 0 = unlimited.
        try:
            if hasattr(llm_engine, "set_max_concurrent_calls"):
                llm_engine.set_max_concurrent_calls(
                    int(getattr(cfg.llm, "max_concurrent_calls", 4))
                )
        except Exception as _sem_exc:
            logger.warning("LLM concurrency cap wiring failed: %s", _sem_exc)
        logger.info("LLM engine: %s/%s", cfg.llm.backend, cfg.llm.model)
        logger.info("LLM capabilities: thinking_tag=%r format_compliance=%s "
                    "max_context=%d native_tools=%s",
                    cfg.llm.capabilities.thinking_tag,
                    cfg.llm.capabilities.format_compliance,
                    cfg.llm.capabilities.max_context_chars,
                    cfg.llm.capabilities.supports_native_tools)

        # Export capabilities as env vars so module-independent helpers
        # (agent_memory.fact_extractor, agent_memory.user_model, etc.) can
        # consume them without importing config — preserves the module
        # independence contract while still giving them the right values.
        # Only sets if the env var isn't already set (operator's manual
        # override on the command line still wins).
        import os as _cap_os
        _cap_os.environ.setdefault("LLM_THINKING_TAG", cfg.llm.capabilities.thinking_tag or "")
        _cap_os.environ.setdefault("LLM_FORMAT_COMPLIANCE", cfg.llm.capabilities.format_compliance)

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
            from integrations.clients.embedder import build_embedder
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
        _loader = _ToolLoader(mode=cfg.mode, profile=cfg.agent.profile)
        read_stored_fn, process_chunks_fn = make_read_stored_result_tool(tool_store)
        tool_registry_local = _loader.build_callables()
        tool_registry_local["read_stored_result"]    = read_stored_fn
        tool_registry_local["process_stored_chunks"] = process_chunks_fn
        # Store loader on services so llm_engine can build the dynamic tool section
        services["tool_loader"] = _loader
        logger.info("ToolLoader[%s]: %d tools assembled", cfg.mode, len(tool_registry_local))

        # 6d. MCP client
        # The built-in "netops" mock MCP server + "netops_api" OpenAPI mock are
        # LAN-business integrations (get_device_status, get_devices, …). They
        # must NOT load for non-LAN profiles, or a dc/default agent would gain
        # LAN device tools and break role isolation. Only the lan profile (or
        # pragmatic mode with explicitly-configured real servers) wires them.
        # Tracked: when dc needs its own MCP/OpenAPI, declare per-profile
        # integration config (TODO.md "profile integrations").
        _profile_id = (cfg.agent.profile or "default").strip().lower()
        _load_builtin_netops = (_profile_id == "lan") or cfg.is_pragmatic
        mcp_client = None
        api_client = None
        if _load_builtin_netops:
            mcp_client = await _build_mcp_client(MCPClient)
            await mcp_client.connect_all()
            services["mcp_client"] = mcp_client

            # 6e. OpenAPI client (mock in both modes unless explicitly configured)
            api_client = await _build_openapi_client(OpenAPIClient)
            services["api_client"] = api_client
        else:
            logger.info(
                "Profile %r: skipping built-in netops MCP+OpenAPI mock "
                "(LAN-business integrations; not loaded for this profile)",
                _profile_id,
            )

        # 6f. Pragmatic extra MCP servers
        extra_mcp_clients = []
        if cfg.is_pragmatic and cfg.pragmatic.mcp_servers:
            extra_mcp_clients = await _load_pragmatic_mcp_servers(MCPClient)

        # 6g. ToolRouter
        router = ToolRouter(tool_store=tool_store)
        if mcp_client is not None:
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
            from integrations.adapters.hitl_executor import HitlExecutor
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

            # Build tool_metadata for action_type fast-path.
            # services["tool_loader"] is set in step 6c (line ~368) — already
            # available here. Fail-soft: if loader is missing, just don't
            # populate the fast-path (PolicyEngine still works without it).
            try:
                _tool_md = services["tool_loader"].build_metadata()
            except (KeyError, AttributeError):
                _tool_md = None

            _policy_engine = PolicyEngine(
                policies    = _policy_defs,
                llm_call    = _policy_llm_call,
                cache_ttl_s = 120,
                # Wire tool metadata so classify_action_type() fast-path works.
                # Tools that declare action_type (read_only/reversible/
                # destructive) will bypass LLM evaluation entirely. See
                # runtime/policy_engine.py classify_action_type() for rationale.
                tool_metadata = _tool_md,
            )
            set_policy_engine(_policy_engine)
            # Wire trust_mode from cfg.hitl (graduated-trust spectrum).
            # set_trust_mode validates input and falls back to 'cautious'
            # for unknown values, so this is fail-soft even with bad config.
            try:
                _tm = getattr(cfg.hitl, "trust_mode", "cautious") or "cautious"
                _policy_engine.set_trust_mode(_tm)
            except Exception as _tm_exc:
                logger.warning(
                    "PolicyEngine: trust_mode wire failed (%s) — using 'cautious'",
                    _tm_exc,
                )
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
            from skills import SkillCatalogService, SkillLoader
            _skill_loader = SkillLoader(mode=cfg.mode, profile=cfg.agent.profile)
            _skill_defs   = _skill_loader.skill_definitions()
            services["skill_loader"] = _skill_loader
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

            # Evolved skills are per-agent state (an LAN agent shouldn't
            # inherit a DC agent's auto-evolved skills), so they live under
            # the agent's private data dir. The golden set used by the A/B
            # bench below is a SHARED fixture and stays at data/.
            _skills_dir = _agent_data_dir
            _skill_evolver = SkillEvolver(
                catalog    = services["skill_catalog"],
                llm_fn     = _async_llm_for_skills,
                skills_dir = str(_pl.Path(_skills_dir) / "skills"),
            )
            services["skill_evolver"] = _skill_evolver

            # Wire the evolver into the HITL executor so the batch
            # finalizer can mint skills from successful multi-target
            # batches. The executor was constructed earlier (before
            # the evolver existed) — use the deferred setter.
            try:
                _exec = services.get("executor")
                if _exec is not None and hasattr(_exec, "set_skill_evolver"):
                    _exec.set_skill_evolver(_skill_evolver)
                    logger.info(
                        "SkillEvolver: wired into HITL executor (batch finalizer hook)"
                    )
            except Exception as _wire_exc:
                logger.warning(
                    "SkillEvolver: failed to wire into executor (%s) — "
                    "batch resolution will skip skill evolution",
                    _wire_exc,
                )

            # ── A/B safety net for SkillEvolver.apply_feedback ──────────────
            # Wire a bench runner that runs a per-skill subset of the
            # tool-compliance golden set on both old + new content. If
            # args_ok would drop, apply_feedback rolls back the patch.
            # Wire is best-effort: missing golden set or engine just leaves
            # the safety net disabled (legacy unchecked path remains).
            #
            # See skills/evolver.py:set_bench_runner docstring for contract.
            try:
                from evaluation.tool_compliance_bench import ToolComplianceBench
                from evaluation.tool_compliance_types import ToolCallCase
                import json as _json_skbench

                # Compliance golden set is a SHARED fixture (not per-agent).
                # Look in the base data dir, not the agent's private subtree.
                _golden_path = _pl.Path(cfg.memory.data_dir or "./data") / "tool_compliance_set.jsonl"
                if not _golden_path.exists():
                    _golden_path = _pl.Path("data/tool_compliance_set.jsonl")

                if _golden_path.exists() and llm_engine is not None:
                    # Load cases once at startup
                    _cases_all: list[ToolCallCase] = []
                    for _line in _golden_path.read_text().splitlines():
                        _line = _line.strip()
                        if not _line or _line.startswith("#"):
                            continue
                        try:
                            _d = _json_skbench.loads(_line)
                            _cases_all.append(ToolCallCase(**_d))
                        except Exception:
                            continue

                    # Construct a per-skill bench-runner closure. Subsetting
                    # by tool name keeps the run cheap (3-5 cases per skill);
                    # see skills/evolver.py:set_bench_runner contract.
                    async def _ab_bench_runner(skill_id: str, candidate_content: str):
                        # Filter cases: match any tool the skill probably
                        # touches. The simplest heuristic — case is included
                        # if its expected_tool name appears in the candidate
                        # content. Skills that reference no recognised tools
                        # bail out to None (skip gate, don't trap improvement).
                        _matched = [
                            c for c in _cases_all
                            if c.expected_tool in candidate_content
                        ][:5]
                        if not _matched:
                            return None
                        try:
                            _bench = ToolComplianceBench(
                                engine = llm_engine,
                                name   = f"ab-safety-net/{skill_id}",
                            )
                            return await _bench.run(_matched)
                        except Exception as _bench_inner:
                            logger.debug(
                                "A/B bench runner exception for %s: %s",
                                skill_id, _bench_inner,
                            )
                            return None

                    _skill_evolver.set_bench_runner(_ab_bench_runner)
                    logger.info(
                        "SkillEvolver: A/B safety-net wired (%d compliance cases loaded)",
                        len(_cases_all),
                    )
                else:
                    logger.info(
                        "SkillEvolver: A/B safety-net DISABLED — "
                        "golden set %s exists=%s, engine=%s",
                        _golden_path, _golden_path.exists(),
                        "yes" if llm_engine else "no",
                    )
            except Exception as _ab_exc:
                logger.warning(
                    "SkillEvolver: A/B safety-net wire failed (%s) — "
                    "feedback patches will skip the regression gate",
                    _ab_exc,
                )

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
                _meta = _TL(mode=cfg.mode, profile=cfg.agent.profile).build_metadata()
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

        # ── Cross-module: Journal → MemoryFacts (Tier 1 #1) ───────────
        # Independent of runtime. Disabled by default; enable via config.
        try:
            _xm = getattr(cfg, "cross_module", None)
            _jtf_cfg = getattr(_xm, "journal_to_facts", None) if _xm else None
            if _jtf_cfg and _jtf_cfg.enabled:
                from integrations.adapters.memory_facts_adapter import JournalToFactsAdapter
                from runtime.skill_journal import get_journal_store
                _mem_adapter = services.get("memory")
                if _mem_adapter is not None:
                    jtf = JournalToFactsAdapter(
                        journal_store=get_journal_store(),
                        fact_writer=_mem_adapter,
                        interval_s=_jtf_cfg.interval_s,
                        min_observations=_jtf_cfg.min_observations,
                        dormant_threshold=_jtf_cfg.dormant_threshold,
                        success_threshold=_jtf_cfg.success_threshold,
                        fact_ttl_days=_jtf_cfg.fact_ttl_days,
                        max_facts_per_scan=_jtf_cfg.max_facts_per_scan,
                        target_user_id=_jtf_cfg.target_user_id,
                        target_session_id=_jtf_cfg.target_session_id,
                    )
                    services["journal_to_facts_adapter"] = jtf
                    _prev_start = services.get("_start_consumer")
                    _prev_stop  = services.get("_stop_consumer")
                    async def _start_adapters():
                        if _prev_start: await _prev_start()
                        await jtf.start()
                    async def _stop_adapters():
                        await jtf.stop()
                        if _prev_stop: await _prev_stop()
                    services["_start_consumer"] = _start_adapters
                    services["_stop_consumer"]  = _stop_adapters
                    logger.info(
                        "JournalToFactsAdapter wired (interval=%ds) — bridges skill journal → memory",
                        _jtf_cfg.interval_s,
                    )
                else:
                    logger.info("JournalToFactsAdapter skipped — no memory adapter in services")
            else:
                logger.debug("JournalToFactsAdapter disabled (cfg.cross_module.journal_to_facts.enabled=false)")
        except Exception as _jtf_exc:
            logger.warning("JournalToFactsAdapter setup failed: %s", _jtf_exc)

        # ── Cross-module: Fact conflict detection (Tier 1 #2) ─────────
        # Wraps memory.add_fact via a detector instance attached to services.
        # The detector is exposed as services["fact_conflict_detector"];
        # call sites that want conflict-aware insertion use it explicitly.
        try:
            _fcd_cfg = getattr(_xm, "fact_conflict_detection", None) if _xm else None
            if _fcd_cfg and _fcd_cfg.enabled:
                from integrations.adapters.fact_conflict_detector import FactConflictDetector
                _mem_adapter = services.get("memory")
                if _mem_adapter is not None:
                    _llm_for_reconcile = None
                    if _fcd_cfg.llm_reconcile_enabled:
                        _engine = services.get("llm_engine") or services.get("engine")
                        if _engine and hasattr(_engine, "complete"):
                            async def _llm_reconcile(system: str, user: str) -> str:
                                return await _engine.complete(system=system, user=user, max_tokens=256)
                            _llm_for_reconcile = _llm_reconcile
                    fcd = FactConflictDetector(
                        memory=_mem_adapter,
                        llm_fn=_llm_for_reconcile,
                        similarity_threshold=_fcd_cfg.similarity_threshold,
                        equivalence_threshold=_fcd_cfg.equivalence_threshold,
                        llm_reconcile_enabled=_fcd_cfg.llm_reconcile_enabled,
                        llm_timeout_s=_fcd_cfg.llm_timeout_s,
                        top_k_candidates=_fcd_cfg.top_k_candidates,
                        confidence_boost=_fcd_cfg.confidence_boost,
                        contradiction_demote=_fcd_cfg.contradiction_demote,
                    )
                    services["fact_conflict_detector"] = fcd
                    # Wire into MemoryAdapter so add_fact() routes through
                    # the detector instead of doing a direct mid_term insert.
                    # This is the LAST mile of the FactConflictDetector
                    # feature — without it the detector is constructed,
                    # registered in services, and never called by anyone
                    # (the "ghost service" anti-pattern caught by
                    # audit_wiring.py).
                    if hasattr(_mem_adapter, "set_conflict_detector"):
                        _mem_adapter.set_conflict_detector(fcd)
                    else:
                        logger.warning(
                            "FactConflictDetector wired in services but "
                            "MemoryAdapter lacks set_conflict_detector — "
                            "facts will still take the direct path. "
                            "Upgrade MemoryAdapter or wire callers manually."
                        )
                    logger.info(
                        "FactConflictDetector wired (sim_thr=%.2f, llm_reconcile=%s)",
                        _fcd_cfg.similarity_threshold, _fcd_cfg.llm_reconcile_enabled,
                    )
                else:
                    logger.info("FactConflictDetector skipped — no memory adapter in services")
            else:
                logger.debug("FactConflictDetector disabled (cfg.cross_module.fact_conflict_detection.enabled=false)")
        except Exception as _fcd_exc:
            logger.warning("FactConflictDetector setup failed: %s", _fcd_exc)

        # ── Evaluation bench on startup (Tier 2 #4) ──────────────────
        try:
            _eval_cfg = getattr(cfg, "evaluation", None)
            if _eval_cfg and _eval_cfg.bench_on_startup and _eval_cfg.golden_set_path:
                from evaluation import (
                    load_golden_set, RetrievalBench, format_text_report,
                )
                _golden = load_golden_set(_eval_cfg.golden_set_path)
                if _golden:
                    _sr = services.get("skill_retriever")
                    if _sr:
                        rep = RetrievalBench(
                            retriever=_sr, golden_set=_golden,
                            top_k=_eval_cfg.bench_top_k,
                        ).run()
                        logger.info("Startup bench:\n%s", format_text_report(rep))
                        if _eval_cfg.fail_below_mrr > 0 and rep.mrr < _eval_cfg.fail_below_mrr:
                            raise RuntimeError(
                                f"Startup bench MRR {rep.mrr:.3f} below threshold "
                                f"{_eval_cfg.fail_below_mrr:.3f} — refusing to start"
                            )
                else:
                    logger.warning(
                        "evaluation.golden_set_path=%r produced 0 cases — bench skipped",
                        _eval_cfg.golden_set_path,
                    )
        except RuntimeError:
            raise   # honour fail_below_mrr gate
        except Exception as _eval_exc:
            logger.warning("Startup eval bench failed: %s", _eval_exc)

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
            _tool_meta = _TL(mode=cfg.mode, profile=cfg.agent.profile).build_metadata()
            from skills import SkillLoader as _SL
            _skill_defs = _SL(mode=cfg.mode, profile=cfg.agent.profile).skill_definitions()

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

                    # IMPORTANT: real_registry was snapshotted earlier from
                    # router.registry. The snapshot is a fresh dict — additions
                    # to the router AFTER that point don't propagate. Now that
                    # meta-tools (list_tools, list_skills, tool_details) are
                    # registered, re-snap and update every consumer that holds
                    # the old reference.
                    try:
                        _refreshed_registry = router.registry
                        # Update runtime loop
                        _rt_loop = services.get("runtime_loop")
                        if _rt_loop is not None and hasattr(_rt_loop, "_tool_registry"):
                            _rt_loop._tool_registry = _refreshed_registry
                        # Update HITL executor
                        _exec = services.get("executor")
                        if _exec is not None and hasattr(_exec, "_tool_registry"):
                            _exec._tool_registry = _refreshed_registry
                        # Re-patch HITL graph if used
                        try:
                            from integrations.clients.llm_engine import patch_hitl_graph
                            patch_hitl_graph(llm_engine, tool_registry=_refreshed_registry)
                        except Exception:
                            pass
                        logger.info(
                            "Tool registry refreshed across consumers — "
                            "now %d callable(s) including meta-tools",
                            len(_refreshed_registry),
                        )
                    except Exception as _refresh_exc:
                        logger.warning(
                            "Tool registry refresh failed (meta-tools may not "
                            "be dispatchable): %s", _refresh_exc,
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

    # C2 (Sprint 3): FastAPI auto-instrumentation. Must run AFTER
    # build_services() (which calls tracing.configure()), and is a no-op
    # when tracing is disabled / the instrumentation package is absent.
    try:
        from runtime.tracing import instrument_fastapi
        instrument_fastapi(app)
    except Exception as _fi_exc:
        logger.warning("FastAPI instrumentation wiring failed: %s", _fi_exc)

    from a2a import create_a2a_app, InMemoryTaskStore
    a2a_app = create_a2a_app(
        base_url   = cfg.server.a2a_base_url,
        executor   = _services["executor"],
        task_store = InMemoryTaskStore(),
        identity   = cfg.agent,        # Phase-1: publish real agent_id + caps
    )
    app.mount("/api/v1/a2a", a2a_app)

    # Mount the legacy /hitl/* router only when running on the langgraph
    # backend. In core mode, /hitl/* endpoints are served by webui/backend
    # (which speaks to hitl_core.HitlRouter directly via services).
    if _services.get("hitl_router") is not None:
        try:
            from hitl.router import create_hitl_router
            from hitl.review import get_sse_channel
        except ImportError as e:
            logger.warning(
                "Legacy /hitl/* router not mountable: hitl.router/review "
                "modules not packaged in this build (%s). Falling back to "
                "core backend endpoints in webui/backend.", e,
            )
        else:
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

    # Start HITL chunk-queue idle watchdog — auto-completes streams that
    # have gone silent for > idle_timeout seconds. Prevents a hung HITL
    # resumer from leaking stale chunks into a subsequent chat_stream
    # turn on the same session.  See AUDIT_REPORT issue D.
    try:
        from hitl_core.chunk_queue import get_chunk_queue_registry
        get_chunk_queue_registry().start_idle_watchdog()
    except Exception as _cq_exc:
        logger.warning("ChunkQueue idle watchdog start failed: %s", _cq_exc)

    # ── Phase 1 multi-agent: periodic peer-card refresh ────────────────
    # We've already fetched peer AgentCards at registry construction time.
    # This loop re-fetches them on a schedule so capability changes /
    # restarts / health flips become visible without operator action.
    # Skipped when no peers are configured or refresh is disabled.
    peer_refresh_task = None
    _peers_to_refresh = list(_services.get("_peer_urls") or [])
    _refresh_interval = int(cfg.agent.peer_refresh_interval_s)
    if _peers_to_refresh and _refresh_interval > 0:
        async def _peer_refresh_loop():
            """Periodically re-fetch peer AgentCards and update registry.

            Two phases:
              1. Fast bootstrap retries — peers may not be up yet when THIS
                 agent starts (common when launching both processes at once).
                 Retry every 5s for the first ~30s so they find each other
                 quickly instead of waiting a full refresh interval.
              2. Steady-state refresh every peer_refresh_interval_s.
            """
            registry = _services["registry"]
            # Re-fetch with the same source label the initial registration
            # used (STATIC — these peers come from config/env). Passing
            # source=None breaks AgentEntry validation (the enum rejects
            # None); register() upserts by agent_id so re-registering with
            # STATIC is idempotent and refreshes the card. (Bug fixed
            # 2026-05: the loop previously passed source=None.)
            from registry.schemas import RegistrationSource as _RS

            async def _do_refresh() -> int:
                if not hasattr(registry, "register_from_urls"):
                    return 0
                refreshed = await registry.register_from_urls(
                    _peers_to_refresh, source=_RS.STATIC,
                )
                return len(refreshed or [])

            # Phase 1: fast bootstrap. Stop early once all peers are found.
            _bootstrap_attempts = 6      # 6 × 5s = 30s
            for _i in range(_bootstrap_attempts):
                try:
                    await asyncio.sleep(5)
                    _n = await _do_refresh()
                    if _n >= len(_peers_to_refresh):
                        logger.info(
                            "Peer bootstrap: all %d peer(s) discovered", _n,
                        )
                        break
                    logger.debug(
                        "Peer bootstrap attempt %d: %d/%d peer(s) found",
                        _i + 1, _n, len(_peers_to_refresh),
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as _pr_exc:
                    logger.debug("Peer bootstrap attempt %d failed: %s", _i + 1, _pr_exc)

            # Phase 2: steady-state refresh
            while True:
                try:
                    await asyncio.sleep(_refresh_interval)
                    await _do_refresh()
                except asyncio.CancelledError:
                    raise
                except Exception as _pr_exc:
                    # Never tank the agent because a peer was momentarily
                    # unreachable. Log + try again next interval.
                    logger.debug("Peer refresh failed (will retry): %s", _pr_exc)
        peer_refresh_task = asyncio.create_task(_peer_refresh_loop())
        logger.info(
            "Peer refresh loop started (every %ds, %d peer(s))",
            _refresh_interval, len(_peers_to_refresh),
        )

    # ── Graceful shutdown plumbing (Sprint-3-pre, 2026-05) ──────────────
    # Track in-flight LLM/tool work so the shutdown drain (after `yield`)
    # can wait for it before the process exits. Without this, a tool that
    # already executed against a real device but whose result hadn't yet
    # returned to the LLM would leave device state inconsistent with our
    # memory record.
    #
    # IMPORTANT: we deliberately do NOT install our own SIGINT/SIGTERM
    # handler. uvicorn already installs handlers that (a) stop accepting
    # connections and (b) run the ASGI lifespan shutdown — i.e. the code
    # after `yield` below. If we call loop.add_signal_handler() here we
    # OVERRIDE uvicorn's handler; unless we then re-trigger uvicorn's
    # shutdown ourselves (we can't cleanly), the server never stops and
    # Ctrl+C hangs. The correct integration point is the post-`yield`
    # drain block, which uvicorn invokes for us on signal. (Bug fixed
    # 2026-05: the earlier add_signal_handler approach broke Ctrl+C.)
    _services["in_flight_tasks"] = set()

    yield

    # ─── Graceful shutdown sequence ───────────────────────────────────
    # Reached when uvicorn receives SIGINT/SIGTERM and runs lifespan
    # shutdown. We drain in-flight work, then flush HITL state.
    # 1. Drain in-flight LLM/tool work (best effort, 30s cap)
    _in_flight = _services.get("in_flight_tasks") or set()
    if _in_flight:
        # Drain timeout is configurable; default 10s keeps interactive
        # Ctrl+C snappy while still giving real device operations a
        # chance to finish. Production can raise it (e.g. 30-60s) where
        # long-running tool calls matter more than fast restart.
        import os as _os_drain
        _drain_timeout = float(_os_drain.getenv("SHUTDOWN_DRAIN_TIMEOUT_S", "10"))
        logger.info(
            "Waiting for %d in-flight request(s) to complete (max %.0fs)…",
            len(_in_flight), _drain_timeout,
        )
        try:
            await asyncio.wait_for(
                asyncio.gather(*list(_in_flight), return_exceptions=True),
                timeout=_drain_timeout,
            )
            logger.info("All in-flight requests drained")
        except asyncio.TimeoutError:
            logger.warning(
                "Forced shutdown after %.0fs drain timeout — %d task(s) "
                "still pending; their state may not be persisted",
                _drain_timeout, len(_in_flight),
            )
        except Exception as _drain_exc:
            logger.warning("Drain hit unexpected error: %s", _drain_exc)

    # 2. Flush HITL checkpoint store so pending interrupts survive restart
    try:
        _hs = _services.get("hitl_store")
        if _hs is not None:
            # Both sqlite & redis stores expose flush()/close(); memory
            # store has neither and is irrelevant (data already lost).
            if hasattr(_hs, "flush"):
                _res = _hs.flush()
                if asyncio.iscoroutine(_res):
                    await _res
                logger.info("HITL checkpoint store flushed")
            elif hasattr(_hs, "close"):
                _res = _hs.close()
                if asyncio.iscoroutine(_res):
                    await _res
                logger.info("HITL checkpoint store closed")
    except Exception as _flush_exc:
        logger.warning("HITL checkpoint flush failed: %s", _flush_exc)

    # Stop SkillJournalConsumer before other teardown
    try:
        _stop = _services.get("_stop_consumer")
        if _stop:
            await _stop()
    except Exception as _jc_exc:
        logger.warning("SkillJournalConsumer stop failed: %s", _jc_exc)

    if peer_refresh_task is not None:
        peer_refresh_task.cancel()
        try:
            await peer_refresh_task
        except asyncio.CancelledError:
            pass

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


@app.get("/metrics", include_in_schema=False)
async def metrics():
    """Prometheus / OpenMetrics text exposition (C1, Sprint 3, 2026-05).

    Returns live counters/gauges/histograms in OpenMetrics format for a
    Prometheus scraper. Refreshes the pending-HITL gauge on each scrape so
    it reflects current state without a background poller.

    When prometheus_client isn't installed, returns a plaintext notice
    (still 200 so scrapers don't hard-fail).
    """
    from fastapi.responses import Response
    from runtime import metrics as _metrics

    # Refresh point-in-time gauges on scrape.
    try:
        _rtr = _services.get("hitl_core_router") or _services.get("hitl_router")
        if _rtr is not None and hasattr(_rtr, "_payload_store"):
            _pending = sum(
                1 for p in _rtr._payload_store.values()
                if getattr(getattr(p, "status", None), "value", None) == "pending"
            )
            _metrics.set_hitl_pending(_pending)
    except Exception:
        pass

    body, content_type = _metrics.render_latest()
    return Response(content=body, media_type=content_type)


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