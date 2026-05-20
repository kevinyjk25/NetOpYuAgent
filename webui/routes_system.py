"""
webui/routes_system.py — System diagnostics + integrations endpoints.

EXTRACTED FROM webui/backend.py during audit refactor D-4 (see
AUDIT_REPORT.md). Covers:
    /system/status   — wired-modules summary
    /system/wiring   — detailed component health
    /system/log-level — get/set per-logger levels
    /hitl/debug      — raw dump of HITL store (debug only)
    /hermes/stats    — Hermes telemetry
    /integrations/*  — tool router status / metrics / test

Public API:
    register_system_routes(app, services)
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

# Pydantic body-model imports MUST be at module level — see
# webui/schemas.py docstring for why. The late-import-inside-function
# pattern fails because FastAPI resolves type annotations via the
# enclosing module's globals.
from webui.schemas import ToolCallRequest

logger = logging.getLogger(__name__)


def register_system_routes(app: FastAPI, services: dict[str, Any]) -> None:
    """Attach /system/*, /hitl/debug, /hermes/*, /integrations/* endpoints."""
    @app.get("/system/status")
    async def system_status(
    ) -> JSONResponse:
        store    = services.get("tool_store")
        catalog  = services.get("skill_catalog")
        registry = services.get("registry")
        agents   = await registry.list_agents() if registry else []
        router   = services.get("tool_router")
        # tool_registry is wired into `services` by backend.create_webui_app
        # (line ~278) so this route can access it without sharing closures.
        tool_registry = services.get("tool_registry", {})

        return JSONResponse(content={
            "runtime_loop":    "ready",
            "tool_registry":   list(tool_registry.keys()),
            "tools_cached":    store.stored_count if store else 0,
            "skills_loaded":   catalog.skill_count if catalog else 0,
            "registry_agents": len(agents),
            "memory":          "ready" if services.get("memory") else "stub",
            "hitl":            "ready" if services.get("hitl_core_router") else "stub",
            "integrations": {
                "llm_engine":   type(services.get("llm_engine", "")).__name__,
                "mcp_tools":    router.tool_count().get("mcp", 0) if router else 0,
                "openapi_tools":router.tool_count().get("openapi", 0) if router else 0,
                "local_tools":  router.tool_count().get("local", 0) if router else 0,
            },
        })

    @app.get("/hitl/debug")
    async def hitl_debug() -> JSONResponse:
        """
        Raw dump of the HITL store — use this to diagnose HITL tab issues.
        Call GET /webui/hitl/debug after triggering a HITL interrupt.

        Speaks to hitl_core. The legacy backend (hitl/* LangGraph) was
        retired; see AUDIT_REPORT.md task A.
        """
        hitl_core_router = services.get("hitl_core_router")
        if hitl_core_router is None:
            return JSONResponse(content={"error": "hitl_core_router not wired"})
        entries = await hitl_core_router.list_pending(limit=1000)
        items = []
        for entry in entries:
            p = entry.payload
            items.append({
                "interrupt_id": p.interrupt_id,
                "thread_id":    p.thread_id,
                "trigger_kind": p.trigger_kind.value,
                "risk_level":   p.risk_level.value,
                "user_query":   (p.user_query or "")[:80],
                "intent":       (p.intent_summary or "")[:80],
            })
        return JSONResponse(content={
            "store_size": len(items),
            "interrupts": items,
            "router_id":  id(hitl_core_router),
        })

    @app.get("/system/peers")
    async def system_peers() -> JSONResponse:
        """
        List peer agents discovered through Phase-1 multi-agent setup.

        Returns this agent's identity + all peers known to the registry,
        EXCLUDING self. Use this from the WebUI / curl to confirm two
        agent processes can see each other before attempting any
        cross-agent work in later phases.

        Sample response:
        {
          "self": {"agent_id": "lan-agent", "url": "http://localhost:8000/...",
                   "capabilities": ["lan_diagnose", "lan_config"]},
          "peers": [
            {"agent_id": "wan-agent", "url": "http://localhost:8001/...",
             "health": "healthy", "capabilities": [...]}
          ]
        }
        """
        # Import the config singleton lazily so tests / re-imports work.
        from config import cfg
        registry = services.get("registry")
        if registry is None:
            return JSONResponse(
                status_code=503,
                content={"error": "registry not available"},
            )

        # Self description — built from cfg.agent + AgentCard
        self_block = {
            "agent_id":     cfg.agent.agent_id,
            "display_name": cfg.agent.display_name,
            "url":          cfg.server.a2a_base_url,
            "capabilities": [c.skill_id for c in cfg.agent.capabilities],
        }

        # All agents in the registry, then filter out self by agent_id.
        try:
            agents = await registry.list_agents()
        except Exception as exc:
            logger.warning("/system/peers: registry list failed: %s", exc)
            return JSONResponse(
                status_code=503,
                content={"self": self_block, "peers": [],
                         "error": f"registry list failed: {exc}"},
            )

        peers_out = []
        own_id = cfg.agent.agent_id
        for a in agents:
            # AgentEntry exposes agent_id, agent_url, health, skills (list of AgentSkill)
            a_id = getattr(a, "agent_id", "") or ""
            if a_id == own_id:
                continue   # don't list self
            skills_obj = getattr(a, "skills", []) or []
            caps = []
            for s in skills_obj:
                # AgentSkill may have either `id` or `skill_id` depending on
                # registry source — handle both.
                sid = getattr(s, "skill_id", "") or getattr(s, "id", "")
                if sid:
                    caps.append(sid)
            health = getattr(a, "health", None)
            health_val = (
                health.value if hasattr(health, "value")
                else str(health) if health is not None
                else "unknown"
            )
            peers_out.append({
                "agent_id":     a_id,
                "agent_url":    getattr(a, "agent_url", ""),
                "display_name": getattr(a, "agent_name", "") or a_id,
                "health":       health_val,
                "capabilities": caps,
            })

        return JSONResponse(content={
            "self":  self_block,
            "peers": peers_out,
            "peer_refresh_interval_s": cfg.agent.peer_refresh_interval_s,
        })

    @app.get("/system/wiring")
    async def system_wiring() -> JSONResponse:
        """
        Returns what is actually wired vs stub.
        Check this first when diagnosing why LLM / HITL / Memory don't work.

        The legacy "fts_store / memory_curator / user_model / dtm" names are
        kept in the response shape for UI compatibility, but the actual checks
        now look at the unified MemoryManager subsystems:
          fts_store      → memory._mgr.long_term       (FTS5 + TF-IDF over chunks)
          memory_curator → memory._mgr.extractor       (FactExtractor with LLM)
          user_model     → memory._mgr.user_model      (UserModelEngine)
          dtm            → recall_orchestrator         (always available since it's a module)
        """
        backend  = services.get("_llm_backend", "unknown")
        model    = services.get("_llm_model",   "unknown")
        has_real_llm = backend not in ("mock", "unknown")

        # Resolve the new memory subsystems through MemoryAdapter._mgr
        memory_adapter = services.get("memory")
        mgr = getattr(memory_adapter, "_mgr", None) if memory_adapter else None

        fts_alive       = bool(mgr and getattr(mgr, "long_term", None))
        extractor_alive = bool(mgr and getattr(mgr, "extractor", None))
        # FactExtractor with LLM is the spiritual successor to the old MemoryCurator
        extractor_has_llm = bool(
            extractor_alive and getattr(mgr.extractor, "_llm_fn", None) is not None
        )
        user_model_alive = bool(mgr and getattr(mgr, "user_model", None))
        # DTM has been replaced by recall_orchestrator (always available as a module)
        dtm_alive = False
        try:
            from agent_memory.retrieval import recall_orchestrator as _orch  # noqa: F401
            dtm_alive = True
        except Exception:
            dtm_alive = False

        # Stats — show recall orchestrator's tunables so the UI confirms
        # which version of the algorithm is wired.
        dtm_stats: dict = {}
        if dtm_alive:
            try:
                from agent_memory.retrieval.recall_orchestrator import (
                    TRACK_B_WEIGHT, MMR_LAMBDA,
                    SHALLOW_NUDGE_INTERVAL, DEEP_NUDGE_INTERVAL,
                )
                dtm_stats = {
                    "engine":              "recall_orchestrator",
                    "track_b_weight":      TRACK_B_WEIGHT,
                    "mmr_lambda":          MMR_LAMBDA,
                    "shallow_nudge_every": SHALLOW_NUDGE_INTERVAL,
                    "deep_nudge_every":    DEEP_NUDGE_INTERVAL,
                }
            except Exception:
                pass

        # Memory storage path — sourced from MemoryManager directly
        db_path = "not initialised"
        if mgr is not None:
            db_path = str(getattr(mgr, "_db_path", services.get("_hermes_data", "unknown")))

        return JSONResponse(content={
            "llm": {
                "backend":      backend,
                "model":        model,
                "real":         has_real_llm,
                "note": "Set LLM_BACKEND=ollama LLM_MODEL=qwen3.5:27b LLM_BASE_URL=http://localhost:11434" if not has_real_llm else "real LLM active",
            },
            # PERF-3: surface frontend poll timings so index.html can pick them up
            # without a hardcoded setInterval(...). Falls back to defaults if config
            # is missing.
            "webui": (lambda: (
                lambda _w: {
                    "hitl_poll_interval_ms":  int(getattr(_w, "hitl_poll_interval_ms",  3000)) if _w else 3000,
                    "stats_poll_interval_ms": int(getattr(_w, "stats_poll_interval_ms", 20000)) if _w else 20000,
                }
            )(getattr(__import__("config").cfg, "webui", None)))(),
            "hermes": {
                "fts_store":      fts_alive,                   # → memory._mgr.long_term
                "memory_curator": extractor_alive and extractor_has_llm,  # → FactExtractor with LLM wired
                "user_model":     user_model_alive,            # → UserModelEngine
                "skill_evolver":  services.get("skill_evolver") is not None,
                "dtm":            dtm_alive,                   # → recall_orchestrator module
                "db_path":        db_path,
                "dtm_stats":      dtm_stats,
            },
            "executor": {
                "wired":            services.get("executor") is not None,
                "hitl_core_router": services.get("hitl_core_router") is not None,
                "tool_router":      services.get("tool_router") is not None,
                "skill_catalog":    services.get("skill_catalog") is not None,
            },
            "startup_env": {
                "LLM_BACKEND":     backend,
                "LLM_MODEL":       model,
                "MCP_USE_MOCK":    str(services.get("_mcp_mock", True)),
                "HERMES_DATA_DIR": services.get("_hermes_data", "./data"),
            },
        })

    @app.get("/system/log-level")
    async def get_log_level() -> JSONResponse:
        """Return current effective log level for each key logger."""
        import logging as _logging
        loggers = [
            "integrations.llm_engine",
            "runtime.loop",
            "hitl.graph",
            "hitl.a2a_integration",
            "agent_memory.memory_manager",
            "webui.backend",
        ]
        return JSONResponse(content={
            name: _logging.getLevelName(_logging.getLogger(name).getEffectiveLevel())
            for name in loggers
        })

    @app.post("/system/log-level")
    async def set_log_level(req: Request) -> JSONResponse:
        """
        Toggle log verbosity at runtime — no restart required.

        Body: {"mode": "normal" | "llm" | "verbose"}
          normal  — INFO for everything (default)
          llm     — DEBUG for LLM messages, tool args, and tool results; INFO elsewhere
          verbose — DEBUG everywhere

        Or set a specific logger:
          {"logger": "integrations.llm_engine", "level": "DEBUG"}

        Examples:
          curl -X POST http://localhost:8000/webui/system/log-level \\
               -H 'Content-Type: application/json' -d '{"mode": "llm"}'

          curl -X POST http://localhost:8000/webui/system/log-level \\
               -H 'Content-Type: application/json' \\
               -d '{"logger": "runtime.loop", "level": "DEBUG"}'
        """
        import logging as _logging
        body = await req.json()
        mode        = body.get("mode", "")
        logger_name = body.get("logger", "")
        level_name  = body.get("level", "DEBUG").upper()

        if logger_name:
            # Set a specific logger
            lg    = _logging.getLogger(logger_name)
            level = getattr(_logging, level_name, _logging.INFO)
            lg.setLevel(level)
            return JSONResponse(content={
                "set": logger_name,
                "level": _logging.getLevelName(lg.getEffectiveLevel()),
            })

        # Mode-based — use logging_config if available
        try:
            import logging_config as _lc
            _lc.configure(mode=mode or "normal")
        except ImportError:
            # Fallback if logging_config.py isn't in path
            root_level = _logging.DEBUG if mode == "verbose" else _logging.INFO
            _logging.getLogger().setLevel(root_level)
            if mode == "llm":
                for name in ("integrations.llm_engine", "runtime.loop"):
                    _logging.getLogger(name).setLevel(_logging.DEBUG)

        return JSONResponse(content={"mode": mode or "normal", "status": "ok"})

    @app.get("/hermes/stats")
    async def hermes_stats() -> JSONResponse:
        """Live stats from Hermes learning loop modules."""
        fts_store = services.get("fts_store")
        evolver   = services.get("skill_evolver")
        try:
            fts_data = await fts_store.get_stats() if fts_store else {}
        except Exception:
            fts_data = {}
        evolver_stats = evolver.get_all_skill_stats() if evolver else []
        return JSONResponse(content={
            "total_turns":    fts_data.get("total_turns", 0),
            "total_sessions": fts_data.get("total_sessions", 0),
            "db_size_kb":     fts_data.get("db_size_kb", 0),
            "auto_skills":    len(evolver_stats),
            "fts_ready":      fts_store is not None,
            "curator_ready":  services.get("memory_curator") is not None,
            "user_model_ready": services.get("user_model") is not None,
            "evolver_ready":  evolver is not None,
        })

    @app.get("/integrations/status")
    async def integrations_status() -> JSONResponse:
        """Detailed status of all integration components."""
        mcp_client = services.get("mcp_client")
        api_client = services.get("api_client")
        llm_engine = services.get("llm_engine")
        router     = services.get("tool_router")

        mcp_tools = []
        if mcp_client:
            mcp_tools = [
                {"name": t.name, "server": t.server_name,
                 "description": t.description[:80], "returns_large": t.returns_large}
                for t in mcp_client.list_tools()
            ]

        openapi_ops = []
        if api_client:
            openapi_ops = [
                {"tool_name": op.tool_name(), "method": op.method,
                 "path": op.path, "summary": op.summary[:80]}
                for op in api_client.list_operations()
            ]

        return JSONResponse(content={
            "llm": {
                "engine":  type(llm_engine).__name__ if llm_engine else "not configured",
                "model":   getattr(llm_engine, "model", "—"),
                "backend": type(llm_engine).__name__.replace("Engine", "").lower() if llm_engine else "—",
            },
            "mcp": {
                "servers":    mcp_client.server_names if mcp_client else [],
                "tool_count": len(mcp_tools),
                "tools":      mcp_tools,
            },
            "openapi": {
                "client":    api_client.name if api_client else "not configured",
                "op_count":  len(openapi_ops),
                "operations": openapi_ops,
            },
            "tool_router": {
                "total_tools": sum(router.tool_count().values()) if router else 0,
                "by_source":   router.tool_count() if router else {},
            },
        })

    @app.get("/integrations/metrics")
    async def integrations_metrics() -> JSONResponse:
        """Per-tool call metrics (latency, error rate, circuit breaker status)."""
        router = services.get("tool_router")
        if not router:
            return JSONResponse(content={"error": "ToolRouter not initialised"})
        return JSONResponse(content={"tools": router.get_metrics()})

    @app.post("/integrations/test/{tool_name}")
    async def test_tool(tool_name: str, req: ToolCallRequest) -> JSONResponse:
        """Test any registered tool (MCP, OpenAPI, or local) directly."""
        router = services.get("tool_router")
        if not router:
            raise HTTPException(status_code=503, detail="ToolRouter not initialised")
        reg = router.registry
        if tool_name not in reg:
            raise HTTPException(status_code=404, detail=f"Tool {tool_name!r} not in ToolRouter")
        try:
            result = await reg[tool_name](req.args)
            return JSONResponse(content={"tool": tool_name, "result": result[:2000],
                                         "truncated": len(result) > 2000})
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))