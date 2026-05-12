"""
webui/backend.py
-----------------
Self-contained FastAPI backend for the IT Ops Agent WebUI.

Mount this in main.py:
    from webui.backend import create_webui_app
    app.mount("/webui", create_webui_app(_services))

Or run standalone (dev only):
    python webui/backend.py

Endpoints
---------
  GET  /webui/              → serves index.html
  GET  /webui/static/*      → static assets

  POST /webui/chat          → non-streaming query (returns full JSON response)
  POST /webui/chat/stream   → SSE streaming query
  GET  /webui/chat/history  → session message history

  GET  /webui/tools         → list available mock tools
  POST /webui/tools/{name}  → call a tool directly (for testing)
  GET  /webui/tools/result/{ref_id}  → P0: retrieve stored tool result
                                        ?offset=0&length=2000

  GET  /webui/skills        → list all skills (Level 1 summaries)
  GET  /webui/skills/{id}   → get skill detail (Level 2, on-demand)

  GET  /webui/hitl/pending  → list pending HITL interrupts
  POST /webui/hitl/{id}/approve  → approve a HITL interrupt
  POST /webui/hitl/{id}/reject   → reject a HITL interrupt

  GET  /webui/memory        → recent memory for current session
  GET  /webui/session       → current session state (facts, working set)
  GET  /webui/system/status → health of all sub-systems
  GET  /webui/ws            → WebSocket HITL + live events channel
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import time
import uuid
from typing import Annotated, Any, AsyncIterator, Optional

from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

# Auth + redaction + per-operator memory scoping
from auth import (
    Identity, verify_identity, require_role, AUTH_DISABLED,
)
# ── Auth helper ──────────────────────────────────────────────────────────────
# Endpoints used to receive `identity: Identity = Depends(verify_identity)` as
# a parameter, but FastAPI's parameter inference (with the dataclass Identity)
# kept treating it as a body field, causing 422 errors. The identity parameter
# is now resolved inline inside each endpoint via this helper, which keeps the
# endpoint signature limited to the request body alone.
async def _identity() -> Identity:
    """Resolve the current identity. Honors cfg.auth.enabled."""
    return await verify_identity()



from log_redaction import redact_text
# Rate-limit / concurrency dependencies are intentionally NOT wired right now.
# They were previously injected via FastAPI Depends() but the parameter-inference
# path conflicted with the body-param recognition for ChatRequest. To re-enable:
# 1) import per_operator_limit, global_concurrency from rate_limit
# 2) call them inline at the start of each rate-limited endpoint, e.g.:
#       async def chat_stream(req: ChatRequest):
#           await per_operator_limit_check(operator_id, rate_per_min=20)
#           ...
# This keeps the body-only signature that FastAPI parses cleanly.
from memory import set_current_operator, get_current_operator

logger = logging.getLogger(__name__)


def _streaming_cfg():
    """Lazy load AppConfig.streaming; returns None if config not loaded."""
    try:
        from config import cfg as _app_cfg
        return getattr(_app_cfg, "streaming", None)
    except Exception:
        return None


def _truncation_cfg_webui():
    try:
        from config import cfg as _app_cfg
        return getattr(_app_cfg, "truncation", None)
    except Exception:
        return None


_STATIC_DIR = pathlib.Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query:           str            = Field(..., min_length=1, max_length=8_000,
                                           description="User query — max 8 000 chars")
    session_id:      Optional[str]  = Field(None, pattern=r"^[a-zA-Z0-9_-]{1,128}$",
                                           description="Session ID — 1-128 chars of [a-zA-Z0-9_-]")
    confirmed_facts: list[str]      = Field(default_factory=list, max_length=60,
                                           description="Carry-forward facts — max 60 items")
    working_set:     list[dict]     = Field(default_factory=list, max_length=20)
    env_context:     dict           = Field(default_factory=dict)
    delegation_mode: str            = Field("fresh", pattern=r"^(fresh|forked)$")

    @field_validator("confirmed_facts")
    @classmethod
    def cap_fact_length(cls, v: list[str]) -> list[str]:
        """Prevent individual facts from inflating the LLM context."""
        return [f[:500] for f in v]

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        return v.strip()


class ToolCallRequest(BaseModel):
    args: dict[str, Any] = {}


class HitlDecisionRequest(BaseModel):
    operator_id:     str = "webui-operator"
    comment:         Optional[str] = None
    parameter_patch: Optional[dict] = None
    # For DecisionKind.CHOOSE — id of the operator-picked option
    selected_choice_id: Optional[str] = None
    # For DecisionKind.ANSWER — operator's answers to clarification fields
    clarification_answers: Optional[dict[str, str]] = None


# ---------------------------------------------------------------------------
# WebUI factory
# ---------------------------------------------------------------------------

def create_webui_app(services: dict[str, Any]) -> FastAPI:
    """
    Build and return the WebUI FastAPI sub-application.

    Expects 'services' to contain keys from main.py's build_services():
        executor, hitl_router, hitl_audit, memory, registry, task_system
    Plus runtime-specific keys added below:
        runtime_loop, tool_store, skill_catalog
    """
    app = FastAPI(title="IT Ops Agent WebUI", docs_url="/docs")


    # 422 validation error handler — logs the failure so the user can see WHY
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse as _JSON422

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request, exc: RequestValidationError):
        logger.warning(
            "422 on %s — validation errors: %s", request.url.path, exc.errors()
        )
        return _JSON422(
            status_code=422,
            content={"detail": exc.errors(), "body": str(exc.body)[:500]},
        )


    # CORS — allow only configured origins. In production set
    # NETOPYU_ALLOWED_ORIGINS="https://ops.company.com,https://admin.company.com"
    import os as _os_cors
    from fastapi.middleware.cors import CORSMiddleware
    _allowed = [o.strip() for o in _os_cors.getenv(
        "NETOPYU_ALLOWED_ORIGINS", "http://localhost:8001"
    ).split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins     = _allowed,
        allow_credentials = True,
        allow_methods     = ["GET", "POST", "DELETE"],
        allow_headers     = ["Authorization", "Content-Type", "X-API-Key"],
    )

    # ── Inject runtime components if not already present ──────────────
    from runtime import AgentRuntimeLoop, RuntimeConfig, ToolResultStore
    from skills import SkillCatalogService
    from tools import make_read_stored_result_tool

    if "tool_store" not in services:
        services["tool_store"] = ToolResultStore()

    if "skill_catalog" not in services:
        # Use ToolLoader so only mode-appropriate skills are registered.
        # No filter_to_registry needed here — ToolLoader already returns the right set.
        import config as _cfg
        from tools.loader import ToolLoader as _TL
        from skills import SkillLoader as _SL
        _tl = _TL(mode=_cfg.cfg.mode)
        _skill_loader = _SL(mode=_cfg.cfg.mode)
        catalog = SkillCatalogService()
        catalog.register_all(_skill_loader.skill_definitions())
        services["skill_catalog"] = catalog

    # Inject skill_evolver for upload/persist capability if not already provided by main.py
    if "skill_evolver" not in services:
        import os as _os, pathlib as _pl
        from skills.evolver import SkillEvolver
        _skills_dir = _os.getenv("HERMES_DATA_DIR", "./data")

        # Wire the LLM into SkillEvolver — without this, ALL skill content
        # generation falls through to _stub_llm() and returns hardcoded
        # "Network Diagnostic Procedure" boilerplate regardless of input.
        # SkillEvolver._call_llm signature: async (system, user) -> str.
        _llm_engine_for_skills = services.get("llm_engine")
        _skill_llm_fn = None
        if _llm_engine_for_skills is not None:
            async def _async_llm_for_skills(system: str, user: str) -> str:
                """Async LLM wrapper for SkillEvolver. Uses the engine's
                _chat primitive when available so we can pass system+user
                cleanly; falls back to engine.call() otherwise."""
                if hasattr(_llm_engine_for_skills, "_chat"):
                    messages = [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ]
                    return await _llm_engine_for_skills._chat(messages)
                return await _llm_engine_for_skills.call(user, system, state=None)
            _skill_llm_fn = _async_llm_for_skills

        services["skill_evolver"] = SkillEvolver(
            catalog    = services["skill_catalog"],
            skills_dir = str(_pl.Path(_skills_dir) / "skills"),
            llm_fn     = _skill_llm_fn,
        )
        if _skill_llm_fn:
            logger.info(
                "SkillEvolver: LLM-driven generation enabled (engine=%s)",
                type(_llm_engine_for_skills).__name__,
            )
        else:
            logger.warning(
                "SkillEvolver: NO llm_engine in services — content generation "
                "will use stub placeholder. Skill auto-creation and "
                "/skills/generate will produce fixed boilerplate."
            )

    # Wire read_stored_result and process_stored_chunks tools with the live store
    # Build mode-appropriate tool registry (no mock tools in pragmatic mode)
    import config as _cfg_be
    from tools.loader import ToolLoader as _TL_be
    tool_registry = _TL_be(mode=_cfg_be.cfg.mode).build_callables()
    _read_fn, _process_fn = make_read_stored_result_tool(services["tool_store"])
    tool_registry["read_stored_result"]    = _read_fn
    tool_registry["process_stored_chunks"] = _process_fn

    # Track runtime-uploaded skills/tools so they appear correctly in the left panel
    _uploaded_skill_ids: set[str] = set()
    _uploaded_tool_names: set[str] = set()

    # If main.py already built a ToolRouter, use its full registry
    # (MCP + OpenAPI + local) instead of the mock-only dict above
    tool_router = services.get("tool_router")
    if tool_router and hasattr(tool_router, "registry"):
        tool_registry = tool_router.registry
        logger.info("WebUI: using real ToolRouter registry (%d tools)", len(tool_registry))

    # Store tool_registry reference in services so upload_tool and /tools endpoint
    # share the exact same dict — uploaded tools appear in /tools immediately
    services["tool_registry"] = tool_registry

    if "runtime_loop" not in services:
        import os as _os
        # HITL tool watch-list — used by runtime/loop.py to gate LLM-proposed
        # destructive tool calls before execution. Two sources, env wins:
        #   1. HITL_TOOL_NAMES env var (comma-separated)
        #   2. cfg.tools.hitl_tool_names from config.yaml (list)
        # Without this, destructive tools execute unsupervised.
        _env_ht = _os.getenv("HITL_TOOL_NAMES", "").strip()
        if _env_ht:
            _hitl_tools = frozenset(
                t.strip() for t in _env_ht.split(",") if t.strip()
            )
        else:
            try:
                from config import cfg as _app_cfg
                _hitl_tools = frozenset(_app_cfg.tools.hitl_tool_names or [])
            except Exception as _exc:
                logger.warning("backend: cfg fallback for hitl_tool_names failed: %s", _exc)
                _hitl_tools = frozenset()
        logger.info("backend: runtime_loop hitl_tool_names=%s", sorted(_hitl_tools))
        services["runtime_loop"] = AgentRuntimeLoop(
            memory_router=services.get("memory"),
            config=RuntimeConfig(hitl_tool_names=_hitl_tools),
            tool_store=services["tool_store"],
            skill_catalog=services["skill_catalog"],
        )
    else:
        # Re-inject tool store and catalog into existing loop
        loop = services["runtime_loop"]
        loop._store   = services["tool_store"]
        loop._budget._store = services["tool_store"]
        loop._skill_catalog = services["skill_catalog"]

    # If LLM engine was already patched by main.py, it's already in the loop.
    # If not (webui started standalone), patch now with whatever engine is available.
    llm_engine = services.get("llm_engine")
    if llm_engine:
        try:
            from integrations import patch_runtime_loop
            patch_runtime_loop(services["runtime_loop"], llm_engine)
            logger.info("WebUI: runtime loop patched with LLM engine (%s/%s)",
                        llm_engine.__class__.__name__, llm_engine.model)
        except Exception as _e:
            logger.warning("WebUI: LLM patch skipped: %s", _e)

    # Session message history (in-memory, keyed by session_id)
    _message_history: dict[str, list[dict]] = {}

    # ── Static files ───────────────────────────────────────────────────
    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # ── Serve index.html ───────────────────────────────────────────────
    @app.get("/", response_class=HTMLResponse)
    async def serve_index():
        index = _STATIC_DIR / "index.html"
        if index.exists():
            return HTMLResponse(content=index.read_text(encoding="utf-8"))
        return HTMLResponse(content="<h1>IT Ops Agent WebUI</h1><p>Static files not found.</p>")

    # ==================================================================
    # Chat endpoints
    # ==================================================================

    @app.post("/chat")
    async def chat(
        req: ChatRequest,
    ) -> JSONResponse:
        set_current_operator((await _identity()).operator_id)
        """
        Non-streaming chat. Returns the full response as JSON.
        """
        session_id = req.session_id or str(uuid.uuid4())
        loop       = services["runtime_loop"]

        from runtime import DelegationMode
        dm = DelegationMode.FORKED if req.delegation_mode == "forked" else DelegationMode.FRESH

        start = time.time()
        result = await loop.run(
            query=req.query,
            session_id=session_id,
            env_context=req.env_context,
            confirmed_facts=req.confirmed_facts,
            working_set=_parse_working_set(req.working_set),
            tool_registry=tool_registry,
            delegation_mode=dm,
        )
        elapsed = round(time.time() - start, 3)

        msg = {
            "role":            "assistant",
            "content":         result.final_response,
            "session_id":      session_id,
            "turns":           result.turns_taken,
            "confirmed_facts": result.confirmed_facts,
            "stop_outcome":    result.outcome.value,
            "tool_summaries":  result.tool_summaries,
            "elapsed_s":       elapsed,
            "timestamp":       time.time(),
        }
        _push_history(session_id, {"role": "user", "content": req.query}, _message_history)
        _push_history(session_id, msg, _message_history)
        return JSONResponse(content=msg)

    @app.post("/chat/stream")
    async def chat_stream(
        req: ChatRequest,
    ) -> StreamingResponse:
        """
        SSE streaming chat — routes through ITOpsHitlAgentExecutor.

        Routing:
          SIMPLE   → executor._execute_simple()  → AgentRuntimeLoop with real LLM
          COMPLEX  → executor._execute_complex() → HITL graph, may interrupt
          Any      → post-turn Hermes hooks fire (FTS5, curation, user_model, skill_evolver)

        All executor paths use the real LLM (patched by main.py Step 6),
        the real tool registry (ToolRouter with MCP + OpenAPI + local),
        and the real HITL interrupt mechanism.

        The right panel tabs update live:
          Flow  — one event per module invocation
          Cache — auto-opens when large tool results are stored
          HITL  — interrupt card appears; Approve/Reject resumes the task
        """
        set_current_operator((await _identity()).operator_id)
        import uuid as _uuid
        session_id  = req.session_id or str(_uuid.uuid4())
        task_id     = "task-" + _uuid.uuid4().hex[:12]
        context_id  = session_id

        executor    = services.get("executor")
        loop        = services["runtime_loop"]
        # New unified memory backend (agent_memory.MemoryManager via adapter).
        # Replaces the old DTM/curator/fts/user_model split.
        memory      = services.get("memory")
        # Backward-compat aliases — all old service names resolve to the unified adapter.
        # The MemoryAdapter exposes recall_for_session(), after_turn(), get_stats() shims.
        dtm         = memory
        curator     = memory
        fts         = memory
        user_model  = None   # behaviour now embedded in adapter.update_user_profile()
        evolver     = services.get("skill_evolver")

        from runtime import DelegationMode
        dm = DelegationMode.FORKED if req.delegation_mode == "forked" else DelegationMode.FRESH

        async def generate() -> AsyncIterator[str]:
            tokens: list[str] = []
            full_text = ""
            decision  = None
            # _hitl_intercepted is read by the post-stream memory-write block
            # below, but only the SIMPLE path actually sets it (line ~545).
            # The COMPLEX/executor path never enters that branch, so the
            # variable would be undefined at the read site → UnboundLocalError.
            # Initialise it here so every path has a defined value.
            _hitl_intercepted = False
            try:
                # ── Step 1: Classify ──────────────────────────────────────
                decision = await loop.classify_async(req.query)
                yield f"data: {json.dumps({'type':'classify','complexity':decision.complexity.value,'tier':decision.model_tier,'reason':decision.reason[:100]})}\n\n"
                await asyncio.sleep(0)

                # ── Step 2: Cross-session recall (DTM v4) ────────────────
                # Recall happens before agent execution so the LLM sees prior
                # session context (e.g. "this device" in the user query refers
                # to ap-02 from the prior turn). Recalled context is injected
                # into env_context["_fts_context"] for the runtime loop.
                recall_text = ""
                memory_items: list = []
                if memory:
                    try:
                        # operator_id was set by `set_current_operator((await _identity())…)`
                        # at the top of this endpoint — recall sees the same user_id as writes.
                        recall_result = await memory.recall(req.query, session_id, max_chars=1200)
                        recall_text   = recall_result.prompt_context
                        stats = {"chunk_count": recall_result.chunk_count, "fact_count": recall_result.fact_count}
                        # MemoryAdapter.recall returns already-serialized dicts in .results
                        memory_items = recall_result.results
                        yield f"data: {json.dumps({'type':'recall','chars':len(recall_text),'sessions_searched':stats.get('total_sessions',0),'has_context':bool(recall_text),'track_a':recall_result.track_a_count,'track_b':recall_result.track_b_count,'winner':recall_result.winner,'preview':recall_text[:200],'memory_items':memory_items})}\n\n"
                        await asyncio.sleep(0)
                    except Exception as _e:
                        logger.warning("Memory recall failed (no recall event sent to FE): %s", _e, exc_info=True)
                elif curator and fts:
                    try:
                        recall_text = await curator.recall_for_session(req.query, session_id)
                        stats = await fts.get_stats()
                        yield f"data: {json.dumps({'type':'recall','chars':len(recall_text),'sessions_searched':stats.get('total_sessions',0),'has_context':bool(recall_text),'preview':recall_text[:200]})}\n\n"
                        await asyncio.sleep(0)
                    except Exception as _e:
                        logger.debug("FTS5 recall skipped: %s", _e)

                # ── Step 3: Execute via loop (all queries) ────────────────
                # Uses the real patched LLM + real ToolRouter registry.
                # Destructive-action gating is enforced by the loop's
                # tool watch-list (cfg.tools.hitl_tool_names) — there is
                # no separate pre_verify policy run here.
                real_registry = getattr(services.get("tool_router"), "registry", tool_registry)

                # Inject recalled context + user profile into env_context
                env_ctx = dict(req.env_context or {})
                if recall_text:
                    env_ctx["_fts_context"] = recall_text
                elif curator and fts:
                    try:
                        rt = await curator.recall_for_session(req.query, session_id)
                        if rt:
                            env_ctx["_fts_context"] = rt
                    except Exception:
                        pass
                if user_model:
                    try:
                        profile_section = user_model.get_prompt_section(session_id)
                        if profile_section:
                            env_ctx["_user_profile"] = profile_section
                    except Exception:
                        pass

                turns_taken  = 0
                stop_outcome = "done"

                # ── Single execution path: HitlExecutor wraps the loop ────
                # All queries (simple read-only and complex destructive) go
                # through the executor. Destructive-action gating is handled
                # by the loop's tool watch-list (cfg.tools.hitl_tool_names);
                # when the LLM proposes a watched tool, the executor raises
                # a HITL interrupt with the LLM's concrete tool_args.
                #
                # The executor calls back via on_chunk for every loop chunk
                # so we can forward steps/recall/tokens to SSE without
                # re-implementing chunk handling.
                if executor is None:
                    _err_msg = "HitlExecutor not wired (services['executor'] is None)"
                    yield f"data: {json.dumps({'type':'error','error':_err_msg})}\n\n"
                    return

                yield f"data: {json.dumps({'type':'routing','path':'hitl_executor','reason':'all queries route through hitl_executor'})}\n\n"
                await asyncio.sleep(0)

                _stream_env = dict(env_ctx)
                if decision is not None:
                    _stream_env["_initial_confidence"] = float(getattr(decision, "confidence", 1.0))

                # Build a queue we can push to from on_chunk and drain via SSE
                _chunk_queue: asyncio.Queue = asyncio.Queue()

                async def _on_chunk(ch: dict) -> None:
                    """Receive every chunk from runtime/loop.stream,
                    forward to SSE. The executor itself separately inspects
                    HITL signals on the same chunks."""
                    if "token" in ch:
                        tokens.append(ch["token"])
                    if isinstance(ch.get("node_step"), str) and ch["node_step"].startswith("Turn "):
                        nonlocal_turns[0] = nonlocal_turns[0] + 1
                    if ch.get("message"):
                        nonlocal_outcome[0] = ch["message"][:60]
                    if ch.get("stop_hitl"):
                        # Surface HITL trigger to the FE for the routing log.
                        nonlocal_outcome[0] = "stop_hitl: awaiting operator approval"
                        nonlocal_intercepted[0] = True
                    await _chunk_queue.put(("chunk", ch))

                # Mutable holders so the nested function can update outer state
                nonlocal_turns       = [0]
                nonlocal_outcome     = ["done"]
                nonlocal_intercepted = [False]

                async def _run_executor() -> None:
                    try:
                        result = await executor.execute_query(
                            query=req.query,
                            session_id=session_id,
                            confirmed_facts=list(req.confirmed_facts or []),
                            env_context=_stream_env,
                            on_chunk=_on_chunk,
                        )
                        # Surface the result via the queue so the SSE drain
                        # loop can finalise.
                        await _chunk_queue.put(("done", result))
                    except Exception as exc:
                        logger.exception("executor.execute_query failed: %s", exc)
                        await _chunk_queue.put(("error", str(exc)))

                exec_task = asyncio.create_task(_run_executor())

                # Drain queue → SSE
                _final_text = ""
                _final_interrupt: Optional[str] = None
                while True:
                    try:
                        _scfg = _streaming_cfg()
                        _stall_to = float(getattr(_scfg, "sse_stall_timeout_seconds", 180.0)) if _scfg else 180.0
                        kind, payload = await asyncio.wait_for(_chunk_queue.get(), timeout=_stall_to)
                    except asyncio.TimeoutError:
                        logger.warning("executor.execute_query stream stalled (%.1fs)", _stall_to)
                        break
                    if kind == "chunk":
                        yield f"data: {json.dumps(payload)}\n\n"
                        await asyncio.sleep(0)
                    elif kind == "done":
                        _final_text       = payload.get("text", "")
                        _final_interrupt  = payload.get("interrupt_id") if payload.get("interrupted") else None
                        if _final_interrupt:
                            yield f"data: {json.dumps({'type':'hitl_interrupt','hitl_interrupt':True,'interrupt_id':_final_interrupt})}\n\n"
                            await asyncio.sleep(0)
                        break
                    elif kind == "error":
                        yield f"data: {json.dumps({'type':'error','error':str(payload)})}\n\n"
                        break

                try:
                    _drain_to = (lambda _s: float(getattr(_s, "exec_task_drain_timeout_seconds", 5.0)) if _s else 5.0)(_streaming_cfg())
                    await asyncio.wait_for(exec_task, timeout=_drain_to)
                except (asyncio.TimeoutError, Exception):
                    pass

                turns_taken     = nonlocal_turns[0]
                stop_outcome    = nonlocal_outcome[0]
                _hitl_intercepted = nonlocal_intercepted[0] or bool(_final_interrupt)
                full_text       = _final_text or "".join(tokens)

                _push_history(session_id, {"role": "user",      "content": req.query},   _message_history)
                _push_history(session_id, {"role": "assistant", "content": full_text},    _message_history)

                # ── Step 5: Post-turn Hermes hooks ────────────────────────
                # IMPORTANT: when this turn was intercepted by HITL, the work
                # is NOT done — the operator still has to approve, and the
                # actual answer arrives later via _submit_hitl_decision. Do
                # NOT persist this turn to memory yet. If we did, recall on
                # later turns would surface the bare "⚠ HITL interrupt …
                # awaiting approval" text as if it were the final answer,
                # making the agent think the action is still pending even
                # after it has actually completed. _submit_hitl_decision
                # writes the proper {user_query, synthesis} pair when the
                # operator approves / rejects / escalates.
                if _hitl_intercepted:
                    logger.info(
                        "chat_stream: skipping memory write for session=%s "
                        "(HITL intercepted; _submit_hitl_decision will persist "
                        "the completed turn after operator decision)",
                        session_id[:12],
                    )
                else:
                    import re as _re
                    tc = [{"tool": m} for m in _re.findall(r"\[TOOL:(\w+)\]", full_text)]

                    if dtm:
                        # v4 path: DTM.after_turn() handles Track A (FTS5 + daily
                        # .md compaction) and Track B (curator → facts.jsonl).
                        try:
                            memories = await dtm.after_turn(
                                session_id     = session_id,
                                user_text      = req.query,
                                assistant_text = full_text,
                                tool_calls     = tc,
                                importance     = 0.7,
                            )
                            yield f"data: {json.dumps({'type':'hermes_write','session_id':session_id[:12],'track':'A+B'})}\n\n"
                            await asyncio.sleep(0)
                            if memories:
                                _types = [getattr(m, 'fact_type', getattr(m, 'memory_type', 'fact')) for m in memories[:5]]
                                yield f"data: {json.dumps({'type':'hermes_curate','memories_count':len(memories),'types':_types})}\n\n"
                                await asyncio.sleep(0)
                        except Exception as _e:
                            logger.debug("DTM after_turn skipped: %s", _e)
                    else:
                        # v3 fallback: individual hooks when DTM not wired
                        if fts:
                            try:
                                await fts.write_turn(session_id, req.query, full_text, tool_calls=tc, importance=0.7)
                                yield f"data: {json.dumps({'type':'hermes_write','session_id':session_id[:12]})}\n\n"
                                await asyncio.sleep(0)
                            except Exception as _e:
                                logger.debug("FTS5 write skipped: %s", _e)

                        if curator:
                            try:
                                memories = await curator.after_turn(session_id, req.query, full_text, tc)
                                _types = [getattr(m, 'fact_type', getattr(m, 'memory_type', 'fact')) for m in memories[:5]]
                                yield f"data: {json.dumps({'type':'hermes_curate','memories_count':len(memories),'types':_types})}\n\n"
                                await asyncio.sleep(0)
                            except Exception as _e:
                                logger.debug("Curation skipped: %s", _e)

                    # User model always runs (not inside DTM scope)
                    if user_model:
                        try:
                            profile = await user_model.after_turn(session_id, req.query, full_text, tc)
                            yield f"data: {json.dumps({'type':'hermes_umodel','technical_level':profile.technical_level.value,'domain_counts':dict(list(profile.domain_counts.items())[:5]),'trait_count':len(profile.traits)})}\n\n"
                            await asyncio.sleep(0)
                        except Exception as _e:
                            logger.debug("User model skipped: %s", _e)

                    if evolver and decision and decision.complexity.value == "complex":
                        try:
                            proposal = await evolver.after_task(
                                task_description=req.query, solution_summary=full_text[:400],
                                tools_used=[t["tool"] for t in tc], solution_steps=[],
                                key_observations=[], complexity=7.0, session_id=session_id,
                            )
                            yield f"data: {json.dumps({'type':'hermes_skill','created':proposal is not None,'skill_id':proposal.skill_id if proposal else None})}\n\n"
                            await asyncio.sleep(0)
                        except Exception as _e:
                            logger.debug("Skill evolver skipped: %s", _e)

                # Include confirmed_facts so frontend can carry them to next query
                _done_facts = getattr(loop, '_last_confirmed_facts', []) or []

                # Distinguish HITL-pending terminal state from a real "done".
                # When the agent is paused waiting for operator approval, do NOT
                # emit a 'done' event (that fires the ✅ Done step in the UI).
                # Emit 'awaiting_hitl' instead — the UI will show ⏸ status and
                # finalize the run only after the operator decides + post-action runs.
                _is_hitl_pending = (
                    "hitl" in str(stop_outcome).lower()
                    or "approval" in str(stop_outcome).lower()
                    or "interrupt" in str(stop_outcome).lower()
                )
                if _is_hitl_pending:
                    yield f"data: {json.dumps({'type':'awaiting_hitl','session_id':session_id,'turns':turns_taken,'stop_outcome':stop_outcome,'confirmed_facts':_done_facts})}\n\n"
                else:
                    yield f"data: {json.dumps({'type':'done','session_id':session_id,'turns':turns_taken,'stop_outcome':stop_outcome,'confirmed_facts':_done_facts})}\n\n"
                yield "data: [DONE]\n\n"

            except Exception as exc:
                logger.exception("Stream error: %s", exc)
                yield f"data: {json.dumps({'type':'error','error':str(exc)})}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/chat/history")
    async def chat_history(session_id: str = "default") -> JSONResponse:
        """Returns message history — prefers FTS5 (persistent) over in-memory cache."""
        fts = services.get("fts_store")
        if fts:
            try:
                turns = await fts.get_session_turns(session_id, limit=100)
                turns = list(reversed(turns))
                messages = []
                for t in turns:
                    messages.append({"role": "user",      "content": t.user_text,      "ts": t.ts})
                    messages.append({"role": "assistant",  "content": t.assistant_text, "ts": t.ts})
                return JSONResponse(content=messages)
            except Exception:
                pass
        return JSONResponse(content=_message_history.get(session_id, []))

    # ==================================================================
    # Tools endpoints
    # ==================================================================


    @app.post("/tools/upload")
    async def upload_tool(request: Request,
    ) -> JSONResponse:
        """
        Upload a Python tool file (.py).  The file must define one or more
        async functions and a TOOL_REGISTRY dict mapping names → functions,
        OR export individual functions whose names are the tool IDs.

        SECURITY: This endpoint executes uploaded Python code. It is gated
        behind the 'admin' role (when auth.enabled=true) and runs an AST
        denylist check before exec to block obvious shell-out / file-system
        escape attempts. The AST check is best-effort, NOT a sandbox — only
        deploy this endpoint inside a trusted operator network, never expose
        it to the public internet.
        """
        # Require admin role. When auth.enabled=false, _identity() returns
        # the dev-user with admin role anyway, so dev mode still works.
        ident = await _identity()
        if not ident.has_role("admin"):
            raise HTTPException(
                status_code=403,
                detail="Tool upload requires the 'admin' role",
            )

        try:
            form = await request.form()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to parse form data: {exc}")

        upload = form.get("file")
        if upload is None:
            raise HTTPException(status_code=400, detail="No file field in form data — field name must be 'file'")

        filename = getattr(upload, "filename", None) or "uploaded_tool.py"
        if not filename.endswith(".py"):
            raise HTTPException(status_code=400, detail="Only .py files are supported")

        try:
            content_bytes = await upload.read()
            source = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File must be UTF-8 encoded")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to read file: {exc}")

        # AST denylist — block obviously dangerous imports/calls before exec.
        # This is NOT a sandbox; a determined attacker can bypass it. Pair with
        # admin-role gating + trusted-network deployment.
        import ast as _ast_mod
        _DENIED_IMPORTS = {
            "os", "subprocess", "sys", "shutil", "socket", "ctypes",
            "multiprocessing", "pty", "popen2", "commands",
            "_winreg", "winreg",
        }
        _DENIED_CALLS = {"eval", "exec", "compile", "__import__", "open"}
        try:
            tree = _ast_mod.parse(source, filename=filename)
        except SyntaxError as exc:
            raise HTTPException(status_code=400, detail=f"Syntax error: {exc}")

        for node in _ast_mod.walk(tree):
            if isinstance(node, _ast_mod.Import):
                for n in node.names:
                    if n.name.split(".")[0] in _DENIED_IMPORTS:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Import of {n.name!r} is not allowed in uploaded tools",
                        )
            elif isinstance(node, _ast_mod.ImportFrom):
                if node.module and node.module.split(".")[0] in _DENIED_IMPORTS:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Import from {node.module!r} is not allowed in uploaded tools",
                    )
            elif isinstance(node, _ast_mod.Call):
                fn = node.func
                if isinstance(fn, _ast_mod.Name) and fn.id in _DENIED_CALLS:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Call to {fn.id!r} is not allowed in uploaded tools",
                    )

        # Compile (we already parsed; compile catches anything ast.parse missed)
        try:
            code = compile(source, filename, "exec")
        except SyntaxError as exc:
            raise HTTPException(status_code=400, detail=f"Syntax error: {exc}")

        # Execute in an isolated namespace
        import asyncio as _asyncio_mod, inspect as _inspect_mod
        ns: dict = {"__builtins__": __builtins__}
        try:
            exec(code, ns)  # noqa: S102
        except Exception as exc:
            logger.exception("upload_tool: exec failed for %s", filename)
            raise HTTPException(
                status_code=400,
                detail=f"Execution error in {filename}: {type(exc).__name__}: {exc}",
            )

        # Extract tools — prefer explicit TOOL_REGISTRY, else all async callables
        new_tools: dict = {}
        if "TOOL_REGISTRY" in ns and isinstance(ns["TOOL_REGISTRY"], dict):
            new_tools = {k: v for k, v in ns["TOOL_REGISTRY"].items() if callable(v)}
        else:
            import asyncio as _asyncio, inspect as _inspect
            new_tools = {
                name: fn
                for name, fn in ns.items()
                if not name.startswith("_") and callable(fn) and _inspect.iscoroutinefunction(fn)
            }

        if not new_tools:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No tools found. Define a TOOL_REGISTRY dict or "
                    "at least one top-level async function."
                ),
            )

        # Register into ALL live registries so uploaded tools are immediately callable
        # and visible in the Quick Tools panel (/tools endpoint reads services["tool_registry"])
        loop = services.get("runtime_loop")
        if loop and hasattr(loop, "_tool_registry"):
            loop._tool_registry.update(new_tools)

        tool_router_svc = services.get("tool_router")
        if tool_router_svc:
            # .registry is a @property that rebuilds each call — must write to _callables
            if hasattr(tool_router_svc, "_callables"):
                for name, fn in new_tools.items():
                    tool_router_svc._callables[name] = fn
                    # Initialise meta entry so the circuit-breaker wrapper works
                    if hasattr(tool_router_svc, "_meta") and name not in tool_router_svc._meta:
                        from integrations.router.tool_router import ToolMeta  # noqa
                        tool_router_svc._meta[name] = ToolMeta(name, source)

        # Update the shared tool_registry in services (read by /tools and chat/stream)
        svc_reg = services.get("tool_registry")
        if svc_reg is not None:
            svc_reg.update(new_tools)
        tool_registry.update(new_tools)

        registered = list(new_tools.keys())
        logger.info("Tool(s) uploaded and registered: %s (%d tools)", registered, len(registered))
        return JSONResponse(content={
            "status":     "registered",
            "tools":      registered,
            "chars":      len(source),
            "tool_count": len(registered),
        })

    @app.post("/tools/{tool_name}")
    async def call_tool(tool_name: str, req: ToolCallRequest) -> JSONResponse:
        """
        Directly invoke a mock tool and return its raw output.
        Large outputs are stored and a ref label is returned alongside.
        """
        fn = tool_registry.get(tool_name)
        if fn is None:
            raise HTTPException(status_code=404, detail=f"Tool {tool_name!r} not found")
        try:
            raw = await fn(req.args)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        store = services["tool_store"]
        stored_label = store.store(tool_name, raw)
        is_stored    = stored_label != raw
        ref_id       = None
        if is_stored:
            # Extract ref_id from label: [STORED:tool:ref_id]
            import re
            m = re.search(r"\[STORED:[^:]+:([^\]]+)\]", stored_label)
            if m:
                ref_id = m.group(1)

        return JSONResponse(content={
            "tool":       tool_name,
            "args":       req.args,
            "output":     stored_label,
            "raw_length": len(raw),
            "is_stored":  is_stored,
            "ref_id":     ref_id,
            "retrieve_url": f"/webui/tools/result/{ref_id}" if ref_id else None,
        })

    @app.get("/tools/result/{ref_id}")
    async def get_stored_result(
        ref_id: str,
        offset: int = 0,
        length: int = 2000,
    ) -> JSONResponse:
        """
        P0: Retrieve a page of a large tool result by reference ID.

        Example flow:
          1. POST /webui/tools/syslog_search  → {"is_stored": true, "ref_id": "abc123"}
          2. GET  /webui/tools/result/abc123?offset=0       → first 2KB
          3. GET  /webui/tools/result/abc123?offset=2000    → next 2KB
          4. GET  /webui/tools/result/abc123?offset=4000    → etc.

        Response fields:
          ref_id      : the reference ID
          offset      : start of this page
          length      : bytes in this page
          total_chars : full stored size
          has_more    : whether there is more data after this page
          next_offset : offset to use for the next request
          content     : the text slice
        """
        store = services["tool_store"]
        chunk = store.read(ref_id, offset=offset, length=length)
        if chunk is None:
            raise HTTPException(
                status_code=404,
                detail=f"No stored result found for ref_id={ref_id!r}. "
                       "Results may have been cleared or the ref_id is invalid."
            )
        # Normalise ref_id in case it includes tool_name prefix
        _norm_ref = ref_id.strip("[]")
        if ":" in _norm_ref:
            _norm_ref = _norm_ref.rsplit(":", 1)[-1].strip()
        total      = len(store._store.get(_norm_ref, ""))
        next_off   = offset + len(chunk)
        has_more   = next_off < total

        return JSONResponse(content={
            "ref_id":      ref_id,
            "offset":      offset,
            "length":      len(chunk),
            "total_chars": total,
            "has_more":    has_more,
            "next_offset": next_off if has_more else None,
            "content":     chunk,
        })

    # ==================================================================
    # Skills endpoints
    # ==================================================================

    
    @app.get("/skill_journal/recent")
    async def skill_journal_recent(limit: int = 20):
        """Return the most recent SkillJournal entries (newest first)."""
        try:
            from config import cfg as _app_cfg
            _so = getattr(_app_cfg, "skill_orchestration", None)
            if not _so or not getattr(_so, "journal_api_enabled", True):
                raise HTTPException(503, "Skill journal API disabled")
            from runtime.skill_journal import get_journal_store
            store = get_journal_store()
            return JSONResponse(content={"entries": store.list_recent(limit=min(max(1, limit), 100))})
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("/skill_journal/recent failed: %s", exc)
            raise HTTPException(500, str(exc))

    @app.get("/skill_journal/stats")
    async def skill_journal_stats():
        """Aggregate stats: outcomes, per-skill use count, dormancy."""
        try:
            from config import cfg as _app_cfg
            _so = getattr(_app_cfg, "skill_orchestration", None)
            if not _so or not getattr(_so, "journal_api_enabled", True):
                raise HTTPException(503, "Skill journal API disabled")
            from runtime.skill_journal import get_journal_store
            return JSONResponse(content=get_journal_store().stats())
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("/skill_journal/stats failed: %s", exc)
            raise HTTPException(500, str(exc))

    @app.get("/skill_journal/filter")
    async def skill_journal_filter(
        skill_id:   Optional[str]  = None,
        outcome:    Optional[str]  = None,
        ambiguous:  Optional[bool] = None,
        limit:      int            = 50,
    ):
        """Filter journal entries by skill, outcome, or ambiguity flag."""
        try:
            from config import cfg as _app_cfg
            _so = getattr(_app_cfg, "skill_orchestration", None)
            if not _so or not getattr(_so, "journal_api_enabled", True):
                raise HTTPException(503, "Skill journal API disabled")
            from runtime.skill_journal import get_journal_store
            return JSONResponse(content={
                "entries": get_journal_store().filter(
                    skill_id=skill_id, outcome=outcome,
                    ambiguous=ambiguous, limit=min(max(1, limit), 200),
                )
            })
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("/skill_journal/filter failed: %s", exc)
            raise HTTPException(500, str(exc))

    @app.get("/skill_journal/recent")
    async def skill_journal_recent(limit: int = 20):
        """Return the most recent SkillJournal entries (newest first)."""
        try:
            from config import cfg as _app_cfg
            _so = getattr(_app_cfg, "skill_orchestration", None)
            if not _so or not getattr(_so, "journal_api_enabled", True):
                raise HTTPException(503, "Skill journal API disabled")
            from runtime.skill_journal import get_journal_store
            store = get_journal_store()
            return JSONResponse(content={"entries": store.list_recent(limit=min(max(1, limit), 100))})
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("/skill_journal/recent failed: %s", exc)
            raise HTTPException(500, str(exc))

    @app.get("/skill_journal/stats")
    async def skill_journal_stats():
        """Aggregate stats: outcomes, per-skill use count, dormancy."""
        try:
            from config import cfg as _app_cfg
            _so = getattr(_app_cfg, "skill_orchestration", None)
            if not _so or not getattr(_so, "journal_api_enabled", True):
                raise HTTPException(503, "Skill journal API disabled")
            from runtime.skill_journal import get_journal_store
            return JSONResponse(content=get_journal_store().stats())
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("/skill_journal/stats failed: %s", exc)
            raise HTTPException(500, str(exc))

    @app.get("/skill_journal/filter")
    async def skill_journal_filter(
        skill_id:  Optional[str]  = None,
        outcome:   Optional[str]  = None,
        ambiguous: Optional[bool] = None,
        limit:     int            = 50,
    ):
        """Filter journal entries by skill, outcome, or ambiguity flag."""
        try:
            from config import cfg as _app_cfg
            _so = getattr(_app_cfg, "skill_orchestration", None)
            if not _so or not getattr(_so, "journal_api_enabled", True):
                raise HTTPException(503, "Skill journal API disabled")
            from runtime.skill_journal import get_journal_store
            return JSONResponse(content={
                "entries": get_journal_store().filter(
                    skill_id=skill_id, outcome=outcome,
                    ambiguous=ambiguous, limit=min(max(1, limit), 200),
                )
            })
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("/skill_journal/filter failed: %s", exc)
            raise HTTPException(500, str(exc))

    @app.get("/skills")
    async def list_skills() -> JSONResponse:
        catalog    = services["skill_catalog"]
        skill_evol = services.get("skill_evolver")
        import pathlib as _pl

        # Which skills live on disk (evolved / uploaded — not just built-in)
        evolved_ids: set = set()
        if skill_evol and getattr(skill_evol, "_skills_dir", None):
            skills_dir = _pl.Path(skill_evol._skills_dir)
            if skills_dir.exists():
                evolved_ids = {p.stem for p in skills_dir.glob("*.md")}

        return JSONResponse(content=[
            {
                "skill_id":      s.skill_id,
                "name":          s.name,
                "purpose":       s.purpose,
                "risk_level":    s.risk_level,
                "requires_hitl": s.requires_hitl,
                "tags":          s.tags,
                "is_evolved":    s.skill_id in evolved_ids,
            }
            for s in catalog.list_skills()
        ])

    @app.get("/skills/{skill_id}")
    async def get_skill_detail(skill_id: str) -> JSONResponse:
        """Load skill Level 2 detail on demand — progressive disclosure."""
        catalog = services["skill_catalog"]
        detail  = catalog.load_detail(skill_id)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"Skill {skill_id!r} not found")
        summary = catalog.get_summary(skill_id)
        return JSONResponse(content={
            "skill_id":      skill_id,
            "requires_hitl": catalog.requires_hitl(skill_id),
            "detail":        detail,
            "risk_level":    summary.risk_level if summary else "unknown",
        })

    @app.post("/skills/upload")
    async def upload_skill(request: Request,
    ) -> JSONResponse:
        """
        Upload a skill markdown file (.md) or JSON definition (.json).
        The skill is registered in the catalog and persisted to HERMES_DATA_DIR/skills/.
        Uses Request directly (not File()) to work correctly in mounted sub-apps.
        Gated behind 'admin' role.
        """
        ident = await _identity()
        if not ident.has_role("admin"):
            raise HTTPException(
                status_code=403,
                detail="Skill upload requires the 'admin' role",
            )

        catalog    = services.get("skill_catalog")
        skill_evol = services.get("skill_evolver")
        if not catalog:
            raise HTTPException(status_code=503, detail="Skill catalog not available")

        try:
            form = await request.form()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to parse form data: {exc}")

        upload = form.get("file")
        if upload is None:
            raise HTTPException(status_code=400, detail="No file field in form data — field name must be 'file'")

        try:
            content_bytes = await upload.read()
            content = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File must be UTF-8 encoded text")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to read file: {exc}")

        filename  = getattr(upload, "filename", None) or "uploaded_skill"
        skill_id  = filename.removesuffix(".md").removesuffix(".json")
        # Sanitise: only alphanumeric + underscore
        import re as _re
        skill_id  = _re.sub(r"[^a-zA-Z0-9_]", "_", skill_id).strip("_") or "uploaded_skill"

        if filename.endswith(".json"):
            import json as _json
            try:
                defn = _json.loads(content)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}")
            skill_id = defn.get("skill_id", skill_id)
            try:
                catalog.register_all({skill_id: defn})
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Registration failed: {exc}")
        else:
            # Markdown — use SkillEvolver parser if available, else minimal parse
            if skill_evol and hasattr(skill_evol, "_parse_markdown_to_definition"):
                defn = skill_evol._parse_markdown_to_definition(skill_id, content)
                catalog.register_all({skill_id: defn})
            else:
                # Minimal fallback: register with raw content as description
                catalog.register_all({skill_id: {
                    "name":          skill_id.replace("_", " ").title(),
                    "purpose":       content.split("\n")[0].lstrip("# ").strip()[:200],
                    "description":   content,
                    "risk_level":    "low",
                    "requires_hitl": False,
                    "tags":          [],
                    "parameters":    {},
                    "returns":       "string",
                    "examples":      [],
                    "constraints":   [],
                    "estimated_size": "small",
                    "returns_large":  False,
                }})

        # Persist to disk via SkillEvolver if available
        if skill_evol and hasattr(skill_evol, "_save_skill_to_disk"):
            skill_evol._save_skill_to_disk(skill_id, content)

        logger.info("Skill uploaded and registered: %s (persisted=%s)", skill_id,
                     bool(skill_evol and getattr(skill_evol, "_skills_dir", None)))
        return JSONResponse(content={
            "status":   "registered",
            "skill_id": skill_id,
            "chars":    len(content),
            "persisted": bool(skill_evol and getattr(skill_evol, "_skills_dir", None)),
        })

    @app.post("/skills/generate")
    async def generate_skill_from_text(request: Request) -> JSONResponse:
        """
        Generate a skill markdown draft from a free-form conversation snippet.

        The user pastes a chat excerpt (or any prose describing a procedure);
        the LLM converts it to the standard skill format. The draft is
        RETURNED but NOT registered — the user reviews + edits it in the UI,
        then POSTs to /skills/upload to actually register it.

        Body (JSON):
          {
            "text":       "<conversation excerpt or procedure description>",
            "hint_name":  "<optional desired skill name>",
            "hint_tags":  ["optional", "tags"]
          }

        Returns:
          {
            "skill_id":   "<auto-generated stable id>",
            "markdown":   "<draft markdown content>",
            "similar_to": "<existing_id>" | null,   // if Jaccard ≥ 0.35
            "similarity": 0.42                      // Jaccard score
          }

        The frontend can then:
          - Show the draft in an editable textarea
          - If `similar_to` is set, offer "Merge into <existing>" instead of new
          - On confirm, send the (possibly edited) markdown to /skills/upload
        """
        ident = await _identity()
        if not ident.has_role("admin"):
            raise HTTPException(
                status_code=403,
                detail="Skill generation requires the 'admin' role",
            )

        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Body must be JSON")

        text = (body or {}).get("text", "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="`text` field is required")
        if len(text) > 10_000:
            raise HTTPException(status_code=400, detail="`text` exceeds 10000 chars")

        hint_name = (body or {}).get("hint_name", "").strip()
        hint_tags = (body or {}).get("hint_tags", []) or []

        skill_evol = services.get("skill_evolver")
        catalog    = services.get("skill_catalog")
        if skill_evol is None:
            raise HTTPException(status_code=503, detail="SkillEvolver not configured")

        # Diagnostic: warn early if SkillEvolver has no LLM wired — without
        # this the response would be the static stub regardless of input.
        if getattr(skill_evol, "_llm_fn", None) is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "SkillEvolver has no LLM configured — generation would "
                    "produce hardcoded boilerplate. Check server logs for "
                    "'SkillEvolver: NO llm_engine in services' and ensure "
                    "llm_engine is registered in services."
                ),
            )

        # 1. Generate the skill markdown FIRST. Doing this before we compute
        #    the skill_id and similarity lets us derive the id from the
        #    actual generated title (semantically meaningful) instead of
        #    from arbitrary tokens in the user's raw input (which may be
        #    JSON keys, prose, or just "故障诊断").
        from skills.evolver import _SKILL_WRITE_SYSTEM
        user_content = (
            f"Source text (operator-supplied conversation/procedure):\n"
            f"-----\n{text[:6000]}\n-----\n\n"
            f"Desired skill name hint: {hint_name or '(infer from text)'}\n"
            f"Desired tags hint: {', '.join(hint_tags) if hint_tags else '(infer)'}\n\n"
            f"Convert the source text above into a standard skill markdown file. "
            f"Capture the actionable steps, identify which tools/parameters are used, "
            f"and infer a reasonable Risk and HITL level. The Tags line MUST contain "
            f"3-5 short English/lowercase keywords describing the skill domain "
            f"(e.g. [network, dns, troubleshooting]). Keep total length under 1500 chars."
        )
        try:
            raw = await skill_evol._call_llm(_SKILL_WRITE_SYSTEM, user_content)
            import re as _re_local
            markdown = _re_local.sub(r"^```(?:markdown)?\s*\n?", "", raw.strip()).rstrip("```").strip()
        except Exception as exc:
            logger.warning("/skills/generate LLM call failed: %s", exc)
            raise HTTPException(status_code=502, detail=f"LLM generation failed: {exc}")

        if not markdown or len(markdown) < 30:
            raise HTTPException(status_code=502, detail="LLM produced empty/too-short content")

        # Detect stub fallback: if the response equals the well-known stub
        # output, refuse instead of returning misleading boilerplate.
        if "Network Diagnostic Procedure" in markdown[:80] and "get_device_status" in markdown:
            # Cross-check: was the input actually about generic network diagnostic?
            text_lower = text.lower()
            looks_legitimately_about_topic = any(
                k in text_lower for k in ("network diagnostic", "diagnose network", "get_device_status")
            )
            if not looks_legitimately_about_topic:
                logger.error(
                    "/skills/generate: LLM appears to have returned the stub "
                    "fallback (Network Diagnostic Procedure) — input did NOT "
                    "request that. LLM call likely failed silently."
                )
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "LLM returned stub-fallback content (hardcoded "
                        "'Network Diagnostic Procedure'). Your input was about "
                        "something else. Check llm_engine connectivity in server logs."
                    ),
                )

        # 2. Derive a stable, meaningful skill_id from the generated markdown.
        #    Priority: hint_name → H1 title → text fallback.
        title_source = hint_name
        if not title_source:
            m = _re_local.match(r"^\s*#\s+(.+)$", markdown, flags=_re_local.MULTILINE)
            if m:
                title_source = m.group(1).strip()
        skill_id = skill_evol._generate_skill_id(title_source or text[:200])

        # 3. Run similarity check on the GENERATED skill signature (H1 + tags
        #    parsed from the markdown). This is far more accurate than running
        #    similarity on raw user input — generated skills always include
        #    standardised English tag keywords that match catalog entries.
        signature_for_sim = title_source or text[:200]
        # Augment with parsed tags from the generated markdown
        tag_match = _re_local.search(
            r"\*\*Tags:\*\*\s*\[([^\]]*)\]", markdown, flags=_re_local.IGNORECASE,
        )
        if tag_match:
            signature_for_sim += " " + tag_match.group(1)

        similar = await skill_evol._find_similar_skill(signature_for_sim)
        similar_id      = similar[0] if similar else None
        similar_score   = similar[1] if similar else 0.0
        similar_summary = None
        if similar_id and catalog:
            sm = catalog.get_summary(similar_id)
            if sm:
                similar_summary = {
                    "name":    sm.name,
                    "purpose": sm.purpose,
                    "tags":    sm.tags,
                }

        # 4. Detect explicit id collision (same skill_id already registered)
        #    so the UI can warn even when similarity is below threshold.
        id_collides = False
        if catalog and skill_id:
            try:
                id_collides = catalog.get_summary(skill_id) is not None
            except Exception:
                id_collides = False

        logger.info(
            "/skills/generate: id=%s draft_chars=%d similar_to=%s (j=%.2f) id_collides=%s",
            skill_id, len(markdown), similar_id, similar_score, id_collides,
        )
        return JSONResponse(content={
            "skill_id":         skill_id,
            "markdown":         markdown,
            "similar_to":       similar_id,
            "similarity":       round(similar_score, 3),
            "similar_summary":  similar_summary,    # name/purpose/tags of conflict, for UI
            "id_collides":      id_collides,        # exact id already exists
        })

    @app.get("/skills/{skill_id}/content")
    async def get_skill_raw_content(skill_id: str) -> JSONResponse:
        """
        Return the human-readable markdown content of a skill.
        Priority:
          1. Disk file (HERMES_DATA_DIR/skills/<id>.md) — evolved/uploaded skills
          2. catalog.as_markdown()                       — built-in skills synthesised as markdown
          3. 404 if not registered at all
        """
        skill_evol = services.get("skill_evolver")
        raw_content = None
        source = "unknown"

        # 1. Try disk file first (evolved / uploaded skills)
        if skill_evol and getattr(skill_evol, "_skills_dir", None):
            import pathlib as _pl
            path = _pl.Path(skill_evol._skills_dir) / f"{skill_id}.md"
            if path.exists():
                raw_content = path.read_text(encoding="utf-8")
                source = "disk"

        # 2. Fall back to catalog.as_markdown() — works for built-in skills too
        if raw_content is None:
            catalog = services.get("skill_catalog")
            if catalog and hasattr(catalog, "as_markdown"):
                raw_content = catalog.as_markdown(skill_id)
                if raw_content:
                    source = "catalog"

        if raw_content is None:
            raise HTTPException(status_code=404, detail=f"Skill {skill_id!r} not found")

        return JSONResponse(content={
            "skill_id": skill_id,
            "content":  raw_content,
            "source":   source,
        })

    @app.get("/tools")
    async def list_tools() -> JSONResponse:
        """List tools valid for the current running mode (no mock tools in pragmatic)."""
        from tools.loader import ToolLoader
        import config as _cfg
        # ToolLoader returns only tools for the current mode (mock vs pragmatic)
        _loader = ToolLoader(mode=_cfg.cfg.mode)
        all_tools = {}
        for name, meta in _loader.build_metadata().items():
            entry = dict(meta)
            entry["uploaded"] = False
            all_tools[name] = entry
        # Also surface any live-registered/uploaded tools not in the static metadata
        live_reg = services.get("tool_registry") or {}
        for name in live_reg:
            if name not in all_tools:
                all_tools[name] = {
                    "description": f"Uploaded tool: {name}",
                    "parameters":  {},
                    "returns_large": False,
                    "example":     {},
                    "uploaded":    True,
                }
        logger.debug("/tools: returning %d tools (%d uploaded)",
                     len(all_tools),
                     sum(1 for t in all_tools.values() if t.get("uploaded")))
        return JSONResponse(content=all_tools)


    # ==================================================================
    # HITL endpoints
    # ==================================================================

    @app.get("/hitl/pending")
    async def list_pending_hitl(
    ) -> JSONResponse:
        # Prefer hitl_core if wired up (HITL_BACKEND=core); fall back to
        # legacy hitl_router. The two paths return the same JSON shape so
        # the frontend doesn't need to change.
        hitl_core_router = services.get("hitl_core_router")
        if hitl_core_router is not None:
            entries = await hitl_core_router.list_pending(limit=100)
            result = []
            for entry in entries:
                p = entry.payload
                result.append({
                    "interrupt_id":   p.interrupt_id,
                    "thread_id":      p.thread_id,
                    "trigger_kind":   p.trigger_kind.value,
                    "risk_level":     p.risk_level.value,
                    "user_query":     p.user_query,
                    "intent_summary": p.intent_summary,
                    "sla_seconds":    p.sla_seconds,
                    "proposed_action": (
                        p.proposed_action.model_dump()
                        if p.proposed_action else {}
                    ),
                    "choices":               [c.model_dump() for c in (p.choices or [])],
                    "clarification_fields":  [f.model_dump() for f in (p.clarification_fields or [])],
                    "editable_param_keys":   list(p.editable_param_keys or []),
                })
            # PERF-3: only log at INFO when there's actually something pending,
            # otherwise DEBUG to keep the log clean.
            try:
                from config import cfg as _app_cfg
                _info_always = bool(getattr(getattr(_app_cfg, "webui", None), "hitl_pending_log_at_info", False))
            except Exception:
                _info_always = False
            if len(result) > 0 or _info_always:
                logger.info("/hitl/pending [core]: returning %d", len(result))
            else:
                logger.debug("/hitl/pending [core]: returning 0")
            return JSONResponse(content=result)

        # Legacy path
        hitl_router = services.get("hitl_router")
        if not hitl_router:
            logger.warning("/hitl/pending: no HITL router available")
            return JSONResponse(content=[])
        from hitl.schemas import InterruptState
        store_size = len(hitl_router._payload_store)
        logger.info(
            "/hitl/pending: store_size=%d ids=%s",
            store_size,
            [
                f"{k[:8]}…={getattr(v.status,'value',v.status)}"
                for k, v in hitl_router._payload_store.items()
            ],
        )
        result = []
        for p in hitl_router._payload_store.values():
            # Compare robustly: status may be enum or raw string
            status_val = p.status.value if hasattr(p.status, "value") else str(p.status)
            if status_val in ("pending", InterruptState.PENDING.value):
                try:
                    dumped = p.model_dump()
                except Exception:
                    # Fallback for non-pydantic objects
                    dumped = {
                        "interrupt_id":   getattr(p, "interrupt_id", ""),
                        "trigger_kind":   getattr(p.trigger_kind, "value", str(getattr(p, "trigger_kind", ""))),
                        "risk_level":     getattr(p.risk_level,   "value", str(getattr(p, "risk_level",   ""))),
                        "user_query":     getattr(p, "user_query",     ""),
                        "intent_summary": getattr(p, "intent_summary", ""),
                        "sla_seconds":    getattr(p, "sla_seconds",    600),
                        "proposed_action": (
                            p.proposed_action.model_dump()
                            if hasattr(p, "proposed_action") and p.proposed_action and hasattr(p.proposed_action, "model_dump")
                            else getattr(p, "proposed_action", {}) or {}
                        ),
                        "choices": [
                            c.model_dump() if hasattr(c, "model_dump") else c
                            for c in (getattr(p, "choices", []) or [])
                        ],
                        "clarification_fields": [
                            f.model_dump() if hasattr(f, "model_dump") else f
                            for f in (getattr(p, "clarification_fields", []) or [])
                        ],
                        "editable_param_keys": list(getattr(p, "editable_param_keys", []) or []),
                    }
                result.append(dumped)
        try:
            from config import cfg as _app_cfg
            _info_always_2 = bool(getattr(getattr(_app_cfg, "webui", None), "hitl_pending_log_at_info", False))
        except Exception:
            _info_always_2 = False
        if len(result) > 0 or _info_always_2:
            logger.info("/hitl/pending: returning %d pending interrupts", len(result))
        else:
            logger.debug("/hitl/pending: returning 0 pending interrupts")
        return JSONResponse(content=result)

    @app.post("/hitl/{interrupt_id}/approve")
    async def approve_hitl(
        interrupt_id: str,
        req: HitlDecisionRequest,
    ) -> JSONResponse:
        # Override client-supplied operator_id with the verified identity
        # → audit log records the actual approver, not whoever the client claims
        req.operator_id = (await _identity()).operator_id
        return await _submit_hitl_decision(
            interrupt_id, "approve", req, services
        )

    @app.post("/hitl/{interrupt_id}/reject")
    async def reject_hitl(
        interrupt_id: str,
        req: HitlDecisionRequest,
    ) -> JSONResponse:
        # Override client-supplied operator_id with the verified identity
        # → audit log records the actual approver, not whoever the client claims
        req.operator_id = (await _identity()).operator_id
        return await _submit_hitl_decision(
            interrupt_id, "reject", req, services
        )

    @app.post("/hitl/{interrupt_id}/edit")
    async def edit_hitl(
        interrupt_id: str,
        req: HitlDecisionRequest,
    ) -> JSONResponse:
        # Override client-supplied operator_id with the verified identity
        # → audit log records the actual approver, not whoever the client claims
        req.operator_id = (await _identity()).operator_id
        return await _submit_hitl_decision(
            interrupt_id, "edit", req, services
        )

    @app.post("/hitl/{interrupt_id}/choose")
    async def choose_hitl(
        interrupt_id: str,
        req: HitlDecisionRequest,
    ) -> JSONResponse:
        """Operator picked one of the offered choices. The selection id
        comes in via req.selected_choice_id (set by the frontend)."""
        req.operator_id = (await _identity()).operator_id
        return await _submit_hitl_decision(
            interrupt_id, "choose", req, services
        )

    @app.get("/hitl/{interrupt_id}/stream")
    async def hitl_stream(interrupt_id: str) -> StreamingResponse:
        """Live SSE stream of chunks emitted by the HITL resumer.

        Frontend usage:
          1. POST /hitl/<id>/choose (or approve/etc) — returns final result
          2. IN PARALLEL: open EventSource on /hitl/<id>/stream to see
             progressive chunks (turn-by-turn LLM responses, tool calls,
             nested HITL interrupts) as they happen.

        Stream emits the same `data: {...}` JSON line format as
        /webui/chat/stream, so the frontend can reuse its existing
        chunk-rendering pipeline.
        """
        from hitl_core.chunk_queue import get_chunk_queue_registry
        chunk_queue = get_chunk_queue_registry()

        async def _generator():
            try:
                async for chunk in chunk_queue.subscribe(interrupt_id):
                    yield f"data: {json.dumps(chunk, default=str)}\n\n"
                # Signal end-of-stream
                yield f"data: {json.dumps({'type': 'done', 'interrupt_id': interrupt_id})}\n\n"
            except Exception as exc:
                logger.warning("hitl_stream[%s] generator error: %s", interrupt_id, exc)
                yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
            finally:
                # Don't close the registry entry here — the /choose response
                # handler may still read history. Periodic gc handles cleanup.
                pass

        return StreamingResponse(
            _generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",   # disable nginx proxy buffering
            },
        )

    @app.post("/hitl/{interrupt_id}/answer")
    async def answer_hitl(
        interrupt_id: str,
        req: HitlDecisionRequest,
    ) -> JSONResponse:
        """Operator answered the agent's clarification questions. The
        answers dict comes in via req.clarification_answers."""
        req.operator_id = (await _identity()).operator_id
        return await _submit_hitl_decision(
            interrupt_id, "answer", req, services
        )

    # ==================================================================
    # Session management endpoints  (persistent via FTS5 store)
    # ==================================================================

    @app.get("/sessions")
    async def list_sessions_endpoint(limit: int = 50,
    ) -> JSONResponse:
        """
        List all conversation sessions ordered by most recent activity.
        Reads from MemoryAdapter → agent_memory.long_term_chunks (SQLite).
        Survives server restarts.
        """
        memory = services.get("memory")
        if not memory or not hasattr(memory, "list_sessions"):
            return JSONResponse(content=[])
        try:
            # Bind operator before reading — memory is per-user-isolated
            set_current_operator((await _identity()).operator_id)
            sessions = await memory.list_sessions(limit=limit)
            return JSONResponse(content=sessions)
        except Exception as exc:
            logger.warning("/sessions failed: %s", exc)
            return JSONResponse(content=[])

    @app.get("/sessions/{session_id}/history")
    async def get_session_history(session_id: str, limit: int = 100) -> JSONResponse:
        """
        Retrieve full turn history for a session from MemoryAdapter long-term store.
        Returns chunks as {role, content, ts} pairs for the frontend chat panel.
        """
        memory = services.get("memory")
        if not memory or not hasattr(memory, "get_session_history"):
            return JSONResponse(content=_message_history.get(session_id, []))
        try:
            # Bind operator before reading — memory is per-user-isolated
            set_current_operator((await _identity()).operator_id)
            chunks = await memory.get_session_history(session_id)
            messages = []
            for c in chunks[-limit:]:
                text = c.get("text", "")
                ts   = c.get("created_at", 0)
                # Split "User: ...\nAssistant: ..." back into two messages
                if "User:" in text and "Assistant:" in text:
                    parts    = text.split("Assistant:", 1)
                    user_msg = parts[0].replace("User:", "", 1).strip()
                    asst_msg = parts[1].strip() if len(parts) > 1 else ""
                    if user_msg:
                        messages.append({"role": "user",      "content": user_msg, "ts": ts})
                    if asst_msg:
                        messages.append({"role": "assistant", "content": asst_msg, "ts": ts})
                else:
                    messages.append({"role": "assistant", "content": text, "ts": ts})
            return JSONResponse(content=messages)
        except Exception as exc:
            logger.warning("/sessions/%s/history failed: %s", session_id, exc)
            return JSONResponse(content=_message_history.get(session_id, []))

    @app.post("/sessions")
    async def create_session(request: Request,
    ) -> JSONResponse:
        """
        Create (or re-open) a named session. Returns the session_id.
        Body: {"name": "optional display name"} — name becomes the topic_summary.
        """
        try:
            body = await request.json()
        except Exception:
            body = {}
        name       = body.get("name", "").strip()
        session_id = "sess-" + uuid.uuid4().hex[:12]
        fts = services.get("fts_store")
        if fts and name:
            try:
                await fts.update_session_topic(session_id, name)
            except Exception:
                pass
        return JSONResponse(content={
            "session_id":    session_id,
            "topic_summary": name or session_id,
            "created_at":    time.time(),
        })

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str,
    ) -> JSONResponse:
        """
        Delete a session and all its turns from FTS5 state.db.
        Also removes from in-memory history cache.
        """
        fts = services.get("fts_store")
        if fts:
            try:
                await fts.delete_session(session_id)
            except Exception as exc:
                logger.warning("delete_session FTS5 failed: %s", exc)
        _message_history.pop(session_id, None)
        return JSONResponse(content={"deleted": session_id})

    # ==================================================================
    # Memory / Session endpoints
    # ==================================================================

    @app.get("/memory")
    async def get_memory(session_id: str = "default", limit: int = 10,
    ) -> JSONResponse:
        memory = services.get("memory")
        if not memory:
            return JSONResponse(content=[])
        try:
            set_current_operator((await _identity()).operator_id)
            recalled = await memory.recall_for_session("", session_id)
            return JSONResponse(content=[{"session_id": session_id, "recalled": recalled[:2000]}])
        except Exception as exc:
            return JSONResponse(content={"error": str(exc)}, status_code=500)

    @app.get("/session")
    async def get_session(session_id: str = "default") -> JSONResponse:
        """Return current session state including confirmed facts and working set."""
        task_sys = services.get("task_system")
        if not task_sys:
            return JSONResponse(content={"session_id": session_id, "facts": [], "working_set": []})
        try:
            session = await task_sys.session_mgr.get_or_create(context_id=session_id)
            return JSONResponse(content={
                "session_id":     session_id,
                "turn_count":     session.turn_count,
                "confirmed_facts": session.multi_round.confirmed_facts,
                "pending_hitl":   session.multi_round.pending_hitl_ids,
                "open_questions": session.multi_round.open_questions,
            })
        except Exception as exc:
            return JSONResponse(content={"error": str(exc)}, status_code=500)

    # ==================================================================
    # System status
    # ==================================================================

    @app.get("/system/status")
    async def system_status(
    ) -> JSONResponse:
        store    = services.get("tool_store")
        catalog  = services.get("skill_catalog")
        registry = services.get("registry")
        agents   = await registry.list_agents() if registry else []
        router   = services.get("tool_router")

        return JSONResponse(content={
            "runtime_loop":    "ready",
            "tool_registry":   list(tool_registry.keys()),
            "tools_cached":    store.stored_count if store else 0,
            "skills_loaded":   catalog.skill_count if catalog else 0,
            "registry_agents": len(agents),
            "memory":          "ready" if services.get("memory") else "stub",
            "hitl":            "ready" if services.get("hitl_router") else "stub",
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
        Raw dump of _payload_store — use this to diagnose HITL tab issues.
        Call GET /webui/hitl/debug after triggering a HITL interrupt.
        """
        hitl_router = services.get("hitl_router")
        if not hitl_router:
            return JSONResponse(content={"error": "hitl_router not in services"})
        store = hitl_router._payload_store
        items = []
        for iid, p in store.items():
            status_val = p.status.value if hasattr(p.status, "value") else str(p.status)
            items.append({
                "interrupt_id": iid,
                "status":       status_val,
                "trigger_kind": getattr(p.trigger_kind, "value", str(getattr(p, "trigger_kind", ""))),
                "risk_level":   getattr(p.risk_level, "value", str(getattr(p, "risk_level", ""))),
                "user_query":   getattr(p, "user_query", "")[:80],
            })
        return JSONResponse(content={
            "store_size":   len(store),
            "interrupts":   items,
            "router_id":    id(hitl_router),
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
                "wired":         services.get("executor") is not None,
                "hitl_router":   services.get("hitl_router") is not None,
                "tool_router":   services.get("tool_router") is not None,
                "skill_catalog": services.get("skill_catalog") is not None,
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

    # ==================================================================
    # WebSocket — live events + HITL decisions
    # ==================================================================

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_text(json.dumps({
            "type":    "connected",
            "message": "IT Ops Agent WebUI WebSocket ready",
        }))
        try:
            while True:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                msg = json.loads(raw)
                msg_type = msg.get("type", "")

                if msg_type == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))

                elif msg_type == "chat":
                    query      = msg.get("query", "")
                    session_id = msg.get("session_id", str(uuid.uuid4()))
                    loop       = services["runtime_loop"]
                    async for chunk in loop.stream(
                        query=query, session_id=session_id,
                        tool_registry=tool_registry,
                    ):
                        await websocket.send_text(json.dumps({
                            "type": "chunk", "data": chunk,
                        }))
                    await websocket.send_text(json.dumps({"type": "done"}))

                elif msg_type == "hitl_decision":
                    result = await _submit_hitl_decision(
                        msg["interrupt_id"],
                        msg["decision"],
                        HitlDecisionRequest(
                            operator_id=msg.get("operator_id", "ws-operator"),
                            comment=msg.get("comment"),
                            parameter_patch=msg.get("parameter_patch"),
                        ),
                        services,
                    )
                    await websocket.send_text(json.dumps({
                        "type":    "hitl_ack",
                        "result":  result.body.decode() if hasattr(result, "body") else "{}",
                    }))

        except asyncio.TimeoutError:
            await websocket.send_text(json.dumps({"type": "ping"}))
        except WebSocketDisconnect:
            logger.info("WebUI WebSocket disconnected")
        except Exception as exc:
            logger.exception("WebUI WebSocket error: %s", exc)


    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _submit_hitl_decision(
    interrupt_id: str,
    decision_kind: str,
    req: HitlDecisionRequest,
    services: dict,
) -> JSONResponse:
    # CRITICAL: bind operator at the TOP so that the post-HITL callback
    # (whichever backend is active) runs inside the same operator's
    # memory namespace as the original chat_stream turn.
    set_current_operator((await _identity()).operator_id)

    # ── Core backend path ────────────────────────────────────────────
    hitl_core_router = services.get("hitl_core_router")
    if hitl_core_router is not None:
        from hitl_core import (
            DecisionKind, DecisionValidationError, HitlDecision, ResumeError,
        )
        # Map legacy decision_kind strings to hitl_core DecisionKind enum
        kind_map = {
            "approve":  DecisionKind.APPROVE,
            "reject":   DecisionKind.REJECT,
            "edit":     DecisionKind.EDIT,
            "choose":   DecisionKind.CHOOSE,
            "answer":   DecisionKind.ANSWER,
            "escalate": DecisionKind.ESCALATE,
        }
        if decision_kind not in kind_map:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown decision_kind: {decision_kind!r}",
            )
        decision = HitlDecision(
            interrupt_id=interrupt_id,
            decision=kind_map[decision_kind],
            operator_id=req.operator_id,
            comment=req.comment,
            parameter_patch=req.parameter_patch,
            selected_choice_id=req.selected_choice_id,
            clarification_answers=req.clarification_answers,
        )
        try:
            outcome = await hitl_core_router.deliver(decision)
        except DecisionValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except ResumeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

        # Map hitl_core outcome shape to legacy result_dict shape so the
        # frontend's existing rendering code keeps working.
        # `outcome["result"]` is whatever the resumer returned. The
        # tool_call_resumer returns a dict with .result/.tool_result
        # keys (string-typed); the agent_loop_resumer returns a dict
        # with .text. Plain strings or None also possible. Normalise
        # to a string here so the frontend doesn't get "[object Object]"
        # nor crash on `.slice` calls.
        _raw_result = outcome.get("result")
        logger.info(
            "_submit_hitl_decision: outcome shape — outcome_keys=%s, "
            "_raw_result_type=%s, _raw_result_keys=%s",
            list(outcome.keys()),
            type(_raw_result).__name__,
            (list(_raw_result.keys())[:10] if isinstance(_raw_result, dict)
             else None),
        )
        if isinstance(_raw_result, dict):
            _tool_result_str = (
                _raw_result.get("tool_result")
                or _raw_result.get("result")
                or _raw_result.get("text")
                or _raw_result.get("error")
                or ""
            )
        elif _raw_result is None:
            _tool_result_str = ""
        else:
            _tool_result_str = str(_raw_result)

        result_dict = {
            "interrupt_id":     outcome.get("interrupt_id"),
            "decision":         outcome.get("outcome"),
            "already_resolved": outcome.get("already_resolved", False),
            "tool_result":      _tool_result_str,
            # tool_name will be filled in by the synthesis block below
            # by introspecting the resume_handle if needed.
            "tool_name":        "",
        }
        # The resumer can return additional thinking-trace chunks (sub-loop
        # steps that ran AFTER operator approval — TURN N, Tool Call,
        # nested HITL, etc). Surface them so the UI can replay them into
        # the thinking trace; without this, the trace dies at "HITL-WAIT"
        # and the user sees nothing of what happened post-approval.
        if isinstance(_raw_result, dict):
            _sub = _raw_result.get("chunks")
            if isinstance(_sub, list) and _sub:
                result_dict["sub_chunks"] = _sub
                logger.info(
                    "_submit_hitl_decision: forwarding %d sub_chunks to UI "
                    "(first kinds: %s)",
                    len(_sub),
                    [c.get("node_step") or c.get("type") or list(c.keys())[:2] for c in _sub[:5]],
                )
            else:
                logger.info(
                    "_submit_hitl_decision: NO sub_chunks in resumer outcome "
                    "(_raw_result keys=%s)", list(_raw_result.keys()),
                )
            # If the sub-stream raised a NESTED HITL (e.g. user_choice
            # resolved → loop ran → emitted another stop_hitl for tool
            # watch-list), surface that so the UI re-pivots to HITL tab.
            if _raw_result.get("interrupted"):
                result_dict["nested_hitl"] = {
                    "interrupted":  True,
                    "interrupt_id": _raw_result.get("interrupt_id"),
                }
                logger.info(
                    "_submit_hitl_decision: nested HITL detected — "
                    "interrupt_id=%s", _raw_result.get("interrupt_id"),
                )
        # Look up the resolved entry to recover tool_name + the original
        # user_query for synthesis. We fetch from the store rather than
        # holding a reference because outcome doesn't carry it.
        try:
            entry = await hitl_core_router.load(interrupt_id)
            if entry is not None:
                result_dict["tool_name"] = entry.resume_handle.state.get(
                    "tool_name", "agent_loop",
                )
                _user_query_local = entry.payload.user_query or ""
            else:
                _user_query_local = ""
        except Exception as exc:
            logger.debug("post-deliver entry lookup failed: %s", exc)
            _user_query_local = ""

        # Synthesis is the same for both backends; share that block by
        # falling through to the existing logic below. We mock up
        # `payload` to satisfy the legacy variable references. The shim
        # populates every attribute the post-synthesis code (memory
        # writeback, audit, etc.) reads off the legacy pydantic payload.
        _context_id_local = ""
        _thread_id_local  = ""
        try:
            if entry is not None:
                _context_id_local = entry.payload.context_id or ""
                _thread_id_local  = entry.payload.thread_id or ""
        except Exception:
            pass

        class _PayloadShim:
            user_query   = _user_query_local
            context_id   = _context_id_local
            thread_id    = _thread_id_local
            interrupt_id_str = interrupt_id  # avoid name shadowing in inner class
        payload = _PayloadShim()
        # Re-bind locals the legacy synthesis block uses
        _decision    = result_dict["decision"]
        _tool_result = result_dict.get("tool_result", "")
        _tool_name   = result_dict.get("tool_name", "agent_loop")
        _user_query  = _user_query_local
        # Continue to the LLM synthesis section (line below).
    else:
        # ── Legacy backend path ─────────────────────────────────────
        hitl_router = services.get("hitl_router")
        if not hitl_router:
            raise HTTPException(status_code=503, detail="HITL router not available")

        payload = hitl_router._payload_store.get(interrupt_id)
        if payload is None:
            raise HTTPException(status_code=404, detail=f"Interrupt {interrupt_id!r} not found")

        from hitl.schemas import HitlDecision
        decision = HitlDecision(
            interrupt_id=interrupt_id,
            thread_id=payload.thread_id,
            decision=decision_kind,
            operator_id=req.operator_id,
            comment=req.comment,
            parameter_patch=req.parameter_patch,
            selected_choice_id=req.selected_choice_id,
            clarification_answers=req.clarification_answers,
        )
        result      = await hitl_router.handle_decision(decision)
        result_dict = result.to_dict()

        _decision    = result_dict.get("decision")
        _tool_result = result_dict.get("tool_result", "")
        _tool_name   = result_dict.get("tool_name", "the approved action")
        _user_query  = payload.user_query or ""

    # Post-HITL synthesis: run one LLM call to summarise the tool result so
    # the chat shows a meaningful response after the operator approves.
    # SKIP synthesis when tool_name == "agent_loop": the post-HITL agent
    # loop already produced a complete user-facing markdown answer; running
    # synthesis on top of it would just be a degraded summary-of-summary
    # that wastes tokens and loses information.
    _llm = services.get("llm_engine")
    if (
        _llm and _tool_result and _decision == "approve"
        and _tool_name != "agent_loop"
    ):
        try:
            _prompt = (
                f"The operator just approved this request:\n"
                f"  {_user_query}\n\n"
                f"The tool '{_tool_name}' executed and returned:\n"
                f"-----\n{str(_tool_result)[:4000]}\n-----\n\n"
                f"Write a clear answer to the operator in the SAME LANGUAGE as the original request. "
                f"Use markdown: a brief verdict, key findings as bullets, and a short next-step "
                f"recommendation. Do not invent data — only summarise what the tool actually returned."
            )
            messages = [{"role": "user", "content": _prompt}]
            if hasattr(_llm, "_chat"):
                synth = await _llm._chat(messages)
            else:
                synth = await _llm.call(_prompt, "", state=None)
            if synth and synth.strip():
                result_dict["synthesis"] = synth.strip()
        except Exception as _e:
            logger.warning("Post-HITL synthesis failed: %s", _e)
    elif _tool_name == "agent_loop" and _tool_result:
        # The agent loop output IS the user-facing answer — promote it to
        # `synthesis` so the frontend renders it once (not twice).
        result_dict["synthesis"] = str(_tool_result).strip()

    # Memory write + curation for every post-HITL turn (approve / reject /
    # escalate are all auditable operations facts worth persisting).
    # Frontend reads memory_write / memory_curate flags to render FTS5 Write
    # and Memory Curation steps.
    _memory = services.get("memory")
    logger.info(
        "_submit_hitl_decision: memory write check — _memory=%s _decision=%r "
        "tool_name=%r user_query=%r synthesis_len=%d tool_result_len=%d",
        bool(_memory), _decision, _tool_name,
        (_user_query or "")[:80],
        len(result_dict.get("synthesis") or ""),
        len(str(_tool_result) if _tool_result else ""),
    )
    if _memory and _decision in ("approve", "reject", "escalate", "choose", "answer"):
        try:
            # Build assistant_text. For approves with tool output use the
            # synthesised answer (or raw tool output if synthesis was skipped).
            # For rejects / escalates write a concise audit-style message so
            # the conversation history reflects the operator's decision.
            #
            # Prefix approves with an explicit "[HITL APPROVED — completed]"
            # tag so future recall makes it unambiguous that this turn FINISHED
            # successfully. Without the tag, an LLM looking at recall context
            # can confuse a synthesis answer with a pending interrupt note
            # (both contain the same user query) and report the action as
            # still awaiting approval — see screenshot bug report.
            if _decision in ("approve", "choose", "answer"):
                _body = (
                    result_dict.get("synthesis")
                    or (str(_tool_result)[:2000] if _tool_result else "")
                    or f"[HITL approved — {_tool_name}]"
                )
                # Detect whether the body actually contains completion markers.
                # If not, label this turn so future recall surfaces it as
                # "executed but result inconclusive" rather than as a
                # successful change. Without this discriminator, Path A's
                # plan-only output (e.g. "我将修复 ap-01... 第一步: ...")
                # would get the same [APPROVED & COMPLETED] tag as a real
                # successful completion, and the LLM next turn can't tell
                # them apart.
                _has_done_marker = any(
                    kw in _body
                    for kw in (
                        "已修复", "已完成", "已应用", "修复成功", "修复完成",
                        "配置已", "已生效", "已优化", "执行完毕",
                        "completed", "applied", "fixed successfully",
                        "has been updated", "configuration updated",
                    )
                )
                _status_tag = (
                    "[HITL APPROVED & COMPLETED"
                    if _has_done_marker
                    else "[HITL APPROVED & EXECUTED — result inconclusive"
                )
                # Add an explicit one-line summary at the very top so the
                # LLM seeing this in recall context can immediately classify
                # the turn without parsing the full body.
                _summary_line = (
                    f"{_status_tag} — {_tool_name}] "
                    f"operator={req.operator_id or 'unknown'} "
                    f"request={(_user_query or '')[:80]!r}"
                )
                assistant_text = f"{_summary_line}\n{_body}"
            else:
                _comment = (req.comment or "").strip()
                assistant_text = (
                    f"[HITL {_decision.upper()} — {_tool_name}] "
                    f"operator={req.operator_id or 'unknown'}"
                    + (f" — {_comment}" if _comment else "")
                )
            # operator is already bound at the top of _submit_hitl_decision —
            # no need to re-set here.
            # Reuse the original session_id from the interrupt payload for continuity.
            session_id = payload.context_id or f"hitl__{interrupt_id[:8]}"
            logger.info(
                "_submit_hitl_decision: writing memory — session_id=%r "
                "operator=%s assistant_text_len=%d preview=%r",
                session_id, get_current_operator(),
                len(assistant_text), assistant_text[:120],
            )
            new_facts = await _memory.after_turn(
                session_id      = session_id,
                user_text       = _user_query,
                assistant_text  = assistant_text,
                tool_calls      = [{"tool": _tool_name}] if _tool_name else [],
                importance      = 0.7 if _decision in ("approve", "choose", "answer") else 0.5,
            )
            result_dict["memory_write"] = {"session_id": session_id, "ok": True}
            logger.info(
                "_submit_hitl_decision: memory write OK — %d new facts",
                len(new_facts) if new_facts else 0,
            )
            if new_facts:
                _types = [getattr(f, "fact_type", getattr(f, "memory_type", "fact"))
                          for f in new_facts[:5]]
                result_dict["memory_curate"] = {
                    "memories_count": len(new_facts),
                    "types":          _types,
                }
        except Exception as _e:
            # Log the FULL traceback so we can see where it actually fails.
            # The previous `logger.warning("Post-HITL memory write failed: %s")`
            # was eaten silently in many cases.
            logger.error(
                "_submit_hitl_decision: memory write FAILED — %s",
                _e, exc_info=True,
            )
    else:
        logger.warning(
            "_submit_hitl_decision: SKIPPING memory write — _memory=%s "
            "_decision=%r (not in approve/reject/escalate)",
            bool(_memory), _decision,
        )

    # ── Skill evolution after approve ────────────────────────────────
    # If the operator approved a destructive action and a tool actually
    # executed, ask SkillEvolver whether the request+solution forms a
    # reusable pattern worth canonicalizing. Surfaces in thinking trace
    # via result_dict["skill_evolved"].
    if _decision in ("approve", "choose", "answer") and _tool_name and _tool_name != "agent_loop":
        _evolver = services.get("skill_evolver")
        if _evolver is not None:
            try:
                proposal = await _evolver.after_task(
                    task_description = _user_query,
                    solution_summary = (result_dict.get("synthesis") or "")[:400],
                    tools_used       = [_tool_name],
                    solution_steps   = [],
                    key_observations = [],
                    complexity       = 7.0,
                    session_id       = session_id,
                )
                if proposal is not None:
                    result_dict["skill_evolved"] = {
                        "skill_id":   proposal.skill_id,
                        "name":       getattr(proposal, "name", proposal.skill_id),
                        "registered": True,
                    }
                    logger.info(
                        "_submit_hitl_decision: skill evolved — id=%s",
                        proposal.skill_id,
                    )
            except Exception as _e:
                logger.debug("Post-HITL skill evolver skipped: %s", _e)

    # Diagnostic: log exactly what we send back to the frontend so we can
    # tell whether the issue is server-side (we never sent it) or client-side
    # (frontend received it but didn't render).
    _synth_len = len(result_dict.get("synthesis") or "")
    _tres_len  = len(str(result_dict.get("tool_result") or ""))
    logger.info(
        "_submit_hitl_decision: returning to frontend — interrupt=%s "
        "decision=%s tool_name=%r synthesis=%d chars tool_result=%d chars "
        "memory_write=%s memory_curate=%s",
        interrupt_id[:12], _decision, _tool_name, _synth_len, _tres_len,
        bool(result_dict.get("memory_write")),
        bool(result_dict.get("memory_curate")),
    )
    return JSONResponse(content=result_dict)


def _push_history(session_id: str, msg: dict, store: dict) -> None:
    if session_id not in store:
        store[session_id] = []
    store[session_id].append(msg)
    # Keep last 100 messages
    if len(store[session_id]) > 100:
        store[session_id] = store[session_id][-100:]


def _parse_working_set(raw: list[dict]) -> list:
    from runtime import DeviceRef
    return [
        DeviceRef(id=d["id"], label=d.get("label", d["id"]))
        for d in raw
        if isinstance(d, dict) and "id" in d
    ]


# ---------------------------------------------------------------------------
# Standalone dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    # Minimal services for standalone run
    _dev_services: dict[str, Any] = {}
    _app = create_webui_app(_dev_services)

    uvicorn.run(_app, host="0.0.0.0", port=8001, reload=False)