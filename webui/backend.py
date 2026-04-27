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
from memory import set_current_operator

logger = logging.getLogger(__name__)

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
        _tl = _TL(mode=_cfg.cfg.mode)
        catalog = SkillCatalogService()
        catalog.register_all(_tl.skill_definitions())
        services["skill_catalog"] = catalog

    # Inject skill_evolver for upload/persist capability if not already provided by main.py
    if "skill_evolver" not in services:
        import os as _os, pathlib as _pl
        from skills.evolver import SkillEvolver
        _skills_dir = _os.getenv("HERMES_DATA_DIR", "./data")
        services["skill_evolver"] = SkillEvolver(
            catalog=services["skill_catalog"],
            skills_dir=str(_pl.Path(_skills_dir) / "skills"),
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
        _hitl_tools = frozenset(
            t.strip() for t in _os.getenv("HITL_TOOL_NAMES", "").split(",") if t.strip()
        )
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
            try:
                # ── Step 1: Classify ──────────────────────────────────────
                decision = await loop.classify_async(req.query)
                yield f"data: {json.dumps({'type':'classify','complexity':decision.complexity.value,'tier':decision.model_tier,'reason':decision.reason[:100]})}\n\n"
                await asyncio.sleep(0)

                # ── Step 2: Pre-verify ────────────────────────────────────
                pre = await loop.pre_verify(req.query, req.confirmed_facts, req.env_context)
                yield f"data: {json.dumps({'type':'pre_verify','passed':pre.passed,'reason':pre.reason[:150]})}\n\n"
                await asyncio.sleep(0)

                # ── Step 3: Cross-session recall (DTM v4 or FTS5 v3) ─────
                recall_text = ""
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

                # ── Step 4: Execute via loop (all queries) ────────────────
                # Uses the real patched LLM + real ToolRouter registry
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

                # COMPLEX queries: route through HITL graph if executor is available
                if decision.complexity.value == "complex" and executor is not None:
                    yield f"data: {json.dumps({'type':'routing','path':'hitl_executor','reason':'complex query routed to HITL graph'})}\n\n"
                    await asyncio.sleep(0)

                    from a2a.event_queue import EventQueue, RequestContext
                    from a2a.schemas import Message, TextPart

                    eq  = EventQueue()
                    ctx = RequestContext(
                        task_id=task_id,
                        context_id=context_id,
                        message=Message(role="user", parts=[TextPart(text=req.query)]),
                        metadata={
                            "session_id":      session_id,
                            "env_context":     env_ctx,
                            "confirmed_facts": list(req.confirmed_facts or []),
                            "working_set":     list(req.working_set or []),
                        },
                    )

                    # Run executor in background, stream events as SSE
                    exec_task = asyncio.create_task(executor.execute(ctx, eq))

                    async for event in eq.consume():
                        kind = event.kind  # camelCase: taskStatusUpdate, taskArtifactUpdate, message

                        if kind == "taskStatusUpdate":
                            state_val = event.status.state.value if event.status else "unknown"
                            yield f"data: {json.dumps({'type':'task_status','state':state_val,'task_id':task_id})}\n\n"
                            await asyncio.sleep(0)

                        elif kind == "message":
                            # Terminal event — extract text and emit as token stream
                            for part in (event.message.parts if event.message else []):
                                if hasattr(part, "text") and part.text:
                                    tokens.append(part.text)
                                    # Stream the text in 80-char chunks, PRESERVING newlines and
                                    # all whitespace exactly. The frontend renders markdown after
                                    # the stream completes, so block structure (\n\n, table rows,
                                    # ```fences, ## headers) must survive transmission.
                                    _txt = part.text
                                    for _i in range(0, len(_txt), 80):
                                        yield f"data: {json.dumps({'token': _txt[_i:_i+80]})}\n\n"
                                    await asyncio.sleep(0)

                        elif kind == "taskArtifactUpdate":
                            if event.artifact:
                                for part in event.artifact.parts:
                                    if hasattr(part, "data") and isinstance(part.data, dict):
                                        data = part.data
                                        art_kind = data.get("tag") or data.get("kind") or ""
                                        if art_kind == "hitl_interrupt":
                                            # Real HITL interrupt — emit and switch console to HITL tab
                                            yield f"data: {json.dumps({'type':'hitl_interrupt','hitl_interrupt':True,**data})}\n\n"
                                            await asyncio.sleep(0)
                                            # Mark stream terminal state so terminal
                                            # event becomes 'awaiting_hitl' (⏸) not 'done' (✅).
                                            stop_outcome = "stop_hitl: awaiting operator approval"
                                        elif data.get("node_step"):
                                            yield f"data: {json.dumps({'node_step':data['node_step'],'node':data.get('node','')})}\n\n"
                                            await asyncio.sleep(0)
                                        elif data.get("node_result"):
                                            yield f"data: {json.dumps({'node_result':data['node_result']})}\n\n"
                                            await asyncio.sleep(0)
                                        else:
                                            yield f"data: {json.dumps({'type':'artifact','data':data})}\n\n"
                                            await asyncio.sleep(0)

                    try:
                        # Wait for exec_task to finish (it completes once MessageEvent sent)
                        await asyncio.wait_for(exec_task, timeout=120.0)
                    except asyncio.TimeoutError:
                        logger.warning("Executor task timed out after 120s — likely HITL pending")
                    except Exception as exc:
                        logger.debug("Executor task ended: %s", exc)
                    full_text = "".join(tokens)

                else:
                    # SIMPLE path: loop.stream() with real LLM + ToolRouter
                    _hitl_intercepted = False
                    async for chunk in loop.stream(
                        query=req.query,
                        session_id=session_id,
                        env_context=env_ctx,
                        confirmed_facts=list(req.confirmed_facts or []),
                        working_set=_parse_working_set(list(req.working_set or [])),
                        tool_registry=real_registry,
                        delegation_mode=dm,
                    ):
                        if "token" in chunk:
                            tokens.append(chunk["token"])
                        if isinstance(chunk.get("node_step"), str) and chunk["node_step"].startswith("Turn "):
                            turns_taken += 1
                        if chunk.get("message"):
                            stop_outcome = chunk["message"][:60]

                        # HITL gate: skill-ambiguity or tool-watchlist triggered from SIMPLE path
                        # Re-route to executor so HITL graph fires and approval card appears
                        if chunk.get("stop_hitl") and executor is not None:
                            yield f"data: {json.dumps(chunk)}\n\n"
                            await asyncio.sleep(0)
                            yield f"data: {json.dumps({'type':'routing','path':'hitl_executor','reason':chunk.get('message','HITL gate triggered')[:80]})}\n\n"
                            await asyncio.sleep(0)
                            from a2a.event_queue import EventQueue, RequestContext
                            from a2a.schemas import Message, TextPart
                            eq  = EventQueue()
                            # Pass the tool name and args that triggered HITL so
                            # the graph can force the interrupt and replay after approval.
                            _hitl_tool = chunk.get("tool_name", "")
                            _hitl_args = chunk.get("tool_args", {})
                            ctx = RequestContext(
                                task_id=task_id,
                                context_id=context_id,
                                message=Message(role="user", parts=[TextPart(text=req.query)]),
                                metadata={
                                    "session_id":      session_id,
                                    "env_context":     env_ctx,
                                    "confirmed_facts": list(req.confirmed_facts or []),
                                    "working_set":     list(req.working_set or []),
                                    "force_hitl_tool": _hitl_tool,   # bypass LLM trigger eval
                                    "force_hitl_args": _hitl_args,   # replay args after approval
                                    "action_type":     f"tool_call:{_hitl_tool}",
                                },
                            )
                            exec_task = asyncio.create_task(executor.execute(ctx, eq))
                            async for event in eq.consume():
                                kind = event.kind
                                if kind == "taskStatusUpdate":
                                    state_val = event.status.state.value if event.status else "unknown"
                                    yield f"data: {json.dumps({'type':'task_status','state':state_val,'task_id':task_id})}\n\n"
                                    await asyncio.sleep(0)
                                elif kind == "message":
                                    for part in (event.message.parts if event.message else []):
                                        if hasattr(part, "text") and part.text:
                                            tokens.append(part.text)
                                            _txt = part.text
                                            for _i in range(0, len(_txt), 80):
                                                yield f"data: {json.dumps({'token': _txt[_i:_i+80]})}\n\n"
                                            await asyncio.sleep(0)
                                elif kind == "taskArtifactUpdate":
                                    if event.artifact:
                                        for part in event.artifact.parts:
                                            if hasattr(part, "data") and isinstance(part.data, dict):
                                                data = part.data
                                                if (data.get("tag") or data.get("kind")) == "hitl_interrupt":
                                                    yield f"data: {json.dumps({'type':'hitl_interrupt','hitl_interrupt':True,**data})}\n\n"
                                                    await asyncio.sleep(0)
                            try:
                                await asyncio.wait_for(exec_task, timeout=120.0)
                            except (asyncio.TimeoutError, Exception):
                                pass
                            _hitl_intercepted = True
                            # Mark this stream's terminal state so the outer code
                            # emits 'awaiting_hitl' (⏸) instead of 'done' (✅).
                            stop_outcome = "stop_hitl: awaiting operator approval"
                            break

                        yield f"data: {json.dumps(chunk)}\n\n"
                        await asyncio.sleep(0)
                    full_text = "".join(tokens)

                _push_history(session_id, {"role": "user",      "content": req.query},   _message_history)
                _push_history(session_id, {"role": "assistant", "content": full_text},    _message_history)

                # ── Step 5: Post-turn Hermes hooks ────────────────────────
                import re as _re
                tc = [{"tool": m} for m in _re.findall(r"\[TOOL:(\w+)\]", full_text)]

                if dtm:
                    # v4 path: DTM.after_turn() handles Track A (FTS5 + daily .md
                    # compaction) and Track B (curator → facts.jsonl) in one call.
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
                            # MemoryFact uses .fact_type (not .memory_type)
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
                        from integrations.tool_router import ToolMeta  # noqa
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
        hitl_router = services.get("hitl_router")
        if not hitl_router:
            logger.warning("/hitl/pending: hitl_router not in services")
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
                    }
                result.append(dumped)
        logger.info("/hitl/pending: returning %d pending interrupts", len(result))
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
        Check this first when diagnosing why LLM / HITL / Hermes don't work.
        """
        backend  = services.get("_llm_backend", "unknown")
        model    = services.get("_llm_model",   "unknown")
        has_real_llm = backend not in ("mock", "unknown")
        return JSONResponse(content={
            "llm": {
                "backend":      backend,
                "model":        model,
                "real":         has_real_llm,
                "note": "Set LLM_BACKEND=ollama LLM_MODEL=qwen3.5:27b LLM_BASE_URL=http://localhost:11434" if not has_real_llm else "real LLM active",
            },
            "hermes": {
                "fts_store":      services.get("fts_store") is not None,
                "memory_curator": services.get("memory_curator") is not None,
                "user_model":     services.get("user_model") is not None,
                "skill_evolver":  services.get("skill_evolver") is not None,
                "dtm":            services.get("dtm") is not None,
                "db_path":        services.get("_hermes_data", "not initialised"),
                "dtm_stats":      services.get("dtm").stats() if services.get("dtm") else {},
            },
            "executor": {
                "wired":       services.get("executor") is not None,
                "hitl_router": services.get("hitl_router") is not None,
                "tool_router": services.get("tool_router") is not None,
                "skill_catalog": services.get("skill_catalog") is not None,
            },
            "startup_env": {
                "LLM_BACKEND":   backend,
                "LLM_MODEL":     model,
                "MCP_USE_MOCK":  str(services.get("_mcp_mock", True)),
                "HERMES_DATA_DIR": services.get("_hermes_data", "./data/state.db"),
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
    )
    result     = await hitl_router.handle_decision(decision)
    result_dict = result.to_dict()

    # Post-HITL synthesis: run one LLM call to summarise the tool result so
    # the chat shows a meaningful response after the operator approves.
    # Uses llm_engine._chat directly (no full agent loop needed).
    _llm = services.get("llm_engine")
    _tool_result = result_dict.get("tool_result", "")
    _tool_name   = result_dict.get("tool_name", "the approved action")
    if _llm and _tool_result and result_dict.get("decision") == "approve":
        try:
            user_query = ""
            payload2 = hitl_router._payload_store.get(interrupt_id)
            if payload2:
                user_query = payload2.user_query or ""
            _prompt = (
                f"The operator just approved this request:\n"
                f"  {user_query}\n\n"
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