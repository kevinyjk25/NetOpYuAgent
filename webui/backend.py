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

P0 demo flow (shown in the UI)
--------------------------------
  1. POST /webui/chat  {"query": "search syslogs for errors on ap-01"}
  2. Runtime Loop calls syslog_search → 300 lines (~6000 chars)
  3. ToolResultStore stores it → returns [STORED:syslog_search:abc12345]
  4. Response shows the ref label + preview
  5. GET /webui/tools/result/abc12345?offset=0  → first 2KB
  6. GET /webui/tools/result/abc12345?offset=2000 → next 2KB
  7. Each page shows "Has more: True/False" and next offset
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
from rate_limit import per_operator_limit, global_concurrency
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


class DemoRunRequest(BaseModel):
    scenario: str
    params:   dict = {}


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
                        # Serialize all DTMResult items for the Memory tab
                        memory_items = [
                            {
                                "track":       r.track,
                                "score":       round(r.score, 3),
                                "source":      r.source,
                                "memory_type": r.memory_type,
                                "content":     r.content[:500],
                                "recency_ts":  r.recency_ts,
                                "tags":        r.tags[:6],
                            }
                            for r in recall_result.results
                        ]
                        yield f"data: {json.dumps({'type':'recall','chars':len(recall_text),'sessions_searched':stats.get('total_sessions',0),'has_context':bool(recall_text),'track_a':recall_result.track_a_count,'track_b':recall_result.track_b_count,'winner':recall_result.winner,'preview':recall_text[:200],'memory_items':memory_items})}\n\n"
                        await asyncio.sleep(0)
                    except Exception as _e:
                        logger.debug("DTM recall skipped: %s", _e)
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
                                    for word in part.text.split():
                                        yield f"data: {json.dumps({'token': word + ' '})}\n\n"
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
                                            for word in part.text.split():
                                                yield f"data: {json.dumps({'token': word + ' '})}\n\n"
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
                            yield f"data: {json.dumps({'type':'hermes_curate','memories_count':len(memories),'types':[m.memory_type.value for m in memories[:5]]})}\n\n"
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
        Uses Request directly (not File()) to work correctly in mounted sub-apps.
        """
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

        # Compile first to catch syntax errors before exec
        try:
            code = compile(source, filename, "exec")
        except SyntaxError as exc:
            raise HTTPException(status_code=400, detail=f"Syntax error: {exc}")

        # Execute in an isolated namespace with standard library pre-imported
        # (exec'd functions need __builtins__ and common stdlib available)
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
        """
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

    # ==================================================================
    # Demo runner endpoint
    # ==================================================================

    @app.post("/demos/run")
    async def run_demo(req: DemoRunRequest) -> JSONResponse:
        """Execute a named demo scenario and return structured results."""
        import re as _re
        loop    = services["runtime_loop"]
        catalog = services["skill_catalog"]
        store   = services["tool_store"]
        sid     = f"demo-{req.scenario}-{uuid.uuid4().hex[:6]}"
        result  = {}

        # P0-A: Large payload cache
        if req.scenario == "p0_large_cache":
            tool_name = req.params.get("tool", "syslog_search")
            tool_args = req.params.get("args", {"host": "ap-*", "keyword": "error", "lines": 300})
            fn  = tool_registry.get(tool_name)
            raw = await fn(tool_args) if fn else "[Tool not found]"
            label = store.store(tool_name, raw)
            is_stored = label != raw
            ref_id = None
            if is_stored:
                m = _re.search(r"\[STORED:[^:]+:([^\]]+)\]", label)
                ref_id = m.group(1) if m else None
            page1 = store.read(ref_id, 0, 2000) if ref_id else None
            result = {
                "tool": tool_name, "args": tool_args,
                "raw_chars": len(raw), "is_stored": is_stored,
                "ref_id": ref_id, "label": label[:200],
                "first_page_chars": len(page1) if page1 else 0,
                "first_page_preview": page1[:300] if page1 else None,
                "total_chars": len(raw),
                "pages_needed": (len(raw) // 2000) + 1 if is_stored else 1,
                "retrieve_url": f"/webui/tools/result/{ref_id}" if ref_id else None,
            }

        # P0-B: Context budget assembly
        elif req.scenario == "p0_context_budget":
            from runtime.context_budget import ContextBudgetManager, BudgetConfig, DeviceRef as DR
            mgr   = ContextBudgetManager(BudgetConfig(total_cap_tokens=800))
            facts = ["device: ap-01, status: healthy", "config: validated OK"]
            ws    = [DR(id="ap-01", label="AP-01 Site-A"), DR(id="sw-core", label="Core Switch")]
            env   = {"site": "Site-A", "change_window": False}
            # Synthetic payloads — no live tool call needed to test budget assembly
            sm  = "payments.internal -> 10.0.1.42 (A record, TTL 300)"
            big = "\n".join(f"2024-01-01 {h:02d}:00 ap-01 ERROR auth fail" for h in range(50))
            sm_ref  = mgr.store_tool_result("get_device_status", sm)
            big_ref = mgr.store_tool_result("get_syslog", big)
            ctx = mgr.assemble(confirmed_facts=facts, working_set=ws,
                               tool_outputs={"get_device_status": sm_ref, "get_syslog": big_ref},
                               env_context=env)
            result = {
                "budget_cap_tokens": 800,
                "dns_chars": len(sm), "dns_inline": "[STORED" not in sm_ref,
                "syslog_chars": len(big), "syslog_stored": "[STORED" in big_ref,
                "assembled_tokens": len(ctx) // 4,
                "context_preview": ctx[:500],
                "explanation": "dns_lookup (small) returned inline; syslog_search (large) stored externally — only a reference label entered the prompt",
            }

        # P0-C: Stop policy trace
        elif req.scenario == "p0_stop_policy":
            from runtime.stop_policy import StopPolicy, StopPolicyConfig, LoopState
            cfg    = StopPolicyConfig(
                max_turns=req.params.get("max_turns", 4),
                max_tool_calls=req.params.get("max_tool_calls", 6),
                token_budget=req.params.get("token_budget", 1200),
                confidence_floor=0.5, low_confidence_turns=2, max_no_progress_turns=2,
            )
            policy = StopPolicy(cfg)
            trace  = []
            state  = LoopState()
            for turn in range(10):
                state.turns         = turn + 1
                state.tool_calls    = (turn + 1) * 2
                state.tokens_consumed = (turn + 1) * 250
                if turn >= 2:
                    state.no_progress_turns = turn - 1
                dec = policy.evaluate(state)
                trace.append({"turn": state.turns, "tool_calls": state.tool_calls,
                               "tokens": state.tokens_consumed, "no_progress": state.no_progress_turns,
                               "outcome": dec.outcome.value, "stopped": dec.should_stop, "reason": dec.reason})
                if dec.should_stop:
                    break
            result = {"config": {"max_turns": cfg.max_turns, "max_tool_calls": cfg.max_tool_calls,
                                 "token_budget": cfg.token_budget}, "trace": trace}

        # P1-A: Skill catalog progressive disclosure
        elif req.scenario == "p1_skill_catalog":
            skill_id = req.params.get("skill_id", "syslog_search")
            summary  = catalog.format_summary()
            detail   = catalog.load_detail(skill_id)
            s        = catalog.get_summary(skill_id)
            result   = {
                "skill_id": skill_id,
                "total_skills": catalog.skill_count,
                "requires_hitl": catalog.requires_hitl(skill_id),
                "risk_level": s.risk_level if s else "unknown",
                "level1_tokens": len(summary) // 4,
                "level2_tokens": len(detail) // 4 if detail else 0,
                "token_saved_pct": round((1 - len(detail or "") / max(len(summary), 1)) * 100, 1) if detail else 0,
                "level1_preview": summary[:400],
                "level2_detail": detail or "(not found)",
                "explanation": "Level 1 injected every turn (~compact). Level 2 loaded only when model requests [SKILL_LOAD:skill_id].",
            }

        # P1-B: Forked delegation
        elif req.scenario == "p1_forked_delegation":
            from runtime.loop import DelegationMode, ForkContextPolicy
            from runtime.stop_policy import LoopState
            from runtime.context_budget import DeviceRef as DR
            parent = LoopState()
            parent.confirmed_facts = [
                "INC-1291: RADIUS timeout on ap-03",
                "DNS confirmed OK — not a DNS issue",
                "40 wireless clients affected at Site-A",
            ]
            setattr(parent, "working_set", [DR(id="ap-03", label="AP-03 Site-A"),
                                             DR(id="radius-01", label="RADIUS Server")])
            fork_fo = loop.build_fork_context(parent, ForkContextPolicy.FACTS_ONLY)
            fork_ws = loop.build_fork_context(parent, ForkContextPolicy.WORKING_SET)
            fork_fl = loop.build_fork_context(parent, ForkContextPolicy.FULL)
            res = await loop.run(query="check RADIUS server health", session_id=sid,
                                 delegation_mode=DelegationMode.FORKED,
                                 parent_state=parent, tool_registry=tool_registry)
            result = {
                "parent_facts": parent.confirmed_facts,
                "parent_working_set": [str(w) for w in getattr(parent, "working_set", [])],
                "policies": {
                    "FACTS_ONLY":   {"keys": list(fork_fo.keys()), "facts_inherited": len(fork_fo.get("confirmed_facts", []))},
                    "WORKING_SET":  {"keys": list(fork_ws.keys()), "facts_inherited": len(fork_ws.get("confirmed_facts", []))},
                    "FULL":         {"keys": list(fork_fl.keys())},
                },
                "forked_query": "check RADIUS server health",
                "response_preview": res.final_response[:300],
                "inherited_facts_in_response": len(res.confirmed_facts),
                "turns": res.turns_taken,
                "explanation": "Forked sub-agent inherited parent facts without re-transmitting full context. Fresh agent would have started from zero.",
            }

        # P1-C: Pre/Post verification
        elif req.scenario == "p1_verification":
            cases = [
                ("safe DNS query",         "check DNS for payments.internal", {}, True),
                ("destructive — no env",   "restart payments-service",        {}, False),
                ("closed change window",   "rollback deployment to v2.1",     {"change_window": False}, False),
                ("open window + allowed",  "check service health",            {"change_window": True, "allow_destructive": True}, True),
            ]
            pre_results = []
            for label, q, env, expected in cases:
                r = await loop.pre_verify(q, [], env)
                pre_results.append({"label": label, "query": q, "env": env,
                                    "passed": r.passed, "expected": expected,
                                    "correct": r.passed == expected, "reason": r.reason})
            post_cases = [
                ("dns_lookup",    "A → 10.0.1.42 (TTL 300)"),
                ("service_health","Status: error — connection refused"),
                ("alert_summary", "P1 (high): 2 active incidents"),
            ]
            post_results = []
            for tool, output in post_cases:
                r = await loop.post_verify(tool, output, [])
                post_results.append({"tool": tool, "output_preview": output,
                                     "passed": r.passed, "warnings": r.warnings})
            result = {"pre_verification": pre_results, "post_verification": post_results,
                      "explanation": "Pre-verify runs before execution; post-verify runs after each tool call to catch error states."}

        # P1-D: Confirmed Facts + Working Set context priority
        elif req.scenario == "p1_working_set":
            from runtime.context_budget import ContextBudgetManager, BudgetConfig, DeviceRef as DR
            facts = ["INC-1291 RADIUS timeout ap-03", "DNS OK", "40 clients affected"]
            ws    = [DR(id="ap-03", label="AP-03 Site-A (focus)"),
                     DR(id="radius-01", label="RADIUS-01 (suspect)"),
                     DR(id="sw-access", label="SW-Access-02 (upstream)")]
            env   = {"site": "Site-A", "change_window": False}
            mgr   = ContextBudgetManager(BudgetConfig(total_cap_tokens=500))
            no_ctx   = mgr.assemble(env_context=env)
            with_ctx = mgr.assemble(confirmed_facts=facts, working_set=ws, env_context=env)
            result = {
                "facts": facts, "working_set": [str(w) for w in ws],
                "baseline_tokens": len(no_ctx) // 4,
                "enriched_tokens": len(with_ctx) // 4,
                "baseline_preview": no_ctx[:150] or "(only env injected)",
                "enriched_preview": with_ctx[:500],
                "priority_order": ["1. Confirmed Facts (highest)", "2. Working Set",
                                   "3. Memory", "4. Tool outputs", "5. Environment"],
                "explanation": "Confirmed Facts and Working Set are injected before memory results. LLM sees the most critical structured knowledge first.",
            }

        # P2-A: Model tier routing classification
        elif req.scenario == "p2_model_tier":
            queries = [
                ("check DNS for site-a",                   "simple / fast_model"),
                ("status of payments-service",             "simple / fast_model"),
                ("search syslogs for errors across all sites", "complex — parallel keyword"),
                ("P1 outage: auth service is down",        "complex — P1 keyword"),
                ("restart the payments-service in prod",   "complex — destructive keyword"),
                ("analyse auth failures and predict trend","complex — parallel/multi-step"),
                ("what is the uptime of ap-01",            "simple / fast_model"),
                ("compare latency across all regions",     "complex — parallel keyword"),
            ]
            rows = []
            for q, expected in queries:
                d = loop.classify(q)
                rows.append({"query": q, "complexity": d.complexity.value,
                             "tier": d.model_tier, "confidence": d.confidence,
                             "reason": d.reason[:60], "expected_label": expected})
            result = {
                "rows": rows,
                "simple": sum(1 for r in rows if r["complexity"] == "simple"),
                "complex": sum(1 for r in rows if r["complexity"] == "complex"),
                "fast_model": sum(1 for r in rows if r["tier"] == "fast_model"),
                "full_model": sum(1 for r in rows if r["tier"] == "full_model"),
                "explanation": "fast_model routes to a lighter model (e.g. haiku). full_model uses the analysis model. Neither is actually called here — this is the routing decision layer.",
            }

        # P2-B: Prompt cache-friendly ordering
        elif req.scenario == "p2_prompt_cache":
            stable   = catalog.format_summary()
            facts    = ["RADIUS timeout on ap-03", "DNS OK"]
            ws_text  = "AP-03 Site-A, RADIUS-01"
            volatile = "site=Site-A change_window=False"
            total    = len(stable) + len(facts[0]) + len(ws_text) + len(volatile)
            result   = {
                "sections": [
                    {"name": "Skill Catalog Summary", "chars": len(stable),
                     "pct": round(len(stable)/total*100,1),
                     "stability": "stable — same every turn (cacheable prefix)",
                     "preview": stable[:200]},
                    {"name": "Confirmed Facts + Working Set", "chars": len(facts[0])+len(ws_text),
                     "pct": round((len(facts[0])+len(ws_text))/total*100,1),
                     "stability": "semi-stable — changes when facts update"},
                    {"name": "Environment Context", "chars": len(volatile),
                     "pct": round(len(volatile)/total*100,1),
                     "stability": "volatile — may change each turn"},
                ],
                "explanation": "Stable prefix (skill catalog) is placed first so the prompt cache hits on repeated turns. Volatile sections at the end don't invalidate the cache.",
            }

        # HITL flow simulation
        elif req.scenario == "hitl_flow":
            q = req.params.get("query", "restart the payments-service in production")
            d = loop.classify(q)
            pre = await loop.pre_verify(q, ["payments P1 active"],
                                        {"change_window": True, "allow_destructive": True})
            result = {
                "query": q,
                "classify": {"complexity": d.complexity.value, "reason": d.reason},
                "pre_verify": {"passed": pre.passed, "reason": pre.reason},
                "path": "HITL Graph" if d.complexity.value == "complex" else "Runtime Loop",
                "trigger_chain": [
                    {"priority": 1, "trigger": "DestructiveActionTrigger",
                     "fires": "restart" in q.lower(), "reason": "restart in destructive_action_types"},
                    {"priority": 2, "trigger": "SeverityTrigger",
                     "fires": False, "reason": "not evaluated — higher priority trigger hit first"},
                ],
                "notifications": ["A2A webhook", "Slack Block Kit", "PagerDuty Events API v2",
                                  "SSE dashboard broadcast", "WebSocket external agents"],
                "decision_options": ["approve", "reject", "edit (patch parameters)", "escalate", "timeout (SLA)"],
                "sla_map": {"critical": "300s", "high": "600s", "medium": "900s", "low": "1800s"},
            }

        # Memory module flow
        elif req.scenario == "memory_flow":
            mem = services.get("memory")
            result = {
                "query": req.params.get("query", "authentication failure analysis"),
                "four_tiers": [
                    {"tier": "L1 REALTIME",   "backend": "in-process list",        "ttl": "request lifetime", "search": "sequential scan"},
                    {"tier": "L2 SHORT_TERM", "backend": "Redis sorted set",       "ttl": "86400s",           "search": "time-ordered ZREVRANGE"},
                    {"tier": "L3 MID_TERM",   "backend": "ChromaDB vector index",  "ttl": "30 days",          "search": "cosine + time decay"},
                    {"tier": "L4 LONG_TERM",  "backend": "PostgreSQL pg_trgm",     "ttl": "permanent",        "search": "trigram similarity"},
                ],
                "ingestion_thresholds": {"L1+L2": "importance < 0.40", "L1+L2+L3": "0.40 ≤ importance < 0.75", "all 4": "importance ≥ 0.75"},
                "retrieval_pipeline": ["embed(query)", "4-layer asyncio.gather", "blend scores (rel×0.7 + recency×0.3)", "MMR dedup λ=0.6", "token trim"],
                "consolidation": ["summarise last N turns via LLM", "extract named entities", "decay low-access mid-term records"],
                "memory_available": mem is not None,
            }

        # Registry flow
        elif req.scenario == "registry_flow":
            reg    = services.get("registry")
            agents = await reg.list_agents() if reg else []
            result = {
                "registered_agents": len(agents),
                "discovery_paths": ["/.well-known/agent-card.json", "/agent-card",
                                    "/api/v1/a2a/.well-known/agent-card.json", "/api/v1/a2a/agent-card"],
                "lb_strategies": {"round_robin": "default, stateful last-index", "random": "stateless",
                                  "least_loaded": "uses record_task_start/end counters"},
                "health_check_interval": "60s",
                "card_refresh_interval": "300s",
                "health_states": ["unknown", "healthy", "degraded (still routed)", "unhealthy (excluded)"],
                "agent_list": [{"id": a.agent_id[:8] + "…", "name": a.card.name,
                                "health": a.health.value,
                                "skills": list(a.skill_index.keys())[:4]}
                               for a in agents[:5]],
            }

        # E2E simple query
        elif req.scenario == "e2e_simple":
            q   = req.params.get("query", "why is user authentication failing?")
            cls = loop.classify(q)
            pre = await loop.pre_verify(q, [], {})
            res = await loop.run(query=q, session_id=sid,
                                 env_context={"site": "Site-A"},
                                 tool_registry=tool_registry)
            result = {
                "query": q,
                "classify": {"complexity": cls.complexity.value, "tier": cls.model_tier},
                "pre_verify": {"passed": pre.passed},
                "path": "Runtime Loop (no LangGraph)",
                "turns": res.turns_taken,
                "stop_outcome": res.outcome.value,
                "tools_called": res.tool_summaries,
                "response_preview": res.final_response[:400],
                "modules": ["runtime/loop.py", "runtime/context_budget.py", "runtime/stop_policy.py",
                            "skills/catalog.py", "tools/mock_tools.py"],
            }

        # E2E complex query (classify only — HITL graph not run in demo)
        elif req.scenario == "e2e_complex":
            q   = req.params.get("query", "restart the payments-service in production")
            cls = loop.classify(q)
            pre = await loop.pre_verify(q, [], {})
            result = {
                "query": q,
                "classify": {"complexity": cls.complexity.value, "tier": cls.model_tier, "reason": cls.reason},
                "pre_verify": {"passed": pre.passed, "reason": pre.reason},
                "path": "HITL Graph + TaskPlanner DAG",
                "hitl_steps": {
                    "1": "evaluate_triggers() → DestructiveActionTrigger fires",
                    "2": "LangGraph interrupt() → state serialised to MemorySaver",
                    "3": "HitlReviewService.notify() → 5 channels concurrently",
                    "4": "Operator submits via /hitl/{id}/approve",
                    "5": "graph.update_state() + ainvoke(None) → resumes",
                    "6": "_verify_action_result() → post-action health check",
                },
                "modules": ["hitl/a2a_integration.py", "hitl/graph.py", "hitl/triggers.py",
                            "hitl/decision.py", "hitl/review.py", "hitl/audit.py",
                            "task/intra/planner.py", "registry/registry.py"],
            }

        # ── Hermes: FTS5 session write + search ──────────────────────────
        elif req.scenario == "hermes_fts_search":
            fts = services.get("fts_store")
            query = req.params.get("query", "RADIUS authentication failure")
            if fts is None:
                result = {"error": "FTS5SessionStore not initialised — check HERMES_DATA_DIR"}
            else:
                stats_before = await fts.get_stats()
                # Write a synthetic turn so search has something to find
                await fts.write_turn(
                    session_id=sid,
                    user_text=f"Demo turn: {query} on ap-03 and ap-07",
                    assistant_text=(
                        "Searched syslogs — RADIUS certificate expired. "
                        "Renewed cert via admin console, auth restored for 40 clients."
                    ),
                    tool_calls=[{"tool": "syslog_search", "result": "cert expired 2 days ago"}],
                    importance=0.85,
                    tags=["demo", "radius", "auth"],
                )
                results = await fts.search(query=query, limit=5)
                stats_after = await fts.get_stats()
                summary = await fts.summarize_results(results, query)
                sessions = await fts.list_sessions()
                result = {
                    "query":           query,
                    "turns_before":    stats_before["total_turns"],
                    "turns_after":     stats_after["total_turns"],
                    "sessions_total":  stats_after["total_sessions"],
                    "db_size_kb":      stats_after["db_size_kb"],
                    "search_results":  len(results),
                    "top_result": {
                        "session_id":    results[0].session_id if results else None,
                        "snippet":       results[0].snippet if results else None,
                        "rank":          results[0].rank if results else None,
                        "user_text":     results[0].user_text[:120] if results else None,
                    } if results else None,
                    "llm_summary":     summary,
                    "recent_sessions": [
                        {"session_id": s.session_id[:16]+"…",
                         "turn_count": s.turn_count, "topic": s.topic_summary}
                        for s in sessions[:5]
                    ],
                    "explanation": (
                        "Every turn is written to SQLite FTS5. "
                        "New sessions search past turns by keyword before building the prompt — "
                        "only the top-K relevant excerpts are injected, not the full history."
                    ),
                }

        # ── Hermes: Memory curation (per-turn + nudge) ───────────────────
        elif req.scenario == "hermes_curation":
            curator = services.get("memory_curator")
            if curator is None:
                result = {"error": "MemoryCurator not initialised"}
            else:
                turns = req.params.get("turns", [
                    ("RADIUS auth failing for all APs at site-a",
                     "Found expired cert on RADIUS-01. Renewed — auth restored.",
                     [{"tool": "syslog_search", "result": "cert_expired"}]),
                    ("I prefer httpx over requests for all HTTP calls",
                     "Noted. Using httpx for all outbound HTTP.",
                     []),
                    ("Check BGP summary for router-01",
                     "BGP summary: 3 neighbors established, 1 active, 0 idle.",
                     [{"tool": "get_bgp_summary", "result": "3 established"}]),
                ])
                all_memories = []
                for user_t, asst_t, tc in turns:
                    memories = await curator.after_turn(sid, user_t, asst_t, tc)
                    all_memories.extend(memories)
                counter = curator._turn_counter.get(sid, 0)
                result = {
                    "session_id":       sid,
                    "turns_processed":  len(turns),
                    "turn_counter":     counter,
                    "memories_curated": len(all_memories),
                    "curated": [
                        {
                            "content":     m.content[:120],
                            "type":        m.memory_type.value,
                            "confidence":  round(m.confidence, 2),
                            "tags":        m.tags,
                        }
                        for m in all_memories
                    ],
                    "nudge_status": (
                        f"Nudge fires at intervals {curator._shallow_n} (shallow) "
                        f"and {curator._deep_n} (deep). "
                        f"Current turn: {counter}. "
                        f"Next shallow at turn {((counter // curator._shallow_n) + 1) * curator._shallow_n}."
                    ),
                    "explanation": (
                        "After each turn, the LLM is asked: 'What is worth remembering?' "
                        "Only distilled facts (not raw chat) are written to long-term memory. "
                        "Every 5 turns a shallow nudge scans for missed facts. "
                        "Every 20 turns a deep nudge re-evaluates preferences and detects contradictions."
                    ),
                }

        # ── Hermes: User model behavioral inference ──────────────────────
        elif req.scenario == "hermes_user_model":
            user_model = services.get("user_model")
            if user_model is None:
                result = {"error": "UserModelEngine not initialised"}
            else:
                # Run a representative sequence of turns
                demo_turns = [
                    ("RADIUS auth failing on ap-03 and ap-07",
                     "Searched syslogs — cert expired on RADIUS-01.",
                     [{"tool": "syslog_search"}, {"tool": "get_device_status"}]),
                    ("WiFi RSSI below threshold on 5GHz channels at site-a",
                     "Channel 6 congested. Recommend switching to channel 1.",
                     [{"tool": "get_device_status"}]),
                    ("I prefer httpx over requests for all HTTP tool calls",
                     "Noted, using httpx going forward.",
                     []),
                    ("Show BGP summary and check for route flapping",
                     "BGP: 3 neighbors. Route to 10.0.0.0/8 flapping every 30s.",
                     [{"tool": "get_bgp_summary"}, {"tool": "syslog_search"}]),
                    ("List open P1 incidents",
                     "2 open P1 incidents: INC-1291 (RADIUS), INC-1305 (DNS).",
                     [{"tool": "list_incidents"}]),
                ]
                for user_t, asst_t, tc in demo_turns:
                    await user_model.after_turn(sid, user_t, asst_t, tc)
                profile = user_model.get_profile(sid)
                prompt_section = user_model.get_prompt_section(sid)
                result = {
                    "session_id":       sid,
                    "turns_processed":  len(demo_turns),
                    "technical_level":  profile.technical_level.value,
                    "comm_style":       profile.communication_style.value,
                    "total_turns":      profile.total_turns,
                    "domain_counts":    dict(sorted(profile.domain_counts.items(), key=lambda x: -x[1])),
                    "tool_usage":       dict(sorted(profile.tool_usage.items(), key=lambda x: -x[1])),
                    "stated_prefs":     profile.stated_preferences,
                    "revealed_prefs":   profile.revealed_preferences,
                    "contradictions":   profile.contradictions,
                    "trait_count":      len(profile.traits),
                    "top_traits": [
                        {"trait": k, "value": str(v.value)[:80],
                         "confidence": round(v.confidence, 2),
                         "contradicted": v.contradicted}
                        for k, v in sorted(profile.traits.items(),
                                           key=lambda x: -x[1].confidence)[:5]
                    ],
                    "hourly_activity":  {str(h): c for h, c in sorted(profile.hourly_activity.items())},
                    "prompt_section":   prompt_section,
                    "explanation": (
                        "The user model tracks REVEALED preferences (actual tool choices, query patterns) "
                        "separately from STATED preferences (what the operator says). "
                        "Contradictions are flagged when behavior doesn't match claims. "
                        "The [OPERATOR PROFILE] section is injected as hidden context every session."
                    ),
                }

        # ── Hermes: Skill auto-creation ──────────────────────────────────
        elif req.scenario == "hermes_skill_creation":
            evolver = services.get("skill_evolver")
            if evolver is None:
                result = {"error": "SkillEvolver not initialised (needs skill_catalog in services)"}
            else:
                complexity = float(req.params.get("complexity", 7.5))
                proposal = await evolver.after_task(
                    task_description=req.params.get(
                        "task", "RADIUS certificate renewal for production AP network"
                    ),
                    solution_summary=(
                        "Identified expired certificate on RADIUS-01 via syslog. "
                        "Renewed cert via admin console. Verified auth restored for 40 clients."
                    ),
                    tools_used=["syslog_search", "get_device_status", "get_bgp_summary"],
                    solution_steps=[
                        "Search syslogs with severity=error for RADIUS errors",
                        "Identify cert expiry timestamp from log message",
                        "Access RADIUS admin console and navigate to Certificates",
                        "Renew certificate and restart RADIUS service",
                        "Verify AP auth by checking syslog for 'Auth OK' messages",
                        "Confirm all 40 affected clients have reconnected",
                    ],
                    key_observations=[
                        "Certificate had expired 2 days prior — no auto-renewal configured",
                        "Auth restored within 90 seconds of cert renewal",
                        "All 40 clients reconnected without manual intervention",
                    ],
                    complexity=complexity,
                    session_id=sid,
                )
                if proposal:
                    versions = evolver.get_version_history(proposal.skill_id)
                    stats = evolver.get_all_skill_stats()
                    result = {
                        "created":         True,
                        "skill_id":        proposal.skill_id,
                        "reuse_potential": round(proposal.reuse_potential, 2),
                        "complexity_score": round(proposal.complexity_score, 2),
                        "rationale":       proposal.rationale,
                        "version_count":   len(versions),
                        "version_history": versions,
                        "markdown_preview": proposal.markdown_content[:800],
                        "total_auto_skills": len(stats),
                        "eligibility_threshold": {
                            "min_complexity":      evolver._min_complex,
                            "min_reuse_potential": evolver._min_reuse,
                            "passed":              True,
                        },
                        "explanation": (
                            "After each complex task, the agent asks: 'Should this be a reusable Skill?' "
                            f"This task scored complexity={complexity} (threshold={evolver._min_complex}) "
                            f"and reuse_potential={round(proposal.reuse_potential, 2)} "
                            f"(threshold={evolver._min_reuse}). "
                            "The markdown skill was written and registered in SkillCatalogService."
                        ),
                    }
                else:
                    result = {
                        "created":  False,
                        "reason":   f"Below threshold: complexity={complexity} (min={evolver._min_complex if evolver else 'N/A'})",
                        "eligibility_threshold": {
                            "min_complexity":      evolver._min_complex if evolver else None,
                            "min_reuse_potential": evolver._min_reuse if evolver else None,
                            "passed":              False,
                        },
                    }

        # ── Hermes: Skill self-improvement via feedback ──────────────────
        elif req.scenario == "hermes_skill_feedback":
            evolver = services.get("skill_evolver")
            if evolver is None:
                result = {"error": "SkillEvolver not initialised"}
            else:
                # Create a skill first if none exist
                skill_id = req.params.get("skill_id", "")
                if not skill_id or not evolver.get_version_history(skill_id):
                    proposal = await evolver.after_task(
                        task_description="RADIUS certificate renewal procedure",
                        solution_summary="Renewed cert, auth restored",
                        tools_used=["syslog_search"],
                        solution_steps=["Check syslog", "Renew cert", "Verify"],
                        key_observations=["Cert expired 2 days ago"],
                        complexity=6.0,
                        session_id=sid,
                    )
                    skill_id = proposal.skill_id if proposal else ""

                feedback = req.params.get(
                    "feedback",
                    "Step 1 should also check interface metrics before syslog to rule out network issues first.",
                )
                if skill_id:
                    v_before = len(evolver.get_version_history(skill_id))
                    fb_result = await evolver.apply_feedback(
                        skill_id=skill_id,
                        feedback=feedback,
                        success=True,
                        problem_step="Check syslog",
                    )
                    v_after = len(evolver.get_version_history(skill_id))
                    history = evolver.get_version_history(skill_id)
                    result = {
                        "skill_id":      skill_id,
                        "feedback":      feedback,
                        "version_before": v_before,
                        "version_after":  v_after,
                        "changes":        fb_result.changes if fb_result else [],
                        "quality_delta":  round(fb_result.quality_delta, 3) if fb_result else 0,
                        "quality_improved": (fb_result.quality_delta > 0) if fb_result else False,
                        "version_history": history,
                        "explanation": (
                            "Operator feedback triggers an LLM patch of the skill's steps. "
                            "Only the identified problem step is changed — the rest is preserved. "
                            "A new SkillVersion is created with the diff summary. "
                            "Rollback to any previous version is available."
                        ),
                    }
                else:
                    result = {"error": "Could not create or find a skill to apply feedback to"}

        # ── Hermes: Full learning loop (all 5 nodes in one demo) ─────────
        elif req.scenario == "hermes_full_loop":
            fts     = services.get("fts_store")
            curator = services.get("memory_curator")
            umodel  = services.get("user_model")
            evolver = services.get("skill_evolver")
            loop_sid = f"loop-demo-{sid}"

            steps = []

            # Node 1: FTS5 recall
            if fts:
                recall_results = await fts.search("RADIUS authentication", limit=3)
                steps.append({
                    "node": "1 — FTS5 Cross-Session Recall",
                    "status": "executed",
                    "detail": f"Searched past sessions for 'RADIUS authentication'. Found {len(recall_results)} relevant turns.",
                    "recall_count": len(recall_results),
                })
            else:
                steps.append({"node": "1 — FTS5 Cross-Session Recall", "status": "skipped", "detail": "FTS5 not initialised"})

            # Node 2: Simulate turn + write to FTS5
            user_q = "RADIUS auth failing for wireless users at site-a. Check ap-03 and ap-07."
            asst_r = "Searched syslogs — RADIUS-01 certificate expired 2 days ago. Renewing now. Auth should restore in 60s."
            tool_calls_demo = [{"tool": "syslog_search", "result": "cert_expired_RADIUS-01"}]
            if fts:
                await fts.write_turn(loop_sid, user_q, asst_r, tool_calls=tool_calls_demo, importance=0.9)
                stats = await fts.get_stats()
                steps.append({
                    "node": "2 — FTS5 Turn Write",
                    "status": "executed",
                    "detail": f"Turn written. DB now has {stats['total_turns']} total turns across {stats['total_sessions']} sessions.",
                    "db_size_kb": stats["db_size_kb"],
                })
            else:
                steps.append({"node": "2 — FTS5 Turn Write", "status": "skipped", "detail": "FTS5 not initialised"})

            # Node 3: Memory curation
            curated_count = 0
            if curator:
                memories = await curator.after_turn(loop_sid, user_q, asst_r, tool_calls_demo)
                curated_count = len(memories)
                steps.append({
                    "node": "3 — Memory Curation",
                    "status": "executed",
                    "detail": f"LLM curated {curated_count} memories from this turn.",
                    "memories": [{"content": m.content[:100], "type": m.memory_type.value, "confidence": round(m.confidence,2)} for m in memories],
                })
            else:
                steps.append({"node": "3 — Memory Curation", "status": "skipped", "detail": "MemoryCurator not initialised"})

            # Node 4: User model update
            if umodel:
                profile = await umodel.after_turn(loop_sid, user_q, asst_r, tool_calls_demo)
                steps.append({
                    "node": "4 — User Model Update",
                    "status": "executed",
                    "detail": f"Profile updated. Domains: {dict(list(profile.domain_counts.items())[:3])}. Tools: {dict(list(profile.tool_usage.items())[:3])}.",
                    "technical_level": profile.technical_level.value,
                    "domain_counts":   profile.domain_counts,
                })
            else:
                steps.append({"node": "4 — User Model Update", "status": "skipped", "detail": "UserModelEngine not initialised"})

            # Node 5: Skill auto-creation
            if evolver:
                proposal = await evolver.after_task(
                    task_description="RADIUS authentication certificate renewal",
                    solution_summary=asst_r,
                    tools_used=["syslog_search"],
                    solution_steps=["Search syslog", "Identify cert expiry", "Renew cert", "Verify auth"],
                    key_observations=["Cert expired 2 days ago", "Auth restored in 60s"],
                    complexity=6.5,
                    session_id=loop_sid,
                )
                steps.append({
                    "node": "5 — Skill Auto-Creation",
                    "status": "executed" if proposal else "skipped (below threshold)",
                    "detail": (
                        f"Created skill '{proposal.skill_id}' (reuse={round(proposal.reuse_potential,2)})"
                        if proposal else "Task complexity below threshold for skill creation"
                    ),
                    "skill_id":    proposal.skill_id if proposal else None,
                    "reuse":       round(proposal.reuse_potential, 2) if proposal else None,
                })
            else:
                steps.append({"node": "5 — Skill Auto-Creation", "status": "skipped", "detail": "SkillEvolver not initialised"})

            result = {
                "session_id":    loop_sid,
                "query":         user_q,
                "assistant":     asst_r,
                "loop_steps":    steps,
                "modules_active": {
                    "fts_store":      fts is not None,
                    "memory_curator": curator is not None,
                    "user_model":     umodel is not None,
                    "skill_evolver":  evolver is not None,
                },
                "explanation": (
                    "This is the complete Hermes 5-node learning loop run in sequence for one turn. "
                    "In production, all nodes fire after every turn. "
                    "The loop is what makes the agent get better with use — not just remember more."
                ),
            }

        # ── Live skill evolution: run query → get result → evolve skill ─────
        elif req.scenario == "skill_evolve_live":
            evolver = services.get("skill_evolver")
            loop    = services.get("runtime_loop")
            if evolver is None:
                result = {"error": "SkillEvolver not initialised"}
            else:
                query = req.params.get("query", "why is RADIUS authentication failing for wireless users?")
                # Step 1: Run a real loop query to get actual tool results
                step1_result = "not run"
                step1_tools  = []
                try:
                    res = await loop.run(
                        query=query,
                        session_id=sid,
                        tool_registry=tool_registry,
                    )
                    step1_result = res.final_response[:400]
                    step1_tools  = res.tool_summaries or []
                except Exception as exc:
                    step1_result = f"Loop error: {exc}"

                # Step 2: Trigger skill evolution from the completed task
                proposal = await evolver.after_task(
                    task_description=query,
                    solution_summary=step1_result,
                    tools_used=step1_tools[:4] or ["syslog_search", "get_device_status"],
                    solution_steps=[
                        "Search syslogs for authentication errors",
                        "Check RADIUS server certificate expiry",
                        "Verify device connectivity to RADIUS server",
                        "Confirm auth restoration after remediation",
                    ],
                    key_observations=[
                        "RADIUS certificate expiry is the most common cause",
                        "syslog_search with severity=error reveals the root cause",
                        "Auth restores within 60s of certificate renewal",
                    ],
                    complexity=req.params.get("complexity", 7.0),
                    session_id=sid,
                )

                # Step 3: Apply simulated operator feedback if skill was created
                feedback_result = None
                if proposal:
                    fb = req.params.get("feedback",
                        "Step 1 should also check interface metrics before syslog to rule out network-layer issues.")
                    if fb:
                        feedback_result = await evolver.apply_feedback(
                            skill_id=proposal.skill_id,
                            feedback=fb,
                            success=True,
                            problem_step="Search syslogs for authentication errors",
                        )

                result = {
                    "query":          query,
                    "step1_loop_result": step1_result,
                    "step1_tools_called": step1_tools,
                    "step2_skill_created": proposal is not None,
                    "skill_id":       proposal.skill_id if proposal else None,
                    "reuse_potential": round(proposal.reuse_potential, 2) if proposal else None,
                    "markdown_preview": proposal.markdown_content[:600] if proposal else None,
                    "step3_feedback_applied": feedback_result is not None,
                    "version_after_feedback": feedback_result.new_version if feedback_result else None,
                    "quality_delta":  round(feedback_result.quality_delta, 3) if feedback_result else None,
                    "version_history": evolver.get_version_history(proposal.skill_id) if proposal else [],
                    "explanation": (
                        "This demo runs a real loop query, feeds its output to SkillEvolver, "
                        "creates a skill if complexity threshold is met, then applies feedback "
                        "to self-improve the skill — the complete Hermes §05 flow end-to-end."
                    ),
                }

        else:
            return JSONResponse(content={"error": f"Unknown scenario: {req.scenario}"}, status_code=400)

        return JSONResponse(content={"scenario": req.scenario, "result": result})

    @app.get("/demos", response_class=HTMLResponse)
    async def serve_demos():
        page = _STATIC_DIR / "demos.html"
        if page.exists():
            return HTMLResponse(content=page.read_text(encoding="utf-8"))
        return HTMLResponse(content="<h1>demos.html not found</h1>")


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

    # Post-HITL synthesis: run one LLM turn to summarise the tool execution result
    # so the chat shows a meaningful response after the operator approves.
    _loop = services.get("runtime_loop")
    _tool_result = result_dict.get("tool_result", "")
    _tool_name   = result_dict.get("tool_name", "the approved tool")
    if _loop and _tool_result and result_dict.get("decision") == "approve":
        try:
            _synthesis_query = (
                f"The HITL-approved tool '{_tool_name}' has just been executed. "
                f"Summarise the result for the operator in 2-3 sentences."
            )
            _synthesis_facts = [f"TOOL_RESULT: {str(_tool_result)[:800]}"]
            _full_text = ""
            async for _chunk in _loop.stream(
                query           = _synthesis_query,
                session_id      = f"hitl__{interrupt_id[:8]}",
                confirmed_facts = _synthesis_facts,
            ):
                if _chunk.get("type") == "token":
                    _full_text += _chunk.get("token", "")
            if _full_text.strip():
                result_dict["synthesis"] = _full_text.strip()
        except Exception as _e:
            logger.debug("Post-HITL synthesis failed: %s", _e)

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