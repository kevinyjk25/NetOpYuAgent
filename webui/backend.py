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

from runtime.stop_policy import StopOutcome

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


# ---------------------------------------------------------------------------
# Async HITL (H2) SSE notification bridge (2026-05)
# ---------------------------------------------------------------------------
#
# Maps session_id → emit_fn(chunk_dict). Used by H2 on_resolved callbacks
# to push a soft-notify chunk into a currently-active chat SSE stream so
# the operator sees "RADIUS result arrived" while still on the page.
#
# Populated by chat_stream() while the SSE stream is alive; removed on
# stream end. If the operator has navigated away (no entry), the H2 ack
# still writes a confirmed_fact via the inject queue (see runtime/loop.py
# drain_async_inject) — the SSE notify is purely additive.
#
# Thread-safety: chat_stream / SSE runs in the asyncio event loop, and
# H2 on_resolved callbacks ALSO run in the asyncio event loop (they're
# triggered by router.deliver which is async). So dict ops are safe
# without a lock — no concurrent mutation.

_session_sse_emit: dict[str, "Callable[[dict], None]"] = {}


# ── A2A Phase 3 (P3-b): cross-agent HITL resume ───────────────────────────
# When a peer (e.g. dc) resolves a delegated HITL, it calls this agent's
# /api/v1/a2a/hitl_resolved endpoint. That endpoint hands the result here so
# we can (1) inject it into the originating session and (2) ACTIVELY drive a
# synthesis turn (A2 — no user input needed) and buffer the answer for the
# frontend's poll. Buffer is in-memory (persistence deferred to P3-d).
#
# session_id -> list of {text, peer_agent, correlation_id, ts} resumption items
_pending_resumptions: dict[str, list[dict]] = {}
# Set by create_webui_app so the module-level a2a callback can drive a turn.
_resume_driver: "Optional[Callable]" = None
# Strong refs to detached cross-agent resume tasks (asyncio holds only weak
# refs; without this a fire-and-forget resume turn could be GC'd mid-flight).
_resume_tasks: set = set()


def register_resume_driver(fn) -> None:
    """Called by create_webui_app to publish the active-resume coroutine
    (drives one synthesis turn for a session and buffers the answer)."""
    global _resume_driver
    _resume_driver = fn


async def handle_cross_agent_resume(
    *, local_session_id: str, peer_agent: str, result_text: str,
    decision: str, correlation_id: str = "",
) -> bool:
    """Entry point invoked by the a2a /hitl_resolved endpoint.

    SCHEDULES the active resume (A2) on a background task and returns
    immediately. The resume turn drives a full LLM synthesis on THIS agent
    (30-60s on a local model); the peer's callback POST must NOT block on it,
    or the POST's HTTP timeout fires long before the turn finishes — leaving
    the peer's resumer hung mid-approval and this agent's turn result orphaned
    (the bug that left dc's inbound task stuck and lan never resuming).

    Returns True when a resume was SCHEDULED (driver present), False when no
    driver is registered (we buffer the raw result for next-turn merge-back).
    The completed answer reaches the UI via _pending_resumptions + the
    /chat/resumptions poll (and a live SSE push if the stream is still open).
    """
    if _resume_driver is None:
        logger.warning(
            "handle_cross_agent_resume: no resume driver registered "
            "(session=%s peer=%s) — buffering raw result only",
            local_session_id, peer_agent,
        )
        _pending_resumptions.setdefault(local_session_id, []).append({
            "text": result_text, "peer_agent": peer_agent,
            "correlation_id": correlation_id, "driven": False,
        })
        return False

    # Fire-and-forget: schedule the synthesis turn, return at once so the
    # peer's POST /hitl_resolved gets a fast 200. The driver has its own
    # try/except + logging, and buffers its answer into _pending_resumptions
    # on completion, so a detached task losing its result is safe.
    async def _run_resume() -> None:
        try:
            await _resume_driver(
                local_session_id=local_session_id, peer_agent=peer_agent,
                result_text=result_text, decision=decision,
                correlation_id=correlation_id,
            )
        except Exception as exc:  # never let a detached task die silently
            logger.exception(
                "cross-agent resume task failed (session=%s peer=%s): %s",
                local_session_id, peer_agent, exc,
            )

    import asyncio as _asyncio
    _task = _asyncio.create_task(
        _run_resume(), name=f"xa_resume_{(local_session_id or '?')[:8]}",
    )
    # Keep a reference so the task isn't GC'd mid-flight (asyncio only holds
    # a weak ref to tasks); drop it from the set when done.
    _resume_tasks.add(_task)
    _task.add_done_callback(_resume_tasks.discard)
    return True


def register_session_sse_emit(session_id: str, emit_fn) -> None:
    """Called by chat_stream while its SSE stream is alive."""
    if session_id:
        _session_sse_emit[session_id] = emit_fn


def unregister_session_sse_emit(session_id: str) -> None:
    """Called by chat_stream on stream end (finally block)."""
    if session_id:
        _session_sse_emit.pop(session_id, None)


def emit_async_hitl_notify(session_id: str, chunk: dict) -> bool:
    """Push an async-hitl-resolved chunk into the session's SSE stream if active.

    Returns True if delivered to a live stream, False if no stream (operator
    has navigated away — caller should rely on the confirmed_facts inject
    queue alone for next-turn merge-back).

    Called by H2 on_resolved callbacks (typically in skills / tools that
    wire up an async approval). Safe to call when SSE not active — silent
    no-op.
    """
    emit_fn = _session_sse_emit.get(session_id) if session_id else None
    if emit_fn is None:
        return False
    try:
        emit_fn(chunk)
        return True
    except Exception as exc:
        logger.warning(
            "emit_async_hitl_notify: emit_fn failed for session %s: %s",
            session_id, exc,
        )
        return False


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
# Request / response models — DEFINITIONS live in webui/schemas.py to keep
# them importable at module level by route extraction files (routes_hitl,
# routes_system, etc) without circular imports via backend.py.
#
# Re-exported here for backwards compat: any historical code doing
# `from webui.backend import HitlDecisionRequest` still works.
# ---------------------------------------------------------------------------
from webui.schemas import ChatRequest, ToolCallRequest, HitlDecisionRequest   # noqa: E402,F401


# ---------------------------------------------------------------------------
# WebUI factory
# ---------------------------------------------------------------------------

def create_webui_app(services: dict[str, Any]) -> FastAPI:
    """
    Build and return the WebUI FastAPI sub-application.

    Expects 'services' to contain keys from main.py's build_services():
        executor, hitl_core_router, hitl_core_audit, memory, registry, task_system
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
        _tl = _TL(mode=_cfg.cfg.mode, profile=_cfg.cfg.agent.profile)
        _skill_loader = _SL(mode=_cfg.cfg.mode, profile=_cfg.cfg.agent.profile)
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
    tool_registry = _TL_be(mode=_cfg_be.cfg.mode, profile=_cfg_be.cfg.agent.profile).build_callables()
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
        # editable_hitl_tools: business map injected by the active profile's
        # config (L1). Empty if unset → L0 stays domain-free (Stage A).
        _editable_hitl = {}
        try:
            from config import cfg as _app_cfg2
            _editable_hitl = dict(getattr(_app_cfg2.tools, "editable_hitl_tools", {}) or {})
        except Exception as _exc:
            logger.warning("backend: cfg fallback for editable_hitl_tools failed: %s", _exc)
        # L0/L1 Stage B: ask the profile layer for its batch HITL resolver
        # (network profiles return the device-prose resolver; default → None).
        _batch_resolver = None
        try:
            from profiles import get_batch_resolver_for_profile
            _batch_resolver = get_batch_resolver_for_profile(_cfg_be.cfg.agent.profile)
            if _batch_resolver:
                logger.info("backend: batch_resolver injected for profile=%s", _cfg_be.cfg.agent.profile)
        except Exception as _exc:
            logger.warning("backend: batch_resolver injection skipped: %s", _exc)
        services["runtime_loop"] = AgentRuntimeLoop(
            memory_router=services.get("memory"),
            config=RuntimeConfig(hitl_tool_names=_hitl_tools, editable_hitl_tools=_editable_hitl),
            tool_store=services["tool_store"],
            skill_catalog=services["skill_catalog"],
            delegate_fn=services.get("delegate_fn"),
            batch_resolver_fn=_batch_resolver,
        )
    else:
        # Re-inject tool store and catalog into existing loop
        loop = services["runtime_loop"]
        loop._store   = services["tool_store"]
        loop._budget._store = services["tool_store"]
        loop._skill_catalog = services["skill_catalog"]
        # Re-inject delegation hook (Phase 2B) if present.
        if services.get("delegate_fn") is not None:
            loop._delegate_fn = services["delegate_fn"]

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
    # Publish into services so module-level handlers (e.g.
    # _submit_hitl_decision, which is NOT a closure of this factory and only
    # receives `services`) can read the session transcript. Without this the
    # H2 async-HITL follow-up raised NameError('_message_history') because it
    # referenced this closure local from the wrong scope.
    services["_message_history"] = _message_history

    # ── Static files ───────────────────────────────────────────────────
    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # ── Serve index.html ───────────────────────────────────────────────
    # No-cache headers: index.html changes often during dev sessions, and
    # browsers cache it aggressively (memory cache lasts the whole tab
    # session, disk cache 24h+). After backend updates, operators were
    # debugging against stale HTML — we shipped a fix but their browser
    # never loaded it. These headers tell the browser to re-fetch every
    # time. The cost is one round-trip per page load — trivial since this
    # is a single-operator dev UI.
    @app.get("/", response_class=HTMLResponse)
    async def serve_index():
        index = _STATIC_DIR / "index.html"
        if index.exists():
            return HTMLResponse(
                content=index.read_text(encoding="utf-8"),
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma":        "no-cache",
                    "Expires":       "0",
                },
            )
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

    # ── A2A Phase 3 (P3-b): active cross-agent HITL resume (A2) ────────
    # Drives ONE synthesis turn for a session after a peer resolved a
    # delegated HITL, with no user input, and buffers the answer for the
    # frontend poll (/chat/resumptions). The peer's result is injected via
    # the existing H2 async-inject queue so the loop's turn-start drain
    # merges it into confirmed_facts before assembling the synthesis prompt.
    async def _drive_cross_agent_resume(
        *, local_session_id: str, peer_agent: str, result_text: str,
        decision: str, correlation_id: str = "",
    ) -> bool:
        _executor = services.get("executor")
        if _executor is None or not local_session_id:
            logger.warning("cross-agent resume: no executor / session — skip")
            return False
        # 1. Inject the peer's result as a confirmed fact (H2 reuse).
        try:
            from runtime.loop import enqueue_async_inject
            _fact = (
                f"[跨 Agent 委派结果 from {peer_agent} — operator {decision}] "
                f"{result_text}".strip()
            )
            enqueue_async_inject(local_session_id, _fact)
        except Exception as _inj_exc:
            logger.warning("cross-agent resume: inject failed: %s", _inj_exc)
        # 2. Actively drive ONE synthesis turn; collect the answer.
        _parts: list[str] = []
        async def _collect(ch: dict) -> None:
            tok = ch.get("token") or ""
            if tok:
                _parts.append(str(tok))
        _synth_q = (
            f"[系统] 委派到 {peer_agent} 的人在环审批已完成,结果已注入上下文。"
            f"请综合本地已有信息与该结果,给出针对原始请求的最终完整回答。"
        )
        try:
            await _executor.execute_query(
                query=_synth_q,
                session_id=local_session_id,
                confirmed_facts=[],
                env_context={"_cross_agent_resume": True},
                on_chunk=_collect,
            )
        except Exception as _ex:
            logger.exception("cross-agent resume: synthesis turn failed: %s", _ex)
            return False
        _answer = "".join(_parts).strip()
        # 3. Buffer for the frontend poll + push to a live SSE if still open.
        _pending_resumptions.setdefault(local_session_id, []).append({
            "text": _answer, "peer_agent": peer_agent,
            "correlation_id": correlation_id, "decision": decision,
            "driven": True,
        })
        try:
            emit_async_hitl_notify(local_session_id, {
                "type": "cross_agent_resume", "peer_agent": peer_agent,
                "text": _answer, "correlation_id": correlation_id,
            })
        except Exception:
            pass
        logger.info(
            "cross-agent resume: drove synthesis turn for session=%s peer=%s "
            "answer_chars=%d", local_session_id[:12], peer_agent, len(_answer),
        )
        return True

    register_resume_driver(_drive_cross_agent_resume)

    @app.get("/chat/resumptions")
    async def get_resumptions(session_id: str) -> JSONResponse:
        """Frontend polls this to pick up cross-agent HITL resume answers that
        arrived after the original SSE stream closed (A2A Phase 3). Returns +
        clears the buffered items for the session."""
        items = _pending_resumptions.pop(session_id, [])
        return JSONResponse(content=items)

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

        # Close any HITL sub-streams still live from a PRIOR turn on this
        # same session. Without this, the SSE history of a long-running
        # HITL resumer (e.g. agent_loop_resumer that hung mid-execution)
        # can leak chunks into the UI of THIS new turn — operators see
        # stale "HITL approve" / "Skills matched" / Turn-N traces mixed
        # into their fresh query.  See AUDIT_REPORT issue D.
        try:
            from hitl_core.chunk_queue import get_chunk_queue_registry
            _closed = await get_chunk_queue_registry().close_session_streams(session_id)
            if _closed:
                logger.info(
                    "chat_stream: closed %d stale HITL stream(s) for session=%s "
                    "before starting new turn",
                    _closed, session_id[:12],
                )
        except Exception as _close_exc:
            # Never let lifecycle hygiene block the actual chat — log and continue
            logger.debug("chat_stream: close_session_streams skipped: %s", _close_exc)

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

                # ── Async-HITL SSE bridge (H2, 2026-05) ─────────────────
                # Register a non-async emit_fn so H2 on_resolved callbacks
                # (running in the same event loop but outside this generator)
                # can push a soft-notify chunk into this stream. Use a
                # nowait put because the emit is fire-and-forget; queue is
                # unbounded so the put can't fail in practice.
                def _h2_emit(chunk: dict) -> None:
                    try:
                        _chunk_queue.put_nowait(("chunk", chunk))
                    except Exception as exc:
                        logger.warning(
                            "_h2_emit: put_nowait failed for session=%s: %s",
                            session_id, exc,
                        )

                register_session_sse_emit(session_id, _h2_emit)

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

                # ── Graceful-shutdown plumbing (Sprint-3-pre, 2026-05) ──
                # Register this task so main.py's lifespan drain phase can
                # wait for it (up to 30s) before killing the process. Without
                # this, a SIGTERM mid-tool-execute would orphan device state.
                # set.discard on completion guarantees we don't leak entries
                # (asyncio task callbacks run inline on completion).
                _in_flight = services.get("in_flight_tasks")
                if _in_flight is not None:
                    _in_flight.add(exec_task)
                    exec_task.add_done_callback(_in_flight.discard)

                # Drain queue → SSE
                _final_text = ""
                _final_interrupt: Optional[str] = None
                _stalled = False
                _user_cancelled = False
                try:
                    while True:
                        try:
                            _scfg = _streaming_cfg()
                            _stall_to = float(getattr(_scfg, "sse_stall_timeout_seconds", 180.0)) if _scfg else 180.0
                            kind, payload = await asyncio.wait_for(_chunk_queue.get(), timeout=_stall_to)
                        except asyncio.TimeoutError:
                            logger.warning("executor.execute_query stream stalled (%.1fs)", _stall_to)
                            _stalled = True
                            # Tell the frontend WHY the stream ended so the UI
                            # can show a meaningful message instead of just
                            # losing the response. The single most common
                            # cause is a slow LLM backend (e.g. Ollama on a
                            # large model + 8K+ prompt taking 3-5 minutes on
                            # consumer hardware); without this signal, the
                            # operator sees the trace freeze and can't tell
                            # whether the agent crashed or is still working.
                            yield f"data: {json.dumps({'type':'stall','stalled':True,'timeout_s':_stall_to,'message':'LLM backend did not respond within {:.0f}s — likely a slow model or large prompt. The request was cancelled.'.format(_stall_to)})}\n\n"
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
                except (asyncio.CancelledError, GeneratorExit):
                    # The client aborted the request mid-stream (operator hit
                    # the Stop button → fetch().abort() / reader.cancel()).
                    # FastAPI propagates that as a cancellation at our yield.
                    # Treat it as an explicit user cancel: stop the executor,
                    # mark the outcome, and fall through to the finally-style
                    # cleanup below so any partial answer + audit are still
                    # persisted. We deliberately swallow it (don't re-raise)
                    # so the post-turn hooks run; the connection is already
                    # gone so further yields are no-ops.
                    _user_cancelled = True
                    nonlocal_outcome[0] = StopOutcome.USER_CANCELLED.value
                    logger.info("chat_stream: user cancelled session=%s after %d turn(s)",
                                session_id, nonlocal_turns[0])

                try:
                    _drain_to = (lambda _s: float(getattr(_s, "exec_task_drain_timeout_seconds", 5.0)) if _s else 5.0)(_streaming_cfg())
                    if _stalled or _user_cancelled:
                        # Stalled LLM OR user cancel: the executor task is
                        # almost certainly still blocked inside httpx awaiting
                        # Ollama. Cancel it so the underlying request aborts and
                        # resources are reclaimed promptly — otherwise the
                        # orphaned task keeps running (and the user's next query
                        # could be processed by a backend still busy with the
                        # old one).
                        exec_task.cancel()
                    await asyncio.wait_for(exec_task, timeout=_drain_to)
                except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                    pass

                turns_taken     = nonlocal_turns[0]
                stop_outcome    = nonlocal_outcome[0]
                _hitl_intercepted = nonlocal_intercepted[0] or bool(_final_interrupt)
                _was_cancelled  = (stop_outcome == StopOutcome.USER_CANCELLED.value)
                full_text       = _final_text or "".join(tokens)
                if _was_cancelled and full_text:
                    # Mark the partial answer so a later read of the session
                    # transcript is honest about what happened.
                    full_text = full_text.rstrip() + "\n\n_[已取消 — 部分回答]_"

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
                #
                # Likewise, when the operator CANCELLED mid-stream, the
                # partial answer is not a trustworthy fact — keep it in the
                # session transcript (above) for context, but do NOT distil it
                # into durable memory / facts where recall would resurface a
                # half-finished answer as if complete.
                if _hitl_intercepted or _was_cancelled:
                    logger.info(
                        "chat_stream: skipping memory write for session=%s "
                        "(%s)",
                        session_id[:12],
                        "HITL intercepted; _submit_hitl_decision will persist "
                        "after operator decision" if _hitl_intercepted
                        else "user cancelled; partial answer not distilled to memory",
                    )
                else:
                    import re as _re
                    # Use centralized parser for whitespace tolerance
                    from runtime.directive_parser import find_tool_names as _ftn
                    tc = [{"tool": m} for m in _ftn(full_text)]

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
            finally:
                # Unregister async-HITL SSE bridge (H2, 2026-05)
                try:
                    unregister_session_sse_emit(session_id)
                except Exception as _u_exc:
                    logger.debug(
                        "unregister_session_sse_emit failed: %s", _u_exc,
                    )

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
    # Skills + skill_journal endpoints — registered from webui/routes_skills.py
    # ==================================================================
    from webui.routes_skills import register_skills_routes
    register_skills_routes(app, services)


    @app.get("/tools")
    async def list_tools() -> JSONResponse:
        """List tools valid for the current running mode (no mock tools in pragmatic)."""
        from tools.loader import ToolLoader
        import config as _cfg
        # ToolLoader returns only tools for the current mode (mock vs pragmatic)
        _loader = ToolLoader(mode=_cfg.cfg.mode, profile=_cfg.cfg.agent.profile)
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
    # HITL endpoints — registered from webui/routes_hitl.py
    # (Extracted during audit refactor D-4 to keep this file focused.)
    # ==================================================================
    from webui.routes_hitl import register_hitl_routes
    register_hitl_routes(app, services)

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

    @app.get("/delegations/inbound")
    async def list_inbound_delegations() -> JSONResponse:
        """List every INBOUND delegation this agent has received from peers,
        across ALL sessions.

        Rationale: when LAN delegates to DC, the inbound TaskDefinition on
        the DC side is keyed by the *LAN* session_id (it has to be — that's
        the join key for end-to-end correlation). But the DC operator
        opening DC's webui doesn't know LAN's session_id; they need a view
        that just lists "all the work peers have asked me to do".

        This endpoint serves that view. Outbound delegations remain on
        /delegations/{session_id} (filtered by the local user's session).

        NOTE: must be registered BEFORE /delegations/{session_id} or FastAPI
        will route this path's "inbound" literal as a session_id parameter.
        """
        task_system = services.get("task_system")
        if task_system is None or not hasattr(task_system, "store"):
            return JSONResponse(content=[])
        try:
            # Scan all local tasks. InMemory backend exposes _local dict;
            # Redis backend would need a session_id index — we degrade to
            # whatever's in the local fallback cache.
            all_tasks = (
                list(task_system.store._local.values())
                if hasattr(task_system.store, "_local") else []
            )
        except Exception as exc:
            logger.warning("/delegations/inbound scan failed: %s", exc)
            return JSONResponse(content=[])
        out = []
        for t in all_tasks:
            md = dict(t.metadata or {})
            if md.get("direction") != "inbound":
                continue
            assignment = getattr(t, "assignment", None)
            peer_agent = md.get("source_agent") or "?"
            _created_at = getattr(t, "created_at", "") or ""
            _completed_at = getattr(t, "completed_at", "") or ""
            latency_ms = None
            try:
                if _created_at and _completed_at:
                    from datetime import datetime as _dt
                    latency_ms = int(1000 * (
                        _dt.fromisoformat(_completed_at).timestamp()
                        - _dt.fromisoformat(_created_at).timestamp()
                    ))
            except Exception:
                latency_ms = None
            out.append({
                "task_id":      t.task_id,
                "session_id":   getattr(t, "session_id", "") or "",
                "direction":    "inbound",
                "peer_agent":   peer_agent,
                "target_agent": peer_agent,   # backward-compat alias
                "description":  t.description,
                "state":        getattr(t.state, "value", str(t.state)),
                "created_at":   _created_at,
                "completed_at": _completed_at,
                "latency_ms":   latency_ms,
                "result":       t.result if getattr(t, "result", None) else None,
                "error":        getattr(t, "error", None),
                "forked":       bool(md.get("forked", False)),
                "source_query": md.get("source_query") or "",
                "shared_facts_count": int(md.get("shared_facts_count", 0)),
                "awaiting_hitl_id":   md.get("awaiting_hitl_id") or None,
            })
        out.sort(key=lambda d: d.get("created_at") or "", reverse=True)
        return JSONResponse(content=out)

    @app.get("/delegations/{session_id}")
    async def list_delegations(session_id: str) -> JSONResponse:
        """List every [DELEGATE:] dispatched in this session, with status.

        Phase 2B+: parent-side Delegations tab data source. For each task
        the LAN-side coordinator sent over A2A, return:
          - task_id / target agent / subtask description
          - state (pending / running / completed / failed)
          - created/completed timestamps + latency
          - result (when completed) or error (when failed)
          - source_query (original user query at time of dispatch)
          - forked / shared_facts_count provenance flags

        Frontends not aware of this endpoint just don't render the tab;
        no other UI code path depends on this data.
        """
        task_system = services.get("task_system")
        if task_system is None or not hasattr(task_system, "store"):
            return JSONResponse(content=[])
        try:
            tasks = await task_system.store.get_by_session(session_id)
        except Exception as exc:
            logger.warning("/delegations/%s failed: %s", session_id, exc)
            return JSONResponse(content=[])
        out = []
        for t in tasks:
            # TaskDefinition.metadata carries delegation provenance —
            # see task/delegation.py:delegate_fn for the canonical keys.
            md = dict(t.metadata or {})
            # direction = "outbound" (this agent dispatched it) or
            # "inbound" (a peer sent it to this agent — recorded by
            # integrations/adapters/hitl_executor.py._record_inbound_delegation
            # when the A2A request carried source_agent metadata).
            # Default outbound for legacy rows written before Phase 2B+.
            direction = md.get("direction", "outbound")
            assignment = getattr(t, "assignment", None)
            if direction == "inbound":
                # peer_agent is who SENT it to us — flip the visual semantics
                # so the row reads "← lan-agent" instead of "→ <some target>"
                peer_agent = md.get("source_agent") or "?"
            else:
                peer_agent = (
                    (assignment.agent_id if assignment is not None else None)
                    or md.get("delegated_to")
                    or "?"
                )
            _created_at = getattr(t, "created_at", "") or ""
            _completed_at = getattr(t, "completed_at", "") or ""
            # Compute latency_ms when both ends available
            latency_ms = None
            try:
                if _created_at and _completed_at:
                    from datetime import datetime as _dt
                    latency_ms = int(1000 * (
                        _dt.fromisoformat(_completed_at).timestamp()
                        - _dt.fromisoformat(_created_at).timestamp()
                    ))
            except Exception:
                latency_ms = None
            out.append({
                "task_id":      t.task_id,
                "direction":    direction,
                "peer_agent":   peer_agent,
                # Keep target_agent for backward compat with any caller
                # built against the pre-direction endpoint; it just
                # mirrors peer_agent.
                "target_agent": peer_agent,
                "description":  t.description,
                "state":        getattr(t.state, "value", str(t.state)),
                "created_at":   _created_at,
                "completed_at": _completed_at,
                "latency_ms":   latency_ms,
                "result":       t.result if getattr(t, "result", None) else None,
                "error":        getattr(t, "error", None),
                "forked":       bool(md.get("forked", False)),
                "source_query": md.get("source_query") or "",
                "shared_facts_count": int(md.get("shared_facts_count", 0)),
                # When inbound + state=PENDING with awaiting_hitl_id set,
                # this delegation is waiting on a local operator decision —
                # the UI should highlight it.
                "awaiting_hitl_id": md.get("awaiting_hitl_id") or None,
                # A2A Phase 3: when state=awaiting_peer_hitl, this outbound
                # delegation is blocked on the PEER's operator (mode B —
                # approval happens on the peer's console, NOT here). The UI
                # renders a read-only "⏳ waiting for <peer> approval" item
                # and must NOT offer approve/reject locally.
                "peer_hitl_pending":  bool(md.get("peer_hitl_pending", False)),
                "peer_interrupt_id":  md.get("peer_interrupt_id") or None,
            })
        # Newest first — operators usually want the most recent delegation
        out.sort(key=lambda d: d.get("created_at") or "", reverse=True)
        return JSONResponse(content=out)

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
    # System diagnostics + integrations endpoints — webui/routes_system.py
    # ==================================================================
    from webui.routes_system import register_system_routes
    register_system_routes(app, services)


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

async def _batch_decision_fanout(
    batch_id: str,
    decision_kind: str,                  # "approve" | "reject"
    req: "HitlDecisionRequest",
    services: dict,
    _identity_fn,                        # closure from caller
) -> JSONResponse:
    """Common implementation for POST /hitl/batch/{id}/{approve_all,reject_all}.

    Loads the batch, builds one HitlDecision per still-pending child,
    fans out through router.deliver_batch which validates each row and
    invokes the child's resumer. Returns the deliver_batch result dict
    (results + errors) so the caller can show per-child outcomes.

    Rerun-safe: already-decided children are skipped so retrying after
    a partial network failure won't double-trigger anything.
    """
    hitl_core_router = services.get("hitl_core_router")
    if hitl_core_router is None:
        raise HTTPException(404, "hitl_core not configured")

    snapshot = await hitl_core_router.load_batch(batch_id)
    if snapshot is None:
        raise HTTPException(404, f"Batch {batch_id!r} not found")

    op_id = (await _identity_fn()).operator_id
    set_current_operator(op_id)

    from hitl_core.schema import (
        BatchSubmission, HitlDecision, DecisionKind,
    )
    kind = (
        DecisionKind.APPROVE if decision_kind == "approve" else DecisionKind.REJECT
    )

    decisions = []
    for child in snapshot.children:
        # Skip already-decided children — operator may have approved a
        # subset individually before clicking 'approve all'.
        if child.decision is not None:
            continue
        decisions.append(HitlDecision(
            interrupt_id=child.interrupt_id,
            decision=kind,
            operator_id=op_id,
            comment=req.comment,
        ))

    if not decisions:
        return JSONResponse(content={
            "batch_id":   batch_id,
            "applied":    0,
            "message":    "all children already decided",
        })

    submission = BatchSubmission(
        batch_id=batch_id, operator_id=op_id,
        comment=req.comment, decisions=decisions,
    )
    result = await hitl_core_router.deliver_batch(submission)
    return JSONResponse(content=result)


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

        # Detect: was this interrupt a batch member? If yes, the router
        # returns None synchronously (the BatchCoordinator owns resumption
        # — see hitl_core/router.py:_dispatch path 0). The tool execution
        # happens AFTER the LAST batch sibling is approved, in a
        # background task spawned by _raise_tool_hitl_batch. The frontend
        # must (a) NOT close its SSE stream yet, (b) keep showing
        # "executing" state, and (c) consume tool_call/tool_result chunks
        # as they arrive over /hitl/{id}/stream.
        #
        # We signal this with `batch_pending` in the response so the
        # frontend can branch on it. Also include the sibling list so
        # the UI can show "waiting for 1/2 more approvals" if applicable.
        _batch_pending_info = None
        try:
            entry = await hitl_core_router.load(interrupt_id)
            if entry and entry.payload.context_snapshot:
                from hitl_core.batch import BATCH_ID_KEY
                _bid = entry.payload.context_snapshot.get(BATCH_ID_KEY)
                if _bid:
                    _batch_pending_info = {
                        "batch_id": _bid,
                        "interrupt_id": interrupt_id,
                        "message": (
                            "Decision recorded. Tool execution will run "
                            "asynchronously once all batch siblings have "
                            "been decided. Watch the live stream for results."
                        ),
                    }
                    logger.info(
                        "_submit_hitl_decision: batch member detected — "
                        "batch_id=%s, signalling batch_pending to UI",
                        _bid[:12],
                    )
        except Exception as exc:
            logger.debug("batch_pending detection failed: %s", exc)

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
        # Surface batch_pending state to frontend so it knows to keep
        # the SSE stream open and not declare the turn "done".
        if _batch_pending_info is not None:
            result_dict["batch_pending"] = _batch_pending_info
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

        # ── H2 async-HITL follow-up turn (2026-05) ─────────────────────
        # When the resumer returned {"async_resolved": True, ...} the
        # H2 fact has been enqueued into runtime/loop._async_inject_queue
        # by the on_resolved callback, but no turn is currently running
        # for this session to drain it (the original stream finished
        # before the operator decided). Fire a fresh agent turn so the
        # LLM sees the fact (via turn-start drain → state.confirmed_facts)
        # and can revise/complete its answer. The follow-up result is
        # returned in result_dict["async_followup"] so the FE can render
        # it as an additional agent message.
        if _raw_result.get("async_resolved"):
            try:
                _diverged = bool(_raw_result.get("diverged"))
                _session_for_followup = _thread_id_local or req.session_id or ""
                _runtime_loop = services.get("runtime_loop")
                if _runtime_loop is not None and _session_for_followup:
                    # Synthetic follow-up query — the agent_memory turn-start
                    # drain will inject the H2 result into confirmed_facts
                    # before this turn's prompt is assembled. We ALSO thread
                    # the original user query + previous answer back in (debt
                    # #7) so the LLM's follow-up stays connected to what the
                    # user actually asked, instead of a context-less synthetic.
                    # _message_history lives in the create_webui_app closure;
                    # this module-level handler reads it via services (published
                    # there at factory time). Fixes the H2 follow-up NameError.
                    _msg_hist = services.get("_message_history", {})
                    _hist = _msg_hist.get(_session_for_followup, [])
                    _orig_q = ""
                    _prev_a = ""
                    for _m in reversed(_hist):
                        if not _prev_a and _m.get("role") == "assistant":
                            _prev_a = _m.get("content", "")
                        elif not _orig_q and _m.get("role") == "user":
                            _orig_q = _m.get("content", "")
                        if _orig_q and _prev_a:
                            break
                    _div_note = (
                        "注意:实际结果可能跟你之前的假设(permission_ok)不一致,"
                        "请相应修正诊断方向。"
                        if _diverged else
                        "(注:实际结果跟初始假设一致,可确认原诊断。)"
                    )
                    from runtime.loop import build_resumption_query as _brq
                    _followup_query = _brq(
                        "异步 HITL 已返回 RADIUS 检查结果(见 confirmed_facts)。",
                        original_query=_orig_q,
                        previous_answer=_prev_a,
                        divergence_note=_div_note,
                    )
                    from runtime import DelegationMode as _DM
                    _t_followup_start = time.time()
                    _followup = await _runtime_loop.run(
                        query           = _followup_query,
                        session_id      = _session_for_followup,
                        env_context     = {},
                        confirmed_facts = [],            # facts come from inject queue
                        working_set     = None,
                        tool_registry   = services.get("tool_registry") or {},
                        delegation_mode = _DM.FRESH,
                    )
                    _followup_elapsed = round(time.time() - _t_followup_start, 2)
                    _followup_text = getattr(_followup, "final_response", "") or ""
                    result_dict["async_followup"] = {
                        "answer":     _followup_text,
                        "diverged":   _diverged,
                        "elapsed_s":  _followup_elapsed,
                        "session_id": _session_for_followup,
                    }
                    logger.info(
                        "_submit_hitl_decision: H2 follow-up complete — "
                        "interrupt=%s diverged=%s elapsed=%.1fs answer_chars=%d",
                        interrupt_id, _diverged, _followup_elapsed,
                        len(_followup_text),
                    )
                else:
                    logger.info(
                        "_submit_hitl_decision: H2 follow-up skipped — "
                        "runtime_loop=%s session=%r",
                        _runtime_loop is not None, _session_for_followup,
                    )
            except Exception as _fu_exc:
                logger.warning(
                    "_submit_hitl_decision: H2 follow-up failed for "
                    "interrupt=%s: %s",
                    interrupt_id, _fu_exc,
                )
                result_dict["async_followup"] = {"error": str(_fu_exc)}

        # Continue to the LLM synthesis section (line below).
    else:
        # No HITL router wired. Legacy LangGraph backend was retired
        # (see AUDIT_REPORT.md task A) — only hitl_core is supported now.
        raise HTTPException(
            status_code=503,
            detail="HITL_BACKEND=core is required; legacy backend retired",
        )

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
            # For BATCH members: don't run LLM-bound fact extraction on
            # every sibling approval. The batch executor will write a
            # proper summary once tools complete. importance=0.4 means
            # the chunk is stored but no LLM distillation runs — fact
            # extraction takes 30s per call (Ollama serialized), so on
            # a 2-child batch this added 60s+ of operator wait time
            # AFTER each approve click. The actual tool execution
            # happens in the batch executor's background task.
            _is_batch_member = _batch_pending_info is not None
            if _is_batch_member:
                _writeback_importance = 0.4
            else:
                _writeback_importance = (
                    0.7 if _decision in ("approve", "choose", "answer") else 0.5
                )
            new_facts = await _memory.after_turn(
                session_id      = session_id,
                user_text       = _user_query,
                assistant_text  = assistant_text,
                tool_calls      = [{"tool": _tool_name}] if _tool_name else [],
                importance      = _writeback_importance,
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
    #
    # SKIP for BATCH members: SkillEvolver is an LLM call (8s+ on local
    # Ollama) that blocks the POST response. For an N-target batch the
    # operator clicks approve N times — running SkillEvolver N times
    # for the SAME logical request is wasteful AND each invocation
    # delays its member's POST visible response by another LLM
    # round-trip. The skill fires ONCE after the whole batch resolves —
    # wired in hitl_executor.py _batch_execute_after_resolution, which
    # calls self._skill_evolver.after_task with the unioned tools_used
    # and a representative solution summary across all successful children.
    # See `set_skill_evolver` in HitlExecutor for the deferred injection
    # point used by main.py.
    _is_batch_member_for_evolver = _batch_pending_info is not None
    if (
        _decision in ("approve", "choose", "answer")
        and _tool_name and _tool_name != "agent_loop"
        and not _is_batch_member_for_evolver
    ):
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
    elif _is_batch_member_for_evolver:
        logger.debug(
            "_submit_hitl_decision: skipping SkillEvolver for batch member "
            "(would block POST; batch executor will handle the request as a whole)"
        )

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