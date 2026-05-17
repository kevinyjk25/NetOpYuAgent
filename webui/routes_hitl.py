"""
webui/routes_hitl.py — HITL endpoints registered on the WebUI FastAPI app.

EXTRACTED FROM webui/backend.py during the post-batch-HITL audit (see
AUDIT_REPORT.md task D-4). All HITL-related routes live here so backend.py
stays focused on the app factory + chat/tool/session/memory routes.

Public API:
    register_hitl_routes(app, services)

Behaviour is identical to the inline definitions that used to live in
backend.py — same paths, same responses, same auth flow. The functions
`_submit_hitl_decision` and `_batch_decision_fanout` remain in
backend.py because they're also reachable from the chat-stream HITL
intercept path; this module imports them.

Routes registered:
    GET   /hitl/pending
    POST  /hitl/{interrupt_id}/approve
    POST  /hitl/{interrupt_id}/reject
    POST  /hitl/{interrupt_id}/edit
    POST  /hitl/{interrupt_id}/choose
    POST  /hitl/{interrupt_id}/answer
    GET   /hitl/{interrupt_id}/stream
    GET   /hitl/batch/{batch_id}
    POST  /hitl/batch/{batch_id}/approve_all
    POST  /hitl/batch/{batch_id}/reject_all
"""
from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

# Pydantic body-model imports MUST be at module level (not inside
# register_hitl_routes). FastAPI resolves the `req: HitlDecisionRequest`
# annotation by looking up the name in the route function's enclosing
# *module globals* — if HitlDecisionRequest is only imported inside the
# register function, FastAPI can't find it, falls back to treating
# `req` as a query parameter, and every POST returns 422
# {"loc": ("query", "req"), "msg": "Field required"}.
#
# Models live in webui/schemas.py (a dependency-free module) to avoid
# a circular import via webui/backend.py.
from webui.schemas import HitlDecisionRequest

logger = logging.getLogger(__name__)


def register_hitl_routes(app: FastAPI, services: dict[str, Any]) -> None:
    """Attach all /hitl/* endpoints to `app`. Idempotent within an app
    instance — FastAPI raises on duplicate path registration so don't
    call twice on the same app."""
    # Late imports for helpers (used only inside route bodies, where
    # module-globals visibility isn't required).
    from webui.backend import (
        _identity,
        _submit_hitl_decision,
        _batch_decision_fanout,
    )

    @app.get("/hitl/pending")
    async def list_pending_hitl() -> JSONResponse:
        # Single backend path (hitl_core). The legacy LangGraph-based
        # hitl_router was retired in the audit refactor; see
        # AUDIT_REPORT.md task A for the rationale.
        hitl_core_router = services.get("hitl_core_router")
        if hitl_core_router is not None:
            entries = await hitl_core_router.list_pending(limit=100)
            # Resolve batch_id for each entry (None if not part of a batch).
            # Older frontends ignore unknown fields; newer ones can group
            # cards by batch_id to render a single "Approve all" affordance.
            from hitl_core.batch import BATCH_ID_KEY
            result = []
            for entry in entries:
                p = entry.payload
                _batch_id = (
                    p.context_snapshot.get(BATCH_ID_KEY)
                    if p.context_snapshot else None
                )
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
                    "choices":              [c.model_dump() for c in (p.choices or [])],
                    "clarification_fields": [f.model_dump() for f in (p.clarification_fields or [])],
                    "editable_param_keys":  list(p.editable_param_keys or []),
                    # Present when this interrupt is part of a batch
                    # (multiple destructive calls in one LLM turn). Frontends
                    # that don't know about batches still render N independent
                    # cards as before; newer UIs can group on this key.
                    "batch_id":             _batch_id,
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

        # No HITL router available — happens only if HITL was disabled entirely
        # (the legacy LangGraph backend was deleted along with the hitl/* stub
        # package; see AUDIT_REPORT.md task A).
        logger.warning("/hitl/pending: no HITL router wired")
        return JSONResponse(content=[])

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
        req.operator_id = (await _identity()).operator_id
        return await _submit_hitl_decision(
            interrupt_id, "reject", req, services
        )

    @app.post("/hitl/{interrupt_id}/edit")
    async def edit_hitl(
        interrupt_id: str,
        req: HitlDecisionRequest,
    ) -> JSONResponse:
        req.operator_id = (await _identity()).operator_id
        return await _submit_hitl_decision(
            interrupt_id, "edit", req, services
        )

    # ── Batch HITL endpoints ──────────────────────────────────────
    # When the LLM emits multiple destructive [TOOL:] calls in one turn
    # (e.g. push_config to 2+ devices), the executor raises a HitlBatch
    # so children appear together. These endpoints let operators act on
    # the whole group at once instead of clicking through N cards.
    # Single-decision endpoints (/hitl/{id}/approve etc) still work on
    # individual children — these batch ones are a convenience layer.

    @app.get("/hitl/batch/{batch_id}")
    async def get_hitl_batch(batch_id: str) -> JSONResponse:
        """Return the batch envelope + all child payloads + decision counts."""
        hitl_core_router = services.get("hitl_core_router")
        if hitl_core_router is None:
            raise HTTPException(404, "hitl_core not configured")
        snapshot = await hitl_core_router.load_batch(batch_id)
        if snapshot is None:
            raise HTTPException(404, f"Batch {batch_id!r} not found")
        return JSONResponse(content={
            "batch":         snapshot.batch.model_dump(),
            "children":      [c.model_dump() for c in snapshot.children],
            "decided_count": snapshot.decided_count,
            "pending_count": snapshot.pending_count,
        })

    @app.post("/hitl/batch/{batch_id}/approve_all")
    async def approve_batch_all(
        batch_id: str, req: HitlDecisionRequest,
    ) -> JSONResponse:
        """Approve every still-pending child of a batch in one call."""
        return await _batch_decision_fanout(
            batch_id, "approve", req, services, _identity
        )

    @app.post("/hitl/batch/{batch_id}/reject_all")
    async def reject_batch_all(
        batch_id: str, req: HitlDecisionRequest,
    ) -> JSONResponse:
        """Reject every still-pending child of a batch in one call."""
        return await _batch_decision_fanout(
            batch_id, "reject", req, services, _identity
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