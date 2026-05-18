"""
hitl_core.transport.http_adapter — FastAPI router factory.

OPTIONAL component. Building it requires `pip install fastapi`.
hitl_core itself does not depend on fastapi; this module is purely
a convenience for hosts that want the canonical HTTP API shape
without writing it from scratch.

Endpoints (all prefixed with the router's prefix):

  GET  /pending                        list pending interrupts
  GET  /pending/batches                list pending batches
  GET  /{interrupt_id}                 load one interrupt's payload
  POST /{interrupt_id}/approve         operator approves
  POST /{interrupt_id}/reject          operator rejects
  POST /{interrupt_id}/edit            operator approves with parameter_patch
  POST /{interrupt_id}/choose          operator selects from payload.choices
  POST /{interrupt_id}/answer          operator submits clarification
  GET  /batch/{batch_id}               load batch snapshot
  POST /batch/{batch_id}/submit        submit a BatchSubmission

All requests/responses are JSON. Validation errors return HTTP 422
with a structured body: {"error": "...", "field": "..."}.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..router import DecisionValidationError, HitlRouter, ResumeError
from ..schema import (
    BatchSubmission,
    DecisionKind,
    HitlDecision,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Factory — returns a configured APIRouter
# ---------------------------------------------------------------------------

def build_http_router(
    hitl_router: HitlRouter,
    *,
    prefix: str = "/hitl",
    tags: Optional[list[str]] = None,
):
    """Build a FastAPI APIRouter wired to the given HitlRouter.

    Caller mounts it on their FastAPI app:

        app.include_router(build_http_router(hitl_router))

    Lazy import of fastapi so hitl_core's hard deps stay minimal.
    """
    try:
        from fastapi import APIRouter, Body, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError(
            "build_http_router requires fastapi: pip install fastapi"
        ) from exc

    api = APIRouter(prefix=prefix, tags=tags or ["hitl"])

    # ── Request bodies ──────────────────────────────────────────────
    # We accept partial decision shapes here for ergonomics; the router
    # synthesises a full HitlDecision before dispatch.

    class _DecisionBody(BaseModel):
        operator_id: str = "unknown"
        comment: Optional[str] = None

    class _EditBody(_DecisionBody):
        parameter_patch: dict[str, Any] = Field(default_factory=dict)

    class _ChooseBody(_DecisionBody):
        selected_choice_id: str

    class _AnswerBody(_DecisionBody):
        clarification_answers: dict[str, str] = Field(default_factory=dict)

    # ── Helpers ─────────────────────────────────────────────────────

    def _wrap_validation(exc: DecisionValidationError) -> HTTPException:
        return HTTPException(status_code=422, detail={"error": str(exc)})

    def _wrap_resume_err(exc: ResumeError) -> HTTPException:
        return HTTPException(status_code=409, detail={"error": str(exc)})

    async def _do(decision: HitlDecision) -> dict[str, Any]:
        try:
            return await hitl_router.deliver(decision)
        except DecisionValidationError as exc:
            raise _wrap_validation(exc)
        except ResumeError as exc:
            raise _wrap_resume_err(exc)

    # ── Routes ──────────────────────────────────────────────────────

    @api.get("/pending")
    async def list_pending(
        limit: int = 100, thread_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        entries = await hitl_router.list_pending(
            limit=limit, thread_id=thread_id,
        )
        return [e.model_dump(mode="json") for e in entries]

    @api.get("/pending/batches")
    async def list_pending_batches(
        limit: int = 50, thread_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        batches = await hitl_router.list_pending_batches(
            limit=limit, thread_id=thread_id,
        )
        return [b.model_dump(mode="json") for b in batches]

    @api.get("/{interrupt_id}")
    async def get_interrupt(interrupt_id: str) -> dict[str, Any]:
        entry = await hitl_router.load(interrupt_id)
        if entry is None:
            raise HTTPException(status_code=404, detail={"error": "not found"})
        return entry.model_dump(mode="json")

    @api.post("/{interrupt_id}/approve")
    async def approve(
        interrupt_id: str, body: _DecisionBody = Body(default=None),
    ) -> dict[str, Any]:
        body = body or _DecisionBody()
        return await _do(HitlDecision(
            interrupt_id=interrupt_id,
            decision=DecisionKind.APPROVE,
            operator_id=body.operator_id,
            comment=body.comment,
        ))

    @api.post("/{interrupt_id}/reject")
    async def reject(
        interrupt_id: str, body: _DecisionBody = Body(default=None),
    ) -> dict[str, Any]:
        body = body or _DecisionBody()
        return await _do(HitlDecision(
            interrupt_id=interrupt_id,
            decision=DecisionKind.REJECT,
            operator_id=body.operator_id,
            comment=body.comment,
        ))

    @api.post("/{interrupt_id}/edit")
    async def edit(
        interrupt_id: str, body: _EditBody,
    ) -> dict[str, Any]:
        return await _do(HitlDecision(
            interrupt_id=interrupt_id,
            decision=DecisionKind.EDIT,
            operator_id=body.operator_id,
            comment=body.comment,
            parameter_patch=body.parameter_patch,
        ))

    @api.post("/{interrupt_id}/choose")
    async def choose(
        interrupt_id: str, body: _ChooseBody,
    ) -> dict[str, Any]:
        return await _do(HitlDecision(
            interrupt_id=interrupt_id,
            decision=DecisionKind.CHOOSE,
            operator_id=body.operator_id,
            comment=body.comment,
            selected_choice_id=body.selected_choice_id,
        ))

    @api.post("/{interrupt_id}/answer")
    async def answer(
        interrupt_id: str, body: _AnswerBody,
    ) -> dict[str, Any]:
        return await _do(HitlDecision(
            interrupt_id=interrupt_id,
            decision=DecisionKind.ANSWER,
            operator_id=body.operator_id,
            comment=body.comment,
            clarification_answers=body.clarification_answers,
        ))

    @api.get("/batch/{batch_id}")
    async def get_batch(batch_id: str) -> dict[str, Any]:
        snap = await hitl_router.load_batch(batch_id)
        if snap is None:
            raise HTTPException(status_code=404, detail={"error": "not found"})
        return snap.model_dump(mode="json")

    @api.post("/batch/{batch_id}/submit")
    async def submit_batch(
        batch_id: str, submission: BatchSubmission,
    ) -> dict[str, Any]:
        # Allow path param to override / set batch_id (clients sometimes
        # only fill the path).
        if not submission.batch_id:
            submission.batch_id = batch_id
        if submission.batch_id != batch_id:
            raise HTTPException(
                status_code=422,
                detail={"error": f"path {batch_id} != body {submission.batch_id}"},
            )
        try:
            return await hitl_router.deliver_batch(submission)
        except DecisionValidationError as exc:
            raise _wrap_validation(exc)

    return api