"""
hitl/router.py
--------------
FastAPI router that exposes all HITL HTTP endpoints.

Mount in your main FastAPI app:

    from hitl.router import create_hitl_router
    router = create_hitl_router(decision_router=..., audit=..., sse_channel=...)
    app.include_router(router, prefix="/hitl")

Endpoints
---------
  POST /hitl/decisions/{interrupt_id}     Accept operator decision
  GET  /hitl/interrupts/{interrupt_id}    Fetch interrupt details
  GET  /hitl/interrupts                   List recent interrupts
  GET  /hitl/audit/{interrupt_id}         Fetch audit trail for an interrupt
  GET  /hitl/stream                       SSE stream of live HITL events
  GET  /hitl/health                       Health check
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

from .audit import HitlAuditService
from .decision import DecisionResult, HitlDecisionRouter
from .review import WebDashboardSSEChannel, WebSocketHitlManager, get_sse_channel, get_ws_manager
from .schemas import HitlDecision, HitlPayload, InterruptState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dependency injection helpers
# ---------------------------------------------------------------------------

# These are set by create_hitl_router; injected via FastAPI Depends
_router_instance: Optional[HitlDecisionRouter] = None
_audit_instance:  Optional[HitlAuditService]   = None
_sse_instance:    Optional[WebDashboardSSEChannel] = None
_ws_manager_instance: Optional[WebSocketHitlManager] = None


def _get_decision_router() -> HitlDecisionRouter:
    if _router_instance is None:
        raise RuntimeError("HitlDecisionRouter not initialised")
    return _router_instance


def _get_audit() -> HitlAuditService:
    if _audit_instance is None:
        raise RuntimeError("HitlAuditService not initialised")
    return _audit_instance


def _get_sse() -> WebDashboardSSEChannel:
    return _sse_instance or get_sse_channel()


def _get_ws_manager() -> WebSocketHitlManager:
    return _ws_manager_instance or get_ws_manager()


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------

def create_hitl_router(
    decision_router: HitlDecisionRouter,
    audit: HitlAuditService,
    sse_channel: Optional[WebDashboardSSEChannel] = None,
    ws_manager: Optional[WebSocketHitlManager] = None,
) -> APIRouter:
    """
    Build and return the FastAPI APIRouter for all HITL endpoints.

    Parameters
    ----------
    decision_router : HitlDecisionRouter
        Injected decision handler (carries the compiled LangGraph graph).
    audit : HitlAuditService
        Audit log service.
    sse_channel : WebDashboardSSEChannel, optional
        SSE broadcast channel; uses the global singleton if None.
    ws_manager : WebSocketHitlManager, optional
        WebSocket connection manager; uses the global singleton if None.
    """
    global _router_instance, _audit_instance, _sse_instance, _ws_manager_instance
    _router_instance     = decision_router
    _audit_instance      = audit
    _sse_instance        = sse_channel or get_sse_channel()
    _ws_manager_instance = ws_manager or get_ws_manager()

    api = APIRouter(tags=["HITL"])

    # ------------------------------------------------------------------
    # POST /decisions/{interrupt_id}
    # ------------------------------------------------------------------

    @api.post("/decisions/{interrupt_id}", summary="Submit operator decision")
    async def submit_decision(
        interrupt_id: str,
        body: HitlDecision,
        router: HitlDecisionRouter = Depends(_get_decision_router),
    ) -> JSONResponse:
        """
        Accept an operator decision for a pending HITL interrupt.

        The decision is validated, the LangGraph graph is resumed (or aborted),
        and an audit record is written.

        Body
        ----
        HitlDecision JSON:
          {
            "interrupt_id": "...",
            "thread_id": "...",
            "decision": "approve|reject|edit|escalate",
            "operator_id": "sre-alice",
            "comment": "Looks safe, proceeding.",
            "parameter_patch": {"rolling": false}   // for 'edit' only
          }
        """
        if body.interrupt_id != interrupt_id:
            raise HTTPException(
                status_code=422,
                detail="interrupt_id in path and body must match",
            )

        result: DecisionResult = await router.handle_decision(body)

        if result.error:
            raise HTTPException(status_code=409, detail=result.error)

        return JSONResponse(content=result.to_dict())

    # ------------------------------------------------------------------
    # GET /interrupts/{interrupt_id}
    # ------------------------------------------------------------------

    @api.get("/interrupts/{interrupt_id}", summary="Get interrupt details")
    async def get_interrupt(
        interrupt_id: str,
        router: HitlDecisionRouter = Depends(_get_decision_router),
    ) -> JSONResponse:
        payload: Optional[HitlPayload] = router._payload_store.get(interrupt_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="Interrupt not found")
        return JSONResponse(content=payload.model_dump())

    # ------------------------------------------------------------------
    # GET /interrupts   (list pending / recent)
    # ------------------------------------------------------------------

    @api.get("/interrupts", summary="List interrupts")
    async def list_interrupts(
        status: Optional[str] = None,
        limit: int = 20,
        router: HitlDecisionRouter = Depends(_get_decision_router),
    ) -> JSONResponse:
        """
        List interrupts from the in-process store.
        Filter by ``status`` (pending | resolved | timed_out | escalated).
        """
        all_payloads = list(router._payload_store.values())

        if status:
            try:
                target = InterruptState(status)
                all_payloads = [p for p in all_payloads if p.status == target]
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status. Must be one of: {[s.value for s in InterruptState]}",
                )

        # Sort by created_at descending
        all_payloads.sort(key=lambda p: p.created_at, reverse=True)
        return JSONResponse(content=[p.model_dump() for p in all_payloads[:limit]])

    # ------------------------------------------------------------------
    # GET /audit/{interrupt_id}
    # ------------------------------------------------------------------

    @api.get("/audit/{interrupt_id}", summary="Get audit trail")
    async def get_audit_trail(
        interrupt_id: str,
        audit_svc: HitlAuditService = Depends(_get_audit),
    ) -> JSONResponse:
        records = await audit_svc.get_by_interrupt(interrupt_id)
        if not records:
            raise HTTPException(
                status_code=404,
                detail=f"No audit records for interrupt_id={interrupt_id!r}",
            )
        return JSONResponse(content=[r.model_dump() for r in records])

    # ------------------------------------------------------------------
    # GET /audit/thread/{thread_id}
    # ------------------------------------------------------------------

    @api.get("/audit/thread/{thread_id}", summary="Get audit trail by thread")
    async def get_audit_by_thread(
        thread_id: str,
        audit_svc: HitlAuditService = Depends(_get_audit),
    ) -> JSONResponse:
        records = await audit_svc.get_by_thread(thread_id)
        return JSONResponse(content=[r.model_dump() for r in records])

    # ------------------------------------------------------------------
    # GET /stream  — SSE live feed
    # ------------------------------------------------------------------

    @api.get("/stream", summary="SSE live event stream")
    async def sse_stream(
        request: Request,
        sse: WebDashboardSSEChannel = Depends(_get_sse),
    ) -> StreamingResponse:
        """
        Server-Sent Events stream delivering live HITL events to the dashboard.

        Connect with EventSource in the browser:
            const es = new EventSource('/hitl/stream');
            es.onmessage = (e) => console.log(JSON.parse(e.data));
        """
        queue = sse.subscribe()

        async def generate() -> AsyncIterator[str]:
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=30.0)
                        yield f"data: {event}\n\n"
                    except asyncio.TimeoutError:
                        # Keepalive ping
                        yield ": keepalive\n\n"
            finally:
                sse.unsubscribe(queue)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # ------------------------------------------------------------------
    # WebSocket /ws  — bidirectional channel for external agent systems
    # ------------------------------------------------------------------

    @api.websocket("/ws")
    async def hitl_websocket(
        websocket: WebSocket,
        client_id: Optional[str] = None,
    ) -> None:
        """
        WebSocket endpoint for external agent systems that need to:
          - RECEIVE live HITL interrupt events (server → client)
          - SEND decisions back over the same connection (client → server)

        Connect
        -------
            ws://host/hitl/ws                        # auto-generated client_id
            ws://host/hitl/ws?client_id=my-agent     # explicit client_id

        Server → client messages
        -------------------------
        Interrupt event:
            {
                "type": "hitl_interrupt",
                "interrupt_id": "...",
                "thread_id": "...",
                "trigger_kind": "destructive_action",
                "risk_level": "high",
                "intent_summary": "Restart payments-service in prod",
                "proposed_action": {"action_type": "restart_service", ...},
                "sla_seconds": 600,
                "created_at": "2025-01-01T00:00:00Z"
            }

        Keepalive ping (every 30 s):
            {"type": "ping"}

        Acknowledgement of a submitted decision:
            {"type": "decision_ack", "interrupt_id": "...", "resumed": true, "error": null}

        Client → server messages
        -------------------------
        Submit a decision:
            {
                "type": "hitl_decision",
                "interrupt_id": "...",
                "thread_id": "...",
                "decision": "approve",
                "operator_id": "external-agent-sre",
                "comment": "Approved by automated SRE agent",
                "parameter_patch": {}
            }

        Keepalive pong (optional):
            {"type": "pong"}
        """
        import uuid as _uuid

        ws_manager: WebSocketHitlManager = _get_ws_manager()
        router:     HitlDecisionRouter   = _get_decision_router()

        # Assign a client_id if not provided via query param
        effective_client_id = client_id or str(_uuid.uuid4())

        await ws_manager.connect(websocket, effective_client_id)

        # Send connection acknowledgement
        await websocket.send_text(json.dumps({
            "type":      "connected",
            "client_id": effective_client_id,
            "message":   "Connected to HITL WebSocket. Awaiting interrupt events.",
        }))

        try:
            while True:
                # Wait for a message from the client (with keepalive ping every 30 s)
                try:
                    raw = await asyncio.wait_for(
                        websocket.receive_text(), timeout=30.0
                    )
                except asyncio.TimeoutError:
                    # Send keepalive ping; client may optionally reply with pong
                    await websocket.send_text(json.dumps({"type": "ping"}))
                    continue

                # Parse incoming message
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await websocket.send_text(json.dumps({
                        "type":  "error",
                        "error": "Invalid JSON",
                    }))
                    continue

                msg_type = msg.get("type", "")

                if msg_type == "pong":
                    # Client keepalive reply — no action needed
                    continue

                elif msg_type == "hitl_decision":
                    # Build and process a HitlDecision from the WebSocket message
                    try:
                        decision = HitlDecision(
                            interrupt_id      = msg["interrupt_id"],
                            thread_id         = msg["thread_id"],
                            decision          = msg["decision"],
                            operator_id       = msg.get("operator_id", effective_client_id),
                            comment           = msg.get("comment"),
                            parameter_patch   = msg.get("parameter_patch"),
                            escalation_target = msg.get("escalation_target"),
                        )
                    except (KeyError, ValueError) as exc:
                        await websocket.send_text(json.dumps({
                            "type":  "error",
                            "error": f"Invalid decision payload: {exc}",
                        }))
                        continue

                    result: DecisionResult = await router.handle_decision(decision)
                    await websocket.send_text(json.dumps({
                        "type":         "decision_ack",
                        "interrupt_id": decision.interrupt_id,
                        "resumed":      result.resumed,
                        "error":        result.error,
                    }))
                    logger.info(
                        "WebSocket decision received interrupt_id=%s decision=%s "
                        "client_id=%s resumed=%s",
                        decision.interrupt_id, decision.decision,
                        effective_client_id, result.resumed,
                    )

                else:
                    await websocket.send_text(json.dumps({
                        "type":  "error",
                        "error": f"Unknown message type: {msg_type!r}. "
                                 f"Expected: hitl_decision | pong",
                    }))

        except WebSocketDisconnect:
            logger.info(
                "WebSocket client disconnected normally client_id=%s",
                effective_client_id,
            )
        except Exception as exc:
            logger.exception(
                "WebSocket error client_id=%s: %s", effective_client_id, exc
            )
        finally:
            await ws_manager.disconnect(effective_client_id)

    # ------------------------------------------------------------------
    # GET /health
    # ------------------------------------------------------------------

    @api.get("/health", summary="HITL module health check")
    async def health(
        router: HitlDecisionRouter = Depends(_get_decision_router),
    ) -> JSONResponse:
        pending = sum(
            1 for p in router._payload_store.values()
            if p.status == InterruptState.PENDING
        )
        ws_manager = _get_ws_manager()
        return JSONResponse({
            "status": "ok",
            "pending_interrupts":      pending,
            "websocket_clients":       ws_manager.connection_count,
            "module": "hitl",
        })

    return api