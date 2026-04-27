"""
hitl/decision.py
----------------
Layer 4 — Decision intake and graph resumption.

HitlDecisionRouter
  - Receives operator decisions (approve / reject / edit / escalate)
  - Validates the decision against the stored interrupt
  - Applies graph.update_state() for edits
  - Resumes the LangGraph graph via graph.ainvoke(None, config)
  - Writes audit records

HitlTimeoutWatchdog
  - Background asyncio task
  - Polls pending interrupts every 60 s
  - Auto-escalates when SLA is breached

Usage (from FastAPI route)
--------------------------
    router = HitlDecisionRouter(graph=compiled_graph, audit=audit_service)
    result = await router.handle_decision(decision)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from .audit import HitlAuditService
from .schemas import (
    AuditEventKind,
    DecisionKind,
    HitlAuditRecord,
    HitlDecision,
    HitlPayload,
    InterruptState,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decision outcome
# ---------------------------------------------------------------------------

class DecisionResult:
    """Returned by HitlDecisionRouter.handle_decision()."""
    def __init__(
        self,
        interrupt_id: str,
        decision: DecisionKind,
        resumed: bool,
        graph_result: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        self.interrupt_id = interrupt_id
        self.decision     = decision
        self.resumed      = resumed
        self.graph_result = graph_result
        self.error        = error

    def to_dict(self) -> dict[str, Any]:
        d = {
            "interrupt_id": self.interrupt_id,
            "decision":     self.decision.value,
            "resumed":      self.resumed,
            "error":        self.error,
        }
        # Include tool execution result when callback ran successfully
        if self.graph_result:
            if "result" in self.graph_result:
                d["tool_result"] = self.graph_result["result"]
            if "tool" in self.graph_result:
                d["tool_name"] = self.graph_result["tool"]
            if "error" in self.graph_result and not d["error"]:
                d["error"] = self.graph_result["error"]
        return d


# ---------------------------------------------------------------------------
# Decision Router
# ---------------------------------------------------------------------------

class HitlDecisionRouter:
    """
    Central class for processing operator decisions.

    Parameters
    ----------
    graph:
        The compiled LangGraph StateGraph (from build_hitl_graph()).
    audit:
        HitlAuditService for writing audit records.
    payload_store:
        Dict-like store mapping interrupt_id → HitlPayload.
        In production, back this with Redis or PostgreSQL.
    """

    def __init__(
        self,
        graph: Any,
        audit: "HitlAuditService",
        payload_store: Optional[dict[str, HitlPayload]] = None,
    ) -> None:
        self._graph         = graph
        self._audit         = audit
        self._payload_store:    dict[str, HitlPayload] = payload_store or {}
        # interrupt_id → (callback, registered_at) — pruned after 30 min
        self._direct_callbacks: dict = {}  # {interrupt_id: (callback, monotonic_ts)}

    # ------------------------------------------------------------------
    # Public API called by FastAPI route
    # ------------------------------------------------------------------

    async def register_interrupt(self, payload: HitlPayload) -> None:
        """Called when a new HITL interrupt fires. Stores the payload. Idempotent."""
        if payload.interrupt_id in self._payload_store:
            logger.debug("register_interrupt: %s already registered, skipping",
                        payload.interrupt_id[:12])
            return
        self._payload_store[payload.interrupt_id] = payload
        logger.info(
            "HITL registered: interrupt_id=%s trigger=%s risk=%s status=%s store_size=%d",
            payload.interrupt_id,
            getattr(payload.trigger_kind, "value", payload.trigger_kind),
            getattr(payload.risk_level, "value", payload.risk_level),
            getattr(payload.status, "value", payload.status),
            len(self._payload_store),
        )
        await self._audit.write(HitlAuditRecord(
            interrupt_id=payload.interrupt_id,
            thread_id=payload.thread_id,
            event_kind=AuditEventKind.INTERRUPT_FIRED,
            actor="agent",
            payload=payload.model_dump(),
        ))

    async def handle_decision(self, decision: HitlDecision) -> DecisionResult:
        """
        Process an operator's decision and resume (or abort) the graph.

        Steps
        -----
        1. Validate the interrupt exists and is still pending.
        2. Write a DECISION_RECEIVED audit record.
        3. For EDIT decisions: patch the graph state via update_state().
        4. For APPROVE / EDIT: resume the graph by calling ainvoke(None).
        5. For REJECT / ESCALATE / TIMEOUT: mark the interrupt resolved without resuming.
        6. Write a GRAPH_RESUMED or GRAPH_ABORTED audit record.
        """
        payload = self._payload_store.get(decision.interrupt_id)
        if payload is None:
            logger.warning("Decision for unknown interrupt_id=%s", decision.interrupt_id)
            return DecisionResult(
                interrupt_id=decision.interrupt_id,
                decision=decision.decision,
                resumed=False,
                error=f"interrupt_id {decision.interrupt_id!r} not found",
            )

        if payload.status != InterruptState.PENDING:
            return DecisionResult(
                interrupt_id=decision.interrupt_id,
                decision=decision.decision,
                resumed=False,
                error=f"Interrupt already {payload.status.value}",
            )

        # Mark resolved
        payload.status      = InterruptState.RESOLVED
        payload.resolved_at = datetime.now(timezone.utc).isoformat()
        payload.resolved_by = decision.operator_id

        await self._audit.write(HitlAuditRecord(
            interrupt_id=decision.interrupt_id,
            thread_id=payload.thread_id,
            event_kind=AuditEventKind.DECISION_RECEIVED,
            actor=decision.operator_id,
            payload=decision.model_dump(),
        ))

        thread_cfg = {"configurable": {"thread_id": decision.thread_id}}

        if decision.decision == DecisionKind.APPROVE:
            return await self._resume(decision, payload, thread_cfg)

        elif decision.decision == DecisionKind.EDIT:
            return await self._resume_with_edit(decision, payload, thread_cfg)

        elif decision.decision == DecisionKind.REJECT:
            return await self._abort(
                decision, payload, thread_cfg, reason="operator_rejected"
            )

        elif decision.decision == DecisionKind.ESCALATE:
            return await self._escalate(decision, payload, thread_cfg)

        elif decision.decision == DecisionKind.TIMEOUT:
            return await self._abort(
                decision, payload, thread_cfg, reason="sla_timeout"
            )

        else:
            return DecisionResult(
                interrupt_id=decision.interrupt_id,
                decision=decision.decision,
                resumed=False,
                error=f"Unknown decision kind: {decision.decision}",
            )

    # ------------------------------------------------------------------
    # Private handlers
    # ------------------------------------------------------------------

    async def _resume(
        self,
        decision: HitlDecision,
        payload: HitlPayload,
        thread_cfg: dict,
    ) -> DecisionResult:
        """Resume the graph after APPROVE — or run direct callback for non-graph interrupts."""
        try:
            # Direct callback path (force_hitl_tool interrupts that bypassed LangGraph)
            _cb_entry = self._direct_callbacks.pop(decision.interrupt_id, None)
            callback = _cb_entry[0] if _cb_entry else None
            if callback:
                logger.info("_resume: running direct callback for interrupt %s", decision.interrupt_id[:12])
                graph_result = await callback()
                await self._audit.write(HitlAuditRecord(
                    interrupt_id=decision.interrupt_id,
                    thread_id=payload.thread_id,
                    event_kind=AuditEventKind.GRAPH_RESUMED,
                    actor=decision.operator_id,
                    payload={"decision": decision.decision.value, "path": "direct_callback"},
                ))
                return DecisionResult(
                    interrupt_id=decision.interrupt_id,
                    decision=decision.decision,
                    resumed=True,
                    graph_result=graph_result or {},
                )
            # LangGraph resume path
            self._graph.update_state(
                thread_cfg,
                {"hitl_decision": decision.model_dump()},
            )
            graph_result: dict = await self._graph.ainvoke(None, thread_cfg)

            await self._audit.write(HitlAuditRecord(
                interrupt_id=decision.interrupt_id,
                thread_id=payload.thread_id,
                event_kind=AuditEventKind.GRAPH_RESUMED,
                actor=decision.operator_id,
                payload={"decision": decision.decision.value},
            ))
            return DecisionResult(
                interrupt_id=decision.interrupt_id,
                decision=decision.decision,
                resumed=True,
                graph_result=graph_result,
            )
        except Exception as exc:
            logger.exception("Graph resume failed: %s", exc)
            return DecisionResult(
                interrupt_id=decision.interrupt_id,
                decision=decision.decision,
                resumed=False,
                error=str(exc),
            )

    async def _resume_with_edit(
        self,
        decision: HitlDecision,
        payload: HitlPayload,
        thread_cfg: dict,
    ) -> DecisionResult:
        """Patch state with operator edits, then resume."""
        patch: dict[str, Any] = {"hitl_decision": decision.model_dump()}

        if decision.parameter_patch:
            # Merge the operator's edits into the proposed_action parameters
            raw_action = dict(payload.proposed_action.model_dump())
            raw_action["parameters"] = {
                **raw_action.get("parameters", {}),
                **decision.parameter_patch,
            }
            patch["proposed_action"] = raw_action
            logger.info(
                "Operator edit — interrupt_id=%s patch=%s",
                decision.interrupt_id,
                decision.parameter_patch,
            )

        self._graph.update_state(thread_cfg, patch)
        return await self._resume(decision, payload, thread_cfg)

    async def _abort(
        self,
        decision: HitlDecision,
        payload: HitlPayload,
        thread_cfg: dict,
        reason: str,
    ) -> DecisionResult:
        """Abort the graph without resuming."""
        try:
            # Signal the graph to end without executing
            self._graph.update_state(
                thread_cfg,
                {
                    "hitl_decision": decision.model_dump(),
                    "__end__": True,
                },
            )
        except Exception as exc:
            logger.warning("Graph abort state update failed: %s", exc)

        await self._audit.write(HitlAuditRecord(
            interrupt_id=decision.interrupt_id,
            thread_id=payload.thread_id,
            event_kind=AuditEventKind.GRAPH_ABORTED,
            actor=decision.operator_id,
            payload={"reason": reason, "comment": decision.comment},
        ))
        return DecisionResult(
            interrupt_id=decision.interrupt_id,
            decision=decision.decision,
            resumed=False,
        )

    async def _escalate(
        self,
        decision: HitlDecision,
        payload: HitlPayload,
        thread_cfg: dict,
    ) -> DecisionResult:
        """Escalate to a higher-tier operator; abort the current graph run."""
        payload.status = InterruptState.ESCALATED

        await self._audit.write(HitlAuditRecord(
            interrupt_id=decision.interrupt_id,
            thread_id=payload.thread_id,
            event_kind=AuditEventKind.ESCALATION_SENT,
            actor=decision.operator_id,
            payload={
                "target": decision.escalation_target,
                "comment": decision.comment,
            },
        ))
        logger.info(
            "Escalation — interrupt_id=%s → target=%s",
            decision.interrupt_id,
            decision.escalation_target,
        )
        # In production: trigger PagerDuty escalation policy or on-call rotation
        return await self._abort(decision, payload, thread_cfg, reason="escalated")


# ---------------------------------------------------------------------------
# Timeout Watchdog  (background asyncio task)
# ---------------------------------------------------------------------------

class HitlTimeoutWatchdog:
    """
    Background task that monitors pending interrupts and auto-escalates
    those that exceed their SLA.

    Usage
    -----
        watchdog = HitlTimeoutWatchdog(router=router, poll_interval=60)
        asyncio.create_task(watchdog.run())   # in FastAPI lifespan

    Cancellation
    ------------
        watchdog.stop()
    """

    def __init__(
        self,
        router: HitlDecisionRouter,
        poll_interval: float = 60.0,
    ) -> None:
        self._router        = router
        self._poll_interval = poll_interval
        self._running       = False

    async def run(self) -> None:
        """Main loop — call this as an asyncio task."""
        self._running = True
        logger.info("HitlTimeoutWatchdog started (poll=%ds)", self._poll_interval)
        while self._running:
            try:
                await self._check_all()
            except Exception as exc:
                logger.exception("Watchdog poll error: %s", exc)
            await asyncio.sleep(self._poll_interval)

    def stop(self) -> None:
        self._running = False
        logger.info("HitlTimeoutWatchdog stopped")

    async def _check_all(self) -> None:
        now = datetime.now(timezone.utc)
        expired: list[HitlPayload] = []

        for payload in list(self._router._payload_store.values()):
            from .schemas import InterruptState
            if payload.status != InterruptState.PENDING:
                continue
            created = datetime.fromisoformat(payload.created_at)
            elapsed = (now - created).total_seconds()
            if elapsed >= payload.sla_seconds:
                expired.append(payload)

        for payload in expired:
            logger.warning(
                "SLA breached — interrupt_id=%s elapsed=%.0fs sla=%ds",
                payload.interrupt_id,
                (now - datetime.fromisoformat(payload.created_at)).total_seconds(),
                payload.sla_seconds,
            )
            timeout_decision = HitlDecision(
                interrupt_id=payload.interrupt_id,
                thread_id=payload.thread_id,
                decision=DecisionKind.TIMEOUT,
                operator_id="timeout_watchdog",
                comment=f"SLA of {payload.sla_seconds}s exceeded — auto-escalating",
            )
            await self._router.handle_decision(timeout_decision)

            await self._router._audit.write(HitlAuditRecord(
                interrupt_id=payload.interrupt_id,
                thread_id=payload.thread_id,
                event_kind=AuditEventKind.TIMEOUT_TRIGGERED,
                actor="timeout_watchdog",
                payload={"sla_seconds": payload.sla_seconds},
            ))