"""
task/inter/hitl_bridge.py
--------------------------
HitlTaskBridge — thin adapter connecting the Task module to the HITL module.

When a task triggers a HITL interrupt:
  1. Sets task state → WAITING_HITL
  2. Registers the interrupt with hitl.HitlDecisionRouter
  3. Records interrupt_id in session.multi_round.pending_hitl_ids
  4. On resume: sets task state → PENDING, removes from pending list
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from ..schemas import (
    SessionRecord,
    TaskAuditRecord,
    TaskDefinition,
    TaskEventKind,
    TaskState,
)

if TYPE_CHECKING:
    from hitl.decision import HitlDecisionRouter
    from hitl.review import HitlReviewService
    from hitl.schemas import HitlDecision, HitlPayload

logger = logging.getLogger(__name__)


class HitlTaskBridge:
    """
    Connects TaskDefinition lifecycle to the HITL module.

    Parameters
    ----------
    hitl_router  : hitl.decision.HitlDecisionRouter
    review_svc   : hitl.review.HitlReviewService
    task_store   : task.intra.store.TaskStore
    session_mgr  : task.inter.session.SessionManager
    """

    def __init__(
        self,
        hitl_router:  "HitlDecisionRouter",
        review_svc:   "HitlReviewService",
        task_store:   object,    # TaskStore – avoid circular import
        session_mgr:  object,    # SessionManager
    ) -> None:
        self._router  = hitl_router
        self._review  = review_svc
        self._store   = task_store
        self._session = session_mgr

    async def suspend_for_review(
        self,
        task: TaskDefinition,
        payload: "HitlPayload",
        session: Optional[SessionRecord] = None,
    ) -> None:
        """
        Suspend a task pending human review.

        Steps
        -----
        1. Set task → WAITING_HITL
        2. Register interrupt with HITL router
        3. Fan-out notifications
        4. Update session multi-round context
        """
        task.state = TaskState.WAITING_HITL
        await self._store.save(task)
        await self._store.write_audit(TaskAuditRecord(
            task_id=task.task_id,
            session_id=task.session_id,
            event_kind=TaskEventKind.HITL_PAUSE,
            actor="hitl_task_bridge",
            payload={"interrupt_id": payload.interrupt_id},
        ))

        await self._router.register_interrupt(payload)

        import asyncio
        asyncio.create_task(self._review.notify(payload))

        if session:
            ctx = session.multi_round
            if payload.interrupt_id not in ctx.pending_hitl_ids:
                ctx.pending_hitl_ids.append(payload.interrupt_id)
            await self._session.update_multi_round(session, ctx)

        logger.info(
            "HitlTaskBridge: task %s suspended interrupt_id=%s",
            task.task_id, payload.interrupt_id,
        )

    async def resume_task(
        self,
        task: TaskDefinition,
        decision: "HitlDecision",
        session: Optional[SessionRecord] = None,
    ) -> TaskDefinition:
        """
        Called after the operator submits a decision.
        Re-queues (approve/edit) or cancels (reject/timeout) the task.
        """
        from hitl.schemas import DecisionKind

        if decision.decision in (DecisionKind.APPROVE, DecisionKind.EDIT):
            task.state = TaskState.PENDING
            if decision.parameter_patch:
                task.parameters.update(decision.parameter_patch)
            logger.info("HitlTaskBridge: task %s approved, re-queuing", task.task_id)
        else:
            task.state = TaskState.CANCELLED
            task.error = f"Rejected by operator {decision.operator_id}: {decision.comment}"
            logger.info("HitlTaskBridge: task %s rejected/cancelled", task.task_id)

        await self._store.save(task)
        await self._store.write_audit(TaskAuditRecord(
            task_id=task.task_id,
            session_id=task.session_id,
            event_kind=TaskEventKind.HITL_RESUME,
            actor=decision.operator_id,
            payload={"decision": decision.decision.value},
        ))

        # Clean up session pending list
        if session:
            ctx = session.multi_round
            interrupt_id = decision.interrupt_id
            ctx.pending_hitl_ids = [i for i in ctx.pending_hitl_ids if i != interrupt_id]
            await self._session.update_multi_round(session, ctx)

        return task
