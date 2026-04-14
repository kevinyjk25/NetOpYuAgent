"""
task/inter/coordinator.py
--------------------------
A2ATaskDispatcher      – send subtasks to remote agents via A2A message/stream
MultiRoundCoordinator  – correlate responses back to TaskDefinitions across turns
ResultAggregator       – merge partial results from N parallel agents
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, AsyncIterator, Optional

import httpx

from ..schemas import (
    AgentAssignment,
    SessionRecord,
    TaskAuditRecord,
    TaskDefinition,
    TaskEventKind,
    TaskState,
)
from task.inter.session import SessionManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# A2A Task Dispatcher
# ---------------------------------------------------------------------------

class A2ATaskDispatcher:
    """
    Sends a TaskDefinition to a remote A2A agent via message/stream.

    Returns an async generator that yields chunk dicts compatible with
    the 6-processor chain in a2a/agent_executor.py + hitl/a2a_integration.py.
    """

    def __init__(self, http_timeout: float = 120.0) -> None:
        self._timeout = http_timeout

    async def dispatch(
        self,
        task: TaskDefinition,
        assignment: AgentAssignment,
        store: Any,     # TaskStore — avoids circular import
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Stream subtask execution from a remote A2A agent.
        Yields chunk dicts (token, message, node_result, hitl_interrupt, …).
        """
        task.state = TaskState.RUNNING
        await store.save(task)
        await store.write_audit(TaskAuditRecord(
            task_id=task.task_id, session_id=task.session_id,
            event_kind=TaskEventKind.DISPATCHED, actor="a2a_dispatcher",
            payload={"agent_url": assignment.agent_url, "skill_id": assignment.skill_id},
        ))

        body = {
            "jsonrpc": "2.0",
            "method":  "message/stream",
            "params": {
                "message": {
                    "kind":    "message",
                    "role":    "user",
                    "message_id": str(uuid.uuid4()),
                    "parts": [{"kind": "text", "text": task.description}],
                },
                "context_id": task.context_id,
                "metadata": {
                    "task_id":    task.task_id,
                    "session_id": task.session_id,
                    **task.parameters,
                },
            },
            "id": 1,
        }

        async for chunk in self._stream_request(assignment.agent_url, body):
            yield chunk

    async def _stream_request(
        self,
        agent_url: str,
        body: dict,
    ) -> AsyncIterator[dict[str, Any]]:
        stream_url = agent_url.rstrip("/") + "/stream"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                async with client.stream("POST", stream_url, json=body) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            return
                        try:
                            import json
                            yield json.loads(data)
                        except Exception:
                            yield {"token": data + " "}
        except Exception as exc:
            logger.error("A2ATaskDispatcher stream failed: %s", exc)
            yield {"message": f"Remote agent error: {exc}", "node": "dispatcher"}


# ---------------------------------------------------------------------------
# Multi-round coordinator
# ---------------------------------------------------------------------------

class MultiRoundCoordinator:
    """
    Tracks open tasks across multiple A2A turns in a session.

    On each new turn it:
    1. Checks for deferred tasks (from previous turns) that can now be answered.
    2. Carries open questions and confirmed facts forward.
    3. Updates MultiRoundContext on the session.
    """

    def __init__(self, session_mgr: SessionManager) -> None:
        self._session_mgr = session_mgr

    async def on_turn_start(
        self,
        session: SessionRecord,
        user_text: str,
    ) -> SessionRecord:
        """Called at the start of each A2A turn. Updates session multi-round context."""
        ctx = session.multi_round

        # Carry last response forward as confirmed fact if it was definitive
        if ctx.last_agent_response and len(ctx.last_agent_response) > 50:
            fact = ctx.last_agent_response[:200]
            if fact not in ctx.confirmed_facts:
                ctx.confirmed_facts.append(fact)

        # Check if user is answering an open question
        for q in list(ctx.open_questions):
            if any(kw in user_text.lower() for kw in ["yes", "no", "confirm", "proceed"]):
                ctx.open_questions.remove(q)
                logger.debug("MultiRound: resolved open question: %s", q)

        await self._session_mgr.update_multi_round(session, ctx)
        return session

    async def on_turn_end(
        self,
        session: SessionRecord,
        agent_response: str,
        deferred_task_ids: Optional[list[str]] = None,
        open_questions: Optional[list[str]] = None,
    ) -> SessionRecord:
        """Called at end of each turn. Saves state for next turn."""
        ctx = session.multi_round
        ctx.last_agent_response = agent_response

        if deferred_task_ids:
            ctx.deferred_task_ids.extend(deferred_task_ids)

        if open_questions:
            ctx.open_questions.extend(open_questions)

        await self._session_mgr.update_multi_round(session, ctx)
        await self._session_mgr.increment_turn(session)
        return session

    def build_context_prefix(self, session: SessionRecord) -> str:
        """
        Construct a context prefix for the LLM prompt summarising
        the multi-round state (open questions, confirmed facts, etc.).
        """
        ctx   = session.multi_round
        lines = [f"[Turn {session.turn_count}]"]

        if ctx.confirmed_facts:
            lines.append("Confirmed context:")
            for f in ctx.confirmed_facts[-3:]:   # last 3
                lines.append(f"  - {f[:120]}")

        if ctx.open_questions:
            lines.append("Open questions awaiting answer:")
            for q in ctx.open_questions:
                lines.append(f"  ? {q}")

        if ctx.deferred_task_ids:
            lines.append(f"Deferred tasks: {', '.join(ctx.deferred_task_ids)}")

        if ctx.pending_hitl_ids:
            lines.append(f"Pending human review: {', '.join(ctx.pending_hitl_ids)}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Result Aggregator
# ---------------------------------------------------------------------------

class ResultAggregator:
    """
    Merges partial results from N parallel subtask agents into a single
    ranked result dict, and streams the merged output via A2A artifacts.
    """

    def __init__(self) -> None:
        self._results: dict[str, list[dict[str, Any]]] = {}   # task_id → chunks

    def record_chunk(self, task_id: str, chunk: dict[str, Any]) -> None:
        self._results.setdefault(task_id, []).append(chunk)

    async def aggregate(
        self,
        tasks: list[TaskDefinition],
    ) -> dict[str, Any]:
        """
        Produce the final merged result.
        Override for domain-specific ranking logic.
        """
        merged: dict[str, Any] = {
            "task_count": len(tasks),
            "completed":  sum(1 for t in tasks if t.state == TaskState.COMPLETED),
            "failed":     sum(1 for t in tasks if t.state == TaskState.FAILED),
            "subtask_results": {},
        }
        for task in tasks:
            if task.result:
                merged["subtask_results"][task.task_id] = {
                    "description": task.description,
                    "result": task.result,
                }
        logger.info(
            "ResultAggregator: merged %d tasks (%d completed, %d failed)",
            merged["task_count"], merged["completed"], merged["failed"],
        )
        return merged

    async def stream_merged(
        self,
        tasks: list[TaskDefinition],
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield A2A-compatible chunk dicts for the merged result."""
        summary = await self.aggregate(tasks)

        yield {"node_step": "Aggregating results from all agents", "node": "aggregator"}

        for tid, data in summary["subtask_results"].items():
            yield {
                "node_result": {
                    "summary": data["result"].get("output", str(data["result"])),
                },
                "node": "aggregator_result",
            }

        status_line = (
            f"Completed {summary['completed']}/{summary['task_count']} subtasks"
            + (f" ({summary['failed']} failed)" if summary["failed"] else "")
        )
        for word in status_line.split():
            yield {"token": word + " "}
