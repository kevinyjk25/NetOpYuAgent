"""
a2a/request_handler.py
----------------------
DefaultRequestHandler: routes incoming JSON-RPC calls to the AgentExecutor,
manages tasks in the TaskStore, and fires push notifications.

Supported methods
-----------------
  message/send     – synchronous: waits for all events, returns final Task
  message/stream   – streaming: returns an async generator of SSE lines
  tasks/get        – fetch a stored task by ID
  tasks/cancel     – cancel a running task
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, AsyncIterator

from a2a.agent_executor import AgentExecutor
from a2a.event_queue import EventQueue, RequestContext
from a2a.push_notifications import PushNotificationService
from .schemas import (
    Artifact,
    Message,
    MessageEvent,
    Part,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)
from .task_store import TaskStore

logger = logging.getLogger(__name__)


class DefaultRequestHandler:
    """
    Routes A2A JSON-RPC method calls to the AgentExecutor.

    Parameters
    ----------
    agent_executor:
        Your concrete AgentExecutor implementation.
    task_store:
        Persistent or in-memory store for task state.
    push_service:
        Optional push-notification sender; defaults to a new instance.
    """

    def __init__(
        self,
        agent_executor: AgentExecutor,
        task_store: TaskStore,
        push_service: PushNotificationService | None = None,
    ) -> None:
        self._executor = agent_executor
        self._store = task_store
        self._push = push_service or PushNotificationService()

    # ------------------------------------------------------------------
    # Public routing entry-point
    # ------------------------------------------------------------------

    async def handle(self, method: str, params: dict[str, Any]) -> Any:
        """
        Dispatch a decoded JSON-RPC call.

        Returns
        -------
        For ``message/send``  → serialisable dict (Task)
        For ``message/stream``→ AsyncIterator[str]  (SSE lines)
        For ``tasks/get``     → serialisable dict (Task) or error dict
        For ``tasks/cancel``  → serialisable dict (Task) or error dict
        """
        match method:
            case "message/send":
                return await self._handle_send(params)
            case "message/stream":
                return self._handle_stream(params)
            case "tasks/get":
                return await self._handle_tasks_get(params)
            case "tasks/cancel":
                return await self._handle_tasks_cancel(params)
            case _:
                return {"error": {"code": -32601, "message": f"Method not found: {method}"}}

    # ------------------------------------------------------------------
    # message/send  (synchronous)
    # ------------------------------------------------------------------

    async def _handle_send(self, params: dict[str, Any]) -> dict[str, Any]:
        context, event_queue, task = self._build_context_and_task(params)
        await self._store.save(task)

        # Run executor and collect all events
        exec_task = asyncio.create_task(self._executor.execute(context, event_queue))

        async for event in event_queue.consume():
            task = await self._apply_event(task, event)
            await self._store.save(task)
            await self._maybe_push(context, task, event)

        await exec_task
        return task.model_dump()

    # ------------------------------------------------------------------
    # message/stream  (SSE streaming)
    # ------------------------------------------------------------------

    async def _handle_stream(self, params: dict[str, Any]) -> AsyncIterator[str]:
        """Returns an async generator of SSE-formatted strings."""
        context, event_queue, task = self._build_context_and_task(params)
        await self._store.save(task)

        # Fire executor concurrently
        asyncio.create_task(self._executor.execute(context, event_queue))

        async for event in event_queue.consume():
            task = await self._apply_event(task, event)
            await self._store.save(task)
            await self._maybe_push(context, task, event)
            yield f"data: {event.model_dump_json()}\n\n"

        yield "data: [DONE]\n\n"

    # ------------------------------------------------------------------
    # tasks/get
    # ------------------------------------------------------------------

    async def _handle_tasks_get(self, params: dict[str, Any]) -> dict[str, Any]:
        task_id: str = params.get("id", "")
        task = await self._store.get(task_id)
        if task is None:
            return {"error": {"code": -32001, "message": f"Task not found: {task_id}"}}
        return task.model_dump()

    # ------------------------------------------------------------------
    # tasks/cancel
    # ------------------------------------------------------------------

    async def _handle_tasks_cancel(self, params: dict[str, Any]) -> dict[str, Any]:
        task_id: str = params.get("id", "")
        task = await self._store.get(task_id)
        if task is None:
            return {"error": {"code": -32001, "message": f"Task not found: {task_id}"}}

        # Build a minimal context for the cancel call
        cancel_queue = EventQueue()
        context = RequestContext(
            task_id=task_id,
            context_id=task.context_id or task_id,
            message=Message(role="user", parts=[TextPart(text="cancel")]),
        )
        await self._executor.cancel(context, cancel_queue)

        task.status = TaskStatus(state=TaskState.CANCELED)
        await self._store.save(task)
        return task.model_dump()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_context_and_task(
        self, params: dict[str, Any]
    ) -> tuple[RequestContext, EventQueue, Task]:
        raw_msg = params.get("message", {})
        parts = self._parse_parts(raw_msg.get("parts", []))
        message = Message(
            message_id=raw_msg.get("message_id", str(uuid.uuid4())),
            role=raw_msg.get("role", "user"),
            parts=parts,
            metadata=raw_msg.get("metadata"),
        )
        task_id = str(uuid.uuid4())
        context_id = params.get("context_id", str(uuid.uuid4()))
        metadata = params.get("metadata", {})
        webhook_url = params.get("webhook_url")

        context = RequestContext(
            task_id=task_id,
            context_id=context_id,
            message=message,
            metadata=metadata,
            webhook_url=webhook_url,
        )
        event_queue = EventQueue()
        task = Task(
            id=task_id,
            context_id=context_id,
            status=TaskStatus(state=TaskState.SUBMITTED),
        )
        return context, event_queue, task

    @staticmethod
    def _parse_parts(raw_parts: list[dict[str, Any]]) -> list[Part]:
        from .schemas import DataPart, FilePart, TextPart

        result: list[Part] = []
        for p in raw_parts:
            kind = p.get("kind", "text")
            if kind == "text":
                result.append(TextPart(text=p.get("text", "")))
            elif kind == "file":
                result.append(
                    FilePart(
                        name=p.get("name", ""),
                        mime_type=p.get("mime_type", "application/octet-stream"),
                        data=p.get("data", ""),
                    )
                )
            elif kind == "data":
                result.append(DataPart(data=p.get("data", {})))
        return result or [TextPart(text="")]

    @staticmethod
    async def _apply_event(task: Task, event: Any) -> Task:
        """Mutate task state based on the incoming event."""
        if isinstance(event, TaskStatusUpdateEvent):
            task.status = event.status
        elif isinstance(event, TaskArtifactUpdateEvent):
            task.artifacts.append(event.artifact)
        elif isinstance(event, MessageEvent):
            task.status = TaskStatus(state=TaskState.COMPLETED)
        return task

    async def _maybe_push(
        self, context: RequestContext, task: Task, event: Any
    ) -> None:
        if not context.webhook_url:
            return
        from .schemas import PushNotificationPayload

        payload = PushNotificationPayload(
            task_id=task.id,
            context_id=task.context_id or task.id,
            state=task.status.state,
        )
        if isinstance(event, TaskArtifactUpdateEvent):
            payload.artifact = event.artifact

        await self._push.notify(context.webhook_url, payload)
