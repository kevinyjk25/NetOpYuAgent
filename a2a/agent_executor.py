"""
a2a/agent_executor.py  [MODIFIED]
----------------------------------
Changes from v1:
  1. Added MemoryAwareAgentExecutor mixin that injects memory context
     into every _run_agent() call before LangGraph execution.
  2. ITOpsAgentExecutor now inherits from MemoryAwareAgentExecutor
     so all A2A calls automatically retrieve + write back memory.
  3. Added session_id extraction from RequestContext.metadata.

All original processors and the abstract AgentExecutor base are unchanged.
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from a2a.event_queue import EventQueue, RequestContext
from .schemas import (
    Artifact, DataPart, Message, MessageEvent,
    TaskArtifactUpdateEvent, TaskState, TaskStatus,
    TaskStatusUpdateEvent, TextPart,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base (unchanged)
# ---------------------------------------------------------------------------

class AgentExecutor(ABC):
    @abstractmethod
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None: ...

    @abstractmethod
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None: ...


# ---------------------------------------------------------------------------
# Strategy-pattern processors (unchanged from v1)
# ---------------------------------------------------------------------------

class A2AEventProcessor(ABC):
    @abstractmethod
    async def process(self, chunk, event_queue, task_id, context_id) -> None: ...


class A2ATokenProcessor(A2AEventProcessor):
    async def process(self, chunk, event_queue, task_id, context_id):
        token = chunk.get("token")
        if not token:
            return
        await event_queue.enqueue_event(TaskArtifactUpdateEvent(
            task_id=task_id, context_id=context_id,
            artifact=Artifact(
                name="llm_token",
                parts=[DataPart(data={"token": token, "type": "token"})],
            ),
        ))


class A2ABatchTokenProcessor(A2AEventProcessor):
    async def process(self, chunk, event_queue, task_id, context_id):
        tokens = chunk.get("tokens", [])
        if not tokens:
            return
        await event_queue.enqueue_event(TaskArtifactUpdateEvent(
            task_id=task_id, context_id=context_id,
            artifact=Artifact(
                name="llm_tokens_batch",
                parts=[DataPart(data={"tokens": tokens, "type": "tokens_batch"})],
            ),
        ))


class A2AMessageProcessor(A2AEventProcessor):
    async def process(self, chunk, event_queue, task_id, context_id):
        text = chunk.get("message")
        node = chunk.get("node", "unknown")
        if not text:
            return
        await event_queue.enqueue_event(TaskArtifactUpdateEvent(
            task_id=task_id, context_id=context_id,
            artifact=Artifact(
                name="node_message",
                parts=[DataPart(data={"text": text, "node": node, "type": "message"})],
            ),
        ))


class A2ANodeResultProcessor(A2AEventProcessor):
    NODE_RESULT_TAG_MAP: dict[str, dict[str, str]] = {
        "alert_analysis_result": {"summary": "markdown", "chart": "chart", "raw_data": "code"},
        "incident_result":       {"summary": "markdown", "recommendations": "buttons"},
        "prediction_result":     {"forecast_chart": "chart", "explanation": "markdown"},
        "multi_dataset_result":  {"correlation_chart": "chart", "conclusion": "markdown"},
        "aggregator_result":     {"summary": "markdown"},           # ← NEW: task aggregator
        "executor_result":       {"summary": "markdown", "raw_data": "code"},  # ← NEW
    }

    async def process(self, chunk, event_queue, task_id, context_id):
        node_result = chunk.get("node_result")
        node = chunk.get("node", "unknown")
        if not node_result:
            return
        tag_map = self.NODE_RESULT_TAG_MAP.get(node, {})
        for field_name, value in node_result.items():
            tag = tag_map.get(field_name, "markdown")
            await event_queue.enqueue_event(TaskArtifactUpdateEvent(
                task_id=task_id, context_id=context_id,
                artifact=Artifact(
                    name=f"{node}_{field_name}",
                    parts=[DataPart(data={
                        "tag": tag, "node": node,
                        "field": field_name, "content": value,
                        "type": "node_result",
                    })],
                ),
            ))


class A2ANodeStepProcessor(A2AEventProcessor):
    async def process(self, chunk, event_queue, task_id, context_id):
        step = chunk.get("node_step")
        node = chunk.get("node", "unknown")
        if not step:
            return
        await event_queue.enqueue_event(TaskArtifactUpdateEvent(
            task_id=task_id, context_id=context_id,
            artifact=Artifact(
                name="node_step",
                parts=[DataPart(data={"step": step, "node": node, "type": "node_step"})],
            ),
        ))


DEFAULT_PROCESSORS: list[A2AEventProcessor] = [
    A2ATokenProcessor(),
    A2ABatchTokenProcessor(),
    A2AMessageProcessor(),
    A2ANodeResultProcessor(),
    A2ANodeStepProcessor(),
]


# ---------------------------------------------------------------------------
# NEW: Memory-aware mixin
# ---------------------------------------------------------------------------

class MemoryAwareMixin:
    """
    Mixin that adds memory retrieval + write-back to any AgentExecutor.

    Wire up by passing a MemoryRouter instance at construction time.
    If no router is provided the mixin is a no-op (backward compatible).
    """

    def __init_memory__(self, memory_router: Any | None) -> None:
        self._memory = memory_router

    async def _build_memory_context(self, context: RequestContext) -> str:
        """Retrieve relevant memory and return a formatted context string."""
        if self._memory is None:
            return ""
        try:
            session_id = context.metadata.get("session_id", context.context_id)
            return await self._memory.recall_for_session(
                context.get_user_input(), session_id,
            )
        except Exception as exc:
            logger.warning("Memory retrieval failed: %s", exc)
            return ""

    async def _write_back_memory(
        self,
        context: RequestContext,
        user_text: str,
        assistant_text: str,
    ) -> None:
        """Write the completed turn back to memory."""
        if self._memory is None:
            return
        try:
            session_id = context.metadata.get("session_id", context.context_id)
            await self._memory.ingest_turn(session_id, user_text, assistant_text)
        except Exception as exc:
            logger.warning("Memory write-back failed: %s", exc)


# ---------------------------------------------------------------------------
# ITOpsAgentExecutor (updated — adds memory awareness)
# ---------------------------------------------------------------------------

class ITOpsAgentExecutor(AgentExecutor, MemoryAwareMixin):
    """
    IT Ops agent executor with memory context injection.

    v2 changes vs v1
    ----------------
    - Accepts optional ``memory_router`` for automatic memory R/W.
    - Passes ``memory_context`` string into ``_run_agent()``.
    - Writes completed turn back to memory after streaming.
    - Session ID extracted from context.metadata["session_id"]
      (falls back to context_id for backward compatibility).
    """

    def __init__(
        self,
        memory_router: Any | None = None,
    ) -> None:
        self.__init_memory__(memory_router)
        self._cancelled: dict[str, bool] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id    = context.task_id
        context_id = context.context_id
        query      = context.get_user_input()

        logger.info("ITOpsAgentExecutor.execute task_id=%s", task_id)
        self._cancelled[task_id] = False

        # Retrieve memory context before starting
        memory_context = await self._build_memory_context(context)

        await event_queue.enqueue_event(TaskStatusUpdateEvent(
            task_id=task_id, context_id=context_id,
            status=TaskStatus(state=TaskState.WORKING),
        ))

        response_chunks: list[str] = []

        try:
            async for chunk in self._run_agent(query, context, memory_context):
                if self._cancelled.get(task_id):
                    break
                for processor in DEFAULT_PROCESSORS:
                    await processor.process(chunk, event_queue, task_id, context_id)
                # Collect token chunks to reconstruct assistant response for memory
                if "token" in chunk:
                    response_chunks.append(chunk["token"])

            await event_queue.enqueue_event(TaskStatusUpdateEvent(
                task_id=task_id, context_id=context_id,
                status=TaskStatus(state=TaskState.COMPLETED),
            ))
            await event_queue.enqueue_event(MessageEvent(
                task_id=task_id, context_id=context_id,
                message=Message(role="assistant", parts=[TextPart(text="Task completed.")]),
            ))

            # Write turn back to memory
            assistant_text = "".join(response_chunks) or "Task completed."
            await self._write_back_memory(context, query, assistant_text)

        except Exception as exc:
            logger.exception("ITOpsAgentExecutor error task_id=%s: %s", task_id, exc)
            await event_queue.enqueue_event(TaskStatusUpdateEvent(
                task_id=task_id, context_id=context_id,
                status=TaskStatus(state=TaskState.FAILED, message=str(exc)),
            ))
            await event_queue.enqueue_event(MessageEvent(
                task_id=task_id, context_id=context_id,
                message=Message(role="assistant", parts=[TextPart(text=f"Task failed: {exc}")]),
            ))
        finally:
            self._cancelled.pop(task_id, None)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        self._cancelled[context.task_id] = True
        await event_queue.enqueue_event(TaskStatusUpdateEvent(
            task_id=context.task_id, context_id=context.context_id,
            status=TaskStatus(state=TaskState.CANCELED),
        ))
        await event_queue.enqueue_event(MessageEvent(
            task_id=context.task_id, context_id=context.context_id,
            message=Message(role="assistant", parts=[TextPart(text="Task cancelled.")]),
        ))

    async def _run_agent(
        self,
        query: str,
        context: RequestContext,
        memory_context: str = "",   # ← NEW parameter
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Stub — replace with your real LangGraph / HITL graph call:

            async for chunk in run_with_hitl(
                query=query,
                thread_id=context.context_id,
                context_id=context.context_id,
                task_id=context.task_id,
                user_metadata={**context.metadata, "memory_context": memory_context},
            ):
                yield chunk
        """
        if memory_context:
            yield {"message": "Memory context loaded", "node": "memory"}
            await asyncio.sleep(0)

        yield {"node_step": "Routing query…", "node": "intent"}
        await asyncio.sleep(0)

        for word in f"Processing: {query}".split():
            yield {"token": word + " "}
            await asyncio.sleep(0)

        yield {
            "node_result": {
                "summary": f"**Query:** {query}\n\nConnect your LangGraph graph here.",
            },
            "node": "alert_analysis_result",
        }
