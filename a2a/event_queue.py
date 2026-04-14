"""
a2a/event_queue.py
------------------
Thread-safe async EventQueue and RequestContext.

EventQueue:
  - Producers call ``enqueue_event()``
  - The SSE / streaming layer calls ``consume()`` to drain events
  - Sending a MessageEvent marks the queue as *finalised*; further
    consumption will raise StopAsyncIteration

RequestContext:
  - Wraps the incoming JSON-RPC params so AgentExecutor implementations
    stay clean and testable.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Optional

from .schemas import A2AEvent, Message, MessageEvent, Part, TextPart

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EventQueue
# ---------------------------------------------------------------------------

class EventQueue:
    """
    Async FIFO queue for A2A streaming events.

    Usage::

        queue = EventQueue()

        # producer side (AgentExecutor)
        await queue.enqueue_event(TaskArtifactUpdateEvent(...))
        await queue.enqueue_event(MessageEvent(...))   # finalises the stream

        # consumer side (SSE handler)
        async for event in queue.consume():
            send_sse(event.model_dump_json())
    """

    _SENTINEL = object()

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._finalised = False

    # ------------------------------------------------------------------
    # Producer API
    # ------------------------------------------------------------------

    async def enqueue_event(self, event: A2AEvent) -> None:
        """
        Enqueue an event.  After a :class:`MessageEvent` is enqueued the
        queue is sealed – subsequent enqueue calls are no-ops (with a
        warning logged).
        """
        if self._finalised:
            logger.warning(
                "EventQueue already finalised; dropping event %s", type(event).__name__
            )
            return

        await self._queue.put(event)

        # MessageEvent is the terminal signal
        if event.kind == "message":
            self._finalised = True
            await self._queue.put(self._SENTINEL)

    # ------------------------------------------------------------------
    # Consumer API
    # ------------------------------------------------------------------

    async def consume(self) -> AsyncIterator[A2AEvent]:
        """Async generator that yields events until the queue is finalised."""
        while True:
            item = await self._queue.get()
            if item is self._SENTINEL:
                return
            yield item  # type: ignore[misc]

    def is_finalised(self) -> bool:
        return self._finalised


# ---------------------------------------------------------------------------
# RequestContext
# ---------------------------------------------------------------------------

class RequestContext:
    """
    Wraps the decoded JSON-RPC ``params`` from an incoming A2A request.

    Provides convenience helpers used inside AgentExecutor.execute().
    """

    def __init__(
        self,
        task_id: str,
        context_id: str,
        message: Message,
        metadata: Optional[dict[str, Any]] = None,
        webhook_url: Optional[str] = None,
    ) -> None:
        self.task_id = task_id
        self.context_id = context_id
        self.message = message
        self.metadata: dict[str, Any] = metadata or {}
        self.webhook_url = webhook_url  # for push notifications

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_user_input(self) -> str:
        """Return the concatenated text from all TextPart items in the message."""
        texts: list[str] = []
        for part in self.message.parts:
            if isinstance(part, TextPart):
                texts.append(part.text)
        return "\n".join(texts)

    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    def __repr__(self) -> str:
        return (
            f"<RequestContext task_id={self.task_id!r} "
            f"context_id={self.context_id!r}>"
        )
