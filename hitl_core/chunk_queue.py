"""
hitl_core/chunk_queue.py
------------------------
Per-interrupt async chunk queue for live HITL resume streaming.

When a HITL interrupt is resumed via `/hitl/<id>/choose` (or approve/edit/...),
the resumer runs the full agent loop synchronously and only returns when done.
For long-running resumes (e.g. paginated reading + slow LLM), the frontend
would otherwise see nothing until the resumer finishes — looking frozen.

This module provides a simple per-interrupt-id chunk queue:

  - Resumer pushes chunks via `push(interrupt_id, chunk_dict)` as they happen
  - Frontend SSE endpoint subscribes via `subscribe(interrupt_id)` to get
    an async iterator of those chunks in real time
  - `complete(interrupt_id)` signals end-of-stream; subscribers exit cleanly
  - `close(interrupt_id)` removes the queue (cleanup after final response)

Independence:
  - Pure asyncio primitives (Queue + Event), no FastAPI dependency
  - Used only by webui/backend.py + integrations/adapters/hitl_executor.py
  - Disabling is a no-op: pushes silently drop if no subscriber exists
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Optional

logger = logging.getLogger(__name__)


class _InterruptStream:
    """Single per-interrupt chunk stream."""

    def __init__(self, interrupt_id: str, max_buffer: int = 500):
        self.interrupt_id = interrupt_id
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_buffer)
        self.done: asyncio.Event = asyncio.Event()
        self.created_at: float = time.time()
        # Optional: keep the last N chunks for late subscribers, so a frontend
        # that connects AFTER the resumer started doesn't miss the early trace.
        self.history: list[dict] = []
        self.history_cap: int = 200
        # Track whether any subscriber connected — for cleanup heuristics
        self.subscriber_count: int = 0

    def push(self, chunk: dict) -> None:
        """Non-blocking push. Drops chunk if queue is full (caller logs)."""
        try:
            self.queue.put_nowait(chunk)
        except asyncio.QueueFull:
            logger.debug("InterruptStream[%s]: queue full, dropping chunk", self.interrupt_id)
        # Always preserve in history (capped)
        self.history.append(chunk)
        if len(self.history) > self.history_cap:
            self.history.pop(0)

    def complete(self) -> None:
        """Signal that no more chunks will be pushed."""
        self.done.set()


class HitlChunkQueueRegistry:
    """Process-wide registry of per-interrupt chunk streams.

    Lifecycle:
      1. Resumer calls ensure(interrupt_id) before pushing.
      2. Resumer pushes via push(interrupt_id, chunk_dict).
      3. Subscriber (SSE endpoint) calls subscribe(interrupt_id) to async-iterate.
      4. Resumer calls complete(interrupt_id) when done.
      5. Periodic cleanup or explicit close(interrupt_id) removes the stream.
    """

    def __init__(self, ttl_seconds: float = 1800.0):
        self._streams: dict[str, _InterruptStream] = {}
        self._lock = asyncio.Lock()
        self._ttl = float(ttl_seconds)

    async def ensure(self, interrupt_id: str) -> _InterruptStream:
        """Create-or-get the stream for an interrupt."""
        async with self._lock:
            s = self._streams.get(interrupt_id)
            if s is None:
                s = _InterruptStream(interrupt_id)
                self._streams[interrupt_id] = s
                logger.debug("ChunkQueue: created stream for %s", interrupt_id)
            return s

    def push(self, interrupt_id: str, chunk: dict) -> None:
        """Non-blocking push. No-op if stream doesn't exist (subscriber-less)."""
        s = self._streams.get(interrupt_id)
        if s is not None:
            s.push(chunk)

    def complete(self, interrupt_id: str) -> None:
        """Mark end-of-stream so subscribers exit."""
        s = self._streams.get(interrupt_id)
        if s is not None:
            s.complete()
            logger.debug("ChunkQueue: completed stream for %s", interrupt_id)

    async def subscribe(
        self,
        interrupt_id: str,
        *,
        replay_history: bool = True,
        poll_timeout: float = 0.5,
    ) -> AsyncIterator[dict]:
        """Async iterator over chunks for this interrupt.

        Args:
          replay_history: if True, yield any chunks that arrived BEFORE
                          subscription (so a frontend that connected late
                          still sees the early trace).
          poll_timeout:   max seconds to wait for next chunk before checking
                          done flag (lets us exit promptly when resumer ends).
        """
        s = await self.ensure(interrupt_id)
        s.subscriber_count += 1

        try:
            # 1. Replay history (snapshot at subscribe time)
            if replay_history and s.history:
                for ch in list(s.history):
                    yield ch

            # 2. Live tail
            while True:
                try:
                    chunk = await asyncio.wait_for(s.queue.get(), timeout=poll_timeout)
                    yield chunk
                except asyncio.TimeoutError:
                    if s.done.is_set() and s.queue.empty():
                        break
                    continue
        finally:
            s.subscriber_count -= 1

    async def close(self, interrupt_id: str) -> None:
        """Remove the stream (call after final response is sent)."""
        async with self._lock:
            s = self._streams.pop(interrupt_id, None)
            if s is not None:
                s.complete()   # wake any stragglers
                logger.debug("ChunkQueue: closed stream for %s", interrupt_id)

    async def gc(self) -> int:
        """Garbage-collect streams older than TTL with no subscribers.
        Returns number of streams removed."""
        now = time.time()
        async with self._lock:
            to_remove = [
                iid for iid, s in self._streams.items()
                if (now - s.created_at) > self._ttl and s.subscriber_count == 0
            ]
            for iid in to_remove:
                self._streams.pop(iid, None)
        if to_remove:
            logger.debug("ChunkQueue: gc'd %d expired stream(s)", len(to_remove))
        return len(to_remove)


# Process-wide singleton — webui/backend.py + hitl_executor.py both use this
_GLOBAL: Optional[HitlChunkQueueRegistry] = None


def get_chunk_queue_registry() -> HitlChunkQueueRegistry:
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = HitlChunkQueueRegistry()
    return _GLOBAL
