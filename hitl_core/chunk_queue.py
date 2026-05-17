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
    """Single per-interrupt chunk stream.

    Storage model: a single append-only `history` list plus an asyncio.Event
    that signals "new chunk arrived". Subscribers track their own index into
    history. This avoids the previous double-storage bug (chunks pushed to
    both queue and history → subscribers saw each chunk twice: once via
    history replay, once via queue drain).

    Lifecycle fields (added during the audit fix for issue D — chunk_queue
    leakage across chat_stream turns):
      - session_id: which chat session owns this stream. Lets a new
        chat_stream turn close orphan streams from prior turns on the
        same session.
      - last_activity_at: monotonic timestamp of the most recent push.
        Used by the idle-timeout watchdog to auto-complete streams that
        have gone quiet (e.g. agent_loop_resumer hung mid-execution).
    """

    def __init__(
        self,
        interrupt_id: str,
        max_buffer: int = 500,
        session_id: str = "",
    ):
        self.interrupt_id = interrupt_id
        self.session_id = session_id
        self.done: asyncio.Event = asyncio.Event()
        self.new_chunk: asyncio.Event = asyncio.Event()
        self.created_at: float = time.time()
        self.last_activity_at: float = time.time()
        # Single source of truth for chunks. Capped to avoid unbounded growth
        # if a resumer pushes faster than any subscriber drains; the cap is
        # high enough that real conversations never trim, but a runaway
        # resumer is bounded.
        self.history: list[dict] = []
        self.history_cap: int = max_buffer
        # Track whether any subscriber connected — for cleanup heuristics
        self.subscriber_count: int = 0

    def push(self, chunk: dict) -> None:
        """Non-blocking push. Trims oldest history when over cap (rare)."""
        self.history.append(chunk)
        self.last_activity_at = time.time()
        if len(self.history) > self.history_cap:
            # Drop oldest. Late subscribers will lose these, but that's
            # better than blocking the resumer or unbounded memory growth.
            overflow = len(self.history) - self.history_cap
            del self.history[:overflow]
            logger.debug(
                "InterruptStream[%s]: history capped at %d, dropped %d",
                self.interrupt_id, self.history_cap, overflow,
            )
        # Wake any subscriber currently awaiting new_chunk
        self.new_chunk.set()

    def complete(self) -> None:
        """Signal that no more chunks will be pushed."""
        self.done.set()
        # Also wake subscribers blocked on new_chunk so they observe done
        self.new_chunk.set()


class HitlChunkQueueRegistry:
    """Process-wide registry of per-interrupt chunk streams.

    Lifecycle:
      1. Resumer calls ensure(interrupt_id, session_id) before pushing.
      2. Resumer pushes via push(interrupt_id, chunk_dict).
      3. Subscriber (SSE endpoint) calls subscribe(interrupt_id) to async-iterate.
      4. Resumer calls complete(interrupt_id) when done.
      5. Periodic cleanup or explicit close(interrupt_id) removes the stream.

    Idle-timeout guard (added during audit fix D):
      A background task wakes every `idle_check_interval` seconds and
      auto-completes any stream whose last_activity_at is older than
      `idle_timeout`. This prevents an agent_loop_resumer that hung mid-
      execution from leaking chunks into the UI of a subsequent chat_stream
      turn (the original bug: 3-min resumer push storm collided with
      operator's '请继续' query).

    Session-scoped close (added during audit fix D):
      `close_session_streams(session_id)` completes all live streams for
      a session. chat_stream calls this when starting a new operator turn,
      ensuring stale HITL sub-streams from prior turns don't dribble
      chunks into the new turn's display.
    """

    def __init__(
        self,
        ttl_seconds: float = 1800.0,
        idle_timeout: float = 120.0,
        idle_check_interval: float = 30.0,
    ):
        self._streams: dict[str, _InterruptStream] = {}
        self._lock = asyncio.Lock()
        self._ttl = float(ttl_seconds)
        self._idle_timeout = float(idle_timeout)
        self._idle_check_interval = float(idle_check_interval)
        self._idle_task: Optional[asyncio.Task] = None

    def _ensure_sync(
        self, interrupt_id: str, session_id: str = "",
    ) -> _InterruptStream:
        """Sync stream create-or-get, callable from non-async resumer code.

        Doesn't acquire self._lock (an asyncio.Lock — illegal in sync
        context). Relies on the GIL for atomic dict ops + a small
        check-then-set window; concurrent first-pushers would each create
        an _InterruptStream and only the latest wins, but since the queue
        is brand-new and empty either way, the loser is harmless.

        For the corresponding subscribe path, ensure() (async, locked)
        still serialises stream creation, so subscribers always see
        the same instance the pusher writes to.
        """
        s = self._streams.get(interrupt_id)
        if s is None:
            s = _InterruptStream(interrupt_id, session_id=session_id)
            # Last-write-wins is fine: any concurrent push of the same
            # interrupt_id is using the same chunk_log already, and the
            # subscriber side reads from self._streams[interrupt_id]
            # after ensure(), so it sees whatever's there at that point.
            self._streams[interrupt_id] = s
            logger.debug("ChunkQueue: lazy-created stream for %s (sync push path)", interrupt_id)
        elif session_id and not s.session_id:
            # Late-arriving session_id (e.g. ensure called sync first
            # without it, then async with it) — record it so close_session
            # can target this stream.
            s.session_id = session_id
        return s

    async def ensure(
        self, interrupt_id: str, session_id: str = "",
    ) -> _InterruptStream:
        """Create-or-get the stream for an interrupt."""
        async with self._lock:
            s = self._streams.get(interrupt_id)
            if s is None:
                s = _InterruptStream(interrupt_id, session_id=session_id)
                self._streams[interrupt_id] = s
                logger.debug("ChunkQueue: created stream for %s", interrupt_id)
            elif session_id and not s.session_id:
                s.session_id = session_id
            return s

    def push(self, interrupt_id: str, chunk: dict, session_id: str = "") -> None:
        """Non-blocking push. Auto-creates the stream if absent so the
        first chunk (which usually arrives before any subscriber connects)
        is preserved in `history` for replay on subscribe()."""
        s = self._ensure_sync(interrupt_id, session_id=session_id)
        s.push(chunk)
        # Debug log: helps trace whether the issue is push-side (no chunk
        # pushed) or deliver-side (chunk pushed but subscriber didn't get
        # it). Includes subscriber count so we can tell if the stream is
        # going to a live SSE or just into history for later replay.
        _kind = chunk.get("node") or chunk.get("type") or chunk.get("node_step", "?")[:30]
        logger.info(
            "ChunkQueue[%s]: push kind=%s subscribers=%d history_len=%d",
            interrupt_id[:12], _kind, s.subscriber_count, len(s.history),
        )

    def complete(self, interrupt_id: str) -> None:
        """Mark end-of-stream so subscribers exit."""
        s = self._streams.get(interrupt_id)
        if s is not None:
            s.complete()
            logger.info(
                "ChunkQueue[%s]: completed stream (history_len=%d subscribers=%d)",
                interrupt_id[:12], len(s.history), s.subscriber_count,
            )

    async def subscribe(
        self,
        interrupt_id: str,
        *,
        replay_history: bool = True,
        poll_timeout: float = 0.5,
    ) -> AsyncIterator[dict]:
        """Async iterator over chunks for this interrupt.

        Implementation: each subscriber keeps a private index into the
        stream's history list. On each iteration, yield any new history
        entries since last seen, then await new_chunk.set() (or done).

        Args:
          replay_history: if True (default), start the cursor at 0 so the
                          subscriber sees every chunk emitted so far. If
                          False, skip to the current end (tail-only mode).
          poll_timeout:   how long to wait for the new_chunk event before
                          re-checking the done flag. Mostly cosmetic — done
                          also sets new_chunk so wake-ups are immediate in
                          practice.
        """
        s = await self.ensure(interrupt_id)
        s.subscriber_count += 1

        # Private cursor: index of next chunk to yield from history.
        cursor = 0 if replay_history else len(s.history)

        try:
            while True:
                # Drain any history entries we haven't yielded yet
                while cursor < len(s.history):
                    yield s.history[cursor]
                    cursor += 1

                # All caught up — exit if stream is done, else wait for more
                if s.done.is_set():
                    break

                # Clear the event before waiting so we only wake on NEW arrivals.
                # If a chunk arrives between our check above and clear() here,
                # we'll catch it on the next loop iteration via the cursor.
                s.new_chunk.clear()
                try:
                    await asyncio.wait_for(s.new_chunk.wait(), timeout=poll_timeout)
                except asyncio.TimeoutError:
                    # Loop and re-check done / history again
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

    async def close_session_streams(self, session_id: str) -> int:
        """Complete + drop every live stream tagged with this session_id.

        Called by chat_stream when a new operator turn starts on a
        session — any HITL sub-streams from prior turns are no longer
        relevant to the UI and should not race into the new turn's
        display. Returns the number of streams closed.

        Streams without a recorded session_id are left alone (we can't
        tell which session they belong to).
        """
        if not session_id:
            return 0
        async with self._lock:
            to_close = [
                iid for iid, s in self._streams.items()
                if s.session_id == session_id and not s.done.is_set()
            ]
            for iid in to_close:
                s = self._streams.get(iid)
                if s is not None:
                    s.complete()
                    # Don't pop from _streams — late SSE subscribers may
                    # still want to see the trailing chunks via history
                    # replay. gc/close will reap them later.
        if to_close:
            logger.info(
                "ChunkQueue: closed %d stream(s) for session=%s (new turn started)",
                len(to_close), session_id[:12],
            )
        return len(to_close)

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

    async def _idle_sweep_once(self) -> int:
        """Single pass of idle-timeout watchdog.

        Completes (but does not drop) streams whose last_activity_at is
        older than idle_timeout. The next gc() pass reaps them once
        subscribers detach.
        """
        now = time.time()
        async with self._lock:
            to_idle = [
                (iid, s) for iid, s in self._streams.items()
                if not s.done.is_set()
                and (now - s.last_activity_at) > self._idle_timeout
            ]
        for iid, s in to_idle:
            # Inject a synthetic terminal chunk so subscribers see WHY
            # the stream ended — otherwise the UI shows an abrupt cutoff.
            s.push({
                "type":   "idle_timeout",
                "message": (
                    f"HITL sub-stream timed out after "
                    f"{int(self._idle_timeout)}s with no activity. "
                    f"The agent loop may have hung. Check server logs "
                    f"for the underlying LLM call."
                ),
                "interrupt_id": iid,
            })
            s.complete()
            logger.warning(
                "ChunkQueue[%s]: idle-timeout after %.1fs of silence "
                "(subscribers=%d history_len=%d session=%s) — forcing complete",
                iid[:12], max(0.0, now - s.last_activity_at),
                s.subscriber_count, len(s.history),
                (s.session_id[:12] if s.session_id else "?"),
            )
        return len(to_idle)

    async def _idle_watchdog(self) -> None:
        """Background loop: every idle_check_interval seconds, sweep idle
        streams. Started by start_idle_watchdog() at app startup."""
        try:
            while True:
                await asyncio.sleep(self._idle_check_interval)
                try:
                    await self._idle_sweep_once()
                except Exception as exc:
                    logger.warning("ChunkQueue idle watchdog sweep failed: %s", exc)
        except asyncio.CancelledError:
            logger.info("ChunkQueue idle watchdog cancelled")
            raise

    def start_idle_watchdog(self) -> None:
        """Start the background idle-timeout watchdog task.

        Idempotent — safe to call multiple times (e.g. across app
        reload). The watchdog cancels itself when the event loop shuts
        down via asyncio.CancelledError.

        Must be called from within a running event loop (typically from
        a FastAPI startup hook).
        """
        if self._idle_task is not None and not self._idle_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
            self._idle_task = loop.create_task(
                self._idle_watchdog(),
                name="ChunkQueueIdleWatchdog",
            )
            logger.info(
                "ChunkQueue: idle watchdog started "
                "(timeout=%.0fs, check_interval=%.0fs)",
                self._idle_timeout, self._idle_check_interval,
            )
        except RuntimeError:
            # No running loop — caller should invoke from startup hook
            logger.debug(
                "ChunkQueue.start_idle_watchdog called outside event loop; "
                "deferring until first push or subscribe."
            )


# Process-wide singleton — webui/backend.py + hitl_executor.py both use this
_GLOBAL: Optional[HitlChunkQueueRegistry] = None


def get_chunk_queue_registry() -> HitlChunkQueueRegistry:
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = HitlChunkQueueRegistry()
    return _GLOBAL