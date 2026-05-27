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

# httpx is imported lazily inside _stream_request (the only place it's used)
# so that importing this module — and the task package — does not hard-require
# httpx. This lets delegation wiring + unit tests load in environments without
# httpx installed; the dependency is only needed when actually dispatching to
# a remote peer over HTTP.

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
# Heartbeat wrapper for streaming delegation
# ---------------------------------------------------------------------------

async def _with_heartbeat(
    upstream: "AsyncIterator[dict[str, Any]]",
    *,
    heartbeat_s: float,
    agent_id: str,
) -> "AsyncIterator[dict[str, Any]]":
    """Forward chunks from ``upstream`` and inject a heartbeat node_step
    every ``heartbeat_s`` seconds of silence.

    Design rationale
    ----------------
    The delegating side's SSE pipeline cancels the request when the
    chunk_queue is idle longer than ``sse_stall_timeout_seconds``
    (default 300s). A peer running a slow local LLM can be silent for
    minutes before its first token, so we synthesise a no-op chunk
    periodically. The chunk carries only a ``node_step`` string and
    intentionally NO ``token`` / ``message`` / ``node_result`` — the
    parent's _handle_delegate uses tokens/message text to build the
    final synthesis context, so a heartbeat with token content would
    pollute that. The Flow tab on the parent UI gets a "peer working"
    event though, which is what the operator actually wants to see.

    Implementation: background task drains ``upstream`` into an asyncio
    Queue; the foreground does ``wait_for(queue.get, heartbeat_s)`` and
    falls back to a heartbeat chunk on timeout. Done sentinel signals
    end-of-stream. Exceptions in the drainer are forwarded as a special
    sentinel so they're re-raised in the foreground.
    """
    import asyncio

    _END  = object()    # normal end of stream
    _ERR  = object()    # exception sentinel; followed by the exception

    queue: "asyncio.Queue[Any]" = asyncio.Queue()

    async def _drainer() -> None:
        try:
            async for chunk in upstream:
                await queue.put(chunk)
        except BaseException as exc:  # noqa: BLE001 — forward everything incl. CancelledError
            await queue.put(_ERR)
            await queue.put(exc)
            return
        await queue.put(_END)

    drain_task = asyncio.create_task(_drainer())
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=heartbeat_s)
            except asyncio.TimeoutError:
                # No real chunk in the heartbeat window — emit one ourselves
                # so the parent's chunk_queue stays alive. Keep the cadence
                # going by looping back to wait_for again.
                yield {
                    "node_step": f"peer {agent_id} still working...",
                    "node":       "delegate",
                    "heartbeat":  True,
                }
                continue
            if item is _END:
                return
            if item is _ERR:
                exc = await queue.get()
                raise exc
            yield item
    finally:
        if not drain_task.done():
            drain_task.cancel()
            try:
                await drain_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# A2A Task Dispatcher
# ---------------------------------------------------------------------------

class A2ATaskDispatcher:
    """
    Sends a TaskDefinition to a remote A2A agent via message/stream.

    Returns an async generator that yields chunk dicts compatible with
    the 6-processor chain in a2a/agent_executor.py + hitl/a2a_integration.py.
    """

    def __init__(self, http_timeout: float = 300.0) -> None:
        # 300s default: a delegated peer runs its FULL agent loop (query
        # classification + multiple LLM turns + tools) before the first
        # content token, and on local models (e.g. Ollama qwen3.5:27b on
        # consumer hardware) that gap can be 2-3 minutes. The peer emits a
        # `working` status quickly, but the read-gap until the first real
        # token is the whole loop — so this must be generous. Override via
        # build_delegate_fn if needed. _stream_request builds a granular
        # httpx.Timeout so connect stays short while read is lenient.
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

        # Build the dispatch payload. We pack `task.metadata` (which carries
        # delegated_by / forked / shared_facts_count, plus anything the
        # caller put there) into params.metadata so the peer's
        # agent_executor can read it from `context.metadata` and attach
        # provenance to any HITL cards it creates. Without this, the peer
        # has no way to know it's processing a delegation — the operator
        # sees a bare approval card with no hint of who's upstream.
        # task_id / session_id stay at the top level for compatibility
        # with the peer-side keys that already exist; metadata
        # spread comes last so it can't accidentally overwrite them.
        _meta: dict[str, Any] = {
            "task_id":    task.task_id,
            "session_id": task.session_id,
        }
        # task.parameters can be empty; only spread when actually set
        if task.parameters:
            _meta.update(task.parameters)
        # task.metadata carries delegated_by / forked / shared_facts_count
        if task.metadata:
            for _k, _v in task.metadata.items():
                # Don't let metadata clobber our required keys
                if _k not in ("task_id", "session_id"):
                    _meta[_k] = _v

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
                "metadata": _meta,
            },
            "id": 1,
        }

        # Wrap the peer stream with a heartbeat layer so the delegating
        # side's SSE chunk_queue gets at least one chunk every
        # `heartbeat_s` seconds even when the peer is silently crunching
        # its own LLM call. Without this, a slow peer (e.g. qwen3.5:27b
        # doing query classification + Turn 1) can silence the parent
        # for 3-5 minutes — long enough to trip the parent's SSE
        # `sse_stall_timeout_seconds` (default 300s) and cancel the
        # whole delegation before the peer even streams its first token
        # (observed 2026-05: LAN → dc-agent → 5 min silence → "LLM
        # backend did not respond within 300s"). The heartbeat yields a
        # bare node_step the parent runtime loop's _handle_delegate
        # ignores for synthesis purposes but the parent SSE chunk_queue
        # counts as activity, so the request stays alive. heartbeat_s
        # is intentionally well under sse_stall_timeout_seconds.

        # Accumulate the peer's token stream so we can write a meaningful
        # `result` to the outbound TaskDefinition on completion. Without
        # this, the LAN-side Delegations tab shows a task stuck in RUNNING
        # forever even after the peer is done (observed 2026-05).
        _result_parts: list[str] = []
        _peer_hitl    = False
        _peer_interrupt_id = None
        _peer_error: Optional[str] = None
        _stream_ok    = True
        try:
            async for chunk in _with_heartbeat(
                self._stream_request(assignment.agent_url, body),
                heartbeat_s = 30.0,
                agent_id    = assignment.agent_id,
            ):
                # Mirror what runtime/loop.py:_handle_delegate does for its
                # _result_parts accumulator, in case the runtime loop isn't
                # involved (some callers consume dispatcher directly).
                if isinstance(chunk, dict):
                    if chunk.get("type") == "hitl_interrupt" or chunk.get("hitl_interrupt"):
                        _peer_hitl = True
                        # Capture the peer's interrupt_id so the outbound task
                        # can be correlated when the peer's operator resolves it
                        # (A2A Phase 3 cross-agent HITL passthrough).
                        _iid = chunk.get("interrupt_id")
                        if _iid:
                            _peer_interrupt_id = _iid
                    if chunk.get("error"):
                        _peer_error = str(chunk["error"])
                    tok = chunk.get("token") or chunk.get("message") or ""
                    if tok:
                        _result_parts.append(str(tok))
                yield chunk
        except Exception as exc:
            _stream_ok = False
            _peer_error = str(exc) or type(exc).__name__
            # Re-raise so callers (delegate_fn's try/except) can also see it
            raise
        finally:
            # Update terminal state on the outbound TaskDefinition so the
            # LAN-side Delegations tab transitions away from RUNNING. Done
            # in finally so we always write SOMETHING — even on cancellation
            # or unexpected exception — rather than leaving a ghost row.
            try:
                from datetime import datetime, timezone
                _result_text = "".join(_result_parts).strip()
                if _peer_error:
                    task.state = TaskState.FAILED
                    task.error = _peer_error[:500]
                elif _peer_hitl:
                    # Peer is awaiting ITS OWN operator's approval (mode B:
                    # the delegated-to side approves). Mark a dedicated
                    # non-terminal state + carry the peer agent_id and the
                    # peer's interrupt_id so the cross-agent resume callback
                    # (A2A Phase 3 P3-b) can correlate the resolution back to
                    # this outbound delegation.
                    task.state = TaskState.AWAITING_PEER_HITL
                    task.metadata["peer_hitl_pending"] = True
                    task.metadata["peer_agent"] = assignment.agent_id
                    if _peer_interrupt_id:
                        task.metadata["peer_interrupt_id"] = _peer_interrupt_id
                        # Register the awaiting record so the peer's later
                        # /hitl_resolved callback can find this local session
                        # to resume (A2A Phase 3 P3-b).
                        try:
                            from task.inter.cross_agent_hitl import (
                                get_cross_agent_hitl_bridge,
                            )
                            get_cross_agent_hitl_bridge().record_awaiting_peer(
                                local_session_id=task.session_id,
                                peer_agent=assignment.agent_id,
                                peer_interrupt_id=_peer_interrupt_id,
                                correlation_id=task.metadata.get("correlation_id", ""),
                                original_query=task.metadata.get("source_query", ""),
                            )
                        except Exception as _br_exc:
                            logger.debug("awaiting-peer record skipped: %s", _br_exc)
                elif _stream_ok:
                    task.state = TaskState.COMPLETED
                    if _result_text:
                        task.result = {"text": _result_text}
                else:
                    task.state = TaskState.FAILED
                task.completed_at = datetime.now(timezone.utc).isoformat()
                await store.save(task)
                await store.write_audit(TaskAuditRecord(
                    task_id=task.task_id, session_id=task.session_id,
                    event_kind=TaskEventKind.COMPLETED
                        if task.state == TaskState.COMPLETED
                        else TaskEventKind.FAILED,
                    actor="a2a_dispatcher",
                    payload={"state": task.state.value,
                             "result_chars": len(_result_text),
                             "error": _peer_error or ""},
                ))
            except Exception as _save_exc:
                logger.debug("dispatcher terminal save failed: %s", _save_exc)

    async def _stream_request(
        self,
        agent_url: str,
        body: dict,
    ) -> AsyncIterator[dict[str, Any]]:
        stream_url = agent_url.rstrip("/") + "/stream"
        try:
            import httpx   # lazy: only needed when actually dispatching
            # Granular timeout: fail fast on connect, but allow a long gap
            # before the first/next byte since the peer's agent loop is slow.
            _to = httpx.Timeout(self._timeout, connect=10.0)
            async with httpx.AsyncClient(timeout=_to) as client:
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
                            raw = json.loads(data)
                        except Exception:
                            yield {"token": data + " "}
                            continue
                        # The peer streams A2A protocol events (Task / Message /
                        # Artifact update envelopes), not the runtime loop's
                        # flat {token, node_step, ...} chunks. Unwrap them into
                        # the flat shape the delegating loop's _handle_delegate
                        # understands. Without this, the loop sees the raw
                        # envelope (no top-level 'token') and accumulates
                        # nothing — the delegation appears to "succeed" but
                        # returns empty (observed 2026-05).
                        for chunk in self._unwrap_a2a_event(raw):
                            yield chunk
        except Exception as exc:
            # Surface a meaningful error — bare str(exc) is empty for several
            # httpx exception types (e.g. ReadTimeout, RemoteProtocolError),
            # which made this failure undebuggable in the field.
            _msg = str(exc) or f"{type(exc).__name__} (no message)"
            logger.error("A2ATaskDispatcher stream to %s failed: %s",
                         stream_url, _msg)
            # Yield an error chunk so the delegating loop can degrade
            # gracefully (inject a "delegation failed" note) instead of just
            # silently getting nothing.
            yield {"node_step": f"peer stream failed: {_msg}",
                   "node": "delegate", "error": _msg}

    @staticmethod
    def _unwrap_a2a_event(raw: dict[str, Any]) -> "list[dict[str, Any]]":
        """Translate one A2A protocol event dict into flat loop chunks.

        The peer's request_handler yields `event.model_dump_json()` for
        TaskArtifactUpdateEvent / TaskStatusUpdateEvent / MessageEvent. The
        token/message text lives at artifact.parts[*].data. We pull it up to
        the top level so the delegating runtime loop can forward + accumulate
        it. Unknown / structural events (status transitions) yield nothing
        (they're not user-facing content).
        """
        out: list[dict[str, Any]] = []
        if not isinstance(raw, dict):
            return out
        # Explicit peer-side error envelope (server sse_generator on exception).
        if "error" in raw and "artifact" not in raw and "status" not in raw:
            out.append({"node_step": f"peer error: {raw['error']}",
                        "node": "delegate", "error": str(raw["error"])})
            return out
        artifact = raw.get("artifact")
        if isinstance(artifact, dict):
            for part in artifact.get("parts", []) or []:
                data = part.get("data") if isinstance(part, dict) else None
                if not isinstance(data, dict):
                    # A plain text part.
                    txt = part.get("text") if isinstance(part, dict) else None
                    if txt:
                        out.append({"token": txt})
                    continue
                ptype = data.get("type")
                if ptype == "token" and data.get("token"):
                    out.append({"token": data["token"]})
                elif ptype == "tokens_batch":
                    for t in data.get("tokens", []) or []:
                        out.append({"token": str(t)})
                elif ptype == "message" and data.get("text"):
                    out.append({"message": data["text"],
                                "node": data.get("node", "peer")})
                elif data.get("token"):
                    out.append({"token": data["token"]})
                elif data.get("text"):
                    out.append({"message": data["text"]})
            return out
        # MessageEvent (final assistant message) — pull parts text as tokens.
        msg = raw.get("message")
        if isinstance(msg, dict):
            for part in msg.get("parts", []) or []:
                txt = part.get("text") if isinstance(part, dict) else None
                if txt and txt not in ("Task completed.",):
                    out.append({"token": txt})
            return out
        # TaskStatusUpdateEvent — surface progress so the delegating side
        # has something to push into its SSE chunk_queue while the peer is
        # still working on its own LLM call. Without this, only token /
        # message chunks generate output, and on a slow local model the peer
        # can take 3-5 minutes BEFORE its first token, during which the
        # delegating side's chunk_queue sees nothing and SSE
        # `sse_stall_timeout_seconds` (default 300s) cancels the request
        # (observed 2026-05: LAN delegates to dc-agent → 5 min silence →
        # "LLM backend did not respond within 300s" → cancel before peer's
        # first token streams back). We map non-terminal status states to a
        # node_step "peer working ..." chunk; terminal FAILED state keeps
        # its existing error-chunk behavior.
        status = raw.get("status")
        if isinstance(status, dict):
            st = (status.get("state") or "").lower()
            if st in ("failed",) and status.get("message"):
                out.append({"node_step": f"peer task failed: {status['message']}",
                            "node": "delegate", "error": str(status["message"])})
            elif st in ("input-required", "input_required"):
                # A2A Phase 3: the peer raised a HITL while serving our
                # delegation and is awaiting ITS OWN operator (mode B). The
                # peer's local interrupt_id is carried in status.message. Surface
                # it as a hitl_interrupt chunk so the dispatcher can mark the
                # outbound task AWAITING_PEER_HITL and register the awaiting
                # record (keyed by this interrupt_id) for the later
                # /hitl_resolved resume callback. Previously this status was
                # dropped, so the delegating side never learned the interrupt_id.
                _iid = str(status.get("message") or "").strip()
                out.append({
                    "type": "hitl_interrupt",
                    "hitl_interrupt": True,
                    "interrupt_id": _iid,
                    "node": "delegate",
                    "node_step": "peer requires operator approval (HITL)",
                })
            elif st in ("submitted", "working", "running"):
                # Progress signal — keeps the delegating-side SSE alive and
                # gives the operator a Flow event showing the peer is busy.
                # Brief on purpose (no payload) so a chatty peer can't flood
                # the parent's stream with status spam.
                out.append({"node_step": f"peer status: {st}",
                            "node": "delegate"})
            # canceled / completed: nothing to surface (caller already sees
            # MessageEvent / artifact stream end).
        return out


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
