"""
hitl_core.transport.sse_adapter — Convert pipeline events to Server-Sent Events.

OPTIONAL component. The pipeline yields async events; web UIs typically
want them as SSE chunks. This adapter handles the format conversion and
JSON serialisation.

Usage:

    @app.get("/agent/run")
    async def run_endpoint(query: str):
        pipeline = build_my_pipeline()
        state    = PipelineState(user_query=query)
        return EventSourceResponse(
            stream_pipeline_as_sse(pipeline, state)
        )

The adapter handles three event types:

  • token              → "data: {\"type\":\"token\",\"text\":\"...\"}\\n\\n"
  • interrupt          → "data: {\"type\":\"interrupt\",\"payload\":{...}}\\n\\n"
  • batch_interrupt    → "data: {\"type\":\"batch_interrupt\",\"batch\":{...}}\\n\\n"
  • done               → "data: {\"type\":\"done\",\"state\":{...}}\\n\\n"
  • aborted            → "data: {\"type\":\"aborted\",\"reason\":\"...\"}\\n\\n"

Custom event types can be added by passing an `extra_handler` callable.
"""

from __future__ import annotations


def _pipeline_poll_interval_ms() -> int:
    """Lazy-load pipeline poll interval from cfg.concurrency."""
    try:
        from config import cfg as _app_cfg
        return int(getattr(getattr(_app_cfg, "concurrency", None), "hitl_pipeline_poll_interval_ms", 50))
    except Exception:
        return 50



import json
import logging
from typing import Any, AsyncIterator, Callable, Optional

from ..pipeline import HitlPipeline, PipelineState

logger = logging.getLogger(__name__)


def _format_sse(data: dict[str, Any], event: Optional[str] = None) -> str:
    """Format a dict as one SSE message. event= sets the SSE event name
    (UI uses this to dispatch handlers); data is JSON-encoded."""
    out = ""
    if event:
        out += f"event: {event}\n"
    out += f"data: {json.dumps(data, default=str)}\n\n"
    return out


async def stream_pipeline_as_sse(
    pipeline: HitlPipeline,
    state: PipelineState,
    *,
    extra_handler: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
) -> AsyncIterator[str]:
    """Drive a pipeline and yield SSE-formatted strings.

    For HTTP responses, wrap with FastAPI's StreamingResponse:

        return StreamingResponse(
            stream_pipeline_as_sse(pipeline, state),
            media_type="text/event-stream",
        )

    Args:
      pipeline: a configured HitlPipeline.
      state:    the initial PipelineState.
      extra_handler: optional translator for unknown event types. Given
                     the raw event dict, returns the dict to send to the
                     UI (or None to skip).
    """
    try:
        async for event in pipeline.run(state, poll_interval_ms=_pipeline_poll_interval_ms()):
            etype = event.get("type")

            if etype == "token":
                # Pipelines emit tokens via ctx.emit_token; the run loop
                # drains them through the on_token hook (not via this
                # event). We don't expect "token" events here typically.
                # If a step explicitly yields a token-shaped event, just
                # forward it.
                yield _format_sse(event, event="token")

            elif etype == "interrupt":
                payload = event["payload"]
                yield _format_sse(
                    {
                        "type":         "interrupt",
                        "interrupt_id": payload.interrupt_id,
                        "payload":      payload.model_dump(mode="json"),
                    },
                    event="interrupt",
                )

            elif etype == "batch_interrupt":
                batch = event["batch"]
                yield _format_sse(
                    {
                        "type":     "batch_interrupt",
                        "batch_id": batch.batch_id,
                        "batch":    batch.model_dump(mode="json"),
                    },
                    event="batch_interrupt",
                )

            elif etype == "done":
                final = event["state"]
                # state may carry non-JSON things; we strip to a safe view
                yield _format_sse(
                    {
                        "type":     "done",
                        "metadata": final.metadata,
                        "tokens":   "".join(final.tokens),
                        "decisions": [d.model_dump(mode="json") for d in final.decisions],
                    },
                    event="done",
                )

            elif etype == "aborted":
                yield _format_sse(
                    {
                        "type":   "aborted",
                        "reason": event.get("reason", ""),
                        "kind":   event.get("kind", "aborted"),
                    },
                    event="aborted",
                )

            else:
                # Unknown event type — give the host a chance to handle
                if extra_handler is not None:
                    try:
                        translated = extra_handler(event)
                        if translated:
                            yield _format_sse(translated)
                    except Exception as exc:
                        logger.warning("extra_handler raised %s", exc)
                else:
                    logger.debug("Unhandled pipeline event: %s", etype)
    except Exception as exc:
        # Pipeline blew up — surface as an error event to the UI
        logger.exception("Pipeline streaming error")
        yield _format_sse(
            {
                "type":  "error",
                "error": str(exc),
            },
            event="error",
        )


# ---------------------------------------------------------------------------
# Token emit hook — wires ctx.emit_token directly into an SSE queue
# ---------------------------------------------------------------------------

class TokenSseChannel:
    """Bridge from pipeline ctx.emit_token to an SSE stream.

    Pipelines stream tokens via the on_token hook on PipelineContext
    (set when constructing the context). This class is a lightweight
    queue + iterator; pass `channel.on_token` as the hook, and the
    SSE consumer yields from `channel.iterate()`.

    Typical pattern (multiplexing tokens + lifecycle events):

        channel = TokenSseChannel()
        ctx     = PipelineContext(state=..., store=..., on_token=channel.on_token)

        # Run pipeline.run(state) in a task; merge channel + lifecycle
        # events into one SSE stream.

    Most hosts won't need this — emitting tokens through on_token and
    collecting them via state.tokens for the final "done" event is
    simpler. Use this only when real-time per-token streaming is required.
    """
    def __init__(self) -> None:
        import asyncio
        self._queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

    async def on_token(self, token: str) -> None:
        await self._queue.put(token)

    def close(self) -> None:
        self._queue.put_nowait(None)

    async def iterate(self) -> AsyncIterator[str]:
        while True:
            tok = await self._queue.get()
            if tok is None:
                return
            yield _format_sse({"type": "token", "text": tok}, event="token")