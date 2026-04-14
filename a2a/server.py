"""
a2a/server.py
-------------
FastAPI sub-application that implements the full A2A HTTP surface:

  GET  /.well-known/agent-card.json  – agent discovery
  POST /                             – JSON-RPC (message/send, tasks/get, tasks/cancel)
  POST /stream                       – JSON-RPC streaming (message/stream → SSE)

Mount it into your main FastAPI app::

    from a2a.server import create_a2a_app
    app.mount("/api/v1/a2a", create_a2a_app(base_url="http://localhost:8000/api/v1/a2a"))
"""

from __future__ import annotations

import logging
import sys

# ── Guard against direct execution ───────────────────────────────────────────
# server.py is a sub-module of the `a2a` package.  Running it directly
# (e.g. `python a2a/server.py`) strips the package context so every
# relative import (`from .schemas import ...`) raises:
#   ImportError: attempted relative import with no known parent package
#
# The correct entry point is main.py at the project root:
#   uvicorn main:app --reload          # recommended
#   python  main.py                    # also works
if __name__ == "__main__":
    print(
        "\n[ERROR] a2a/server.py cannot be run directly.\n"
        "        It is an internal sub-module of the `a2a` package.\n\n"
        "        Run the application from the project root instead:\n"
        "          uvicorn main:app --reload\n"
        "          python  main.py\n",
        file=sys.stderr,
    )
    sys.exit(1)

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .agent_card import get_agent_card
from .agent_executor import ITOpsAgentExecutor
from .request_handler import DefaultRequestHandler
from .schemas import JsonRpcResponse
from .task_store import InMemoryTaskStore

logger = logging.getLogger(__name__)


def create_a2a_app(
    base_url: str,
    executor: ITOpsAgentExecutor | None = None,
    task_store: InMemoryTaskStore | None = None,
) -> FastAPI:
    """
    Build and return a self-contained A2A FastAPI sub-application.

    Parameters
    ----------
    base_url:
        Public URL at which this sub-app is reachable, e.g.
        ``"https://myservice.com/api/v1/a2a"``.
        Used in the AgentCard ``url`` field.
    executor:
        Optional custom AgentExecutor.  Defaults to ``ITOpsAgentExecutor()``.
    task_store:
        Optional custom TaskStore.  Defaults to ``InMemoryTaskStore()``.

    Returns
    -------
    FastAPI
        Mount this with ``app.mount("/api/v1/a2a", create_a2a_app(...))``.
    """
    _executor = executor or ITOpsAgentExecutor()
    _store = task_store or InMemoryTaskStore()
    _handler = DefaultRequestHandler(agent_executor=_executor, task_store=_store)
    _agent_card = get_agent_card(base_url)

    a2a = FastAPI(title="IT Ops A2A Agent", version="1.0.0", docs_url=None)

    a2a.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Agent discovery
    # ------------------------------------------------------------------

    @a2a.get("/.well-known/agent-card.json", tags=["A2A"])
    async def agent_card() -> JSONResponse:
        """Return the AgentCard so that clients can discover this agent."""
        return JSONResponse(content=_agent_card)

    # Alias used by A2A Inspector
    @a2a.get("/agent-card", tags=["A2A"])
    async def agent_card_alias() -> JSONResponse:
        return JSONResponse(content=_agent_card)

    # ------------------------------------------------------------------
    # JSON-RPC  (sync: message/send, tasks/get, tasks/cancel)
    # ------------------------------------------------------------------

    @a2a.post("/", tags=["A2A"])
    async def jsonrpc(request: Request) -> JSONResponse:
        body = await _parse_body(request)
        method: str = body.get("method", "")
        params: dict = body.get("params", {})
        req_id = body.get("id")

        if method == "message/stream":
            raise HTTPException(
                status_code=400,
                detail="Use POST /stream for streaming requests.",
            )

        try:
            result = await _handler.handle(method, params)
            return JSONResponse(
                content=JsonRpcResponse(result=result, id=req_id).model_dump()
            )
        except Exception as exc:
            logger.exception("JSON-RPC error method=%s: %s", method, exc)
            return JSONResponse(
                status_code=500,
                content=JsonRpcResponse(
                    error={"code": -32603, "message": str(exc)},
                    id=req_id,
                ).model_dump(),
            )

    # ------------------------------------------------------------------
    # Streaming endpoint  (message/stream → SSE)
    # ------------------------------------------------------------------

    @a2a.post("/stream", tags=["A2A"])
    async def jsonrpc_stream(request: Request) -> StreamingResponse:
        body = await _parse_body(request)
        params: dict = body.get("params", {})

        async def sse_generator():
            try:
                async for chunk in await _handler.handle("message/stream", params):
                    yield chunk
            except Exception as exc:
                logger.exception("Streaming error: %s", exc)
                yield f"data: {{\"error\": \"{exc}\"}}\n\n"

        return StreamingResponse(
            sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # disable nginx buffering
            },
        )

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    @a2a.get("/health", tags=["A2A"])
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok", "agent": _agent_card["name"]})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _parse_body(request: Request) -> dict:
        try:
            return await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

    return a2a
