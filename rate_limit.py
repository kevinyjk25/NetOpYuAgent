"""
rate_limit.py
─────────────
Lightweight in-process rate limiting and concurrency control. No external
dependencies — uses asyncio primitives.

Two layers:
  1. Global concurrency cap on long-running operations (e.g. LLM streams).
     Prevents one client from exhausting LLM connections by holding many
     SSE streams open simultaneously.
  2. Per-operator request rate (token bucket). Enforces "X requests per
     minute" without needing Redis.

Both are FastAPI-compatible — used as Depends().

Usage
-----
    from rate_limit import per_operator_limit, global_concurrency

    @app.post("/chat/stream")
    async def chat_stream(
        req: ChatRequest,
        identity: Identity = Depends(verify_identity),
        _rate:    None     = Depends(per_operator_limit(rate_per_min=10)),
        _conc:    None     = Depends(global_concurrency(name="chat", limit=4)),
    ):
        ...
"""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Awaitable, Callable

from fastapi import HTTPException

from auth_core import Identity


# ── Per-operator token bucket ────────────────────────────────────────────────

class _TokenBucket:
    """Simple token-bucket per (operator, scope). Refills `rate` tokens
    over a `window` (seconds). Burst capacity == rate."""

    def __init__(self, rate: int, window: float = 60.0) -> None:
        self.rate     = rate
        self.window   = window
        self._tokens: dict[str, float] = defaultdict(lambda: float(rate))
        self._last:   dict[str, float] = defaultdict(lambda: time.monotonic())
        self._lock    = asyncio.Lock()

    async def consume(self, key: str, cost: float = 1.0) -> bool:
        async with self._lock:
            now      = time.monotonic()
            elapsed  = now - self._last[key]
            # Refill
            self._tokens[key] = min(
                self.rate,
                self._tokens[key] + elapsed * (self.rate / self.window),
            )
            self._last[key] = now
            if self._tokens[key] >= cost:
                self._tokens[key] -= cost
                return True
            return False


def per_operator_limit(rate_per_min: int = 30) -> Callable:
    """Factory: returns a FastAPI dependency that enforces `rate_per_min`
    requests per minute per operator. Returns 429 when exceeded."""
    bucket = _TokenBucket(rate=rate_per_min, window=60.0)

    async def _dep(identity: Identity) -> None:
        ok = await bucket.consume(identity.operator_id)
        if not ok:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded ({rate_per_min}/min). "
                        "Wait a moment and retry.",
                headers={"Retry-After": "60"},
            )
        return None

    return _dep


# ── Global concurrency cap ───────────────────────────────────────────────────

_semaphores: dict[str, asyncio.Semaphore] = {}


def global_concurrency(name: str, limit: int) -> Callable:
    """Factory: returns a dependency that holds a global semaphore for the
    duration of the request. Caps the total in-flight requests system-wide
    for the named scope.

    Important: FastAPI dependencies that yield (generator) wrap the request,
    so this is the canonical way to do per-request resource gates without
    middleware.
    """
    if name not in _semaphores:
        _semaphores[name] = asyncio.Semaphore(limit)
    sem = _semaphores[name]

    async def _dep():
        # Block until a slot is free; raise 503 if we've waited too long
        try:
            await asyncio.wait_for(sem.acquire(), timeout=30.0)
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=503,
                detail=f"Service overloaded — {limit} concurrent {name} requests in flight. "
                        "Try again in a few seconds.",
            )
        try:
            yield
        finally:
            sem.release()

    return _dep
