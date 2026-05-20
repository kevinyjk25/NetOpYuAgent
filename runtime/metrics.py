"""
runtime/metrics.py — Prometheus metrics (optional, graceful no-op)
==================================================================

C1 (Sprint 3, 2026-05): expose agent metrics in Prometheus / OpenMetrics
text format at GET /metrics.

Design mirrors runtime/tracing.py:
  - prometheus_client is an OPTIONAL dependency. If it isn't installed,
    every record_*() call becomes a no-op and /metrics returns a short
    plaintext notice instead of crashing.
  - Counters / gauges / histograms are module-level singletons created
    lazily on first use, so importing this module is cheap and side-effect
    free when metrics aren't wired.

What's instrumented
───────────────────
  netopyu_llm_calls_total{model, outcome}        Counter
  netopyu_llm_call_duration_seconds{model}        Histogram
  netopyu_tool_calls_total{tool, outcome}         Counter
  netopyu_hitl_pending                            Gauge
  netopyu_active_llm_calls                         Gauge (in-flight, D1 semaphore)

Call sites record via the thin helpers (record_llm_call, record_tool_call,
set_hitl_pending, …). Those helpers no-op when prometheus_client is absent,
so call sites never branch on availability.

Enabling
────────
  pip install prometheus-client
  Then GET /metrics returns the live text exposition. No config flag —
  if the package is present, metrics are collected; the endpoint always
  exists (returns a notice when the package is missing).

Module independence
───────────────────
  Imports nothing from other internal modules. Safe to import from
  webui/routes_system.py + any call site.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Probe for prometheus_client ──────────────────────────────────────────
_PROM_AVAILABLE: bool = False
try:
    from prometheus_client import (              # type: ignore
        Counter, Gauge, Histogram,
        CONTENT_TYPE_LATEST, generate_latest,
        CollectorRegistry,
    )
    _PROM_AVAILABLE = True
except ImportError:
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"  # type: ignore


# ── Collectors (created once, lazily) ────────────────────────────────────
_REGISTRY: Any = None
_llm_calls: Any = None
_llm_duration: Any = None
_tool_calls: Any = None
_hitl_pending: Any = None
_active_llm: Any = None
_INITIALIZED = False


def _init() -> None:
    """Create the collectors. Idempotent. No-op when prometheus absent."""
    global _REGISTRY, _llm_calls, _llm_duration, _tool_calls
    global _hitl_pending, _active_llm, _INITIALIZED
    if _INITIALIZED or not _PROM_AVAILABLE:
        _INITIALIZED = True
        return

    # Dedicated registry (not the global default) so we don't pick up
    # process/GC collectors we didn't ask for, and so tests can reset.
    _REGISTRY = CollectorRegistry()
    _llm_calls = Counter(
        "netopyu_llm_calls_total",
        "Total LLM calls by model and outcome.",
        ["model", "outcome"],
        registry=_REGISTRY,
    )
    _llm_duration = Histogram(
        "netopyu_llm_call_duration_seconds",
        "LLM call wall-clock duration in seconds, by model.",
        ["model"],
        # Buckets tuned for local Ollama: sub-second to ~2min.
        buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 40, 80, 120),
        registry=_REGISTRY,
    )
    _tool_calls = Counter(
        "netopyu_tool_calls_total",
        "Total tool dispatches by tool name and outcome.",
        ["tool", "outcome"],
        registry=_REGISTRY,
    )
    _hitl_pending = Gauge(
        "netopyu_hitl_pending",
        "Current number of pending HITL approvals.",
        registry=_REGISTRY,
    )
    _active_llm = Gauge(
        "netopyu_active_llm_calls",
        "Current number of in-flight LLM calls (gated by the D1 semaphore).",
        registry=_REGISTRY,
    )
    _INITIALIZED = True
    logger.info("Metrics: prometheus_client collectors initialised")


def is_available() -> bool:
    """True iff prometheus_client is installed."""
    return _PROM_AVAILABLE


# ── Recording helpers — all no-op when prometheus absent ─────────────────

def record_llm_call(model: str, outcome: str, duration_s: Optional[float] = None) -> None:
    """Record one LLM call. outcome ∈ {ok, error, timeout}."""
    if not _PROM_AVAILABLE:
        return
    _init()
    try:
        _llm_calls.labels(model=model or "unknown", outcome=outcome).inc()
        if duration_s is not None:
            _llm_duration.labels(model=model or "unknown").observe(max(0.0, duration_s))
    except Exception as exc:
        logger.debug("metrics.record_llm_call failed: %s", exc)


def record_tool_call(tool: str, outcome: str) -> None:
    """Record one tool dispatch. outcome ∈ {ok, error, not_found}."""
    if not _PROM_AVAILABLE:
        return
    _init()
    try:
        _tool_calls.labels(tool=tool or "unknown", outcome=outcome).inc()
    except Exception as exc:
        logger.debug("metrics.record_tool_call failed: %s", exc)


def set_hitl_pending(n: int) -> None:
    """Set the pending-HITL gauge."""
    if not _PROM_AVAILABLE:
        return
    _init()
    try:
        _hitl_pending.set(max(0, int(n)))
    except Exception as exc:
        logger.debug("metrics.set_hitl_pending failed: %s", exc)


@contextmanager
def track_active_llm():
    """Context manager: inc the in-flight LLM gauge for the block's duration."""
    if not _PROM_AVAILABLE:
        yield
        return
    _init()
    try:
        _active_llm.inc()
    except Exception:
        pass
    try:
        yield
    finally:
        try:
            _active_llm.dec()
        except Exception:
            pass


@contextmanager
def time_llm_call(model: str):
    """Context manager: time an LLM call + record outcome automatically.

    Usage:
        with time_llm_call("qwen3.5:27b"):
            result = await engine._chat_impl(...)
    Records outcome=ok on clean exit, outcome=error on exception.
    """
    _start = time.monotonic()
    _outcome = "ok"
    try:
        yield
    except Exception:
        _outcome = "error"
        raise
    finally:
        record_llm_call(model, _outcome, time.monotonic() - _start)


def render_latest() -> tuple[bytes, str]:
    """Return (body, content_type) for the /metrics endpoint.

    When prometheus_client isn't installed, returns a short plaintext
    notice (still a 200 so scrapers don't alarm on a hard failure) with
    the standard content type.
    """
    if not _PROM_AVAILABLE:
        body = (
            b"# prometheus_client not installed.\n"
            b"# pip install prometheus-client to enable metrics.\n"
        )
        return body, "text/plain; charset=utf-8"
    _init()
    try:
        return generate_latest(_REGISTRY), CONTENT_TYPE_LATEST
    except Exception as exc:
        logger.warning("metrics.render_latest failed: %s", exc)
        return (f"# metrics render error: {exc}\n".encode(), "text/plain; charset=utf-8")
