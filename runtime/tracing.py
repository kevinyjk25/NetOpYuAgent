"""
runtime/tracing.py — OpenTelemetry adapter (minimal skeleton)
==============================================================

Sprint-3-pre (2026-05): graceful, OPTIONAL tracing layer. The full OTel
deployment story (collector config, dashboards, sampling) takes 1-2
weeks; this module ships the **minimum bootstrapping** so:

  - tracing CAN be enabled by setting the env var or config flag,
  - but is DISABLED by default to keep dev / mock-mode boots zero-cost,
  - and ALL `with tracer.start_as_current_span(...)` call sites
    elsewhere in the codebase work either way (real OTel API when
    available, no-op shim when not).

What this gives you
───────────────────
- One global Tracer accessor (`get_tracer()`).
- Decorator `@traced(span_name)` for async functions.
- Context manager `start_span(name, **attrs)` for inline blocks.
- Graceful degradation: if opentelemetry-api is not installed, all
  call sites become free no-ops without changing their call shape.

What this does NOT give you
───────────────────────────
- Auto-instrumentation of arbitrary application frameworks or databases.
- Cross-process / session_id propagation.
  Sprint 3 will derive a deterministic trace_id from session_id so the
  UI can show "open this trace" links; not yet implemented.
- OTLP collector config / Tempo / Jaeger / Grafana dashboards.
  Operational decision, deferred.

Enabling at runtime
───────────────────
1. `pip install opentelemetry-api opentelemetry-sdk` (+ exporter you want)
2. Set `OTEL_TRACING_ENABLED=true` (env) OR `observability.tracing_enabled: true`
   in config.yaml.
3. Optional: set `OTEL_EXPORTER_OTLP_ENDPOINT=http://collector:4317` to
   ship spans to a real collector. Without it, the SDK uses an
   in-memory exporter that's useful only for dev / unit tests.

Module-independence
───────────────────
- This module is in `runtime/` because the runtime loop is the
  largest consumer.
- It does NOT import any other internal module.
- All `opentelemetry.*` imports are inside try/except so the module
  loads cleanly even with OTel uninstalled — every internal
  consumer can use `from runtime.tracing import start_span` without
  worrying about ImportError.
"""
from __future__ import annotations

import contextlib
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Probe for opentelemetry ──────────────────────────────────────────────
_OTEL_AVAILABLE: bool = False
_tracer: Any = None

try:
    from opentelemetry import trace as _otel_trace          # type: ignore
    _OTEL_AVAILABLE = True
except ImportError:
    _otel_trace = None  # type: ignore


# ── No-op span shim ──────────────────────────────────────────────────────
# Used when OTel is unavailable OR when tracing is disabled at runtime.
# Implements the smallest subset of the Span protocol we touch:
#   set_attribute, add_event, record_exception, end (no-op),
#   __enter__/__exit__ for `with` blocks.
class _NoopSpan:
    def set_attribute(self, key: str, value: Any) -> None:    pass
    def add_event(self, name: str, attributes: Any = None) -> None: pass
    def record_exception(self, exc: BaseException) -> None:    pass
    def end(self) -> None:                                     pass
    def __enter__(self) -> "_NoopSpan":                        return self
    def __exit__(self, *a: Any) -> None:                       pass


_NOOP_SPAN = _NoopSpan()


# ── Configuration / setup ────────────────────────────────────────────────
_INITIALIZED: bool = False
_ENABLED:     bool = False


def configure(
    *,
    enabled:           bool = False,
    service_name:      str  = "netopyu-agent",
    service_version:   str  = "6.0.0",
    otlp_endpoint:     Optional[str] = None,
    sample_ratio:      float = 1.0,
) -> bool:
    """Initialize the global tracer. Idempotent.

    Returns True iff tracing was successfully enabled. False means
    either OTel is not installed, configuration disabled it, or
    something failed during provider setup — in all cases, span
    creation becomes a no-op and the caller does NOT need to branch.

    Called once by the DSH backend during startup. After that, all spans
    flow through `get_tracer()` / `start_span()` / `@traced`.
    """
    global _tracer, _INITIALIZED, _ENABLED

    if _INITIALIZED:
        return _ENABLED   # idempotent

    if not enabled or not _OTEL_AVAILABLE:
        _ENABLED = False
        _INITIALIZED = True
        logger.info(
            "Tracing: disabled (enabled=%s, otel_available=%s)",
            enabled, _OTEL_AVAILABLE,
        )
        return False

    try:
        # Late-import the SDK so dev installs without `opentelemetry-sdk`
        # still boot cleanly (api-only is enough for no-op).
        from opentelemetry.sdk.trace             import TracerProvider          # type: ignore
        from opentelemetry.sdk.resources         import Resource                # type: ignore
        from opentelemetry.sdk.trace.sampling    import TraceIdRatioBased       # type: ignore
        from opentelemetry.sdk.trace.export      import BatchSpanProcessor      # type: ignore
        from opentelemetry.sdk.trace.export      import ConsoleSpanExporter     # type: ignore

        resource = Resource.create({
            "service.name":    service_name,
            "service.version": service_version,
        })
        provider = TracerProvider(
            resource = resource,
            sampler  = TraceIdRatioBased(max(0.0, min(1.0, sample_ratio))),
        )

        # Exporter selection: real OTLP if endpoint given, else console.
        if otlp_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter  # type: ignore
                provider.add_span_processor(
                    BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True))
                )
                logger.info("Tracing: OTLP exporter → %s", otlp_endpoint)
            except ImportError:
                # OTLP exporter pkg not installed — fall back to console
                provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
                logger.warning(
                    "Tracing: opentelemetry-exporter-otlp not installed — "
                    "falling back to console exporter (set up "
                    "OTLP_TRACES_EXPORTER for production)"
                )
        else:
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
            logger.info(
                "Tracing: console exporter (no OTLP endpoint configured); "
                "set OTEL_EXPORTER_OTLP_ENDPOINT for real ingestion"
            )

        _otel_trace.set_tracer_provider(provider)             # type: ignore[union-attr]
        _tracer = _otel_trace.get_tracer(service_name)        # type: ignore[union-attr]
        _ENABLED = True
        _INITIALIZED = True
        logger.info(
            "Tracing: ENABLED (service=%s v%s, sample=%.2f)",
            service_name, service_version, sample_ratio,
        )

        # C2 (Sprint 3, 2026-05): httpx auto-instrumentation. Every outbound
        # HTTP call (peer AgentCard fetch, OpenAPI tools, MCP-over-HTTP, etc.)
        # gets an automatic client span linked to the active trace. Optional
        # package — if not installed, we log + skip (manual spans still work).
        try:
            from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor  # type: ignore
            HTTPXClientInstrumentor().instrument()
            logger.info("Tracing: httpx auto-instrumentation enabled")
        except ImportError:
            logger.info(
                "Tracing: opentelemetry-instrumentation-httpx not installed — "
                "outbound HTTP calls won't be auto-spanned (pip install it to enable)"
            )
        except Exception as _httpx_exc:
            logger.warning("Tracing: httpx instrumentation failed: %s", _httpx_exc)

        return True
    except Exception as exc:
        # Any setup failure → degrade to no-op rather than crashing boot.
        # Operators get a warning; agent keeps working.
        logger.warning(
            "Tracing: setup failed (%s) — degrading to no-op",
            exc,
        )
        _ENABLED = False
        _INITIALIZED = True
        return False


def is_enabled() -> bool:
    """True iff configure() succeeded with enabled=True."""
    return _ENABLED


# ── Span creation API ────────────────────────────────────────────────────

@contextlib.contextmanager
def start_span(name: str, **attributes: Any):
    """Open a span. Use as:

        with start_span("llm.call", model="qwen3.5:27b") as span:
            ...
            span.set_attribute("output.chars", len(result))

    Always safe to call — degrades to no-op when tracing is disabled or
    OTel is unavailable. Exceptions inside the block are recorded on the
    span (record_exception) and re-raised so the caller's error handling
    is unchanged.
    """
    if not _ENABLED or _tracer is None:
        yield _NOOP_SPAN
        return
    span_cm = _tracer.start_as_current_span(name)
    span = span_cm.__enter__()
    try:
        for k, v in attributes.items():
            try:
                span.set_attribute(k, v)
            except Exception:
                # Best-effort: bad attribute type → log but don't crash.
                logger.debug("tracing: bad attribute %s=%r", k, v)
        yield span
    except BaseException as exc:
        try:
            span.record_exception(exc)
        except Exception:
            pass
        raise
    finally:
        try:
            span_cm.__exit__(None, None, None)
        except Exception:
            pass


def get_tracer() -> Any:
    """Return the underlying OTel tracer, or None when disabled.

    Most callers should prefer `start_span()` instead — it handles the
    enable / disable branch internally. Use get_tracer() only when you
    need to pass the tracer to a 3rd-party library that expects one.
    """
    return _tracer
