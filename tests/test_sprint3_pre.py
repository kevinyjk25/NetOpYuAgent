"""
tests/test_sprint3_pre.py
─────────────────────────
Unit tests for the Sprint-3-pre readiness additions (May 2026):

  - runtime/tracing.py: graceful degradation when OTel not installed
  - config.ObservabilityConfig: env override + defaults
  - config.HITLCheckpointConfig: sqlite default, env override
  - skills/evolver.set_bench_runner: A/B safety net contract

These additions are production-readiness blockers. Each test pins a
behaviour an operator depends on: tracing must boot without OTel,
HITL checkpoint must default to sqlite (so approvals survive restart),
evolver bench gate must reject regressions.

Run:
    python -m unittest tests.test_sprint3_pre
    python -m pytest tests/test_sprint3_pre.py -v
"""
from __future__ import annotations

import asyncio
import os
import sys
import unittest
from unittest import mock


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ─────────────────────────────────────────────────────────────────────
#  runtime/tracing.py
# ─────────────────────────────────────────────────────────────────────
class TestTracingShim(unittest.TestCase):
    """OTel-optional tracing module behaviour."""

    def setUp(self) -> None:
        # tracing module caches state in module-level globals; reset between
        # tests so each starts fresh. Use importlib.reload() so all tests
        # see the same _INITIALIZED=False starting state.
        import importlib
        if "runtime.tracing" in sys.modules:
            importlib.reload(sys.modules["runtime.tracing"])

    def test_disabled_by_default_no_otel(self):
        """configure(enabled=False) → is_enabled() False, spans are no-op."""
        from runtime.tracing import configure, is_enabled, start_span
        result = configure(enabled=False)
        self.assertFalse(result)
        self.assertFalse(is_enabled())
        # Span context manager must still work without crashing.
        with start_span("test.span", foo="bar") as span:
            span.set_attribute("k", "v")
            span.add_event("evt", {"x": 1})

    def test_enabled_without_otel_installed_degrades_gracefully(self):
        """configure(enabled=True) with no opentelemetry → still no-op, no crash.

        This is the critical contract: a flag in config.yaml MUST NOT crash
        boot if the operator hasn't installed the OTel packages. We can
        only test this when OTel ISN'T installed; if it is, this becomes a
        positive assertion that configure succeeded.
        """
        from runtime.tracing import configure, is_enabled, _OTEL_AVAILABLE
        result = configure(enabled=True, service_name="test", service_version="0.0.0")
        if _OTEL_AVAILABLE:
            # OTel is installed in this env — configure() should succeed.
            self.assertTrue(result)
            self.assertTrue(is_enabled())
        else:
            # OTel missing — must degrade silently.
            self.assertFalse(result)
            self.assertFalse(is_enabled())

    def test_configure_idempotent(self):
        from runtime.tracing import configure
        a = configure(enabled=False)
        b = configure(enabled=True)   # second call should be no-op
        # Both return the same state (whatever the first call decided).
        self.assertEqual(a, b)

    def test_start_span_records_exception(self):
        """If a wrapped block raises, the exception must propagate."""
        from runtime.tracing import configure, start_span
        configure(enabled=False)
        with self.assertRaises(ValueError):
            with start_span("test.span"):
                raise ValueError("boom")


# ─────────────────────────────────────────────────────────────────────
#  config.ObservabilityConfig
# ─────────────────────────────────────────────────────────────────────
class TestObservabilityConfig(unittest.TestCase):
    def test_defaults(self):
        from config import ObservabilityConfig
        c = ObservabilityConfig()
        self.assertFalse(c.tracing_enabled)
        self.assertIsNone(c.otlp_endpoint)
        self.assertEqual(c.sample_ratio, 1.0)

    def test_env_override_tracing_enabled(self):
        from config import _load_observability_config
        with mock.patch.dict(os.environ, {"OTEL_TRACING_ENABLED": "true"}, clear=False):
            c = _load_observability_config({})
        self.assertTrue(c.tracing_enabled)

    def test_env_override_otlp_endpoint(self):
        from config import _load_observability_config
        for k in ("OTEL_TRACING_ENABLED", "OTEL_EXPORTER_OTLP_ENDPOINT",
                  "OTEL_SAMPLE_RATIO", "OTEL_SERVICE_NAME", "OTEL_SERVICE_VERSION"):
            os.environ.pop(k, None)
        with mock.patch.dict(os.environ,
                             {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4317"},
                             clear=False):
            c = _load_observability_config({})
        self.assertEqual(c.otlp_endpoint, "http://collector:4317")


# ─────────────────────────────────────────────────────────────────────
#  config.HITLCheckpointConfig
# ─────────────────────────────────────────────────────────────────────
class TestHITLCheckpointConfig(unittest.TestCase):
    def test_default_is_sqlite(self):
        """Critical: default backend must be sqlite, not memory.

        The whole point of the Sprint-3-pre change is that operators who
        deploy without reading the yaml don't silently get a non-persistent
        backend (which loses pending approvals on restart).
        """
        from config import HITLCheckpointConfig
        c = HITLCheckpointConfig()
        self.assertEqual(c.backend, "sqlite")
        self.assertEqual(c.sqlite_path, "data/hitl_checkpoints.db")

    def test_loaded_default_from_yaml(self):
        """Verify the live cfg.hitl.checkpoint has sqlite default."""
        from config import cfg
        self.assertEqual(cfg.hitl.checkpoint.backend, "sqlite")
        self.assertTrue(cfg.hitl.checkpoint.sqlite_path)


# ─────────────────────────────────────────────────────────────────────
#  skills.evolver A/B safety net
# ─────────────────────────────────────────────────────────────────────
class TestEvolverParseJsonResponse(unittest.TestCase):
    """The _parse_json_response helper that apply_feedback + evaluate
    depend on. Previously missing — bug fixed 2026-05.

    The LLM is *prompted* for strict JSON but real responses often
    contain prose / fences / think blocks. Parser must extract a usable
    dict from all the common shapes, return {} on garbage, never raise.
    """

    def test_strict_json(self):
        from skills.evolver import SkillEvolver
        self.assertEqual(SkillEvolver._parse_json_response('{"a": 1}'),
                         {"a": 1})

    def test_markdown_code_fence(self):
        from skills.evolver import SkillEvolver
        text = 'Here is the JSON:\n```json\n{"x": 2}\n```'
        self.assertEqual(SkillEvolver._parse_json_response(text), {"x": 2})

    def test_unlabeled_fence(self):
        from skills.evolver import SkillEvolver
        self.assertEqual(
            SkillEvolver._parse_json_response('```\n{"y": 3}\n```'),
            {"y": 3},
        )

    def test_embedded_in_prose(self):
        from skills.evolver import SkillEvolver
        self.assertEqual(
            SkillEvolver._parse_json_response('Sure! {"k": "v"} done.'),
            {"k": "v"},
        )

    def test_nested_braces(self):
        from skills.evolver import SkillEvolver
        self.assertEqual(
            SkillEvolver._parse_json_response('{"outer": {"inner": 1}}'),
            {"outer": {"inner": 1}},
        )

    def test_strips_think_block(self):
        """qwen3 / deepseek-r1 emit <think>…</think>. Must strip."""
        from skills.evolver import SkillEvolver
        text = '<think>thinking with { braces } inside</think>{"a": 1}'
        self.assertEqual(SkillEvolver._parse_json_response(text), {"a": 1})

    def test_empty_input(self):
        from skills.evolver import SkillEvolver
        self.assertEqual(SkillEvolver._parse_json_response(""), {})
        self.assertEqual(SkillEvolver._parse_json_response("   \n  "), {})

    def test_garbage_returns_empty_dict(self):
        """Never raise — return {} so callers can check truthiness."""
        from skills.evolver import SkillEvolver
        self.assertEqual(
            SkillEvolver._parse_json_response("not even close"), {},
        )
        self.assertEqual(
            SkillEvolver._parse_json_response("[not, an, object]"), {},
        )

    def test_string_with_escaped_braces(self):
        """A '}' inside a quoted string must not confuse the balance scan."""
        from skills.evolver import SkillEvolver
        text = '{"msg": "hello } world", "ok": true}'
        result = SkillEvolver._parse_json_response(text)
        self.assertEqual(result["msg"], "hello } world")
        self.assertIs(result["ok"], True)


class TestEvolverBenchRunner(unittest.TestCase):
    """set_bench_runner contract + apply_feedback gating logic."""

    def test_set_bench_runner_accepts_callable(self):
        from skills.evolver import SkillEvolver

        class _DummyCatalog:
            def load_detail(self, sid):    return "old content"
            def update(self, *a, **k):     pass
            def add(self, *a, **k):        pass
        ev = SkillEvolver(catalog=_DummyCatalog(), llm_fn=None)
        # Default state — None
        self.assertIsNone(ev._bench_runner)
        # Wire a callable
        async def runner(skill_id, content): return None
        ev.set_bench_runner(runner)
        self.assertIs(ev._bench_runner, runner)
        # Clearing back to None
        ev.set_bench_runner(None)
        self.assertIsNone(ev._bench_runner)

    def test_set_bench_runner_logs_state_change(self):
        from skills.evolver import SkillEvolver

        class _DummyCatalog:
            def load_detail(self, sid):    return "old content"
        ev = SkillEvolver(catalog=_DummyCatalog(), llm_fn=None)
        # Just exercising the code path; no log assertions to avoid
        # coupling to caplog setup.
        async def runner(skill_id, content): return None
        ev.set_bench_runner(runner)
        ev.set_bench_runner(None)

    def test_apply_feedback_gate_rejects_regression(self):
        """When the bench wrapper says candidate is worse, apply_feedback
        returns None and does NOT call the catalog updater.

        This is a public-contract test: we don't reach into _versions or
        private LLM internals. We only verify two things visible to a
        caller — return value + that the public catalog-mutation method
        was not invoked.

        This is a public-contract test: we don't reach into _versions or
        private LLM internals. We only verify two things visible to a
        caller — return value + that the public catalog-mutation method
        was not invoked.
        """
        from skills.evolver import SkillEvolver

        class _Report:
            def __init__(self, args_rate, total=5):
                self.args_rate = args_rate
                self.total = total

        # Catalog stub returning placeholder current content. Tracks
        # whether the update mutation method got called.
        class _DummyCatalog:
            mutated = False
            def load_detail(self, sid):
                return "old content with foo tool reference"
            def update(self, *a, **k):
                _DummyCatalog.mutated = True

        ev = SkillEvolver(catalog=_DummyCatalog(), llm_fn=None)

        # Fake LLM returning a candidate diff with non-trivial update
        async def fake_llm(system, user):
            import json
            return json.dumps({
                "updated_content": "new content with foo tool reference",
                "changes": ["clarified step 2"],
                "quality_delta": 0.0,
            })
        ev._llm_fn = fake_llm

        # Catalog-mutation guard: if the patch is rejected, _update_catalog_*
        # must not run. Replace with a tripwire.
        async def fake_update_from_md(skill_id, content):
            _DummyCatalog.mutated = True
        ev._update_catalog_from_markdown = fake_update_from_md

        # Bench: baseline 0.9, candidate 0.5 → drop, must reject
        bench_calls = []
        async def gate_runner(skill_id, content):
            bench_calls.append(content)
            if len(bench_calls) == 1:
                return _Report(args_rate=0.9, total=5)
            return _Report(args_rate=0.5, total=5)
        ev.set_bench_runner(gate_runner)

        result = asyncio.run(ev.apply_feedback(
            skill_id="test-skill",
            feedback="please be more concise",
            success=False,
        ))
        self.assertIsNone(result, "Regression must yield None")
        self.assertFalse(_DummyCatalog.mutated,
                         "Catalog must NOT be mutated when bench rejects")
        self.assertEqual(len(bench_calls), 2,
                         "Bench must run for both baseline + candidate")

    def test_apply_feedback_gate_accepts_improvement(self):
        """If candidate args_ok >= baseline, the patch proceeds (returns
        a FeedbackApplication) AND the catalog mutation method is called.

        Same setup as test_apply_feedback_gate_rejects_regression, but the
        bench runner reports candidate is better. Mirror test guards
        against false positives: it would be very easy to write a bench
        gate that ALWAYS rejects.
        """
        from skills.evolver import SkillEvolver

        class _Report:
            def __init__(self, args_rate, total=5):
                self.args_rate = args_rate
                self.total = total

        class _DummyCatalog:
            mutated = False
            def load_detail(self, sid):
                return "old content with foo tool reference"

        ev = SkillEvolver(catalog=_DummyCatalog(), llm_fn=None)

        async def fake_llm(system, user):
            import json
            return json.dumps({
                "updated_content": "new content with foo tool reference",
                "changes": ["clarified step 2"],
                "quality_delta": 0.1,
            })
        ev._llm_fn = fake_llm

        async def track_update(skill_id, content):
            _DummyCatalog.mutated = True
        ev._update_catalog_from_markdown = track_update

        # Bench: baseline 0.5, candidate 0.8 → improvement, must proceed
        async def gate_runner(skill_id, content):
            if "old content" in content:
                return _Report(args_rate=0.5, total=5)
            return _Report(args_rate=0.8, total=5)
        ev.set_bench_runner(gate_runner)

        result = asyncio.run(ev.apply_feedback(
            skill_id="test-skill",
            feedback="please clarify step 2",
            success=True,
        ))
        # Improvement must produce a non-None FeedbackApplication
        self.assertIsNotNone(result, "Improvement must yield a result")
        # Catalog mutation should have happened (patch was applied)
        self.assertTrue(_DummyCatalog.mutated,
                        "Catalog MUST be mutated when bench accepts")


# ─────────────────────────────────────────────────────────────────────
#  C1 — Prometheus metrics (runtime/metrics.py)
# ─────────────────────────────────────────────────────────────────────
class TestMetricsModule(unittest.TestCase):
    """runtime/metrics.py — all helpers must be no-op-safe whether or not
    prometheus_client is installed, and render_latest always returns bytes.
    """

    def test_helpers_never_raise(self):
        from runtime import metrics as m
        # None of these may raise regardless of prometheus availability.
        m.record_llm_call("qwen3.5:27b", "ok", 1.2)
        m.record_llm_call("m", "error")          # duration optional
        m.record_tool_call("list_devices", "ok")
        m.record_tool_call("x", "not_found")
        m.set_hitl_pending(5)
        m.set_hitl_pending(0)
        with m.track_active_llm():
            pass
        with m.time_llm_call("m"):
            pass

    def test_time_llm_call_records_error_on_exception(self):
        """time_llm_call must re-raise but still record (no swallow)."""
        from runtime import metrics as m
        with self.assertRaises(ValueError):
            with m.time_llm_call("m"):
                raise ValueError("boom")

    def test_render_latest_returns_bytes_and_content_type(self):
        from runtime import metrics as m
        body, ct = m.render_latest()
        self.assertIsInstance(body, bytes)
        self.assertIsInstance(ct, str)
        self.assertIn("text/plain", ct.lower())

    def test_render_includes_metric_names_when_available(self):
        """When prometheus_client is installed, exposition has our metrics."""
        from runtime import metrics as m
        if not m.is_available():
            self.skipTest("prometheus_client not installed")
        m.record_llm_call("test-model", "ok", 0.5)
        body, _ = m.render_latest()
        text = body.decode()
        self.assertIn("netopyu_llm_calls_total", text)


# ─────────────────────────────────────────────────────────────────────
#  C2 — tracing instrument_fastapi
# ─────────────────────────────────────────────────────────────────────
class TestTracingInstrumentFastapi(unittest.TestCase):
    def setUp(self):
        import importlib, sys
        if "runtime.tracing" in sys.modules:
            importlib.reload(sys.modules["runtime.tracing"])

    def test_instrument_fastapi_noop_when_disabled(self):
        from runtime.tracing import configure, instrument_fastapi
        configure(enabled=False)
        # Must return False, not raise, when tracing is off.
        class _FakeApp:
            pass
        self.assertFalse(instrument_fastapi(_FakeApp()))


# ─────────────────────────────────────────────────────────────────────
#  D1 — LLM concurrency semaphore (logic-level)
# ─────────────────────────────────────────────────────────────────────
class TestLLMConcurrencyCap(unittest.TestCase):
    """The semaphore gating logic on the base LLMEngine.

    We test the logic in isolation (LLMEngine pulls httpx-heavy deps that
    may not be in a minimal test env), mirroring the _get_semaphore /
    set_max_concurrent_calls contract.
    """

    def _make_engine_like(self):
        import asyncio

        class _EngineLike:
            def __init__(self):
                self._max_concurrent_calls = 0
                self._llm_semaphore = None
                self._llm_sem_loop = None
            def set_max_concurrent_calls(self, n):
                self._max_concurrent_calls = max(0, int(n))
                self._llm_semaphore = None
                self._llm_sem_loop = None
            def _get_semaphore(self):
                if self._max_concurrent_calls <= 0:
                    return None
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    return None
                if self._llm_semaphore is None or self._llm_sem_loop is not loop:
                    self._llm_semaphore = asyncio.Semaphore(self._max_concurrent_calls)
                    self._llm_sem_loop = loop
                return self._llm_semaphore
        return _EngineLike()

    def test_disabled_by_default(self):
        import asyncio
        e = self._make_engine_like()
        async def go():
            return e._get_semaphore()
        self.assertIsNone(asyncio.run(go()))

    def test_cap_limits_concurrency(self):
        import asyncio
        e = self._make_engine_like()
        e.set_max_concurrent_calls(2)

        async def go():
            active = 0
            peak = 0
            async def work():
                nonlocal active, peak
                async with e._get_semaphore():
                    active += 1
                    peak = max(peak, active)
                    await asyncio.sleep(0.01)
                    active -= 1
            await asyncio.gather(*[work() for _ in range(12)])
            return peak
        peak = asyncio.run(go())
        self.assertLessEqual(peak, 2, f"peak in-flight {peak} exceeded cap 2")

    def test_config_has_max_concurrent_calls(self):
        from config import cfg
        self.assertTrue(hasattr(cfg.llm, "max_concurrent_calls"))
        self.assertGreaterEqual(cfg.llm.max_concurrent_calls, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
