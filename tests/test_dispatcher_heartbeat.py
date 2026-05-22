"""tests/test_dispatcher_heartbeat.py
=====================================

Regression test for _with_heartbeat in task/inter/coordinator.py.

Background (2026-05): LAN agent delegates to dc-agent. The dc-agent peer
runs its own slow LLM call (query classification + Turn 1, ~3-5 minutes
on qwen3.5:27b consumer hardware) before streaming the first token back.
During that gap the LAN-side SSE chunk_queue saw nothing — and
sse_stall_timeout_seconds (default 300s) cancelled the request before
the peer's first token arrived. Symptom on the operator UI:
"LLM backend did not respond within 300s — the request was cancelled."
even though dc-agent was healthy and busy.

Fix: wrap dispatcher's stream with a heartbeat layer that injects a
no-op node_step chunk every `heartbeat_s` seconds of silence. The
chunk carries no `token`/`message` content (so it doesn't pollute the
delegating side's synthesis context), only a `node_step` string and
`heartbeat=True` flag — but the parent's SSE chunk_queue counts it as
activity and stays alive.

These tests verify:
1. heartbeats appear when upstream is slow
2. heartbeats DO NOT appear when upstream is fast (no flooding)
3. real chunks pass through unchanged
4. end-of-stream propagates cleanly
5. exceptions from upstream propagate (don't get swallowed)
"""

from __future__ import annotations

import asyncio
import unittest


def _load_with_heartbeat():
    """Load _with_heartbeat without triggering task.__init__ → task.schemas →
    pydantic (which isn't installed in the sandbox). We exec-load the function
    out of the source file with the right globals."""
    import os, importlib.util, sys, types

    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "task", "inter", "coordinator.py",
    )
    # Read just the function we need rather than executing the whole module
    # (which has top-level imports that pull in pydantic).
    with open(path, "r") as f:
        src = f.read()

    # Slice from "async def _with_heartbeat" through (and including) the
    # blank line + comment that follows the function. We anchor by the next
    # top-level marker comment after the function.
    start = src.find("async def _with_heartbeat(")
    if start == -1:
        return None
    end_marker = "# A2A Task Dispatcher"
    end = src.find(end_marker, start)
    if end == -1:
        return None
    # Walk back to the "# ---" line preceding the marker.
    end = src.rfind("# ---", start, end)
    if end == -1:
        return None
    fn_src = src[start:end]

    # Provide the names the function references at runtime.
    ns: dict = {"AsyncIterator": __import__("typing").AsyncIterator}
    try:
        exec(fn_src, ns)
    except Exception:
        return None
    return ns.get("_with_heartbeat")


class TestHeartbeatInjected(unittest.IsolatedAsyncioTestCase):
    """When upstream is silent longer than heartbeat_s, the wrapper
    must inject a heartbeat chunk so the delegating SSE doesn't stall."""

    async def asyncSetUp(self):
        self._with_heartbeat = _load_with_heartbeat()
        if self._with_heartbeat is None:
            self.skipTest("coordinator not importable")

    async def test_heartbeat_emitted_during_silence(self):
        async def _slow():
            # 0.5s wait, then a single chunk, then end
            await asyncio.sleep(0.5)
            yield {"token": "hello"}

        chunks = []
        async for ch in self._with_heartbeat(_slow(),
                                              heartbeat_s=0.1,
                                              agent_id="dc-agent"):
            chunks.append(ch)

        # Expect: several heartbeats then the real {"token": "hello"}
        heartbeats = [c for c in chunks if c.get("heartbeat")]
        tokens     = [c for c in chunks if c.get("token")]
        self.assertGreaterEqual(len(heartbeats), 2,
            f"expected >=2 heartbeats during 0.5s silence at heartbeat_s=0.1, got {chunks}")
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0]["token"], "hello")

    async def test_heartbeat_node_step_does_not_carry_token(self):
        """A heartbeat must NOT carry token/message — otherwise the
        delegating loop's _handle_delegate would accumulate it into the
        synthesis context, polluting the final answer with 'peer working
        peer working peer working ...'."""
        async def _slow():
            await asyncio.sleep(0.3)
            yield {"token": "real"}

        async for ch in self._with_heartbeat(_slow(),
                                              heartbeat_s=0.1,
                                              agent_id="dc-agent"):
            if ch.get("heartbeat"):
                self.assertNotIn("token",   ch)
                self.assertNotIn("message", ch)
                self.assertIn   ("node_step", ch)


class TestHeartbeatPassthroughWhenFast(unittest.IsolatedAsyncioTestCase):
    """When upstream is fast, no heartbeats — just passthrough."""

    async def asyncSetUp(self):
        self._with_heartbeat = _load_with_heartbeat()
        if self._with_heartbeat is None:
            self.skipTest("coordinator not importable")

    async def test_fast_stream_no_heartbeat(self):
        async def _fast():
            for i in range(5):
                yield {"token": f"tok-{i}"}

        chunks = []
        async for ch in self._with_heartbeat(_fast(),
                                              heartbeat_s=1.0,
                                              agent_id="dc-agent"):
            chunks.append(ch)

        heartbeats = [c for c in chunks if c.get("heartbeat")]
        self.assertEqual(len(heartbeats), 0,
            "fast stream must not produce heartbeats")
        self.assertEqual(len(chunks), 5)


class TestEndOfStream(unittest.IsolatedAsyncioTestCase):
    """Stream end must be propagated cleanly."""

    async def asyncSetUp(self):
        self._with_heartbeat = _load_with_heartbeat()
        if self._with_heartbeat is None:
            self.skipTest("coordinator not importable")

    async def test_clean_end_of_stream(self):
        async def _short():
            yield {"token": "only"}

        chunks = []
        async for ch in self._with_heartbeat(_short(),
                                              heartbeat_s=10.0,
                                              agent_id="x"):
            chunks.append(ch)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["token"], "only")

    async def test_empty_upstream(self):
        async def _empty():
            if False:
                yield {}   # makes it an async generator
        chunks = []
        async for ch in self._with_heartbeat(_empty(),
                                              heartbeat_s=10.0,
                                              agent_id="x"):
            chunks.append(ch)
        self.assertEqual(chunks, [])


class TestExceptionPropagation(unittest.IsolatedAsyncioTestCase):
    """Exceptions from upstream must propagate, not be swallowed."""

    async def asyncSetUp(self):
        self._with_heartbeat = _load_with_heartbeat()
        if self._with_heartbeat is None:
            self.skipTest("coordinator not importable")

    async def test_upstream_raises(self):
        class _MyErr(RuntimeError): pass

        async def _bad():
            yield {"token": "one"}
            raise _MyErr("boom")

        chunks = []
        with self.assertRaises(_MyErr):
            async for ch in self._with_heartbeat(_bad(),
                                                  heartbeat_s=10.0,
                                                  agent_id="x"):
                chunks.append(ch)
        # First chunk should have made it through before the raise
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["token"], "one")


if __name__ == "__main__":
    unittest.main(verbosity=2)
