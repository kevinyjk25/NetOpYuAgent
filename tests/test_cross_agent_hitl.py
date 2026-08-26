"""
tests/test_cross_agent_hitl.py
==============================

A2A Phase 3 (P3-b) — cross-agent HITL passthrough, mode B.

Exercises the correlation chain end-to-end WITHOUT two live agents:
  1. lan unwraps dc's INPUT_REQUIRED A2A status into a hitl_interrupt chunk
     that carries dc's interrupt_id (the fix that closed the correlation gap).
  2. lan registers an awaiting-peer record keyed by (dc, interrupt_id).
  3. dc records an inbound-HITL record under the SAME interrupt_id.
  4. The /hitl_resolved resolve path finds lan's awaiting record (and guards
     double-resume).

The key invariant: the interrupt_id dc raises locally == the id lan keys its
awaiting record by == the id dc calls back with. If any layer rewrote it the
chain would silently break.
"""
import unittest

import pytest
# task.* and hitl_core.* pull in pydantic; skip the whole module if it's
# absent (minimal CI runner) instead of failing collection with ImportError.
pytest.importorskip("pydantic")

from task.inter.coordinator import A2ATaskDispatcher
from task.inter.cross_agent_hitl import (
    CrossAgentHitlBridge, get_cross_agent_hitl_bridge,
)


class TestUnwrapInputRequired(unittest.TestCase):
    def test_input_required_status_becomes_hitl_interrupt_chunk(self):
        # dc emits TaskStatusUpdateEvent(state=input-required, message=<iid>)
        raw = {"status": {"state": "input-required", "message": "iid-abc123"}}
        chunks = A2ATaskDispatcher._unwrap_a2a_event(raw)
        hit = [c for c in chunks if c.get("hitl_interrupt")]
        self.assertEqual(len(hit), 1, f"expected one hitl_interrupt chunk, got {chunks}")
        self.assertEqual(hit[0]["interrupt_id"], "iid-abc123")

    def test_underscore_spelling_also_handled(self):
        raw = {"status": {"state": "input_required", "message": "iid-x"}}
        chunks = A2ATaskDispatcher._unwrap_a2a_event(raw)
        self.assertTrue(any(c.get("interrupt_id") == "iid-x" for c in chunks))

    def test_working_status_is_not_a_hitl_interrupt(self):
        raw = {"status": {"state": "working", "message": ""}}
        chunks = A2ATaskDispatcher._unwrap_a2a_event(raw)
        self.assertFalse(any(c.get("hitl_interrupt") for c in chunks))


class TestCrossAgentCorrelation(unittest.TestCase):
    def test_full_correlation_chain(self):
        bridge = CrossAgentHitlBridge()
        iid = "iid-shared-777"

        # dc side: raised a HITL while serving lan's delegation.
        cid = bridge.record_inbound_hitl(
            interrupt_id=iid, source_agent="lan-agent",
            source_session_id="lan-sess-1", source_query="check Alice app access",
        )

        # lan side: saw the hitl_interrupt chunk, registered awaiting record
        # keyed by (peer_agent, interrupt_id) — SAME iid.
        bridge.record_awaiting_peer(
            local_session_id="lan-sess-1", peer_agent="dc-agent",
            peer_interrupt_id=iid, correlation_id=cid,
        )

        # dc operator approves → dc looks up its inbound record to know who to
        # call back.
        inbound = bridge.pop_inbound_hitl(iid)
        self.assertIsNotNone(inbound)
        self.assertEqual(inbound.source_agent, "lan-agent")
        self.assertEqual(inbound.source_session_id, "lan-sess-1")

        # lan receives the /hitl_resolved callback → resolves awaiting record.
        awaiting = bridge.resolve_awaiting_peer(
            peer_agent="dc-agent", peer_interrupt_id=iid,
        )
        self.assertIsNotNone(awaiting)
        self.assertEqual(awaiting.local_session_id, "lan-sess-1")
        self.assertEqual(awaiting.correlation_id, cid)

        # double callback is guarded
        again = bridge.resolve_awaiting_peer(peer_agent="dc-agent", peer_interrupt_id=iid)
        self.assertIsNone(again)

    def test_unknown_interrupt_returns_none(self):
        bridge = CrossAgentHitlBridge()
        self.assertIsNone(
            bridge.resolve_awaiting_peer(peer_agent="dc-agent", peer_interrupt_id="nope")
        )


class TestResumeDriverContract(unittest.TestCase):
    def test_handle_cross_agent_resume_buffers_when_no_driver(self):
        # When no resume driver is registered, the result is buffered (not lost).
        import asyncio
        try:
            import webui.backend as be   # needs fastapi
        except ImportError as e:
            raise unittest.SkipTest(f"webui.backend import needs fastapi: {e}")
        # Ensure no driver (save/restore to not disturb other tests).
        saved = be._resume_driver
        be._resume_driver = None
        try:
            ok = asyncio.run(be.handle_cross_agent_resume(
                local_session_id="s-buf", peer_agent="dc-agent",
                result_text="app perm denied", decision="approve",
            ))
            self.assertFalse(ok)
            items = be._pending_resumptions.get("s-buf", [])
            self.assertTrue(any("app perm denied" in i["text"] for i in items))
        finally:
            be._resume_driver = saved
            be._pending_resumptions.pop("s-buf", None)

    def test_handle_cross_agent_resume_does_not_block_on_slow_driver(self):
        # The callback receiver must SCHEDULE the resume turn and return at
        # once — never block on the (30-60s) LLM synthesis turn, or the peer's
        # POST /hitl_resolved times out mid-approval (the dc-stuck/lan-orphan
        # bug). Returns True (scheduled); the answer arrives via the buffer
        # once the background turn finishes.
        import asyncio
        import time
        try:
            import webui.backend as be   # needs fastapi
        except ImportError as e:
            raise unittest.SkipTest(f"webui.backend import needs fastapi: {e}")

        async def _slow_driver(*, local_session_id, peer_agent, result_text,
                               decision, correlation_id="", phase="result",
                               outbound_task_id=""):
            await asyncio.sleep(1.5)   # simulate a long synthesis turn
            be._pending_resumptions.setdefault(local_session_id, []).append(
                {"text": "final synthesized answer", "driven": True,
                 "peer_agent": peer_agent}
            )
            return True

        async def _go():
            t0 = time.monotonic()
            ok = await be.handle_cross_agent_resume(
                local_session_id="s-fast", peer_agent="dc-agent",
                result_text="granted", decision="approve",
            )
            dt = time.monotonic() - t0
            # Returned fast (scheduled), not after the 1.5s driver.
            self.assertTrue(ok)
            self.assertLess(dt, 0.5, f"blocked {dt:.2f}s on slow driver")
            # Buffer not populated yet (turn still running in background).
            self.assertFalse(be._pending_resumptions.get("s-fast"))
            # After the background turn completes, the answer is buffered.
            await asyncio.sleep(1.8)
            items = be._pending_resumptions.get("s-fast", [])
            self.assertTrue(
                any("final synthesized answer" in i["text"] for i in items),
                "background resume turn must buffer its answer",
            )

        saved = be._resume_driver
        be._resume_driver = _slow_driver
        try:
            asyncio.run(_go())
        finally:
            be._resume_driver = saved
            be._pending_resumptions.pop("s-fast", None)


class TestTwoPhasePassback(unittest.TestCase):
    """A2A Phase 3 case-3: HITL with async follow-up sends TWO callbacks
    (approval then result). The bridge must keep the record alive across the
    intermediate approval push and only consume it on the terminal result."""

    def _fresh_bridge(self):
        from task.inter import cross_agent_hitl as m
        m._BRIDGE = None
        return m.get_cross_agent_hitl_bridge()

    def test_inbound_peek_keeps_record_pop_consumes(self):
        b = self._fresh_bridge()
        b.record_inbound_hitl(
            interrupt_id="int-1", source_agent="lan-agent",
            source_session_id="s-lan", correlation_id="cid-1",
        )
        self.assertIsNotNone(b.peek_inbound_hitl("int-1"))
        self.assertIsNotNone(b.peek_inbound_hitl("int-1"))   # still there
        self.assertIsNotNone(b.pop_inbound_hitl("int-1"))
        self.assertIsNone(b.peek_inbound_hitl("int-1"))      # consumed

    def test_awaiting_resolve_nonterminal_then_terminal(self):
        b = self._fresh_bridge()
        b.record_awaiting_peer(
            local_session_id="s-lan", peer_agent="dc-agent",
            peer_interrupt_id="int-1", correlation_id="cid-1",
        )
        rec1 = b.resolve_awaiting_peer(
            peer_agent="dc-agent", peer_interrupt_id="int-1", terminal=False,
        )
        self.assertIsNotNone(rec1)
        self.assertEqual(rec1.local_session_id, "s-lan")
        rec2 = b.resolve_awaiting_peer(
            peer_agent="dc-agent", peer_interrupt_id="int-1", terminal=True,
        )
        self.assertIsNotNone(rec2)
        rec3 = b.resolve_awaiting_peer(
            peer_agent="dc-agent", peer_interrupt_id="int-1", terminal=True,
        )
        self.assertIsNone(rec3)   # duplicate terminal rejected

    def test_case2_single_terminal_push_still_works(self):
        b = self._fresh_bridge()
        b.record_awaiting_peer(
            local_session_id="s-lan", peer_agent="dc-agent",
            peer_interrupt_id="int-2", correlation_id="cid-2",
        )
        self.assertIsNotNone(b.resolve_awaiting_peer(
            peer_agent="dc-agent", peer_interrupt_id="int-2", terminal=True,
        ))
        self.assertIsNone(b.resolve_awaiting_peer(
            peer_agent="dc-agent", peer_interrupt_id="int-2", terminal=True,
        ))


if __name__ == "__main__":
    unittest.main(verbosity=2)
