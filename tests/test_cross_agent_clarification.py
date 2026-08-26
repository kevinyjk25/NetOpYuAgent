"""Cross-agent clarification loop (Option A).

When a delegated peer (dc) needs a clarification, it bubbles the question up
to the ORIGIN user's conversation (lan). The user answers in their session;
the answer routes back to dc so its parked interrupt resumes.

    lan user ── delegates ──▶ dc
                              dc needs info → POST /peer_clarification → lan
    lan surfaces card ◀───────────────────────────────────────────────┘
    user answers → POST /chat/peer_clarification/answer (lan)
                 → POST /peer_clarification_answer (dc) → dc resumes
"""
import unittest

from task.inter.cross_agent_hitl import get_cross_agent_hitl_bridge


class TestBridgeClarificationRoundTrip(unittest.TestCase):
    def setUp(self):
        # fresh bridge state
        b = get_cross_agent_hitl_bridge()
        b._peer_clarifications.clear()

    def test_record_and_lookup_by_correlation(self):
        b = get_cross_agent_hitl_bridge()
        b.record_peer_clarification(
            correlation_id="cid-1", peer_agent="dc-agent",
            peer_interrupt_id="int-9", local_session_id="lan-sess",
            question="which CRM instance?")
        rec = b.get_peer_clarification("cid-1")
        self.assertIsNotNone(rec)
        self.assertEqual(rec.peer_agent, "dc-agent")
        self.assertEqual(rec.peer_interrupt_id, "int-9")
        self.assertFalse(rec.answered)

    def test_lookup_by_session_finds_unanswered(self):
        b = get_cross_agent_hitl_bridge()
        b.record_peer_clarification(
            correlation_id="cid-2", peer_agent="dc-agent",
            peer_interrupt_id="int-2", local_session_id="s2", question="q")
        rec = b.peer_clarification_for_session("s2")
        self.assertIsNotNone(rec)
        self.assertEqual(rec.correlation_id, "cid-2")

    def test_resolve_marks_answered_and_is_idempotent(self):
        b = get_cross_agent_hitl_bridge()
        b.record_peer_clarification(
            correlation_id="cid-3", peer_agent="dc", peer_interrupt_id="i3",
            local_session_id="s3", question="q")
        first = b.resolve_peer_clarification("cid-3")
        self.assertIsNotNone(first)
        self.assertTrue(first.answered)
        # second resolve is a no-op (already answered)
        self.assertIsNone(b.resolve_peer_clarification("cid-3"))
        # session lookup no longer returns it (answered)
        self.assertIsNone(b.peer_clarification_for_session("s3"))


class TestOriginBufferAndDrain(unittest.TestCase):
    """The origin buffers a peer clarification and drains it for the poll."""
    def test_push_then_drain(self):
        import asyncio
        from webui.backend import (push_peer_clarification_to_session,
                                   drain_peer_clarifications)

        async def go():
            ok = await push_peer_clarification_to_session(
                session_id="sx", peer_agent="dc-agent", correlation_id="cid-x",
                question="which app?",
                clarification_fields=[{"key": "app", "prompt": "which app?"}])
            self.assertTrue(ok)
            items = drain_peer_clarifications("sx")
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["type"], "peer_clarification")
            self.assertEqual(items[0]["peer_agent"], "dc-agent")
            self.assertEqual(items[0]["correlation_id"], "cid-x")
            # drained — second drain is empty
            self.assertEqual(drain_peer_clarifications("sx"), [])
        asyncio.run(go())


class TestEndpointsWireUp(unittest.TestCase):
    """The a2a endpoints exist and validate input."""
    def test_a2a_has_clarification_endpoints(self):
        import inspect
        import a2a.server as srv
        src = inspect.getsource(srv)
        self.assertIn("/peer_clarification", src)
        self.assertIn("/peer_clarification_answer", src)
        # origin surfaces via backend push
        self.assertIn("push_peer_clarification_to_session", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
