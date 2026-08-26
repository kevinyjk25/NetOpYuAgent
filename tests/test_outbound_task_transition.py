"""Regression: when a cross-agent HITL is resolved (case3 terminal callback),
the LAN-side outbound delegation task must transition from
AWAITING_PEER_HITL → COMPLETED. Without this, the UI shows "PEER HITL
awaiting" forever even though DC has already approved and pushed back the
result — the state-disagree the user observed in the DELEG panel where LAN's
outbound stayed PEER HITL while DC's inbound was DONE.

We verify two things:
  1. AwaitingPeerRecord carries the outbound_task_id (so /hitl_resolved can
     find the task to transition).
  2. record_awaiting_peer accepts outbound_task_id and stores it on the
     record; the round-trip through resolve_awaiting_peer preserves it.
"""
import unittest


class TestOutboundTaskTransitionPlumbing(unittest.TestCase):
    def _fresh_bridge(self):
        from task.inter import cross_agent_hitl as m
        m._BRIDGE = None
        return m.get_cross_agent_hitl_bridge()

    def test_awaiting_record_carries_outbound_task_id(self):
        from task.inter.cross_agent_hitl import AwaitingPeerRecord
        rec = AwaitingPeerRecord(
            local_session_id="s-lan", peer_agent="dc-agent",
            peer_interrupt_id="int-1", correlation_id="cid",
            outbound_task_id="task-abc",
        )
        self.assertEqual(rec.outbound_task_id, "task-abc")

    def test_record_awaiting_peer_stores_task_id(self):
        b = self._fresh_bridge()
        b.record_awaiting_peer(
            local_session_id="s-lan", peer_agent="dc-agent",
            peer_interrupt_id="int-1", correlation_id="cid",
            outbound_task_id="task-xyz",
        )
        rec = b.resolve_awaiting_peer(
            peer_agent="dc-agent", peer_interrupt_id="int-1", terminal=False,
        )
        self.assertIsNotNone(rec)
        self.assertEqual(rec.outbound_task_id, "task-xyz")

    def test_record_awaiting_peer_optional_task_id(self):
        """case1/2 callers (no cross-agent HITL) may omit the task_id."""
        b = self._fresh_bridge()
        b.record_awaiting_peer(
            local_session_id="s-lan", peer_agent="dc-agent",
            peer_interrupt_id="int-2", correlation_id="cid",
        )
        rec = b.resolve_awaiting_peer(
            peer_agent="dc-agent", peer_interrupt_id="int-2", terminal=True,
        )
        self.assertIsNotNone(rec)
        self.assertEqual(rec.outbound_task_id, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
