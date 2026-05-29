"""Regression: LAN must suppress repeat DELEGATE to the same target.

Without this guard, when a peer (e.g. dc-agent) returns an inconclusive
intermediate result (e.g. its own LLM is in a loop and never calls any
tool/HITL), the LAN LLM mis-reads "no conclusion" as "task not done" and
re-emits [DELEGATE:dc-agent], spawning a new inbound task on the peer every
turn — the storm we observed (lan/dc task state diverges, 5+ inbound RUNNING
tasks accumulate on the peer).

The guard tracks per-target delegation count on env_ctx (persists across
turns within one stream). At >=2 we suppress and tell the LLM to stop.
"""
import unittest
from pathlib import Path


class TestRepeatDelegationGuard(unittest.TestCase):
    def test_per_target_count_tracked(self):
        src = Path("runtime/loop.py").read_text(encoding="utf-8")
        self.assertIn('"_delegate_target_counts"', src,
                      "expected per-target delegation count on env_ctx")

    def test_repeat_suppress_threshold(self):
        src = Path("runtime/loop.py").read_text(encoding="utf-8")
        # The threshold check (≥2) must be present.
        self.assertIn("_prev_n >= 2", src,
                      "expected repeat-suppress threshold at 2 prior delegations")

    def test_two_trip_wires_independent(self):
        """The HITL-pending guard and the repeat-count guard must be separate
        conditions, so non-HITL inconclusive peers still get suppressed."""
        src = Path("runtime/loop.py").read_text(encoding="utf-8")
        # Pending HITL path
        self.assertIn("_peer_hitl_pending_targets", src)
        # Repeat-count path
        self.assertIn("_delegate_target_counts", src)
        # Both lead to a single _suppress_reason switch
        self.assertIn("_suppress_reason", src,
                      "expected a unified suppression switch for both paths")


if __name__ == "__main__":
    unittest.main(verbosity=2)
