"""Regression: LAN must suppress repeat DELEGATE to the same target.

Without this guard, when a peer (e.g. dc-agent) returns an inconclusive
intermediate result, the LAN LLM mis-reads "no conclusion" as "task not
done" and re-emits [DELEGATE:dc-agent], spawning a new inbound task on the
peer every turn — the storm we observed (lan/dc task state diverges, 5+
inbound RUNNING tasks accumulate on the peer).

DESIGN EVOLUTION (2026-06): the original guard tracked a per-target count
on env_ctx and suppressed at >=2 prior delegations, with a separate
_peer_hitl_pending_targets trip wire. It was REPLACED by a stricter
mechanism, live-verified on the dual-agent setup:

  1. `_delegated_targets_this_request: set` — per-stream; the FIRST repeat
     to the same target is suppressed (>=1, stricter than the old >=2).
  2. On suppression, a durable synthesis instruction is recorded via
     state.record_new_fact (survives the per-turn context rebuild) and
     `_needs_synthesis_turn` forces one more LLM turn so the user gets a
     final synthesized answer instead of a dangling suppression.
  3. Peer-HITL (case2) no longer uses a pending-targets set: the stream is
     PARKED entirely (`_parked_peer_hitl` → return) and the stage-2 result
     callback drives a dedicated synthesis turn, where `_cross_agent_resume`
     hard-blocks any DELEGATE. No further turns exist to re-delegate from —
     strictly stronger than marking targets pending.

These tests assert the CURRENT invariants (still via source inspection, as
the guard logic is deep inside _stream_impl's turn loop and a full
behavioural harness needs a live LLM + peer).
"""
import unittest
from pathlib import Path


def _src() -> str:
    return Path("runtime/loop.py").read_text(encoding="utf-8")


class TestRepeatDelegationGuard(unittest.TestCase):
    def test_per_request_target_set_tracked(self):
        """Per-stream set of already-delegated targets must exist and be
        populated when a delegation is honored."""
        src = _src()
        self.assertIn("_delegated_targets_this_request", src,
                      "expected per-request delegated-target set")
        self.assertIn("_delegated_targets_this_request.add(", src,
                      "expected the set to be populated on a honored delegate")

    def test_first_repeat_suppressed_with_synthesis(self):
        """The membership check must suppress the FIRST repeat (stricter than
        the old >=2 count) and force a synthesis turn with a DURABLE note
        (record_new_fact, not context_str += which is rebuilt per turn)."""
        src = _src()
        self.assertIn("in _delegated_targets_this_request", src,
                      "expected membership check suppressing the first repeat")
        self.assertIn("_needs_synthesis_turn = True", src,
                      "expected suppression to force a synthesis turn")
        self.assertIn("forcing synthesis turn after suppressed re-delegate", src,
                      "expected the synthesis-turn gate to be wired")
        # The durable-injection fix: the synthesis note must be recorded as a
        # fact (survives context rebuild), in the suppression block.
        _i = src.find("in _delegated_targets_this_request")
        _block = src[_i:_i + 2500]
        self.assertIn("record_new_fact", _block,
                      "synthesis note must be durable (record_new_fact), not context_str +=")

    def test_two_trip_wires_independent(self):
        """Two independent guards must coexist:
        (a) per-request repeat suppress (any peer, incl. non-HITL inconclusive)
        (b) cross-agent-resume hard block — the synthesis turn driven by the
            stage-2 result callback must never delegate.
        Plus the park mechanism that removes the re-delegate window entirely
        while a peer HITL is pending."""
        src = _src()
        # (a) repeat path
        self.assertIn("_delegated_targets_this_request", src)
        # (b) synthesis-turn hard block
        self.assertIn("_cross_agent_resume", src,
                      "expected cross-agent resume synthesis turns to hard-block DELEGATE")
        # park-on-peer-HITL replaces the old _peer_hitl_pending_targets set
        self.assertIn("_parked_peer_hitl", src,
                      "expected stream parking while peer HITL is pending")
        self.assertIn("cross_agent_parked", src,
                      "expected the parked event to be surfaced to the UI")


if __name__ == "__main__":
    unittest.main(verbosity=2)
