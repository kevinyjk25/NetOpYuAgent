"""tests/test_delegation_outbound_state.py
===========================================

Regressions for two parent-side delegation state bugs observed v12:

1. LAN-side outbound TaskDefinition never transitioned past RUNNING.
   The dispatcher set state=RUNNING + saved at dispatch time but no
   code ever updated the state on stream end. Result: LAN's Delegations
   tab showed the row stuck in RUNNING forever, while DC's tab showed
   the same task as COMPLETED. Operator couldn't tell when a delegation
   actually finished.

2. Delegation result rendered as raw JSON ({"text": "..."}) in the
   Delegations tab detail panel. Operators saw a JSON blob instead of
   the markdown answer.

Both tests are grep-based — they assert the protective code paths
exist, without needing a live A2A round-trip.
"""

from __future__ import annotations
import os
import unittest


def _read(rel: str) -> str:
    p = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        rel,
    )
    with open(p) as f:
        return f.read()


class TestOutboundDispatcherTerminalState(unittest.TestCase):
    """Dispatcher must write a terminal state to the outbound task on
    stream end, in a try/finally so cancellation / exceptions still
    update the row."""

    def setUp(self):
        self.src = _read("task/inter/coordinator.py")

    def test_dispatcher_has_terminal_state_block(self):
        """A 'finally' must write task.completed_at + state."""
        # Look in the dispatch method
        i = self.src.find("async def dispatch")
        self.assertGreater(i, 0, "dispatch method not found")
        block = self.src[i:i+8000]
        self.assertIn(
            "finally:", block,
            "dispatch() must update terminal state in a finally block",
        )
        self.assertIn(
            "task.completed_at", block,
            "dispatch() must stamp completed_at on terminal",
        )

    def test_dispatcher_marks_completed_on_success(self):
        i = self.src.find("async def dispatch")
        block = self.src[i:i+8000]
        # The success path sets COMPLETED + writes result text
        self.assertIn(
            "TaskState.COMPLETED", block,
            "dispatch() must mark task COMPLETED on successful stream",
        )

    def test_dispatcher_marks_failed_on_error(self):
        i = self.src.find("async def dispatch")
        block = self.src[i:i+8000]
        self.assertIn(
            "TaskState.FAILED", block,
            "dispatch() must mark task FAILED when peer returns error",
        )

    def test_dispatcher_writes_audit_record_on_terminal(self):
        """Auditing requirement — terminal state must produce an
        audit record so the trail isn't broken."""
        i = self.src.find("async def dispatch")
        block = self.src[i:i+8000]
        # Existing code already had write_audit(DISPATCHED); we add COMPLETED/FAILED
        self.assertIn(
            "TaskEventKind.COMPLETED", block,
            "Terminal state needs a COMPLETED audit record",
        )

    def test_dispatcher_handles_peer_hitl(self):
        """When peer raises HITL, outbound state should stay non-terminal
        (PENDING with peer_hitl_pending flag) so the row clearly shows
        'still waiting on peer' rather than COMPLETED."""
        i = self.src.find("async def dispatch")
        block = self.src[i:i+8000]
        self.assertIn(
            "peer_hitl_pending", block,
            "Peer HITL must keep outbound row in non-terminal state",
        )


class TestDelegationResultRendering(unittest.TestCase):
    """Frontend must extract result.text for display rather than dumping
    the whole {text: "..."} JSON object."""

    def setUp(self):
        self.front = _read("webui/index.html")

    def test_frontend_extracts_result_text_field(self):
        """refreshDelegations detail panel must check r.result.text before
        falling back to JSON dump."""
        self.assertIn(
            "r.result.text", self.front,
            "Detail panel must extract .text from result object — "
            "otherwise operator sees raw JSON blob",
        )

    def test_frontend_renders_result_as_markdown(self):
        """Result text typically contains markdown (bullets, bold, code
        fences). Must use _renderMarkdown rather than escape-and-pre
        — pre with escaped text strips formatting."""
        self.assertIn(
            "_renderMarkdown(resultText)", self.front,
            "Result section must use _renderMarkdown — pre+escape "
            "strips markdown formatting (bold / lists / code)",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
