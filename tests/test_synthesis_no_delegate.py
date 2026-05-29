"""Regression: the cross-agent resume synthesis turn must NOT issue a fresh
DELEGATE. Without this guard, the loop is:

    user query → LAN delegates to DC → DC raises HITL → operator approves
    → DC callback → LAN spawns synthesis execute_query (fresh env_ctx, count=0)
    → LLM emits [DELEGATE:dc-agent] AGAIN → DC raises another HITL → ...

This is the 5-delegation storm the user observed despite the count guard,
because the count was scoped to env_ctx (fresh per execute_query) while the
synthesis turn starts a NEW execute_query.

The fix: when env_ctx['_cross_agent_resume'] is set, ALL DELEGATE directives
are suppressed regardless of count, with a clear log anchor.
"""
import unittest
from pathlib import Path


class TestSynthesisNoDelegate(unittest.TestCase):
    def test_cross_agent_resume_blocks_delegate(self):
        src = Path("runtime/loop.py").read_text(encoding="utf-8")
        # The guard reads the env_ctx flag set by the resume driver.
        self.assertIn(
            '(env_ctx or {}).get("_cross_agent_resume")', src,
            "expected synthesis-turn DELEGATE block via _cross_agent_resume "
            "env_ctx flag",
        )
        # The guard sets _suppress_reason (unified switch with the other two).
        self.assertIn("cross-agent resume synthesis turn", src,
                      "expected diagnostic anchor in suppression reason")

    def test_resume_driver_sets_the_flag(self):
        """The driver MUST pass env_context={'_cross_agent_resume': True} so
        the loop's guard activates."""
        src = Path("webui/backend.py").read_text(encoding="utf-8")
        self.assertIn('"_cross_agent_resume": True', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
