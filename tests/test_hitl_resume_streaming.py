"""Lock the HITL resume-streaming fix (2026-06).

Bug: after a HITL pause, the resumer streamed continuation tokens but the
frontend dropped them — appendToken() early-returned because streamState was
nulled by finalizeStream(), so the operator saw no streaming, only the final
result appearing at once.

Fix: park the original bubble at stop_hitl and reattach on the resume
stream's first token so continuation renders IN THE SAME bubble.

These are source assertions (the frontend has no JS test runner); they guard
the wiring from silent removal. The behavioral proof is the node logic-test
run during development.
"""
import re
import unittest
from pathlib import Path


def _html() -> str:
    return Path("webui/index.html").read_text(encoding="utf-8")


class TestHitlResumeStreaming(unittest.TestCase):
    def setUp(self):
        self.html = _html()

    def test_parked_stream_holder_declared(self):
        self.assertIn("_parkedStream", self.html,
                      "need a parked-stream holder to resume into the same bubble")

    def test_stop_hitl_parks_the_bubble(self):
        # parking must happen via a shared helper called from every pause path
        self.assertIn("function parkCurrentStream", self.html,
                      "need a shared parkCurrentStream helper")
        # awaiting_hitl (skill-selection / clarification) must park too — this
        # is the path the screenshot bug hit (skill selection → no streaming)
        self.assertRegex(
            self.html,
            r"awaiting_hitl[\s\S]{0,1200}parkCurrentStream\(\)",
            "awaiting_hitl (skill-selection) must park the bubble for resume")
        # stop_hitl (tool approval) parks too
        self.assertRegex(
            self.html,
            r"stop_hitl[\s\S]{0,1200}parkCurrentStream\(\)",
            "stop_hitl (approval) must park the bubble for resume")

    def test_append_token_reattaches_on_resume(self):
        # appendToken must handle the "no streamState but parked" case by
        # rebuilding streamState pointing at the parked bubble
        self.assertRegex(
            self.html,
            r"if\s*\(!streamState\s*&&\s*_parkedStream\)",
            "appendToken must reattach to the parked bubble on resume")
        self.assertIn("resumed: true", self.html,
                      "reattached streamState should be flagged resumed")

    def test_resume_done_finalizes_reattached_bubble(self):
        # on the resume stream's done, if we resumed we finalize the bubble
        self.assertRegex(
            self.html,
            r"streamState\.resumed[\s\S]{0,80}finalizeStream",
            "resume done must finalize the reattached bubble")

    def test_fresh_stream_clears_stale_park(self):
        # a new main stream must clear any un-resumed park so it doesn't bleed
        self.assertRegex(
            self.html,
            r"_parkedStream\s*=\s*null;\s*//[^\n]*new query",
            "startStream must clear a stale parked stream")


if __name__ == "__main__":
    unittest.main(verbosity=2)
