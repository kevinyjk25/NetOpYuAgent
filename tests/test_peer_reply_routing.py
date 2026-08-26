"""Lock: peer-HITL post-approval messages route to the delegation window.

When a delegated task triggers HITL and the operator approves, the peer's
subsequent reply belongs in the per-peer delegation thread — NOT the user's
main conversation. The origin's final synthesis stays in the main chat.

Source assertions (no JS runner); the behavioral proof is the node
logic-test run during development.
"""
import re
import unittest
from pathlib import Path


def _html() -> str:
    return Path("webui/index.html").read_text(encoding="utf-8")


class TestPeerReplyRouting(unittest.TestCase):
    def setUp(self):
        self.html = _html()

    def test_router_fn_exists(self):
        self.assertIn("function routePeerMessageToDelegation", self.html)
        self.assertIn("_dlgExtraMsgs", self.html)

    def test_resume_poll_routes_peer_interim_to_delegation(self):
        # the resume poll must send peer approval messages to the delegation
        # window, not addMessage('agent', ...) in the main chat
        self.assertRegex(
            self.html,
            r"isPeerInterim[\s\S]{0,200}routePeerMessageToDelegation",
            "peer interim messages must route to the delegation window")

    def test_final_synthesis_still_goes_to_main_chat(self):
        # the non-peer terminal result still uses addMessage for the main chat
        self.assertRegex(
            self.html,
            r"else if \(txt\)[\s\S]{0,120}addMessage\('agent'",
            "origin's final synthesis stays in the main chat")

    def test_delegate_tab_badge_on_peer_reply(self):
        self.assertIn("_updateDelegateTabBadge", self.html)
        self.assertIn("_dlgTabUnread", self.html)

    def test_thread_renders_extra_peer_messages(self):
        # _renderDlgThread must append the buffered post-approval messages
        self.assertRegex(
            self.html,
            r"_dlgExtraMsgs\[peer\][\s\S]{0,800}审批后回复",
            "delegation thread must render post-approval peer messages")


if __name__ == "__main__":
    unittest.main(verbosity=2)
