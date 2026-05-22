"""tests/test_delegation_no_double_count.py
============================================

Regression: when DC handles a [DELEGATE:] from LAN, it must NOT emit
the runtime loop's final_text TWICE — once as streamed tokens via
TaskArtifactUpdateEvent, then again inside the sealing MessageEvent.
The earlier bug (observed 2026-05) put final_text in BOTH paths, and
LAN's dispatcher _unwrap_a2a_event mapped MessageEvent body to a
{token: ...} chunk, so the parent LLM saw the same analysis 2-3x and
composed a final answer repeating itself.

Fix: in integrations/adapters/hitl_executor.py.HitlExecutor.execute,
when tokens were streamed via _emit_token, _finalize MUST seal the
MessageEvent with a generic marker ("Task completed.") that LAN's
_unwrap_a2a_event already filters out (see task/inter/coordinator.py
line ~312: txt not in ("Task completed.",)). Only when no tokens were
streamed should the MessageEvent carry final_text.

This test is grep-based — it asserts the protective code paths exist,
without needing a live LLM or A2A round-trip.
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


class TestNoDoubleCount(unittest.TestCase):
    """Verify the protective code paths in hitl_executor.execute."""

    def setUp(self):
        self.src = _read("integrations/adapters/hitl_executor.py")

    def test_streamed_token_flag_declared(self):
        """_streamed_any_token must be initialised once before _emit_token."""
        self.assertIn(
            "_streamed_any_token = False",
            self.src,
            "_streamed_any_token guard flag missing — _finalize cannot "
            "tell whether tokens streamed and will always include "
            "final_text in MessageEvent, causing LAN double-count",
        )

    def test_emit_token_sets_streamed_flag(self):
        """_emit_token must flip _streamed_any_token so _finalize sees it."""
        # Find the _emit_token block and check it sets the flag
        i = self.src.find("async def _emit_token(tok: str)")
        self.assertGreater(i, 0, "_emit_token not found")
        block = self.src[i:i+800]
        self.assertIn(
            "_streamed_any_token = True",
            block,
            "_emit_token must set _streamed_any_token=True",
        )

    def test_finalize_skips_final_text_when_streamed(self):
        """_finalize must conditionally use final_text only when no
        tokens were streamed."""
        # Look for the seal_text gate
        self.assertIn(
            'if _streamed_any_token else',
            self.src,
            "_finalize lacks streaming guard — MessageEvent will "
            "always include final_text",
        )

    def test_lan_unwrap_filters_task_completed(self):
        """LAN-side: _unwrap_a2a_event must filter out 'Task completed.'
        sentinel so the seal MessageEvent (when tokens were already
        streamed) does NOT inject empty/duplicate content into the
        parent's _result_parts.

        This is the contract the DC-side fix relies on. If anyone
        removes the filter, the fix breaks silently.
        """
        coord_src = _read("task/inter/coordinator.py")
        self.assertIn(
            '"Task completed."',
            coord_src,
            "LAN _unwrap_a2a_event must filter 'Task completed.' "
            "sentinel — DC-side seal relies on this",
        )


class TestInboundDelegationRecording(unittest.TestCase):
    """Phase 2B+: when DC handles an inbound A2A request with
    metadata.source_agent set, HitlExecutor must record a TaskDefinition
    with metadata.direction='inbound' to the local task_store so the
    Delegations tab can show 'received from <agent>'."""

    def setUp(self):
        self.src  = _read("integrations/adapters/hitl_executor.py")
        self.main = _read("main.py")

    def test_set_task_store_method_exists(self):
        self.assertIn(
            "def set_task_store(self, task_store)",
            self.src,
            "HitlExecutor.set_task_store deferred-wiring setter missing",
        )

    def test_record_inbound_helper_exists(self):
        self.assertIn(
            "async def _record_inbound_delegation",
            self.src,
            "Inbound recording helper missing",
        )

    def test_record_inbound_writes_direction_inbound(self):
        """The inbound recorder must tag direction='inbound' so the
        /delegations endpoint can distinguish from outbound rows."""
        self.assertIn(
            '"direction":         "inbound"',
            self.src,
            "Inbound record must carry direction='inbound' metadata",
        )

    def test_main_wires_set_task_store(self):
        """main.py must call executor.set_task_store(task_system.store)
        after task_system is built. Without this, DC never sees inbound
        delegations in its Delegations tab."""
        self.assertIn(
            "executor.set_task_store",
            self.main,
            "main.py must wire executor.set_task_store(task_system.store)",
        )

    def test_fail_inbound_helper_exists(self):
        """When the runtime loop raises (e.g. Ollama 300s timeout), the
        inbound TaskDefinition must be marked FAILED — not left in RUNNING
        forever. Without this, dc-agent's Delegations tab shows '← lan-agent
        ... RUNNING' indefinitely even after the agent has crashed.
        """
        self.assertIn(
            "async def _fail_inbound_delegation",
            self.src,
            "_fail_inbound_delegation helper missing — failure path will "
            "leave inbound row stuck in RUNNING",
        )

    def test_execute_calls_fail_on_exception(self):
        """The except clause in HitlExecutor.execute (around the
        execute_query call) must invoke _fail_inbound_delegation."""
        # Find the except block following execute_query
        i = self.src.find("HitlExecutor.execute_a2a failed")
        self.assertGreater(i, 0, "execute_query except block not found")
        block = self.src[i:i+1500]
        self.assertIn(
            "_fail_inbound_delegation",
            block,
            "execute()'s execute_query except clause must call "
            "_fail_inbound_delegation — otherwise inbound row stays "
            "in RUNNING after the agent crashes",
        )


class TestInboundDelegationsEndpoint(unittest.TestCase):
    """The /delegations/inbound endpoint must list inbound rows across
    ALL sessions. Required because DC's local operator doesn't know
    LAN's session_id (which is what inbound TaskDefinitions are keyed by).
    """

    def setUp(self):
        self.backend = _read("webui/backend.py")
        self.front   = _read("webui/index.html")

    def test_inbound_endpoint_registered_before_session(self):
        """/delegations/inbound must come BEFORE /delegations/{session_id}
        in the file — FastAPI matches paths in registration order, and
        a {session_id} placeholder would otherwise capture 'inbound' as
        a literal session id."""
        i_inbound = self.backend.find('"/delegations/inbound"')
        i_session = self.backend.find('"/delegations/{session_id}"')
        self.assertGreater(i_inbound, 0, "inbound endpoint missing")
        self.assertGreater(i_session, 0, "by-session endpoint missing")
        self.assertLess(
            i_inbound, i_session,
            "FastAPI route ordering: /delegations/inbound must be "
            "registered BEFORE /delegations/{session_id} or 'inbound' "
            "will be parsed as a literal session_id parameter",
        )

    def test_inbound_endpoint_filters_direction(self):
        """The endpoint must filter on metadata.direction == 'inbound'
        so outbound rows from other sessions don't leak in."""
        # Look for the direction filter in the endpoint body
        i = self.backend.find('"/delegations/inbound"')
        body = self.backend[i:i+3000]
        self.assertIn(
            'md.get("direction") != "inbound"',
            body,
            "Inbound endpoint must filter direction='inbound' explicitly",
        )

    def test_frontend_fetches_both_inbound_and_session(self):
        """refreshDelegations must fetch /delegations/inbound — without
        this, DC's UI can't show '← lan-agent' rows because LAN-side
        session_id isn't known locally."""
        self.assertIn(
            "/delegations/inbound",
            self.front,
            "Frontend must call /delegations/inbound for DC-side view",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
