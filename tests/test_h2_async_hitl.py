"""tests/test_h2_async_hitl.py
==============================

Regression tests for the H2 async-HITL demo tool — `tools.mock_tools.query_radius_logs`.

These tests catch the bug class that production deploy hit: any pydantic
ValidationError or wiring failure in the H2 setup path used to propagate
out as a "tool error", causing the LLM to retry the same call N times.

The tool MUST always return a usable string. Setup failures are caught
and surfaced as degraded-mode strings the LLM can act on.

Tests are deliberately tolerant about whether main._services is wired —
in the test environment it isn't. That's fine; the tool's degraded paths
should kick in and we verify the degraded responses are sane.
"""

from __future__ import annotations
import asyncio
import unittest


class TestQueryRadiusLogsAlwaysReturns(unittest.TestCase):
    """The tool must NEVER raise — only return strings.

    This is the primary regression. The v6-h2 deploy had query_radius_logs
    raising pydantic ValidationError due to a ProposedAction field mismatch
    (used tool_name/tool_args; schema requires action_type/target). The
    ValidationError propagated out as a tool error and the LLM looped
    retrying the same call 3 turns before giving up.

    After the fix the tool catches its own setup errors and returns a
    "degraded mode" string the LLM can synthesise an answer from.
    """

    def _load_tool(self):
        """Import the tool, skipping cleanly when test env can't load it.

        Use profiles.load_profile() (NOT the LAN-specific import path) so
        the audit_profiles check that forbids hard-importing a specific
        business profile stays clean. Tests may know which profile they're
        testing, but they ask for it by id at runtime.
        """
        try:
            from profiles import load_profile
            p = load_profile("lan")
            tool = p.tool_callables.get("query_radius_logs")
            if tool is None:
                self.skipTest("query_radius_logs not registered in lan profile")
                return None
            return tool
        except ImportError as exc:
            self.skipTest(f"profiles package not importable: {exc}")
            return None  # unreachable, but keeps the type checker happy

    def test_basic_args_returns_string_does_not_raise(self):
        """The most basic call — must produce a string, not raise."""
        tool = self._load_tool()
        result = asyncio.run(tool({"user_id": "alice", "minutes": 60}))
        self.assertIsInstance(result, str)
        # Whatever path (full H2, degraded, fallback) was taken, the
        # response must mention the user_id the LLM asked about.
        self.assertIn("alice", result)
        # And it should NEVER contain a raw pydantic validation error
        # leaked out as the tool result.
        self.assertNotIn("validation errors for ProposedAction", result)
        self.assertNotIn("validation errors for HitlPayload", result)

    def test_unknown_user_returns_string(self):
        """Missing user_id falls back to 'unknown_user' default."""
        tool = self._load_tool()
        result = asyncio.run(tool({}))
        self.assertIsInstance(result, str)
        self.assertIn("unknown_user", result)

    def test_with_session_id_returns_string(self):
        """The runtime injects _session_id; tool must accept it cleanly."""
        tool = self._load_tool()
        result = asyncio.run(tool({
            "user_id":     "bob",
            "minutes":     30,
            "_session_id": "sess-test-xyz",
        }))
        self.assertIsInstance(result, str)
        self.assertIn("bob", result)

    def test_demo_autoreply_disabled_still_returns(self):
        """_demo_autoreply=0 disables the autoresponder. Tool must still return."""
        tool = self._load_tool()
        result = asyncio.run(tool({
            "user_id":          "carol",
            "_demo_autoreply":  "0",
        }))
        self.assertIsInstance(result, str)
        self.assertIn("carol", result)


class TestProposedActionFieldContract(unittest.TestCase):
    """Sanity-check the ProposedAction field names that the tool depends on.

    If schema.py changes these field names again, the tool needs to be
    updated in lock-step. This test catches drift early.
    """

    def test_proposed_action_required_fields_are_action_type_and_target(self):
        try:
            from hitl_core.schema import ProposedAction
        except ImportError:
            self.skipTest("hitl_core.schema not importable in this env")
            return
        # Pydantic v2 exposes fields via model_fields. The names matter:
        # query_radius_logs constructs ProposedAction(action_type=..., target=...,
        # parameters=..., risk_level=..., reversible=...). If any of these
        # required field names ever change, the tool's setup will raise.
        required = {
            name for name, field in ProposedAction.model_fields.items()
            if field.is_required()
        }
        # The minimum required set we depend on:
        self.assertIn("action_type", required,
            "ProposedAction.action_type is required — tool depends on this name")
        self.assertIn("target", required,
            "ProposedAction.target is required — tool depends on this name")
        # 'tool_name' / 'tool_args' must NOT have crept in (we don't use them):
        all_fields = set(ProposedAction.model_fields.keys())
        self.assertNotIn("tool_name", all_fields,
            "If ProposedAction.tool_name was added, the demo tool should adopt it")
        self.assertNotIn("tool_args", all_fields)


if __name__ == "__main__":
    unittest.main(verbosity=2)
