"""tests/test_delegation_provenance.py
=======================================

End-to-end regression: when LAN delegates to dc-agent and dc-agent's
runtime loop triggers a HITL card, the card MUST carry source_agent /
source_session_id / source_query so the peer operator sees who's
upstream and what the original user asked.

Provenance flow:

  1. task/delegation.py.delegate_fn builds TaskDefinition.metadata with
     source_agent / source_session_id / source_query
  2. task/inter/coordinator.py.A2ATaskDispatcher.dispatch packs the
     full task.metadata into A2A request params.metadata
  3. integrations/adapters/hitl_executor.py.HitlExecutor.execute
     extracts these fields off meta.* into env_context
  4. _raise_tool_hitl / _raise_tool_hitl_batch / _raise_multi_mode
     each call _extract_delegation_provenance(env_context) and stamp
     the HitlPayload

This test pins the schema contract + the helper. The full A2A wire test
needs httpx + a live server; for unit-level coverage we stage what the
peer receives by hand and verify the right HitlPayload fields fire.
"""

from __future__ import annotations
import unittest


def _try_import_schema():
    try:
        from hitl_core.schema import (
            HitlPayload, ProposedAction, TriggerKind, RiskLevel,
        )
        return HitlPayload, ProposedAction, TriggerKind, RiskLevel
    except ImportError:
        return None, None, None, None


def _try_import_helper():
    """Load _extract_delegation_provenance without dragging hitl_core/pydantic.

    The helper itself only uses dict.get + bool — no pydantic, no schemas.
    Direct importing pulls the module __init__ which transitively requires
    pydantic; we exec just the function body instead so the test runs in
    sandbox environments without pydantic installed.
    """
    try:
        from integrations.adapters.hitl_executor import (
            _extract_delegation_provenance,
        )
        return _extract_delegation_provenance
    except ImportError:
        pass
    # Fallback: parse the function out of the source.
    import os
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "integrations", "adapters", "hitl_executor.py",
    )
    try:
        with open(path) as f:
            src = f.read()
    except OSError:
        return None
    start = src.find("def _extract_delegation_provenance(")
    if start == -1:
        return None
    # End: next top-level helper or class (start of line)
    import re
    rest = src[start:]
    m = re.search(r"\n(?:def |class )", rest)
    if m is None:
        return None
    fn_src = rest[: m.start()]
    # Replace the typing import with `Optional = type(None) hack`
    ns: dict = {
        "Optional": __import__("typing").Optional,
        "Any":      __import__("typing").Any,
    }
    try:
        exec(fn_src, ns)
    except Exception:
        return None
    return ns.get("_extract_delegation_provenance")


class TestHitlPayloadSchema(unittest.TestCase):
    """HitlPayload must carry the 3 provenance fields, all Optional."""

    def setUp(self):
        HitlPayload, ProposedAction, TriggerKind, RiskLevel = _try_import_schema()
        if HitlPayload is None:
            self.skipTest("hitl_core.schema not importable (pydantic missing)")
        self.HitlPayload      = HitlPayload
        self.ProposedAction   = ProposedAction
        self.TriggerKind      = TriggerKind
        self.RiskLevel        = RiskLevel

    def test_provenance_fields_default_none(self):
        """Backward compat: omitting all 3 fields must work."""
        p = self.HitlPayload(
            thread_id="s1", context_id="s1",
            user_query="check spine-1",
            proposed_action=self.ProposedAction(
                action_type="diagnostic", target="-", parameters={},
            ),
        )
        self.assertIsNone(p.source_agent)
        self.assertIsNone(p.source_session_id)
        self.assertIsNone(p.source_query)

    def test_provenance_fields_round_trip(self):
        """Setting all 3 fields persists through serialisation."""
        p = self.HitlPayload(
            thread_id="s1", context_id="s1",
            user_query="check spine-1",
            proposed_action=self.ProposedAction(
                action_type="diagnostic", target="-", parameters={},
            ),
            source_agent="lan-agent",
            source_session_id="sess-lan-abc",
            source_query="spine-1 的 BGP EVPN 邻居状态",
        )
        self.assertEqual(p.source_agent, "lan-agent")
        self.assertEqual(p.source_session_id, "sess-lan-abc")
        self.assertEqual(p.source_query, "spine-1 的 BGP EVPN 邻居状态")

        # Serialise + reload — schema must accept its own output
        dumped = p.model_dump()
        self.assertEqual(dumped["source_agent"],      "lan-agent")
        self.assertEqual(dumped["source_session_id"], "sess-lan-abc")
        self.assertEqual(dumped["source_query"],      "spine-1 的 BGP EVPN 邻居状态")

        p2 = self.HitlPayload(**dumped)
        self.assertEqual(p2.source_agent,      p.source_agent)
        self.assertEqual(p2.source_session_id, p.source_session_id)
        self.assertEqual(p2.source_query,      p.source_query)


class TestExtractDelegationProvenance(unittest.TestCase):
    """The helper that pulls source_xxx out of env_context."""

    def setUp(self):
        self._extract = _try_import_helper()
        if self._extract is None:
            self.skipTest("hitl_executor not importable")

    def test_none_env_context(self):
        self.assertEqual(self._extract(None), (None, None, None))

    def test_empty_env_context(self):
        self.assertEqual(self._extract({}), (None, None, None))

    def test_user_query_not_delegated(self):
        """Normal (non-delegated) request — no source_* keys present.
        All three components must be None so the HITL card UI knows to
        skip the 'Delegated from' banner."""
        env = {
            "_fts_context": "some memory",
            "_confirmed_facts": [],
        }
        self.assertEqual(self._extract(env), (None, None, None))

    def test_delegated_request_full_provenance(self):
        """Peer-side env_context populated by HitlExecutor.execute when
        meta carried source_xxx."""
        env = {
            "source_agent":      "lan-agent",
            "source_session_id": "sess-lan-abc",
            "source_query":      "spine-1 的 BGP EVPN 邻居状态",
        }
        result = self._extract(env)
        self.assertEqual(result, (
            "lan-agent", "sess-lan-abc", "spine-1 的 BGP EVPN 邻居状态",
        ))

    def test_partial_provenance(self):
        """Some delegations might not have an original_query (e.g. when
        runtime stash didn't fire). The helper must return None for the
        missing piece, not break."""
        env = {
            "source_agent":      "lan-agent",
            "source_session_id": "sess-lan-abc",
            # source_query missing
        }
        result = self._extract(env)
        self.assertEqual(result, ("lan-agent", "sess-lan-abc", None))

    def test_empty_strings_treated_as_none(self):
        """Provenance is meaningful only when non-empty. Empty strings
        from `delegate_fn(original_query="")` should not produce a
        misleading 'Delegated from ""' banner."""
        env = {
            "source_agent":      "",
            "source_session_id": "",
            "source_query":      "",
        }
        # Helper uses `or None` so falsy → None
        result = self._extract(env)
        self.assertEqual(result, (None, None, None))


class TestTaskDefinitionMetadataShape(unittest.TestCase):
    """Verify delegate_fn writes the right keys into TaskDefinition.metadata.

    Inspects the source of task/delegation.py to confirm the canonical
    field names — if a refactor renames or drops one, this catches it.
    """

    def test_delegation_module_writes_expected_keys(self):
        import os
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "task", "delegation.py",
        )
        with open(path) as f:
            src = f.read()
        # These are the canonical metadata keys read by _extract_delegation_provenance
        for key in ("source_agent", "source_session_id", "source_query"):
            self.assertIn(
                f'"{key}"', src,
                f"task/delegation.py must write metadata['{key}'] — "
                "renaming/removing breaks peer-side HITL provenance",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
