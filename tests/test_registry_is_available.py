"""tests/test_registry_is_available.py
========================================

Regression test for AgentEntry.is_available — the eligibility check
used by delegation, capability resolution, and routing.

Background (2026-05): the LAN agent emitted [DELEGATE:dc-agent] but the
backend logged 'delegate: agent dc-agent not available' even though
dc-agent was registered, reachable, and serving health checks. Root
cause: is_available returned False on UNKNOWN, but newly-discovered
peers start in UNKNOWN and only transition to HEALTHY after the health
watcher's first tick (~60s). So in the first minute after startup, the
peer was BOTH listed in the LLM's AVAILABLE PEERS prompt AND rejected by
delegate_fn — a 60s blind window where delegation just silently failed.

Fix: treat UNKNOWN as available (optimistic). Only UNHEALTHY excludes.

These tests pin the contract so a future change to "strict-health-only"
fails CI rather than reintroducing the silent-failure window.
"""

from __future__ import annotations
import unittest


def _try_import():
    try:
        from registry.schemas import AgentEntry, AgentHealthState, RawAgentCard
        return AgentEntry, AgentHealthState, RawAgentCard
    except ImportError:
        return None, None, None


class TestIsAvailableTreatsUnknownAsAvailable(unittest.TestCase):

    def setUp(self):
        AgentEntry, AgentHealthState, RawAgentCard = _try_import()
        if AgentEntry is None:
            self.skipTest("registry.schemas not importable (pydantic missing in sandbox)")
        self.AgentEntry      = AgentEntry
        self.AgentHealthState = AgentHealthState
        self.RawAgentCard    = RawAgentCard

    def _make_entry(self, health):
        return self.AgentEntry(
            agent_id="dc-agent",
            card=self.RawAgentCard(name="DC Agent", url="http://localhost:8001"),
            health=health,
        )

    def test_unknown_is_available(self):
        """The critical regression — UNKNOWN must NOT exclude.
        
        This is the case that hit production: peer just registered, health
        watcher hasn't ticked yet, registry still shows UNKNOWN, but the
        peer is actually reachable. Returning False here breaks delegation
        for the first ~60s after every startup.
        """
        e = self._make_entry(self.AgentHealthState.UNKNOWN)
        self.assertTrue(e.is_available,
            "UNKNOWN must be treated as available — see ARCHITECTURE §8.8 "
            "follow-up. Excluding UNKNOWN creates a 60s blind window at startup.")

    def test_healthy_is_available(self):
        e = self._make_entry(self.AgentHealthState.HEALTHY)
        self.assertTrue(e.is_available)

    def test_degraded_is_available(self):
        e = self._make_entry(self.AgentHealthState.DEGRADED)
        self.assertTrue(e.is_available)

    def test_unhealthy_is_not_available(self):
        """The watcher has actively failed it — exclude."""
        e = self._make_entry(self.AgentHealthState.UNHEALTHY)
        self.assertFalse(e.is_available)


if __name__ == "__main__":
    unittest.main(verbosity=2)
