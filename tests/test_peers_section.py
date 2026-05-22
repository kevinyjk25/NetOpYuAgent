"""tests/test_peers_section.py
==============================

Regression tests for LLMEngine._build_peers_section / _extract_peer_keywords.

The first bug class caught here: AgentEntry.agent_id is on the entry (not
the card), but the original code read `card.agent_id` which is always None
on a RawAgentCard. Result: every entry got filtered out and the peers
section came out empty — the LAN agent, asked about `spine-1 BGP EVPN`,
exhausted local tools instead of delegating.

These tests work with shape-mimicking fakes so we don't need a live
registry; they exercise the exact attribute access paths
_build_peers_section relies on.
"""

from __future__ import annotations
import unittest


class _FakeSkill:
    def __init__(self, sid, name, desc=""):
        self.id = sid
        self.name = name
        self.description = desc
        self.tags = []
        self.examples = []


class _FakeCard:
    """Matches RawAgentCard: name (not agent_id), description, skills, capabilities."""
    def __init__(self, name, skills, description=""):
        self.name = name
        self.skills = skills
        self.description = description
        self.capabilities = {}


class _FakeHealth:
    """Mimics AgentHealthState enum — has .value attribute."""
    def __init__(self, value): self.value = value


class _FakeEntry:
    """Matches AgentEntry: agent_id at top level, card nested, health at top level."""
    def __init__(self, agent_id, card, health_value="healthy"):
        self.agent_id = agent_id
        self.card = card
        self.health = _FakeHealth(health_value)


class _FakeStore:
    def __init__(self, entries: dict):
        self._store = entries


class _FakeRegistry:
    """Mimics AgentRegistry: registry._store._store is the agent_id → AgentEntry dict."""
    def __init__(self, entries: dict):
        self._store = _FakeStore(entries)


def _load_engine_methods():
    """Pull _build_peers_section + _extract_peer_keywords off LLMEngine without
    importing the full module (which transitively requires pydantic, not
    available in the sandbox)."""
    try:
        from integrations.clients.llm_engine import LLMEngine
        return LLMEngine
    except ImportError:
        return None


class TestPeersSectionWithRealEntryShape(unittest.TestCase):
    """The single most important regression: AgentEntry.agent_id is the
    entry's attribute, NOT the card's. If _build_peers_section reads
    `card.agent_id` it gets None and the entry silently disappears."""

    def setUp(self):
        self.LLMEngine = _load_engine_methods()
        if self.LLMEngine is None:
            self.skipTest("LLMEngine not importable (pydantic missing in sandbox)")
        # Build the engine bypassing __init__ — we only need the two slots.
        self.engine = self.LLMEngine.__new__(self.LLMEngine)
        # Two agents — lan-agent is self, dc-agent is the peer we want listed.
        self.engine._peer_registry = _FakeRegistry({
            "lan-agent": _FakeEntry("lan-agent", _FakeCard(
                name="LAN Agent",
                skills=[
                    _FakeSkill("lan_diagnose", "LAN Diag", "LAN diag"),
                ],
                description="LAN agent",
            )),
            "dc-agent": _FakeEntry("dc-agent", _FakeCard(
                name="DC Agent",
                skills=[
                    _FakeSkill("dc_fabric_diagnose", "DC FD",
                               "Diagnose DC fabric: spine-leaf topology, BGP-EVPN, VXLAN"),
                    _FakeSkill("dc_fabric_config", "DC FC",
                               "Push DC fabric config on spine and leaf"),
                    _FakeSkill("dc_loadbalancer", "DC LB",
                               "Manage F5/NSX load balancer"),
                ],
                description="DC fabric IT ops — spine-leaf, BGP-EVPN, VXLAN",
            )),
        })
        self.engine._self_agent_id = "lan-agent"

    def test_self_filtered_out_and_peer_included(self):
        """lan-agent (self) must be excluded; dc-agent must appear with its id."""
        out = self.engine._build_peers_section()
        self.assertIn("dc-agent", out)
        # 'lan-agent' should NOT appear as a peer entry. The agent_id may
        # incidentally appear as a token inside `owns:` or `capabilities:`,
        # so check the per-peer marker `  - lan-agent` instead.
        self.assertNotIn("  - lan-agent", out)

    def test_peers_section_nonempty_when_peer_exists(self):
        """The previous bug returned "" even with peers populated. Catch that."""
        out = self.engine._build_peers_section()
        self.assertTrue(out.strip(),
            "peers section must not be empty when at least one healthy peer is registered")
        self.assertIn("AVAILABLE PEERS", out)

    def test_capabilities_listed_from_skill_ids(self):
        """Each peer line must list capability IDs (used by [DELEGATE:*cap])."""
        out = self.engine._build_peers_section()
        self.assertIn("dc_fabric_diagnose", out)
        self.assertIn("dc_fabric_config", out)
        self.assertIn("dc_loadbalancer", out)

    def test_owns_hints_extract_entity_tokens(self):
        """The owns: line must surface entity tokens the user might type."""
        out = self.engine._build_peers_section()
        self.assertIn("owns:", out)
        # Distinctive tokens from skill descriptions
        self.assertIn("spine", out.lower())
        self.assertIn("leaf",  out.lower())
        # Hyphenated technical token preserved
        self.assertIn("BGP-EVPN", out)
        self.assertIn("VXLAN",    out)

    def test_health_unhealthy_filtered_out(self):
        """Peers marked unhealthy must not appear."""
        # Mark dc-agent unhealthy
        self.engine._peer_registry._store._store["dc-agent"].health = _FakeHealth("unhealthy")
        out = self.engine._build_peers_section()
        self.assertNotIn("dc-agent", out)

    def test_unknown_health_still_included(self):
        """Newly-discovered peers with health='unknown' should still appear —
        otherwise peer discovery → first prompt has a stale gap."""
        self.engine._peer_registry._store._store["dc-agent"].health = _FakeHealth("unknown")
        out = self.engine._build_peers_section()
        self.assertIn("dc-agent", out)


class TestEmptyRegistry(unittest.TestCase):
    """No registry / no peers / no self_agent_id edge cases."""

    def setUp(self):
        self.LLMEngine = _load_engine_methods()
        if self.LLMEngine is None:
            self.skipTest("LLMEngine not importable")
        self.engine = self.LLMEngine.__new__(self.LLMEngine)

    def test_no_registry_returns_empty(self):
        self.engine._peer_registry = None
        self.engine._self_agent_id = ""
        self.assertEqual(self.engine._build_peers_section(), "")

    def test_empty_registry_returns_empty(self):
        self.engine._peer_registry = _FakeRegistry({})
        self.engine._self_agent_id = "lan-agent"
        self.assertEqual(self.engine._build_peers_section(), "")

    def test_only_self_returns_empty(self):
        """If the only entry is self, peers section must be empty (no point
        offering to delegate to yourself)."""
        self.engine._peer_registry = _FakeRegistry({
            "lan-agent": _FakeEntry("lan-agent", _FakeCard("LAN Agent", [], "")),
        })
        self.engine._self_agent_id = "lan-agent"
        self.assertEqual(self.engine._build_peers_section(), "")


class TestExtractPeerKeywords(unittest.TestCase):
    """The keyword extractor — minimal sanity checks on the heuristic."""

    def setUp(self):
        self.LLMEngine = _load_engine_methods()
        if self.LLMEngine is None:
            self.skipTest("LLMEngine not importable")

    def test_stopwords_dropped(self):
        """Generic IT-ops words must not pollute the owns: hint."""
        out = self.LLMEngine._extract_peer_keywords(
            agent_id="x", cap_ids=["lan_diagnose"],
            all_descs=["Diagnose the LAN network device"],
        )
        # All these are in _PEER_HINT_STOPWORDS
        for stopword in ("diagnose", "network", "device", "the"):
            self.assertNotIn(stopword, out,
                f"stopword '{stopword}' leaked into hints {out}")

    def test_hyphenated_technical_tokens_preserved(self):
        """BGP-EVPN must stay as one token, not split into BGP + EVPN."""
        out = self.LLMEngine._extract_peer_keywords(
            agent_id="x", cap_ids=[],
            all_descs=["spine-leaf topology with BGP-EVPN and VXLAN"],
        )
        self.assertIn("spine-leaf", out)
        self.assertIn("BGP-EVPN", out)
        self.assertIn("VXLAN", out)

    def test_dedup_case_insensitive(self):
        """Same token in different case shouldn't appear twice."""
        out = self.LLMEngine._extract_peer_keywords(
            agent_id="x", cap_ids=["spine_diagnose"],
            all_descs=["spine and Spine and SPINE"],
        )
        # 'spine' should appear exactly once (case-insensitive dedupe)
        lowered = [t.lower() for t in out]
        self.assertEqual(lowered.count("spine"), 1)

    def test_cap_at_12_tokens(self):
        """Token budget — should not blow up arbitrary-length descriptions."""
        # 30 unique non-stopword tokens
        words = [f"foobar{i}" for i in range(30)]
        out = self.LLMEngine._extract_peer_keywords(
            agent_id="x", cap_ids=[],
            all_descs=[" ".join(words)],
        )
        self.assertLessEqual(len(out), 12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
