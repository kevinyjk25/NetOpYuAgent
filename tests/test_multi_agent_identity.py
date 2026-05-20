"""
tests/test_multi_agent_identity.py
─────────────────────────────────
Unit tests for Phase 1 multi-agent foundation:

  - AgentIdentityConfig dataclass + AgentSkillSpec
  - _load_agent_identity_config YAML loader
  - Env var overrides (AGENT_ID, AGENT_DISPLAY_NAME, AGENT_PEERS, etc.)
  - a2a/agent_card.py — both legacy path AND identity-driven path
  - Peer URL merge (cfg.registry.agent_urls + cfg.agent.peer_urls, deduped)

Phase 1 does not include actual inter-agent dispatch — that's Phase 2.
These tests cover identity + discovery static configuration only.

Run:
    python -m unittest tests.test_multi_agent_identity
    python -m pytest tests/test_multi_agent_identity.py -v
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock


# Make the project root importable in unittest mode (pytest does this
# automatically via pytest.ini; unittest does not).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestAgentIdentityConfigDataclass(unittest.TestCase):
    """Basic shape + defaults of AgentIdentityConfig."""

    def test_defaults(self):
        from config import AgentIdentityConfig
        cfg = AgentIdentityConfig()
        self.assertEqual(cfg.agent_id, "default-agent")
        self.assertEqual(cfg.display_name, "IT Ops Agent")
        self.assertIn("IT operations", cfg.description)
        self.assertEqual(cfg.capabilities, [])
        self.assertEqual(cfg.peer_urls, [])
        self.assertEqual(cfg.peer_refresh_interval_s, 120)

    def test_agent_skill_spec_defaults_name_from_id(self):
        from config import AgentSkillSpec
        s = AgentSkillSpec(skill_id="lan_diagnose")
        # name defaults to skill_id if blank — `__post_init__` handles this.
        self.assertEqual(s.name, "lan_diagnose")
        # tags / examples / description all default to empty.
        self.assertEqual(s.tags, [])
        self.assertEqual(s.examples, [])
        self.assertEqual(s.description, "")

    def test_agent_skill_spec_preserves_explicit_name(self):
        from config import AgentSkillSpec
        s = AgentSkillSpec(skill_id="lan_diagnose", name="LAN Diagnose")
        self.assertEqual(s.name, "LAN Diagnose")


class TestAgentIdentityYAMLLoader(unittest.TestCase):
    """_load_agent_identity_config + env override semantics."""

    def test_empty_yaml_returns_defaults(self):
        from config import _load_agent_identity_config
        with mock.patch.dict(os.environ, {}, clear=False):
            # Defensive clear of any AGENT_* env that might leak in.
            for k in ("AGENT_ID", "AGENT_DISPLAY_NAME", "AGENT_DESCRIPTION",
                      "AGENT_PEERS", "AGENT_PEER_REFRESH_S"):
                os.environ.pop(k, None)
            cfg = _load_agent_identity_config({})
        self.assertEqual(cfg.agent_id, "default-agent")
        self.assertEqual(cfg.peer_urls, [])

    def test_full_yaml_section(self):
        from config import _load_agent_identity_config
        for k in ("AGENT_ID", "AGENT_DISPLAY_NAME", "AGENT_DESCRIPTION",
                  "AGENT_PEERS", "AGENT_PEER_REFRESH_S"):
            os.environ.pop(k, None)
        cfg = _load_agent_identity_config({
            "agent_id":     "lan-agent",
            "display_name": "LAN Operations Agent",
            "description":  "Internal LAN devices",
            "capabilities": [
                {"skill_id": "lan_diagnose", "tags": ["lan", "switch"],
                 "description": "Diagnose LAN"},
                {"skill_id": "lan_config",   "tags": ["lan", "destructive"]},
            ],
            "peers": ["http://localhost:8001", "http://localhost:8002"],
            "peer_refresh_interval_s": 60,
        })
        self.assertEqual(cfg.agent_id, "lan-agent")
        self.assertEqual(cfg.display_name, "LAN Operations Agent")
        self.assertEqual(cfg.description, "Internal LAN devices")
        self.assertEqual(len(cfg.capabilities), 2)
        self.assertEqual(cfg.capabilities[0].skill_id, "lan_diagnose")
        self.assertEqual(cfg.capabilities[0].tags, ["lan", "switch"])
        self.assertEqual(cfg.capabilities[1].skill_id, "lan_config")
        self.assertEqual(cfg.peer_urls, ["http://localhost:8001", "http://localhost:8002"])
        self.assertEqual(cfg.peer_refresh_interval_s, 60)

    def test_yaml_with_peer_urls_alias(self):
        """Accept both `peers` and `peer_urls` as keys."""
        from config import _load_agent_identity_config
        cfg = _load_agent_identity_config({
            "agent_id": "x",
            "peer_urls": ["http://a", "http://b"],
        })
        self.assertEqual(cfg.peer_urls, ["http://a", "http://b"])

    def test_env_override_agent_id(self):
        from config import _load_agent_identity_config
        with mock.patch.dict(os.environ, {"AGENT_ID": "wan-agent"}, clear=False):
            cfg = _load_agent_identity_config({"agent_id": "lan-agent"})
        # Env should win.
        self.assertEqual(cfg.agent_id, "wan-agent")

    def test_env_override_peers_comma_separated(self):
        from config import _load_agent_identity_config
        with mock.patch.dict(
            os.environ,
            {"AGENT_PEERS": "http://a:8000,http://b:8001 , http://c:8002"},
            clear=False,
        ):
            cfg = _load_agent_identity_config({"peers": ["http://yaml-peer"]})
        # Env wins; whitespace is stripped.
        self.assertEqual(
            cfg.peer_urls,
            ["http://a:8000", "http://b:8001", "http://c:8002"],
        )

    def test_env_override_refresh(self):
        from config import _load_agent_identity_config
        with mock.patch.dict(os.environ, {"AGENT_PEER_REFRESH_S": "45"}, clear=False):
            cfg = _load_agent_identity_config({"peer_refresh_interval_s": 120})
        self.assertEqual(cfg.peer_refresh_interval_s, 45)

    def test_malformed_capabilities_silently_skipped(self):
        """A cap entry missing skill_id should drop, not blow up."""
        from config import _load_agent_identity_config
        cfg = _load_agent_identity_config({
            "capabilities": [
                {"skill_id": "valid"},
                {"tags": ["no-id-here"]},           # invalid: missing skill_id
                "not even a dict",                  # invalid: not a dict
                {"skill_id": "", "tags": ["empty-id"]},  # invalid: empty id
                {"skill_id": "also_valid"},
            ],
        })
        ids = [c.skill_id for c in cfg.capabilities]
        self.assertEqual(ids, ["valid", "also_valid"])


class TestAgentCardLegacyPath(unittest.TestCase):
    """get_agent_card() with no identity arg = old single-agent behaviour."""

    def _load_agent_card_module(self):
        """Load agent_card.py directly, bypassing a2a/__init__.py.

        a2a/__init__.py pulls in PushNotificationService which imports httpx.
        That's fine in production but means a minimal-deps test env (e.g.
        a CI runner that hasn't `pip install`-ed yet) can't import the
        full a2a package. We load the file directly so the test stays
        focused on agent_card behaviour, not on a2a/'s outer imports.
        """
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "agent_card_test_load",
            os.path.join(os.path.dirname(__file__), "..", "a2a", "agent_card.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_legacy_card_shape(self):
        mod = self._load_agent_card_module()
        card = mod.get_agent_card("http://x/api/v1/a2a")
        self.assertEqual(card["url"], "http://x/api/v1/a2a")
        self.assertIn("name", card)
        self.assertIn("skills", card)
        self.assertGreater(len(card["skills"]), 0)
        # Phase-1 addition: agent_id field always present even in legacy
        # path so peer registries can index consistently.
        self.assertEqual(card["agent_id"], "default-agent")


class TestAgentCardIdentityDriven(unittest.TestCase):
    """get_agent_card(..., identity=...) uses configured fields."""

    def _load_agent_card_module(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "agent_card_test_load2",
            os.path.join(os.path.dirname(__file__), "..", "a2a", "agent_card.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_card_uses_identity_fields(self):
        mod = self._load_agent_card_module()
        from config import AgentIdentityConfig, AgentSkillSpec
        identity = AgentIdentityConfig(
            agent_id     = "lan-agent",
            display_name = "LAN Ops",
            description  = "Internal LAN handler",
            capabilities = [
                AgentSkillSpec(skill_id="lan_diagnose", description="Diagnose",
                               tags=["lan", "switch"]),
                AgentSkillSpec(skill_id="lan_config",   description="Configure",
                               tags=["lan", "destructive"]),
            ],
        )
        card = mod.get_agent_card("http://x/api/v1/a2a", identity=identity)
        self.assertEqual(card["name"], "LAN Ops")
        self.assertEqual(card["agent_id"], "lan-agent")
        self.assertEqual(card["description"], "Internal LAN handler")
        # Skills come from identity, NOT from legacy SKILLS list.
        skill_ids = [s["id"] for s in card["skills"]]
        self.assertEqual(skill_ids, ["lan_diagnose", "lan_config"])

    def test_identity_with_empty_capabilities_falls_back_to_legacy_skills(self):
        """If capabilities=[], we DON'T strip skills entirely — use legacy.

        Rationale: empty capabilities probably means the operator started
        from the default yaml and just changed agent_id. Hiding all
        skills would silently break retrieval / matching for that agent.
        Set capabilities=[] only if you genuinely want a zero-skill agent.
        """
        mod = self._load_agent_card_module()
        from config import AgentIdentityConfig
        identity = AgentIdentityConfig(
            agent_id="lan-agent", display_name="LAN", capabilities=[]
        )
        card = mod.get_agent_card("http://x/api/v1/a2a", identity=identity)
        # Inherit legacy skill list.
        self.assertGreater(len(card["skills"]), 0)
        # But agent_id is the configured one.
        self.assertEqual(card["agent_id"], "lan-agent")


class TestA2AServerPublishesIdentity(unittest.TestCase):
    """Regression: the PUBLISHED AgentCard (served at
    /.well-known/agent-card.json) must carry the configured agent_id.

    Bug fixed 2026-05: create_a2a_app() called get_agent_card(base_url)
    with NO identity, so peers always discovered us as "default-agent"
    regardless of AGENT_ID. The /system/peers self block was correct
    (read straight from cfg.agent) but the published card — which is
    what peers actually fetch — was wrong. This test guards the wiring.

    We assert on source rather than booting FastAPI (which needs the
    full dependency tree). The contract: create_a2a_app must accept an
    `identity` param AND pass it to get_agent_card.
    """

    def test_create_a2a_app_accepts_identity_param(self):
        import inspect
        # Read the function signature from source without importing a2a
        # (which pulls httpx). Use the AST-free approach: read the file.
        import os
        server_path = os.path.join(
            os.path.dirname(__file__), "..", "a2a", "server.py",
        )
        with open(server_path) as f:
            src = f.read()
        # create_a2a_app must declare an identity parameter
        self.assertIn("identity", src.split("def create_a2a_app")[1].split(")")[0],
                      "create_a2a_app must accept an identity parameter")
        # and must forward it to get_agent_card
        self.assertIn("get_agent_card(base_url, identity=identity)", src,
                      "create_a2a_app must pass identity to get_agent_card")

    def test_main_passes_identity_to_create_a2a_app(self):
        import os
        main_path = os.path.join(os.path.dirname(__file__), "..", "main.py")
        with open(main_path) as f:
            src = f.read()
        # The create_a2a_app call in main.py must pass identity=cfg.agent
        self.assertIn("identity   = cfg.agent", src,
                      "main.py must pass identity=cfg.agent to create_a2a_app")


class TestAgentDiscoveryParse(unittest.TestCase):
    """`AgentDiscovery._parse` must read agent_id from the raw card JSON.

    Bug fixed 2026-05: before this, _parse never read the top-level
    `agent_id` field that the Phase-1 AgentCard publishes, so every
    discovered peer got a fresh UUID per fetch — registry dedup was
    unreliable and /system/peers showed garbage IDs in the UI.
    """

    def _load_disc(self):
        """Load discovery module bypassing registry/__init__ httpx import."""
        try:
            from registry.discovery import AgentDiscovery
            return AgentDiscovery
        except ImportError:
            self.skipTest("registry.discovery deps missing in test env")

    def test_agent_id_from_card_field(self):
        """When card publishes agent_id, AgentEntry must reuse it."""
        AgentDiscovery = self._load_disc()
        from registry.schemas import RegistrationSource
        card = {
            "name": "WAN Agent",
            "agent_id": "wan-agent",
            "description": "WAN ops",
            "url": "http://localhost:8001/api/v1/a2a",
            "skills": [{"id": "wan_diagnose", "tags": ["wan"]}],
        }
        entry = AgentDiscovery._parse(
            card, "http://localhost:8001/api/v1/a2a",
            RegistrationSource.STATIC,
        )
        self.assertEqual(entry.agent_id, "wan-agent")

    def test_no_agent_id_falls_back_to_uuid(self):
        """Legacy / non-Phase-1 cards keep UUID fallback for backwards compat."""
        AgentDiscovery = self._load_disc()
        from registry.schemas import RegistrationSource
        import uuid
        card = {"name": "Legacy", "url": "http://x/api/v1/a2a", "skills": []}
        entry = AgentDiscovery._parse(
            card, "http://x/api/v1/a2a", RegistrationSource.STATIC,
        )
        # Must parse as a UUID — no exception
        uuid.UUID(entry.agent_id)

    def test_blank_agent_id_falls_back_to_uuid(self):
        """Whitespace agent_id must not be treated as valid."""
        AgentDiscovery = self._load_disc()
        from registry.schemas import RegistrationSource
        import uuid
        card = {"name": "Weird", "agent_id": "   ", "url": "http://x"}
        entry = AgentDiscovery._parse(
            card, "http://x", RegistrationSource.STATIC,
        )
        uuid.UUID(entry.agent_id)

    def test_source_none_is_rejected(self):
        """_parse must NOT accept source=None.

        Regression: the peer-refresh loop in main.py passed source=None
        thinking it meant 'keep existing labels', but RegistrationSource
        is a required enum — None fails AgentEntry validation, fetch_many
        swallows the error per-URL, and the peer silently never registers
        (showed as peers: [] in /system/peers). The loop now passes
        RegistrationSource.STATIC. This test guards _parse's contract.
        """
        AgentDiscovery = self._load_disc()
        card = {"name": "X", "agent_id": "x-agent", "url": "http://x"}
        with self.assertRaises(Exception):
            AgentDiscovery._parse(card, "http://x", None)

    def test_main_refresh_loop_uses_valid_source(self):
        """main.py's peer-refresh loop must not pass source=None.

        Source-level guard — the actual call must use a real enum value.
        """
        import os
        main_path = os.path.join(os.path.dirname(__file__), "..", "main.py")
        with open(main_path) as f:
            src = f.read()
        # The actual register_from_urls call must pass a valid enum, not None.
        # We check the call expression specifically (not comments, which may
        # mention source=None when explaining the historical bug).
        self.assertIn("source=_RS.STATIC", src,
                      "peer refresh loop must pass a valid RegistrationSource")
        # Guard against the buggy call expression returning. We look for the
        # call-site pattern `register_from_urls(` followed (within a small
        # window) by `source=None` — comments don't contain that sequence.
        import re
        # Find the register_from_urls(...) call body
        m = re.search(r"register_from_urls\((.*?)\)", src, re.DOTALL)
        self.assertIsNotNone(m, "register_from_urls call not found in main.py")
        call_args = m.group(1)
        self.assertNotIn("source=None", call_args,
                         "register_from_urls call must not pass source=None")


class TestPeerURLMerge(unittest.TestCase):
    """The merge logic in main.py: union(registry.agent_urls, agent.peer_urls)."""

    def test_dedupe_preserves_first_occurrence_order(self):
        # Reproduce the merge logic from main.py inline so we don't have to
        # import build_services (which pulls in fastapi / httpx / etc.).
        registry_urls = ["http://a", "http://b"]
        agent_urls    = ["http://b", "http://c"]   # b overlaps
        merged: list[str] = []
        seen: set[str] = set()
        for u in registry_urls + agent_urls:
            u = u.strip()
            if u and u not in seen:
                seen.add(u)
                merged.append(u)
        self.assertEqual(merged, ["http://a", "http://b", "http://c"])

    def test_empty_inputs_yield_empty_list(self):
        registry_urls = []
        agent_urls    = []
        merged: list[str] = []
        seen: set[str] = set()
        for u in registry_urls + agent_urls:
            u = u.strip()
            if u and u not in seen:
                seen.add(u)
                merged.append(u)
        self.assertEqual(merged, [])

    def test_whitespace_stripped_and_blanks_dropped(self):
        registry_urls = ["  http://a  ", "", "  "]
        agent_urls    = ["http://b"]
        merged: list[str] = []
        seen: set[str] = set()
        for u in registry_urls + agent_urls:
            u = u.strip()
            if u and u not in seen:
                seen.add(u)
                merged.append(u)
        self.assertEqual(merged, ["http://a", "http://b"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
