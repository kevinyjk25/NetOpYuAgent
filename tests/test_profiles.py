"""
tests/test_profiles.py
──────────────────────
Tests for the business profile layer (2A — profile decoupling, 2026-05).

Verifies:
  - All three profiles load (default / lan / dc)
  - default profile has NO business tools/skills (decoupling proof)
  - lan and dc tool sets are DISJOINT (role isolation — the precondition
    for meaningful A2A delegation in Phase 2B)
  - Each profile's tool_callables and tool_metadata keys align
  - Unknown profile id falls back to default (safe degradation)
  - ToolLoader / SkillLoader honour the profile parameter

Run:
    python -m unittest tests.test_profiles
    python -m pytest tests/test_profiles.py -v
"""
from __future__ import annotations

import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestProfileLoading(unittest.TestCase):
    def test_all_known_profiles_load(self):
        from profiles import load_profile, available_profiles
        self.assertEqual(set(available_profiles()), {"default", "lan", "dc", "wan"})
        for pid in available_profiles():
            p = load_profile(pid)
            self.assertEqual(p.profile_id, pid)

    def test_wan_profile_tools_and_disjoint(self):
        """WAN is a real third role: has its own tools, disjoint from LAN/DC."""
        from profiles import load_profile
        wan = load_profile("wan")
        lan = load_profile("lan")
        dc  = load_profile("dc")
        wan_tools = set(wan.tool_callables)
        self.assertIn("wan_path_sla", wan_tools)
        self.assertIn("wan_tunnel_status", wan_tools)
        # tool isolation — cross-domain work must go through delegation
        self.assertEqual(wan_tools & set(lan.tool_callables), set())
        self.assertEqual(wan_tools & set(dc.tool_callables), set())
        # a destructive WAN tool exists and is HITL-gated
        self.assertTrue(wan.tool_metadata["wan_failover_path"]["hitl"])

    def test_default_profile_has_no_business(self):
        """The whole point: default = framework only, zero business logic."""
        from profiles import load_profile
        p = load_profile("default")
        self.assertEqual(len(p.tool_callables), 0,
                         "default profile must have NO business tools")
        self.assertEqual(len(p.skill_defs), 0,
                         "default profile must have NO business skills")
        # It still advertises a generic capability so peers see *something*.
        self.assertGreaterEqual(len(p.capabilities), 1)

    def test_lan_profile_populated(self):
        from profiles import load_profile
        p = load_profile("lan")
        self.assertGreater(len(p.tool_callables), 0)
        self.assertIn("list_devices", p.tool_callables)
        self.assertIn("edit_device_config", p.tool_callables)

    def test_dc_profile_populated(self):
        from profiles import load_profile
        p = load_profile("dc")
        self.assertGreater(len(p.tool_callables), 0)
        self.assertIn("dc_bgp_evpn_status", p.tool_callables)
        self.assertIn("dc_config_push", p.tool_callables)

    def test_unknown_profile_falls_back_to_default(self):
        from profiles import load_profile
        p = load_profile("nonexistent-profile-xyz")
        self.assertEqual(p.profile_id, "default")

    def test_empty_profile_id_falls_back_to_default(self):
        from profiles import load_profile
        self.assertEqual(load_profile("").profile_id, "default")
        self.assertEqual(load_profile(None).profile_id, "default")  # type: ignore


class TestProfileIsolation(unittest.TestCase):
    """Role isolation — lan and dc must not share tools. This is what makes
    A2A delegation meaningful: each agent physically lacks the other's tools.
    """

    def test_lan_dc_tools_disjoint(self):
        from profiles import load_profile
        lan = set(load_profile("lan").tool_callables)
        dc = set(load_profile("dc").tool_callables)
        overlap = lan & dc
        self.assertEqual(overlap, set(),
                         f"LAN and DC tools must be disjoint; overlap={overlap}")

    def test_lan_dc_skills_disjoint(self):
        from profiles import load_profile
        lan = set(load_profile("lan").skill_defs)
        dc = set(load_profile("dc").skill_defs)
        self.assertEqual(lan & dc, set())

    def test_dc_tools_all_prefixed(self):
        """DC tools use a dc_ prefix so they're unmistakable in logs / prompts."""
        from profiles import load_profile
        for name in load_profile("dc").tool_callables:
            self.assertTrue(name.startswith("dc_"),
                            f"DC tool {name!r} should start with 'dc_'")


class TestProfileConsistency(unittest.TestCase):
    """Each profile's callables and metadata must reference the same tools."""

    def test_callable_metadata_alignment(self):
        from profiles import load_profile
        for pid in ("default", "lan", "dc"):
            p = load_profile(pid)
            cb = set(p.tool_callables)
            md = set(p.tool_metadata)
            self.assertEqual(
                cb, md,
                f"profile {pid}: callable/metadata mismatch — "
                f"callable-only={cb - md}, metadata-only={md - cb}",
            )

    def test_tools_are_callable(self):
        from profiles import load_profile
        p = load_profile("dc")
        for name, fn in p.tool_callables.items():
            self.assertTrue(callable(fn), f"{name} is not callable")

    def test_dc_tool_actually_runs(self):
        from profiles import load_profile
        p = load_profile("dc")
        out = asyncio.run(p.tool_callables["dc_list_fabric"]({}))
        self.assertIsInstance(out, str)
        self.assertIn("spine", out.lower())


class TestLoadersHonourProfile(unittest.TestCase):
    """ToolLoader / SkillLoader must load only the requested profile's tools."""

    def test_tool_loader_default_has_only_common(self):
        from tools.loader import ToolLoader
        meta = ToolLoader(mode="mock", profile="default").build_metadata()
        # default has only the common builtin tools (read_stored_result, …)
        # — no business tools.
        self.assertIn("read_stored_result", meta)
        self.assertNotIn("list_devices", meta)
        self.assertNotIn("dc_bgp_evpn_status", meta)

    def test_tool_loader_lan_has_lan_not_dc(self):
        from tools.loader import ToolLoader
        meta = ToolLoader(mode="mock", profile="lan").build_metadata()
        self.assertIn("list_devices", meta)
        self.assertNotIn("dc_bgp_evpn_status", meta)

    def test_tool_loader_dc_has_dc_not_lan(self):
        from tools.loader import ToolLoader
        meta = ToolLoader(mode="mock", profile="dc").build_metadata()
        self.assertIn("dc_bgp_evpn_status", meta)
        self.assertNotIn("list_devices", meta)

    def test_tool_loader_callables_match_profile(self):
        from tools.loader import ToolLoader
        cb = ToolLoader(mode="mock", profile="dc").build_callables()
        self.assertIn("dc_config_push", cb)
        self.assertNotIn("edit_device_config", cb)

    def test_skill_loader_honours_profile(self):
        from skills.loader import SkillLoader
        lan_skills = SkillLoader(mode="mock", profile="lan").skill_definitions()
        dc_skills = SkillLoader(mode="mock", profile="dc").skill_definitions()
        # builtin skills appear in both; business skills don't cross over.
        self.assertIn("dc_evpn_troubleshoot", dc_skills)
        self.assertNotIn("dc_evpn_troubleshoot", lan_skills)

    def test_all_business_skills_have_name_field(self):
        """SkillCatalogService.register_all requires 'name'; a skill missing
        it crashes catalog build at boot (regression: DC skills shipped
        without 'name' and broke the DC agent's catalog)."""
        from profiles import load_profile
        for pid in ("lan", "dc"):
            for sid, sk in load_profile(pid).skill_defs.items():
                self.assertTrue(
                    sk.get("name"),
                    f"profile {pid} skill {sid} missing required 'name' field",
                )

class TestAgentDataIsolation(unittest.TestCase):
    """cfg.agent_data_dir() must give each agent_id its own data subtree."""

    def test_data_dir_includes_agent_id(self):
        import importlib, os
        # Reload config under a specific AGENT_ID
        os.environ["AGENT_ID"] = "lan-agent"
        os.environ.pop("AGENT_DATA_DIR", None)
        import config as _cfg
        importlib.reload(_cfg)
        d = _cfg.cfg.agent_data_dir()
        self.assertIn("lan-agent", d)
        self.assertIn("agents", d)
        # cleanup
        import shutil, pathlib
        p = pathlib.Path(d)
        if p.exists() and "agents" in str(p):
            shutil.rmtree(p, ignore_errors=True)
        os.environ.pop("AGENT_ID", None)
        importlib.reload(_cfg)

    def test_different_agents_get_different_dirs(self):
        import importlib, os, shutil, pathlib
        os.environ.pop("AGENT_DATA_DIR", None)
        import config as _cfg

        os.environ["AGENT_ID"] = "lan-agent"
        importlib.reload(_cfg)
        lan = _cfg.cfg.agent_data_dir()

        os.environ["AGENT_ID"] = "dc-agent"
        importlib.reload(_cfg)
        dc = _cfg.cfg.agent_data_dir()

        self.assertNotEqual(lan, dc)
        for d in (lan, dc):
            p = pathlib.Path(d)
            if p.exists() and "agents" in str(p):
                shutil.rmtree(p, ignore_errors=True)
        os.environ.pop("AGENT_ID", None)
        importlib.reload(_cfg)

    def test_explicit_override_wins(self):
        import importlib, os, shutil, pathlib, tempfile
        tmp = tempfile.mkdtemp()
        os.environ["AGENT_DATA_DIR"] = tmp
        import config as _cfg
        importlib.reload(_cfg)
        self.assertEqual(_cfg.cfg.agent_data_dir(), tmp)
        os.environ.pop("AGENT_DATA_DIR", None)
        shutil.rmtree(tmp, ignore_errors=True)
        importlib.reload(_cfg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
