"""
tests/test_access_scenario.py
=============================

Locks in the cross-agent HITL fault scenario data (2026-05): "user alice
cannot access application CRM". The diagnosis must cross both agents:

  - LAN side: alice is fully admitted (RADIUS/802.1X/NAC ok) → LAN is NOT the
    cause; the agent should delegate to DC.
  - DC side: alice holds no role on CRM → DENIED → the real root cause; the
    fix (dc_grant_app_access) is HITL-gated on the DC side (mode B).

If these datasets drift apart the scenario stops making sense, so we assert the
key facts directly. Method 甲: queries read-only, only grant/revoke are HITL.
"""
import asyncio
import unittest


def _run(coro):
    return asyncio.run(coro)


class TestLanSideClean(unittest.TestCase):
    def test_alice_is_admitted_on_lan(self):
        from profiles.lan import tools as t
        out = _run(t.get_user_access({"user_id": "alice"}))
        self.assertIn("ADMITTED", out)
        self.assertNotIn("BLOCKED", out)
        # points the operator/agent toward the DC for app-layer causes
        self.assertIn("DC", out)

    def test_lan_users_include_alice(self):
        from profiles.lan import tools as t
        out = _run(t.list_users({}))
        self.assertIn("alice", out)


class TestDcSideRootCause(unittest.TestCase):
    def test_alice_denied_on_crm(self):
        from profiles.dc import tools as t
        out = _run(t.dc_check_user_app_access({"user_id": "alice", "app_id": "crm"}))
        self.assertIn("DENIED", out)

    def test_bob_allowed_on_crm(self):
        from profiles.dc import tools as t
        out = _run(t.dc_check_user_app_access({"user_id": "bob", "app_id": "crm"}))
        self.assertIn("ALLOWED", out)

    def test_grant_then_allowed(self):
        from profiles.dc import tools as t
        # grant alice a CRM role, then re-check → now allowed
        _run(t.dc_grant_app_access({"user_id": "alice2", "app_id": "crm",
                                    "role": "sales-rep", "reason": "test"}))
        out = _run(t.dc_check_user_app_access({"user_id": "alice2", "app_id": "crm"}))
        self.assertIn("ALLOWED", out)


class TestHitlGatingClassification(unittest.TestCase):
    """Method 甲: queries read-only, only writes HITL-gated."""

    def test_dc_metadata_hitl_flags(self):
        from profiles.dc.tool_meta import TOOLS
        self.assertFalse(TOOLS["dc_check_user_app_access"]["hitl"])
        self.assertFalse(TOOLS["dc_get_app_acl"]["hitl"])
        self.assertTrue(TOOLS["dc_grant_app_access"]["hitl"])
        self.assertTrue(TOOLS["dc_revoke_app_access"]["hitl"])

    def test_lan_metadata_hitl_flags(self):
        from profiles.lan.tool_meta import TOOLS
        self.assertFalse(TOOLS["get_user_access"]["hitl"])
        self.assertFalse(TOOLS["list_users"]["hitl"])
        self.assertTrue(TOOLS["grant_user_access"]["hitl"])
        self.assertTrue(TOOLS["revoke_user_access"]["hitl"])

    def test_callable_metadata_consistency(self):
        from profiles.lan import PROFILE as LAN
        from profiles.dc import PROFILE as DC
        for p in (LAN, DC):
            self.assertEqual(
                set(p.tool_callables.keys()), set(p.tool_metadata.keys()),
                f"{p.profile_id}: callable/metadata key mismatch",
            )


class TestAccessDiagnosisSkills(unittest.TestCase):
    """The access-diagnosis skills are what steer the agent to permission tools
    instead of fabric/VNI tools (the 2026-05 mis-routing fix). Lock them in."""

    def test_dc_has_app_access_skill_binding_permission_tools(self):
        from profiles.dc import PROFILE as DC
        self.assertIn("dc_app_access_diagnose", DC.skill_defs)
        sk = DC.skill_defs["dc_app_access_diagnose"]
        # must bind the permission tools, not fabric tools
        self.assertIn("dc_check_user_app_access", sk["tool_deps"])
        self.assertIn("dc_grant_app_access", sk["tool_deps"])
        # purpose must signal access/permission (so it out-ranks fabric skills
        # for an access query)
        blob = (sk["purpose"] + sk["description"]).lower()
        self.assertIn("access", blob)
        self.assertIn("permission", blob)

    def test_lan_access_skill_steers_to_delegation(self):
        from profiles.lan import PROFILE as LAN
        self.assertIn("lan_user_access_diagnose", LAN.skill_defs)
        sk = LAN.skill_defs["lan_user_access_diagnose"]
        self.assertIn("get_user_access", sk["tool_deps"])
        # description must tell lan to delegate the app-layer part to the DC
        # and NOT pre-frame as a network/VNI problem
        desc = sk["description"].lower()
        self.assertIn("delegate", desc)
        self.assertTrue("data center" in desc or "dc agent" in desc)

    def test_onboarding_skill_pins_complete_dc_delegate_arguments(self):
        from pathlib import Path

        text = (Path(__file__).parents[1]
                / "profiles/lan/skills/lan-new-employee-onboarding-access/SKILL.md").read_text()
        self.assertIn('target="dc-agent"', text)
        self.assertIn('description="Grant <user_id> <app> application access"', text)
        self.assertIn('self-contained `prompt=', text)
        self.assertIn("never report end-to-end success", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
