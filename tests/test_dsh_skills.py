from __future__ import annotations

import unittest

from dsh_adapter.skills import build_skill_manifest


class TestDshSkillProjection(unittest.TestCase):
    def test_every_profile_projects_exact_legacy_catalog(self):
        expected = {"default": 1, "lan": 12, "dc": 5, "wan": 1}
        for profile, count in expected.items():
            with self.subTest(profile=profile):
                manifest = build_skill_manifest(profile, "mock")
                self.assertEqual(len(manifest["skills"]), count)
                self.assertEqual(len({skill["name"] for skill in manifest["skills"]}), count)
                for skill in manifest["skills"]:
                    self.assertTrue(skill["content"])
                    self.assertTrue(skill["path"].endswith("SKILL.md"))

    def test_dc_profile_contains_only_common_and_dc_skills(self):
        names = {skill["name"] for skill in build_skill_manifest("dc", "mock")["skills"]}
        self.assertEqual(names, {
            "read-stored-result", "dc-app-access-diagnose", "dc-evpn-troubleshoot",
            "dc-lb-health-check", "dc-path-troubleshoot",
        })
