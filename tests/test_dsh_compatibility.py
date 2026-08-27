from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class TestDshCompatibilityContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads(
            (ROOT / "dsh-plugin-netopyu" / "compatibility.json").read_text(encoding="utf-8")
        )
        cls.package = json.loads(
            (ROOT / "dsh-plugin-netopyu" / "package.json").read_text(encoding="utf-8")
        )
        cls.launcher = (ROOT / "scripts" / "netopyu-dsh").read_text(encoding="utf-8")

    def test_launcher_version_is_the_tested_default(self):
        match = re.search(r'^DSH_VERSION="([^"]+)"$', self.launcher, re.MULTILINE)
        self.assertIsNotNone(match)
        version = match.group(1)
        self.assertEqual(version, self.contract["dsh"]["default"])
        self.assertIn(version, self.contract["dsh"]["tested"])

    def test_node_engine_matches_the_contract(self):
        self.assertEqual(self.package["engines"]["node"], self.contract["node"]["engines"])
        self.assertIn("22.19.0", self.contract["node"]["tested"])
        self.assertIn("24.x", self.contract["node"]["tested"])

    def test_python_matrix_is_explicit(self):
        self.assertEqual(self.contract["python"]["tested"], ["3.11", "3.12"])

    def test_plugin_is_private_and_has_a_stable_entrypoint(self):
        self.assertTrue(self.package["private"])
        self.assertEqual(self.package["main"], "./src/index.js")


if __name__ == "__main__":
    unittest.main()
