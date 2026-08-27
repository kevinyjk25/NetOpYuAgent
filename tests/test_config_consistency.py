from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import load


class TestDshConfig(unittest.TestCase):
    def test_minimal_repository_config_loads(self):
        config = load("config.yaml")
        self.assertEqual(config.mode, "mock")
        self.assertEqual(config.agent.profile, "lan")
        self.assertIn("restart_service", config.tools.editable_hitl_tools)
        self.assertEqual(config.pragmatic.device_inventory, [])

    def test_dsh_environment_overrides(self):
        with patch.dict(os.environ, {
            "NETOPYU_DSH_BACKEND": "pragmatic",
            "NETOPYU_PROFILE": "dc",
            "NETOPYU_DSH_A2A_PEERS": "http://one:1/,http://two:2",
            "MCP_CONFIG_JSON": "{}",
        }, clear=False):
            config = load("config.yaml")
        self.assertEqual(config.mode, "pragmatic")
        self.assertEqual(config.agent.profile, "dc")
        self.assertEqual(config.agent.peer_urls, ["http://one:1", "http://two:2"])
        self.assertEqual(config.tools.mcp.config_json, "{}")

    def test_pragmatic_device_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "config.yaml")
            path.write_text("""
mode: pragmatic
pragmatic:
  device_inventory:
    - id: sw-1
      device_type: cisco_ios
      host: 127.0.0.1
      username: local
      password: simulated
""", encoding="utf-8")
            config = load(path)
        self.assertEqual(config.pragmatic.device_inventory[0].device_type, "cisco_ios")
        self.assertEqual(config.pragmatic.device_inventory[0].port, 22)
