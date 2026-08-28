import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dsh_adapter.backend import open_backend, resolve_backend_mode
from dsh_adapter.bridge import build_manifest, invoke_tool
from dsh_adapter.scoped_services import recall_memory, search_capabilities
from network_runtime.contracts import ApprovalError


class TestDshManifest(unittest.TestCase):
    def test_network_runtime_imports_in_a_clean_interpreter(self):
        completed = subprocess.run(
            [sys.executable, "-c", "from network_runtime import NetworkRuntime; print(NetworkRuntime.__name__)"],
            cwd=Path(__file__).parents[1], text=True, capture_output=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "NetworkRuntime")

    def test_default_manifest_exports_only_read_only_lan_tools(self):
        manifest = build_manifest("lan")
        names = {tool["name"] for tool in manifest["tools"]}
        self.assertIn("list_devices", names)
        self.assertIn("get_user_access", names)
        self.assertNotIn("restart_service", names)
        self.assertNotIn("grant_user_access", names)
        self.assertTrue(all(not tool["requires_approval"] for tool in manifest["tools"]))

    def test_manifest_compiles_required_and_typed_parameters(self):
        manifest = build_manifest("lan")
        tools = {tool["name"]: tool for tool in manifest["tools"]}
        self.assertTrue(tools["get_user_access"]["parameters"]["user_id"]["required"])
        self.assertEqual(tools["syslog_search"]["parameters"]["lines"]["type"], "integer")

    def test_destructive_manifest_is_explicit_opt_in(self):
        manifest = build_manifest("lan", include_destructive=True)
        tools = {tool["name"]: tool for tool in manifest["tools"]}
        self.assertTrue(tools["restart_service"]["requires_approval"])
        self.assertEqual(tools["restart_service"]["action_type"], "destructive")
        self.assertEqual(tools["restart_service"]["l0_skill_id"], "network.service.restart")
        self.assertEqual(tools["restart_service"]["l0_skill_version"], "1.0.0")
        self.assertTrue(tools["restart_service"]["l0_contract_hash"].startswith("sha256:"))
        self.assertEqual(tools["restart_service"]["intent_kind"], "restart_service")
        self.assertEqual(tools["restart_service"]["execution_boundary"], "domain_effect_runtime")
        l0_skills = {item["skill_id"]: item for item in manifest["l0_skills"]}
        self.assertIn("network.service.restart", l0_skills)
        self.assertEqual(l0_skills["network.service.restart"]["tool_name"], "restart_service")
        self.assertEqual(
            tools["edit_device_config"]["editable_parameters"],
            ["config_lines", "reason"],
        )
        # Stale profile config keys are not exported as editable parameters.
        self.assertEqual(tools["restart_service"]["editable_parameters"], [])


class TestDshInvocation(unittest.TestCase):
    def test_read_only_tool_invokes_existing_profile_callable(self):
        result = asyncio.run(invoke_tool("lan", "get_user_access", {"user_id": "alice"}))
        self.assertIn("alice", result.lower())

    def test_lan_inventory_tag_filter_matches_declared_schema(self):
        result = asyncio.run(invoke_tool(
            "lan", "list_devices", {"type": "switch", "tag": "site-a"},
        ))
        self.assertIn("sw-core-01", result)
        self.assertIn("sw-acc-01", result)
        self.assertNotIn("sw-core-02", result)
        self.assertNotIn("site-b", result)

    def test_device_info_uses_inventory_model_for_requested_device(self):
        result = asyncio.run(invoke_tool(
            "lan", "device_info", {"device_id": "sw-core-01"},
        ))
        self.assertIn("Cisco Catalyst 9500-48Y4C", result)
        self.assertIn("Type:         switch", result)
        self.assertNotIn("Clients (2.4GHz)", result)

    def test_destructive_tool_is_denied_without_durable_hitl(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NETOPYU_DSH_ALLOW_DESTRUCTIVE", None)
            with self.assertRaises(ApprovalError):
                asyncio.run(invoke_tool("lan", "restart_service", {"service": "crm"}))

    def test_common_result_tools_are_available_for_every_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = {
                "NETOPYU_DSH_BACKEND": "mock",
                "NETOPYU_DSH_TOOL_RESULT_STORE": str(Path(directory, "results.sqlite")),
            }
            with patch.dict(os.environ, environment):
                for profile in ("default", "lan", "dc", "wan"):
                    backend = asyncio.run(open_backend(profile))
                    try:
                        self.assertIn("read_stored_result", backend.callables)
                        self.assertIn("process_stored_chunks", backend.callables)
                        self.assertEqual(backend.sources["read_stored_result"], "netopyu-runtime")
                    finally:
                        asyncio.run(backend.close())

    def test_large_result_survives_across_bridge_process_lifecycles(self):
        from runtime import ToolResultStore

        with tempfile.TemporaryDirectory() as directory:
            environment = {
                "NETOPYU_DSH_BACKEND": "mock",
                "NETOPYU_DSH_TOOL_RESULT_STORE": str(Path(directory, "results.sqlite")),
            }
            with patch.dict(os.environ, environment), patch.object(
                ToolResultStore, "MAX_INLINE_CHARS", 1,
            ):
                stored = asyncio.run(invoke_tool(
                    "lan", "get_user_access", {"user_id": "alice"},
                ))
                self.assertTrue(stored.startswith("[STORED:get_user_access:"))
                page = asyncio.run(invoke_tool(
                    "lan", "read_stored_result", {"ref_id": stored, "length": 8000},
                ))
            self.assertIn("alice", page.lower())
            self.assertIn("Has more: False", page)


class TestDshScopedServices(unittest.TestCase):
    def test_missing_memory_store_degrades_without_creating_it(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory, "missing")
            result = asyncio.run(recall_memory(
                memory_dir=str(missing), operator_id="alice", session_id="s1",
                query="anything",
            ))
            self.assertFalse(result["available"])
            self.assertFalse(missing.exists())

    def test_memory_recall_is_operator_and_session_scoped(self):
        from agent_memory import MemoryManager

        with tempfile.TemporaryDirectory() as directory:
            manager = MemoryManager(data_dir=directory, enable_user_model=False)
            manager.remember("alice", "s1", "alpha private router note")
            manager.remember("bob", "s1", "beta other operator note")
            manager.remember("alice", "s2", "gamma other session note")
            manager.close()
            result = asyncio.run(recall_memory(
                memory_dir=directory, operator_id="alice", session_id="s1",
                query="current router note", recent_turns=4,
            ))
        self.assertTrue(result["available"])
        self.assertIn("alpha private router note", result["prompt_context"])
        self.assertNotIn("beta other operator note", result["prompt_context"])
        self.assertNotIn("gamma other session note", result["prompt_context"])

    def test_capability_search_reuses_active_profile_catalog(self):
        with patch.dict(os.environ, {"NETOPYU_DSH_BACKEND": "mock"}):
            result = asyncio.run(search_capabilities(
                profile_id="lan", query="restart production service",
                top_k=5, kinds=["tool"],
            ))
        self.assertEqual(result["backend"], "bm25")
        self.assertTrue(any(match["id"] == "restart_service" for match in result["matches"]))
        self.assertTrue(all(match["kind"] == "tool" for match in result["matches"]))

    def test_capability_search_cannot_surface_tools_hidden_by_dsh(self):
        with patch.dict(os.environ, {"NETOPYU_DSH_BACKEND": "mock"}):
            result = asyncio.run(search_capabilities(
                profile_id="lan", query="restart production service",
                top_k=20, kinds=["tool"], allowed_tool_names=["service_health"],
            ))
        self.assertEqual([match["id"] for match in result["matches"]], ["service_health"])


class TestPragmaticBackend(unittest.TestCase):
    def test_manifest_uses_real_registry_without_mock_fallback(self):
        with patch.dict(os.environ, {"NETOPYU_DSH_BACKEND": "pragmatic"}):
            manifest = build_manifest("lan", include_destructive=True)
        names = {tool["name"] for tool in manifest["tools"]}
        self.assertEqual(manifest["backend"]["mode"], "pragmatic")
        self.assertFalse(manifest["backend"]["ready"])
        self.assertIn("get_device_status", names)
        self.assertIn("edit_device_config", names)
        self.assertNotIn("get_user_access", names)
        self.assertTrue(all(tool["source"] != "profile-mock" for tool in manifest["tools"]))

    def test_empty_real_inventory_is_diagnostic_not_simulated(self):
        with patch.dict(os.environ, {"NETOPYU_DSH_BACKEND": "pragmatic"}):
            result = asyncio.run(invoke_tool("lan", "list_devices", {}))
        self.assertIn("No devices found", result)
        self.assertIn("pragmatic.device_inventory", result)

    def test_configured_inventory_is_registered_as_real_source(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory, "config.yaml")
            config_path.write_text(
                """
mode: pragmatic
pragmatic:
  device_inventory:
    - id: edge-lab-1
      device_type: cisco_ios
      host: 192.0.2.10
      username: netops
      password: test-only
      label: Lab Edge
      tags: [lab]
  mcp_servers: []
""",
                encoding="utf-8",
            )
            environment = {
                "NETOPYU_DSH_BACKEND": "pragmatic",
                "NETOPYU_CONFIG_PATH": str(config_path),
            }
            with patch.dict(os.environ, environment):
                manifest = build_manifest("lan")
                result = asyncio.run(invoke_tool("lan", "list_devices", {}))
        self.assertTrue(manifest["backend"]["ready"])
        self.assertEqual(manifest["backend"]["device_count"], 1)
        self.assertIn("edge-lab-1", result)
        self.assertIn("192.0.2.10", result)

    def test_pragmatic_write_still_requires_durable_hitl(self):
        with patch.dict(os.environ, {"NETOPYU_DSH_BACKEND": "pragmatic"}, clear=False):
            os.environ.pop("NETOPYU_DSH_ALLOW_DESTRUCTIVE", None)
            with self.assertRaises(ApprovalError):
                asyncio.run(invoke_tool("lan", "edit_device_config", {
                    "device_id": "sw-1", "changes": {}, "reason": "test",
                }))

    def test_unknown_backend_is_rejected(self):
        with patch.dict(os.environ, {"NETOPYU_DSH_BACKEND": "typo"}):
            with self.assertRaises(ValueError):
                resolve_backend_mode()

    def test_pragmatic_backend_rejects_explicit_mock_mcp_transport(self):
        environment = {
            "NETOPYU_DSH_BACKEND": "pragmatic",
            "MCP_CONFIG_JSON": json.dumps({"fake": {"transport": "mock"}}),
        }
        with patch.dict(os.environ, environment):
            with self.assertRaisesRegex(ValueError, "refuses MCP transport=mock"):
                build_manifest("lan")


if __name__ == "__main__":
    unittest.main()
