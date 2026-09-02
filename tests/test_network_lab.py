from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dsh_adapter.backend import BackendSession, open_backend
from dsh_adapter.skills import build_skill_manifest
from network_lab.containerlab import (
    CommandResult,
    ContainerlabProvider,
    LabCommandError,
    normalize_frr_config,
)
from network_lab.manifest import ManifestError, load_manifest
from network_runtime import NetworkRuntime
from network_runtime.contracts import PlanState
from network_runtime.l0_skills import REGISTRY as L0_SKILLS
from runtime import ToolResultStore


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "labs" / "p075-a-frr" / "lab.yaml"


def run(value):
    return asyncio.run(value)


class RecordingRunner:
    def __init__(self, outputs: dict[tuple[str, ...], CommandResult] | None = None) -> None:
        self.outputs = outputs or {}
        self.calls: list[tuple[str, ...]] = []

    async def run(self, argv, *, cwd=None, timeout=30.0):
        key = tuple(argv)
        self.calls.append(key)
        return self.outputs.get(key, CommandResult(key, 0, "ok\n", ""))


class TestLabManifestAndProvider(unittest.TestCase):
    def test_reviewed_manifest_is_strict_and_complete(self) -> None:
        manifest = load_manifest(MANIFEST)
        self.assertEqual(manifest.name, "netopyu-p075a")
        self.assertEqual(set(manifest.devices), {"branch-r1", "dc-r1"})
        self.assertEqual(set(manifest.probes), {"branch-to-dc", "dc-to-branch"})
        self.assertEqual(
            manifest.container_name("branch-r1"), "clab-netopyu-p075a-branch-r1",
        )

    def test_manifest_rejects_topology_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lab.yaml").write_text(
                "schema_version: 1\nname: safe-lab\ntopology: ../outside.yml\n"
                "devices: {r1: {node: r1, platform: frr}}\n"
                "endpoints: {}\nprobes: {}\nfault_targets: {}\n",
                encoding="utf-8",
            )
            (root.parent / "outside.yml").write_text("name: outside\n", encoding="utf-8")
            with self.assertRaises(ManifestError):
                load_manifest(root / "lab.yaml")
            (root.parent / "outside.yml").unlink(missing_ok=True)

    def test_provider_never_uses_a_shell_and_rejects_unreviewed_commands(self) -> None:
        runner = RecordingRunner()
        provider = ContainerlabProvider(load_manifest(MANIFEST), runner=runner)
        output = run(provider.show("branch-r1", "show ip route"))
        self.assertEqual(output.strip(), "ok")
        self.assertEqual(runner.calls[-1], (
            "docker", "exec", "clab-netopyu-p075a-branch-r1",
            "vtysh", "-c", "show ip route",
        ))
        for command in (
            "show ip route; rm -rf /tmp/x",
            "bash -c id",
            "configure terminal",
            "show ip route\nreload",
        ):
            with self.assertRaises(LabCommandError):
                run(provider.show("branch-r1", command))
        self.assertEqual(len(runner.calls), 1)

    def test_config_allowlist_and_normalization_are_deterministic(self) -> None:
        allowed = ContainerlabProvider.validate_config_lines((
            "interface eth2", "ip ospf cost 10", "description managed-by-netopyu",
        ))
        self.assertEqual(allowed[1], "ip ospf cost 10")
        for line in (
            "reload", "do show run", "shell /bin/sh", "interface eth0",
            "no interface eth2", "no router ospf", "ip address 999.1.1.1/24",
        ):
            with self.assertRaises(LabCommandError):
                ContainerlabProvider.validate_config_lines((
                    "interface eth2", line,
                ) if line.startswith("ip address") else (line,))
        with self.assertRaises(LabCommandError):
            ContainerlabProvider.validate_config_lines(("exit",))
        first = normalize_frr_config("Building configuration...\nCurrent configuration:\ninterface eth2\n!\n")
        second = normalize_frr_config("interface eth2\n!\n")
        self.assertEqual(first, second)


class TestLabRuntimeIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.journal = Path(self.temp.name) / "runtime.sqlite"

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def backend_factory(*, fail_verification: bool):
        state = {
            "config": "! Configuration\ninterface eth2\n ip ospf cost 20\n!\n",
            "snapshot": None,
            "restore_calls": 0,
        }

        async def factory(profile_id: str) -> BackendSession:
            async def get_config(args):
                return state["config"]

            async def edit(args):
                state["snapshot"] = state["config"]
                cost = 30 if fail_verification else 10
                state["config"] = f"! Configuration\ninterface eth2\n ip ospf cost {cost}\n!\n"
                return "Config applied to lab device branch-r1"

            async def restore(args):
                state["restore_calls"] += 1
                state["config"] = state["snapshot"]
                return "Restored exact execution-session snapshot for branch-r1"

            async def lab_probe(args):
                return (
                    '{"ok": true, "probe_id": "branch-to-dc", '
                    '"transmitted": 3, "received": 3}'
                )

            metadata = {
                "edit_device_config": {
                    "description": "lab write",
                    "parameters": {
                        "device_id": "id", "config_lines": "lines", "reason": "reason",
                        "verification_probe_id": "probe",
                    },
                    "required": ["device_id", "config_lines", "reason"],
                    "hitl": True,
                    "action_type": "destructive",
                },
                "get_device_config": {
                    "description": "read config",
                    "parameters": {"device_id": "id", "section": "section"},
                    "required": ["device_id"],
                    "hitl": False,
                    "action_type": "read_only",
                },
            }
            callables = {
                "edit_device_config": edit,
                "get_device_config": get_config,
                "restore_device_config": restore,
                "lab_probe": lab_probe,
            }
            return BackendSession(
                mode="pragmatic",
                profile_id=profile_id,
                callables=callables,
                metadata=metadata,
                sources={name: "network-lab" for name in callables},
                report={"ready": True, "lab": True},
                _tool_store=ToolResultStore(":memory:"),
            )

        return factory, state

    def _execute(self, *, fail_verification: bool):
        factory, state = self.backend_factory(fail_verification=fail_verification)
        runtime = NetworkRuntime(self.journal, backend_factory=factory)
        contract = L0_SKILLS.for_tool("lan", "edit_device_config")
        prepared = run(runtime.prepare(
            "lan",
            "edit_device_config",
            {
                "device_id": "branch-r1",
                "config_lines": ["interface eth2", "ip ospf cost 10"],
                "reason": "P0.75-A integration test",
                "verification_probe_id": "branch-to-dc",
            },
            l0_skill_id=contract.skill_id,
        ))
        self.assertEqual(prepared["status"], "plan_ready", prepared)
        plan = prepared["plan"]
        outcome = run(runtime.execute(
            plan_id=plan["plan_id"],
            plan_hash=plan["plan_hash"],
            execution_nonce=prepared["execution_nonce"],
            approval_request_id="lab-test-approval",
            approval_actor="unit-test-operator",
            allow_destructive=True,
        ))
        return runtime, state, plan, outcome

    def test_l0_lab_write_is_verified_by_fresh_configuration_read(self) -> None:
        runtime, state, plan, outcome = self._execute(fail_verification=False)
        self.assertEqual(outcome.state, PlanState.VERIFIED_SUCCESS)
        self.assertEqual(state["restore_calls"], 0)
        self.assertIn("traffic_postcondition", {item.evidence_type for item in outcome.evidence})
        self.assertTrue(runtime.audit(plan["plan_id"])["ok"])

    def test_pragmatic_backend_projects_lab_tools_and_l1_workflow_only_when_enabled(self) -> None:
        config_path = Path(self.temp.name) / "lab-config.yaml"
        config_path.write_text(
            "mode: pragmatic\n"
            "agent: {profile: lan}\n"
            "tools:\n"
            "  schema_validation_enabled: true\n"
            "  mcp: {config_json: ''}\n"
            "  openapi: {spec_url: '', base_url: ''}\n"
            "  editable_hitl_tools: {edit_device_config: [config_lines, reason]}\n"
            "pragmatic:\n"
            "  device_inventory: []\n"
            "  mcp_servers: []\n"
            "  lab:\n"
            "    enabled: true\n"
            "    provider: containerlab\n"
            f"    manifest: {MANIFEST}\n",
            encoding="utf-8",
        )
        environment = {
            "NETOPYU_DSH_BACKEND": "pragmatic",
            "NETOPYU_CONFIG_PATH": str(config_path),
            "NETOPYU_TOOL_RESULT_STORE": str(Path(self.temp.name) / "results.sqlite"),
        }
        with patch.dict(os.environ, environment, clear=False):
            backend = run(open_backend("lan"))
            try:
                self.assertTrue(backend.report["ready"])
                self.assertTrue(backend.report["lab_enabled"])
                self.assertEqual(backend.sources["edit_device_config"], "network-lab")
                self.assertIn("lab_probe", backend.callables)
                self.assertNotIn("restore_device_config", backend.metadata)
            finally:
                run(backend.close())
            skills = build_skill_manifest("lan", "pragmatic")
            selected = {item["name"]: item for item in skills["skills"]}
            self.assertIn("lab-ospf-path-remediation", selected)
            self.assertNotIn("lan-new-employee-onboarding-access", selected)
            workflow = selected["lab-ospf-path-remediation"]["network_workflow"]
            self.assertEqual(workflow["version"], "1.0.0")
            self.assertIn("edit_device_config", workflow["write_requirements"])

    def test_failed_postcondition_restores_exact_snapshot_and_audits_it(self) -> None:
        runtime, state, plan, outcome = self._execute(fail_verification=True)
        self.assertEqual(outcome.state, PlanState.ROLLBACK_VERIFIED)
        self.assertEqual(state["restore_calls"], 1)
        self.assertIn("ip ospf cost 20", state["config"])
        evidence_types = {item.evidence_type for item in outcome.evidence}
        self.assertIn("rollback", evidence_types)
        self.assertIn("rollback_postcondition", evidence_types)
        self.assertTrue(runtime.audit(plan["plan_id"])["ok"])


if __name__ == "__main__":
    unittest.main()
