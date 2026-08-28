from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from dsh_adapter.backend import BackendSession
from dsh_adapter.skills import build_skill_manifest
from network_lab.containerlab import CommandResult, ContainerlabProvider, LabCommandError
from network_lab.manifest import ManifestError, load_manifest
from network_runtime import NetworkRuntime
from network_runtime.contracts import PlanState
from network_runtime.l0_skills import REGISTRY as L0_SKILLS
from runtime import ToolResultStore


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "labs" / "p075-c-evpn-vxlan" / "lab.yaml"
CONFIG = ROOT / "config.evpn-vxlan-lab.yaml"


def run(value):
    return asyncio.run(value)


class FabricRunner:
    """Small stateful runner that proves the provider emits only reviewed argv."""

    def __init__(self) -> None:
        self.vlan = 10
        self.bridge = "br10"
        self.calls: list[tuple[str, ...]] = []

    async def run(self, argv, *, cwd=None, timeout=30.0):
        values = tuple(argv)
        self.calls.append(values)
        inner = values[3:] if values[:2] == ("docker", "exec") else ()
        if inner == ("bridge", "-j", "vlan", "show", "dev", "eth3"):
            payload = [{
                "ifname": "eth3",
                "vlans": [{"vlan": self.vlan, "flags": ["PVID", "Egress Untagged"]}],
            }]
            return CommandResult(values, 0, json.dumps(payload), "")
        if inner == ("ip", "-j", "link", "show", "dev", "eth3"):
            return CommandResult(values, 0, json.dumps([
                {"ifname": "eth3", "master": self.bridge},
            ]), "")
        if inner[:7] == ("ip", "link", "set", "dev", "eth3", "master", "br20"):
            self.bridge = "br20"
        elif inner[:7] == ("ip", "link", "set", "dev", "eth3", "master", "br10"):
            self.bridge = "br10"
        elif inner[:7] == ("bridge", "vlan", "add", "dev", "eth3", "vid", "20"):
            self.vlan = 20
        elif inner[:7] == ("bridge", "vlan", "add", "dev", "eth3", "vid", "10"):
            self.vlan = 10
        return CommandResult(values, 0, "", "")


class FabricManifestAndProviderTests(unittest.TestCase):
    def test_manifest_declares_real_l2_evpn_contract_without_l3_claim(self) -> None:
        manifest = load_manifest(MANIFEST)
        self.assertIsNotNone(manifest.fabric)
        fabric = manifest.fabric
        self.assertEqual(fabric.mode, "evpn-vxlan-l2")
        self.assertEqual(fabric.route_reflectors, ("spine-1", "spine-2"))
        self.assertEqual(fabric.vteps, ("leaf-1", "leaf-2"))
        self.assertEqual(set(fabric.vlans), {10, 20})
        self.assertEqual(fabric.vlans[10].l2vni, 10010)
        self.assertEqual(fabric.vlans[20].l2vni, 10020)
        self.assertEqual(fabric.attachments["host-a1"].interface, "eth3")
        self.assertEqual(fabric.attachments["trunk-1"].mode, "trunk")

    def test_manifest_rejects_unqualified_l3vpn_and_single_vlan_trunk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "topology.clab.yml").write_text(
                (MANIFEST.parent / "topology.clab.yml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            payload = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
            payload["fabric"]["mode"] = "evpn-vxlan-l2-l3"
            path = root / "lab.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ManifestError, "L3VPN is not qualified"):
                load_manifest(path)
            payload["fabric"]["mode"] = "evpn-vxlan-l2"
            payload["fabric"]["attachments"]["trunk-1"]["vlans"] = [10]
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ManifestError, "trunk attachments require at least two"):
                load_manifest(path)

    def test_access_vlan_provider_uses_fixed_argv_and_restores_exact_state(self) -> None:
        runner = FabricRunner()
        provider = ContainerlabProvider(load_manifest(MANIFEST), runner=runner)
        changed = json.loads(run(provider.set_fabric_access_vlan("leaf-1", "eth3", 20)))
        self.assertEqual(changed["current_vlan"], 20)
        self.assertEqual((runner.vlan, runner.bridge), (20, "br20"))
        restored = json.loads(run(provider.restore_fabric_access_vlan("leaf-1", "eth3")))
        self.assertEqual(restored["restored_vlan"], 10)
        self.assertEqual((runner.vlan, runner.bridge), (10, "br10"))
        self.assertIn((
            "docker", "exec", "clab-netopyu-evpn-vxlan-leaf-1",
            "bridge", "vlan", "add", "dev", "eth3", "vid", "20", "pvid", "untagged",
        ), runner.calls)
        self.assertTrue(all(call[:2] == ("docker", "exec") for call in runner.calls))
        with self.assertRaises(LabCommandError):
            run(provider.set_fabric_access_vlan("leaf-1", "eth5", 20))
        with self.assertRaises(LabCommandError):
            run(provider.set_fabric_access_vlan("leaf-1", "eth3", 999))


class FabricNetworkRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.journal = Path(self.temp.name) / "runtime.sqlite"
        self.environment = patch.dict(os.environ, {
            "NETOPYU_DSH_BACKEND": "pragmatic",
            "NETOPYU_CONFIG_PATH": str(CONFIG),
            "NETOPYU_TOOL_RESULT_STORE": str(Path(self.temp.name) / "results.sqlite"),
        }, clear=False)
        self.environment.start()

    def tearDown(self) -> None:
        self.environment.stop()
        self.temp.cleanup()

    @staticmethod
    def backend_factory(*, probe_ok: bool):
        state = {"vlan": 10, "snapshot": None, "restore_calls": 0}

        def observed() -> str:
            vlan = int(state["vlan"])
            return json.dumps({
                "ok": True,
                "device_id": "leaf-1",
                "interface": "eth3",
                "mode": "access",
                "current_vlan": vlan,
                "bridge": f"br{vlan}",
                "vlans": [{"vlan_id": vlan, "pvid": True, "untagged": True}],
            }, sort_keys=True)

        async def factory(profile_id: str) -> BackendSession:
            async def get_access_vlan(args):
                return observed()

            async def set_access_vlan(args):
                state["snapshot"] = state["vlan"]
                state["vlan"] = int(args["vlan_id"])
                return json.dumps({"ok": True, "current_vlan": state["vlan"]})

            async def restore_access_vlan(args):
                state["restore_calls"] += 1
                state["vlan"] = state["snapshot"]
                return json.dumps({"ok": True, "restored_vlan": state["vlan"]})

            async def probe(args):
                return json.dumps({
                    "ok": probe_ok,
                    "probe_id": args["probe_id"],
                    "transmitted": 3,
                    "received": 3 if probe_ok else 0,
                })

            metadata = {
                "fabric_set_access_vlan": {
                    "description": "typed fabric write",
                    "parameters": {
                        "device_id": "device", "interface": "access interface",
                        "vlan_id": {"type": "integer"}, "reason": "audit reason",
                        "verification_probe_id": "manifest probe",
                    },
                    "required": ["device_id", "interface", "vlan_id", "reason"],
                    "hitl": True,
                    "action_type": "destructive",
                },
                "lab_get_access_vlan": {
                    "description": "typed fabric read",
                    "parameters": {"device_id": "device", "interface": "interface"},
                    "required": ["device_id", "interface"],
                    "hitl": False,
                    "action_type": "read_only",
                },
            }
            callables = {
                "fabric_set_access_vlan": set_access_vlan,
                "fabric_restore_access_vlan": restore_access_vlan,
                "lab_get_access_vlan": get_access_vlan,
                "lab_probe": probe,
            }
            return BackendSession(
                mode="pragmatic", profile_id=profile_id,
                callables=callables, metadata=metadata,
                sources={name: "network-lab" for name in callables},
                report={"ready": True}, _tool_store=ToolResultStore(":memory:"),
            )

        return factory, state

    def execute(self, *, probe_ok: bool):
        factory, state = self.backend_factory(probe_ok=probe_ok)
        runtime = NetworkRuntime(self.journal, backend_factory=factory)
        contract = L0_SKILLS.for_tool("dc", "fabric_set_access_vlan")
        prepared = run(runtime.prepare(
            "dc", "fabric_set_access_vlan", {
                "device_id": "leaf-1", "interface": "eth3", "vlan_id": 20,
                "reason": "runtime integration test",
                "verification_probe_id": "tenant-a-l2vpn",
            }, l0_skill_id=contract.skill_id,
        ))
        self.assertEqual(prepared["status"], "plan_ready", prepared)
        plan = prepared["plan"]
        outcome = run(runtime.execute(
            plan_id=plan["plan_id"], plan_hash=plan["plan_hash"],
            execution_nonce=prepared["execution_nonce"],
            approval_request_id="fabric-test-approval",
            approval_actor="unit-test-operator", allow_destructive=True,
        ))
        return runtime, state, plan, outcome

    def test_runtime_verifies_real_access_vlan_postcondition(self) -> None:
        runtime, state, plan, outcome = self.execute(probe_ok=True)
        self.assertEqual(outcome.state, PlanState.VERIFIED_SUCCESS)
        self.assertEqual(state["vlan"], 20)
        self.assertEqual(state["restore_calls"], 0)
        self.assertTrue(runtime.audit(plan["plan_id"])["ok"])

    def test_failed_traffic_postcondition_restores_exact_vlan_snapshot(self) -> None:
        runtime, state, plan, outcome = self.execute(probe_ok=False)
        self.assertEqual(outcome.state, PlanState.ROLLBACK_VERIFIED, outcome.to_dict())
        self.assertEqual(state["vlan"], 10)
        self.assertEqual(state["restore_calls"], 1)
        self.assertIn("rollback_postcondition", {item.evidence_type for item in outcome.evidence})
        self.assertTrue(runtime.audit(plan["plan_id"])["ok"])

    def test_invalid_vlan_and_missing_reason_fail_before_write(self) -> None:
        factory, state = self.backend_factory(probe_ok=True)
        runtime = NetworkRuntime(self.journal, backend_factory=factory)
        contract = L0_SKILLS.for_tool("dc", "fabric_set_access_vlan")
        invalid = run(runtime.prepare(
            "dc", "fabric_set_access_vlan", {
                "device_id": "leaf-1", "interface": "eth3", "vlan_id": 4095,
                "reason": "invalid boundary test",
            }, l0_skill_id=contract.skill_id,
        ))
        self.assertEqual(invalid["status"], "rejected")
        missing = run(runtime.prepare(
            "dc", "fabric_set_access_vlan", {
                "device_id": "leaf-1", "interface": "eth3", "vlan_id": 20,
            }, l0_skill_id=contract.skill_id,
        ))
        self.assertEqual(missing["status"], "clarification_required")
        self.assertIn("reason", missing["missing"])
        self.assertEqual(state["vlan"], 10)

    def test_dsh_projects_fabric_skills_and_reviewed_workflow(self) -> None:
        environment = {
            "NETOPYU_DSH_BACKEND": "pragmatic",
            "NETOPYU_CONFIG_PATH": str(CONFIG),
            "NETOPYU_TOOL_RESULT_STORE": str(Path(self.temp.name) / "results.sqlite"),
        }
        with patch.dict(os.environ, environment, clear=False):
            manifest = build_skill_manifest("dc", "pragmatic")
        selected = {item["name"]: item for item in manifest["skills"]}
        self.assertIn("lab-evpn-vxlan-operations", selected)
        self.assertIn("lab-fabric-access-vlan-change", selected)
        workflow = selected["lab-fabric-access-vlan-change"]["network_workflow"]
        self.assertIn("fabric_set_access_vlan", workflow["write_requirements"])


if __name__ == "__main__":
    unittest.main()
