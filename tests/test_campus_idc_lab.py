from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dsh_adapter.skills import build_skill_manifest
from network_lab.containerlab import CommandResult, ContainerlabProvider, LabCommandError
from network_lab.manifest import ManifestError, load_manifest
from network_lab.tools import LabToolAdapter, lab_access_metadata
from network_runtime.policies import resolve_contract


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "labs" / "p075-a-campus-idc" / "lab.yaml"
CONFIG = ROOT / "config.campus-idc-lab.yaml"


def run(value):
    return asyncio.run(value)


class AdmissionRunner:
    def __init__(self) -> None:
        self.up = True
        self.route = "10.20.0.0/16 via 10.10.20.1 dev eth1"

    async def run(self, argv, *, cwd=None, timeout=30.0):
        values = tuple(argv)
        inner = values[3:] if values[:2] == ("docker", "exec") else ()
        if inner == ("cat", "/sys/class/net/eth1/operstate"):
            output = "up\n" if self.up else "down\n"
        elif inner == ("ip", "route", "show"):
            output = f"{self.route}\n" if self.route else ""
        else:
            output = ""
        if inner == ("ip", "link", "set", "dev", "eth1", "down"):
            self.up = False
            self.route = ""
        elif inner == ("ip", "link", "set", "dev", "eth1", "up"):
            self.up = True
        elif inner == (
            "ip", "route", "replace", "10.20.0.0/16",
            "via", "10.10.20.1", "dev", "eth1",
        ):
            self.route = "10.20.0.0/16 via 10.10.20.1 dev eth1"
        return CommandResult(values, 0, output, "")


class ApplicationPolicyRunner:
    def __init__(self) -> None:
        self.route: str | None = None

    async def run(self, argv, *, cwd=None, timeout=30.0):
        values = tuple(argv)
        inner = values[3:] if values[:2] == ("docker", "exec") else ()
        output = ""
        if inner == ("ip", "route", "show", "10.10.20.10/32"):
            output = f"{self.route}\n" if self.route else ""
        elif inner == (
            "ip", "route", "replace", "blackhole", "10.10.20.10/32",
        ):
            self.route = "blackhole 10.10.20.10"
        elif inner == (
            "ip", "route", "del", "blackhole", "10.10.20.10/32",
        ):
            self.route = None
        return CommandResult(values, 0, output, "")


class TestCampusIdcManifest(unittest.TestCase):
    def test_access_targets_are_typed_and_bound_to_declared_endpoints(self) -> None:
        manifest = load_manifest(MANIFEST)
        self.assertEqual(manifest.name, "netopyu-campus-idc")
        self.assertEqual(set(manifest.users), {"erin", "bob"})
        self.assertEqual(set(manifest.applications), {"crm", "wiki"})
        self.assertEqual(manifest.users["erin"].endpoint, "erin-client")
        self.assertEqual(manifest.users["erin"].interface, "eth1")
        self.assertEqual(manifest.applications["crm"].address, "10.20.10.20")
        self.assertEqual(manifest.applications["crm"].roles, ("sales-rep",))

    def test_manifest_rejects_access_endpoint_outside_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "topology.clab.yml").write_text("name: test\n", encoding="utf-8")
            (root / "lab.yaml").write_text(
                "schema_version: 1\nname: access-test\ntopology: topology.clab.yml\n"
                "devices: {r1: {node: r1, platform: frr}}\n"
                "endpoints: {client: {node: client}, server: {node: server}}\n"
                "probes: {}\nfault_targets: {}\naccess:\n  users:\n"
                "    erin: {endpoint: invented, interface: eth1, address: 10.0.0.1, prefix_length: 24, gateway: 10.0.0.254, application_prefix: 10.1.0.0/16, vlan: 20}\n"
                "  applications:\n"
                "    crm: {endpoint: server, address: 10.1.0.1, port: 80, path: /, roles: [reader]}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ManifestError, "endpoint is not declared"):
                load_manifest(root / "lab.yaml")

    def test_admission_snapshot_restores_interface_and_every_reviewed_route(self) -> None:
        runner = AdmissionRunner()
        provider = ContainerlabProvider(load_manifest(MANIFEST), runner=runner)
        snapshot = run(provider.user_admission_snapshot("erin"))
        self.assertTrue(snapshot["admitted"])
        self.assertEqual(snapshot["routes"], {"10.20.0.0/16": True})
        run(provider.set_user_admission("erin", admitted=False))
        self.assertFalse(run(provider.user_admission_snapshot("erin"))["admitted"])
        run(provider.restore_user_admission_snapshot("erin", snapshot))
        self.assertEqual(run(provider.user_admission_snapshot("erin")), snapshot)
        runner.route = "10.20.0.0/16 via 10.10.20.254 dev eth1"
        with self.assertRaisesRegex(LabCommandError, "outside the reviewed"):
            run(provider.user_admission_snapshot("erin"))

    def test_application_snapshot_rejects_unknown_route_and_restores_exact_policy(self) -> None:
        runner = ApplicationPolicyRunner()
        provider = ContainerlabProvider(load_manifest(MANIFEST), runner=runner)
        snapshot = run(provider.application_access_snapshot("erin", "crm"))
        self.assertTrue(snapshot["allowed"])
        self.assertIsNone(snapshot["route"])
        run(provider.set_application_access("erin", "crm", allowed=False))
        self.assertFalse(run(provider.application_access_snapshot("erin", "crm"))["allowed"])
        run(provider.restore_application_access_snapshot("erin", "crm", snapshot))
        self.assertEqual(run(provider.application_access_snapshot("erin", "crm")), snapshot)
        runner.route = "10.10.20.10 via 10.20.10.1 dev eth1"
        with self.assertRaisesRegex(LabCommandError, "outside the reviewed"):
            run(provider.application_access_snapshot("erin", "crm"))


class TestCampusIdcProjection(unittest.TestCase):
    def test_profile_projection_does_not_cross_lan_dc_write_boundary(self) -> None:
        manifest = load_manifest(MANIFEST)
        adapter = LabToolAdapter(type("Provider", (), {"manifest": manifest})())
        lan = adapter.callables("lan")
        dc = adapter.callables("dc")
        self.assertIn("grant_user_access", lan)
        self.assertNotIn("dc_grant_app_access", lan)
        # Cross-domain reconciliation may share read-only observed-state
        # probes, while profile-specific legacy mutation tools stay isolated.
        self.assertIn("lab_app_probe", lan)
        self.assertIn("network_get_app_enforcement", lan)
        self.assertIn("dc_grant_app_access", dc)
        self.assertIn("lab_app_probe", dc)
        self.assertIn("network_get_app_enforcement", dc)
        self.assertNotIn("grant_user_access", dc)
        self.assertTrue(lab_access_metadata("lan")["grant_user_access"]["hitl"])
        self.assertTrue(lab_access_metadata("dc")["dc_grant_app_access"]["hitl"])

    def test_pragmatic_lab_contracts_are_source_bound(self) -> None:
        allowed = resolve_contract(
            "grant_user_access", action_type="destructive", requires_approval=True,
            mode="pragmatic", source="network-lab",
        )
        blocked = resolve_contract(
            "grant_user_access", action_type="destructive", requires_approval=True,
            mode="pragmatic", source="pragmatic-device",
        )
        self.assertIsNotNone(allowed)
        self.assertIsNone(blocked)

    def test_pragmatic_lab_loads_only_profile_specific_access_skill(self) -> None:
        environment = {
            "NETOPYU_DSH_BACKEND": "pragmatic",
            "NETOPYU_CONFIG_PATH": str(CONFIG),
        }
        with patch.dict(os.environ, environment, clear=False):
            lan = {item["name"] for item in build_skill_manifest("lan", "pragmatic")["skills"]}
            dc = {item["name"] for item in build_skill_manifest("dc", "pragmatic")["skills"]}
        self.assertIn("lan-new-employee-onboarding-access", lan)
        self.assertNotIn("dc-app-access-diagnose", lan)
        self.assertIn("dc-app-access-diagnose", dc)
        self.assertNotIn("lan-new-employee-onboarding-access", dc)


if __name__ == "__main__":
    unittest.main()
