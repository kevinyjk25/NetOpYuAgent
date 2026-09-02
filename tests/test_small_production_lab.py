from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from dsh_adapter.skills import build_skill_manifest
from network_lab.containerlab import CommandResult, ContainerlabProvider
from network_lab.cli import _established_bgp_neighbors
from network_lab.manifest import ManifestError, load_manifest
from network_lab.tools import LabToolAdapter, lab_tool_metadata


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "labs" / "p075-b-small-production" / "lab.yaml"
CONFIG = ROOT / "config.small-production-lab.yaml"


def run(value):
    return asyncio.run(value)


class PathRunner:
    def __init__(self, *, unknown_hop: bool = False) -> None:
        self.unknown_hop = unknown_hop
        self.calls: list[tuple[str, ...]] = []

    async def run(self, argv, *, cwd=None, timeout=30.0):
        values = tuple(argv)
        self.calls.append(values)
        if "traceroute" in values:
            middle = "203.0.113.9" if self.unknown_hop else "10.0.0.2"
            output = (
                "traceroute to 10.20.10.20 (10.20.10.20), 16 hops max\n"
                " 1  10.10.20.1  0.010 ms\n"
                f" 2  {middle}  0.020 ms\n"
                " 3  10.0.0.38  0.030 ms\n"
                " 4  10.20.10.20  0.040 ms\n"
            )
            return CommandResult(values, 0, output, "")
        if values[-1:] == ("/sys/class/net/eth1/operstate",):
            return CommandResult(values, 0, "up\n", "")
        if values[-4:] == ("ip", "route", "show", "10.10.20.10/32"):
            return CommandResult(values, 0, "", "")
        return CommandResult(values, 1, "", f"unexpected test command: {values}")


class TestSmallProductionManifest(unittest.TestCase):
    def test_inventory_topology_and_security_baseline_are_complete(self) -> None:
        manifest = load_manifest(MANIFEST)
        topology = yaml.safe_load(manifest.topology_path.read_text(encoding="utf-8"))
        topology_nodes = set(topology["topology"]["nodes"])

        self.assertEqual(manifest.name, "netopyu-small-production")
        self.assertEqual(len(manifest.devices), 10)
        self.assertEqual(len(manifest.endpoints), 10)
        self.assertEqual(len(manifest.nodes), 20)
        self.assertEqual(topology_nodes, set(manifest.nodes))
        self.assertEqual(set(manifest.users), {"erin", "bob", "carol", "guest"})
        self.assertEqual(
            set(manifest.applications), {"crm", "wiki", "monitoring", "portal"},
        )
        self.assertFalse(manifest.probes["guest-to-crm"].expected)
        self.assertTrue(manifest.probes["guest-to-portal"].expected)
        self.assertFalse(manifest.probes["internet-to-crm"].expected)
        self.assertTrue(manifest.probes["internet-to-portal"].expected)
        self.assertIn("primary-internet-uplink", manifest.fault_targets)
        self.assertIn("crm-app-egress", manifest.fault_targets)
        self.assertEqual(len(manifest.links), 26)
        self.assertEqual(len(manifest.address_index), 52)
        self.assertEqual(
            manifest.address_index["10.0.0.45"],
            ("campus-core-1", "eth7", "core1-dmz"),
        )
        self.assertEqual(manifest.devices["security-edge-1"].zone, "enterprise-edge")
        self.assertEqual(manifest.endpoints["crm-server"].role, "application-server")
        self.assertEqual(
            manifest.simulation.security_edge,
            "routed-wan-edge-no-stateful-firewall",
        )

    def test_user_routes_and_control_plane_expectations_are_typed(self) -> None:
        manifest = load_manifest(MANIFEST)
        self.assertEqual(
            manifest.users["erin"].route_prefixes,
            ("10.0.0.0/8", "198.51.100.0/24"),
        )
        self.assertEqual(manifest.devices["campus-core-1"].expected_ospf_neighbors, 7)
        self.assertEqual(manifest.devices["security-edge-1"].expected_bgp_neighbors, 1)
        self.assertEqual(manifest.devices["isp-1"].expected_bgp_neighbors, 2)

    def test_pragmatic_projection_includes_lan_and_dc_lab_skills(self) -> None:
        environment = {
            "NETOPYU_DSH_BACKEND": "pragmatic",
            "NETOPYU_CONFIG_PATH": str(CONFIG),
        }
        with patch.dict(os.environ, environment, clear=False):
            lan = {item["name"] for item in build_skill_manifest("lan", "pragmatic")["skills"]}
            dc = {item["name"] for item in build_skill_manifest("dc", "pragmatic")["skills"]}
        self.assertIn("lan-new-employee-onboarding-access", lan)
        self.assertIn("lab-ospf-path-remediation", lan)
        self.assertIn("dc-app-access-diagnose", dc)
        self.assertIn("lab-deterministic-path-query", lan)
        self.assertIn("lab-deterministic-path-query", dc)

    def test_typed_links_must_exactly_match_containerlab_wiring(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
            raw["topology"] = "topology.clab.yml"
            raw["links"].pop("core1-dmz")
            (root / "lab.yaml").write_text(
                yaml.safe_dump(raw, sort_keys=False), encoding="utf-8",
            )
            (root / "topology.clab.yml").write_text(
                (MANIFEST.parent / "topology.clab.yml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ManifestError, "do not exactly match"):
                load_manifest(root / "lab.yaml")


class TestDeterministicTopologyAndPath(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = load_manifest(MANIFEST)

    def test_graph_and_endpoint_are_typed_not_inferred(self) -> None:
        provider = ContainerlabProvider(self.manifest, runner=PathRunner())
        graph = provider.topology_graph()
        endpoint = provider.endpoint_detail("erin-client")
        self.assertTrue(graph["ok"])
        self.assertEqual(graph["graph_validation"], "exact-match-with-containerlab-topology")
        self.assertFalse(graph["truth_boundary"]["stateful_firewall"])
        self.assertEqual(endpoint["links"][0]["peer_node"], "access-wired-1")
        self.assertEqual(endpoint["links"][0]["local_address"], "10.10.20.10/24")

    def test_trace_resolves_every_hop_and_exact_adjacency(self) -> None:
        provider = ContainerlabProvider(self.manifest, runner=PathRunner())
        result = run(provider.trace_path("erin-client", "crm-server"))
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["node_path"], [
            "erin-client", "access-wired-1", "campus-core-1",
            "idc-leaf-1", "crm-server",
        ])
        self.assertEqual(result["link_path"], [
            "erin-wired", "wired-core1", "core1-idc", "crm-service",
        ])
        self.assertTrue(all(item["adjacency_verified"] for item in result["hops"]))

    def test_trace_fails_closed_for_unknown_hop(self) -> None:
        provider = ContainerlabProvider(
            self.manifest, runner=PathRunner(unknown_hop=True),
        )
        result = run(provider.trace_path("erin-client", "crm-server"))
        self.assertFalse(result["ok"])
        self.assertTrue(result["fail_closed"])
        self.assertIn(
            "address-not-in-reviewed-manifest",
            {item["reason"] for item in result["unresolved"]},
        )

    def test_enforcement_path_names_actual_simulations(self) -> None:
        provider = ContainerlabProvider(self.manifest, runner=PathRunner())
        result = run(provider.enforcement_path("erin", "crm"))
        self.assertTrue(result["traffic_allowed_and_verified"], result)
        self.assertEqual(
            [item["implementation"] for item in result["enforcement_points"]],
            [
                "simulated-endpoint-interface-state",
                "linux-server-source-blackhole-route",
            ],
        )
        self.assertFalse(
            result["security_boundary"]["security_edges_in_observed_internal_path"],
        )

    def test_tools_are_projected_to_both_profiles(self) -> None:
        provider = ContainerlabProvider(self.manifest, runner=PathRunner())
        for profile in ("lan", "dc"):
            callables = LabToolAdapter(provider).callables(profile)
            metadata = lab_tool_metadata(
                profile, access_enabled=True, topology_enabled=True,
            )
            for name in (
                "lab_get_topology_graph", "lab_get_endpoint", "lab_trace_path",
                "lab_get_enforcement_path",
            ):
                self.assertIn(name, callables)
                self.assertIn(name, metadata)
        payload = json.loads(run(
            LabToolAdapter(provider).lab_get_endpoint({"endpoint_id": "crm-server"}),
        ))
        self.assertEqual(payload["applications"], ["crm"])


class TestBgpSummaryParsing(unittest.TestCase):
    def test_only_established_rows_are_counted(self) -> None:
        output = """
Neighbor V AS MsgRcvd MsgSent TblVer InQ OutQ Up/Down State/PfxRcd PfxSnt Desc
192.0.2.1 4 64501 23 23 3 0 0 00:00:55 2 3 N/A
192.0.2.5 4 64502 0 0 0 0 0 never Active 0 N/A
"""
        self.assertEqual(_established_bgp_neighbors(output), 1)


if __name__ == "__main__":
    unittest.main()
