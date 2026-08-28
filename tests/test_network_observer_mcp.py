from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from dsh_adapter.backend import _mcp_metadata, open_backend
from integrations.clients.mcp_client import (
    MCPClient,
    MCPToolSpec,
    validate_evidence_envelope,
)
from network_runtime.contracts import sha256_json
from network_runtime.provider_contracts import EVIDENCE_CONTRACT, REGISTRY


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
MANIFEST = ROOT / "labs" / "p075-b-small-production" / "lab.yaml"


def run(value):
    return asyncio.run(value)


def observer_config(*, expected_name: str = "netopyu.network-observer") -> dict:
    return {
        "network-observer": {
            "domain": "network",
            "transport": "stdio",
            "command": [
                str(PYTHON), "-m", "network_provider.mcp_observer",
                "--profile", "lan", "--manifest", str(MANIFEST),
            ],
            "cwd": str(ROOT),
            "expected_server_name": expected_name,
            "expected_server_version": "1.0.0",
        }
    }


class NetworkObserverMCPTests(unittest.TestCase):
    def test_capability_registry_is_unique_and_separates_observer_from_actor(self) -> None:
        contracts = REGISTRY.contracts()
        self.assertEqual(len(contracts), len({item.capability_id for item in contracts}))
        self.assertEqual(
            REGISTRY.for_tool("lab_trace_path").capability_id,
            "network.path.trace",
        )
        self.assertEqual(REGISTRY.for_tool("lab_trace_path").provider_role, "observer")
        self.assertEqual(REGISTRY.for_tool("edit_device_config").provider_role, "actor")

    def test_stdio_observer_exposes_only_reads_and_unwraps_verified_evidence(self) -> None:
        async def exercise():
            client = MCPClient.from_config(observer_config())
            await client.connect_all()
            try:
                names = {item.name for item in client.list_tools()}
                self.assertEqual(len(names), 22)
                self.assertIn("lab_get_topology_graph", names)
                self.assertNotIn("edit_device_config", names)
                spec = client.get_tool_spec("lab_get_topology_graph")
                self.assertEqual(
                    spec.meta["netopyu"]["capability_id"],
                    "network.topology.graph.get",
                )
                result = await client.call_tool("lab_get_topology_graph", {})
                self.assertFalse(result.is_error)
                self.assertEqual(json.loads(result.content)["lab"], "netopyu-small-production")
                self.assertEqual(
                    result.evidence_envelope["provider_identity"],
                    "netopyu.network-observer@1.0.0",
                )
                single = await client.call_tool(
                    "multi_device_check",
                    {"device_ids": "campus-core-1", "command": "show version"},
                )
                self.assertEqual(
                    list(json.loads(single.content)),
                    ["campus-core-1"],
                )
            finally:
                await client.disconnect_all()

        run(exercise())

    def test_identity_mismatch_fails_before_any_tool_is_available(self) -> None:
        async def exercise():
            client = MCPClient.from_config(observer_config(expected_name="wrong.observer"))
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                await client.connect_all()
            await client.disconnect_all()

        run(exercise())

    def test_network_provider_without_identity_pins_is_rejected(self) -> None:
        async def exercise():
            config = observer_config()
            config["network-observer"].pop("expected_server_name")
            config["network-observer"].pop("expected_server_version")
            client = MCPClient.from_config(config)
            with self.assertRaisesRegex(ValueError, "requires expected_server_name"):
                await client.connect_all()
            await client.disconnect_all()

        run(exercise())

    def test_evidence_digest_and_capability_mismatch_fail_closed(self) -> None:
        meta = {
            "netopyu": {
                "domain": "network",
                "provider_role": "observer",
                "provider_kind": "network-observer-mcp",
                "action_type": "read_only",
                "capability_id": "network.topology.graph.get",
                "capability_version": "1.0.0",
                "result_contract": EVIDENCE_CONTRACT,
            }
        }
        spec = MCPToolSpec(
            name="lab_get_topology_graph",
            description="topology",
            server_name="network-observer",
            server_identity="netopyu.network-observer",
            server_version="1.0.0",
            meta=meta,
            identity_pinned=True,
        )
        payload = {"ok": True, "lab": "test"}
        envelope = {
            "ok": True,
            "code": "ok",
            "correlation_id": "corr-test",
            "observed_at": "2026-08-28T00:00:00+00:00",
            "simulation": True,
            "provider_identity": "netopyu.network-observer@1.0.0",
            "capability_id": "network.topology.graph.get",
            "capability_version": "1.0.0",
            "payload_digest": sha256_json(payload),
            "content_type": "application/json",
            "payload": payload,
        }
        content, _ = validate_evidence_envelope(
            spec, envelope,
            server_identity="netopyu.network-observer", server_version="1.0.0",
            now=datetime(2026, 8, 28, tzinfo=timezone.utc),
        )
        self.assertEqual(json.loads(content), payload)
        with self.assertRaisesRegex(RuntimeError, "digest mismatch"):
            validate_evidence_envelope(
                spec, {**envelope, "payload_digest": "sha256:" + "0" * 64},
                server_identity="netopyu.network-observer", server_version="1.0.0",
                now=datetime(2026, 8, 28, tzinfo=timezone.utc),
            )
        with self.assertRaisesRegex(RuntimeError, "freshness limit"):
            validate_evidence_envelope(
                spec, envelope,
                server_identity="netopyu.network-observer", server_version="1.0.0",
                now=datetime(2026, 8, 28, tzinfo=timezone.utc) + timedelta(minutes=6),
            )
        with self.assertRaisesRegex(ValueError, "capability mismatch"):
            _mcp_metadata(MCPToolSpec(
                name="lab_get_topology_graph",
                description="bad declaration",
                server_name="network-observer",
                server_identity="netopyu.network-observer",
                server_version="1.0.0",
                meta={"netopyu": {**meta["netopyu"], "capability_id": "network.fake"}},
                identity_pinned=True,
            ))
        with self.assertRaisesRegex(ValueError, "omitted domain=network"):
            _mcp_metadata(MCPToolSpec(
                name="lab_get_topology_graph",
                description="missing declaration",
                server_name="network-observer",
                server_identity="netopyu.network-observer",
                server_version="1.0.0",
                configured_domain="network",
                identity_pinned=True,
            ))

    def test_backend_prefers_observer_mcp_reads_and_keeps_actor_local(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "observer.yaml"
            config.write_text(
                "mode: pragmatic\n"
                "agent: {profile: lan}\n"
                "tools:\n"
                "  schema_validation_enabled: true\n"
                "  mcp: {config_json: ''}\n"
                "  openapi: {spec_url: '', base_url: ''}\n"
                "  editable_hitl_tools: {edit_device_config: [config_lines, reason]}\n"
                "pragmatic:\n"
                "  device_inventory: []\n"
                "  mcp_servers:\n"
                "    - name: network-observer\n"
                "      domain: network\n"
                "      transport: stdio\n"
                f"      cwd: {ROOT}\n"
                f"      command: [{PYTHON}, -m, network_provider.mcp_observer, --profile, lan, --manifest, {MANIFEST}]\n"
                "      expected_server_name: netopyu.network-observer\n"
                "      expected_server_version: 1.0.0\n"
                "  lab:\n"
                "    enabled: true\n"
                "    provider: containerlab\n"
                f"    manifest: {MANIFEST}\n",
                encoding="utf-8",
            )
            environment = {
                "NETOPYU_CONFIG_PATH": str(config),
                "NETOPYU_DSH_BACKEND": "pragmatic",
                "NETOPYU_TOOL_RESULT_STORE": str(Path(directory) / "results.sqlite"),
            }
            with patch.dict(os.environ, environment, clear=False):
                async def exercise():
                    backend = await open_backend("lan")
                    try:
                        self.assertEqual(
                            backend.sources["lab_get_topology_graph"],
                            "mcp:network-observer",
                        )
                        self.assertEqual(backend.sources["edit_device_config"], "network-lab")
                        self.assertEqual(
                            backend.metadata["lab_get_topology_graph"]["capability_id"],
                            "network.topology.graph.get",
                        )
                        self.assertEqual(
                            backend.metadata["edit_device_config"]["capability_id"],
                            "network.device.config.edit",
                        )
                        topology = json.loads(
                            await backend.callables["lab_get_topology_graph"]({})
                        )
                        self.assertEqual(topology["lab"], "netopyu-small-production")
                    finally:
                        await backend.close()

                run(exercise())


if __name__ == "__main__":
    unittest.main(verbosity=2)
