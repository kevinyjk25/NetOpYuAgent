from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dsh_adapter.backend import open_backend
from dsh_adapter.bridge import build_manifest
from effect_runtime import EffectRuntime
from integrations.clients.mcp_client import MCPCallResult, MCPClient
from integrations.router.tool_router import ToolRouter
from network_runtime import PlanState
from network_runtime.evidence import failed_output
from network_runtime.l0_skills import REGISTRY as L0_SKILLS
from service_layer.store import ServiceStore, ServiceStoreError


def run(value):
    return asyncio.run(value)


class ServiceMCPTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        server = {
            "transport": "stdio",
            "command": [
                ".venv/bin/python", "-m", "service_layer.mcp_server", "--domain", "all",
            ],
            "env": {"NETOPYU_SERVICE_STORE": str(root / "service.sqlite")},
            "cwd": str(Path(__file__).parents[1]),
            "trusted_for_writes": True,
            "expected_server_name": "netopyu.all",
            "expected_server_version": "1.0.0",
        }
        self.environment = {
            "NETOPYU_DSH_BACKEND": "pragmatic",
            "NETOPYU_CONFIG_PATH": "config.yaml",
            "MCP_CONFIG_JSON": json.dumps({"access-policy-service": server}),
            "NETOPYU_DSH_NETWORK_RUNTIME_STORE": str(root / "runtime.sqlite"),
            "NETOPYU_DSH_TOOL_RESULT_STORE": str(root / "results.sqlite"),
        }
        self.env_patch = patch.dict(os.environ, self.environment)
        self.env_patch.start()

    def tearDown(self) -> None:
        self.env_patch.stop()
        self.temp.cleanup()

    def _prepare_grant(self, runtime: EffectRuntime) -> dict:
        contract = L0_SKILLS.for_tool("lan", "access_policy_grant_entitlement")
        assert contract is not None
        return run(runtime.prepare(
            "lan", "access_policy_grant_entitlement",
            {
                "user_id": "alice", "app_id": "crm", "role": "sales-rep",
                "change_id": "CHG-1001", "expected_revision": 0,
                "reason": "unit-test approved access",
            },
            l0_skill_id=contract.skill_id,
        ))

    @staticmethod
    def _execute(runtime: EffectRuntime, prepared: dict):
        plan = prepared["plan"]
        return run(runtime.execute(
            plan_id=plan["plan_id"], plan_hash=plan["plan_hash"],
            execution_nonce=prepared["execution_nonce"],
            approval_request_id="approval-service-test",
            approval_actor="unit-test-operator",
            allow_destructive=True,
        ))

    def test_official_stdio_transport_preserves_structured_content(self) -> None:
        config = json.loads(self.environment["MCP_CONFIG_JSON"])
        client = MCPClient.from_config(config)

        async def exercise():
            await client.connect_all()
            try:
                spec = client.get_tool_spec("identity_get_user")
                self.assertIsNotNone(spec)
                self.assertEqual(spec.server_identity, "netopyu.all")
                self.assertTrue(spec.input_schema_digest.startswith("sha256:"))
                result = await client.call_tool("identity_get_user", {"user_id": "alice"})
                self.assertFalse(result.is_error)
                self.assertEqual(result.structured_content["user"]["status"], "active")
            finally:
                await client.disconnect_all()

        run(exercise())

    def test_internal_compensator_tool_is_not_projected_to_harness(self) -> None:
        manifest = build_manifest("lan", include_destructive=True)
        names = {item["name"] for item in manifest["tools"]}
        self.assertIn("access_policy_grant_entitlement", names)
        self.assertNotIn("access_policy_restore_entitlement", names)

    def test_structured_semantic_failure_is_fail_closed(self) -> None:
        self.assertTrue(failed_output('{"ok": false, "error": "revision conflict"}'))
        self.assertFalse(failed_output('{"ok": true, "revision": 2}'))

        class SemanticFailureClient:
            async def call_tool(self, tool_name, arguments):
                return MCPCallResult(
                    tool_name=tool_name,
                    server_name="test",
                    content='{"ok": false}',
                    structured_content={"ok": False},
                )

        with self.assertRaisesRegex(RuntimeError, "structured ok=false"):
            run(ToolRouter._make_mcp_fn(
                SemanticFailureClient(), "unsafe_semantic_result",
            )({}))

    def test_revision_and_idempotency_are_safe_across_store_connections(self) -> None:
        path = Path(self.temp.name) / "atomic.sqlite"
        first = ServiceStore(path)
        second = ServiceStore(path)
        try:
            arguments = {
                "operation": "grant", "user_id": "alice", "app_id": "crm",
                "role": "sales-rep", "roles": None, "change_id": "CHG-1001",
                "reason": "atomic test", "expected_revision": 0,
                "correlation_id": "corr-atomic-1",
            }
            granted = first.mutate_entitlement(**arguments)
            self.assertEqual(granted["revision"], 1)
            replayed = second.mutate_entitlement(**arguments)
            self.assertTrue(replayed["replayed"])

            second.mutate_entitlement(
                operation="revoke", user_id="alice", app_id="crm", role=None,
                roles=None, change_id="CHG-1001", reason="advance state",
                expected_revision=1, correlation_id="corr-atomic-2",
            )
            with self.assertRaisesRegex(ServiceStoreError, "target state changed later"):
                first.mutate_entitlement(**arguments)
            with self.assertRaisesRegex(ServiceStoreError, "expected revision 0, observed 2"):
                first.mutate_entitlement(**{
                    **arguments,
                    "role": "sales-admin",
                    "correlation_id": "corr-atomic-3",
                })
        finally:
            second.close()
            first.close()

    def test_revoked_seed_entitlement_is_not_resurrected_on_reopen(self) -> None:
        path = Path(self.temp.name) / "seed-once.sqlite"
        with ServiceStore(path) as first:
            first.mutate_entitlement(
                operation="revoke", user_id="bob", app_id="crm", role=None,
                roles=None, change_id="CHG-1001", reason="seed persistence test",
                expected_revision=0, correlation_id="corr-seed-1",
            )
            self.assertFalse(first.entitlement("bob", "crm")["allowed"])
        with ServiceStore(path) as reopened:
            current = reopened.entitlement("bob", "crm")
            self.assertFalse(current["allowed"])
            self.assertEqual(current["roles"], [])
            self.assertEqual(current["revision"], 1)

    def test_trusted_service_write_is_bound_verified_and_audited(self) -> None:
        runtime = EffectRuntime()
        prepared = self._prepare_grant(runtime)
        self.assertEqual(prepared["status"], "plan_ready")
        plan = prepared["plan"]
        self.assertEqual(plan["schema_version"], 9)
        self.assertTrue(plan["provider_release_digest"].startswith("unmanaged-local:"))
        self.assertEqual(plan["provider_deployment_digest"], "unmanaged-local")
        self.assertEqual(
            plan["provider_identity"],
            "mcp:access-policy-service:netopyu.all@1.0.0",
        )
        outcome = self._execute(runtime, prepared)
        self.assertEqual(outcome.state, PlanState.VERIFIED_SUCCESS)
        current = json.loads(run(runtime.invoke_read(
            "lan", "access_policy_get_entitlement", {"user_id": "alice", "app_id": "crm"},
        )))
        self.assertTrue(current["allowed"])
        self.assertEqual(current["roles"], ["sales-rep"])
        self.assertTrue(runtime.audit(plan["plan_id"])["ok"])

    def test_untrusted_mcp_write_fails_closed(self) -> None:
        config = json.loads(self.environment["MCP_CONFIG_JSON"])
        config["access-policy-service"]["trusted_for_writes"] = False
        with patch.dict(os.environ, {"MCP_CONFIG_JSON": json.dumps(config)}):
            prepared = self._prepare_grant(EffectRuntime())
        self.assertEqual(prepared["status"], "rejected")
        self.assertIn("no reviewed execution contract", prepared["errors"][0])

    def test_schema_or_provider_drift_after_approval_is_rejected(self) -> None:
        opens = 0

        async def backend_factory(profile: str):
            nonlocal opens
            opens += 1
            backend = await open_backend(profile)
            if opens >= 2:
                backend.metadata["access_policy_grant_entitlement"]["input_schema_digest"] = (
                    "sha256:" + "0" * 64
                )
            return backend

        runtime = EffectRuntime(backend_factory=backend_factory)
        prepared = self._prepare_grant(runtime)
        outcome = self._execute(runtime, prepared)
        self.assertEqual(outcome.state, PlanState.PRECONDITION_CHANGED)
        self.assertIn("write was not sent", outcome.error)

    def test_failed_service_postcondition_restores_exact_role_snapshot(self) -> None:
        async def backend_factory(profile: str):
            backend = await open_backend(profile)
            original = backend.callables["access_policy_get_entitlement"]
            reads = 0

            async def fail_only_verification(arguments):
                nonlocal reads
                reads += 1
                value = await original(arguments)
                # Execution-time preflight is read 1; verifier is read 2;
                # rollback verification is read 3.
                if reads == 2:
                    payload = json.loads(value)
                    payload.update({"allowed": False, "roles": []})
                    return json.dumps(payload, sort_keys=True)
                return value

            backend.callables["access_policy_get_entitlement"] = fail_only_verification
            return backend

        runtime = EffectRuntime(backend_factory=backend_factory)
        prepared = self._prepare_grant(runtime)
        outcome = self._execute(runtime, prepared)
        self.assertEqual(outcome.state, PlanState.ROLLBACK_VERIFIED)
        current = json.loads(run(runtime.invoke_read(
            "lan", "access_policy_get_entitlement", {"user_id": "alice", "app_id": "crm"},
        )))
        self.assertFalse(current["allowed"])
        self.assertEqual(current["roles"], [])
        self.assertTrue(any(item.evidence_type == "rollback_postcondition" for item in outcome.evidence))


if __name__ == "__main__":
    unittest.main(verbosity=2)
