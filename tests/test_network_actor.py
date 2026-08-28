from __future__ import annotations

import asyncio
import tempfile
import unittest
import uuid
from pathlib import Path

from dsh_adapter.backend import _mcp_metadata
from integrations.clients.mcp_client import MCPClient
from network_lab import ContainerlabProvider, load_manifest
from network_provider.actor import ActorCrashSimulated, DurableNetworkActor, EffectContext
from network_provider.actor_store import ActorStore, ActorStoreError
from network_runtime.contracts import sha256_json


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
MANIFEST = ROOT / "labs" / "p075-b-small-production" / "lab.yaml"


def run(value):
    return asyncio.run(value)


class MemoryAccessProvider(ContainerlabProvider):
    """Manifest-bound provider whose one effect is deterministic in memory."""

    def __init__(self) -> None:
        super().__init__(load_manifest(MANIFEST))
        self.allowed: dict[tuple[str, str], bool] = {("erin", "crm"): True}
        self.effect_calls = 0

    async def application_access_blocked(self, user_id: str, app_id: str) -> bool:
        self._user(user_id)
        self._application(app_id)
        return not self.allowed.get((user_id, app_id), True)

    async def set_application_access(
        self, user_id: str, app_id: str, *, allowed: bool,
    ) -> str:
        self._user(user_id)
        self._application(app_id)
        self.effect_calls += 1
        self.allowed[(user_id, app_id)] = allowed
        return f"application={app_id} allowed={str(allowed).lower()}"

    async def application_access_snapshot(
        self, user_id: str, app_id: str,
    ) -> dict[str, object]:
        user = self._user(user_id)
        app = self._application(app_id)
        allowed = self.allowed.get((user_id, app_id), True)
        return {
            "user_id": user_id,
            "app_id": app_id,
            "source_prefix": f"{user.address}/32",
            "application_endpoint": app.endpoint,
            "allowed": allowed,
            "route": None if allowed else f"blackhole {user.address}",
        }

    async def restore_application_access_snapshot(
        self, user_id: str, app_id: str, snapshot: dict[str, object],
    ) -> str:
        result = await self.set_application_access(
            user_id, app_id, allowed=bool(snapshot["allowed"]),
        )
        if await self.application_access_snapshot(user_id, app_id) != snapshot:
            raise RuntimeError("memory snapshot restoration failed")
        return result


def effect_context(operation_id: str, preflight: dict, phase: str = "execute") -> EffectContext:
    return EffectContext.create(
        operation_id=operation_id,
        plan_hash=sha256_json({"plan": operation_id}),
        intent_hash=sha256_json({"intent": operation_id}),
        approved_preflight=preflight,
        effect_phase=phase,
    )


class DurableNetworkActorTests(unittest.TestCase):
    def test_crash_recovery_idempotent_replay_and_exact_restore(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store_path = Path(directory) / "actor.sqlite"
            provider = MemoryAccessProvider()
            operation_id = str(uuid.uuid4())
            arguments = {
                "user_id": "erin",
                "app_id": "crm",
                "change_id": "CHG-1001",
                "reason": "incident containment",
            }

            async def crash_after_effect(point: str, _operation_id: str) -> None:
                if point == "after_effect_before_commit":
                    raise ActorCrashSimulated()

            first_store = ActorStore(store_path, lease_seconds=10)
            first_actor = DurableNetworkActor(
                provider, first_store, fault_hook=crash_after_effect,
            )
            preflight = run(first_actor._observe_preflight(
                "network_revoke_app_enforcement", arguments,
            ))
            context = effect_context(operation_id, preflight)
            with self.assertRaises(ActorCrashSimulated):
                run(first_actor.execute(
                    "network_revoke_app_enforcement", arguments, context,
                ))
            self.assertFalse(provider.allowed[("erin", "crm")])
            self.assertEqual(first_store.get(operation_id)["state"], "executing")
            first_store.close()

            recovered_store = ActorStore(store_path, lease_seconds=10)
            recovered_actor = DurableNetworkActor(provider, recovered_store)
            reconciled = run(recovered_actor.reconcile_startup())
            self.assertEqual(reconciled, [{"operation_id": operation_id, "state": "applied"}])

            effect_calls = provider.effect_calls
            replay = run(recovered_actor.execute(
                "network_revoke_app_enforcement", arguments, context,
            ))
            self.assertTrue(replay["replayed"])
            self.assertTrue(replay["reconciled"])
            self.assertEqual(provider.effect_calls, effect_calls)

            restore_args = {
                "user_id": "erin", "app_id": "crm", "allowed": True,
                "change_id": "CHG-1001", "reason": "verified rollback",
            }
            restored = run(recovered_actor.execute(
                "network_restore_app_enforcement",
                restore_args,
                effect_context(operation_id, preflight, "compensate"),
            ))
            self.assertEqual(restored["state"], "restored")
            self.assertTrue(provider.allowed[("erin", "crm")])
            finalized = run(recovered_actor.finalize(
                operation_id, context.plan_hash, "rollback_verified",
            ))
            self.assertEqual(finalized["state"], "restored")
            self.assertTrue(finalized["audit_chain_valid"])
            self.assertTrue(recovered_store.verify_event_chain(operation_id))
            self.assertEqual(
                recovered_store.events(operation_id)[-1]["event_type"],
                "runtime_finalized",
            )
            recovered_store.close()

    def test_operation_id_reuse_with_changed_immutable_content_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ActorStore(Path(directory) / "actor.sqlite", lease_seconds=10)
            provider = MemoryAccessProvider()
            actor = DurableNetworkActor(provider, store)
            operation_id = str(uuid.uuid4())
            arguments = {
                "user_id": "erin", "app_id": "crm", "change_id": "CHG-1",
                "reason": "test",
            }
            preflight = run(actor._observe_preflight(
                "network_revoke_app_enforcement", arguments,
            ))
            context = effect_context(operation_id, preflight)
            run(actor.execute("network_revoke_app_enforcement", arguments, context))
            changed = {**arguments, "reason": "different immutable request"}
            with self.assertRaisesRegex(ActorStoreError, "reused"):
                run(actor.execute("network_revoke_app_enforcement", changed, context))
            self.assertEqual(provider.effect_calls, 1)
            store.close()

    def test_unreviewed_dc_role_is_rejected_before_durable_effect(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ActorStore(Path(directory) / "actor.sqlite", lease_seconds=10)
            provider = MemoryAccessProvider()
            actor = DurableNetworkActor(provider, store)
            operation_id = str(uuid.uuid4())
            arguments = {
                "user_id": "erin", "app_id": "crm", "role": "domain-admin",
                "reason": "unsafe role",
            }
            with self.assertRaisesRegex(ActorStoreError, "not reviewed"):
                run(actor.execute(
                    "dc_grant_app_access", arguments,
                    effect_context(operation_id, {}),
                ))
            self.assertIsNone(store.find(operation_id))
            self.assertEqual(provider.effect_calls, 0)
            store.close()

    def test_operation_row_tampering_breaks_actor_audit_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ActorStore(Path(directory) / "actor.sqlite", lease_seconds=10)
            provider = MemoryAccessProvider()
            actor = DurableNetworkActor(provider, store)
            operation_id = str(uuid.uuid4())
            arguments = {
                "user_id": "erin", "app_id": "crm", "change_id": "CHG-2",
                "reason": "tamper test",
            }
            preflight = run(actor._observe_preflight(
                "network_revoke_app_enforcement", arguments,
            ))
            run(actor.execute(
                "network_revoke_app_enforcement",
                arguments,
                effect_context(operation_id, preflight),
            ))
            self.assertTrue(store.verify_event_chain(operation_id))
            store._db.execute(
                "UPDATE actor_operations SET desired_json=? WHERE operation_id=?",
                ('{"allowed":true}', operation_id),
            )
            store._db.commit()
            self.assertFalse(store.verify_event_chain(operation_id))
            with self.assertRaisesRegex(ActorStoreError, "record digest mismatch"):
                store.get(operation_id)
            store.close()

    def test_manual_intervention_quarantines_target_across_operations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ActorStore(Path(directory) / "actor.sqlite", lease_seconds=10)
            provider = MemoryAccessProvider()
            actor = DurableNetworkActor(provider, store)
            first_id = str(uuid.uuid4())
            arguments = {
                "user_id": "erin", "app_id": "crm", "change_id": "CHG-3",
                "reason": "manual quarantine test",
            }
            preflight = run(actor._observe_preflight(
                "network_revoke_app_enforcement", arguments,
            ))
            first_context = effect_context(first_id, preflight)
            run(actor.execute(
                "network_revoke_app_enforcement", arguments, first_context,
            ))
            finalized = run(actor.finalize(
                first_id, first_context.plan_hash, "manual_intervention_required",
            ))
            self.assertEqual(finalized["state"], "manual_intervention")
            current_preflight = run(actor._observe_preflight(
                "network_revoke_app_enforcement", arguments,
            ))
            with self.assertRaisesRegex(ActorStoreError, "quarantined"):
                run(actor.execute(
                    "network_revoke_app_enforcement",
                    {**arguments, "change_id": "CHG-4"},
                    effect_context(str(uuid.uuid4()), current_preflight),
                ))
            self.assertEqual(provider.effect_calls, 1)
            store.close()


class NetworkActorMCPTests(unittest.TestCase):
    def test_stdio_actor_is_identity_pinned_and_hides_runtime_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "network-actor": {
                    "domain": "network",
                    "transport": "stdio",
                    "command": [
                        str(PYTHON), "-m", "network_provider.mcp_actor",
                        "--profile", "all", "--manifest", str(MANIFEST),
                        "--store", str(Path(directory) / "actor.sqlite"),
                    ],
                    "cwd": str(ROOT),
                    "trusted_for_writes": True,
                    "expected_server_name": "netopyu.network-actor",
                    "expected_server_version": "1.0.0",
                }
            }

            async def exercise() -> None:
                client = MCPClient.from_config(config)
                await client.connect_all()
                try:
                    names = {item.name for item in client.list_tools()}
                    self.assertIn("edit_device_config", names)
                    self.assertIn("network_actor_finalize", names)
                    self.assertNotIn("lab_get_topology_graph", names)
                    spec = client.get_tool_spec("edit_device_config")
                    metadata = _mcp_metadata(spec)
                    self.assertEqual(metadata["provider_role"], "actor")
                    self.assertEqual(metadata["provider_kind"], "network-actor-mcp")
                    self.assertNotIn("operation_id", metadata["parameters"])
                    self.assertNotIn("approved_preflight", metadata["parameters"])
                    self.assertEqual(
                        set(metadata["internal_parameters"]),
                        {
                            "operation_id", "plan_hash", "intent_hash",
                            "approved_preflight", "effect_phase",
                        },
                    )
                    self.assertEqual(metadata["profiles"], ["lan", "dc"])
                    self.assertEqual(
                        _mcp_metadata(client.get_tool_spec("grant_user_access"))["profiles"],
                        ["lan"],
                    )
                    self.assertEqual(
                        _mcp_metadata(client.get_tool_spec("dc_grant_app_access"))["profiles"],
                        ["dc"],
                    )
                    self.assertTrue(
                        _mcp_metadata(client.get_tool_spec("network_actor_finalize"))["internal_only"]
                    )
                finally:
                    await client.disconnect_all()

            run(exercise())


if __name__ == "__main__":
    unittest.main(verbosity=2)
