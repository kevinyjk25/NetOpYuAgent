from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from dsh_adapter.backend import BackendSession
from effect_runtime import EffectRuntime
from effect_runtime.saga import (
    SagaCoordinator,
    SagaDefinition,
    SagaState,
    SagaStepSpec,
)
from network_runtime.access import ObservationAccessContext, ObservationPolicy
from network_runtime.capabilities import (
    CapabilityContract,
    CapabilityKind,
    DataSensitivity,
)
from network_runtime.contracts import (
    ApprovalError,
    Evidence,
    ExecutionOutcome,
    PlanState,
)


def run(value):
    return asyncio.run(value)


class CapabilityAndReadPolicyTests(unittest.TestCase):
    def test_transport_neutral_contract_and_read_scope_are_enforced(self) -> None:
        contract = CapabilityContract.from_metadata(
            "get_device_config",
            {
                "action_type": "read_only",
                "domain": "network",
                "capability_id": "network.device.config.get",
                "capability_version": "1.0.0",
                "provider_role": "observer",
                "provider_identity": "mcp:observer@1.0.0",
                "provider_kind": "mcp",
                "parameters": {"device_id": {"type": "string"}},
                "required_roles": ["operations-reader"],
                "sensitivity": "restricted",
            },
            source="mcp:network-observer",
        )
        self.assertEqual(contract.kind, CapabilityKind.OBSERVATION)
        self.assertEqual(contract.scope_fields, ("device_id",))
        policy = ObservationPolicy()
        denied = policy.authorize(
            contract,
            {"device_id": "core-1"},
            ObservationAccessContext(
                subject_id="viewer",
                roles=frozenset({"operations-reader"}),
                scopes=frozenset({"device_id:edge-1"}),
                purpose="incident-response",
                clearance=DataSensitivity.RESTRICTED,
            ),
        )
        self.assertFalse(denied.allowed)
        self.assertEqual(denied.code, "scope_denied")
        wrong_role = policy.authorize(
            contract,
            {"device_id": "core-1"},
            ObservationAccessContext(
                subject_id="viewer",
                roles=frozenset({"guest"}),
                scopes=frozenset({"device_id:core-1"}),
                purpose="incident-response",
                clearance=DataSensitivity.RESTRICTED,
            ),
        )
        self.assertEqual(wrong_role.code, "role_denied")
        under_cleared = policy.authorize(
            contract,
            {"device_id": "core-1"},
            ObservationAccessContext(
                subject_id="viewer",
                roles=frozenset({"operations-reader"}),
                scopes=frozenset({"device_id:core-1"}),
                purpose="incident-response",
                clearance=DataSensitivity.INTERNAL,
            ),
        )
        self.assertEqual(under_cleared.code, "clearance_denied")
        allowed = policy.authorize(
            contract,
            {"device_id": "core-1"},
            ObservationAccessContext(
                subject_id="viewer",
                roles=frozenset({"operations-reader"}),
                scopes=frozenset({"device_id:core-1"}),
                purpose="incident-response",
                clearance=DataSensitivity.RESTRICTED,
            ),
        )
        self.assertTrue(allowed.allowed)

    def test_runtime_denies_unauthenticated_read_before_provider_call(self) -> None:
        calls = 0

        async def read_tool(arguments):
            nonlocal calls
            calls += 1
            return json.dumps({"ok": True, "arguments": arguments})

        async def backend_factory(profile):
            return BackendSession(
                mode="mock",
                profile_id=profile,
                callables={"safe_read": read_tool},
                metadata={
                    "safe_read": {
                        "description": "read",
                        "parameters": {},
                        "required": [],
                        "action_type": "read_only",
                        "sensitivity": "internal",
                    },
                },
                sources={"safe_read": "unit-test"},
                report={"mode": "mock"},
            )

        runtime = EffectRuntime(backend_factory=backend_factory)
        with self.assertRaisesRegex(ApprovalError, "unauthenticated"):
            run(runtime.invoke_read(
                "lan", "safe_read", {},
                access_context={
                    "subject_id": "",
                    "authenticated": False,
                    "purpose": "test",
                },
            ))
        self.assertEqual(calls, 0)


class TerminalEnvelopeTests(unittest.TestCase):
    def test_actor_intermediate_state_is_not_exposed_to_model(self) -> None:
        outcome = ExecutionOutcome(
            plan_id="plan-1",
            plan_hash="sha256:plan",
            state=PlanState.VERIFIED_SUCCESS,
            result='{"state":"applied","fence_token":7}',
            evidence=(Evidence(
                evidence_type="postcondition",
                source="network-observer",
                target="erin",
                observed_at="2026-08-28T00:00:00+00:00",
                value={"admitted": True},
                passed=True,
                predicate="user is admitted",
                expected=True,
            ),),
        )
        envelope = outcome.terminal_envelope()
        self.assertTrue(envelope["terminal"])
        self.assertEqual(envelope["state"], "verified_success")
        self.assertIn("independently verified", envelope["summary"])
        self.assertNotIn("applied", json.dumps(envelope))
        self.assertTrue(envelope["provider_result_digest"].startswith("sha256:"))


class DurableSagaTests(unittest.TestCase):
    @staticmethod
    def definition() -> SagaDefinition:
        return SagaDefinition.create(
            "service-network-access",
            "1.0.0",
            (
                SagaStepSpec(
                    "service-entitlement", "service",
                    "service.access.entitlement.revoke",
                    compensation_capability_id="service.access.entitlement.grant",
                ),
                SagaStepSpec(
                    "network-enforcement", "network",
                    "network.application.enforcement.revoke",
                    depends_on=("service-entitlement",),
                    compensation_capability_id="network.application.enforcement.apply",
                ),
            ),
        )

    def test_cross_provider_failure_recovers_in_reverse_order_after_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite"
            with SagaCoordinator(path) as saga:
                started = saga.start(self.definition(), correlation_id="change-1001")
                saga_id = started["saga_id"]
                self.assertEqual(started["next_action"]["step_id"], "service-entitlement")
                with self.assertRaisesRegex(Exception, "dependencies"):
                    saga.bind_plan(
                        saga_id, "network-enforcement",
                        plan_id="plan-network", plan_hash="sha256:network",
                    )
                saga.bind_plan(
                    saga_id, "service-entitlement",
                    plan_id="plan-service", plan_hash="sha256:service",
                )
                with self.assertRaisesRegex(Exception, "different immutable plan"):
                    saga.bind_plan(
                        saga_id, "service-entitlement",
                        plan_id="plan-other", plan_hash="sha256:other",
                    )
                first = saga.record_outcome(
                    saga_id, "service-entitlement", terminal_state="verified_success",
                )
                self.assertEqual(first["next_action"]["step_id"], "network-enforcement")
                saga.bind_plan(
                    saga_id, "network-enforcement",
                    plan_id="plan-network", plan_hash="sha256:network",
                )
                failed = saga.record_outcome(
                    saga_id, "network-enforcement",
                    terminal_state="precondition_changed", error="drift",
                )
                self.assertEqual(failed["state"], SagaState.COMPENSATING.value)
                self.assertEqual(failed["next_action"]["step_id"], "service-entitlement")
                self.assertTrue(failed["audit"]["ok"])

            with SagaCoordinator(path) as recovered:
                resumable = recovered.recoverable()
                self.assertEqual([item["saga_id"] for item in resumable], [saga_id])
                recovered.bind_compensation_plan(
                    saga_id, "service-entitlement",
                    plan_id="plan-compensate", plan_hash="sha256:compensate",
                )
                final = recovered.record_compensation_outcome(
                    saga_id, "service-entitlement", terminal_state="verified_success",
                )
                self.assertEqual(final["state"], SagaState.COMPENSATED.value)
                self.assertIsNone(final["next_action"])
                self.assertTrue(final["audit"]["ok"])

    def test_uncompensatable_verified_step_requires_manual_intervention(self) -> None:
        definition = SagaDefinition.create(
            "irreversible-example", "1.0.0",
            (
                SagaStepSpec("irreversible", "external", "external.delete"),
                SagaStepSpec("later", "network", "network.change", depends_on=("irreversible",)),
            ),
        )
        with tempfile.TemporaryDirectory() as directory, SagaCoordinator(
            Path(directory) / "runtime.sqlite"
        ) as saga:
            started = saga.start(definition, correlation_id="irreversible-test")
            saga_id = started["saga_id"]
            saga.bind_plan(saga_id, "irreversible", plan_id="p1", plan_hash="h1")
            saga.record_outcome(saga_id, "irreversible", terminal_state="verified_success")
            saga.bind_plan(saga_id, "later", plan_id="p2", plan_hash="h2")
            final = saga.record_outcome(
                saga_id, "later", terminal_state="precondition_changed",
            )
            self.assertEqual(final["state"], SagaState.MANUAL_INTERVENTION_REQUIRED.value)

    def test_saga_event_tampering_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory, SagaCoordinator(
            Path(directory) / "runtime.sqlite"
        ) as saga:
            started = saga.start(self.definition(), correlation_id="tamper-test")
            saga_id = started["saga_id"]
            self.assertTrue(saga.verify_event_chain(saga_id)["ok"])
            saga.db.execute(
                "UPDATE effect_saga_events SET payload_json='{}' WHERE saga_id=?",
                (saga_id,),
            )
            saga.db.commit()
            self.assertFalse(saga.verify_event_chain(saga_id)["ok"])

    def test_cross_step_assertion_can_request_compensation(self) -> None:
        with tempfile.TemporaryDirectory() as directory, SagaCoordinator(
            Path(directory) / "runtime.sqlite"
        ) as saga:
            started = saga.start(self.definition(), correlation_id="checkpoint-failure")
            saga_id = started["saga_id"]
            saga.bind_plan(
                saga_id, "service-entitlement", plan_id="p1", plan_hash="h1",
            )
            saga.record_outcome(
                saga_id, "service-entitlement", terminal_state="verified_success",
            )
            requested = saga.request_compensation(
                saga_id, reason="cross-layer checkpoint did not converge",
            )
            self.assertEqual(requested["state"], SagaState.COMPENSATING.value)
            self.assertEqual(requested["next_action"]["step_id"], "service-entitlement")


if __name__ == "__main__":
    unittest.main(verbosity=2)
