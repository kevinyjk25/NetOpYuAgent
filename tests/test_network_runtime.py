from __future__ import annotations

import asyncio
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from dsh_adapter.backend import open_backend
from dsh_adapter.bridge import invoke_tool
from network_runtime import NetworkRuntime, PlanState
from network_runtime.contracts import (
    ApprovalError,
    OutcomeIndeterminateError,
    PlanIntegrityError,
    PreparedPlan,
    ResourceConflictError,
    StateTransitionError,
    sha256_json,
)
from network_runtime.journal import NetworkJournal
from network_runtime.compensators import REGISTRY as COMPENSATORS
from network_runtime.l0_skills import REGISTRY as L0_SKILLS
from network_runtime.policies import reviewed_contracts
from network_runtime.verifiers import REGISTRY as VERIFIERS
from network_runtime.workflows import WorkflowRuntime, compile_workflow_templates


def run(coro):
    return asyncio.run(coro)


class NetworkRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.journal = root / "network-runtime.sqlite"
        self.tool_results = root / "tool-results.sqlite"
        self.old_environment = {
            name: os.environ.get(name)
            for name in (
                "NETOPYU_DSH_BACKEND",
                "NETOPYU_DSH_NETWORK_RUNTIME_STORE",
                "NETOPYU_DSH_TOOL_RESULT_STORE",
                "NETOPYU_DSH_ALLOW_DESTRUCTIVE",
            )
        }
        os.environ["NETOPYU_DSH_BACKEND"] = "mock"
        os.environ["NETOPYU_DSH_NETWORK_RUNTIME_STORE"] = str(self.journal)
        os.environ["NETOPYU_DSH_TOOL_RESULT_STORE"] = str(self.tool_results)
        from profiles.dc import tools as dc_tools
        from profiles.lan import tools as lan_tools
        from profiles.wan import tools as wan_tools

        lan_tools._DEVICE_STATE.clear()
        lan_tools._LAN_ACCESS_CHANGES.clear()
        lan_tools._MOCK_OPERATION_STATE.clear()
        dc_tools._FABRIC_STATE.clear()
        dc_tools._DC_ACCESS_CHANGES.clear()
        # Restore the canonical ACL member lists that other tests or runtime
        # executions may have mutated in this interpreter.
        dc_tools._DC_APP_ACL["crm"] = {
            "sales-rep": ["bob", "carol"], "sales-admin": ["dave"],
        }
        wan_tools._WAN_TUNNEL_STATE.clear()
        wan_tools._WAN_TUNNEL_STATE.update({
            item["id"]: item["transport"] for item in wan_tools._WAN_TUNNELS
        })

    def tearDown(self) -> None:
        for name, value in self.old_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self.temp.cleanup()

    def runtime(self, **kwargs) -> NetworkRuntime:
        return NetworkRuntime(self.journal, **kwargs)

    def prepare(self, profile: str, tool: str, arguments: dict):
        return self.prepare_with(self.runtime(), profile, tool, arguments)

    def prepare_with(
        self, runtime: NetworkRuntime, profile: str, tool: str, arguments: dict,
        *, session_id: str | None = None,
    ):
        contract = L0_SKILLS.for_tool(profile, tool)
        return run(runtime.prepare(
            profile, tool, arguments, session_id=session_id,
            l0_skill_id=contract.skill_id if contract else None,
        ))

    def execute(self, prepared: dict, **overrides):
        plan = prepared["plan"]
        values = {
            "plan_id": plan["plan_id"],
            "plan_hash": plan["plan_hash"],
            "execution_nonce": prepared["execution_nonce"],
            "approval_request_id": "approval-test",
            "approval_actor": "unit-test-operator",
            "allow_destructive": True,
        }
        values.update(overrides)
        return run(self.runtime().execute(**values))

    def test_missing_safety_parameter_requires_clarification_before_approval(self) -> None:
        prepared = self.prepare("lan", "edit_device_config", {
            "device_id": "ap-01", "config_lines": ["radius-server timeout 3"],
        })
        self.assertEqual(prepared["status"], "clarification_required")
        self.assertIn("reason", prepared["missing"])
        self.assertFalse(self.journal.exists())

    def test_unknown_parameter_and_inventory_entity_fail_closed(self) -> None:
        unknown = self.prepare("lan", "grant_user_access", {
            "user_id": "erin", "reason": "test", "surprise": True,
        })
        self.assertEqual(unknown["status"], "rejected")
        self.assertIn("unknown parameter: surprise", unknown["errors"])
        unresolved = self.prepare("lan", "edit_device_config", {
            "device_id": "not-a-device", "config_lines": ["ntp server 10.0.0.5"], "reason": "test",
        })
        self.assertEqual(unresolved["status"], "clarification_required")
        self.assertEqual(unresolved["clarification"]["code"], "intent_target_ambiguous")
        self.assertEqual(
            unresolved["clarification"]["l0_skill_id"], "network.device.config.edit",
        )
        self.assertTrue(any("does not resolve uniquely" in item for item in unresolved["errors"]))

    def test_direct_write_bridge_is_retired_even_with_legacy_boolean(self) -> None:
        os.environ["NETOPYU_DSH_ALLOW_DESTRUCTIVE"] = "1"
        with self.assertRaises(ApprovalError):
            run(invoke_tool(
                "lan", "grant_user_access", {"user_id": "erin", "reason": "bypass"},
                allow_destructive=True,
            ))

    def test_write_prepare_requires_exact_l0_skill_entrypoint(self) -> None:
        runtime = self.runtime()
        arguments = {"service": "crm", "environment": "staging"}
        missing = run(runtime.prepare("lan", "restart_service", arguments))
        self.assertEqual(missing["status"], "rejected")
        self.assertEqual(missing["expected_l0_skill_id"], "network.service.restart")
        wrong = run(runtime.prepare(
            "lan", "restart_service", arguments,
            l0_skill_id="network.service.rollback",
        ))
        self.assertEqual(wrong["status"], "rejected")
        self.assertEqual(runtime.recent(), [])

    def test_schema_v10_plan_binds_identity_policy_provider_release_intent_and_l0_steps(self) -> None:
        prepared = self.prepare("lan", "restart_service", {
            "service": "crm", "environment": "staging",
        })
        plan = prepared["plan"]
        self.assertEqual(plan["schema_version"], 10)
        self.assertIsNone(plan["l1_decision_binding"])
        self.assertEqual(plan["requester_identity"]["subject_id"], "local-effect-runtime")
        self.assertEqual(plan["requester_digest"], prepared["identity_binding"]["requester_digest"])
        self.assertEqual(plan["approval_mode"], "single")
        self.assertEqual(plan["approval_policy_id"], "netopyu.effect-approval")
        self.assertTrue(plan["approval_policy_hash"].startswith("sha256:"))
        self.assertEqual(plan["provider_role"], "actor")
        self.assertTrue(plan["capability_id"])
        self.assertTrue(plan["capability_version"])
        self.assertEqual(plan["provider_identity"], "profile-mock")
        self.assertTrue(plan["provider_release_digest"].startswith("unmanaged-local:"))
        self.assertEqual(plan["provider_manifest_digest"], "unmanaged-local")
        self.assertEqual(plan["provider_qualification_digest"], "unmanaged-local")
        self.assertEqual(plan["provider_deployment_digest"], "unmanaged-local")
        self.assertTrue(plan["input_schema_digest"].startswith("sha256:"))
        self.assertTrue(plan["output_schema_digest"].startswith("sha256:"))
        self.assertEqual(plan["l0_skill_id"], "network.service.restart")
        self.assertEqual(plan["intent_spec"]["operation"], "restart_service")
        self.assertEqual(
            plan["intent_spec"]["desired_state"],
            {"service_health": "healthy", "rollout": "complete"},
        )
        self.assertEqual(plan["intent_hash"], plan["intent_spec"]["intent_hash"])
        self.assertEqual(
            [item["step_id"] for item in plan["step_contract"]],
            [
                "snapshot", "precheck", "approval", "revalidate", "execute",
                "verify", "commit", "reconcile", "abort", "escalate",
            ],
        )
        outcome = self.execute(prepared)
        self.assertEqual(outcome.state, PlanState.VERIFIED_SUCCESS)
        inspection = self.runtime().inspect(plan["plan_id"])
        events = inspection["events"]
        l0_steps = [
            event["payload"]["step_id"]
            for event in events
            if event["event_type"] in {"l0_step_started", "l0_step_completed"}
        ]
        self.assertEqual(l0_steps, [
            "validate_parameters", "compile_intent", "preflight", "approval",
            "revalidate", "execute", "execute", "verify", "verify", "audit",
        ])
        graph = inspection["graph_execution"]
        self.assertTrue(graph["complete"])
        self.assertEqual(graph["terminal_phase"], "commit")
        self.assertEqual(
            [item["node_id"] for item in graph["results"]],
            [
                "snapshot", "precheck", "approval", "revalidate", "execute",
                "verify", "commit",
            ],
        )
        latency = inspection["graph_latency"]
        self.assertGreaterEqual(latency["runtime_active_ms"], 0.0)
        self.assertGreaterEqual(latency["approval_wait_ms"], 0.0)
        self.assertIn("excludes Reasoning/LLM", latency["claim_boundary"])
        provenance = inspection["provenance"]
        self.assertGreaterEqual(provenance["coverage"]["evidence_nodes"], 2)
        self.assertEqual(provenance["coverage"]["traceability_rate"], 1.0)
        self.assertEqual(
            provenance["integrity"], {"acyclic": True, "dangling_edges": 0},
        )
        self.assertTrue(all(
            "identity" not in node and "object" not in node
            for node in provenance["nodes"]
        ))
        self.assertTrue(any(
            edge["relation"] == "depends_on"
            for edge in provenance["edges"]
        ))

    def test_schema_v9_plan_hash_remains_read_compatible(self) -> None:
        prepared = self.prepare("lan", "restart_service", {
            "service": "crm", "environment": "staging",
        })
        legacy = dict(prepared["plan"])
        legacy["schema_version"] = 9
        legacy.pop("l1_decision_binding")
        immutable = dict(legacy)
        immutable.pop("plan_hash")
        immutable.pop("state")
        legacy["plan_hash"] = sha256_json(immutable)
        loaded = PreparedPlan.from_dict(legacy)
        self.assertEqual(loaded.schema_version, 9)
        self.assertIsNone(loaded.l1_decision_binding)

    def test_schema_v8_plan_hash_remains_read_compatible(self) -> None:
        prepared = self.prepare("lan", "restart_service", {
            "service": "crm", "environment": "staging",
        })
        legacy = dict(prepared["plan"])
        legacy["schema_version"] = 8
        legacy.pop("provider_deployment_digest")
        legacy.pop("l1_decision_binding")
        immutable = dict(legacy)
        immutable.pop("plan_hash")
        immutable.pop("state")
        legacy["plan_hash"] = sha256_json(immutable)
        loaded = PreparedPlan.from_dict(legacy)
        self.assertEqual(loaded.schema_version, 8)
        self.assertEqual(loaded.provider_deployment_digest, "legacy-unbound")

    def test_signed_provider_release_must_authorize_selected_l0_contract(self) -> None:
        async def backend_factory(profile):
            backend = await open_backend(profile)
            backend.metadata["restart_service"] = dict(backend.metadata["restart_service"])
            backend.metadata["restart_service"].update({
                "provider_release_digest": "sha256:" + "a" * 64,
                "provider_manifest_digest": "sha256:" + "b" * 64,
                "provider_qualification_digest": "sha256:" + "c" * 64,
                "provider_l0_contract_hashes": ["sha256:" + "0" * 64],
            })
            return backend

        runtime = self.runtime(backend_factory=backend_factory)
        prepared = self.prepare_with(runtime, "lan", "restart_service", {
            "service": "crm", "environment": "staging",
        })
        self.assertEqual(prepared["status"], "rejected")
        self.assertIn("does not authorize this L0 contract", prepared["errors"][0])
        self.assertEqual(runtime.recent(), [])

    def test_provider_release_switch_after_approval_is_terminal_without_write(self) -> None:
        opens = 0
        writes = 0
        l0 = L0_SKILLS.for_tool("lan", "restart_service")
        self.assertIsNotNone(l0)

        async def backend_factory(profile):
            nonlocal opens, writes
            opens += 1
            backend = await open_backend(profile)
            metadata = dict(backend.metadata["restart_service"])
            backend.metadata["restart_service"] = metadata
            metadata.update({
                "provider_release_digest": "sha256:" + ("a" if opens == 1 else "d") * 64,
                "provider_manifest_digest": "sha256:" + "b" * 64,
                "provider_qualification_digest": "sha256:" + "c" * 64,
                "provider_l0_contract_hashes": [l0.contract_hash],
            })
            original = backend.callables["restart_service"]

            async def counted(arguments):
                nonlocal writes
                writes += 1
                return await original(arguments)

            backend.callables["restart_service"] = counted
            return backend

        runtime = self.runtime(backend_factory=backend_factory)
        prepared = self.prepare_with(runtime, "lan", "restart_service", {
            "service": "crm", "environment": "staging",
        })
        self.assertEqual(prepared["status"], "plan_ready")
        plan = prepared["plan"]
        outcome = run(runtime.execute(
            plan_id=plan["plan_id"], plan_hash=plan["plan_hash"],
            execution_nonce=prepared["execution_nonce"],
            approval_request_id="release-drift",
            approval_actor="operator",
            allow_destructive=True,
        ))
        self.assertEqual(outcome.state, PlanState.PRECONDITION_CHANGED)
        self.assertEqual(writes, 0)
        self.assertIn("write was not sent", outcome.error)
        inspected = runtime.inspect(plan["plan_id"])
        self.assertEqual(inspected["plan"]["state"], "precondition_changed")
        self.assertTrue(runtime.audit(plan["plan_id"])["ok"])

    def test_provider_deployment_switch_after_approval_is_terminal_without_write(self) -> None:
        opens = 0
        writes = 0
        l0 = L0_SKILLS.for_tool("lan", "restart_service")
        self.assertIsNotNone(l0)

        async def backend_factory(profile):
            nonlocal opens, writes
            opens += 1
            backend = await open_backend(profile)
            metadata = dict(backend.metadata["restart_service"])
            backend.metadata["restart_service"] = metadata
            metadata.update({
                "provider_release_digest": "sha256:" + "a" * 64,
                "provider_manifest_digest": "sha256:" + "b" * 64,
                "provider_qualification_digest": "sha256:" + "c" * 64,
                "provider_deployment_digest": (
                    "sha256:" + ("d" if opens == 1 else "e") * 64
                ),
                "provider_l0_contract_hashes": [l0.contract_hash],
            })
            original = backend.callables["restart_service"]

            async def counted(arguments):
                nonlocal writes
                writes += 1
                return await original(arguments)

            backend.callables["restart_service"] = counted
            return backend

        runtime = self.runtime(backend_factory=backend_factory)
        prepared = self.prepare_with(runtime, "lan", "restart_service", {
            "service": "crm", "environment": "staging",
        })
        plan = prepared["plan"]
        self.assertEqual(plan["provider_deployment_digest"], "sha256:" + "d" * 64)
        outcome = run(runtime.execute(
            plan_id=plan["plan_id"], plan_hash=plan["plan_hash"],
            execution_nonce=prepared["execution_nonce"],
            approval_request_id="deployment-drift",
            approval_actor="operator",
            allow_destructive=True,
        ))
        self.assertEqual(outcome.state, PlanState.PRECONDITION_CHANGED)
        self.assertEqual(writes, 0)
        self.assertTrue(runtime.audit(plan["plan_id"])["ok"])

    def test_plan_hash_nonce_and_terminal_state_are_one_shot(self) -> None:
        prepared = self.prepare("lan", "grant_user_access", {
            "user_id": "erin", "reason": "approved change",
        })
        self.assertEqual(prepared["status"], "plan_ready")
        with self.assertRaises(PlanIntegrityError):
            self.execute(prepared, plan_hash="sha256:" + "0" * 64)
        outcome = self.execute(prepared)
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.state, PlanState.VERIFIED_SUCCESS)
        with self.assertRaises(ApprovalError):
            self.execute(prepared)
        with NetworkJournal(self.journal) as journal:
            with self.assertRaises(StateTransitionError):
                journal.transition(
                    prepared["plan"]["plan_id"], PlanState.ROLLING_BACK, "illegal", {},
                )

    def test_approval_rejection_is_terminal_and_audited(self) -> None:
        prepared = self.prepare("lan", "grant_user_access", {
            "user_id": "erin", "reason": "operator may reject",
        })
        plan = prepared["plan"]
        rejected = self.runtime().reject(
            plan_id=plan["plan_id"], plan_hash=plan["plan_hash"], reason="operator rejected",
        )
        self.assertEqual(rejected["state"], PlanState.REJECTED.value)
        inspected = self.runtime().inspect(plan["plan_id"])
        self.assertEqual(inspected["plan"]["state"], PlanState.REJECTED.value)
        self.assertEqual(inspected["events"][-1]["event_type"], "l0_step_completed")
        self.assertEqual(inspected["events"][-1]["payload"]["step_id"], "audit")
        with self.assertRaises(StateTransitionError):
            self.execute(prepared)

    def test_target_lock_prevents_concurrent_approved_writes(self) -> None:
        first = self.prepare("lan", "grant_user_access", {"user_id": "erin", "reason": "one"})
        second = self.prepare("lan", "revoke_user_access", {"user_id": "erin", "reason": "two"})
        with NetworkJournal(self.journal) as journal:
            journal.claim_for_execution(
                plan_id=first["plan"]["plan_id"], plan_hash=first["plan"]["plan_hash"],
                execution_nonce=first["execution_nonce"], approval_request_id="a1", approval_actor="op1",
            )
            with self.assertRaises(ResourceConflictError):
                journal.claim_for_execution(
                    plan_id=second["plan"]["plan_id"], plan_hash=second["plan"]["plan_hash"],
                    execution_nonce=second["execution_nonce"], approval_request_id="a2", approval_actor="op2",
                )

    def test_execution_aborts_if_target_state_drifted_after_approval(self) -> None:
        from profiles.lan import tools as lan_tools

        writes = 0

        async def backend_factory(profile):
            nonlocal writes
            backend = await open_backend(profile)
            original = backend.callables["grant_user_access"]

            async def counted(arguments):
                nonlocal writes
                writes += 1
                return await original(arguments)

            backend.callables["grant_user_access"] = counted
            return backend

        runtime = self.runtime(backend_factory=backend_factory)
        prepared = self.prepare_with(runtime, "lan", "grant_user_access", {
            "user_id": "erin", "reason": "approved while blocked",
        })
        # A different actor changes the target during the approval window.
        lan_tools._LAN_ACCESS_CHANGES.append({
            "user_id": "erin", "op": "external-change",
            "changes": {"radius": "pass", "dot1x": "authorized", "nac": "compliant", "vlan": 55},
            "reason": "state drift",
        })
        plan = prepared["plan"]
        outcome = run(runtime.execute(
            plan_id=plan["plan_id"], plan_hash=plan["plan_hash"],
            execution_nonce=prepared["execution_nonce"], approval_request_id="drift-approval",
            approval_actor="operator", allow_destructive=True,
        ))
        self.assertEqual(outcome.state, PlanState.PRECONDITION_CHANGED)
        self.assertEqual(writes, 0, "a stale approved plan must not send its write")
        self.assertIn("write was not sent", outcome.error)
        self.assertTrue(all(item.evidence_type == "execution_precondition" for item in outcome.evidence))
        with NetworkJournal(self.journal) as journal:
            self.assertEqual(journal.record(plan["plan_id"])["state"], "precondition_changed")

    def test_dc_config_execution_aborts_on_approved_snapshot_drift(self) -> None:
        from profiles.dc import tools as dc_tools

        writes = 0

        async def backend_factory(profile):
            nonlocal writes
            backend = await open_backend(profile)
            original = backend.callables["dc_config_push"]

            async def counted(arguments):
                nonlocal writes
                writes += 1
                return await original(arguments)

            backend.callables["dc_config_push"] = counted
            return backend

        runtime = self.runtime(backend_factory=backend_factory)
        prepared = self.prepare_with(runtime, "dc", "dc_config_push", {
            "node": "leaf-1", "config_lines": ["mtu 9216"], "reason": "approved state",
        })
        dc_tools._FABRIC_STATE["leaf-1"] = {"applied": ["external change"]}
        plan = prepared["plan"]
        outcome = run(runtime.execute(
            plan_id=plan["plan_id"], plan_hash=plan["plan_hash"],
            execution_nonce=prepared["execution_nonce"], approval_request_id="dc-drift",
            approval_actor="operator", allow_destructive=True,
        ))
        self.assertEqual(outcome.state, PlanState.PRECONDITION_CHANGED)
        self.assertEqual(writes, 0)

    def test_failed_postcondition_triggers_verified_semantic_rollback(self) -> None:
        from profiles.lan import tools as lan_tools

        def inject(stage, _plan):
            if stage == "before_verify":
                lan_tools._LAN_ACCESS_CHANGES.append({
                    "user_id": "erin", "op": "fault",
                    "changes": {"radius": "fail", "dot1x": "rejected", "nac": "quarantine", "vlan": None},
                    "reason": "fault injection",
                })

        runtime = self.runtime(fault_hook=inject)
        prepared = self.prepare_with(runtime, "lan", "grant_user_access", {
            "user_id": "erin", "reason": "rollback test",
        })
        plan = prepared["plan"]
        outcome = run(runtime.execute(
            plan_id=plan["plan_id"], plan_hash=plan["plan_hash"],
            execution_nonce=prepared["execution_nonce"], approval_request_id="rollback-approval",
            approval_actor="operator", allow_destructive=True,
        ))
        self.assertEqual(outcome.state, PlanState.ROLLBACK_VERIFIED)
        self.assertFalse(outcome.ok)
        self.assertTrue(any(item.evidence_type == "rollback" for item in outcome.evidence))
        self.assertIn("BLOCKED at network layer", run(lan_tools.get_user_access({"user_id": "erin"})))
        inspection = runtime.inspect(plan["plan_id"])
        step_events = [
            (event["event_type"], event["payload"].get("step_id"))
            for event in inspection["events"]
            if event["event_type"].startswith("l0_step_")
        ]
        self.assertIn(("l0_step_completed", "compensate"), step_events)
        self.assertEqual(step_events[-1], ("l0_step_completed", "audit"))
        self.assertEqual(
            [item["node_id"] for item in inspection["graph_execution"]["results"]],
            [
                "snapshot", "precheck", "approval", "revalidate", "execute",
                "verify", "compensate", "verify_recovery", "abort",
            ],
        )
        self.assertEqual(
            inspection["graph_execution"]["terminal_phase"], "abort",
        )

    def test_unexpected_pre_effect_runtime_failure_closes_graph_without_effect(self) -> None:
        async def unavailable_backend(_profile):
            raise RuntimeError("provider unavailable")

        runtime = self.runtime(backend_factory=unavailable_backend)
        prepared = self.prepare_with(self.runtime(), "lan", "restart_service", {
            "service": "crm", "environment": "staging",
        })
        plan = prepared["plan"]
        outcome = run(runtime.execute(
            plan_id=plan["plan_id"], plan_hash=plan["plan_hash"],
            execution_nonce=prepared["execution_nonce"],
            approval_request_id="provider-down", approval_actor="operator",
            allow_destructive=True,
        ))
        self.assertEqual(outcome.state, PlanState.MANUAL_INTERVENTION_REQUIRED)
        graph = runtime.inspect(plan["plan_id"])["graph_execution"]
        self.assertTrue(graph["complete"])
        self.assertEqual(graph["terminal_phase"], "abort")
        self.assertNotIn("execute", [item["node_id"] for item in graph["results"]])

    def test_rollback_never_claims_success_without_exact_typed_restoration(self) -> None:
        from profiles.lan import tools as lan_tools

        def inject(stage, _plan):
            if stage == "before_verify":
                lan_tools._LAN_ACCESS_CHANGES.append({
                    "user_id": "carol", "op": "fault",
                    "changes": {"radius": "pass", "dot1x": "authorized", "nac": "compliant", "vlan": 20},
                    "reason": "force verification failure with a non-original VLAN",
                })

        runtime = self.runtime(fault_hook=inject)
        prepared = self.prepare_with(runtime, "lan", "revoke_user_access", {
            "user_id": "carol", "reason": "exact rollback test",
        })
        plan = prepared["plan"]
        outcome = run(runtime.execute(
            plan_id=plan["plan_id"], plan_hash=plan["plan_hash"],
            execution_nonce=prepared["execution_nonce"], approval_request_id="exact-rollback",
            approval_actor="operator", allow_destructive=True,
        ))
        self.assertEqual(outcome.state, PlanState.MANUAL_INTERVENTION_REQUIRED)
        self.assertTrue(any(
            item.evidence_type == "postcondition" and isinstance(item.value, dict)
            for item in outcome.evidence
        ))

    def test_tool_reported_rollback_requires_independent_read(self) -> None:
        async def backend_factory(profile):
            backend = await open_backend(profile)

            async def reports_rollback_without_change(_arguments):
                return "[ROLLED BACK] simulator restored its snapshot"

            backend.callables["edit_device_config"] = reports_rollback_without_change
            return backend

        runtime = self.runtime(backend_factory=backend_factory)
        prepared = self.prepare_with(runtime, "lan", "edit_device_config", {
            "device_id": "ap-01", "config_lines": ["ntp server 10.0.0.7"], "reason": "test",
        })
        plan = prepared["plan"]
        outcome = run(runtime.execute(
            plan_id=plan["plan_id"], plan_hash=plan["plan_hash"],
            execution_nonce=prepared["execution_nonce"], approval_request_id="self-report",
            approval_actor="operator", allow_destructive=True,
        ))
        self.assertEqual(outcome.state, PlanState.ROLLBACK_VERIFIED)
        restoration = [item for item in outcome.evidence if item.evidence_type == "rollback_postcondition"]
        self.assertEqual(len(restoration), 1)
        self.assertTrue(restoration[0].passed)
        self.assertIsInstance(restoration[0].value, dict)

    def test_indeterminate_transport_reconciles_without_retry(self) -> None:
        calls = 0

        async def backend_factory(profile):
            nonlocal calls
            backend = await open_backend(profile)
            original = backend.callables["grant_user_access"]

            async def apply_then_disconnect(arguments):
                nonlocal calls
                calls += 1
                await original(arguments)
                raise OutcomeIndeterminateError("connection lost after device accepted request")

            backend.callables["grant_user_access"] = apply_then_disconnect
            return backend

        runtime = self.runtime(backend_factory=backend_factory)
        prepared = self.prepare_with(runtime, "lan", "grant_user_access", {
            "user_id": "erin", "reason": "disconnect test",
        })
        plan = prepared["plan"]
        outcome = run(runtime.execute(
            plan_id=plan["plan_id"], plan_hash=plan["plan_hash"],
            execution_nonce=prepared["execution_nonce"], approval_request_id="disconnect",
            approval_actor="operator", allow_destructive=True,
        ))
        self.assertEqual(calls, 1, "an indeterminate write must never be retried")
        self.assertEqual(outcome.state, PlanState.VERIFIED_SUCCESS)
        events = self.runtime().inspect(plan["plan_id"])["events"]
        self.assertIn("outcome_indeterminate", [event["to_state"] for event in events])

    def test_unreconciled_indeterminate_outcome_requires_manual_intervention(self) -> None:
        calls = 0

        async def backend_factory(profile):
            nonlocal calls
            backend = await open_backend(profile)

            async def disconnect_before_effect(_arguments):
                nonlocal calls
                calls += 1
                raise OutcomeIndeterminateError("transport did not disclose acceptance")

            backend.callables["grant_user_access"] = disconnect_before_effect
            return backend

        runtime = self.runtime(backend_factory=backend_factory)
        prepared = self.prepare_with(runtime, "lan", "grant_user_access", {
            "user_id": "erin", "reason": "unknown outcome test",
        })
        plan = prepared["plan"]
        outcome = run(runtime.execute(
            plan_id=plan["plan_id"], plan_hash=plan["plan_hash"],
            execution_nonce=prepared["execution_nonce"], approval_request_id="disconnect-unknown",
            approval_actor="operator", allow_destructive=True,
        ))
        self.assertEqual(calls, 1)
        self.assertEqual(outcome.state, PlanState.MANUAL_INTERVENTION_REQUIRED)
        self.assertIn("no retry performed", outcome.error)

    def test_worker_startup_marks_inflight_plan_indeterminate(self) -> None:
        prepared = self.prepare("lan", "grant_user_access", {"user_id": "erin", "reason": "crash"})
        with NetworkJournal(self.journal) as journal:
            journal.claim_for_execution(
                plan_id=prepared["plan"]["plan_id"], plan_hash=prepared["plan"]["plan_hash"],
                execution_nonce=prepared["execution_nonce"], approval_request_id="crash-a",
                approval_actor="operator",
            )
        with NetworkJournal(self.journal, recover_crashed=True) as recovered:
            self.assertEqual(
                recovered.get(prepared["plan"]["plan_id"]).state,
                PlanState.OUTCOME_INDETERMINATE,
            )

    def test_startup_reconciliation_reads_state_without_replaying_write(self) -> None:
        from profiles.lan import tools as lan_tools

        prepared = self.prepare("lan", "grant_user_access", {"user_id": "erin", "reason": "crash"})
        plan = prepared["plan"]
        with NetworkJournal(self.journal) as journal:
            journal.claim_for_execution(
                plan_id=plan["plan_id"], plan_hash=plan["plan_hash"],
                execution_nonce=prepared["execution_nonce"], approval_request_id="crash-a",
                approval_actor="operator",
            )
        # The simulator accepted the write, then the worker died before it
        # could journal verification. Startup is only allowed to re-read.
        run(lan_tools.grant_user_access({"user_id": "erin", "reason": "accepted before crash"}))
        with NetworkJournal(self.journal, recover_crashed=True):
            pass
        outcomes = run(self.runtime().recover_inflight())
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0]["state"], PlanState.VERIFIED_SUCCESS.value)
        inspection = self.runtime().inspect(plan["plan_id"])
        events = inspection["events"]
        self.assertEqual(sum(event["event_type"] == "execution_started" for event in events), 1)
        graph = inspection["graph_execution"]
        self.assertTrue(graph["complete"])
        self.assertEqual(graph["terminal_phase"], "commit")
        self.assertEqual(
            [
                (item["node_id"], item["outcome"])
                for item in graph["results"]
            ],
            [
                ("snapshot", "succeeded"),
                ("precheck", "succeeded"),
                ("approval", "succeeded"),
                ("revalidate", "skipped"),
                ("execute", "indeterminate"),
                ("reconcile", "succeeded"),
                ("verify", "succeeded"),
                ("commit", "succeeded"),
            ],
        )

    def test_compiled_skill_workflow_blocks_skipped_or_cross_skill_write_steps(self) -> None:
        templates = compile_workflow_templates("lan", "mock")
        self.assertIn("lan-new-employee-onboarding-access", templates)
        self.assertIn("restart-service", templates)
        with WorkflowRuntime(self.journal) as workflows:
            started = workflows.start(
                session_id="workflow-session", profile="lan", mode="mock",
                skill_name="lan-new-employee-onboarding-access",
            )
        self.assertTrue(started["active"])
        skipped = self.prepare_with(self.runtime(),
            "lan", "grant_user_access", {"user_id": "erin", "reason": "skipped checks"},
            session_id="workflow-session",
        )
        self.assertEqual(skipped["status"], "rejected")
        self.assertTrue(any("workflow prerequisites" in item for item in skipped["errors"]))
        cross_skill = self.prepare_with(self.runtime(),
            "lan", "restart_service", {"service": "crm", "environment": "staging"},
            session_id="workflow-session",
        )
        self.assertEqual(cross_skill["status"], "rejected")
        self.assertTrue(any("does not allow write" in item for item in cross_skill["errors"]))

    def test_compiled_skill_workflow_binds_verified_prerequisites_into_plan(self) -> None:
        session = "workflow-happy"
        with WorkflowRuntime(self.journal) as workflows:
            started = workflows.start(
                session_id=session, profile="lan", mode="mock",
                skill_name="lan-new-employee-onboarding-access",
            )
        access = run(self.runtime().invoke_read("lan", "get_user_access", {"user_id": "erin"}))
        nac = run(self.runtime().invoke_read("lan", "check_nac_policy", {"user_id": "erin"}))
        with WorkflowRuntime(self.journal) as workflows:
            workflows.observe(
                session_id=session, tool_name="get_user_access", arguments={"user_id": "erin"},
                result=access, success=True, mutating=False,
            )
            workflows.observe(
                session_id=session, tool_name="check_nac_policy", arguments={"user_id": "erin"},
                result=nac, success=True, mutating=False,
            )
        prepared = self.prepare_with(self.runtime(),
            "lan", "grant_user_access", {"user_id": "erin", "reason": "onboarding"},
            session_id=session,
        )
        self.assertEqual(prepared["status"], "plan_ready", prepared)
        self.assertEqual(prepared["plan"]["workflow_run_id"], started["run_id"])
        self.assertEqual(
            prepared["plan"]["workflow_template_hash"],
            started["template"]["template_hash"],
        )
        outcome = self.execute(prepared)
        self.assertEqual(outcome.state, PlanState.VERIFIED_SUCCESS)

    def test_workflow_observations_are_target_bound(self) -> None:
        session = "workflow-target-binding"
        with WorkflowRuntime(self.journal) as workflows:
            workflows.start(
                session_id=session, profile="lan", mode="mock",
                skill_name="lan-user-access-diagnose",
            )
            workflows.observe(
                session_id=session, tool_name="get_user_access", arguments={"user_id": "erin"},
                result="network admission: ❌ BLOCKED at network layer", success=True, mutating=False,
            )
            workflows.observe(
                session_id=session, tool_name="check_nac_policy", arguments={"user_id": "erin"},
                result="result         : DENY", success=True, mutating=False,
            )
        wrong_target = self.prepare_with(self.runtime(),
            "lan", "grant_user_access", {"user_id": "alice", "reason": "wrong target"},
            session_id=session,
        )
        self.assertEqual(wrong_target["status"], "rejected")

    def test_all_local_write_contracts_prepare_and_verify(self) -> None:
        samples = [
            ("lan", "edit_device_config", {"device_id": "ap-01", "config_lines": ["radius-server timeout 3"], "reason": "test"}),
            ("lan", "restart_service", {"service": "crm", "environment": "staging"}),
            ("lan", "push_config", {"device_id": "ap-02", "config_text": "ntp server 10.0.0.8"}),
            ("lan", "rollback_deploy", {"deploy_id": "deploy-1"}),
            ("lan", "drain_node", {"node_id": "node-1"}),
            ("lan", "failover", {"resource_id": "db-1", "target": "db-2"}),
            ("lan", "delete_resource", {"resource_id": "unused-1"}),
            ("lan", "rollback_service", {"service": "billing", "version": "3.2.1", "environment": "staging"}),
            ("lan", "grant_user_access", {"user_id": "erin", "reason": "test"}),
            ("lan", "revoke_user_access", {"user_id": "alice", "reason": "test"}),
            ("dc", "dc_config_push", {"node": "leaf-1", "config_lines": ["interface Ethernet1/1", "mtu 9216"], "reason": "test"}),
            ("dc", "dc_grant_app_access", {"user_id": "alice", "app_id": "crm", "role": "sales-rep", "reason": "test"}),
            ("dc", "dc_revoke_app_access", {"user_id": "bob", "app_id": "crm", "reason": "test"}),
            ("wan", "wan_failover_path", {"tunnel": "tun-sf-dc", "to_transport": "lte"}),
        ]
        runtime = self.runtime()
        for index, (profile, tool, arguments) in enumerate(samples):
            with self.subTest(tool=tool):
                prepared = self.prepare_with(runtime, profile, tool, arguments)
                self.assertEqual(prepared["status"], "plan_ready", prepared)
                plan = prepared["plan"]
                outcome = run(runtime.execute(
                    plan_id=plan["plan_id"], plan_hash=plan["plan_hash"],
                    execution_nonce=prepared["execution_nonce"],
                    approval_request_id=f"matrix-{index}", approval_actor="matrix-operator",
                    allow_destructive=True,
                ))
                self.assertEqual(outcome.state, PlanState.VERIFIED_SUCCESS, outcome.to_dict())
                self.assertTrue(outcome.evidence)

    def test_all_reviewed_write_contracts_have_registered_runtime_handlers(self) -> None:
        for tool_name, contract in reviewed_contracts().items():
            with self.subTest(tool=tool_name):
                self.assertIn(contract.verifier, VERIFIERS.contract_ids())
                if contract.rollback_tool:
                    self.assertIsNotNone(contract.compensator)
                    self.assertIn(contract.compensator, COMPENSATORS.contract_ids())

    def test_event_hash_chain_detects_sqlite_payload_tampering(self) -> None:
        prepared = self.prepare("lan", "grant_user_access", {
            "user_id": "erin", "reason": "audit chain",
        })
        outcome = self.execute(prepared)
        self.assertEqual(outcome.state, PlanState.VERIFIED_SUCCESS)
        plan_id = prepared["plan"]["plan_id"]
        audit = self.runtime().audit(plan_id)
        self.assertTrue(audit["ok"], audit)
        self.assertGreaterEqual(audit["events"], 4)

        with sqlite3.connect(self.journal) as database:
            database.execute(
                "UPDATE plan_events SET payload_json=? WHERE plan_id=? AND event_id=("
                "SELECT MIN(event_id) FROM plan_events WHERE plan_id=?)",
                ('{"tampered":true}', plan_id, plan_id),
            )
            database.commit()
        damaged = self.runtime().audit(plan_id)
        self.assertFalse(damaged["ok"])
        self.assertTrue(any(
            item["error"] == "event_hash_mismatch" for item in damaged["errors"]
        ))


if __name__ == "__main__":
    unittest.main()
