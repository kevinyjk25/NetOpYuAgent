from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from hermes_adapter.comparison import FakeHermesContext, InProcessWorkerClient, run_comparison
from hermes_adapter.pending import PendingActions
from hermes_adapter.plugin import HermesAdapterConfig, NetOpYuHermesAdapter


class _NoCommandContext:
    def register_tool(self, **definition):
        del definition


class _RemoteClient:
    def __init__(self):
        self.resumed = None

    def request(self, command, *, args=None, **fields):
        del fields
        if command == "a2a-delegate" and not (args or {}).get("resume_interrupt_id"):
            return {
                "ok": False,
                "status": "input-required",
                "peer": "dc-agent",
                "interrupt_id": "remote-interrupt-1",
                "approval": {
                    "kind": "network-l0-plan",
                    "plan_id": "remote-plan",
                    "plan_hash": "sha256:" + "a" * 64,
                    "tool_name": "dc_grant_app_access",
                },
            }
        if command == "a2a-delegate":
            self.resumed = dict(args or {})
            return {"ok": True, "status": "completed", "text": "verified remote success"}
        raise AssertionError(command)


class HermesAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.previous = {
            key: os.environ.get(key)
            for key in (
                "NETOPYU_BACKEND", "NETOPYU_NETWORK_RUNTIME_STORE",
                "NETOPYU_TOOL_RESULT_STORE",
            )
        }
        os.environ["NETOPYU_BACKEND"] = "mock"
        os.environ["NETOPYU_NETWORK_RUNTIME_STORE"] = str(root / "runtime.sqlite")
        os.environ["NETOPYU_TOOL_RESULT_STORE"] = str(root / "results.sqlite")
        from profiles.lan import tools as lan_tools
        lan_tools._LAN_ACCESS_CHANGES.clear()
        self.config = HermesAdapterConfig(
            profile="lan",
            socket_path=root / "unused.sock",
            include_destructive=True,
            operator_id="local:test-operator",
            own_agent_id="hermes-lan",
            peer_urls=(),
            timeout_seconds=30,
        )

    def tearDown(self) -> None:
        for key, value in self.previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.temp.cleanup()

    def adapter(self):
        adapter = NetOpYuHermesAdapter(InProcessWorkerClient(), self.config)
        context = FakeHermesContext()
        adapter.register(context)
        return adapter, context

    def test_official_plugin_surface_projects_profile_tools_and_skills(self):
        adapter, context = self.adapter()
        self.assertEqual(adapter.manifest["profile"], "lan")
        self.assertIn("get_user_access", context.tools)
        self.assertIn("restart_service", context.tools)
        self.assertIn("netopyu_runtime_audit", context.tools)
        self.assertIn("netopyu_delegate", context.tools)
        self.assertIn("netopyu_skill_view", context.tools)
        self.assertIn("netopyu_capability_search", context.tools)
        self.assertIn("netopyu_memory_recall", context.tools)
        self.assertIn("lan-new-employee-onboarding-access", context.skills)
        self.assertTrue(
            context.skills["lan-new-employee-onboarding-access"].endswith("SKILL.md")
        )
        self.assertIn("netopyu-approve", context.commands)
        self.assertIn("pre_tool_call", context.hooks)

    def test_read_tool_uses_same_strict_runtime_projection(self):
        _, context = self.adapter()
        result = json.loads(context.tools["get_user_access"]["handler"](
            {"user_id": "alice"}, task_id="hermes-read",
        ))
        self.assertTrue(result["ok"])
        self.assertIn("alice", result["result"].lower())

    def test_write_only_prepares_until_exact_user_slash_command(self):
        _, context = self.adapter()
        text = context.tools["restart_service"]["handler"](
            {"service": "crm", "environment": "staging"}, task_id="hermes-write",
        )
        prepared = json.loads(text)
        self.assertEqual(prepared["status"], "approval_required")
        self.assertFalse(prepared["executed"])
        self.assertNotIn("execution_nonce", text)
        self.assertEqual(prepared["plan"]["l0_skill_id"], "network.service.restart")

        command_name, raw = prepared["approval"]["command"][1:].split(" ", 1)
        plan_id, plan_hash = raw.split()
        wrong = json.loads(context.commands[command_name]["handler"](
            f"{plan_id} sha256:{'0' * 64}",
        ))
        self.assertFalse(wrong["ok"])
        outcome = json.loads(context.commands[command_name]["handler"](f"{plan_id} {plan_hash}"))
        self.assertTrue(outcome["ok"])
        self.assertEqual(outcome["state"], "verified_success")

        duplicate = json.loads(context.commands[command_name]["handler"](f"{plan_id} {plan_hash}"))
        self.assertFalse(duplicate["ok"])

    def test_pending_nonce_is_lost_safely_on_adapter_restart(self):
        adapter, context = self.adapter()
        prepared = json.loads(context.tools["restart_service"]["handler"](
            {"service": "crm", "environment": "staging"}, task_id="restart-loss",
        ))
        plan = prepared["plan"]
        replacement = NetOpYuHermesAdapter(
            InProcessWorkerClient(), self.config, pending=PendingActions(),
        )
        denied = json.loads(replacement.approve(f"{plan['plan_id']} {plan['plan_hash']}"))
        self.assertFalse(denied["ok"])
        inspected = adapter.client.request(
            "runtime-inspect", args={"plan_id": plan["plan_id"]}, profile="lan",
        )
        self.assertEqual(inspected["plan"]["state"], "plan_ready")

    def test_mutation_projection_fails_closed_without_slash_command_api(self):
        adapter = NetOpYuHermesAdapter(InProcessWorkerClient(), self.config)
        with self.assertRaises(RuntimeError):
            adapter.register(_NoCommandContext())

    def test_remote_network_plan_also_requires_exact_slash_approval(self):
        client = _RemoteClient()
        adapter = NetOpYuHermesAdapter(client, self.config)
        delegated = json.loads(adapter._delegate({
            "description": "grant app access",
            "prompt": "self-contained request",
            "target": "dc-agent",
        }, task_id="remote-test"))
        self.assertEqual(delegated["status"], "input-required")
        self.assertIn("/netopyu-a2a-approve", delegated["approval_command"])
        command, raw = delegated["approval_command"][1:].split(" ", 1)
        self.assertEqual(command, "netopyu-a2a-approve")
        outcome = json.loads(adapter.approve_remote(raw))
        self.assertTrue(outcome["ok"])
        self.assertEqual(client.resumed["operator_decision"], "approve")

    def test_hermes_skill_view_enforces_reviewed_l1_observations_before_l0(self):
        _, context = self.adapter()
        task_id = "hermes-reviewed-workflow"
        loaded = json.loads(context.tools["netopyu_skill_view"]["handler"](
            {"name": "lan-new-employee-onboarding-access"}, task_id=task_id,
        ))
        self.assertTrue(loaded["ok"])
        self.assertTrue(loaded["workflow"]["active"])

        blocked = json.loads(context.tools["grant_user_access"]["handler"](
            {"user_id": "erin", "reason": "local adapter workflow test"},
            task_id=task_id,
        ))
        self.assertEqual(blocked["status"], "rejected")
        self.assertIn("workflow prerequisites", blocked["errors"][0])

        access = json.loads(context.tools["get_user_access"]["handler"](
            {"user_id": "erin"}, task_id=task_id,
        ))
        policy = json.loads(context.tools["check_nac_policy"]["handler"](
            {"user_id": "erin"}, task_id=task_id,
        ))
        self.assertTrue(access["ok"])
        self.assertTrue(policy["ok"])
        prepared = json.loads(context.tools["grant_user_access"]["handler"](
            {"user_id": "erin", "reason": "local adapter workflow test"},
            task_id=task_id,
        ))
        self.assertEqual(prepared["status"], "approval_required")
        self.assertIsNotNone(prepared["plan"]["workflow_run_id"])
        _, raw = prepared["approval"]["deny_command"][1:].split(" ", 1)
        denied = json.loads(context.commands["netopyu-deny"]["handler"](raw))
        self.assertEqual(denied["state"], "rejected")

    def test_builtin_hermes_skill_view_hook_starts_reviewed_workflow(self):
        _, context = self.adapter()
        context.hooks["pre_tool_call"][0](
            tool_name="skill_view",
            args={"name": "netopyu:restart-service"},
            task_id="builtin-skill-view",
        )
        prepared = json.loads(context.tools["restart_service"]["handler"](
            {"service": "crm", "environment": "staging"},
            task_id="builtin-skill-view",
        ))
        self.assertEqual(prepared["status"], "approval_required")
        self.assertIsNotNone(prepared["plan"]["workflow_run_id"])

    def test_dsh_and_hermes_keep_network_runtime_invariants_equal(self):
        result = run_comparison()
        self.assertTrue(result["ok"], result)
        self.assertTrue(result["runtime_invariants_equal"])
        self.assertFalse(result["hermes"]["nonce_exposed_to_model"])
        self.assertTrue(result["hermes"]["duplicate_blocked"])


if __name__ == "__main__":
    unittest.main()
