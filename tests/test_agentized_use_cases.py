from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dsh_adapter.agentized_authoring import (
    L1_TEMPLATE,
    authoring_template,
    authoring_trace,
    capture_authoring,
    submit_authoring,
)
from dsh_adapter.skills import build_skill_manifest
from network_runtime.l0.promotion import PromotionError


def reviewed_translation() -> dict:
    argument = lambda name: "${arguments." + name + "}"
    return {
        "l0_id": "network.lan.user-access.grant.agent-demo",
        "profiles": ["lan"],
        "parameters": {
            "user_id": {
                "type": "string", "required": True, "maxLength": 128,
            },
            "reason": {
                "type": "string", "required": True, "minLength": 1,
                "maxLength": 512,
            },
        },
        "effect_capability": "network.lan.user-access.grant",
        "observation_capability": "network.access.user.get",
        "verification_capability": "netopyu.verifier.lan-access-granted",
        "compensation_capability": "network.lan.user-access.revoke",
        "compensation_verification_capability": "netopyu.rollback-verifier.inverse-tool-v1",
        "effect_request": {
            "user_id": argument("user_id"), "reason": argument("reason"),
        },
        "intent": {
            "kind": "grant_network_access",
            "target_fields": ["user_id"],
            "desired_state": {"admitted": True},
        },
        "preflight": {
            "arguments": {"user_id": argument("user_id")},
            "snapshot_fields": ["facts"],
            "predicates": [
                {"field": "facts", "operator": "exists"},
                {"field": "facts.status", "operator": "equals", "expected": "active"},
            ],
        },
        "verification_arguments": {"user_id": argument("user_id")},
        "verification_predicates": [
            {"field": "passed", "operator": "equals", "expected": True},
        ],
        "compensation_arguments": {"user_id": argument("user_id")},
        "compensation_verification_arguments": {"user_id": argument("user_id")},
        "compensation_verification_predicates": [
            {"field": "restored", "operator": "equals", "expected": True},
        ],
        "risk": "high",
        "approval_mode": "single",
        "translation_logic": [
            "user_id is the immutable target and reason is mandatory audit context",
            "the effect is independently observed before and after the write",
            "verification failure restores the exact prior admission state",
        ],
    }


class AgentizedAuthoringTests(unittest.TestCase):
    def test_template_separates_model_and_runtime_authority(self) -> None:
        value = authoring_template()
        self.assertIn("model_must_translate", value)
        self.assertIn("runtime_owns", value)
        self.assertFalse(value["activation"]["automatic"])
        self.assertIn("network.lan.user-access.grant", {
            item["id"] for item in value["trusted_capabilities"]
        })

    def test_agent_translation_packages_visible_non_executable_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"NETOPYU_L0_PROPOSALS_DIR": directory}, clear=False,
        ):
            value = submit_authoring({
                "catalog_id": "lan-user-access",
                "skill_markdown": L1_TEMPLATE,
                "translation": reviewed_translation(),
            })
            trace = authoring_trace(value["attempt_id"])
            proposal = Path(value["proposal_directory"])
            stages = [
                item["stage"] for item in __import__("json").loads(
                    (proposal / "trajectory.json").read_text(encoding="utf-8")
                )["stages"]
            ]
            artifact_paths = value["artifact_paths"]
            trace_paths = trace["artifact_paths"]
            self.assertEqual(artifact_paths, trace_paths)
            self.assertTrue(all(Path(path).exists() for path in artifact_paths.values()))
            self.assertEqual(
                Path(artifact_paths["promotion_trajectory"]).name,
                "trajectory.json",
            )
            self.assertEqual(
                Path(artifact_paths["semantic_review_workbench"]).name,
                "semantic-review.html",
            )
            self.assertNotIn(
                "proposal-trajectory.md", "\n".join(artifact_paths.values()),
            )
        self.assertTrue(value["ok"])
        self.assertEqual(value["status"], "ready_for_review")
        self.assertFalse(value["auto_activated"])
        self.assertFalse(trace["activation"]["execution_authority"])
        self.assertEqual(trace["runtime_stage"]["validation_status"], "ready_for_review")
        self.assertEqual(trace["runtime_stage"]["semantic_coverage"]["gate"], "passed")
        self.assertEqual(value["semantic_coverage"]["summary"]["blockingRequirements"], 0)
        self.assertEqual(stages, ["L1", "L0.5", "L0-authoring", "L0-compiled"])

    def test_captured_source_and_flat_small_model_schema_build_same_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"NETOPYU_L0_PROPOSALS_DIR": directory}, clear=False,
        ):
            captured = capture_authoring({"skill_markdown": L1_TEMPLATE})
            value = submit_authoring({
                "draft_id": captured["draft_id"],
                "catalog_id": "lan-user-access",
                **reviewed_translation(),
            })
            trace = authoring_trace(captured["draft_id"])
        self.assertTrue(value["ok"])
        self.assertEqual(value["attempt_id"], captured["draft_id"])
        self.assertEqual(trace["runtime_stage"]["validation_status"], "ready_for_review")
        self.assertFalse(trace["activation"]["automatic"])

    def test_legacy_trace_gets_only_existing_authoritative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"NETOPYU_L0_PROPOSALS_DIR": directory}, clear=False,
        ):
            value = submit_authoring({
                "skill_markdown": L1_TEMPLATE,
                "translation": reviewed_translation(),
            })
            trace_path = Path(directory) / value["attempt_id"] / "agent-trace.json"
            legacy = __import__("json").loads(trace_path.read_text(encoding="utf-8"))
            legacy.pop("artifact_paths")
            trace_path.write_text(__import__("json").dumps(legacy), encoding="utf-8")
            trace = authoring_trace(value["attempt_id"])
            self.assertTrue(trace["path_reporting_policy"]["inferred_paths_forbidden"])
            self.assertTrue(all(
                Path(path).exists() for path in trace["artifact_paths"].values()
            ))
            self.assertEqual(
                Path(trace["artifact_paths"]["promotion_trajectory"]).name,
                "trajectory.json",
            )

    def test_invented_capability_is_rejected_before_compilation(self) -> None:
        translation = reviewed_translation()
        translation["effect_capability"] = "model.invented.superuser.grant"
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"NETOPYU_L0_PROPOSALS_DIR": directory}, clear=False,
        ):
            with self.assertRaisesRegex(PromotionError, "invents capability"):
                submit_authoring({
                    "skill_markdown": L1_TEMPLATE, "translation": translation,
                })


class AgentizedSkillProjectionTests(unittest.TestCase):
    def test_mock_runtime_projects_real_agent_remediation_and_authoring(self) -> None:
        names = {item["name"] for item in build_skill_manifest("lan", "mock")["skills"]}
        self.assertIn("agentized-lan-access-remediation", names)
        self.assertIn("l1-to-l0-agent-authoring", names)

    def test_service_only_runtime_projects_external_mcp_agent_without_lab_skill(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with patch.dict("os.environ", {
            "NETOPYU_CONFIG_PATH": str(root / "config.service-lab.yaml"),
        }, clear=False):
            names = {
                item["name"]
                for item in build_skill_manifest("lan", "pragmatic")["skills"]
            }
        self.assertIn("enterprise-access-mcp-agent", names)
        self.assertIn("l1-to-l0-agent-authoring", names)
        self.assertNotIn("service-network-access-reconcile", names)


if __name__ == "__main__":
    unittest.main(verbosity=2)
