from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from effect_runtime import (
    EffectSemantics,
    ModelConfidence,
    RiskTier,
    SkillEdge,
    SkillLevel,
    SkillNode,
    decide_progressive_execution,
    build_skill_disclosure_packet,
    inspect_skill_package,
    validate_skill_graph,
)
from evaluation.progressive_skill_suite import evaluate_progressive_skill_suite
from evaluation.progressive_determinization import build_progressive_report


def _assessment(score: float = 95.0, **summary_overrides: object) -> dict[str, object]:
    summary: dict[str, object] = {
        "averageMappingConfidence": score,
        "averageL1ToL05Confidence": score,
        "averageL05ToL0Confidence": score,
        "blockingRequirements": 0,
        "missing": 0,
        "ambiguous": 0,
    }
    summary.update(summary_overrides)
    return {
        "schema": "netopyu.io/l0-promotion-report/v2",
        "status": "ready_for_review",
        "findings": [],
        "semanticCoverage": {"gate": "passed", "summary": summary},
    }


def _package(gate: str = "passed", coverage: float = 100.0) -> dict[str, object]:
    return {
        "schema": "effect-runtime.io/skill-package-report/v1",
        "gate": gate,
        "summary": {"referenceCoveragePercent": coverage},
    }


class SkillPackageTests(unittest.TestCase):
    def _write_skill(self, root: Path, *, script: str = "print('ok')\n") -> Path:
        (root / "scripts").mkdir()
        (root / "references").mkdir()
        (root / "SKILL.md").write_text(
            """---
name: portable-change
description: Apply one portable reviewed change.
metadata:
  effect-runtime-script-roles: scripts/apply.py=provider_adapter
---
# Portable change

Read [the contract](references/contract.md), then use `scripts/apply.py`.
""",
            encoding="utf-8",
        )
        (root / "references" / "contract.md").write_text("# Contract\n", encoding="utf-8")
        (root / "scripts" / "apply.py").write_text(script, encoding="utf-8")
        return root

    def test_package_is_hashed_and_resource_graph_is_visible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._write_skill(Path(directory))
            report = inspect_skill_package(
                root, bound_scripts=["scripts/apply.py=generic.change.apply"],
            )
        self.assertEqual(report["gate"], "passed")
        self.assertTrue(str(report["packageDigest"]).startswith("sha256:"))
        self.assertEqual(report["summary"]["referenceCoveragePercent"], 100.0)
        self.assertEqual(len(report["referenceGraph"]), 2)
        script = next(item for item in report["resources"] if item["kind"] == "scripts")
        self.assertEqual(script["script_role"], "provider_adapter")
        self.assertTrue(script["capability_bound"])
        self.assertEqual(script["capability_binding"], "generic.change.apply")

    def test_disclosure_packet_contains_only_reachable_resources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._write_skill(Path(directory))
            root.joinpath("assets").mkdir()
            root.joinpath("assets", "unused.txt").write_text("unused", encoding="utf-8")
            packet = build_skill_disclosure_packet(
                root, bound_scripts=["scripts/apply.py=generic.change.apply"],
            )
        self.assertEqual(packet["gate"], "passed")
        self.assertEqual(
            {item["path"] for item in packet["resources"]},
            {"references/contract.md", "scripts/apply.py"},
        )
        self.assertTrue(all(item["executable"] is False for item in packet["resources"]))

    def test_unbound_effect_script_fails_closed_without_execution(self) -> None:
        marker: Path
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "executed"
            self._write_skill(
                root,
                script=f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n",
            )
            report = inspect_skill_package(root)
            self.assertFalse(marker.exists())
        self.assertEqual(report["gate"], "blocked")
        self.assertIn("SCRIPT_EFFECT_UNBOUND", {item["code"] for item in report["findings"]})

    def test_script_path_without_capability_id_is_not_a_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._write_skill(Path(directory))
            report = inspect_skill_package(
                root, bound_scripts=["scripts/apply.py"],
            )
        self.assertEqual(report["gate"], "blocked")
        self.assertIn("SCRIPT_BINDING_INVALID", {
            item["code"] for item in report["findings"]
        })

    def test_missing_resource_and_path_traversal_are_hard_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.joinpath("SKILL.md").write_text(
                """---
name: broken-package
description: Broken package used to test gates.
---
[missing](references/no.md)
[escape](../../outside.md)
""",
                encoding="utf-8",
            )
            report = inspect_skill_package(root)
        codes = {item["code"] for item in report["findings"]}
        self.assertEqual(report["gate"], "blocked")
        self.assertIn("RESOURCE_REFERENCE_MISSING", codes)
        self.assertIn("RESOURCE_REFERENCE_UNSAFE", codes)


class ProgressiveDecisionTests(unittest.TestCase):
    def _decide(self, **overrides: object) -> dict[str, object]:
        arguments: dict[str, object] = {
            "assessment": _assessment(),
            "package_report": _package(),
            "risk": RiskTier.MEDIUM,
            "effect_semantics": EffectSemantics.REVERSIBLE,
            "l0_active": True,
            "l0_artifact_digest": "sha256:" + "a" * 64,
            "repeat_stability": 1.0,
            "simulation_pass_rate": 1.0,
        }
        arguments.update(overrides)
        return decide_progressive_execution(**arguments)  # type: ignore[arg-type]

    def test_active_l0_routes_writes_only_through_runtime(self) -> None:
        decision = self._decide()
        self.assertEqual(decision["route"], "l0_runtime")
        self.assertFalse(decision["controls"]["l1DirectWriteAllowed"])
        self.assertFalse(decision["controls"]["autoActivationAllowed"])
        self.assertFalse(decision["controls"]["autoExecutionAllowed"])

    def test_low_confidence_write_becomes_proposal_not_direct_l1_write(self) -> None:
        decision = self._decide(assessment=_assessment(45.0))
        self.assertEqual(decision["route"], "proposal_only")
        self.assertFalse(decision["controls"]["l1DirectWriteAllowed"])

    def test_low_confidence_read_can_stay_in_read_only_l1(self) -> None:
        decision = self._decide(
            assessment=_assessment(45.0),
            effect_semantics=EffectSemantics.READ_ONLY,
            l0_active=False,
        )
        self.assertEqual(decision["route"], "l1_read_only")

    def test_l1_can_orchestrate_only_explicit_active_l0_references(self) -> None:
        decision = self._decide(
            assessment=_assessment(75.0),
            l0_active=False,
            referenced_l0=("inventory.read@1.2.0#sha256:" + "b" * 64,),
        )
        self.assertEqual(decision["route"], "hybrid_l1_l0")

    def test_semantic_gap_requires_clarification(self) -> None:
        decision = self._decide(assessment=_assessment(95.0, missing=1))
        self.assertEqual(decision["route"], "clarification_required")

    def test_package_hard_gate_precedes_confidence(self) -> None:
        decision = self._decide(package_report=_package("blocked"))
        self.assertEqual(decision["route"], "blocked")
        self.assertIn("PACKAGE_GATE_BLOCKED", decision["hardGate"]["failures"])

    def test_active_boolean_without_digest_cannot_route_to_runtime(self) -> None:
        decision = self._decide(l0_artifact_digest=None)
        self.assertEqual(decision["route"], "proposal_only")
        self.assertIn("ACTIVE_L0_DIGEST_MISSING", {
            item["code"] for item in decision["findings"]
        })

    def test_write_requires_repeat_and_simulation_evidence(self) -> None:
        decision = self._decide(repeat_stability=None, simulation_pass_rate=None)
        self.assertEqual(decision["route"], "proposal_only")

    def test_privileged_effect_requires_review_and_approval_control(self) -> None:
        blocked = self._decide(
            risk=RiskTier.HIGH,
            effect_semantics=EffectSemantics.DESTRUCTIVE,
            assessment=_assessment(100.0),
        )
        allowed = self._decide(
            risk=RiskTier.HIGH,
            effect_semantics=EffectSemantics.DESTRUCTIVE,
            assessment=_assessment(100.0),
            activation_reviewed=True,
            approval_control_available=True,
        )
        self.assertEqual(blocked["route"], "proposal_only")
        self.assertEqual(allowed["route"], "l0_runtime")

    def test_uncalibrated_model_score_is_excluded(self) -> None:
        decision = self._decide(
            assessment=_assessment(45.0),
            model_confidence=ModelConfidence(score=1.0, calibrated=False),
        )
        self.assertEqual(decision["route"], "proposal_only")
        self.assertIn("MODEL_SIGNAL_EXCLUDED", {item["code"] for item in decision["findings"]})
        self.assertNotIn(
            "calibrated_model_judge",
            {item["name"] for item in decision["confidence"]["signals"]},
        )


class SkillGraphTests(unittest.TestCase):
    def test_l1_to_active_l0_is_allowed(self) -> None:
        report = validate_skill_graph(
            [
                SkillNode("diagnose", SkillLevel.L1),
                SkillNode(
                    "inventory.read", SkillLevel.L0, active=True,
                    version="1.2.0", artifact_digest="sha256:" + "a" * 64,
                ),
            ],
            [SkillEdge("diagnose", "inventory.read")],
        )
        self.assertEqual(report["gate"], "passed")

    def test_l0_to_l1_is_forbidden(self) -> None:
        report = validate_skill_graph(
            [
                SkillNode(
                    "change.apply", SkillLevel.L0, active=True,
                    version="1.0.0", artifact_digest="sha256:" + "a" * 64,
                ),
                SkillNode("replan", SkillLevel.L1),
            ],
            [SkillEdge("change.apply", "replan")],
        )
        self.assertEqual(report["gate"], "blocked")
        self.assertIn("L0_TO_L1_FORBIDDEN", {item["code"] for item in report["findings"]})

    def test_active_l0_reference_requires_version_and_digest(self) -> None:
        report = validate_skill_graph(
            [
                SkillNode("diagnose", SkillLevel.L1),
                SkillNode("inventory.read", SkillLevel.L0, active=True),
            ],
            [SkillEdge("diagnose", "inventory.read")],
        )
        self.assertEqual(report["gate"], "blocked")
        self.assertIn("L0_ARTIFACT_BINDING_MISSING", {
            item["code"] for item in report["findings"]
        })


class HeterogeneousAnthropicSkillSuiteTests(unittest.TestCase):
    def test_fixture_suite_covers_common_skill_structures(self) -> None:
        report = evaluate_progressive_skill_suite()
        self.assertEqual(report["summary"]["cases"], 10)
        self.assertEqual(report["summary"]["failed"], 0)
        self.assertGreaterEqual(report["summary"]["domains"], 7)
        self.assertTrue({
            "reference", "approval", "condition-branch", "multi-step", "script",
            "verification", "compensation", "l1-l0-composition", "negative-gate",
        }.issubset(set(report["features"])))

    def test_network_anchor_uses_the_same_generic_policy(self) -> None:
        report = build_progressive_report()
        anchor = report["networkAnchor"]
        self.assertEqual(anchor["semanticGate"], "passed")
        self.assertEqual(anchor["packageGate"], "passed")
        self.assertEqual(anchor["decision"]["route"], "l0_runtime")
        self.assertFalse(anchor["decision"]["controls"]["l1DirectWriteAllowed"])


if __name__ == "__main__":
    unittest.main()
