from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from network_runtime.l0.promotion import (
    PromotionError,
    assess_promotion,
    build_l05_spec,
    inspect_skill,
    l05_yaml,
    package_promotion,
    promotion_prompt,
    review_promotion,
)


ROOT = Path(__file__).resolve().parents[1]
PROMOTION = ROOT / "network_runtime" / "l0" / "promotion_examples" / "url1-network-access"
SKILL = PROMOTION / "SKILL.md"
CAPABILITIES = PROMOTION / "capabilities.yaml"
L05 = PROMOTION / "L0.5.yaml"
CANDIDATE = ROOT / "network_runtime" / "l0" / "examples" / "s1-network-access-grant.yaml"


class L0PromotionTests(unittest.TestCase):
    def test_inspect_and_prompt_bind_official_skill_and_trusted_catalog(self) -> None:
        inspected = inspect_skill(SKILL)
        self.assertEqual(inspected["name"], "url1-network-access")
        self.assertEqual(
            inspected["declared_tools"], ["url1_grant_network_access"],
        )
        packet = json.loads(promotion_prompt(
            skill_path=SKILL, capability_catalog_path=CAPABILITIES, l05_path=L05,
        ))
        self.assertEqual(packet["sourceSkill"]["sha256"], inspected["sha256"])
        self.assertIn("atomic", packet["outputSchemas"])
        self.assertEqual(
            packet["structuredSkill"]["document"]["apiVersion"],
            "netopyu.io/l0.5-structured-skill/v1",
        )
        self.assertIn("do not guess", " ".join(packet["trustBoundary"]).lower())

    def test_l05_is_structured_human_readable_and_source_bound(self) -> None:
        spec = build_l05_spec(
            skill_path=SKILL,
            capability_catalog_path=CAPABILITIES,
        )
        self.assertEqual(spec.skill_id, "url1_network_access")
        self.assertEqual(spec.capabilities.effects, ("rest.url1.network-access.grant",))
        self.assertEqual(spec.unresolved_questions, ())
        self.assertIn("workflow:", l05_yaml(spec))

    def test_candidate_is_source_bound_and_ready_only_for_review(self) -> None:
        assessment = assess_promotion(
            skill_path=SKILL, candidate_path=CANDIDATE,
            capability_catalog_path=CAPABILITIES, l05_path=L05,
        )
        self.assertEqual(assessment.report["status"], "ready_for_review")
        self.assertEqual(assessment.report["summary"], {"errors": 0, "warnings": 0})
        self.assertFalse(assessment.report["executionEligible"])
        self.assertFalse(assessment.report["autoActivated"])
        self.assertEqual(
            assessment.bound_manifest.metadata.labels["source-skill"],
            "url1-network-access",
        )
        self.assertTrue(
            assessment.bound_manifest.metadata.labels["l0.5-sha256"].startswith("sha256:"),
        )
        self.assertEqual(
            assessment.report["structuredSkill"]["previousStageSha256"],
            assessment.report["sourceSkill"]["sha256"],
        )
        self.assertTrue(assessment.report["candidate"]["compiledHash"].startswith("sha256:"))
        coverage = assessment.report["semanticCoverage"]
        self.assertEqual(coverage["gate"], "passed")
        self.assertEqual(coverage["summary"]["blockingRequirements"], 0)
        self.assertGreater(coverage["summary"]["totalRequirements"], 10)
        self.assertGreater(coverage["summary"]["averageMappingConfidence"], 80)
        self.assertGreater(coverage["summary"]["averageL1ToL05Confidence"], 80)
        self.assertGreater(coverage["summary"]["averageL05ToL0Confidence"], 80)
        self.assertEqual(
            coverage["summary"]["lowConfidenceRequirements"],
            coverage["summary"]["attentionRequirements"],
        )
        self.assertTrue(all(
            item["source"]["path"]
            and item["verdict"]
            and item["mappingConfidence"]["method"] == "deterministic_evidence_v1"
            and 0 <= item["mappingConfidence"]["score"] <= 100
            and 0 <= item["languageLoss"]["riskPercent"] <= 100
            for item in coverage["requirements"]
        ))
        self.assertTrue(all(
            set(item["transitionAssessments"]) == {"l1ToL05", "l05ToL0"}
            and item["transitionAssessments"]["l1ToL05"]["toStage"] == "L0.5"
            and item["transitionAssessments"]["l05ToL0"]["toStage"] == "L0"
            for item in coverage["requirements"]
        ))
        self.assertTrue(all(
            round(sum(
                item["mappingConfidence"]["components"][name] * weight
                for name, weight in item["mappingConfidence"]["weights"].items()
            )) == item["mappingConfidence"]["score"]
            for item in coverage["requirements"]
        ))

    def test_semantic_gate_blocks_l05_that_drops_active_identity_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = yaml.safe_load(L05.read_text(encoding="utf-8"))
            raw["parameters"]["user_id"] = "User identifier."
            l05 = Path(directory) / "L0.5.yaml"
            l05.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
            assessment = assess_promotion(
                skill_path=SKILL,
                l05_path=l05,
                candidate_path=CANDIDATE,
                capability_catalog_path=CAPABILITIES,
            )
        self.assertEqual(assessment.report["status"], "blocked")
        requirement = next(
            item for item in assessment.report["semanticCoverage"]["requirements"]
            if item["category"] == "precondition"
            and item["source"]["path"] == "parameters.user_id#precondition"
        )
        self.assertEqual(requirement["verdict"], "missing")
        self.assertEqual(requirement["fix"]["file"], "02-L0.5.yaml")

    def test_semantic_gate_locates_weakened_active_preflight_predicate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = yaml.safe_load(CANDIDATE.read_text(encoding="utf-8"))
            raw["spec"]["preflight"][0]["predicates"] = [
                {"field": "revision", "operator": "exists"},
            ]
            candidate = Path(directory) / "candidate.yaml"
            candidate.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
            assessment = assess_promotion(
                skill_path=SKILL,
                l05_path=L05,
                candidate_path=candidate,
                capability_catalog_path=CAPABILITIES,
            )
        self.assertEqual(assessment.report["status"], "blocked")
        requirement = next(
            item for item in assessment.report["semanticCoverage"]["requirements"]
            if item["category"] == "precondition"
            and item["source"]["path"] == "parameters.user_id#precondition"
        )
        self.assertEqual(requirement["verdict"], "weakened")
        self.assertTrue(requirement["blocksPromotion"])
        self.assertEqual(requirement["mappingConfidence"]["band"], "low")
        self.assertEqual(requirement["languageLoss"]["type"], "semantic_weakening")
        self.assertTrue(requirement["attentionRequired"])
        self.assertEqual(requirement["alertLevel"], "critical")
        self.assertEqual(
            requirement["transitionAssessments"]["l05ToL0"]["verdict"],
            "weakened",
        )
        self.assertLess(
            requirement["transitionAssessments"]["l05ToL0"]["score"], 65,
        )
        self.assertEqual(requirement["fix"]["path"], "spec.preflight.predicates")
        self.assertIn("active", requirement["fix"]["hint"])

    def test_semantic_gate_blocks_undeclared_extra_effect_tool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = yaml.safe_load(CANDIDATE.read_text(encoding="utf-8"))
            raw["spec"]["effect"]["tool"] = "model_invented_effect"
            candidate = Path(directory) / "candidate.yaml"
            candidate.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
            assessment = assess_promotion(
                skill_path=SKILL,
                l05_path=L05,
                candidate_path=candidate,
                capability_catalog_path=CAPABILITIES,
            )
        self.assertEqual(assessment.report["semanticCoverage"]["gate"], "blocked")
        self.assertEqual(
            assessment.report["semanticCoverage"]["extraEffects"],
            ["model_invented_effect"],
        )
        self.assertIn(
            "SEMANTIC_EXTRA_EFFECT",
            {item["code"] for item in assessment.report["findings"]},
        )

    def test_missing_l1_parameter_blocks_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "url1-network-access"
            root.mkdir()
            text = SKILL.read_text(encoding="utf-8").replace(
                "- `reason`: Human-readable business reason of 5 through 512 characters.\n", "",
            )
            skill = root / "SKILL.md"
            skill.write_text(text, encoding="utf-8")
            assessment = assess_promotion(
                skill_path=skill, candidate_path=CANDIDATE,
                capability_catalog_path=CAPABILITIES,
            )
            self.assertEqual(assessment.report["status"], "blocked")
            self.assertIn(
                "L1_PARAMETER_COVERAGE_MISSING",
                {item["code"] for item in assessment.report["findings"]},
            )

    def test_l05_cannot_drop_parameters_or_weaken_approval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = yaml.safe_load(L05.read_text(encoding="utf-8"))
            raw["parameters"].pop("reason")
            raw["safety"]["approvalRequired"] = False
            l05 = Path(directory) / "L0.5.yaml"
            l05.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
            assessment = assess_promotion(
                skill_path=SKILL,
                l05_path=l05,
                candidate_path=CANDIDATE,
                capability_catalog_path=CAPABILITIES,
            )
            self.assertEqual(assessment.report["status"], "blocked")
            codes = {item["code"] for item in assessment.report["findings"]}
            self.assertIn("L05_PARAMETER_DRIFT", codes)
            self.assertIn("L05_APPROVAL_WEAKENED_FROM_L1", codes)

    def test_capability_role_mismatch_blocks_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = yaml.safe_load(CAPABILITIES.read_text(encoding="utf-8"))
            catalog["capabilities"][1]["role"] = "effect"
            path = Path(directory) / "capabilities.yaml"
            path.write_text(yaml.safe_dump(catalog), encoding="utf-8")
            assessment = assess_promotion(
                skill_path=SKILL, candidate_path=CANDIDATE,
                capability_catalog_path=path,
            )
            self.assertEqual(assessment.report["status"], "blocked")
            self.assertIn(
                "CAPABILITY_ROLE_MISMATCH",
                {item["code"] for item in assessment.report["findings"]},
            )

    def test_review_is_one_shot_integrity_checked_and_never_activates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            proposal = Path(directory) / "proposal"
            packaged = package_promotion(
                skill_path=SKILL, candidate_path=CANDIDATE,
                capability_catalog_path=CAPABILITIES, output_directory=proposal,
                l05_path=L05,
            )
            self.assertFalse(packaged["auto_activated"])
            trajectory = json.loads((proposal / "trajectory.json").read_text(encoding="utf-8"))
            self.assertEqual(
                [item["stage"] for item in trajectory["stages"]],
                ["L1", "L0.5", "L0-authoring", "L0-compiled"],
            )
            self.assertTrue((proposal / "02-L0.5.yaml").is_file())
            reviewed = review_promotion(
                proposal_directory=proposal, reviewer="network-reviewer",
                decision="approve", reason="schema and rollback reviewed",
            )
            self.assertFalse(reviewed["auto_activated"])
            with self.assertRaisesRegex(PromotionError, "already been reviewed"):
                review_promotion(
                    proposal_directory=proposal, reviewer="second-reviewer",
                    decision="reject", reason="late decision",
                )

    def test_tampered_proposal_cannot_be_reviewed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            proposal = Path(directory) / "proposal"
            package_promotion(
                skill_path=SKILL, candidate_path=CANDIDATE,
                capability_catalog_path=CAPABILITIES, output_directory=proposal,
            )
            with (proposal / "02-L0.5.yaml").open("a", encoding="utf-8") as stream:
                stream.write("# tampered\n")
            with self.assertRaisesRegex(PromotionError, "integrity check failed"):
                review_promotion(
                    proposal_directory=proposal, reviewer="network-reviewer",
                    decision="approve", reason="should fail",
                )

    def test_legacy_report_without_semantic_gate_cannot_be_reviewed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            proposal = Path(directory) / "proposal"
            package_promotion(
                skill_path=SKILL, candidate_path=CANDIDATE,
                capability_catalog_path=CAPABILITIES, output_directory=proposal,
                l05_path=L05,
            )
            report_path = proposal / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report.pop("semanticCoverage")
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(PromotionError, "semantic coverage gate"):
                review_promotion(
                    proposal_directory=proposal, reviewer="network-reviewer",
                    decision="approve", reason="legacy report",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
