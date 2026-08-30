from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from network_runtime.l0.production import CATALOG, PRODUCTION_DEFINITIONS
from network_runtime.l0.production_trajectory import (
    DEFAULT_ARCHIVE_ROOT,
    ProductionTrajectoryError,
    build_production_trajectories,
    validate_production_trajectories,
)
from network_runtime.l0.promotion import StructuredNaturalLanguageSkill
from skills.skill_format import parse_skill_md


class ProductionL0TrajectoryTests(unittest.TestCase):
    def test_all_production_l0_have_readable_exact_archives(self) -> None:
        result = validate_production_trajectories()
        self.assertTrue(result["ok"])
        self.assertEqual(result["contracts"], 21)
        self.assertEqual(result["promotion_ready"], 21)
        self.assertEqual(result["exact_round_trips"], 21)
        self.assertEqual(
            {item["skill_id"] for item in result["items"]},
            {item.skill_id for item in PRODUCTION_DEFINITIONS},
        )

    def test_each_archive_is_human_readable_and_explainable(self) -> None:
        for definition in PRODUCTION_DEFINITIONS:
            with self.subTest(skill=definition.skill_id):
                root = DEFAULT_ARCHIVE_ROOT / definition.skill_id
                l1 = parse_skill_md(
                    (root / "01-L1-SKILL.md").read_text(encoding="utf-8")
                )
                l05 = StructuredNaturalLanguageSkill.model_validate(
                    yaml.safe_load((root / "02-L0.5.yaml").read_text(encoding="utf-8"))
                )
                report = json.loads((root / "report.json").read_text(encoding="utf-8"))
                production = CATALOG.require(definition.skill_id, definition.version)
                self.assertIn(definition.skill_id, l1.body)
                self.assertEqual(l05.unresolved_questions, ())
                self.assertEqual(len(l05.semantic_intents), 1)
                self.assertEqual(
                    l05.semantic_intents[0].intent_spec(),
                    production.spec.intent,
                )
                self.assertEqual(report["promotionAssessment"]["findings"], [])
                self.assertTrue(report["roundTrip"]["semanticParity"])
                self.assertEqual(
                    report["roundTrip"]["productionContractHash"],
                    production.contract_hash,
                )

    def test_archive_generation_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            generated = Path(directory) / "trajectories"
            result = build_production_trajectories(generated)
            self.assertEqual(result["contracts"], 21)
            validate_production_trajectories(generated)
            expected_files = sorted(
                path.relative_to(DEFAULT_ARCHIVE_ROOT)
                for path in DEFAULT_ARCHIVE_ROOT.rglob("*") if path.is_file()
            )
            actual_files = sorted(
                path.relative_to(generated)
                for path in generated.rglob("*") if path.is_file()
            )
            self.assertEqual(actual_files, expected_files)
            for relative in expected_files:
                self.assertEqual(
                    (generated / relative).read_bytes(),
                    (DEFAULT_ARCHIVE_ROOT / relative).read_bytes(),
                    relative,
                )

    def test_any_stage_tampering_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "trajectories"
            shutil.copytree(DEFAULT_ARCHIVE_ROOT, archive)
            target = archive / PRODUCTION_DEFINITIONS[0].skill_id / "02-L0.5.yaml"
            with target.open("a", encoding="utf-8") as stream:
                stream.write("# tampered\n")
            with self.assertRaisesRegex(
                ProductionTrajectoryError,
                "archive file hash mismatch",
            ):
                validate_production_trajectories(archive)


if __name__ == "__main__":
    unittest.main(verbosity=2)
