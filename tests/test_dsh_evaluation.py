from __future__ import annotations

import asyncio
import os
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from dsh_adapter.evaluation import parity_report
from evaluation import load_golden_set
from skills import SkillLoader


class TestDshEvaluation(unittest.TestCase):
    def test_current_lan_golden_set_passes_retirement_gate(self):
        root = Path(__file__).parents[1]
        with patch.dict(os.environ, {"NETOPYU_DSH_BACKEND": "mock"}):
            report = asyncio.run(parity_report(
                profile_id="lan", golden_path=str(root / "data" / "golden_set.jsonl"),
            ))
        self.assertTrue(report["ok"])
        self.assertEqual(report["retirement_gate"], "pass")
        self.assertEqual(report["metrics"]["cases"], 100)
        self.assertEqual(report["metrics"]["recall_at_3"], 1.0)
        self.assertGreaterEqual(report["metrics"]["mrr"], 0.90)
        self.assertEqual(report["failures"], [])

    def test_golden_set_is_balanced_valid_and_unique(self):
        root = Path(__file__).parents[1]
        cases = load_golden_set(root / "data" / "golden_set.jsonl")
        active = set(SkillLoader(mode="mock", profile="lan").skill_definitions())
        queries = [case.query.casefold().strip() for case in cases]
        languages = Counter(case.language for case in cases)
        primary_skills = Counter(case.expected_ids[0] for case in cases)

        self.assertEqual(len(cases), 100)
        self.assertEqual(len(queries), len(set(queries)))
        self.assertGreaterEqual(languages["en"], 40)
        self.assertGreaterEqual(languages["zh"], 30)
        self.assertGreaterEqual(languages["mixed"], 5)
        self.assertTrue(all(case.kind == "skill" for case in cases))
        self.assertTrue(all(set(case.expected_ids) <= active for case in cases))
        self.assertTrue(all(count >= 8 for count in primary_skills.values()))
