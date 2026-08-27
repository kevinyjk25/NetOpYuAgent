from __future__ import annotations

import asyncio
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from dsh_adapter.evaluation import parity_report


class TestDshMigration(unittest.TestCase):
    def test_current_lan_golden_set_passes_retirement_gate(self):
        root = Path(__file__).parents[1]
        with patch.dict(os.environ, {"NETOPYU_DSH_BACKEND": "mock"}):
            report = asyncio.run(parity_report(
                profile_id="lan", golden_path=str(root / "data" / "golden_set.jsonl"),
            ))
        self.assertTrue(report["ok"])
        self.assertEqual(report["retirement_gate"], "pass")
        self.assertGreaterEqual(report["metrics"]["recall_at_3"], 0.95)
        self.assertGreaterEqual(report["metrics"]["mrr"], 0.90)
        self.assertEqual(report["failures"], [])
