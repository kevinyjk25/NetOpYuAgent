from __future__ import annotations

import unittest
from pathlib import Path

from dsh_adapter.reliability import run_local_reliability


class TestDshLocalReliability(unittest.TestCase):
    def test_complete_local_retirement_rehearsal(self):
        root = Path(__file__).parents[1]
        report = run_local_reliability(
            project_root=str(root), python_executable=str(root / ".venv" / "bin" / "python"),
            request_count=8, concurrency=4,
        )
        self.assertTrue(report["ok"], report)
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["real_network_actions"], 0)
        self.assertTrue(report["temporary_state_removed"])
