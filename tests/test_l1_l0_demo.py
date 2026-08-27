from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import unittest
from pathlib import Path

from network_runtime.demo import run_l1_l0_access_demo


class TestL1L0Demo(unittest.TestCase):
    def test_cross_domain_access_problem_is_resolved_and_audited(self) -> None:
        report = asyncio.run(run_l1_l0_access_demo(approve_local_simulation=True))
        self.assertTrue(report["ok"], report)
        self.assertEqual(
            [item["l0_skill_id"] for item in report["writes"]],
            ["network.lan.user-access.grant", "network.dc.app-access.grant"],
        )
        self.assertTrue(all(item["terminal_state"] == "verified_success" for item in report["writes"]))
        self.assertTrue(all(item["audit"]["ok"] for item in report["writes"]))
        for item in report["writes"]:
            events = [
                (event["event_type"], event["step_id"])
                for event in item["l0_events"]
            ]
            self.assertIn(("l0_step_completed", "execute"), events)
            self.assertIn(("l0_step_skipped", "compensate"), events)
            self.assertEqual(events[-1], ("l0_step_completed", "audit"))
        self.assertEqual(report["guarantees_review"]["unbound_writes"], 0)
        self.assertTrue(report["guarantees_review"]["problem_resolved"])

    def test_demo_requires_explicit_local_write_authorization(self) -> None:
        with self.assertRaises(PermissionError):
            asyncio.run(run_l1_l0_access_demo(approve_local_simulation=False))

    def test_cli_runs_from_a_script_entrypoint(self) -> None:
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [sys.executable, "scripts/l1_l0_demo.py", "--approve-local-simulation"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)
        report = json.loads(completed.stdout)
        self.assertTrue(report["ok"])
        self.assertEqual(len(report["writes"]), 2)


if __name__ == "__main__":
    unittest.main()
