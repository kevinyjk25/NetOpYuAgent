from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from dsh_adapter.learning import mine_candidates, review_candidate


class TestDshOfflineLearning(unittest.TestCase):
    def _source(self, root: Path) -> Path:
        source = root / "hitl.sqlite"
        with sqlite3.connect(source) as connection:
            connection.execute("""
                CREATE TABLE trajectory_events (
                  id INTEGER PRIMARY KEY, session_id TEXT, event_type TEXT,
                  payload_json TEXT, created_at TEXT
                )
            """)
            for session in ("s1", "s2", "s3"):
                for tool in ("list_devices", "device_info"):
                    connection.execute(
                        "INSERT INTO trajectory_events(session_id,event_type,payload_json,created_at) VALUES(?,?,?,?)",
                        (session, "tool:start", json.dumps({"tool_name": tool, "argument_keys": ["secret"]}), "now"),
                    )
        return source

    def test_dry_run_is_non_mutating_and_apply_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, review = self._source(root), root / "learning.sqlite"
            dry = mine_candidates(source_database=str(source), review_database=str(review))
            self.assertFalse(review.exists())
            self.assertEqual(dry["candidates"][0]["tool_sequence"], ["list_devices", "device_info"])
            self.assertNotIn("secret", json.dumps(dry))
            first = mine_candidates(source_database=str(source), review_database=str(review), apply_changes=True)
            second = mine_candidates(source_database=str(source), review_database=str(review), apply_changes=True)
            self.assertEqual(first["new_candidates_stored"], 1)
            self.assertEqual(second["new_candidates_stored"], 0)

    def test_review_is_one_shot_and_approval_only_creates_proposal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, review = self._source(root), root / "learning.sqlite"
            report = mine_candidates(source_database=str(source), review_database=str(review), apply_changes=True)
            candidate_id = report["candidates"][0]["candidate_id"]
            result = review_candidate(
                review_database=str(review), candidate_id=candidate_id, decision="approve",
                reviewer="local-reviewer", proposal_directory=str(root / "proposals"),
            )
            self.assertTrue(Path(result["proposal_path"], "SKILL.md").is_file())
            self.assertFalse(result["auto_installed"])
            with self.assertRaises(RuntimeError):
                review_candidate(
                    review_database=str(review), candidate_id=candidate_id, decision="reject",
                    reviewer="second-reviewer",
                )
