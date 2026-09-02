from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from evaluation.runtime_comparison import (
    BASELINE_NAME,
    RUNTIME_NAME,
    append_history,
    evaluate_trend,
    load_baseline,
    read_history,
    render_html,
    render_markdown,
    run_benchmark,
    write_report,
)


class RuntimeComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = asyncio.run(run_benchmark(iterations=2))

    def test_campaign_is_explicit_and_runtime_passes_every_oracle(self) -> None:
        self.assertEqual(self.report["scenario_count"], 72)
        self.assertEqual(self.report["scenario_matrix"], {
            "valid_completion": 8,
            "parameter_intent": 12,
            "approval_binding": 12,
            "read_policy": 8,
            "result_recovery": 12,
            "compensation": 8,
            "saga": 6,
            "evidence_integrity": 6,
        })
        self.assertTrue(self.report["methodology"]["llm_and_l1_selection_excluded"])
        self.assertTrue(all(
            scenario[RUNTIME_NAME]["passed"] for scenario in self.report["scenarios"]
        ))
        baseline = next(
            metric for metric in self.report["metrics"][BASELINE_NAME]
            if metric["metric_id"] == "control_effectiveness"
        )
        guarded = next(
            metric for metric in self.report["metrics"][RUNTIME_NAME]
            if metric["metric_id"] == "control_effectiveness"
        )
        self.assertLess(baseline["rate"], guarded["rate"])
        self.assertEqual(guarded["rate"], 100.0)

    def test_reports_are_machine_and_human_readable(self) -> None:
        markdown = render_markdown(self.report)
        rendered = render_html(self.report)
        self.assertIn("故障/风险控制有效率", markdown)
        self.assertIn("DSH only vs DSH + Network Runtime", rendered)
        self.assertIn("LLM/Skill", rendered)
        with tempfile.TemporaryDirectory() as temporary:
            paths = write_report(self.report, Path(temporary))
            payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "netopyu.runtime-ab@1.0.0")
            self.assertTrue(Path(paths["markdown"]).is_file())
            self.assertTrue(Path(paths["html"]).is_file())

    def test_three_unique_iterations_detect_stable_improved_and_regressed_trends(self) -> None:
        baseline = load_baseline("data/runtime_ab_baseline.json")

        def sample(index: int, *, p50: float = 7.9, p95: float = 9.0,
                   controls: float = 100.0, valid: float = 100.0,
                   passed: bool = True, count: int = 10) -> dict:
            return {
                "campaign_id": "core72-v1",
                "source_fingerprint": f"sha256:{index:064d}",
                "all_oracles_passed": passed,
                "control_effectiveness_rate": controls,
                "control_scenarios": count,
                "valid_completion_rate": valid,
                "p50_ms": p50,
                "p95_ms": p95,
            }

        legacy = {
            "source_fingerprint": f"sha256:{0:064d}",
            "all_oracles_passed": False,
        }
        collecting = evaluate_trend(baseline, [legacy, sample(1), sample(2)])
        self.assertEqual(collecting["status"], "collecting")
        stable = evaluate_trend(baseline, [sample(1), sample(2), sample(3)])
        self.assertEqual(stable["status"], "stable")
        improved = evaluate_trend(baseline, [
            sample(1, p50=5.8, p95=6.8),
            sample(2, p50=5.9, p95=6.9),
            sample(3, p50=6.0, p95=7.0),
        ])
        self.assertEqual(improved["status"], "improved")
        regressed = evaluate_trend(baseline, [
            sample(1), sample(2), sample(3, controls=90.0, passed=False),
        ])
        self.assertEqual(regressed["status"], "regressed")

    def test_history_counts_only_unique_implementation_fingerprints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            history = Path(temporary) / "history.jsonl"
            legacy = {
                "source_fingerprint": "sha256:" + "a" * 64,
                "all_oracles_passed": True,
            }
            snapshot = {
                "campaign_id": "core72-v1",
                "source_fingerprint": "sha256:" + "a" * 64,
                "all_oracles_passed": True,
            }
            legacy_result = append_history(history, legacy)
            first = append_history(history, snapshot)
            duplicate = append_history(history, snapshot)
            self.assertTrue(legacy_result["recorded"])
            self.assertTrue(first["recorded"])
            self.assertFalse(duplicate["recorded"])
            self.assertEqual(len(read_history(history)), 2)


if __name__ == "__main__":
    unittest.main()
