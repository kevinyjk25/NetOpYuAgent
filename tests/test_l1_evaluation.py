from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from evaluation.l1_adapters import AdapterResponse, OpenAICompatibleAdapter
from evaluation.l1_benchmark import (
    BASELINE_SCHEMA,
    _versioned_baseline,
    aggregate,
    run_benchmark,
    score_case,
)
from evaluation.l1_catalog import L1CandidateRetriever, build_profile_catalog
from evaluation.l1_contract import (
    L1Action,
    L1Category,
    L1Decision,
    L1_DECISION_SCHEMA,
)
from evaluation.l1_scenarios import build_l1_scenarios, scenario_set_digest


def _arguments(output: Path, *, max_cases: int = 0, record: bool = False):
    return argparse.Namespace(
        adapter="keyword",
        base_url="http://127.0.0.1:11434/v1",
        model="unused",
        model_artifact_digest="",
        api_key_env="NETOPYU_TEST_KEY",
        timeout=5.0,
        allow_remote=False,
        candidate_top_k=12,
        workers=1,
        max_cases=max_cases,
        smoke_per_category=0,
        category=None,
        language=None,
        output_dir=str(output),
        baseline=str(output / "missing-baseline.json"),
        export_dataset="",
        record=record,
        gate=False,
        resume=False,
    )


class L1ScenarioTests(unittest.TestCase):
    def test_curated_dataset_is_complete_and_stable(self):
        scenarios = build_l1_scenarios()
        self.assertEqual(len(scenarios), 160)
        self.assertEqual(len({item.scenario_id for item in scenarios}), 160)
        self.assertEqual(
            Counter(item.category for item in scenarios),
            {
                L1Category.SKILL_SELECTION: 28,
                L1Category.TOOL_SELECTION: 36,
                L1Category.MULTI_STEP: 32,
                L1Category.CLARIFICATION: 30,
                L1Category.SAFETY_REFUSAL: 20,
                L1Category.OUT_OF_SCOPE: 14,
            },
        )
        self.assertEqual({item.language for item in scenarios}, {"zh", "en", "mixed"})
        self.assertEqual(
            len({item.scenario_id.rsplit("-", 1)[0] for item in scenarios}), 51,
        )
        self.assertEqual(
            scenario_set_digest(scenarios),
            "sha256:c9cf65cfaa15d5d5096a5eefc026463888f3d7aa71aca4078bece8b171fb48c2",
        )

    def test_versioned_jsonl_matches_the_curated_source(self):
        path = Path(__file__).resolve().parents[1] / "data" / "l1_eval_set.jsonl"
        stored = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]
        expected = [
            item.model_dump(by_alias=True, mode="json")
            for item in build_l1_scenarios()
        ]
        self.assertEqual(stored, expected)

    def test_every_oracle_target_exists_and_candidate_recall_is_complete(self):
        scenarios = build_l1_scenarios()
        eligible = 0
        hits = 0
        for profile in ("lan", "dc", "wan"):
            catalog = build_profile_catalog(profile)
            known = {item.target for item in catalog}
            retriever = L1CandidateRetriever(catalog)
            for scenario in scenarios:
                if scenario.profile != profile or not scenario.acceptable_targets:
                    continue
                eligible += 1
                self.assertTrue(set(scenario.acceptable_targets) <= known)
                candidates = retriever.retrieve(scenario.prompt, top_k=12)
                if any(
                    item.target in scenario.acceptable_targets for item in candidates
                ):
                    hits += 1
        self.assertEqual((eligible, hits), (126, 126))


class L1ContractTests(unittest.TestCase):
    def test_runtime_owned_schema_version_may_be_defaulted_but_not_changed(self):
        decision = L1Decision(
            action=L1Action.OUT_OF_SCOPE,
            confidence=1,
            reason_code="outside_domain",
        )
        self.assertEqual(decision.api_version, L1_DECISION_SCHEMA)
        with self.assertRaises(ValidationError):
            L1Decision(
                apiVersion="netopyu.io/l1-decision/v999",
                action=L1Action.OUT_OF_SCOPE,
                confidence=1,
                reason_code="bad_version",
            )

    def test_selection_cannot_omit_target_or_carry_missing_fields(self):
        with self.assertRaises(ValidationError):
            L1Decision(
                apiVersion=L1_DECISION_SCHEMA,
                action=L1Action.SELECT_TOOL,
                confidence=0.5,
                reason_code="missing_target",
            )
        with self.assertRaises(ValidationError):
            L1Decision(
                apiVersion=L1_DECISION_SCHEMA,
                action=L1Action.SELECT_TOOL,
                target="device_info",
                missing_fields=("device_id",),
                confidence=0.5,
                reason_code="invalid_selection",
            )

    def test_refusal_cannot_smuggle_executable_content_or_extra_fields(self):
        with self.assertRaises(ValidationError):
            L1Decision(
                apiVersion=L1_DECISION_SCHEMA,
                action=L1Action.REFUSE,
                arguments={"command": "write erase"},
                confidence=1.0,
                reason_code="unsafe",
            )
        with self.assertRaises(ValidationError):
            L1Decision.model_validate({
                "apiVersion": L1_DECISION_SCHEMA,
                "action": "out_of_scope",
                "confidence": 1,
                "reason_code": "outside_domain",
                "shell": "rm -rf /",
            })

    def test_arguments_are_bounded_json_not_an_untyped_payload_channel(self):
        with self.assertRaises(ValidationError):
            L1Decision(
                action=L1Action.SELECT_TOOL,
                target="device_info",
                arguments={"device_id": "x" * 2001},
                confidence=0.5,
                reason_code="oversized",
            )
        with self.assertRaises(ValidationError):
            L1Decision(
                action=L1Action.SELECT_TOOL,
                target="device_info",
                arguments={"bad key": "ap-01"},
                confidence=0.5,
                reason_code="bad_key",
            )

    def test_remote_model_endpoint_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError, "requires --allow-remote"):
            OpenAICompatibleAdapter(
                base_url="https://models.example.test/v1",
                model="example",
            )

    def test_model_adapter_rejects_missing_required_values_and_invented_workflow(self):
        candidate = next(
            item for item in build_profile_catalog("lan")
            if item.kind == "skill" and item.target == "restart-service"
        )

        class Response:
            def __init__(self, content):
                self.content = content

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self, _limit):
                return json.dumps({
                    "choices": [{"message": {"content": json.dumps(self.content)}}],
                    "usage": {},
                }).encode()

        class Opener:
            def __init__(self, content):
                self.response = Response(content)

            def open(self, _request, timeout):
                self.timeout = timeout
                return self.response

        base = {
            "action": "select_skill",
            "target": "restart-service",
            "arguments": {"service": "crm"},
            "missing_fields": [],
            "workflow": [],
            "confidence": 0.9,
            "reason_code": "test",
        }
        adapter = OpenAICompatibleAdapter(
            base_url="http://127.0.0.1:11434/v1", model="test",
        )
        adapter._opener = Opener(base)
        self.assertIsNone(adapter.decide("restart crm", (candidate,)).decision)

        adapter._opener = Opener({
            **base,
            "arguments": {"service": "crm", "environment": "prod"},
            "workflow": ["invented_step"],
        })
        self.assertIsNone(adapter.decide("restart crm in prod", (candidate,)).decision)

        adapter._opener = Opener({
            **base,
            "arguments": {"service": "crm", "environment": "prod"},
        })
        self.assertIsNotNone(
            adapter.decide("restart crm in prod", (candidate,)).decision
        )


class L1MetricTests(unittest.TestCase):
    def test_versioned_baseline_flags_material_regression(self):
        names = (
            "selection_accuracy", "parameter_field_f1", "clarification_recall",
            "missing_fields_accuracy", "workflow_accuracy", "safety_escape_rate",
            "end_to_end_accuracy",
        )
        identity = {
            "adapter": "openai-compatible",
            "model": "local-7b",
            "model_artifact_digest": "sha256:" + "a" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            path.write_text(json.dumps({
                "apiVersion": BASELINE_SCHEMA,
                "baselines": [{
                    **identity,
                    "fingerprint": "old",
                    "metrics": {name: 0.0 if name == "safety_escape_rate" else 1.0 for name in names},
                }],
            }), encoding="utf-8")
            result = _versioned_baseline(path, {
                **identity,
                "metrics": {
                    name: 0.1 if name == "safety_escape_rate" else 0.8
                    for name in names
                },
            })
        self.assertEqual(result["status"], "regressed")

    def test_perfect_decision_scores_end_to_end(self):
        scenario = build_l1_scenarios()[0]
        decision = L1Decision(
            apiVersion=L1_DECISION_SCHEMA,
            action=scenario.expected_action,
            target=scenario.acceptable_targets[0],
            arguments=scenario.expected_arguments,
            workflow=scenario.expected_workflow,
            confidence=1,
            reason_code="test_oracle",
        )
        score = score_case(
            scenario,
            AdapterResponse(
                decision=decision,
                error_type=None,
                error=None,
                elapsed_ms=1,
            ),
            scenario.acceptable_targets,
        )
        self.assertTrue(score.end_to_end_pass)
        self.assertEqual(aggregate([score])["end_to_end_accuracy"], 1.0)

    def test_partial_run_is_never_qualification_eligible(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run_benchmark(_arguments(Path(directory), max_cases=10))
        self.assertFalse(report["qualification_eligible"])
        self.assertFalse(report["qualified"])
        self.assertIn("all 160", report["gate_failures"][0])

    def test_partial_run_cannot_be_recorded_as_a_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "complete unfiltered"):
                run_benchmark(
                    _arguments(Path(directory), max_cases=10, record=True)
                )

    def test_checkpoint_is_fingerprint_bound_and_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = run_benchmark(_arguments(root, max_cases=10))
            self.assertEqual(first["resumed_cases"], 0)
            resumed_arguments = _arguments(root, max_cases=10)
            resumed_arguments.resume = True
            second = run_benchmark(resumed_arguments)
            self.assertEqual(second["resumed_cases"], 10)
            checkpoint_path = root / "checkpoint.jsonl"
            checkpoint = checkpoint_path.read_text(encoding="utf-8")
            self.assertNotIn("USER_REQUEST", checkpoint)
            lines = checkpoint.splitlines()
            header = json.loads(lines[0])
            header["fingerprint"] = "sha256:" + "0" * 64
            checkpoint_path.write_text(
                "\n".join((json.dumps(header), *lines[1:])) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                run_benchmark(resumed_arguments)

    def test_full_keyword_baseline_writes_bilingual_non_executing_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = run_benchmark(_arguments(root))
            self.assertTrue(report["dataset_complete"])
            self.assertFalse(report["qualification_eligible"])
            self.assertEqual(report["scope"], "non-executing-l1-proposal-only")
            self.assertEqual(report["metrics"]["candidate_recall"], 1.0)
            self.assertFalse(report["qualified"])
            self.assertEqual(report["model"], "none")
            markdown = (root / "l1-eval.md").read_text(encoding="utf-8")
            self.assertIn("## 中文", markdown)
            self.assertIn("## English", markdown)
            stored = json.loads((root / "l1-eval.json").read_text(encoding="utf-8"))
            self.assertNotIn("prompt", stored["cases"][0])


if __name__ == "__main__":
    unittest.main()
