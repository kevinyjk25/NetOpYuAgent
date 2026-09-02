from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from network_runtime.contracts import sha256_json

from evaluation.dsh_controlled_tool import (
    CONTROLLER_CONTRACT,
    CONTROLLED_SYSTEM_PREFIX,
    EXPECTED_TOOLS,
    REQUIRED_DISABLED_IDS_C1,
    SAFE_ACTIVE_IDS_C1,
    ProcessResult,
    _compile_controlled_decision,
    aggregate_protocol,
    audit_c1_dumped_config,
    build_controlled_system_prompt,
    controlled_evaluator_fingerprint,
    materialize_c1_patch,
    project_controlled_transcript,
)
from evaluation.dsh_shadow import DSH_TESTED_VERSION
from evaluation.l1_catalog import build_profile_catalog


PROJECT = Path(__file__).resolve().parents[1]
PLUGIN = PROJECT / "dsh-plugin-l1-protocol-controller/src/index.js"
SKILL = PROJECT / "dsh-plugin-l1-protocol-controller/skills/l1-controlled-decision/SKILL.md"
SKILL_DIGEST = "sha256:" + __import__("hashlib").sha256(SKILL.read_bytes()).hexdigest()


def _config(*, activate: str | None = None, plugin: Path = PLUGIN) -> str:
    blocks: list[str] = []
    for entry_id in sorted(SAFE_ACTIVE_IDS_C1 | REQUIRED_DISABLED_IDS_C1):
        disabled = entry_id in REQUIRED_DISABLED_IDS_C1 and entry_id != activate
        name = str(plugin.resolve()) if entry_id == "l1-protocol-controller" else f"@dsh/{entry_id}"
        block = f"- id: {entry_id}\n  name: '{name}'\n"
        if disabled:
            block += "  disabled: true\n"
        if entry_id == "system-prompt":
            block += (
                "  config:\n"
                "    persona: !!js process.env.NETOPYU_L1_CONTROLLED_SYSTEM_PROMPT\n"
            )
        if entry_id == "l1-protocol-controller":
            block += (
                "  config:\n"
                "    preloadedSkillDigest: !!js process.env.NETOPYU_L1_PRELOADED_SKILL_DIGEST\n"
            )
        blocks.append(block)
    return "".join(blocks)


def _tool_result(call_id: str, text: str, *, error: bool = False) -> dict:
    return {
        "type": "tool/result",
        "seq": 4,
        "data": {
            "message": {
                "content": [{
                    "type": "tool-result",
                    "toolCallId": call_id,
                    "content": [{"type": "text", "text": text}],
                    "isError": error,
                }]
            }
        },
    }


def _valid_events() -> tuple[list[dict], tuple]:
    candidate = next(
        item for item in build_profile_catalog("lan")
        if item.kind == "skill" and item.target == "restart-service"
    )
    typed_arguments = {
        "target": "restart-service",
        "arguments": {"environment": "prod", "service": "crm"},
        "confidence": 0.9,
        "reason_code": "explicit_restart",
    }
    receipt = json.dumps({
        "accepted": True,
        "contract": CONTROLLER_CONTRACT,
        "digest": sha256_json({
            "tool": "propose_l1_skill", "arguments": typed_arguments,
        }),
        "preloadedSkillDigest": SKILL_DIGEST,
    })
    return [
        {"type": "session"},
        {
            "type": "request/header",
            "seq": 1,
            "data": {"header": {"tools": [
                {"name": name} for name in EXPECTED_TOOLS
            ]}},
        },
        {
            "type": "tool/call",
            "seq": 2,
            "data": {
                "step": 1,
                "callId": "capture-1",
                "name": "propose_l1_skill",
                "arguments": json.dumps(typed_arguments),
            },
        },
        _tool_result("capture-1", receipt),
        {"type": "turn/end", "seq": 5, "data": {"reason": {"kind": "stop"}}},
    ], (candidate,)


class C1ConfigTests(unittest.TestCase):
    def test_versioned_full_observation_binds_current_evaluator(self):
        payload = json.loads(
            (PROJECT / "data/l1_dsh_controlled_tool_observations.json").read_text()
        )
        self.assertEqual(
            payload["apiVersion"],
            "netopyu.io/l1-dsh-controlled-tool-observations/v1",
        )
        self.assertEqual(len(payload["observations"]), 1)
        observation = payload["observations"][0]
        self.assertEqual(
            observation["evaluator_fingerprint"],
            controlled_evaluator_fingerprint(PROJECT),
        )
        self.assertEqual(observation["evaluated_cases"], 160)
        self.assertTrue(observation["dataset_complete"])
        self.assertTrue(observation["qualification_eligible"])
        self.assertFalse(observation["qualified"])
        self.assertEqual(observation["protocol_metrics"]["forbidden_tool_call_rate"], 0.0)
        self.assertGreater(
            payload["comparison_to_b2_same_model_artifact"]["end_to_end_accuracy_delta"],
            0,
        )

    def test_reviewed_config_is_exact_and_fail_closed(self):
        audit = audit_c1_dumped_config(
            _config(), dsh_version=DSH_TESTED_VERSION, expected_plugin_path=PLUGIN,
        )
        self.assertEqual(set(audit.active_ids), SAFE_ACTIVE_IDS_C1)
        self.assertTrue(audit.config_digest.startswith("sha256:"))
        with self.assertRaisesRegex(ValueError, "did not disable"):
            audit_c1_dumped_config(
                _config(activate="tool-bash"),
                dsh_version=DSH_TESTED_VERSION,
                expected_plugin_path=PLUGIN,
            )

    def test_plugin_path_version_and_environment_bindings_are_pinned(self):
        with self.assertRaisesRegex(ValueError, "plugin path"):
            audit_c1_dumped_config(
                _config(plugin=PROJECT / "not-reviewed.js"),
                dsh_version=DSH_TESTED_VERSION,
                expected_plugin_path=PLUGIN,
            )
        with self.assertRaisesRegex(ValueError, "reviewed version"):
            audit_c1_dumped_config(
                _config(), dsh_version="0.1.2", expected_plugin_path=PLUGIN,
            )
        with self.assertRaisesRegex(ValueError, "PRELOADED_SKILL_DIGEST"):
            audit_c1_dumped_config(
                _config().replace("NETOPYU_L1_PRELOADED_SKILL_DIGEST", "MISSING"),
                dsh_version=DSH_TESTED_VERSION,
                expected_plugin_path=PLUGIN,
            )

    def test_patch_materialization_and_preloaded_skill_are_deterministic(self):
        template = (PROJECT / "evaluation/dsh_controlled_tool.patch.yml").read_text()
        materialized = materialize_c1_patch(template, PLUGIN)
        self.assertNotIn("__NETOPYU_L1_PROTOCOL_CONTROLLER_PLUGIN__", materialized)
        self.assertIn(json.dumps(str(PLUGIN.resolve())), materialized)
        self.assertIn("NETOPYU_L1_PRELOADED_SKILL_DIGEST", materialized)
        for entry_id in REQUIRED_DISABLED_IDS_C1:
            self.assertRegex(
                materialized,
                rf"(?m)^- id: {re.escape(entry_id)}\n  disabled: true$",
            )
        prompt = build_controlled_system_prompt(SKILL.read_text())
        self.assertTrue(prompt.startswith(CONTROLLED_SYSTEM_PREFIX))
        self.assertIn("<reviewed_l1_skill>", prompt)
        self.assertNotIn("metadata:\n", prompt)

    def test_controller_has_no_effect_adapter_and_fingerprint_binds_dependencies(self):
        source = PLUGIN.read_text(encoding="utf-8")
        for forbidden in (
            "network_runtime", "dsh-plugin-netopyu", "child_process",
            "node:http", "node:https", "node:net", "fetch(",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("export const inject = ['tools']", source)
        self.assertRegex(controlled_evaluator_fingerprint(PROJECT), r"^sha256:[0-9a-f]{64}$")


class C1TranscriptTests(unittest.TestCase):
    def test_controller_derives_workflow_and_missing_fields_from_catalog(self):
        candidates = build_profile_catalog("lan")
        selected = _compile_controlled_decision(
            "propose_l1_skill",
            {
                "target": "restart-service",
                "arguments": {"service": "crm"},
                "confidence": 0.8,
                "reason_code": "restart_requested",
            },
            candidates,
        )
        self.assertEqual(selected.action.value, "clarify")
        self.assertEqual(selected.missing_fields, ("environment",))
        self.assertEqual(selected.workflow, ())

        workflow = _compile_controlled_decision(
            "propose_l1_skill",
            {
                "target": "branch-app-reachability",
                "arguments": {},
                "confidence": 0.8,
                "reason_code": "branch_diagnosis",
            },
            candidates,
        )
        expected = next(
            item.workflow_hint for item in candidates
            if item.target == "branch-app-reachability"
        )
        self.assertEqual(workflow.workflow, expected)

    def test_single_capture_projects_a_digest_bound_decision(self):
        events, candidates = _valid_events()
        projected = project_controlled_transcript(
            events,
            scenario_id="fixture",
            candidates=candidates,
            expected_skill_digest=SKILL_DIGEST,
            session_digest="sha256:" + "b" * 64,
            process_result=ProcessResult(0, "proposal captured\n", "", 12.0),
        )
        self.assertEqual(projected.response.decision.target, "restart-service")
        self.assertIsNone(projected.trace.error_type)
        self.assertTrue(projected.trace.preloaded_skill_digest_match)
        self.assertEqual(projected.trace.tool_calls, ("propose_l1_skill",))
        metrics = aggregate_protocol([projected.trace])
        self.assertEqual(metrics["single_capture_accuracy"], 1.0)
        self.assertEqual(metrics["forbidden_tool_call_rate"], 0.0)

    def test_duplicate_forbidden_or_skill_digest_mismatch_fails_closed(self):
        events, candidates = _valid_events()
        duplicate = dict(events[2])
        duplicate["seq"] = 3
        duplicate["data"] = dict(duplicate["data"], callId="capture-2")
        events.insert(-1, duplicate)
        projected = project_controlled_transcript(
            events,
            scenario_id="fixture",
            candidates=candidates,
            expected_skill_digest=SKILL_DIGEST,
            session_digest=None,
            process_result=ProcessResult(0, "proposal captured\n", "", 12.0),
        )
        self.assertIsNone(projected.response.decision)
        self.assertEqual(projected.trace.error_type, "DuplicateCaptureCall")

        events, candidates = _valid_events()
        events.insert(-1, {
            "type": "tool/call", "seq": 4,
            "data": {"callId": "bad", "name": "tool-bash", "arguments": "{}"},
        })
        projected = project_controlled_transcript(
            events,
            scenario_id="fixture",
            candidates=candidates,
            expected_skill_digest=SKILL_DIGEST,
            session_digest=None,
            process_result=ProcessResult(0, "proposal captured\n", "", 12.0),
        )
        self.assertEqual(projected.trace.error_type, "ForbiddenToolCall")

        events, candidates = _valid_events()
        projected = project_controlled_transcript(
            events,
            scenario_id="fixture",
            candidates=candidates,
            expected_skill_digest="sha256:" + "c" * 64,
            session_digest=None,
            process_result=ProcessResult(0, "proposal captured\n", "", 12.0),
        )
        self.assertEqual(projected.trace.error_type, "PreloadedSkillDigestMismatch")

    def test_invalid_candidate_contract_is_not_projected(self):
        events, candidates = _valid_events()
        invalid = json.loads(events[2]["data"]["arguments"])
        invalid["target"] = "invented-skill"
        events[2]["data"]["arguments"] = json.dumps(invalid)
        receipt = json.loads(events[3]["data"]["message"]["content"][0]["content"][0]["text"])
        normalized = {
            "tool": "propose_l1_skill",
            "arguments": invalid,
        }
        receipt["digest"] = sha256_json(normalized)
        events[3]["data"]["message"]["content"][0]["content"][0]["text"] = json.dumps(receipt)
        projected = project_controlled_transcript(
            events,
            scenario_id="fixture",
            candidates=candidates,
            expected_skill_digest=SKILL_DIGEST,
            session_digest=None,
            process_result=ProcessResult(0, "proposal captured\n", "", 12.0),
        )
        self.assertIsNone(projected.response.decision)
        self.assertEqual(projected.trace.error_type, "ProposalContractInvalid")


if __name__ == "__main__":
    unittest.main()
