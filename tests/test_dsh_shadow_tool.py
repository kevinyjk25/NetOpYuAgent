from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

import zstandard

from network_runtime.contracts import sha256_json

from evaluation.dsh_shadow import DSH_TESTED_VERSION
from evaluation.dsh_shadow_tool import (
    B2_SYSTEM_PROMPT,
    CAPTURE_CONTRACT,
    CAPTURE_SKILL,
    CAPTURE_TOOL,
    EXPECTED_TOOLS,
    REQUIRED_DISABLED_IDS_B2,
    SAFE_ACTIVE_IDS_B2,
    ProcessResult,
    _parse_preflight_response,
    _read_transcript,
    aggregate_protocol,
    audit_b2_dumped_config,
    materialize_b2_patch,
    project_transcript,
    tool_shadow_evaluator_fingerprint,
)
from evaluation.l1_catalog import build_profile_catalog


PROJECT = Path(__file__).resolve().parents[1]
PLUGIN = PROJECT / "dsh-plugin-l1-shadow-capture/src/index.js"


def _config(*, activate: str | None = None, plugin: Path = PLUGIN) -> str:
    blocks: list[str] = []
    for entry_id in sorted(SAFE_ACTIVE_IDS_B2 | REQUIRED_DISABLED_IDS_B2):
        disabled = entry_id in REQUIRED_DISABLED_IDS_B2 and entry_id != activate
        name = str(plugin.resolve()) if entry_id == "l1-shadow-capture" else f"@dsh/{entry_id}"
        block = f"- id: {entry_id}\n  name: '{name}'\n"
        if disabled:
            block += "  disabled: true\n"
        if entry_id == "system-prompt":
            block += (
                "  config:\n"
                "    persona: !!js process.env.NETOPYU_L1_TOOL_SHADOW_SYSTEM_PROMPT\n"
            )
        blocks.append(block)
    return "".join(blocks)


def _tool_result(call_id: str, text: str, *, error: bool = False) -> dict:
    return {
        "type": "tool/result",
        "seq": 1,
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
    decision = {
        "apiVersion": "netopyu.io/l1-decision/v1",
        "action": "select_skill",
        "target": "restart-service",
        "arguments": {"environment": "prod", "service": "crm"},
        "missing_fields": [],
        "workflow": [],
        "confidence": 0.9,
        "reason_code": "explicit_restart",
    }
    receipt = json.dumps({
        "accepted": True,
        "contract": CAPTURE_CONTRACT,
        "digest": sha256_json(decision),
    })
    events = [
        {"type": "session"},
        {
            "type": "user/message",
            "seq": 1,
            "data": {"source": {
                "kind": "skill-catalog",
                "entries": [{"name": CAPTURE_SKILL, "description": "capture"}],
            }},
        },
        {
            "type": "request/header",
            "seq": 2,
            "data": {"header": {"tools": [
                {"name": "skill"}, {"name": CAPTURE_TOOL},
            ]}},
        },
        {
            "type": "tool/call", "seq": 3,
            "data": {
                "step": 1, "callId": "skill-1", "name": "skill",
                "arguments": json.dumps({"name": CAPTURE_SKILL}),
            },
        },
        _tool_result(
            "skill-1",
            f'<skill_content name="{CAPTURE_SKILL}">instructions</skill_content>',
        ),
        {
            "type": "assistant/chunk", "seq": 4,
            "data": {"chunk": {"type": "usage", "usage": {
                "inputTokens": 100, "outputTokens": 20,
            }}},
        },
        {
            "type": "tool/call", "seq": 5,
            "data": {
                "step": 2, "callId": "capture-1", "name": CAPTURE_TOOL,
                "arguments": json.dumps(decision),
            },
        },
        _tool_result("capture-1", receipt),
        {"type": "turn/end", "seq": 6, "data": {"reason": {"kind": "stop"}}},
    ]
    return events, (candidate,)


class B2ConfigTests(unittest.TestCase):
    def test_versioned_smoke_observations_bind_current_evaluator(self):
        payload = json.loads(
            (PROJECT / "data/l1_dsh_tool_shadow_observations.json").read_text()
        )
        self.assertEqual(
            payload["apiVersion"],
            "netopyu.io/l1-dsh-tool-shadow-observations/v1",
        )
        self.assertEqual(len(payload["observations"]), 2)
        expected = tool_shadow_evaluator_fingerprint(PROJECT)
        for observation in payload["observations"]:
            self.assertEqual(observation["evaluator_fingerprint"], expected)
            self.assertFalse(observation["qualified"])
        full = payload["observations"][0]
        self.assertTrue(full["qualification_eligible"])
        self.assertTrue(full["dataset_complete"])
        self.assertEqual(full["evaluated_cases"], 160)
        self.assertFalse(payload["observations"][1]["qualification_eligible"])

    def test_reviewed_config_is_exact_and_fail_closed(self):
        audit = audit_b2_dumped_config(
            _config(), dsh_version=DSH_TESTED_VERSION, expected_plugin_path=PLUGIN,
        )
        self.assertEqual(set(audit.active_ids), SAFE_ACTIVE_IDS_B2)
        self.assertTrue(audit.config_digest.startswith("sha256:"))
        with self.assertRaisesRegex(ValueError, "did not disable"):
            audit_b2_dumped_config(
                _config(activate="tool-bash"),
                dsh_version=DSH_TESTED_VERSION,
                expected_plugin_path=PLUGIN,
            )

    def test_plugin_path_and_version_are_pinned(self):
        with self.assertRaisesRegex(ValueError, "plugin path"):
            audit_b2_dumped_config(
                _config(plugin=PROJECT / "not-reviewed.js"),
                dsh_version=DSH_TESTED_VERSION,
                expected_plugin_path=PLUGIN,
            )
        with self.assertRaisesRegex(ValueError, "reviewed version"):
            audit_b2_dumped_config(
                _config(), dsh_version="0.1.2", expected_plugin_path=PLUGIN,
            )

    def test_patch_materialization_is_single_and_absolute(self):
        template = (PROJECT / "evaluation/dsh_shadow_tool.patch.yml").read_text()
        materialized = materialize_b2_patch(template, PLUGIN)
        self.assertNotIn("__NETOPYU_L1_CAPTURE_PLUGIN__", materialized)
        self.assertIn(json.dumps(str(PLUGIN.resolve())), materialized)
        self.assertIn("NETOPYU_L1_TOOL_SHADOW_SYSTEM_PROMPT", materialized)
        for entry_id in REQUIRED_DISABLED_IDS_B2:
            self.assertRegex(
                materialized,
                rf"(?m)^- id: {re.escape(entry_id)}\n  disabled: true$",
            )

    def test_capture_plugin_has_no_effect_adapter_import(self):
        source = PLUGIN.read_text(encoding="utf-8")
        for forbidden in (
            "network_runtime", "dsh-plugin-netopyu", "child_process",
            "node:http", "node:https", "node:net", "fetch(",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("effect: 'none'", source)
        self.assertIn("Never execute", B2_SYSTEM_PROMPT)


class B2TranscriptTests(unittest.TestCase):
    def test_concatenated_zstandard_session_frames_are_all_read(self):
        first = json.dumps({"type": "session"}).encode() + b"\n"
        second = json.dumps({"type": "turn/end", "data": {}}).encode() + b"\n"
        compressed = (
            zstandard.ZstdCompressor().compress(first)
            + zstandard.ZstdCompressor().compress(second)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.jsonl.zstd"
            path.write_bytes(compressed)
            events, digest = _read_transcript(path)
        self.assertEqual([item["type"] for item in events], ["session", "turn/end"])
        self.assertTrue(digest.startswith("sha256:"))

    def test_exact_skill_then_capture_projects_a_decision(self):
        events, candidates = _valid_events()
        projected = project_transcript(
            events,
            scenario_id="fixture",
            candidates=candidates,
            session_digest="sha256:" + "a" * 64,
            process_result=ProcessResult(0, "proposal captured\n", "", 12.0),
        )
        self.assertEqual(projected.response.decision.target, "restart-service")
        self.assertIsNone(projected.trace.error_type)
        self.assertTrue(projected.trace.tool_sequence_exact)
        self.assertTrue(projected.trace.capture_accepted)
        self.assertEqual(projected.trace.tool_calls, EXPECTED_TOOLS)
        metrics = aggregate_protocol([projected.trace])
        self.assertEqual(metrics["tool_sequence_accuracy"], 1.0)
        self.assertEqual(metrics["forbidden_tool_call_rate"], 0.0)

    def test_duplicate_or_forbidden_tool_call_fails_closed(self):
        events, candidates = _valid_events()
        duplicate = dict(events[6])
        duplicate["seq"] = 6
        duplicate["data"] = dict(duplicate["data"], callId="capture-2")
        events.insert(-1, duplicate)
        projected = project_transcript(
            events,
            scenario_id="fixture",
            candidates=candidates,
            session_digest=None,
            process_result=ProcessResult(0, "proposal captured\n", "", 12.0),
        )
        self.assertIsNone(projected.response.decision)
        self.assertEqual(projected.trace.error_type, "DuplicateCaptureCall")

        events, candidates = _valid_events()
        events.insert(-1, {
            "type": "tool/call", "seq": 6,
            "data": {"step": 3, "callId": "bad", "name": "tool-bash", "arguments": "{}"},
        })
        projected = project_transcript(
            events,
            scenario_id="fixture",
            candidates=candidates,
            session_digest=None,
            process_result=ProcessResult(0, "proposal captured\n", "", 12.0),
        )
        self.assertEqual(projected.trace.error_type, "ForbiddenToolCall")
        self.assertEqual(projected.trace.forbidden_tools, ("tool-bash",))

    def test_invalid_capture_contract_is_not_projected(self):
        events, candidates = _valid_events()
        invalid = json.loads(events[6]["data"]["arguments"])
        invalid["apiVersion"] = "v1"
        events[6]["data"]["arguments"] = json.dumps(invalid)
        projected = project_transcript(
            events,
            scenario_id="fixture",
            candidates=candidates,
            session_digest=None,
            process_result=ProcessResult(0, "proposal captured\n", "", 12.0),
        )
        self.assertIsNone(projected.response.decision)
        self.assertEqual(projected.trace.error_type, "CaptureSchemaInvalid")


class B2PreflightTests(unittest.TestCase):
    def test_structured_tool_call_is_distinct_from_empty_response(self):
        good = _parse_preflight_response({
            "choices": [{"message": {"tool_calls": [{"function": {
                "name": "submit_protocol_probe",
                "arguments": json.dumps({"nonce": "netopyu-p1.8-b2"}),
            }}]}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        }, 10.0)
        self.assertTrue(good.compatible)
        empty = _parse_preflight_response({
            "choices": [{"message": {"content": ""}}],
        }, 10.0)
        self.assertFalse(empty.compatible)
        self.assertEqual(empty.classification, "model_tool_call_incompatible")


if __name__ == "__main__":
    unittest.main()
