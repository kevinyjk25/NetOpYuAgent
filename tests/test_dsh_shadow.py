from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from evaluation.dsh_shadow import (
    DSHShadowAdapter,
    DSH_TESTED_VERSION,
    REQUIRED_DISABLED_IDS,
    SAFE_ACTIVE_IDS,
    _validated_decision,
    _reference_delta,
    audit_dumped_config,
    parse_dumped_config,
    shadow_evaluator_fingerprint,
)
from evaluation.l1_catalog import build_profile_catalog


def _config(*, activate: str | None = None, omit: str | None = None) -> str:
    blocks: list[str] = []
    for entry_id in sorted(SAFE_ACTIVE_IDS | REQUIRED_DISABLED_IDS):
        if entry_id == omit:
            continue
        disabled = entry_id in REQUIRED_DISABLED_IDS and entry_id != activate
        block = f"- id: {entry_id}\n  name: '@deepseek-ai/dsh-{entry_id}'\n"
        if disabled:
            block += "  disabled: true\n"
        if entry_id == "system-prompt":
            block += (
                "  config:\n"
                "    persona: !!js process.env.NETOPYU_L1_SHADOW_SYSTEM_PROMPT\n"
            )
        blocks.append(block)
    return "".join(blocks)


class DSHShadowConfigTests(unittest.TestCase):
    def test_versioned_baseline_is_bound_to_current_shadow_evaluator(self):
        project = Path(__file__).resolve().parents[1]
        payload = json.loads(
            (project / "data/l1_dsh_shadow_baselines.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["apiVersion"], "netopyu.io/l1-dsh-shadow-baselines/v1")
        baseline = payload["baselines"][0]
        self.assertEqual(
            baseline["evaluator_fingerprint"], shadow_evaluator_fingerprint(project),
        )
        self.assertFalse(baseline["qualified"])

    def test_reviewed_config_is_accepted(self):
        text = _config()
        parsed = parse_dumped_config(text)
        audit = audit_dumped_config(text, dsh_version=DSH_TESTED_VERSION)
        self.assertEqual(len(parsed), len(SAFE_ACTIVE_IDS | REQUIRED_DISABLED_IDS))
        self.assertEqual(set(audit.active_ids), SAFE_ACTIVE_IDS)
        self.assertTrue(audit.config_digest.startswith("sha256:"))

    def test_new_or_reactivated_plugin_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "did not disable"):
            audit_dumped_config(
                _config(activate="tool-bash"), dsh_version=DSH_TESTED_VERSION,
            )
        text = _config() + "- id: new-network-tool\n  name: '@x/new-network-tool'\n"
        with self.assertRaisesRegex(ValueError, "allowlist mismatch"):
            audit_dumped_config(text, dsh_version=DSH_TESTED_VERSION)

    def test_version_change_requires_review(self):
        with self.assertRaisesRegex(ValueError, "reviewed version"):
            audit_dumped_config(_config(), dsh_version="0.1.2")

    def test_source_patch_explicitly_disables_every_required_entry(self):
        patch = (
            Path(__file__).resolve().parents[1] / "evaluation/dsh_shadow.patch.yml"
        ).read_text(encoding="utf-8")
        for entry_id in REQUIRED_DISABLED_IDS:
            self.assertRegex(
                patch,
                rf"(?m)^- id: {re.escape(entry_id)}\n  disabled: true$",
                msg=f"{entry_id} is not fail-closed in the source patch",
            )

    def test_remote_endpoint_is_rejected_before_dsh_starts(self):
        with self.assertRaisesRegex(ValueError, "loopback"):
            DSHShadowAdapter(
                project_root=Path(__file__).resolve().parents[1],
                model="test",
                base_url="https://models.example.test",
            )


class DSHShadowDecisionTests(unittest.TestCase):
    def test_partial_metrics_are_not_compared_to_the_full_reference(self):
        result = _reference_delta(
            Path("does-not-matter.json"),
            dataset_complete=False,
            model="qwen2.5:7b",
            model_artifact_digest="sha256:" + "a" * 64,
            metrics={},
        )
        self.assertEqual(result["status"], "subset-not-comparable")

    def test_selection_reuses_reference_candidate_contract(self):
        candidate = next(
            item for item in build_profile_catalog("lan")
            if item.target == "restart-service" and item.kind == "skill"
        )
        valid = {
            "action": "select_skill",
            "target": "restart-service",
            "arguments": {"service": "crm", "environment": "prod"},
            "missing_fields": [],
            "workflow": [],
            "confidence": 0.9,
            "reason_code": "explicit_restart",
        }
        decision = _validated_decision(json.dumps(valid), (candidate,))
        self.assertEqual(decision.target, "restart-service")

        invalid = {**valid, "arguments": {"service": "crm"}}
        with self.assertRaisesRegex(ValueError, "omitted required"):
            _validated_decision(json.dumps(invalid), (candidate,))


if __name__ == "__main__":
    unittest.main()
