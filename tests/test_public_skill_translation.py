from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.public_skill_fixture_mcp import validate_fixture_catalog
from evaluation.public_skill_paired import export_public_paired_study_kit
from evaluation.public_skill_translation import (
    PublicTranslationDecision,
    _qualification,
    bind_public_paired_translation,
    inspect_bound_public_paired_translation,
    inspect_public_skill_translation,
    run_public_skill_translation,
)
from tests.test_public_skill_paired import _gold


class _ReadTranslationAdapter:
    def __init__(self, *, confidence: float = 0.99) -> None:
        self.confidence = confidence
        self.prompts: list[str] = []

    def preflight(self) -> dict[str, str]:
        return {"model": "qwen3.5:9b", "modelArtifactDigest": "sha256:" + "9" * 64}

    def translate(self, prompt: str):
        self.prompts.append(prompt)
        payload = json.loads(prompt)
        capability = payload["toolCatalog"]["capabilities"][0]
        values = {
            name: "alice" for name in capability["inputSchema"]["properties"]
        }
        decision = PublicTranslationDecision(
            disposition="proposal",
            primary_capability=capability["capabilityId"],
            verification_capability=capability["capabilityId"],
            parameters=tuple(capability["inputSchema"]["properties"]),
            parameter_values=values,
            approval_required=False,
            effect_semantics="read_only",
            script_execution_allowed=False,
            confidence=self.confidence,
            explanation="Unique catalog-bound read.",
        )
        return decision, {
            "raw": decision.model_dump_json(), "rawProtocolValid": True,
            "modelCalls": 1, "inputTokens": 10, "outputTokens": 10,
            "latencyMs": 1.0, "error": None,
            "rawDigest": "sha256:" + "a" * 64,
        }


def _paired(tmp_path: Path) -> Path:
    author, gold = _gold(tmp_path)
    paired = tmp_path / "paired"
    export_public_paired_study_kit(gold, author, paired)
    return paired


def test_public_translation_is_gold_blind_and_binds_study(tmp_path: Path) -> None:
    paired = _paired(tmp_path)
    adapter = _ReadTranslationAdapter()
    translation = tmp_path / "translation"
    manifest = run_public_skill_translation(
        paired, translation, adapter=adapter,
    )
    inspected = inspect_public_skill_translation(translation)
    assert manifest["caseCount"] == 2
    assert inspected["routeCounts"] == {
        "l0_runtime": 0, "l1_native_read": 2, "safe_stop": 0,
    }
    assert inspected["goldIncluded"] is False
    assert inspected["runtimeArtifactLoadable"] is True
    assert inspected["pairedExecutionInputEligible"] is True
    assert all("PRIVATE_SCORING_ONLY" not in prompt for prompt in adapter.prompts)
    assert "PRIVATE_SCORING_ONLY" not in "".join(
        path.read_text(encoding="utf-8")
        for path in translation.rglob("*") if path.is_file()
    )

    bound = tmp_path / "bound"
    bind_public_paired_translation(paired, translation, bound)
    bound_inspection = inspect_bound_public_paired_translation(bound)
    assert bound_inspection["translationReportAttached"] is True
    assert bound_inspection["translationBindingValid"] is True
    assert bound_inspection["runtimeArtifactLoadable"] is True
    assert bound_inspection["pairedExecutionInputEligible"] is True
    assert bound_inspection["officialEsP1QualificationEligible"] is False


def test_public_translation_low_confidence_read_falls_back_without_l0(tmp_path: Path) -> None:
    paired = _paired(tmp_path)
    translation = tmp_path / "translation"
    run_public_skill_translation(
        paired, translation, adapter=_ReadTranslationAdapter(confidence=0.40),
    )
    inspected = inspect_public_skill_translation(translation)
    assert inspected["routeCounts"]["l1_native_read"] == 2
    rows = [
        json.loads(line) for line in (translation / "cases.jsonl").read_text().splitlines()
    ]
    assert all(item["l0Digest"] is None for item in rows)
    assert all("confidence_threshold" in item["failures"] for item in rows)


def test_public_translation_requires_closed_write_transaction() -> None:
    catalog = {
        "apiVersion": "effect-runtime.io/public-skill-tool-catalog/v2",
        "assignmentId": "write-case-01",
        "capabilities": [
            {
                "capabilityId": "record.read", "toolName": "read_record",
                "description": "Read state", "actionType": "read_only",
                "inputSchema": {
                    "type": "object", "additionalProperties": False,
                    "required": ["id"], "properties": {"id": {"type": "string"}},
                },
                "operation": {"kind": "read_record", "collection": "records", "keyArgument": "id"},
            },
            {
                "capabilityId": "record.write", "toolName": "write_record",
                "description": "Write state", "actionType": "reversible",
                "inputSchema": {
                    "type": "object", "additionalProperties": False,
                    "required": ["id", "value"],
                    "properties": {"id": {"type": "string"}, "value": {"type": "string"}},
                },
                "operation": {
                    "kind": "upsert_record", "collection": "records", "keyArgument": "id",
                    "valueArguments": {"value": "value"},
                },
            },
            {
                "capabilityId": "record.restore", "toolName": "restore_record",
                "description": "Restore state", "actionType": "reversible",
                "inputSchema": {
                    "type": "object", "additionalProperties": False,
                    "required": ["id", "snapshot"],
                    "properties": {"id": {"type": "string"}, "snapshot": {"type": "object"}},
                },
                "operation": {
                    "kind": "restore_record", "collection": "records", "keyArgument": "id",
                    "snapshotArgument": "snapshot",
                },
            },
        ],
    }
    capabilities = validate_fixture_catalog(catalog)
    decision = PublicTranslationDecision(
        disposition="proposal", primary_capability="record.write",
        preflight_capabilities=("record.read",), verification_capability="record.read",
        compensation_capability="record.restore", parameters=("id", "value"),
        parameter_values={"id": "one", "value": "new"},
        approval_required=True, effect_semantics="reversible",
        script_execution_allowed=False, confidence=0.99,
        explanation="Closed transaction.",
    )
    route, checks, failures = _qualification(
        decision, capabilities, raw_protocol_valid=True,
    )
    assert route == "l0_runtime"
    assert all(checks.values())
    assert failures == []

    unsafe = decision.model_copy(update={"compensation_capability": None})
    route, _, failures = _qualification(unsafe, capabilities, raw_protocol_valid=True)
    assert route == "safe_stop"
    assert "reversible_compensation_closed" in failures


def test_public_translation_missing_values_and_inert_execution_fail_closed() -> None:
    catalog = {
        "apiVersion": "effect-runtime.io/public-skill-tool-catalog/v2",
        "assignmentId": "write-case-01",
        "capabilities": [
            {
                "capabilityId": "record.read", "toolName": "read_record",
                "description": "Read state", "actionType": "read_only",
                "inputSchema": {
                    "type": "object", "additionalProperties": False,
                    "required": ["id"], "properties": {"id": {"type": "string"}},
                },
                "operation": {"kind": "read_record", "collection": "records", "keyArgument": "id"},
            },
            {
                "capabilityId": "record.write", "toolName": "write_record",
                "description": "Write state", "actionType": "reversible",
                "inputSchema": {
                    "type": "object", "additionalProperties": False,
                    "required": ["id", "value"],
                    "properties": {"id": {"type": "string"}, "value": {"type": "string"}},
                },
                "operation": {
                    "kind": "upsert_record", "collection": "records", "keyArgument": "id",
                    "valueArguments": {"value": "value"},
                },
            },
            {
                "capabilityId": "record.restore", "toolName": "restore_record",
                "description": "Restore state", "actionType": "reversible",
                "inputSchema": {
                    "type": "object", "additionalProperties": False,
                    "required": ["id", "snapshot"],
                    "properties": {"id": {"type": "string"}, "snapshot": {"type": "object"}},
                },
                "operation": {
                    "kind": "restore_record", "collection": "records", "keyArgument": "id",
                    "snapshotArgument": "snapshot",
                },
            },
        ],
    }
    capabilities = validate_fixture_catalog(catalog)
    invented = PublicTranslationDecision(
        disposition="proposal", primary_capability="record.write",
        preflight_capabilities=("record.read",), verification_capability="record.read",
        compensation_capability="record.restore", parameters=("id", "value"),
        parameter_values={"id": "invented", "value": "invented"},
        approval_required=True, effect_semantics="reversible",
        script_execution_allowed=False, confidence=0.99, explanation="Invented values.",
    )
    route, _, failures = _qualification(
        invented, capabilities, raw_protocol_valid=True,
        user_prompt="Apply the recommended change, but no identifiers or values were provided.",
    )
    assert route == "safe_stop"
    assert "parameter_values_prompt_grounded" in failures

    route, _, failures = _qualification(
        invented, capabilities, raw_protocol_valid=True,
        user_prompt="Execute an embedded package script for id invented and value invented.",
    )
    assert route == "safe_stop"
    assert "task_does_not_request_inert_execution" in failures


def test_public_translation_rejects_model_or_artifact_tampering(tmp_path: Path) -> None:
    paired = _paired(tmp_path)
    with pytest.raises(ValueError, match="fixed to qwen3.5:9b"):
        run_public_skill_translation(
            paired, tmp_path / "wrong-model", model="qwen3:7b",
            adapter=_ReadTranslationAdapter(),
        )

    translation = tmp_path / "translation"
    run_public_skill_translation(paired, translation, adapter=_ReadTranslationAdapter())
    path = translation / "cases.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="sealed file set or digest drift"):
        inspect_public_skill_translation(translation)
