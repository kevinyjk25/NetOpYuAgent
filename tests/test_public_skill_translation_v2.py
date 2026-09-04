from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation.public_skill_fixture_mcp import validate_fixture_catalog
from evaluation.public_skill_paired import export_public_paired_study_kit
from evaluation.public_skill_translation_v2 import (
    BlockedDecision,
    ClarificationDecision,
    ParameterEvidence,
    ReadProposal,
    TranslationEnvelope,
    WriteProposal,
    _named_parameter_evidence,
    inspect_public_skill_translation_v2,
    link_catalog,
    run_public_skill_translation_v2,
    should_repair,
    translate_one,
)
from tests.test_public_skill_paired import _gold


def _catalog(*, duplicate_write: bool = False) -> tuple[Any, ...]:
    capabilities = [
        {
            "capabilityId": "record.read",
            "toolName": "read_record",
            "description": "Observe one record",
            "actionType": "read_only",
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["resource_id"],
                "properties": {"resource_id": {"type": "string"}},
            },
            "operation": {
                "kind": "read_record",
                "collection": "records",
                "keyArgument": "resource_id",
            },
        },
        {
            "capabilityId": "record.change",
            "toolName": "change_record",
            "description": "Change one record",
            "actionType": "reversible",
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["resource_id", "desired_state", "expected_revision"],
                "properties": {
                    "resource_id": {"type": "string"},
                    "desired_state": {"type": "string"},
                    "expected_revision": {"type": "integer", "minimum": 1},
                },
            },
            "operation": {
                "kind": "upsert_record",
                "collection": "records",
                "keyArgument": "resource_id",
                "valueArguments": {"state": "desired_state"},
                "revisionArgument": "expected_revision",
            },
        },
        {
            "capabilityId": "record.restore",
            "toolName": "restore_record",
            "description": "Restore one pre-change snapshot",
            "actionType": "reversible",
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["resource_id", "snapshot"],
                "properties": {
                    "resource_id": {"type": "string"},
                    "snapshot": {"type": "object"},
                },
            },
            "operation": {
                "kind": "restore_record",
                "collection": "records",
                "keyArgument": "resource_id",
                "snapshotArgument": "snapshot",
            },
        },
    ]
    if duplicate_write:
        duplicate = dict(capabilities[1])
        duplicate["capabilityId"] = "record.change.alternate"
        duplicate["toolName"] = "change_record_alternate"
        capabilities.insert(2, duplicate)
    return validate_fixture_catalog({
        "apiVersion": "effect-runtime.io/public-skill-tool-catalog/v2",
        "assignmentId": "translator-v2-test",
        "capabilities": capabilities,
    })


def _write(confidence: float = 0.01, hint: str | None = None) -> TranslationEnvelope:
    return TranslationEnvelope(decision=WriteProposal(
        kind="write_proposal",
        operation_intent="change",
        effect_semantics="reversible",
        capability_hint=hint,
        parameter_evidence=(),
        confidence=confidence,
        explanation="The user requests a reversible state transition.",
    ))


def test_named_parameter_evidence_accepts_sentence_punctuation() -> None:
    prompt = "Change resource_id=r1 and expected_revision=1."
    result = _named_parameter_evidence(
        prompt, "expected_revision", {"type": "integer", "minimum": 1},
    )
    assert result is not None
    value, source = result
    assert value == 1
    assert prompt[source["start"]:source["end"]] == "1"


def test_catalog_linker_closes_write_without_apply_keyword_or_model_ids() -> None:
    prompt = (
        "Please perform the approved transition with resource_id=r1, "
        "desired_state=ready, and expected_revision=1. Verify it independently."
    )
    link = link_catalog(_write(), _catalog(), prompt)
    assert link.status == "linked"
    assert link.route == "l0_write"
    assert link.primary_capability == "record.change"
    assert link.preflight_capabilities == ("record.read",)
    assert link.verification_capability == "record.read"
    assert link.compensation_capability == "record.restore"
    assert link.parameter_values == {
        "resource_id": "r1", "desired_state": "ready", "expected_revision": 1,
    }
    assert all(
        prompt[item["start"]:item["end"]] == item["sourceText"]
        for item in link.parameter_sources.values()
    )
    # A low model confidence cannot deny or grant authority; deterministic closure decides.
    assert _write().decision.confidence == 0.01


def test_catalog_linker_requires_unique_capability_and_allows_exact_catalog_hint() -> None:
    prompt = "Change resource_id=r1, desired_state=ready, expected_revision=1."
    ambiguous = link_catalog(_write(), _catalog(duplicate_write=True), prompt)
    assert ambiguous.status == "unlinked"
    assert ambiguous.route == "safe_stop"
    assert "primary_candidate_unique" in ambiguous.failures

    linked = link_catalog(
        _write(hint="record.change.alternate"), _catalog(duplicate_write=True), prompt,
    )
    assert linked.status == "linked"
    assert linked.primary_capability == "record.change.alternate"


def test_model_span_is_reconstructed_and_uniquely_realigned() -> None:
    prompt = "Inspect resource_id=router-17 now."
    evidence = ParameterEvidence(
        name="resource_id", value="router-17", source_text="router-17", start=0, end=9,
    )
    envelope = TranslationEnvelope(decision=ReadProposal(
        kind="read_proposal",
        operation_intent="inspect",
        parameter_evidence=(evidence,),
        confidence=0.8,
        explanation="Read request.",
    ))
    link = link_catalog(envelope, _catalog(), prompt)
    assert link.route == "l0_read"
    assert link.parameter_sources["resource_id"]["method"] == "model_span_uniquely_realigned"


def test_clarification_and_blocked_decisions_are_not_forced_into_repair() -> None:
    clarification = TranslationEnvelope(decision=ClarificationDecision(
        kind="clarification",
        missing_information=("resource_id",),
        confidence=0.9,
        explanation="Target absent.",
    ))
    blocked = TranslationEnvelope(decision=BlockedDecision(
        kind="blocked",
        reason_code="inert_script_execution",
        confidence=0.99,
        explanation="The task asks to execute package code.",
    ))
    for envelope in (clarification, blocked):
        link = link_catalog(envelope, _catalog(), "Do something unspecified")
        assert link.route == "safe_stop"
        assert should_repair(envelope, link) is False

    closed_prompt = "Change resource_id=r1, desired_state=ready, expected_revision=1."
    closed_link = link_catalog(clarification, _catalog(), closed_prompt)
    assert should_repair(clarification, closed_link, _catalog(), closed_prompt) is True


class _RepairingAdapter:
    def __init__(self) -> None:
        self.feedback: dict[str, Any] | None = None

    def preflight(self) -> dict[str, str]:
        return {"model": "qwen3.5:9b", "modelArtifactDigest": "sha256:" + "2" * 64}

    def translate(self, prompt: str):
        envelope = TranslationEnvelope(decision=WriteProposal(
            kind="write_proposal",
            operation_intent="delete",
            effect_semantics="reversible",
            parameter_evidence=(),
            confidence=0.4,
            explanation="Initial semantic miss.",
        ))
        return envelope, _telemetry(envelope)

    def repair(self, prompt: str, decision: TranslationEnvelope | None, feedback: dict[str, Any]):
        self.feedback = feedback
        envelope = _write()
        return envelope, _telemetry(envelope)


def _telemetry(envelope: TranslationEnvelope) -> dict[str, Any]:
    return {
        "raw": envelope.model_dump_json(),
        "rawProtocolValid": True,
        "modelCalls": 1,
        "inputTokens": 1,
        "outputTokens": 1,
        "latencyMs": 1.0,
        "error": None,
        "rawDigest": "sha256:" + "1" * 64,
    }


def test_semantic_repair_uses_deterministic_failures_without_gold_or_expected_route() -> None:
    adapter = _RepairingAdapter()
    _, link, telemetry, attempts = translate_one(
        adapter,
        "opaque model prompt",
        "Change resource_id=r1, desired_state=ready, expected_revision=1.",
        _catalog(),
    )
    assert link.route == "l0_write"
    assert telemetry["semanticRepairCount"] == 1
    assert len(attempts) == 2
    assert adapter.feedback is not None
    assert adapter.feedback["goldIncluded"] is False
    assert adapter.feedback["expectedCapabilityOrRoute"] is None


class _ReadAdapter:
    def preflight(self) -> dict[str, str]:
        return {"model": "qwen3.5:9b", "modelArtifactDigest": "sha256:" + "3" * 64}

    def translate(self, prompt: str):
        payload = json.loads(prompt)
        user_prompt = payload["case"]["userPrompt"]
        marker = "alice"
        start = user_prompt.index(marker)
        envelope = TranslationEnvelope(decision=ReadProposal(
            kind="read_proposal",
            operation_intent="read",
            parameter_evidence=(ParameterEvidence(
                name="user_id", value=marker, source_text=marker,
                start=start, end=start + len(marker),
            ),),
            confidence=0.2,
            explanation="Read one user.",
        ))
        return envelope, _telemetry(envelope)

    def repair(self, prompt: str, decision: TranslationEnvelope | None, feedback: dict[str, Any]):
        raise AssertionError("qualified read must not be repaired")


def test_v2_runner_is_gold_blind_and_seals_l1_l05_l0_trajectory(tmp_path: Path) -> None:
    author, gold = _gold(tmp_path)
    paired = tmp_path / "paired"
    export_public_paired_study_kit(gold, author, paired)
    output = tmp_path / "translation-v2"
    manifest = run_public_skill_translation_v2(paired, output, adapter=_ReadAdapter())
    inspected = inspect_public_skill_translation_v2(output)
    assert manifest["caseCount"] == 2
    assert inspected["routeCounts"] == {"l0_read": 2}
    assert inspected["goldIncluded"] is False
    assert inspected["runtimeOrDshExecuted"] is False
    assert all((path / "04-l0.json").is_file() for path in (output / "trajectories").iterdir())
    assert "PRIVATE_SCORING_ONLY" not in "".join(
        path.read_text(encoding="utf-8")
        for path in output.rglob("*") if path.is_file()
    )
