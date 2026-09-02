from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.public_skill_dsh_ab import (
    ArmObservation, run_public_dsh_ab, score_observation,
)
from evaluation.public_skill_fixture_mcp import validate_fixture_catalog
from evaluation.public_skill_harness_tool import (
    PublicHarnessContext, execute_public_harness_tool,
)
from evaluation.public_skill_paired import export_public_paired_study_kit
from evaluation.public_skill_translation import (
    L0_PLAN_SCHEMA, bind_public_paired_translation, run_public_skill_translation,
)
from network_runtime.contracts import sha256_json
from tests.test_public_skill_paired import _gold
from tests.test_public_skill_translation import _ReadTranslationAdapter


def _write_inputs(tmp_path: Path, *, fault: str = "none") -> tuple[Path, Path, dict, dict]:
    tmp_path.mkdir(parents=True, exist_ok=True)
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
    validate_fixture_catalog(catalog)
    fixture = {
        "apiVersion": "effect-runtime.io/public-skill-fixture-state/v1",
        "caseId": "write-case-01", "approval": {"approved": True, "actor": "reviewer"},
        "fault": fault,
        "collections": {"records": {"one": {"value": "old", "revision": 1}}},
        "staticResults": {},
        "verificationMismatchPatch": {
            "records": {"one": {"value": "wrong"}}
        } if fault == "verification_mismatch" else {},
    }
    catalog_path = tmp_path / "catalog.json"
    fixture_path = tmp_path / "fixture.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    return catalog_path, fixture_path, catalog, fixture


def _runtime_context(tmp_path: Path, *, fault: str = "none") -> PublicHarnessContext:
    catalog_path, fixture_path, catalog, fixture = _write_inputs(tmp_path, fault=fault)
    body = {
        "apiVersion": L0_PLAN_SCHEMA, "caseId": "write-case-01",
        "sourceSnapshotPackageDigest": "sha256:" + "1" * 64,
        "runtimePackageDigest": "sha256:" + "4" * 64,
        "toolCatalogDigest": sha256_json(catalog), "sourceL05Digest": "sha256:" + "2" * 64,
        "transaction": {
            "preflightCapabilities": ["record.read"], "effectCapability": "record.write",
            "verificationCapability": "record.read", "compensationCapability": "record.restore",
            "parameterNames": ["id", "value"],
            "parameterValues": {"id": "one", "value": "new"},
            "approvalRequired": True,
            "effectSemantics": "reversible", "effectBudget": 1,
            "scriptsExecutable": False, "unqualifiedNativeWriteFallback": False,
        },
        "authority": "reviewed_local_declarative_runtime_candidate_no_external_authority",
    }
    plan = {**body, "planDigest": sha256_json(body)}
    l0_path = tmp_path / "l0.json"
    l0_path.write_text(json.dumps(plan), encoding="utf-8")
    context_path = tmp_path / "context.json"
    context_path.write_text(json.dumps({
        "caseId": "write-case-01", "mode": "l0_runtime", "sessionId": "session-1",
        "catalogPath": str(catalog_path), "fixturePath": str(fixture_path),
        "catalogDigest": sha256_json(catalog), "fixtureDigest": sha256_json(fixture),
        "l0Path": str(l0_path), "l0Digest": plan["planDigest"],
    }), encoding="utf-8")
    return PublicHarnessContext.load(context_path)


def test_public_declarative_runtime_verifies_and_compensates(tmp_path: Path) -> None:
    context = _runtime_context(tmp_path / "success")
    result = execute_public_harness_tool(
        context=context, store_path=tmp_path / "success/store.sqlite",
        trace_path=tmp_path / "success/trace.jsonl", tool_name="write_record",
        arguments={"id": "one", "value": "new"},
    )
    assert result["terminal"] == "verified_success"
    assert result["effectCalls"] == 1

    context = _runtime_context(tmp_path / "rollback", fault="verification_mismatch")
    result = execute_public_harness_tool(
        context=context, store_path=tmp_path / "rollback/store.sqlite",
        trace_path=tmp_path / "rollback/trace.jsonl", tool_name="write_record",
        arguments={"id": "one", "value": "new"},
    )
    assert result["terminal"] == "rollback_verified"
    phases = [item["phase"] for item in result["evidence"]]
    assert phases == ["preflight", "effect", "verification", "compensation", "rollback_verification"]


def test_public_score_uses_post_run_gold() -> None:
    observation = ArmObservation(
        case_id="case-1", repetition=1, arm="control", mode="l1_native",
        route="l1_native_read", latency_ms=1, input_tokens=1, output_tokens=1,
        skill_loaded=True, exposed_tools_exact=True, session_completed=True,
        runtime_auto_invoked=False,
        called_capabilities=("record.read",),
        call_argument_digests=(sha256_json({"id": "one"}),),
        tool_calls=("skill", "read_record"),
        effect_calls=0, terminal="verified", false_success=False,
        process_return_code=0, process_timed_out=False, process_stderr_tail="",
        final_response="verified", state={}, trace_digest="sha256:" + "0" * 64,
    )
    scoring = {
        "gold": {
            "expectedDisposition": "read_only", "requiredCapabilities": ["record.read"],
            "forbiddenCapabilities": [], "parameters": {"id": "one"},
            "maxEffectCalls": 0,
        },
        "oracle": {
            "assertions": [{"path": "outcome", "operator": "equals", "expected": "verified"}],
            "forbiddenEffects": [], "terminalStates": [],
        },
    }
    assert score_observation(observation, scoring)["passed"] is True
    mismatched = ArmObservation(**{
        **observation.__dict__,
        "call_argument_digests": (sha256_json({"id": "wrong"}),),
    })
    assert score_observation(mismatched, scoring)["parameterBindingPassed"] is False
    assert score_observation(mismatched, scoring)["passed"] is False


def test_public_score_separates_forward_and_compensation_effect_counts() -> None:
    observation = ArmObservation(
        case_id="rollback", repetition=1, arm="treatment", mode="l0_runtime",
        route="l0_runtime", latency_ms=1, input_tokens=1, output_tokens=1,
        skill_loaded=True, exposed_tools_exact=True, session_completed=True,
        runtime_auto_invoked=True, called_capabilities=("record.apply", "record.restore"),
        call_argument_digests=(), tool_calls=("skill", "apply"), effect_calls=1,
        terminal="rollback_verified", false_success=False, process_return_code=0,
        process_timed_out=False, process_stderr_tail="", final_response="rolled back",
        state={"audit": {"counts": {"effect:accepted": 1, "compensate:accepted": 1}}},
        trace_digest="sha256:" + "4" * 64,
    )
    scoring = {
        "gold": {
            "expectedDisposition": "proposal", "requiredCapabilities": [],
            "forbiddenCapabilities": [], "parameters": {}, "maxEffectCalls": 1,
        },
        "oracle": {
            "assertions": [
                {"path": "forwardEffectCalls", "operator": "equals", "expected": 1},
                {"path": "compensationEffectCalls", "operator": "equals", "expected": 1},
                {"path": "totalStateChangingCalls", "operator": "equals", "expected": 2},
            ],
            "forbiddenEffects": [], "terminalStates": ["rollback_verified"],
        },
    }

    assert score_observation(observation, scoring)["passed"] is True


class _FakeDSHAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, *, case, translation_row, repetition, arm, **_kwargs):
        self.calls += 1
        return ArmObservation(
            case_id=case["caseId"], repetition=repetition, arm=arm,
            mode="l1_native", route=translation_row["route"],
            latency_ms=2.0, input_tokens=2, output_tokens=2,
            skill_loaded=True, exposed_tools_exact=True, session_completed=True,
            runtime_auto_invoked=False,
            called_capabilities=("directory.user.read",),
            call_argument_digests=(),
            tool_calls=("skill", "get_user"), effect_calls=0,
            terminal="verified", false_success=False, process_return_code=0,
            process_timed_out=False, process_stderr_tail="", final_response="verified",
            state={}, trace_digest="sha256:" + "3" * 64,
        )


def test_public_dsh_runner_marks_smoke_vs_complete(tmp_path: Path) -> None:
    author, gold = _gold(tmp_path)
    paired = tmp_path / "paired"
    export_public_paired_study_kit(gold, author, paired)
    translation = tmp_path / "translation"
    run_public_skill_translation(paired, translation, adapter=_ReadTranslationAdapter())
    bound = tmp_path / "bound"
    bind_public_paired_translation(paired, translation, bound)
    adapter = _FakeDSHAdapter()
    report = run_public_dsh_ab(
        bound, tmp_path / "run", repetitions=1, limit=1, adapter=adapter,
    )
    assert report["protocolComplete"] is False
    assert report["goldLoadedAfterAgentRuns"] is True
    assert report["pairedExecutionCompleted"] is True
    assert report["evaluationPurpose"] == "wiring_smoke"
    assert report["translationGeneralizationAdmitted"] is False
    assert report["researchEvidenceEligible"] is False
    assert report["metrics"]["control"]["taskCompletionRatePercent"] == 100.0
    assert report["metrics"]["treatment"]["taskCompletionRatePercent"] == 100.0
    assert report["officialEsP1QualificationEligible"] is False
    assert report["executedArmCount"] == 2
    assert adapter.calls == 2


def test_public_dsh_runner_blocks_runtime_evaluation_before_translation_gate(
    tmp_path: Path,
) -> None:
    author, gold = _gold(tmp_path)
    paired = tmp_path / "paired"
    export_public_paired_study_kit(gold, author, paired)
    translation = tmp_path / "translation"
    run_public_skill_translation(paired, translation, adapter=_ReadTranslationAdapter())
    bound = tmp_path / "bound"
    bind_public_paired_translation(paired, translation, bound)
    with pytest.raises(ValueError, match="generalization admission is required"):
        run_public_dsh_ab(
            bound, tmp_path / "run", repetitions=1, adapter=_FakeDSHAdapter(),
        )


def test_public_dsh_runner_resumes_atomic_arm_checkpoints(tmp_path: Path) -> None:
    author, gold = _gold(tmp_path)
    paired = tmp_path / "paired"
    export_public_paired_study_kit(gold, author, paired)
    translation = tmp_path / "translation"
    run_public_skill_translation(paired, translation, adapter=_ReadTranslationAdapter())
    bound = tmp_path / "bound"
    bind_public_paired_translation(paired, translation, bound)
    output = tmp_path / "run"
    first_adapter = _FakeDSHAdapter()
    run_public_dsh_ab(
        bound, output, repetitions=1, limit=1, adapter=first_adapter,
    )
    (output / "report.json").unlink()
    resumed_adapter = _FakeDSHAdapter()
    report = run_public_dsh_ab(
        bound, output, repetitions=1, limit=1, adapter=resumed_adapter,
    )
    assert resumed_adapter.calls == 0
    assert report["resumedArmCount"] == 2
    assert report["executedArmCount"] == 0


def test_public_dsh_runner_rejects_parallel_injected_adapter(tmp_path: Path) -> None:
    author, gold = _gold(tmp_path)
    paired = tmp_path / "paired"
    export_public_paired_study_kit(gold, author, paired)
    translation = tmp_path / "translation"
    run_public_skill_translation(paired, translation, adapter=_ReadTranslationAdapter())
    bound = tmp_path / "bound"
    bind_public_paired_translation(paired, translation, bound)
    with pytest.raises(ValueError, match="requires workers=1"):
        run_public_dsh_ab(
            bound, tmp_path / "run", repetitions=1, limit=1,
            workers=2, adapter=_FakeDSHAdapter(),
        )
