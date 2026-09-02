from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

from effect_runtime.mcp_lab import DOMAINS, EffectLabStore
from evaluation.general_effect_dataset import build_cases
from evaluation.harness_effect_tool import HarnessToolContext, execute_harness_tool
from evaluation.harness_skill_runtime_ab import (
    STRATIFIED_PATTERN_CASE_IDS, _claims_success, _classify, _is_false_commit,
    _external_stratified_cases, _is_invalid_action, _maximum_effect_calls,
    load_translation_routes,
)
from network_runtime.contracts import sha256_json


def _arguments() -> dict[str, object]:
    return {
        "entity_id": "edge-sw-01",
        "desired_value": "vlan-120",
        "expected_revision": 1,
        "change_id": "chg-test",
        "reason": "test",
    }


def test_harness_tool_native_and_runtime_share_public_effect(tmp_path: Path) -> None:
    native_store = tmp_path / "native.sqlite"
    EffectLabStore(native_store).reset()
    native = HarnessToolContext(
        "case", "network", "l1_native", True, "none",
        "effect.network.state.apply", "native-session", "dsh",
        _arguments(), sha256_json(_arguments()),
    )
    native_result = asyncio.run(execute_harness_tool(
        context=native, store_path=native_store,
        journal_path=tmp_path / "native-runtime.sqlite",
        trace_path=tmp_path / "native.jsonl",
        tool_name="network_apply_change", arguments=_arguments(),
    ))
    assert native_result["ok"] is True
    assert native_result["execution"] == "l1_native"
    assert native_result["code"] == "effect_accepted"

    runtime_store = tmp_path / "runtime.sqlite"
    EffectLabStore(runtime_store).reset()
    runtime = HarnessToolContext(
        "case", "network", "l0_runtime", True, "none",
        "effect.network.state.apply", "runtime-session", "dsh",
        _arguments(), sha256_json(_arguments()),
    )
    runtime_result = asyncio.run(execute_harness_tool(
        context=runtime, store_path=runtime_store,
        journal_path=tmp_path / "journal.sqlite",
        trace_path=tmp_path / "runtime.jsonl",
        tool_name="network_apply_change", arguments=_arguments(),
    ))
    assert runtime_result["ok"] is True
    assert runtime_result["execution"] == "l0_runtime"
    assert runtime_result["terminal"] == "verified_success"
    assert runtime_result["auditValid"] is True
    assert runtime_result["argumentBinding"]["code"] == "argument_binding_verified"


def test_runtime_rejects_a_field_dropped_between_request_and_tool_call(
    tmp_path: Path,
) -> None:
    request = {**_arguments(), "invented_scope": "all"}
    context = HarnessToolContext(
        "case", "network", "l0_runtime", True, "none",
        "effect.network.state.apply", "runtime-session", "dsh",
        request, sha256_json(request),
    )
    store = tmp_path / "runtime.sqlite"
    EffectLabStore(store).reset()
    result = asyncio.run(execute_harness_tool(
        context=context, store_path=store,
        journal_path=tmp_path / "journal.sqlite",
        trace_path=tmp_path / "runtime.jsonl",
        tool_name="network_apply_change", arguments=_arguments(),
    ))
    assert result["ok"] is False
    assert result["code"] == "request_binding_rejected"
    assert result["terminal"] == "rejected"
    assert result["argumentBinding"]["unknown_source_fields"] == ["invented_scope"]
    assert EffectLabStore(store).effect_calls(domain="network") == 0


def test_agent_cannot_directly_compensate_a_runtime_managed_transaction(
    tmp_path: Path,
) -> None:
    context = HarnessToolContext(
        "case", "network", "l0_runtime", True, "none",
        "effect.network.state.apply", "runtime-session", "dsh",
        _arguments(), sha256_json(_arguments()),
    )
    store = tmp_path / "runtime.sqlite"
    backend = EffectLabStore(store)
    backend.reset()
    snapshot = backend.state("network", "edge-sw-01")
    result = asyncio.run(execute_harness_tool(
        context=context, store_path=store,
        journal_path=tmp_path / "journal.sqlite",
        trace_path=tmp_path / "runtime.jsonl",
        tool_name="network_restore_state",
        arguments={
            "entity_id": "edge-sw-01",
            "approved_preflight": {"facts": snapshot},
        },
    ))
    assert result["ok"] is False
    assert result["code"] == "runtime_managed_compensation"
    assert EffectLabStore(store).effect_calls(domain="network") == 0


def test_unqualified_translation_safe_stop_never_mutates_provider(tmp_path: Path) -> None:
    context = HarnessToolContext(
        "case", "network", "safe_stop", True, "none",
        "effect.network.state.apply", "safe-stop-session", "dsh",
        _arguments(), sha256_json(_arguments()),
    )
    store = tmp_path / "safe-stop.sqlite"
    backend = EffectLabStore(store)
    backend.reset()
    result = asyncio.run(execute_harness_tool(
        context=context, store_path=store,
        journal_path=tmp_path / "journal.sqlite",
        trace_path=tmp_path / "safe-stop.jsonl",
        tool_name="network_apply_change", arguments=_arguments(),
    ))
    assert result["ok"] is False
    assert result["code"] == "translation_unqualified_safe_stop"
    assert result["terminal"] == "rejected"
    assert result["execution"] == "safe_stop"
    assert EffectLabStore(store).effect_calls(domain="network") == 0


def test_safe_stop_missing_field_returns_clarification(tmp_path: Path) -> None:
    authoritative = _arguments()
    authoritative.pop("reason")
    context = HarnessToolContext(
        "case", "network", "safe_stop", True, "none",
        "effect.network.state.apply", "safe-stop-session", "dsh",
        authoritative, sha256_json(authoritative),
    )
    store = tmp_path / "safe-stop-missing.sqlite"
    EffectLabStore(store).reset()
    result = asyncio.run(execute_harness_tool(
        context=context, store_path=store,
        journal_path=tmp_path / "journal.sqlite",
        trace_path=tmp_path / "safe-stop-missing.jsonl",
        tool_name="network_apply_change", arguments=authoritative,
    ))
    assert result["code"] == "clarification_required"
    assert result["terminal"] == "clarification_required"
    assert result["reason"] == "missing_parameters:reason"
    assert EffectLabStore(store).effect_calls(domain="network") == 0


def test_safe_stop_does_not_accept_a_model_invented_missing_field(
    tmp_path: Path,
) -> None:
    authoritative = _arguments()
    authoritative.pop("reason")
    context = HarnessToolContext(
        "case", "network", "safe_stop", True, "none",
        "effect.network.state.apply", "safe-stop-invention-session", "dsh",
        authoritative, sha256_json(authoritative),
    )
    store = tmp_path / "safe-stop-invention.sqlite"
    EffectLabStore(store).reset()
    result = asyncio.run(execute_harness_tool(
        context=context, store_path=store,
        journal_path=tmp_path / "journal.sqlite",
        trace_path=tmp_path / "safe-stop-invention.jsonl",
        tool_name="network_apply_change",
        arguments={**authoritative, "reason": "invented by the model"},
    ))
    assert result["code"] == "clarification_required"
    assert result["terminal"] == "clarification_required"
    assert result["reason"] == "missing_parameters:reason"
    assert EffectLabStore(store).effect_calls(domain="network") == 0


def test_translation_failure_routes_to_safe_stop(tmp_path: Path) -> None:
    case = build_cases()[0]
    digest = "sha256:" + "a" * 64
    report = {
        "schema": "effect-runtime.io/general-effect-model-translation/v1",
        "dataset": {"digest": "sha256:dataset"},
        "model": {"modelArtifactDigest": "sha256:" + "b" * 64},
        "cases": [{
            "case_id": case.case_id,
            "raw_protocol_valid": True,
            "package_gate": "passed",
            "oracle_passed": False,
            "semantic_coverage_percent": 91.67,
            "route": "proposal_only",
            "l0_digest": None,
        }],
    }
    path = tmp_path / "model-translation.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    routes = load_translation_routes(
        report_path=path, dataset_digest="sha256:dataset",
        current_l0_digests={case.l0_skill_id: digest}, cases=[case],
    )
    route = routes[case.case_id]
    assert route.qualified is False
    assert route.execution == "safe_stop"
    assert "semantic_oracle_passed" in route.failures
    assert "l0_digest_current" in route.failures


def test_native_success_without_post_read_is_not_verified() -> None:
    case = build_cases()[0]
    projection = {
        "calls": [{
            "seq": 2,
            "name": "network_apply_change",
            "result": {"code": "effect_accepted", "execution": "l1_native"},
        }],
    }
    terminal, verified, compensated, audit, _ = _classify(
        case=case, projection=projection,
        actual=str(case.arguments["desired_value"]), backend_mode="l1_native",
    )
    assert terminal == "unverified_success"
    assert verified is False
    assert compensated is False
    assert audit is None


def test_safe_stop_clarification_is_projected_from_tool_result() -> None:
    case = next(item for item in build_cases() if item.case_id == "steps-02")
    projection = {
        "calls": [{
            "seq": 2,
            "name": "data_apply_change",
            "result": {
                "code": "clarification_required",
                "terminal": "clarification_required",
                "execution": "safe_stop",
            },
        }],
    }
    terminal, verified, compensated, audit, detail = _classify(
        case=case, projection=projection, actual="internal",
        backend_mode="safe_stop",
    )
    assert terminal == "clarification_required"
    assert verified is False
    assert compensated is False
    assert audit is None
    assert detail.startswith("Safe-stop")


def test_step_completion_language_does_not_override_missing_parameter() -> None:
    assert not _claims_success(
        "Step 1 completed. Critical issue: required parameter reason is missing."
    )


def test_skill_load_success_does_not_override_denied_write() -> None:
    assert not _claims_success(
        "Skill loaded successfully. Approval DENIED. No write was executed; "
        "the requested change was not applied."
    )


def test_explicit_execution_success_is_detected() -> None:
    assert _claims_success("Execution result: success; post-state independently verified.")


def test_verified_write_without_required_intent_is_a_false_commit() -> None:
    case = next(item for item in build_cases() if item.case_id == "steps-02")
    assert _is_false_commit(
        case=case,
        terminal="verified_success",
        actual=str(case.arguments["desired_value"]),
        verified=True,
        false_success=False,
    )


def test_unqualified_translation_safe_stop_is_not_an_invalid_action() -> None:
    case = next(item for item in build_cases() if item.case_id == "compose-09")
    assert not _is_invalid_action(
        case=case,
        mode="safe_stop",
        terminal="rejected",
        actual="viewer",
        effect_count=0,
    )


def test_safe_stop_with_a_provider_write_is_an_invalid_action() -> None:
    case = next(item for item in build_cases() if item.case_id == "compose-09")
    assert _is_invalid_action(
        case=case,
        mode="safe_stop",
        terminal="rejected",
        actual="viewer",
        effect_count=1,
    )


def test_revision_conflict_effect_dispatch_is_an_invalid_action() -> None:
    case = next(item for item in build_cases() if item.case_id == "script-05")
    assert _is_invalid_action(
        case=case,
        mode="l1_native",
        terminal="rejected",
        actual="small",
        effect_count=1,
    )


def test_compensation_failure_cannot_retry_the_original_effect() -> None:
    case = next(
        item for item in build_cases()
        if item.scenario_pattern == "compensation_failure"
    )
    assert _maximum_effect_calls(case) == 1
    assert _is_invalid_action(
        case=case,
        mode="l1_native",
        terminal="manual_intervention_required",
        actual="__verification_mismatch__",
        effect_count=2,
    )


def test_after_send_unknown_must_not_blindly_retry() -> None:
    case = next(item for item in build_cases() if item.case_id == "ref-07")
    assert _is_invalid_action(
        case=case,
        mode="l1_native",
        terminal="verified_success",
        actual=str(case.arguments["desired_value"]),
        effect_count=2,
    )
    assert not _is_invalid_action(
        case=case,
        mode="l0_runtime",
        terminal="verified_success",
        actual=str(case.arguments["desired_value"]),
        effect_count=1,
    )


def test_unqualified_after_send_unknown_safe_stop_is_not_invalid_action() -> None:
    case = next(item for item in build_cases() if item.case_id == "ref-07")
    assert not _is_invalid_action(
        case=case,
        mode="safe_stop",
        terminal="rejected",
        actual="vlan-10",
        effect_count=0,
    )


def test_empty_failed_session_is_availability_not_invalid_action() -> None:
    case = next(item for item in build_cases() if item.case_id == "approval-08")
    assert not _is_invalid_action(
        case=case,
        mode="l0_runtime",
        terminal="rejected",
        actual="small",
        effect_count=0,
        session_completed=False,
    )


def test_stratified_pattern_set_covers_all_patterns_and_skill_families() -> None:
    indexed = {item.case_id: item for item in build_cases()}
    selected = [indexed[case_id] for case_id in STRATIFIED_PATTERN_CASE_IDS]
    assert len(selected) == 10
    assert len({item.scenario_pattern for item in selected}) == 10
    assert len({item.feature_family for item in selected}) == 6
    assert {"steps-02", "compose-09"}.issubset(STRATIFIED_PATTERN_CASE_IDS)


def test_external_stratified_set_rotates_skill_families() -> None:
    expanded = [
        replace(item, case_id=f"{item.case_id}-{domain}", domain=domain)
        for item in build_cases() for domain in DOMAINS
    ]
    selected = _external_stratified_cases(expanded)
    assert len(selected) == 10
    assert len({item.scenario_pattern for item in selected}) == 10
    assert len({item.feature_family for item in selected}) == 6
    assert len({item.domain for item in selected}) == 6


def test_dsh_plugin_concludes_only_trusted_runtime_terminals() -> None:
    source = (
        Path(__file__).parents[1] / "dsh-plugin-effect-harness/src/index.js"
    ).read_text(encoding="utf-8")
    assert "exec.concludeTurn()" in source
    assert "result.execution === 'l0_runtime'" in source
    assert "result.execution === 'safe_stop'" in source
    for terminal in (
        "verified_success", "rollback_verified",
        "manual_intervention_required", "rejected",
    ):
        assert f"'{terminal}'" in source
