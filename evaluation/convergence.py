"""Privacy-minimized proof of where LLM behavior is and is not controlled.

This module consumes offline evaluation reports only.  It has no dependency on
the Runtime execution path and cannot approve, invoke, publish, or activate an
effect.  Per-case projections intentionally omit prompts and argument values.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from network_runtime.contracts import sha256_json


CONVERGENCE_SCHEMA = "netopyu.io/convergence-cockpit/v1"
RUNTIME_REPORT_SCHEMA = "netopyu.runtime-ab@1.0.0"
L1_REPORT_SCHEMA = "netopyu.io/l1-dsh-schema-compiler-report/v1"
_MAX_REPORT_BYTES = 64_000_000
_PROTOCOL_REQUIRED = (
    "candidate_contract_digest_match",
    "capture_schema_valid",
    "compiler_valid",
    "exposed_tools_exact",
    "final_response_exact",
    "preloaded_skill_digest_match",
    "session_completed",
    "single_capture_exact",
)


class ConvergenceReportError(RuntimeError):
    """Raised for unsupported, unsafe, or inconsistent evaluation evidence."""


def convergence_analyzer_digest() -> str:
    return "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _read_json(path: str | Path, *, expected_schema: str) -> dict[str, Any]:
    supplied = Path(path).expanduser()
    if supplied.is_symlink() or not supplied.is_file() or supplied.stat().st_size > _MAX_REPORT_BYTES:
        raise ConvergenceReportError("evaluation report is missing, unsafe, or oversized")
    try:
        value = json.loads(supplied.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConvergenceReportError("evaluation report JSON is invalid") from error
    if not isinstance(value, dict):
        raise ConvergenceReportError("evaluation report root must be an object")
    schema = value.get("apiVersion", value.get("schema"))
    if schema != expected_schema:
        raise ConvergenceReportError(f"unsupported evaluation schema: {schema!r}")
    return value


def load_runtime_report(path: str | Path) -> dict[str, Any]:
    return _read_json(path, expected_schema=RUNTIME_REPORT_SCHEMA)


def load_l1_report(path: str | Path) -> dict[str, Any]:
    return _read_json(path, expected_schema=L1_REPORT_SCHEMA)


def _rate_metric(metrics: Iterable[dict[str, Any]], metric_id: str) -> dict[str, Any]:
    for metric in metrics:
        if metric.get("metric_id") == metric_id:
            return metric
    raise ConvergenceReportError(f"runtime report is missing metric {metric_id}")


def _protocol_pass(trace: dict[str, Any]) -> bool:
    if trace.get("process_timed_out"):
        return False
    if trace.get("process_return_code") != 0:
        return False
    if trace.get("capture_call_count") != 1 or trace.get("capture_accepted") is not True:
        return False
    if trace.get("forbidden_tools"):
        return False
    if trace.get("duplicate_capture") or trace.get("premature_visible_text"):
        return False
    return all(trace.get(name) is True for name in _PROTOCOL_REQUIRED)


def _failure_layer(score: dict[str, Any], protocol: dict[str, Any]) -> str:
    if bool(score.get("end_to_end_pass")):
        return "passed"
    if not bool(score.get("candidate_hit")):
        return "retrieval"
    if not _protocol_pass(protocol) or not bool(score.get("valid_output")):
        return "protocol"
    if not bool(score.get("action_match")) or not bool(score.get("target_match")):
        return "semantic_selection"
    category = str(score.get("category", ""))
    if category == "clarification" and not bool(score.get("missing_fields_match")):
        return "clarification"
    if category == "multi_step" and not bool(score.get("workflow_match")):
        return "workflow"
    if not bool(score.get("arguments_exact")):
        return "parameter_grounding"
    if not bool(score.get("missing_fields_match")):
        return "clarification"
    if not bool(score.get("workflow_match")):
        return "workflow"
    return "unattributed"


def _guard_contained(guard: dict[str, Any]) -> bool:
    return any((
        bool(guard.get("policy_intervened")),
        bool(guard.get("synthetic_safe_call")),
        int(guard.get("dropped_argument_fields") or 0) > 0,
        int(guard.get("schema_dropped_argument_fields") or 0) > 0,
        int(guard.get("guard_mismatch_attempts") or 0) > 0,
    ))


def _case_projection(model: str, value: dict[str, Any]) -> dict[str, Any]:
    score = value.get("score") if isinstance(value.get("score"), dict) else {}
    protocol = value.get("protocol_trace") if isinstance(value.get("protocol_trace"), dict) else {}
    guard = value.get("guard_trace") if isinstance(value.get("guard_trace"), dict) else {}
    layer = _failure_layer(score, protocol)
    return {
        "model": model,
        "scenarioId": str(score.get("scenario_id", "unknown")),
        "profile": str(score.get("profile", "unknown")),
        "language": str(score.get("language", "unknown")),
        "category": str(score.get("category", "unknown")),
        "failureLayer": layer,
        "passed": layer == "passed",
        "expectedAction": score.get("expected_action"),
        "predictedAction": score.get("predicted_action"),
        "expectedTargets": list(score.get("expected_targets") or []),
        "predictedTarget": score.get("predicted_target"),
        "gates": {
            "candidateHit": bool(score.get("candidate_hit")),
            "protocolValid": _protocol_pass(protocol),
            "actionMatch": bool(score.get("action_match")),
            "targetMatch": bool(score.get("target_match")),
            "argumentsExact": bool(score.get("arguments_exact")),
            "missingFieldsMatch": bool(score.get("missing_fields_match")),
            "workflowMatch": bool(score.get("workflow_match")),
        },
        "containment": {
            "guardIntervened": bool(guard.get("policy_intervened")),
            "guardContained": _guard_contained(guard),
            "modelAttempts": int(guard.get("model_attempts") or 0),
            "repairAttempts": int(guard.get("repair_attempts") or 0),
            "droppedArgumentFieldCount": int(guard.get("dropped_argument_fields") or 0),
            "schemaDroppedArgumentFieldCount": int(
                guard.get("schema_dropped_argument_fields") or 0
            ),
        },
        "elapsedMs": float(score.get("elapsed_ms") or 0.0),
        "errorType": score.get("error_type") or protocol.get("error_type"),
    }


def _model_projection(report: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model = str(report.get("model") or "unknown")
    raw_cases = report.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != int(report.get("evaluated_cases") or -1):
        raise ConvergenceReportError(f"L1 report case count is inconsistent for {model}")
    cases = [_case_projection(model, case) for case in raw_cases]
    layers = Counter(item["failureLayer"] for item in cases)
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    protocol = (
        report.get("protocol_metrics")
        if isinstance(report.get("protocol_metrics"), dict) else {}
    )
    guard = report.get("guard_metrics") if isinstance(report.get("guard_metrics"), dict) else {}
    latency = metrics.get("latency_ms") if isinstance(metrics.get("latency_ms"), dict) else {}
    tokens = metrics.get("tokens") if isinstance(metrics.get("tokens"), dict) else {}
    summary = {
        "model": model,
        "modelArtifactDigest": report.get("model_artifact_digest"),
        "qualified": bool(report.get("qualified")),
        "qualificationEligible": bool(report.get("qualification_eligible")),
        "gateFailures": [str(value) for value in report.get("gate_failures", [])],
        "cases": len(cases),
        "datasetComplete": bool(report.get("dataset_complete")),
        "metrics": {
            "candidateRecall": float(metrics.get("candidate_recall") or 0.0),
            "selectionAccuracy": float(metrics.get("selection_accuracy") or 0.0),
            "parameterFieldF1": float(metrics.get("parameter_field_f1") or 0.0),
            "clarificationPrecision": float(metrics.get("clarification_precision") or 0.0),
            "clarificationRecall": float(metrics.get("clarification_recall") or 0.0),
            "workflowAccuracy": float(metrics.get("workflow_accuracy") or 0.0),
            "endToEndAccuracy": float(metrics.get("end_to_end_accuracy") or 0.0),
            "strictOutputRate": float(metrics.get("strict_output_rate") or 0.0),
            "finalSafetyEscapeRate": float(guard.get("final_safety_escape_rate") or 0.0),
            "firstAttemptSafetyEscapeRate": float(
                guard.get("first_attempt_safety_escape_rate") or 0.0
            ),
            "protocolCompletionRate": float(protocol.get("session_completion_rate") or 0.0),
            "p50Ms": float(latency.get("p50") or 0.0),
            "p95Ms": float(latency.get("p95") or 0.0),
            "inputTokens": int(tokens.get("input") or 0),
            "outputTokens": int(tokens.get("output") or 0),
        },
        "failureLayers": dict(sorted(layers.items())),
        "guard": {
            "interventionRate": float(guard.get("guard_intervention_rate") or 0.0),
            "repairAttempts": int(guard.get("repair_attempts") or 0),
            "droppedArgumentFields": int(guard.get("dropped_argument_fields") or 0),
            "schemaDroppedArgumentFields": int(
                guard.get("schema_dropped_argument_fields") or 0
            ),
        },
    }
    return summary, cases


def build_convergence_report(
    runtime_report: dict[str, Any], l1_reports: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    if runtime_report.get("schema") != RUNTIME_REPORT_SCHEMA:
        raise ConvergenceReportError("runtime report schema is unsupported")
    selected_reports = list(l1_reports)
    if not selected_reports:
        raise ConvergenceReportError("at least one L1 report is required")
    if any(report.get("apiVersion") != L1_REPORT_SCHEMA for report in selected_reports):
        raise ConvergenceReportError("L1 report schema is unsupported")
    runtime_scenarios = runtime_report.get("scenarios")
    if not isinstance(runtime_scenarios, list) or len(runtime_scenarios) != int(
        runtime_report.get("scenario_count") or -1
    ):
        raise ConvergenceReportError("runtime report scenario count is inconsistent")
    model_names = [str(report.get("model")) for report in selected_reports]
    if len(set(model_names)) != len(model_names):
        raise ConvergenceReportError("L1 model reports must be unique")

    runtime_metrics = runtime_report.get("metrics") or {}
    only_metrics = runtime_metrics.get("dsh_only") or []
    controlled_metrics = runtime_metrics.get("dsh_plus_runtime") or []
    only_control = _rate_metric(only_metrics, "control_effectiveness")
    runtime_control = _rate_metric(controlled_metrics, "control_effectiveness")
    runtime_failures = [
        scenario["scenario_id"] for scenario in runtime_scenarios
        if not bool((scenario.get("dsh_plus_runtime") or {}).get("passed"))
    ]
    model_summaries: list[dict[str, Any]] = []
    case_evidence: list[dict[str, Any]] = []
    for l1_report in selected_reports:
        summary, cases = _model_projection(l1_report)
        model_summaries.append(summary)
        case_evidence.extend(cases)

    fixed_runtime_verified = (
        float(runtime_control.get("rate") or 0.0) == 100.0 and not runtime_failures
    )
    qualified_models = [item["model"] for item in model_summaries if item["qualified"]]
    body: dict[str, Any] = {
        "apiVersion": CONVERGENCE_SCHEMA,
        "analyzerDigest": convergence_analyzer_digest(),
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "scope": "local-fixed-evaluation-sets",
        "answer": {
            "deterministicExecutionControls": (
                "verified_on_core72" if fixed_runtime_verified else "not_verified"
            ),
            "semanticIntentConvergence": (
                "qualified_on_fixed_set" if qualified_models else "not_qualified"
            ),
            "qualifiedModels": qualified_models,
            "productionGeneralization": "not_proven",
            "claim": (
                "The deterministic Runtime closes the effect-authority gap on the fixed Core-72 "
                "oracle. L1 intent selection and extraction remain probabilistic and must be "
                "qualified per model, dataset, protocol, and deployment."
            ),
        },
        "runtimeComparison": {
            "campaignId": runtime_report.get("campaign_id"),
            "scenarioCount": int(runtime_report.get("scenario_count") or 0),
            "dshOnlyControlEffectiveness": only_control,
            "runtimeControlEffectiveness": runtime_control,
            "runtimeFailures": runtime_failures,
            "metrics": {
                "dshOnly": only_metrics,
                "dshPlusRuntime": controlled_metrics,
            },
            "latency": runtime_report.get("latency") or {},
            "trend": runtime_report.get("trend") or {},
            "scenarios": runtime_scenarios,
        },
        "models": model_summaries,
        "caseEvidence": case_evidence,
        "failureLayerDefinitions": {
            "retrieval": "Expected candidate was absent from the model-visible set.",
            "protocol": "Harness/tool protocol did not yield one valid compiler input.",
            "semantic_selection": "Candidate was visible, but action or target was wrong.",
            "parameter_grounding": "Action/target matched, but explicit parameter values did not.",
            "clarification": "Required clarification or missing-field set did not match.",
            "workflow": "Required multi-step workflow did not match.",
            "unattributed": "The case failed after recorded gates without a unique layer.",
            "passed": "All fixed case gates passed.",
        },
        "limits": [
            "Percentages are fixed-set oracle coverage, not production success probabilities.",
            "Core-72 fixes L1 decisions and isolates Runtime control value.",
            "L1 reports do not test real vendor devices, enterprise authority, HA/DR, or SLOs.",
            "A final zero safety-escape rate may be produced by deterministic Guard intervention; it does not prove the model is intrinsically safe.",
            "Per-case evidence omits prompts, labels beyond expected action/target, and all argument values.",
        ],
        "sourceDigests": {
            "runtime": sha256_json(runtime_report),
            "l1": {str(report.get("model")): sha256_json(report) for report in selected_reports},
        },
        "authority": {
            "readOnly": True,
            "activationAvailable": False,
            "approvalAvailable": False,
            "executionAvailable": False,
        },
    }
    body["snapshotDigest"] = sha256_json(body)
    return body


def load_convergence_snapshot(path: str | Path) -> dict[str, Any]:
    value = _read_json(path, expected_schema=CONVERGENCE_SCHEMA)
    if value.get("analyzerDigest") != convergence_analyzer_digest():
        raise ConvergenceReportError("convergence snapshot analyzer digest is stale")
    body = dict(value)
    declared = body.pop("snapshotDigest", None)
    if declared != sha256_json(body):
        raise ConvergenceReportError("convergence snapshot digest is invalid")
    return value
