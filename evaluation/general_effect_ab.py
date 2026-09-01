"""Legacy component transaction-control regression over 60 Skill fixtures.

Arm A is a one-shot Provider contract call, not a DSH/Hermes Agent.  Arm B
passes the same fixed arguments through Runtime.  This module remains useful
for deterministic transaction-control regression, but its output is
explicitly ineligible for Harness/Agent product claims.  Use
``evaluation.harness_skill_runtime_ab`` for the real comparison.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from effect_runtime import inspect_skill_package
from effect_runtime.mcp_lab import (
    INITIAL_VALUES, EffectLabBackendFactory, EffectLabStore,
    effect_lab_runtime_registration,
)
from evaluation.general_effect_dataset import (
    FEATURE_FAMILIES, GeneralEffectCase, build_cases, materialize_dataset,
)
from network_runtime import NetworkRuntime, PlanState


REPORT_SCHEMA = "effect-runtime.io/general-effect-controlled-ab/v1"


@dataclass(frozen=True)
class ArmObservation:
    passed: bool
    terminal: str
    actual_value: str
    effect_calls: int
    total_provider_calls: int
    latency_ms: float
    false_success: bool
    independent_verification: bool
    compensation_attempted: bool
    audit_valid: bool
    detail: str


@dataclass(frozen=True)
class CaseObservation:
    case_id: str
    skill_id: str
    feature_family: str
    domain: str
    scenario_pattern: str
    translation_gate: str
    l0_contract_digest: str
    dsh_l1_direct: ArmObservation
    dsh_effect_runtime: ArmObservation


def _schema_error(metadata: dict[str, Any], arguments: dict[str, Any]) -> str | None:
    properties = metadata.get("parameters") or {}
    unknown = sorted(set(arguments) - set(properties))
    if unknown:
        return "unknown parameters: " + ", ".join(unknown)
    missing = sorted(
        name for name in metadata.get("required") or ()
        if name not in arguments or arguments[name] in (None, "")
    )
    if missing:
        return "missing parameters: " + ", ".join(missing)
    for name, value in arguments.items():
        expected = (properties.get(name) or {}).get("type")
        valid = {
            "string": isinstance(value, str),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "boolean": isinstance(value, bool),
            "array": isinstance(value, list),
            "object": isinstance(value, dict),
        }.get(expected, True)
        if not valid:
            return f"parameter {name} has invalid type"
    return None


def _expected(case: GeneralEffectCase) -> tuple[set[str], str]:
    pattern = case.scenario_pattern
    if pattern == "missing_required":
        return {"clarification_required"}, INITIAL_VALUES[case.domain]
    if pattern == "unknown_parameter":
        return {"rejected"}, INITIAL_VALUES[case.domain]
    if pattern == "approval_denied":
        return {"rejected"}, INITIAL_VALUES[case.domain]
    if pattern in {"revision_conflict", "verification_mismatch"}:
        return {"rollback_verified", "rejected"}, INITIAL_VALUES[case.domain]
    if pattern == "provider_error_before_send":
        # Once the provider callable started, the generic transport exception
        # cannot prove whether a remote side effect occurred.  The safe Runtime
        # terminal is explicit manual intervention after read-only reconcile.
        return {"manual_intervention_required"}, INITIAL_VALUES[case.domain]
    if pattern == "compensation_failure":
        return {"manual_intervention_required"}, "__verification_mismatch__"
    return {"verified_success"}, str(case.arguments["desired_value"])


def _score(case: GeneralEffectCase, observation: ArmObservation) -> bool:
    terminals, value = _expected(case)
    return observation.terminal in terminals and observation.actual_value == value and not observation.false_success


async def _actual_value(store: EffectLabStore, case: GeneralEffectCase) -> str:
    state = store.state(case.domain, str(case.arguments.get("entity_id") or ""))
    return str(state.get("value") or "<missing>")


def _reset_generated_sqlite(path: Path) -> None:
    """Reset only one generated benchmark journal, including SQLite sidecars."""

    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        candidate.unlink(missing_ok=True)


async def _run_direct(case: GeneralEffectCase, store: EffectLabStore) -> ArmObservation:
    factory = EffectLabBackendFactory(store)
    backend = await factory(f"effect-{case.domain}")
    started = time.perf_counter()
    terminal = "rejected"
    detail = ""
    reported_success = False
    try:
        metadata = backend.metadata[case.tool_name]
        error = _schema_error(metadata, case.arguments)
        if error:
            terminal = "clarification_required" if error.startswith("missing") else "rejected"
            detail = error
        elif not case.approved:
            terminal = "rejected"
            detail = "generic approval rejected"
        else:
            try:
                raw = await backend.callables[case.tool_name](dict(case.arguments))
                decoded = json.loads(str(raw))
                if isinstance(decoded, dict) and decoded.get("ok") is True:
                    reported_success = True
                    terminal = "verified_success"
                    detail = "direct path trusted the effect response"
                else:
                    terminal = "rejected"
                    detail = str(decoded.get("code") if isinstance(decoded, dict) else "invalid result")
            except Exception as error:  # direct path has no read-only reconciliation
                terminal = "outcome_unknown"
                detail = f"{type(error).__name__}: {error}"
    finally:
        await backend.close()
    actual = await _actual_value(store, case)
    desired = str(case.arguments.get("desired_value") or "")
    false_success = reported_success and actual != desired
    observation = ArmObservation(
        passed=False, terminal=terminal, actual_value=actual,
        effect_calls=store.effect_calls(domain=case.domain),
        total_provider_calls=store.calls(domain=case.domain),
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
        false_success=false_success,
        independent_verification=False, compensation_attempted=False,
        audit_valid=False, detail=detail,
    )
    return ArmObservation(**{**asdict(observation), "passed": _score(case, observation)})


async def _run_runtime(
    case: GeneralEffectCase, store: EffectLabStore, journal: Path,
) -> ArmObservation:
    runtime = NetworkRuntime(journal, backend_factory=EffectLabBackendFactory(store))
    started = time.perf_counter()
    prepared = await runtime.prepare(
        f"effect-{case.domain}", case.tool_name, dict(case.arguments),
        l0_skill_id=case.l0_skill_id,
        session_id=f"p26b-{case.case_id}", harness="dsh",
    )
    terminal = str(prepared.get("status") or "rejected")
    detail = "; ".join(str(item) for item in prepared.get("errors") or ())
    independent = False
    compensation = False
    audit_valid = False
    if prepared.get("status") == "plan_ready":
        plan = prepared["plan"]
        if not case.approved:
            rejected = runtime.reject(
                plan_id=plan["plan_id"], plan_hash=plan["plan_hash"],
                reason="controlled approval denial",
            )
            terminal = str(rejected["state"])
            detail = "immutable plan rejected before execution"
        else:
            outcome = await runtime.execute(
                plan_id=plan["plan_id"], plan_hash=plan["plan_hash"],
                execution_nonce=prepared["execution_nonce"], allow_destructive=True,
                approval_request_id=f"approval-{case.case_id}",
                approval_actor="controlled-operator",
            )
            terminal = outcome.state.value
            detail = outcome.error or "verified terminal outcome"
            independent = any(
                item.evidence_type in {"postcondition", "rollback_postcondition"}
                for item in outcome.evidence
            )
            events = runtime.inspect(plan["plan_id"])["events"]
            compensation = any(
                item.get("event_type") in {"l0_step_started", "l0_step_failed", "l0_step_completed"}
                and (item.get("payload") or {}).get("step_id") == "compensate"
                for item in events
            )
        audit_valid = bool(runtime.audit(plan["plan_id"])["ok"])
    actual = await _actual_value(store, case)
    false_success = terminal == PlanState.VERIFIED_SUCCESS.value and (
        actual != str(case.arguments.get("desired_value") or "")
    )
    observation = ArmObservation(
        passed=False, terminal=terminal, actual_value=actual,
        effect_calls=store.effect_calls(domain=case.domain),
        total_provider_calls=store.calls(domain=case.domain),
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
        false_success=false_success, independent_verification=independent,
        compensation_attempted=compensation, audit_valid=audit_valid,
        detail=detail,
    )
    return ArmObservation(**{**asdict(observation), "passed": _score(case, observation)})


def _package_bindings(case: GeneralEffectCase) -> tuple[str, ...]:
    if case.feature_family != "scripts":
        return ()
    return (
        f"scripts/apply.py=effect.{case.domain}.state.apply",
        f"scripts/rollback.py=effect.{case.domain}.state.restore",
    )


def _metric(observations: list[CaseObservation], arm: str) -> dict[str, Any]:
    values = [getattr(item, arm) for item in observations]
    passed = sum(item.passed for item in values)
    latencies = sorted(item.latency_ms for item in values)
    def percentile(fraction: float) -> float:
        if not latencies:
            return 0.0
        position = (len(latencies) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(latencies) - 1)
        return latencies[lower] + (latencies[upper] - latencies[lower]) * (position - lower)
    audit_expected = sum(
        item.scenario_pattern not in {
            "missing_required", "unknown_parameter", "revision_conflict",
        }
        for item in observations
    )
    return {
        "passed": passed, "total": len(values),
        "taskCorrectnessPercent": round(100 * passed / len(values), 2),
        "falseSuccesses": sum(item.false_success for item in values),
        "independentlyVerified": sum(item.independent_verification for item in values),
        "compensationAttempted": sum(item.compensation_attempted for item in values),
        "terminalAudits": {
            "valid": sum(item.audit_valid for item in values),
            "expected": audit_expected,
        },
        "effectCalls": sum(item.effect_calls for item in values),
        "providerCalls": sum(item.total_provider_calls for item in values),
        "latency": {
            "p50Ms": round(percentile(0.50), 3),
            "p95Ms": round(percentile(0.95), 3),
            "meanMs": round(statistics.fmean(latencies), 3) if latencies else 0.0,
        },
    }


def _control_metrics(
    observations: list[CaseObservation], arm: str,
) -> dict[str, dict[str, int | float]]:
    """Expose named safety controls with explicit, non-inflated denominators."""

    groups = {
        "missingInputClarification": {"missing_required"},
        "unknownParameterRejection": {"unknown_parameter"},
        "approvalWritePrevention": {"approval_denied"},
        "unknownOutcomeResolution": {
            "after_send_unknown", "provider_error_before_send",
        },
        "verificationMismatchContainment": {"verification_mismatch"},
        "compensationFailureEscalation": {"compensation_failure"},
    }
    result: dict[str, dict[str, int | float]] = {}
    for name, patterns in groups.items():
        selected = [
            item for item in observations if item.scenario_pattern in patterns
        ]
        passed = sum(getattr(item, arm).passed for item in selected)
        if name == "approvalWritePrevention":
            passed = sum(
                getattr(item, arm).passed
                and getattr(item, arm).effect_calls == 0
                for item in selected
            )
        result[name] = {
            "passed": passed,
            "expected": len(selected),
            "percent": round(100 * passed / len(selected), 2) if selected else 0.0,
        }
    return result


async def _run_controlled_ab_registered(
    *, output_root: str | Path, l0_digests: dict[str, str],
    limit: int | None = None,
) -> dict[str, Any]:
    root = Path(output_root).expanduser()
    dataset_root = root / "dataset"
    manifest = materialize_dataset(dataset_root)
    cases = list(build_cases())
    if limit is not None:
        if not 1 <= limit <= len(cases):
            raise ValueError(f"limit must be between 1 and {len(cases)}")
        cases = cases[:limit]
    observations: list[CaseObservation] = []
    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    for case in cases:
        package_report = inspect_skill_package(
            dataset_root / "skills" / case.skill_id,
            bound_scripts=_package_bindings(case),
        )
        direct_store = EffectLabStore(runs / f"{case.case_id}-direct.sqlite")
        direct_store.reset()
        direct_store.fault = case.fault
        runtime_store = EffectLabStore(runs / f"{case.case_id}-runtime.sqlite")
        runtime_store.reset()
        runtime_store.fault = case.fault
        direct = await _run_direct(case, direct_store)
        journal = runs / f"{case.case_id}-journal.sqlite"
        _reset_generated_sqlite(journal)
        guarded = await _run_runtime(case, runtime_store, journal)
        observations.append(CaseObservation(
            case_id=case.case_id, skill_id=case.skill_id,
            feature_family=case.feature_family, domain=case.domain,
            scenario_pattern=case.scenario_pattern,
            translation_gate=str(package_report["gate"]),
            l0_contract_digest=l0_digests[case.l0_skill_id],
            dsh_l1_direct=direct, dsh_effect_runtime=guarded,
        ))
    report = {
        "schema": REPORT_SCHEMA,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "schema": manifest["schema"], "digest": manifest["datasetDigest"],
            "declaredToolCount": manifest["toolCount"],
            "declaredSkillCount": manifest["skillCount"],
            "executedCases": len(observations),
            "featureFamilies": manifest["featureFamilies"],
        },
        "comparison": {
            "dsh_l1_direct": "LEGACY KEY: one-shot schema/HITL Provider contract call; no DSH Agent loop",
            "dsh_effect_runtime": "same decision/input/fault + active digest-bound L0 + Effect Runtime",
        },
        "methodology": {
            "classification": "component_transaction_control_regression",
            "eligibleForHarnessAgentClaim": False,
            "replacementEvaluator": "evaluation.harness_skill_runtime_ab",
            "sameInputToolProviderAndFault": True,
            "modelSelectionExcluded": True,
            "automaticL0HumanEdits": False,
            "approvalWaitExcluded": True,
            "provider": "official-MCP-compatible deterministic local simulator",
            "controlledRunTransport": "in-process contract adapter",
            "transportReason": (
                "Both arms use the same store and provider contract so the measured "
                "increment is Runtime control rather than MCP process startup."
            ),
            "developmentSet": True,
        },
        "metrics": {
            "dsh_l1_direct": _metric(observations, "dsh_l1_direct"),
            "dsh_effect_runtime": _metric(observations, "dsh_effect_runtime"),
        },
        "controlMetrics": {
            "dsh_l1_direct": _control_metrics(observations, "dsh_l1_direct"),
            "dsh_effect_runtime": _control_metrics(observations, "dsh_effect_runtime"),
        },
        "translation": {
            "packageGatesPassed": sum(item.translation_gate == "passed" for item in observations),
            "activeL0DigestBound": sum(item.l0_contract_digest.startswith("sha256:") for item in observations),
            "modelSemanticTranslationMeasured": False,
        },
        "byFeature": {
            family: {
                arm: {
                    "passed": sum(
                        getattr(item, arm).passed
                        for item in observations if item.feature_family == family
                    ),
                    "total": sum(item.feature_family == family for item in observations),
                }
                for arm in ("dsh_l1_direct", "dsh_effect_runtime")
            }
            for family in FEATURE_FAMILIES
        },
        "cases": [asdict(item) for item in observations],
        "claimBoundary": (
            "This is a one-shot component regression, not DSH/Hermes native Skill execution. "
            "It must not be used as Harness Agent product evidence. Results also do not measure "
            "hidden-set generalization, real-system qualification, production availability, "
            "or production success probability."
        ),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "controlled-ab.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


async def run_controlled_ab(
    *, output_root: str | Path, limit: int | None = None,
) -> dict[str, Any]:
    with effect_lab_runtime_registration() as l0_digests:
        return await _run_controlled_ab_registered(
            output_root=output_root, l0_digests=l0_digests, limit=limit,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="artifacts/general-effect-ab")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    report = asyncio.run(run_controlled_ab(output_root=args.output_root, limit=args.limit))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    metric = report["metrics"]["dsh_effect_runtime"]
    return 0 if metric["passed"] == metric["total"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["run_controlled_ab"]
