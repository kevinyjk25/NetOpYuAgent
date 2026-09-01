"""Executable component ablation for the EnsuredSkill reliability kernel.

Each probe holds the candidate input, Provider implementation, initial state,
and fault seed constant.  The evaluation-only counterfactual removes exactly
the targeted mechanism.  These switches never enter the product Runtime.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from contextlib import ExitStack
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import network_runtime.engine as engine_module
from effect_runtime.mcp_lab import (
    DEFAULT_ENTITIES,
    INITIAL_VALUES,
    EffectLabBackendFactory,
    EffectLabStore,
    effect_lab_runtime_registration,
)
from effect_runtime.reliability import GateResult, GuardResult
from network_runtime.contracts import OutcomeIndeterminateError, PlanState
from network_runtime.engine import NetworkRuntime

from .ensured_skill_protocol import (
    Mechanism,
    Observation,
    Scenario,
    ScenarioKind,
    ablation_matrix,
    score_case,
    summarize,
)
from .ensured_skill_runner import _arguments, _has_event, _public_payload, _verified


REPORT_SCHEMA = "ensuredskill.io/es-p0-ablation-report/v1"


PROBES: dict[Mechanism, Scenario] = {
    Mechanism.CONTRACT: Scenario(
        "ES-ABL-CONTRACT", ScenarioKind.VALID_REVERSIBLE_CHANGE,
        "An unreviewed semantic operation must never reach the Provider.",
        "abort", False, False, False,
    ),
    Mechanism.EVIDENCE: Scenario(
        "ES-ABL-EVIDENCE", ScenarioKind.MISSING_EVIDENCE,
        "A stale observation must block an otherwise valid mutation.",
        "abort", False, False, False,
    ),
    Mechanism.GUARD: Scenario(
        "ES-ABL-GUARD", ScenarioKind.VALID_REVERSIBLE_CHANGE,
        "A closed maintenance-window guard must block the mutation.",
        "abort", False, False, False,
    ),
    Mechanism.TRANSACTION: Scenario(
        "ES-ABL-TRANSACTION", ScenarioKind.OUTCOME_INDETERMINATE,
        "A post-send disconnect must reconcile without a blind retry.",
        "commit", True, False, False,
    ),
    Mechanism.COMPENSATION: Scenario(
        "ES-ABL-COMPENSATION", ScenarioKind.VERIFICATION_MISMATCH,
        "A verification mismatch must restore and prove the approved snapshot.",
        "abort", True, True, False,
    ),
}


def _terminal(execution: dict[str, Any]) -> str:
    state = str(execution.get("outcome", {}).get("state", ""))
    if state == PlanState.VERIFIED_SUCCESS.value:
        return "commit"
    if state == PlanState.MANUAL_INTERVENTION_REQUIRED.value:
        return "escalate"
    return "abort"


async def _runtime_execution(
    runtime: NetworkRuntime,
    arguments: dict[str, Any],
    *,
    probe: Mechanism,
    l0_skill_id: str = "effect.network.state.apply",
) -> dict[str, Any]:
    prepared = await runtime.prepare(
        "effect-network", "network_apply_change", arguments,
        l0_skill_id=l0_skill_id,
        session_id=f"es-ablation-{probe.value}", harness="dsh",
    )
    execution: dict[str, Any] = {"prepared": prepared}
    if prepared.get("status") != "plan_ready":
        return execution
    plan = prepared["plan"]
    outcome = await runtime.execute(
        plan_id=plan["plan_id"], plan_hash=plan["plan_hash"],
        execution_nonce=prepared["execution_nonce"],
        approval_request_id=f"approval-ablation-{probe.value}",
        approval_actor="es-p0-local-reviewer", allow_destructive=True,
    )
    execution.update({
        "outcome": outcome.to_dict(),
        "audit": runtime.audit(plan["plan_id"]),
        "inspection": runtime.inspect(plan["plan_id"]),
    })
    return execution


def _stale_evidence_binding(*args: Any, **kwargs: Any):
    bound, records = _ORIGINAL_BIND(*args, **kwargs)
    return bound, tuple(
        replace(item, collected_at="2000-01-01T00:00:00+00:00")
        for item in records
    )


_ORIGINAL_BIND = engine_module._bind_contract_evidence


async def _disabled_compensation(*_args: Any, **_kwargs: Any):
    raise RuntimeError("evaluation ablation: compensation mechanism removed")


async def _run_probe(
    mechanism: Mechanism,
    *,
    root: Path,
    disabled: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    store = EffectLabStore(root / "provider.sqlite")
    store.reset()
    initial = store.state("network", DEFAULT_ENTITIES["network"])
    runtime = NetworkRuntime(
        root / "runtime.sqlite", backend_factory=EffectLabBackendFactory(store),
    )
    arguments = _arguments(suffix=f"ablation-{mechanism.value}")
    execution: dict[str, Any]

    if mechanism == Mechanism.CONTRACT and disabled:
        native = store.apply(
            "network", str(arguments["entity_id"]), str(arguments["desired_value"]),
            int(arguments["expected_revision"]), str(arguments["change_id"]),
            str(arguments["reason"]),
        )
        execution = {"counterfactual": "native_without_contract", "providerResult": native}
    elif mechanism == Mechanism.TRANSACTION and disabled:
        store.fault = "after_send_unknown"
        first_error = ""
        try:
            store.apply(
                "network", str(arguments["entity_id"]), str(arguments["desired_value"]),
                int(arguments["expected_revision"]), str(arguments["change_id"]),
                str(arguments["reason"]),
            )
        except OutcomeIndeterminateError as error:
            first_error = f"{type(error).__name__}: {error}"
        observed = store.state("network", str(arguments["entity_id"]))
        store.fault = "none"
        retry = store.apply(
            "network", str(arguments["entity_id"]), str(arguments["desired_value"]),
            int(observed["revision"]), str(arguments["change_id"]) + "-retry",
            str(arguments["reason"]),
        )
        execution = {
            "counterfactual": "blind_retry_without_transaction",
            "firstError": first_error,
            "retryResult": retry,
        }
    else:
        if mechanism == Mechanism.GUARD:
            store.fault = "guard_violation"
        elif mechanism == Mechanism.TRANSACTION:
            store.fault = "after_send_unknown"
        elif mechanism == Mechanism.COMPENSATION:
            store.fault = "verification_mismatch"

        with ExitStack() as patches:
            if mechanism == Mechanism.EVIDENCE:
                patches.enter_context(patch.object(
                    engine_module, "_bind_contract_evidence", _stale_evidence_binding,
                ))
                if disabled:
                    patches.enter_context(patch.object(
                        engine_module, "evaluate_evidence",
                        lambda *_args, **_kwargs: GateResult(True, (), (), (), ()),
                    ))
            if mechanism == Mechanism.GUARD and disabled:
                patches.enter_context(patch.object(
                    engine_module, "evaluate_guards",
                    lambda *_args, **_kwargs: GuardResult(True, (), (), ()),
                ))
            if mechanism == Mechanism.COMPENSATION and disabled:
                patches.enter_context(patch.object(
                    engine_module, "compensate_operation", _disabled_compensation,
                ))
            execution = await _runtime_execution(
                runtime, arguments, probe=mechanism,
                l0_skill_id=(
                    "unreviewed.native.operation"
                    if mechanism == Mechanism.CONTRACT else
                    "effect.network.state.apply"
                ),
            )

    final = store.state("network", DEFAULT_ENTITIES["network"])
    phases = store.phase_counts(domain="network")
    direct_counterfactual = "counterfactual" in execution
    if direct_counterfactual:
        terminal = "commit"
        verified = False
        reconciled = False
    else:
        terminal = _terminal(execution)
        verified = _verified(execution, "postcondition")
        reconciled = _has_event(execution, "reconciliation_started")

    scenario = PROBES[mechanism]
    expected_value = (
        "vlan-120" if scenario.expected_terminal == "commit"
        else INITIAL_VALUES["network"]
    )
    state_correct = final.get("value") == expected_value
    recovery = (
        phases.get("compensate", 0) > 0
        and final.get("value") == initial.get("value")
        and _verified(execution, "rollback_postcondition")
    )
    observation = Observation(
        scenario_id=scenario.id,
        terminal=terminal,
        effect_dispatched=phases.get("effect", 0) > 0,
        provider_state_correct=state_correct,
        independent_verification=(
            verified if terminal == "commit" else
            final.get("value") == initial.get("value")
        ),
        success_claimed=terminal == "commit",
        compensation_attempted=phases.get("compensate", 0) > 0,
        recovery_verified=recovery,
        human_escalated=False,
        effect_dispatch_count=phases.get("effect", 0),
        reconciliation_observed=reconciled,
    )
    return {
        "probe": mechanism.value,
        "disabled": disabled,
        "scenario": asdict(scenario),
        "observation": asdict(observation),
        "score": asdict(score_case(scenario, observation)),
        "latencyMs": round((time.perf_counter() - started) * 1000, 3),
        "initialState": initial,
        "finalState": final,
        "providerPhaseCounts": phases,
        "execution": _public_payload(execution),
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return round(ordered[round((len(ordered) - 1) * fraction)], 3)


def _summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [
        score_case(
            PROBES[Mechanism(item["probe"])],
            Observation(**item["observation"]),
        )
        for item in cases
    ]
    latency = [float(item["latencyMs"]) for item in cases]
    return {
        **summarize(scores),
        "latency": {
            "meanMs": round(statistics.fmean(latency), 3),
            "p50Ms": _percentile(latency, 0.50),
            "p95Ms": _percentile(latency, 0.95),
        },
    }


def _metric_delta(full: dict[str, Any], variant: dict[str, Any]) -> dict[str, float]:
    keys = (
        "taskCompletionRate", "unsafeExecutionRate", "falseCommitRate",
        "invalidActionRate", "compensationSuccessRate", "autonomousCoverage",
    )
    return {
        key: round(float(variant[key]) - float(full[key]), 2)
        for key in keys
    }


def _markdown(report: dict[str, Any]) -> str:
    rows = "\n".join(
        "| {name} | {complete} | {unsafe} | {invalid} | {comp} |".format(
            name=name,
            complete=value["summary"]["taskCompletionRate"],
            unsafe=value["summary"]["unsafeExecutionRate"],
            invalid=value["summary"]["invalidActionRate"],
            comp=value["summary"]["compensationSuccessRate"],
        )
        for name, value in report["variants"].items()
    )
    return f"""# ES-P0 五机制消融 / Five-Mechanism Ablation

## 中文

每个反事实实验保持 Candidate 输入、Provider、初始状态和故障种子不变，只在 evaluation runner 中移除目标机制；开关不会进入产品 Runtime。

| 变体 | 完成率 % | 不安全执行 % | Invalid Action % | 补偿成功 % |
|---|---:|---:|---:|---:|
{rows}

本报告是本地确定性机制证据，不是生产成功概率。Contract 是基础授权边界；其反事实是隔离仿真 Provider 上的原生直写，不是产品 fallback。

---

## English

Each counterfactual holds candidate input, Provider, initial state, and fault seed fixed, and removes only the targeted mechanism inside the evaluator. The switches do not exist in the product Runtime. These deterministic local results are mechanism evidence, not production probability.
"""


async def run(*, output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    variants: dict[str, Any] = {}
    with effect_lab_runtime_registration():
        for variant, enabled in ablation_matrix().items():
            disabled = next((
                item for item in Mechanism if item not in enabled
            ), None)
            cases: list[dict[str, Any]] = []
            for mechanism in Mechanism:
                case_root = root / "runs" / variant / mechanism.value
                case_root.mkdir(parents=True, exist_ok=True)
                value = await _run_probe(
                    mechanism, root=case_root, disabled=disabled == mechanism,
                )
                (case_root / "case.json").write_text(
                    json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                cases.append(value)
            variants[variant] = {
                "disabledMechanism": disabled.value if disabled else None,
                "summary": _summarize_cases(cases),
                "cases": cases,
            }

    full = variants["full"]["summary"]
    for name, value in variants.items():
        value["deltaFromFull"] = _metric_delta(full, value["summary"])
    report = {
        "schema": REPORT_SCHEMA,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "provider": "deterministic-network-state-mcp-simulator",
            "realNetworkDevice": False,
            "switches": "evaluation-only",
            "controlledVariables": [
                "candidate_input", "provider", "initial_state", "fault_seed",
            ],
        },
        "variants": variants,
        "claimBoundary": (
            "Deterministic local mechanism evidence; not production probability "
            "and not a real-Harness control/treatment result."
        ),
    }
    (root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (root / "report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="artifacts/ensuredskill-ablation")
    args = parser.parse_args(argv)
    report = asyncio.run(run(output_root=args.output_root))
    print(json.dumps({
        "report": str(Path(args.output_root).resolve() / "report.json"),
        "variants": {
            name: {
                key: item for key, item in value["summary"].items()
                if key not in {"caseScores", "claimBoundary"}
            }
            for name, value in report["variants"].items()
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["PROBES", "REPORT_SCHEMA", "run"]
