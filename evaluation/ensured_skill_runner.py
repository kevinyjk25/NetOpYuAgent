"""Executable ES-P0 network transaction scenarios.

The runner uses the real NetworkRuntime, reviewed L0 compiler, approval path,
independent verifier, compensation manager, journal, and Saga coordinator. The
Infrastructure plane is a deterministic network-state MCP contract simulator;
live Containerlab qualification is recorded separately and is never implied by
these results.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from effect_runtime.mcp_lab import (
    DEFAULT_ENTITIES,
    INITIAL_VALUES,
    EffectLabBackendFactory,
    EffectLabStore,
    effect_lab_runtime_registration,
)
from effect_runtime.saga import SagaCoordinator, SagaDefinition, SagaStepSpec
from network_runtime.contracts import PlanState, sha256_json
from network_runtime.engine import NetworkRuntime

from .ensured_skill_protocol import (
    Observation,
    Scenario,
    ScenarioKind,
    load_scenarios,
    score_case,
    summarize,
)


REPORT_SCHEMA = "ensuredskill.io/es-p0-execution-report/v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIOS = PROJECT_ROOT / "data" / "ensured_skill_scenarios.yaml"


def _arguments(
    *,
    desired_value: str = "vlan-120",
    expected_revision: int = 1,
    entity_id: str | None = None,
    suffix: str = "case",
) -> dict[str, Any]:
    return {
        "entity_id": entity_id or DEFAULT_ENTITIES["network"],
        "desired_value": desired_value,
        "expected_revision": expected_revision,
        "change_id": f"chg-es-p0-{suffix}",
        "reason": f"ES-P0 deterministic network scenario {suffix}",
    }


def _semantic_terminal(state: str) -> str:
    if state == PlanState.VERIFIED_SUCCESS.value:
        return "commit"
    if state == PlanState.MANUAL_INTERVENTION_REQUIRED.value:
        return "escalate"
    return "abort"


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return round(ordered[index], 3)


async def _prepare(
    runtime: NetworkRuntime,
    arguments: dict[str, Any],
    *,
    scenario_id: str,
) -> dict[str, Any]:
    return await runtime.prepare(
        "effect-network",
        "network_apply_change",
        arguments,
        l0_skill_id="effect.network.state.apply",
        session_id=f"es-p0-{scenario_id}",
        harness="dsh",
    )


async def _execute_prepared(
    runtime: NetworkRuntime,
    prepared: dict[str, Any],
    *,
    scenario_id: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {"prepared": prepared}
    if prepared.get("status") != "plan_ready":
        return result
    plan = prepared["plan"]
    outcome = await runtime.execute(
        plan_id=plan["plan_id"],
        plan_hash=plan["plan_hash"],
        execution_nonce=prepared["execution_nonce"],
        approval_request_id=f"approval-{scenario_id}",
        approval_actor="es-p0-local-reviewer",
        allow_destructive=True,
    )
    result.update({
        "outcome": outcome.to_dict(),
        "audit": runtime.audit(plan["plan_id"]),
        "inspection": runtime.inspect(plan["plan_id"]),
    })
    return result


async def _execute(
    runtime: NetworkRuntime,
    arguments: dict[str, Any],
    *,
    scenario_id: str,
) -> dict[str, Any]:
    prepared = await _prepare(runtime, arguments, scenario_id=scenario_id)
    return await _execute_prepared(runtime, prepared, scenario_id=scenario_id)


def _phase_counts(store: EffectLabStore) -> dict[str, int]:
    return store.phase_counts(domain="network")


def _public_payload(value: Any) -> Any:
    """Do not persist one-time execution credentials in evaluation artifacts."""
    if isinstance(value, dict):
        return {
            key: ("<withheld>" if key == "execution_nonce" else _public_payload(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_public_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_public_payload(item) for item in value]
    return value


def _has_event(execution: dict[str, Any], event_type: str) -> bool:
    return any(
        item.get("event_type") == event_type
        for item in execution.get("inspection", {}).get("events", [])
    )


def _verified(execution: dict[str, Any], *types: str) -> bool:
    accepted = set(types)
    return any(
        item.get("evidence_type") in accepted and item.get("passed") is True
        for item in execution.get("outcome", {}).get("evidence", [])
    )


async def _single_case(
    scenario: Scenario,
    *,
    root: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    store = EffectLabStore(root / "provider.sqlite")
    store.reset()
    initial = store.state("network", DEFAULT_ENTITIES["network"])
    runtime = NetworkRuntime(
        root / "runtime.sqlite",
        backend_factory=EffectLabBackendFactory(store),
    )
    execution: dict[str, Any]
    human = False
    terminal = "abort"

    if scenario.kind == ScenarioKind.MISSING_EVIDENCE:
        execution = await _execute(
            runtime,
            _arguments(entity_id="edge-sw-missing", suffix=scenario.id),
            scenario_id=scenario.id,
        )
    elif scenario.kind == ScenarioKind.HIGH_RISK:
        prepared = await runtime.prepare(
            "effect-network",
            "network_apply_change",
            _arguments(suffix=scenario.id),
            l0_skill_id="effect.network.state.apply",
            session_id=f"es-p0-{scenario.id}",
            harness="dsh",
        )
        execution = {"prepared": prepared}
        if prepared.get("status") == "plan_ready":
            plan = prepared["plan"]
            rejected = runtime.reject(
                plan_id=plan["plan_id"],
                plan_hash=plan["plan_hash"],
                reason="ES-P0 human reviewer held the high-risk change",
            )
            execution.update({
                "outcome": rejected,
                "audit": runtime.audit(plan["plan_id"]),
                "inspection": runtime.inspect(plan["plan_id"]),
            })
        human = True
        terminal = "escalate"
    elif scenario.kind == ScenarioKind.OUTCOME_INDETERMINATE:
        store.fault = "after_send_unknown"
        execution = await _execute(
            runtime, _arguments(suffix=scenario.id), scenario_id=scenario.id,
        )
        terminal = _semantic_terminal(str(execution.get("outcome", {}).get("state", "")))
    elif scenario.kind == ScenarioKind.VERIFICATION_MISMATCH:
        store.fault = "verification_mismatch"
        execution = await _execute(
            runtime, _arguments(suffix=scenario.id), scenario_id=scenario.id,
        )
        terminal = _semantic_terminal(str(execution.get("outcome", {}).get("state", "")))
    else:
        execution = await _execute(
            runtime, _arguments(suffix=scenario.id), scenario_id=scenario.id,
        )
        terminal = _semantic_terminal(str(execution.get("outcome", {}).get("state", "")))

    final = store.state("network", DEFAULT_ENTITIES["network"])
    phases = _phase_counts(store)
    desired = "vlan-120"
    is_commit = terminal == "commit"
    if scenario.kind in {
        ScenarioKind.VERIFICATION_MISMATCH,
        ScenarioKind.MISSING_EVIDENCE,
        ScenarioKind.HIGH_RISK,
    }:
        state_correct = final.get("value") == initial.get("value")
    else:
        state_correct = final.get("value") == desired
    independent = (
        _verified(execution, "postcondition")
        if is_commit else
        _verified(execution, "rollback_postcondition")
        or scenario.kind in {ScenarioKind.MISSING_EVIDENCE, ScenarioKind.HIGH_RISK}
    )
    observation = Observation(
        scenario_id=scenario.id,
        terminal=terminal,
        effect_dispatched=phases.get("effect", 0) > 0,
        provider_state_correct=state_correct,
        independent_verification=independent,
        success_claimed=is_commit,
        compensation_attempted=phases.get("compensate", 0) > 0,
        recovery_verified=_verified(execution, "rollback_postcondition"),
        human_escalated=human,
        effect_dispatch_count=phases.get("effect", 0),
        reconciliation_observed=_has_event(execution, "reconciliation_started"),
    )
    score = score_case(scenario, observation)
    duration = round((time.perf_counter() - started) * 1000, 3)
    return {
        "scenario": asdict(scenario),
        "observation": asdict(observation),
        "score": asdict(score),
        "latencyMs": duration,
        "initialState": initial,
        "finalState": final,
        "providerPhaseCounts": phases,
        "transactionAssertions": {
            "outcomeIndeterminateObserved": _has_event(
                execution, "write_outcome_indeterminate",
            ),
            "readOnlyReconciliationObserved": _has_event(
                execution, "reconciliation_started",
            ),
            "blindRetryPrevented": (
                scenario.kind != ScenarioKind.OUTCOME_INDETERMINATE
                or phases.get("effect", 0) == 1
            ),
        },
        "execution": execution,
    }


async def _partial_failure_case(
    scenario: Scenario,
    *,
    root: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    store = EffectLabStore(root / "provider.sqlite")
    store.reset()
    initial = store.state("network", DEFAULT_ENTITIES["network"])
    runtime = NetworkRuntime(
        root / "runtime.sqlite",
        backend_factory=EffectLabBackendFactory(store),
    )
    definition = SagaDefinition.create(
        "es-p0-partial-network-change",
        "1.0.0",
        (
            SagaStepSpec(
                "access-vlan", "network", "effect.network.state.apply",
                compensation_capability_id="effect.network.state.apply",
            ),
            SagaStepSpec(
                "dependent-policy", "network", "effect.network.state.apply",
                depends_on=("access-vlan",),
                compensation_capability_id="effect.network.state.apply",
            ),
        ),
    )
    traces: dict[str, Any] = {}
    compensation_order_verified = False
    with SagaCoordinator(root / "saga.sqlite") as saga:
        started_saga = saga.start(definition, correlation_id=scenario.id)
        saga_id = started_saga["saga_id"]

        first_prepared = await _prepare(
            runtime,
            _arguments(desired_value="vlan-120", expected_revision=1, suffix="step-1"),
            scenario_id=f"{scenario.id}-step-1",
        )
        first_plan = first_prepared["plan"]
        saga.bind_plan(
            saga_id, "access-vlan",
            plan_id=first_plan["plan_id"], plan_hash=first_plan["plan_hash"],
        )
        first = await _execute_prepared(
            runtime, first_prepared, scenario_id=f"{scenario.id}-step-1",
        )
        saga.record_outcome(
            saga_id, "access-vlan",
            terminal_state=str(first["outcome"]["state"]),
        )

        store.fault = "verification_mismatch"
        second_prepared = await _prepare(
            runtime,
            _arguments(desired_value="vlan-130", expected_revision=2, suffix="step-2"),
            scenario_id=f"{scenario.id}-step-2",
        )
        second_plan = second_prepared["plan"]
        saga.bind_plan(
            saga_id, "dependent-policy",
            plan_id=second_plan["plan_id"], plan_hash=second_plan["plan_hash"],
        )
        second = await _execute_prepared(
            runtime, second_prepared, scenario_id=f"{scenario.id}-step-2",
        )
        after_failure = saga.record_outcome(
            saga_id, "dependent-policy",
            terminal_state=str(second["outcome"]["state"]),
            error=str(second["outcome"].get("error") or "dependent step failed"),
        )
        next_action = after_failure.get("next_action") or {}
        compensation_order_verified = (
            next_action.get("action") == "compensate"
            and next_action.get("step_id") == "access-vlan"
        )

        store.fault = "none"
        reverse_prepared = await _prepare(
            runtime,
            _arguments(desired_value=str(initial["value"]), expected_revision=2, suffix="reverse-1"),
            scenario_id=f"{scenario.id}-reverse-1",
        )
        reverse_plan = reverse_prepared["plan"]
        saga.bind_compensation_plan(
            saga_id, "access-vlan",
            plan_id=reverse_plan["plan_id"], plan_hash=reverse_plan["plan_hash"],
        )
        reverse = await _execute_prepared(
            runtime, reverse_prepared, scenario_id=f"{scenario.id}-reverse-1",
        )
        final_saga = saga.record_compensation_outcome(
            saga_id, "access-vlan",
            terminal_state=str(reverse["outcome"]["state"]),
        )
        traces = {
            "first": first,
            "failedDependent": second,
            "reverse": reverse,
            "saga": final_saga,
        }

    final = store.state("network", DEFAULT_ENTITIES["network"])
    phases = _phase_counts(store)
    audit_ok = all(
        item.get("audit", {}).get("ok") is True
        for item in (traces["first"], traces["failedDependent"], traces["reverse"])
    ) and traces["saga"].get("audit", {}).get("ok") is True
    recovered = final.get("value") == initial.get("value") and audit_ok
    observation = Observation(
        scenario_id=scenario.id,
        terminal="abort",
        effect_dispatched=phases.get("effect", 0) > 0,
        provider_state_correct=recovered,
        independent_verification=audit_ok,
        success_claimed=False,
        compensation_attempted=True,
        recovery_verified=recovered and compensation_order_verified,
        human_escalated=False,
        effect_dispatch_count=phases.get("effect", 0),
        compensation_order_verified=compensation_order_verified,
    )
    return {
        "scenario": asdict(scenario),
        "observation": asdict(observation),
        "score": asdict(score_case(scenario, observation)),
        "latencyMs": round((time.perf_counter() - started) * 1000, 3),
        "initialState": initial,
        "finalState": final,
        "providerPhaseCounts": phases,
        "transactionAssertions": {
            "reverseDependencyOrderVerified": compensation_order_verified,
            "allRuntimeAuditsValid": audit_ok,
            "semanticBaselineRestored": final.get("value") == initial.get("value"),
        },
        "execution": traces,
    }


def _markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    rows = "\n".join(
        "| {id} | {terminal} | {completed} | {unsafe} | {false_commit} | {latency} |".format(
            id=item["scenario"]["id"],
            terminal=item["observation"]["terminal"],
            completed="✅" if item["score"]["task_completed"] else "❌",
            unsafe="是" if item["score"]["unsafe_execution"] else "否",
            false_commit="是" if item["score"]["false_commit"] else "否",
            latency=item["latencyMs"],
        )
        for item in report["latestCases"]
    )
    return f"""# ES-P0 可执行场景报告 / Executable Scenario Report

## 中文

本报告执行真实 `NetworkRuntime`、受审 L0、审批、Evidence/Guard、Verifier、Compensator、Journal 与 Saga；Provider 是本地确定性网络状态 MCP 合同模拟器，不是厂商设备或生产概率。

| 场景 | 终态 | 完成 | 不安全执行 | 假提交 | 时延 ms |
|---|---|---:|---:|---:|---:|
{rows}

- 运行次数：{report['iterations']} × 6 场景
- Task Completion：{summary['taskCompletionRate']}%
- Unsafe Execution：{summary['unsafeExecutionRate']}%
- False Commit：{summary['falseCommitRate']}%
- Invalid Action：{summary['invalidActionRate']}%
- Compensation Success：{summary['compensationSuccessRate']}%
- Autonomous Coverage：{summary['autonomousCoverage']}%
- Human Escalation：{summary['humanEscalationRate']}%
- p50 / p95：{report['latency']['p50Ms']} / {report['latency']['p95Ms']} ms

固定场景结果只证明这些机制在该本地协议下按 Oracle 工作，不是生产成功概率。

---

## English

This report executes the real NetworkRuntime, reviewed L0, approval, Evidence/Guard, verifier, compensator, journal, and Saga. The provider is a deterministic local network-state MCP contract simulator, not vendor-device or production-probability evidence.

Task completion is {summary['taskCompletionRate']}%, unsafe execution {summary['unsafeExecutionRate']}%, false commit {summary['falseCommitRate']}%, compensation success {summary['compensationSuccessRate']}%, and autonomous coverage {summary['autonomousCoverage']}%. Local p50/p95 is {report['latency']['p50Ms']}/{report['latency']['p95Ms']} ms.
"""


async def run(
    *,
    output_root: str | Path,
    iterations: int = 1,
    scenarios_path: str | Path = DEFAULT_SCENARIOS,
) -> dict[str, Any]:
    if not 1 <= iterations <= 100:
        raise ValueError("iterations must be between 1 and 100")
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    scenarios = load_scenarios(scenarios_path)
    cases: list[dict[str, Any]] = []
    with effect_lab_runtime_registration() as l0_digests:
        for iteration in range(1, iterations + 1):
            for scenario in scenarios:
                case_root = root / "runs" / f"{iteration:03d}-{scenario.id.lower()}"
                case_root.mkdir(parents=True, exist_ok=True)
                if scenario.kind == ScenarioKind.PARTIAL_MULTI_STEP_FAILURE:
                    value = await _partial_failure_case(scenario, root=case_root)
                else:
                    value = await _single_case(scenario, root=case_root)
                value["execution"] = _public_payload(value["execution"])
                value["iteration"] = iteration
                case_path = case_root / "case.json"
                case_path.write_text(
                    json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                value["caseArtifact"] = str(case_path.relative_to(root))
                cases.append(value)

    scores = [score_case(
        next(item for item in scenarios if item.id == value["scenario"]["id"]),
        Observation(**value["observation"]),
    ) for value in cases]
    latency = [float(item["latencyMs"]) for item in cases]
    report = {
        "schema": REPORT_SCHEMA,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "iterations": iterations,
        "scenarioManifestDigest": sha256_json([
            asdict(item) for item in scenarios
        ]),
        "runtime": {
            "reasoningHarness": "DSH contract-compatible candidate input",
            "provider": "deterministic-network-state-mcp-simulator",
            "realNetworkDevice": False,
            "l0Digests": l0_digests,
        },
        "summary": summarize(scores),
        "latency": {
            "meanMs": round(statistics.fmean(latency), 3),
            "p50Ms": _percentile(latency, 0.50),
            "p95Ms": _percentile(latency, 0.95),
        },
        "latestCases": [
            item for item in cases if item["iteration"] == iterations
        ],
        "allCaseArtifacts": [item["caseArtifact"] for item in cases],
        "claimBoundary": (
            "Fixed local scenarios are executable mechanism evidence, not a "
            "production success probability or live vendor-device qualification."
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
    parser.add_argument("--output-root", default="artifacts/ensuredskill-es-p0")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--scenarios", default=str(DEFAULT_SCENARIOS))
    args = parser.parse_args(argv)
    report = asyncio.run(run(
        output_root=args.output_root,
        iterations=args.iterations,
        scenarios_path=args.scenarios,
    ))
    summary = {
        key: value for key, value in report["summary"].items()
        if key not in {"caseScores", "claimBoundary"}
    }
    print(json.dumps({
        "report": str(Path(args.output_root).resolve() / "report.json"),
        "summary": summary,
        "latency": report["latency"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["summary"]["taskCompletionRate"] == 100.0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["DEFAULT_SCENARIOS", "REPORT_SCHEMA", "run"]
