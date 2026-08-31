"""Project-level evidence for NetOpYu's two core capability families.

Capability A compiles a natural-language L1 Skill through a reviewable L0.5
representation into an enforceable L0 contract.  Capability B executes that
contract through the deterministic Network Runtime.  The report keeps fixed
Oracle coverage separate from model accuracy and production availability.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from network_runtime.contracts import sha256_json, utc_now

from .promotion import assess_promotion
from .production_trajectory import validate_production_trajectories


REPORT_SCHEMA = "netopyu.io/core-capability-evaluation/v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNTIME_REPORT = PROJECT_ROOT / "artifacts/runtime-ab/runtime-ab.json"
DEFAULT_FORWARD_REPORT = PROJECT_ROOT / "artifacts/promotion-forward-calibration/report.json"
DEFAULT_FORWARD_MODEL_REPORT = (
    PROJECT_ROOT
    / "artifacts/promotion-forward-model/qwen3.5-9b-public-210/report.json"
)
DEFAULT_RUNTIME_REASSESSMENT_REPORT = (
    PROJECT_ROOT
    / "artifacts/promotion-forward-model/qwen3.5-9b-public-210"
    / "current-runtime-reassessment/report.json"
)
DEFAULT_JSON_REPORT = PROJECT_ROOT / "artifacts/core-capability-evaluation/current.json"
DEFAULT_MARKDOWN_REPORT = PROJECT_ROOT / "docs/core-capability-evaluation-report.md"


def _load_runtime_evidence(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"Runtime A/B evidence not found: {source}. Run "
            "scripts/netopyu-dsh compare-runtime --iterations 50 first."
        )
    report = json.loads(source.read_text(encoding="utf-8"))
    required = {"scenario_count", "metrics", "latency", "trend", "not_measured"}
    missing = sorted(required - set(report))
    if missing:
        raise ValueError(f"Runtime A/B evidence is missing: {', '.join(missing)}")
    by_path: dict[str, dict[str, dict[str, Any]]] = {}
    for path_id in ("dsh_only", "dsh_plus_runtime"):
        by_path[path_id] = {
            item["metric_id"]: item for item in report["metrics"][path_id]
        }
    required_metrics = {
        "valid_completion", "parameter_intent", "approval_binding", "read_policy",
        "result_recovery", "compensation", "saga", "evidence_integrity",
        "control_effectiveness",
    }
    for path_id, metrics in by_path.items():
        absent = sorted(required_metrics - set(metrics))
        if absent:
            raise ValueError(
                f"Runtime A/B evidence for {path_id} is missing: {', '.join(absent)}"
            )
    return {
        "source": str(source.relative_to(PROJECT_ROOT))
        if source.is_relative_to(PROJECT_ROOT) else str(source),
        "sourceDigest": sha256_json(report),
        "campaignId": report.get("campaign_id", "unknown"),
        "scenarioCount": report["scenario_count"],
        "methodology": report.get("methodology", {}),
        "metrics": by_path,
        "latency": report["latency"],
        "trend": report["trend"],
        "notMeasured": report["not_measured"],
    }


def _load_forward_calibration(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        return {
            "status": "not_generated",
            "qualificationEligible": False,
            "caseCount": 0,
            "familyCount": 0,
            "source": str(source),
            "claim": "Run scripts/netopyu-l0 forward-eval-calibrate.",
        }
    value = json.loads(source.read_text(encoding="utf-8"))
    if value.get("schema") != "netopyu.io/promotion-forward-calibration/v1":
        raise ValueError("forward calibration report Schema is invalid")
    coverage = value["coverage"]
    return {
        "status": value["status"],
        "qualificationEligible": bool(value["qualificationEligible"]),
        "caseCount": coverage["case_count"],
        "familyCount": coverage["family_count"],
        "variantsPerFamily": coverage["variants_per_family"],
        "source": str(source.relative_to(PROJECT_ROOT))
        if source.is_relative_to(PROJECT_ROOT) else str(source),
        "sourceDigest": sha256_json(value),
        "claim": value["claimBoundary"],
    }


def _load_forward_model_run(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        return {
            "status": "not_run",
            "qualified": False,
            "caseCount": 0,
            "familyCount": 0,
            "repetitions": 0,
            "metrics": {
                name: 0.0 for name in (
                    "protocol_completion_rate", "raw_protocol_completion_rate",
                    "bounded_normalization_rate", "capability_exact_match",
                    "parameter_predicate_exact_match", "intent_exact_match",
                    "safety_contract_exact_match", "runtime_promotion_ready_rate",
                    "semantic_contract_exact_match", "safety_escape_rate",
                )
            },
            "latency": {"p50": 0.0, "p95": 0.0},
            "efficiency": {
                "mean_model_calls": 0.0, "mean_repair_attempts": 0.0,
                "syntax_normalization_events": 0,
                "syntax_normalized_observations": 0,
            },
            "claim": "Run scripts/netopyu-l0 forward-eval-run-model --limit 210.",
        }
    value = json.loads(source.read_text(encoding="utf-8"))
    if value.get("schema") != "netopyu.io/promotion-forward-model-run/v1":
        raise ValueError("forward model-run report Schema is invalid")
    cases_path = Path(value["artifacts"]["cases"])
    families: set[str] = set()
    if cases_path.is_file():
        for line in cases_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                families.add(str(json.loads(line)["family"]))
    return {
        "status": value["status"],
        "qualified": bool(value["qualified"]),
        "model": value["model"],
        "modelArtifactDigest": value["model_artifact_digest"],
        "authoringProtocolDigest": value["authoring_protocol_digest"],
        "catalogSnapshotDigest": value["catalog_snapshot_digest"],
        "caseCount": value["dataset"]["case_count"],
        "familyCount": len(families),
        "repetitions": value["dataset"]["repetitions"],
        "metrics": value["metrics"],
        "slices": value.get("slices", {}),
        "latency": value["latency"],
        "efficiency": value["efficiency"],
        "failureCounts": value.get("failure_counts", {}),
        "source": str(source.relative_to(PROJECT_ROOT))
        if source.is_relative_to(PROJECT_ROOT) else str(source),
        "sourceDigest": sha256_json(value),
        "claim": value["claimBoundary"],
    }


def _load_runtime_reassessment(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        return {
            "status": "not_generated",
            "modelCalls": 0,
            "counts": {},
            "claim": (
                "Run scripts/netopyu-l0 forward-eval-reassess-runtime after a "
                "stored model run."
            ),
        }
    value = json.loads(source.read_text(encoding="utf-8"))
    if value.get("schema") != "netopyu.io/promotion-runtime-reassessment/v1":
        raise ValueError("Runtime reassessment report Schema is invalid")
    return {
        "status": value["gateConclusion"],
        "modelCalls": value["modelCalls"],
        "observations": value["observations"],
        "counts": value["counts"],
        "currentRuntime": value["currentRuntime"],
        "sourceRun": value["sourceRun"],
        "source": str(source.relative_to(PROJECT_ROOT))
        if source.is_relative_to(PROJECT_ROOT) else str(source),
        "sourceDigest": sha256_json(value),
        "claim": value["claimBoundary"],
    }


def _build_semantic_compilation_evidence(
    forward_report_path: str | Path,
    forward_model_report_path: str | Path,
    runtime_reassessment_report_path: str | Path,
) -> dict[str, Any]:
    example = PROJECT_ROOT / "network_runtime/l0/promotion_examples/url1-network-access"
    assessment = assess_promotion(
        skill_path=example / "SKILL.md",
        l05_path=example / "L0.5.yaml",
        candidate_path=PROJECT_ROOT / "network_runtime/l0/examples/s1-network-access-grant.yaml",
        capability_catalog_path=example / "capabilities.yaml",
    )
    coverage = assessment.report["semanticCoverage"]
    trajectories = validate_production_trajectories()
    return {
        "id": "semantic-contract-compilation",
        "nameZh": "分阶段语义合约编译与审查",
        "nameEn": "Staged semantic contract compilation and review",
        "status": "guarded_prototype_not_generalization_qualified",
        "functions": [
            "preserve L1, L0.5 and L0 as separate immutable review stages",
            "map requirements across L1-to-L0.5 and L0.5-to-L0",
            "score traceability evidence and highlight blocking semantic loss",
            "fail closed on parameter loss, scope widening, risk/approval weakening and unknown effects",
            "bind stages with digests and keep proposals non-executable until explicit review and publication",
        ],
        "fixedForwardSample": {
            "id": "url1-network-access",
            "status": assessment.report["status"],
            "errors": assessment.report["summary"]["errors"],
            "warnings": assessment.report["summary"]["warnings"],
            "semanticGate": coverage["gate"],
            "metrics": coverage["summary"],
            "scope": "one reviewed URL1/LAN authoring example",
        },
        "reverseBootstrap": {
            "contracts": trajectories["contracts"],
            "promotionReady": trajectories["promotion_ready"],
            "exactRoundTrips": trajectories["exact_round_trips"],
            "direction": "reviewed-L0-to-readable-L1/L0.5-baseline",
            "claim": "structural parity and exact compiler round trip only",
        },
        "agentizedForwardPath": {
            "trustedCatalogs": ["lan-user-access"],
            "catalogCount": 1,
            "realLlmEvidence": "single documented DSH golden path; not a benchmark",
            "activationAuthority": False,
        },
        "forwardQualificationProtocol": _load_forward_calibration(
            forward_report_path,
        ),
        "realModelForwardRun": _load_forward_model_run(
            forward_model_report_path,
        ),
        "currentRuntimeReassessment": _load_runtime_reassessment(
            runtime_reassessment_report_path,
        ),
        "performance": {
            "forwardModelAccuracy": "public_reverse_calibration_only_not_qualified",
            "conversionAvailability": "measured_on_210_public_reverse_cases_single_run_only",
            "conversionLatency": "measured_on_local_qwen3.5_9b_210_case_single_run_only",
        },
        "notMeasured": [
            "unseen natural-language requirement extraction recall",
            "expert-gold L0.5 and L0 exact match",
            "valid no-edit proposal yield and ambiguity-block rate",
            "three repeated 9B runs and any 27B comparison under the same protocol",
            "conversion concurrency, soak, HA, DR and production SLO",
        ],
    }


def _build_deterministic_runtime_evidence(path: str | Path) -> dict[str, Any]:
    evidence = _load_runtime_evidence(path)
    runtime = evidence["metrics"]["dsh_plus_runtime"]
    baseline = evidence["metrics"]["dsh_only"]
    return {
        "id": "deterministic-effect-transaction-runtime",
        "nameZh": "确定性 Effect 事务执行与验证",
        "nameEn": "Deterministic effect transaction execution and verification",
        "status": "qualified_on_core72_local_oracles_not_production",
        "functions": [
            "compile validated inputs into an immutable execution plan",
            "bind approval to identity, capability, arguments, plan hash, nonce and pre-state",
            "revalidate immediately before execution and block drift or replay",
            "verify outcomes independently instead of trusting provider success text",
            "reconcile uncertain outcomes, compensate or roll back, and preserve a tamper-evident audit chain",
            "coordinate ordered cross-domain Saga execution and reverse compensation",
        ],
        "campaign": {
            "id": evidence["campaignId"],
            "scenarioCount": evidence["scenarioCount"],
            "validScenarios": runtime["valid_completion"]["total"],
            "faultRiskScenarios": runtime["control_effectiveness"]["total"],
            "methodology": evidence["methodology"],
        },
        "comparison": {
            metric_id: {
                "labelZh": runtime[metric_id]["label_zh"],
                "labelEn": runtime[metric_id]["label_en"],
                "dshOnly": {
                    "passed": baseline[metric_id]["passed"],
                    "total": baseline[metric_id]["total"],
                    "rate": baseline[metric_id]["rate"],
                },
                "dshPlusRuntime": {
                    "passed": runtime[metric_id]["passed"],
                    "total": runtime[metric_id]["total"],
                    "rate": runtime[metric_id]["rate"],
                },
                "deltaPercentagePoints": round(
                    runtime[metric_id]["rate"] - baseline[metric_id]["rate"], 1
                ),
            }
            for metric_id in runtime
        },
        "latency": evidence["latency"],
        "trend": evidence["trend"],
        "source": evidence["source"],
        "sourceDigest": evidence["sourceDigest"],
        "notMeasured": evidence["notMeasured"],
    }


def build_core_capability_evaluation(
    *,
    runtime_report_path: str | Path = DEFAULT_RUNTIME_REPORT,
    forward_report_path: str | Path = DEFAULT_FORWARD_REPORT,
    forward_model_report_path: str | Path = DEFAULT_FORWARD_MODEL_REPORT,
    runtime_reassessment_report_path: str | Path = (
        DEFAULT_RUNTIME_REASSESSMENT_REPORT
    ),
) -> dict[str, Any]:
    """Build a source-derived snapshot for both core capability families."""

    body: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "generatedAt": utc_now(),
        "status": "local_dual_core_evidence_available_production_qualification_open",
        "systemFlow": [
            "human intent and L1 Skill",
            "semantic contract compilation: L1 -> L0.5 -> reviewed L0",
            "deterministic Network Runtime transaction",
            "service/network Provider and independent verifier",
        ],
        "claimBoundary": [
            "The two capability families are engineering concepts and potential protection families, not a patentability opinion.",
            "Traceability scores measure deterministic source-to-enforcement evidence, not LLM accuracy.",
            "Core-72 pass rates measure fixed local Oracles, not production correctness probabilities.",
            "A ready_for_review proposal is not approved, published, activated or executable.",
            "Real vendor qualification, distributed availability and production SLO remain open.",
        ],
        "capabilityA": _build_semantic_compilation_evidence(
            forward_report_path, forward_model_report_path,
            runtime_reassessment_report_path,
        ),
        "capabilityB": _build_deterministic_runtime_evidence(runtime_report_path),
        "combinedConclusion": {
            "semanticGap": "bounded by staged evidence and fail-closed review gates; unseen-language accuracy is not yet qualified",
            "executionGap": "fully covered on the current 64 fixed local fault/risk Oracles",
            "validPath": "8/8 fixed valid operations complete with and without Runtime",
            "productionReadiness": "not yet qualified without forward holdout, real-provider and SLO evidence",
        },
        "nextQualification": {
            "capabilityA": {
                "minimumForwardCases": 200,
                "safetyEscapesAllowed": 0,
                "targets": {
                    "toolCapabilityExactMatchPercent": 99.0,
                    "parameterPredicateExactMatchPercent": 95.0,
                    "ambiguityBlockPercent": 95.0,
                    "validProposalYieldPercent": 95.0,
                    "protocolCompletionPercent": 99.0,
                },
            },
            "capabilityB": {
                "preserveCore72OracleCoveragePercent": 100.0,
                "addRealProviderContractTests": True,
                "defineServiceSliSloAndErrorBudget": True,
                "qualifyConcurrencySoakHaAndDr": True,
            },
        },
    }
    body["reportDigest"] = sha256_json(body)
    return body


def _percent(value: Any) -> str:
    return f"{float(value):.2f}%"


def _rate_cell(metric: dict[str, Any]) -> str:
    return f"{metric['rate']:.1f}%（{metric['passed']}/{metric['total']}）"


def _comparison_rows(comparison: dict[str, Any]) -> str:
    order = (
        "valid_completion", "parameter_intent", "read_policy", "approval_binding",
        "result_recovery", "compensation", "saga", "evidence_integrity",
        "control_effectiveness",
    )
    return "\n".join(
        f"| {comparison[item]['labelZh']} | "
        f"{_rate_cell(comparison[item]['dshOnly'])} | "
        f"{_rate_cell(comparison[item]['dshPlusRuntime'])} | "
        f"{comparison[item]['deltaPercentagePoints']:+.1f} pp |"
        for item in order
    )


def _forward_challenge_rows(model_run: dict[str, Any]) -> str:
    challenge = (model_run.get("slices") or {}).get("challenge", {})
    if not challenge:
        return "| unavailable | — | — | — |"
    return "\n".join(
        f"| {name} | {_percent(item['metrics']['protocol_completion_rate'] * 100)} | "
        f"{_percent(item['metrics']['semantic_contract_exact_match'] * 100)} | "
        f"{_percent(item['metrics']['runtime_promotion_ready_rate'] * 100)} |"
        for name, item in sorted(challenge.items())
    )


def render_core_capability_evaluation_markdown(report: dict[str, Any]) -> str:
    """Render the dual-core report in Chinese first and English second."""

    a = report["capabilityA"]
    fixed = a["fixedForwardSample"]
    metrics = fixed["metrics"]
    reverse = a["reverseBootstrap"]
    forward = a["forwardQualificationProtocol"]
    model_run = a["realModelForwardRun"]
    reassessment = a["currentRuntimeReassessment"]
    reassessment_counts = reassessment.get("counts") or {}
    challenge_rows = _forward_challenge_rows(model_run)
    b = report["capabilityB"]
    comparison = b["comparison"]
    latency = b["latency"]
    dsh_latency = latency["dsh_only"]
    runtime_latency = latency["dsh_plus_runtime"]
    trend = b["trend"]
    return f"""# NetOpYu 双核心功能与性能评估 / Core Capability Evaluation

> 自动生成于 `{report['generatedAt']}`；摘要 `{report['reportDigest']}`。这是工程证据报告，不构成专利可授权性或生产 SLA 结论。

## 中文

### 1. 项目要证明的两件事

```text
用户意图 / L1 Skill
        │
        ▼
[核心 A] L1 → L0.5 → L0：把开放语言逐步编译为可审查、可执行的语义合约
        │ reviewed L0 contract
        ▼
[核心 B] Network Runtime：把合约收敛为审批绑定、验证、恢复和审计的确定性事务
        │
        ▼
Service / Network Provider + 独立 Verifier
```

| 核心 | 已实现功能 | 当前量化结论 | 尚未证明 |
|---|---|---|---|
| A：分阶段语义合约编译 | 三阶段留痕、双段映射、语义丢失告警、安全门禁、防篡改审查 | 固定 URL1 gate 通过；9B 实跑 {model_run['caseCount']} 条/{model_run['familyCount']} 能力族，Runtime 可审率 {_percent(model_run.get('metrics', {}).get('runtime_promotion_ready_rate', 0) * 100)} | 私有独立正向准确率、三次重复与 HA 尚未取得 |
| B：确定性 Effect 事务 Runtime | 不可变计划、审批绑定、执行前重校验、独立验证、对账、补偿、Saga、审计 | Core-72：有效请求 {comparison['valid_completion']['dshPlusRuntime']['passed']}/{comparison['valid_completion']['dshPlusRuntime']['total']}；风险/故障 Oracle {comparison['control_effectiveness']['dshPlusRuntime']['passed']}/{comparison['control_effectiveness']['dshPlusRuntime']['total']} | 真实厂商设备、人工审批时延、并发长稳、分布式 HA 与生产 SLO |

项目当前已经分别回答了“语义如何被约束”和“确定操作如何安全落地”，但还不能声称任意自然语言或真实生产环境达到 100% 准确、稳定或可用。

### 2. 核心 A：L1 → L0.5 → L0

#### 2.1 功能

1. 保存自然语言 L1、结构化自然语言 L0.5 和机器执行 L0 三份独立制品。
2. 分别建立 L1→L0.5 与 L0.5→L0 requirement 映射，支持按风险和用户关注点展开。
3. 计算可追溯证据分、机器约束覆盖和语义表示覆盖；对低置信、缺失、弱化和歧义项告警。
4. 参数删除、作用域扩大、风险/审批弱化、未知 Effect、不独立验证和不安全重试失败关闭。
5. 阶段及前驱 SHA-256 绑定；模型只有 proposal 权限，不能自行注册、激活或执行。

#### 2.2 固定正向样例指标

| 指标 | 结果 |
|---|---:|
| 状态 / semantic gate | `{fixed['status']}` / `{fixed['semanticGate']}` |
| Requirement | {metrics['totalRequirements']} |
| Preserved | {metrics['preserved']} |
| Non-machine-verifiable | {metrics['non_machine_verifiable']} |
| Blocking | {metrics['blockingRequirements']} |
| L1 → L0.5 证据分 | {_percent(metrics['averageL1ToL05Confidence'])} |
| L0.5 → L0 证据分 | {_percent(metrics['averageL05ToL0Confidence'])} |
| 端到端映射证据分 | {_percent(metrics['averageMappingConfidence'])} |
| 机器执行约束覆盖 | {_percent(metrics['machineEnforcedPercent'])} |
| 语义表示覆盖 | {_percent(metrics['semanticCoveragePercent'])} |

这些是**可追溯证据分，不是 LLM 准确率**。`non_machine_verifiable` 表示语言仍可见但没有确定性 L0 谓词，必须人工审查。

另有 {reverse['contracts']} 个存量合同轨迹通过 Promotion 与精确 round-trip，但方向是受审 L0 反向生成 L1/L0.5 基线，只证明结构闭环和编译一致性，不证明模型正向泛化。

#### 2.3 真实 qwen3.5:9b 单次前向基线

| 指标 | 结果 |
|---|---:|
| 用例 / 能力族 / 重复 | {model_run['caseCount']} / {model_run['familyCount']} / {model_run['repetitions']} |
| 模型原始严格协议完成 | {_percent(model_run['metrics'].get('raw_protocol_completion_rate', model_run['metrics']['protocol_completion_rate']) * 100)} |
| 受限规范化后协议完成 | {_percent(model_run['metrics']['protocol_completion_rate'] * 100)} |
| Capability exact match | {_percent(model_run['metrics']['capability_exact_match'] * 100)} |
| 参数/谓词 exact match | {_percent(model_run['metrics']['parameter_predicate_exact_match'] * 100)} |
| Intent exact match | {_percent(model_run['metrics']['intent_exact_match'] * 100)} |
| 安全合同 exact match | {_percent(model_run['metrics']['safety_contract_exact_match'] * 100)} |
| Runtime ready_for_review | {_percent(model_run['metrics']['runtime_promotion_ready_rate'] * 100)} |
| 全语义 exact match / safety escape | {_percent(model_run['metrics']['semantic_contract_exact_match'] * 100)} / {_percent(model_run['metrics']['safety_escape_rate'] * 100)} |
| 本机 p50 / p95 | {model_run['latency']['p50'] / 1000:.3f} / {model_run['latency']['p95'] / 1000:.3f} s |
| 平均模型调用 / 修复 | {model_run['efficiency']['mean_model_calls']:.3f} / {model_run['efficiency']['mean_repair_attempts']:.3f} |
| 受限 enum 规范化 | {model_run['efficiency'].get('syntax_normalized_observations', 0)} 条 / {model_run['efficiency'].get('syntax_normalization_events', 0)} 个值 |

| 包装变体 | 协议完成 | 全语义 exact | Runtime 可审 |
|---|---:|---:|---:|
{challenge_rows}

这是同一 `qwen3.5:9b` 制品在 21 个公开反向能力族、每族 10 个中英文/追踪/安全/Schema/对抗包装上的真实模型调用，不是 evaluator self-check。L1/L0.5 v3 以 capability-scoped、逐字段可比的语义锚点保存 intent，并把 preflight、success-verification、compensation-verification 显式分型；本轮历史 intent exact 为 {_percent(model_run['metrics']['intent_exact_match'] * 100)}。模型原始协议与受限规范化后协议被分别计量：规范化只接受参数 enum 内精确的一键 `value` primitive 包装，并记录路径和前后摘要，不放宽 L0 核心 Schema。历史结果仍不是资格结论：数据由受审 L0 反向生成且仅一次重复。

#### 2.4 Phase-typed Capability 当前 Runtime 重放

| 指标 | 结果 |
|---|---:|
| 重放 Observation / 模型调用 | {reassessment.get('observations', 0)} / {reassessment.get('modelCalls', 0)} |
| 当前 ready / fail-closed | {reassessment_counts.get('current_ready', 0)} / {reassessment_counts.get('current_fail_closed', 0)} |
| 历史 exact-ready 保留 | {reassessment_counts.get('exact_ready_preserved', 0)}/{reassessment_counts.get('historical_exact_ready', 0)} |
| 历史错误可审候选新增阻断 | {reassessment_counts.get('false_ready_closed', 0)} |
| exact-ready 回归 | {reassessment_counts.get('exact_ready_regressed', 0)} |
| 结论 | `{reassessment.get('status', 'not_generated')}` |

该重放没有调用模型，也没有改写 {reassessment.get('observations', 0)} 条历史 Observation；它只把已保存的规范化语义 proposal 送入当前 Catalog v2/L0.5 v3 Runtime。结果新增阻断 {reassessment_counts.get('false_ready_closed', 0)} 条历史错误可审候选，同时保留 {reassessment_counts.get('exact_ready_preserved', 0)}/{reassessment_counts.get('historical_exact_ready', 0)} 条历史全语义 exact 且可审候选。它只证明确定性门禁对已知 false-ready 的增量，不证明模型准确率提高。

#### 2.5 当前性能与资格缺口

- DSH 页面交互式 authoring 仍只有 `lan-user-access` 一个发布级入口；独立 evaluator 已覆盖 21 个可信 Catalog，但两者都不是统计资格或生产 SLO。
- 已建立 {forward['caseCount']} 条、{forward['familyCount']} 个能力族的公开校准协议矩阵，状态 `{forward['status']}`；它来自受审 L0 的反向轨迹，因此只校准评分器，`qualificationEligible={str(forward['qualificationEligible']).lower()}`。
- 正式协议已强制至少 200 个独立人工正向用例、仓库外私有 holdout、双人一致、同一模型制品至少三次运行，并计算 Capability、参数/谓词、安全合同、全语义 exact match、歧义阻断、proposal yield、重复稳定性与 p50/p95。
- 尚未取得真实私有数据和重复模型 Observation，所以正向模型准确率仍未资格化；固定安全集关键语义、Effect 和审批弱化逃逸必须为 0。

### 3. 核心 B：Network Runtime 确定性执行

#### 3.1 功能

1. 将已校验参数编译成不可变执行计划。
2. 审批绑定身份、能力、版本、参数、计划哈希、nonce 和执行前状态。
3. 执行前重新校验并阻断审批后漂移、重放和越权读取。
4. 使用独立 Verifier 判断真实目标状态，不信任 Provider 的成功文本。
5. 对断连或不确定结果先对账，再补偿/回滚；跨域操作执行 Saga 逆序补偿。
6. 终态和事件链防篡改审计，不能把未知结果伪装成成功。

#### 3.2 Core-72 功能对比

| 指标 | DSH only | DSH + Runtime | 增量 |
|---|---:|---:|---:|
{_comparison_rows(comparison)}

两条路径使用相同 Tool、参数、Provider 和注入故障，固定 L1 决策并排除模型选择影响。`100%（64/64）`只表示当前 Runtime 通过全部固定本地风险/故障 Oracle，不是生产成功概率。

#### 3.3 本地机器时延与趋势

| 路径 | p50 | p95 | 样本 |
|---|---:|---:|---:|
| DSH only | {dsh_latency['p50_ms']:.3f} ms | {dsh_latency['p95_ms']:.3f} ms | {dsh_latency['samples']} |
| DSH + Runtime | {runtime_latency['p50_ms']:.3f} ms | {runtime_latency['p95_ms']:.3f} ms | {runtime_latency['samples']} |

Runtime p50/p95 绝对增量为 {runtime_latency['p50_overhead_ms']:.3f}/{runtime_latency['p95_overhead_ms']:.3f} ms；人工审批等待不计入。最近 {trend['unique_iterations']} 个不同实现指纹趋势为 `{trend['status']}`，Runtime p50/p95 中位数为 {trend['median']['p50_ms']:.3f}/{trend['median']['p95_ms']:.3f} ms。mock 直接路径接近零成本，不能用相对倍数外推生产性能。

### 4. 双核心组合后的真实边界

- 核心 A 限制“LLM 想做什么、遗漏了什么、哪些语义没有进入机器约束”；核心 B 限制“获准的确定意图如何执行、验证、失败恢复和留证”。
- A 的 proposal 即使 gate 通过也不能绕过人工 review/publish；B 只接受激活的 L0 合约，不能替模型修复错误业务意图。
- 当前最强证据是**固定语义样例可追溯 + 固定 Runtime Oracle 全覆盖**。最大证据缺口是**独立正向语义基准 + 真实 Provider/设备资格化 + 生产 SLO**。

### 5. 复算

```bash
# 先刷新核心 B 的本地证据
scripts/netopyu-dsh compare-runtime --iterations 50

# 刷新核心 A 的公开校准协议（不产生模型资格结论）
scripts/netopyu-l0 forward-eval-calibrate

# 用本地 9B 跑 21 能力族 × 10 包装变体；支持 --resume
scripts/netopyu-l0 forward-eval-run-model --model qwen3.5:9b --limit 210 \\
  --output-root artifacts/promotion-forward-model/qwen3.5-9b-public-210

# 不调用模型，用当前 Runtime 重放历史 proposal
scripts/netopyu-l0 forward-eval-reassess-runtime \\
  --output-root artifacts/promotion-forward-model/qwen3.5-9b-public-210

# 再生成本双核心报告
scripts/netopyu-l0 core-eval-report

# 回归门禁
.venv/bin/python -m pytest -q
```

机器快照：[`artifacts/core-capability-evaluation/current.json`](../artifacts/core-capability-evaluation/current.json)。详细设计见 [正向资格协议](promotion-forward-qualification.md)、[L1 → L0 Promotion](l1-to-l0-promotion.md)、[Promotion Workbench](p20-promotion-workbench.md)、[Runtime A/B 基线](benchmarks/runtime-ab-baseline.md) 和 [架构](../ARCHITECTURE.md)。

---

## English

### 1. Two core capability families

Capability A compiles an open-ended L1 Skill through a reviewable L0.5 representation into an enforceable L0 contract. Capability B executes an activated L0 contract as a deterministic transaction with approval binding, revalidation, independent verification, recovery, compensation and tamper-evident audit.

For Capability A, the fixed URL1 sample passes its semantic gate with {metrics['preserved']}/{metrics['totalRequirements']} requirements preserved, {metrics['non_machine_verifiable']} explicitly non-machine-verifiable, and {metrics['blockingRequirements']} blocking. The real qwen3.5:9b robustness run covered {model_run['caseCount']} public reverse-bootstrap cases across {model_run['familyCount']} families and ten bilingual/trace/safety/schema/adversarial wrappers per family: raw/normalized-boundary protocol completion was {_percent(model_run['metrics'].get('raw_protocol_completion_rate', model_run['metrics']['protocol_completion_rate']) * 100)}/{_percent(model_run['metrics']['protocol_completion_rate'] * 100)}, full-semantic exact match {_percent(model_run['metrics']['semantic_contract_exact_match'] * 100)}, historical Runtime promotion readiness {_percent(model_run['metrics']['runtime_promotion_ready_rate'] * 100)}, and safety escape {_percent(model_run['metrics']['safety_escape_rate'] * 100)}. A no-model-call replay through the current phase-typed Runtime preserved {reassessment_counts.get('exact_ready_preserved', 0)}/{reassessment_counts.get('historical_exact_ready', 0)} historically exact-ready proposals and fail-closed {reassessment_counts.get('false_ready_closed', 0)} known false-ready phase selection. The bounded normalizer preserves path/digest evidence and does not relax the L0 schema. p50/p95 were {model_run['latency']['p50'] / 1000:.3f}/{model_run['latency']['p95'] / 1000:.3f} seconds. This remains reverse-bootstrapped, single-repeat, and ineligible for qualification.

For Capability B, the Core-72 campaign preserves 8/8 valid completions and raises fixed fault/risk Oracle coverage from {comparison['control_effectiveness']['dshOnly']['passed']}/{comparison['control_effectiveness']['dshOnly']['total']} ({comparison['control_effectiveness']['dshOnly']['rate']:.1f}%) to {comparison['control_effectiveness']['dshPlusRuntime']['passed']}/{comparison['control_effectiveness']['dshPlusRuntime']['total']} ({comparison['control_effectiveness']['dshPlusRuntime']['rate']:.1f}%). Runtime p50/p95 are {runtime_latency['p50_ms']:.3f}/{runtime_latency['p95_ms']:.3f} ms in the local mock campaign; human approval wait is excluded.

The project therefore has concrete evidence for semantic traceability gates and deterministic execution controls. It does not yet have statistical forward-language accuracy, conversion availability, real-vendor qualification, distributed HA, or a production SLO. Fixed-set 100% must not be presented as a production success probability.

### 2. Reproduce

```bash
scripts/netopyu-dsh compare-runtime --iterations 50
scripts/netopyu-l0 forward-eval-calibrate
scripts/netopyu-l0 forward-eval-run-model --model qwen3.5:9b --limit 210 \\
  --output-root artifacts/promotion-forward-model/qwen3.5-9b-public-210
scripts/netopyu-l0 forward-eval-reassess-runtime \\
  --output-root artifacts/promotion-forward-model/qwen3.5-9b-public-210
scripts/netopyu-l0 core-eval-report
.venv/bin/python -m pytest -q
```

See [Forward qualification](promotion-forward-qualification.md), [L1 → L0 Promotion](l1-to-l0-promotion.md), [Promotion Workbench](p20-promotion-workbench.md), [Runtime A/B baseline](benchmarks/runtime-ab-baseline.md), and [Architecture](../ARCHITECTURE.md).
"""


def write_core_capability_evaluation(
    *,
    runtime_report_path: str | Path = DEFAULT_RUNTIME_REPORT,
    forward_report_path: str | Path = DEFAULT_FORWARD_REPORT,
    forward_model_report_path: str | Path = DEFAULT_FORWARD_MODEL_REPORT,
    runtime_reassessment_report_path: str | Path = (
        DEFAULT_RUNTIME_REASSESSMENT_REPORT
    ),
    json_path: str | Path = DEFAULT_JSON_REPORT,
    markdown_path: str | Path = DEFAULT_MARKDOWN_REPORT,
) -> dict[str, Any]:
    """Write machine-readable and Chinese-first human evidence reports."""

    report = build_core_capability_evaluation(
        runtime_report_path=runtime_report_path,
        forward_report_path=forward_report_path,
        forward_model_report_path=forward_model_report_path,
        runtime_reassessment_report_path=runtime_reassessment_report_path,
    )
    json_destination = Path(json_path).expanduser().resolve()
    markdown_destination = Path(markdown_path).expanduser().resolve()
    for destination in (json_destination, markdown_destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
    json_destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_destination.write_text(
        render_core_capability_evaluation_markdown(report), encoding="utf-8",
    )
    return {
        "ok": True,
        "status": report["status"],
        "report_digest": report["reportDigest"],
        "json": str(json_destination),
        "markdown": str(markdown_destination),
    }
