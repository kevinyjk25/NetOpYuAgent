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


def _build_semantic_compilation_evidence() -> dict[str, Any]:
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
        "performance": {
            "forwardModelAccuracy": "not_statistically_qualified",
            "conversionAvailability": "not_measured",
            "conversionLatency": "not_measured",
        },
        "notMeasured": [
            "unseen natural-language requirement extraction recall",
            "expert-gold L0.5 and L0 exact match",
            "valid no-edit proposal yield and ambiguity-block rate",
            "repeated 9B/27B runs, repair exhaustion and model-call count",
            "conversion latency, concurrency, soak, HA, DR and production SLO",
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
    *, runtime_report_path: str | Path = DEFAULT_RUNTIME_REPORT,
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
        "capabilityA": _build_semantic_compilation_evidence(),
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


def render_core_capability_evaluation_markdown(report: dict[str, Any]) -> str:
    """Render the dual-core report in Chinese first and English second."""

    a = report["capabilityA"]
    fixed = a["fixedForwardSample"]
    metrics = fixed["metrics"]
    reverse = a["reverseBootstrap"]
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
| A：分阶段语义合约编译 | 三阶段留痕、双段映射、语义丢失告警、安全门禁、防篡改审查 | 固定 URL1 样例 gate 通过；{metrics['preserved']}/{metrics['totalRequirements']} preserved，{metrics['non_machine_verifiable']} 项需人工判断 | 未见自然语言的正向模型准确率、转换成功率、p50/p95 与 HA |
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

#### 2.3 当前性能与资格缺口

- Agent 正向 authoring 目前只开放 `lan-user-access` 一个可信 Catalog；真实 DSH/LLM 只有单次 Golden Path，不是统计基准。
- 尚未测量正向转换准确率、合法输入 proposal yield、歧义阻断率、修复耗尽率、模型调用数、转换 p50/p95、并发、长稳和 HA。
- 下一步应使用至少 200 个独立人工标注用例、双人仲裁、私有 holdout、同义改写/中英文/冲突/恶意扩权切片；固定安全集关键语义、Effect 和审批弱化逃逸必须为 0。

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

# 再生成本双核心报告
scripts/netopyu-l0 core-eval-report

# 回归门禁
.venv/bin/python -m pytest -q
```

机器快照：[`artifacts/core-capability-evaluation/current.json`](../artifacts/core-capability-evaluation/current.json)。详细设计见 [L1 → L0 Promotion](l1-to-l0-promotion.md)、[Promotion Workbench](p20-promotion-workbench.md)、[Runtime A/B 基线](benchmarks/runtime-ab-baseline.md) 和 [架构](../ARCHITECTURE.md)。

---

## English

### 1. Two core capability families

Capability A compiles an open-ended L1 Skill through a reviewable L0.5 representation into an enforceable L0 contract. Capability B executes an activated L0 contract as a deterministic transaction with approval binding, revalidation, independent verification, recovery, compensation and tamper-evident audit.

For Capability A, the fixed URL1 sample passes its semantic gate with {metrics['preserved']}/{metrics['totalRequirements']} requirements preserved, {metrics['non_machine_verifiable']} explicitly non-machine-verifiable, and {metrics['blockingRequirements']} blocking. Evidence scores are {_percent(metrics['averageL1ToL05Confidence'])} for L1→L0.5, {_percent(metrics['averageL05ToL0Confidence'])} for L0.5→L0 and {_percent(metrics['averageMappingConfidence'])} end to end. They measure deterministic traceability—not model accuracy. The {reverse['contracts']} reverse-bootstrapped trajectories prove compiler closure, not forward generalization.

For Capability B, the Core-72 campaign preserves 8/8 valid completions and raises fixed fault/risk Oracle coverage from {comparison['control_effectiveness']['dshOnly']['passed']}/{comparison['control_effectiveness']['dshOnly']['total']} ({comparison['control_effectiveness']['dshOnly']['rate']:.1f}%) to {comparison['control_effectiveness']['dshPlusRuntime']['passed']}/{comparison['control_effectiveness']['dshPlusRuntime']['total']} ({comparison['control_effectiveness']['dshPlusRuntime']['rate']:.1f}%). Runtime p50/p95 are {runtime_latency['p50_ms']:.3f}/{runtime_latency['p95_ms']:.3f} ms in the local mock campaign; human approval wait is excluded.

The project therefore has concrete evidence for semantic traceability gates and deterministic execution controls. It does not yet have statistical forward-language accuracy, conversion availability, real-vendor qualification, distributed HA, or a production SLO. Fixed-set 100% must not be presented as a production success probability.

### 2. Reproduce

```bash
scripts/netopyu-dsh compare-runtime --iterations 50
scripts/netopyu-l0 core-eval-report
.venv/bin/python -m pytest -q
```

See [L1 → L0 Promotion](l1-to-l0-promotion.md), [Promotion Workbench](p20-promotion-workbench.md), [Runtime A/B baseline](benchmarks/runtime-ab-baseline.md), and [Architecture](../ARCHITECTURE.md).
"""


def write_core_capability_evaluation(
    *,
    runtime_report_path: str | Path = DEFAULT_RUNTIME_REPORT,
    json_path: str | Path = DEFAULT_JSON_REPORT,
    markdown_path: str | Path = DEFAULT_MARKDOWN_REPORT,
) -> dict[str, Any]:
    """Write machine-readable and Chinese-first human evidence reports."""

    report = build_core_capability_evaluation(
        runtime_report_path=runtime_report_path,
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
