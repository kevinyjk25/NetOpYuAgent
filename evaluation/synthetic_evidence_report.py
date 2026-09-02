"""Build a digest-bound summary for the repository-external synthetic study.

The summary is intentionally descriptive.  It never upgrades model-authored
synthetic evidence into official ES-P1 qualification or a production success
probability.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from evaluation.synthetic_skill_holdout import (
    EVIDENCE_CLASS, inspect_synthetic_packages, load_synthetic_dataset,
)
from network_runtime.contracts import sha256_json


REPORT_SCHEMA = "effect-runtime.io/synthetic-evidence-summary/v1"


def _percent(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0


def _dimension_summary(
    rows: Iterable[dict[str, Any]], key: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    result: dict[str, dict[str, Any]] = {}
    for value, items in sorted(grouped.items()):
        total = len(items)
        qualified = sum(item["route"] == "l0_runtime" for item in items)
        result[value] = {
            "total": total,
            "rawProtocolValid": sum(bool(item["raw_protocol_valid"]) for item in items),
            "trustedOracleQualified": qualified,
            "trustedOracleQualifiedPercent": _percent(qualified, total),
            "fallbacks": total - qualified,
            "meanSemanticCoveragePercent": round(statistics.fmean(
                float(item["semantic_coverage_percent"]) for item in items
            ), 2),
        }
    return result


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"report must be a JSON object: {path}")
    return value


def build_synthetic_evidence_summary(
    *, dataset_root: str | Path, translation_report: str | Path,
    output_root: str | Path, dsh_report: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(dataset_root).expanduser().resolve()
    manifest, cases = load_synthetic_dataset(root)
    inspection = inspect_synthetic_packages(root)
    translation_path = Path(translation_report).expanduser().resolve()
    translation = _load_json(translation_path)
    if translation.get("dataset", {}).get("digest") != manifest["datasetDigest"]:
        raise ValueError("translation report does not bind the sealed synthetic data set")
    if translation.get("dataset", {}).get("executedCases") != len(cases):
        raise ValueError("translation report is not a complete synthetic-corpus run")

    case_index = {case.case_id: case for case in cases}
    rows: list[dict[str, Any]] = []
    for item in translation.get("cases") or []:
        case_id = str(item.get("case_id") or "")
        case = case_index.get(case_id)
        if case is None:
            raise ValueError(f"translation report contains unknown case: {case_id}")
        rows.append({
            **item,
            "scenario_pattern": case.scenario_pattern,
            "language": case.language,
        })
    if len(rows) != len(cases) or len({item["case_id"] for item in rows}) != len(cases):
        raise ValueError("translation report case coverage is incomplete or duplicated")

    route_violations = sum(
        item["route"] == "l0_runtime" and not item["oracle_passed"] for item in rows
    )
    field_failures: Counter[str] = Counter()
    for item in rows:
        field_failures.update(
            name for name, passed in item["field_results"].items() if not passed
        )
    summary: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "evidenceClass": EVIDENCE_CLASS,
        "officialEsP1QualificationEligible": False,
        "dataset": {
            "manifestDigest": manifest["manifestDigest"],
            "datasetDigest": manifest["datasetDigest"],
            "cases": len(cases),
            "coverage": manifest["coverage"],
            "renderer": manifest["renderer"],
            "packageInspection": inspection,
        },
        "translation": {
            "source": str(translation_path),
            "sourceDigest": sha256_json(translation),
            "model": translation.get("model"),
            "metrics": translation["metrics"],
            "runtimeRouteOracleViolations": route_violations,
            "fieldFailureCounts": dict(sorted(field_failures.items())),
            "byFeature": _dimension_summary(rows, "feature_family"),
            "byScenario": _dimension_summary(rows, "scenario_pattern"),
            "byDomain": _dimension_summary(rows, "domain"),
            "byLanguage": _dimension_summary(rows, "language"),
        },
        "dshPairedExecution": None,
        "claimBoundary": (
            "Repository-external, context-isolated, model-authored sealed synthetic "
            "holdout evidence. It is not independently human-authored ES-P1 truth, "
            "a production success probability, or real-network qualification."
        ),
    }
    if dsh_report is not None:
        dsh_path = Path(dsh_report).expanduser().resolve()
        dsh = _load_json(dsh_path)
        if dsh.get("dataset", {}).get("digest") != manifest["datasetDigest"]:
            raise ValueError("DSH report does not bind the sealed synthetic data set")
        summary["dshPairedExecution"] = {
            "source": str(dsh_path),
            "sourceDigest": sha256_json(dsh),
            "dataset": dsh.get("dataset"),
            "metrics": dsh.get("metrics"),
            "coverageEffectCurve": dsh.get("coverageEffectCurve"),
            "byScenario": dsh.get("byScenario"),
            "byFeature": dsh.get("byFeature"),
        }

    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "synthetic-evidence-summary.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    translation_metrics = summary["translation"]["metrics"]
    dsh_metrics = (summary["dshPairedExecution"] or {}).get("metrics")
    markdown = [
        "# 合成证据摘要 / Synthetic Evidence Summary",
        "",
        "## 中文",
        "",
        f"- 密封用例：{len(cases)}；数据摘要：`{manifest['datasetDigest']}`",
        f"- Skill 包门禁：{inspection['packageGates']}；发现：{inspection['findingCounts']}",
        f"- 原始协议有效：{translation_metrics['rawProtocolValid']}/{translation_metrics['total']}",
        f"- 可信 Oracle 合格：{translation_metrics['oraclePassed']}/{translation_metrics['total']}；fallback：{translation_metrics['fallbacks']}",
        f"- Runtime 路由违反 Oracle：{route_violations}",
        f"- 转译时延 p50 / p95：{translation_metrics['latency']['p50Ms']} / {translation_metrics['latency']['p95Ms']} ms",
    ]
    if dsh_metrics:
        control = dsh_metrics["dsh_l1_native"]
        treatment = dsh_metrics["dsh_l0_auto_runtime"]
        markdown.extend([
            f"- DSH 任务完成率：Control {control['taskCompletionRate']}%，Treatment {treatment['taskCompletionRate']}%",
            f"- DSH 执行精度：Control {control['executionPrecision']}%，Treatment {treatment['executionPrecision']}%",
            f"- 不安全执行：Control {control['unsafeExecutions']}，Treatment {treatment['unsafeExecutions']}；错误提交：Control {control['falseCommits']}，Treatment {treatment['falseCommits']}",
            f"- 无效动作：Control {control['invalidActions']}，Treatment {treatment['invalidActions']}；进程失败：Control {control['processFailures']}，Treatment {treatment['processFailures']}",
            f"- DSH p50 / p95：Control {control['latency']['p50Ms']} / {control['latency']['p95Ms']} ms；Treatment {treatment['latency']['p50Ms']} / {treatment['latency']['p95Ms']} ms",
            f"- Runtime 审计：{treatment['runtimeAudits']['valid']}/{treatment['runtimeAudits']['applicable']} 有效",
        ])
    markdown.extend([
        "",
        "以上是模型生成的仓库外合成证据，不是独立人工 ES-P1、生产成功概率或真实网络资格。",
        "",
        "## English",
        "",
        f"The sealed corpus contains {len(cases)} cases. Raw protocol validity is {translation_metrics['rawProtocolValid']}/{translation_metrics['total']}; {translation_metrics['oraclePassed']} translations passed every trusted Catalog Oracle and {translation_metrics['fallbacks']} remained fallback-only.",
        f"Runtime-route Oracle violations: {route_violations}.",
        *([] if not dsh_metrics else [
            f"Across the paired DSH runs, task completion was {dsh_metrics['dsh_l1_native']['taskCompletionRate']}% for Control and {dsh_metrics['dsh_l0_auto_runtime']['taskCompletionRate']}% for Treatment; unsafe executions were {dsh_metrics['dsh_l1_native']['unsafeExecutions']} and {dsh_metrics['dsh_l0_auto_runtime']['unsafeExecutions']}, respectively.",
        ]),
        "This is model-authored sealed synthetic evidence, not independent human ES-P1 qualification, a production success probability, or real-network qualification.",
        "",
    ])
    (output / "synthetic-evidence-summary.md").write_text(
        "\n".join(markdown), encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--translation-report", required=True)
    parser.add_argument("--dsh-report")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    report = build_synthetic_evidence_summary(
        dataset_root=args.dataset_root,
        translation_report=args.translation_report,
        dsh_report=args.dsh_report,
        output_root=args.output_root,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_synthetic_evidence_summary"]
