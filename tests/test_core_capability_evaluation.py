from __future__ import annotations

import json
import subprocess
from pathlib import Path

from network_runtime.contracts import sha256_json
from network_runtime.l0.core_capability_evaluation import (
    REPORT_SCHEMA,
    build_core_capability_evaluation,
    render_core_capability_evaluation_markdown,
    write_core_capability_evaluation,
)


ROOT = Path(__file__).resolve().parents[1]


def _runtime_report(path: Path) -> Path:
    metrics = [
        ("valid_completion", "有效请求完成率", "Valid request completion", 8, 8, 8),
        ("parameter_intent", "参数与意图收口率", "Parameter and intent closure", 2, 12, 12),
        ("approval_binding", "审批后漂移阻断率", "Post-approval drift blocking", 1, 12, 12),
        ("read_policy", "越权读取阻断率", "Unauthorized-read blocking", 2, 8, 8),
        ("result_recovery", "结果判定与恢复率", "Outcome resolution and recovery", 0, 12, 12),
        ("compensation", "补偿与回滚正确率", "Compensation and rollback correctness", 0, 8, 8),
        ("saga", "跨域 Saga 控制率", "Cross-domain Saga control", 0, 6, 6),
        ("evidence_integrity", "终态与审计完整率", "Terminal and audit integrity", 0, 6, 6),
        ("control_effectiveness", "故障/风险控制有效率", "Fault/risk control effectiveness", 5, 64, 64),
    ]

    def build(path_id: str) -> list[dict[str, object]]:
        result = []
        for metric_id, zh, en, dsh_passed, total, runtime_passed in metrics:
            passed = dsh_passed if path_id == "dsh_only" else runtime_passed
            result.append({
                "metric_id": metric_id,
                "label_zh": zh,
                "label_en": en,
                "passed": passed,
                "total": total,
                "rate": round(passed / total * 100, 1),
            })
        return result

    value = {
        "campaign_id": "core72-v1",
        "scenario_count": 72,
        "metrics": {"dsh_only": build("dsh_only"), "dsh_plus_runtime": build("runtime")},
        "methodology": {"fixed_inputs": True, "llm_and_l1_selection_excluded": True},
        "latency": {
            "unit": "milliseconds",
            "approval_wait_excluded": True,
            "dsh_only": {"samples": 50, "p50_ms": 0.313, "p95_ms": 0.393},
            "dsh_plus_runtime": {
                "samples": 50, "p50_ms": 7.704, "p95_ms": 8.681,
                "p50_overhead_ms": 7.391, "p95_overhead_ms": 8.288,
            },
        },
        "trend": {
            "status": "stable", "unique_iterations": 3,
            "median": {"p50_ms": 7.599, "p95_ms": 8.680}, "reasons": ["stable"],
        },
        "not_measured": ["real vendor-device success rate", "production SLO"],
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_report_covers_both_core_capabilities_without_overclaiming(tmp_path: Path) -> None:
    report = build_core_capability_evaluation(
        runtime_report_path=_runtime_report(tmp_path / "runtime.json"),
    )
    assert report["schema"] == REPORT_SCHEMA
    assert report["capabilityA"]["status"] == (
        "guarded_prototype_not_generalization_qualified"
    )
    fixed = report["capabilityA"]["fixedForwardSample"]
    assert fixed["status"] == "ready_for_review"
    assert fixed["metrics"]["totalRequirements"] == 28
    assert fixed["metrics"]["averageL1ToL05Confidence"] == 91.43
    assert fixed["metrics"]["averageL05ToL0Confidence"] == 80.61
    assert report["capabilityA"]["reverseBootstrap"]["exactRoundTrips"] == 21

    runtime = report["capabilityB"]
    assert runtime["campaign"]["scenarioCount"] == 72
    control = runtime["comparison"]["control_effectiveness"]
    assert control["dshOnly"] == {"passed": 5, "total": 64, "rate": 7.8}
    assert control["dshPlusRuntime"] == {"passed": 64, "total": 64, "rate": 100.0}
    assert runtime["latency"]["dsh_plus_runtime"]["p50_ms"] == 7.704
    body = dict(report)
    digest = body.pop("reportDigest")
    assert digest == sha256_json(body)


def test_markdown_is_chinese_first_and_shows_function_performance_boundary(
    tmp_path: Path,
) -> None:
    report = build_core_capability_evaluation(
        runtime_report_path=_runtime_report(tmp_path / "runtime.json"),
    )
    rendered = render_core_capability_evaluation_markdown(report)
    assert rendered.index("## 中文") < rendered.index("## English")
    assert "[核心 A] L1 → L0.5 → L0" in rendered
    assert "[核心 B] Network Runtime" in rendered
    assert "不是 LLM 准确率" in rendered
    assert "7.8%（5/64）" in rendered
    assert "100.0%（64/64）" in rendered
    assert "不是生产成功概率" in rendered
    assert "scripts/netopyu-l0 core-eval-report" in rendered


def test_writer_and_cli_emit_dual_core_reports(tmp_path: Path) -> None:
    runtime_path = _runtime_report(tmp_path / "runtime.json")
    json_path = tmp_path / "evaluation.json"
    markdown_path = tmp_path / "evaluation.md"
    result = write_core_capability_evaluation(
        runtime_report_path=runtime_path,
        json_path=json_path,
        markdown_path=markdown_path,
    )
    assert result["ok"] is True
    assert json.loads(json_path.read_text(encoding="utf-8"))["schema"] == REPORT_SCHEMA
    assert markdown_path.read_text(encoding="utf-8").startswith(
        "# NetOpYu 双核心功能与性能评估"
    )

    cli_json = tmp_path / "cli.json"
    cli_markdown = tmp_path / "cli.md"
    completed = subprocess.run(
        [
            str(ROOT / "scripts/netopyu-l0"), "core-eval-report",
            "--runtime-report", str(runtime_path),
            "--json", str(cli_json), "--markdown", str(cli_markdown),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    assert cli_json.is_file()
    assert cli_markdown.is_file()
