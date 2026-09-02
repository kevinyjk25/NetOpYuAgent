"""Build digest-bound ES-P1-AI-External evidence and clustered statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from evaluation.ai_external_study import EVIDENCE_CLASS, PROVENANCE_SCHEMA
from evaluation.public_skill_simulation_report import (
    _expected_route,
    _failure_taxonomy,
    _group_metrics,
    _percent,
)
from evaluation.public_skill_translation import inspect_bound_public_execution_inputs
from network_runtime.contracts import sha256_json


SUMMARY_SCHEMA = "effect-runtime.io/es-p1-ai-external-evidence/v1"
MANIFEST_SCHEMA = "effect-runtime.io/es-p1-ai-external-report-manifest/v1"
BOOTSTRAP_SEED = 20260902
BOOTSTRAP_ITERATIONS = 10_000
ROUTES = ("l0_runtime", "l1_native_read", "safe_stop")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"AI-External report input must be an object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_digest(value: dict[str, Any], key: str, label: str) -> None:
    if value.get(key) != sha256_json({name: item for name, item in value.items() if name != key}):
        raise ValueError(f"{label} digest mismatch")


def _arm_completion(rows: Iterable[dict[str, Any]], arm: str) -> float:
    values = list(rows)
    if not values:
        return 0.0
    return sum(bool(row[arm]["score"]["passed"]) for row in values) / len(values)


def _cluster_statistics(
    rows: list[dict[str, Any]], cluster_by_case: dict[str, str], *,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[cluster_by_case[row["caseId"]]].append(row)
    names = sorted(grouped)
    deltas = {
        name: 100.0 * (
            _arm_completion(grouped[name], "treatment") - _arm_completion(grouped[name], "control")
        )
        for name in names
    }
    rng = random.Random(BOOTSTRAP_SEED)
    samples = [
        statistics.fmean(deltas[rng.choice(names)] for _ in names)
        for _ in range(iterations)
    ] if names else []
    samples.sort()
    low = samples[int(0.025 * (len(samples) - 1))] if samples else 0.0
    high = samples[int(0.975 * (len(samples) - 1))] if samples else 0.0
    return {
        "clusterCount": len(names),
        "macroDeltaPercentagePoints": round(statistics.fmean(deltas.values()), 2) if deltas else 0.0,
        "bootstrap95CiPercentagePoints": [round(low, 2), round(high, 2)],
        "bootstrapIterations": iterations, "bootstrapSeed": BOOTSTRAP_SEED,
        "perClusterDeltaPercentagePoints": {
            name: round(value, 2) for name, value in sorted(deltas.items())
        },
    }


def _translation_route_statistics(
    translations: dict[str, dict[str, Any]], expected_routes: dict[str, str],
) -> dict[str, Any]:
    confusion = {
        expected: {actual: 0 for actual in ROUTES}
        for expected in ROUTES
    }
    matches = 0
    unsafe_accepts = 0
    over_rejects = 0
    for case_id, expected in expected_routes.items():
        actual = str(translations[case_id]["route"])
        if expected not in confusion or actual not in confusion[expected]:
            raise ValueError(f"unsupported AI-External route: {expected} -> {actual}")
        confusion[expected][actual] += 1
        matches += actual == expected
        unsafe_accepts += actual == "l0_runtime" and expected != "l0_runtime"
        over_rejects += actual == "safe_stop" and expected != "safe_stop"
    total = len(expected_routes)
    runtime_expected = sum(expected == "l0_runtime" for expected in expected_routes.values())
    runtime_accepted = confusion["l0_runtime"]["l0_runtime"]
    return {
        "expectedRouteCounts": {
            route: sum(expected == route for expected in expected_routes.values())
            for route in ROUTES
        },
        "confusionMatrix": confusion,
        "expectedRouteMatches": matches,
        "expectedRouteMatchPercent": _percent(matches, total),
        "unsafeRuntimeAccepts": unsafe_accepts,
        "overSafeStops": over_rejects,
        "runtimeEligibleRecallPercent": _percent(runtime_accepted, runtime_expected),
    }


def _render_markdown(summary: dict[str, Any]) -> str:
    corpus = summary["corpus"]
    execution = summary["pairedExecution"]
    control = execution["metrics"]["control"]
    treatment = execution["metrics"]["treatment"]
    skill_ci = summary["generalization"]["skillClustered"]
    repo_ci = summary["generalization"]["repositoryClustered"]
    return "\n".join([
        "# ES-P1-AI-External 证据报告 / Evidence Report", "",
        "## 中文", "",
        "> 本报告来自角色隔离 GPT 外部模拟，不是真实独立人员、私有隐藏集、真实系统或正式 ES-P1 资格证据。ES-P1-Private-Human 保留但跳过。", "",
        "### 覆盖与证据等级", "",
        f"- {corpus['skillCount']} 个 Skill、{corpus['repositoryCount']} 个仓库、{corpus['caseCount']} 个 Case、{execution['repetitions']} 次重复。",
        f"- 当前等级：`{summary['generalization']['evidenceTier']}`；初步跨 Skill 说服力门槛为 ≥50 Skill / ≥15 仓库 / ≥8–10 领域。",
        f"- Skill 聚类完成率差值 95% bootstrap CI：{skill_ci['bootstrap95CiPercentagePoints']} 个百分点；仓库聚类：{repo_ci['bootstrap95CiPercentagePoints']}。",
        "- Case 数不是独立样本数；统计结论以 Skill/仓库聚类口径为主。", "",
        "### 结果", "",
        f"- 转译路由一致：{summary['translation']['expectedRouteMatches']}/{summary['translation']['total']}（{summary['translation']['expectedRouteMatchPercent']}%）；不安全 Runtime 误接纳 {summary['translation']['unsafeRuntimeAccepts']}；过度安全停止 {summary['translation']['overSafeStops']}。",
        f"- 应进入 Runtime 的召回率：{summary['translation']['runtimeEligibleRecallPercent']}%。安全门禁和可用性必须分开报告。",
        f"- Task Completion：Control {control['taskCompletionRatePercent']}%，Treatment {treatment['taskCompletionRatePercent']}%，差值 {execution['deltas']['taskCompletionPercentagePoints']} 个百分点。",
        f"- Oracle：Control {control['oraclePassRatePercent']}%，Treatment {treatment['oraclePassRatePercent']}%；参数精确：Control {control['parameterBindingPassRatePercent']}%，Treatment {treatment['parameterBindingPassRatePercent']}%。",
        f"- 不安全执行 Control/Treatment：{control['unsafeExecutions']}/{treatment['unsafeExecutions']}；错误提交：{control['falseCommits']}/{treatment['falseCommits']}。",
        f"- p50/p95：Control {control['latencyMs']['p50']}/{control['latencyMs']['p95']} ms；Treatment {treatment['latencyMs']['p50']}/{treatment['latencyMs']['p95']} ms。", "",
        "Control 为 DSH + 原始 L1 Skill + 9B 原生 Tool 编排；Treatment 的唯一功能增量是 Gold-blind L1→L0 转译门、L0 auto Runtime 及原生 Agent fallback。", "",
        "## English", "",
        "> Role-isolated GPT external simulation only—not independent-human, private-holdout, real-system, production-probability, or formal ES-P1 qualification evidence. ES-P1-Private-Human remains retained and skipped.", "",
        f"Coverage: {corpus['skillCount']} Skills, {corpus['repositoryCount']} repositories, {corpus['caseCount']} cases, and {execution['repetitions']} repetitions. Cases are not treated as independent generalization units; Skill- and repository-clustered intervals are primary.",
        f"Control completion is {control['taskCompletionRatePercent']}%; Treatment is {treatment['taskCompletionRatePercent']}%. Skill-clustered 95% bootstrap CI for the delta is {skill_ci['bootstrap95CiPercentagePoints']} percentage points.", "",
    ])


def _render_html(summary: dict[str, Any]) -> str:
    data = json.dumps(summary, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ES-P1-AI-External</title><style>
body{{margin:0;background:#09111f;color:#edf3ff;font:14px/1.5 system-ui,sans-serif}}main{{max-width:1180px;margin:auto;padding:30px}}.warn{{border:1px solid #f5b94c;background:#2b2112;padding:14px;border-radius:12px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:20px 0}}.card{{background:#121d31;border:1px solid #293956;border-radius:12px;padding:15px}}.num{{font-size:27px;font-weight:750}}.muted{{color:#9dafcf}}table{{width:100%;border-collapse:collapse;background:#121d31}}th,td{{padding:9px;border-bottom:1px solid #293956;text-align:left}}code{{color:#9ecbff}}
</style></head><body><main><h1>ES-P1-AI-External</h1><div class="warn">GPT 角色隔离模拟；不是 Private-Human 或正式资格证据。<br>Role-isolated GPT simulation; not Private-Human qualification.</div><div id="cards" class="grid"></div><h2>Clustered generalization</h2><table><thead><tr><th>Unit</th><th>Clusters</th><th>Macro delta</th><th>95% bootstrap CI</th></tr></thead><tbody id="clusters"></tbody></table><h2>Per Skill</h2><table><thead><tr><th>Skill</th><th>Delta (pp)</th></tr></thead><tbody id="skills"></tbody></table><p class="muted">Digest <code id="digest"></code></p></main><script id="data" type="application/json">{data}</script><script>
const D=JSON.parse(document.getElementById('data').textContent),E=D.pairedExecution,C=E.metrics.control,T=E.metrics.treatment;const cards=[['Skills',D.corpus.skillCount],['Repositories',D.corpus.repositoryCount],['Cases',D.corpus.caseCount],['Control completion',C.taskCompletionRatePercent+'%'],['Treatment completion',T.taskCompletionRatePercent+'%'],['Unsafe C / T',C.unsafeExecutions+' / '+T.unsafeExecutions],['Route agreement',D.translation.expectedRouteMatchPercent+'%'],['Over-safe-stops',D.translation.overSafeStops],['Runtime recall',D.translation.runtimeEligibleRecallPercent+'%']];for(const [k,v] of cards){{const n=document.createElement('div');n.className='card';n.innerHTML='<div class="muted">'+k+'</div><div class="num">'+v+'</div>';document.getElementById('cards').append(n)}}for(const [name,x] of Object.entries({{Skill:D.generalization.skillClustered,Repository:D.generalization.repositoryClustered}})){{const tr=document.createElement('tr');tr.innerHTML='<td>'+name+'</td><td>'+x.clusterCount+'</td><td>'+x.macroDeltaPercentagePoints+'</td><td>'+x.bootstrap95CiPercentagePoints.join(' … ')+'</td>';document.getElementById('clusters').append(tr)}}for(const [name,v] of Object.entries(D.generalization.skillClustered.perClusterDeltaPercentagePoints)){{const tr=document.createElement('tr');tr.innerHTML='<td>'+name+'</td><td>'+v+'</td>';document.getElementById('skills').append(tr)}}document.getElementById('digest').textContent=D.reportDigest;
</script></body></html>"""


def build_ai_external_evidence_report(
    study_root: str | Path, result_root: str | Path, output_root: str | Path,
    *, require_complete: bool = True,
) -> dict[str, Any]:
    study = Path(study_root).expanduser().resolve()
    provenance = _json(study / "ai-external-provenance.json")
    _verify_digest(provenance, "provenanceDigest", "AI-External provenance")
    if any((
        provenance.get("apiVersion") != PROVENANCE_SCHEMA,
        provenance.get("evidenceClass") != EVIDENCE_CLASS,
        provenance.get("humanIndependent") is not False,
        provenance.get("privateHumanStage") != "skipped_retained_open",
        provenance.get("externalGptRoleSeparation") is not True,
        provenance.get("officialEsP1QualificationEligible") is not False,
    )):
        raise ValueError("AI-External provenance authority boundary mismatch")
    bound = inspect_bound_public_execution_inputs(study / "bound-study")
    translation_report = _json(study / "translation/report.json")
    _verify_digest(translation_report, "reportDigest", "AI-External translation report")
    dsh = _json(Path(result_root).expanduser().resolve() / "report.json")
    _verify_digest(dsh, "reportDigest", "AI-External DSH report")
    if require_complete and not dsh.get("protocolComplete"):
        raise ValueError("AI-External paired execution is not complete")
    if any((
        dsh.get("sourceBoundStudyDigest") != bound["workspaceDigest"],
        dsh.get("modelArtifactDigest") != bound["modelArtifactDigest"],
        dsh.get("goldLoadedAfterAgentRuns") is not True,
        dsh.get("officialEsP1QualificationEligible") is not False,
    )):
        raise ValueError("AI-External DSH binding or authority mismatch")
    paired = study / "bound-study/study"
    cases = _jsonl(paired / "agent/cases.jsonl")
    scoring = {item["caseId"]: item for item in _jsonl(paired / "scoring/gold.jsonl")}
    translations = {
        item["caseId"]: item for item in _jsonl(study / "bound-study/translation/cases.jsonl")
    }
    rows = list(dsh.get("rows") or [])
    expected_count = int(dsh["caseCount"]) * int(dsh["repetitions"])
    if len(rows) != expected_count or len({(x["caseId"], x["repetition"]) for x in rows}) != expected_count:
        raise ValueError("AI-External DSH row coverage mismatch")
    expected_routes = {
        case["caseId"]: _expected_route(scoring[case["caseId"]]) for case in cases
    }
    route_statistics = _translation_route_statistics(translations, expected_routes)
    challenge_by_case = {case["caseId"]: case["challenge"] for case in cases}
    skill_by_case = {
        case["caseId"]: (
            f"{case['skill']['repository']}:{case['skill']['sourcePath']}"
            f"@{case['skill']['commitSha'][:12]}"
        )
        for case in cases
    }
    repository_by_case = {case["caseId"]: case["skill"]["repository"] for case in cases}
    enriched = [{
        **row, "challenge": challenge_by_case[row["caseId"]],
        "skill": skill_by_case[row["caseId"]],
        "repository": repository_by_case[row["caseId"]],
    } for row in rows]
    outcomes: Counter[str] = Counter()
    for row in rows:
        control_pass = bool(row["control"]["score"]["passed"])
        treatment_pass = bool(row["treatment"]["score"]["passed"])
        outcomes[
            "bothPass" if control_pass and treatment_pass else
            "treatmentWins" if treatment_pass else
            "controlWins" if control_pass else "bothFail"
        ] += 1
    control = dsh["metrics"]["control"]
    treatment = dsh["metrics"]["treatment"]
    skill_count = len(set(skill_by_case.values()))
    repository_count = len(set(repository_by_case.values()))
    report_body = {
        "apiVersion": SUMMARY_SCHEMA, "generatedAt": datetime.now(timezone.utc).isoformat(),
        "status": "ai_external_protocol_complete" if dsh.get("protocolComplete") else "ai_external_smoke_only",
        "evidenceClass": EVIDENCE_CLASS, "humanIndependent": False,
        "privateHumanStage": "skipped_retained_open",
        "officialEsP1QualificationEligible": False,
        "corpus": {
            "skillCount": skill_count, "repositoryCount": repository_count,
            "caseCount": len(cases),
            "challengeCounts": dict(sorted(Counter(case["challenge"] for case in cases).items())),
            "languageCounts": dict(sorted(Counter(case["language"] for case in cases).items())),
            "casesPerSkillMean": round(len(cases) / skill_count, 2),
        },
        "generalization": {
            "evidenceTier": "mechanism_feasibility_15_skills",
            "minimumPersuasiveTarget": {
                "skills": 50, "repositories": 15, "domains": "8-10", "cases": "600-800",
            },
            "skillClustered": _cluster_statistics(rows, skill_by_case),
            "repositoryClustered": _cluster_statistics(rows, repository_by_case),
        },
        "translation": {
            "model": translation_report["model"], "routeCounts": translation_report["routeCounts"],
            "total": len(cases), **route_statistics,
            "meanConfidence": round(statistics.fmean(
                float(item["confidence"]) for item in translations.values()
            ), 4),
            "reportDigest": translation_report["reportDigest"],
        },
        "pairedExecution": {
            "model": dsh["model"], "modelArtifactDigest": dsh["modelArtifactDigest"],
            "repetitions": dsh["repetitions"], "workers": dsh.get("workers", 1),
            "pairedObservationCount": len(rows), "armExecutionCount": len(rows) * 2,
            "protocolComplete": dsh["protocolComplete"], "metrics": dsh["metrics"],
            "deltas": {
                "taskCompletionPercentagePoints": round(
                    treatment["taskCompletionRatePercent"] - control["taskCompletionRatePercent"], 2,
                ),
                "oraclePercentagePoints": round(
                    treatment["oraclePassRatePercent"] - control["oraclePassRatePercent"], 2,
                ),
                "parameterBindingPercentagePoints": round(
                    treatment["parameterBindingPassRatePercent"]
                    - control["parameterBindingPassRatePercent"], 2,
                ),
            },
            "pairedOutcomes": {name: outcomes[name] for name in (
                "treatmentWins", "controlWins", "bothPass", "bothFail",
            )},
            "failureTaxonomy": {
                "control": _failure_taxonomy(rows, "control"),
                "treatment": _failure_taxonomy(rows, "treatment"),
            },
            "byChallenge": _group_metrics(enriched, "challenge"),
            "bySkill": _group_metrics(enriched, "skill"),
            "byRepository": _group_metrics(enriched, "repository"),
            "sourceReportDigest": dsh["reportDigest"],
        },
        "bindings": {
            "provenanceDigest": provenance["provenanceDigest"],
            "boundStudyDigest": bound["workspaceDigest"],
            "translationReportDigest": translation_report["reportDigest"],
            "dshReportDigest": dsh["reportDigest"],
        },
        "claimBoundary": (
            "Role-isolated GPT external simulation only. Not independent-human private holdout, "
            "production probability, real-system evidence, or formal ES-P1 qualification."
        ),
    }
    summary = {**report_body, "reportDigest": sha256_json(report_body)}
    output = Path(output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("AI-External report output must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (output / "report.md").write_text(_render_markdown(summary), encoding="utf-8")
    (output / "report.html").write_text(_render_html(summary), encoding="utf-8")
    manifest_body = {
        "apiVersion": MANIFEST_SCHEMA, "evidenceClass": EVIDENCE_CLASS,
        "officialEsP1QualificationEligible": False, "summaryDigest": summary["reportDigest"],
        "files": {
            name: _file_digest(output / name) for name in ("summary.json", "report.md", "report.html")
        },
    }
    manifest = {**manifest_body, "manifestDigest": sha256_json(manifest_body)}
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return summary


def inspect_ai_external_evidence_report(root_path: str | Path) -> dict[str, Any]:
    root = Path(root_path).expanduser().resolve()
    expected = {"summary.json", "report.md", "report.html", "manifest.json"}
    if {path.name for path in root.iterdir() if path.is_file()} != expected:
        raise ValueError("AI-External report file set mismatch")
    manifest = _json(root / "manifest.json")
    _verify_digest(manifest, "manifestDigest", "AI-External report manifest")
    if manifest.get("evidenceClass") != EVIDENCE_CLASS or manifest.get(
        "officialEsP1QualificationEligible"
    ) is not False:
        raise ValueError("AI-External report authority boundary mismatch")
    for name, digest in manifest["files"].items():
        if _file_digest(root / name) != digest:
            raise ValueError(f"AI-External report file digest mismatch: {name}")
    summary = _json(root / "summary.json")
    _verify_digest(summary, "reportDigest", "AI-External report summary")
    if summary["reportDigest"] != manifest["summaryDigest"]:
        raise ValueError("AI-External report summary binding mismatch")
    return {
        "status": summary["status"], "skillCount": summary["corpus"]["skillCount"],
        "repositoryCount": summary["corpus"]["repositoryCount"],
        "caseCount": summary["corpus"]["caseCount"],
        "reportDigest": summary["reportDigest"],
        "officialEsP1QualificationEligible": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("study_root")
    build.add_argument("result_root")
    build.add_argument("--output-root", required=True)
    build.add_argument("--allow-smoke", action="store_true")
    inspect = commands.add_parser("inspect")
    inspect.add_argument("root")
    args = parser.parse_args(argv)
    if args.command == "build":
        result = build_ai_external_evidence_report(
            args.study_root, args.result_root, args.output_root,
            require_complete=not args.allow_smoke,
        )
        brief = {
            "status": result["status"], "skillCount": result["corpus"]["skillCount"],
            "repositoryCount": result["corpus"]["repositoryCount"],
            "caseCount": result["corpus"]["caseCount"],
            "reportDigest": result["reportDigest"],
            "officialEsP1QualificationEligible": False,
        }
    else:
        brief = inspect_ai_external_evidence_report(args.root)
    print(json.dumps(brief, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_ai_external_evidence_report", "inspect_ai_external_evidence_report"]
