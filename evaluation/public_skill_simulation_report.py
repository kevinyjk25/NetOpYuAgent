"""Build a digest-bound ES-P1-Wild simulation evidence report.

The report makes the virtual role separation, translation route, paired DSH
outcome, and claim boundary inspectable.  It can never upgrade simulation-only
inputs into independent-human ES-P1 qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from evaluation.public_skill_translation import inspect_bound_public_execution_inputs
from network_runtime.contracts import sha256_json


SUMMARY_SCHEMA = "effect-runtime.io/es-p1-wild-simulation-evidence/v1"
MANIFEST_SCHEMA = "effect-runtime.io/es-p1-wild-simulation-report-manifest/v1"
EVIDENCE_CLASS = "simulation_only_not_independent_human_evidence"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"ES-P1-Wild report input must be an object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_object_digest(value: dict[str, Any], key: str, label: str) -> None:
    body = {name: item for name, item in value.items() if name != key}
    if value.get(key) != sha256_json(body):
        raise ValueError(f"{label} digest mismatch")


def _percent(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0


def _expected_route(scoring: dict[str, Any]) -> str:
    disposition = scoring["gold"]["expectedDisposition"]
    if disposition == "read_only":
        return "l1_native_read"
    if disposition == "proposal":
        return "l0_runtime"
    if disposition in {"safe_stop_clarify", "safe_stop_reject"}:
        return "safe_stop"
    raise ValueError(f"unsupported simulated Gold disposition: {disposition}")


def _slice_metrics(rows: Iterable[dict[str, Any]], arm: str) -> dict[str, Any]:
    values = [row[arm] for row in rows]
    total = len(values)
    passed = sum(bool(item["score"]["passed"]) for item in values)
    return {
        "total": total,
        "passed": passed,
        "taskCompletionRatePercent": _percent(passed, total),
        "oraclePassRatePercent": _percent(
            sum(bool(item["score"]["oraclePassed"]) for item in values), total,
        ),
        "parameterBindingPassRatePercent": _percent(
            sum(bool(item["score"]["parameterBindingPassed"]) for item in values), total,
        ),
        "protocolPassRatePercent": _percent(
            sum(bool(item["score"]["protocolPassed"]) for item in values), total,
        ),
        "unsafeExecutions": sum(bool(item["score"]["unsafeExecution"]) for item in values),
        "falseCommits": sum(bool(item["score"]["falseCommit"]) for item in values),
        "runtimeAutoInvocations": sum(
            bool(item["observation"].get("runtime_auto_invoked")) for item in values
        ),
    }


def _group_metrics(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    return {
        name: {
            "control": _slice_metrics(items, "control"),
            "treatment": _slice_metrics(items, "treatment"),
        }
        for name, items in sorted(grouped.items())
    }


_SCORE_GATES = (
    "protocolPassed", "requiredCapabilitiesPassed", "forbiddenCapabilitiesPassed",
    "parameterBindingPassed", "effectBudgetPassed", "dispositionPassed", "oraclePassed",
)


def _failure_taxonomy(rows: Iterable[dict[str, Any]], arm: str) -> dict[str, int]:
    failures: Counter[str] = Counter()
    for row in rows:
        score = row[arm]["score"]
        observation = row[arm]["observation"]
        for gate in _SCORE_GATES:
            if not bool(score.get(gate)):
                failures[gate] += 1
        if bool(score.get("unsafeExecution")):
            failures["unsafeExecution"] += 1
        if bool(score.get("falseCommit")):
            failures["falseCommit"] += 1
        if bool(observation.get("process_timed_out")):
            failures["processTimedOut"] += 1
        if not bool(observation.get("session_completed")):
            failures["sessionIncomplete"] += 1
    return dict(sorted(failures.items()))


def _render_markdown(summary: dict[str, Any]) -> str:
    execution = summary["pairedExecution"]
    control = execution["metrics"]["control"]
    treatment = execution["metrics"]["treatment"]
    translation = summary["translation"]
    lines = [
        "# ES-P1-Wild 模拟独立测试报告 / Simulated Independent Test Report",
        "",
        "## 中文",
        "",
        "> 证据边界：这是角色隔离模拟，不是真实独立人员、私有隐藏集、真实系统、生产成功概率或正式 ES-P1 资格证据。",
        "",
        "### 结论",
        "",
        f"- 虚拟 Case Author：`{summary['roleSimulation']['caseAuthorId']}`；虚拟 Gold Author：`{summary['roleSimulation']['goldAuthorId']}`；两者只是隔离协议输入，不是两名真人。",
        f"- 公开 Skill：{summary['corpus']['skillCount']}；案例：{summary['corpus']['caseCount']}；重复：{execution['repetitions']}；成对观察：{execution['pairedObservationCount']}；实验臂执行：{execution['armExecutionCount']}。",
        f"- 9B 转译路由：L0 Runtime {translation['routeCounts'].get('l0_runtime', 0)}，原生只读 {translation['routeCounts'].get('l1_native_read', 0)}，安全停止 {translation['routeCounts'].get('safe_stop', 0)}。",
        f"- 后验路由一致：{translation['expectedRouteMatches']}/{translation['total']}（{translation['expectedRouteMatchPercent']}%）；不安全 Runtime 误接纳：{translation['unsafeRuntimeAccepts']}。",
        f"- Task Completion：Control {control['taskCompletionRatePercent']}%，Treatment {treatment['taskCompletionRatePercent']}%，差值 {execution['deltas']['taskCompletionPercentagePoints']} 个百分点。",
        f"- Oracle：Control {control['oraclePassRatePercent']}%，Treatment {treatment['oraclePassRatePercent']}%；参数精确：Control {control['parameterBindingPassRatePercent']}%，Treatment {treatment['parameterBindingPassRatePercent']}%。",
        f"- 不安全执行：Control {control['unsafeExecutions']}，Treatment {treatment['unsafeExecutions']}；错误提交：Control {control['falseCommits']}，Treatment {treatment['falseCommits']}。",
        f"- p50/p95：Control {control['latencyMs']['p50']}/{control['latencyMs']['p95']} ms；Treatment {treatment['latencyMs']['p50']}/{treatment['latencyMs']['p95']} ms。",
        f"- 配对结果：Treatment 胜 {execution['pairedOutcomes']['treatmentWins']}，Control 胜 {execution['pairedOutcomes']['controlWins']}，均通过 {execution['pairedOutcomes']['bothPass']}，均失败 {execution['pairedOutcomes']['bothFail']}。",
        f"- 失败门分类：Control {execution['failureTaxonomy']['control']}；Treatment {execution['failureTaxonomy']['treatment']}。",
        "",
        "### 解释",
        "",
        "Control 是 DSH + 原始 L1 Skill + LLM 原生 Tool 编排；Treatment 使用相同模型、Skill、任务、Tool Catalog、fixture、审批和故障，仅增加 Gold-blind 转译门与 L0 auto Runtime。可信写计划由 Runtime 收口；不可信写安全停止；只读保持原生 L1。",
        "",
        "本报告中的 Gold 与 Oracle 由隔离的虚拟角色生成，并在两臂结束后才加载。它们可验证协议、代码和评测机械链路，但不能替代真实外部人员的独立判断。",
        "",
        "## English",
        "",
        "> Evidence boundary: this is a role-separated simulation, not independent-human, private-holdout, real-system, production-probability, or formal ES-P1 qualification evidence.",
        "",
        "### Result",
        "",
        f"Virtual roles are `{summary['roleSimulation']['caseAuthorId']}` and `{summary['roleSimulation']['goldAuthorId']}`. They are isolated protocol inputs, not two human reviewers.",
        f"The study covers {summary['corpus']['skillCount']} public Skills, {summary['corpus']['caseCount']} cases, {execution['repetitions']} repetitions, {execution['pairedObservationCount']} paired observations, and {execution['armExecutionCount']} arm executions.",
        f"Control task completion is {control['taskCompletionRatePercent']}% and Treatment is {treatment['taskCompletionRatePercent']}%. Unsafe executions are {control['unsafeExecutions']} and {treatment['unsafeExecutions']}; false commits are {control['falseCommits']} and {treatment['falseCommits']}.",
        f"Post-run translation-route agreement with simulated Gold is {translation['expectedRouteMatches']}/{translation['total']} ({translation['expectedRouteMatchPercent']}%), with {translation['unsafeRuntimeAccepts']} unsafe Runtime accepts.",
        "",
        "Control is DSH plus the original L1 Skill and LLM-native Tool orchestration. Treatment receives identical model, Skill, task, Catalog, fixture, approval, and fault inputs; the only functional addition is the Gold-blind translation gate and L0 auto Runtime. Qualified writes are controlled by Runtime, unqualified writes safe-stop, and reads retain native L1 fallback.",
        "",
        "The simulated Gold and Oracles are loaded only after both arms terminate. They validate mechanics and prototype behavior, but do not replace independent external judgment.",
        "",
    ]
    return "\n".join(lines)


def _render_html(summary: dict[str, Any]) -> str:
    safe_data = json.dumps(summary, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    return f"""<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>ES-P1-Wild Simulation Evidence</title><style>
:root{{--bg:#0b1020;--panel:#121a2f;--muted:#9aa8c7;--text:#eef3ff;--blue:#65a7ff;--green:#62d49c;--red:#ff7f8b;--amber:#ffc66d;--line:#283552}}*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(135deg,#08101e,#111932);color:var(--text);font:14px/1.5 ui-sans-serif,system-ui,sans-serif}}main{{max-width:1320px;margin:auto;padding:32px}}h1{{font-size:30px;margin:0 0 8px}}h2{{margin-top:34px}}.boundary{{border:1px solid var(--amber);background:#2a2113;padding:14px;border-radius:12px;color:#ffe0a0}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:18px 0}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px}}.num{{font-size:28px;font-weight:750}}.muted{{color:var(--muted)}}table{{width:100%;border-collapse:collapse;background:var(--panel);border-radius:12px;overflow:hidden}}th,td{{padding:10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{position:sticky;top:0;background:#19233c}}.pass{{color:var(--green)}}.fail{{color:var(--red)}}.win{{background:#123126}}.loss{{background:#381d25}}select,input{{background:#121a2f;color:var(--text);border:1px solid var(--line);padding:9px;border-radius:8px;margin:0 8px 12px 0}}code{{color:#b8d7ff}}details{{max-width:460px}}summary{{cursor:pointer;color:var(--blue)}}@media(max-width:800px){{main{{padding:16px}}table{{font-size:12px}}}}
</style></head><body><main><h1>ES-P1-Wild 模拟独立测试</h1><p class=\"muted\">Simulated independent-role evidence · digest-bound · local DSH</p><div class=\"boundary\">这是角色隔离模拟，不是真实独立人员资格、生产成功概率或真实系统证据。<br>This is role-separated simulation evidence, not independent-human qualification or production probability.</div><p id=\"runmeta\" class=\"muted\"></p><div id=\"cards\" class=\"grid\"></div><h2>重复轮次 / Repetition slices</h2><div id=\"repetitions\"></div><h2>路由分层 / Route slices</h2><div id=\"routes\"></div><h2>挑战分层 / Challenge slices</h2><div id=\"challenges\"></div><h2>Skill 分层 / Skill slices</h2><div id=\"skills\"></div><h2>案例审计 / Case audit</h2><input id=\"search\" placeholder=\"case / skill / repo\"><select id=\"filter\"><option value=\"all\">全部 / All</option><option value=\"treatment-win\">Treatment 胜</option><option value=\"treatment-fail\">Treatment 失败</option><option value=\"route-mismatch\">路由不一致</option></select><table><thead><tr><th>Case</th><th>Skill / challenge</th><th>Route</th><th>Control</th><th>Treatment</th><th>Evidence</th></tr></thead><tbody id=\"cases\"></tbody></table><p class=\"muted\">Report digest: <code id=\"digest\"></code></p></main><script id=\"evidence\" type=\"application/json\">{safe_data}</script><script>
const D=JSON.parse(document.getElementById('evidence').textContent), E=D.pairedExecution, C=E.metrics.control, T=E.metrics.treatment;
const el=(t,c,x)=>{{const n=document.createElement(t);if(c)n.className=c;if(x!==undefined)n.textContent=String(x);return n}};
document.getElementById('runmeta').textContent='Model: '+E.model+' · '+E.invocationProfile+' · workers '+E.workers+' · virtual roles '+D.roleSimulation.caseAuthorId+' / '+D.roleSimulation.goldAuthorId;
const cards=[['Skills',D.corpus.skillCount],['Cases',D.corpus.caseCount],['Paired observations',E.pairedObservationCount],['Arm executions',E.armExecutionCount],['Control completion',C.taskCompletionRatePercent+'%'],['Treatment completion',T.taskCompletionRatePercent+'%'],['Treatment wins',E.pairedOutcomes.treatmentWins],['Unsafe C / T',C.unsafeExecutions+' / '+T.unsafeExecutions],['Route agreement',D.translation.expectedRouteMatchPercent+'%']];
for(const [k,v] of cards){{const c=el('div','card');c.append(el('div','muted',k),el('div','num',v));document.getElementById('cards').append(c)}}
function sliceTable(data,label){{const t=el('table');const h=el('thead');h.innerHTML='<tr><th>'+label+'</th><th>Control pass</th><th>Treatment pass</th><th>Parameter C/T</th><th>Unsafe C/T</th></tr>';t.append(h);const b=el('tbody');for(const [name,x] of Object.entries(data)){{const tr=el('tr');for(const v of [name,x.control.passed+'/'+x.control.total,x.treatment.passed+'/'+x.treatment.total,x.control.parameterBindingPassRatePercent+'% / '+x.treatment.parameterBindingPassRatePercent+'%',x.control.unsafeExecutions+' / '+x.treatment.unsafeExecutions])tr.append(el('td','',v));b.append(tr)}}t.append(b);return t}}document.getElementById('repetitions').append(sliceTable(E.byRepetition,'Repetition'));document.getElementById('routes').append(sliceTable(E.byRoute,'Route'));document.getElementById('challenges').append(sliceTable(E.byChallenge,'Challenge'));document.getElementById('skills').append(sliceTable(E.bySkill,'Skill'));
const body=document.getElementById('cases'), search=document.getElementById('search'), filter=document.getElementById('filter');
function render(){{body.textContent='';const q=search.value.toLowerCase();for(const x of D.cases){{const cp=x.controlPasses, tp=x.treatmentPasses, n=x.repetitions;const tag=tp>cp?'treatment-win':tp<n?'treatment-fail':'other';if(filter.value!=='all'&&filter.value!==tag&&!(filter.value==='route-mismatch'&&!x.routeMatchesExpected))continue;const hay=(x.caseId+' '+x.skill+' '+x.repository).toLowerCase();if(q&&!hay.includes(q))continue;const tr=el('tr',tp>cp?'win':cp>tp?'loss':'');const c1=el('td');c1.append(el('code','',x.caseId));const c2=el('td');c2.append(el('div','',x.skill),el('div','muted',x.challenge));const c3=el('td');c3.append(el('div',x.routeMatchesExpected?'pass':'fail',x.route),el('div','muted','expected '+x.expectedRoute));const c4=el('td',cp===n?'pass':'fail',cp+'/'+n+' · '+x.controlTerminals.join(', '));const c5=el('td',tp===n?'pass':'fail',tp+'/'+n+' · '+x.treatmentTerminals.join(', '));const c6=el('td');const d=el('details');d.append(el('summary','','checks / digests'),el('div','muted','auto runtime: '+x.runtimeAutoInvocations),el('div','muted','control failures: '+JSON.stringify(x.controlFailureChecks)),el('div','muted','treatment failures: '+JSON.stringify(x.treatmentFailureChecks)),el('div','muted','translation: '+x.translationFailures.join(', ')),el('code','',x.l0Digest||'fallback'));c6.append(d);for(const c of [c1,c2,c3,c4,c5,c6])tr.append(c);body.append(tr)}}}}
search.addEventListener('input',render);filter.addEventListener('change',render);document.getElementById('digest').textContent=D.reportDigest;render();
</script></body></html>"""


def build_simulation_evidence_report(
    simulation_root: str | Path, result_root: str | Path, output_root: str | Path,
    *, require_complete: bool = True,
) -> dict[str, Any]:
    simulation = Path(simulation_root).expanduser().resolve()
    result_path = Path(result_root).expanduser().resolve() / "report.json"
    provenance = _json(simulation / "simulation-provenance.json")
    _verify_object_digest(provenance, "provenanceDigest", "simulation provenance")
    if (
        provenance.get("evidenceClass") != EVIDENCE_CLASS
        or provenance.get("humanIndependent") is not False
        or provenance.get("virtualRoleSeparation") is not True
        or provenance.get("officialEsP1QualificationEligible") is not False
    ):
        raise ValueError("simulation provenance authority boundary is invalid")
    bound = simulation / "bound-study"
    bound_inspection = inspect_bound_public_execution_inputs(bound)
    translation_report = _json(simulation / "translation/report.json")
    _verify_object_digest(translation_report, "reportDigest", "translation report")
    dsh = _json(result_path)
    _verify_object_digest(dsh, "reportDigest", "DSH paired report")
    if require_complete and not dsh.get("protocolComplete"):
        raise ValueError("ES-P1-Wild paired execution is not protocol-complete")
    if (
        dsh.get("sourceBoundStudyDigest") != bound_inspection["workspaceDigest"]
        or dsh.get("modelArtifactDigest") != bound_inspection["modelArtifactDigest"]
        or dsh.get("goldLoadedAfterAgentRuns") is not True
        or dsh.get("officialEsP1QualificationEligible") is not False
    ):
        raise ValueError("DSH paired report binding or authority boundary is invalid")
    cases = _jsonl(bound / "study/agent/cases.jsonl")
    scoring = {item["caseId"]: item for item in _jsonl(bound / "study/scoring/gold.jsonl")}
    translations = {item["caseId"]: item for item in _jsonl(bound / "translation/cases.jsonl")}
    rows = list(dsh.get("rows") or [])
    expected_rows = int(dsh["caseCount"]) * int(dsh["repetitions"])
    if len(rows) != expected_rows or len({(x["caseId"], x["repetition"]) for x in rows}) != expected_rows:
        raise ValueError("DSH paired report row coverage is incomplete or duplicated")
    executed_case_ids = {row["caseId"] for row in rows}
    if len(executed_case_ids) != int(dsh["caseCount"]):
        raise ValueError("DSH paired report case coverage mismatch")
    expected_routes = {case["caseId"]: _expected_route(scoring[case["caseId"]]) for case in cases}
    route_matches = sum(
        translations[case_id]["route"] == expected for case_id, expected in expected_routes.items()
    )
    unsafe_runtime_accepts = sum(
        translations[case_id]["route"] == "l0_runtime" and expected != "l0_runtime"
        for case_id, expected in expected_routes.items()
    )
    conservative_read_fallbacks = sum(
        translations[case_id]["route"] == "safe_stop" and expected == "l1_native_read"
        for case_id, expected in expected_routes.items()
    )
    challenge_by_case = {case["caseId"]: case["challenge"] for case in cases}
    skill_by_case = {case["caseId"]: case["skill"]["name"] for case in cases}
    enriched_rows = [{
        **row,
        "challenge": challenge_by_case[row["caseId"]],
        "skill": skill_by_case[row["caseId"]],
    } for row in rows]
    pair_outcomes = Counter()
    for row in rows:
        control_pass = bool(row["control"]["score"]["passed"])
        treatment_pass = bool(row["treatment"]["score"]["passed"])
        pair_outcomes[
            "bothPass" if control_pass and treatment_pass else
            "treatmentWins" if treatment_pass else
            "controlWins" if control_pass else "bothFail"
        ] += 1
    case_rows: list[dict[str, Any]] = []
    for case in cases:
        case_id = case["caseId"]
        if case_id not in executed_case_ids:
            continue
        observations = [row for row in rows if row["caseId"] == case_id]
        translation = translations[case_id]
        case_rows.append({
            "caseId": case_id, "assignmentId": case["assignmentId"],
            "skill": case["skill"]["name"], "repository": case["skill"]["repository"],
            "challenge": case["challenge"], "language": case["language"],
            "route": translation["route"], "expectedRoute": expected_routes[case_id],
            "routeMatchesExpected": translation["route"] == expected_routes[case_id],
            "translationConfidence": translation["confidence"],
            "translationFailures": translation["failures"],
            "l0Digest": translation.get("l0Digest"),
            "controlPasses": sum(row["control"]["score"]["passed"] for row in observations),
            "treatmentPasses": sum(row["treatment"]["score"]["passed"] for row in observations),
            "repetitions": len(observations),
            "controlTerminals": [row["control"]["observation"]["terminal"] for row in observations],
            "treatmentTerminals": [row["treatment"]["observation"]["terminal"] for row in observations],
            "runtimeAutoInvocations": sum(
                bool(row["treatment"]["observation"].get("runtime_auto_invoked"))
                for row in observations
            ),
            "controlFailureChecks": _failure_taxonomy(observations, "control"),
            "treatmentFailureChecks": _failure_taxonomy(observations, "treatment"),
        })
    control = dsh["metrics"]["control"]
    treatment = dsh["metrics"]["treatment"]
    repair_counts = [
        int(item.get("telemetry", {}).get("semanticRepairCount") or 0)
        for item in translations.values()
    ]
    report_body: dict[str, Any] = {
        "apiVersion": SUMMARY_SCHEMA,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "status": "simulation_protocol_complete" if dsh.get("protocolComplete") else "simulation_smoke_only",
        "evidenceClass": EVIDENCE_CLASS,
        "humanIndependent": False,
        "virtualRoleSeparation": True,
        "officialEsP1QualificationEligible": False,
        "roleSimulation": {
            "caseAuthorId": provenance["caseAuthorId"],
            "goldAuthorId": provenance["goldAuthorId"],
            "strictRoleDeclarationsAreSimulatedProtocolInputs": provenance[
                "strictRoleDeclarationsAreSimulatedProtocolInputs"
            ],
        },
        "corpus": {
            "skillCount": len({case["packageId"] for case in cases}),
            "caseCount": len(cases),
            "challengeCounts": dict(sorted(Counter(case["challenge"] for case in cases).items())),
            "languageCounts": dict(sorted(Counter(case["language"] for case in cases).items())),
        },
        "translation": {
            "model": translation_report["model"],
            "goldReadByTranslator": translation_report["goldReadByTranslator"],
            "routeCounts": translation_report["routeCounts"],
            "total": len(cases),
            "rawProtocolValid": sum(item["rawProtocolValid"] for item in translations.values()),
            "expectedRouteMatches": route_matches,
            "expectedRouteMatchPercent": _percent(route_matches, len(cases)),
            "unsafeRuntimeAccepts": unsafe_runtime_accepts,
            "conservativeReadFallbacks": conservative_read_fallbacks,
            "semanticRepairCases": sum(count > 0 for count in repair_counts),
            "semanticRepairCalls": sum(repair_counts),
            "meanConfidence": round(statistics.fmean(
                float(item["confidence"]) for item in translations.values()
            ), 4),
            "reportDigest": translation_report["reportDigest"],
        },
        "pairedExecution": {
            "model": dsh["model"], "modelArtifactDigest": dsh["modelArtifactDigest"],
            "invocationProfile": dsh.get("invocationProfile"), "workers": dsh.get("workers", 1),
            "repetitions": dsh["repetitions"],
            "executedCaseCount": dsh["caseCount"],
            "pairedObservationCount": len(rows),
            "armExecutionCount": len(rows) * 2,
            "protocolComplete": dsh["protocolComplete"],
            "metrics": dsh["metrics"],
            "deltas": {
                "taskCompletionPercentagePoints": round(
                    treatment["taskCompletionRatePercent"] - control["taskCompletionRatePercent"], 2
                ),
                "oraclePercentagePoints": round(
                    treatment["oraclePassRatePercent"] - control["oraclePassRatePercent"], 2
                ),
                "parameterBindingPercentagePoints": round(
                    treatment["parameterBindingPassRatePercent"]
                    - control["parameterBindingPassRatePercent"], 2
                ),
                "p50LatencyMs": round(
                    treatment["latencyMs"]["p50"] - control["latencyMs"]["p50"], 3
                ),
                "p95LatencyMs": round(
                    treatment["latencyMs"]["p95"] - control["latencyMs"]["p95"], 3
                ),
            },
            "pairedOutcomes": {name: pair_outcomes[name] for name in (
                "treatmentWins", "controlWins", "bothPass", "bothFail",
            )},
            "failureTaxonomy": {
                "control": _failure_taxonomy(rows, "control"),
                "treatment": _failure_taxonomy(rows, "treatment"),
            },
            "byRepetition": _group_metrics(rows, "repetition"),
            "byRoute": _group_metrics(rows, "route"),
            "byChallenge": _group_metrics(enriched_rows, "challenge"),
            "bySkill": _group_metrics(enriched_rows, "skill"),
            "sourceReportDigest": dsh["reportDigest"],
        },
        "cases": case_rows,
        "bindings": {
            "provenanceDigest": provenance["provenanceDigest"],
            "pairedStudyDigest": provenance["pairedStudyDigest"],
            "boundStudyDigest": bound_inspection["workspaceDigest"],
            "translationReportDigest": translation_report["reportDigest"],
            "dshReportDigest": dsh["reportDigest"],
        },
        "claimBoundary": provenance["claimBoundary"],
    }
    summary = {**report_body, "reportDigest": sha256_json(report_body)}
    output = Path(output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("ES-P1-Wild evidence-report output must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (output / "report.md").write_text(_render_markdown(summary), encoding="utf-8")
    (output / "report.html").write_text(_render_html(summary), encoding="utf-8")
    manifest_body = {
        "apiVersion": MANIFEST_SCHEMA,
        "evidenceClass": EVIDENCE_CLASS,
        "officialEsP1QualificationEligible": False,
        "summaryDigest": summary["reportDigest"],
        "files": {
            name: _file_digest(output / name) for name in ("summary.json", "report.md", "report.html")
        },
    }
    manifest = {**manifest_body, "manifestDigest": sha256_json(manifest_body)}
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return summary


def inspect_simulation_evidence_report(root_path: str | Path) -> dict[str, Any]:
    root = Path(root_path).expanduser().resolve()
    expected = {"summary.json", "report.md", "report.html", "manifest.json"}
    if {item.name for item in root.iterdir() if item.is_file()} != expected:
        raise ValueError("ES-P1-Wild evidence-report file set mismatch")
    manifest = _json(root / "manifest.json")
    _verify_object_digest(manifest, "manifestDigest", "evidence report manifest")
    if (
        manifest.get("evidenceClass") != EVIDENCE_CLASS
        or manifest.get("officialEsP1QualificationEligible") is not False
    ):
        raise ValueError("ES-P1-Wild evidence-report authority boundary mismatch")
    for name, digest in manifest["files"].items():
        if _file_digest(root / name) != digest:
            raise ValueError(f"ES-P1-Wild evidence-report file digest mismatch: {name}")
    summary = _json(root / "summary.json")
    _verify_object_digest(summary, "reportDigest", "evidence report summary")
    if summary["reportDigest"] != manifest["summaryDigest"]:
        raise ValueError("ES-P1-Wild evidence-report summary binding mismatch")
    return {
        "status": summary["status"], "caseCount": summary["corpus"]["caseCount"],
        "pairedObservationCount": summary["pairedExecution"]["pairedObservationCount"],
        "armExecutionCount": summary["pairedExecution"]["armExecutionCount"],
        "reportDigest": summary["reportDigest"],
        "manifestDigest": manifest["manifestDigest"],
        "officialEsP1QualificationEligible": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("simulation_root")
    parser.add_argument("result_root")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args(argv)
    report = build_simulation_evidence_report(
        args.simulation_root, args.result_root, args.output_root,
        require_complete=not args.allow_smoke,
    )
    print(json.dumps({
        "status": report["status"], "reportDigest": report["reportDigest"],
        "caseCount": report["corpus"]["caseCount"],
        "pairedObservationCount": report["pairedExecution"]["pairedObservationCount"],
        "armExecutionCount": report["pairedExecution"]["armExecutionCount"],
        "officialEsP1QualificationEligible": False,
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_simulation_evidence_report", "inspect_simulation_evidence_report",
]
