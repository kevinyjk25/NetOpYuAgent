"""Readable bilingual view of digest-bound offline diagnostic reports."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from evaluation.flow_diagnostics import PROTOCOL
from network_runtime.contracts import sha256_json

LABELS = {
    'source': '原文与输入', 'representation': '流程表达/合同检查',
    'parent_semantic_review': '第一阶段源文审查', 'mapping': '分类与来源映射',
    'lowering_consistency': '映射前后 L0 一致性', 'semantic_review': '完整语义审查', 'runtime': 'Runtime 执行',
    'objective_kind_conflict': '目的与自身分类冲突', 'node_evidence_missing': '已有节点缺来源证据',
    'unresolved_requirement': '模型标为未解决的要求', 'declared_missing_host_capability': '模型报告宿主缺能力（待核实）',
}


def cell(value):
    text = html.escape(str(value), quote=True).replace('\\', '\\\\')
    for char in '|`[]*_#':
        text = text.replace(char, '\\' + char)
    return text.replace('\n', '<br>')


def verify(report):
    if report.get('protocol') != PROTOCOL or report.get('reportDigest') != sha256_json({k: v for k, v in report.items() if k != 'reportDigest'}):
        raise ValueError('diagnostic protocol/digest mismatch')


def render(report):
    verify(report)
    rows = report.get('cases', [dict(case='single-case', diagnosis=report)])
    for row in rows:
        verify(row['diagnosis'])
    lines = ['# 转译分层诊断 / Layered Translation Diagnostics', '', '## 中文', '',
        '本视图由冻结输入的离线诊断生成，不修复答案、不重新评分、不调用模型或执行工具。', '',
        '**判读边界：已有节点缺证据 ≠ 节点不存在；引文精确 ≠ 原意完整；模型报告缺能力 ≠ 已独立验证；未评估 ≠ 0% 成功。**', '',
        '这里的要求条目由原模型生成，不是独立审阅的语义义务总表，因此准确率、语义丢失率和置信概率均不提供。', '',
        f'诊断摘要：`{report["reportDigest"]}`', '']
    if 'originalReportDigest' in report:
        lines += [f'原报告摘要：`{report["originalReportDigest"]}`；原资格保持 **流程 {report["originalFlowQualified"]}/{len(rows)}、映射 {report["originalMappingQualified"]}/{len(rows)}**。', '',
            '| 独立诊断类别 | 数量（不是错误概率） |', '|---|---:|']
        lines += [f'| {cell(LABELS.get(k, k))} | {v} |' for k, v in report['findingsByCode'].items()]
        lines += ['']
    for row in rows:
        r = row['diagnosis']
        lines += [f'### {cell(row["case"])}', '',
            f'最早观测到的失败层：**{cell(LABELS.get(r["earliestFailedStage"], r["earliestFailedStage"] or "无机械失败；不代表语义接受"))}**。', '',
            '| 阶段 | 检查状态 |', '|---|---|']
        lines += [f'| {cell(LABELS.get(k, k))} | {cell(v["status"])} |' for k, v in r['stages'].items()]
        lines += ['', '| 阻断项 | 定位 | 关联来源 |', '|---|---|---|']
        errors = [f for f in r['findings'] if f['severity'] == 'error']
        lines += [f'| {cell(LABELS.get(f["code"], f["code"]))} | {cell(f["pointer"] or "根对象")} | {cell(", ".join(f["sourceClauseIds"]))} |' for f in errors]
        if not errors:
            lines += ['| 无观测到的机械/已提供审阅阻断 | 不等于转译成功 | 仍需完整源审查 |']
        missing = [n for n in r['nodeTrace'] if not n['eligibleCandidatePointers']]
        if missing:
            lines += ['', '下列节点**已经存在并下沉**，缺的是可用来源映射；父节点引用仅作定位线索，不能作为正确性证明。', '',
                '| L1 原文线索 | L0.5 节点 | L0 位置 | 应检查什么 |', '|---|---|---|---|']
            for node in missing:
                source = next((s['exactQuote'] for s in r['sourceLines'] if s['sourceId'] == node['parentSourceId']), '')
                lines += [f'| {cell(source)} | {cell(node["treePointer"])} ({cell(node["node"]["kind"])}) | {cell(node["l0Pointer"])} | 原文是否支持此节点；补忠实证据或标记节点无依据，不自动补图 |']
        lines += ['', f'未解决候选：{r["counts"]["unresolvedCandidates"]}；声明事项：{len(r["declaredIssues"])}。完整逐条解释、建议和来源偏移见同名 JSON。', '']
    lines += ['## English', '',
        'This is an offline diagnostic view, not a rescore. No model, Runtime, provider or script execution occurs. '
        'A missing node citation is not a missing node. Exact text matching is not source entailment. '
        'Declared capability gaps and unresolved candidates require independent source/host verification. '
        'Downstream not_evaluated is not zero-percent success.', '',
        'Source/host, tree and mapping digests bind the input. Findings aggregate independent mechanical defects with exact source spans and L0.5/L0 pointers. '
        'Optional digest-bound first-pass reviews remain usable even when mapping is blocked. Complete mapping reviews require successful compilation. '
        'An assisted witness tests bounded constructibility without replacing or rescoring the original. '
        'Projection equality checks mapped versus parent L0, not arbitrary compiler correctness or source fidelity.', '',
        'No calibrated semantic confidence, semantic-loss rate, independent obligation denominator or automatic admission is claimed. '
        'Candidate requirement counts and development fixtures are not independent Gold, public-Skill generalization, or production success probabilities.', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('output exists; preserve previous evidence')
    result = render(json.loads(args.report.read_text()))
    with args.output.open('x', encoding='utf-8') as handle:
        handle.write(result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
