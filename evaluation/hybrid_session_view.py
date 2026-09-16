"""Show host status, source selection and native prose separately; no AI judge."""
from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

from dsh_adapter.hybrid_session import _sealed
from evaluation.stage2_batch import digest
from skill_authoring.artifacts import write_artifacts
from skill_authoring.contracts import seal


def build(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(root):
        raise ValueError("use a new view directory outside the frozen run")
    freeze = _sealed(root / "freeze.json")
    sources, sections = {"freeze.json": digest(root / "freeze.json")}, []
    def take(path):
        sources[str(path.relative_to(root))] = digest(path)
        return _sealed(path)
    for case in freeze["knownCases"]:
        folder = root / case
        task = freeze["inputsAndExpectations"][case][1]["task"]
        stdout = folder / "dsh-stdout.txt"
        sources[str(stdout.relative_to(root))] = digest(stdout)
        host_rows, selections = [], []
        for session in sorted((folder / "sessions").glob("*")):
            take(session / "request.json")
            contract_path = session / "delivery-contract/report.json"
            if contract_path.is_file():
                contract = take(contract_path)
                for row in contract["requirements"]:
                    selections.append("<li><b>" + escape(row["id"] + " · " + row["kind"]) + "</b><pre>" +
                                      escape(row["quote"]) + "</pre></li>")
            report_path = session / "draft/result/report.json"
            if not report_path.is_file():
                host_rows.append("<p class='alert'>没有终态交付回执 / No terminal delivery receipt.</p>")
                continue
            report = take(report_path)
            evidence = take(session / "draft/evidence.json")
            if report["evidenceDigest"] != evidence["reportDigest"]:
                raise ValueError("delivery/evidence binding drift")
            outcome = report.get("hostResult")
            if outcome is None:
                raise ValueError("explicit hostResult required; do not infer legacy success")
            candidate = report["task"].get("delivery")
            if outcome["deliveryDigest"] != (candidate["reportDigest"] if candidate else None):
                raise ValueError("host result/delivery binding drift")
            host_rows.append("<h3>" + escape(outcome["state"]) + "</h3><p>" + escape(outcome["message"]) +
                "</p><p>语义未批准 / Semantic approval: false · Task success: unknown</p><pre>" +
                escape(candidate["rendered"] if candidate else "没有获准交付文本 / No admitted delivery text") + "</pre>")
            if report.get("diagnostic"):
                host_rows.append("<pre>" + escape(str(report["diagnostic"])) + "</pre>")
        sections.append("<section><h2>" + escape(case) + "</h2><details><summary>原任务及模型所选职责 / Task & selected requirements</summary>"
            "<pre>" + escape(task) + "</pre><p>锚点存在不证明类型正确或覆盖完整。 / Membership is not interpretation or coverage proof.</p><ol>" +
            "".join(selections) + "</ol></details><div class='columns'><article><h3>宿主回执 / Host receipt</h3>" +
            ("".join(host_rows) or "<p class='alert'>没有有效会话 / No valid session</p>") +
            "</article><article><h3>Agent 最终文字 / Native final prose</h3><p>以下文字不能覆盖左侧状态。未在本视图自动判断业务正确性。</p><pre>" +
            escape(stdout.read_text(encoding="utf-8")) + "</pre></article></div></section>")
    html = """<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>Host receipt / Agent prose</title><style>body{max-width:1280px;margin:30px auto;padding:0 20px;font:16px/1.55 system-ui;color:#172437;background:#f4f6f8}h1{font-size:26px}.columns{display:grid;grid-template-columns:1fr 1fr;gap:16px}article,details{padding:18px;background:white;border:1px solid #c8d1dc;border-radius:8px;min-width:0}section{margin:28px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.55 ui-monospace,monospace}summary{cursor:pointer}.alert{color:#9b1c1c}@media(max-width:800px){.columns{grid-template-columns:1fr}}</style>
<h1>宿主结果与 Agent 声明分离 / Host results are not Agent claims</h1>
<p>只读诊断视图；不执行来源脚本、不重跑、不提供自动语义评分。可展开原任务，核对模型所选职责是否遗漏。</p>""" + "".join(sections) + "</html>"
    report = seal({"freezeDigest": freeze["reportDigest"], "sourceRoot": str(root), "sourceDigests": sources,
                   "semanticAssessment": "not_performed", "sourceScriptsExecuted": False})
    write_artifacts(output, {"report.json": report})
    (output / "host-result.html").write_text(html, encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("output")
    args = parser.parse_args()
    build(args.run, args.output)
