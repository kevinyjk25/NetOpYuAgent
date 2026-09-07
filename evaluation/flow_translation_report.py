"""Replay a bounded development experiment; never turn retries into accuracy."""

import argparse
import hashlib
import json
from pathlib import Path

from evaluation.flow_translation import FlowDraft, FlowRevision, FlowSources, IndexedDraft, assess, diagnose, load_run
from evaluation.read_l05_review import ReadL05Review
from network_runtime.contracts import sha256_json


def build_report(attempts: list[Path]) -> dict:
    if len(attempts) != 3 or len({path.resolve() for path in attempts}) != 3:
        raise ValueError("this report requires three distinct development protocol attempts")
    rows = []
    artifacts = {}
    for root in attempts:
        sources = FlowSources.model_validate_json((root / "sources.json").read_text())
        draft = FlowDraft.model_validate_json((root / "draft.json").read_text())
        response = json.loads((root / "response.json").read_text())
        model = json.loads(response["body"])
        request = json.loads((root / "request.json").read_text())
        original_text = model["message"]["content"]
        reconstructed = IndexedDraft.model_validate_json(original_text).named() if request.get("draftProtocol") == "indexed-v1" else FlowDraft.model_validate_json(original_text)
        if reconstructed != draft:
            raise ValueError("attempt draft differs from original model reply")
        rows.append({"attempt": root.name, "model": request["model"], "latencyMs": response["latencyMs"],
                     "inputTokens": model["prompt_eval_count"], "outputTokens": model["eval_count"],
                     "structuralIssues": diagnose(sources, draft),
                     "sourceDigest": sha256_json(sources.source_text),
                     "requestDigest": sha256_json(request), "responseDigest": sha256_json(response)})
        artifacts[root.name] = {file.name: "sha256:" + hashlib.sha256(file.read_bytes()).hexdigest()
                                for file in sorted(root.glob("*.json"))}
    root = attempts[-1]
    if len({row["sourceDigest"] for row in rows}) != 1:
        raise ValueError("single-source development report cannot combine different sources")
    sources, parent = load_run(root)
    original = assess(sources, parent, ReadL05Review.model_validate_json((root / "review-original.json").read_text()))
    revision = FlowRevision.model_validate_json((root / "text-revision.json").read_text())
    revised = assess(sources, parent, ReadL05Review.model_validate_json((root / "review-revised.json").read_text()), revision)
    for name, expected in (("report-original.json", original), ("report-revised.json", revised)):
        if json.loads((root / name).read_text()) != expected:
            raise ValueError("saved semantic report differs from replay")
    executions = [json.loads((root / name).read_text()) for name in ("run-campus.json", "run-idc.json")]
    for execution in executions:
        result = execution["execution"]
        if result["reportDigest"] != sha256_json({key: value for key, value in result.items() if key != "reportDigest"}):
            raise ValueError("execution receipt digest mismatch")
        if execution["reviewReportDigest"] != revised["reportDigest"]:
            raise ValueError("execution refers to another review")
    body = {"evidenceRole": "known_development_single_flow_protocol_iteration",
            "sourceCount": 1, "publicSkillCount": 0, "modelCalls": len(rows), "attempts": rows,
            "modelLatencyTotalMs": sum(row["latencyMs"] for row in rows),
            "inputTokensTotal": sum(row["inputTokens"] for row in rows),
            "outputTokensTotal": sum(row["outputTokens"] for row in rows),
            "rawOriginalSemanticStatus": original["status"],
            "originalClaimVerdicts": original["assessment"]["verdictCounts"],
            "assistedSemanticStatus": revised["status"], "assistedClaimVerdicts": revised["assessment"]["verdictCounts"],
            "assistantTextEdits": len(revision.edits), "graphAndArgumentsEdited": False,
            "executions": [{"status": value["execution"]["status"], "reportDigest": value["execution"]["reportDigest"]} for value in executions],
            "actualLocalReads": sum(row["kind"] == "read" for value in executions for row in value["execution"]["trace"]),
            "effectCalls": 0, "dshAgentLoopExecuted": False, "publicWholeSkillTranslationCompleted": False,
            "independentHumanEvidence": False, "artifacts": artifacts,
            "originalReportDigest": original["reportDigest"], "revisedReportDigest": revised["reportDigest"],
            "boundary": "Three different development protocols on one known fixture, not repeated independent trials. No calibrated accuracy, p50/p95, unseen generalization or autonomous success claim. Editor and reviewer are the same assistant."}
    return {**body, "reportDigest": sha256_json(body)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempts", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.attempts)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
