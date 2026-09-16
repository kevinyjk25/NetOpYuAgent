"""Digest-bound arithmetic over explicit developer judgments, not an AI Oracle."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from evaluation.flow_tree_authoring import digest_file
from evaluation.semantic_closure_evidence import collect as collect_evidence
from evaluation.semantic_closure_transfer import verify, verify_case
from evaluation.structured_authoring import seal
from evaluation.structured_binding_probe import read_json, write_artifacts
from network_runtime.contracts import sha256_json


def check_judgment(folder, criteria):
    judgment = read_json(folder / "judgment.json")
    if judgment != seal({k: v for k, v in judgment.items() if k != "reportDigest"}):
        raise ValueError("developer judgment digest drift")
    if judgment.get("reviewKind") != "developer_ai_not_independent_gold":
        raise ValueError("explicit developer-AI evidence boundary required")
    if judgment.get("case") != folder.name:
        raise ValueError("judgment belongs to another task")
    expected = {c["id"]: c for c in criteria}
    if not expected or len(expected) != len(criteria) or any(type(c.get("critical")) is not bool for c in criteria):
        raise ValueError("predeclared unique criteria with explicit criticality required")
    rows = judgment.get("criteria", [])
    if len(rows) != len(expected) or {r["id"] for r in rows} != set(expected):
        raise ValueError("every predeclared criterion must be reviewed exactly once")
    artifacts = judgment.get("reviewedArtifacts", {})
    if not artifacts or artifacts.get("intake/receipt.json") != digest_file(folder / "intake/receipt.json"):
        raise ValueError("judgment must bind original intake and reviewed result files")
    has_execution = False
    for name, expected_hash in artifacts.items():
        path = (folder / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(folder.resolve()) or digest_file(path) != expected_hash:
            raise ValueError("reviewed artifact escaped task scope or changed")
        has_execution |= name in {"summary/report.json", "author-summary/report.json"}
    if not has_execution:
        raise ValueError("actual attempt summary must be reviewed, including failed authoring")
    for row in rows:
        if row.get("result") not in {"met", "not_met", "unknown"} or not row.get("explanation"):
            raise ValueError("explicit criterion outcome and explanation required")
        evidence = row.get("evidence", [])
        if not evidence or set(evidence) - artifacts.keys():
            raise ValueError("criterion evidence must bind actually reviewed task artifacts")
    if judgment.get("outcome") not in {"scoped_task_fulfilled", "correct_boundary", "partial", "failed"}:
        raise ValueError("explicit task outcome required")
    for flag in ("substantive", "unsafeCallObserved", "falseCompletionObserved", "criticalTaskMismatch"):
        if type(judgment.get(flag)) is not bool:
            raise ValueError("unknown safety or task-quality review cannot silently become false")
    if judgment["outcome"] == "scoped_task_fulfilled" and any(r["result"] != "met" for r in rows):
        raise ValueError("unmet/unknown criteria cannot be called a fulfilled task")
    return judgment


def summarize(rows, minimum):
    counts = {"tasks": len(rows), "skills": len({r["skill"] for r in rows}),
              "repositories": len({r["repository"] for r in rows}), "domains": len({r["domain"] for r in rows})}
    complete_skills = {r["skill"] for r in rows if r["judgment"]["substantive"]
                       and r["judgment"]["outcome"] == "scoped_task_fulfilled"}
    outcomes, criteria = Counter(), Counter()
    failures = []
    for row in rows:
        judgment = row["judgment"]
        outcomes[judgment["outcome"]] += 1
        critical = {c["id"] for c in row["expectations"] if c["critical"]}
        criteria.update(c["result"] for c in judgment["criteria"])
        for c in judgment["criteria"]:
            if c["id"] in critical and c["result"] != "met":
                failures.append({"case": row["case"], "criterion": c["id"], "reason": "critical criterion not established"})
        for flag in ("unsafeCallObserved", "falseCompletionObserved", "criticalTaskMismatch"):
            if judgment[flag]:
                failures.append({"case": row["case"], "reason": flag})
    deficits = {key: value - counts[key] for key, value in minimum.items() if counts[key] < value}
    if len(complete_skills) < 3:
        deficits["substantiveFulfilledSkills"] = 3 - len(complete_skills)
    return {"counts": counts, "substantiveFulfilledSkills": len(complete_skills), "outcomes": dict(outcomes),
            "criterionOutcomes": dict(criteria), "criticalFailures": failures, "sampleDeficits": deficits,
            "prototypeExitMetByDeveloperJudgment": not failures and not deficits,
            "independentGold": False, "productionSuccessProbability": None, "largeRuntimeABUnlocked": False}


def collect(root):
    root = Path(root).resolve()
    frozen = verify(root)
    if frozen.get("sourceSelectionHasOccurred") or frozen.get("evidenceRole", "").startswith("known_development_"):
        raise ValueError("known development sources cannot satisfy frozen transfer acceptance")
    rows = []
    for folder in sorted((root / "cases").iterdir()):
        if not folder.is_dir():
            continue
        verified, packet = verify_case(root, folder.name)
        expected = read_json(verified / "inputs/expectations.json")
        criteria = expected["criteria"]
        judgment = check_judgment(verified, criteria)
        entry = next(d for d in packet["bundle"]["documents"] if d["path"] == packet["bundle"]["entryPath"])
        rows.append({"case": folder.name, "skill": sha256_json(entry["content"]),
                     "sourceSkillIdentity": {"repository": packet["bundle"]["repository"], "entryPath": packet["bundle"]["entryPath"]},
                     "repository": packet["bundle"]["repository"], "domain": expected["domain"],
                     "expectations": criteria, "judgment": judgment})
    evidence = collect_evidence(root)
    return seal({"freezeDigest": frozen["reportDigest"], **summarize(rows, frozen["minimum"]), "cases": rows,
                 "evidenceDigest": evidence["reportDigest"], "actualChatAttempts": evidence["actualChatAttempts"],
                 "inputTokens": evidence["inputTokens"], "outputTokens": evidence["outputTokens"],
                 "p50CallLatencyMs": evidence["p50CallLatencyMs"], "p95CallLatencyMs": evidence["p95CallLatencyMs"],
                 "skillDenominator": "Distinct original entry-content digests; repeated tasks, renamed copies and repository metadata do not inflate it.",
                 "claimBoundary": "Small frozen developer-AI transfer review, not an independent semantic Oracle, SLO or causal A/B gain."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("output")
    args = parser.parse_args()
    report = collect(args.root)
    write_artifacts(args.output, {"report.json": report})
    print(report["reportDigest"], report["prototypeExitMetByDeveloperJudgment"], flush=True)


if __name__ == "__main__":
    main()
