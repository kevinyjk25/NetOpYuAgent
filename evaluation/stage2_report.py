"""Digest-bound first-construction facts plus explicit developer-AI review."""
from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.stage1_evidence import percentile
from evaluation.stage2_batch import digest
from evaluation.flow_checkpoint import verify_receipt
from evaluation.structured_authoring import seal
from evaluation.structured_binding_probe import read_json, write_artifacts


def collect(preparation, run, reviews):
    preparation, run, reviews = Path(preparation), Path(run), Path(reviews)
    m = read_json(preparation / "freeze/manifest.json")
    summary = read_json(run / "summary/report.json")
    review = read_json(reviews)
    if summary["preparationDigest"] != m["reportDigest"] or review["preparationDigest"] != m["reportDigest"]:
        raise ValueError("review/batch preparation mismatch")
    if summary != seal({k: v for k, v in summary.items() if k != "reportDigest"}):
        raise ValueError("construction summary digest mismatch")
    ids = [r["id"] for r in summary["rows"]]
    if len(ids) != len(set(ids)) or set(ids) != {r["id"] for r in m["cases"]}:
        raise ValueError("construction summary must retain every selected Skill")
    if review["reviewKind"] != "developer_ai_not_independent_gold":
        raise ValueError("no independent Gold claims for developer review")
    by_id = {r["id"]: r for r in review["reviews"]}
    if len(by_id) != len(review["reviews"]) or set(by_id) != {r["id"] for r in m["cases"]}:
        raise ValueError("all selected Skills need distinct explicit review dispositions")
    if m != seal({k: v for k, v in m.items() if k != "reportDigest"}):
        raise ValueError("preparation digest mismatch")
    if any(digest(preparation / p) != d for p, d in m["artifactDigests"].items()):
        raise ValueError("source/task/review-input drift")
    rows, calls, paths = [], [], [preparation / "freeze/manifest.json", run / "summary/report.json", reviews]
    for row in summary["rows"]:
        case = row["id"]
        decision = by_id[case]
        if not decision.get("findings") or not decision.get("limitations"):
            raise ValueError("explicit findings and scope limits required")
        report_path = run / case / "final-report/report.json"
        if report_path.exists():
            final = read_json(report_path)
            if decision["reportDigest"] != final["reportDigest"]:
                raise ValueError("review is not bound to the final report")
            if row["status"] != final["status"] or row.get("compiled") != final["compiled"]:
                raise ValueError("summary differs from construction outcome")
            replay = read_json(run / case / "replay-report/report.json")
            if replay != final:
                raise ValueError("original-version offline replay differs")
        elif decision.get("reportDigest") is not None:
            raise ValueError("review claims a nonexistent model report")
        if decision["verdict"] == "accepted_read_region" and not row.get("compiled"):
            raise ValueError("cannot accept an uncompiled region")
        compilations = list((run / case).glob("round-*/compilation.json"))
        if compilations and decision.get("treeDigest") != read_json(compilations[-1])["treeDigest"]:
            raise ValueError("semantic review must bind the compiled tree")
        for path in sorted((run / case).glob("round-*/result.json")):
            verify_receipt(path.parent)
            result = read_json(path)
            calls.append({"case": case, **result})
        rows.append({**row, "review": decision})
    paths.extend(preparation / p for p in m["artifactDigests"])
    paths.extend(p for p in run.rglob("*") if p.is_file())
    times = [r["latencyMs"] for r in calls if r.get("latencyMs") is not None]
    unknown = sum(r.get("inputTokens") is None or r.get("outputTokens") is None for r in calls)
    report = {"stage": 2, "preparationDigest": m["reportDigest"], "protocol": m["protocol"],
        "skillCount": m["skillCount"], "repositoryCount": m["repositoryCount"], "domainCount": m["domainCount"],
        "toolBearingSkills": sum(c["status"] != "l1_only_no_external_read_not_run" for c in m["cases"]),
        "initialRequestsWithinBudget": sum(c["status"] == "ready_for_first_construction" for c in m["cases"]),
        "modelAttemptedSkills": sum(bool(list((run / c["id"]).glob("round-*/request.json"))) for c in m["cases"]),
        "structurallyCompiledSkills": sum(bool(r.get("compiled")) for r in rows),
        "acceptedReadRegions": sum(r["review"]["verdict"] == "accepted_read_region" for r in rows),
        "rows": rows, "modelCalls": len(calls), "receiptsWithUnknownTokenUsage": unknown,
        "unreceiptedRequests": len(list(run.glob("*/round-*/request.json"))) - len(list(run.glob("*/round-*/receipt.json"))),
        "recordedInputTokens": sum(r.get("inputTokens") or 0 for r in calls),
        "recordedOutputTokens": sum(r.get("outputTokens") or 0 for r in calls),
        "summedRequestLatencyMs": sum(times), "requestLatencyMs": {"p50": percentile(times, .5), "p95": percentile(times, .95)},
        "runtimeLatencyMs": None, "runtimeBehaviorTests": 0, "parameterOracleAccuracy": None,
        "correctStopRate": None, "overStopRate": None, "dutyRetentionRate": None,
        "usefulRegionCoverage": None, "wholeSkillAccuracy": None, "unseenGeneralizationProven": False,
        "sourceScriptCalls": 0, "providerCalls": 0, "runtimeAuthorityGranted": False,
        "largeRuntimeABUnlocked": False, "reviewKind": review["reviewKind"],
        "limitations": review["limitations"], "reporterDigest": digest(__file__),
        "artifactDigests": {str(p): digest(p) for p in sorted(set(paths))}}
    return seal(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preparation")
    parser.add_argument("run")
    parser.add_argument("reviews")
    parser.add_argument("output")
    args = parser.parse_args()
    result = collect(args.preparation, args.run, args.reviews)
    write_artifacts(args.output, {"report.json": result})
    print(result["reportDigest"])


if __name__ == "__main__":
    main()
