"""Digest-bound mixed development evidence; no metric inflation or activation."""
from __future__ import annotations

import argparse
import statistics
from collections import Counter
from pathlib import Path

from evaluation.stage2_batch import digest
from evaluation.flow_checkpoint import verify_receipt
from evaluation.structured_authoring import seal
from evaluation.structured_binding_probe import read_json, write_artifacts


def summarize(preparation, run, reviews, output):
    preparation, run, reviews, output = map(Path, (preparation, run, reviews, output))
    manifest = read_json(preparation / "freeze/manifest.json")
    report = read_json(run / "summary/report.json")
    review = read_json(reviews)
    if report["preparationDigest"] != manifest["reportDigest"] or review.get("runDigest") != report["reportDigest"]:
        raise ValueError("review belongs to another frozen model run")
    for value in (manifest, report, review):
        if value != seal({k: v for k, v in value.items() if k != "reportDigest"}):
            raise ValueError("manifest/report/review seal drift")
    if any(digest(preparation / path) != expected for path, expected in manifest["artifactDigests"].items()):
        raise ValueError("frozen preparation artifact drift")
    rows = {r["id"]: r for r in report["rows"]}
    expected = {c["id"] for c in manifest["cases"]}
    judgments = {r["id"]: r for r in review["cases"]}
    if (set(rows) != expected or set(judgments) != expected or len(review["cases"]) != len(expected)
            or len(report["rows"]) != len(expected)):
        raise ValueError("retain every preselected Skill, including all failures")
    for key, row in rows.items():
        if judgments[key].get("reviewKind") != "developer_ai_not_independent_gold":
            raise ValueError("developer review cannot claim independent gold")
        if judgments[key].get("scopedPlanAccepted") is True and row["status"] != "compiled_mixed_candidate_requires_review":
            raise ValueError("uncompiled plans cannot be accepted")
        for index, cost in enumerate(row["rounds"]):
            folder = run / key / f"round-{index:02d}"
            verify_receipt(folder)
            if cost != {"round": index, **read_json(folder / "result.json")}:
                raise ValueError("summary differs from original model receipt")
        requests = list((run / key).glob("round-*/request.json"))
        if len(requests) != len(row["rounds"]):
            raise ValueError("unaccounted model request; preserve and diagnose before publication")
    calls = [r for row in rows.values() for r in row["rounds"]]
    latencies = [r["latencyMs"] for r in calls if r.get("latencyMs") is not None]
    observed = [r for r in judgments.values() if r.get("scopedPlanAccepted") is True]
    artifacts = {}
    for label, folder in (("preparation", preparation), ("construction", run)):
        artifacts.update({label + "/" + str(p.relative_to(folder)): digest(p) for p in sorted(folder.rglob("*.json"))})
    artifacts["review"] = digest(reviews)
    records = []
    for case in manifest["cases"]:
        row, judgment = rows[case["id"]], judgments[case["id"]]
        records.append({**case, "construction": row, "review": judgment})
    result = seal({"protocol": manifest["protocol"], "preparationDigest": manifest["reportDigest"], "runDigest": report["reportDigest"],
        "reviewDigest": review["reportDigest"], "skillCount": len(expected), "repositoryCount": len({c["repository"] for c in manifest["cases"]}),
        "domainCount": len({c["domain"] for c in manifest["cases"]}), "calls": len(calls),
        "compiledSkillCount": sum(r["status"] == "compiled_mixed_candidate_requires_review" for r in rows.values()),
        "reviewedUsefulPlanCount": len(observed), "records": records,
        "reviewedUsefulPlanKinds": dict(Counter(r.get("planKind", "unclassified") for r in observed)),
        "wholeRequestedTaskPlanAcceptedCount": sum(r.get("wholeRequestedTaskPlanAccepted") is True for r in judgments.values()),
        "wholeOriginalSkillAcceptedCount": sum(r.get("wholeOriginalSkillAccepted") is True for r in judgments.values()),
        "inputTokens": sum(r["inputTokens"] for r in calls) if all(r.get("inputTokens") is not None for r in calls) else None,
        "outputTokens": sum(r["outputTokens"] for r in calls) if all(r.get("outputTokens") is not None for r in calls) else None,
        "requestLatencyMs": {"p50": statistics.median(latencies) if latencies else None,
            "p95": statistics.quantiles(latencies, n=100, method="inclusive")[94] if len(latencies) >= 2 else None,
            "sum": sum(latencies), "kind": "authoring_model_requests_not_runtime_latency"},
        "artifactDigests": artifacts, "sourceScriptCalls": 0, "businessProviderCallsDuringAuthoring": 0,
        "reviewKind": "developer_ai_not_independent_gold", "semanticAccuracyOnUnseenSkills": None,
        "wholeSkillDeterministicConversionRate": None, "largeRuntimeABUnlocked": False,
        "claimBoundary": "Known public development only. Mixed useful plans are not wholly deterministic L0 or production success probabilities."})
    write_artifacts(output, {"report.json": result})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("preparation", "run", "review", "output"):
        parser.add_argument(name)
    args = parser.parse_args()
    print(summarize(args.preparation, args.run, args.review, args.output)["reportDigest"])


if __name__ == "__main__":
    main()
