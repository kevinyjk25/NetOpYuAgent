"""Join real model, generated graph and explicit draft-review evidence.

Developer review remains developer review. Counts of useful partial work,
requested-task fulfillment and whole-Skill conversion are never interchangeable.
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from evaluation.flow_checkpoint import verify_receipt
from evaluation.stage2_batch import digest
from evaluation.structured_authoring import seal
from evaluation.structured_binding_probe import read_json, write_artifacts


def sealed(path):
    value = read_json(path)
    if value != seal({k: v for k, v in value.items() if k != "reportDigest"}):
        raise ValueError("evidence seal drift")
    return value


def collect(construction_path, cases):
    """cases contains (live directory, sealed draft review, counterfactual report)."""
    construction_path = Path(construction_path)
    construction = sealed(construction_path)
    accepted = {r["id"] for r in construction["records"] if r["review"].get("scopedPlanAccepted") is True}
    artifacts = {str(construction_path): digest(construction_path)}
    results, seen = [], set()
    for folder, review_path, check_path in cases:
        folder, review_path, check_path = map(Path, (folder, review_path, check_path))
        live = sealed(folder / "summary/report.json")
        review, check = sealed(review_path), sealed(check_path)
        case = live["case"]
        if case in seen or case not in accepted:
            raise ValueError("unique source-reviewed compiled Skill required")
        seen.add(case)
        if (review.get("case") != case or review.get("executionDigest") != live["reportDigest"]
                or review.get("reviewKind") != "developer_ai_not_independent_gold"
                or not review.get("findings") or not review.get("limitations")
                or review.get("decision") not in {"useful_partial", "fulfilled_requested_task", "not_accepted"}):
            raise ValueError("explicit actual-draft review bound to execution required")
        if (check["case"] != case or check["compilationDigest"] != live["compilationDigest"]
                or check["actualModelCalls"] != 0):
            raise ValueError("separate synthetic counterfactual evidence required")
        freeze = sealed(folder / "freeze/inputs.json")
        if freeze["compilationDigest"] != live["compilationDigest"] or not freeze["fixtureIsInProcessNotRealNetwork"]:
            raise ValueError("disclosed frozen business fixture required")
        requests = sorted(folder.glob("model/*/request.json"))
        if len(requests) != len(live["modelCalls"]):
            raise ValueError("pending or unaccounted actual model call")
        for path in requests:
            verify_receipt(path.parent)
            cost = read_json(path.parent / "result.json")
            if {"node": path.parent.name, **cost} not in live["modelCalls"]:
                raise ValueError("live cost differs from original model receipt")
        completed = live["execution"]["status"] == "governed_graph_completed"
        if review["decision"] != "not_accepted" and not completed:
            raise ValueError("blocked graph cannot count as a useful completed draft")
        # These expectations were frozen BEFORE live execution, not inferred
        # from the model answer. This checks this fixture only, not generality.
        args = [{"tool": call["tool"], "arguments": call["arguments"],
            "matchesFrozenFixture": call["tool"] in freeze["fixture"]
                and call["arguments"] == freeze["fixture"][call["tool"]][0]} for call in live["providerCalls"]]
        mixed = bool(args) and bool(live["modelCalls"])
        results.append({"case": case, "executionDigest": live["reportDigest"], "compilationDigest": live["compilationDigest"],
            "graphCompleted": completed, "actualModelCalls": len(requests), "strictReadPlusRealReason": mixed,
            "runtimeWallLatencyMs": live["runtimeWallLatencyMs"], "modelCosts": live["modelCalls"],
            "checkedArguments": args, "draftReview": review, "outputs": live["execution"].get("outputs"),
            "counterfactualPassed": check["passed"], "counterfactualTotal": check["total"],
            "sourceScriptCalls": live["sourceScriptCalls"], "effectCalls": live["effectCalls"]})
        artifacts.update({str(p): digest(p) for p in sorted(folder.rglob("*.json"))})
        artifacts.update({str(p): digest(p) for p in (review_path, check_path)})
    costs = [c for r in results for c in r["modelCosts"]]
    times = [r["runtimeWallLatencyMs"] for r in results]
    useful = [r for r in results if r["strictReadPlusRealReason"] and r["draftReview"]["decision"] != "not_accepted"]
    return seal({"evidenceRole": "known_public_development_real_local_9b_with_synthetic_hosts",
        "constructionDigest": construction["reportDigest"], "skillCount": construction["skillCount"],
        "repositoryCount": construction["repositoryCount"], "domainCount": construction["domainCount"],
        "compiledSkillCount": construction["compiledSkillCount"],
        "reviewedUsefulPlanKinds": construction["reviewedUsefulPlanKinds"], "liveSkills": len(results),
        "reviewedUsefulMixedExecutions": len(useful),
        "fulfilledRequestedTasks": sum(r["draftReview"]["decision"] == "fulfilled_requested_task" for r in results),
        "partialUsefulTasks": sum(r["draftReview"]["decision"] == "useful_partial" for r in results),
        "actualModelCalls": len(costs), "records": results,
        "inputTokens": sum(c["inputTokens"] for c in costs) if all(c.get("inputTokens") is not None for c in costs) else None,
        "outputTokens": sum(c["outputTokens"] for c in costs) if all(c.get("outputTokens") is not None for c in costs) else None,
        "runtimeWallLatencyMs": {"p50": statistics.median(times) if times else None,
            "p95": statistics.quantiles(times, n=100, method="inclusive")[94] if len(times) > 1 else None,
            "sampleCount": len(times), "includesRealModel": True, "SLOProven": False},
        "wholeSkillConversionRate": None, "semanticAccuracyOnUnseenSkills": None,
        "newDSHRuntimeAB": False, "largeRuntimeABUnlocked": False,
        "threeUsefulMixedDemonstrationsReached": len(useful) >= 3,
        "stageComplete": None, "stageCompleteMeaning": "Requires separate final QA, documentation and manual exit audit.",
        "artifactDigests": artifacts})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("construction")
    parser.add_argument("output")
    parser.add_argument("--case", nargs=3, action="append", required=True, metavar=("LIVE", "REVIEW", "CHECK"))
    a = parser.parse_args()
    report = collect(a.construction, a.case)
    write_artifacts(a.output, {"report.json": report})
    print(report["reportDigest"])


if __name__ == "__main__":
    main()
