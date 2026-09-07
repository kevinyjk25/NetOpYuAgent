"""Recompute a frozen batch and separately authored source reviews; no inference."""

import argparse
import json
from pathlib import Path

from evaluation.flow_translation import _write, assess, load_run
from evaluation.flow_translation_batch import report
from evaluation.read_l05_review import ReadL05Review
from network_runtime.contracts import sha256_json


def build_evidence(root: Path, review_root: Path) -> dict:
    structural = report(root)
    if any(row["status"] not in {"blocked", "awaiting_source_review"} for row in structural["rows"]):
        raise ValueError("batch not complete")
    reviews = []
    counts = {"supported": 0, "contradicted": 0, "insufficient_evidence": 0}
    for row in structural["rows"]:
        if row["status"] != "awaiting_source_review":
            continue
        review = ReadL05Review.model_validate_json((review_root / (row["id"] + ".json")).read_text())
        rebuilt = assess(*load_run(root / row["id"]), review)
        saved = json.loads((review_root / (row["id"] + "-report.json")).read_text())
        if rebuilt != saved:
            raise ValueError("saved semantic assessment differs from source-bound replay")
        for key, value in rebuilt["assessment"]["verdictCounts"].items():
            counts[key] += value
        reviews.append({"id": row["id"], "status": rebuilt["status"],
            "inputDigest": rebuilt["inputDigest"], "reviewDigest": rebuilt["reviewDigest"],
            "reportDigest": rebuilt["reportDigest"], "reviewerId": rebuilt["reviewerId"],
            "reviewerKind": rebuilt["reviewerKind"], "verdictCounts": rebuilt["assessment"]["verdictCounts"],
            "unresolvedQuestions": rebuilt["unresolvedQuestions"],
            "findings": [{"claimId": finding["claimId"], "pointer": finding["l05Pointer"],
                          "verdict": finding["verdict"], "explanation": finding["rationale"]}
                         for finding in rebuilt["blockers"]]})
    body = {"evidenceRole": "known_development_same_assistant_review_not_independent",
        "structuralReportDigest": structural["reportDigest"], "manifestDigest": structural["manifestDigest"],
        "model": structural["model"], "flowCount": structural["flowCount"], "publicSkillCount": 0,
        "domainCount": 1, "modelResponses": sum("httpStatus" in row for row in structural["rows"]),
        "modelResponseTotalMs": sum(row.get("latencyMs", 0) for row in structural["rows"]),
        "inputTokens": sum(row.get("inputTokens") or 0 for row in structural["rows"]),
        "outputTokens": sum(row.get("outputTokens") or 0 for row in structural["rows"]),
        "structurallyQualified": structural["structurallyQualified"],
        "sourceReviewedFlows": len(reviews), "claimVerdicts": counts,
        "reviewSupportedInactiveFlows": sum(row["status"] == "review_supported_inactive_flow" for row in reviews),
        "runtimeExecutions": 0, "writes": 0, "scriptExecutions": 0, "assistedEdits": 0,
        "reviews": reviews, "independentHumanEvidence": False,
        "boundary": "Structurally qualified drafts receive complete claim review; structurally invalid replies do not. Claim counts overlap and are not translation accuracy. No repair/retry or execution, no production/generalization claim. Latency excludes source review and testing."}
    return {**body, "reportDigest": sha256_json(body)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("review_root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    _write(args.output, build_evidence(args.root, args.review_root))


if __name__ == "__main__":
    main()
