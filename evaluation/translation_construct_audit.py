"""Audit sealed authoring constructs under current rules without relabeling them."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation.translation_case_authoring import (
    AnchoredBundle, authoring_implementation_digest, inspect_anchored_case_authoring,
    validate_anchored_bundle,
)
from network_runtime.contracts import sha256_json


def audit_constructs(
    corpus_root: Path, authoring_roots: list[Path], output_path: Path,
) -> dict[str, Any]:
    """Create a new diagnostic artifact, not replacement Gold or a revised score."""

    corpus = corpus_root.resolve()
    roots = [root.resolve() for root in authoring_roots]
    output = output_path.resolve()
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("provide unique nonempty authoring roots")
    if any(output.is_relative_to(root) for root in [corpus, *roots]):
        raise ValueError("audit output must be outside sealed inputs")
    if output.exists():
        raise ValueError("audit output already exists; choose a new version")
    skills = {
        skill["packageId"]: skill
        for skill in json.loads((corpus / "index.json").read_text())["skills"]
    }
    sources, rows = [], []
    for root in roots:
        inspection = inspect_anchored_case_authoring(root, corpus)
        sources.append({"path": str(root), "workspaceDigest": inspection["workspaceDigest"]})
        for line in (root / "candidates.jsonl").read_text().splitlines():
            if not line:
                continue
            row = json.loads(line)
            if row["candidate"] is None:
                continue
            validation = validate_anchored_bundle(
                skills[row["packageId"]], row["assignmentId"],
                AnchoredBundle.model_validate(row["candidate"]), validation_version="v3",
            )
            rows.append({
                "sourceWorkspaceDigest": inspection["workspaceDigest"],
                "assignmentId": row["assignmentId"], "packageId": row["packageId"],
                "skillName": row["skillName"], "sourceStatus": row["status"],
                "candidateDigest": sha256_json(row["candidate"]),
                "currentMechanicalChecksPassed": validation["passed"],
                "failures": validation["failures"],
                "constructChecks": validation["constructChecks"],
            })
    body = {
        "apiVersion": "effect-runtime.io/translation-construct-audit/v1",
        "authoringImplementationDigest": authoring_implementation_digest(),
        "sources": sources, "rows": rows,
        "candidateCount": len(rows),
        "uniqueSkillCount": len({row["packageId"] for row in rows}),
        "sourceAcceptedCount": sum(row["sourceStatus"] == "accepted_candidate" for row in rows),
        "currentMechanicalPassCount": sum(row["currentMechanicalChecksPassed"] for row in rows),
        "newlyBlockedPreviouslyAcceptedCount": sum(
            row["sourceStatus"] == "accepted_candidate" and not row["currentMechanicalChecksPassed"]
            for row in rows
        ),
        "failureCounts": dict(sorted(Counter(
            failure.split(":", 1)[-1] if failure.startswith("development-") else failure
            for row in rows for failure in row["failures"]
        ).items())),
        "semanticAlignmentProven": False, "goldAuthored": False,
        "runtimeOrDshExecuted": False, "thirdPartyExecutionAttempted": False,
        "claimBoundary": (
            "Known-development static re-audit only. Source artifacts and labels unchanged. "
            "Mechanical pass counts are neither translation accuracy nor semantic validation."
        ),
    }
    report = {**body, "reportDigest": sha256_json(body)}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents an accidental overwrite after a race.
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_root", type=Path)
    parser.add_argument("authoring_roots", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_constructs(args.corpus_root, args.authoring_roots, args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
