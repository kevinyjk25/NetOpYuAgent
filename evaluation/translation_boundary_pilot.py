"""Source-bound intake for heterogeneous known Skills, not translation scoring.

Review annotations describe why environment/flow support is required. They are
not reference answers, inferred API schemas, calibrated labels or permission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from evaluation.translation_corpus import inspect_translation_corpus
from network_runtime.contracts import sha256_json


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceQuote(StrictModel):
    path: str
    exact_quote: str = Field(min_length=1)


class BoundaryEntry(StrictModel):
    package_id: str
    structures: tuple[Literal[
        "no_argument_call", "optional_parameters", "references", "conditional_branch",
        "multi_step", "script_dependency", "approval", "freeform_cli",
        "read_write_mixed", "open_ended_reasoning",
    ], ...] = Field(min_length=1)
    required_environment: tuple[str, ...] = Field(min_length=1)
    single_read_gap: str = Field(min_length=12)
    citations: tuple[SourceQuote, ...] = Field(min_length=1)


class BoundaryPilot(StrictModel):
    corpus_index_digest: str
    reviewer_kind: Literal["ai_role_simulation", "test_fixture"]
    reviewer_id: str
    evidence_role: Literal["known_development_intake_only"]
    entries: tuple[BoundaryEntry, ...] = Field(min_length=8, max_length=12)


def build_boundary_report(index: dict, pilot: BoundaryPilot) -> dict:
    pilot = BoundaryPilot.model_validate(pilot.model_dump())
    if index.get("indexDigest") != sha256_json({k: v for k, v in index.items() if k != "indexDigest"}):
        raise ValueError("corpus index digest mismatch")
    if pilot.corpus_index_digest != index["indexDigest"]:
        raise ValueError("pilot is bound to another corpus index")
    ids = [entry.package_id for entry in pilot.entries]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate pilot Skill")
    skills = {skill["packageId"]: skill for skill in index["skills"]}
    if len(skills) != len(index["skills"]):
        raise ValueError("duplicate corpus package identity")
    rows = []
    for entry in pilot.entries:
        if entry.package_id not in skills:
            raise ValueError("unknown pilot package")
        skill = skills[entry.package_id]
        if not skill["primaryTranslationEligible"]:
            raise ValueError("pilot selection requires a primary known-development Skill")
        files = {item["path"]: item for item in skill["files"]}
        if len(files) != len(skill["files"]):
            raise ValueError("duplicate corpus file path")
        for file in files.values():
            if file["content"] is not None and "sha256:" + hashlib.sha256(file["content"].encode()).hexdigest() != file["sha256"]:
                raise ValueError("source file text differs from pinned digest")
        citations = []
        for quote in entry.citations:
            file = files.get(quote.path)
            if not file or file["content"] is None or file["content"].count(quote.exact_quote) != 1:
                raise ValueError("source quote must occur exactly once in its declared file")
            start = file["content"].index(quote.exact_quote)
            citations.append({"path": quote.path, "start": start, "end": start + len(quote.exact_quote),
                              "exactQuote": quote.exact_quote, "sourceDigest": file["sha256"]})
        rows.append({
            "packageId": entry.package_id, "name": skill["name"], "repository": skill["repository"],
            "commitSha": skill["commitSha"], "packageDigest": skill["packageDigest"], "domain": skill["domain"],
            "structures": sorted(set(entry.structures)), "requiredEnvironment": list(entry.required_environment),
            "singleReadGap": entry.single_read_gap, "citations": citations,
            "status": "needs_environment_and_flow_qualification",
            "toolEnvironmentSuppliedToThisPilot": False,
            "translationOutcome": "not_run", "runtimeAuthorityGranted": False,
        })
    body = {
        "apiVersion": "effect-runtime.io/translation-boundary-intake/v1",
        "indexDigest": index["indexDigest"], "pilotDigest": sha256_json(pilot.model_dump(mode="json")),
        "reviewerKind": pilot.reviewer_kind, "reviewerId": pilot.reviewer_id,
        "evidenceRole": pilot.evidence_role, "skillCount": len(rows),
        "repositoryCount": len({row["repository"] for row in rows}),
        "domainCount": len({row["domain"] for row in rows}),
        "structureCounts": dict(sorted(Counter(x for row in rows for x in row["structures"]).items())),
        "rows": rows, "translationRuns": 0, "runtimeRuns": 0, "modelCalls": 0,
        "proofCohortEligible": False, "thirdPartyExecutionAttempted": False,
        "boundary": (
            "Intake annotations only. No environment contracts were supplied to this run; "
            "this is NOT zero translation accuracy, invalid Skills, or intrinsic untranslatability. "
            "Corpora package-format readiness does not establish executable tool availability."
        ),
    }
    return {**body, "reportDigest": sha256_json(body)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inspect_translation_corpus(args.corpus)
    report = build_boundary_report(json.loads((args.corpus / "index.json").read_text()),
                                   BoundaryPilot.model_validate_json(args.pilot.read_text()))
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
