"""Private sealed-holdout and two-reviewer adjudication contracts for P1.9."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from network_runtime.contracts import sha256_json

from .contracts import L1Decision, L1DecisionAction, StrictModel


HOLDOUT_CASE_SCHEMA = "netopyu.io/l1-holdout-case/v1"
HOLDOUT_LABEL_SCHEMA = "netopyu.io/l1-holdout-label/v1"
HOLDOUT_MANIFEST_SCHEMA = "netopyu.io/l1-holdout-manifest/v1"
_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_MAX_BYTES = 10 * 1024 * 1024
_MAX_CASES = 1_000


class HoldoutCase(StrictModel):
    api_version: Literal[HOLDOUT_CASE_SCHEMA] = Field(
        default=HOLDOUT_CASE_SCHEMA, alias="apiVersion",
    )
    case_id: str
    profile: Literal["lan", "dc", "wan"]
    category: str
    language: Literal["zh", "en", "mixed"]
    prompt: str = Field(min_length=1, max_length=4_000)

    @model_validator(mode="after")
    def validate_identifiers(self) -> "HoldoutCase":
        if not _ID.fullmatch(self.case_id) or not _ID.fullmatch(self.category):
            raise ValueError("holdout case identifiers are invalid")
        return self


class HoldoutLabel(StrictModel):
    api_version: Literal[HOLDOUT_LABEL_SCHEMA] = Field(
        default=HOLDOUT_LABEL_SCHEMA, alias="apiVersion",
    )
    case_id: str
    reviewer_id: str
    action: L1DecisionAction
    target: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    missing_fields: tuple[str, ...] = ()
    workflow: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_semantics(self) -> "HoldoutLabel":
        if not _ID.fullmatch(self.case_id) or not _ID.fullmatch(self.reviewer_id):
            raise ValueError("holdout label identifiers are invalid")
        L1Decision(
            action=self.action,
            target=self.target,
            arguments=self.arguments,
            missing_fields=self.missing_fields,
            workflow=self.workflow,
            confidence=1.0,
            reason_code="holdout_adjudication",
        )
        return self

    def semantic_payload(self) -> dict[str, Any]:
        payload = self.model_dump(by_alias=True, mode="json")
        payload.pop("reviewer_id")
        return payload


def _load_jsonl(path: Path, model: type[HoldoutCase] | type[HoldoutLabel]) -> list[Any]:
    if path.stat().st_size > _MAX_BYTES:
        raise ValueError("holdout input exceeds 10 MiB")
    output: list[Any] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                continue
            if len(output) >= _MAX_CASES:
                raise ValueError("holdout input exceeds 1000 cases")
            try:
                output.append(model.model_validate_json(raw))
            except Exception as error:
                raise ValueError(f"invalid holdout line {line_number}: {type(error).__name__}") from error
    if not output:
        raise ValueError("holdout input is empty")
    identifiers = [item.case_id for item in output]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("holdout case ids must be unique")
    return output


def _case_payload(cases: list[HoldoutCase]) -> list[dict[str, Any]]:
    return [
        item.model_dump(by_alias=True, mode="json")
        for item in sorted(cases, key=lambda value: value.case_id)
    ]


def seal_cases(path: Path, *, dataset_id: str, version: str) -> dict[str, Any]:
    if not _ID.fullmatch(dataset_id) or not _ID.fullmatch(version):
        raise ValueError("holdout dataset identity/version is invalid")
    cases: list[HoldoutCase] = _load_jsonl(path, HoldoutCase)
    categories = Counter(item.category for item in cases)
    profiles = Counter(item.profile for item in cases)
    languages = Counter(item.language for item in cases)
    prompt_digests = {
        sha256_json({"prompt": item.prompt}) for item in cases
    }
    requirements = {
        "at_least_50_cases": len(cases) >= 50,
        "at_least_10_categories": len(categories) >= 10,
        "all_profiles": all(profiles[name] > 0 for name in ("lan", "dc", "wan")),
        "zh_and_en": languages["zh"] >= 10 and languages["en"] >= 10,
        "unique_prompts": len(prompt_digests) == len(cases),
    }
    if not all(requirements.values()):
        raise ValueError("holdout coverage requirements failed: " + json.dumps(
            requirements, sort_keys=True,
        ))
    body = {
        "dataset_id": dataset_id,
        "version": version,
        "case_count": len(cases),
        "category_counts": dict(sorted(categories.items())),
        "profile_counts": dict(sorted(profiles.items())),
        "language_counts": dict(sorted(languages.items())),
        "coverage_requirements": requirements,
        "cases_digest": sha256_json(_case_payload(cases)),
        "privacy": "manifest_contains_no_prompt_or_label",
    }
    return {
        "apiVersion": HOLDOUT_MANIFEST_SCHEMA,
        **body,
        "manifest_digest": sha256_json(body),
    }


def adjudicate(
    cases_path: Path,
    manifest_path: Path,
    first_labels_path: Path,
    second_labels_path: Path,
) -> dict[str, Any]:
    cases: list[HoldoutCase] = _load_jsonl(cases_path, HoldoutCase)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("apiVersion") != HOLDOUT_MANIFEST_SCHEMA
        or set(manifest) != {
            "apiVersion", "dataset_id", "version", "case_count",
            "category_counts", "profile_counts", "language_counts",
            "coverage_requirements", "cases_digest", "privacy", "manifest_digest",
        }
    ):
        raise ValueError("holdout manifest Schema is invalid")
    manifest_body = {
        key: value
        for key, value in manifest.items()
        if key not in {"apiVersion", "manifest_digest"}
    }
    if manifest["manifest_digest"] != sha256_json(manifest_body):
        raise ValueError("holdout manifest integrity check failed")
    cases_digest = sha256_json(_case_payload(cases))
    if manifest["cases_digest"] != cases_digest or manifest["case_count"] != len(cases):
        raise ValueError("holdout cases do not match the sealed manifest")
    first: list[HoldoutLabel] = _load_jsonl(first_labels_path, HoldoutLabel)
    second: list[HoldoutLabel] = _load_jsonl(second_labels_path, HoldoutLabel)
    case_ids = {item.case_id for item in cases}
    first_by_id = {item.case_id: item for item in first}
    second_by_id = {item.case_id: item for item in second}
    if set(first_by_id) != case_ids or set(second_by_id) != case_ids:
        raise ValueError("both reviewers must label the exact sealed case set")
    first_reviewers = {item.reviewer_id for item in first}
    second_reviewers = {item.reviewer_id for item in second}
    if len(first_reviewers) != 1 or len(second_reviewers) != 1:
        raise ValueError("each adjudication file must contain one stable reviewer id")
    if first_reviewers == second_reviewers:
        raise ValueError("holdout adjudication requires two distinct reviewers")
    disagreements: list[str] = []
    consensus: list[dict[str, Any]] = []
    for case_id in sorted(case_ids):
        left = first_by_id[case_id].semantic_payload()
        right = second_by_id[case_id].semantic_payload()
        if left != right:
            disagreements.append(sha256_json({"case_id": case_id}))
        else:
            consensus.append(left)
    ready = not disagreements
    return {
        "apiVersion": "netopyu.io/l1-holdout-adjudication/v1",
        "ready_for_holdout_run": ready,
        "case_count": len(cases),
        "consensus_count": len(consensus),
        "disagreement_count": len(disagreements),
        "disagreement_case_digests": disagreements,
        "reviewer_digests": sorted(
            sha256_json({"reviewer_id": next(iter(reviewers))})
            for reviewers in (first_reviewers, second_reviewers)
        ),
        "cases_digest": cases_digest,
        "sealed_manifest_digest": manifest["manifest_digest"],
        "consensus_labels_digest": sha256_json(consensus) if ready else None,
        "warning": (
            "Distinct reviewer ids are a local process control, not enterprise identity proof."
        ),
    }


def load_consensus(
    cases_path: Path,
    manifest_path: Path,
    first_labels_path: Path,
    second_labels_path: Path,
) -> tuple[tuple[HoldoutCase, ...], dict[str, HoldoutLabel], dict[str, Any]]:
    """Load private consensus labels after the public adjudication gate passes.

    The returned labels are intentionally in-memory only.  Callers must emit
    aggregate metrics and case-id digests rather than prompts or label values.
    """
    report = adjudicate(
        cases_path, manifest_path, first_labels_path, second_labels_path,
    )
    if not report["ready_for_holdout_run"]:
        raise ValueError("holdout reviewer disagreements must be resolved before scoring")
    cases: list[HoldoutCase] = _load_jsonl(cases_path, HoldoutCase)
    first: list[HoldoutLabel] = _load_jsonl(first_labels_path, HoldoutLabel)
    return tuple(cases), {item.case_id: item for item in first}, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    seal = subparsers.add_parser("seal")
    seal.add_argument("cases", type=Path)
    seal.add_argument("--dataset-id", required=True)
    seal.add_argument("--version", required=True)
    review = subparsers.add_parser("adjudicate")
    review.add_argument("cases", type=Path)
    review.add_argument("manifest", type=Path)
    review.add_argument("reviewer_one", type=Path)
    review.add_argument("reviewer_two", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.command == "seal":
        result = seal_cases(
            arguments.cases, dataset_id=arguments.dataset_id, version=arguments.version,
        )
        ok = True
    else:
        result = adjudicate(
            arguments.cases, arguments.manifest,
            arguments.reviewer_one, arguments.reviewer_two,
        )
        ok = bool(result["ready_for_holdout_run"])
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
