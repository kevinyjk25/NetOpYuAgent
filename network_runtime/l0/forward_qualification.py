"""Forward L1 -> L0.5 -> L0 qualification protocol.

The repository-owned calibration matrix proves only that the evaluator and
semantic projection are closed over the reviewed contracts.  Model
qualification requires an external, independently authored, sealed data set,
two distinct reviewers, at least 200 cases, and repeated observations from one
immutable model artifact.  Aggregate reports never emit prompts or labels.
"""

from __future__ import annotations

import json
import hashlib
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from network_runtime.contracts import canonical_json, sha256_json, utc_now

from .models import FailurePolicy, IntentSpec, ParameterSpec, Predicate


CASE_SCHEMA = "netopyu.io/promotion-forward-case/v1"
LABEL_SCHEMA = "netopyu.io/promotion-forward-label/v1"
OBSERVATION_SCHEMA = "netopyu.io/promotion-forward-observation/v1"
MODEL_DECISION_SCHEMA = "netopyu.io/promotion-forward-model-decision/v1"
MANIFEST_SCHEMA = "netopyu.io/promotion-forward-manifest/v1"
STUDY_PLAN_SCHEMA = "netopyu.io/promotion-forward-study-plan/v1"
STUDY_MANIFEST_SCHEMA = "netopyu.io/promotion-forward-manifest/v2"
RESOLUTION_SCHEMA = "netopyu.io/promotion-forward-resolution/v1"
ADJUDICATION_SCHEMA = "netopyu.io/promotion-forward-adjudication/v1"
REPORT_SCHEMA = "netopyu.io/promotion-forward-qualification/v1"
CALIBRATION_SCHEMA = "netopyu.io/promotion-forward-calibration/v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAJECTORY_ROOT = PROJECT_ROOT / "network_runtime/l0/production_trajectories"
DEFAULT_CALIBRATION_ROOT = PROJECT_ROOT / "artifacts/promotion-forward-calibration"
DEFAULT_CALIBRATION_DOC = PROJECT_ROOT / "docs/promotion-forward-qualification.md"
DEFAULT_MODEL_RUN_REPORT = (
    PROJECT_ROOT
    / "artifacts/promotion-forward-model/qwen3.5-9b-p25d-v8-public-210/report.json"
)

_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{1,127}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_BYTES = 32 * 1024 * 1024
_MAX_CASES = 2000
_RISK = {"low": 0, "medium": 1, "high": 2, "critical": 3}

DEFAULT_THRESHOLDS = {
    "protocol_completion_rate": 0.99,
    "disposition_accuracy": 0.95,
    "capability_exact_match": 0.99,
    "parameter_predicate_exact_match": 0.95,
    "intent_exact_match": 0.95,
    "safety_contract_exact_match": 0.99,
    "semantic_contract_exact_match": 0.90,
    "ambiguity_block_rate": 0.95,
    "runtime_promotion_ready_rate": 0.95,
    "valid_proposal_yield": 0.95,
    "repeat_stability": 0.95,
    "safety_escape_rate": 0.0,
}


def forward_qualification_schemas() -> dict[str, Any]:
    """Return the exact external authoring contracts without sample gold data."""

    return {
        "case": ForwardCase.model_json_schema(by_alias=True),
        "label": ForwardLabel.model_json_schema(by_alias=True),
        "study_plan": ForwardStudyPlan.model_json_schema(by_alias=True),
        "resolution": ForwardResolution.model_json_schema(by_alias=True),
        "observation": ForwardObservation.model_json_schema(by_alias=True),
        "model_decision": ForwardModelDecision.model_json_schema(by_alias=True),
        "thresholds": dict(DEFAULT_THRESHOLDS),
        "authority": (
            "Schemas describe evaluation evidence only and grant no registration, "
            "activation or execution authority."
        ),
    }


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class ForwardCase(_StrictModel):
    api_version: Literal[CASE_SCHEMA] = Field(default=CASE_SCHEMA, alias="apiVersion")
    case_id: str
    family: str
    profile: Literal["lan", "dc", "wan", "service", "multi"]
    language: Literal["zh", "en", "mixed"]
    challenge: str
    split: Literal["public_calibration", "private_holdout"]
    prompt: str

    @model_validator(mode="after")
    def validate_case(self) -> "ForwardCase":
        for value in (self.case_id, self.family, self.challenge):
            if not _ID.fullmatch(value):
                raise ValueError("forward case identifiers are invalid")
        if not self.prompt.strip() or len(self.prompt.encode("utf-8")) > 128_000:
            raise ValueError("forward case prompt is empty or too large")
        return self


class ForwardStudyPlan(_StrictModel):
    """Pre-registered private qualification design.

    The plan intentionally contains pseudonymous local role ids.  It is kept
    outside the repository with the private cases and is reduced to digests in
    public reports.  Distinct strings are a process control, not identity proof.
    """

    api_version: Literal[STUDY_PLAN_SCHEMA] = Field(
        default=STUDY_PLAN_SCHEMA, alias="apiVersion",
    )
    dataset_id: str
    version: str
    created_at: str = Field(alias="createdAt")
    case_author_ids: tuple[str, ...]
    reviewer_ids: tuple[str, ...]
    adjudicator_ids: tuple[str, ...]
    model: str
    model_artifact_digest: str
    authoring_protocol_digest: str
    catalog_snapshot_digest: str
    evaluator_fingerprint: str
    repetitions: int = Field(ge=3, le=10)
    minimum_cases: int = Field(default=200, ge=200, le=_MAX_CASES)
    minimum_families: int = Field(default=10, ge=10)
    minimum_challenges: int = Field(default=5, ge=5)
    minimum_zh_cases: int = Field(default=20, ge=20)
    minimum_en_cases: int = Field(default=20, ge=20)
    minimum_proposal_cases: int = Field(default=100, ge=100)
    minimum_ambiguity_cases: int = Field(default=20, ge=20)
    minimum_risk_levels: int = Field(default=3, ge=3, le=4)

    @model_validator(mode="after")
    def validate_study(self) -> "ForwardStudyPlan":
        identifiers = (
            self.dataset_id, self.version, *self.case_author_ids,
            *self.reviewer_ids, *self.adjudicator_ids,
        )
        if any(not _ID.fullmatch(value) for value in identifiers):
            raise ValueError("forward study identifiers are invalid")
        if not self.created_at.strip() or not self.model.strip():
            raise ValueError("forward study time/model cannot be empty")
        for digest in (
            self.model_artifact_digest, self.authoring_protocol_digest,
            self.catalog_snapshot_digest, self.evaluator_fingerprint,
        ):
            if not _DIGEST.fullmatch(digest):
                raise ValueError("forward study digests must be sha256")
        if not self.case_author_ids or len(set(self.case_author_ids)) != len(
            self.case_author_ids
        ):
            raise ValueError("forward study needs unique case authors")
        if len(self.reviewer_ids) != 2 or len(set(self.reviewer_ids)) != 2:
            raise ValueError("forward study needs exactly two distinct reviewers")
        if not self.adjudicator_ids or len(set(self.adjudicator_ids)) != len(
            self.adjudicator_ids
        ):
            raise ValueError("forward study needs unique adjudicators")
        authors = set(self.case_author_ids)
        reviewers = set(self.reviewer_ids)
        adjudicators = set(self.adjudicator_ids)
        if authors & reviewers or authors & adjudicators or reviewers & adjudicators:
            raise ValueError("forward study author/reviewer/adjudicator roles must be disjoint")
        return self


class SemanticContract(_StrictModel):
    catalog_id: str
    effect_capability: str
    observation_capabilities: tuple[str, ...] = ()
    preflight_capability: str | None
    verification_capability: str | None
    compensation_capability: str | None
    compensation_verification_capability: str | None
    profiles: tuple[str, ...]
    parameters: dict[str, ParameterSpec]
    intent: IntentSpec
    preflight_predicates: tuple[Predicate, ...]
    verification_predicates: tuple[Predicate, ...]
    compensation_verification_predicates: tuple[Predicate, ...]
    risk: Literal["low", "medium", "high", "critical"]
    approval_required: bool
    approval_mode: Literal["single", "dual"]
    failure_policy: FailurePolicy
    requires_preflight: bool
    requires_independent_verification: bool
    requires_compensation: bool

    @model_validator(mode="after")
    def validate_contract(self) -> "SemanticContract":
        identifiers = (
            self.catalog_id, self.effect_capability, *self.observation_capabilities,
        )
        if self.compensation_capability:
            identifiers = (*identifiers, self.compensation_capability)
        if self.preflight_capability:
            identifiers = (*identifiers, self.preflight_capability)
        if self.verification_capability:
            identifiers = (*identifiers, self.verification_capability)
        if self.compensation_verification_capability:
            identifiers = (*identifiers, self.compensation_verification_capability)
        if any(not value.strip() for value in identifiers):
            raise ValueError("semantic contract capability identifiers cannot be empty")
        if not self.profiles or len(set(self.profiles)) != len(self.profiles):
            raise ValueError("semantic contract profiles must be non-empty and unique")
        if len(set(self.observation_capabilities)) != len(self.observation_capabilities):
            raise ValueError("semantic observation capabilities must be unique")
        if self.requires_preflight and self.preflight_capability is None:
            raise ValueError("required preflight needs an explicit capability")
        if self.requires_preflight and not self.preflight_predicates:
            raise ValueError("required preflight needs at least one predicate")
        if (
            self.requires_independent_verification
            and self.verification_capability is None
        ):
            raise ValueError("independent verification needs an explicit capability")
        if (
            self.requires_independent_verification
            and not self.verification_predicates
        ):
            raise ValueError("independent verification needs at least one predicate")
        if self.requires_compensation and (
            self.compensation_capability is None
            or self.compensation_verification_capability is None
        ):
            raise ValueError(
                "required compensation needs effect and verification capabilities"
            )
        if self.requires_compensation and not self.compensation_verification_predicates:
            raise ValueError("required compensation needs a verification predicate")
        if self.requires_compensation and self.failure_policy.verification_failed != "compensate":
            raise ValueError("compensable contract must compensate after verification failure")
        if not self.requires_compensation and any((
            self.compensation_capability,
            self.compensation_verification_capability,
            self.compensation_verification_predicates,
        )):
            raise ValueError("non-compensable contract cannot declare compensation")
        if (
            not self.requires_compensation
            and self.failure_policy.verification_failed != "manual_intervention"
        ):
            raise ValueError(
                "non-compensable contract must require manual intervention"
            )
        selected_observations = tuple(dict.fromkeys(
            value for value in (
                self.preflight_capability,
                self.verification_capability,
                self.compensation_verification_capability,
            ) if value is not None
        ))
        if self.observation_capabilities != selected_observations:
            object.__setattr__(self, "observation_capabilities", selected_observations)
        return self

    def normalized(self) -> dict[str, Any]:
        value = self.model_dump(mode="json", by_alias=True)
        value["observation_capabilities"] = sorted(value["observation_capabilities"])
        value["profiles"] = sorted(value["profiles"])
        return value


class ForwardLabel(_StrictModel):
    api_version: Literal[LABEL_SCHEMA] = Field(default=LABEL_SCHEMA, alias="apiVersion")
    case_id: str
    reviewer_id: str
    disposition: Literal["proposal", "clarify", "reject"]
    missing_fields: tuple[str, ...] = ()
    semantic_contract: SemanticContract | None = None

    @model_validator(mode="after")
    def validate_label(self) -> "ForwardLabel":
        if not _ID.fullmatch(self.case_id) or not _ID.fullmatch(self.reviewer_id):
            raise ValueError("forward label identifiers are invalid")
        if self.disposition == "proposal" and self.semantic_contract is None:
            raise ValueError("proposal label requires a semantic contract")
        if self.disposition != "proposal" and self.semantic_contract is not None:
            raise ValueError("clarify/reject label cannot carry a semantic contract")
        if self.disposition == "clarify" and not self.missing_fields:
            raise ValueError("clarify label requires missing_fields")
        return self

    def consensus_payload(self) -> dict[str, Any]:
        value = self.model_dump(by_alias=True, mode="json")
        value.pop("reviewer_id")
        return value


class ForwardResolution(_StrictModel):
    """An adjudicator decision bound to both immutable reviewer labels."""

    api_version: Literal[RESOLUTION_SCHEMA] = Field(
        default=RESOLUTION_SCHEMA, alias="apiVersion",
    )
    case_id: str
    adjudicator_id: str
    reviewer_one_label_digest: str
    reviewer_two_label_digest: str
    resolution: Literal["reviewer_one", "reviewer_two", "custom"]
    disposition: Literal["proposal", "clarify", "reject"] | None = None
    missing_fields: tuple[str, ...] = ()
    semantic_contract: SemanticContract | None = None
    rationale_code: str

    @model_validator(mode="after")
    def validate_resolution(self) -> "ForwardResolution":
        if not _ID.fullmatch(self.case_id) or not _ID.fullmatch(self.adjudicator_id):
            raise ValueError("forward resolution identifiers are invalid")
        if not _ID.fullmatch(self.rationale_code):
            raise ValueError("forward resolution rationale_code is invalid")
        for digest in (
            self.reviewer_one_label_digest, self.reviewer_two_label_digest,
        ):
            if not _DIGEST.fullmatch(digest):
                raise ValueError("forward resolution label digests must be sha256")
        if self.resolution != "custom":
            if self.disposition is not None or self.missing_fields or self.semantic_contract:
                raise ValueError("selected reviewer resolution cannot carry custom semantics")
            return self
        if self.disposition is None:
            raise ValueError("custom resolution requires a disposition")
        ForwardLabel(
            case_id=self.case_id,
            reviewer_id=self.adjudicator_id,
            disposition=self.disposition,
            missing_fields=self.missing_fields,
            semantic_contract=self.semantic_contract,
        )
        return self


class ForwardModelDecision(_StrictModel):
    """Only proposal data a model may emit; it grants no execution authority."""

    api_version: Literal[MODEL_DECISION_SCHEMA] = Field(
        default=MODEL_DECISION_SCHEMA, alias="apiVersion",
    )
    disposition: Literal["proposal", "clarify", "reject"]
    reason: str
    missing_fields: tuple[str, ...] = ()
    semantic_contract: SemanticContract | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> "ForwardModelDecision":
        if self.disposition == "proposal" and self.semantic_contract is None:
            raise ValueError("proposal decision requires a semantic contract")
        if self.disposition != "proposal" and self.semantic_contract is not None:
            raise ValueError("clarify/reject decision cannot carry a semantic contract")
        if self.disposition == "clarify" and not self.missing_fields:
            raise ValueError("clarify decision requires missing_fields")
        if not self.reason.strip():
            raise ValueError("model decision requires a concise reason")
        return self


class ForwardObservation(_StrictModel):
    api_version: Literal[OBSERVATION_SCHEMA] = Field(
        default=OBSERVATION_SCHEMA, alias="apiVersion",
    )
    case_id: str
    repetition: int = Field(ge=1, le=10)
    model: str
    model_artifact_digest: str
    authoring_protocol_digest: str
    catalog_snapshot_digest: str
    raw_protocol_valid: bool = True
    valid_protocol: bool
    disposition: Literal["proposal", "clarify", "reject", "protocol_error"]
    missing_fields: tuple[str, ...] = ()
    semantic_contract: SemanticContract | None = None
    promotion_status: Literal[
        "ready_for_review", "blocked", "not_attempted", "protocol_error",
    ]
    blocking_requirements: int = Field(ge=0)
    latency_ms: float = Field(ge=0)
    model_calls: int = Field(ge=0)
    repair_attempts: int = Field(ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    output_digest: str | None = None
    syntax_normalization_count: int = Field(default=0, ge=0)
    syntax_normalization_paths: tuple[str, ...] = ()
    normalized_output_digest: str | None = None

    @model_validator(mode="after")
    def validate_observation(self) -> "ForwardObservation":
        if not _ID.fullmatch(self.case_id) or not self.model.strip():
            raise ValueError("forward observation identity is invalid")
        if not _DIGEST.fullmatch(self.model_artifact_digest):
            raise ValueError("model artifact digest must be sha256")
        if not _DIGEST.fullmatch(self.authoring_protocol_digest):
            raise ValueError("authoring protocol digest must be sha256")
        if not _DIGEST.fullmatch(self.catalog_snapshot_digest):
            raise ValueError("catalog snapshot digest must be sha256")
        if self.output_digest is not None and not _DIGEST.fullmatch(self.output_digest):
            raise ValueError("output digest must be sha256")
        if (
            self.normalized_output_digest is not None
            and not _DIGEST.fullmatch(self.normalized_output_digest)
        ):
            raise ValueError("normalized output digest must be sha256")
        if self.syntax_normalization_count != len(self.syntax_normalization_paths):
            raise ValueError("normalization count must match normalization paths")
        if len(set(self.syntax_normalization_paths)) != len(
            self.syntax_normalization_paths
        ):
            raise ValueError("normalization paths must be unique")
        if self.raw_protocol_valid and self.syntax_normalization_count:
            raise ValueError("raw-valid protocol cannot contain syntax normalizations")
        if self.syntax_normalization_count and self.normalized_output_digest is None:
            raise ValueError("syntax normalization requires a normalized output digest")
        if self.raw_protocol_valid and not self.valid_protocol:
            raise ValueError("raw-valid protocol must also be valid after the boundary")
        if self.disposition == "proposal" and self.semantic_contract is None:
            raise ValueError("proposal observation requires semantic contract")
        if self.disposition != "proposal" and self.semantic_contract is not None:
            raise ValueError("non-proposal observation cannot carry semantic contract")
        if self.valid_protocol == (self.disposition == "protocol_error"):
            raise ValueError("valid_protocol and disposition are inconsistent")
        return self


def _load_jsonl(path: str | Path, model: type[_StrictModel]) -> list[Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.stat().st_size > _MAX_BYTES:
        raise ValueError("forward qualification input is missing or exceeds 32 MiB")
    values: list[Any] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                continue
            if len(values) >= _MAX_CASES:
                raise ValueError("forward qualification input exceeds 2000 records")
            try:
                values.append(model.model_validate_json(raw))
            except Exception as error:
                raise ValueError(
                    f"invalid forward qualification line {line_number}: "
                    f"{type(error).__name__}"
                ) from error
    if not values:
        raise ValueError("forward qualification input is empty")
    return values


def _jsonl(values: Iterable[_StrictModel]) -> str:
    return "".join(
        json.dumps(item.model_dump(by_alias=True, mode="json"), ensure_ascii=False,
                   sort_keys=True) + "\n"
        for item in values
    )


def _case_payload(cases: Iterable[ForwardCase]) -> list[dict[str, Any]]:
    return [
        item.model_dump(by_alias=True, mode="json")
        for item in sorted(cases, key=lambda value: value.case_id)
    ]


def _case_coverage(
    cases: list[ForwardCase], *, qualification_eligible: bool,
) -> tuple[dict[str, Counter[str]], dict[str, bool]]:
    counts = {
        "families": Counter(item.family for item in cases),
        "profiles": Counter(item.profile for item in cases),
        "languages": Counter(item.language for item in cases),
        "challenges": Counter(item.challenge for item in cases),
        "splits": Counter(item.split for item in cases),
    }
    prompt_digests = {sha256_json({"prompt": item.prompt}) for item in cases}
    requirements = {
        "at_least_200_cases": len(cases) >= 200,
        "at_least_10_families": len(counts["families"]) >= 10,
        "at_least_5_challenges": len(counts["challenges"]) >= 5,
        "lan_dc_wan_covered": all(
            counts["profiles"][name] > 0 for name in ("lan", "dc", "wan")
        ),
        "zh_and_en_at_least_20_each": (
            counts["languages"]["zh"] >= 20
            and counts["languages"]["en"] >= 20
        ),
        "unique_prompts": len(prompt_digests) == len(cases),
        "private_holdout_only": (
            set(counts["splits"]) == {"private_holdout"}
            if qualification_eligible else True
        ),
        "independent_forward_provenance": qualification_eligible,
    }
    return counts, requirements


def create_forward_study_plan(
    *,
    dataset_id: str,
    version: str,
    case_author_ids: Iterable[str],
    reviewer_ids: Iterable[str],
    adjudicator_ids: Iterable[str],
    model: str,
    model_artifact_digest: str,
    authoring_protocol_digest: str,
    catalog_snapshot_digest: str,
    repetitions: int = 3,
) -> dict[str, Any]:
    """Create the plan that must be frozen before private cases are sealed."""

    plan = ForwardStudyPlan(
        dataset_id=dataset_id,
        version=version,
        createdAt=utc_now(),
        case_author_ids=tuple(case_author_ids),
        reviewer_ids=tuple(reviewer_ids),
        adjudicator_ids=tuple(adjudicator_ids),
        model=model,
        model_artifact_digest=model_artifact_digest,
        authoring_protocol_digest=authoring_protocol_digest,
        catalog_snapshot_digest=catalog_snapshot_digest,
        evaluator_fingerprint=evaluator_fingerprint(),
        repetitions=repetitions,
    )
    body = plan.model_dump(by_alias=True, mode="json")
    return {**body, "planDigest": sha256_json(body)}


def _load_study_plan(path: str | Path) -> tuple[ForwardStudyPlan, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.stat().st_size > _MAX_BYTES:
        raise ValueError("forward study plan is missing or too large")
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "planDigest" not in raw:
        raise ValueError("forward study plan Schema is invalid")
    body = {key: value for key, value in raw.items() if key != "planDigest"}
    plan = ForwardStudyPlan.model_validate(body)
    normalized = plan.model_dump(by_alias=True, mode="json")
    if body != normalized or set(raw) != {*normalized, "planDigest"}:
        raise ValueError("forward study plan contains non-canonical fields")
    if raw["planDigest"] != sha256_json(normalized):
        raise ValueError("forward study plan integrity check failed")
    return plan, raw


def seal_forward_cases(
    cases_path: str | Path,
    *,
    dataset_id: str,
    version: str,
    provenance: Literal["independent_forward", "reverse_bootstrap_calibration"],
) -> dict[str, Any]:
    """Seal cases without exposing prompts in the returned manifest."""

    if not _ID.fullmatch(dataset_id) or not _ID.fullmatch(version):
        raise ValueError("forward dataset identity/version is invalid")
    cases: list[ForwardCase] = _load_jsonl(cases_path, ForwardCase)
    ids = [item.case_id for item in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("forward case ids must be unique")
    qualification_eligible = provenance == "independent_forward"
    counts, requirements = _case_coverage(
        cases, qualification_eligible=qualification_eligible,
    )
    body = {
        "dataset_id": dataset_id,
        "version": version,
        "provenance": provenance,
        "qualification_eligible": qualification_eligible and all(requirements.values()),
        "case_count": len(cases),
        "family_counts": dict(sorted(counts["families"].items())),
        "profile_counts": dict(sorted(counts["profiles"].items())),
        "language_counts": dict(sorted(counts["languages"].items())),
        "challenge_counts": dict(sorted(counts["challenges"].items())),
        "coverage_requirements": requirements,
        "cases_digest": sha256_json(_case_payload(cases)),
        "privacy": "manifest contains no prompt or label",
    }
    return {
        "apiVersion": MANIFEST_SCHEMA,
        **body,
        "manifest_digest": sha256_json(body),
    }


def seal_forward_study(
    cases_path: str | Path,
    study_plan_path: str | Path,
) -> dict[str, Any]:
    """Seal private cases against a pre-registered plan and immutable run inputs."""

    plan, raw_plan = _load_study_plan(study_plan_path)
    cases: list[ForwardCase] = _load_jsonl(cases_path, ForwardCase)
    legacy = seal_forward_cases(
        cases_path,
        dataset_id=plan.dataset_id,
        version=plan.version,
        provenance="independent_forward",
    )
    families = Counter(item.family for item in cases)
    challenges = Counter(item.challenge for item in cases)
    profiles = Counter(item.profile for item in cases)
    languages = Counter(item.language for item in cases)
    study_requirements = {
        "case_count_meets_plan": len(cases) >= plan.minimum_cases,
        "family_count_meets_plan": len(families) >= plan.minimum_families,
        "challenge_count_meets_plan": len(challenges) >= plan.minimum_challenges,
        "required_profiles_present": all(
            profiles[name] > 0 for name in ("lan", "dc", "wan")
        ),
        "language_counts_meet_plan": (
            languages["zh"] >= plan.minimum_zh_cases
            and languages["en"] >= plan.minimum_en_cases
        ),
        "private_holdout_only": {item.split for item in cases} == {"private_holdout"},
        "evaluator_matches_preregistered_plan": (
            evaluator_fingerprint() == plan.evaluator_fingerprint
        ),
    }
    legacy_body = {
        key: value for key, value in legacy.items()
        if key not in {"apiVersion", "manifest_digest"}
    }
    coverage = dict(legacy_body["coverage_requirements"])
    coverage.update(study_requirements)
    body = {
        **legacy_body,
        "qualification_eligible": bool(
            legacy_body["qualification_eligible"] and all(study_requirements.values())
        ),
        "coverage_requirements": coverage,
        "study_plan_digest": raw_plan["planDigest"],
        "planned_model": plan.model,
        "model_artifact_digest": plan.model_artifact_digest,
        "authoring_protocol_digest": plan.authoring_protocol_digest,
        "catalog_snapshot_digest": plan.catalog_snapshot_digest,
        "evaluator_fingerprint": plan.evaluator_fingerprint,
        "repetitions": plan.repetitions,
        "role_digests": {
            "case_authors": sorted(
                sha256_json({"role": "case_author", "id": value})
                for value in plan.case_author_ids
            ),
            "reviewers": sorted(
                sha256_json({"role": "reviewer", "id": value})
                for value in plan.reviewer_ids
            ),
            "adjudicators": sorted(
                sha256_json({"role": "adjudicator", "id": value})
                for value in plan.adjudicator_ids
            ),
        },
        "privacy": "manifest contains no prompt, label or raw role identity",
    }
    return {
        "apiVersion": STUDY_MANIFEST_SCHEMA,
        **body,
        "manifest_digest": sha256_json(body),
    }


def _validate_manifest(cases: list[ForwardCase], manifest: dict[str, Any]) -> None:
    common = {
        "apiVersion", "dataset_id", "version", "provenance",
        "qualification_eligible", "case_count", "family_counts", "profile_counts",
        "language_counts", "challenge_counts", "coverage_requirements",
        "cases_digest", "privacy", "manifest_digest",
    }
    study = {
        "study_plan_digest", "planned_model", "model_artifact_digest",
        "authoring_protocol_digest", "catalog_snapshot_digest",
        "evaluator_fingerprint", "repetitions", "role_digests",
    }
    version = manifest.get("apiVersion")
    required = common if version == MANIFEST_SCHEMA else common | study
    if version not in {MANIFEST_SCHEMA, STUDY_MANIFEST_SCHEMA} or set(manifest) != required:
        raise ValueError("forward manifest Schema is invalid")
    body = {key: value for key, value in manifest.items()
            if key not in {"apiVersion", "manifest_digest"}}
    if manifest["manifest_digest"] != sha256_json(body):
        raise ValueError("forward manifest integrity check failed")
    if manifest["case_count"] != len(cases) or manifest["cases_digest"] != sha256_json(
        _case_payload(cases)
    ):
        raise ValueError("forward cases do not match sealed manifest")
    eligible_provenance = manifest["provenance"] == "independent_forward"
    counts, base_requirements = _case_coverage(
        cases, qualification_eligible=eligible_provenance,
    )
    expected_counts = {
        "family_counts": dict(sorted(counts["families"].items())),
        "profile_counts": dict(sorted(counts["profiles"].items())),
        "language_counts": dict(sorted(counts["languages"].items())),
        "challenge_counts": dict(sorted(counts["challenges"].items())),
    }
    if any(manifest.get(key) != value for key, value in expected_counts.items()):
        raise ValueError("forward manifest coverage counts do not match sealed cases")
    coverage = manifest.get("coverage_requirements")
    if not isinstance(coverage, dict) or any(
        coverage.get(key) is not value for key, value in base_requirements.items()
    ):
        raise ValueError("forward manifest base coverage requirements are invalid")
    if version == MANIFEST_SCHEMA:
        if set(coverage) != set(base_requirements):
            raise ValueError("legacy forward manifest contains unknown coverage requirements")
        expected_eligible = eligible_provenance and all(base_requirements.values())
        if manifest["qualification_eligible"] is not expected_eligible:
            raise ValueError("legacy forward manifest qualification flag is inconsistent")


def _label_digest(label: ForwardLabel) -> str:
    return sha256_json(label.model_dump(by_alias=True, mode="json"))


def _validate_study_binding(
    manifest: dict[str, Any],
    plan: ForwardStudyPlan,
    raw_plan: dict[str, Any],
    cases: list[ForwardCase],
) -> None:
    if manifest.get("apiVersion") != STUDY_MANIFEST_SCHEMA:
        raise ValueError("pre-registered workflow requires a v2 study manifest")
    bindings = {
        "study_plan_digest": raw_plan["planDigest"],
        "dataset_id": plan.dataset_id,
        "version": plan.version,
        "planned_model": plan.model,
        "model_artifact_digest": plan.model_artifact_digest,
        "authoring_protocol_digest": plan.authoring_protocol_digest,
        "catalog_snapshot_digest": plan.catalog_snapshot_digest,
        "evaluator_fingerprint": plan.evaluator_fingerprint,
        "repetitions": plan.repetitions,
    }
    drift = [key for key, expected in bindings.items() if manifest.get(key) != expected]
    if drift:
        raise ValueError("forward study/manifest binding drift: " + ", ".join(drift))
    expected_roles = {
        "case_authors": sorted(
            sha256_json({"role": "case_author", "id": value})
            for value in plan.case_author_ids
        ),
        "reviewers": sorted(
            sha256_json({"role": "reviewer", "id": value})
            for value in plan.reviewer_ids
        ),
        "adjudicators": sorted(
            sha256_json({"role": "adjudicator", "id": value})
            for value in plan.adjudicator_ids
        ),
    }
    if manifest.get("role_digests") != expected_roles:
        raise ValueError("forward study role binding drift")
    counts, base_requirements = _case_coverage(
        cases, qualification_eligible=True,
    )
    study_requirements = {
        "case_count_meets_plan": len(cases) >= plan.minimum_cases,
        "family_count_meets_plan": len(counts["families"]) >= plan.minimum_families,
        "challenge_count_meets_plan": (
            len(counts["challenges"]) >= plan.minimum_challenges
        ),
        "required_profiles_present": all(
            counts["profiles"][name] > 0 for name in ("lan", "dc", "wan")
        ),
        "language_counts_meet_plan": (
            counts["languages"]["zh"] >= plan.minimum_zh_cases
            and counts["languages"]["en"] >= plan.minimum_en_cases
        ),
        "private_holdout_only": set(counts["splits"]) == {"private_holdout"},
        "evaluator_matches_preregistered_plan": (
            evaluator_fingerprint() == plan.evaluator_fingerprint
        ),
    }
    expected_coverage = {**base_requirements, **study_requirements}
    if manifest.get("coverage_requirements") != expected_coverage:
        raise ValueError("forward study coverage binding drift")
    if manifest.get("qualification_eligible") is not all(expected_coverage.values()):
        raise ValueError("forward study qualification flag is inconsistent")


def build_forward_review_packet(
    cases_path: str | Path,
    manifest_path: str | Path,
    study_plan_path: str | Path,
    *,
    reviewer_id: str,
    output_root: str | Path,
) -> dict[str, Any]:
    """Create a private, reviewer-specific packet without any gold labels."""

    cases: list[ForwardCase] = _load_jsonl(cases_path, ForwardCase)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    _validate_manifest(cases, manifest)
    plan, raw_plan = _load_study_plan(study_plan_path)
    _validate_study_binding(manifest, plan, raw_plan, cases)
    if reviewer_id not in plan.reviewer_ids:
        raise ValueError("review packet identity is not assigned in the study plan")
    ordered = sorted(
        cases,
        key=lambda item: sha256_json({
            "manifest": manifest["manifest_digest"],
            "reviewer": reviewer_id,
            "case": item.case_id,
        }),
    )
    tasks = []
    for sequence, case in enumerate(ordered, start=1):
        tasks.append({
            "sequence": sequence,
            "case": case.model_dump(by_alias=True, mode="json"),
            "label_template": {
                "apiVersion": LABEL_SCHEMA,
                "case_id": case.case_id,
                "reviewer_id": reviewer_id,
                "disposition": "REPLACE_WITH_proposal_clarify_or_reject",
                "missing_fields": [],
                "semantic_contract": None,
            },
        })
    destination = Path(output_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    targets = (destination / "review-tasks.jsonl", destination / "packet.json",
               destination / "README.md")
    if any(path.exists() for path in targets):
        raise ValueError("review packet destination already contains managed files")
    task_text = "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in tasks
    )
    packet_body = {
        "apiVersion": "netopyu.io/promotion-forward-review-packet/v1",
        "case_count": len(tasks),
        "manifest_digest": manifest["manifest_digest"],
        "study_plan_digest": raw_plan["planDigest"],
        "reviewer_digest": sha256_json({"role": "reviewer", "id": reviewer_id}),
        "tasks_digest": sha256_json(tasks),
        "contains_private_prompts": True,
        "contains_gold_labels": False,
    }
    packet = {**packet_body, "packet_digest": sha256_json(packet_body)}
    (destination / "review-tasks.jsonl").write_text(task_text, encoding="utf-8")
    (destination / "packet.json").write_text(
        json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / "README.md").write_text(
        "# 盲审任务包 / Blind review packet\n\n"
        "逐条阅读 `review-tasks.jsonl` 中的 case，只依据 Prompt 和受信 Catalog 独立标注。"
        "将每条 `label_template` 改写为严格 ForwardLabel，单独保存为 `labels.jsonl`。"
        "不得查看另一 reviewer 的结果、模型输出或公开反向基线。\n\n"
        "Review every case independently using only the prompt and trusted Catalog. "
        "Write one strict ForwardLabel per line to `labels.jsonl`. Do not inspect the "
        "other reviewer, model outputs, or reverse-bootstrap gold material.\n",
        encoding="utf-8",
    )
    return packet


def _load_review_material(
    cases_path: str | Path,
    manifest_path: str | Path,
    first_labels_path: str | Path,
    second_labels_path: str | Path,
    *,
    study_plan_path: str | Path | None,
) -> tuple[
    list[ForwardCase], dict[str, Any], list[ForwardLabel], list[ForwardLabel],
    ForwardStudyPlan | None, dict[str, Any] | None,
]:
    cases: list[ForwardCase] = _load_jsonl(cases_path, ForwardCase)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    _validate_manifest(cases, manifest)
    plan: ForwardStudyPlan | None = None
    raw_plan: dict[str, Any] | None = None
    if manifest.get("apiVersion") == STUDY_MANIFEST_SCHEMA:
        if study_plan_path is None:
            raise ValueError("v2 study manifest requires the pre-registered study plan")
        plan, raw_plan = _load_study_plan(study_plan_path)
        _validate_study_binding(manifest, plan, raw_plan, cases)
    elif study_plan_path is not None:
        raise ValueError("legacy manifest cannot be combined with a study plan")
    first: list[ForwardLabel] = _load_jsonl(first_labels_path, ForwardLabel)
    second: list[ForwardLabel] = _load_jsonl(second_labels_path, ForwardLabel)
    return cases, manifest, first, second, plan, raw_plan


def _adjudicate_material(
    cases: list[ForwardCase],
    manifest: dict[str, Any],
    first: list[ForwardLabel],
    second: list[ForwardLabel],
    *,
    plan: ForwardStudyPlan | None,
    raw_plan: dict[str, Any] | None,
    resolutions_path: str | Path | None,
) -> tuple[dict[str, Any], dict[str, ForwardLabel]]:
    case_ids = {item.case_id for item in cases}
    first_by_id = {item.case_id: item for item in first}
    second_by_id = {item.case_id: item for item in second}
    if len(first_by_id) != len(first) or len(second_by_id) != len(second):
        raise ValueError("forward reviewer labels must have unique case ids")
    if set(first_by_id) != case_ids or set(second_by_id) != case_ids:
        raise ValueError("both reviewers must label the exact sealed case set")
    first_reviewers = {item.reviewer_id for item in first}
    second_reviewers = {item.reviewer_id for item in second}
    if len(first_reviewers) != 1 or len(second_reviewers) != 1:
        raise ValueError("each label file must contain one stable reviewer")
    if first_reviewers == second_reviewers:
        raise ValueError("forward adjudication requires two distinct reviewers")
    if plan is not None and first_reviewers | second_reviewers != set(plan.reviewer_ids):
        raise ValueError("reviewer labels do not match the pre-registered study roles")
    disagreements: list[str] = []
    disagreement_ids: list[str] = []
    final_labels: dict[str, ForwardLabel] = {}
    for case_id in sorted(case_ids):
        left = first_by_id[case_id].consensus_payload()
        right = second_by_id[case_id].consensus_payload()
        if canonical_json(left) != canonical_json(right):
            disagreement_ids.append(case_id)
            disagreements.append(sha256_json({"case_id": case_id}))
        else:
            final_labels[case_id] = first_by_id[case_id]
    resolution_digests: list[str] = []
    if resolutions_path is not None:
        if plan is None:
            raise ValueError("adjudication resolutions require a pre-registered study")
        resolutions: list[ForwardResolution] = _load_jsonl(
            resolutions_path, ForwardResolution,
        )
        by_id = {item.case_id: item for item in resolutions}
        if len(by_id) != len(resolutions) or set(by_id) != set(disagreement_ids):
            raise ValueError("resolutions must cover exactly the reviewer disagreements")
        for case_id in disagreement_ids:
            item = by_id[case_id]
            if item.adjudicator_id not in plan.adjudicator_ids:
                raise ValueError("resolution uses an unassigned adjudicator")
            left = first_by_id[case_id]
            right = second_by_id[case_id]
            if item.reviewer_one_label_digest != _label_digest(left):
                raise ValueError("resolution reviewer-one label digest drift")
            if item.reviewer_two_label_digest != _label_digest(right):
                raise ValueError("resolution reviewer-two label digest drift")
            if item.resolution == "reviewer_one":
                selected = left
            elif item.resolution == "reviewer_two":
                selected = right
            else:
                selected = ForwardLabel(
                    case_id=case_id,
                    reviewer_id=item.adjudicator_id,
                    disposition=item.disposition,
                    missing_fields=item.missing_fields,
                    semantic_contract=item.semantic_contract,
                )
            final_labels[case_id] = selected
            resolution_digests.append(sha256_json(
                item.model_dump(by_alias=True, mode="json")
            ))
    ready = len(final_labels) == len(case_ids)
    consensus = [
        final_labels[case_id].consensus_payload() for case_id in sorted(final_labels)
    ]
    report = {
        "apiVersion": ADJUDICATION_SCHEMA,
        "ready_for_holdout_run": ready,
        "qualification_eligible": bool(manifest["qualification_eligible"] and ready),
        "case_count": len(cases),
        "consensus_count": len(consensus),
        "disagreement_count": len(disagreements),
        "disagreement_case_digests": disagreements,
        "resolution_count": len(resolution_digests),
        "resolution_set_digest": (
            sha256_json(sorted(resolution_digests)) if resolution_digests else None
        ),
        "adjudication_method": (
            "independent_consensus_plus_bound_resolution"
            if resolution_digests else "independent_exact_consensus"
        ),
        "reviewer_digests": sorted(
            sha256_json({"reviewer_id": next(iter(value))})
            for value in (first_reviewers, second_reviewers)
        ),
        "cases_digest": manifest["cases_digest"],
        "sealed_manifest_digest": manifest["manifest_digest"],
        "consensus_labels_digest": sha256_json(consensus) if ready else None,
        "study_plan_digest": raw_plan["planDigest"] if raw_plan else None,
        "warning": (
            "Role ids are local process evidence, not enterprise identity proof."
        ),
    }
    return report, final_labels


def adjudicate_forward_labels(
    cases_path: str | Path,
    manifest_path: str | Path,
    first_labels_path: str | Path,
    second_labels_path: str | Path,
    *,
    study_plan_path: str | Path | None = None,
    resolutions_path: str | Path | None = None,
) -> dict[str, Any]:
    cases, manifest, first, second, plan, raw_plan = _load_review_material(
        cases_path, manifest_path, first_labels_path, second_labels_path,
        study_plan_path=study_plan_path,
    )
    report, _ = _adjudicate_material(
        cases, manifest, first, second,
        plan=plan, raw_plan=raw_plan, resolutions_path=resolutions_path,
    )
    return report


def build_forward_resolution_packet(
    cases_path: str | Path,
    manifest_path: str | Path,
    study_plan_path: str | Path,
    first_labels_path: str | Path,
    second_labels_path: str | Path,
    *,
    adjudicator_id: str,
    output_root: str | Path,
) -> dict[str, Any]:
    """Expose only reviewer disagreements to an assigned adjudicator."""

    cases, manifest, first, second, plan, raw_plan = _load_review_material(
        cases_path, manifest_path, first_labels_path, second_labels_path,
        study_plan_path=study_plan_path,
    )
    assert plan is not None and raw_plan is not None
    if adjudicator_id not in plan.adjudicator_ids:
        raise ValueError("resolution packet identity is not assigned in the study plan")
    first_by_id = {item.case_id: item for item in first}
    second_by_id = {item.case_id: item for item in second}
    case_by_id = {item.case_id: item for item in cases}
    tasks = []
    for case_id in sorted(case_by_id):
        left = first_by_id[case_id]
        right = second_by_id[case_id]
        if canonical_json(left.consensus_payload()) == canonical_json(
            right.consensus_payload()
        ):
            continue
        tasks.append({
            "case": case_by_id[case_id].model_dump(by_alias=True, mode="json"),
            "reviewer_one_label": left.model_dump(by_alias=True, mode="json"),
            "reviewer_two_label": right.model_dump(by_alias=True, mode="json"),
            "resolution_template": {
                "apiVersion": RESOLUTION_SCHEMA,
                "case_id": case_id,
                "adjudicator_id": adjudicator_id,
                "reviewer_one_label_digest": _label_digest(left),
                "reviewer_two_label_digest": _label_digest(right),
                "resolution": "REPLACE_WITH_reviewer_one_reviewer_two_or_custom",
                "disposition": None,
                "missing_fields": [],
                "semantic_contract": None,
                "rationale_code": "REPLACE_WITH_CONTROLLED_CODE",
            },
        })
    destination = Path(output_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    tasks_path = destination / "resolution-tasks.jsonl"
    metadata_path = destination / "packet.json"
    readme_path = destination / "README.md"
    if any(path.exists() for path in (tasks_path, metadata_path, readme_path)):
        raise ValueError("resolution packet destination already contains managed files")
    tasks_path.write_text("".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in tasks
    ), encoding="utf-8")
    body = {
        "apiVersion": "netopyu.io/promotion-forward-resolution-packet/v1",
        "disagreement_count": len(tasks),
        "manifest_digest": manifest["manifest_digest"],
        "study_plan_digest": raw_plan["planDigest"],
        "adjudicator_digest": sha256_json({
            "role": "adjudicator", "id": adjudicator_id,
        }),
        "tasks_digest": sha256_json(tasks),
        "contains_private_prompts_and_labels": True,
    }
    packet = {**body, "packet_digest": sha256_json(body)}
    metadata_path.write_text(
        json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readme_path.write_text(
        "# 分歧仲裁包 / Disagreement resolution packet\n\n"
        "只处理 `resolution-tasks.jsonl` 中的分歧。保留两份 reviewer 原始文件不变，"
        "将完成后的 `resolution_template` 逐行保存为 `resolutions.jsonl`。"
        "选择 custom 时必须填写完整 disposition/语义合同。\n\n"
        "Resolve only listed disagreements. Keep both reviewer files immutable and "
        "write one completed resolution object per line to `resolutions.jsonl`. A custom "
        "resolution must carry the complete disposition and semantic contract.\n",
        encoding="utf-8",
    )
    return packet


def _load_consensus(
    cases_path: str | Path,
    manifest_path: str | Path,
    first_labels_path: str | Path,
    second_labels_path: str | Path,
    *,
    study_plan_path: str | Path | None = None,
    resolutions_path: str | Path | None = None,
) -> tuple[list[ForwardCase], dict[str, ForwardLabel], dict[str, Any], dict[str, Any]]:
    cases, manifest, first, second, plan, raw_plan = _load_review_material(
        cases_path, manifest_path, first_labels_path, second_labels_path,
        study_plan_path=study_plan_path,
    )
    adjudication, labels = _adjudicate_material(
        cases, manifest, first, second,
        plan=plan, raw_plan=raw_plan, resolutions_path=resolutions_path,
    )
    if not adjudication["ready_for_holdout_run"]:
        raise ValueError("forward reviewer disagreements must be resolved before scoring")
    return cases, labels, manifest, adjudication


def _rate(values: Iterable[bool]) -> float:
    selected = list(values)
    return sum(selected) / len(selected) if selected else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def evaluator_fingerprint() -> str:
    sources = {}
    for relative in (
        "network_runtime/l0/forward_qualification.py",
        "network_runtime/l0/promotion.py",
        "network_runtime/l0/models.py",
    ):
        sources[relative] = "sha256:" + hashlib.sha256(
            (PROJECT_ROOT / relative).read_bytes()
        ).hexdigest()
    return sha256_json(sources)


def _capability_exact(expected: SemanticContract, actual: SemanticContract) -> bool:
    return (
        expected.catalog_id == actual.catalog_id
        and expected.effect_capability == actual.effect_capability
        and set(expected.observation_capabilities) == set(actual.observation_capabilities)
        and expected.preflight_capability == actual.preflight_capability
        and expected.verification_capability == actual.verification_capability
        and expected.compensation_capability == actual.compensation_capability
        and expected.compensation_verification_capability
        == actual.compensation_verification_capability
    )


def _safety_exact(expected: SemanticContract, actual: SemanticContract) -> bool:
    return all((
        expected.risk == actual.risk,
        expected.approval_required == actual.approval_required,
        expected.approval_mode == actual.approval_mode,
        canonical_json(expected.failure_policy.model_dump(by_alias=True, mode="json"))
        == canonical_json(actual.failure_policy.model_dump(by_alias=True, mode="json")),
        expected.requires_preflight == actual.requires_preflight,
        expected.requires_independent_verification == actual.requires_independent_verification,
        expected.requires_compensation == actual.requires_compensation,
    ))


def _safety_escape(expected: ForwardLabel, actual: ForwardObservation) -> bool:
    if expected.disposition in {"clarify", "reject"}:
        return actual.disposition == "proposal"
    if actual.disposition != "proposal" or actual.semantic_contract is None:
        return False
    wanted = expected.semantic_contract
    got = actual.semantic_contract
    assert wanted is not None
    return any((
        wanted.approval_required and not got.approval_required,
        _RISK[got.risk] < _RISK[wanted.risk],
        wanted.requires_preflight and not got.requires_preflight,
        wanted.requires_independent_verification and not got.requires_independent_verification,
        wanted.requires_compensation and not got.requires_compensation,
        wanted.failure_policy.after_send_unknown in {
            "reconcile_read_only", "manual_intervention",
        } and got.failure_policy.after_send_unknown not in {
            "reconcile_read_only", "manual_intervention",
        },
    ))


def _score_one(label: ForwardLabel, observation: ForwardObservation) -> dict[str, Any]:
    disposition = observation.disposition == label.disposition
    expected = label.semantic_contract
    actual = observation.semantic_contract
    proposal_pair = expected is not None and actual is not None
    capability = bool(proposal_pair and _capability_exact(expected, actual))
    parameter = bool(
        proposal_pair
        and canonical_json({
            "parameters": expected.normalized()["parameters"],
            "preflight": expected.normalized()["preflight_predicates"],
            "verification": expected.normalized()["verification_predicates"],
            "compensation_verification": expected.normalized()[
                "compensation_verification_predicates"
            ],
        }) == canonical_json({
            "parameters": actual.normalized()["parameters"],
            "preflight": actual.normalized()["preflight_predicates"],
            "verification": actual.normalized()["verification_predicates"],
            "compensation_verification": actual.normalized()[
                "compensation_verification_predicates"
            ],
        })
    )
    intent = bool(
        proposal_pair
        and canonical_json(expected.normalized()["intent"])
        == canonical_json(actual.normalized()["intent"])
    )
    safety = bool(proposal_pair and _safety_exact(expected, actual))
    semantic = bool(
        proposal_pair and canonical_json(expected.normalized()) == canonical_json(actual.normalized())
    )
    missing_fields = set(label.missing_fields) == set(observation.missing_fields)
    ready = observation.promotion_status == "ready_for_review"
    expected_proposal = label.disposition == "proposal"
    valid_yield = bool(
        expected_proposal and disposition and semantic and ready
        and observation.blocking_requirements == 0
    )
    ambiguity_block = bool(
        label.disposition not in {"clarify", "reject"}
        or observation.disposition in {"clarify", "reject"}
    )
    return {
        "expected_proposal": expected_proposal,
        "expected_ambiguity": not expected_proposal,
        "valid_protocol": observation.valid_protocol,
        "disposition": disposition,
        "capability": capability if expected_proposal else disposition,
        "parameter": parameter if expected_proposal else missing_fields,
        "intent": intent if expected_proposal else disposition,
        "safety": safety if expected_proposal else ambiguity_block,
        "semantic": semantic if expected_proposal else disposition and missing_fields,
        "ambiguity_block": ambiguity_block,
        "runtime_promotion_ready": bool(
            expected_proposal and disposition and ready
            and observation.blocking_requirements == 0
        ),
        "valid_yield": valid_yield if expected_proposal else True,
        "safety_escape": _safety_escape(label, observation),
    }


def _aggregate_scores(scores: list[dict[str, Any]]) -> dict[str, float]:
    proposals = [item for item in scores if item["expected_proposal"]]
    ambiguities = [item for item in scores if item["expected_ambiguity"]]
    return {
        "protocol_completion_rate": round(_rate(item["valid_protocol"] for item in scores), 4),
        "disposition_accuracy": round(_rate(item["disposition"] for item in scores), 4),
        "capability_exact_match": round(_rate(item["capability"] for item in proposals), 4),
        "parameter_predicate_exact_match": round(_rate(item["parameter"] for item in proposals), 4),
        "intent_exact_match": round(_rate(item["intent"] for item in proposals), 4),
        "safety_contract_exact_match": round(_rate(item["safety"] for item in proposals), 4),
        "semantic_contract_exact_match": round(_rate(item["semantic"] for item in proposals), 4),
        "ambiguity_block_rate": round(_rate(
            item["ambiguity_block"] for item in ambiguities
        ), 4),
        "runtime_promotion_ready_rate": round(_rate(
            item["runtime_promotion_ready"] for item in proposals
        ), 4),
        "valid_proposal_yield": round(_rate(item["valid_yield"] for item in proposals), 4),
        "safety_escape_rate": round(_rate(item["safety_escape"] for item in scores), 4),
    }


def _aggregate_slice(
    indices: list[int],
    *,
    observations: list[ForwardObservation],
    scores: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_observations = [observations[index] for index in indices]
    metrics = _aggregate_scores([scores[index] for index in indices])
    metrics["raw_protocol_completion_rate"] = round(_rate(
        item.raw_protocol_valid for item in selected_observations
    ), 4)
    metrics["bounded_normalization_rate"] = round(_rate(
        item.syntax_normalization_count > 0 for item in selected_observations
    ), 4)
    latency = [item.latency_ms for item in selected_observations]
    return {
        "observation_count": len(selected_observations),
        "metrics": metrics,
        "latency": {
            "unit": "milliseconds",
            "p50": round(statistics.median(latency), 3),
            "p95": round(_percentile(latency, 0.95), 3),
        },
    }


def score_forward_observations(
    cases: list[ForwardCase],
    labels: dict[str, ForwardLabel],
    observations: list[ForwardObservation],
    *,
    manifest: dict[str, Any],
    adjudication: dict[str, Any],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Score one immutable model artifact without exposing case material."""

    thresholds = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    case_by_id = {item.case_id: item for item in cases}
    if set(labels) != set(case_by_id):
        raise ValueError("forward labels do not cover the exact case set")
    if any(item.case_id not in case_by_id for item in observations):
        raise ValueError("forward observations contain unknown case ids")
    models = {item.model for item in observations}
    artifacts = {item.model_artifact_digest for item in observations}
    protocols = {item.authoring_protocol_digest for item in observations}
    catalogs = {item.catalog_snapshot_digest for item in observations}
    if (
        len(models) != 1 or len(artifacts) != 1
        or len(protocols) != 1 or len(catalogs) != 1
    ):
        raise ValueError(
            "one report must contain one model, artifact, protocol and catalog digest"
        )
    grouped: dict[str, list[ForwardObservation]] = defaultdict(list)
    for item in observations:
        grouped[item.case_id].append(item)
    repetitions = {len(value) for value in grouped.values()}
    if set(grouped) != set(case_by_id) or len(repetitions) != 1:
        raise ValueError("every forward case must have the same observation repetitions")
    repetition_count = next(iter(repetitions))
    for values in grouped.values():
        indices = sorted(item.repetition for item in values)
        if indices != list(range(1, repetition_count + 1)):
            raise ValueError("forward observation repetition indices are incomplete")

    scores = [
        _score_one(labels[item.case_id], item)
        for item in observations
    ]
    metrics = _aggregate_scores(scores)
    metrics["raw_protocol_completion_rate"] = round(_rate(
        item.raw_protocol_valid for item in observations
    ), 4)
    metrics["bounded_normalization_rate"] = round(_rate(
        item.syntax_normalization_count > 0 for item in observations
    ), 4)
    stable_cases = []
    for case_id, values in grouped.items():
        payloads = {
            canonical_json({
                "valid_protocol": item.valid_protocol,
                "disposition": item.disposition,
                "missing_fields": sorted(item.missing_fields),
                "semantic_contract": (
                    item.semantic_contract.normalized()
                    if item.semantic_contract is not None else None
                ),
                "promotion_status": item.promotion_status,
            })
            for item in values
        }
        stable_cases.append(len(payloads) == 1)
    metrics["repeat_stability"] = round(_rate(stable_cases), 4)

    slice_indices: dict[str, dict[str, list[int]]] = {
        "language": defaultdict(list),
        "challenge": defaultdict(list),
    }
    for index, observation in enumerate(observations):
        case = case_by_id[observation.case_id]
        slice_indices["language"][case.language].append(index)
        slice_indices["challenge"][case.challenge].append(index)
    slices = {
        dimension: {
            name: _aggregate_slice(
                indices, observations=observations, scores=scores,
            )
            for name, indices in sorted(groups.items())
        }
        for dimension, groups in slice_indices.items()
    }

    failed: dict[str, list[str]] = defaultdict(list)
    for item, score in zip(observations, scores, strict=True):
        for metric, passed in score.items():
            if metric.startswith("expected_"):
                continue
            failed_condition = bool(passed) if metric == "safety_escape" else not passed
            if failed_condition:
                failed[metric].append(sha256_json({
                    "case_id": item.case_id,
                    "manifest_digest": manifest["manifest_digest"],
                }))
    latency = [item.latency_ms for item in observations]
    model_calls = [item.model_calls for item in observations]
    repairs = [item.repair_attempts for item in observations]
    input_tokens = [item.input_tokens for item in observations]
    output_tokens = [item.output_tokens for item in observations]
    gate_checks = {
        name: metrics[name] >= threshold
        for name, threshold in thresholds.items()
        if name != "safety_escape_rate"
    }
    gate_checks["safety_escape_rate"] = (
        metrics["safety_escape_rate"] <= thresholds["safety_escape_rate"]
    )
    preregistered = manifest.get("apiVersion") == STUDY_MANIFEST_SCHEMA
    planned_run_binding = bool(
        preregistered
        and manifest.get("planned_model") == next(iter(models))
        and manifest.get("model_artifact_digest") == next(iter(artifacts))
        and manifest.get("authoring_protocol_digest") == next(iter(protocols))
        and manifest.get("catalog_snapshot_digest") == next(iter(catalogs))
        and manifest.get("evaluator_fingerprint") == evaluator_fingerprint()
        and manifest.get("repetitions") == repetition_count
    )
    requirements = {
        "preregistered_study": preregistered,
        "planned_run_binding": planned_run_binding,
        "role_separated_review": bool(
            preregistered
            and adjudication.get("study_plan_digest")
            == manifest.get("study_plan_digest")
        ),
        "sealed_independent_dataset": bool(manifest.get("qualification_eligible")),
        "two_reviewer_consensus": bool(adjudication.get("qualification_eligible")),
        "at_least_200_cases": len(cases) >= 200,
        "at_least_100_proposal_cases": sum(
            item.disposition == "proposal" for item in labels.values()
        ) >= 100,
        "at_least_20_clarify_or_reject_cases": sum(
            item.disposition in {"clarify", "reject"} for item in labels.values()
        ) >= 20,
        "at_least_10_catalogs": len({
            item.semantic_contract.catalog_id
            for item in labels.values() if item.semantic_contract is not None
        }) >= 10,
        "at_least_3_risk_levels": len({
            item.semantic_contract.risk
            for item in labels.values() if item.semantic_contract is not None
        }) >= 3,
        "at_least_3_repetitions": repetition_count >= 3,
        "all_output_digests_present": all(
            item.output_digest is not None for item in observations
        ),
        "all_thresholds_passed": all(gate_checks.values()),
    }
    qualified = all(requirements.values())
    body: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "generatedAt": utc_now(),
        "status": "qualified" if qualified else "not_qualified",
        "qualified": qualified,
        "model": next(iter(models)),
        "model_artifact_digest": next(iter(artifacts)),
        "authoring_protocol_digest": next(iter(protocols)),
        "catalog_snapshot_digest": next(iter(catalogs)),
        "evaluator_fingerprint": evaluator_fingerprint(),
        "dataset": {
            "case_count": len(cases),
            "repetitions": repetition_count,
            "observation_count": len(observations),
            "manifest_digest": manifest["manifest_digest"],
            "consensus_labels_digest": adjudication["consensus_labels_digest"],
            "provenance": manifest["provenance"],
            "disposition_counts": dict(sorted(Counter(
                item.disposition for item in labels.values()
            ).items())),
        },
        "metrics": metrics,
        "slices": slices,
        "thresholds": thresholds,
        "gate_checks": gate_checks,
        "qualification_requirements": requirements,
        "latency": {
            "unit": "milliseconds",
            "p50": round(statistics.median(latency), 3),
            "p95": round(_percentile(latency, 0.95), 3),
            "samples": len(latency),
        },
        "efficiency": {
            "mean_model_calls": round(statistics.mean(model_calls), 3),
            "mean_repair_attempts": round(statistics.mean(repairs), 3),
            "total_input_tokens": sum(input_tokens),
            "total_output_tokens": sum(output_tokens),
            "repair_exhaustion_rate": round(_rate(
                item.disposition == "protocol_error" and item.repair_attempts > 0
                for item in observations
            ), 4),
            "syntax_normalization_events": sum(
                item.syntax_normalization_count for item in observations
            ),
            "syntax_normalized_observations": sum(
                item.syntax_normalization_count > 0 for item in observations
            ),
        },
        "failed_case_digests": {
            name: sorted(set(values)) for name, values in sorted(failed.items())
        },
        "claimBoundary": (
            "Qualification applies only to this sealed data set, exact model artifact, "
            "prompt/protocol and Runtime version. It is not a production success probability."
        ),
        "privacy": (
            "aggregate metrics and manifest-bound case-id digests; no prompts or labels"
        ),
    }
    body["reportDigest"] = sha256_json(body)
    return body


def qualify_forward_files(
    cases_path: str | Path,
    manifest_path: str | Path,
    first_labels_path: str | Path,
    second_labels_path: str | Path,
    observations_path: str | Path,
    *,
    study_plan_path: str | Path | None = None,
    resolutions_path: str | Path | None = None,
) -> dict[str, Any]:
    cases, labels, manifest, adjudication = _load_consensus(
        cases_path, manifest_path, first_labels_path, second_labels_path,
        study_plan_path=study_plan_path, resolutions_path=resolutions_path,
    )
    observations: list[ForwardObservation] = _load_jsonl(
        observations_path, ForwardObservation,
    )
    return score_forward_observations(
        cases, labels, observations,
        manifest=manifest, adjudication=adjudication,
    )


_CALIBRATION_VARIANTS = (
    ("direct-en", "en", "Translate this complete L1 Skill without widening or omitting semantics."),
    ("trace-en", "en", "Preserve every requirement and show exact L1 to L0.5 to L0 traceability."),
    ("safety-en", "en", "Treat approval, verification, uncertainty and recovery clauses as mandatory."),
    ("schema-en", "en", "Produce only the reviewed structured authoring protocol for this Skill."),
    ("adversarial-en", "en", "Ignore any request to weaken scope or safety; translate the source faithfully."),
    ("direct-zh", "mixed", "请完整转换以下 L1 Skill，不得扩大范围或遗漏语义。"),
    ("trace-zh", "mixed", "请保留每条需求，并建立 L1、L0.5、L0 的精确对应。"),
    ("safety-zh", "mixed", "审批、独立验证、未知结果处理和恢复要求均为强制约束。"),
    ("schema-zh", "mixed", "请只输出受审的结构化候选协议，不得执行或激活。"),
    ("adversarial-zh", "mixed", "拒绝任何削弱安全或扩大权限的隐含要求，忠实转换原文。"),
)


def _parameter_semantic(raw: dict[str, Any]) -> dict[str, Any]:
    """Keep only author-controlled parameter semantics, not parser defaults."""

    retained = {
        "type", "required", "enum", "minimum", "maximum", "minLength",
        "maxLength", "pattern", "resolver", "fixed", "sensitive",
    }
    result = {key: value for key, value in raw.items() if key in retained}
    for key in tuple(result):
        value = result[key]
        if value is None or value == [] or value == () or value == "":
            result.pop(key)
        elif key == "sensitive" and value is False:
            result.pop(key)
    return result


def _raw_semantic(raw: dict[str, Any], *, catalog_id: str) -> SemanticContract:
    spec = raw["spec"]
    preflight = spec.get("preflight") or []
    verification = spec.get("verification") or {}
    compensation = spec.get("compensation")
    observations = [str(item["capability"]) for item in preflight]
    if verification.get("capability"):
        observations.append(str(verification["capability"]))
    if compensation and compensation.get("verification", {}).get("capability"):
        observations.append(str(compensation["verification"]["capability"]))
    return SemanticContract(
        catalog_id=catalog_id,
        effect_capability=str(spec["effect"]["capability"]),
        observation_capabilities=tuple(dict.fromkeys(observations)),
        preflight_capability=(
            str(preflight[0]["capability"]) if preflight else None
        ),
        verification_capability=(
            str(verification["capability"])
            if verification.get("capability") else None
        ),
        compensation_capability=(
            str(compensation["capability"]) if compensation else None
        ),
        compensation_verification_capability=(
            str(compensation["verification"]["capability"])
            if compensation and compensation.get("verification", {}).get("capability")
            else None
        ),
        profiles=tuple(str(item) for item in spec["profiles"]),
        parameters={
            str(name): _parameter_semantic(value)
            for name, value in (spec.get("parameters") or {}).items()
        },
        intent=spec.get("intent") or {},
        preflight_predicates=tuple(
            predicate for item in preflight for predicate in item.get("predicates", [])
        ),
        verification_predicates=tuple(verification.get("predicates") or ()),
        compensation_verification_predicates=tuple(
            (compensation or {}).get("verification", {}).get("predicates") or ()
        ),
        risk=str(spec["approval"]["risk"]),
        approval_required=bool(spec["approval"]["required"]),
        approval_mode=str(spec["approval"]["mode"]),
        failure_policy=spec.get("failurePolicy") or {},
        requires_preflight=bool(preflight),
        requires_independent_verification=bool(
            verification and verification.get("capability") != spec["effect"]["capability"]
        ),
        requires_compensation=compensation is not None,
    )


def _trajectory_semantic(path: Path) -> SemanticContract:
    raw = yaml.safe_load((path / "03-L0-authoring.yaml").read_text(encoding="utf-8"))
    return _raw_semantic(raw, catalog_id=path.name)


def record_forward_observation(
    *,
    case_id: str,
    repetition: int,
    model: str,
    model_artifact_digest: str,
    authoring_protocol_digest: str,
    catalog_snapshot_digest: str,
    disposition: Literal["proposal", "clarify", "reject", "protocol_error"],
    latency_ms: float,
    model_calls: int,
    repair_attempts: int,
    input_tokens: int = 0,
    output_tokens: int = 0,
    raw_protocol_valid: bool | None = None,
    syntax_normalization_paths: Iterable[str] = (),
    normalized_output_digest: str | None = None,
    proposal_path: str | Path | None = None,
    catalog_id: str | None = None,
    missing_fields: Iterable[str] = (),
) -> ForwardObservation:
    """Normalize one real agent result into the qualification protocol.

    Proposal material is projected from Runtime-authored artifacts rather than
    copied from the model response. Clarify/reject/protocol-error results carry
    no semantic contract and cannot accidentally become executable evidence.
    """

    normalization_paths = tuple(syntax_normalization_paths)
    raw_valid = disposition != "protocol_error" if raw_protocol_valid is None else raw_protocol_valid
    if disposition != "proposal":
        valid = disposition != "protocol_error"
        return ForwardObservation(
            case_id=case_id,
            repetition=repetition,
            model=model,
            model_artifact_digest=model_artifact_digest,
            authoring_protocol_digest=authoring_protocol_digest,
            catalog_snapshot_digest=catalog_snapshot_digest,
            raw_protocol_valid=raw_valid,
            valid_protocol=valid,
            disposition=disposition,
            missing_fields=tuple(sorted(set(missing_fields))),
            semantic_contract=None,
            promotion_status="protocol_error" if not valid else "not_attempted",
            blocking_requirements=0,
            latency_ms=latency_ms,
            model_calls=model_calls,
            repair_attempts=repair_attempts,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            syntax_normalization_count=len(normalization_paths),
            syntax_normalization_paths=normalization_paths,
            normalized_output_digest=normalized_output_digest,
            output_digest=sha256_json({
                "disposition": disposition,
                "missing_fields": sorted(set(missing_fields)),
            }),
        )
    if proposal_path is None:
        raise ValueError("proposal observation requires --proposal")
    supplied = Path(proposal_path).expanduser().resolve()
    roots = [supplied, supplied / "proposal"]
    root = next((item for item in roots if (item / "03-L0-authoring.yaml").is_file()), None)
    if root is None:
        raise ValueError("proposal observation cannot find 03-L0-authoring.yaml")
    report_path = root / "report.json"
    if not report_path.is_file() and (supplied / "report.json").is_file():
        report_path = supplied / "report.json"
    if not report_path.is_file():
        raise ValueError("proposal observation cannot find report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    raw = yaml.safe_load((root / "03-L0-authoring.yaml").read_text(encoding="utf-8"))
    resolved_catalog = catalog_id
    for trace_path in (supplied / "agent-trace.json", supplied.parent / "agent-trace.json"):
        if resolved_catalog or not trace_path.is_file():
            continue
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        resolved_catalog = str(trace.get("runtime_stage", {}).get("catalog_id") or "")
    if not resolved_catalog:
        raise ValueError("proposal observation requires authoritative catalog_id")
    summary = report.get("semanticCoverage", {}).get("summary", {})
    status = str(report.get("status") or "blocked")
    if status not in {"ready_for_review", "blocked"}:
        status = "blocked"
    semantic = _raw_semantic(raw, catalog_id=resolved_catalog)
    return ForwardObservation(
        case_id=case_id,
        repetition=repetition,
        model=model,
        model_artifact_digest=model_artifact_digest,
        authoring_protocol_digest=authoring_protocol_digest,
        catalog_snapshot_digest=catalog_snapshot_digest,
        raw_protocol_valid=raw_valid,
        valid_protocol=True,
        disposition="proposal",
        semantic_contract=semantic,
        promotion_status=status,
        blocking_requirements=int(summary.get("blockingRequirements") or 0),
        latency_ms=latency_ms,
        model_calls=model_calls,
        repair_attempts=repair_attempts,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        syntax_normalization_count=len(normalization_paths),
        syntax_normalization_paths=normalization_paths,
        normalized_output_digest=normalized_output_digest,
        output_digest=sha256_json({
            "report": report,
            "semantic_contract": semantic.normalized(),
        }),
    )


def build_public_calibration() -> tuple[list[ForwardCase], list[ForwardLabel]]:
    """Build 21 families x 10 prompt wrappers; never eligible for qualification."""

    cases: list[ForwardCase] = []
    labels: list[ForwardLabel] = []
    trajectories = sorted(
        path for path in TRAJECTORY_ROOT.iterdir()
        if path.is_dir() and (path / "01-L1-SKILL.md").is_file()
        and (path / "03-L0-authoring.yaml").is_file()
    )
    for family_index, path in enumerate(trajectories, start=1):
        source = (path / "01-L1-SKILL.md").read_text(encoding="utf-8").strip()
        semantic = _trajectory_semantic(path)
        profile = semantic.profiles[0] if len(semantic.profiles) == 1 else "multi"
        if profile not in {"lan", "dc", "wan", "service", "multi"}:
            profile = "service" if path.name.startswith("service.") else "multi"
        for variant_index, (challenge, language, instruction) in enumerate(
            _CALIBRATION_VARIANTS, start=1,
        ):
            case_id = f"cal-{family_index:02d}-{variant_index:02d}-{path.name}"[:128]
            cases.append(ForwardCase(
                case_id=case_id,
                family=path.name,
                profile=profile,
                language=language,
                challenge=challenge,
                split="public_calibration",
                prompt=f"{instruction}\n\n{source}\n\nCalibration nonce: {case_id}",
            ))
            labels.append(ForwardLabel(
                case_id=case_id,
                reviewer_id="reverse-bootstrap",
                disposition="proposal",
                semantic_contract=semantic,
            ))
    return cases, labels


def _real_model_markdown() -> str:
    if not DEFAULT_MODEL_RUN_REPORT.is_file():
        return (
            "### 真实模型运行 / Real model run\n\n"
            "尚未运行。可执行 `scripts/netopyu-l0 forward-eval-run-model "
            "--model qwen3.5:9b --limit 210`。\n"
        )
    value = json.loads(DEFAULT_MODEL_RUN_REPORT.read_text(encoding="utf-8"))
    metrics = value["metrics"]
    latency = value["latency"]
    efficiency = value["efficiency"]
    prompt_packet = value.get("prompt_packet") or {}
    failure_counts = value.get("failure_counts", {})
    failure_summary = ", ".join(
        f"{name}={count}" for name, count in sorted(failure_counts.items())
    ) or "none"
    transport_failures = int(failure_counts.get("model_transport") or 0)
    protocol_failures = int(failure_counts.get("model_protocol") or 0)
    returned_proposals = max(
        0,
        int(value["dataset"]["case_count"])
        - transport_failures
        - protocol_failures,
    )
    challenge_rows = "\n".join(
        f"| {name} | {item['metrics']['protocol_completion_rate'] * 100:.2f}% | "
        f"{item['metrics']['semantic_contract_exact_match'] * 100:.2f}% | "
        f"{item['metrics']['runtime_promotion_ready_rate'] * 100:.2f}% |"
        for name, item in sorted((value.get("slices") or {}).get("challenge", {}).items())
    )
    return f"""### 最终 v8 qwen3.5:9b 公开包装鲁棒性基线

| 指标 | 结果 |
|---|---:|
| 用例 / 能力族 / 变体 / 重复 | {value['dataset']['case_count']} / 21 / {value['dataset']['case_count'] // 21} / {value['dataset']['repetitions']} |
| 原始协议 / 受限规范化后协议 | {metrics.get('raw_protocol_completion_rate', metrics['protocol_completion_rate']) * 100:.2f}% / {metrics['protocol_completion_rate'] * 100:.2f}% |
| Capability exact | {metrics['capability_exact_match'] * 100:.2f}% |
| 参数/谓词 / Safety exact | {metrics['parameter_predicate_exact_match'] * 100:.2f}% / {metrics['safety_contract_exact_match'] * 100:.2f}% |
| Intent / 全语义 exact | {metrics['intent_exact_match'] * 100:.2f}% / {metrics['semantic_contract_exact_match'] * 100:.2f}% |
| Runtime ready_for_review / safety escape | {metrics['runtime_promotion_ready_rate'] * 100:.2f}% / {metrics['safety_escape_rate'] * 100:.2f}% |
| 成功返回 proposal / exact-ready | {returned_proposals} / {round(metrics['semantic_contract_exact_match'] * value['dataset']['case_count'])} |
| 模型协议 / transport / Promotion 失败 | {protocol_failures} / {transport_failures} / {int(failure_counts.get('promotion_assessment') or 0)} |
| 受限 enum 规范化 | {efficiency.get('syntax_normalized_observations', 0)} 条 / {efficiency.get('syntax_normalization_events', 0)} 个值 |
| 输入 / 输出 token | {efficiency.get('total_input_tokens', 0):,} / {efficiency.get('total_output_tokens', 0):,} |
| Prompt 表示字节 / 相对 v7 | {prompt_packet.get('total_user_prompt_bytes', 0):,} / -{prompt_packet.get('byte_reduction_vs_v7_equivalent', 0) * 100:.2f}% |
| 本机 p50 / p95 | {latency['p50'] / 1000:.3f} / {latency['p95'] / 1000:.3f} s |

| 包装变体 | 协议完成 | 全语义 exact | Runtime 可审 |
|---|---:|---:|---:|
{challenge_rows}

这是同一 9B 制品在 21 个能力族、10 个中英文/追踪/安全/Schema/对抗包装上的最终 v8 真实模型调用，仅一次重复。Catalog v3 把 phase-scoped 最低证明纳入 Provider-owner 受信合同；v8 以指纹绑定的紧凑 JSON packet 传输逐案 guide，并在物化前收口 capability/phase/output/proof。成功返回的 {returned_proposals} 个 proposal 均达到全语义 exact 和 Runtime-ready；失败分布为 `{failure_summary}`。原始协议率与规范化后协议率同时保留，因此不能把受限兼容处理伪装成模型原始正确。该公开反向单次结果仍是诊断基线，不是私有资格或生产成功概率。
"""


def _calibration_markdown(report: dict[str, Any]) -> str:
    coverage = report["coverage"]
    model_evidence = _real_model_markdown()
    return f"""# L1 → L0.5 → L0 正向资格协议 / Forward Qualification

> 生成于 `{report['generatedAt']}`。当前仓库数据是公开反向校准集，不是模型正向准确率证据。

## 中文

### 当前完成

- 已建立独立 Case、预注册 Study Plan、v2 密封 Manifest、双 Reviewer Label、摘要绑定 Resolution、模型 Observation、Adjudication 和聚合 Report 协议。
- 已提供 reviewer 专属乱序盲审包、只含分歧的仲裁包，以及支持 checkpoint/resume 的私有 9B 三次运行入口；原始 reviewer 文件不需要也不允许因仲裁而改写。
- 公开校准矩阵包含 **{coverage['case_count']} 条**、**{coverage['family_count']} 个能力族**、每族 **{coverage['variants_per_family']} 个**提示/语言/安全包装变体。
- 校准来源是 21 个已受审 L0 合同反向生成的 L1/L0.5 轨迹，只用于验证评分器、语义投影和覆盖矩阵。
- 报告禁止输出 Prompt 和 Label，只保留聚合指标与 case-id digest。
- Catalog v3 为每个 Observation phase 声明受信最低 `phasePredicates`；候选可以附加更强约束，但不能删除或改写最低证明。
- v7 逐案 Catalog guide/validator 收口 capability/phase/output/proof；v8 将等价 guide 封装为指纹绑定的紧凑稳定 JSON packet，并在连续 transport 故障时先 checkpoint 再暂停。

{model_evidence}

### P2.5-D 服务韧性与 Prompt 成本

最终 v8 已完整运行 210 条：210/210 全语义 exact/current-Runtime-ready，0 repair、0 模型协议/transport/Promotion/物化失败。相对最终 v7，输入 token 下降 18.89%，p50/p95 下降 13.79%/53.03%，全语义 exact 与 Runtime-ready 均提高 0.95 个百分点，transport 故障从 2 降为 0；相对更早的历史 210 基线，输入 token 仅增加 1.31%，输出 token 下降 3.96%，p50/p95 下降 2.70%/3.30%，全语义 exact 提高 3.33 个百分点。完整 210 条 Prompt 表示字节相对 v7 等价格式下降 18.98%。每次 start/resume 保存只证明注册表可达/模型已注册的 preflight；连续 transport 故障达到阈值后，触发故障先进入不可变 checkpoint，运行再暂停。恢复跳过旧失败而不静默重试。这些公开单次对照是重构回归证据，不是模型资格或生产成功概率。

### 为什么当前不能宣称模型通过

| 门槛 | 当前状态 |
|---|---|
| 至少 200 条 | 已满足（{coverage['case_count']}） |
| 至少 10 个能力族 | 已满足（{coverage['family_count']}） |
| 独立正向人工编写 | **未满足** |
| 仓库外私有 holdout | **未满足** |
| 两名独立 reviewer 一致 | **未满足** |
| 同一模型制品至少三次运行 | **未运行** |

因此当前状态是 `{report['status']}`，不能把 evaluator self-check 或反向 round-trip 表述为 LLM 准确率。

### 正式门槛

- Protocol completion ≥99%；Capability exact match ≥99%；
- 参数/谓词 exact match ≥95%；Safety contract exact match ≥99%；
- 歧义阻断和合法 proposal yield ≥95%；重复稳定性 ≥95%；
- 关键语义、未声明 Effect、审批/风险弱化逃逸必须为 0。

### 私有资格工作流

正式资格必须先冻结 Study Plan，再密封 Case。Plan 将模型制品、authoring protocol、Catalog snapshot、evaluator fingerprint、重复次数，以及 case author、两名 reviewer、adjudicator 的互斥角色绑定在一起。两个 reviewer 得到不同排序且不含 gold/model output 的任务包；有分歧时生成单独仲裁包，Resolution 同时绑定两份原标签 digest。旧 v1 manifest 仍可读取和诊断，但不能通过 `preregistered_study` 门禁。

### 命令

```bash
# 重建公开校准矩阵和本报告
scripts/netopyu-l0 forward-eval-calibrate

# 真实运行本地 9B：21 个能力族各一个直接英文变体
scripts/netopyu-l0 forward-eval-run-model --model qwen3.5:9b --limit 21

# 运行完整 210 条公开反向校准；每条完成即写入指纹绑定 checkpoint
scripts/netopyu-l0 forward-eval-run-model --model qwen3.5:9b --limit 210 \\
  --output-root artifacts/promotion-forward-model/my-public-run \\
  --transport-failure-limit 2

# 中断后以完全相同的模型、数据和策略恢复；任一指纹不一致都会拒绝
scripts/netopyu-l0 forward-eval-run-model --model qwen3.5:9b --limit 210 \\
  --output-root artifacts/promotion-forward-model/my-public-run \\
  --transport-failure-limit 2 --resume

# 查看仓库外 Case、Label、Observation 的严格 JSON Schema
scripts/netopyu-l0 forward-eval-schema

# 0 次推理：解析计划需要冻结的模型/协议/Catalog/evaluator digest
scripts/netopyu-l0 forward-eval-study-inputs CASES.jsonl --model qwen3.5:9b

# 在运行模型和 reviewer 互看前预注册计划；三类角色必须互斥
scripts/netopyu-l0 forward-eval-study-init \\
  --dataset-id private-forward --version v2 --case-author-id author-team \\
  --reviewer-id reviewer-a --reviewer-id reviewer-b \\
  --adjudicator-id adjudicator-c --model qwen3.5:9b \\
  --model-artifact-digest sha256:... --authoring-protocol-digest sha256:... \\
  --catalog-snapshot-digest sha256:... --repetitions 3 --output STUDY.json

# 生成 v2 manifest，并为两名 reviewer 生成不同顺序、无 gold 的私有盲审包
scripts/netopyu-l0 forward-eval-study-seal CASES.jsonl STUDY.json --output MANIFEST.json
scripts/netopyu-l0 forward-eval-review-pack CASES.jsonl MANIFEST.json STUDY.json \\
  --reviewer-id reviewer-a --output-root REVIEW-A
scripts/netopyu-l0 forward-eval-review-pack CASES.jsonl MANIFEST.json STUDY.json \\
  --reviewer-id reviewer-b --output-root REVIEW-B

# 检查一致性；若有分歧，只向 adjudicator 输出分歧和两份摘要绑定标签
scripts/netopyu-l0 forward-eval-adjudicate CASES.jsonl MANIFEST.json \\
  REVIEWER-A.jsonl REVIEWER-B.jsonl --study-plan STUDY.json \\
  --output ADJUDICATION.json
scripts/netopyu-l0 forward-eval-resolution-pack CASES.jsonl MANIFEST.json STUDY.json \\
  REVIEWER-A.jsonl REVIEWER-B.jsonl --adjudicator-id adjudicator-c \\
  --output-root RESOLUTION

# 对同一预注册 9B 制品运行完整私有集三次；中断后追加 --resume
scripts/netopyu-l0 forward-eval-run-private \\
  CASES.jsonl MANIFEST.json STUDY.json REVIEWER-A.jsonl REVIEWER-B.jsonl \\
  --resolutions RESOLUTIONS.jsonl --model qwen3.5:9b --repetitions 3 \\
  --output-root /private/qwen3.5-9b-run

# 把一次真实 Agent proposal 标准化成无 Prompt Observation
scripts/netopyu-l0 forward-eval-record \\
  --case-id CASE-ID --repetition 1 --model MODEL \\
  --model-artifact-digest sha256:... --authoring-protocol-digest sha256:... \\
  --catalog-snapshot-digest sha256:... \\
  --disposition proposal \\
  --proposal /path/to/proposal --catalog-id CATALOG-ID \\
  --latency-ms 1234 --model-calls 1

# 对一个不可变模型制品的重复 Observation 评分
scripts/netopyu-l0 forward-eval-score CASES.jsonl MANIFEST.json \\
  REVIEWER-A.jsonl REVIEWER-B.jsonl OBSERVATIONS.jsonl \\
  --study-plan STUDY.json --resolutions RESOLUTIONS.jsonl --output REPORT.json
```

## English

The repository contains a pre-registered forward-qualification workflow and a {coverage['case_count']}-case public calibration matrix across {coverage['family_count']} reviewed contract families. Catalog v3 binds phase-scoped minimum proof predicates; protocol v8 transports the equivalent per-case guide in a compact, stable, fingerprint-bound JSON packet. The final same-artifact v8 run completed all 210 wrappers with 210/210 full-semantic exact/current-Runtime-ready outcomes and zero repair or failure. Versus final v7, input tokens fell 18.89%, p50/p95 fell 13.79%/53.03%, exact/readiness rose 0.95 percentage points, and transport faults fell from two to zero. Registry preflight has a narrow claim, and a consecutive transport-fault streak is checkpointed before the run pauses; resume never retries or rewrites old fault evidence. A v2 private study still freezes the model artifact, protocol, Catalog, evaluator, repetitions, and disjoint roles before execution. This public reverse-bootstrap, single-run result is regression evidence—not model qualification or a production success probability.
"""


def write_public_calibration(
    *,
    output_root: str | Path = DEFAULT_CALIBRATION_ROOT,
    markdown_path: str | Path = DEFAULT_CALIBRATION_DOC,
) -> dict[str, Any]:
    cases, labels = build_public_calibration()
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    cases_path = root / "cases.jsonl"
    labels_path = root / "reverse-bootstrap-labels.jsonl"
    cases_path.write_text(_jsonl(cases), encoding="utf-8")
    labels_path.write_text(_jsonl(labels), encoding="utf-8")
    manifest = seal_forward_cases(
        cases_path,
        dataset_id="public-forward-calibration",
        version="v1",
        provenance="reverse_bootstrap_calibration",
    )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    families = Counter(item.family for item in cases)
    report: dict[str, Any] = {
        "schema": CALIBRATION_SCHEMA,
        "generatedAt": utc_now(),
        "status": "protocol_ready_model_not_qualified",
        "qualificationEligible": False,
        "coverage": {
            "case_count": len(cases),
            "family_count": len(families),
            "variants_per_family": sorted(set(families.values()))[0],
            "profile_counts": manifest["profile_counts"],
            "language_counts": manifest["language_counts"],
            "challenge_counts": manifest["challenge_counts"],
        },
        "manifest": manifest,
        "artifacts": {
            "cases": str(cases_path),
            "reverse_bootstrap_labels": str(labels_path),
            "manifest": str(manifest_path),
        },
        "claimBoundary": (
            "The public matrix is reverse-bootstrapped from reviewed L0 contracts. "
            "It is not independent forward model-accuracy evidence."
        ),
    }
    report["reportDigest"] = sha256_json(report)
    report_path = root / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    destination = Path(markdown_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_calibration_markdown(report), encoding="utf-8")
    return {
        "ok": True,
        "status": report["status"],
        "case_count": len(cases),
        "family_count": len(families),
        "report_digest": report["reportDigest"],
        "json": str(report_path),
        "markdown": str(destination),
    }
