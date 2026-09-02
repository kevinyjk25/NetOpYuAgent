"""Strict, authority-free contracts for the production L1 decision plane."""

from __future__ import annotations

import json
import math
import re
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from network_runtime.contracts import sha256_json


L1_DECISION_SCHEMA = "netopyu.io/l1-decision/v1"
L1_DECISION_ENVELOPE_SCHEMA = "netopyu.io/l1-decision-envelope/v1"
L1_DECISION_EVIDENCE_SCHEMA = "netopyu.io/l1-decision-evidence/v1"
_IDENTIFIER = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _validate_json(value: Any, *, depth: int = 0) -> None:
    if depth > 4:
        raise ValueError("L1 arguments exceed four nesting levels")
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("L1 arguments cannot contain non-finite numbers")
        return
    if isinstance(value, str):
        if len(value) > 2_000:
            raise ValueError("L1 argument string exceeds 2000 characters")
        return
    if isinstance(value, list):
        if len(value) > 32:
            raise ValueError("L1 argument list exceeds 32 items")
        for item in value:
            _validate_json(item, depth=depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > 32:
            raise ValueError("L1 argument object exceeds 32 fields")
        for key, item in value.items():
            if not isinstance(key, str) or not _IDENTIFIER.fullmatch(key):
                raise ValueError("L1 argument keys must be bounded identifiers")
            _validate_json(item, depth=depth + 1)
        return
    raise ValueError("L1 arguments must contain bounded JSON values only")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class L1DecisionAction(StrEnum):
    SELECT_SKILL = "select_skill"
    SELECT_TOOL = "select_tool"
    CLARIFY = "clarify"
    REFUSE = "refuse"
    OUT_OF_SCOPE = "out_of_scope"


class L1Decision(StrictModel):
    """A proposal only; validation deliberately grants no execution authority."""

    api_version: Literal[L1_DECISION_SCHEMA] = Field(
        default=L1_DECISION_SCHEMA, alias="apiVersion",
    )
    action: L1DecisionAction
    target: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    missing_fields: tuple[str, ...] = ()
    workflow: tuple[str, ...] = ()
    confidence: float = Field(ge=0.0, le=1.0)
    reason_code: str

    @model_validator(mode="after")
    def validate_shape(self) -> "L1Decision":
        if not self.reason_code.strip() or len(self.reason_code) > 80:
            raise ValueError("reason_code must contain 1..80 characters")
        if self.target is not None and not _IDENTIFIER.fullmatch(self.target):
            raise ValueError("L1 target must be a bounded identifier")
        if len(self.arguments) > 32 or len(self.missing_fields) > 16:
            raise ValueError("L1 decision exceeds bounded field counts")
        if len(self.workflow) > 16:
            raise ValueError("L1 workflow exceeds 16 steps")
        if len(set(self.missing_fields)) != len(self.missing_fields):
            raise ValueError("L1 missing fields must be unique")
        if any(not _IDENTIFIER.fullmatch(item) for item in self.missing_fields):
            raise ValueError("L1 missing fields must be bounded identifiers")
        if any(not _IDENTIFIER.fullmatch(item) for item in self.workflow):
            raise ValueError("L1 workflow steps must be bounded identifiers")
        _validate_json(self.arguments)
        encoded = json.dumps(
            self.arguments, ensure_ascii=False, allow_nan=False, separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > 16_384:
            raise ValueError("L1 arguments exceed 16 KiB")
        if self.action in {
            L1DecisionAction.SELECT_SKILL, L1DecisionAction.SELECT_TOOL,
        }:
            if not self.target or self.missing_fields:
                raise ValueError("selection requires a target and no missing fields")
        elif self.action == L1DecisionAction.CLARIFY:
            if not self.target or not self.missing_fields or self.workflow:
                raise ValueError("clarification requires target and missing fields only")
        elif self.target is not None or self.arguments or self.missing_fields or self.workflow:
            raise ValueError("terminal decisions cannot carry executable content")
        return self

    @property
    def digest(self) -> str:
        return sha256_json(self.model_dump(by_alias=True, mode="json"))


class L1DecisionEvidence(StrictModel):
    api_version: Literal[L1_DECISION_EVIDENCE_SCHEMA] = Field(
        default=L1_DECISION_EVIDENCE_SCHEMA, alias="apiVersion",
    )
    prompt_digest: str
    catalog_digest: str
    candidate_digest: str
    policy_digest: str
    model: str | None
    model_attempts: int = Field(ge=0, le=2)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    token_usage_complete: bool
    selected_candidate_index: int | None = Field(default=None, ge=0, le=11)
    candidate_ids: tuple[str, ...]
    dropped_argument_fields: tuple[str, ...] = ()
    normalized_argument_fields: tuple[str, ...] = ()
    attempt_error_types: tuple[str, ...] = ()
    guard_action: Literal["allow", "refuse", "out_of_scope"]
    guard_reason: str
    protocol_valid: bool
    duration_ms: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_evidence(self) -> "L1DecisionEvidence":
        for value in (
            self.prompt_digest, self.catalog_digest,
            self.candidate_digest, self.policy_digest,
        ):
            if not _DIGEST.fullmatch(value):
                raise ValueError("L1 evidence digests must be sha256 values")
        if not 0 <= len(self.candidate_ids) <= 12:
            raise ValueError("L1 evidence exceeds candidate bound")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("L1 evidence candidate identities must be unique")
        if len(self.attempt_error_types) > self.model_attempts:
            raise ValueError("L1 attempt errors exceed model attempt count")
        return self


class L1DecisionEnvelope(StrictModel):
    api_version: Literal[L1_DECISION_ENVELOPE_SCHEMA] = Field(
        default=L1_DECISION_ENVELOPE_SCHEMA, alias="apiVersion",
    )
    decision_id: str
    mode: Literal["shadow", "canary", "enforced"]
    profile: Literal["lan", "dc", "wan"]
    session_id: str
    harness: str
    status: Literal["decided", "policy_terminal", "protocol_failure"]
    decision: L1Decision | None
    evidence: L1DecisionEvidence
    decision_digest: str | None
    evidence_digest: str
    authority: Literal["proposal_only"] = "proposal_only"

    @model_validator(mode="after")
    def validate_binding(self) -> "L1DecisionEnvelope":
        if not self.decision_id.strip() or len(self.decision_id) > 128:
            raise ValueError("L1 decision id is invalid")
        if not self.session_id.strip() or len(self.session_id) > 256:
            raise ValueError("L1 session id is invalid")
        if not self.harness.strip() or len(self.harness) > 64:
            raise ValueError("L1 harness is invalid")
        if self.decision is None:
            if self.status != "protocol_failure" or self.decision_digest is not None:
                raise ValueError("missing decision must be a protocol failure")
        elif self.decision_digest != self.decision.digest:
            raise ValueError("L1 decision digest does not bind the decision")
        if not _DIGEST.fullmatch(self.evidence_digest):
            raise ValueError("L1 evidence digest is invalid")
        expected_evidence = sha256_json(
            self.evidence.model_dump(by_alias=True, mode="json"),
        )
        if expected_evidence != self.evidence_digest:
            raise ValueError("L1 evidence digest does not bind the evidence")
        return self


__all__ = [
    "L1_DECISION_ENVELOPE_SCHEMA",
    "L1_DECISION_EVIDENCE_SCHEMA",
    "L1_DECISION_SCHEMA",
    "L1Decision",
    "L1DecisionAction",
    "L1DecisionEnvelope",
    "L1DecisionEvidence",
]
