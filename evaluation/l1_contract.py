"""Strict contracts for P1.8 natural-language L1 qualification.

The contract is deliberately a proposal only.  It cannot authorize or execute a
tool; accepted write proposals still enter the L0/Domain Effect Runtime.
"""

from __future__ import annotations

import json
import math
import re
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from network_runtime.contracts import sha256_json


L1_DECISION_SCHEMA = "netopyu.io/l1-decision/v1"
L1_SCENARIO_SCHEMA = "netopyu.io/l1-eval-scenario/v1"
_IDENTIFIER = re.compile(r"[A-Za-z0-9_.:-]{1,128}")


def _validate_json_value(value: Any, *, depth: int = 0) -> None:
    if depth > 4:
        raise ValueError("L1 argument nesting exceeds four levels")
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("L1 arguments cannot contain non-finite numbers")
        return
    if isinstance(value, str):
        if len(value) > 2000:
            raise ValueError("L1 argument string exceeds 2000 characters")
        return
    if isinstance(value, list):
        if len(value) > 32:
            raise ValueError("L1 argument list exceeds 32 items")
        for item in value:
            _validate_json_value(item, depth=depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > 32:
            raise ValueError("L1 nested object exceeds 32 fields")
        for key, item in value.items():
            if not isinstance(key, str) or not _IDENTIFIER.fullmatch(key):
                raise ValueError("L1 argument keys must be bounded identifiers")
            _validate_json_value(item, depth=depth + 1)
        return
    raise ValueError("L1 arguments must contain bounded JSON values only")


def _validate_arguments(arguments: dict[str, Any]) -> None:
    _validate_json_value(arguments)
    encoded = json.dumps(
        arguments, ensure_ascii=False, allow_nan=False, separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > 16_384:
        raise ValueError("L1 arguments exceed 16 KiB")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class L1Action(StrEnum):
    SELECT_SKILL = "select_skill"
    SELECT_TOOL = "select_tool"
    CLARIFY = "clarify"
    REFUSE = "refuse"
    OUT_OF_SCOPE = "out_of_scope"


class L1Category(StrEnum):
    SKILL_SELECTION = "skill_selection"
    TOOL_SELECTION = "tool_selection"
    MULTI_STEP = "multi_step"
    CLARIFICATION = "clarification"
    SAFETY_REFUSAL = "safety_refusal"
    OUT_OF_SCOPE = "out_of_scope"


class L1Decision(StrictModel):
    # The transport version is a Runtime-owned constant, not a model choice.
    # Omitting it is safe to normalize; supplying any other value still fails.
    api_version: Literal[L1_DECISION_SCHEMA] = Field(
        default=L1_DECISION_SCHEMA,
        alias="apiVersion",
    )
    action: L1Action
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
        if len(self.arguments) > 32 or len(self.missing_fields) > 16:
            raise ValueError("L1 decision exceeds bounded argument or missing-field count")
        if len(self.workflow) > 16 or len(set(self.missing_fields)) != len(self.missing_fields):
            raise ValueError("L1 decision workflow or missing fields are invalid")
        if self.target is not None and not _IDENTIFIER.fullmatch(self.target):
            raise ValueError("L1 target must be a bounded identifier")
        if any(not _IDENTIFIER.fullmatch(item) for item in self.missing_fields):
            raise ValueError("L1 missing fields must be bounded identifiers")
        if any(not _IDENTIFIER.fullmatch(item) for item in self.workflow):
            raise ValueError("L1 workflow steps must be bounded identifiers")
        _validate_arguments(self.arguments)
        if self.action in {L1Action.SELECT_SKILL, L1Action.SELECT_TOOL}:
            if not self.target or self.missing_fields:
                raise ValueError("selection requires a target and no missing fields")
        elif self.action == L1Action.CLARIFY:
            if not self.missing_fields:
                raise ValueError("clarification requires explicit missing fields")
            if self.workflow:
                raise ValueError("clarification cannot carry an executable workflow")
        elif self.target is not None or self.arguments or self.missing_fields or self.workflow:
            raise ValueError("refusal/out-of-scope decisions cannot carry executable content")
        return self


class L1Scenario(StrictModel):
    api_version: Literal[L1_SCENARIO_SCHEMA] = Field(alias="apiVersion")
    scenario_id: str
    profile: Literal["lan", "dc", "wan"]
    language: Literal["zh", "en", "mixed"]
    category: L1Category
    prompt: str
    expected_action: L1Action
    acceptable_targets: tuple[str, ...] = ()
    expected_arguments: dict[str, Any] = Field(default_factory=dict)
    expected_missing_fields: tuple[str, ...] = ()
    expected_workflow: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_oracle(self) -> "L1Scenario":
        if not self.scenario_id.strip() or not self.prompt.strip():
            raise ValueError("scenario id and prompt are required")
        if len(self.prompt) > 4000 or len(self.expected_arguments) > 32:
            raise ValueError("scenario exceeds bounded prompt or argument size")
        _validate_arguments(self.expected_arguments)
        if self.expected_action in {L1Action.SELECT_SKILL, L1Action.SELECT_TOOL}:
            if not self.acceptable_targets or self.expected_missing_fields:
                raise ValueError("selection oracle requires a target and no missing fields")
        elif self.expected_action == L1Action.CLARIFY:
            if not self.expected_missing_fields:
                raise ValueError("clarification oracle requires missing fields")
        elif (
            self.acceptable_targets or self.expected_arguments
            or self.expected_missing_fields or self.expected_workflow
        ):
            raise ValueError("non-routing oracle cannot contain executable content")
        return self

    @property
    def digest(self) -> str:
        return sha256_json(self.model_dump(by_alias=True, mode="json"))


__all__ = [
    "L1Action",
    "L1Category",
    "L1Decision",
    "L1Scenario",
    "L1_DECISION_SCHEMA",
    "L1_SCENARIO_SCHEMA",
]
