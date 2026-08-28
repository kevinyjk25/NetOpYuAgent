"""Strict serializable contracts for L0 v2 authoring and compiled catalogs."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


API_VERSION = "netopyu.io/l0-effect/v2"
COMPILED_API_VERSION = "netopyu.io/l0-effect-compiled/v2"
SEMVER = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
IDENTIFIER = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class Metadata(StrictModel):
    id: str
    version: str
    owner: str
    description: str = ""
    labels: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_identity(self) -> "Metadata":
        if not IDENTIFIER.fullmatch(self.id):
            raise ValueError("metadata.id must be a lowercase dotted identifier")
        if not SEMVER.fullmatch(self.version):
            raise ValueError("metadata.version must be strict MAJOR.MINOR.PATCH")
        if not self.owner.strip():
            raise ValueError("metadata.owner is required")
        return self


class SkillRef(StrictModel):
    id: str
    version: str

    @model_validator(mode="after")
    def validate_ref(self) -> "SkillRef":
        if not IDENTIFIER.fullmatch(self.id) or not SEMVER.fullmatch(self.version):
            raise ValueError("skillRef requires a valid id and exact semantic version")
        return self

    @property
    def key(self) -> tuple[str, str]:
        return self.id, self.version


ParameterType = Literal["string", "integer", "number", "boolean", "array", "object"]


def value_matches_type(value: Any, expected: ParameterType) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    return isinstance(value, dict)


class ParameterSpec(StrictModel):
    type: ParameterType
    required: bool = False
    description: str = ""
    enum: tuple[Any, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    min_length: int | None = Field(default=None, alias="minLength")
    max_length: int | None = Field(default=None, alias="maxLength")
    pattern: str | None = None
    resolver: str | None = None
    fixed: Any | None = None
    sensitive: bool = False

    @model_validator(mode="after")
    def validate_constraints(self) -> "ParameterSpec":
        if (self.minimum is not None or self.maximum is not None) and self.type not in {
            "integer", "number",
        }:
            raise ValueError("minimum/maximum require an integer or number parameter")
        if (self.min_length is not None or self.max_length is not None) and self.type not in {
            "string", "array",
        }:
            raise ValueError("minLength/maxLength require a string or array parameter")
        if self.pattern is not None and self.type != "string":
            raise ValueError("pattern requires a string parameter")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("parameter minimum cannot exceed maximum")
        if self.min_length is not None and self.max_length is not None and self.min_length > self.max_length:
            raise ValueError("parameter minLength cannot exceed maxLength")
        if self.pattern is not None:
            re.compile(self.pattern)
        if any(not value_matches_type(item, self.type) for item in self.enum):
            raise ValueError("enum value has the wrong parameter type")
        if self.fixed is not None:
            if not value_matches_type(self.fixed, self.type):
                raise ValueError("fixed parameter value has the wrong type")
            if self.enum and self.fixed not in self.enum:
                raise ValueError("fixed parameter value is not in enum")
            if self.minimum is not None and self.fixed < self.minimum:
                raise ValueError("fixed parameter value is below minimum")
            if self.maximum is not None and self.fixed > self.maximum:
                raise ValueError("fixed parameter value is above maximum")
            if self.min_length is not None and len(self.fixed) < self.min_length:
                raise ValueError("fixed parameter value is shorter than minLength")
            if self.max_length is not None and len(self.fixed) > self.max_length:
                raise ValueError("fixed parameter value is longer than maxLength")
            if self.pattern is not None and not re.fullmatch(self.pattern, self.fixed):
                raise ValueError("fixed parameter value does not match pattern")
        return self


class ParameterPatch(StrictModel):
    required: bool | None = None
    enum: tuple[Any, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None
    min_length: int | None = Field(default=None, alias="minLength")
    max_length: int | None = Field(default=None, alias="maxLength")
    pattern: str | None = None
    fixed: Any | None = None
    sensitive: bool | None = None


class Predicate(StrictModel):
    field: str
    operator: Literal[
        "equals", "not_equals", "in", "contains", "greater_than",
        "greater_than_or_equal", "less_than", "less_than_or_equal",
        "exists", "exact_snapshot",
    ]
    expected: Any = None


class Observation(StrictModel):
    id: str
    capability: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    snapshot_fields: tuple[str, ...] = Field(default_factory=tuple, alias="snapshotFields")
    predicates: tuple[Predicate, ...] = ()


class EffectBinding(StrictModel):
    capability: str
    tool: str | None = None
    request: dict[str, Any] = Field(default_factory=dict)
    internal: dict[str, Any] = Field(default_factory=dict)


class IntentSpec(StrictModel):
    kind: str
    target_fields: tuple[str, ...] = Field(alias="targetFields")
    desired_state: dict[str, Any] = Field(alias="desiredState")

    @model_validator(mode="after")
    def require_targets(self) -> "IntentSpec":
        if not self.target_fields:
            raise ValueError("intent.targetFields cannot be empty")
        return self


class VerificationSpec(StrictModel):
    capability: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    predicates: tuple[Predicate, ...]

    @model_validator(mode="after")
    def require_predicate(self) -> "VerificationSpec":
        if not self.predicates:
            raise ValueError("verification requires at least one predicate")
        return self


class CompensationSpec(StrictModel):
    capability: str
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    verification: VerificationSpec


RISK_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}
APPROVAL_RANK = {"single": 0, "dual": 1}


class ApprovalSpec(StrictModel):
    required: bool = True
    risk: Literal["low", "medium", "high", "critical"] = "medium"
    mode: Literal["single", "dual"] = "single"


class FailurePolicy(StrictModel):
    before_send: Literal["abort"] = Field(default="abort", alias="beforeSend")
    after_send_unknown: Literal["reconcile_read_only", "manual_intervention"] = Field(
        default="reconcile_read_only", alias="afterSendUnknown",
    )
    verification_failed: Literal["compensate", "manual_intervention"] = Field(
        default="compensate", alias="verificationFailed",
    )
    compensation_failed: Literal["manual_intervention"] = Field(
        default="manual_intervention", alias="compensationFailed",
    )


class AtomicEffectSpec(StrictModel):
    template: str
    profiles: tuple[str, ...]
    effect: EffectBinding
    intent: IntentSpec
    parameters: dict[str, ParameterSpec]
    preflight: tuple[Observation, ...]
    verification: VerificationSpec
    compensation: CompensationSpec | None = None
    approval: ApprovalSpec = Field(default_factory=ApprovalSpec)
    failure_policy: FailurePolicy = Field(default_factory=FailurePolicy, alias="failurePolicy")

    @model_validator(mode="after")
    def validate_atomic(self) -> "AtomicEffectSpec":
        if not self.profiles:
            raise ValueError("atomic effect requires at least one profile")
        if not self.preflight:
            raise ValueError("atomic effect requires an independent preflight observation")
        if any(item.capability == self.effect.capability for item in self.preflight):
            raise ValueError("preflight must use an observation capability, not the effect capability")
        if self.verification.capability == self.effect.capability:
            raise ValueError("verification must use an independent observation capability")
        preflight_ids = [item.id for item in self.preflight]
        if len(set(preflight_ids)) != len(preflight_ids):
            raise ValueError("preflight observation ids must be unique")
        unknown_targets = set(self.intent.target_fields) - set(self.parameters)
        if unknown_targets:
            raise ValueError("intent target fields are not declared parameters: " + ", ".join(sorted(unknown_targets)))
        if not self.approval.required:
            raise ValueError("L0 effects cannot disable approval")
        if self.failure_policy.verification_failed == "compensate" and self.compensation is None:
            raise ValueError("compensate failure policy requires a compensation contract")
        return self


class AtomicEffectManifest(StrictModel):
    api_version: Literal[API_VERSION] = Field(alias="apiVersion")
    kind: Literal["AtomicEffect"]
    metadata: Metadata
    spec: AtomicEffectSpec


class DerivedEffectSpec(StrictModel):
    constrain_parameters: dict[str, ParameterPatch] = Field(
        default_factory=dict, alias="constrainParameters",
    )
    add_parameters: dict[str, ParameterSpec] = Field(default_factory=dict, alias="addParameters")
    add_preflight: tuple[Observation, ...] = Field(default_factory=tuple, alias="addPreflight")
    add_verification: tuple[Predicate, ...] = Field(default_factory=tuple, alias="addVerification")
    desired_state_additions: dict[str, Any] = Field(
        default_factory=dict, alias="desiredStateAdditions",
    )
    intent_kind: str | None = Field(default=None, alias="intentKind")
    approval: ApprovalSpec | None = None


class DerivedEffectManifest(StrictModel):
    api_version: Literal[API_VERSION] = Field(alias="apiVersion")
    kind: Literal["DerivedEffect"]
    metadata: Metadata
    extends: SkillRef
    mode: Literal["constraint", "extension"]
    spec: DerivedEffectSpec

    @model_validator(mode="after")
    def validate_mode(self) -> "DerivedEffectManifest":
        if self.mode == "constraint" and (
            self.spec.add_parameters or self.spec.add_preflight
            or self.spec.add_verification or self.spec.desired_state_additions
        ):
            raise ValueError("constraint derivation may only narrow parameters and strengthen approval")
        if not self.spec.constrain_parameters and self.spec.approval is None and self.mode == "constraint":
            raise ValueError("constraint derivation does not change the parent")
        return self


class CompositeStep(StrictModel):
    id: str
    skill_ref: SkillRef = Field(alias="skillRef")
    depends_on: tuple[str, ...] = Field(default_factory=tuple, alias="dependsOn")
    arguments: dict[str, Any]
    domain: str = "network"


class Checkpoint(StrictModel):
    id: str
    after: tuple[str, ...]
    observations: tuple[Observation, ...]


class CompositeEffectSpec(StrictModel):
    inputs: dict[str, ParameterSpec]
    steps: tuple[CompositeStep, ...]
    checkpoints: tuple[Checkpoint, ...] = ()
    approval: ApprovalSpec = Field(default_factory=lambda: ApprovalSpec(risk="high"))
    compensation_order: Literal["reverse"] = Field(default="reverse", alias="compensationOrder")

    @model_validator(mode="after")
    def validate_steps(self) -> "CompositeEffectSpec":
        ids = [step.id for step in self.steps]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("composite step ids must be non-empty and unique")
        positions = {step_id: index for index, step_id in enumerate(ids)}
        for index, step in enumerate(self.steps):
            invalid = {
                dependency for dependency in step.depends_on
                if dependency not in positions or positions[dependency] >= index
            }
            if invalid:
                raise ValueError(
                    f"step {step.id!r} has missing or forward dependencies: {sorted(invalid)}"
                )
        for checkpoint in self.checkpoints:
            unknown = set(checkpoint.after) - set(ids)
            if unknown:
                raise ValueError(f"checkpoint {checkpoint.id!r} references unknown steps: {sorted(unknown)}")
        checkpoint_ids = [item.id for item in self.checkpoints]
        if len(set(checkpoint_ids)) != len(checkpoint_ids):
            raise ValueError("checkpoint ids must be unique")
        if not self.approval.required:
            raise ValueError("composite effects cannot disable approval")
        return self


class CompositeEffectManifest(StrictModel):
    api_version: Literal[API_VERSION] = Field(alias="apiVersion")
    kind: Literal["CompositeEffect"]
    metadata: Metadata
    spec: CompositeEffectSpec


class BoundCompositeStep(StrictModel):
    id: str
    skill_ref: SkillRef = Field(alias="skillRef")
    contract_hash: str = Field(alias="contractHash")
    capability: str
    compensation_capability: str | None = Field(default=None, alias="compensationCapability")
    depends_on: tuple[str, ...] = Field(default_factory=tuple, alias="dependsOn")
    arguments: dict[str, Any]
    domain: str


class CompiledAtomicEffect(StrictModel):
    api_version: Literal[COMPILED_API_VERSION] = Field(alias="apiVersion")
    kind: Literal["CompiledAtomicEffect"]
    metadata: Metadata
    derivation: Literal["base", "constraint", "extension"]
    lineage: tuple[SkillRef, ...] = ()
    spec: AtomicEffectSpec
    contract_hash: str = Field(alias="contractHash")


class CompiledCompositeEffect(StrictModel):
    api_version: Literal[COMPILED_API_VERSION] = Field(alias="apiVersion")
    kind: Literal["CompiledCompositeEffect"]
    metadata: Metadata
    inputs: dict[str, ParameterSpec]
    steps: tuple[BoundCompositeStep, ...]
    checkpoints: tuple[Checkpoint, ...]
    approval: ApprovalSpec
    compensation_order: Literal["reverse"] = Field(alias="compensationOrder")
    definition_hash: str = Field(alias="definitionHash")


AuthoringManifest = AtomicEffectManifest | DerivedEffectManifest | CompositeEffectManifest
CompiledContract = CompiledAtomicEffect | CompiledCompositeEffect
