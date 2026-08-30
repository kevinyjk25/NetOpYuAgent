"""Review-gated promotion of an Agent Skill into an L0 v2 candidate.

The model may draft a candidate, but this module owns deterministic parsing,
capability cross-checks, compilation, provenance binding, and human review.
Approval creates no Runtime registration and grants no execution authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from network_runtime.contracts import canonical_json, sha256_json, utc_now
from skills.skill_format import ParsedSkill, parse_skill_md, to_flat_dict

from .catalog import L0Catalog
from .compiler import L0CompileError, compile_documents, load_documents, parse_document
from .models import (
    AtomicEffectManifest,
    AuthoringManifest,
    CompiledAtomicEffect,
    CompiledCompositeEffect,
    CompiledContract,
    CompositeEffectManifest,
    DerivedEffectManifest,
    IntentSpec,
    ParameterSpec,
    value_matches_type,
)


PROMOTION_SCHEMA = "netopyu.io/l0-promotion-report/v2"
CAPABILITY_API_VERSION = "netopyu.io/capability-catalog/v1"
L05_API_VERSION = "netopyu.io/l0.5-structured-skill/v2"
TRAJECTORY_SCHEMA = "netopyu.io/l0-promotion-trajectory/v1"
SEMANTIC_COVERAGE_SCHEMA = "netopyu.io/l0-semantic-coverage/v1"
_DIRECT_ARGUMENT = re.compile(r"^\$\{\s*arguments\.([A-Za-z_][A-Za-z0-9_]*)\s*\}$")
_L1_INTENT_BLOCK = re.compile(
    r"<!--\s*netopyu:semantic-intents/v1\s*-->\s*"
    r"```ya?ml\s*\n(?P<body>.*?)\n```",
    re.DOTALL | re.IGNORECASE,
)


class PromotionError(ValueError):
    pass


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class CapabilityDefinition(_StrictModel):
    id: str
    role: Literal["observation", "effect", "compensation"]
    tool: str | None = None
    profiles: tuple[str, ...]
    inputs: dict[str, ParameterSpec]
    outputs: dict[str, ParameterSpec]

    @model_validator(mode="after")
    def require_profile(self) -> "CapabilityDefinition":
        if not self.profiles:
            raise ValueError("capability profiles cannot be empty")
        return self


class CapabilityCatalogManifest(_StrictModel):
    api_version: Literal[CAPABILITY_API_VERSION] = Field(alias="apiVersion")
    provider: str
    version: str
    capabilities: tuple[CapabilityDefinition, ...]

    @model_validator(mode="after")
    def unique_capabilities(self) -> "CapabilityCatalogManifest":
        ids = [item.id for item in self.capabilities]
        if len(set(ids)) != len(ids):
            raise ValueError("capability ids must be unique")
        if not self.provider.strip() or not self.version.strip():
            raise ValueError("capability provider and version are required")
        return self

    def by_id(self) -> dict[str, CapabilityDefinition]:
        return {item.id: item for item in self.capabilities}


class L05WorkflowStep(_StrictModel):
    phase: Literal[
        "validate", "preflight", "approval", "effect", "verification",
        "compensation",
    ]
    instruction: str
    capability_options: tuple[str, ...] = Field(alias="capabilityOptions")


class L05CapabilityOptions(_StrictModel):
    effects: tuple[str, ...]
    observations: tuple[str, ...]
    compensations: tuple[str, ...]


class L05Safety(_StrictModel):
    risk: Literal["low", "medium", "high", "critical"]
    approval_required: bool = Field(alias="approvalRequired")
    unknown_outcome_policy: str = Field(alias="unknownOutcomePolicy")
    stop_conditions: tuple[str, ...] = Field(alias="stopConditions")
    non_compensable_justification: str | None = Field(
        default=None,
        alias="nonCompensableJustification",
    )


class L05Outcomes(_StrictModel):
    success: str
    failure: str
    rollback: str


class L05SemanticIntent(_StrictModel):
    """Exact, capability-scoped intent preserved across L1 -> L0.5 -> L0."""

    effect_capability: str = Field(alias="effectCapability")
    kind: str
    target_fields: tuple[str, ...] = Field(alias="targetFields")
    desired_state: dict[str, Any] = Field(alias="desiredState")

    @model_validator(mode="after")
    def validate_intent(self) -> "L05SemanticIntent":
        if not self.effect_capability.strip() or not self.kind.strip():
            raise ValueError("semantic intent capability and kind cannot be empty")
        if not self.target_fields or len(set(self.target_fields)) != len(self.target_fields):
            raise ValueError("semantic intent targetFields must be non-empty and unique")
        return self

    def intent_spec(self) -> IntentSpec:
        return IntentSpec(
            kind=self.kind,
            targetFields=self.target_fields,
            desiredState=self.desired_state,
        )


class StructuredNaturalLanguageSkill(_StrictModel):
    """Human-reviewable bridge between free-form L1 and executable L0."""

    api_version: Literal[L05_API_VERSION] = Field(alias="apiVersion")
    kind: Literal["StructuredNaturalLanguageSkill"]
    skill_id: str = Field(alias="skillId")
    title: str
    purpose: str
    profiles: tuple[str, ...]
    parameters: dict[str, str]
    semantic_intents: tuple[L05SemanticIntent, ...] = Field(alias="semanticIntents")
    constraints: tuple[str, ...]
    workflow: tuple[L05WorkflowStep, ...]
    capabilities: L05CapabilityOptions
    safety: L05Safety
    outcomes: L05Outcomes
    source_skill_sha256: str = Field(alias="sourceSkillSha256")
    capability_catalog_sha256: str = Field(alias="capabilityCatalogSha256")
    previous_stage_sha256: str = Field(alias="previousStageSha256")
    unresolved_questions: tuple[str, ...] = Field(alias="unresolvedQuestions")

    @model_validator(mode="after")
    def validate_stage_link(self) -> "StructuredNaturalLanguageSkill":
        if self.previous_stage_sha256 != self.source_skill_sha256:
            raise ValueError("L0.5 previousStageSha256 must bind the L1 source hash")
        if not self.profiles:
            raise ValueError("L0.5 profiles cannot be empty")
        capabilities = [item.effect_capability for item in self.semantic_intents]
        if len(set(capabilities)) != len(capabilities):
            raise ValueError("L0.5 semanticIntents must be unique per effect capability")
        return self


@dataclass(frozen=True)
class SkillSource:
    path: Path
    text: str
    parsed: ParsedSkill
    skill_id: str
    definition: dict[str, Any]
    digest: str

    @property
    def declared_tools(self) -> frozenset[str]:
        return frozenset({
            *(str(item) for item in self.definition.get("allowed_tools", ())),
            *(str(item) for item in self.definition.get("tool_deps", ())),
        })


@dataclass(frozen=True)
class PromotionAssessment:
    report: dict[str, Any]
    bound_manifest: AuthoringManifest
    compiled_contract: CompiledContract | None


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _text_digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_skill_source(path: str | Path) -> SkillSource:
    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.name != "SKILL.md":
        raise PromotionError("promotion source must be an existing SKILL.md")
    text = source.read_text(encoding="utf-8")
    parsed = parse_skill_md(text)
    skill_id, definition = to_flat_dict(parsed, skill_id_hint=source.parent.name.replace("-", "_"))
    if parsed.name != source.parent.name:
        raise PromotionError("SKILL.md name must match its containing directory")
    return SkillSource(source, text, parsed, skill_id, definition, _text_digest(text))


def load_capability_catalog(path: str | Path) -> tuple[CapabilityCatalogManifest, str, Path]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise PromotionError(f"capability catalog does not exist: {source}")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise PromotionError("capability catalog must be a YAML object")
        catalog = CapabilityCatalogManifest.model_validate(raw)
    except (yaml.YAMLError, ValidationError) as error:
        raise PromotionError(f"invalid capability catalog: {error}") from error
    return catalog, _file_digest(source), source


def extract_l1_semantic_intents(source: SkillSource) -> tuple[L05SemanticIntent, ...]:
    """Read the optional machine-checkable intent block from an L1 Skill.

    Arbitrary prose is never guessed into executable intent.  Authors keep the
    surrounding Skill natural-language-first, but exact effect/target/state
    semantics use one small, visibly marked YAML block that can be compared at
    every later stage.
    """

    matches = list(_L1_INTENT_BLOCK.finditer(source.parsed.body))
    if not matches:
        return ()
    if len(matches) != 1:
        raise PromotionError("L1 Skill must contain at most one semantic intent block")
    try:
        raw = yaml.safe_load(matches[0].group("body"))
        if not isinstance(raw, list) or not raw:
            raise PromotionError("L1 semantic intent block must be a non-empty YAML list")
        return tuple(L05SemanticIntent.model_validate(item) for item in raw)
    except (yaml.YAMLError, ValidationError, TypeError) as error:
        raise PromotionError(f"invalid L1 semantic intent block: {error}") from error


def build_l05_spec(
    *, skill_path: str | Path, capability_catalog_path: str | Path,
) -> StructuredNaturalLanguageSkill:
    """Create a deterministic, reviewable L0.5 semantic worksheet.

    The worksheet preserves the L1 prose and exposes only trusted capability
    choices. It intentionally does not guess which observer or compensation is
    correct when the provider catalog offers multiple options.
    """

    source = load_skill_source(skill_path)
    catalog, catalog_digest, _ = load_capability_catalog(capability_catalog_path)
    grouped = {
        role: tuple(sorted(item.id for item in catalog.capabilities if item.role == role))
        for role in ("observation", "effect", "compensation")
    }
    declared_effects = tuple(sorted(
        item.id for item in catalog.capabilities
        if item.role == "effect" and item.tool in source.declared_tools
    ))
    effect_options = declared_effects or grouped["effect"]
    semantic_intents = extract_l1_semantic_intents(source)
    constraints = tuple(str(item) for item in source.definition.get("constraints", ()))
    non_compensable_justification = next((
        item.split(":", 1)[1].strip()
        for item in constraints
        if item.lower().startswith("non-compensable justification:")
    ), None)
    unresolved: list[str] = []
    if len(effect_options) != 1:
        unresolved.append(
            "Select exactly one primary effect capability; the trusted catalog "
            f"currently offers {list(effect_options)}."
        )
    if not semantic_intents:
        unresolved.append(
            "Declare the exact capability-scoped semantic intent in the marked "
            "L1 semantic-intents block; Runtime will not infer intent from prose."
        )
    unknown_intent_effects = sorted({
        item.effect_capability for item in semantic_intents
        if item.effect_capability not in effect_options
    })
    if unknown_intent_effects:
        unresolved.append(
            "Semantic intent references effect capabilities outside the trusted L1/catalog "
            f"intersection: {unknown_intent_effects}."
        )
    if not grouped["observation"]:
        unresolved.append(
            "Declare an independent observation capability for preflight and verification."
        )
    if not grouped["compensation"] and not non_compensable_justification:
        unresolved.append(
            "Declare a compensation capability or explicitly justify a non-compensable effect."
        )
    workflow = (
        L05WorkflowStep(
            phase="validate",
            instruction=(
                "校验每个已记录参数并解析关键身份，不得推断。 / Validate every "
                "documented parameter and resolve critical identities without inference."
            ),
            capabilityOptions=(),
        ),
        L05WorkflowStep(
            phase="preflight",
            instruction=(
                "通过独立 Observation 读取当前状态并保留 revision 与回滚证据。 / "
                "Read current state through an independent observation and preserve "
                "revision plus rollback evidence."
            ),
            capabilityOptions=grouped["observation"],
        ),
        L05WorkflowStep(
            phase="approval",
            instruction=(
                "在任何 Effect 前展示并绑定精确不可变变更。 / Present and bind "
                "the exact immutable change before any effect."
            ),
            capabilityOptions=(),
        ),
        L05WorkflowStep(
            phase="effect",
            instruction=(
                "只用已批准且通过 Schema 的输入调用一次受审 Effect。 / Invoke one "
                "reviewed effect with only approved and schema-valid inputs."
            ),
            capabilityOptions=effect_options,
        ),
        L05WorkflowStep(
            phase="verification",
            instruction=(
                "通过独立 Observation 判定成功，禁止只依赖写响应。 / Determine "
                "success from an independent observation, never from the write response alone."
            ),
            capabilityOptions=grouped["observation"],
        ),
        L05WorkflowStep(
            phase="compensation",
            instruction=(
                "验证失败时恢复精确前态并独立验证恢复结果。 / On failed verification, "
                "restore the exact prior state and independently verify restoration."
            ),
            capabilityOptions=grouped["compensation"],
        ),
    )
    return StructuredNaturalLanguageSkill(
        apiVersion=L05_API_VERSION,
        kind="StructuredNaturalLanguageSkill",
        skillId=source.skill_id,
        title=str(source.definition.get("name") or source.parsed.name),
        purpose=str(
            source.definition.get("purpose")
            or source.parsed.frontmatter["description"]
        ),
        profiles=tuple(str(item) for item in source.definition.get("profiles", ("default",))),
        parameters={
            str(name): str(value)
            for name, value in (source.definition.get("parameters") or {}).items()
        },
        semanticIntents=semantic_intents,
        constraints=constraints,
        workflow=workflow,
        capabilities=L05CapabilityOptions(
            effects=effect_options,
            observations=grouped["observation"],
            compensations=grouped["compensation"],
        ),
        safety=L05Safety(
            risk=str(source.definition.get("risk_level", "low")),
            approvalRequired=bool(source.definition.get("requires_hitl")),
            unknownOutcomePolicy=(
                "任何重试前先只读对账，禁止盲目重复 Effect。 / Read-only reconciliation "
                "before any retry; never blindly repeat an effect."
            ),
            stopConditions=tuple(
                item for item in constraints
                if any(
                    token in item.lower()
                    for token in ("never", "must", "mandatory", "required")
                )
            ),
            nonCompensableJustification=non_compensable_justification,
        ),
        outcomes=L05Outcomes(
            success=str(source.definition.get("returns") or "Independently verified desired state."),
            failure=(
                "未验证不得成功；失败关闭或进入明确恢复态。 / No unverified success; "
                "fail closed or enter an explicit recovery state."
            ),
            rollback=(
                "补偿后必须独立验证精确前态。 / Exact pre-change state independently "
                "verified after compensation."
            ),
        ),
        sourceSkillSha256=source.digest,
        capabilityCatalogSha256=catalog_digest,
        previousStageSha256=source.digest,
        unresolvedQuestions=tuple(unresolved),
    )


def load_l05_spec(
    path: str | Path,
    *, expected_source_digest: str,
    expected_capability_digest: str,
) -> StructuredNaturalLanguageSkill:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise PromotionError(f"L0.5 specification does not exist: {source}")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise PromotionError("L0.5 specification must be a YAML object")
        spec = StructuredNaturalLanguageSkill.model_validate(raw)
    except (yaml.YAMLError, ValidationError) as error:
        raise PromotionError(f"invalid L0.5 specification: {error}") from error
    if spec.source_skill_sha256 != expected_source_digest:
        raise PromotionError("L0.5 source Skill hash does not match the current L1")
    if spec.capability_catalog_sha256 != expected_capability_digest:
        raise PromotionError("L0.5 Capability Catalog hash does not match the current catalog")
    return spec


def l05_yaml(spec: StructuredNaturalLanguageSkill) -> str:
    return yaml.safe_dump(
        spec.model_dump(by_alias=True, mode="json"),
        sort_keys=False,
        allow_unicode=True,
    )


def _l05_digest(spec: StructuredNaturalLanguageSkill) -> str:
    return _text_digest(l05_yaml(spec))


def _resolve_l05(
    *,
    skill_path: str | Path,
    capability_catalog_path: str | Path,
    l05_path: str | Path | None,
) -> StructuredNaturalLanguageSkill:
    source = load_skill_source(skill_path)
    _, capability_digest, _ = load_capability_catalog(capability_catalog_path)
    if l05_path is None:
        return build_l05_spec(
            skill_path=skill_path,
            capability_catalog_path=capability_catalog_path,
        )
    return load_l05_spec(
        l05_path,
        expected_source_digest=source.digest,
        expected_capability_digest=capability_digest,
    )


def _bind_source(
    manifest: AuthoringManifest,
    source: SkillSource,
    capability_digest: str,
    l05_digest: str,
) -> AuthoringManifest:
    raw = manifest.model_dump(by_alias=True, mode="python")
    labels = dict(raw["metadata"].get("labels") or {})
    labels.update({
        "source-skill": source.parsed.name,
        "source-sha256": source.digest,
        "capability-catalog-sha256": capability_digest,
        "l0.5-sha256": l05_digest,
        "promotion-state": "candidate",
    })
    raw["metadata"]["labels"] = labels
    return parse_document(raw, source="source-bound promotion candidate")


def _finding(
    findings: list[dict[str, Any]],
    severity: Literal["error", "warning", "info"],
    code: str,
    message: str,
    evidence: Any = None,
) -> None:
    value: dict[str, Any] = {"severity": severity, "code": code, "message": message}
    if evidence is not None:
        value["evidence"] = evidence
    findings.append(value)


def _normalized_requirement_text(value: str) -> str:
    return " ".join(value.casefold().split()).strip(" .;,:，。；：")


def _requirement_id(category: str, source_path: str, text: str) -> str:
    digest = hashlib.sha256(
        f"{category}\n{source_path}\n{_normalized_requirement_text(text)}".encode("utf-8")
    ).hexdigest()[:12]
    return f"REQ-{category.upper().replace('_', '-')}-{digest}"


def _classify_requirement(text: str) -> str:
    value = _normalized_requirement_text(text)
    if any(token in value for token in (
        "unknown write", "uncertain write", "indeterminate", "blind retry",
        "read-only reconciliation", "不确定", "盲目重试", "只读对账",
    )):
        return "unknown_outcome"
    if any(token in value for token in (
        "approval", "approve", "hitl", "审批", "批准",
    )):
        return "approval"
    if any(token in value for token in (
        "rollback", "restore", "restoration", "compensat", "previous state",
        "pre-change", "回滚", "恢复", "补偿", "前态",
    )):
        return "compensation"
    if any(token in value for token in (
        "write response", "api response", "provider response", "readback",
        "independent verification", "independently verify", "verify the new",
        "verification", "verifier", "after the write", "after write",
        "写响应", "返回不能", "独立验证", "回读", "写入后",
    )):
        return "verification"
    if any(token in value for token in (
        "current state", "read first", "snapshot", "revision", "preflight",
        "re-read", "drift", "读取当前", "先读取", "快照", "版本号", "重读", "漂移",
    )):
        return "preflight"
    if any(token in value for token in (
        "inactive", "active user", "active employee", "identity is unknown",
        "existing user", "身份", "用户存在", "活跃", "禁用", "停用",
    )):
        return "precondition"
    if any(token in value for token in (
        "never infer", "do not infer", "without inference", "without inferring", "do not invent",
        "不得推断", "禁止推断", "不得编造",
    )):
        return "parameter_integrity"
    if any(token in value for token in (
        "exactly once", "grant access once", "send once", "invoke once",
        "effect once", "调用一次", "只发送一次", "发送一次", "写入一次",
    )):
        return "effect"
    return "business_rule"


def _requirement_criticality(category: str) -> str:
    if category in {
        "approval", "risk", "precondition", "parameter_integrity", "effect",
        "verification", "unknown_outcome", "compensation", "semantic_intent",
    }:
        return "safety_critical"
    if category in {"parameter", "profile", "preflight"}:
        return "operational"
    return "business"


def _body_directives(source: SkillSource) -> list[str]:
    """Extract reviewable prose statements from the complete Skill body.

    The Runtime does not pretend to understand arbitrary prose.  This bounded
    extraction makes otherwise easy-to-lose safety and business sentences
    visible.  Example sections and code fences are excluded; unclassified
    prose remains a human-review responsibility in the coverage matrix.
    """

    paragraphs: list[str] = []
    current: list[str] = []
    section = ""
    in_code = False

    def flush() -> None:
        if current:
            paragraphs.append(" ".join(current))
            current.clear()

    for raw in source.parsed.body.splitlines():
        stripped = raw.strip()
        if stripped.startswith("```"):
            flush()
            in_code = not in_code
            continue
        if in_code:
            continue
        if stripped.startswith("## "):
            flush()
            section = stripped[3:].strip().casefold()
            continue
        if stripped.startswith("#"):
            flush()
            continue
        if not stripped:
            flush()
            continue
        if any(token in section for token in (
            "example", "示例", "样例", "parameter", "参数",
        )):
            continue
        cleaned = re.sub(r"^(?:>|[-*+]\s+|\d+[.)]\s+)", "", stripped).strip()
        if cleaned:
            current.append(cleaned)
    flush()
    statements: list[str] = []
    for paragraph in paragraphs:
        for item in re.split(r"(?<=[.!?。！？])\s+", paragraph):
            statement = item.strip()
            if not statement or (statement.startswith("/") and statements):
                continue
            statements.append(statement)
    return statements


def _source_requirements(source: SkillSource) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(category: str, source_path: str, text: str) -> None:
        normalized = _normalized_requirement_text(text)
        key = (category, normalized)
        if not normalized or key in seen:
            return
        seen.add(key)
        values.append({
            "id": _requirement_id(category, source_path, text),
            "category": category,
            "criticality": _requirement_criticality(category),
            "source": {
                "stage": "L1",
                "file": "01-L1-SKILL.md",
                "path": source_path,
                "text": text,
                "sha256": _text_digest(text),
            },
        })

    for name, description in (source.definition.get("parameters") or {}).items():
        text = str(description)
        add("parameter", f"parameters.{name}", f"{name}: {text}")
        if _classify_requirement(text) == "precondition":
            add("precondition", f"parameters.{name}#precondition", text)
    for index, intent in enumerate(extract_l1_semantic_intents(source)):
        add(
            "semantic_intent",
            f"semanticIntents[{index}]",
            canonical_json(intent.model_dump(by_alias=True, mode="json")),
        )
    profiles = tuple(str(item) for item in source.definition.get("profiles") or ("default",))
    add("profile", "metadata.profiles", "Allowed profiles: " + ", ".join(profiles))
    source_risk = str(source.definition.get("risk_level", "low"))
    add("risk", "metadata.risk_level", f"Risk must be at least {source_risk}.")
    if bool(source.definition.get("requires_hitl")):
        add("approval", "metadata.requires_hitl", "Human approval is required.")
    for index, text in enumerate(source.definition.get("constraints") or ()):
        add(_classify_requirement(str(text)), f"constraints[{index}]", str(text))
    for index, text in enumerate(source.definition.get("steps") or ()):
        add(_classify_requirement(str(text)), f"steps[{index}]", str(text))
    for index, text in enumerate(_body_directives(source)):
        add(_classify_requirement(text), f"body[{index}]", text)
    return values


def _evidence(stage: str, path: str, value: Any) -> dict[str, Any]:
    return {"stage": stage, "path": path, "value": value}


def _l05_evidence(
    requirement: dict[str, Any], l05: StructuredNaturalLanguageSkill,
) -> list[dict[str, Any]]:
    category = requirement["category"]
    text = str(requirement["source"]["text"])
    normalized = _normalized_requirement_text(text)
    values: list[dict[str, Any]] = []
    if category == "semantic_intent":
        try:
            index = int(requirement["source"]["path"].split("[", 1)[1].rstrip("]"))
        except (ValueError, IndexError):
            index = -1
        if 0 <= index < len(l05.semantic_intents):
            values.append(_evidence(
                "L0.5",
                f"semanticIntents[{index}]",
                l05.semantic_intents[index].model_dump(by_alias=True, mode="json"),
            ))
    elif category == "parameter":
        name = requirement["source"]["path"].split(".", 1)[1]
        if name in l05.parameters:
            values.append(_evidence("L0.5", f"parameters.{name}", l05.parameters[name]))
    elif category == "profile":
        values.append(_evidence("L0.5", "profiles", list(l05.profiles)))
    elif category == "risk":
        values.append(_evidence("L0.5", "safety.risk", l05.safety.risk))
    elif category == "approval":
        values.append(_evidence(
            "L0.5", "safety.approvalRequired", l05.safety.approval_required,
        ))
    elif category == "unknown_outcome":
        values.append(_evidence(
            "L0.5", "safety.unknownOutcomePolicy", l05.safety.unknown_outcome_policy,
        ))
    elif category == "precondition":
        corpus = [
            *l05.constraints, *l05.safety.stop_conditions,
            *(item.instruction for item in l05.workflow if item.phase in {"validate", "preflight"}),
            *l05.parameters.values(),
        ]
        identity_tokens = (
            "active", "inactive", "identity", "existing", "unknown",
            "身份", "活跃", "禁用", "停用", "存在",
        )
        for index, item in enumerate(corpus):
            item_normalized = _normalized_requirement_text(str(item))
            if (
                item_normalized == normalized
                or any(token in normalized and token in item_normalized for token in identity_tokens)
            ):
                values.append(_evidence("L0.5", f"preconditionEvidence[{index}]", item))
    elif category == "parameter_integrity":
        for index, step in enumerate(l05.workflow):
            if step.phase == "validate" and any(token in step.instruction.casefold() for token in (
                "without inference", "不得推断", "不得", "validate every",
            )):
                values.append(_evidence("L0.5", f"workflow[{index}].instruction", step.instruction))
    elif category == "preflight":
        for index, step in enumerate(l05.workflow):
            if step.phase == "preflight":
                values.append(_evidence("L0.5", f"workflow[{index}]", step.model_dump(by_alias=True, mode="json")))
    elif category == "verification":
        for index, step in enumerate(l05.workflow):
            if step.phase == "verification":
                values.append(_evidence("L0.5", f"workflow[{index}]", step.model_dump(by_alias=True, mode="json")))
    elif category == "compensation":
        for index, step in enumerate(l05.workflow):
            if step.phase == "compensation":
                values.append(_evidence("L0.5", f"workflow[{index}]", step.model_dump(by_alias=True, mode="json")))
        values.append(_evidence("L0.5", "outcomes.rollback", l05.outcomes.rollback))
    elif category == "effect":
        for index, step in enumerate(l05.workflow):
            if step.phase == "effect":
                values.append(_evidence("L0.5", f"workflow[{index}]", step.model_dump(by_alias=True, mode="json")))
    else:
        for index, item in enumerate(l05.constraints):
            if _normalized_requirement_text(item) == normalized:
                values.append(_evidence("L0.5", f"constraints[{index}]", item))
    return values


def _compiled_atomics(
    compiled: CompiledContract | None, catalog: L0Catalog | None,
) -> list[CompiledAtomicEffect]:
    if isinstance(compiled, CompiledAtomicEffect):
        return [compiled]
    if isinstance(compiled, CompiledCompositeEffect) and catalog is not None:
        return [
            child for step in compiled.steps
            if isinstance((child := catalog.require(step.skill_ref.id, step.skill_ref.version)), CompiledAtomicEffect)
        ]
    return []


def _number_range(value: str) -> tuple[float, float] | None:
    match = re.search(
        r"(?:from\s+)?(-?\d+(?:\.\d+)?)\s*(?:through|to|[-~至到])\s*(-?\d+(?:\.\d+)?)",
        value.casefold(),
    )
    return (float(match.group(1)), float(match.group(2))) if match else None


def _semantic_requirement_mapping(
    requirement: dict[str, Any],
    *,
    l05: StructuredNaturalLanguageSkill,
    atomics: list[CompiledAtomicEffect],
) -> dict[str, Any]:
    category = requirement["category"]
    source_text = str(requirement["source"]["text"])
    normalized = _normalized_requirement_text(source_text)
    l05_evidence = _l05_evidence(requirement, l05)
    l0_evidence: list[dict[str, Any]] = []
    verdict = "preserved"
    reason = "The requirement is represented in L0.5 and enforced by the compiled L0 contract."
    fix_stage = "none"
    fix_path = ""
    hint = "No change required."

    def fail(value: str, why: str, stage: str, path: str, action: str) -> None:
        nonlocal verdict, reason, fix_stage, fix_path, hint
        verdict, reason, fix_stage, fix_path, hint = value, why, stage, path, action

    if category == "business_rule":
        verdict = "non_machine_verifiable"
        reason = (
            "The business-language rule has no deterministic L0 proof; it remains "
            "visible for independent review."
            if not l05_evidence else
            "The business-language constraint is preserved in L0.5 but has no "
            "deterministic L0 predicate mapping."
        )
        fix_stage = "L0.5"
        fix_path = "constraints"
        hint = "Map this rule to a predicate/capability or explicitly certify it as manual."
    elif not l05_evidence:
        l05_fix_paths = {
            "semantic_intent": "semanticIntents",
            "parameter": "parameters",
            "profile": "profiles",
            "risk": "safety.risk",
            "approval": "safety.approvalRequired",
            "precondition": "constraints / safety.stopConditions",
            "parameter_integrity": "workflow[phase=validate]",
            "preflight": "workflow[phase=preflight]",
            "verification": "workflow[phase=verification]",
            "unknown_outcome": "safety.unknownOutcomePolicy",
            "compensation": "workflow[phase=compensation] / outcomes.rollback",
            "effect": "workflow[phase=effect]",
        }
        fail(
            "missing",
            "The L1 requirement has no explicit L0.5 representation.",
            "L0.5", l05_fix_paths.get(category, "constraints"),
            "Add an explicit structured constraint/workflow mapping, then regenerate L0.",
        )
    elif category == "semantic_intent":
        try:
            expected = json.loads(source_text)
        except json.JSONDecodeError:
            expected = {}
        l05_value = (
            l05_evidence[0]["value"] if l05_evidence else {}
        )
        compiled_values = [
            {
                "effectCapability": item.spec.effect.capability,
                **item.spec.intent.model_dump(by_alias=True, mode="json"),
            }
            for item in atomics
        ]
        l0_evidence.extend(
            _evidence("L0", f"atomics[{index}].spec.intent", value)
            for index, value in enumerate(compiled_values)
        )
        if canonical_json(l05_value) != canonical_json(expected):
            fail(
                "weakened",
                "L0.5 semantic intent is not an exact copy of the marked L1 intent.",
                "L0.5", "semanticIntents",
                "Restore kind, targetFields, desiredState, and effectCapability exactly from L1.",
            )
        elif not any(
            canonical_json(value) == canonical_json(expected)
            for value in compiled_values
        ):
            fail(
                "weakened",
                "No compiled L0 atomic effect exactly preserves this L0.5 semantic intent.",
                "L0", "spec.intent",
                "Align effect capability, intent kind, targetFields, and desiredState with L0.5.",
            )
    elif category == "parameter":
        name = requirement["source"]["path"].split(".", 1)[1]
        specs = [item.spec.parameters.get(name) for item in atomics]
        specs = [item for item in specs if item is not None]
        for index, spec in enumerate(specs):
            l0_evidence.append(_evidence(
                "L0", f"atomics[{index}].spec.parameters.{name}",
                spec.model_dump(by_alias=True, mode="json"),
            ))
        expected_range = _number_range(source_text)
        source_required = not any(token in normalized for token in (
            "optional", "可选", "非必填",
        ))
        if not specs:
            fail("missing", f"Parameter {name!r} is absent from L0.", "L0", f"spec.parameters.{name}", "Declare and bind the parameter in L0 authoring.")
        elif source_required and any(not item.required for item in specs):
            fail("weakened", f"L1 parameter {name!r} is not required by every L0 effect.", "L0", f"spec.parameters.{name}.required", "Make the parameter required in every affected L0 contract.")
        elif expected_range and any(
            (
                item.type in {"integer", "number"}
                and (
                    item.minimum is None or item.maximum is None
                    or item.minimum < expected_range[0] or item.maximum > expected_range[1]
                )
            )
            or (
                item.type in {"string", "array"}
                and (
                    item.min_length is None or item.max_length is None
                    or item.min_length < expected_range[0] or item.max_length > expected_range[1]
                )
            )
            for item in specs
        ):
            fail("weakened", f"L0 does not preserve the documented range {expected_range} for {name!r}.", "L0", f"spec.parameters.{name}", "Narrow minimum/maximum to the L1 range.")
    elif category == "profile":
        l0_profiles = {profile for item in atomics for profile in item.spec.profiles}
        l0_evidence.append(_evidence("L0", "spec.profiles", sorted(l0_profiles)))
        source_profiles = {
            item.strip() for item in source_text.split(":", 1)[-1].split(",") if item.strip()
        }
        if not l0_profiles:
            fail("missing", "L0 has no profile scope.", "L0", "spec.profiles", "Declare the exact supported profiles.")
        elif not l0_profiles.issubset(set(l05.profiles)) or not set(l05.profiles).issubset(source_profiles):
            fail("weakened", "The profile scope is wider than an earlier stage.", "L0.5", "profiles", "Narrow L0.5 and L0 profiles to the L1 scope.")
    elif category == "risk":
        ranks = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        risks = [item.spec.approval.risk for item in atomics]
        l0_evidence.append(_evidence("L0", "spec.approval.risk", risks))
        source_risk = source_text.rstrip(".").rsplit(" ", 1)[-1]
        if ranks[l05.safety.risk] < ranks.get(source_risk, 0):
            fail("weakened", "L0.5 risk is weaker than L1.", "L0.5", "safety.risk", "Raise L0.5 risk to at least the L1 level.")
        elif not risks:
            fail("missing", "L0 has no enforceable approval risk.", "L0", "spec.approval.risk", "Declare approval risk in L0.")
        elif any(ranks[item] < ranks.get(source_risk, 0) for item in risks):
            fail("weakened", "L0 risk is weaker than L1.", "L0", "spec.approval.risk", "Raise L0 approval risk to at least the L1 level.")
        elif any(ranks[item] > ranks.get(source_risk, 0) for item in risks):
            verdict = "strengthened"
            reason = "L0 enforces a stronger risk classification than L1."
    elif category == "approval":
        approvals = [item.spec.approval.required for item in atomics]
        l0_evidence.append(_evidence("L0", "spec.approval.required", approvals))
        if not l05.safety.approval_required:
            fail("weakened", "L0.5 removes mandatory L1 approval.", "L0.5", "safety.approvalRequired", "Set approvalRequired to true.")
        elif not approvals or not all(approvals):
            fail("weakened", "L0 does not require approval for every effect.", "L0", "spec.approval.required", "Require approval for every affected L0 effect.")
    elif category == "precondition":
        predicates = [
            (atomic_index, observation_index, predicate_index, predicate)
            for atomic_index, atomic in enumerate(atomics)
            for observation_index, observation in enumerate(atomic.spec.preflight)
            for predicate_index, predicate in enumerate(observation.predicates)
        ]
        for ai, oi, pi, predicate in predicates:
            l0_evidence.append(_evidence(
                "L0", f"atomics[{ai}].spec.preflight[{oi}].predicates[{pi}]",
                predicate.model_dump(mode="json"),
            ))
        needs_active = any(token in normalized for token in (
            "active", "inactive", "活跃", "禁用", "停用",
        ))
        active_enforced = any(
            (
                "active" in predicate.field.casefold()
                and predicate.operator == "equals" and predicate.expected is True
            ) or (
                "status" in predicate.field.casefold()
                and predicate.operator == "equals"
                and str(predicate.expected).casefold() == "active"
            )
            for _, _, _, predicate in predicates
        )
        existence_enforced = active_enforced or any(
            predicate.operator == "exists"
            and any(token in predicate.field.casefold() for token in (
                "user", "identity", "status", "active", "facts", "record",
            ))
            for _, _, _, predicate in predicates
        )
        if not predicates:
            fail("missing", "L0 has no machine-checkable precondition predicate.", "L0", "spec.preflight.predicates", "Add an independent preflight predicate for this requirement.")
        elif needs_active and not active_enforced:
            fail("weakened", "L0 proves only that data exists; it does not prove the identity is active.", "L0", "spec.preflight.predicates", "Add an explicit status=active or active=true predicate backed by a declared observation field.")
        elif not needs_active and not existence_enforced:
            fail("ambiguous", "The identity/existence requirement is not explicitly proved by L0.", "L0", "spec.preflight.predicates", "Add an explicit existence predicate or mark the requirement for reviewed manual certification.")
    elif category == "parameter_integrity":
        parameter_items = [
            (name, parameter)
            for item in atomics for name, parameter in item.spec.parameters.items()
        ]
        parameters = [parameter for _, parameter in parameter_items]
        direct_bindings = [
            value for item in atomics for value in item.spec.effect.request.values()
            if isinstance(value, str) and _DIRECT_ARGUMENT.fullmatch(value)
        ]
        l0_evidence.append(_evidence("L0", "spec.parameters", {
            "required": sum(item.required for item in parameters),
            "directEffectBindings": len(direct_bindings),
        }))
        named_parameters = {
            name for name, _ in parameter_items if name.casefold() in normalized
        }
        if not parameters:
            fail("missing", "L0 has no parameter contract for values L1 forbids the model to infer.", "L0", "spec.parameters", "Declare strict parameters and their sources.")
        elif any(
            name in named_parameters and not parameter.required
            for name, parameter in parameter_items
        ):
            fail("weakened", "A parameter explicitly required by L1 can be omitted in L0.", "L0", "spec.parameters", "Require every explicitly named no-inference parameter.")
    elif category == "preflight":
        observations = [item for atomic in atomics for item in atomic.spec.preflight]
        l0_evidence.extend(
            _evidence("L0", f"spec.preflight[{index}]", item.model_dump(by_alias=True, mode="json"))
            for index, item in enumerate(observations)
        )
        l05_preflight = " ".join(str(item["value"]) for item in l05_evidence).casefold()
        if not any(token in l05_preflight for token in (
            "read", "observation", "snapshot", "revision", "读取", "快照", "前态",
        )):
            fail("weakened", "L0.5 no longer requires observable preflight state.", "L0.5", "workflow[phase=preflight]", "Require independent read, predicates, and rollback evidence.")
        elif not observations:
            fail("missing", "L0 has no independent preflight observation.", "L0", "spec.preflight", "Add preflight observation, predicates, and snapshot fields.")
        elif any(not item.snapshot_fields or not item.predicates for item in observations):
            fail("weakened", "L0 preflight does not both preserve state and enforce a predicate.", "L0", "spec.preflight", "Add snapshot fields and machine-checkable predicates.")
    elif category == "verification":
        verifications = [item.spec.verification for item in atomics]
        l0_evidence.extend(
            _evidence("L0", f"atomics[{index}].spec.verification", item.model_dump(mode="json"))
            for index, item in enumerate(verifications)
        )
        l05_verification = " ".join(str(item["value"]) for item in l05_evidence).casefold()
        if not any(token in l05_verification for token in (
            "independent", "never from", "write response", "readback", "独立", "写响应", "回读",
        )):
            fail("weakened", "L0.5 no longer requires independent verification.", "L0.5", "workflow[phase=verification]", "State that success requires independent observation rather than the write response.")
        elif not verifications:
            fail("missing", "L0 has no independent result verification.", "L0", "spec.verification", "Add an observation-backed verification contract.")
        elif any(
            verification.capability == atomics[index].spec.effect.capability
            or not verification.predicates
            for index, verification in enumerate(verifications)
        ):
            fail("weakened", "L0 verification is not independent and predicate-backed.", "L0", "spec.verification", "Use an independent observation capability with explicit predicates.")
    elif category == "unknown_outcome":
        policies = [item.spec.failure_policy.after_send_unknown for item in atomics]
        l0_evidence.append(_evidence("L0", "spec.failurePolicy.afterSendUnknown", policies))
        l05_policy = l05.safety.unknown_outcome_policy.casefold()
        l05_safe = any(token in l05_policy for token in (
            "read-only", "reconciliation", "never blind", "只读", "对账", "禁止盲目",
        ))
        if not l05_safe:
            fail("weakened", "L0.5 no longer forbids blind retry after an uncertain write.", "L0.5", "safety.unknownOutcomePolicy", "Require read-only reconciliation or manual intervention before retry.")
        elif not policies or any(item not in {"reconcile_read_only", "manual_intervention"} for item in policies):
            fail("weakened", "L0 permits unsafe handling of an unknown write outcome.", "L0", "spec.failurePolicy.afterSendUnknown", "Use reconcile_read_only or manual_intervention.")
    elif category == "compensation":
        compensations = [item.spec.compensation for item in atomics]
        l0_evidence.extend(
            _evidence("L0", f"atomics[{index}].spec.compensation", (
                item.model_dump(mode="json") if item is not None else None
            ))
            for index, item in enumerate(compensations)
        )
        l05_compensation = " ".join(str(item["value"]) for item in l05_evidence).casefold()
        if not any(token in l05_compensation for token in (
            "restore", "compensat", "rollback", "恢复", "补偿", "回滚",
        )):
            fail("weakened", "L0.5 no longer represents the L1 recovery requirement.", "L0.5", "workflow[phase=compensation]", "Describe compensation and independent restoration verification.")
        elif not compensations or any(item is None for item in compensations):
            manual_terminal = bool(l05.safety.non_compensable_justification) and all(
                item.spec.failure_policy.verification_failed == "manual_intervention"
                for item in atomics
            )
            if manual_terminal:
                l0_evidence.append(_evidence(
                    "L0", "spec.failurePolicy.verificationFailed",
                    [item.spec.failure_policy.verification_failed for item in atomics],
                ))
                reason = (
                    "The effect is explicitly justified as non-compensable and L0 "
                    "fails into manual intervention rather than claiming restoration."
                )
            else:
                fail("missing", "L1 requires recovery but L0 has no compensation for every effect.", "L0", "spec.compensation", "Declare compensation or explicitly justify a non-compensable effect in L0.5 and use manual_intervention.")
        elif any(not item.verification.predicates for item in compensations if item is not None):
            fail("weakened", "Compensation is not independently verified.", "L0", "spec.compensation.verification", "Add independent restoration verification predicates.")
        elif any(token in normalized for token in ("exact snapshot", "exact pre-change", "精确前态", "精确快照")):
            exact = all(
                any("preflight" in str(value) for value in item.arguments.values())
                and any(predicate.operator == "exact_snapshot" for predicate in item.verification.predicates)
                for item in compensations if item is not None
            )
            if not exact:
                fail("ambiguous", "L1 requires exact prior-state restoration but L0 does not bind and compare the snapshot exactly.", "L0", "spec.compensation", "Bind the preflight snapshot into compensation and verify with exact_snapshot.")
    elif category == "effect":
        effects = [item.spec.effect for item in atomics]
        l0_evidence.extend(
            _evidence("L0", f"atomics[{index}].spec.effect", item.model_dump(mode="json"))
            for index, item in enumerate(effects)
        )
        if not effects:
            fail("missing", "L0 has no bounded effect for the L1 action.", "L0", "spec.effect", "Bind the action to one reviewed effect capability.")
    else:
        verdict = "non_machine_verifiable"
        reason = "No deterministic mapper exists for this requirement category."
        fix_stage = "L0.5"
        fix_path = "constraints"
        hint = "Map this rule to a predicate/capability or explicitly certify it as manual."

    if verdict == "preserved" and not l0_evidence:
        fail("missing", "The requirement has no concrete L0 enforcement evidence.", "L0", "03-L0-authoring.yaml", "Add an explicit L0 binding or predicate.")
    blocks = verdict in {"missing", "weakened"} or (
        verdict == "ambiguous" and requirement["criticality"] == "safety_critical"
    )
    fidelity_scores = {
        "preserved": 95,
        "strengthened": 100,
        "non_machine_verifiable": 20,
        "ambiguous": 35,
        "weakened": 10,
        "missing": 0,
    }
    confidence_components = {
        "sourceTraceability": 100,
        "l05Representation": 100 if l05_evidence else 0,
        "l0Enforcement": 100 if l0_evidence else 0,
        "semanticFidelity": fidelity_scores[verdict],
    }
    confidence_weights = {
        "sourceTraceability": 0.10,
        "l05Representation": 0.15,
        "l0Enforcement": 0.20,
        "semanticFidelity": 0.55,
    }
    confidence_score = round(sum(
        confidence_components[name] * weight
        for name, weight in confidence_weights.items()
    ))
    confidence_band = (
        "high" if confidence_score >= 85 else
        "medium" if confidence_score >= 65 else
        "low"
    )
    confidence_basis = [
        "The L1 requirement is source-bound to an immutable path and digest.",
        (
            f"{len(l05_evidence)} explicit L0.5 evidence item(s) were found."
            if l05_evidence else
            "No explicit L0.5 evidence was found."
        ),
        (
            f"{len(l0_evidence)} compiled L0 enforcement evidence item(s) were found."
            if l0_evidence else
            "No compiled L0 enforcement evidence was found."
        ),
        f"The deterministic semantic verdict is {verdict!r}.",
    ]
    if verdict == "preserved":
        loss_type, loss_risk, loss_explanation = (
            "none", 0,
            "No detected semantic loss: the requirement is explicit in L0.5 and enforced in L0.",
        )
    elif verdict == "strengthened":
        loss_type, loss_risk, loss_explanation = (
            "strengthened_no_loss", 0,
            "No detected loss: L0 narrows or strengthens the earlier safety requirement.",
        )
    elif verdict == "non_machine_verifiable":
        loss_type, loss_risk, loss_explanation = (
            "manual_semantic_gap", 55,
            "The language is retained for review, but no deterministic L0 predicate proves it.",
        )
    elif verdict == "ambiguous":
        loss_type, loss_risk, loss_explanation = (
            "ambiguous_mapping", 75,
            "The available evidence permits more than one interpretation and needs review.",
        )
    elif verdict == "weakened":
        loss_type, loss_risk, loss_explanation = (
            "semantic_weakening", 80,
            "A restriction exists in later stages but is weaker than the L1 requirement.",
        )
    elif l05_evidence:
        loss_type, loss_risk, loss_explanation = (
            "l05_to_l0_loss", 100,
            "The requirement reached L0.5 but has no concrete compiled L0 enforcement.",
        )
    else:
        loss_type, loss_risk, loss_explanation = (
            "l1_to_l05_loss", 100,
            "The requirement was lost before or during the L1 to L0.5 translation.",
        )
    attention_required = confidence_band == "low" or verdict == "non_machine_verifiable"
    l1_to_l05_verdict = (
        "missing" if not l05_evidence else
        verdict if verdict in {"weakened", "ambiguous"} and fix_stage == "L0.5" else
        "preserved"
    )
    l1_to_l05_fidelity = {
        "preserved": 100,
        "ambiguous": 35,
        "weakened": 10,
        "missing": 0,
    }[l1_to_l05_verdict]
    l1_to_l05_score = round(40 + 0.60 * l1_to_l05_fidelity)
    l1_to_l05_loss = {
        "preserved": 0,
        "ambiguous": 75,
        "weakened": 80,
        "missing": 100,
    }[l1_to_l05_verdict]
    if not l05_evidence:
        l05_to_l0_verdict = "upstream_missing"
        l05_to_l0_score = 0
        l05_to_l0_loss = 100
        l05_to_l0_explanation = (
            "L0.5 has no source representation, so L0 enforcement cannot be "
            "meaningfully compared until the upstream gap is fixed."
        )
    elif l1_to_l05_verdict in {"weakened", "ambiguous"}:
        l05_to_l0_verdict = "upstream_unresolved"
        l05_to_l0_score = 0
        l05_to_l0_loss = 100
        l05_to_l0_explanation = (
            "The L0.5 input is already weakened or ambiguous; repair L1 to L0.5 "
            "before treating the downstream comparison as authoritative."
        )
    else:
        l05_to_l0_verdict = verdict
        l05_to_l0_score = round(
            15
            + confidence_components["l0Enforcement"] * 0.30
            + confidence_components["semanticFidelity"] * 0.55
        )
        l05_to_l0_loss = loss_risk
        l05_to_l0_explanation = loss_explanation

    def transition_band(score: int) -> str:
        return "high" if score >= 85 else "medium" if score >= 65 else "low"

    return {
        **requirement,
        "l05Evidence": l05_evidence,
        "l0Evidence": l0_evidence,
        "verdict": verdict,
        "blocksPromotion": blocks,
        "reason": reason,
        "mappingConfidence": {
            "score": confidence_score,
            "band": confidence_band,
            "method": "deterministic_evidence_v1",
            "components": confidence_components,
            "weights": confidence_weights,
            "basis": confidence_basis,
            "claimBoundary": (
                "This is a deterministic traceability score, not model confidence or a "
                "production success probability."
            ),
        },
        "languageLoss": {
            "type": loss_type,
            "riskPercent": loss_risk,
            "explanation": loss_explanation,
        },
        "transitionAssessments": {
            "l1ToL05": {
                "fromStage": "L1",
                "toStage": "L0.5",
                "score": l1_to_l05_score,
                "band": transition_band(l1_to_l05_score),
                "verdict": l1_to_l05_verdict,
                "lossRiskPercent": l1_to_l05_loss,
                "explanation": (
                    "The L1 clause has explicit structured L0.5 evidence."
                    if l1_to_l05_verdict == "preserved" else reason
                ),
            },
            "l05ToL0": {
                "fromStage": "L0.5",
                "toStage": "L0",
                "score": l05_to_l0_score,
                "band": transition_band(l05_to_l0_score),
                "verdict": l05_to_l0_verdict,
                "lossRiskPercent": l05_to_l0_loss,
                "explanation": l05_to_l0_explanation,
            },
        },
        "attentionRequired": attention_required,
        "alertLevel": "critical" if blocks else (
            "warning" if attention_required else "none"
        ),
        "fix": {
            "stage": fix_stage,
            "file": "02-L0.5.yaml" if fix_stage == "L0.5" else (
                "03-L0-authoring.yaml" if fix_stage == "L0" else None
            ),
            "path": fix_path,
            "hint": hint,
        },
    }


def _semantic_coverage(
    *,
    source: SkillSource,
    l05: StructuredNaturalLanguageSkill,
    compiled: CompiledContract | None,
    catalog: L0Catalog | None,
) -> dict[str, Any]:
    requirements = _source_requirements(source)
    atomics = _compiled_atomics(compiled, catalog)
    mappings = [
        _semantic_requirement_mapping(item, l05=l05, atomics=atomics)
        for item in requirements
    ]
    counts = {
        verdict: sum(item["verdict"] == verdict for item in mappings)
        for verdict in (
            "preserved", "strengthened", "weakened", "missing", "ambiguous",
            "non_machine_verifiable",
        )
    }
    total = len(mappings)
    represented = counts["preserved"] + counts["strengthened"] + counts["non_machine_verifiable"]
    machine_enforced = counts["preserved"] + counts["strengthened"]
    blocked = sum(bool(item["blocksPromotion"]) for item in mappings)
    attention = sum(bool(item["attentionRequired"]) for item in mappings)
    low_confidence = sum(
        item["mappingConfidence"]["band"] == "low" for item in mappings
    )
    language_loss = sum(
        item["languageLoss"]["riskPercent"] > 0 for item in mappings
    )
    average_confidence = (
        round(sum(item["mappingConfidence"]["score"] for item in mappings) / total, 2)
        if total else 100.0
    )
    average_l1_to_l05_confidence = (
        round(sum(
            item["transitionAssessments"]["l1ToL05"]["score"] for item in mappings
        ) / total, 2)
        if total else 100.0
    )
    average_l05_to_l0_confidence = (
        round(sum(
            item["transitionAssessments"]["l05ToL0"]["score"] for item in mappings
        ) / total, 2)
        if total else 100.0
    )
    source_tools = set(source.declared_tools)
    extra_effects = sorted({
        atomic.spec.effect.tool
        for atomic in atomics
        if source_tools and atomic.spec.effect.tool and atomic.spec.effect.tool not in source_tools
    })
    return {
        "schema": SEMANTIC_COVERAGE_SCHEMA,
        "gate": "blocked" if blocked or extra_effects else "passed",
        "claimBoundary": (
            "Deterministic coverage of extracted requirements is not a proof of arbitrary "
            "natural-language equivalence. Non-machine-verifiable rules require independent review."
        ),
        "summary": {
            "totalRequirements": total,
            **counts,
            "blockingRequirements": blocked,
            "attentionRequirements": attention,
            "lowConfidenceRequirements": low_confidence,
            "languageLossRequirements": language_loss,
            "averageMappingConfidence": average_confidence,
            "averageL1ToL05Confidence": average_l1_to_l05_confidence,
            "averageL05ToL0Confidence": average_l05_to_l0_confidence,
            "semanticCoveragePercent": round(100.0 * represented / total, 2) if total else 100.0,
            "machineEnforcedPercent": round(100.0 * machine_enforced / total, 2) if total else 100.0,
            "extraEffects": len(extra_effects),
        },
        "requirements": mappings,
        "extraEffects": extra_effects,
    }


def _parameter_contract_is_subset(source: ParameterSpec, target: ParameterSpec) -> bool:
    if source.type != target.type or (target.required and not source.required):
        return False
    if target.enum and (not source.enum or not set(source.enum).issubset(set(target.enum))):
        return False
    if target.minimum is not None and (source.minimum is None or source.minimum < target.minimum):
        return False
    if target.maximum is not None and (source.maximum is None or source.maximum > target.maximum):
        return False
    if target.min_length is not None and (
        source.min_length is None or source.min_length < target.min_length
    ):
        return False
    if target.max_length is not None and (
        source.max_length is None or source.max_length > target.max_length
    ):
        return False
    if target.pattern is not None and source.pattern != target.pattern:
        return False
    if target.sensitive and not source.sensitive:
        return False
    return True


def _literal_satisfies(value: Any, parameter: ParameterSpec) -> bool:
    if not value_matches_type(value, parameter.type):
        return False
    if parameter.enum and value not in parameter.enum:
        return False
    if parameter.minimum is not None and value < parameter.minimum:
        return False
    if parameter.maximum is not None and value > parameter.maximum:
        return False
    if parameter.min_length is not None and len(value) < parameter.min_length:
        return False
    if parameter.max_length is not None and len(value) > parameter.max_length:
        return False
    if parameter.pattern is not None and not re.fullmatch(parameter.pattern, value):
        return False
    return True


def _validate_call(
    *,
    capability: CapabilityDefinition,
    arguments: dict[str, Any],
    l0_parameters: dict[str, ParameterSpec],
    context: str,
    findings: list[dict[str, Any]],
) -> None:
    unknown = sorted(set(arguments) - set(capability.inputs))
    missing = sorted(
        name for name, parameter in capability.inputs.items()
        if parameter.required and parameter.fixed is None and name not in arguments
    )
    if unknown:
        _finding(findings, "error", "API_UNKNOWN_INPUT", f"{context} sends unknown API inputs", unknown)
    if missing:
        _finding(findings, "error", "API_REQUIRED_INPUT_MISSING", f"{context} misses API inputs", missing)
    for name, value in arguments.items():
        target = capability.inputs.get(name)
        if target is None:
            continue
        match = _DIRECT_ARGUMENT.fullmatch(value) if isinstance(value, str) else None
        if match:
            parameter = l0_parameters.get(match.group(1))
            if parameter is None or not _parameter_contract_is_subset(parameter, target):
                _finding(
                    findings, "error", "API_INPUT_CONTRACT_MISMATCH",
                    f"{context}.{name} is not guaranteed to satisfy the API input contract",
                    {"argument": match.group(1)},
                )
        elif "${" not in str(value) and not _literal_satisfies(value, target):
            _finding(
                findings, "error", "API_LITERAL_INVALID",
                f"{context}.{name} literal violates the API input contract",
            )


def _validate_observation(
    *,
    capability_id: str,
    arguments: dict[str, Any],
    fields: Iterable[str],
    profiles: tuple[str, ...],
    l0_parameters: dict[str, ParameterSpec],
    capabilities: dict[str, CapabilityDefinition],
    context: str,
    findings: list[dict[str, Any]],
) -> None:
    capability = capabilities.get(capability_id)
    if capability is None:
        _finding(findings, "error", "CAPABILITY_UNKNOWN", f"{context} capability is not declared", capability_id)
        return
    if capability.role != "observation":
        _finding(findings, "error", "CAPABILITY_ROLE_MISMATCH", f"{context} must use an observation capability", capability.role)
    if not set(profiles).issubset(set(capability.profiles)):
        _finding(findings, "error", "CAPABILITY_PROFILE_MISMATCH", f"{context} does not support every L0 profile")
    _validate_call(
        capability=capability, arguments=arguments, l0_parameters=l0_parameters,
        context=context, findings=findings,
    )
    unknown_fields = sorted({field.split(".", 1)[0] for field in fields} - set(capability.outputs))
    if unknown_fields:
        _finding(findings, "error", "API_OUTPUT_FIELD_UNKNOWN", f"{context} relies on undeclared output fields", unknown_fields)


def _validate_atomic_capabilities(
    contract: CompiledAtomicEffect,
    capabilities: dict[str, CapabilityDefinition],
    declared_tools: frozenset[str],
    findings: list[dict[str, Any]],
) -> None:
    spec = contract.spec
    effect = capabilities.get(spec.effect.capability)
    if effect is None:
        _finding(findings, "error", "CAPABILITY_UNKNOWN", "effect capability is not declared", spec.effect.capability)
    else:
        if effect.role != "effect":
            _finding(findings, "error", "CAPABILITY_ROLE_MISMATCH", "primary effect must use an effect capability", effect.role)
        if not set(spec.profiles).issubset(set(effect.profiles)):
            _finding(findings, "error", "CAPABILITY_PROFILE_MISMATCH", "effect does not support every L0 profile")
        if spec.effect.tool and effect.tool and spec.effect.tool != effect.tool:
            _finding(findings, "error", "CAPABILITY_TOOL_MISMATCH", "L0 tool and API catalog tool differ")
        model_tool = spec.effect.tool or effect.tool
        if model_tool and model_tool not in declared_tools:
            _finding(
                findings, "error", "L1_TOOL_BOUNDARY_MISMATCH",
                "L0 primary effect tool is not declared by the source Skill",
                {"tool": model_tool, "declared": sorted(declared_tools)},
            )
        _validate_call(
            capability=effect, arguments=spec.effect.request,
            l0_parameters=spec.parameters, context="effect", findings=findings,
        )
    for observation in spec.preflight:
        _validate_observation(
            capability_id=observation.capability,
            arguments=observation.arguments,
            fields=(*observation.snapshot_fields, *(item.field for item in observation.predicates)),
            profiles=spec.profiles, l0_parameters=spec.parameters,
            capabilities=capabilities, context=f"preflight.{observation.id}", findings=findings,
        )
    _validate_observation(
        capability_id=spec.verification.capability,
        arguments=spec.verification.arguments,
        fields=(item.field for item in spec.verification.predicates),
        profiles=spec.profiles, l0_parameters=spec.parameters,
        capabilities=capabilities, context="verification", findings=findings,
    )
    if spec.compensation is not None:
        compensation = capabilities.get(spec.compensation.capability)
        if compensation is None:
            _finding(findings, "error", "CAPABILITY_UNKNOWN", "compensation capability is not declared", spec.compensation.capability)
        else:
            if compensation.role != "compensation":
                _finding(findings, "error", "CAPABILITY_ROLE_MISMATCH", "compensation must use a compensation capability", compensation.role)
            _validate_call(
                capability=compensation, arguments=spec.compensation.arguments,
                l0_parameters=spec.parameters, context="compensation", findings=findings,
            )
        _validate_observation(
            capability_id=spec.compensation.verification.capability,
            arguments=spec.compensation.verification.arguments,
            fields=(item.field for item in spec.compensation.verification.predicates),
            profiles=spec.profiles, l0_parameters=spec.parameters,
            capabilities=capabilities, context="compensation.verification", findings=findings,
        )


def _load_candidate(path: str | Path) -> tuple[AuthoringManifest, str, Path]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise PromotionError(f"candidate manifest does not exist: {source}")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise PromotionError(f"invalid candidate YAML: {error}") from error
    if not isinstance(raw, dict):
        raise PromotionError("candidate manifest must be one YAML object")
    return parse_document(raw, source=str(source)), _file_digest(source), source


def _dependencies(paths: Iterable[str | Path]) -> list[AuthoringManifest]:
    values: list[AuthoringManifest] = []
    for path in paths:
        values.extend(load_documents(path))
    return values


def _validate_l05_contract(
    *,
    l05: StructuredNaturalLanguageSkill,
    compiled: CompiledContract,
    catalog: L0Catalog,
    findings: list[dict[str, Any]],
) -> None:
    if l05.unresolved_questions:
        _finding(
            findings,
            "error",
            "L05_UNRESOLVED_QUESTIONS",
            "L0.5 contains unresolved semantic questions",
            list(l05.unresolved_questions),
        )
    atomics: list[CompiledAtomicEffect] = []
    if isinstance(compiled, CompiledAtomicEffect):
        atomics.append(compiled)
    else:
        for step in compiled.steps:
            child = catalog.require(step.skill_ref.id, step.skill_ref.version)
            if isinstance(child, CompiledAtomicEffect):
                atomics.append(child)
    intents_by_effect = {
        item.effect_capability: item for item in l05.semantic_intents
    }
    compiled_effects = {item.spec.effect.capability for item in atomics}
    unimplemented_intents = sorted(set(intents_by_effect) - compiled_effects)
    if unimplemented_intents:
        _finding(
            findings, "error", "L05_INTENT_UNIMPLEMENTED",
            "L0.5 declares semantic intents with no compiled L0 effect",
            unimplemented_intents,
        )
    for atomic in atomics:
        spec = atomic.spec
        staged_intent = intents_by_effect.get(spec.effect.capability)
        if staged_intent is None:
            _finding(
                findings, "error", "L05_INTENT_MISSING",
                "L0.5 has no exact semantic intent for this L0 effect capability",
                spec.effect.capability,
            )
        elif canonical_json(staged_intent.intent_spec().model_dump(
            by_alias=True, mode="json",
        )) != canonical_json(spec.intent.model_dump(by_alias=True, mode="json")):
            _finding(
                findings, "error", "L05_INTENT_DRIFT",
                "L0 intent must exactly preserve L0.5 kind, targetFields and desiredState",
                {
                    "effectCapability": spec.effect.capability,
                    "l0.5": staged_intent.intent_spec().model_dump(
                        by_alias=True, mode="json",
                    ),
                    "l0": spec.intent.model_dump(by_alias=True, mode="json"),
                },
            )
        required_parameters = {
            name for name, parameter in spec.parameters.items()
            if parameter.required and parameter.fixed is None
        }
        missing_parameters = sorted(required_parameters - set(l05.parameters))
        if missing_parameters:
            _finding(
                findings, "error", "L05_PARAMETER_COVERAGE_MISSING",
                "L0.5 does not document every required L0 parameter",
                missing_parameters,
            )
        if spec.effect.capability not in l05.capabilities.effects:
            _finding(
                findings, "error", "L05_EFFECT_MISMATCH",
                "L0 effect is not an allowed L0.5 capability",
                spec.effect.capability,
            )
        observations = {
            *(item.capability for item in spec.preflight),
            spec.verification.capability,
        }
        if spec.compensation is not None:
            observations.add(spec.compensation.verification.capability)
            if spec.compensation.capability not in l05.capabilities.compensations:
                _finding(
                    findings, "error", "L05_COMPENSATION_MISMATCH",
                    "L0 compensation is not an allowed L0.5 capability",
                    spec.compensation.capability,
                )
        elif not l05.safety.non_compensable_justification:
            _finding(
                findings, "error", "L05_NON_COMPENSABLE_UNJUSTIFIED",
                "L0 has no compensation and L0.5 contains no explicit justification",
            )
        unknown_observations = sorted(observations - set(l05.capabilities.observations))
        if unknown_observations:
            _finding(
                findings, "error", "L05_OBSERVATION_MISMATCH",
                "L0 observations are not allowed by L0.5",
                unknown_observations,
            )
        if not set(spec.profiles).issubset(set(l05.profiles)):
            _finding(
                findings, "error", "L05_PROFILE_MISMATCH",
                "L0 profiles are broader than the L0.5 scope",
                {"l0.5": list(l05.profiles), "l0": list(spec.profiles)},
            )
        if l05.safety.approval_required and not spec.approval.required:
            _finding(
                findings, "error", "L05_APPROVAL_MISMATCH",
                "L0 removes approval required by L0.5",
            )
        risk_rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        if risk_rank[spec.approval.risk] < risk_rank[l05.safety.risk]:
            _finding(
                findings, "error", "L05_RISK_WEAKENED",
                "L0 approval risk is weaker than L0.5",
            )


def _validate_l05_source(
    *,
    source: SkillSource,
    l05: StructuredNaturalLanguageSkill,
    findings: list[dict[str, Any]],
) -> None:
    if l05.skill_id != source.skill_id:
        _finding(
            findings, "error", "L05_SKILL_ID_MISMATCH",
            "L0.5 skillId does not match the L1 Skill",
        )
    source_intents = extract_l1_semantic_intents(source)
    if canonical_json([
        item.model_dump(by_alias=True, mode="json") for item in l05.semantic_intents
    ]) != canonical_json([
        item.model_dump(by_alias=True, mode="json") for item in source_intents
    ]):
        _finding(
            findings, "error", "L05_INTENT_DRIFT_FROM_L1",
            "L0.5 semanticIntents must exactly preserve the marked L1 intent block",
            {
                "l1": [item.model_dump(by_alias=True, mode="json") for item in source_intents],
                "l0.5": [
                    item.model_dump(by_alias=True, mode="json")
                    for item in l05.semantic_intents
                ],
            },
        )
    source_parameters = set(source.definition.get("parameters") or {})
    if set(l05.parameters) != source_parameters:
        _finding(
            findings, "error", "L05_PARAMETER_DRIFT",
            "L0.5 parameter names must exactly preserve the L1 parameter set",
            {
                "missing": sorted(source_parameters - set(l05.parameters)),
                "added": sorted(set(l05.parameters) - source_parameters),
            },
        )
    source_profiles = set(source.definition.get("profiles") or ("default",))
    if not set(l05.profiles).issubset(source_profiles):
        _finding(
            findings, "error", "L05_PROFILE_WIDENED",
            "L0.5 profiles are broader than the L1 Skill",
        )
    risk_rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    source_risk = str(source.definition.get("risk_level", "low"))
    if risk_rank[l05.safety.risk] < risk_rank[source_risk]:
        _finding(
            findings, "error", "L05_RISK_WEAKENED_FROM_L1",
            "L0.5 risk is weaker than the L1 Skill",
        )
    if bool(source.definition.get("requires_hitl")) and not l05.safety.approval_required:
        _finding(
            findings, "error", "L05_APPROVAL_WEAKENED_FROM_L1",
            "L0.5 removes approval required by the L1 Skill",
        )


def assess_promotion(
    *,
    skill_path: str | Path,
    candidate_path: str | Path,
    capability_catalog_path: str | Path,
    dependency_paths: Iterable[str | Path] = (),
    l05_path: str | Path | None = None,
) -> PromotionAssessment:
    source = load_skill_source(skill_path)
    capabilities, capability_digest, capability_path = load_capability_catalog(
        capability_catalog_path,
    )
    l05 = _resolve_l05(
        skill_path=skill_path,
        capability_catalog_path=capability_catalog_path,
        l05_path=l05_path,
    )
    l05_digest = _l05_digest(l05)
    candidate, candidate_digest, candidate_source = _load_candidate(candidate_path)
    bound = _bind_source(candidate, source, capability_digest, l05_digest)
    findings: list[dict[str, Any]] = []
    _validate_l05_source(source=source, l05=l05, findings=findings)
    compiled: CompiledContract | None = None
    catalog: L0Catalog | None = None
    try:
        contracts = compile_documents([*_dependencies(dependency_paths), bound])
        catalog = L0Catalog(contracts)
        compiled = catalog.require(bound.metadata.id, bound.metadata.version)
    except (L0CompileError, KeyError, ValidationError, ValueError) as error:
        _finding(findings, "error", "L0_COMPILE_FAILED", str(error))

    parameters = (
        compiled.spec.parameters if isinstance(compiled, CompiledAtomicEffect)
        else compiled.inputs if isinstance(compiled, CompiledCompositeEffect)
        else {}
    )
    source_parameters = set(source.definition.get("parameters") or {})
    required_parameters = {
        name for name, parameter in parameters.items()
        if parameter.required and parameter.fixed is None
    }
    missing_in_skill = sorted(required_parameters - source_parameters)
    if missing_in_skill:
        _finding(
            findings, "error", "L1_PARAMETER_COVERAGE_MISSING",
            "source Skill does not document every required L0 parameter", missing_in_skill,
        )
    extra_in_skill = sorted(source_parameters - set(parameters))
    if extra_in_skill:
        _finding(
            findings, "warning", "L1_PARAMETER_NOT_BOUND",
            "source Skill parameters are not represented by the candidate", extra_in_skill,
        )
    if compiled is not None:
        if catalog is None:
            raise AssertionError("compiled promotion requires a catalog")
        _validate_l05_contract(
            l05=l05,
            compiled=compiled,
            catalog=catalog,
            findings=findings,
        )
        atomics: list[CompiledAtomicEffect] = []
        if isinstance(compiled, CompiledAtomicEffect):
            atomics.append(compiled)
        elif catalog is not None:
            for step in compiled.steps:
                child = catalog.require(step.skill_ref.id, step.skill_ref.version)
                if isinstance(child, CompiledAtomicEffect):
                    atomics.append(child)
        for atomic in atomics:
            _validate_atomic_capabilities(
                atomic, capabilities.by_id(), source.declared_tools, findings,
            )
        candidate_risk = max(
            ({"low": 0, "medium": 1, "high": 2, "critical": 3}[item.spec.approval.risk]
             for item in atomics),
            default=0,
        )
        source_risk = {"low": 0, "medium": 1, "high": 2, "critical": 3}.get(
            str(source.definition.get("risk_level", "low")), 0,
        )
        if source_risk > candidate_risk:
            _finding(findings, "error", "L0_RISK_WEAKER_THAN_L1", "candidate approval risk is weaker than the source Skill")
        if not bool(source.definition.get("requires_hitl")):
            _finding(findings, "warning", "L1_HITL_NOT_DECLARED", "source Skill does not declare HITL; L0 approval remains mandatory")

    semantic_coverage = _semantic_coverage(
        source=source,
        l05=l05,
        compiled=compiled,
        catalog=catalog,
    )
    for requirement in semantic_coverage["requirements"]:
        if not requirement["blocksPromotion"]:
            continue
        verdict = str(requirement["verdict"])
        _finding(
            findings,
            "error",
            f"SEMANTIC_REQUIREMENT_{verdict.upper()}",
            requirement["reason"],
            {
                "requirementId": requirement["id"],
                "category": requirement["category"],
                "source": requirement["source"],
                "fix": requirement["fix"],
            },
        )
    if semantic_coverage["extraEffects"]:
        _finding(
            findings,
            "error",
            "SEMANTIC_EXTRA_EFFECT",
            "L0 contains an effect outside the L1 declared tool boundary.",
            {
                "extraEffects": semantic_coverage["extraEffects"],
                "fix": {
                    "stage": "L0",
                    "file": "03-L0-authoring.yaml",
                    "path": "spec.effect",
                    "hint": "Remove the undeclared effect or explicitly review and declare it in L1.",
                },
            },
        )

    error_count = sum(item["severity"] == "error" for item in findings)
    warning_count = sum(item["severity"] == "warning" for item in findings)
    compiled_hash = (
        getattr(compiled, "contract_hash", getattr(compiled, "definition_hash", None))
        if compiled is not None else None
    )
    report: dict[str, Any] = {
        "schema": PROMOTION_SCHEMA,
        "status": "blocked" if error_count else "ready_for_review",
        "executionEligible": False,
        "autoActivated": False,
        "sourceSkill": {
            "name": source.parsed.name,
            "skillId": source.skill_id,
            "path": str(source.path),
            "sha256": source.digest,
            "declaredTools": sorted(source.declared_tools),
            "documentedParameters": sorted(source_parameters),
        },
        "structuredSkill": {
            "apiVersion": l05.api_version,
            "skillId": l05.skill_id,
            "sha256": l05_digest,
            "previousStageSha256": l05.previous_stage_sha256,
            "unresolvedQuestions": list(l05.unresolved_questions),
        },
        "candidate": {
            "id": bound.metadata.id,
            "version": bound.metadata.version,
            "kind": bound.kind,
            "path": str(candidate_source),
            "inputSha256": candidate_digest,
            "boundManifestSha256": _text_digest(canonical_json(
                bound.model_dump(by_alias=True, mode="json"),
            )),
            "compiledHash": compiled_hash,
        },
        "capabilityCatalog": {
            "provider": capabilities.provider,
            "version": capabilities.version,
            "path": str(capability_path),
            "sha256": capability_digest,
        },
        "summary": {"errors": error_count, "warnings": warning_count},
        "findings": findings,
        "semanticCoverage": semantic_coverage,
        "manualCertificationRequired": [
            "API authentication and provider identity qualification",
            "independent observer freshness and schema qualification",
            "approval identity and separation-of-duty test",
            "effect timeout/indeterminate-outcome fault injection",
            "verification failure and exact rollback exercise",
            "DSH/Hermes semantic-entrypoint and model-selection evaluation",
        ],
    }
    report["proposalHash"] = sha256_json(report)
    return PromotionAssessment(report, bound, compiled)


def inspect_skill(skill_path: str | Path) -> dict[str, Any]:
    source = load_skill_source(skill_path)
    return {
        "ok": True,
        "name": source.parsed.name,
        "skill_id": source.skill_id,
        "description": source.parsed.frontmatter["description"],
        "sha256": source.digest,
        "declared_tools": sorted(source.declared_tools),
        "parameters": source.definition.get("parameters") or {},
        "risk_level": source.definition.get("risk_level"),
        "requires_hitl": source.definition.get("requires_hitl"),
    }


def promotion_prompt(
    *,
    skill_path: str | Path,
    capability_catalog_path: str | Path,
    l05_path: str | Path | None = None,
) -> str:
    source = load_skill_source(skill_path)
    capabilities, digest, path = load_capability_catalog(capability_catalog_path)
    l05 = _resolve_l05(
        skill_path=skill_path,
        capability_catalog_path=capability_catalog_path,
        l05_path=l05_path,
    )
    packet = {
        "schema": "netopyu.io/l0-promotion-prompt/v2",
        "task": (
            "Draft exactly one NetOpYu L0 v2 authoring YAML manifest from the "
            "source Agent Skill and trusted capability catalog."
        ),
        "trustBoundary": [
            "The Agent Skill is domain guidance, not execution authorization.",
            "Use only declared capability ids, tools, inputs, outputs, and profiles.",
            "Do not invent observers, rollback, fields, defaults, parameters, or approvals.",
            "If required semantics are missing, return NEEDS_CLARIFICATION with a list; do not guess.",
            "The output is an untrusted candidate and can never activate itself.",
            (
                "Treat the L0.5 structured Skill as the reviewed semantic bridge; "
                "do not widen its capability, profile, approval, or recovery boundaries."
            ),
        ],
        "requiredSemantics": [
            "exact effect and independent observation capabilities",
            "parameter types and limits",
            "target fields and desired state",
            "preflight snapshots/predicates",
            "postcondition predicates",
            "compensation and independent compensation verification",
            "risk and approval mode",
            "failure policy",
        ],
        "sourceSkill": {
            "path": str(source.path), "sha256": source.digest,
            "frontmatter": source.parsed.frontmatter, "body": source.parsed.body,
        },
        "structuredSkill": {
            "sha256": _l05_digest(l05),
            "document": l05.model_dump(by_alias=True, mode="json"),
        },
        "capabilityCatalog": {
            "path": str(path), "sha256": digest,
            "document": capabilities.model_dump(by_alias=True, mode="json"),
        },
        "outputSchemas": {
            "atomic": AtomicEffectManifest.model_json_schema(by_alias=True),
            "derived": DerivedEffectManifest.model_json_schema(by_alias=True),
            "composite": CompositeEffectManifest.model_json_schema(by_alias=True),
        },
    }
    return json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def package_promotion(
    *,
    skill_path: str | Path,
    candidate_path: str | Path,
    capability_catalog_path: str | Path,
    output_directory: str | Path,
    dependency_paths: Iterable[str | Path] = (),
    l05_path: str | Path | None = None,
) -> dict[str, Any]:
    assessment = assess_promotion(
        skill_path=skill_path, candidate_path=candidate_path,
        capability_catalog_path=capability_catalog_path,
        dependency_paths=dependency_paths,
        l05_path=l05_path,
    )
    if assessment.report["status"] != "ready_for_review" or assessment.compiled_contract is None:
        raise PromotionError("promotion is blocked; run promote-assess and resolve every error")
    destination = Path(output_directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    source = load_skill_source(skill_path)
    l05 = _resolve_l05(
        skill_path=skill_path,
        capability_catalog_path=capability_catalog_path,
        l05_path=l05_path,
    )
    capability_source = Path(capability_catalog_path).expanduser().resolve()
    capability_text = capability_source.read_text(encoding="utf-8")
    l05_text = l05_yaml(l05)
    bound_text = yaml.safe_dump(
        assessment.bound_manifest.model_dump(by_alias=True, mode="json"),
        sort_keys=False, allow_unicode=True,
    )
    compiled_text = json.dumps(
        assessment.compiled_contract.model_dump(by_alias=True, mode="json"),
        ensure_ascii=False, indent=2, sort_keys=True,
    ) + "\n"
    stage_files = {
        "00-capability-catalog.yaml": capability_text,
        "01-L1-SKILL.md": source.text,
        "02-L0.5.yaml": l05_text,
        "03-L0-authoring.yaml": bound_text,
        "04-L0-compiled.json": compiled_text,
    }
    for name, content in stage_files.items():
        (destination / name).write_text(content, encoding="utf-8")

    l1_digest = _text_digest(source.text)
    l05_digest = _l05_digest(l05)
    l0_authoring_digest = _text_digest(bound_text)
    l0_compiled_digest = _text_digest(compiled_text)
    trajectory: dict[str, Any] = {
        "schema": TRAJECTORY_SCHEMA,
        "executionEligible": False,
        "autoActivated": False,
        "capabilityCatalog": {
            "file": "00-capability-catalog.yaml",
            "sha256": _text_digest(capability_text),
        },
        "stages": [
            {
                "stage": "L1",
                "format": "anthropic-skill-markdown",
                "file": "01-L1-SKILL.md",
                "sha256": l1_digest,
                "previousSha256": None,
            },
            {
                "stage": "L0.5",
                "format": L05_API_VERSION,
                "file": "02-L0.5.yaml",
                "sha256": l05_digest,
                "previousSha256": l1_digest,
            },
            {
                "stage": "L0-authoring",
                "format": assessment.bound_manifest.api_version,
                "file": "03-L0-authoring.yaml",
                "sha256": l0_authoring_digest,
                "previousSha256": l05_digest,
            },
            {
                "stage": "L0-compiled",
                "format": assessment.compiled_contract.api_version,
                "file": "04-L0-compiled.json",
                "sha256": l0_compiled_digest,
                "previousSha256": l0_authoring_digest,
                "contractHash": assessment.report["candidate"]["compiledHash"],
            },
        ],
    }
    trajectory["trajectoryHash"] = sha256_json(trajectory)
    trajectory_text = json.dumps(
        trajectory, ensure_ascii=False, indent=2, sort_keys=True,
    ) + "\n"
    (destination / "trajectory.json").write_text(trajectory_text, encoding="utf-8")
    report = dict(assessment.report)
    report["packageFiles"] = {
        name: _text_digest(content)
        for name, content in {**stage_files, "trajectory.json": trajectory_text}.items()
    }
    report["trajectoryHash"] = trajectory["trajectoryHash"]
    report["proposalHash"] = sha256_json({
        key: value for key, value in report.items() if key != "proposalHash"
    })
    report_text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (destination / "report.json").write_text(report_text, encoding="utf-8")
    return {
        "ok": True, "status": "ready_for_review", "proposal": str(destination),
        "proposal_hash": report["proposalHash"],
        "trajectory_hash": trajectory["trajectoryHash"],
        "semantic_coverage": report["semanticCoverage"]["summary"],
        "auto_activated": False,
    }


def _validate_packaged_trajectory(destination: Path, expected_hash: str) -> None:
    trajectory_path = destination / "trajectory.json"
    try:
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PromotionError("proposal trajectory integrity check failed") from error
    if trajectory.get("schema") != TRAJECTORY_SCHEMA:
        raise PromotionError("proposal trajectory schema is invalid")
    stored_hash = trajectory.get("trajectoryHash")
    calculated_hash = sha256_json({
        key: value for key, value in trajectory.items() if key != "trajectoryHash"
    })
    if stored_hash != calculated_hash or stored_hash != expected_hash:
        raise PromotionError("proposal trajectory integrity check failed")
    previous: str | None = None
    expected_stages = ("L1", "L0.5", "L0-authoring", "L0-compiled")
    stages = trajectory.get("stages")
    if not isinstance(stages, list) or tuple(item.get("stage") for item in stages) != expected_stages:
        raise PromotionError("proposal trajectory stage order is invalid")
    for stage in stages:
        file_path = destination / str(stage.get("file", ""))
        if (
            stage.get("previousSha256") != previous
            or not file_path.is_file()
            or _file_digest(file_path) != stage.get("sha256")
        ):
            raise PromotionError(f"proposal trajectory stage integrity check failed: {stage.get('stage')}")
        previous = str(stage["sha256"])


def review_promotion(
    *, proposal_directory: str | Path, reviewer: str, decision: str, reason: str,
) -> dict[str, Any]:
    destination = Path(proposal_directory).expanduser().resolve()
    reviewer = reviewer.strip()
    decision = decision.strip().lower()
    if not reviewer:
        raise PromotionError("reviewer is required")
    if decision not in {"approve", "reject"}:
        raise PromotionError("decision must be approve or reject")
    report_path = destination / "report.json"
    if not report_path.is_file():
        raise PromotionError("proposal report.json does not exist")
    review_path = destination / "review.json"
    if review_path.exists():
        raise PromotionError("proposal has already been reviewed")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "ready_for_review":
        raise PromotionError("only a ready_for_review proposal can be reviewed")
    semantic_coverage = report.get("semanticCoverage")
    if (
        not isinstance(semantic_coverage, dict)
        or semantic_coverage.get("schema") != SEMANTIC_COVERAGE_SCHEMA
        or semantic_coverage.get("gate") != "passed"
        or not isinstance(semantic_coverage.get("summary"), dict)
        or semantic_coverage["summary"].get("blockingRequirements") != 0
        or semantic_coverage["summary"].get("extraEffects") != 0
    ):
        raise PromotionError(
            "proposal lacks a passing requirement-level semantic coverage gate; "
            "regenerate and reassess it"
        )
    stored_hash = report.get("proposalHash")
    calculated_hash = sha256_json({
        key: value for key, value in report.items() if key != "proposalHash"
    })
    if stored_hash != calculated_hash:
        raise PromotionError("proposal report integrity check failed")
    for name, expected in report.get("packageFiles", {}).items():
        file_path = destination / name
        if not file_path.is_file() or _file_digest(file_path) != expected:
            raise PromotionError(f"proposal file integrity check failed: {name}")
    _validate_packaged_trajectory(destination, str(report.get("trajectoryHash", "")))
    review = {
        "schema": "netopyu.io/l0-promotion-review/v2",
        "proposalHash": stored_hash,
        "decision": decision,
        "reviewer": reviewer,
        "reason": reason.strip(),
        "reviewedAt": utc_now(),
        "activatesRuntime": False,
        "grantsExecutionAuthority": False,
    }
    review["reviewHash"] = sha256_json(review)
    review_path.write_text(
        json.dumps(review, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "ok": True, "decision": decision, "reviewer": reviewer,
        "review_hash": review["reviewHash"], "auto_activated": False,
    }
