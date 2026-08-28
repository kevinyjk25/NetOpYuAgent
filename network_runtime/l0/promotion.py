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
    ParameterSpec,
    value_matches_type,
)


PROMOTION_SCHEMA = "netopyu.io/l0-promotion-report/v1"
CAPABILITY_API_VERSION = "netopyu.io/capability-catalog/v1"
_DIRECT_ARGUMENT = re.compile(r"^\$\{\s*arguments\.([A-Za-z_][A-Za-z0-9_]*)\s*\}$")


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


def _bind_source(
    manifest: AuthoringManifest,
    source: SkillSource,
    capability_digest: str,
) -> AuthoringManifest:
    raw = manifest.model_dump(by_alias=True, mode="python")
    labels = dict(raw["metadata"].get("labels") or {})
    labels.update({
        "source-skill": source.parsed.name,
        "source-sha256": source.digest,
        "capability-catalog-sha256": capability_digest,
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


def assess_promotion(
    *,
    skill_path: str | Path,
    candidate_path: str | Path,
    capability_catalog_path: str | Path,
    dependency_paths: Iterable[str | Path] = (),
) -> PromotionAssessment:
    source = load_skill_source(skill_path)
    capabilities, capability_digest, capability_path = load_capability_catalog(
        capability_catalog_path,
    )
    candidate, candidate_digest, candidate_source = _load_candidate(candidate_path)
    bound = _bind_source(candidate, source, capability_digest)
    findings: list[dict[str, Any]] = []
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
    *, skill_path: str | Path, capability_catalog_path: str | Path,
) -> str:
    source = load_skill_source(skill_path)
    capabilities, digest, path = load_capability_catalog(capability_catalog_path)
    packet = {
        "schema": "netopyu.io/l0-promotion-prompt/v1",
        "task": "Draft exactly one NetOpYu L0 v2 authoring YAML manifest from the source Agent Skill and trusted capability catalog.",
        "trustBoundary": [
            "The Agent Skill is domain guidance, not execution authorization.",
            "Use only declared capability ids, tools, inputs, outputs, and profiles.",
            "Do not invent observers, rollback, fields, defaults, parameters, or approvals.",
            "If required semantics are missing, return NEEDS_CLARIFICATION with a list; do not guess.",
            "The output is an untrusted candidate and can never activate itself.",
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
) -> dict[str, Any]:
    assessment = assess_promotion(
        skill_path=skill_path, candidate_path=candidate_path,
        capability_catalog_path=capability_catalog_path,
        dependency_paths=dependency_paths,
    )
    if assessment.report["status"] != "ready_for_review" or assessment.compiled_contract is None:
        raise PromotionError("promotion is blocked; run promote-assess and resolve every error")
    destination = Path(output_directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    source = load_skill_source(skill_path)
    bound_text = yaml.safe_dump(
        assessment.bound_manifest.model_dump(by_alias=True, mode="json"),
        sort_keys=False, allow_unicode=True,
    )
    compiled_text = json.dumps(
        assessment.compiled_contract.model_dump(by_alias=True, mode="json"),
        ensure_ascii=False, indent=2, sort_keys=True,
    ) + "\n"
    (destination / "SKILL.md").write_text(source.text, encoding="utf-8")
    (destination / "candidate.yaml").write_text(bound_text, encoding="utf-8")
    (destination / "compiled.json").write_text(compiled_text, encoding="utf-8")
    report = dict(assessment.report)
    report["packageFiles"] = {
        name: _text_digest(content) for name, content in {
            "SKILL.md": source.text,
            "candidate.yaml": bound_text,
            "compiled.json": compiled_text,
        }.items()
    }
    report["proposalHash"] = sha256_json({
        key: value for key, value in report.items() if key != "proposalHash"
    })
    report_text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (destination / "report.json").write_text(report_text, encoding="utf-8")
    return {
        "ok": True, "status": "ready_for_review", "proposal": str(destination),
        "proposal_hash": report["proposalHash"], "auto_activated": False,
    }


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
    review = {
        "schema": "netopyu.io/l0-promotion-review/v1",
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
