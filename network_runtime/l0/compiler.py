"""Compile strict L0 v2 manifests into flattened immutable contracts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import yaml
from pydantic import ValidationError

from network_runtime.contracts import sha256_json

from .models import (
    API_VERSION,
    APPROVAL_RANK,
    RISK_RANK,
    AtomicEffectManifest,
    AtomicEffectSpec,
    AuthoringManifest,
    BoundCompositeStep,
    CompiledAtomicEffect,
    CompiledCompositeEffect,
    CompiledContract,
    CompositeEffectManifest,
    DerivedEffectManifest,
    Metadata,
    ParameterPatch,
    ParameterSpec,
    SkillRef,
    VerificationSpec,
    IntentSpec,
    value_matches_type,
)


class L0CompileError(ValueError):
    pass


_REFERENCE = re.compile(r"\$\{\s*([^{}]+?)\s*\}")


def _references(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield from (match.group(1) for match in _REFERENCE.finditer(value))
    elif isinstance(value, dict):
        for child in value.values():
            yield from _references(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _references(child)


def _validate_atomic_references(spec: AtomicEffectSpec) -> None:
    preflight_ids = {item.id for item in spec.preflight}
    values: list[Any] = [
        spec.effect.request,
        spec.effect.internal,
        spec.intent.desired_state,
        *(item.arguments for item in spec.preflight),
        *(predicate.expected for item in spec.preflight for predicate in item.predicates),
        spec.verification.arguments,
        *(item.expected for item in spec.verification.predicates),
    ]
    if spec.compensation is not None:
        values.extend([
            spec.compensation.arguments,
            spec.compensation.verification.arguments,
            *(item.expected for item in spec.compensation.verification.predicates),
        ])
    for reference in _references(values):
        path = reference.split()[0]
        if path.startswith("arguments."):
            name = path.split(".", 1)[1]
            if name not in spec.parameters:
                raise L0CompileError(f"template references unknown argument {name!r}")
        elif path.startswith("preflight."):
            parts = path.split(".")
            if len(parts) < 3 or parts[1] not in preflight_ids:
                raise L0CompileError(f"template references unknown preflight path {path!r}")
        elif not path.startswith(("plan.", "intent.", "verification.")):
            raise L0CompileError(f"template uses unsupported reference root {path!r}")


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


def _input_is_compatible(source: ParameterSpec, target: ParameterSpec) -> bool:
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
    if target.resolver is not None and source.resolver != target.resolver:
        return False
    if target.sensitive and not source.sensitive:
        return False
    return True


def parse_document(value: dict[str, Any], *, source: str = "<memory>") -> AuthoringManifest:
    if value.get("apiVersion") != API_VERSION:
        raise L0CompileError(f"{source}: unsupported apiVersion {value.get('apiVersion')!r}")
    model = {
        "AtomicEffect": AtomicEffectManifest,
        "DerivedEffect": DerivedEffectManifest,
        "CompositeEffect": CompositeEffectManifest,
    }.get(value.get("kind"))
    if model is None:
        raise L0CompileError(f"{source}: unsupported L0 kind {value.get('kind')!r}")
    try:
        return model.model_validate(value)
    except ValidationError as error:
        raise L0CompileError(f"{source}: {error}") from error


def load_documents(path: str | Path) -> list[AuthoringManifest]:
    root = Path(path).expanduser().resolve()
    files = [root] if root.is_file() else sorted({*root.rglob("*.yaml"), *root.rglob("*.yml")})
    documents: list[AuthoringManifest] = []
    for file_path in files:
        raw = yaml.safe_load(file_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise L0CompileError(f"{file_path}: manifest must be a YAML object")
        documents.append(parse_document(raw, source=str(file_path)))
    return documents


def _stable_atomic(
    metadata: Metadata,
    derivation: str,
    lineage: tuple[SkillRef, ...],
    spec: AtomicEffectSpec,
) -> dict[str, Any]:
    return {
        "apiVersion": "netopyu.io/l0-effect-compiled/v2",
        "kind": "CompiledAtomicEffect",
        "metadata": metadata.model_dump(by_alias=True, mode="json"),
        "derivation": derivation,
        "lineage": [item.model_dump(by_alias=True, mode="json") for item in lineage],
        "spec": spec.model_dump(by_alias=True, mode="json"),
    }


def _compile_atomic(manifest: AtomicEffectManifest) -> CompiledAtomicEffect:
    _validate_atomic_references(manifest.spec)
    stable = _stable_atomic(manifest.metadata, "base", (), manifest.spec)
    return CompiledAtomicEffect.model_validate({
        **stable,
        "contractHash": sha256_json(stable),
    })


def _patch_parameter(name: str, parent: ParameterSpec, patch: ParameterPatch) -> ParameterSpec:
    values = parent.model_dump(by_alias=False, mode="python")
    changes = patch.model_dump(exclude_none=True, by_alias=False, mode="python")
    if "required" in changes and parent.required and not changes["required"]:
        raise L0CompileError(f"parameter {name}: derived contract cannot make a required field optional")
    if "enum" in changes:
        child_enum = set(changes["enum"])
        if parent.enum and not child_enum.issubset(set(parent.enum)):
            raise L0CompileError(f"parameter {name}: derived enum must be a subset of the parent")
    if "minimum" in changes and parent.minimum is not None and changes["minimum"] < parent.minimum:
        raise L0CompileError(f"parameter {name}: derived minimum cannot be lower")
    if "maximum" in changes and parent.maximum is not None and changes["maximum"] > parent.maximum:
        raise L0CompileError(f"parameter {name}: derived maximum cannot be higher")
    if "min_length" in changes and parent.min_length is not None and changes["min_length"] < parent.min_length:
        raise L0CompileError(f"parameter {name}: derived minLength cannot be lower")
    if "max_length" in changes and parent.max_length is not None and changes["max_length"] > parent.max_length:
        raise L0CompileError(f"parameter {name}: derived maxLength cannot be higher")
    if "pattern" in changes and parent.pattern is not None and changes["pattern"] != parent.pattern:
        raise L0CompileError(f"parameter {name}: an existing pattern cannot be replaced")
    if "sensitive" in changes and parent.sensitive and not changes["sensitive"]:
        raise L0CompileError(f"parameter {name}: derived contract cannot remove sensitivity")
    if "fixed" in changes:
        fixed = changes["fixed"]
        if not value_matches_type(fixed, parent.type):
            raise L0CompileError(f"parameter {name}: fixed value has the wrong type")
        if parent.fixed is not None and fixed != parent.fixed:
            raise L0CompileError(f"parameter {name}: fixed parent value cannot change")
        if parent.enum and fixed not in parent.enum:
            raise L0CompileError(f"parameter {name}: fixed value is outside the parent enum")
        if parent.minimum is not None and fixed < parent.minimum:
            raise L0CompileError(f"parameter {name}: fixed value is below the parent minimum")
        if parent.maximum is not None and fixed > parent.maximum:
            raise L0CompileError(f"parameter {name}: fixed value is above the parent maximum")
    values.update(changes)
    try:
        return ParameterSpec.model_validate(values)
    except ValidationError as error:
        raise L0CompileError(f"parameter {name}: {error}") from error


def _approval_is_monotonic(parent: Any, child: Any) -> bool:
    return (
        (not parent.required or child.required)
        and RISK_RANK[child.risk] >= RISK_RANK[parent.risk]
        and APPROVAL_RANK[child.mode] >= APPROVAL_RANK[parent.mode]
    )


def _compile_derived(
    manifest: DerivedEffectManifest,
    parent: CompiledAtomicEffect,
) -> CompiledAtomicEffect:
    spec = parent.spec
    parameters = dict(spec.parameters)
    for name, patch in manifest.spec.constrain_parameters.items():
        if name not in parameters:
            raise L0CompileError(f"derived contract constrains unknown parameter {name!r}")
        parameters[name] = _patch_parameter(name, parameters[name], patch)
    if manifest.mode == "constraint" and manifest.spec.add_parameters:
        raise L0CompileError("constraint derivation cannot add parameters")
    overlap = set(parameters) & set(manifest.spec.add_parameters)
    if overlap:
        raise L0CompileError("derived addParameters already exist: " + ", ".join(sorted(overlap)))
    parameters.update(manifest.spec.add_parameters)
    approval = manifest.spec.approval or spec.approval
    if not _approval_is_monotonic(spec.approval, approval):
        raise L0CompileError("derived approval may only increase risk and approval strength")
    verification = VerificationSpec.model_validate({
        **spec.verification.model_dump(by_alias=True, mode="python"),
        "predicates": [
            item.model_dump(by_alias=True, mode="python")
            for item in (*spec.verification.predicates, *manifest.spec.add_verification)
        ],
    })
    overwritten = {
        key for key, value in manifest.spec.desired_state_additions.items()
        if key in spec.intent.desired_state and spec.intent.desired_state[key] != value
    }
    if overwritten:
        raise L0CompileError(
            "derived desiredState additions cannot overwrite parent values: "
            + ", ".join(sorted(overwritten))
        )
    intent = IntentSpec.model_validate({
        **spec.intent.model_dump(by_alias=True, mode="python"),
        "kind": manifest.spec.intent_kind or spec.intent.kind,
        "desiredState": {
            **spec.intent.desired_state,
            **manifest.spec.desired_state_additions,
        },
    })
    compiled_spec = AtomicEffectSpec.model_validate({
        **spec.model_dump(by_alias=True, mode="python"),
        "parameters": {
            key: value.model_dump(by_alias=True, mode="python")
            for key, value in parameters.items()
        },
        "preflight": [
            item.model_dump(by_alias=True, mode="python")
            for item in (*spec.preflight, *manifest.spec.add_preflight)
        ],
        "verification": verification.model_dump(by_alias=True, mode="python"),
        "intent": intent.model_dump(by_alias=True, mode="python"),
        "approval": approval.model_dump(by_alias=True, mode="python"),
    })
    _validate_atomic_references(compiled_spec)
    lineage = (*parent.lineage, SkillRef(id=parent.metadata.id, version=parent.metadata.version))
    stable = _stable_atomic(manifest.metadata, manifest.mode, lineage, compiled_spec)
    return CompiledAtomicEffect.model_validate({
        **stable,
        "contractHash": sha256_json(stable),
    })


def _compile_composite(
    manifest: CompositeEffectManifest,
    resolved: dict[tuple[str, str], CompiledContract],
) -> CompiledCompositeEffect:
    steps: list[BoundCompositeStep] = []
    input_reference = re.compile(r"^\$\{\s*input\.([A-Za-z_][A-Za-z0-9_]*)\s*\}$")
    for step in manifest.spec.steps:
        child = resolved.get(step.skill_ref.key)
        if child is None:
            raise L0CompileError(
                f"composite {manifest.metadata.id} references unavailable {step.skill_ref.id}@{step.skill_ref.version}"
            )
        if not isinstance(child, CompiledAtomicEffect):
            raise L0CompileError("v2 composites may reference atomic/derived effects only; nested composites are rejected")
        unknown_arguments = set(step.arguments) - set(child.spec.parameters)
        if unknown_arguments:
            raise L0CompileError(
                f"composite step {step.id!r} has unknown child arguments: {sorted(unknown_arguments)}"
            )
        missing_arguments = {
            name for name, parameter in child.spec.parameters.items()
            if parameter.required and parameter.fixed is None and name not in step.arguments
        }
        if missing_arguments:
            raise L0CompileError(
                f"composite step {step.id!r} misses required child arguments: {sorted(missing_arguments)}"
            )
        for name, value in step.arguments.items():
            match = input_reference.fullmatch(value) if isinstance(value, str) else None
            referenced_inputs = []
            for reference in _references(value):
                path = reference.split()[0]
                if not path.startswith("input."):
                    raise L0CompileError(
                        f"composite step {step.id!r} uses unsupported reference {path!r}"
                    )
                input_name = path.split(".", 1)[1]
                if input_name not in manifest.spec.inputs:
                    raise L0CompileError(
                        f"composite step {step.id!r} references unknown input {input_name!r}"
                    )
                referenced_inputs.append(input_name)
            if match and not _input_is_compatible(
                manifest.spec.inputs[match.group(1)], child.spec.parameters[name],
            ):
                raise L0CompileError(
                    f"composite input {match.group(1)!r} cannot satisfy step {step.id!r} "
                    f"parameter {name!r}"
                )
            if not referenced_inputs and not _literal_satisfies(value, child.spec.parameters[name]):
                raise L0CompileError(
                    f"composite step {step.id!r} literal for {name!r} violates child contract"
                )
            if referenced_inputs and not match and child.spec.parameters[name].type != "string":
                raise L0CompileError(
                    f"composite step {step.id!r} interpolates input into non-string {name!r}"
                )
            fixed = child.spec.parameters[name].fixed
            if fixed is not None and value != fixed:
                raise L0CompileError(
                    f"composite step {step.id!r} cannot override fixed parameter {name!r}"
                )
        steps.append(BoundCompositeStep(
            id=step.id,
            skillRef=step.skill_ref,
            contractHash=child.contract_hash,
            capability=child.spec.effect.capability,
            compensationCapability=(
                child.spec.compensation.capability if child.spec.compensation else None
            ),
            dependsOn=step.depends_on,
            arguments=step.arguments,
            domain=step.domain,
        ))
    for checkpoint in manifest.spec.checkpoints:
        for observation in checkpoint.observations:
            for reference in _references((
                observation.arguments,
                *(item.expected for item in observation.predicates),
            )):
                path = reference.split()[0]
                if not path.startswith("input.") or path.split(".", 1)[1] not in manifest.spec.inputs:
                    raise L0CompileError(
                        f"checkpoint {checkpoint.id!r} uses unknown reference {path!r}"
                    )
    child_risk = max(RISK_RANK[resolved[step.skill_ref.key].spec.approval.risk] for step in manifest.spec.steps)  # type: ignore[union-attr]
    child_mode = max(APPROVAL_RANK[resolved[step.skill_ref.key].spec.approval.mode] for step in manifest.spec.steps)  # type: ignore[union-attr]
    if (
        RISK_RANK[manifest.spec.approval.risk] < child_risk
        or APPROVAL_RANK[manifest.spec.approval.mode] < child_mode
    ):
        raise L0CompileError("composite approval cannot be weaker than any child contract")
    stable = {
        "apiVersion": "netopyu.io/l0-effect-compiled/v2",
        "kind": "CompiledCompositeEffect",
        "metadata": manifest.metadata.model_dump(by_alias=True, mode="json"),
        "inputs": {
            key: value.model_dump(by_alias=True, mode="json")
            for key, value in sorted(manifest.spec.inputs.items())
        },
        "steps": [item.model_dump(by_alias=True, mode="json") for item in steps],
        "checkpoints": [
            item.model_dump(by_alias=True, mode="json") for item in manifest.spec.checkpoints
        ],
        "approval": manifest.spec.approval.model_dump(by_alias=True, mode="json"),
        "compensationOrder": manifest.spec.compensation_order,
    }
    return CompiledCompositeEffect.model_validate({
        **stable,
        "definitionHash": sha256_json(stable),
    })


def compile_documents(documents: Iterable[AuthoringManifest]) -> list[CompiledContract]:
    manifests: dict[tuple[str, str], AuthoringManifest] = {}
    for manifest in documents:
        key = (manifest.metadata.id, manifest.metadata.version)
        if key in manifests:
            raise L0CompileError(f"duplicate L0 contract {key[0]}@{key[1]}")
        manifests[key] = manifest
    resolved: dict[tuple[str, str], CompiledContract] = {}
    visiting: set[tuple[str, str]] = set()

    def resolve(key: tuple[str, str]) -> CompiledContract:
        if key in resolved:
            return resolved[key]
        if key in visiting:
            raise L0CompileError(f"cyclic L0 derivation at {key[0]}@{key[1]}")
        manifest = manifests.get(key)
        if manifest is None:
            raise L0CompileError(f"unknown L0 contract {key[0]}@{key[1]}")
        visiting.add(key)
        if isinstance(manifest, AtomicEffectManifest):
            compiled: CompiledContract = _compile_atomic(manifest)
        elif isinstance(manifest, DerivedEffectManifest):
            parent = resolve(manifest.extends.key)
            if not isinstance(parent, CompiledAtomicEffect):
                raise L0CompileError("derived effects may extend atomic/derived effects only")
            compiled = _compile_derived(manifest, parent)
        else:
            for step in manifest.spec.steps:
                resolve(step.skill_ref.key)
            compiled = _compile_composite(manifest, resolved)
        visiting.remove(key)
        resolved[key] = compiled
        return compiled

    for key in manifests:
        resolve(key)
    return [resolved[key] for key in sorted(resolved)]
