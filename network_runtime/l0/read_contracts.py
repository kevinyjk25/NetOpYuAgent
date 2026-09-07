"""Inactive read-contract compilation and request drafts; no tool execution.

Source declarations are mechanically cross-checked, not independently certified.
Result shape checking is not verification of business correctness or provenance.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from network_runtime.contracts import sha256_json

from .models import (
    COMPILED_API_VERSION, AtomicReadManifest, CompiledAtomicRead,
    ReadAccessDeclaration, ReadObjectSchema, value_matches_type,
)


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate read source JSON key")
        result[key] = value
    return result


def _source_object(text: str) -> dict[str, Any]:
    value = json.loads(text, object_pairs_hook=_unique_pairs)
    json.dumps(value, allow_nan=False)
    if not isinstance(value, dict):
        raise ValueError("read tool/adapter source must be a JSON object")
    return value


def compile_read(manifest: AtomicReadManifest) -> CompiledAtomicRead:
    # Frozen Pydantic models still contain mutable dicts; revalidate at boundaries.
    manifest = AtomicReadManifest.model_validate(manifest.model_dump(by_alias=True))
    spec = manifest.spec
    sources = {source.role: source for source in spec.sources}
    tool = _source_object(sources["tool"].text)
    adapter = _source_object(sources["adapter"].text)
    if tool.get("name") != spec.tool:
        raise ValueError("read tool name differs from source")
    annotations = tool.get("annotations", {})
    if not isinstance(annotations, dict) or (
        "readOnlyHint" in annotations and annotations["readOnlyHint"] is not True
    ):
        raise ValueError("tool source contradicts read-only declaration")
    for key, expected in (("inputSchema", spec.input_schema), ("outputSchema", spec.output_schema)):
        if ReadObjectSchema.model_validate(tool.get(key)) != expected:
            raise ValueError(f"read {key} differs from source")
    if (
        adapter.get("capability") != spec.capability or adapter.get("tool") != spec.tool
        or adapter.get("effect") != "read_only"
    ):
        raise ValueError("read adapter mapping/effect declaration mismatch")
    if ReadAccessDeclaration.model_validate(adapter.get("access")) != spec.access:
        raise ValueError("read access declaration differs from adapter source")
    stable = {
        "apiVersion": COMPILED_API_VERSION, "kind": "CompiledAtomicRead",
        "metadata": manifest.metadata.model_dump(by_alias=True, mode="json"),
        "spec": spec.model_dump(by_alias=True, mode="json"),
        "runtimeAuthorityGranted": False, "semanticAlignmentProven": False,
    }
    return CompiledAtomicRead.model_validate({**stable, "contractHash": sha256_json(stable)})


def _verified_contract(contract: CompiledAtomicRead) -> CompiledAtomicRead:
    contract = CompiledAtomicRead.model_validate(contract.model_dump(by_alias=True))
    rebuilt = compile_read(AtomicReadManifest(
        apiVersion="netopyu.io/l0-effect/v2", kind="AtomicRead",
        metadata=contract.metadata, spec=contract.spec,
    ))
    if rebuilt.contract_hash != contract.contract_hash:
        raise ValueError("read contract hash mismatch")
    return rebuilt


def _validate_values(values: dict[str, Any], schema: ReadObjectSchema, *, inputs: bool) -> dict[str, Any]:
    if not isinstance(values, dict) or any(not isinstance(key, str) for key in values):
        raise ValueError("read values require an object with string keys")
    if set(values) - set(schema.properties):
        raise ValueError("unknown read fields")
    if set(schema.required) - set(values):
        raise ValueError("missing required read fields")
    for name, value in values.items():
        if not value_matches_type(value, schema.properties[name].type):
            raise ValueError(f"read field type mismatch: {name}")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"non-finite read value: {name}")
        if inputs and isinstance(value, str) and re.search(r"\$[({]|<[^<>]+>|\{\{", value):
            raise ValueError(f"unresolved read input: {name}")
    # Snapshot values; callers cannot mutate the returned draft via their inputs.
    return json.loads(json.dumps(values, ensure_ascii=False, allow_nan=False))


def instantiate_read(contract: CompiledAtomicRead, arguments: dict[str, Any]) -> dict[str, Any]:
    """Create a typed request proposal, not an authorized Runtime invocation."""
    contract = _verified_contract(contract)
    values = _validate_values(arguments, contract.spec.input_schema, inputs=True)
    body = {
        "kind": "ReadRequestDraft", "contractHash": contract.contract_hash,
        "capability": contract.spec.capability, "tool": contract.spec.tool,
        "arguments": values,
        "accessDeclaration": contract.spec.access.model_dump(by_alias=True, mode="json"),
        "status": "awaiting_semantic_review_and_authorization",
        "runtimeAuthorityGranted": False,
    }
    return {**body, "requestDigest": sha256_json(body)}


def validate_read_result_shape(contract: CompiledAtomicRead, result: dict[str, Any]) -> dict[str, Any]:
    """Structural offline check only; no claim about authenticity/freshness/truth."""
    contract = _verified_contract(contract)
    values = _validate_values(result, contract.spec.output_schema, inputs=False)
    return {
        "contractHash": contract.contract_hash, "resultDigest": sha256_json(values),
        "shapeValid": True, "businessCorrectnessProven": False,
        "sourceAuthenticityVerified": False, "runtimeAuthorityGranted": False,
    }
