"""Inactive, source-bound structured read contracts for the existing read gateway."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import Field, model_validator

from network_runtime.contracts import sha256_json

from .models import CompiledAtomicRead, Metadata, ReadAccessDeclaration, ReadSource, StrictModel
from .read_contracts import _source_object, _verified_contract
from .structured_schema import checked_schema, schema_location, schema_types, snapshot_json, validate_data


class StructuredReadSpec(StrictModel):
    capability: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    effect: Literal["read_only"]
    input_schema: dict[str, Any] = Field(alias="inputSchema")
    output_schema: dict[str, Any] = Field(alias="outputSchema")
    resource_scopes: dict[str, str] = Field(alias="resourceScopes")
    access: ReadAccessDeclaration
    sources: tuple[ReadSource, ...]

    @model_validator(mode="after")
    def validate_spec(self):
        inputs, outputs = checked_schema(self.input_schema), checked_schema(self.output_schema)
        if schema_types(schema_location(inputs, "")[0]) != {"object"} or schema_types(schema_location(outputs, "")[0]) != {"object"}:
            raise ValueError("structured tool inputs and outputs require object roots")
        if sorted(s.role for s in self.sources) != ["adapter", "skill", "tool"]:
            raise ValueError("structured read requires exactly three declared source roles")
        if len(self.resource_scopes) > 32 or any(not key.strip() for key in self.resource_scopes):
            raise ValueError("resource scope mapping must be bounded and named")
        for pointer in self.resource_scopes.values():
            node, required = schema_location(inputs, pointer)
            if not required or not schema_types(node) <= {"string", "integer"}:
                raise ValueError("resource scope must bind a required scalar identity")
        return self


class StructuredReadManifest(StrictModel):
    api_version: Literal["netopyu.io/l0-structured-read/v1"] = Field(alias="apiVersion")
    kind: Literal["StructuredRead"]
    metadata: Metadata
    spec: StructuredReadSpec


class CompiledStructuredRead(StrictModel):
    api_version: Literal["netopyu.io/l0-structured-read-compiled/v1"] = Field(alias="apiVersion")
    kind: Literal["CompiledStructuredRead"]
    metadata: Metadata
    spec: StructuredReadSpec
    runtime_authority_granted: Literal[False] = Field(default=False, alias="runtimeAuthorityGranted")
    semantic_alignment_proven: Literal[False] = Field(default=False, alias="semanticAlignmentProven")
    contract_hash: str = Field(alias="contractHash", pattern=r"^sha256:[0-9a-f]{64}$")


def compile_structured_read(manifest: StructuredReadManifest) -> CompiledStructuredRead:
    manifest = StructuredReadManifest.model_validate(manifest.model_dump(by_alias=True))
    spec = manifest.spec
    sources = {s.role: s for s in spec.sources}
    tool, adapter = (_source_object(sources[key].text) for key in ("tool", "adapter"))
    if tool.get("name") != spec.tool:
        raise ValueError("structured tool name differs from source")
    annotations = tool.get("annotations", {})
    if not isinstance(annotations, dict) or ("readOnlyHint" in annotations and annotations["readOnlyHint"] is not True):
        raise ValueError("tool contradicts declared read-only effect")
    for key, schema in (("inputSchema", spec.input_schema), ("outputSchema", spec.output_schema)):
        if checked_schema(tool.get(key)) != schema:
            raise ValueError(f"structured {key} differs from original declaration")
    if (adapter.get("capability") != spec.capability or adapter.get("tool") != spec.tool
            or adapter.get("effect") != "read_only" or adapter.get("resourceScopes") != spec.resource_scopes
            or ReadAccessDeclaration.model_validate(adapter.get("access")) != spec.access):
        raise ValueError("structured adapter, resource scopes or access declaration mismatch")
    body = {"apiVersion": "netopyu.io/l0-structured-read-compiled/v1", "kind": "CompiledStructuredRead",
            "metadata": manifest.metadata.model_dump(by_alias=True, mode="json"),
            "spec": spec.model_dump(by_alias=True, mode="json"),
            "runtimeAuthorityGranted": False, "semanticAlignmentProven": False}
    return CompiledStructuredRead.model_validate({**body, "contractHash": sha256_json(body)})


def parse_read_contract(value: dict) -> CompiledAtomicRead | CompiledStructuredRead:
    cls = CompiledStructuredRead if value.get("kind") == "CompiledStructuredRead" else CompiledAtomicRead
    return cls.model_validate(value)


def verify_read_contract(contract: CompiledAtomicRead | CompiledStructuredRead):
    if isinstance(contract, CompiledAtomicRead):
        return _verified_contract(contract)
    contract = CompiledStructuredRead.model_validate(contract.model_dump(by_alias=True))
    rebuilt = compile_structured_read(StructuredReadManifest(
        apiVersion="netopyu.io/l0-structured-read/v1", kind="StructuredRead",
        metadata=contract.metadata, spec=contract.spec,
    ))
    if rebuilt.contract_hash != contract.contract_hash:
        raise ValueError("structured contract hash mismatch")
    return rebuilt


def read_schema(contract, role: str) -> dict:
    value = getattr(contract.spec, role + "_schema")
    return snapshot_json(value) if isinstance(value, dict) else value.model_dump(by_alias=True, mode="json")


def instantiate_structured_read(contract: CompiledStructuredRead, arguments: dict) -> dict:
    contract = verify_read_contract(contract)
    values = validate_data(contract.spec.input_schema, arguments)
    body = {"kind": "ReadRequestDraft", "contractHash": contract.contract_hash,
            "capability": contract.spec.capability, "tool": contract.spec.tool, "arguments": values,
            "accessDeclaration": contract.spec.access.model_dump(by_alias=True, mode="json"),
            "resourceScopes": dict(contract.spec.resource_scopes),
            "status": "awaiting_semantic_review_and_authorization", "runtimeAuthorityGranted": False}
    return {**body, "requestDigest": sha256_json(body)}


def validate_structured_result(contract: CompiledStructuredRead, result: dict) -> dict:
    contract = verify_read_contract(contract)
    values = validate_data(contract.spec.output_schema, result)
    return {"contractHash": contract.contract_hash, "resultDigest": sha256_json(values),
            "shapeValid": True, "businessCorrectnessProven": False,
            "sourceAuthenticityVerified": False, "runtimeAuthorityGranted": False}


def clone_payload(value):
    """Copy JSON at callback boundaries, preserving legacy scalar representations."""
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
