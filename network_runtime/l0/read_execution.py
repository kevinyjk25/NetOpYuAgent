"""Host-bound read wiring for local experiments; not an activation/token API.

The host supplies a trusted binding and context, never values accepted from an LLM.
No implicit system identity, source-code loading or provider discovery occurs here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from network_runtime.access import ObservationAccessContext, ObservationPolicy
from network_runtime.capabilities import CapabilityContract, CapabilityKind, DataSensitivity
from network_runtime.contracts import sha256_json

from .models import CompiledAtomicRead
from .read_contracts import instantiate_read, validate_read_result_shape


@dataclass(frozen=True)
class HostReadBinding:
    approved_contract_hash: str
    capability: CapabilityContract
    required_scopes: frozenset[str]
    observe: Callable[[dict[str, Any]], tuple[dict[str, Any], dict[str, Any]]]


def execute_host_read(
    contract: CompiledAtomicRead, arguments: dict[str, Any],
    context: ObservationAccessContext, binding: HostReadBinding,
) -> dict[str, Any]:
    contract = CompiledAtomicRead.model_validate(contract.model_dump(by_alias=True))
    draft = instantiate_read(contract, arguments)
    capability = binding.capability
    if (
        draft["contractHash"] != binding.approved_contract_hash
        or capability.kind != CapabilityKind.OBSERVATION
        or capability.capability_id != draft["capability"] or capability.tool_name != draft["tool"]
        or capability.input_schema_digest != sha256_json(contract.spec.input_schema.model_dump(by_alias=True, mode="json"))
        or capability.output_schema_digest != sha256_json(contract.spec.output_schema.model_dump(by_alias=True, mode="json"))
        or capability.sensitivity != DataSensitivity(contract.spec.access.data_classification)
        or binding.required_scopes != frozenset(contract.spec.access.required_scopes)
    ):
        raise ValueError("host read binding differs from the exact reviewed contract")
    if (
        not isinstance(context, ObservationAccessContext) or context.implicit_local_context
        or context.authenticated is not True
    ):
        raise PermissionError("explicit host observation identity is required")
    # Unlike compatibility paths, do not allow wildcard/system shortcuts here.
    if "system" in context.roles or "*" in context.scopes:
        raise PermissionError("local read experiment requires explicit least-privilege scopes")
    if not binding.required_scopes <= context.scopes:
        raise PermissionError("required read capability scope missing")
    if any(name not in draft["arguments"] for name in capability.scope_fields):
        raise PermissionError("resource scope cannot be evaluated without its argument")
    decision = ObservationPolicy().authorize(capability, draft["arguments"], context)
    if not decision.allowed:
        raise PermissionError(decision.code)
    # The provider receives a separate dictionary, not the request evidence object.
    result, evidence = binding.observe(dict(draft["arguments"]))
    shape = validate_read_result_shape(contract, result)
    body = {
        "status": "local_read_completed_shape_valid",
        "contractHash": draft["contractHash"], "requestDigest": draft["requestDigest"],
        "subjectId": context.subject_id, "accessDecision": decision.code,
        "payload": result, "providerEvidence": evidence, "shapeCheck": shape,
        "executionAuthority": "explicit_host_binding_local_experiment_only",
        "contractActivated": False, "businessCorrectnessProven": False,
        "productionIdentityVerified": False,
    }
    return {**body, "receiptDigest": sha256_json(body)}
