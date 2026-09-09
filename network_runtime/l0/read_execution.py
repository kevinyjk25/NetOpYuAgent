"""Host-bound read wiring for local experiments; not an activation/token API.

The host supplies a trusted binding and context, never values accepted from an LLM.
No implicit system identity, source-code loading or provider discovery occurs here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from network_runtime.access import ObservationAccessContext, ObservationPolicy
from network_runtime.capabilities import CapabilityContract, CapabilityKind, DataSensitivity, EffectSemantics
from network_runtime.contracts import sha256_json

from .models import CompiledAtomicRead
from .read_contracts import instantiate_read, validate_read_result_shape
from .structured_reads import (
    CompiledStructuredRead, clone_payload, instantiate_structured_read, read_schema,
    validate_structured_result, verify_read_contract,
)
from .structured_schema import DataBindingError, pointer_value


@dataclass(frozen=True)
class HostReadBinding:
    approved_contract_hash: str
    capability: CapabilityContract
    required_scopes: frozenset[str]
    observe: Callable[[dict[str, Any]], tuple[dict[str, Any], dict[str, Any]]]


def execute_host_read(
    contract: CompiledAtomicRead | CompiledStructuredRead, arguments: dict[str, Any],
    context: ObservationAccessContext, binding: HostReadBinding,
) -> dict[str, Any]:
    contract = verify_read_contract(contract)
    structured = isinstance(contract, CompiledStructuredRead)
    draft = instantiate_structured_read(contract, arguments) if structured else instantiate_read(contract, arguments)
    capability = binding.capability
    if (
        draft["contractHash"] != binding.approved_contract_hash
        or capability.kind != CapabilityKind.OBSERVATION
        or capability.capability_id != draft["capability"] or capability.tool_name != draft["tool"]
        or capability.input_schema_digest != sha256_json(read_schema(contract, "input"))
        or capability.output_schema_digest != sha256_json(read_schema(contract, "output"))
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
    policy_arguments = draft["arguments"]
    if structured:
        if (capability.effect_semantics != EffectSemantics.NONE or capability.action_type != "read_only"
                or set(capability.scope_fields) != set(contract.spec.resource_scopes)):
            raise ValueError("structured observation effect or resource scope contract mismatch")
        policy_arguments = {key: pointer_value(draft["arguments"], pointer)
                            for key, pointer in contract.spec.resource_scopes.items()}
        if any(type(value) not in {str, int} or (isinstance(value, str) and not value.strip())
               for value in policy_arguments.values()):
            raise PermissionError("resource scope requires a nonblank scalar identity")
    if any(name not in policy_arguments for name in capability.scope_fields):
        raise PermissionError("resource scope cannot be evaluated without its argument")
    decision = ObservationPolicy().authorize(capability, policy_arguments, context)
    if not decision.allowed:
        raise PermissionError(decision.code)
    # The provider receives a separate dictionary, not the request evidence object.
    try:
        result, evidence = binding.observe(clone_payload(draft["arguments"]))
    except DataBindingError:
        # Only our own validation errors may expose structured diagnostics.
        # A provider can reuse this exception class with confidential details.
        raise ValueError("provider callback failed") from None
    result, evidence = clone_payload(result), clone_payload(evidence)
    shape = validate_structured_result(contract, result) if structured else validate_read_result_shape(contract, result)
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
