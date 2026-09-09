"""Explicit, digest-bound host correspondences; declarations, never authority.

Source argument names are reviewed claims, not inferred equivalences. This
read-only authoring profile never executes mappings or grants activation.
"""

from __future__ import annotations

from jsonschema import Draft202012Validator

from evaluation.structured_authoring import _obj, parse_read_contract, sha256_json
from evaluation.structured_flow_tree import SourceSpan
from network_runtime.l0.structured_schema import snapshot_json


def binding_packet_declarations(value):
    if (not isinstance(value, dict) or set(value) != {"apiVersion", "bindings"}
            or value["apiVersion"] != "netopyu.io/source-host-bindings/v1" or not isinstance(value["bindings"], list)):
        raise ValueError("versioned source-host binding packet required")
    return value["bindings"]


def validate_bindings(packet, bindings):
    bindings = snapshot_json(bindings)
    text = {"type": "string", "minLength": 1, "maxLength": 600}
    schema = {"type": "array", "maxItems": 16, "items": _obj({
        "id": text, "sourceBundleDigest": text, "source": SourceSpan.model_json_schema(),
        "sourceOperation": text, "hostTool": text, "contractHash": text,
        "inputSchemaDigest": text, "outputSchemaDigest": text,
        "parameterMap": {"type": "array", "minItems": 1, "maxItems": 32,
                         "items": _obj({"sourceArgument": text, "hostParameter": text})},
        "scope": text, "limitations": {"type": "array", "minItems": 1, "maxItems": 16, "items": text},
        "reviewKind": {"const": "developer_reviewed_adapter_declaration_not_independent_gold"}})}
    if next(Draft202012Validator(schema).iter_errors(bindings), None):
        raise ValueError("invalid explicit source-host binding declaration")
    if len({b["id"] for b in bindings}) != len(bindings):
        raise ValueError("duplicate source-host binding id")
    documents = {d["path"]: d for d in packet["bundle"]["documents"]}
    tools = {t["name"]: t for t in packet["catalog"]["tools"]}
    for binding in bindings:
        span = SourceSpan.model_validate(binding["source"])
        doc = documents.get(span.path)
        if (binding["sourceBundleDigest"] != packet["bundle"]["bundleDigest"] or doc is None
                or doc["representation"] != "inert_utf8_text"
                or doc["content"][span.start:span.end] != span.quote
                or span.end - span.start != len(span.quote)
                or binding["sourceOperation"] not in span.quote):
            raise ValueError("source-host binding source drift or absent operation")
        tool = tools.get(binding["hostTool"])
        if tool is None:
            raise ValueError("source-host binding targets an undeclared host tool")
        contract = parse_read_contract(packet["reads"][binding["hostTool"]])
        if (binding["contractHash"] != contract.contract_hash
                or binding["inputSchemaDigest"] != sha256_json(tool["inputSchema"])
                or binding["outputSchemaDigest"] != sha256_json(tool["outputSchema"])):
            raise ValueError("source-host binding host contract/schema drift")
        pairs = binding["parameterMap"]
        if (len({p["sourceArgument"] for p in pairs}) != len(pairs)
                or len({p["hostParameter"] for p in pairs}) != len(pairs)
                or any(p["sourceArgument"] not in span.quote
                       or p["hostParameter"] not in tool["inputSchema"].get("properties", {}) for p in pairs)):
            raise ValueError("ambiguous or ungrounded source-host parameter mapping")
    return bindings


def binding_view(bindings):
    return [{**b, "parameterMapRole": "declared_name_correspondence_not_argument_values_or_transform",
             "semanticEquivalenceProven": False, "runtimeAuthorityGranted": False} for b in bindings]


def authoring_boundary(packet):
    """Expose existing read gates, not a new permission/deferred-duty mechanism."""
    reads = [parse_read_contract(c) for c in packet["reads"].values()]
    return {"operation": "offline_inactive_candidate_authoring", "liveCredentialsRequiredForAuthoring": False,
            "providerExecution": False, "executionPermissionSatisfied": False,
            "runtimeGates": ["exact_host_contract_binding", "explicit_authenticated_identity", "least_privilege_scopes",
                             "resource_observation_policy"],
            "readPrerequisites": [{"tool": c.spec.tool, "contractHash": c.contract_hash,
                                   "access": c.spec.access.model_dump(by_alias=True, mode="json"),
                                   "resourceScopes": getattr(c.spec, "resource_scopes", None)} for c in reads],
            "sourceSpecificPreconditionsAutomaticallyCovered": False}
