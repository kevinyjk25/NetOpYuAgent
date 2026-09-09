"""In-process, versioned Netdata-shaped host. No HTTP, credentials or source scripts.

This is a disclosed evaluation adapter, NOT a captured Netdata server or original
wrapper. One callback performs one info/query operation. It never decodes rows,
aggregates results, follows pagination or retries. Payloads are synthetic.
"""

from __future__ import annotations

import hashlib
import json

from network_runtime.capabilities import CapabilityContract
from network_runtime.contracts import sha256_json
from network_runtime.l0.read_execution import HostReadBinding
from network_runtime.l0.structured_reads import StructuredReadManifest, compile_structured_read
from network_runtime.l0.structured_schema import DataBindingError, snapshot_json, validate_data

HOST_VERSION = "netopyu-isolated-netdata/1"
TOOL = "fixture_netdata_function"
FUNCTION = "snmp:traps"
SEVERITIES = ["emerg", "alert", "crit", "err", "warning", "notice", "info", "debug"]
QUERY_KEYS = {"after", "before", "last", "direction", "selections"}


def obj(properties, *, required=None, extra=False):
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else required, "additionalProperties": extra}


def array(items, limit=256):
    return {"type": "array", "items": items, "maxItems": limit}


def host_catalog():
    text = {"type": "string", "minLength": 1, "maxLength": 128}
    selections = obj({k: {**array(text, 1), "minItems": 1} for k in
                      ("__logs_sources", "TRAP_REPORT_TYPE", "TRAP_CATEGORY", "TRAP_SOURCE_IP")})
    body = obj({"info": {"type": "boolean", "const": True},
                "after": {"type": "integer", "minimum": -86400, "maximum": -1},
                "before": {"type": "integer", "const": 0}, "last": {"type": "integer", "minimum": 1, "maximum": 200},
                "direction": {"type": "string", "const": "backward"}, "selections": selections}, required=[])
    metadata = obj({"index": {"type": "integer", "minimum": 0, "maximum": 127}}, extra=True)
    # Keep original field names and optional/error/discovery envelopes. Shape is
    # deliberately not proof of the conditional domain rules checked separately.
    output = obj({"status": {"type": "integer"}, "v": {"type": "integer"}, "type": text,
                  "has_history": {"type": "boolean"}, "accepted_params": array(text, 64),
                  "required_params": array(obj({"id": text, "name": text, "type": text,
                    "options": array(obj({"id": text, "name": text}, extra=True), 64)}, extra=True), 32),
                  "columns": {"type": "object", "additionalProperties": metadata, "maxProperties": 128},
                  "data": array(array({"type": ["string", "integer", "number", "boolean", "null", "object"]}, 128)),
                  "partial": {"type": "boolean"}, "pagination": obj({}, required=[], extra=True)},
                 required=["status", "v", "type"], extra=True)
    return {"hostVersion": HOST_VERSION, "origin": "synthetic_in_process_adapter_not_netdata_capture",
            "tools": [{"name": TOOL, "description": "One isolated snmp:traps Function call; no network transport or aggregation.",
                       "inputSchema": obj({"node": text, "function": {**text, "const": FUNCTION}, "body": body}),
                       "outputSchema": output, "annotations": {"readOnlyHint": True}}]}


def validate_info(info, listener):
    if (info.get("status") != 200 or type(info.get("v")) is not int or info.get("v") != 3
            or info.get("type") != "logs" or info.get("has_history") is not True or "data" in info):
        raise DataBindingError("netdata_info_envelope", "", "expected this isolated host's v3 discovery envelope")
    params, widgets = info.get("accepted_params"), info.get("required_params")
    if (not isinstance(params, list) or any(not isinstance(p, str) for p in params)
            or len(params) != len(set(params)) or not QUERY_KEYS <= set(params)):
        raise DataBindingError("netdata_parameter_support", "/accepted_params", "required query keys must be advertised without ambiguity")
    if (not isinstance(widgets, list) or len(widgets) != 1 or not isinstance(widgets[0], dict)
            or widgets[0].get("id") != "__logs_sources" or widgets[0].get("type") != "multiselect"):
        raise DataBindingError("netdata_source_widget", "/required_params", "unknown required widgets need explicit review; never ignore them")
    options = widgets[0].get("options")
    if not isinstance(options, list) or not options or any(not isinstance(o, dict) or not isinstance(o.get("id"), str) for o in options):
        raise DataBindingError("netdata_source_options", "/required_params/0/options", "source options required")
    ids = [o["id"] for o in options]
    if len(ids) != len(set(ids)) or listener not in ids:
        raise DataBindingError("netdata_source_unavailable", "/required_params/0/options", "requested listener absent or options ambiguous")


def query_arguments(info, *, node, listener, device_ip, window_seconds=86400, last=200):
    """A disclosed task-specific proposal builder, not an LLM translation result."""
    validate_info(info, listener)
    arguments = {"node": node, "function": FUNCTION, "body": {
        "after": -window_seconds if type(window_seconds) is int else window_seconds,
        "before": 0, "last": last, "direction": "backward", "selections": {
            "__logs_sources": [listener], "TRAP_REPORT_TYPE": ["trap"],
            "TRAP_CATEGORY": ["security"], "TRAP_SOURCE_IP": [device_ip]}}}
    return validate_data(host_catalog()["tools"][0]["inputSchema"], arguments)


class IsolatedNetdataHost:
    def __init__(self, *, node, listener, device_ip, now_seconds=1788912000, info=None, response=None):
        self.node, self.listener, self.device_ip, self.now_seconds = node, listener, device_ip, now_seconds
        self.calls = []
        self.catalog = host_catalog()
        self._info = snapshot_json(info) if info is not None else {
            "status": 200, "v": 3, "type": "logs", "has_history": True,
            "accepted_params": sorted(QUERY_KEYS | {"info"}), "required_params": [{
                "id": "__logs_sources", "name": "Log sources", "type": "multiselect",
                "options": [{"id": listener, "name": "Isolated fixture listener"}]}]}
        # Not a transcription of the inconsistent upstream example. Deliberately
        # non-display order; tests permute metadata and cells together.
        names = ["MESSAGE", "TRAP_SOURCE_IP", "TRAP_SEVERITY", "timestamp", "TRAP_JOB", "TRAP_CATEGORY", "TRAP_REPORT_TYPE"]
        self._response = snapshot_json(response) if response is not None else {
            "status": 200, "v": 3, "type": "logs", "partial": False,
            "columns": {name: {"index": i} for i, name in enumerate(names)},
            "data": [["SYNTHETIC-PRIVATE-MESSAGE", device_ip, severity, (now_seconds - 60 - i) * 1000000,
                      listener, "security", "trap"] for i, severity in enumerate(("warning", "crit", "warning"))],
            "pagination": {"direction": "backward", "last": 200}}

    def observe(self, arguments):
        arguments = validate_data(self.catalog["tools"][0]["inputSchema"], arguments)
        if arguments["node"] != self.node:
            raise PermissionError("isolated node is outside host scope")
        body = arguments["body"]
        if body == {"info": True}:
            phase, response = "info", self._info
        else:
            if set(body) != QUERY_KEYS:
                raise ValueError("query/discovery cannot mix; unsupported parameters are not ignored")
            expected = query_arguments(self._info, node=self.node, listener=self.listener, device_ip=self.device_ip,
                                       window_seconds=-body["after"], last=body["last"])
            if expected != arguments:
                raise PermissionError("query differs from isolated host scope")
            phase, response = "query", self._response
        self.calls.append({"operation": phase, "requestDigest": sha256_json(arguments)})
        return snapshot_json(response), {"source": HOST_VERSION, "synthetic": True, "networkCalls": 0,
                                         "clockSeconds": self.now_seconds, "deviceFreshnessAttested": False}

    def contract_and_binding(self, source_text):
        tool = self.catalog["tools"][0]
        access = {"requiredScopes": ["netdata:logs:read"], "dataClassification": "confidential"}
        scopes = {"node_id": "/node", "function_id": "/function"}
        adapter = {"tool": TOOL, "capability": "fixture.netdata.function", "effect": "read_only",
                   "resourceScopes": scopes, "access": access, "hostVersion": HOST_VERSION,
                   "restriction": "in-process synthetic host; exact node, listener and source IP; no credentials/HTTP"}
        sources = [{"role": role, "origin": "isolated-netdata:" + role, "text": text,
                    "sha256": "sha256:" + hashlib.sha256(text.encode()).hexdigest()}
                   for role, text in (("skill", source_text), ("tool", json.dumps(tool)), ("adapter", json.dumps(adapter)))]
        manifest = StructuredReadManifest.model_validate({"apiVersion": "netopyu.io/l0-structured-read/v1", "kind": "StructuredRead",
            "metadata": {"id": "fixture.netdata-function", "version": "1.0.0", "owner": "local-evaluation"},
            "spec": {**{k: adapter[k] for k in ("tool", "capability", "effect", "resourceScopes", "access")},
                     "inputSchema": tool["inputSchema"], "outputSchema": tool["outputSchema"], "sources": sources}})
        contract = compile_structured_read(manifest)
        capability = CapabilityContract.from_metadata(TOOL, {
            "capability_id": contract.spec.capability, "action_type": "read_only", "domain": "network",
            "input_schema_digest": sha256_json(tool["inputSchema"]), "output_schema_digest": sha256_json(tool["outputSchema"]),
            "required_roles": ["netops"], "sensitivity": "confidential", "scope_fields": list(scopes),
            "freshness_limit_seconds": 10}, source=HOST_VERSION)
        return contract, HostReadBinding(contract.contract_hash, capability, frozenset(access["requiredScopes"]), self.observe)
