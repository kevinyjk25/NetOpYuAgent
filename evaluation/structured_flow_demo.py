"""One synthetic, source-anchored flow wiring fixture using the shared executor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from evaluation.structured_binding_probe import write_artifacts
from evaluation.structured_flow_tree import StructuredFlowTree, compile_structured_tree
from evaluation.translation_intake import _bundle, _document
from network_runtime.access import ObservationAccessContext
from network_runtime.capabilities import CapabilityContract
from network_runtime.contracts import sha256_json
from network_runtime.l0.flow import HostFlowConsent, StructuredEffectTarget, parse_flow, run_read_flow
from network_runtime.l0.read_execution import HostReadBinding
from network_runtime.l0.structured_reads import StructuredReadManifest, compile_structured_read

SOURCE_PATH = "examples/translation-intake/structured-flow-source.md"
ROOT = Path(__file__).resolve().parents[1]


def object_schema(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def reference(source, pointer):
    return {"kind": "reference", "source": source, "pointer": pointer}


def object_expression(**fields):
    return {"kind": "object", "fields": fields}


def fixture():
    text = (ROOT / SOURCE_PATH).read_text()
    bundle = _bundle({"apiVersion": "effect-runtime.io/translation-intake/v1", "candidateId": "local-structured-flow",
                      "repository": "local/development-fixture", "commitSha": "0" * 40,
                      "snapshotDigest": sha256_json(text), "entryPath": SOURCE_PATH,
                      "documents": [_document(SOURCE_PATH, text.encode(), mode="100644", origin="local_development_fixture")],
                      "supplementAttempts": [], "parentBundleDigest": None,
                      "evidenceRole": "synthetic_mechanical_wiring_not_public_skill"})
    device = object_schema({"id": {"type": "string", "minLength": 1}})
    port = object_schema({"name": {"type": "string", "minLength": 1}})
    input_schema = object_schema({"device": device})
    specs = {
        "get_interfaces": (input_schema, object_schema({"interfaces": {"type": "array", "items": object_schema({
            "name": {"type": "string", "minLength": 1}, "adminUp": {"type": "boolean"}})}}), {"device_id": "/device/id"}),
        "get_interface_counters": (object_schema({"device": device, "interface": port}),
                                   object_schema({"counters": object_schema({"errors": {"type": "integer", "minimum": 0}})}),
                                   {"device_id": "/device/id", "interface_name": "/interface/name"}),
    }
    reads = {}
    access = {"requiredScopes": ["network:read"], "dataClassification": "internal"}
    for name, (inputs, outputs, scopes) in specs.items():
        tool = {"name": name, "inputSchema": inputs, "outputSchema": outputs, "annotations": {"readOnlyHint": True}}
        adapter = {"tool": name, "capability": "network." + name, "effect": "read_only", "resourceScopes": scopes, "access": access}
        sources = []
        for role, value in (("skill", text), ("tool", json.dumps(tool)), ("adapter", json.dumps(adapter))):
            sources.append({"role": role, "origin": "synthetic-fixture:" + role, "text": value,
                            "sha256": "sha256:" + hashlib.sha256(value.encode()).hexdigest()})
        manifest = StructuredReadManifest.model_validate({
            "apiVersion": "netopyu.io/l0-structured-read/v1", "kind": "StructuredRead",
            "metadata": {"id": "fixture." + name.replace("_", "-"), "version": "1.0.0", "owner": "local-fixture"},
            "spec": {"tool": name, "capability": "network." + name, "effect": "read_only", "inputSchema": inputs,
                     "outputSchema": outputs, "resourceScopes": scopes, "access": access, "sources": sources},
        })
        reads[name] = compile_structured_read(manifest)
    effects = {"enable": StructuredEffectTarget(profile="fixture", tool="set_interface_admin_state",
        skill_id="fixture.enable-interface", contract_hash="sha256:" + "e" * 64,
        input_schema=object_schema({"device": device, "interface": port, "enabled": {"type": "boolean"}}))}
    lines = text.splitlines()

    def span(prefix):
        quote = next(line for line in lines if line.startswith(prefix))
        start = text.index(quote)
        return {"path": SOURCE_PATH, "start": start, "end": start + len(quote), "quote": quote}

    def end(prefix):
        return {"kind": "end", "source": span(prefix), "outcome": "read_path_completed", "explanation": "Read-only fixture path complete; not business correctness proof."}

    dev = reference("input", "/device")
    interface = object_expression(name=reference("interfaces", "/interfaces/0/name"))
    tree = StructuredFlowTree.model_validate({
        "api_version": "netopyu.io/structured-flow-tree/v1", "source_digest": bundle["bundleDigest"],
        "purpose": "Local nested-data, branch and candidate-only wiring", "input_schema": input_schema, "max_read_age_seconds": 5,
        "steps": [
            {"kind": "read", "source": span("先按"), "tool": "get_interfaces", "bind": "interfaces", "arguments": object_expression(device=dev)},
            {"kind": "if_equal", "source": span("如果"), "left": reference("interfaces", "/interfaces/0/adminUp"), "equals": False,
             "when_equal": [
                 {"kind": "read", "source": span("如果"), "tool": "get_interface_counters", "bind": "counters",
                  "arguments": object_expression(device=dev, interface=interface)},
                 {"kind": "if_equal", "source": span("错误"), "left": reference("counters", "/counters/errors"), "equals": 0,
                  "when_equal": [end("错误")], "otherwise": [{"kind": "effect_candidate", "source": span("错误"), "binding_id": "enable",
                     "arguments": object_expression(device=dev, interface=interface, enabled={"kind": "literal", "value": True})}]},
             ], "otherwise": [end("如果")]},
        ],
    })
    return bundle, tree, reads, effects


def context():
    return ObservationAccessContext(subject_id="local-test-user", roles=frozenset({"netops"}),
                                   scopes=frozenset({"network:read", "device_id:lab-sw1", "interface_name:eth0"}),
                                   purpose="isolated structured wiring test")


def host_bindings(reads, calls, *, interfaces=None, errors=7):
    result = {}
    interfaces = [{"name": "eth0", "adminUp": False}] if interfaces is None else interfaces
    for name, contract in reads.items():
        def observe(arguments, name=name):
            calls.append({"tool": name, "arguments": arguments})
            payload = {"interfaces": interfaces} if name == "get_interfaces" else {"counters": {"errors": errors}}
            return payload, {"source": "inert_local_fixture", "deviceFreshnessAttested": False}
        capability = CapabilityContract.from_metadata(name, {
            "capability_id": contract.spec.capability, "action_type": "read_only", "domain": "network",
            "input_schema_digest": sha256_json(contract.spec.input_schema), "output_schema_digest": sha256_json(contract.spec.output_schema),
            "required_roles": ["netops"], "sensitivity": "internal", "scope_fields": list(contract.spec.resource_scopes),
            "freshness_limit_seconds": 10,
        }, source="local_fixture")
        result[contract.contract_hash] = HostReadBinding(contract.contract_hash, capability, frozenset({"network:read"}), observe)
    return result


def run_demo(output):
    if Path(output).exists():
        raise FileExistsError("output must not exist; preserve previous evidence")
    bundle, tree, reads, effects = fixture()
    compilation = compile_structured_tree(bundle, tree, reads, effects)
    flow = parse_flow(compilation["flow"])
    arguments, calls = {"device": {"id": "lab-sw1"}}, []
    result = run_read_flow(flow, arguments, reads={c.contract_hash: c for c in reads.values()}, effects=effects,
                           bindings=host_bindings(reads, calls), context=context(),
                           consent=HostFlowConsent(compilation["qualification"]["flowDigest"], sha256_json(arguments)))
    if result["status"] != "awaiting_effect_admission" or result["effectExecuted"] or len(calls) != 2:
        raise AssertionError("structured flow wiring failed")
    files = {"source-bundle.json": bundle, "tree.json": tree.model_dump(mode="json"), "compilation.json": compilation,
             "read-contracts.json": {k: v.model_dump(by_alias=True, mode="json") for k, v in reads.items()},
             "effect-targets.json": {k: v.model_dump(mode="json") for k, v in effects.items()},
             "arguments.json": arguments, "execution.json": result, "calls.json": {"reads": calls, "writes": []}}
    report = {"apiVersion": "netopyu.io/structured-flow-demo/v1", "evidenceRole": "synthetic_mechanical_wiring_not_public_skill",
              "status": result["status"], "readCalls": len(calls), "effectCalls": 0, "modelCalls": 0,
              "artifactDigests": {name: sha256_json(data) for name, data in sorted(files.items())},
              "runtimeAuthorityGranted": False, "contractActivated": False, "wholeSkillTranslationProven": False,
              "translationMetrics": None, "runtimeLatencyMetrics": None,
              "claimBoundary": "Shared executor wiring with explicit local host bindings and inert fixtures, not real-device or semantic generalization evidence."}
    report["reportDigest"] = sha256_json(report)
    write_artifacts(output, {**files, "report.json": report})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("output must not exist; preserve previous evidence")
    print(json.dumps(run_demo(args.output), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
