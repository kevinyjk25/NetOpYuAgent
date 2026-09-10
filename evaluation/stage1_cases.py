"""Known synthetic development sources and primitive host contracts, no L0 input.

Separate execution expectations are for developer evaluation only. The model
gets the original six-field source/task/catalog packet, not those expectations.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evaluation.structured_authoring import validate_inputs
from evaluation.structured_flow_demo import fixture, object_schema
from evaluation.translation_intake import _bundle, _document
from network_runtime.contracts import sha256_json
from network_runtime.l0.structured_reads import StructuredReadManifest, compile_structured_read

ROOT = Path(__file__).resolve().parents[1]


def packet(case):
    if case == "wiring":
        bundle, tree, reads, _ = fixture()
        return validate_inputs({"bundle": bundle,
            "task": "Author an inactive read-flow candidate from the supplied original structured wiring source. The future flow must use the caller's input.device, preserve source conditions and access requirements, and hand unrepresented work to L1. Do not execute anything.",
            "taskOrigin": "developer_authored_evaluation_request", "inputSchema": tree.input_schema,
            "catalog": {"tools": [json.loads(next(s.text for s in c.spec.sources if s.role == "tool")) for c in reads.values()]},
            "reads": {k: c.model_dump(by_alias=True, mode="json") for k, c in reads.items()}})
    if case not in {"approval", "reference"}:
        raise ValueError("unknown stage-1 development case")
    directory = ROOT / "evaluation/fixtures/stage1" / case
    entry = str((directory / "SKILL.md").relative_to(ROOT))
    documents = [_document(str(p.relative_to(ROOT)), p.read_bytes(), mode="100644", origin="known_developer_authored_fixture")
                 for p in sorted(directory.rglob("*.md"))]
    bundle = _bundle({"apiVersion": "effect-runtime.io/translation-intake/v1", "candidateId": "stage1-" + case,
        "repository": "local/stage1-development-" + case, "commitSha": "0" * 40,
        "snapshotDigest": sha256_json(documents), "entryPath": entry, "documents": documents,
        "supplementAttempts": [], "parentBundleDigest": None,
        "evidenceRole": "known_development_synthetic_not_public_skill"})
    identifier = {"type": "string", "minLength": 1}
    device = object_schema({"id": identifier})
    if case == "approval":
        inputs = object_schema({"changeId": identifier})
        specs = {"get_change_request": (inputs, object_schema({"approved": {"type": "boolean"}, "device": device}),
                                         {"change_id": "/changeId"}),
                 "get_device_health": (object_schema({"deviceId": identifier}),
                     object_schema({"health": object_schema({"available": {"type": "boolean"}})}), {"device_id": "/deviceId"})}
    else:
        inputs = object_schema({"serviceId": identifier})
        specs = {"lookup_service": (inputs, object_schema({"ready": {"type": "boolean"}, "device": device}), {"service_id": "/serviceId"}),
            "get_device_health": (object_schema({"deviceId": identifier}),
                object_schema({"health": object_schema({"available": {"type": "boolean"}, "alarmId": identifier})}), {"device_id": "/deviceId"}),
            "get_alarm": (object_schema({"alarmId": identifier}), object_schema({"code": identifier, "details": {"type": "string"}}),
                          {"alarm_id": "/alarmId"})}
    access = {"requiredScopes": ["network:read"], "dataClassification": "internal"}
    reads, tools = {}, []
    source_text = "\n\n".join(d["content"] for d in documents)
    for name, (args, result, scopes) in specs.items():
        tool = {"name": name, "inputSchema": args, "outputSchema": result, "annotations": {"readOnlyHint": True}}
        tools.append(tool)
        adapter = {"tool": name, "capability": "network." + name, "effect": "read_only", "resourceScopes": scopes, "access": access}
        sources = [{"role": role, "origin": "synthetic-stage1:" + role, "text": text,
                    "sha256": "sha256:" + hashlib.sha256(text.encode()).hexdigest()}
                   for role, text in (("skill", source_text), ("tool", json.dumps(tool)), ("adapter", json.dumps(adapter)))]
        spec = StructuredReadManifest.model_validate({"apiVersion": "netopyu.io/l0-structured-read/v1", "kind": "StructuredRead",
            "metadata": {"id": "stage1." + name.replace("_", "-"), "version": "1.0.0", "owner": "local-fixture"},
            "spec": {"tool": name, "capability": "network." + name, "effect": "read_only", "inputSchema": args,
                     "outputSchema": result, "resourceScopes": scopes, "access": access, "sources": sources}})
        reads[name] = compile_structured_read(spec).model_dump(by_alias=True, mode="json")
    return validate_inputs({"bundle": bundle, "task": "Translate the source's read-only procedure into an inactive candidate for execution-time caller input. Preserve all conditional decisions, dynamic identifiers, output restrictions and host access requirements. Do not execute anything.",
        "taskOrigin": "developer_authored_evaluation_request", "inputSchema": inputs, "catalog": {"tools": tools}, "reads": reads})
