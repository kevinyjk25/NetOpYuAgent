"""Pure synthetic read-fixture input construction, without research orchestration.

This declares existing read contracts only: no model calls, semantic reviewer,
Gold, executable source scripts, or additional provider authority. Provenance
labels and serialization are retained for compatibility with earlier packets.
"""
from __future__ import annotations

import hashlib
import json

from network_runtime.l0.structured_reads import StructuredReadManifest, compile_structured_read
from skill_authoring import compiler as author


def packet_for(bundle, specification):
    """Declare primitive adapter contracts, not a graph or expected response."""
    text = next(d["content"] for d in bundle["documents"] if d["path"] == bundle["entryPath"])
    reads = {}
    for index, tool in enumerate(specification["tools"]):
        name = tool["name"]
        if name in reads or tool.get("annotations", {}).get("readOnlyHint") is not True:
            raise ValueError("unique explicitly read-only local tools required")
        adapter = {"tool": name, "capability": f"semantic-transfer.read{index}", "effect": "read_only",
                   "resourceScopes": {}, "access": {"requiredScopes": ["stage2:read"], "dataClassification": "internal"},
                   "limitations": "Disclosed synthetic host; finite resource ACL separately enforced before provider invocation."}
        sources = [{"role": role, "origin": "local-transfer-declaration:" + role, "text": source,
                    "sha256": "sha256:" + hashlib.sha256(source.encode()).hexdigest()}
                   for role, source in (("skill", text), ("tool", json.dumps(tool)), ("adapter", json.dumps(adapter)))]
        contract = StructuredReadManifest.model_validate({"apiVersion": "netopyu.io/l0-structured-read/v1", "kind": "StructuredRead",
            "metadata": {"id": f"semantic-transfer.read{index}", "version": "1.0.0", "owner": "local-development-review"},
            "spec": {**{k: adapter[k] for k in ("tool", "capability", "effect", "resourceScopes", "access")},
                     "inputSchema": tool["inputSchema"], "outputSchema": tool["outputSchema"], "sources": sources}})
        reads[name] = compile_structured_read(contract).model_dump(mode="json", by_alias=True)
    packet = {"bundle": bundle, "task": specification["task"], "taskOrigin": "developer_authored_evaluation_request",
              "inputSchema": specification["inputSchema"], "catalog": {"tools": specification["tools"],
              "origin": "disclosed_local_synthetic_host_not_vendor_capture"}, "reads": reads}
    if "taskScope" in specification:
        packet["taskScope"] = specification["taskScope"]
    author.validate_packet(packet)
    return packet
