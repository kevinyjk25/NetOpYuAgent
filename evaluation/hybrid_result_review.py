"""Bind a developer-reviewed result mapping to existing L1/source locations.

This is a host installation adapter, NOT automatic semantic duty extraction or
independent Gold. It reuses SourceSpan and the frozen source bundle. Expected
result values are absent; only source-linked obligations and host projections
are declared. Original Stage 2 compilations are never rewritten.
"""
from __future__ import annotations

from network_runtime.contracts import sha256_json
from network_runtime.l0.hybrid import GovernedHybridFlow, qualify_hybrid
from network_runtime.l0.result_contract import ResultContract, bind_result_candidate, qualify_result_contract
from network_runtime.l0.structured_reads import parse_read_contract
from network_runtime.l0.structured_schema import snapshot_json

from evaluation.structured_flow_tree import SourceSpan


def prepare_binding(packet, compilation, raw_contract, mapping):
    contract = ResultContract.model_validate(snapshot_json(raw_contract))
    mapping = snapshot_json(mapping)
    if set(mapping) != {"sourceDigest", "taskDigest", "reviewKind", "duties"}:
        raise ValueError("result mapping must use the explicit source/duty review profile")
    if (mapping["reviewKind"] != "developer_ai_not_independent_gold"
            or mapping["sourceDigest"] != packet["bundle"]["bundleDigest"]
            or mapping["taskDigest"] != sha256_json(packet["task"])
            or contract.mapping_digest != sha256_json(mapping)):
        raise ValueError("result mapping/source/task digest drift")
    expected = {d.id: d for d in contract.duties}
    rows = mapping["duties"]
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("result mapping must cover every declared duty once")
    seen = set()
    docs = {d["path"]: d.get("content") for d in packet["bundle"]["documents"]}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "source_ref", "statement", "source"}:
            raise ValueError("result mapping row needs the original duty ID, locator, statement and source span")
        duty = expected.get(row["id"])
        if duty is None or duty.id in seen or row["statement"] != duty.statement or row["source_ref"] != duty.source_ref:
            raise ValueError("result mapping duty identity or statement drift")
        seen.add(duty.id)
        span = SourceSpan.model_validate(row["source"])
        text = packet["task"] if span.path == "task" else docs.get(span.path)
        if not isinstance(text, str) or text[span.start:span.end] != span.quote or span.end - span.start != len(span.quote):
            raise ValueError("result mapping source span must match the exact original task or retained Skill")
    reads = {c.contract_hash: c for c in (parse_read_contract(raw) for raw in packet["reads"].values())}
    original = GovernedHybridFlow.model_validate(compilation["flow"])
    flow = bind_result_candidate(contract, original, reads)
    qualification = qualify_result_contract(contract, qualify_hybrid(flow, reads), reads)
    return flow, contract, qualification
