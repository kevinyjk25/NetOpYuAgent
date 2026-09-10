"""Carry model-selected original source IDs with statements, never re-infer them."""
from __future__ import annotations

import copy

from evaluation import source_program, source_program_anchors as anchors, source_program_lines as lines
from evaluation import source_closed_program as closed
from evaluation import structured_authoring as prior
from evaluation.source_blocks import source_span

PROFILE = "typed-program-inline-original-source-map/v1"


def prepare(choice, packet, blocks, *, modes=(), bindings=()):
    choices = anchors.evidence_choices(blocks)
    fragments = anchors.scan_fragments(blocks)
    if set(choice["source_scan"]) != set(fragments):
        raise ValueError("source scan requires every original fragment")
    # The namespace is exactly the original blocks plus mechanically enumerated
    # lines. No fuzzy matching, answer templates, ranking or model rebinding.
    combined = {**copy.deepcopy(blocks), **{key: copy.deepcopy(value["block"]) for key, value in choices.items()}}
    # Internal statement-array callers remain testable; the current model wire
    # accepts only a closed tree. Lowering never discards an unreachable suffix.
    syntax = closed.lower(choice["program"]) if isinstance(choice["program"], dict) else None
    rows = syntax["statements"] if syntax else choice["program"]
    rendered = lines.render(rows, packet["catalog"], modes, bindings, evidence_ids=choices)
    parsed = source_program.parse(rendered, input_schema=packet["inputSchema"], catalog=packet["catalog"])
    normalized = lines.canonicalize(rows)
    origins = {r["normalizedPointer"]: r["originalPointer"] for r in normalized["statementOrigins"]}
    if syntax:
        tree_origins = {r["normalizedPointer"]: r["originalPointer"] for r in syntax["statementOrigins"]}
        origins = {key: tree_origins[value] for key, value in origins.items()}
    mappings = []

    def record(row, pointer, role, origin):
        key = row["source_id"]
        mappings.append({"normalizedPointer": pointer, "originalPointer": origin,
            "role": role, "selectedEvidenceId": key,
            "source": source_span({"block_id": key}, combined),
            "originalBlock": source_span({"block_id": choices[key]["block_id"]}, blocks)})

    def visit(rows, path):
        for index, row in enumerate(rows):
            pointer = path + f"/{index}"
            record(row, pointer, row["op"], origins[pointer])
            for field in ("duties", "restrictions"):
                for i, duty in enumerate(row.get(field, [])):
                    record(duty, pointer + f"/{field}/{i}", field, origins[pointer] + f"/{field}/{i}")
            if row["op"] in lines.BRANCHES:
                for field in ("when_equal", "otherwise"):
                    visit(row[field], pointer + "/" + field)

    visit(normalized["statements"], "/program")
    for field in ("procedure", "business_gaps", "outside_task_duties", "execution_requirements"):
        for row in choice[field]:
            source_span(row["source"], blocks)
    if sum(m["role"] == "read" for m in mappings) > 8:
        raise ValueError("eight-read authoring limit exceeded")
    draft = prior.seal({"profile": PROFILE, "choice": copy.deepcopy(choice), "renderedProgram": rendered,
        "sourceScan": [{"id": key, **choice["source_scan"][key],
                        "source": source_span({"block_id": key}, combined)} for key in fragments],
        "sourceScanMeaning": "complete_fragment_response_not_verified_semantics_or_program_coverage",
        "controlFlowNormalization": normalized, "inlineSourceMap": mappings, "parsedStructure": parsed,
        **({"controlSyntaxLowering": syntax} if syntax else {}),
        "packetDigest": prior.sha256_json(packet), "semanticEntailmentProven": False,
        "runtimeAuthorityGranted": False})
    audit = prior.seal({"profile": PROFILE, "draftDigest": draft["reportDigest"], "anchors": mappings,
        "programStructureChanged": False, "sourceSelectionsInferred": False,
        "additionalSourceBindingModelCalls": 0, "semanticEntailmentProven": False, "runtimeAuthorityGranted": False})
    prepared = {**copy.deepcopy(choice), "program": rendered}
    return prepared, combined, draft, audit
