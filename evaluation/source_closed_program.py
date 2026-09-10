"""Closed model-facing read control trees; syntactic lowering, not semantic repair.

A read has exactly one successor, a decision has exactly two, and a terminal
has none. No implicit exit, unreachable suffix removal or business inference.
The existing statement language/compiler remains the implementation target.
"""
from __future__ import annotations

import copy

from evaluation import source_program_lines as lines
from evaluation import structured_authoring as prior

PROFILE = "inactive-read-plan/closed-control-tree/v2"
TERMINAL_LABELS = {
    "complete": "Source-linked read-only path completed; no broader business outcome is asserted.",
    "handoff": "Source-linked read boundary reached; declared remaining duties have not been executed.",
}


def schema(catalog, modes=(), bindings=(), **kwargs):
    original = lines.schema(catalog, modes, bindings, model_view=True, **kwargs)
    definitions = copy.deepcopy(original["$defs"])
    variants = definitions.pop("ProgramStatement")["oneOf"]
    node = {"$ref": "#/$defs/ProgramNode"}
    for variant in variants:
        properties = variant["properties"]
        op = properties["op"]["const"]
        if "source_id" in properties:
            properties = {"op": properties["op"], "source_id": properties["source_id"],
                          **{k: v for k, v in properties.items() if k not in {"op", "source_id"}}}
            variant["properties"] = properties
        if op == "read":
            # Anchor the invocation before selecting its adapter/output slot;
            # later result processing has a separate source and separate node.
            properties["next"] = node
        elif op in lines.BRANCHES:
            properties["when_equal"] = node
            properties["otherwise"] = node
        if op in TERMINAL_LABELS:
            # Flow status is not an invitation to invent a health/success fact.
            # Actual source-required output work remains in handoff duties.
            properties.pop("explanation")
        variant["required"] = list(properties)
    definitions["ProgramNode"] = {"oneOf": variants}
    return {**node, "$defs": definitions}


def lower(root):
    """Map every node exactly once to existing statement-array positions.

    Validate size/shape before recursive schema work. Source/type/tool checks
    remain in the existing statement renderer, scope checker and compiler.
    """
    pending, count = [(root, 0)], 0
    while pending:
        node, branch_depth = pending.pop()
        count += 1
        if count > 32 or branch_depth > 8 or not isinstance(node, dict):
            raise ValueError("closed program node/depth budget or shape exceeded")
        op = node.get("op")
        children = set(node) & {"next", "when_equal", "otherwise"}
        expected = {"next"} if op == "read" else {"when_equal", "otherwise"} if op in lines.BRANCHES else set()
        if op not in {"read", "if_equal", "if_length_equal", "complete", "handoff"} or children != expected:
            raise ValueError("closed program requires exact successors; terminal continuation is forbidden")
        if op in TERMINAL_LABELS and "explanation" in node:
            raise ValueError("closed terminal status has no model-authored explanation")
        pending.extend((node[key], branch_depth + (op in lines.BRANCHES)) for key in children)
    origins = []

    def visit(node, source, target, index=0):
        pointer = target + f"/{index}"
        row = {k: copy.deepcopy(v) for k, v in node.items() if k not in {"next", "when_equal", "otherwise"}}
        if node["op"] in TERMINAL_LABELS:
            row["explanation"] = TERMINAL_LABELS[node["op"]]
        origins.append({"originalPointer": source, "normalizedPointer": pointer})
        if node["op"] == "read":
            return [row, *visit(node["next"], source + "/next", target, index + 1)]
        if node["op"] in lines.BRANCHES:
            for key in ("when_equal", "otherwise"):
                row[key] = visit(node[key], source + "/" + key, pointer + "/" + key)
        return [row]

    rows = visit(root, "/program", "/program")
    lines.check_size(rows)
    return prior.seal({"profile": PROFILE, "statements": rows, "statementOrigins": origins,
        "nodeCount": count, "businessConditionsInferred": False, "statementsDiscarded": 0,
        "terminalLabels": dict(TERMINAL_LABELS), "businessOutcomesInferred": False,
        "semanticEntailmentProven": False, "runtimeAuthorityGranted": False})
