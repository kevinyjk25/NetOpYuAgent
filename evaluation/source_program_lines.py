"""Typed planning statements -> inert syntax. No semantic completion."""
from __future__ import annotations

import json
import copy

from jsonschema import Draft202012Validator

from evaluation import structured_authoring as prior
from evaluation.source_program import NAME

PROFILE = "inactive-read-plan/typed-statements-v7"
BRANCHES = {"if_equal", "if_length_equal"}
OBSERVATION_SLOTS = tuple(f"obs{i}" for i in range(8))


def schema(catalog, modes=(), bindings=(), *, model_view=False, value_paths=None, evidence_ids=None, observation_names=None):
    text = {"type": "string", "minLength": 8, "maxLength": 400}
    name = {"type": "string", "pattern": "^" + NAME + "$"}
    observation = {"$ref": "#/$defs/ProgramObservation"} if observation_names is not None else name
    read_name = {"$ref": "#/$defs/ProgramReadName"} if observation_names is not None else name
    value = prior._obj({"kind": {"enum": ["field", "length"]}, "source": observation,
                       "pointer": {"type": "string", "maxLength": 300}})
    def row(op, properties):
        fields = {"op": {"const": op}, **properties}
        if evidence_ids is not None:
            origin = {"$ref": "#/$defs/ProgramSourceId"}
            if op in BRANCHES:
                # The source belongs to the predicate, not the last branch
                # instruction. Keep this declaration adjacent to that operand
                # before asking for potentially long nested child programs.
                origin = {"$ref": "#/$defs/ProgramPredicateSourceId"}
                fields = {**{k: v for k, v in fields.items() if k not in {"when_equal", "otherwise"}},
                          "source_id": origin, "when_equal": fields["when_equal"], "otherwise": fields["otherwise"]}
            else:
                fields["source_id"] = origin
        return prior._obj(fields)
    block = {"type": "array", "maxItems": 32, "items": {"$ref": "#/$defs/ProgramStatement"}}
    reads = []
    for tool in catalog["tools"]:
        tool_modes = next((d["modes"] for d in modes if d["hostTool"] == tool["name"]), [None])
        operations = sorted({b["sourceOperation"] for b in bindings if b["hostTool"] == tool["name"]}) or [None]
        for mode in tool_modes:
            for operation in operations:
                properties = {"name": read_name, "tool": {"const": tool["name"]}}
                if mode:
                    properties["operationMode"] = {"const": mode["id"]}
                if operation:
                    properties["sourceOperation"] = {"const": operation}
                reads.append(row("read", properties))
    duty = prior._obj({"when": text, "requirement": text,
                      **({"source_id": {"$ref": "#/$defs/ProgramSourceId"}} if evidence_ids is not None else {})})
    handoff = {"outcome": {"enum": ["needs_l1", "unsupported"]}, "explanation": text,
               "duties": {"type": "array", "minItems": 1, "maxItems": 8, "items": duty}}
    restrictions = {"type": "array", "maxItems": 8, "items": duty}
    compare = {**value, "properties": {**value["properties"], "kind": {"const": "field"}}} if model_view else {
        "oneOf": [value, prior._obj({"kind": {"const": "alias"}, "name": name})]}
    predicates = [
        row("if_equal", {"value": compare,
                         "equals": {"type": ["string", "number", "boolean", "null"]},
                         "when_equal": {**block, "minItems": 1}, "otherwise": block}),
        row("if_length_equal", {"source": observation, "pointer": {"type": "string", "maxLength": 300},
                         "equals": {"type": "integer", "minimum": 0},
                         "when_equal": {**block, "minItems": 1}, "otherwise": block}),
    ]
    if model_view and value_paths is not None:
        # All original type-compatible paths remain selectable. The source
        # observation, comparison value, polarity and branch bodies are NOT
        # chosen here. The original compiler checks the selected source/type.
        predicates = []
        for scalar_type in ("boolean", "number", "string", "null"):
            kinds = {"integer", "number"} if scalar_type == "number" else {scalar_type}
            paths = sorted({p["pointer"] for p in value_paths if set(p["types"]) & kinds
                            and set(p["types"]) <= {"string", "number", "integer", "boolean", "null"}})
            if paths:
                operand = prior._obj({"kind": {"const": "field"}, "source": observation, "pointer": {"enum": paths}})
                predicates.append(row("if_equal", {"value": operand, "equals": {"type": scalar_type},
                    "when_equal": {**block, "minItems": 1}, "otherwise": block}))
        arrays = sorted({p["pointer"] for p in value_paths if p["types"] == ["array"]})
        if arrays:
            operand = prior._obj({"kind": {"const": "length"}, "source": observation, "pointer": {"enum": arrays}})
            predicates.append(row("if_equal", {"value": operand,
                "equals": {"type": "integer", "minimum": 0},
                "when_equal": {**block, "minItems": 1}, "otherwise": block}))
    variants = reads + predicates + [
        row("complete", {"explanation": text}),
        row("handoff", {**handoff, **({"restrictions": restrictions} if model_view else {})}),
    ]
    if not model_view:
        variants.append(row("handoff", {**handoff, "restrictions": restrictions}))
        # Keep hand-authored/internal aliases and old inert end syntax; the
        # model-facing subset needs neither. Direct field pointers retain the
        # same read/condition expressivity without another variable namespace.
        variants += [row("define", {"name": name, "value": value}),
        row("end", {"outcome": {"const": "read_path_completed"}, "explanation": text,
                    "duties": {"type": "array", "maxItems": 0}}),
        row("end", {"outcome": {"enum": ["needs_l1", "unsupported"]}, "explanation": text,
                    "duties": {"type": "array", "minItems": 1, "maxItems": 8, "items": duty}}),
        ]
    return {**block, "minItems": 1, "$defs": {"ProgramStatement": {"oneOf": variants},
        **({"ProgramSourceId": {"enum": list(evidence_ids)},
            "ProgramPredicateSourceId": {"$ref": "#/$defs/ProgramSourceId",
                "description": "Original text defining this predicate and comparison; not a branch action or reference link."}}
           if evidence_ids is not None else {}),
        **({"ProgramObservation": {"enum": ["input", *observation_names]},
            "ProgramReadName": {"enum": list(observation_names)}} if observation_names is not None else {})}}


def check_size(rows):
    # Check shape/depth before recursive JSON Schema traversal, even for a
    # caller that did not pass through the bounded model-response decoder.
    pending, count = [(rows, 0)], 0
    while pending:
        items, depth = pending.pop()
        if not isinstance(items, list) or depth > 8:
            raise ValueError("typed program depth or block shape exceeded")
        count += len(items)
        if count > 32:
            raise ValueError("typed program statement count exceeds 32")
        for item in items:
            if isinstance(item, dict) and item.get("op") in BRANCHES:
                pending.extend((item.get(key), depth + 1) for key in ("when_equal", "otherwise"))
def canonicalize(rows):
    """Move a continuation ONLY into the unique arm that can reach it.

    This is structural continuation passing, not an inferred business guard:
    the other arm has an explicit terminal on every path. Never duplicate a
    statement, remove unreachable work or export possibly undefined values.
    """
    check_size(rows)
    moves, redundant = [], []
    def discardable(rest):
        if len(rest) != 1:
            return False
        value, _ = rest[0]
        return (value["op"] == "complete"
                or (value["op"] == "end" and value["outcome"] == "read_path_completed" and not value["duties"]))
    def record_redundant(rest, predecessor):
        value, pointer = rest[0]
        redundant.append({"originalPointer": pointer, "statement": copy.deepcopy(value),
            "terminatedAfter": predecessor, "rule": "single_unreachable_completion_without_duties",
            "reason": "Every incoming path already explicitly terminates; no observation or handoff is removed."})
    def open_path(items):
        for item in items:
            if item["op"] in {"end", "complete", "handoff"}:
                return False
            if item["op"] in BRANCHES and not any(open_path(item[k]) for k in ("when_equal", "otherwise")):
                return False
        return True
    def pairs(items, path):
        return [(row, path + f"/{i}") for i, row in enumerate(items)]
    def block(items):
        result = []
        for index, (original, origin) in enumerate(items):
            row = copy.deepcopy(original)
            row["_origin"] = origin
            rest = items[index + 1:]
            trim = False
            if row["op"] in {"end", "complete", "handoff"} and rest:
                if not discardable(rest):
                    raise ValueError("unreachable original statements must not be discarded")
                record_redundant(rest, origin)
                trim = True
                rest = []
            if row["op"] in BRANCHES:
                arms = ("when_equal", "otherwise")
                remaining = [k for k in arms if open_path(row[k])]
                if rest and not remaining:
                    if not discardable(rest):
                        raise ValueError("unreachable original continuation must not be discarded")
                    record_redundant(rest, origin)
                    trim = True
                    rest = []
                target = remaining[0] if rest and len(remaining) == 1 else None
                for key in arms:
                    children = pairs(row[key], origin + "/" + key)
                    if key == target:
                        children += rest
                        moves.append({"rule": "append_to_only_fallthrough_arm", "branchOrigin": origin,
                            "targetArm": key, "movedStatementOrigins": [p for _, p in rest]})
                    row[key] = block(children)
                if target:
                    return result + [row]
            result.append(row)
            if trim:
                return result
        return result
    normalized = block(pairs(rows, "/program"))
    origins = []
    def strip(items, path):
        for i, row in enumerate(items):
            pointer = path + f"/{i}"
            origins.append({"normalizedPointer": pointer, "originalPointer": row.pop("_origin")})
            if row["op"] in BRANCHES:
                for key in ("when_equal", "otherwise"):
                    strip(row[key], pointer + "/" + key)
    strip(normalized, "/program")
    check_size(normalized)
    return prior.seal({"statements": normalized, "moves": moves, "statementOrigins": origins,
        "redundantTerminals": redundant,
        "ruleScope": "explicit_terminal_control_flow_only", "businessConditionsInferred": False,
        "semanticEntailmentProven": False, "runtimeAuthorityGranted": False})


def render(rows, catalog, modes=(), bindings=(), *, evidence_ids=None):
    check_size(rows)
    error = next(Draft202012Validator(schema(catalog, modes, bindings, evidence_ids=evidence_ids)).iter_errors(rows), None)
    if error:
        raise ValueError("invalid typed program statement: " + error.message[:200])
    rows = canonicalize(rows)["statements"]
    def quote(value):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    def value(item, source_id=None):
        if item["kind"] == "alias":
            return item["name"]
        return item["kind"] + "(" + item["source"] + ", " + quote(item["pointer"]) + (
            ", " + quote(source_id) if source_id is not None else "") + ")"
    lines = []
    def statement(row, level):
        op = row["op"]
        source_id = row.get("source_id") if evidence_ids is not None else None
        if op == "read":
            optional = "".join(", " + keyword + "=" + quote(row[key])
                for key, keyword in (("operationMode", "operation_mode"), ("sourceOperation", "source_operation")) if key in row)
            origin = (", " + quote(source_id) + ", " + quote("Model-planned observation: " + row["tool"])) if source_id is not None else ""
            text = row["name"] + " = read(" + quote(row["tool"]) + origin + optional + ")"
        elif op == "define":
            text = row["name"] + " = " + value(row["value"], source_id)
        elif op in BRANCHES:
            scalar = row["equals"]
            literal = {"true": "True", "false": "False", "null": "None"}.get(quote(scalar), quote(scalar))
            operand = row["value"] if op == "if_equal" else {"kind": "length", "source": row["source"], "pointer": row["pointer"]}
            text = "if " + value(operand, source_id) + " == " + literal + ":"
            lines.append("    " * level + text)
            for child in row["when_equal"]:
                statement(child, level + 1)
            if row["otherwise"]:
                lines.append("    " * level + "else:")
                for child in row["otherwise"]:
                    statement(child, level + 1)
            return
        else:
            args = [quote(row.get("outcome", "read_path_completed")), quote(row["explanation"])]
            if source_id is not None:
                args.insert(1, quote(source_id))
            duties = row.get("duties", []) + row.get("restrictions", [])
            if duties:
                args.append("[" + ", ".join("(" + ((quote(d["source_id"]) + ", ") if source_id is not None else "")
                    + quote(d["when"]) + ", " + quote(d["requirement"]) + ")" for d in duties) + "]")
            text = "end(" + ", ".join(args) + ")"
        lines.append("    " * level + text)
    for row in rows:
        statement(row, 0)
    return "\n".join(lines) + "\n"
