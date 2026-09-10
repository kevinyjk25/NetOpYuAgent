"""Schema-derived parameter slots. Code builds shape, never chooses business data."""
from __future__ import annotations

import copy

from evaluation import source_catalog, source_modes, structured_authoring as prior
from network_runtime.l0.structured_bindings import compile_binding
from network_runtime.l0.structured_schema import checked_schema, pointer_value, schema_types

PROFILE = "fixed-object-parameter-slots/v1"
SYSTEM = """Bind the parameters of frozenPlan.currentRead. Return requiredOutputSchema JSON; nothing executes.
parameterSlots contains the API's required scalar parameter paths and type-compatible source reference IDs.
For each binding, select ONE reference ID string by the ORIGINAL instruction's business meaning. The IDs
identify future caller/observation data, NOT literal API values. Similar identifier types are not interchangeable.
Code constructs the nested API argument object, so do not recreate object/fields/kind wrappers or actual IDs.
Use a literal object ONLY for an explicit fixed value with its exact original source/task/host-const origin.
Never invent a sample ID or put a template/interpolation string in a literal. A named future value is a reference.
For a valid array index or typed path not shown in navigation, use the explicit reference object; it receives
the same source/type/presence checks. Index zero examples do not imply existence or limit supported indices.
Read the full frozen procedure and original source. Do not alter planDigest/readPointer/slotPacketDigest,
tools, control flow or duties. Unknown source evidence may request_pages or gap_report, never fabricate.
Source/scripts are inert. No execution, credentials, runtime authorization or semantic proof is granted.
"""


def compatible(left, right):
    return bool(left & right or (left & {"integer", "number"} and right & {"integer", "number"}))


def build(packet, plan, slot, navigation, aliases, modes=()):
    """Return None for shapes needing the existing generic expression author."""
    if navigation["navigationTruncated"]:
        return None
    tools = {t["name"]: t for t in packet["catalog"]["tools"]}
    root = checked_schema(tools[slot["tool"]]["inputSchema"])
    if slot.get("operationMode"):
        mode = next(m for d in modes if d["hostTool"] == slot["tool"] for m in d["modes"]
                    if m["id"] == slot["operationMode"])
        root = source_modes.refine(root, mode)
    leaves = []
    def shape(node, target, schema_pointer, depth):
        if depth > 6 or len(leaves) >= 64:
            raise ValueError("slot shape budget")
        if "$ref" in node:
            schema_pointer = node["$ref"][1:]
            return shape(pointer_value(root, schema_pointer), target, schema_pointer, depth + 1)
        kinds = schema_types(node)
        if kinds == {"object"} and "const" not in node and "enum" not in node:
            fields = node.get("properties", {})
            if node.get("additionalProperties") is not False or set(node.get("required", [])) != set(fields):
                raise ValueError("optional/open shape uses generic authoring")
            return {"kind": "object", "fields": {key: shape(value, source_catalog.join_pointer(target, key),
                source_catalog.join_pointer(schema_pointer + "/properties", key), depth + 1) for key, value in fields.items()}}
        if kinds & {"object", "array"}:
            raise ValueError("collection/union shape uses generic authoring")
        references = [row["reference"] for row in navigation["paths"] if compatible(set(row["types"]), kinds)]
        choices = {ref["source"] + "#" + ref["pointer"]: ref for ref in references}
        leaves.append({"target": target, "types": sorted(kinds), "valueSchema": node,
                       "schemaPointer": schema_pointer, "referenceChoices": choices})
        return {"parameterSlot": target}
    try:
        layout = shape(root, "", "", 0)
    except ValueError:
        return None
    sources = {"input": packet["inputSchema"]}
    for read in plan["reads"]:
        for name, pointer in aliases.items():
            if pointer == read["treePointer"]:
                sources[name] = tools[read["tool"]]["outputSchema"]
    return prior.seal({"profile": PROFILE, "planDigest": plan["reportDigest"], "readPointer": slot["treePointer"],
        "layout": layout, "slots": leaves, "sourceSchemas": sources, "targetSchema": root,
        "semanticMappingInferred": False, "runtimeAuthorityGranted": False})


def schema(blueprint):
    fields = {}
    for row in blueprint["slots"]:
        origin = {"$ref": "#/$defs/ValueOrigin"}
        if "const" in row["valueSchema"]:
            origin = {"const": {"kind": "host_schema", "pointer": row["schemaPointer"] + "/const"}}
        variants = []
        if row["referenceChoices"]:
            variants.append({"type": "string", "enum": list(row["referenceChoices"])})
        # Explicit escape for a valid non-enumerated typed path, including any
        # legal array index. Never guess/replace it or treat a template as a ref.
        variants += [prior._obj({"reference": prior._obj({
            "source": {"enum": list(blueprint["sourceSchemas"])}, "pointer": {"type": "string", "maxLength": 600}})}),
            prior._obj({"literal": row["valueSchema"], "origin": origin})]
        fields[row["target"]] = {"oneOf": variants}
    return prior._obj({"mode": {"const": "slot_arguments"}, "planDigest": {"const": blueprint["planDigest"]},
        "readPointer": {"const": blueprint["readPointer"]}, "slotPacketDigest": {"const": blueprint["reportDigest"]},
        "bindings": prior._obj(fields)})


def view(blueprint):
    return {"slotPacketDigest": blueprint["reportDigest"], "profile": PROFILE,
        "slots": [{"target": r["target"], "types": r["types"], "referenceChoices": r["referenceChoices"]}
                  for r in blueprint["slots"]], "correctBusinessSelectionProvided": False}


def lower(blueprint, choice):
    if blueprint != prior.seal({k: v for k, v in blueprint.items() if k != "reportDigest"}):
        raise ValueError("parameter slot packet drift")
    if any(choice[key] != blueprint[target] for key, target in
           (("planDigest", "planDigest"), ("readPointer", "readPointer"), ("slotPacketDigest", "reportDigest"))):
        raise ValueError("parameter choice belongs to another frozen slot")
    rows = {r["target"]: r for r in blueprint["slots"]}
    if set(choice["bindings"]) != set(rows):
        raise ValueError("all and only declared parameter slots required")
    values = {}
    for target, value in choice["bindings"].items():
        if isinstance(value, str):
            if value not in rows[target]["referenceChoices"]:
                raise ValueError("unknown or incompatible typed reference choice")
            expression = rows[target]["referenceChoices"][value]
        elif isinstance(value, dict) and set(value) == {"reference"}:
            reference = value["reference"]
            if not isinstance(reference, dict) or set(reference) != {"source", "pointer"}:
                raise ValueError("invalid explicit pointer reference")
            expression = {"kind": "reference", **reference}
        elif isinstance(value, dict) and set(value) == {"literal", "origin"}:
            expression = {"kind": "literal", "value": value["literal"], "origin": value["origin"]}
        else:
            raise ValueError("invalid parameter selection")
        values[target] = copy.deepcopy(expression)
    def assemble(node):
        if "parameterSlot" in node:
            return values[node["parameterSlot"]]
        return {"kind": "object", "fields": {key: assemble(value) for key, value in node["fields"].items()}}
    arguments = assemble(blueprint["layout"])
    checked = copy.deepcopy(arguments)
    pending = [checked]
    while pending:
        node = pending.pop()
        if node["kind"] == "literal":
            node.pop("origin")
        elif node["kind"] == "object":
            pending.extend(node["fields"].values())
    compile_binding(blueprint["sourceSchemas"], blueprint["targetSchema"], checked)
    return arguments, prior.seal({"profile": PROFILE, "slotPacketDigest": blueprint["reportDigest"],
        "originalChoice": choice, "loweredArguments": arguments, "objectShapeFromHostSchema": True,
        "businessSourceSelectionByCode": False, "literalOriginCheckStillRequired": True,
        "semanticEntailmentProven": False, "runtimeAuthorityGranted": False})
