"""Explicit closed-key host modes for inactive authoring, never permissions.

Only object presence/key constraints are supported. They refine, never replace,
the original host schema. Dynamic values cannot hide the constrained shape.
Declarations are developer-reviewed claims, not authenticated host attestation.
"""
from __future__ import annotations

import copy

from jsonschema import Draft202012Validator

from evaluation.structured_authoring import _obj, parse_read_contract, seal, sha256_json
from network_runtime.l0.structured_schema import checked_schema, join_pointer, pointer_value, snapshot_json

PROFILE = "declared-closed-key-operation-modes/v2"
API_VERSION = "netopyu.io/host-operation-modes/v1"
REVIEW_KIND = "developer_reviewed_host_declaration_not_independent_gold"

INSTRUCTIONS = """HOST OPERATION MODES: each read selects a declared operationMode; modes constrain
parameter combinations, not source intent, prerequisites or permission. Construct object/array
arguments explicitly with leaf origins; dynamic references may not hide a mode-constrained object.
Each read's source.block_id selects its proposed action witness; code retains the exact original
block, so do not retype source text. Block selection is not proof that a paragraph needs a call.
Preserve source order/conditions and an explicit terminal. Do not manufacture one call per paragraph
or remove a necessary repeated observation. A mode declaration does not create an operation in L1.
"""


def packet_declarations(value):
    if (not isinstance(value, dict) or set(value) != {"apiVersion", "declarations"}
            or value["apiVersion"] != API_VERSION or not isinstance(value["declarations"], list)):
        raise ValueError("versioned host-operation-mode packet required")
    return value["declarations"]


def authoring_view(declarations):
    """Do not repeat contract hashes/prose already frozen and validated on disk.

    All mode constraints/descriptions are supplied; the digest binds the full
    developer declaration, not provider identity or semantic approval.
    """
    return {"declarationsDigest": sha256_json(declarations), "role": REVIEW_KIND,
            "tools": [{"hostTool": d["hostTool"], "modes": d["modes"]} for d in declarations]}


def _parts(pointer):
    if pointer == "":
        return []
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("objectPointer must be a canonical JSON Pointer")
    parts = [s.replace("~1", "/").replace("~0", "~") for s in pointer[1:].split("/")]
    rebuilt = ""
    for part in parts:
        rebuilt = join_pointer(rebuilt, part)
    if rebuilt != pointer:
        raise ValueError("objectPointer must be a canonical JSON Pointer")
    return parts


def refine(original, mode):
    """Presence refinements use only the existing restricted schema vocabulary.

    Dereference each changed branch separately: shared $defs must not mutate
    unrelated paths. No arbitrary assertion, value inference or union lowering.
    """
    original = checked_schema(original)
    root = copy.deepcopy(original)

    def object_at(node, parts):
        if "$ref" in node:
            node = copy.deepcopy(pointer_value(original, node["$ref"][1:]))
        # Reference chains are acyclic under checked_schema.
        while "$ref" in node:
            node = copy.deepcopy(pointer_value(original, node["$ref"][1:]))
        if node.get("type") != "object":
            raise ValueError("mode paths require explicit non-nullable object schemas")
        # Aggregate const/enum need intersection reasoning, not discarded values.
        if "const" in node or "enum" in node:
            raise ValueError("aggregate const/enum mode refinement is not supported")
        if parts:
            key = parts[0]
            if key not in node.get("properties", {}):
                raise ValueError("mode path must traverse declared object properties")
            node["required"] = sorted(set(node.get("required", [])) | {key})
            node["properties"][key] = object_at(node["properties"][key], parts[1:])
        else:
            allowed = set(rule["allowedKeys"])
            required = set(node.get("required", [])) | set(rule["requiredKeys"])
            if (not required <= allowed or not allowed <= node.get("properties", {}).keys()
                    or node.get("minProperties", 0) > len(allowed)
                    or len(required) > node.get("maxProperties", 512)):
                raise ValueError("mode conflicts with original required/declared fields or object bounds")
            node["properties"] = {k: v for k, v in node.get("properties", {}).items() if k in allowed}
            node["required"] = sorted(required)
            node["additionalProperties"] = False
        return node

    # Parent restrictions are applied first; a child cannot resurrect a forbidden key.
    for rule in sorted(mode["objects"], key=lambda r: len(_parts(r["objectPointer"]))):
        root = object_at(root, _parts(rule["objectPointer"]))
    return checked_schema(root)


def validate(packet, declarations):
    declarations = snapshot_json(declarations)
    text = {"type": "string", "minLength": 1, "maxLength": 600}
    keys = {"type": "array", "maxItems": 64, "uniqueItems": True,
            "items": {"type": "string", "maxLength": 128}}
    rule = _obj({"objectPointer": {"type": "string", "maxLength": 600},
                 "requiredKeys": keys, "allowedKeys": keys})
    mode = _obj({"id": {"type": "string", "pattern": "^[a-z][a-z0-9_-]{0,47}$"}, "description": text,
                 "objects": {"type": "array", "minItems": 1, "maxItems": 16, "items": rule}})
    schema = {"type": "array", "maxItems": 32, "items": _obj({
        "hostTool": text, "catalogDigest": text, "contractHash": text,
        "inputSchemaDigest": text, "outputSchemaDigest": text,
        "reviewKind": {"const": REVIEW_KIND}, "declarationEvidence": text,
        "modes": {"type": "array", "minItems": 1, "maxItems": 8, "items": mode}})}
    if not Draft202012Validator(schema).is_valid(declarations):
        raise ValueError("invalid host operation-mode declaration")
    if len({d["hostTool"] for d in declarations}) != len(declarations):
        raise ValueError("duplicate host mode declaration")
    tools = {t["name"]: t for t in packet["catalog"]["tools"]}
    for declaration in declarations:
        tool = tools.get(declaration["hostTool"])
        if tool is None:
            raise ValueError("mode targets an undeclared host tool")
        contract = parse_read_contract(packet["reads"][tool["name"]])
        if (declaration["catalogDigest"] != sha256_json(packet["catalog"])
                or declaration["contractHash"] != contract.contract_hash
                or declaration["inputSchemaDigest"] != sha256_json(tool["inputSchema"])
                or declaration["outputSchemaDigest"] != sha256_json(tool["outputSchema"])):
            raise ValueError("host mode catalog/contract/schema drift")
        modes = declaration["modes"]
        if len({m["id"] for m in modes}) != len(modes):
            raise ValueError("duplicate operation mode id")
        for m in modes:
            if len({r["objectPointer"] for r in m["objects"]}) != len(m["objects"]):
                raise ValueError("duplicate object rule in mode")
            refine(tool["inputSchema"], m)
        for index, left in enumerate(modes):
            for right in modes[index + 1:]:
                # Deliberately bounded proof, not a general JSON-schema SAT solver.
                separated = any(a["objectPointer"] == b["objectPointer"] and (
                    set(a["requiredKeys"]) - set(b["allowedKeys"]) or
                    set(b["requiredKeys"]) - set(a["allowedKeys"]))
                    for a in left["objects"] for b in right["objects"])
                if not separated:
                    raise ValueError("operation mode exclusivity is not structurally proven")
    return declarations


def shape_errors(expression, mode):
    """Prove only declared key presence/absence from constructed syntax or constants.

    Binding compilation is type-overlap, NOT subtyping. Do not discard a mode
    after a reference merely appears compatible with its object type.
    """
    errors = []
    for rule in mode["objects"]:
        expr, value, constant = expression, None, False
        try:
            for key in _parts(rule["objectPointer"]):
                if not constant and expr["kind"] == "literal":
                    value, constant = expr["value"], True
                if constant:
                    value = value[key]
                elif expr["kind"] == "object":
                    expr = expr["fields"][key]
                else:
                    raise LookupError("dynamic expression hides a mode-constrained object")
            if not constant and expr["kind"] == "literal":
                value, constant = expr["value"], True
            if constant:
                if not isinstance(value, dict):
                    raise ValueError("mode requires an object")
                keys = set(value)
            elif expr["kind"] == "object":
                keys = set(expr["fields"])
            else:
                raise LookupError("dynamic expression hides a mode-constrained object")
            if not set(rule["requiredKeys"]) <= keys <= set(rule["allowedKeys"]):
                raise ValueError("object keys violate the selected operation mode")
        except (LookupError, TypeError, ValueError) as error:
            errors.append({"code": "operation_mode_shape", "dataPointer": rule["objectPointer"], "detail": str(error)})
    return errors


def plan_report(rows, declarations):
    return seal({"profile": PROFILE, "declarationsDigest": sha256_json(declarations), "operations": rows,
                 "shapeProof": "closed_object_keys_only_not_all_host_business_rules",
                 "sourceActionEntailmentProven": False, "sourceOrderCorrectnessProven": False,
                 "runtimeAuthorityGranted": False, "providerCalls": 0})
