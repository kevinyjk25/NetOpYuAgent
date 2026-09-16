"""Host-shaped parameter authoring, without choosing business parameter values.

Required object shape/types come from the declared tool. The model selects
typed caller references or original literal evidence; source text never runs.
"""
from __future__ import annotations

import re

from network_runtime.l0.structured_schema import checked_schema, pointer_value, schema_types, validate_data


def obj(properties, required=None):
    return {"type": "object", "properties": properties,
        "required": list(properties) if required is None else required, "additionalProperties": False}


def references(input_schema):
    result = {}
    def walk(node, path, depth):
        if depth > 8 or len(result) >= 96:
            return
        if "$ref" in node:
            node = pointer_value(input_schema, node["$ref"][1:])
        result["input#" + path] = (node, {"kind": "reference", "source": "input", "pointer": path})
        for key, child in node.get("properties", {}).items():
            walk(child, path + "/" + key.replace("~", "~0").replace("/", "~1"), depth + 1)
    walk(checked_schema(input_schema), "", 0)
    return result


def compatible(left, right):
    a, b = schema_types(left), schema_types(right)
    return bool(a & b or (a & {"integer", "number"} and b & {"integer", "number"}))


def binding_schema(target, caller, marks):
    choices = references(caller)
    def walk(node, depth):
        if depth > 8:
            raise ValueError("parameter schema depth budget")
        if "$ref" in node:
            node = pointer_value(target, node["$ref"][1:])
        if "const" in node:
            return obj({"host_const": {"const": True}})
        refs = [key for key, (value, _) in choices.items() if compatible(value, node)]
        variants = [obj({"caller": {"enum": refs}})] if refs else []
        variants.append(obj({"observation": {"type": "string", "pattern": "^n[0-7]$"}, "pointer": {"type": "string", "maxLength": 400}}))
        kinds = schema_types(node)
        if "const" in node:
            variants.append(obj({"host_const": {"const": True}}))
        elif kinds <= {"string", "integer", "number", "boolean", "null"}:
            # Exact destination primitive types; avoid one generic literal
            # union that makes the small model stringify numbers/booleans.
            variants.append(obj({"literal": node, "origin": {"enum": marks}, "quote": {"type": "string", "minLength": 1, "maxLength": 4000}}))
        elif kinds == {"object"} and node.get("additionalProperties") is False:
            variants.append(obj({key: walk(child, depth + 1) for key, child in node.get("properties", {}).items()}, node.get("required", [])))
        elif kinds == {"array"}:
            variants.append({"type": "array", "items": walk(node.get("items", {}), depth + 1),
                "minItems": node.get("minItems", 0), "maxItems": min(node.get("maxItems", 32), 32)})
        return {"anyOf": variants}
    return walk(checked_schema(target), 0)


def lower(target, caller, value, origins):
    refs, provenance = references(caller), []
    def walk(node, raw, path):
        if "$ref" in node:
            node = pointer_value(target, node["$ref"][1:])
        if "const" in node and raw != {"host_const": True}:
            raise ValueError("a fixed host constant cannot be replaced by an unrelated caller value")
        if isinstance(raw, dict) and set(raw) == {"caller"}:
            key = raw["caller"]
            if key not in refs or not compatible(refs[key][0], node):
                raise ValueError("unknown or incompatible caller reference")
            provenance.append({"target": path, "origin": "future_caller_data", "reference": key})
            return refs[key][1]
        if isinstance(raw, dict) and set(raw) == {"observation", "pointer"}:
            if not isinstance(raw["observation"], str) or not re.fullmatch("n[0-7]", raw["observation"]):
                raise ValueError("invalid observed source")
            provenance.append({"target": path, "origin": "future_observation", **raw})
            return {"kind": "reference", "source": raw["observation"], "pointer": raw["pointer"]}
        if isinstance(raw, dict) and set(raw) == {"host_const"}:
            if "const" not in node or raw["host_const"] is not True:
                raise ValueError("host constant is absent")
            provenance.append({"target": path, "origin": "host_schema_const", "value": node["const"]})
            return {"kind": "literal", "value": node["const"]}
        if isinstance(raw, dict) and set(raw) == {"literal", "origin", "quote"}:
            item = validate_data(node, raw["literal"])
            if raw["origin"] not in origins:
                raise ValueError("literal origin was not supplied")
            mark = origins[raw["origin"]]
            text = mark.get("text", mark.get("content", ""))
            quote = raw["quote"]
            if not isinstance(quote, str) or not quote or quote not in text:
                raise ValueError("literal needs an exact original source/task quote")
            if isinstance(item, str):
                matches = bool(item) and item in quote
            elif item is None or type(item) is bool:
                matches = re.search(r"\b" + ("null" if item is None else str(item).lower()) + r"\b", quote) is not None
            else:
                matches = re.search(r"(?<![\w.])" + re.escape(str(item)) + r"(?![\w.])", quote) is not None
            if not matches:
                raise ValueError("literal value is not in the cited original quote")
            provenance.append({"target": path, "origin": mark["origin"], "source": raw["origin"],
                "quote": quote, "value": item, "semanticEntailmentProven": False})
            return {"kind": "literal", "value": item}
        if schema_types(node) == {"object"} and isinstance(raw, dict):
            if set(raw) - node.get("properties", {}).keys() or set(node.get("required", [])) - raw.keys():
                raise ValueError("parameters differ from host object shape")
            return {"kind": "object", "fields": {key: walk(node["properties"][key], child, path + "/" + key) for key, child in raw.items()}}
        if schema_types(node) == {"array"} and isinstance(raw, list):
            return {"kind": "array", "items": [walk(node["items"], child, path + "/" + str(index)) for index, child in enumerate(raw)]}
        raise ValueError("unknown typed parameter binding")
    return walk(checked_schema(target), value, ""), provenance
