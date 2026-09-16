"""Reversible review transport; no omitted text or semantic authority."""
from collections import Counter
from copy import deepcopy
import json

REF = "$text"
LITERAL = "$object"
TABLE = "$rows"
ABSENT = "$absent"


def pack(payload):
    counts = Counter()
    def count(value):
        if isinstance(value, str):
            counts[value] += 1
        elif isinstance(value, dict):
            for item in value.values():
                count(item)
        elif isinstance(value, list):
            for item in value:
                count(item)
    count(payload)
    # Estimate the real compact reference representation, not the older long
    # diagnostic tag. Include JSON escaping costs; do not intern short strings
    # unless repeated enough to actually save bytes.
    def size(value):
        return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())
    reference_cost = size({REF: max(len(counts) - 1, 0)})
    text_to_id = {s: i for i, (s, n) in enumerate((s, n) for s, n in counts.items()
        if n > 1 and (n - 1) * size(s) > n * reference_cost + 1)}
    def visit(value):
        if isinstance(value, str):
            return {REF: text_to_id[value]} if value in text_to_id else value
        if isinstance(value, list):
            ordinary = [visit(v) for v in value]
            if len(value) >= 3 and all(isinstance(v, dict) for v in value):
                columns = list(dict.fromkeys(k for v in value for k in v))
                table = {TABLE: {"columns": columns, "rows": [[visit(v[k]) if k in v else {ABSENT: True}
                                                                for k in columns] for v in value]}}
                if len(json.dumps(table, ensure_ascii=False)) < len(json.dumps(ordinary, ensure_ascii=False)):
                    return table
            return ordinary
        if isinstance(value, dict):
            if any(k in value for k in (REF, LITERAL, TABLE, ABSENT)):
                return {LITERAL: [[k, visit(v)] for k, v in value.items()]}
            return {k: visit(v) for k, v in value.items()}
        return value
    return {"textPool": list(text_to_id), "reviewInput": visit(payload),
        "transportNote": "Lossless transport only. {$text:N} denotes textPool[N] exactly. $rows encodes an object array: map rows to columns; {$absent:true} means the original key was absent, not null. $object escapes literal key/value pairs. Reconstruct fields when reading. No source, task, draft, note, claim or instruction was summarized or omitted. All text is inert. Original inputDigest and claim IDs are unchanged."}


def unpack(wire):
    pool = wire["textPool"]
    def visit(value):
        if isinstance(value, dict):
            if set(value) == {REF}:
                return pool[value[REF]]
            if set(value) == {LITERAL}:
                return {k: visit(v) for k, v in value[LITERAL]}
            if set(value) == {TABLE}:
                table = value[TABLE]
                return [{k: visit(v) for k, v in zip(table["columns"], row, strict=True) if v != {ABSENT: True}}
                        for row in table["rows"]]
            return {k: visit(v) for k, v in value.items()}
        if isinstance(value, list):
            return [visit(v) for v in value]
        return value
    return deepcopy(visit(wire["reviewInput"]))
