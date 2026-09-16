from __future__ import annotations
import json
from .source import validate_bundle, review_pages
from network_runtime.contracts import sha256_json
from network_runtime.l0.read_contracts import _source_object
from network_runtime.l0.structured_reads import parse_read_contract, read_schema, verify_read_contract
from network_runtime.l0.structured_schema import checked_schema, snapshot_json
CONTEXT_TOKENS = 49152
OUTPUT_TOKENS = 4096
TEMPLATE_RESERVE = 4096
MAX_WIRE_BYTES = 131072
MAX_PAGE_BYTES = 2048
def seal(body):
    return {**body, "reportDigest": sha256_json(body)}


def validate_inputs(packet):
    packet = snapshot_json(packet)
    if not isinstance(packet, dict) or set(packet) != {"bundle", "task", "taskOrigin", "inputSchema", "catalog", "reads"}:
        raise ValueError("only original source/task/schema/catalog/read declarations allowed; no reviewer answers")
    validate_bundle(packet["bundle"])
    if (not isinstance(packet["task"], str) or not 12 <= len(packet["task"]) <= 4000
            or packet["taskOrigin"] != "developer_authored_evaluation_request"):
        raise ValueError("explicit bounded development task required")
    checked_schema(packet["inputSchema"])
    catalog = packet["catalog"]
    tools = catalog.get("tools") if isinstance(catalog, dict) else None
    if not isinstance(tools, list) or not 1 <= len(tools) <= 32:
        raise ValueError("bounded explicit original catalog required")
    names = [t.get("name") if isinstance(t, dict) else None for t in tools]
    if any(not isinstance(n, str) or not n.strip() for n in names) or len(set(names)) != len(names):
        raise ValueError("unique host tool names required")
    if not isinstance(packet["reads"], dict) or set(packet["reads"]) != set(names):
        raise ValueError("this read-only research profile requires an exact contract for each declared tool")
    for tool in tools:
        c = verify_read_contract(parse_read_contract(packet["reads"][tool["name"]]))
        if (c.spec.tool != tool["name"] or read_schema(c, "input") != tool["inputSchema"]
                or read_schema(c, "output") != tool["outputSchema"]
                or _source_object(next(s.text for s in c.spec.sources if s.role == "tool")) != tool):
            raise ValueError("read contract differs from original host declaration")
    return packet

def _base_pages_for(packet):
    pages = review_pages(packet["bundle"], max_characters=12000)
    return {f"p{i:03d}": row for i, row in enumerate(pages["pages"])}

def pages_for(packet, *, upstream=None):
    """Lossless, bounded, line-first paging; oversized lines use word/UTF-8 cuts.

    Rejoin adjacent upstream chunks first so their old character cuts cannot
    become accidental sentence boundaries. No source text is summarized away.
    """
    result = []
    originals = []
    for page in (_base_pages_for(packet) if upstream is None else upstream).values():
        if (originals and all(originals[-1][key] == page[key] for key in ("path", "sourceDigest"))
                and originals[-1]["end"] == page["start"]):
            originals[-1] = {**originals[-1], "end": page["end"], "text": originals[-1]["text"] + page["text"]}
        else:
            originals.append(dict(page))
    for original in originals:
        text, cuts, start = original["text"], [], 0
        while start < len(text):
            end, size = start, 0
            while end < len(text) and size + len(text[end].encode("utf-8")) <= MAX_PAGE_BYTES:
                size += len(text[end].encode("utf-8"))
                end += 1
            if end < len(text):
                line_end = text.rfind("\n", start, end) + 1
                word_end = max((i + 1 for i in range(start, end) if text[i].isspace()), default=start)
                end = line_end if line_end > start else word_end if word_end > start else end
            cuts.append((start, end))
            start = end
        if not cuts:
            cuts.append((0, 0))
        for start, end in cuts:
            page = {**original, "start": original["start"] + start, "end": original["start"] + end,
                    "text": text[start:end], "startLine": original["startLine"] + text[:start].count("\n"),
                    "endLine": original["startLine"] + text[:end].count("\n")}
            page["referenceIds"] = [r["referenceId"] for r in packet["bundle"]["references"]
                                    if r["sourcePath"] == page["path"] and page["start"] <= r["start"] < page["end"]]
            page["pageId"] = sha256_json({"path": page["path"], "sourceDigest": page["sourceDigest"],
                                               "start": page["start"], "end": page["end"]})
            result.append(page)
    return {f"p{i:03d}": page for i, page in enumerate(result)}


def budget(wire, *, context_tokens=CONTEXT_TOKENS, output_tokens=OUTPUT_TOKENS,
           template_reserve=TEMPLATE_RESERVE, max_wire_bytes=MAX_WIRE_BYTES):
    """Distinct units, no empirical tokens/byte ratio used as a safety guarantee."""
    message_bytes = sum(len(m["content"].encode("utf-8")) for m in wire["messages"])
    schema_bytes = len(json.dumps(wire["format"], ensure_ascii=False).encode("utf-8"))
    wire_bytes = len(json.dumps(wire, ensure_ascii=False).encode("utf-8"))
    output_tokens = wire.get("options", {}).get("num_predict", output_tokens)
    limit = context_tokens - output_tokens - template_reserve
    return {"messageUtf8Bytes": message_bytes, "formatUtf8Bytes": schema_bytes, "wireBytes": wire_bytes,
            "inputByteProxy": message_bytes + schema_bytes, "proxyLimit": limit,
            "accepted": message_bytes + schema_bytes <= limit and wire_bytes <= max_wire_bytes,
            "actualInputTokens": None, "modelContextTokens": context_tokens,
            "outputTokenReserve": output_tokens, "templateTokenReserve": template_reserve,
            "tokenizerAttested": False,
            "meaning": "Conservative byte-based scheduling proxy, not an exact or certified token bound."}
