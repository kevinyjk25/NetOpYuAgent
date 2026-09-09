"""Deterministic, request-bound citation IDs; no model text repair or entailment."""

from __future__ import annotations

import re

MAX_BLOCK_CHARACTERS = 1200


def citation_blocks(pages):
    blocks = {}
    for page_id, page in sorted(pages.items()):
        text, start = page["text"], 0
        boundaries = [m.end() for m in re.finditer(r"\n[ \t]*\n", text)] + [len(text)]
        for end in boundaries:
            while start < end:
                stop = min(end, start + MAX_BLOCK_CHARACTERS)
                key = f"b{len(blocks):04d}"
                blocks[key] = {"page_id": page_id, "path": page["path"], "start": page["start"] + start,
                               "end": page["start"] + stop, "text": text[start:stop]}
                start = stop
    return blocks


def source_span(mark, blocks):
    if not isinstance(mark, dict) or set(mark) != {"block_id"} or mark["block_id"] not in blocks:
        raise ValueError("citation must select one current request-bound source block")
    block = blocks[mark["block_id"]]
    if not 8 <= len(block["text"]) <= MAX_BLOCK_CHARACTERS:
        raise ValueError("source block is too short to establish a citation")
    return {"path": block["path"], "start": block["start"], "end": block["end"], "quote": block["text"]}
