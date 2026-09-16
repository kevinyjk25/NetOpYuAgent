"""Immutable observed quotations inside otherwise unverified model drafts.

Exact source matching protects retained bytes, not relevance, entailment, truth,
freshness or permission. Guidance/candidate prose cannot create an observation.
"""
from collections import Counter
import re


def quote_blocks(text):
    """Single-level blockquotes outside fenced code, with exact draft offsets."""
    result, lines, start, offset, opened = [], [], 0, 0, None

    def flush(end):
        if lines:
            result.append({"start": start, "end": end, "text": "\n".join(lines)})
            lines.clear()

    for line in text.splitlines(keepends=True):
        plain = line.rstrip("\r\n")
        quote = re.fullmatch(r" {0,3}> ?(.*)", plain) if opened is None else None
        if quote and not quote[1].startswith(">"):
            if not lines:
                start = offset
            lines.append(quote[1])
        else:
            flush(offset)
            fence = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", plain)
            if fence:
                marker, tail = fence.groups()
                if opened is None and not (marker[0] == "`" and "`" in tail):
                    opened = marker
                elif opened and marker[0] == opened[0] and len(marker) >= len(opened) and not tail.strip():
                    opened = None
        offset += len(line)
    flush(offset)
    return result


def observed_quote_locks(payload, slot):
    locks = []
    for block in quote_blocks(slot["text"]):
        quote = block["text"]
        if not quote.strip():
            continue
        sources = []
        for source in payload["sourceSpans"]:
            if source["kind"] != "observation":
                continue
            text, cursor = source["exactQuote"], 0
            while (start := text.find(quote, cursor)) >= 0:
                end = start + len(quote)
                # Do not establish a match merely inside a larger word.
                if (not start or not (text[start - 1].isalnum() and quote[0].isalnum())) and (
                        end == len(text) or not (text[end].isalnum() and quote[-1].isalnum())):
                    sources.append({"sourceSpanId": source["source_span_id"], "start": start, "end": end})
                cursor = end
        if sources:
            locks.append({**block, "sources": sources, "entailmentOrRelevanceProven": False})
    return locks


def check_quote_preservation(payload, slot, replacement):
    locked = Counter(row["text"] for row in observed_quote_locks(payload, slot))
    delivered = Counter(row["text"] for row in quote_blocks(replacement))
    if locked - delivered:
        raise ValueError("observed source quotation is read-only in this bounded repair; deletion or paraphrase requires a separate reviewed edit")
