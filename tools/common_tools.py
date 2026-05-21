"""
tools/common_tools.py — Profile-independent (common framework) tools
====================================================================

These tools are NOT business-specific — every profile (default / lan / dc)
gets them. They implement the ToolResultStore paging mechanism that lets the
agent read large tool outputs that were offloaded to disk/cache.

  - read_stored_result   — page through a [STORED:name:ref_id] payload
  - process_stored_chunks — iterate + transform an entire stored payload

Their prompt-facing metadata lives in tools/builtin/registry.py. The factory
`make_read_stored_result_tool(tool_store)` binds them to a ToolResultStore
instance at startup (in main.py:build_services).

`_ts()` is a tiny timestamp helper shared by profile tools that emit
realistic mock log lines; it lives here so profile modules don't duplicate it.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any


def _ts(offset_minutes: int = 0) -> str:
    """UTC timestamp 'Mon DD HH:MM:SS', offset minutes into the past."""
    t = datetime.now(timezone.utc) - timedelta(minutes=offset_minutes)
    return t.strftime("%b %d %H:%M:%S")


def make_read_stored_result_tool(tool_store):
    """
    Factory: returns a tool function bound to a specific ToolResultStore.
    Used to let the LLM retrieve a page of a large cached result.
    """
    async def read_stored_result(args: dict[str, Any]) -> str:
        ref_id = args.get("ref_id", "")
        offset = int(args.get("offset", 0))
        # Default length raised from 2000 to 8000 chars (~2K tokens) so a
        # typical netflow_dump or syslog page-through completes in 5-7 pages
        # instead of 25+. LLM is still free to override with smaller value
        # for sampling, but the default favours fewer turns over fewer tokens
        # per turn — turns are the bottleneck on slow local models.
        # Hard cap at 16000 to keep one page under typical 4K-token budgets.
        length = int(args.get("length", 8000))
        length = max(256, min(length, 16000))

        if not ref_id:
            return "[Error: ref_id is required]"

        # Normalise ref_id: LLM often copies the full "[STORED:tool:uuid]" label
        ref_id = ref_id.strip("[]")
        if ":" in ref_id:
            ref_id = ref_id.rsplit(":", 1)[-1].strip()

        chunk = tool_store.read(ref_id, offset=offset, length=length)
        if chunk is None:
            return f"[Error: no stored result found for ref_id={ref_id!r}]"

        total = len(tool_store._store.get(ref_id, ""))
        next_offset = offset + len(chunk)
        has_more    = next_offset < total

        return (
            f"# Stored result ref_id={ref_id} offset={offset} length={len(chunk)}\n"
            f"# Total size: {total} chars  |  Has more: {has_more}  |  "
            f"Next offset: {next_offset if has_more else 'EOF'}\n"
            f"# {'─'*60}\n"
            f"{chunk}"
        )

    async def process_stored_chunks(args: dict[str, Any]) -> str:
        """
        General-purpose chunk iterator over a stored large result.

        Splits the stored content into fixed-size chunks and applies one of
        several built-in operations to every chunk, accumulating the results.
        This is the right tool whenever you need to process an entire large file
        rather than just reading one page of it.

        Operations (set via `operation` arg):
          "filter"    – keep only lines that contain `match` (case-insensitive)
          "reject"    – keep only lines that do NOT contain `match`
          "extract"   – extract the first regex `pattern` capture group from each line
          "count"     – count lines containing `match` per chunk; return totals
          "summarise" – return the first `head` and last `tail` lines of each chunk
                        (useful for giving an LLM a digest of each section)
          "passthrough" – return every chunk verbatim (same as calling read_stored_result
                          in a loop, but done for you automatically)

        Common usage patterns:
          Check if user X exists anywhere in a log:
            {"ref_id": "...", "operation": "filter", "match": "alice@corp.com"}

          Find all ERROR lines across the whole file:
            {"ref_id": "...", "operation": "filter", "match": "ERROR"}

          Extract all IP addresses from a NetFlow dump:
            {"ref_id": "...", "operation": "extract", "pattern": "(\\d+\\.\\d+\\.\\d+\\.\\d+)"}

          Count how many times each chunk mentions 'timeout':
            {"ref_id": "...", "operation": "count", "match": "timeout"}

          Get a digest of a large prometheus dump:
            {"ref_id": "...", "operation": "summarise", "head": 3, "tail": 2}
        """
        import re as _re

        ref_id    = args.get("ref_id", "")
        operation = args.get("operation", "filter")
        match_str = args.get("match", "")
        pattern   = args.get("pattern", "")
        chunk_size = int(args.get("chunk_size", 3000))   # chars per chunk
        max_output = int(args.get("max_output", 6000))   # cap total output chars
        head_n    = int(args.get("head", 5))
        tail_n    = int(args.get("tail", 5))

        if not ref_id:
            return "[Error: ref_id is required]"

        full = tool_store._store.get(ref_id)
        if full is None:
            return f"[Error: no stored result for ref_id={ref_id!r}]"

        total_chars = len(full)
        all_lines   = full.splitlines()
        total_lines = len(all_lines)

        # Split into line-aligned chunks
        chunks: list[list[str]] = []
        current: list[str] = []
        current_len = 0
        for line in all_lines:
            current.append(line)
            current_len += len(line) + 1
            if current_len >= chunk_size:
                chunks.append(current)
                current = []
                current_len = 0
        if current:
            chunks.append(current)

        output_parts: list[str] = []
        total_matched = 0
        output_chars  = 0

        header = (
            f"# process_stored_chunks ref_id={ref_id} operation={operation!r}\n"
            f"# Total: {total_chars} chars, {total_lines} lines, {len(chunks)} chunk(s)\n"
            f"# {'─'*60}\n"
        )
        output_parts.append(header)
        output_chars += len(header)

        for chunk_idx, chunk_lines in enumerate(chunks):
            if output_chars >= max_output:
                output_parts.append(
                    f"\n# [Output cap {max_output} chars reached — "
                    f"{len(chunks) - chunk_idx} chunk(s) not shown]\n"
                )
                break

            if operation == "filter":
                kw = match_str.lower()
                hits = [l for l in chunk_lines if kw in l.lower()]
                total_matched += len(hits)
                if hits:
                    block = f"\n# chunk {chunk_idx+1}: {len(hits)} match(es)\n" + "\n".join(hits)
                    output_parts.append(block)
                    output_chars += len(block)

            elif operation == "reject":
                kw = match_str.lower()
                kept = [l for l in chunk_lines if kw not in l.lower()]
                total_matched += len(kept)
                block = f"\n# chunk {chunk_idx+1}: {len(kept)} line(s) kept\n" + "\n".join(kept)
                output_parts.append(block)
                output_chars += len(block)

            elif operation == "extract":
                if not pattern:
                    return "[Error: 'pattern' is required for operation='extract']"
                extracted = []
                for line in chunk_lines:
                    m = _re.search(pattern, line)
                    if m:
                        val = m.group(1) if m.lastindex else m.group(0)
                        extracted.append(val)
                seen = sorted(set(extracted))
                total_matched += len(seen)
                if seen:
                    block = f"\n# chunk {chunk_idx+1}: {len(seen)} unique value(s)\n" + "\n".join(seen)
                    output_parts.append(block)
                    output_chars += len(block)

            elif operation == "count":
                kw = match_str.lower()
                count = sum(1 for l in chunk_lines if kw in l.lower())
                total_matched += count
                block = f"\n# chunk {chunk_idx+1}: {count} line(s) contain {match_str!r}"
                output_parts.append(block)
                output_chars += len(block)

            elif operation == "summarise":
                h = chunk_lines[:head_n]
                t = chunk_lines[-tail_n:] if tail_n else []
                mid_omitted = max(0, len(chunk_lines) - head_n - tail_n)
                block = (
                    f"\n# chunk {chunk_idx+1} ({len(chunk_lines)} lines):\n"
                    + "\n".join(h)
                    + (f"\n  … {mid_omitted} lines omitted …\n" if mid_omitted > 0 else "\n")
                    + ("\n".join(t) if t else "")
                )
                output_parts.append(block)
                output_chars += len(block)

            elif operation == "passthrough":
                block = f"\n# chunk {chunk_idx+1}:\n" + "\n".join(chunk_lines)
                output_parts.append(block)
                output_chars += len(block)

            else:
                return f"[Error: unknown operation={operation!r}. Choose: filter, reject, extract, count, summarise, passthrough]"

        # Summary footer
        op_summary = {
            "filter":      f"{total_matched} matching line(s) found",
            "reject":      f"{total_matched} line(s) passed filter",
            "extract":     f"{total_matched} unique value(s) extracted across all chunks",
            "count":       f"{total_matched} total line(s) matched across all chunks",
            "summarise":   f"{len(chunks)} chunk(s) summarised",
            "passthrough": f"{len(chunks)} chunk(s) returned",
        }.get(operation, "")
        footer = f"\n# {'─'*60}\n# Result: {op_summary}\n"
        output_parts.append(footer)

        return "".join(output_parts)

    return read_stored_result, process_stored_chunks


