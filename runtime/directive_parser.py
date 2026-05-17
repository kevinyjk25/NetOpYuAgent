"""
Directive parser for [TOOL:...], [SKILL_LOAD:...], [TOOL_BATCH:...].

Why this module exists
======================
The framework defines an in-band text protocol for tool / skill / batch
invocations (see integrations/clients/llm_engine.py:TOOL_CALL_SYSTEM).
Before this module existed, every consumer (runtime/loop.py, webui/backend.py,
skills/catalog.py, ...) re-implemented the parsing regex with slightly
different strictness:

  - runtime/loop.py:2401   r"\\[TOOL:(\\w+)\\]\\s*(\\{)?"      strict, no space tolerance
  - runtime/loop.py:1533   r"\\s*\\[TOOL:\\w+\\]"               strict, anchor-only
  - runtime/loop.py:2096   r"\\[TOOL:[^\\]]+\\][^\\n]*"        loose, eats trailing text
  - webui/backend.py:647   r"\\[TOOL:(\\w+)\\]"                 strict
  - skills/catalog.py:161  r"\\[TOOL:(\\w+)\\]"                 strict

This drift caused real production failures: when the LLM emitted
"[TOOL: get_device_config]" (note the space after the colon), the parser
returned [] and _is_complete() saw "no tool calls", terminating the loop
mid-task. The model wasn't wrong — the format was within reasonable
human-readable tolerance — the regex was just too strict.

This module centralizes the parsing. Consumers MUST go through these
helpers instead of writing their own regex. To enforce that, scripts/
audit_directive_parsing.py warns on any new \\[TOOL: / \\[SKILL_LOAD: /
\\[TOOL_BATCH: regex outside this file.

Tolerance contract
==================
All parsers tolerate, by design:
  - Whitespace inside the brackets:  [ TOOL : name ]   ==   [TOOL:name]
  - Whitespace between bracket and args:  [TOOL:name]  {}   ==   [TOOL:name]{}
  - Case-insensitive directive keyword:  [tool:name]   ==   [TOOL:name]
  - Trailing punctuation:  [TOOL:name].   parses name correctly
  - Leading text on the same line:  Prefix [TOOL:name]   parses
  - Multiple directives in one response (only the first is "active"
    by policy, but all are reported so the rule check can flag duplicates)

What we DO NOT tolerate (intentionally):
  - Missing brackets:  TOOL:name {}  → not a directive
  - Wrong keyword:     [INVOKE:name] → not a directive
  - Invalid name chars: [TOOL:get device]  →  invalid (space in name)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------
# `re.IGNORECASE` makes the keyword (TOOL/SKILL_LOAD/TOOL_BATCH) case-tolerant.
# Whitespace allowed around brackets and colon. Name is \w+ (letters/digits/_).
# The args portion ({...} or [...]) is captured permissively — exact JSON
# validation happens later in schema/validator.py, not here.

_TOOL_RE = re.compile(
    r"\[\s*TOOL\s*:\s*(?P<name>\w+)\s*\]\s*(?P<args_open>\{)?",
    re.IGNORECASE,
)

_SKILL_LOAD_RE = re.compile(
    r"\[\s*SKILL_LOAD\s*:\s*(?P<name>\w+)\s*\]",
    re.IGNORECASE,
)

_TOOL_BATCH_RE = re.compile(
    r"\[\s*TOOL_BATCH\s*:\s*(?P<name>\w+)\s*\]\s*(?P<args_open>\[)?",
    re.IGNORECASE,
)

# For substitution / stripping a full directive INCLUDING its args.
# These end-anchor at the closing brace/bracket or end-of-line so callers
# can strip directives from a response and inspect the prose around them.
# IMPORTANT: braces inside JSON arg values mean a naive `\{[^}]*\}` will
# stop at the first inner `}` — we use a greedy-stop-at-newline pattern
# instead. For nested JSON the LLM normally puts everything on one line,
# so this is reliable enough.

_TOOL_FULL_RE = re.compile(
    r"\[\s*TOOL\s*:\s*\w+\s*\]\s*(?:\{[^\n]*?\})?",
    re.IGNORECASE,
)

_TOOL_BATCH_FULL_RE = re.compile(
    r"\[\s*TOOL_BATCH\s*:\s*\w+\s*\]\s*(?:\[[^\n]*?\])?",
    re.IGNORECASE,
)

_SKILL_LOAD_FULL_RE = re.compile(
    r"\[\s*SKILL_LOAD\s*:\s*\w+\s*\]",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ToolDirective:
    """One [TOOL:name] directive parsed from LLM output.

    Attributes:
        name: the tool name (lower-case-preserved as emitted)
        start, end: character offsets of the matching directive bracket
                    (NOT including the args JSON; args parsing happens
                    separately because the regex can't reliably bracket-
                    match nested JSON)
        has_args_open: whether the directive was followed by '{' (so the
                    caller knows to look for an args block after end)
    """
    name: str
    start: int
    end: int
    has_args_open: bool


def find_tool_directives(text: str) -> list[ToolDirective]:
    """Return ALL [TOOL:name] directives in `text`, in source order."""
    out: list[ToolDirective] = []
    if not text:
        return out
    for m in _TOOL_RE.finditer(text):
        out.append(ToolDirective(
            name=m.group("name"),
            start=m.start(),
            end=m.end(),
            has_args_open=m.group("args_open") is not None,
        ))
    return out


def find_tool_names(text: str) -> list[str]:
    """Convenience: just the names. Order-preserving, duplicates kept."""
    return [d.name for d in find_tool_directives(text)]


def find_skill_load_names(text: str) -> list[str]:
    """Return SKILL_LOAD ids in source order. Duplicates kept."""
    if not text:
        return []
    return [m.group("name") for m in _SKILL_LOAD_RE.finditer(text)]


@dataclass(frozen=True)
class ToolBatchDirective:
    """One [TOOL_BATCH:name] directive (batch destructive op).

    Same shape as ToolDirective but `has_args_open` looks for '[' (the
    batch args are a list of arg-dicts).
    """
    name: str
    start: int
    end: int
    has_args_open: bool


def find_tool_batch_directives(text: str) -> list[ToolBatchDirective]:
    """Return all [TOOL_BATCH:name] directives."""
    out: list[ToolBatchDirective] = []
    if not text:
        return out
    for m in _TOOL_BATCH_RE.finditer(text):
        out.append(ToolBatchDirective(
            name=m.group("name"),
            start=m.start(),
            end=m.end(),
            has_args_open=m.group("args_open") is not None,
        ))
    return out


# ---------------------------------------------------------------------------
# Stripping helpers — useful for "what's the prose part of the response,
# minus any directives?". Used by _is_complete and friends.
#
# IMPORTANT: directive args (the JSON object after [TOOL:name] or the JSON
# array after [TOOL_BATCH:name]) can contain nested braces/brackets and span
# multiple lines. A naive `r"\{[^}]*\}"` or `r"\[.*?\]"` regex either stops
# at the first inner closer or misses multi-line args entirely. We do a
# proper balance-aware scan instead, identical in spirit to the one in
# runtime/loop.py:_parse_tool_calls.
# ---------------------------------------------------------------------------
def find_balanced_end(text: str, open_pos: int, open_ch: str, close_ch: str) -> int:
    """Given `text[open_pos] == open_ch`, return the index AFTER the matching
    `close_ch`. Tracks string state (so braces inside JSON strings don't
    decrement depth) and handles ``\\``-escaping. Returns -1 if unbalanced.

    Public utility shared between this module (for stripping directives) and
    runtime/loop.py (for extracting the JSON args block of a parsed
    [TOOL:name] or [TOOL_BATCH:name] directive).

    Used to find the end of `[TOOL:name] {...}` or `[TOOL_BATCH:name] [...]`
    blocks for accurate stripping.

    Examples:
        >>> find_balanced_end('{"a": {"b": 1}}', 0, '{', '}')
        15
        >>> find_balanced_end('[1, [2, 3], 4]', 0, '[', ']')
        14
        >>> find_balanced_end('{"x":"}"}', 0, '{', '}')   # close inside string
        9
        >>> find_balanced_end('{ unclosed', 0, '{', '}')
        -1
    """
    if open_pos >= len(text) or text[open_pos] != open_ch:
        return -1
    depth = 0
    in_str = False
    escape = False
    for i in range(open_pos, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i + 1
    return -1


# Backwards-compat alias: internal _strip_directive_with_args still calls
# _find_balanced_end. Removing the alias would force a same-module rename;
# keeping it lets external callers use the public name without breaking the
# internal call sites.
_find_balanced_end = find_balanced_end


def _strip_directive_with_args(
    text: str,
    opener_re: "re.Pattern",
    args_open_ch: str,
    args_close_ch: str,
) -> str:
    """Remove all directive matches PLUS their balanced args block.

    Scans for opener matches in source order, but builds the result string
    by accumulating non-matched ranges. After each opener at `m.start()`:
      - If the directive has an `args_open` group capture (i.e. a `{` or
        `[` immediately followed), use _find_balanced_end to locate the
        true end of the args block.
      - Otherwise the directive is bare (no args) and we strip just the
        bracket part.
    """
    if not text:
        return text
    out: list[str] = []
    cursor = 0
    for m in opener_re.finditer(text):
        # Append everything since the last match
        out.append(text[cursor:m.start()])
        args_open = m.group("args_open") if "args_open" in m.groupdict() else None
        if args_open is not None:
            # The opener regex captures the args_open char as a lookahead
            # group. Find its actual position (may have whitespace between
            # the `]` and the `{`).
            scan_from = m.end() - 1
            while scan_from < len(text) and text[scan_from] != args_open_ch:
                scan_from += 1
            end_pos = _find_balanced_end(text, scan_from, args_open_ch, args_close_ch)
            if end_pos == -1:
                # Unbalanced — strip only the opener bracket, leave the
                # rest intact (probably truncated output from a token limit).
                cursor = m.end()
            else:
                cursor = end_pos
        else:
            cursor = m.end()
    out.append(text[cursor:])
    return "".join(out)


def strip_tool_directives(text: str) -> str:
    """Remove all [TOOL:...] {...} blocks. Use to inspect the prose-only
    portion of a response (e.g. checking if the LLM also wrote analysis
    text, not just emitted a tool call). Handles multi-line + nested JSON."""
    return _strip_directive_with_args(text, _TOOL_RE, "{", "}")


def strip_tool_batch_directives(text: str) -> str:
    """Remove all [TOOL_BATCH:...] [...] blocks. Handles multi-line +
    nested arrays/objects via depth-tracking scanner."""
    return _strip_directive_with_args(text, _TOOL_BATCH_RE, "[", "]")


def strip_skill_load_directives(text: str) -> str:
    """Remove all [SKILL_LOAD:...] tokens. No args — simple regex sub
    is correct here."""
    if not text:
        return text
    return _SKILL_LOAD_FULL_RE.sub("", text)


def strip_all_directives(text: str) -> str:
    """Remove all three directive types — useful for prose-only inspection."""
    text = strip_tool_batch_directives(text)
    text = strip_tool_directives(text)
    text = strip_skill_load_directives(text)
    return text


# ---------------------------------------------------------------------------
# Normalization — some small models emit malformed variants we can fix
# before parsing. Kept narrow on purpose: we only fix things that are
# unambiguous typos, not anything that could be intentional.
# ---------------------------------------------------------------------------
_NORMALIZE_PATTERNS: list[tuple[re.Pattern, str]] = [
    # [TOOL:SKILL_LOAD:X]  →  [SKILL_LOAD:X]
    # qwen variants sometimes wrap SKILL_LOAD inside the TOOL: container.
    (re.compile(r"\[\s*TOOL\s*:\s*SKILL_LOAD\s*:\s*(\w+)\s*\]", re.IGNORECASE),
     r"[SKILL_LOAD:\1]"),
]


def normalize_directives(text: str) -> str:
    """Apply small-model-specific cleanups before parsing.

    Currently fixes:
      - [TOOL:SKILL_LOAD:X] → [SKILL_LOAD:X]
    """
    if not text:
        return text
    for pat, repl in _NORMALIZE_PATTERNS:
        text = pat.sub(repl, text)
    return text


# ---------------------------------------------------------------------------
# Detection helpers (boolean queries)
# ---------------------------------------------------------------------------
def has_any_tool_directive(text: str) -> bool:
    """True if `text` contains at least one [TOOL:...] or [TOOL_BATCH:...]."""
    if not text:
        return False
    return bool(_TOOL_RE.search(text) or _TOOL_BATCH_RE.search(text))


def has_skill_load(text: str) -> bool:
    """True if `text` contains at least one [SKILL_LOAD:...]."""
    if not text:
        return False
    return bool(_SKILL_LOAD_RE.search(text))
