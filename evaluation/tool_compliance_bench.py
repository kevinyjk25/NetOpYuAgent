"""
evaluation/tool_compliance_bench.py
-----------------------------------
Bench runner for tool-call compliance.

For each ToolCallCase:
  1. Send the case's query to the LLM engine with empty context (so we
     isolate the tool-emission behaviour from memory/retrieval effects)
  2. Parse the response with a balanced-brace JSON extractor
  3. Score parse_ok / name_ok / args_ok against expectations
  4. Record elapsed time and any exceptions

The bench is engine-agnostic: it just needs an `engine.call(query,
context, state)` method that returns a string. This lets us A/B native-
tools vs text-protocol by setting `capabilities.supports_native_tools`
on the engine before running.

Typical usage:
    bench = ToolComplianceBench(engine=my_ollama_engine,
                                name="qwen3.5:27b/native")
    report = await bench.run(cases)
    print(f"compliance: {report.compliance:.0%}")

Architectural note: this lives in `evaluation/` because it's a
measurement tool, not an LLM client. Per ARCHITECTURE.md, evaluation
modules MAY depend on `runtime/`, `integrations/`, `schema/` for
constructing engines and parsing — but the bench itself is engine-
neutral and only references the abstract `engine.call(...)` contract.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Optional

from evaluation.tool_compliance_types import (
    ToolCallCase, ToolCallResult, ToolComplianceReport,
)


logger = logging.getLogger(__name__)


# Match a [TOOL:name] directive. Args block is OPTIONAL (some tools take
# no args — e.g. list_dormant_skills). When args are present we capture
# everything up to the next [TOOL: or end-of-line group, then re-parse
# JSON with balance-aware logic so nested braces don't truncate.
_TOOL_DIRECTIVE_RE = re.compile(
    r"\[\s*TOOL\s*:\s*(?P<name>\w+)\s*\]",
    re.IGNORECASE,
)


def _slice_args_json(text: str, start: int) -> tuple[Optional[str], int]:
    """Starting at position `start` in `text`, find the next `{` and return
    the balanced JSON object string + the index after its closing `}`.

    Returns (None, start) if no `{` is found before a `[` or end-of-text
    (i.e. the directive has no args block — valid for args-less tools).
    """
    # Skip whitespace
    i = start
    while i < len(text) and text[i] in " \t":
        i += 1
    # Stop conditions: another directive starts, end of text, or non-{ token
    if i >= len(text) or text[i] != "{":
        return None, start
    # Balance-aware scan. Tracks string state (no brace counting inside
    # quoted strings) and escape sequences. This matches what real JSON
    # parsers do well enough for the compliance bench.
    depth = 0
    in_str = False
    esc = False
    j = i
    while j < len(text):
        ch = text[j]
        if esc:
            esc = False
        elif ch == "\\" and in_str:
            esc = True
        elif ch == '"':
            in_str = not in_str
        elif not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[i:j + 1], j + 1
        j += 1
    # Unbalanced — return what we have so the JSON parser can flag it
    return text[i:], len(text)


def _extract_first_tool_call(response: str) -> tuple[Optional[str], dict[str, Any], bool]:
    """Return (tool_name, args_dict, args_present).

    args_present is True iff a `{...}` block followed the directive
    (regardless of whether it parsed as JSON). This lets the bench
    distinguish "directive with no args" (valid for args-less tools)
    from "directive with malformed args" (failure mode).
    """
    if not response:
        return None, {}, False
    m = _TOOL_DIRECTIVE_RE.search(response)
    if not m:
        return None, {}, False
    name = m.group("name")
    args_json, _end = _slice_args_json(response, m.end())
    if args_json is None:
        return name, {}, False
    try:
        args = json.loads(args_json)
        if not isinstance(args, dict):
            return name, {}, True
        return name, args, True
    except json.JSONDecodeError:
        logger.debug("tool_compliance: args JSON decode failed for %s: %r",
                     name, args_json[:80])
        return name, {}, True


def _check_args(
    case: ToolCallCase, parsed_args: dict[str, Any],
) -> tuple[bool, list[str], list[str], list[str]]:
    """Return (args_ok, missing, wrong_value, forbidden_present)."""
    missing: list[str] = []
    wrong_value: list[str] = []
    forbidden_present: list[str] = []

    # Required keys must be present (with any non-None value if we didn't pin one)
    for k in case.required_arg_names:
        if k not in parsed_args:
            missing.append(k)
            continue
        expected = case.expected_args.get(k)
        if expected is None:
            # Caller pinned the key but not the value — presence is enough.
            continue
        got = parsed_args[k]
        # Type-aware compare: lists/strings of same content are equal;
        # numeric coercion permitted (1 vs "1" — many LLMs stringify
        # numbers and we don't want to fail for that).
        if not _values_equivalent(expected, got):
            wrong_value.append(k)

    for k in case.forbidden_args:
        if k in parsed_args:
            forbidden_present.append(k)

    args_ok = not (missing or wrong_value or forbidden_present)
    return args_ok, missing, wrong_value, forbidden_present


def _values_equivalent(expected: Any, got: Any) -> bool:
    """Loose equivalence for arg values.

    We tolerate:
      - 1 == "1" (string-encoded numbers)
      - True == "true" (case-insensitive booleans)
      - Lists compared element-wise with the same tolerances

    We DON'T tolerate semantic differences ("ap-01" ≠ "ap-02" even though
    they're "similar"). The bench is checking did the model put the
    user's stated value into the right slot — not whether the model
    chose a reasonable value.
    """
    if expected == got:
        return True
    if isinstance(expected, bool) or isinstance(got, bool):
        # Match before numeric coercion since bool is a subclass of int.
        return str(expected).lower() == str(got).lower()
    if isinstance(expected, (int, float)) and isinstance(got, str):
        try:
            return float(expected) == float(got)
        except ValueError:
            return False
    if isinstance(expected, str) and isinstance(got, (int, float)):
        try:
            return float(expected) == float(got)
        except ValueError:
            return False
    if isinstance(expected, list) and isinstance(got, list):
        if len(expected) != len(got):
            return False
        return all(_values_equivalent(a, b) for a, b in zip(expected, got))
    return False


class ToolComplianceBench:
    """Run a batch of ToolCallCase against an engine."""

    def __init__(self, engine: Any, name: str = "engine") -> None:
        self._engine = engine
        self._name = name

    async def run(self, cases: list[ToolCallCase]) -> ToolComplianceReport:
        results: list[ToolCallResult] = []
        parse_ok_count = name_ok_count = args_ok_count = full_count = 0
        errored = 0
        total_elapsed = 0.0

        for case in cases:
            res = await self._run_one(case)
            results.append(res)
            total_elapsed += res.elapsed_ms
            if res.error is not None:
                errored += 1
                continue
            if res.parse_ok: parse_ok_count += 1
            if res.name_ok:  name_ok_count  += 1
            if res.args_ok:  args_ok_count  += 1
            if res.fully_compliant: full_count += 1

        total = len(cases)
        avg_ms = total_elapsed / total if total else 0.0
        return ToolComplianceReport(
            backend_name=self._name,
            total=total,
            parse_ok_count=parse_ok_count,
            name_ok_count=name_ok_count,
            args_ok_count=args_ok_count,
            fully_compliant_count=full_count,
            avg_elapsed_ms=avg_ms,
            errored_count=errored,
            cases=results,
        )

    async def _run_one(self, case: ToolCallCase) -> ToolCallResult:
        t0 = time.perf_counter()
        try:
            # Empty context; no state. Bench measures raw emission behaviour.
            # If `state=None` causes the engine to skip retrieval/prompt
            # decoration, that's fine — we want the cleanest signal.
            response = await self._engine.call(case.query, "", state=None)
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            return ToolCallResult(
                case=case, raw_response="", elapsed_ms=elapsed,
                error=f"{type(exc).__name__}: {exc}",
            )
        elapsed = (time.perf_counter() - t0) * 1000

        name, args, args_present = _extract_first_tool_call(response)
        parse_ok = name is not None
        # name_ok: parsed name must be expected_tool or in acceptable_tools.
        # `acceptable_tools` lets ambiguous queries pass with any of several
        # reasonable picks.
        accepted_names = {case.expected_tool, *case.acceptable_tools}
        name_ok = parse_ok and name in accepted_names

        # args_ok only meaningful if we found a directive AND the right tool.
        # Wrong tool → args don't get checked (different schema applies).
        # Special case: if the case expects no args (e.g. list_dormant_skills),
        # args_ok is True as long as name_ok and either no args block was
        # emitted or it was empty/well-formed.
        if name_ok:
            if not case.required_arg_names and not case.forbidden_args:
                # Args-less tool: presence of an empty {} or no {} is fine.
                # An args block that contains keys is also fine (extra info).
                args_ok, missing, wrong, forbidden = True, [], [], []
            elif not args_present:
                # Directive emitted with NO args block at all, but case
                # requires some — fail at missing-args check.
                args_ok = False
                missing = list(case.required_arg_names)
                wrong, forbidden = [], []
            else:
                args_ok, missing, wrong, forbidden = _check_args(case, args)
        else:
            args_ok, missing, wrong, forbidden = False, [], [], []

        return ToolCallResult(
            case=case, raw_response=response,
            parsed_name=name, parsed_args=args,
            parse_ok=parse_ok, name_ok=name_ok, args_ok=args_ok,
            missing_args=missing, wrong_value_args=wrong,
            forbidden_present=forbidden,
            elapsed_ms=elapsed,
        )
