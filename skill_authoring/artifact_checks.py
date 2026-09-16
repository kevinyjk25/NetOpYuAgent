"""Allowlisted, non-executing artifact checks. Passing is NOT semantic approval.

Only Markdown fences, JSON/Python parsing, literal ratio arithmetic and a narrow
KQL predicate subset are checked. No imports, eval, source scripts or subprocesses.
Unsupported language/interval expressions explicitly remain unverified.
"""
import ast
from decimal import Decimal, InvalidOperation
import json
import re

from .contracts import seal
from . import kql_checks
from network_runtime.contracts import sha256_json

PROFILE = "inert-artifact-checks/v2"
STAMP = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z"
WINDOW = re.compile(rf"\[\s*({STAMP})\s*,\s*({STAMP})\s*\)")


def _row(location, check, status, quote, explanation, **extra):
    return {"location": location, "check": check, "status": status, "artifactQuote": quote,
            "explanation": explanation, "semanticApproval": False, **extra}


def _json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _kql(code, location, windows):
    rows = []
    # Strip comments only. Never interpret arbitrary query text as Python/SQL.
    active = "\n".join(line.split("//", 1)[0] for line in code.splitlines())
    for line in active.splitlines():
        if re.search(r"^\s*\|\s*where\s+\w+\s+between\b", line, re.I):
            valid_shape = re.search(r"\bbetween\s*\(.+?\s*\.\.\s*.+\)\s*$", line, re.I)
            # Only reject the demonstrably malformed single-line SQL-like form.
            # A multiline range or an extra predicate can be valid KQL; abstain.
            malformed = not valid_shape and re.search(r"\bbetween\s*\([^\n]*\)\s+and\s*\(", line, re.I) and ".." not in line
            rows.append(_row(location, "kql_between_shape", "fail" if malformed else "unverified", line.strip(),
                "KQL between requires (lower .. upper), inclusive at both ends. Shape alone does not validate the expression."))
        # A narrow, unambiguous typed-literal position, not strings/comments.
        bad_literal = re.search(rf"(?:>=|<=|>|<|==|\(|\.\.)\s*(datetime(['\"]){STAMP}\2)", line, re.I)
        if bad_literal:
            rows.append(_row(location, "kql_datetime_literal", "fail", bad_literal[1],
                "A KQL datetime literal uses datetime(ISO-date-time), not datetime followed directly by a quoted date. No full parser is implied."))
    rows.append(_row(location, "kql_complete_language", "unverified", code,
                     "No full Kusto parser or database execution; function signatures and complete query validity remain unverified."))
    for state, quote, detail in kql_checks.column_checks(code):
        rows.append(_row(location, "kql_column_lineage", state, quote, detail))
    if len(windows) != 1:
        rows.append(_row(location, "half_open_time_filter", "unverified", code,
                         "Requires exactly one distinct explicit [UTC, UTC) interval in task/observations; never guess."))
        return rows
    _, anchors = next(iter(windows.items()))
    state, explanation = kql_checks.interval(code, windows)
    rows.append(_row(location, "half_open_time_filter", state, code, explanation, sourceAnchors=anchors))
    return rows


def inspect(payload):
    windows = {}
    for source in payload["sourceSpans"]:
        if source["kind"] not in {"task", "observation"}:
            continue
        for match in WINDOW.finditer(source["exactQuote"]):
            windows.setdefault(match.groups(), []).append({"source_span_id": source["source_span_id"], "quote": match[0]})
    rows = []
    for span in payload["draftSpans"]:
        location, content = span["draft_span_id"], span["exactQuote"]
        lines = content.splitlines()
        if not lines:
            continue
        opened = re.fullmatch(r" {0,3}(`{3,}|~{3,})([^\s]*)\s*", lines[0])
        if opened:
            marker, language = opened.groups()
            closed = len(lines) > 1 and bool(re.fullmatch(rf" {{0,3}}{re.escape(marker[0])}{{{len(marker)},}}\s*", lines[-1]))
            rows.append(_row(location, "fence_closed", "pass" if closed else "fail", content,
                             "Structural fence closure only; not a semantic or execution check."))
            if not closed:
                continue
            code = "\n".join(lines[1:-1])
            language = language.lower()
            if language in {"json", "python", "py"}:
                try:
                    if language == "json":
                        json.loads(code, object_pairs_hook=_json_pairs,
                                   parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON constant")))
                    else:
                        ast.parse(code)
                    state, detail = "pass", "Parsed without executing; dependencies, types, safety and behavior NOT established."
                except (ValueError, SyntaxError, RecursionError) as error:
                    state, detail = "fail", f"Syntax-only check: {type(error).__name__}: {str(error)[:200]}"
                rows.append(_row(location, "syntax_" + language, state, code, detail))
            elif language in {"kql", "kusto"}:
                rows.extend(_kql(code, location, windows))
            else:
                rows.append(_row(location, "language_support", "unverified", code, "Language not allowlisted; source is never executed."))
        else:
            # Literal ratios only, no interpretation of natural-language units.
            for match in re.finditer(r"(?<![\w.])([0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+(?:\.[0-9]+)?)\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*%", content):
                try:
                    a, b, percent = map(Decimal, match.groups())
                    correct = b != 0 and a / b * 100 == percent
                except InvalidOperation:
                    correct = False
                rows.append(_row(location, "literal_ratio", "pass" if correct else "fail", match[0],
                                 "Exact decimal ratio only, not correctness of source values, denominator choice or rounded estimates."))
    return seal({"profile": PROFILE, "candidateDigest": payload["completeCandidateDigest"], "checks": rows,
        "unverifiedDimensions": ["task meaning", "full language semantics", "source truth", "general calculations", "note entailment"],
        "sourceScriptsExecuted": False, "completeAnswerApproved": False})


def draft_spans(text):
    """Lossless mechanical addressing; no model-selected edit regions."""
    spans, pending, marker = [], [], None
    for line in text.splitlines(keepends=True):
        opening = re.fullmatch(r" {0,3}(`{3,}|~{3,})([^\s]*)\s*", line.rstrip("\r\n"))
        if marker is None and opening:
            if pending:
                spans.append("".join(pending))
            pending, marker = [line], opening[1]
        else:
            pending.append(line)
            if marker and re.fullmatch(rf" {{0,3}}{re.escape(marker[0])}{{{len(marker)},}}\s*", line.rstrip("\r\n")):
                spans.append("".join(pending))
                pending, marker = [], None
    if pending:
        spans.append("".join(pending))
    return spans


def inspect_candidate(candidate, task, observations):
    """Advisory partial checks on frozen evidence, never complete semantic approval."""
    spans = draft_spans(candidate["draft"])
    source = [{"source_span_id": "task", "kind": "task", "exactQuote": task}]
    source += [{"source_span_id": f"observation-{i}", "kind": "observation",
                "exactQuote": json.dumps(value, ensure_ascii=False)} for i, value in enumerate(observations)]
    result = inspect({"completeCandidateDigest": sha256_json(candidate), "sourceSpans": source,
        "draftSpans": [{"draft_span_id": f"draft-{i}", "exactQuote": text} for i, text in enumerate(spans)]})
    rows = result["checks"]
    return seal({"inspection": result, "sourceEvidenceDigest": sha256_json(source),
        "status": "failed_checks" if any(r["status"] == "fail" for r in rows) else "partial_checks_only" if rows else "no_supported_artifact",
        "semanticApproval": False, "queryExecuted": False,
        "limitation": "No full language parser, function catalog/type check, natural-language entailment or source truth validation. Native harness final edits are not covered by this candidate digest."})
