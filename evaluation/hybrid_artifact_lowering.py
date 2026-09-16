"""Explicit opt-in, source-bound rendering of one narrow typed KQL literal.

Not a general repair engine or parser: only datetime'UTC' -> datetime(UTC) in
already located malformed predicate literals. No timestamp selection, endpoint
change, new query, source-script execution, function invocation or approval.
"""
from copy import deepcopy
from datetime import datetime
import re

from evaluation.hybrid_artifact_checks import STAMP
from evaluation.structured_authoring import seal
from network_runtime.contracts import sha256_json

PROFILE = "source-bound-kql-datetime-render/v1"
LITERAL = re.compile(rf"(?:>=|<=|>|<|==|\(|\.\.)\s*(datetime(['\"])({STAMP})\2)", re.I)


def lower(payload, values, *, remaining_changes):
    if type(remaining_changes) is not int or not 0 <= remaining_changes <= 8:
        raise ValueError("explicit remaining shared change budget from zero to eight required")
    candidate = {**deepcopy(payload["candidate"]), "values": deepcopy(values)}
    if sha256_json(candidate) != payload["completeCandidateDigest"]:
        raise ValueError("original candidate/value binding drift")
    draft, edits, skipped = candidate["draft"], [], []
    for span in payload["draftSpans"]:
        content = span["exactQuote"]
        lines = content.splitlines(keepends=True)
        opened = re.fullmatch(r" {0,3}(`{3,}|~{3,})(?:kql|kusto)\s*", lines[0].rstrip("\r\n"), re.I) if lines else None
        if not opened or len(lines) < 2 or not re.fullmatch(rf" {{0,3}}{re.escape(opened[1][0])}{{{len(opened[1])},}}\s*", lines[-1].rstrip("\r\n")):
            continue
        code = "".join(lines[1:-1]).strip()
        # Captured source code is evidence, not an editable model-owned proposal.
        if any(code and code in s["exactQuote"] for s in payload["sourceSpans"] if s["kind"] in {"skill", "observation"}):
            skipped.append({"location": span["draft_span_id"], "reason": "verbatim_source_code_readonly"})
            continue
        offset = span["start"] + len(lines[0])
        for line in lines[1:-1]:
            active = line.split("//", 1)[0]
            # Do not scan data strings or arbitrary language statements. The
            # whole where body must be a conjunction of supported comparisons.
            where = re.fullmatch(r"\s*\|\s*where\s+(.+?)\s*", active, re.I)
            atom = rf"[A-Za-z_]\w*\s*(?:>=|<=|>|<|==)\s*datetime(?:['\"]{STAMP}['\"]|\(\s*{STAMP}\s*\))"
            if where and all(re.fullmatch(atom, part, re.I) for part in re.split(r"\s+and\s+", where[1], flags=re.I)):
                for match in LITERAL.finditer(active):
                    literal, _, stamp = match.groups()
                    anchors = [{"source_span_id": s["source_span_id"], "quote": stamp}
                               for s in payload["sourceSpans"] if s["kind"] in {"task", "observation"} and stamp in s["exactQuote"]]
                    try:
                        parsed = datetime.fromisoformat(stamp)
                        if parsed.isoformat().replace("+00:00", "Z") != stamp or not anchors:
                            raise ValueError("literal not source-bound or round-trippable")
                    except ValueError:
                        skipped.append({"location": span["draft_span_id"], "quote": literal, "reason": "invalid_or_unbound_UTC_literal"})
                        continue
                    start, end = offset + match.start(1), offset + match.end(1)
                    if draft[start:end] != literal:
                        raise ValueError("host literal location drift")
                    edits.append({"location": span["draft_span_id"], "start": start, "end": end,
                        "before": literal, "after": "datetime(" + stamp + ")", "typedValue": {"type": "datetime_utc", "value": stamp},
                        "sourceAnchors": anchors, "valueAndOperatorsPreserved": True})
            offset += len(line)
    if len(edits) > remaining_changes:
        raise ValueError("literal lowering exceeds remaining shared change budget; atomic no-apply")
    for edit in reversed(edits):
        draft = draft[:edit["start"]] + edit["after"] + draft[edit["end"]:]
    candidate["draft"] = draft
    return candidate, seal({"profile": PROFILE, "priorCandidateDigest": payload["completeCandidateDigest"],
        "candidateDigest": sha256_json(candidate), "edits": edits, "skipped": skipped,
        "remainingChangeBudgetBefore": remaining_changes, "remainingChangeBudgetAfter": remaining_changes - len(edits),
        "modelCalls": 0, "sourceScriptsExecuted": False, "queryExecuted": False,
        "semanticApproval": False, "fullQueryValidityProven": False, "completeAnswerApproved": False})
