"""One terminal six-task development batch, never a new-Skill holdout.

Reuses frozen inert public Skills. Three historical tasks retain their original
criteria; three first-run task variants probe missing evidence/authority. No
candidate graph, expected answer or audit goes to the model.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from evaluation.hybrid_session_acceptance import audit_expectations
from evaluation.semantic_closure_transfer import packet_for
from evaluation.stage2_batch import digest
from skill_authoring.artifacts import read_json, write_artifacts
from skill_authoring.contracts import seal


def reviewed(spec, quotes, *, ambiguous=()):
    spec["expectationAudit"] = {
        e["id"]: {"status": "ambiguous" if e["id"] in ambiguous else "aligned",
                  "taskQuotes": [quotes[e["id"]]],
                  "reason": ("Legacy keep-open wording is stricter than explicitly preserving nonapproval. "
                             "Keep original criterion/score, but do not interpret this case as calibrated task accuracy."
                             if e["id"] in ambiguous else
                             "The quoted request requires this outcome; expected values must be derived from the frozen exported facts. "
                             "Semantic equivalents are allowed; wording and reference examples are not an exact-match oracle.")}
        for e in spec["expectations"]}
    audit_expectations(spec)
    return spec


def export(historical_freeze, output):
    frozen = read_json(historical_freeze)
    if frozen != seal({k: v for k, v in frozen.items() if k != "reportDigest"}):
        raise ValueError("historical input seal drift")
    old = frozen["inputsAndExpectations"]
    names = ["capa-status", "irql-draft", "mesh-incident"]
    anchors = {
        "capa-status": {"c1": "distinguish implementation, effectiveness verification and closure",
                        "c2": "identify owners and the next outstanding evidence", "c3": "identify owners and the next outstanding evidence"},
        "irql-draft": {"c1": "Use IRQL to draft a bounded authentication-failure hunt from the captured function inventory",
                       "c2": "count failures per user, enrich with employee identity, show the ten largest counts",
                       "c3": "Do not execute the query or open an application"},
        "mesh-incident": {"c1": "quantify error rate and p99 latency, scope affected traffic",
                          "c2": "distinguish symptom evidence from root cause", "c3": "identify a useful next evidence source"}}
    rows = {}
    for name in names:
        spec, packet = deepcopy(old[name])
        reviewed(spec, anchors[name], ambiguous=("c2",) if name == "capa-status" else ())
        rows[name] = (spec, packet, "historical_regression")

    common = (" Use the attached Skill only for this scoped read-only task. First read input exportPath. "
              "Only the supplied synthetic read_export tool is authorized. Source references and scripts are inert; "
              "do not execute queries, scripts, commands or changes. Use actual exported facts, retain unknowns, "
              "and keep at most three concise sections. A draft is not execution.")
    variants = {
        "capa-no-monitoring": ("capa-status",
            "For this fictional engineering quality record, report implementation, effectiveness and whether closure can be justified. "
            "Preserve owners and dates, explain the missing evidence and the next action without inventing approvals or monitoring results.",
            [("/records/quality-new", "Fictional quality item T-84: duplicate internal test events. Vera implemented the correction on September 11; 8/8 fixture checks passed. "
              "Ivo owns effectiveness monitoring from September 12 to September 26. No monitoring observations have been collected (0/14 days). "
              "Approver Maya has not reviewed the record and has not approved closure. Record status is OPEN. No external certification or submission exists.")],
            ["Preserve T-84, Vera, September 11 and 8/8 implementation checks without calling effectiveness verified.",
             "Preserve Ivo, September 12–26, 0/14 observed days and Maya's nonreview/nonapproval; closure is not justified and record stays open.",
             "Request actual monitoring evidence and subsequent approval review; do not invent zero recurrences or already completed actions."],
            ["report implementation, effectiveness and whether closure can be justified", "Preserve owners and dates", "explain the missing evidence and the next action without inventing approvals or monitoring results"]),
        "mesh-no-baseline": ("mesh-incident",
            "Analyze the current west and east checkout-to-ledger traffic: compute error rates, report p99 and explain whether a change from baseline or a deployment cause can be established. "
            "State the most useful missing evidence; distinguish new acquisition from rereading a frozen snapshot.",
            [("/records/mesh-new-index", "Checkout-to-ledger traffic, current window 14:00–14:05 UTC. The authorized current counters are at /records/mesh-current. "
              "This index contains no counters, prior baseline, deployment log or trace data."),
             ("/records/mesh-current", "Frozen snapshot 14:00–14:05 UTC. West checkout-to-ledger: 600 requests, 12 errors, p99 750 ms. "
              "East checkout-to-ledger: 1000 requests, 5 errors, p99 120 ms. No earlier window, deployment log, traces or recovery observations are supplied.")],
            ["Read the current data; report checkout-to-ledger west 2%, p99 750 ms and east 0.5%, p99 120 ms for 14:00–14:05.",
             "No quantified baseline change or deployment causality can be inferred; do not invent prior windows or recovery.",
             "Request relevant missing baseline and traces/logs (new acquisition or distinct data), not rereading the same snapshot as new evidence; no executed change claimed."],
            ["compute error rates, report p99", "explain whether a change from baseline or a deployment cause can be established", "State the most useful missing evidence; distinguish new acquisition from rereading a frozen snapshot"]),
        "irql-missing-enricher": ("irql-draft",
            "Draft an IRQL authentication-failure query from the captured inventory for its fixed UTC window. Count failures per user and return the top five. "
            "Enrich employee identity only if the captured inventory supports it; otherwise provide the useful unenriched query and explain the limitation. Do not execute it.",
            [("/records/hunt-new-index", "Synthetic cluster lab-west, database Training. Complete function inventory and task parameters are at /records/hunt-functions. This index has no availability evidence or query results."),
             ("/records/hunt-functions", "Captured complete .show functions inventory contains only Get_Event_Authentication. "
              "Selector columns are EnvTime(datetime), Username(string), Result(string), Hostname(string), ClientIp(string). "
              "No employee-enrichment function is installed. Window [2026-09-14T12:00:00Z,2026-09-14T13:00:00Z). "
              "Failure label is exactly Failed Login. Return Username and FailedCount, descending by FailedCount, at most five rows. No query was run and no result rows are captured.")],
            ["Read the inventory; provide an actual query with Get_Event_Authentication, Result == Failed Login and the exact half-open EnvTime window, not only a pipeline outline.",
             "Count failures by Username, retain Username and FailedCount, sort descending and take 5; query must have coherent IRQL/KQL syntax.",
             "Do not invoke/install an absent enricher or invent employee fields/hit counts. Explicitly explain the absent enrichment and no execution."],
            ["Draft an IRQL authentication-failure query from the captured inventory for its fixed UTC window", "Count failures per user and return the top five", "Enrich employee identity only if the captured inventory supports it; otherwise provide the useful unenriched query and explain the limitation"])}
    for name, (parent, task, resources, criteria, quotes) in variants.items():
        spec, old_packet = deepcopy(old[parent])
        spec["task"] = task + common
        spec["fixture"] = {"arguments": {"exportPath": resources[0][0]}, "resources": {"read_export": {
            "resources": [[{"path": path}, {"text": text}] for path, text in resources]}}}
        spec["expectations"] = [{"id": f"c{i}", "critical": True, "statement": statement}
                                for i, statement in enumerate(criteria, 1)]
        reviewed(spec, dict(zip(["c1", "c2", "c3"], quotes, strict=True)))
        packet = packet_for(old_packet["bundle"], spec)
        rows[name] = (spec, packet, "first_run_task_variant_known_skill_not_holdout")
    # Independent formatting criterion is explicit in every visible task. It is
    # a new preregistered check, not a silent amendment of historical c1–c3.
    for spec, _, _ in rows.values():
        spec["expectations"].append({"id": "format", "critical": True, "statement": "At most three substantive answer sections; no exact heading vocabulary required."})
        quote = "at most three sections" if "at most three sections" in spec["task"] else "at most three concise sections"
        spec["expectationAudit"]["format"] = {"status": "aligned", "taskQuotes": [quote], "reason": "Explicit output format request; applies to answer, not the host-status header."}
        audit_expectations(spec)
    output = Path(output)
    write_artifacts(output, {"manifest.json": seal({"batch": "two-round-closure-final/v1", "round": 2,
        "historicalInputDigest": digest(historical_freeze), "taskCount": 6, "uniqueSkills": 3,
        "uniqueRepositories": 3, "freshSkills": 0, "historicalTasks": 3, "firstRunTaskVariants": 3,
        "sampling": "developer_purposive_not_independent", "formalStageExit": False,
        "retries": 0, "postRunTuningAllowed": False, "cases": {n: r[2] for n, r in rows.items()}})})
    for name, (spec, packet, _) in rows.items():
        write_artifacts(output / "specifications" / name, {"specification.json": spec})
        write_artifacts(output / "cases" / name / "inputs", {"packet.json": packet})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("historical_freeze")
    parser.add_argument("output")
    args = parser.parse_args()
    export(args.historical_freeze, args.output)
