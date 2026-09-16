"""Explicit known-case host-anchor/predicate/independent-edit diagnostic.

python -m evaluation.semantic_typed_duty_probe MANIFEST NEW_OUTPUT [--run]
No new cohort, source execution, business read, retry or admission. At most 16
calls within the previously approved 17 cap: 3 task reviews, 2 predicate passes
before/after (4), 8 owned edits, 1 final task review. Each phase <=8 nodes.
"""
import argparse
from copy import deepcopy
import json

from evaluation import hybrid_duty_contract as legacy
from evaluation import hybrid_typed_duties as typed, hybrid_predicate_review as predicates
from evaluation import hybrid_artifact_checks as local, hybrid_artifact_lowering as lowering
from evaluation import hybrid_note_cells as notes
from evaluation import semantic_duty_probe as probe
from evaluation.hybrid_draft_review import build_review_input
from evaluation.semantic_closure_transfer import fingerprint
from evaluation.semantic_witness_probe import wire_request
from evaluation.source_ledger import budget
from evaluation.structured_authoring import seal
from evaluation.structured_binding_probe import write_artifacts
from network_runtime.contracts import sha256_json

PROFILE = "host-typed-duty-diagnostic/v1"
PHASE_CAPS = {"hostContractModelCalls": 0, "taskChecks": 3, "predicatePassesBefore": 2,
              "sharedBodyNotesEdit": 8, "predicatePassesAfter": 2, "finalTaskCheck": 1}


def independent(folder, specs, pin, failures, *, profile=PROFILE):
    """Independent immutable reason inputs, not recovery from a dependency failure.

Each owns a separate one-node Runtime graph. No failed node is retried or used
as evidence. Drift still aborts all siblings. This is not a general DAG resume.
"""
    if len(specs) > 8 or len({s["id"] for s in specs}) != len(specs):
        raise ValueError("independent batch requires <=8 unique owned nodes")
    outputs = {}
    for spec in specs:
        try:
            outputs.update(probe.phase(folder / spec["id"], [spec], pin, profile=profile))
        except ValueError as error:
            if fingerprint() != pin:
                raise ValueError("source drift; independent siblings must stop") from error
            # Do not hide errors outside an actually retained failed Runtime phase.
            if not (folder / spec["id"] / "summary/report.json").is_file():
                raise
            failures.append({"phase": folder.name, "node": spec["id"], "error": str(error)[:1200],
                             "retried": False, "outputApplied": False})
    return outputs


def note_pass(folder, payload, pin, failures):
    supplied = predicates.extract_input(payload)
    if not supplied["notes"]:
        return seal({"notes": {}, "semanticApproval": False, "modelCalls": 0})
    spec = {"id": "extract", "system": predicates.EXTRACT_SYSTEM, "input": supplied,
            "schema": predicates.extract_schema(supplied), "bind": lambda raw: predicates.bind_extract(supplied, raw)}
    extracted = independent(folder, [spec], pin, failures).get("extract")
    if extracted is not None:
        ctx = predicates.review_input(payload, extracted)
        spec = {"id": "evidence", "system": predicates.REVIEW_SYSTEM, "input": ctx,
            "schema": predicates.review_schema(ctx), "bind": lambda raw: predicates.bind_review(payload, extracted, raw)}
        checked = independent(folder, [spec], pin, failures).get("evidence")
        if checked is not None:
            return checked
    return seal({"notes": {key: {"atoms": [], "needsInspection": True, "allPredicatesEnumeratedProven": False,
                                   "reviewUnavailable": True} for key in supplied["notes"]},
                 "semanticApproval": False, "failedReviewDoesNotEstablishDefect": True})


def task_specs(payload, plan, checks):
    ctx = typed.context(payload, plan)
    return [{"id": d["id"], "system": typed.CHECK_SYSTEM,
        "input": {**ctx, "assignedDuty": d, "localChecks": probe.check_view(checks)},
        "schema": legacy.check_schema(ctx), "bind": lambda raw, d=d: typed.bind_check(ctx, d, raw)} for d in plan["duties"]]


def preflight(inputs):
    payload = build_review_input(inputs)
    plan = typed.contract(payload)
    specs = task_specs(payload, plan, local.inspect(payload))
    note_input = predicates.extract_input(payload)
    return {"hostContractDigest": plan["reportDigest"], "taskAnchorCount": len(plan["taskAnchors"]),
        "hostContractModelCalls": 0, "maxActualModelCalls": sum(PHASE_CAPS.values()),
        "taskCheckBudgets": {s["id"]: budget(wire_request(s["system"], s["input"], s["schema"])) for s in specs},
        "noteExtractionBudget": budget(wire_request(predicates.EXTRACT_SYSTEM, note_input, predicates.extract_schema(note_input)))}


def case_run(output, inputs, pin):
    payload = build_review_input(inputs)
    plan = typed.contract(payload)
    if len(plan["duties"]) > PHASE_CAPS["taskChecks"]:
        raise ValueError("task contract exceeds unchanged review budget; no silent truncation")
    before = local.inspect(payload)
    write_artifacts(output / "before", {"contract.json": plan, "review-input.json": payload, "local-checks.json": before})
    failures = []
    focused = independent(output / "focused", task_specs(payload, plan, before), pin, failures)
    checked_notes = note_pass(output / "notes-before", payload, pin, failures)
    specs, owned, editable, deferred = probe.edit_specs(payload, plan, list(focused.values()), checked_notes, before)
    write_artifacts(output / "editing-plan", {"report.json": seal({"selected": [s["id"] for s in specs],
        "deferred": deferred, "owned": owned, "independentInputs": True, "authorityGranted": False})})
    outputs = independent(output / "edit", specs, pin, failures)
    candidate, application = probe.materialize(payload, inputs["candidate"]["values"], editable, owned, outputs)
    used = len(application["bodyApplication"]["edits"]) + sum(e["changed"] for e in application["noteEdits"])
    intermediate = build_review_input({**inputs, "candidate": candidate})
    lowered, lowering_report = lowering.lower(intermediate, candidate["values"], remaining_changes=8 - used)
    write_artifacts(output / "materialized", {"before-lowering.json": candidate, "candidate.json": lowered,
        "application.json": application, "lowering.json": lowering_report})
    final_payload = build_review_input({**inputs, "candidate": lowered})
    after = local.inspect(final_payload)
    final_notes = note_pass(output / "notes-after", final_payload, pin, failures)
    ctx = typed.context(final_payload, plan)
    schema = {**legacy.obj({d["id"]: {"$ref": "#/$defs/check"} for d in plan["duties"]}),
              "$defs": {"check": legacy.check_schema(ctx)}}
    spec = {"id": "review", "system": typed.CHECK_SYSTEM +
        "\nInspect each host-keyed exact task group on the final artifact. Prior opinions are not supplied. No complete-task approval.",
        "input": {**ctx, "localChecks": probe.check_view(after)}, "schema": schema,
        "bind": lambda raw: seal({"duties": {d["id"]: typed.bind_check(ctx, d, raw[d["id"]]) for d in plan["duties"]},
                                  "semanticApproval": False})}
    final = independent(output / "final", [spec], pin, failures)
    report = seal({"profile": PROFILE, "contract": plan, "candidateDigest": sha256_json(lowered),
        "notesBefore": checked_notes, "notesAfter": final_notes, "final": final, "failures": failures,
        "localBefore": before, "localAfter": after, "loweringDigest": lowering_report["reportDigest"],
        "deferred": deferred, "allScheduledNodesBound": not failures, "knownDevelopmentOnly": True,
        "completeAnswerApproved": False, "semanticSuccess": None, "stageExitMet": False,
        "newBusinessReadCalls": 0, "effectCalls": 0, "sourceScriptCalls": 0})
    write_artifacts(output / "summary", {"report.json": report})
    return report


EVIDENCE_PROFILE = "evidence-first-notes-diagnostic/v1"
EVIDENCE_CAPS = {"predicateExtraction": 1, "evidenceLocation": 1, "evidenceComparison": 1, "ownedNoteProjection": 8}


def evidence_preflight(inputs):
    payload = build_review_input(inputs)
    supplied = predicates.extract_input(payload)
    catalog = predicates.evidence_catalog(payload)
    return {"mode": EVIDENCE_PROFILE, "maxActualModelCalls": 3 + min(8, len(supplied["notes"])),
        "evidenceWindowCount": len(catalog["units"]), "catalogDigest": catalog["reportDigest"],
        "sourceScriptsExecuted": False, "bodyEditingEnabled": False,
        "budget": budget(wire_request(predicates.EXTRACT_SYSTEM, supplied, predicates.extract_schema(supplied)))}


def evidence_case_run(output, inputs, pin):
    """Three evidence passes, then <=8 quote-only note projections; body untouched.

No fresh task review or query-repair retries. Replacing a caveat by a source
view is explicitly not counted as successful reasoning or task fulfillment.
"""
    payload = build_review_input(inputs)
    catalog = predicates.evidence_catalog(payload)
    write_artifacts(output / "before", {"review-input.json": payload, "catalog.json": catalog})
    failures = []

    def one(name, system, supplied, schema, binder):
        result = independent(output / name, [{"id": name, "input": supplied, "schema": schema,
            "system": system, "bind": binder}], pin, failures, profile=EVIDENCE_PROFILE)
        if name not in result:
            raise ValueError(f"required {name} failed; dependent phases not scheduled, no retry")
        return result[name]

    supplied = predicates.extract_input(payload)
    extraction = one("extract", predicates.EXTRACT_SYSTEM, supplied, predicates.extract_schema(supplied),
                     lambda raw: predicates.bind_extract(supplied, raw))
    supplied = predicates.locate_input(payload, extraction)
    located = one("locate", predicates.LOCATE_SYSTEM, predicates.locate_view(supplied), predicates.locate_schema(supplied),
                  lambda raw: predicates.bind_locate(payload, extraction, raw))
    supplied = predicates.compare_input(payload, extraction, located)
    if any(located["selected"].values()):
        checked = one("compare", predicates.COMPARE_SYSTEM, supplied, predicates.compare_schema(supplied),
                      lambda raw: predicates.bind_compare(payload, extraction, located, raw))
    else:
        checked = predicates.bind_compare(payload, extraction, located, {})
        write_artifacts(output / "compare-no-call", {"report.json": checked})
    specs, original_inputs = [], {}
    # Every note is considered, even when a fallible reviewer is positive. The
    # projection selector may keep it; no semantic verdict authorizes deletion.
    for key, row in checked["notes"].items():
        selected = list(dict.fromkeys(eid for atom in row["atoms"] for eid in atom["evidenceIds"]
                                      if catalog["units"][eid]["kind"] == "observation"))
        supplied = notes.projection_input(payload, int(key[1:]), selected)
        original_inputs[key] = supplied
        specs.append({"id": key, "edit": True, "system": notes.PROJECTION_SYSTEM,
            "input": {"readOnlyContext": {k: v for k, v in supplied.items() if k != "ownedNote"},
                      "editTarget": {"ownedNote": supplied["ownedNote"]}}, "schema": notes.projection_schema(supplied),
            "bind": lambda raw, supplied=supplied: notes.validate_projection(payload, supplied, raw)})
    deferred = [s["id"] for s in specs[8:]]
    write_artifacts(output / "projection-plan", {"report.json": seal({"inputs": original_inputs,
        "selected": [s["id"] for s in specs[:8]], "deferred": deferred, "maxSharedChanges": 8})})
    projected = independent(output / "project", specs[:8], pin, failures, profile=EVIDENCE_PROFILE)
    candidate = deepcopy(inputs["candidate"])
    candidate["notes"] = notes.materialize(candidate["notes"], list(projected.values()))
    # Recompute host renderings after materialization; no model can change the
    # quotation in a second phase or convert it into a verified business fact.
    for key, value in projected.items():
        if notes.validate_projection(payload, original_inputs[key], value["proposal"]) != value:
            raise ValueError("post-projection evidence/rendering drift")
    report = seal({"profile": EVIDENCE_PROFILE, "candidateDigest": sha256_json(candidate),
        "priorCandidateDigest": payload["completeCandidateDigest"], "comparison": checked,
        "projections": projected, "failures": failures, "deferred": deferred,
        "allScheduledNodesBound": not failures, "bodyUnchanged": candidate["draft"] == inputs["candidate"]["draft"],
        "valuesUnchanged": candidate["values"] == inputs["candidate"]["values"],
        "originalNotes": payload["candidate"]["notes"], "hostOpenDuties": payload["hostOpenDuties"],
        "hostDutiesCleared": False, "freeTextNoteEdits": 0, "newBusinessReadCalls": 0,
        "sourceScriptCalls": 0, "effectCalls": 0, "semanticSuccess": None,
        "knownDevelopmentOnly": True, "sourceProjectionIsNotReasoningSuccess": True,
        "completeAnswerApproved": False, "stageExitMet": False})
    write_artifacts(output / "materialized", {"candidate.json": candidate})
    write_artifacts(output / "summary", {"report.json": report})
    return report


def run(manifest, output, *, live=False, evidence_first_notes=False):
    return probe.run(manifest, output, live=live, case_runner=evidence_case_run if evidence_first_notes else case_run,
        profile=EVIDENCE_PROFILE if evidence_first_notes else PROFILE,
        phase_caps=EVIDENCE_CAPS if evidence_first_notes else PHASE_CAPS,
        preflight_fn=evidence_preflight if evidence_first_notes else preflight)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--evidence-first-notes", action="store_true", help="Opt-in evidence location/comparison and quote-only notes; no body editing")
    args = parser.parse_args()
    print(json.dumps(run(args.manifest, args.output, live=args.run, evidence_first_notes=args.evidence_first_notes)))
