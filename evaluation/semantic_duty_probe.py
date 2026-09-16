"""Bounded known-case duty/local-check/edit diagnostic, not a new-source gate.

python -m evaluation.semantic_duty_probe MANIFEST NEW_OUTPUT [--run]
Manifest: case -> {source, request, taskScope}; historical receipts are verified.
One plan + <=7 focused checks + <=8 owned editors + one final review = <=17.
All model work goes through existing Runtime reason nodes. No source execution,
new business reads, semantic self-admission, retries or historical overwrites.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import tarfile

from evaluation import hybrid_authoring as author, hybrid_artifact_checks as local
from evaluation import hybrid_duty_contract as duty, hybrid_note_cells as notes
from evaluation import hybrid_repair_cells as cells
from evaluation.flow_checkpoint import author_once
from evaluation.flow_model_transport import decode
from evaluation.hybrid_behavior import context
from evaluation.hybrid_draft_loop import REVIEW_CONFIG
from evaluation.hybrid_draft_review import build_review_input
from evaluation.hybrid_draft_slots import editing_slots, apply_slot_revision
from evaluation.hybrid_review_context import import_candidate_context
from evaluation.hybrid_review_views import editor_view
from evaluation.semantic_closure_evidence import collect
from evaluation.semantic_closure_transfer import ROOT, fingerprint
from evaluation.semantic_witness_probe import wire_request
from evaluation.source_ledger import budget
from evaluation.structured_authoring import seal
from evaluation.structured_binding_probe import read_json, write_artifacts
from network_runtime.contracts import sha256_json
from network_runtime.l0.hybrid import GovernedHybridFlow, qualify_hybrid
from network_runtime.l0.hybrid_execution import HostHybridConsent, HostReasoningBinding, ReasoningReply, context_digest, run_hybrid
from network_runtime.l0.structured_schema import validate_data


def phase(folder, specs, pin, *, profile=duty.PROFILE):
    """Frozen phase <=8 reason nodes; a failed binding stops downstream nodes."""
    if not 1 <= len(specs) <= 8:
        raise ValueError("phase requires one to eight reasoning nodes")
    nodes = []
    for i, spec in enumerate(specs):
        config = author.MODEL_CONFIG if spec.get("edit") else REVIEW_CONFIG
        node = cells.reason_node(spec["id"], spec["input"], cells.schema_of(spec["input"]), spec["schema"],
                                 review=not spec.get("edit"), config=config)
        node.update(instructions=spec["system"], binding_id="worker",
                    depends_on=[specs[i - 1]["id"]] if i else [])
        nodes.append(node)
    flow = GovernedHybridFlow.model_validate({"api_version": "netopyu.io/governed-hybrid/v1",
        "source_digest": sha256_json([s["input"] for s in specs]), "task_digest": sha256_json(profile),
        "purpose": "Bounded unverified semantic diagnostic; no business actions", "input_schema": author.obj({}),
        "nodes": nodes, "outputs": [s["id"] for s in specs], "max_parallel": 1, "max_model_calls": len(specs),
        "timeout_seconds": 1800, "failure_policy": "stop_no_downstream"})
    graph = qualify_hybrid(flow, {})
    write_artifacts(folder / "freeze", {"graph.json": graph})
    bound = {}
    by_id = {s["id"]: s for s in specs}

    def invoke(request):
        if fingerprint() != pin:
            raise ValueError("execution source drift before call")
        spec = by_id[request["nodeId"]]
        wire = wire_request(spec["system"], spec["input"], spec["schema"])
        if spec.get("edit"):
            wire["options"] = {k: v for k, v in author.MODEL_CONFIG.items() if k != "think"}
            target = spec["input"]["editTarget"]
            wire["messages"] = [wire["messages"][0],
                {"role": "user", "content": json.dumps({"readOnlyContext": spec["input"]["readOnlyContext"]}, ensure_ascii=False)},
                {"role": "user", "content": json.dumps({"editTarget": target, "requiredOutputSchema": spec["schema"]}, ensure_ascii=False)}]
        measured = budget(wire)
        if not measured["accepted"]:
            write_artifacts(folder / "preflight" / spec["id"], {"report.json": seal({"status": "context_budget_exceeded", "budget": measured})})
            raise ValueError("unchanged input budget exceeded; no truncation")

        def derive(envelope):
            text, cost = decode("ollama", envelope)
            if text is None:
                return {}, cost
            files = {}
            try:
                raw = validate_data(spec["schema"], json.loads(text))
                files["candidate.json"] = raw
                files["bound.json"] = spec["bind"](deepcopy(raw))
                return files, {**cost, "status": "bound_not_semantic_approval"}
            except (ValueError, TypeError) as error:
                files["invalid.json"] = {"text": text, "reason": str(error)[:1600]}
                return files, {**cost, "status": "invalid_schema_or_binding"}

        result = author_once(folder / "model" / spec["id"], {"wireRequest": wire, "governedRequest": request}, derive,
                             max_new_calls=1, label="bounded focused duty diagnostic")
        if fingerprint() != pin:
            raise ValueError("execution source drift after call")
        print(json.dumps({"phase": str(folder.name), "node": spec["id"], "status": result["result"]["status"]}), flush=True)
        if "bound.json" not in result:
            raise ValueError("one attempted call failed; no automatic retry")
        bound[spec["id"]] = result["bound.json"]
        return ReasoningReply(result["candidate.json"], author.MODEL, request["configurationDigest"],
                              result["result"].get("inputTokens"), result["result"].get("outputTokens"))

    # A phase has either reviewers or editors, never mixed configuration digests.
    config = author.MODEL_CONFIG if specs[0].get("edit") else REVIEW_CONFIG
    if any(bool(s.get("edit")) != bool(specs[0].get("edit")) for s in specs):
        raise ValueError("mixed phase model configurations")
    ctx = context()
    execution = run_hybrid(flow, {}, reads={}, read_bindings={}, gates={}, context=ctx,
        consent=HostHybridConsent(graph["graphDigest"], sha256_json({}), context_digest(ctx)),
        reasoners={"worker": HostReasoningBinding(author.MODEL, sha256_json(config), invoke)})
    write_artifacts(folder / "summary", {"report.json": seal({"execution": execution, "bound": bound,
                                                           "semanticApproval": False})})
    if execution["status"] != "governed_graph_completed":
        raise ValueError("bounded phase stopped; inspect retained Runtime trace")
    return bound


def edit_specs(payload, contract, focused, checked_notes, checks):
    """Host-addressed editing candidates; no caller/model-provided arbitrary paths."""
    editable = editing_slots(payload, complete_sections=True)
    units = cells.source_units(payload, editable)
    spans = {s["draft_span_id"]: s for s in payload["draftSpans"]}
    leads = [lead for report in focused for lead in report["issues"]]
    leads += [{"location": row["location"], "quote": row["artifactQuote"], "explanation": row["explanation"],
               "localCheck": row["check"]} for row in checks["checks"] if row["status"] == "fail"]
    specs, owned = [], {}
    # Notes are inspected regardless of earlier body opinions. Explicit unknown
    # atoms can trigger inspection, never automatic deletion or approval.
    for key, row in checked_notes["notes"].items():
        if not row["needsInspection"]:
            continue
        i = int(key[1:])
        slot = {"id": f"u{i:03d}", "index": i, "pointer": f"/candidate/notes/{i}",
                "text": payload["candidate"]["notes"][i], "findings": row["atoms"]}
        supplied = {"sourceContext": duty.source_input(payload), "readOnlyAnswer": payload["candidate"]["draft"],
                    "hostOpenDuties": payload["hostOpenDuties"], "hostDutiesEditable": False}
        specs.append({"id": slot["id"], "edit": True, "system": notes.SYSTEM,
            "input": {"readOnlyContext": supplied, "editTarget": {"ownedNote": slot}}, "schema": notes.schema(payload),
            "bind": lambda raw, s=slot: notes.validate(payload, s, raw)})
        owned[slot["id"]] = {"kind": "note", "slot": slot}
    for slot in editable["slots"]:
        assigned = []
        for lead in leads:
            loc = lead["location"]
            if not loc or (loc in spans and spans[loc]["start"] < slot["end"] and spans[loc]["end"] > slot["start"]):
                assigned.append(lead)
        if not assigned:
            continue
        local_units = [u for u in units if u["suggestedCell"] in {None, slot["id"]}]
        supplied = {"originalTask": payload["originalTask"], "hostOpenDuties": payload["hostOpenDuties"],
            **({"taskScope": payload["taskScope"]} if "taskScope" in payload else {}),
            "sourceSpans": payload["sourceSpans"], "ownedFragment": slot, "repairFocus": [],
            "completePriorDraftReadOnly": payload["candidate"]["draft"], "sourceRelationIndex": [],
            "reportedConcerns": [], "locatedRepairFindings": assigned, "editableLines": cells.lines_for(slot),
            "sourceUnitCatalog": local_units}
        projected = editor_view(supplied)
        targets = {k: projected.pop(k) for k in ("ownedFragment", "editableLines")}
        projected["unverifiedDutyContract"] = contract
        system = cells.GROUNDED_SYSTEM
        if cells.owns_whole_draft(payload, slot):
            system += "\nYou own the complete draft. At most one missing subsection may be added; preserve existing headings and all valid content."
        specs.append({"id": slot["id"], "edit": True, "system": system,
            "input": {"readOnlyContext": projected, "editTarget": targets},
            "schema": cells.grounded_schema(payload, slot, local_units),
            "bind": lambda raw, s=slot, u=local_units: cells.validate_cell_proposal(payload, s, u, raw, "grounded_patch")})
        owned[slot["id"]] = {"kind": "body", "slot": slot}
    deferred = [s["id"] for s in specs[8:]]
    return specs[:8], owned, editable, deferred


def materialize(payload, values, editable, owned, outputs):
    edits = {s["id"]: {"action": "keep", "replacement": "", "source_span_ids": [], "rationale": ""}
             for s in editable["slots"]}
    note_edits = []
    for key, row in outputs.items():
        if owned[key]["kind"] == "note":
            note_edits.append(row)
        else:
            edits[key] = row
    proposal = {"slots_digest": editable["slotsDigest"], "slots": edits, "notes": payload["candidate"]["notes"],
                "revision_note": "One focused pass, unverified semantics; original duties retained."}
    candidate, application = apply_slot_revision(payload, values, proposal,
                                                  complete_sections=True, preserve_notes=True)
    candidate["notes"] = notes.materialize(payload["candidate"]["notes"], note_edits)
    if len(application["edits"]) + sum(e["changed"] for e in note_edits) > 8:
        raise ValueError("shared eight-change budget exceeded")
    return candidate, seal({"bodyApplication": application, "noteEdits": note_edits,
        "priorCandidateDigest": payload["completeCandidateDigest"], "candidateDigest": sha256_json(candidate),
        "originalNotes": payload["candidate"]["notes"], "hostOpenDuties": payload["hostOpenDuties"],
        "hostDutiesCleared": False, "completeAnswerApproved": False})


def final_schema(ctx, contract):
    return {**duty.obj({"duties": duty.obj({d["id"]: {"$ref": "#/$defs/check"} for d in contract["duties"]}),
                     "notes": duty.notes_schema(ctx)}), "$defs": {"check": duty.check_schema(ctx)}}


def check_view(report):
    """Remove duplicated code from check metadata, never candidate/source text."""
    return {**report, "checks": [{k: v for k, v in row.items() if k != "artifactQuote" or row["status"] == "fail"}
                                 for row in report["checks"]],
            "reportDigestBindsFullAuditNotThisProjection": True,
            "artifactTextLocation": "Complete unchanged text is in candidateLocations; metadata is not new evidence."}


def bind_final(ctx, contract, raw):
    value = validate_data(final_schema(ctx, contract), raw)
    return seal({"duties": {d["id"]: duty.bind_check(ctx, d, value["duties"][d["id"]]) for d in contract["duties"]},
                 "notes": duty.bind_notes(ctx, value["notes"]), "semanticApproval": False})


def case_run(output, inputs, pin):
    payload = build_review_input(inputs)
    source = duty.source_input(payload)
    plan = phase(output / "plan", [{"id": "plan", "system": duty.PLAN_SYSTEM, "input": source,
        "schema": duty.plan_schema(), "bind": lambda raw: duty.bind_plan(source, raw)}], pin)["plan"]
    ctx = duty.context(payload, plan)
    checks = local.inspect(payload)
    write_artifacts(output / "before", {"local-checks.json": checks, "review-input.json": payload})
    specs = [{"id": d["id"], "system": duty.CHECK_SYSTEM,
        "input": {**ctx, "assignedDuty": d, "localChecks": check_view(checks)}, "schema": duty.check_schema(ctx),
        "bind": lambda raw, d=d: duty.bind_check(ctx, d, raw)} for d in plan["duties"]]
    specs.append({"id": "notes", "system": duty.NOTES_SYSTEM, "input": ctx,
                  "schema": duty.notes_schema(ctx), "bind": lambda raw: duty.bind_notes(ctx, raw)})
    reviewed = phase(output / "focused", specs, pin)
    specs, owned, editable, deferred = edit_specs(payload, plan, [reviewed[d["id"]] for d in plan["duties"]], reviewed["notes"], checks)
    write_artifacts(output / "editing-plan", {"report.json": seal({"selected": [s["id"] for s in specs],
        "deferred": deferred, "owned": owned, "contractOverflow": plan["overflow"], "authorityGranted": False})})
    outputs = phase(output / "edit", specs, pin) if specs else {}
    candidate, application = materialize(payload, inputs["candidate"]["values"], editable, owned, outputs)
    write_artifacts(output / "materialized", {"candidate.json": candidate, "application.json": application})
    final_payload = build_review_input({**inputs, "candidate": candidate})
    after = local.inspect(final_payload)
    final_ctx = duty.context(final_payload, plan)
    final = phase(output / "final", [{"id": "review", "system": duty.CHECK_SYSTEM + "\n" + duty.NOTES_SYSTEM +
        "\nFinal review: inspect each keyed duty and every note on the revised artifact. Previous opinions are NOT supplied or evidence.",
        "input": {**final_ctx, "localChecks": check_view(after)}, "schema": final_schema(final_ctx, plan),
        "bind": lambda raw: bind_final(final_ctx, plan, raw)}], pin)["review"]
    report = seal({"candidateDigest": sha256_json(candidate), "contract": plan, "final": final,
        "localBefore": checks, "localAfter": after, "deferred": deferred,
        "knownDevelopmentOnly": True, "semanticSuccess": None, "completeAnswerApproved": False,
        "stageExitMet": False, "sourceScriptCalls": 0, "businessReadCalls": 0, "effectCalls": 0})
    write_artifacts(output / "summary", {"report.json": report})
    return report


def run(manifest, output, *, live=False, case_runner=None, profile=duty.PROFILE, phase_caps=None, preflight_fn=None):
    jobs = read_json(manifest)
    if (not isinstance(jobs, dict) or not 1 <= len(jobs) <= 2
            or any(not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", key) for key in jobs)):
        raise ValueError("one or two explicitly declared known regression tasks only")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("preserve checkpoints and all prior runs; new output required")
    inputs, provenance = {}, {}
    for key, job in jobs.items():
        if set(job) != {"source", "request", "taskScope"}:
            raise ValueError("manifest allows source/request/lossless taskScope only; no expected answers")
        if any(output.is_relative_to(Path(job[k]).resolve()) for k in ("source", "request")):
            raise ValueError("new output must be outside historical evidence")
        _, provenance[key], inputs[key], _ = import_candidate_context(job["source"], job["request"], job["taskScope"])
    pin = fingerprint()
    freeze = seal({"profile": profile, "implementation": pin, "jobs": jobs, "inputs": inputs, "provenance": provenance,
        "live": live, "maxCallsPerTask": 17 if live else 0,
        "phaseCaps": phase_caps or {"plan": 1, "focusedIncludingNotes": 7, "sharedBodyNotesEdit": 8, "final": 1},
        "model": author.MODEL, "reviewConfig": REVIEW_CONFIG, "editorConfig": author.MODEL_CONFIG,
        "newBusinessReads": 0, "effects": 0, "knownDevelopmentOnly": True, "independentGold": False})
    write_artifacts(output / "freeze", {"manifest.json": freeze})
    with tarfile.open(output / "freeze/source-snapshot.tar.gz", "x:gz") as archive:
        for name in pin:
            archive.add(ROOT / name, arcname=name, recursive=False)
    rows = []
    for key, supplied in inputs.items():
        try:
            if not live:
                source = duty.source_input(build_review_input(supplied))
                details = preflight_fn(supplied) if preflight_fn else {
                    "budget": budget(wire_request(duty.PLAN_SYSTEM, source, duty.plan_schema()))}
                row = {"case": key, "status": "preflight_only", **details}
            else:
                report = (case_runner or case_run)(output / key, supplied, pin)
                row = {"case": key, "status": "bounded_diagnostic_completed", "caseReportDigest": report["reportDigest"]}
                if "allScheduledNodesBound" in report:
                    row.update(allScheduledNodesBound=report["allScheduledNodesBound"], failedNodeCount=len(report["failures"]))
        except (ValueError, OSError) as error:
            row = {"case": key, "status": "stopped_without_retry", "error": str(error)[:1200]}
        rows.append(row)
        write_artifacts(output / key / "checkpoint", {"report.json": seal(row)})
        print(json.dumps(row), flush=True)
    if fingerprint() != pin:
        raise ValueError("source drift at completion")
    evidence = collect(output)
    write_artifacts(output / "evidence", {"report.json": evidence})
    summary = seal({"freezeDigest": freeze["reportDigest"], "rows": rows, "evidenceDigest": evidence["reportDigest"],
        **{k: evidence[k] for k in ("actualChatAttempts", "inputTokens", "outputTokens", "callsWithUnknownInputUsage",
            "callsWithUnknownOutputUsage", "p50CallLatencyMs", "p95CallLatencyMs")},
        "historicalGradesChanged": False, "stageExitMet": False, "completeAnswerApproved": False})
    write_artifacts(output / "summary", {"report.json": summary})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.manifest, args.output, live=args.run)))
