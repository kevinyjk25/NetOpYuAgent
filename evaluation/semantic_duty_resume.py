"""Explicit new evidence for unattempted, source-independent diagnostic siblings.

Never retries a model node or regrades its old phase. A failed duty plan cannot
be used, but original-source note review needs no inferred contract. A failed
note blocks its own edit; an unattempted body cell with immutable source input
can run in a NEW graph. Missing duties/failed edits remain explicitly open.
"""
import argparse
import json
from pathlib import Path
import tarfile

from evaluation import semantic_duty_probe as probe, hybrid_duty_contract as duty
from evaluation.flow_tree_authoring import verify_receipt
from evaluation.semantic_closure_evidence import collect
from evaluation.structured_authoring import seal
from evaluation.structured_binding_probe import read_json, write_artifacts
from network_runtime.contracts import sha256_json


def note_input(ctx):
    """Full source retained; only notes are targets, answer is read-only context."""
    return {"sourceContext": ctx["sourceContext"], "readOnlyAnswer": ctx["completeCandidate"]["draft"],
        "candidateDigest": ctx["candidateDigest"], "notesToInspect": {
            k: v for k, v in ctx["candidateLocations"].items() if k.startswith("n")},
        "assignment": "Inspect ONLY these keyed notes, not paragraphs from the read-only answer. Use the exact note key for each quoted atom."}


def admitted_old_output(folder, spec):
    """Recheck an actually successful receipt; never re-derive a failed reply."""
    verify_receipt(folder)
    request = read_json(folder / "request.json")["governedRequest"]
    if request["inputs"] != spec["input"] or request["outputSchema"] != spec["schema"] or request["instructions"] != spec["system"]:
        raise ValueError("original successful node no longer matches its frozen contract")
    if not (folder / "bound.json").exists():
        return None
    bound = spec["bind"](read_json(folder / "candidate.json"))
    if bound != read_json(folder / "bound.json"):
        raise ValueError("successful node derivation drift")
    return bound


def prepare_case(parent, name, inputs):
    root = parent / name
    payload = probe.build_review_input(inputs)
    source = duty.source_input(payload)
    plan_dir = root / "plan/model/plan"
    verify_receipt(plan_dir)
    if read_json(plan_dir / "request.json")["governedRequest"]["inputs"] != source:
        raise ValueError("parent source input differs from the actual model request")
    plan_valid = (plan_dir / "bound.json").exists()
    state = {"priorAttempts": len(list(root.rglob("request.json"))), "failedNodesNotRetried": [],
             "retainedOutputs": {}, "unresolved": [], "newFocusedCalls": 0}
    if plan_valid:
        plan = duty.bind_plan(source, read_json(plan_dir / "candidate.json"))
        if plan != read_json(plan_dir / "bound.json"):
            raise ValueError("original contract drift")
        reviewed = read_json(root / "focused/summary/report.json")
        duty.verify(reviewed)
        if reviewed["execution"]["status"] != "governed_graph_completed":
            raise ValueError("partial focused phase is not supported by this recovery")
        state.update(plan=plan, reviewed=reviewed["bound"])
    else:
        # No old malformed row is repaired, promoted or silently discarded.
        # This is a host marker for absent duties, NOT a translated contract.
        state["failedNodesNotRetried"].append("plan/plan")
        state["unresolved"].append("Duty extraction failed; no business-duty review or complete-answer conclusion is available.")
        state["plan"] = seal({"profile": duty.PROFILE, "sourceDigest": source["reportDigest"],
            "duties": [], "overflow": True, "unrepresentedDuties": "Entire failed duty contract quarantined.",
            "completenessProven": False, "authorityGranted": False, "hostAbsenceMarkerNotModelContract": True})
        if (root / "focused").exists():
            raise ValueError("failed-plan path must not have prior focused calls")
        state["newFocusedCalls"] = 1
    return payload, state


def run(parent, output, *, live=False):
    parent, output = Path(parent).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(parent):
        raise FileExistsError("new recovery evidence outside the original run required")
    # Reject in-flight, tampered or unmatched source/call receipts before work.
    parent_evidence = collect(parent)
    original = read_json(parent / "freeze/manifest.json")
    duty.verify(original)
    if original["profile"] != duty.PROFILE or not original["live"] or original["maxCallsPerTask"] != 17:
        raise ValueError("explicit original two-case 17-call diagnostic required")
    if not 1 <= len(original["inputs"]) <= 2:
        raise ValueError("recovery cannot expand cases")
    prepared = {}
    for name, inputs in original["inputs"].items():
        checkpoint = read_json(parent / name / "checkpoint/report.json")
        duty.verify(checkpoint)
        if checkpoint["status"] != "stopped_without_retry":
            raise ValueError("do not rerun a completed case")
        prepared[name] = prepare_case(parent, name, inputs)
    if live:
        # A sibling claim is outside immutable parent evidence. It is deliberately
        # never auto-deleted: another output name cannot reset the call budget.
        claim = parent.parent / (parent.name + "-resume-claim")
        write_artifacts(claim, {"claim.json": seal({"parentFreezeDigest": original["reportDigest"],
            "output": str(output), "maxCumulativeCallsPerTask": 17, "retriesAuthorized": False})})
    pin = probe.fingerprint()
    freeze = seal({"profile": "unattempted-duty-siblings/v1", "parent": str(parent), "parentFreezeDigest": original["reportDigest"],
        "parentEvidence": parent_evidence, "implementation": pin, "live": live,
        "maxCumulativeCallsPerTask": 17, "failedCallsRetried": False, "historicalGradesChanged": False})
    write_artifacts(output / "freeze", {"manifest.json": freeze})
    with tarfile.open(output / "freeze/source-snapshot.tar.gz", "x:gz") as archive:
        for name in pin:
            archive.add(probe.ROOT / name, arcname=name, recursive=False)
    rows = []
    for name, (payload, state) in prepared.items():
        folder, old = output / name, parent / name
        try:
            ctx = duty.context(payload, state["plan"])
            if not live:
                rows.append({"case": name, "status": "preflight_only", "priorAttempts": state["priorAttempts"]})
                continue
            if "reviewed" not in state:
                if state["priorAttempts"] + 1 + 8 + 1 > 17:
                    raise ValueError("remaining budget cannot cover independent notes path")
                state["reviewed"] = probe.phase(folder / "focused", [{"id": "notes", "system": duty.NOTES_SYSTEM,
                    "input": note_input(ctx), "schema": duty.notes_schema(ctx), "bind": lambda raw: duty.bind_notes(ctx, raw)}], pin)
            focused = [state["reviewed"][d["id"]] for d in state["plan"]["duties"]]
            # Original assignment first, so new local diagnostics cannot invent
            # new edit cells or repeat any attempted old model node.
            old_checks = read_json(old / "before/local-checks.json") if (old / "before/local-checks.json").exists() else probe.local.inspect(payload)
            specs, owned, editable, deferred = probe.edit_specs(payload, state["plan"], focused, state["reviewed"]["notes"], old_checks)
            new_specs = []
            for spec in specs:
                prior = old / "edit/model" / spec["id"]
                if (prior / "request.json").exists():
                    bound = admitted_old_output(prior, spec)
                    if bound is None:
                        state["failedNodesNotRetried"].append("edit/" + spec["id"])
                        state["unresolved"].append(f"Original failed edit {spec['id']} remains unapplied and unreviewed.")
                    else:
                        state["retainedOutputs"][spec["id"]] = bound
                else:
                    spec["input"]["readOnlyContext"]["localChecks"] = probe.check_view(probe.local.inspect(payload))
                    new_specs.append(spec)
            needed = state["priorAttempts"] + state["newFocusedCalls"] + len(new_specs) + 1
            if needed > 17:
                raise ValueError("cumulative model cap exceeded; no further invocation")
            state["unresolved"].extend(f"Deferred edit {key}" for key in deferred)
            write_artifacts(folder / "plan", {"report.json": seal({**state, "newEditCells": [s["id"] for s in new_specs],
                "maxCumulativeCalls": needed, "originalCandidateDigest": payload["completeCandidateDigest"]})})
            edited = probe.phase(folder / "edit", new_specs, pin) if new_specs else {}
            candidate, application = probe.materialize(payload, original["inputs"][name]["candidate"]["values"], editable, owned,
                                                       {**state["retainedOutputs"], **edited})
            write_artifacts(folder / "materialized", {"candidate.json": candidate, "application.json": application})
            after = probe.local.inspect(probe.build_review_input({**original["inputs"][name], "candidate": candidate}))
            final_payload = probe.build_review_input({**original["inputs"][name], "candidate": candidate})
            final_ctx = duty.context(final_payload, state["plan"])
            final = probe.phase(folder / "final", [{"id": "review", "system": duty.CHECK_SYSTEM + "\n" + duty.NOTES_SYSTEM,
                "input": {**final_ctx, "localChecks": probe.check_view(after)}, "schema": probe.final_schema(final_ctx, state["plan"]),
                "bind": lambda raw: probe.bind_final(final_ctx, state["plan"], raw)}], pin)
            row = seal({"case": name, "status": "independent_siblings_completed_not_original_pass", "state": state,
                "candidateDigest": sha256_json(candidate), "localAfter": after, "final": final,
                "stageExitMet": False, "completeAnswerApproved": False, "historicalGradesChanged": False})
        except (ValueError, OSError) as error:
            row = seal({"case": name, "status": "stopped_without_retry", "error": str(error)[:1200], "state": state,
                        "stageExitMet": False, "completeAnswerApproved": False})
        rows.append(row)
        write_artifacts(folder / "checkpoint", {"report.json": row})
        print(json.dumps({k: row[k] for k in ("case", "status")}), flush=True)
    if probe.fingerprint() != pin:
        raise ValueError("recovery execution source drift")
    evidence = collect(output)
    write_artifacts(output / "evidence", {"report.json": evidence})
    summary = seal({"freezeDigest": freeze["reportDigest"], "parentEvidenceDigest": parent_evidence["reportDigest"],
        "rows": rows, "newEvidenceDigest": evidence["reportDigest"],
        **{k: evidence[k] for k in ("actualChatAttempts", "inputTokens", "outputTokens", "callsWithUnknownInputUsage",
            "callsWithUnknownOutputUsage", "p50CallLatencyMs", "p95CallLatencyMs")},
        "sourceScriptsExecuted": False, "newBusinessReads": 0, "effects": 0, "stageExitMet": False,
        "completeAnswerApproved": False, "failedCallsRetried": False, "historicalGradesChanged": False})
    write_artifacts(output / "summary", {"report.json": summary})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent")
    parser.add_argument("output")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.parent, args.output, live=args.run)))
