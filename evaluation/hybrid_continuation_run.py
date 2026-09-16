"""Run a frozen finite number of read continuations, retaining every attempt.

Local 9B + disclosed in-process hosts only. No production connectors or scripts.
Model-selected requests pass current resource policy and the original gateway.
"""
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

from evaluation import hybrid_authoring as author, hybrid_reasoning_transport
from evaluation.flow_checkpoint import author_once, implementation
from evaluation.flow_model_transport import decode
from evaluation.hybrid_behavior import bindings_for, context
from evaluation.hybrid_draft_loop import REVIEW_CONFIG, transport_schema
from evaluation.hybrid_continuation import HostContinuationPolicy, compile_continuation, guarded_bindings, host_gate, selection_transport_schema
from evaluation.source_ledger import budget
from evaluation.structured_authoring import seal
from evaluation.structured_binding_probe import write_artifacts
from network_runtime.contracts import sha256_json
from network_runtime.l0.hybrid import qualify_hybrid
from network_runtime.l0.hybrid_execution import HostHybridConsent, HostReasoningBinding, ReasoningReply, context_digest, run_hybrid
from network_runtime.l0.structured_schema import snapshot_json, validate_data
from network_runtime.l0.structured_reads import instantiate_structured_read, parse_read_contract


def pipeline_implementation():
    return implementation("evaluation/hybrid_continuation.py", "evaluation/hybrid_continuation_run.py",
        "evaluation/hybrid_authoring.py", "evaluation/hybrid_behavior.py", "evaluation/hybrid_reasoning_transport.py",
        "evaluation/hybrid_prefix.py", "evaluation/hybrid_parameters.py", "evaluation/source_ledger.py",
        "evaluation/hybrid_snapshot_review.py", "evaluation/hybrid_draft_loop.py",
        "evaluation/hybrid_draft_slots.py", "evaluation/hybrid_draft_review.py")


def historical_context(report, packet):
    if report != seal({k: v for k, v in report.items() if k != "reportDigest"}):
        raise ValueError("previous report digest drift")
    execution = report.get("execution", report)
    contracts = {r.contract_hash: (name, r) for name, raw in packet["reads"].items() for r in [parse_read_contract(raw)]}
    observations = []
    for event in execution.get("trace", []):
        for row in event.get("regionReport", {}).get("trace", []):
            if row.get("kind") == "read":
                receipt = row["receipt"]
                if receipt != {**{k: v for k, v in receipt.items() if k != "receiptDigest"},
                               "receiptDigest": sha256_json({k: v for k, v in receipt.items() if k != "receiptDigest"})}:
                    raise ValueError("historical read receipt digest drift")
                name, contract = contracts[receipt["contractHash"]]
                arguments = row.get("argumentBinding", {}).get("arguments")
                if arguments is None or instantiate_structured_read(contract, arguments)["requestDigest"] != receipt["requestDigest"]:
                    raise ValueError("historical read lacks exact contract-bound arguments")
                observations.append({"region": event["node"], "readNode": row["node"], "receipt": receipt,
                                     "tool": name, "arguments": arguments})
    return {"previousReportDigest": report["reportDigest"], "historicalContext": {
        "previousStatus": execution["status"], "previousOutputs": execution.get("outputs", {}),
        "observations": observations, "currentActionAuthority": False,
        "claimBoundary": "Recorded prior snapshots and unverified model drafts, not current permission or semantic truth."}}


def fixture_policy(fixtures):
    resources = snapshot_json(fixtures)
    def authorize(name, args, caller):
        resource = resources.get(name)
        if resource is None:
            return False
        entries = resource["resources"] if isinstance(resource, dict) and set(resource) == {"resources"} else [resource]
        return sum(expected == args for expected, _ in entries) == 1
    return HostContinuationPolicy(sha256_json({"scope": "disclosed_in_process_resources_only", "resources": resources}), authorize)


def run(packet, compilation, previous_report, arguments, fixtures, output, *, max_rounds=2, final_review_calls=0):
    output = Path(output)
    if output.exists():
        raise FileExistsError("new evidence directory required; never overwrite or retry prior attempts")
    if type(max_rounds) is not int or not 1 <= max_rounds <= 3:
        raise ValueError("explicit one-to-three round bound required")
    if type(final_review_calls) is not int or final_review_calls not in (0, 3):
        raise ValueError("final review requires an explicit zero or three-call budget")
    review_output = output.with_name(output.name + "-final-review")
    if final_review_calls and review_output.exists():
        raise FileExistsError("final review evidence already exists; refuse before any model work")
    if compilation != author.compile_proposal(packet, compilation["suppliedPages"], compilation["plan"]):
        raise ValueError("original automatic source/task compilation drift")
    arguments = validate_data(packet["inputSchema"], arguments)
    if (previous_report.get("compilationDigest") != compilation["reportDigest"]
            or previous_report.get("execution", {}).get("argumentsDigest") != sha256_json(arguments)):
        raise ValueError("prior run is not bound to this original compilation/caller")
    history = historical_context(previous_report, packet)
    policy = fixture_policy(fixtures)
    files = pipeline_implementation()
    freeze = seal({"packet": packet, "originalCompilationDigest": compilation["reportDigest"],
        "previousReport": previous_report, "arguments": arguments, "fixtures": fixtures,
        "hostPolicyDigest": policy.policy_digest, "maxRounds": max_rounds,
        "maxNewModelCalls": max_rounds * 2 + final_review_calls, "finalReviewCallBudget": final_review_calls,
        "implementation": files, "modelConfig": author.MODEL_CONFIG, "sourceScriptsExecutable": False,
        "noNewObservationPolicy": "retain_exact_prior_candidate_with_unverified_selection_notice_no_writer_call",
        "selectionDecoder": "discriminated_generation_original_runtime_schema_and_gate_unchanged",
        "reviewModelConfig": REVIEW_CONFIG if final_review_calls else None,
        "historicalSnapshotIsCurrentAuthority": False, "automaticWholeSkillSemanticConversionClaimed": False})
    write_artifacts(output / "freeze", {"inputs.json": freeze})
    with tarfile.open(output / "freeze/source-snapshot.tar.gz", "x:gz") as archive:
        for name in files:
            archive.add(Path(__file__).resolve().parents[1] / name, arcname=name, recursive=False)
    reports, all_calls, costs = [], [], []
    used = {sha256_json({r["tool"]: r["arguments"]}) for r in history["historicalContext"]["observations"]}
    for index in range(max_rounds):
        folder, calls = output / f"round-{index + 1}", []
        flow = compile_continuation(packet, compilation["suppliedPages"], history, policy_digest=policy.policy_digest,
                                    preserve_without_read=True)
        reads, bindings = bindings_for(packet, fixtures, calls)
        qualification = qualify_hybrid(flow, reads)
        gate = host_gate(packet, flow, reads, policy)
        ctx = context()
        write_artifacts(folder / "freeze", {"graph.json": qualification, "history.json": history})

        def invoke(request):
            generation_request = request
            generation_format = "json"
            if request["nodeId"] == "select":
                selection_schema = selection_transport_schema(packet)
                generation_request = {**request, "outputSchema": selection_schema}
                generation_format = transport_schema(selection_schema)
            wire = {"model": author.MODEL, "stream": False, "think": False, "format": "json",
                "options": {k: v for k, v in author.MODEL_CONFIG.items() if k != "think"},
                "messages": hybrid_reasoning_transport.messages(generation_request)}
            wire["format"] = generation_format
            if not budget(wire)["accepted"]:
                write_artifacts(folder / "preflight" / request["nodeId"], {"diagnostic.json": seal({
                    "status": "context_budget_exceeded_before_model_call", "wireDigest": sha256_json(wire), "newCalls": 0})})
                raise ValueError("continuation exceeds unchanged context budget; do not truncate")

            def derive(envelope):
                text, cost = decode("ollama", envelope)
                if text is None:
                    return {}, cost
                try:
                    value = validate_data(request["outputSchema"], json.loads(text))
                    return {"candidate.json": value}, {**cost, "status": "unverified_candidate_not_permission"}
                except (ValueError, TypeError) as error:
                    return {"invalid-candidate.json": {"text": text, "diagnostic": str(error)[:1200]}}, {**cost, "status": "invalid_candidate"}

            response = author_once(folder / "model" / request["nodeId"], {"wireRequest": wire, "governedRequest": request},
                derive, max_new_calls=1, label="bounded read continuation")
            costs.append({"round": index + 1, "node": request["nodeId"], **response["result"]})
            if "candidate.json" not in response:
                raise ValueError("continuation candidate invalid; no downstream work")
            candidate = response["candidate.json"]
            if request["nodeId"] == "select" and candidate["decision"] not in {"clarify", "answer"}:
                fingerprint = sha256_json(candidate["requests"])
                if fingerprint in used:
                    raise ValueError("repeated identical read proposal exceeds no-repeat continuation policy")
                used.add(fingerprint)
            return ReasoningReply(candidate, author.MODEL, author.CONFIG_DIGEST,
                response["result"].get("inputTokens"), response["result"].get("outputTokens"))

        outcome = run_hybrid(flow, arguments, reads=reads, read_bindings=guarded_bindings(packet, bindings, policy, arguments),
            reasoners={"local-9b": HostReasoningBinding(author.MODEL, author.CONFIG_DIGEST, invoke)},
            gates={"continuation-policy": gate}, context=ctx,
            consent=HostHybridConsent(qualification["graphDigest"], sha256_json(arguments), context_digest(ctx)))
        report = seal({"execution": outcome, "previousReportDigest": history["previousReportDigest"],
            "providerCalls": snapshot_json(calls), "modelCalls": snapshot_json([c for c in costs if c["round"] == index + 1]),
            "wholeTaskSemanticSuccess": None, "automaticPermission": False})
        write_artifacts(folder / "summary", {"report.json": report})
        reports.append(report)
        all_calls.extend(calls)
        if outcome["status"] != "governed_graph_completed":
            break
        if outcome["outputs"]["select"]["value"]["decision"] in {"clarify", "answer"}:
            break
        newer = historical_context(report, packet)
        # Keep all original observations, but only the latest candidate; earlier
        # full drafts/reports remain on disk and are explicitly digest-linked.
        newer["historicalContext"]["observations"] = [*history["historicalContext"]["observations"],
                                                     *newer["historicalContext"]["observations"]]
        history = newer
    summary = seal({"status": "bounded_continuation_completed" if reports[-1]["execution"]["status"] == "governed_graph_completed" else "blocked",
        "freezeDigest": freeze["reportDigest"], "roundReports": [r["reportDigest"] for r in reports],
        "providerCalls": snapshot_json(all_calls), "modelCalls": snapshot_json(costs), "lastExecution": reports[-1]["execution"],
        "sourceScriptCalls": 0, "effectCalls": 0, "completeAnswerApproved": False,
        "semanticSuccess": None, "wholeSkillConversionRate": None, "largeRuntimeABUnlocked": False})
    write_artifacts(output / "summary", {"report.json": summary})
    if final_review_calls:
        from evaluation.hybrid_snapshot_review import run as review_final

        if files != pipeline_implementation():
            reviewed, review_status = None, "not_run_implementation_drift"
        elif summary["status"] == "bounded_continuation_completed":
            # Separate directory: original continuation evidence remains sealed.
            reviewed = review_final(output, review_output, max_model_calls=3)
            review_status = reviewed["execution"]["status"]
        else:
            reviewed, review_status = None, "not_run_blocked_continuation"
        summary = seal({**{k: v for k, v in summary.items() if k != "reportDigest"},
            "status": "bounded_pipeline_completed_not_semantic_proof" if review_status == "governed_graph_completed" else "bounded_pipeline_incomplete",
            "continuationReportDigest": summary["reportDigest"], "finalReviewStatus": review_status,
            "finalReviewReportDigest": reviewed["reportDigest"] if reviewed else None,
            "semanticSuccess": None, "completeAnswerApproved": False})
        write_artifacts(output / "pipeline-summary", {"report.json": summary})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packet", "compilation", "previous_report", "fixture", "output"):
        parser.add_argument(name)
    parser.add_argument("--max-rounds", type=int, required=True)
    parser.add_argument("--final-review-calls", type=int, choices=(0, 3), default=0)
    args = parser.parse_args()
    def read(path):
        return json.loads(Path(path).read_text())
    fixture = read(args.fixture)
    result = run(read(args.packet), read(args.compilation), read(args.previous_report),
                 fixture["arguments"], fixture["resources"], args.output, max_rounds=args.max_rounds,
                 final_review_calls=args.final_review_calls)
    print(json.dumps({"status": result["status"], "reportDigest": result["reportDigest"],
                      "rounds": len(result["roundReports"]), "modelCalls": len(result["modelCalls"]),
                      "providerCalls": result["providerCalls"], "output": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
