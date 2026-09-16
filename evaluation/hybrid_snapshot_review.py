"""Source-first bounded review of a completed continuation's actual draft.

Reuses the same Runtime scheduler and review/edit binding. No new tool calls,
snapshot refresh, semantic self-approval, case-specific graph or answer values.
"""
from __future__ import annotations

import argparse
import copy
import json
import tarfile
from pathlib import Path

from evaluation import hybrid_authoring as author
from evaluation.flow_checkpoint import implementation
from evaluation.flow_tree_authoring import verify_receipt
from evaluation.hybrid_behavior import context
from evaluation.hybrid_draft_loop import REVIEW_CONFIG, REVIEW_CONFIG_DIGEST, draft_invoker, seal
from evaluation.hybrid_draft_review import REVIEW_SCHEMA, REVIEW_SYSTEM, REVISION_SYSTEM, build_review_input
from evaluation.hybrid_review_context import attach, read_entry, trace_entries
from evaluation.structured_binding_probe import read_json, write_artifacts
from network_runtime.contracts import sha256_json
from network_runtime.l0.hybrid import GovernedHybridFlow, qualify_hybrid
from network_runtime.l0.hybrid_execution import HostHybridConsent, HostReasoningBinding, context_digest, run_hybrid
from network_runtime.l0.structured_schema import validate_data
from network_runtime.l0.structured_reads import parse_read_contract
from network_runtime.l0.structured_bindings import materialize_binding


def prepare(root):
    root = Path(root)
    summary = read_json(root / "summary/report.json")
    if summary != seal({k: v for k, v in summary.items() if k != "reportDigest"}):
        raise ValueError("continuation summary drift")
    if "materializedCandidateDigest" in summary:
        from evaluation.hybrid_historical_candidate import prepare as prepare_historical
        return prepare_historical(root)
    if "compilationDigest" in summary:
        return prepare_initial(root, summary)
    if summary["status"] != "bounded_continuation_completed":
        raise ValueError("no completed continuation draft to review")
    index = len(summary["roundReports"])
    model = root / f"round-{index}/model/deliver"
    freeze = read_json(root / "freeze/inputs.json")
    if freeze != seal({k: v for k, v in freeze.items() if k != "reportDigest"}) or freeze["reportDigest"] != summary["freezeDigest"]:
        raise ValueError("continuation inputs drift")
    packet = freeze["packet"]
    graph = read_json(root / f"round-{index}/freeze/graph.json")
    reads = {r.contract_hash: r for raw in packet["reads"].values() for r in [parse_read_contract(raw)]}
    if (graph != qualify_hybrid(GovernedHybridFlow.model_validate(graph["proposal"]), reads)
            or graph["graphDigest"] != summary["lastExecution"]["graphDigest"]):
        raise ValueError("recorded continuation graph drift")
    available = {"input": freeze["arguments"], **{k: v["value"] for k, v in summary["lastExecution"]["outputs"].items()}}
    binding = graph["inputBindings"]["deliver"]
    inputs = materialize_binding(binding, {k: available[k] for k in binding["requiredSources"]})["arguments"]
    node = next(n for n in graph["proposal"]["nodes"] if n["id"] == "deliver")
    retained = [e for e in summary["lastExecution"]["trace"] if e.get("node") == "deliver" and e.get("modelInvoked") is False]
    if retained:
        from evaluation.hybrid_continuation import prior_candidate

        if len(retained) != 1 or node["kind"] != "reason_if" or model.exists():
            raise ValueError("ambiguous retained-candidate origin or invented model receipt")
        plans = graph["conditionalBindings"]["deliver"]
        value = materialize_binding(plans["condition"], {k: available[k] for k in plans["condition"]["requiredSources"]})["arguments"]
        if value == node["condition"]["equals"]:
            raise ValueError("model-required branch cannot reuse a previous answer")
        candidate = materialize_binding(plans["otherwise"], {k: available[k] for k in plans["otherwise"]["requiredSources"]})["arguments"]
        history = read_json(root / f"round-{index}/freeze/history.json")
        previous = freeze["previousReport"] if index == 1 else read_json(root / f"round-{index - 1}/summary/report.json")
        if (previous != seal({k: v for k, v in previous.items() if k != "reportDigest"})
                or history["previousReportDigest"] != previous["reportDigest"]
                or history["historicalContext"]["previousOutputs"] != previous["execution"].get("outputs", {})
                or candidate != prior_candidate(history)
                or retained[0]["retainedCandidateDigest"] != sha256_json(candidate)
                or retained[0]["inputsDigest"] != sha256_json(inputs)):
            raise ValueError("retained candidate provenance drift")
    else:
        verify_receipt(model)
        request = read_json(model / "request.json")["governedRequest"]
        candidate = read_json(model / "candidate.json")
        if request["graphDigest"] != graph["graphDigest"] or request["inputs"] != inputs:
            raise ValueError("draft input differs from the frozen graph/actual outputs")
    if candidate != summary["lastExecution"]["outputs"]["deliver"]["value"]:
        raise ValueError("recorded draft differs from the actual completed graph")
    values = validate_data(node["output_schema"], candidate)
    declared_input = node["input_schema"]
    observation_values, observation_schemas = {}, {}
    prior_schema = declared_input["properties"]["priorReadResults"]["items"]["properties"]["result"]
    for i, row in enumerate(inputs["priorReadResults"]):
        observation_values[f"prior{i}"] = {"observations": row["result"]}
        observation_schemas[f"prior{i}"] = author.obj({"observations": prior_schema})
    current_schema = declared_input["properties"]["currentObservation"]["properties"]["observations"]
    observation_values["current"] = {"observations": inputs["currentObservation"]["observations"]}
    observation_schemas["current"] = author.obj({"observations": current_schema})
    candidate_schema = author.obj({"values": author.obj({}), "draft": {"type": "string", "maxLength": 12000},
        "notes": {"type": "array", "maxItems": 48, "items": {"type": "string"}}})
    supplied = {"original_task": inputs["original_task"], "source_material": inputs["source_material"],
        "caller": inputs["caller"], "candidate": {"values": {}, "draft": values["draft"], "notes": values["uncertainties"]},
        "observations": observation_values, "open_duties": json.dumps({"originalTask": inputs["original_task"],
            "previousRemainingActions": values["remaining_actions"], "semanticApproval": "not_granted",
            **({"noReadSelection": {"decision": inputs["selection"]["decision"],
                    **({"question": inputs["selection"]["message"]} if inputs["selection"]["decision"] == "clarify" else {})},
                "rawSelectionDigest": sha256_json(inputs["selection"]),
                "selectionOrigin": "unverified_question_only_answer_rationale_retained_in_original_trace_not_new_evidence",
                "answerRetainedWithoutNewModelInvocation": True} if retained else {})}, ensure_ascii=False)}
    schema = author.obj({"original_task": {"type": "string"}, "source_material": {"type": "string"},
        "caller": packet["inputSchema"], "candidate": candidate_schema,
        "observations": author.obj(observation_schemas), "open_duties": {"type": "string"}})
    entries = [read_entry(f"prior{i}", row["tool"], row["tool"], row["arguments"][row["tool"]],
                          row["result"][row["tool"]], row["receiptDigest"])
               for i, row in enumerate(inputs["priorReadResults"])]
    tool_by_hash = {parse_read_contract(raw).contract_hash: name for name, raw in packet["reads"].items()}
    entries.extend(trace_entries(summary["lastExecution"], tool_by_hash,
        {e["node"]: "current" for e in summary["lastExecution"]["trace"] if e.get("kind") == "strict_region"}))
    attach(supplied, schema, packet, entries)
    validate_data(schema, supplied)
    build_review_input(supplied)  # Bound capacity before any new model request.
    return packet, summary, supplied, schema


def prepare_initial(root, summary):
    """Same review input from an actual automatically authored read/reason run."""
    freeze = read_json(root / "freeze/inputs.json")
    if freeze != seal({k: v for k, v in freeze.items() if k != "reportDigest"}):
        raise ValueError("initial run input drift")
    packet, compilation = freeze["packet"], freeze["compilation"]
    if (compilation != author.compile_proposal(packet, compilation["suppliedPages"], compilation["plan"])
            or summary["compilationDigest"] != compilation["reportDigest"] or freeze["resultContract"] is not None):
        raise ValueError("initial review requires the exact automatic compilation, without a replaced result binding")
    flow = GovernedHybridFlow.model_validate(compilation["flow"])
    reads = {r.contract_hash: r for raw in packet["reads"].values() for r in [parse_read_contract(raw)]}
    graph, execution = qualify_hybrid(flow, reads), summary["execution"]
    if (execution["graphDigest"] != graph["graphDigest"] or execution["status"] != "governed_graph_completed"
            or execution["argumentsDigest"] != sha256_json(freeze["arguments"])):
        raise ValueError("initial execution/caller does not match the frozen graph")
    writers = [node for node in flow.nodes if node.kind == "reason" and node.id in flow.outputs]
    if len(writers) != 1:
        raise ValueError("one actual delivered draft required; never guess between parallel writers")
    writer = writers[0]
    folder = root / "model" / writer.id
    verify_receipt(folder)
    request = read_json(folder / "request.json")["governedRequest"]
    candidate = read_json(folder / "candidate.json")
    sources = {"input": freeze["arguments"], **{k: v["value"] for k, v in execution["outputs"].items()}}
    # Non-exported strict values still have exact read receipts in the trace;
    # reconstruct the same observation projection used by the original engine.
    for event in execution["trace"]:
        if event.get("kind") == "strict_region" and event.get("status") == "succeeded":
            region = event["regionReport"]
            if region != seal({k: v for k, v in region.items() if k != "reportDigest"}):
                raise ValueError("strict source region report drift")
            sources[event["node"]] = {"outcome": region["status"], "observations": {
                row["node"]: row["receipt"]["payload"] for row in region["trace"] if row["kind"] == "read"}}
    binding = graph["inputBindings"][writer.id]
    actual = materialize_binding(binding, {k: sources[k] for k in binding["requiredSources"]})["arguments"]
    if (request["graphDigest"] != graph["graphDigest"] or request["inputs"] != actual
            or candidate != execution["outputs"][writer.id]["value"]):
        raise ValueError("delivered candidate or source provenance drift")
    validate_data(writer.output_schema, candidate)
    regions = [node.id for node in flow.nodes if node.kind == "strict_region" and node.id in actual]
    supplied = {"original_task": actual["original_task"], "source_material": actual["source_material"],
        "caller": actual["caller"], "observations": {key: actual[key] for key in regions},
        "candidate": {"values": {}, "draft": candidate["draft"], "notes": candidate["uncertainties"]},
        "open_duties": json.dumps({"originalTask": packet["task"], "authoringBoundaries": actual.get("authoring_boundaries", ""),
                                   "remainingActions": candidate["remaining_actions"], "semanticApproval": "not_granted"})}
    schema = author.obj({"original_task": {"type": "string"}, "source_material": {"type": "string"},
        "caller": packet["inputSchema"], "observations": author.obj({key: writer.input_schema["properties"][key] for key in regions}),
        "candidate": author.obj({"values": author.obj({}), "draft": writer.output_schema["properties"]["draft"],
                                 "notes": writer.output_schema["properties"]["uncertainties"]}),
        "open_duties": {"type": "string"}})
    tool_by_hash = {parse_read_contract(raw).contract_hash: name for name, raw in packet["reads"].items()}
    entries = [e for e in trace_entries(execution, tool_by_hash) if e["region"] in regions]
    attach(supplied, schema, packet, entries)
    validate_data(schema, supplied)
    build_review_input(supplied)
    return packet, summary, supplied, schema


def build_snapshot_flow(packet, inputs, schema):
    projections = {key: author.literal(value) for key, value in inputs.items()}
    nodes = []
    for key in ("review-before", "revise-draft", "review-after"):
        is_review = key != "revise-draft"
        fields, properties = copy.deepcopy(projections), copy.deepcopy(schema["properties"])
        parents = []
        if key == "revise-draft":
            fields["review"], properties["review"] = author.ref("review-before"), REVIEW_SCHEMA
            parents = ["review-before"]
        elif key == "review-after":
            fields["candidate"] = author.ref("revise-draft")
            parents = ["revise-draft"]
        nodes.append({"kind": "reason", "id": key, "depends_on": parents,
            "inputs": author.fields(fields), "input_schema": author.obj(properties),
            "output_schema": REVIEW_SCHEMA if is_review else schema["properties"]["candidate"],
            "instructions": REVIEW_SYSTEM if is_review else REVISION_SYSTEM,
            "binding_id": "local-draft-reviewer" if is_review else "local-9b", "model": author.MODEL,
            "configuration_digest": REVIEW_CONFIG_DIGEST if is_review else author.CONFIG_DIGEST,
            "timeout_seconds": 360, "max_input_bytes": 131072, "max_output_bytes": 48000 if is_review else 24000,
            "max_output_tokens": 4096 if is_review else 2048})
    return GovernedHybridFlow.model_validate({"api_version": "netopyu.io/governed-hybrid/v1",
        "source_digest": packet["bundle"]["bundleDigest"], "task_digest": sha256_json(packet["task"]),
        "purpose": "Historical source-aligned draft review without fresh action evidence", "input_schema": author.obj({}),
        "nodes": nodes, "outputs": ["revise-draft", "review-after"], "max_model_calls": 3, "max_parallel": 1,
        "timeout_seconds": 1200, "failure_policy": "stop_no_downstream"})


def run(root, output, *, max_model_calls=0, candidate_request=None, task_scope=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(root):
        raise FileExistsError("new evidence directory outside prior run required")
    if type(max_model_calls) is not int or max_model_calls not in (0, 1, 3):
        raise ValueError("zero preflight, one located review, or three-call legacy loop required")
    if candidate_request is not None:
        from evaluation.hybrid_review_context import import_candidate_context
        if max_model_calls not in (0, 1):
            raise ValueError("context import permits one new review, never historical loop replay")
        packet, previous, inputs, schema = import_candidate_context(root, candidate_request, task_scope)
    else:
        if task_scope is not None:
            raise ValueError("task scope belongs in the author packet or an explicit diagnostic candidate import")
        packet, previous, inputs, schema = prepare(root)
    if previous.get("kind") == "historical_repair_candidate_data_import" and max_model_calls not in (0, 1):
        raise ValueError("historical edited data permits one fresh review, not a legacy loop or old-opinion reuse")
    flow = build_snapshot_flow(packet, inputs, schema)
    if max_model_calls == 1:
        raw = flow.model_dump(mode="json")
        raw.update(nodes=raw["nodes"][:1], outputs=["review-before"], max_model_calls=1)
        flow = GovernedHybridFlow.model_validate(raw)
    qualification = qualify_hybrid(flow, {})
    files = implementation("evaluation/hybrid_snapshot_review.py", "evaluation/hybrid_draft_loop.py",
        "evaluation/hybrid_draft_slots.py", "evaluation/hybrid_draft_review.py", "evaluation/hybrid_authoring.py",
        "evaluation/source_ledger.py", "evaluation/hybrid_behavior.py")
    freeze = seal({"previousReportDigest": previous["reportDigest"], "inputs": inputs, "graph": qualification,
        **({"contextImport": previous} if candidate_request is not None else {}),
        **({"historicalImportReceipt": previous} if previous.get("kind") == "historical_repair_candidate_data_import" else {}),
        "reviewPhase": "source_review_only" if max_model_calls == 1 else "legacy_three_call_loop",
        "implementation": files, "maxNewModelCalls": max_model_calls, "reviewModelConfig": REVIEW_CONFIG,
        "revisionModelConfig": author.MODEL_CONFIG, "historicalReadSnapshotsOnly": True,
        "newBusinessReadCalls": 0, "authorityGranted": False})
    write_artifacts(output / "freeze", {"inputs.json": freeze})
    with tarfile.open(output / "freeze/source-snapshot.tar.gz", "x:gz") as archive:
        for name in files:
            archive.add(Path(__file__).resolve().parents[1] / name, arcname=name, recursive=False)
    if not max_model_calls:
        return freeze
    calls, assessments = [], {}
    invoke = draft_invoker(output, calls, assessments)
    ctx = context()
    result = run_hybrid(flow, {}, reads={}, read_bindings={}, gates={}, context=ctx,
        consent=HostHybridConsent(qualification["graphDigest"], sha256_json({}), context_digest(ctx)),
        reasoners={"local-draft-reviewer": HostReasoningBinding(author.MODEL, REVIEW_CONFIG_DIGEST, invoke),
                   "local-9b": HostReasoningBinding(author.MODEL, author.CONFIG_DIGEST, invoke)})
    report = seal({"previousReportDigest": previous["reportDigest"], "freezeDigest": freeze["reportDigest"],
        "execution": result, "modelCalls": calls, "reviews": assessments, "newBusinessReadCalls": 0,
        "effectCalls": 0, "semanticSuccess": None, "completeAnswerApproved": False, "largeRuntimeABUnlocked": False})
    write_artifacts(output / "summary", {"report.json": report})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("output")
    parser.add_argument("--max-model-calls", type=int, default=0)
    parser.add_argument("--candidate-request", help="Explicit receipt-bound historical candidate, new diagnostic only")
    parser.add_argument("--task-scope", help="JSON list of explicit lossless task segments; requires candidate request")
    args = parser.parse_args()
    result = run(args.root, args.output, max_model_calls=args.max_model_calls,
                 candidate_request=args.candidate_request, task_scope=read_json(args.task_scope) if args.task_scope else None)
    print(json.dumps({"reportDigest": result["reportDigest"], "status": result.get("execution", {}).get("status", "preflight"),
                      "output": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
