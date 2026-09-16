"""One read/review/revise/review graph over an immutable previous candidate.

Known development only, same 9B in separate calls, synthetic read hosts. No
automatic action approval, new authoring accuracy claim or retry-until-passing.
"""
from __future__ import annotations

import argparse
import copy
import json
import tarfile
import time
from pathlib import Path

from evaluation import hybrid_authoring as author
from evaluation.flow_checkpoint import author_once, implementation
from evaluation.flow_model_transport import decode
from evaluation.flow_tree_authoring import verify_receipt
from evaluation.hybrid_behavior import bindings_for, context, scenario
from evaluation.hybrid_draft_review import (
    REVIEW_SCHEMA, REVIEW_SYSTEM, REVISION_SYSTEM, assess_review, build_review_input,
    obj,
)
from evaluation.hybrid_draft_slots import apply_snapshot_revision, repair_lenses, snapshot_output_schema
from evaluation.hybrid_result_review import prepare_binding
from evaluation.source_ledger import budget
from evaluation import hybrid_review_roles
from evaluation.structured_binding_probe import read_json, write_artifacts
from network_runtime.contracts import sha256_json
from network_runtime.l0.hybrid import GovernedHybridFlow, qualify_hybrid
from network_runtime.l0.hybrid_execution import HostHybridConsent, HostReasoningBinding, ReasoningReply, context_digest, run_hybrid
from network_runtime.l0.result_contract import ResultContract, qualify_result_contract
from network_runtime.l0.structured_schema import validate_data
from network_runtime.l0.read_contracts import _source_object

REVIEW_CONFIG = {**author.MODEL_CONFIG, "num_predict": 4096}
REVIEW_CONFIG_DIGEST = sha256_json(REVIEW_CONFIG)
CASES = ("handoff", "documentation")


def seal(body):
    return {**body, "reportDigest": sha256_json(body)}


def transport_schema(schema):
    """Avoid large string grammar expansion; all limits remain locally enforced."""
    if isinstance(schema, dict):
        return {k: {name: transport_schema(child) for name, child in v.items()} if k in {"properties", "$defs"}
                else transport_schema(v) for k, v in schema.items() if k not in {"minLength", "maxLength"}}
    if isinstance(schema, list):
        return [transport_schema(v) for v in schema]
    return schema


def focused_revision_wire(payload, assessment, raw_review, *, max_output_tokens=2048):
    """Source-first local edit tasks; never carry positive AI grades as facts."""
    source = {k: payload[k] for k in ("sourceSpans", "originalTask", "hostOpenDuties", "completeCandidateDigest")}
    # The old candidate's disclaimers are not evidence or preservation policy.
    # Reconstruct limitations from actual observations instead of passing the
    # writer's own excuses back as instructions to retain unsupported facts.
    note_ids = {unit["claimId"] for unit in payload["claims"] if unit["pointer"].startswith("/candidate/notes")}
    lenses = [lens for lens in repair_lenses(payload, assessment, raw_review) if lens["claimId"] not in note_ids]
    instructions = {text: f"i{index}" for index, text in enumerate(dict.fromkeys(lens["task"] for lens in lenses))}
    actual = {"originalTask": payload["originalTask"], "previousDraft": payload["candidate"]["draft"],
              "draftBlocksForEvidenceCheck": payload["draftSpans"],
              "priorNotesPolicy": "Prior AI disclaimers are retained in the audit only, not supplied as evidence or preservation instructions. Rebuild limitations from observations.",
              "repairInstructionTable": {key: text for text, key in instructions.items()},
              "repairLenses": [{**{k: v for k, v in lens.items() if k != "task"},
                                "instructionRef": instructions[lens["task"]]} for lens in lenses],
              "originalReviewDigest": sha256_json(raw_review), "requiredOutputSchema": snapshot_output_schema(payload),
              "currentObservationEvidence": [s for s in payload["sourceSpans"] if s["kind"] == "observation"],
              "maxOutputTokens": max_output_tokens}
    return {"model": author.MODEL, "stream": False, "think": False, "format": transport_schema(actual["requiredOutputSchema"]),
        "options": {k: v for k, v in author.MODEL_CONFIG.items() if k != "think"}, "messages": [
            {"role": "system", "content": REVISION_SYSTEM + "\nThis is a focused source-to-draft correction, not a whole-draft support grade. "
             "The repairLenses locate actual differences. Compare each original source clause with its actualDraftQuote; "
             "when a responsibility, time, condition or independent entity is absent, restore it in context in the complete draft. "
             "Retain justified paraphrases. Do not say omitted information is implied by an unrelated heading. "
             "Treat draft/Skill template defaults as unverified, not current-project evidence. "
             "No earlier positive AI verdict is evidence; they are deliberately absent here. Original prohibitions still apply."},
            {"role": "user", "content": json.dumps({"complete_original_reference_and_evidence": source}, ensure_ascii=False, separators=(",", ":"))},
            {"role": "user", "content": json.dumps(actual, ensure_ascii=False, separators=(",", ":"))}]}


def build_loop(original, contract, previous_candidate, reads):
    """Host graph constructor; not LLM source-to-control-flow authoring."""
    original = GovernedHybridFlow.model_validate(original.model_dump(mode="json"))
    raw = original.model_dump(mode="json")
    reason = next(n for n in raw["nodes"] if n["id"] == contract.candidate_node)
    regions = [n for n in raw["nodes"] if n["kind"] == "strict_region"]
    if len(regions) != 1 or len(raw["nodes"]) != 2:
        raise ValueError("this local loop adapter requires exactly one original read prefix and one candidate")
    region = regions[0]["id"]
    previous_candidate = validate_data(reason["output_schema"], previous_candidate)
    qualified = qualify_hybrid(original, reads)
    projection = {k: copy.deepcopy(v) for k, v in reason["inputs"]["fields"].items() if k != region}
    projection["observations"] = author.fields({region: author.ref(region)})
    projection["candidate"] = author.literal(previous_candidate)
    projection["open_duties"] = author.literal(json.dumps([d.model_dump(mode="json") for d in contract.duties
                                                          if d.kind == "open_semantics"], ensure_ascii=False, sort_keys=True))
    properties = {k: copy.deepcopy(v) for k, v in reason["input_schema"]["properties"].items() if k != region}
    properties.update(observations=obj({region: qualified["outputSchemas"][region]}),
                      candidate=reason["output_schema"], open_duties={"type": "string"})

    def review_node(key, candidate, parents):
        inputs = {**copy.deepcopy(projection), "candidate": candidate}
        return {**copy.deepcopy(reason), "id": key, "depends_on": parents,
                "instructions": REVIEW_SYSTEM, "inputs": author.fields(inputs), "input_schema": obj(properties),
                "output_schema": REVIEW_SCHEMA, "binding_id": "local-draft-reviewer",
                "configuration_digest": REVIEW_CONFIG_DIGEST, "max_output_tokens": 4096, "max_output_bytes": 48000}

    before = review_node("review-before", author.literal(previous_candidate), [region])
    revise = {**copy.deepcopy(reason), "id": "revise-draft", "depends_on": [region, "review-before"],
              "instructions": REVISION_SYSTEM,
              "inputs": author.fields({**copy.deepcopy(projection), "review": author.ref("review-before")}),
              "input_schema": obj({**properties, "review": REVIEW_SCHEMA})}
    after = review_node("review-after", author.ref("revise-draft"), [region, "revise-draft"])
    raw.update(nodes=[*regions, before, revise, after], outputs=["revise-draft", "review-after"],
               max_model_calls=3, max_parallel=1, timeout_seconds=1200)
    flow = GovernedHybridFlow.model_validate(raw)
    amended = ResultContract.model_validate({**contract.model_dump(mode="json"), "candidate_node": "revise-draft"})
    qualify_result_contract(amended, qualify_hybrid(flow, reads), reads)
    return flow, amended


def prepare_case(previous_root, case):
    folder = Path(previous_root) / case
    inputs = folder / "inputs"
    packet, compilation, raw_contract, mapping = (read_json(inputs / f"{name}.json")
                                                  for name in ("packet", "compilation", "contract", "mapping"))
    if compilation != author.compile_proposal(packet, compilation["suppliedPages"], compilation["plan"]):
        raise ValueError("prior current-version compilation drift")
    prior = read_json(folder / "execution/summary/report.json")
    if prior != seal({k: v for k, v in prior.items() if k != "reportDigest"}):
        raise ValueError("prior run report drift")
    original, contract, qualification = prepare_binding(packet, compilation, raw_contract, mapping)
    if (prior["execution"]["status"] != "governed_graph_completed"
            or prior["execution"]["graphDigest"] != qualification["graphDigest"]
            or prior["execution"]["resultAssessment"]["contractDigest"] != qualification["contractDigest"]):
        raise ValueError("prior candidate is not bound to this graph/result contract")
    model_folder = folder / "execution/model" / contract.candidate_node
    verify_receipt(model_folder)
    previous = prior["execution"]["outputs"][contract.candidate_node]["value"]
    if previous != read_json(model_folder / "candidate.json"):
        raise ValueError("prior candidate differs from recorded model response")
    arguments, fixtures = scenario(case)
    if prior["execution"]["argumentsDigest"] != sha256_json(arguments):
        raise ValueError("caller arguments changed")
    reads, _ = bindings_for(packet, fixtures, [])
    flow, contract = build_loop(original, contract, previous, reads)
    return packet, prior, previous, arguments, fixtures, flow, contract


def draft_invoker(output, model_calls, assessments):
    """Shared bounded model binding for live-prefix and historical review graphs."""
    output = Path(output)

    def invoke(request):
        is_review = request["nodeId"] in {"review-before", "review-after"}
        config = REVIEW_CONFIG if is_review else author.MODEL_CONFIG
        configuration_digest = REVIEW_CONFIG_DIGEST if is_review else author.CONFIG_DIGEST
        payload = build_review_input(request["inputs"])
        if not is_review:
            # Check binding again before accepting feedback from the preceding
            # candidate. This does not turn feedback into observations or policy.
            feedback_assessment = assess_review(payload, request["inputs"]["review"])
        output_schema = hybrid_review_roles.wire_schema(payload) if is_review else snapshot_output_schema(payload)
        model_input = {**hybrid_review_roles.model_input(payload), "maxOutputTokens": request["maxOutputTokens"]}

        wire = {"model": author.MODEL, "stream": False, "think": False, "format": transport_schema(output_schema),
                "options": {k: v for k, v in config.items() if k != "think"},
                "messages": [{"role": "system", "content": request["instructions"]},
                             {"role": "user", "content": json.dumps(model_input, ensure_ascii=False, sort_keys=True, separators=(",", ":"))}]}
        if not is_review:
            wire = focused_revision_wire(payload, feedback_assessment, request["inputs"]["review"],
                                         max_output_tokens=request["maxOutputTokens"])
        wire_budget = budget(wire)
        if not wire_budget["accepted"]:
            write_artifacts(output / "model-preflight" / request["nodeId"], {"diagnostic.json": seal({
                "status": "context_budget_exceeded_before_model_call", "budget": wire_budget,
                "inputDigest": payload["inputDigest"], "wireRequestDigest": sha256_json(wire), "newCalls": 0})})
            raise ValueError("bounded draft loop exceeds unchanged context budget; no truncation")

        def derive(envelope):
            text, cost = decode("ollama", envelope)
            files = {"review-input.json": payload}
            if text is None:
                return files, cost
            try:
                candidate = _source_object(text) if is_review else json.loads(text)
                if is_review:
                    files["role-review.json"] = candidate
                    candidate = hybrid_review_roles.materialize_wire(payload, candidate)
                    candidate = validate_data(request["outputSchema"], candidate)
                    issues = hybrid_review_roles.wire_binding_issues(files["role-review.json"], payload)
                    if issues:
                        files["review-binding-issues.json"] = {"issues": issues, "positiveCoverageGranted": False}
                    files["review-assessment.json"] = assess_review(payload, candidate)
                else:
                    files["revision-proposal.json"] = candidate
                    candidate, files["diff-assessment.json"] = apply_snapshot_revision(payload, request["inputs"]["candidate"]["values"], candidate)
                    candidate = validate_data(request["outputSchema"], candidate)
                files["candidate.json"] = candidate
                return files, {**cost, "status": "located_ai_review_not_truth" if is_review else "unverified_materialized_revision"}
            except (ValueError, TypeError) as error:
                files["invalid-candidate.json"] = {"text": text, "diagnostic": type(error).__name__,
                                                   "reason": str(error)[:1200]}
                return files, {**cost, "status": "invalid_schema_or_review_binding"}

        result = author_once(output / "model" / request["nodeId"], {"wireRequest": wire, "governedRequest": request,
                             **({"reviewWireProfile": "host_keyed_check_cells/v2"} if is_review else {})},
                             derive, max_new_calls=1, label="bounded draft review/revision")
        model_calls.append({"node": request["nodeId"], **result["result"]})
        if "review-assessment.json" in result:
            assessments[request["nodeId"]] = result["review-assessment.json"]
        if "candidate.json" not in result:
            raise ValueError("review/revision failed closed; do not continue downstream")
        return ReasoningReply(result["candidate.json"], author.MODEL, configuration_digest,
                              result["result"].get("inputTokens"), result["result"].get("outputTokens"))

    return invoke


def run_case(prepared, output, *, case):
    packet, prior, previous, arguments, fixtures, flow, contract = prepared
    output = Path(output)
    if output.exists():
        raise FileExistsError("preserve original attempts; use a new reviewed run")
    calls, model_calls, assessments = [], [], {}
    reads, bindings = bindings_for(packet, fixtures, calls)
    qualification = qualify_hybrid(flow, reads)
    result_qualification = qualify_result_contract(contract, qualification, reads)
    ctx = context()
    consent = HostHybridConsent(qualification["graphDigest"], sha256_json(arguments), context_digest(ctx), result_qualification["contractDigest"])
    write_artifacts(output / "freeze", {"inputs.json": seal({"case": case, "flow": flow.model_dump(mode="json"),
        "resultContract": result_qualification, "previousReportDigest": prior["reportDigest"], "previousCandidate": previous,
        "arguments": arguments, "fixture": fixtures, "reviewKind": "developer_ai_not_independent_gold",
        "reviewerIndependence": "same_9b_separate_conversation_not_external_review", "hostConstructedLoop": True,
        "revisionRepresentation": "host_addressed_slots_materialized_as_unverified_candidate",
        "mappedValuesInRevision": "preserved_from_prior_candidate_by_host_not_generated_again",
        "maxNewModelCalls": 3, "maxRevisions": 1, "automaticTaskCompletionApproval": False})})
    invoke = draft_invoker(output, model_calls, assessments)

    started = time.monotonic()
    outcome = run_hybrid(flow, arguments, reads=reads, read_bindings=bindings, context=ctx, consent=consent, gates={},
        reasoners={"local-draft-reviewer": HostReasoningBinding(author.MODEL, REVIEW_CONFIG_DIGEST, invoke),
                   "local-9b": HostReasoningBinding(author.MODEL, author.CONFIG_DIGEST, invoke)}, result_contract=contract)
    report = seal({"case": case, "previousReportDigest": prior["reportDigest"], "previousCandidate": previous,
        "execution": outcome, "reviews": assessments, "modelCalls": model_calls, "providerCalls": calls,
        "latencyMs": (time.monotonic() - started) * 1000, "sourceScriptCalls": 0, "effectCalls": 0,
        "hostConstructedLoop": True, "automaticTranslationAccuracy": None, "semanticDraftAccuracy": None,
        "revisionRepresentation": "host_addressed_slots_materialized_as_unverified_candidate",
        "completeAnswerApproved": False, "largeRuntimeABUnlocked": False})
    write_artifacts(output / "summary", {"report.json": report})
    return report


def execute(previous_root, output, *, max_model_calls=0, cases=CASES):
    output, previous_root = Path(output).resolve(), Path(previous_root).resolve()
    if output.exists() or output.is_relative_to(previous_root):
        raise FileExistsError("new output outside prior evidence required")
    if not cases or len(set(cases)) != len(cases) or set(cases) - set(CASES):
        raise ValueError("select unique known development cases only")
    if type(max_model_calls) is not int or max_model_calls not in (0, 3 * len(cases)):
        raise ValueError("budget must be zero or exactly three for each preselected development loop")
    prepared = {case: prepare_case(previous_root, case) for case in cases}
    files = implementation("evaluation/hybrid_draft_loop.py", "evaluation/hybrid_draft_review.py", "evaluation/hybrid_draft_slots.py",
                           "evaluation/hybrid_authoring.py", "evaluation/hybrid_result_review.py", "evaluation/hybrid_behavior.py",
                           "evaluation/source_ledger.py", "evaluation/structured_binding_probe.py")
    manifest = seal({"cases": list(cases), "previousReports": {c: p[1]["reportDigest"] for c, p in prepared.items()},
                     "implementation": files, "reviewModelConfig": REVIEW_CONFIG, "revisionModelConfig": author.MODEL_CONFIG,
                     "maxNewModelCalls": max_model_calls, "hostConstructedLoop": True, "translationCalls": 0})
    write_artifacts(output / "freeze", {"manifest.json": manifest})
    with tarfile.open(output / "freeze/source-snapshot.tar.gz", "x:gz") as archive:
        repository = Path(__file__).resolve().parents[1]
        for name in files:
            archive.add(repository / name, arcname=name, recursive=False)
    if max_model_calls == 0:
        return manifest
    rows = []
    for case, inputs in prepared.items():
        report = run_case(inputs, output / case, case=case)
        rows.append({"case": case, "reportDigest": report["reportDigest"], "graphStatus": report["execution"]["status"],
                     "calls": len(report["modelCalls"]), "resultStatus": report["execution"].get("resultAssessment", {}).get("status"),
                     "reviewFindings": {key: value["verdictCounts"] for key, value in report["reviews"].items()}})
    summary = seal({"manifestDigest": manifest["reportDigest"], "rows": rows,
                    "semanticDraftAccuracy": None, "completeAnswerApproved": False, "largeRuntimeABUnlocked": False})
    write_artifacts(output / "summary", {"report.json": summary})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous_root")
    parser.add_argument("output")
    parser.add_argument("--max-model-calls", type=int, default=0)
    parser.add_argument("--diagnostic-case", choices=CASES)
    parser.add_argument("--diagnostic", action="store_true", help="One focused diagnostic from input/model/review-before")
    parser.add_argument("--case", choices=CASES)
    a = parser.parse_args()
    if a.diagnostic_case or a.diagnostic:
        if a.max_model_calls != 1 or a.case:
            raise ValueError("focused repair diagnosis requires exactly one new model call")
        report = diagnose_revision(a.previous_root, a.output, case=a.diagnostic_case)
    else:
        report = execute(a.previous_root, a.output, max_model_calls=a.max_model_calls, cases=(a.case,) if a.case else CASES)
    print(json.dumps({"reportDigest": report["reportDigest"], "rows": report.get("rows", []), "output": a.output}, ensure_ascii=False))


def diagnose_revision(previous_root, output, *, case=None):
    """One isolated localization diagnostic, not another runtime acceptance loop."""
    output, previous_root = Path(output).resolve(), Path(previous_root).resolve()
    if output.exists() or output.is_relative_to(previous_root):
        raise FileExistsError("a new diagnostic directory outside earlier evidence is required")
    before = (previous_root / case if case else previous_root) / "model/review-before"
    verify_receipt(before)
    previous_request = read_json(before / "request.json")["governedRequest"]
    payload = build_review_input(previous_request["inputs"])
    if payload != read_json(before / "review-input.json"):
        raise ValueError("focused diagnosis source/candidate projection drift")
    raw_review = read_json(before / "candidate.json")
    assessment = assess_review(payload, raw_review)
    lenses = repair_lenses(payload, assessment, raw_review)
    if not lenses:
        raise ValueError("no located repair lenses; do not make an unnecessary model call")
    wire = focused_revision_wire(payload, assessment, raw_review)
    files = implementation("evaluation/hybrid_draft_loop.py", "evaluation/hybrid_draft_review.py", "evaluation/hybrid_draft_slots.py",
                           "evaluation/hybrid_authoring.py", "evaluation/source_ledger.py")
    freeze = seal({"priorReceipt": read_json(before / "receipt.json"), "inputDigest": payload["inputDigest"],
        "implementation": files, "purpose": "single_call_localization_diagnostic_not_runtime_acceptance",
        "positiveReviewerOpinionsUsedAsFacts": False, "modelConfig": author.MODEL_CONFIG, "maxNewModelCalls": 1})
    write_artifacts(output / "freeze", {"inputs.json": freeze})
    with tarfile.open(output / "freeze/source-snapshot.tar.gz", "x:gz") as archive:
        for name in files:
            archive.add(Path(__file__).resolve().parents[1] / name, arcname=name, recursive=False)
    if not budget(wire)["accepted"]:
        report = seal({"status": "budget_blocked_before_call", "newCalls": 0, "budget": budget(wire)})
        write_artifacts(output / "summary", {"report.json": report})
        return report

    def derive(envelope):
        text, cost = decode("ollama", envelope)
        artifacts = {"repair-lenses.json": lenses, "review-input.json": payload}
        if text is None:
            return artifacts, cost
        try:
            proposal = json.loads(text)
            artifacts["revision-proposal.json"] = proposal
            candidate, application = apply_snapshot_revision(payload, previous_request["inputs"]["candidate"]["values"], proposal)
            artifacts.update({"candidate.json": candidate, "diff-assessment.json": application})
            return artifacts, {**cost, "status": "unverified_materialized_diagnostic"}
        except (ValueError, TypeError) as error:
            artifacts["invalid-candidate.json"] = {"text": text, "reason": str(error)[:1200]}
            return artifacts, {**cost, "status": "invalid_diagnostic_proposal"}

    response = author_once(output / "model", {"wireRequest": wire, "diagnosticFreezeDigest": freeze["reportDigest"]}, derive,
        max_new_calls=1, label="one focused source-to-draft diagnostic")
    report = seal({"freezeDigest": freeze["reportDigest"], "modelCall": response["result"],
        "candidateMaterialized": "candidate.json" in response, "providerCalls": 0, "effectCalls": 0,
        "runtimeAcceptanceRun": False, "semanticSuccess": None, "completeAnswerApproved": False})
    write_artifacts(output / "summary", {"report.json": report})
    return report


if __name__ == "__main__":
    main()
