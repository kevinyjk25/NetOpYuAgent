"""Real local 9B inside reviewed mixed graphs; synthetic business observations.

No automatic semantic approval. A digest-bound, manually supplied developer
review is required. Persist every request once; never retry unknown work.
"""
from __future__ import annotations

import argparse
import copy
import json
import threading
import time
from pathlib import Path

from evaluation import hybrid_authoring as author, hybrid_reasoning_transport
from evaluation.flow_checkpoint import author_once
from evaluation.flow_model_transport import decode
from evaluation.source_ledger import budget
from evaluation.hybrid_behavior import FIXTURE_VERSION, bindings_for, context, scenario
from evaluation.stage2_batch import digest
from evaluation.structured_authoring import seal
from evaluation.structured_binding_probe import read_json, write_artifacts
from network_runtime.contracts import sha256_json
from network_runtime.l0.hybrid import GovernedHybridFlow, qualify_hybrid
from network_runtime.l0.hybrid_execution import HostHybridConsent, HostReasoningBinding, ReasoningReply, context_digest, run_hybrid
from network_runtime.l0.structured_schema import validate_data


def run(packet_path, compilation_path, review_path, output, *, case, max_model_calls=0):
    output = Path(output)
    if output.exists():
        raise FileExistsError("preserve prior execution; late/failed/incomplete requests are never retried")
    packet, compilation, review = map(read_json, (packet_path, compilation_path, review_path))
    if (compilation != seal({k: v for k, v in compilation.items() if k != "reportDigest"})
            or review.get("compilationDigest") != compilation["reportDigest"]
            or review.get("decision") != "admit_local_read_reason_only"
            or review.get("reviewKind") != "developer_ai_not_independent_gold"
            or not review.get("rationale") or review.get("case") != case):
        raise ValueError("explicit source-reviewed, digest-bound local admission required")
    rebuilt = author.compile_proposal(packet, compilation["suppliedPages"], compilation["plan"])
    if compilation != rebuilt:
        raise ValueError("compiled graph/source/task/host drift")
    flow = GovernedHybridFlow.model_validate(compilation["flow"])
    required = sum(n.kind == "reason" for n in flow.nodes)
    if type(max_model_calls) is not int or not required <= max_model_calls <= 8:
        raise ValueError("explicit bounded actual model-call budget required")
    arguments, fixtures = scenario(case)
    freeze = {"case": case, "compilationDigest": compilation["reportDigest"], "review": review,
        "fixtureVersion": FIXTURE_VERSION, "arguments": arguments, "fixture": fixtures,
        "implementation": {name: digest(Path(__file__).parents[1] / name) for name in (
            "evaluation/hybrid_live_demo.py", "evaluation/hybrid_behavior.py", "evaluation/hybrid_authoring.py",
            "evaluation/hybrid_reasoning_transport.py",
            "network_runtime/l0/hybrid.py", "network_runtime/l0/hybrid_execution.py")},
        "maxActualModelCalls": max_model_calls, "fixtureIsInProcessNotRealNetwork": True}
    write_artifacts(output / "freeze", {"inputs.json": seal(freeze)})
    calls, model_costs, lock = [], [], threading.Lock()
    reads, bindings = bindings_for(packet, fixtures, calls)
    def invoke(request):
        wire = {"model": author.MODEL, "stream": False, "think": False, "format": "json",
            "options": {k: v for k, v in author.MODEL_CONFIG.items() if k != "think"},
            "messages": hybrid_reasoning_transport.messages(request)}
        if not budget(wire)["accepted"]:
            raise ValueError("reasoning request exceeds unchanged conservative context budget")
        def derive(envelope):
            text, cost = decode("ollama", envelope)
            if text is None:
                return {}, cost
            try:
                candidate = validate_data(request["outputSchema"], json.loads(text))
                return {"candidate.json": candidate}, {**cost, "status": "schema_valid_candidate_not_semantic_proof"}
            except (ValueError, TypeError):
                return {"invalid-candidate.json": {"text": text}}, {**cost, "status": "invalid_model_candidate"}
        folder = output / "model" / request["nodeId"]
        # One declared node invokes once; author_once additionally seals raw wire,
        # envelope, model artifact identity and derivation receipts.
        result = author_once(folder, {"wireRequest": wire, "governedRequest": request}, derive,
                             max_new_calls=1, label="governed reasoning")
        with lock:
            model_costs.append({"node": request["nodeId"], **result["result"]})
        if "candidate.json" not in result:
            raise ValueError("model candidate not accepted")
        cost = result["result"]
        return ReasoningReply(result["candidate.json"], author.MODEL, author.CONFIG_DIGEST, cost["inputTokens"], cost["outputTokens"])
    ctx = context()
    qualification = qualify_hybrid(flow, reads)
    began = time.monotonic()
    outcome = run_hybrid(flow, arguments, reads=reads, read_bindings=bindings,
        reasoners={"local-9b": HostReasoningBinding(author.MODEL, author.CONFIG_DIGEST, invoke)}, gates={}, context=ctx,
        consent=HostHybridConsent(qualification["graphDigest"], sha256_json(arguments), context_digest(ctx)))
    wall_ms = (time.monotonic() - began) * 1000
    with lock:
        observed_costs = copy.deepcopy(model_costs)
    report = seal({"case": case, "compilationDigest": compilation["reportDigest"], "execution": outcome,
        "providerCalls": copy.deepcopy(calls), "modelCalls": observed_costs, "runtimeWallLatencyMs": wall_ms,
        "sourceScriptCalls": 0, "effectCalls": 0,
        "actualDraftSemanticReview": "pending", "evidenceRole": "real_local_9b_with_disclosed_synthetic_business_hosts",
        "wholeSkillAccuracy": None, "largeRuntimeABUnlocked": False})
    write_artifacts(output / "summary", {"report.json": report})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("packet", "compilation", "review", "output"):
        parser.add_argument(key)
    parser.add_argument("--case", required=True)
    parser.add_argument("--max-model-calls", type=int, default=0)
    a = parser.parse_args()
    report = run(a.packet, a.compilation, a.review, a.output, case=a.case, max_model_calls=a.max_model_calls)
    print(report["reportDigest"], report["execution"]["status"])


if __name__ == "__main__":
    main()
