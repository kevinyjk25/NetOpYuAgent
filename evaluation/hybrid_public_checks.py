"""Counterfactual wiring checks on reviewed generated public graphs.

Uses synthetic callbacks, NOT new LLM trials or semantic accuracy measurements.
The nominal fake draft is only a schema-shaped test double, never reported as
an actual successful language task. Live demo evidence is collected separately.
"""
from __future__ import annotations

import copy
from dataclasses import replace

from evaluation import hybrid_authoring as author
from evaluation.hybrid_behavior import scenario, bindings_for, context
from evaluation.structured_authoring import seal
from network_runtime.contracts import sha256_json
from network_runtime.l0.hybrid import GovernedHybridFlow, qualify_hybrid
from network_runtime.l0.hybrid_execution import HostHybridConsent, HostReasoningBinding, ReasoningReply, context_digest, run_hybrid


def check(packet, compilation, case):
    actual = author.compile_proposal(packet, compilation["suppliedPages"], compilation["plan"])
    if actual != compilation:
        raise ValueError("compiled source/plan drift")
    flow = GovernedHybridFlow.model_validate(compilation["flow"])
    arguments, fixtures = scenario(case)
    variants = ["nominal_mechanics", "bad_input", "scope_denied", "invalid_observation", "provider_error", "model_configuration_drift", "model_authority_injection"]
    rows = []
    for variant in variants:
        calls, models = [], []
        args, ctx = copy.deepcopy(arguments), context()
        fault = variant if variant in {"provider_error", "invalid_observation"} else None
        fault = "invalid_output" if fault == "invalid_observation" else fault
        reads, bindings = bindings_for(packet, fixtures, calls, fault=fault)
        if variant == "scope_denied":
            ctx = replace(ctx, scopes=ctx.scopes - {"stage2:read"})
        if variant == "bad_input":
            args = {**args, "forged_approval": True}
        def invoke(request):
            models.append(request["nodeId"])
            candidate = {"draft": "SCHEMA TEST DOUBLE, not an actual language result", "uncertainties": [], "remaining_actions": []}
            if variant == "model_authority_injection":
                candidate["approved"] = True
            return ReasoningReply(candidate, author.MODEL, author.CONFIG_DIGEST, 0, 0)
        binding = HostReasoningBinding(author.MODEL,
            sha256_json("forged") if variant == "model_configuration_drift" else author.CONFIG_DIGEST, invoke)
        qualification = qualify_hybrid(flow, reads)
        try:
            result = run_hybrid(flow, args, reads=reads, read_bindings=bindings, reasoners={"local-9b": binding}, gates={}, context=ctx,
                consent=HostHybridConsent(qualification["graphDigest"], sha256_json(args), context_digest(ctx)))
        except (PermissionError, ValueError) as error:
            result = {"status": "blocked_before_execution", "errorType": type(error).__name__, "effectExecuted": False}
        expected = "governed_graph_completed" if variant == "nominal_mechanics" else "blocked"
        passed = result["status"] == expected if expected != "blocked" else result["status"] in {"blocked", "blocked_before_execution"}
        if variant in {"scope_denied", "bad_input", "model_configuration_drift"}:
            passed = passed and not calls and not models
        if variant in {"invalid_observation", "provider_error"}:
            passed = passed and not models
        rows.append({"variant": variant, "passed": passed, "expected": expected, "result": result,
            "simulatedReadCalls": calls, "simulatedModelCalls": models, "actualModelCalls": 0})
    return seal({"case": case, "compilationDigest": compilation["reportDigest"], "rows": rows,
        "passed": sum(row["passed"] for row in rows), "total": len(rows), "actualModelCalls": 0,
        "sourceScriptCalls": 0, "effectCalls": 0, "semanticAccuracy": None,
        "evidenceRole": "generated_public_graph_mechanics_with_synthetic_callbacks_not_language_or_device_accuracy"})
