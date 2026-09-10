"""Dual-driver mechanics with synthetic callbacks, never translation accuracy."""
import copy
import threading
import time
from dataclasses import replace

import pytest

from evaluation.structured_flow_demo import fixture, context, host_bindings
from evaluation.structured_flow_tree import compile_structured_tree
from network_runtime.contracts import sha256_json
from network_runtime.l0.hybrid import GovernedHybridFlow, qualify_hybrid
from network_runtime.l0.hybrid_execution import (
    HostCandidateGate, HostHybridConsent, HostReasoningBinding, ReasoningReply, context_digest, run_hybrid,
)


def obj(fields):
    return {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}


def ref(source, pointer=""):
    return {"kind": "reference", "source": source, "pointer": pointer}


def setup():
    bundle, tree, reads_by_name, effects = fixture()
    flow = compile_structured_tree(bundle, tree, reads_by_name, effects)["flow"]
    flow["nodes"] = [flow["nodes"][0], {"kind": "end", "id": "finish", "outcome": "needs_l1", "explanation": "Retained reasoning duty"}]
    flow["nodes"][0]["next"] = "finish"
    config = sha256_json({"temperature": 0})
    read_output = reads_by_name["get_interfaces"].spec.output_schema
    result_schema = obj({"summary": {"type": "string", "maxLength": 500}})
    raw = {"api_version": "netopyu.io/governed-hybrid/v1", "source_digest": bundle["bundleDigest"],
        "task_digest": sha256_json("Synthetic dual-driver task"), "purpose": "Synthetic governed diagnosis only",
        "input_schema": flow["input_schema"], "max_parallel": 1, "max_model_calls": 1, "timeout_seconds": 3,
        "failure_policy": "stop_no_downstream", "outputs": ["draft"], "nodes": [
            {"kind": "strict_region", "id": "observe", "depends_on": [], "inputs": ref("input"), "flow": flow},
            {"kind": "reason", "id": "draft", "depends_on": ["observe"],
             "inputs": ref("observe", "/observations/node-0"), "input_schema": read_output, "output_schema": result_schema,
             "instructions": "Explain only these observed interfaces; do not operate tools or assert repair.",
             "binding_id": "local-model", "model": "synthetic-model", "configuration_digest": config,
             "timeout_seconds": 1, "max_input_bytes": 10000, "max_output_bytes": 1000, "max_output_tokens": 256}]}
    calls, requests = [], []
    bindings = host_bindings(reads_by_name, calls)
    reads = {c.contract_hash: c for c in reads_by_name.values()}
    ctx = replace(context(), scopes=context().scopes | {"reasoning:invoke", "candidate:admit"})
    def model(request):
        requests.append(copy.deepcopy(request))
        return ReasoningReply({"summary": "Candidate explanation; no repair performed."}, "synthetic-model", config, 50, 12)
    reasoners = {"local-model": HostReasoningBinding("synthetic-model", config, model)}
    return raw, reads, bindings, reasoners, ctx, calls, requests


def execute(raw, reads, bindings, reasoners, ctx, *, gates=None, consent=None):
    flow = GovernedHybridFlow.model_validate(raw)
    packet = qualify_hybrid(flow, reads)
    args = {"device": {"id": "lab-sw1"}}
    consent = consent or HostHybridConsent(packet["graphDigest"], sha256_json(args), context_digest(ctx))
    return run_hybrid(flow, args, reads=reads, read_bindings=bindings, reasoners=reasoners, gates=gates or {}, context=ctx, consent=consent)


def test_read_reason_candidates_use_original_gateway_and_exact_input_projection():
    raw, reads, bindings, reasoners, ctx, calls, requests = setup()
    result = execute(raw, reads, bindings, reasoners, ctx)
    assert result["status"] == "governed_graph_completed" and len(calls) == len(requests) == 1
    assert set(requests[0]["inputs"]) == {"interfaces"}
    assert requests[0]["tools"] == [] and not requests[0]["runtimeAuthorityGranted"]
    assert result["outputs"]["draft"]["role"] == "model_candidate"
    assert not result["wholeSkillCorrectnessProven"] and not result["effectExecuted"]
    assert not result["modelOutputsAreFacts"]
    read_event = next(r for r in result["trace"] if r.get("regionReport"))
    assert read_event["regionReport"]["trace"][0]["receipt"]["contractHash"] in reads


@pytest.mark.parametrize("mutation", ["scope", "model", "config", "read_binding", "consent", "identity"])
def test_all_host_bindings_checked_before_any_callback(mutation):
    raw, reads, bindings, reasoners, ctx, calls, requests = setup()
    consent = None
    if mutation == "scope":
        ctx = replace(ctx, scopes=frozenset({"network:read"}))
    elif mutation == "model":
        reasoners["local-model"] = replace(reasoners["local-model"], model="different")
    elif mutation == "config":
        raw["nodes"][1]["configuration_digest"] = sha256_json("changed")
    elif mutation == "read_binding":
        bindings = {}
    elif mutation == "identity":
        ctx = replace(ctx, authenticated=False)
    else:
        consent = HostHybridConsent("wrong", "wrong", "wrong")
    with pytest.raises(PermissionError):
        execute(raw, reads, bindings, reasoners, ctx, consent=consent)
    assert not calls and not requests


@pytest.mark.parametrize("candidate", [{"summary": "claim", "approved": True}, {"summary": 9},
                                      {"summary": "x" * 501}, {"summary": "x", "tools": ["exec"]}])
def test_output_schema_and_authority_injection_never_reaches_downstream(candidate):
    raw, reads, bindings, reasoners, ctx, _, _ = setup()
    binding = reasoners["local-model"]
    reasoners["local-model"] = replace(binding, invoke=lambda request: ReasoningReply(candidate, binding.model, binding.configuration_digest))
    result = execute(raw, reads, bindings, reasoners, ctx)
    assert result["status"] == "blocked" and "outputs" not in result
    assert not result["effectExecuted"]


@pytest.mark.parametrize("mutation", ["cycle", "no_dependency", "unknown_output", "model_budget", "unused_node"])
def test_graph_constraints_are_not_model_editable(mutation):
    raw, reads, *_ = setup()
    if mutation == "cycle":
        raw["nodes"][0]["depends_on"] = ["draft"]
    elif mutation == "no_dependency":
        raw["nodes"][1]["depends_on"] = []
    elif mutation == "unknown_output":
        raw["outputs"] = ["made-up"]
    elif mutation == "model_budget":
        raw["max_model_calls"] = 0
    else:
        raw["outputs"] = ["observe"]
    with pytest.raises(ValueError):
        qualify_hybrid(GovernedHybridFlow.model_validate(raw), reads)


def test_provider_mutation_does_not_change_graph_or_original_observations():
    raw, reads, bindings, reasoners, ctx, _, _ = setup()
    before = copy.deepcopy(raw)
    binding = reasoners["local-model"]
    def mutate(request):
        request["inputs"]["interfaces"][0]["name"] = "forged"
        request["outputSchema"]["properties"]["summary"]["type"] = "integer"
        return ReasoningReply({"summary": "still only a candidate"}, binding.model, binding.configuration_digest)
    reasoners["local-model"] = replace(binding, invoke=mutate)
    result = execute(raw, reads, bindings, reasoners, ctx)
    assert raw == before and result["status"] == "governed_graph_completed"
    event = next(r for r in result["trace"] if r.get("regionReport"))
    assert event["regionReport"]["trace"][0]["receipt"]["payload"]["interfaces"][0]["name"] == "eth0"


def candidate_chain():
    raw, reads, bindings, reasoners, ctx, calls, requests = setup()
    raw["nodes"][1]["output_schema"] = raw["input_schema"]
    binding = reasoners["local-model"]
    reasoners["local-model"] = replace(binding, invoke=lambda request: ReasoningReply(
        {"device": {"id": "lab-sw1"}}, binding.model, binding.configuration_digest))
    policy = sha256_json("strict current-device identity check")
    raw["nodes"].append({"kind": "admit_candidate", "id": "check", "depends_on": ["draft"], "candidate": "draft",
        "inputs": ref("input"), "input_schema": raw["input_schema"], "gate_id": "device-policy", "policy_digest": policy})
    raw["nodes"].append({**copy.deepcopy(raw["nodes"][0]), "id": "observe-again", "depends_on": ["check"], "inputs": ref("check")})
    raw["outputs"] = ["observe-again"]
    packet = qualify_hybrid(GovernedHybridFlow.model_validate(raw), reads)
    gate = HostCandidateGate(policy, lambda candidate, observed: candidate == observed,
                            frozenset({packet["regions"]["observe-again"]["flowDigest"]}))
    return raw, reads, bindings, reasoners, ctx, calls, requests, {"device-policy": gate}


def test_candidate_must_cross_independent_gate_bound_to_exact_strict_region():
    raw, reads, bindings, reasoners, ctx, calls, _, gates = candidate_chain()
    result = execute(raw, reads, bindings, reasoners, ctx, gates=gates)
    assert result["status"] == "governed_graph_completed" and len(calls) == 2
    gate_event = next(r for r in result["trace"] if r.get("kind") == "admit_candidate")
    assert not gate_event["observedFact"] and not gate_event["effectAuthorized"]
    raw["nodes"][-1]["inputs"] = ref("draft")
    with pytest.raises(ValueError, match="directly"):
        qualify_hybrid(GovernedHybridFlow.model_validate(raw), reads)


@pytest.mark.parametrize("decision", [False, "true", {"accepted": True}, None])
def test_rejected_or_forged_gate_verdict_never_runs_next_read(decision):
    raw, reads, bindings, reasoners, ctx, calls, _, gates = candidate_chain()
    gates["device-policy"] = replace(gates["device-policy"], validate=lambda candidate, observed: decision)
    result = execute(raw, reads, bindings, reasoners, ctx, gates=gates)
    assert result["status"] == "blocked" and len(calls) == 1


def test_candidate_cannot_authorize_itself_or_widen_gate_target():
    raw, reads, bindings, reasoners, ctx, calls, _, gates = candidate_chain()
    gates["device-policy"] = replace(gates["device-policy"], allowed_region_digests=frozenset())
    with pytest.raises(PermissionError, match="exact region"):
        execute(raw, reads, bindings, reasoners, ctx, gates=gates)
    assert not calls
    raw["nodes"][2]["inputs"] = ref("draft")
    with pytest.raises(ValueError, match="own independent"):
        qualify_hybrid(GovernedHybridFlow.model_validate(raw), reads)


def test_slow_reasoning_does_not_refresh_evidence_before_admission():
    raw, reads, bindings, reasoners, ctx, calls, _, gates = candidate_chain()
    raw["nodes"][0]["flow"]["max_read_age_seconds"] = 0.03
    binding = reasoners["local-model"]
    original = binding.invoke
    def slow(request):
        time.sleep(0.07)
        return original(request)
    reasoners["local-model"] = replace(binding, invoke=slow)
    result = execute(raw, reads, bindings, reasoners, ctx, gates=gates)
    assert result["status"] == "blocked" and result["blocked"]["code"] == "hybrid_evidence_expired"
    assert len(calls) == 1


def test_late_reply_cannot_revive_a_timed_out_run():
    raw, reads, bindings, reasoners, ctx, _, _ = setup()
    raw["nodes"][1]["timeout_seconds"] = 0.01
    finished = threading.Event()
    binding = reasoners["local-model"]
    def slow(request):
        time.sleep(0.05)
        finished.set()
        return ReasoningReply({"summary": "too late"}, binding.model, binding.configuration_digest)
    reasoners["local-model"] = replace(binding, invoke=slow)
    result = execute(raw, reads, bindings, reasoners, ctx)
    frozen = copy.deepcopy(result)
    assert result["status"] == "blocked" and result["pendingResultsIgnored"] == ["draft"]
    assert not result["providerCancellationProven"] and "outputs" not in result
    assert finished.wait(1) and result == frozen


def test_parallel_model_tasks_join_without_promoting_candidates_to_facts():
    raw, reads, bindings, reasoners, ctx, _, requests = setup()
    raw["max_parallel"], raw["max_model_calls"] = 2, 2
    raw["nodes"].append({**copy.deepcopy(raw["nodes"][1]), "id": "second-draft"})
    result_schema = obj({"first": raw["nodes"][1]["output_schema"], "second": raw["nodes"][1]["output_schema"]})
    raw["nodes"].append({"kind": "join", "id": "join", "depends_on": ["draft", "second-draft"],
        "rule": "all_succeeded", "input_schema": result_schema,
        "inputs": {"kind": "object", "fields": {"first": ref("draft"), "second": ref("second-draft")}}})
    raw["outputs"] = ["join"]
    barrier = threading.Barrier(2)
    original = reasoners["local-model"].invoke
    def simultaneous(request):
        barrier.wait(timeout=0.5)
        return original(request)
    reasoners["local-model"] = replace(reasoners["local-model"], invoke=simultaneous)
    result = execute(raw, reads, bindings, reasoners, ctx)
    assert result["status"] == "governed_graph_completed" and len(requests) == 2
    assert result["outputs"]["join"]["role"] == "model_candidate"
    assert result["completedNodes"] == ["draft", "join", "observe", "second-draft"]


def test_required_parallel_failure_cannot_skip_join():
    raw, reads, bindings, reasoners, ctx, _, requests = setup()
    raw["max_parallel"], raw["max_model_calls"] = 2, 2
    raw["nodes"].append({**copy.deepcopy(raw["nodes"][1]), "id": "failing-draft"})
    raw["nodes"].append({"kind": "join", "id": "join", "depends_on": ["draft", "failing-draft"],
        "rule": "all_succeeded", "input_schema": raw["nodes"][1]["output_schema"], "inputs": ref("draft")})
    raw["outputs"] = ["join"]
    original = reasoners["local-model"].invoke
    def sometimes_fails(request):
        if request["nodeId"] == "failing-draft":
            raise RuntimeError("secret provider error must not be logged")
        return original(request)
    reasoners["local-model"] = replace(reasoners["local-model"], invoke=sometimes_fails)
    result = execute(raw, reads, bindings, reasoners, ctx)
    assert result["status"] == "blocked" and result["blocked"]["node"] == "failing-draft"
    assert not any(r["node"] == "join" for r in result["trace"])
    assert "secret provider" not in str(result) and len(requests) <= 1


def test_join_cannot_launder_a_candidate_into_strict_inputs():
    raw, reads, *_ = candidate_chain()
    raw["nodes"][2] = {"kind": "join", "id": "check", "depends_on": ["draft"],
        "rule": "all_succeeded", "input_schema": raw["input_schema"], "inputs": ref("draft")}
    with pytest.raises(ValueError, match="directly"):
        qualify_hybrid(GovernedHybridFlow.model_validate(raw), reads)


@pytest.mark.parametrize("failure", ["model", "config", "tokens", "bytes", "input_bytes", "false_identity"])
def test_model_receipt_and_budget_checks(failure):
    raw, reads, bindings, reasoners, ctx, _, requests = setup()
    original = reasoners["local-model"].invoke
    def invalid_reply(request):
        reply = original(request)
        if failure == "model":
            return replace(reply, model="unknown")
        if failure == "config":
            return replace(reply, configuration_digest=sha256_json("different"))
        if failure == "tokens":
            return replace(reply, output_tokens=257)
        return reply
    reasoners["local-model"] = replace(reasoners["local-model"], invoke=invalid_reply)
    if failure == "bytes":
        raw["nodes"][1]["max_output_bytes"] = 32
    if failure == "input_bytes":
        raw["nodes"][1]["max_input_bytes"] = 64
    if failure == "false_identity":
        ctx = replace(ctx, authenticated="true")
        with pytest.raises(PermissionError):
            execute(raw, reads, bindings, reasoners, ctx)
        assert not requests
    else:
        result = execute(raw, reads, bindings, reasoners, ctx)
        assert result["status"] == "blocked" and result["blocked"]["node"] == "draft"
        assert not result["effectExecuted"]
        if failure == "input_bytes":
            assert not requests


@pytest.mark.parametrize("admin_up,expected_reads", [(True, 1), (False, 2)])
def test_existing_deterministic_branch_is_preserved_inside_mixed_region(admin_up, expected_reads):
    raw, _, _, reasoners, ctx, calls, requests = setup()
    bundle, tree, named, _ = fixture()
    tree = tree.model_dump(mode="json")
    inner = tree["steps"][0:1] + tree["steps"][1:]
    inner[1]["when_equal"][1]["otherwise"] = [{"kind": "end", "source": inner[1]["when_equal"][1]["source"],
        "outcome": "needs_l1", "explanation": "Keep repair as a separate unapproved task"}]
    from evaluation.structured_flow_tree import StructuredFlowTree
    compiled = compile_structured_tree(bundle, StructuredFlowTree.model_validate(tree), named, {})
    raw["nodes"][0]["flow"] = compiled["flow"]
    reads = {c.contract_hash: c for c in named.values()}
    bindings = host_bindings(named, calls, interfaces=[{"name": "eth0", "adminUp": admin_up}])
    result = execute(raw, reads, bindings, reasoners, ctx)
    assert result["status"] == "governed_graph_completed" and len(calls) == expected_reads and len(requests) == 1
    region = next(r["regionReport"] for r in result["trace"] if "regionReport" in r)
    assert any(r["kind"] == "branch" for r in region["trace"])


def test_read_scope_is_preflighted_even_when_reasoning_scope_exists():
    raw, reads, bindings, reasoners, ctx, calls, requests = setup()
    ctx = replace(ctx, scopes=frozenset({"reasoning:invoke"}))
    with pytest.raises(PermissionError):
        execute(raw, reads, bindings, reasoners, ctx)
    assert not calls and not requests


def test_serial_analysis_may_use_historical_snapshots_but_never_claim_fresh_facts():
    raw, reads, bindings, reasoners, ctx, _, requests = setup()
    raw["nodes"][0]["flow"]["max_read_age_seconds"] = 0.02
    raw["max_model_calls"] = 2
    raw["nodes"].append({**copy.deepcopy(raw["nodes"][1]), "id": "second", "depends_on": ["draft"],
        "inputs": ref("draft"), "input_schema": raw["nodes"][1]["output_schema"]})
    raw["outputs"] = ["second"]
    original = reasoners["local-model"].invoke
    def slow(request):
        time.sleep(0.04)
        return original(request)
    reasoners["local-model"] = replace(reasoners["local-model"], invoke=slow)
    result = execute(raw, reads, bindings, reasoners, ctx)
    assert result["status"] == "governed_graph_completed" and len(requests) == 2
    assert requests[1]["observationAgesAtStartMs"]["observe"] >= 20
    assert requests[1]["evidencePolicy"] == "historical_snapshot_analysis_not_current_action_evidence"
    assert result["outputs"]["second"]["role"] == "model_candidate"


def test_unresolved_strict_region_handoff_cannot_be_bypassed_by_dependency_only():
    raw, reads, bindings, reasoners, ctx, calls, requests = setup()
    first = raw["nodes"][0]
    raw["nodes"] = [first, {**copy.deepcopy(first), "id": "next-read", "depends_on": [first["id"]]}]
    raw["outputs"] = ["next-read"]
    raw["max_model_calls"] = 0
    result = execute(raw, reads, bindings, reasoners, ctx)
    assert result["status"] == "blocked" and result["blocked"]["code"] == "unresolved_region_handoff"
    assert len(calls) == 1 and not requests
