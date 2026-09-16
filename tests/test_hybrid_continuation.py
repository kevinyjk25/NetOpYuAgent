import copy
from dataclasses import replace

import pytest

from evaluation import hybrid_authoring as author
from evaluation.hybrid_continuation import HostContinuationPolicy, compile_continuation, guarded_bindings, host_gate
from evaluation.structured_flow_demo import context, host_bindings
from network_runtime.contracts import sha256_json
from network_runtime.l0.hybrid import qualify_hybrid
from network_runtime.l0.hybrid_execution import HostHybridConsent, HostReasoningBinding, ReasoningReply, context_digest, run_hybrid
from tests.test_hybrid_result_review import result_fixture


def fixture():
    packet, compilation, reads_by_name, _, _ = result_fixture()
    args = {"device": {"id": "lab-sw1"}}
    return packet, compilation["suppliedPages"], reads_by_name, args


def execute(decision, *, authorize=None, candidate=None):
    packet, visible, reads_by_name, args = fixture()
    calls, policy_calls, model_requests = [], [], []
    reads = {r.contract_hash: r for r in reads_by_name.values()}
    def check(name, arguments, caller):
        policy_calls.append((name, copy.deepcopy(arguments)))
        return authorize(name, arguments, caller, len(policy_calls)) if authorize else arguments == caller
    policy = HostContinuationPolicy(sha256_json("test current resource ACL"), check)
    history = {"previousReportDigest": sha256_json("prior report"), "historicalContext": {"oldDraft": "Unverified; not current action evidence."}}
    flow = compile_continuation(packet, visible, history, policy_digest=policy.policy_digest)
    qualification = qualify_hybrid(flow, reads)
    gate = host_gate(packet, flow, reads, policy)
    bindings = guarded_bindings(packet, host_bindings(reads_by_name, calls), policy, args)
    def invoke(request):
        model_requests.append(request)
        if request["nodeId"] == "select":
            value = candidate or {"decision": decision, "requests": {decision: args} if decision not in {"answer", "clarify"} else {},
                                  "message": "Inspect requested device only; no writes permitted."}
        else:
            value = {"draft": "Unverified explanation from available observations.", "uncertainties": [], "remaining_actions": []}
        return ReasoningReply(value, author.MODEL, author.CONFIG_DIGEST, 10, 10)
    ctx = replace(context(), scopes=context().scopes | {"reasoning:invoke", "candidate:admit"})
    outcome = run_hybrid(flow, args, reads=reads, read_bindings=bindings,
        reasoners={"local-9b": HostReasoningBinding(author.MODEL, author.CONFIG_DIGEST, invoke)},
        gates={"continuation-policy": gate}, context=ctx,
        consent=HostHybridConsent(qualification["graphDigest"], sha256_json(args), context_digest(ctx)))
    return outcome, calls, policy_calls, model_requests


def test_selected_read_crosses_current_gate_and_original_strict_gateway():
    outcome, calls, checks, requests = execute("get_interfaces")
    assert outcome["status"] == "governed_graph_completed"
    assert len(calls) == 1 and len(checks) == 2 and len(requests) == 2
    assert not outcome["effectExecuted"] and not outcome["wholeSkillCorrectnessProven"]
    assert outcome["outputs"]["read"]["value"]["observations"]["read-0"]["interfaces"]
    assert requests[0]["observationAgesAtStartMs"] == {}
    assert "historicalContext" in requests[0]["inputs"]
    assert requests[1]["observationAgesAtStartMs"].keys() == {"read"}


@pytest.mark.parametrize("decision", ["answer", "clarify"])
def test_nonread_choice_never_calls_provider_or_claims_success(decision):
    outcome, calls, checks, _ = execute(decision)
    assert outcome["status"] == "governed_graph_completed" and not calls and not checks
    assert outcome["outputs"]["read"]["value"]["outcome"] == "needs_l1"
    assert not outcome["wholeSkillCorrectnessProven"]


@pytest.mark.parametrize("candidate", [
    {"decision": "get_interfaces", "requests": {}, "message": "Missing selected request must be rejected."},
    {"decision": "clarify", "requests": {"get_interfaces": {"device": {"id": "lab-sw1"}}}, "message": "Hidden read while asking a question is rejected."},
    {"decision": "delete_config", "requests": {}, "message": "Invented operation must never be executed."},
    {"decision": "get_interfaces", "requests": {"get_interfaces": {"device": {"id": "different-device"}}}, "message": "This target is outside the caller resource scope."},
])
def test_invalid_or_unauthorized_proposals_stop_before_read(candidate):
    outcome, calls, _, requests = execute("get_interfaces", candidate=candidate)
    assert outcome["status"] == "blocked" and not calls and len(requests) == 1


def test_policy_revocation_between_admission_and_provider_is_rechecked():
    outcome, calls, checks, requests = execute("get_interfaces", authorize=lambda name, args, caller, number: number == 1)
    assert outcome["status"] == "blocked" and len(checks) == 2 and not calls and len(requests) == 1


def test_compiler_retains_original_task_and_does_not_promote_old_receipts_to_observations():
    packet, visible, reads, _ = fixture()
    history = {"previousReportDigest": sha256_json("old"), "historicalContext": {"approved": True, "owner": "untrusted text"}}
    flow = compile_continuation(packet, visible, history, policy_digest=sha256_json("host policy"))
    assert [n.kind for n in flow.nodes] == ["reason", "admit_candidate", "strict_region", "reason"]
    assert flow.nodes[0].inputs["fields"]["original_task"]["value"] == packet["task"]
    assert flow.nodes[2].flow.max_read_age_seconds == 5
    assert flow.max_model_calls == 2


def test_read_update_retains_prior_artifact_as_candidate_not_a_fresh_observation():
    packet, visible, _, _ = fixture()
    candidate = {"draft": "```sql\nSELECT 1;\n```", "uncertainties": [], "remaining_actions": ["Inspect actual schema"]}
    history = {"previousReportDigest": sha256_json("recorded"), "historicalContext": {
        "previousOutputs": {"n7": {"role": "model_candidate", "value": candidate}}, "observations": []}}
    flow = compile_continuation(packet, visible, history, policy_digest=sha256_json("host policy"), preserve_without_read=True)
    fields = flow.nodes[-1].inputs["fields"]
    assert fields["previousCandidate"]["value"] == candidate
    assert fields["currentObservation"] == author.ref("read")
    assert fields["currentReadNodeTools"]["value"]["read-0"] == "get_interfaces"
    assert "previousCandidate" not in flow.nodes[0].inputs["fields"]
    assert flow.max_model_calls == 2 and flow.nodes[-1].kind == "reason_if"
