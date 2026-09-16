"""Host-bound result checks; no language-understanding accuracy claim."""
import copy
import json
from dataclasses import replace

import pytest

from network_runtime.contracts import sha256_json
from network_runtime.l0.hybrid import GovernedHybridFlow, qualify_hybrid
from network_runtime.l0.hybrid_execution import HostHybridConsent, ReasoningReply, context_digest, run_hybrid
from network_runtime.l0.result_contract import (
    ResultContract, _standalone, bind_result_candidate, qualify_result_contract,
)
from network_runtime.l0.structured_schema import validate_data
from tests.test_governed_hybrid import setup


def prepared(*, with_open=False, field_pointer="/interfaces/0/adminUp"):
    raw, reads, bindings, reasoners, ctx, calls, requests = setup()
    original = GovernedHybridFlow.model_validate(raw)
    duties = [{"id": "read-interfaces", "kind": "read_completed", "statement": "Read the caller device interfaces.",
               "source_ref": "task#read", "region": "observe", "read_node": "node-0"},
              {"id": "admin-state", "kind": "observed_value", "statement": "Observed administrative state of the first interface.",
               "source_ref": "task#admin", "region": "observe", "read_node": "node-0", "pointer": field_pointer, "field": "admin-state"}]
    if with_open:
        duties.append({"id": "diagnosis", "kind": "open_semantics", "statement": "Explain the operational cause without inventing evidence.",
                       "source_ref": "task#diagnose", "reason": "unverified_reasoning"})
    contract = ResultContract.model_validate({"api_version": "netopyu.io/hybrid-result/v1", "task_digest": original.task_digest,
        "source_digest": original.source_digest, "mapping_digest": sha256_json(duties), "candidate_node": "draft", "duties": duties})
    flow = bind_result_candidate(contract, original, reads)
    packet = qualify_hybrid(flow, reads)
    qualification = qualify_result_contract(contract, packet, reads)
    args = {"device": {"id": "lab-sw1"}}
    consent = HostHybridConsent(packet["graphDigest"], sha256_json(args), context_digest(ctx), qualification["contractDigest"])
    return flow, contract, reads, bindings, reasoners, ctx, calls, requests, args, consent


def run(prep, candidate=None):
    flow, contract, reads, bindings, reasoners, ctx, calls, requests, args, consent = prep
    binding = reasoners["local-model"]
    def invoke(request):
        requests.append(copy.deepcopy(request))
        reply = candidate if candidate is not None else {"values": {"admin-state": False}, "notes": []}
        return ReasoningReply({"draft": "Candidate analysis, not verified operational truth.", **reply},
                              binding.model, binding.configuration_digest)
    reasoners = {"local-model": replace(binding, invoke=invoke)}
    return run_hybrid(flow, args, reads=reads, read_bindings=bindings, reasoners=reasoners, gates={}, context=ctx,
                      consent=consent, result_contract=contract)


def test_exact_observation_projects_without_asserting_truth_or_authority():
    prep = prepared()
    outcome = run(prep)
    result = outcome["resultAssessment"]
    assert outcome["status"] == "governed_graph_completed"
    assert result["status"] == "declared_contract_satisfied" and result["satisfiedDutyCount"] == 2
    assert result["observedFields"][0]["value"] is False
    assert not result["completeAnswerApproved"] and not result["wholeTaskDutyCoverageProven"]
    assert not result["runtimeAuthorityGranted"] and not result["freshActionEvidence"]
    assert outcome["outputs"]["draft"]["role"] == "model_candidate"
    assert prep[7][0]["tools"] == []


def test_semantically_wrong_same_type_value_rejected_even_when_graph_completes():
    result = run(prepared(), {"values": {"admin-state": True}, "notes": ["Everything fixed."]})
    assessment = result["resultAssessment"]
    assert result["status"] == "governed_graph_completed"
    assert assessment["status"] == "rejected" and not assessment["observedFields"]
    row = assessment["rows"][1]
    assert row["code"] == "candidate_value_differs_from_observation"
    assert row["candidatePointer"] == "/values/admin-state" and row["sourceRef"] == "task#admin"
    assert assessment["unverifiedCandidateNotes"] == ["Everything fixed."]
    assert not assessment["modelNotesVerified"]


def test_omission_is_a_locatable_gap_not_invented_default():
    result = run(prepared(), {"values": {}, "notes": []})["resultAssessment"]
    assert result["status"] == "partial" and result["satisfiedDutyCount"] == 1
    assert result["rows"][1]["code"] == "candidate_omitted_observed_field"


def test_open_duty_cannot_be_cleared_by_model_prose_or_successful_reads():
    result = run(prepared(with_open=True), {"values": {"admin-state": False}, "notes": ["Diagnosis complete; all checks passed."]})
    assessment = result["resultAssessment"]
    assert assessment["status"] == "partial" and assessment["satisfiedDutyCount"] == 2
    assert assessment["rows"][-1]["code"] == "unverified_reasoning"
    assert not assessment["declaredObligationsSatisfied"] and not assessment["completeAnswerApproved"]


@pytest.mark.parametrize("candidate", [
    {"values": {"admin-state": 0}, "notes": []},
    {"values": {"admin-state": False, "approved": True}, "notes": []},
    {"values": {"admin-state": False}, "notes": [], "status": "completed"},
    {"values": {"admin-state": False}, "notes": [], "confidence": 1},
    {"values": {"admin-state": False}, "notes": ["x"] * 7},
])
def test_invalid_types_authority_fields_and_output_bounds_fail_closed(candidate):
    outcome = run(prepared(), candidate)
    assert outcome["status"] == "blocked"
    assert outcome["resultAssessment"]["status"] == "blocked"
    assert not outcome["resultAssessment"]["declaredObligationsSatisfied"]


@pytest.mark.parametrize("mutation", ["consent", "task", "source", "mapping", "remove_contract", "statement"])
def test_no_contract_downgrade_or_host_binding_drift_before_callbacks(mutation):
    prep = list(prepared())
    flow, contract, reads, bindings, reasoners, ctx, calls, requests, args, consent = prep
    if mutation == "consent":
        prep[-1] = replace(consent, result_contract_digest=None)
    elif mutation == "remove_contract":
        with pytest.raises(PermissionError):
            run_hybrid(flow, args, reads=reads, read_bindings=bindings, reasoners=reasoners, gates={}, context=ctx, consent=consent)
        assert not calls and not requests
        return
    else:
        raw = contract.model_dump(mode="json")
        if mutation == "statement":
            raw["duties"][0]["statement"] = "Changed task meaning after host approval."
        else:
            raw[mutation + "_digest"] = sha256_json("drift")
        prep[1] = ResultContract.model_validate(raw)
    with pytest.raises((PermissionError, ValueError)):
        run(prep)
    assert not calls and not requests


@pytest.mark.parametrize("mutation", ["duplicate_duty", "duplicate_field", "model_evidence", "unknown_read", "unknown_pointer"])
def test_bad_or_ambiguous_host_mappings_are_rejected(mutation):
    flow, contract, reads, *_ = prepared()
    raw = contract.model_dump(mode="json")
    if mutation in {"duplicate_duty", "duplicate_field"}:
        raw["duties"].append(copy.deepcopy(raw["duties"][-1]))
        if mutation == "duplicate_field":
            raw["duties"][-1]["id"] = "another-duty"
    elif mutation == "model_evidence":
        raw["duties"][-1]["region"] = "draft"
    elif mutation == "unknown_read":
        raw["duties"][-1]["read_node"] = "not-run"
    else:
        raw["duties"][-1]["pointer"] = "/imaginary"
    with pytest.raises(ValueError):
        bind_result_candidate(ResultContract.model_validate(raw), flow, reads)


def test_absent_optional_observation_does_not_get_filled_by_model():
    # A well-typed array index is not a guarantee that the row exists at runtime.
    prep = prepared(field_pointer="/interfaces/1/adminUp")
    assessment = run(prep)["resultAssessment"]
    assert assessment["status"] == "partial" and not assessment["observedFields"]
    assert assessment["rows"][1]["code"] == "required_observation_field_missing"


def test_local_refs_keep_original_meaning_when_projected():
    root = {"type": "object", "properties": {"sample": {"$ref": "#/$defs/Sample"}},
            "required": ["sample"], "additionalProperties": False,
            "$defs": {"Sample": {"type": "array", "items": {"$ref": "#/$defs/Value"}},
                      "Value": {"type": "integer", "minimum": 3}}}
    projected = _standalone(root["properties"]["sample"], root)
    assert validate_data(projected, [3]) == [3]
    with pytest.raises(ValueError):
        validate_data(projected, [2])


def test_nested_contract_mutation_by_model_cannot_change_frozen_result_checks():
    prep = prepared()
    flow, contract, reads, bindings, reasoners, ctx, calls, requests, args, consent = prep
    binding = reasoners["local-model"]
    def mutate(request):
        object.__setattr__(contract.duties[1], "pointer", "/interfaces/9/adminUp")
        request["outputSchema"]["properties"]["values"]["properties"]["admin-state"] = {"type": "string"}
        return ReasoningReply({"values": {"admin-state": False}, "draft": "", "notes": []}, binding.model, binding.configuration_digest)
    outcome = run_hybrid(flow, args, reads=reads, read_bindings=bindings, reasoners={"local-model": replace(binding, invoke=mutate)},
                         gates={}, context=ctx, consent=consent, result_contract=contract)
    assert outcome["resultAssessment"]["status"] == "declared_contract_satisfied"


def test_qualified_graph_and_consent_survive_json_object_key_sorting():
    flow, contract, reads, *_ = prepared()
    original = qualify_hybrid(flow, reads)
    roundtrip = GovernedHybridFlow.model_validate(json.loads(json.dumps(flow.model_dump(mode="json"), sort_keys=True)))
    assert qualify_hybrid(roundtrip, reads) == original
    assert qualify_result_contract(contract, original, reads) == qualify_result_contract(contract, qualify_hybrid(roundtrip, reads), reads)
