import copy
import json
from dataclasses import replace

import pytest

from evaluation import hybrid_live_demo as demo
from evaluation.hybrid_result_review import prepare_binding
from evaluation.structured_binding_probe import write_artifacts
from evaluation.structured_flow_demo import context, host_bindings
from network_runtime.contracts import sha256_json
from tests.test_hybrid_live_demo import packet_and_compilation


def result_fixture():
    packet, compiled, reads = packet_and_compilation()
    duty = {"id": "interfaces", "kind": "observed_value", "statement": "Preserve the observed interface rows without inventing their state.",
            "source_ref": "task", "region": "n0", "read_node": "read", "pointer": "/interfaces", "field": "interfaces"}
    mapping = {"sourceDigest": packet["bundle"]["bundleDigest"], "taskDigest": sha256_json(packet["task"]),
        "reviewKind": "developer_ai_not_independent_gold", "duties": [{**{k: duty[k] for k in ("id", "source_ref", "statement")},
            "source": {"path": "task", "start": 0, "end": len(packet["task"]), "quote": packet["task"]}}]}
    contract = {"api_version": "netopyu.io/hybrid-result/v1", "source_digest": mapping["sourceDigest"],
        "task_digest": mapping["taskDigest"], "mapping_digest": sha256_json(mapping), "candidate_node": "n1", "duties": [duty]}
    return packet, compiled, reads, contract, mapping


@pytest.mark.parametrize("mutation", ["mapping_digest", "quote", "source", "task", "omitted_duty", "duplicate", "statement", "unknown_extra"])
def test_result_mapping_checks_exact_sources_and_declared_duties(mutation):
    packet, compiled, _, contract, mapping = result_fixture()
    if mutation == "quote":
        mapping["duties"][0]["source"]["quote"] = "Invented evidence."
    elif mutation == "source":
        mapping["sourceDigest"] = sha256_json("wrong source")
    elif mutation == "task":
        mapping["taskDigest"] = sha256_json("wrong task")
    elif mutation == "omitted_duty":
        mapping["duties"] = []
    elif mutation == "duplicate":
        mapping["duties"].append(copy.deepcopy(mapping["duties"][0]))
    elif mutation == "statement":
        mapping["duties"][0]["statement"] = "Silently changed meaning in the review dossier."
    elif mutation == "unknown_extra":
        mapping["expectedResult"] = "Hidden answer must not be accepted."
    if mutation != "mapping_digest":
        contract["mapping_digest"] = sha256_json(mapping)
    else:
        contract["mapping_digest"] = sha256_json("drift")
    with pytest.raises(ValueError):
        prepare_binding(packet, compiled, contract, mapping)


def test_reviewed_result_mapping_and_original_graph_run_together(tmp_path, monkeypatch):
    packet, compiled, reads, contract, mapping = result_fixture()
    _, _, qualified = prepare_binding(packet, compiled, contract, mapping)
    review = {"case": "synthetic", "compilationDigest": compiled["reportDigest"], "resultContractDigest": qualified["contractDigest"],
        "decision": "admit_local_read_reason_only", "reviewKind": "developer_ai_not_independent_gold",
        "rationale": "Explicit local result mapping installation; not automatic semantic translation."}
    inputs = tmp_path / "inputs"
    write_artifacts(inputs, {"packet.json": packet, "compiled.json": compiled, "review.json": review,
                            "contract.json": contract, "mapping.json": mapping})
    calls, requests = [], []
    monkeypatch.setattr(demo, "scenario", lambda _: ({"device": {"id": "lab-sw1"}}, {}))
    monkeypatch.setattr(demo, "bindings_for", lambda p, f, unused: ({c.contract_hash: c for c in reads.values()}, host_bindings(reads, calls)))
    monkeypatch.setattr(demo, "context", lambda: replace(context(), scopes=context().scopes | {"reasoning:invoke"}))
    def fake_once(folder, payload, derive, **kwargs):
        request = payload["governedRequest"]
        requests.append(copy.deepcopy(request))
        candidate = {"values": {"interfaces": [{"name": "eth0", "adminUp": False}]}, "draft": "Unverified diagnostic artifact.", "notes": ["Unverified diagnosis."]}
        return {"candidate.json": candidate, "result": {"inputTokens": 50, "outputTokens": 30}}
    monkeypatch.setattr(demo, "author_once", fake_once)
    result = demo.run(inputs / "packet.json", inputs / "compiled.json", inputs / "review.json", tmp_path / "run",
        case="synthetic", max_model_calls=1, result_contract_path=inputs / "contract.json", result_mapping_path=inputs / "mapping.json")
    assessment = result["execution"]["resultAssessment"]
    assert assessment["status"] == "declared_contract_satisfied" and len(calls) == len(requests) == 1
    assert not assessment["completeAnswerApproved"] and not assessment["modelNotesVerified"]
    freeze = json.loads((tmp_path / "run/freeze/inputs.json").read_text())
    assert freeze["resultMappingIsHostAuthoredNotAutomaticTranslation"]
    with pytest.raises(ValueError, match="silent add/remove"):
        demo.run(inputs / "packet.json", inputs / "compiled.json", inputs / "review.json", tmp_path / "no-contract",
                 case="synthetic", max_model_calls=1)
    assert len(calls) == len(requests) == 1
