import json
from dataclasses import replace

import pytest

from evaluation import hybrid_continuation_run as runner
from evaluation.structured_authoring import seal
from evaluation.structured_flow_demo import context, host_bindings
from network_runtime.contracts import sha256_json
from tests.test_hybrid_live_demo import packet_and_compilation


@pytest.mark.parametrize("mode", ["read_then_answer", "clarify", "repeat", "invalid", "budget"])
def test_bounded_runner_preserves_failures_cost_and_stop_conditions(tmp_path, monkeypatch, mode):
    packet, compilation, reads = packet_and_compilation()
    args = {"device": {"id": "lab-sw1"}}
    prior = {"draft": "Original unverified answer; which observation is missing?", "uncertainties": [], "remaining_actions": []}
    previous = seal({"compilationDigest": compilation["reportDigest"], "execution": {
        "status": "governed_graph_completed", "argumentsDigest": sha256_json(args),
        "outputs": {"draft": {"role": "model_candidate", "value": prior}}, "trace": []}})
    fixtures = {"get_interfaces": [args, {"interfaces": [{"name": "eth0", "adminUp": False}]}]}
    requests, calls = [], []
    monkeypatch.setattr(runner, "context", lambda: replace(context(), scopes=context().scopes | {"reasoning:invoke", "candidate:admit"}))
    monkeypatch.setattr(runner, "bindings_for", lambda p, f, records: ({r.contract_hash: r for r in reads.values()}, host_bindings(reads, calls)))
    if mode == "budget":
        monkeypatch.setattr(runner, "budget", lambda w: {"accepted": False})

    def once(folder, payload, derive, **kwargs):
        request = payload["governedRequest"]
        requests.append(request)
        if request["nodeId"] == "select":
            assert "anyOf" in payload["wireRequest"]["format"]
            selections = sum(r["nodeId"] == "select" for r in requests)
            decision = "clarify" if mode == "clarify" else "answer" if selections == 2 and mode == "read_then_answer" else "get_interfaces"
            value = {"decision": decision, "requests": {decision: args} if decision == "get_interfaces" else {},
                     "message": "Bounded selection from declared tools or a precise question."}
            if mode == "invalid":
                value["approved"] = True
        else:
            value = {"draft": "Candidate with disclosed limitations.", "uncertainties": [], "remaining_actions": []}
        envelope = {"httpStatus": 200, "latencyMs": 1, "body": json.dumps({"model": "qwen3.5:9b", "done": True,
            "done_reason": "stop", "message": {"content": json.dumps(value)}, "prompt_eval_count": 10, "eval_count": 10})}
        files, cost = derive(envelope)
        return {**files, "result": cost}
    monkeypatch.setattr(runner, "author_once", once)
    result = runner.run(packet, compilation, previous, args, fixtures, tmp_path / "run", max_rounds=2)
    assert not result["completeAnswerApproved"] and result["semanticSuccess"] is None
    assert len(requests) <= 4 and len(result["modelCalls"]) == len(requests)
    assert json.loads((tmp_path / "run/summary/report.json").read_text()) == result
    if mode == "read_then_answer":
        assert len(requests) == 3 and len(calls) == 1 and result["status"] == "bounded_continuation_completed"
    elif mode == "clarify":
        assert len(requests) == 1 and not calls
        assert result["lastExecution"]["outputs"]["deliver"]["value"] == prior
        assert any(event.get("modelInvoked") is False for event in result["lastExecution"]["trace"])
    elif mode == "repeat":
        assert len(requests) == 3 and len(calls) == 1 and result["status"] == "blocked"
    else:
        assert not calls and result["status"] == "blocked"
    with pytest.raises(FileExistsError):
        runner.run(packet, compilation, previous, args, fixtures, tmp_path / "run", max_rounds=2)


def test_resource_fixture_matches_arbitrary_registered_objects_not_one_business_answer():
    fixtures = {"read": {"resources": [[{"path": "/project/a"}, {"text": "alpha"}], [{"path": "/project/b"}, {"text": "beta"}]]}}
    policy = runner.fixture_policy(fixtures)
    assert policy.authorize("read", {"path": "/project/a"}, {}) is True
    assert policy.authorize("read", {"path": "/project/b"}, {}) is True
    assert policy.authorize("read", {"path": "/project/../secret"}, {}) is False
    assert policy.authorize("write", {"path": "/project/a"}, {}) is False


def test_review_collision_refuses_before_compilation_or_model_work(tmp_path):
    (tmp_path / "run-final-review").mkdir()
    with pytest.raises(FileExistsError, match="before any model work"):
        runner.run({}, {}, {}, {}, {}, tmp_path / "run", max_rounds=1, final_review_calls=3)
    assert not (tmp_path / "run").exists()


def test_pipeline_freeze_includes_review_and_revision_dependencies():
    pinned = runner.pipeline_implementation()
    assert {"evaluation/hybrid_draft_loop.py", "evaluation/hybrid_draft_review.py",
            "evaluation/hybrid_draft_slots.py", "evaluation/hybrid_snapshot_review.py"} <= pinned.keys()


def test_historical_read_projection_keeps_exact_contract_bound_arguments_and_result():
    from tests.test_hybrid_continuation import execute
    packet, _, _ = packet_and_compilation()
    execution, _, _, _ = execute("get_interfaces")
    report = seal({"execution": execution})
    projected = runner.historical_context(report, packet)
    row = projected["historicalContext"]["observations"][0]
    assert row["tool"] == "get_interfaces" and row["arguments"] == {"device": {"id": "lab-sw1"}}
    assert row["receipt"]["payload"]["interfaces"][0]["name"] == "eth0"
    assert not projected["historicalContext"]["currentActionAuthority"]
    for event in execution["trace"]:
        for step in event.get("regionReport", {}).get("trace", []):
            if step.get("kind") == "read":
                step["argumentBinding"]["arguments"]["device"]["id"] = "wrong-target"
    with pytest.raises(ValueError, match="contract-bound arguments"):
        runner.historical_context(seal({"execution": execution}), packet)


@pytest.mark.parametrize("decision", ["clarify", "answer"])
def test_retained_no_read_answer_has_reviewable_provenance_without_a_fake_model_receipt(tmp_path, monkeypatch, decision):
    from evaluation.flow_tree_authoring import receipt
    from evaluation.hybrid_snapshot_review import prepare
    from evaluation.structured_binding_probe import write_artifacts
    packet, compilation, reads = packet_and_compilation()
    args = {"device": {"id": "lab-sw1"}}
    draft = {"draft": "Preserved answer. Which environment is affected?", "uncertainties": ["Environment is unknown."],
             "remaining_actions": ["Ask the user for the environment."]}
    previous = seal({"compilationDigest": compilation["reportDigest"], "execution": {
        "status": "governed_graph_completed", "argumentsDigest": sha256_json(args),
        "outputs": {"draft": {"role": "model_candidate", "value": draft}}, "trace": []}})
    calls = []
    monkeypatch.setattr(runner, "context", lambda: replace(context(), scopes=context().scopes | {"reasoning:invoke", "candidate:admit"}))
    monkeypatch.setattr(runner, "bindings_for", lambda p, f, records: ({r.contract_hash: r for r in reads.values()}, host_bindings(reads, calls)))
    def once(folder, payload, derive, **kwargs):
        assert payload["governedRequest"]["nodeId"] == "select"
        value = {"decision": decision, "requests": {}, "message": "Which exact environment should be inspected?" if decision == "clarify" else
                 "A selector's invented success must not become answer evidence."}
        envelope = {"httpStatus": 200, "latencyMs": 1, "body": json.dumps({"model": "qwen3.5:9b", "done": True,
            "done_reason": "stop", "message": {"content": json.dumps(value)}, "prompt_eval_count": 10, "eval_count": 10})}
        files, cost = derive(envelope)
        write_artifacts(folder, {"request.json": payload, "response.json": envelope, "result.json": cost, **files})
        (folder / "receipt.json").write_text(json.dumps(receipt(folder)))
        return {**files, "result": cost}
    monkeypatch.setattr(runner, "author_once", once)
    root = tmp_path / "run"
    runner.run(packet, compilation, previous, args, {}, root, max_rounds=1)
    _, _, supplied, _ = prepare(root)
    assert supplied["candidate"]["draft"] == draft["draft"]
    assert supplied["candidate"]["notes"] == draft["uncertainties"]
    assert ("Which exact environment" in supplied["open_duties"]) == (decision == "clarify")
    assert "invented success" not in json.dumps(supplied)
    if decision == "answer":
        recorded = json.loads((root / "round-1/model/select/candidate.json").read_text())
        assert "invented success" in recorded["message"]
    assert not calls and not (root / "round-1/model/deliver").exists()
