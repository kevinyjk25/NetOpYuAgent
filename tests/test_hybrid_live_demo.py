import copy
import json
from dataclasses import replace

import pytest

from evaluation import hybrid_authoring as author, hybrid_live_demo as demo
from evaluation.structured_binding_probe import write_artifacts
from evaluation.structured_flow_demo import fixture, context, host_bindings


def packet_and_compilation():
    bundle, tree, reads, _ = fixture()
    packet = {"bundle": bundle, "task": "Inspect future caller device interfaces and draft findings; do not change devices.",
        "taskOrigin": "developer_authored_evaluation_request", "inputSchema": tree.input_schema,
        "catalog": {"tools": [json.loads(next(s.text for s in c.spec.sources if s.role == "tool")) for c in reads.values()]},
        "reads": {name: c.model_dump(mode="json", by_alias=True) for name, c in reads.items()}}
    visible = list(author.pages_for(packet))
    plan = {"mode": "proposal", "intent_summary": "Read future caller device interfaces through the exact declared tool, then explain observations as an unverified candidate.", "steps": [
        {"kind": "read", "id": "n0", "after": [], "evidence": [visible[0], "task"], "tool": "get_interfaces",
            "arguments": {"device": {"caller": "input#/device"}}},
        {"kind": "reason", "id": "n1", "after": ["n0"], "evidence": [visible[0], "task"],
            "assignment": "Explain observed interfaces without claiming a repair or executing additional tools."}],
        "outputs": ["n1"], "boundaries": []}
    return packet, author.compile_proposal(packet, visible, plan), reads


def test_reviewed_graph_runs_read_and_host_reasoner_with_no_tool_authority(tmp_path, monkeypatch):
    packet, compiled, reads = packet_and_compilation()
    review = {"case": "synthetic", "compilationDigest": compiled["reportDigest"], "decision": "admit_local_read_reason_only",
        "reviewKind": "developer_ai_not_independent_gold", "rationale": "Synthetic compiler/mechanism test, not public semantics."}
    paths = tmp_path / "input"
    write_artifacts(paths, {"packet.json": packet, "compiled.json": compiled, "review.json": review})
    calls, requests = [], []
    monkeypatch.setattr(demo, "scenario", lambda _: ({"device": {"id": "lab-sw1"}}, {}))
    monkeypatch.setattr(demo, "bindings_for", lambda p, f, unused: ({c.contract_hash: c for c in reads.values()}, host_bindings(reads, calls)))
    monkeypatch.setattr(demo, "context", lambda: replace(context(), scopes=context().scopes | {"reasoning:invoke"}))
    def fake_once(folder, payload, derive, **kwargs):
        requests.append(copy.deepcopy(payload))
        assert "tools" not in payload["wireRequest"] and payload["governedRequest"]["tools"] == []
        return {"candidate.json": {"draft": "Synthetic explanation", "uncertainties": [], "remaining_actions": []},
                "result": {"inputTokens": 50, "outputTokens": 10}}
    monkeypatch.setattr(demo, "author_once", fake_once)
    output = tmp_path / "run"
    result = demo.run(paths / "packet.json", paths / "compiled.json", paths / "review.json", output,
                      case="synthetic", max_model_calls=1)
    assert len(calls) == len(requests) == 1 and result["execution"]["status"] == "governed_graph_completed"
    assert result["runtimeWallLatencyMs"] >= 0
    assert json.loads((output / "summary/report.json").read_text()) == result
    with pytest.raises(FileExistsError):
        demo.run(paths / "packet.json", paths / "compiled.json", paths / "review.json", output,
                 case="synthetic", max_model_calls=1)


def test_structural_compilation_alone_never_grants_demo_admission(tmp_path):
    packet, compiled, _ = packet_and_compilation()
    files = tmp_path / "input"
    write_artifacts(files, {"p.json": packet, "c.json": compiled, "r.json": {"decision": "approved"}})
    with pytest.raises(ValueError, match="source-reviewed"):
        demo.run(files / "p.json", files / "c.json", files / "r.json", tmp_path / "run", case="synthetic", max_model_calls=1)


def test_generic_fixture_and_review_reuse_exact_automatic_graph(tmp_path, monkeypatch):
    from evaluation.flow_tree_authoring import receipt
    from evaluation.hybrid_snapshot_review import prepare
    from evaluation.hybrid_behavior import context as local_context
    packet, compilation, _ = packet_and_compilation()
    review = {"case": "new-task", "compilationDigest": compilation["reportDigest"], "decision": "admit_local_read_reason_only",
        "reviewKind": "developer_ai_not_independent_gold", "rationale": "Bounded declared read only; no semantic approval."}
    fixtures = {"arguments": {"device": {"id": "lab-sw1"}}, "resources": {
        "get_interfaces": [{"device": {"id": "lab-sw1"}}, {"interfaces": [{"name": "eth0", "adminUp": False}]}]}}
    write_artifacts(tmp_path / "inputs", {"packet.json": packet, "compilation.json": compilation, "review.json": review, "fixture.json": fixtures})
    monkeypatch.setattr(demo, "context", lambda: replace(local_context(), scopes=local_context().scopes | {"network:read", "device_id:lab-sw1", "interface_name:eth0"}))
    def once(folder, payload, derive, **kwargs):
        candidate = {"draft": "eth0 is administratively down; no change executed.", "uncertainties": [], "remaining_actions": []}
        write_artifacts(folder, {"request.json": payload, "candidate.json": candidate})
        (folder / "receipt.json").write_text(json.dumps(receipt(folder)))
        return {"candidate.json": candidate, "result": {"inputTokens": 10, "outputTokens": 10}}
    monkeypatch.setattr(demo, "author_once", once)
    root = tmp_path / "inputs"
    report = demo.run(root / "packet.json", root / "compilation.json", root / "review.json", tmp_path / "run",
                      case="new-task", max_model_calls=1, fixture_path=root / "fixture.json")
    assert report["execution"]["status"] == "governed_graph_completed"
    recorded, previous, supplied, _ = prepare(tmp_path / "run")
    assert recorded == packet and previous == report
    assert supplied["candidate"]["draft"].startswith("eth0")
    assert supplied["observations"] and supplied["original_task"] == packet["task"]


def test_prefix_read_uses_original_engine_and_keeps_original_task_unmodified():
    packet, original, _ = packet_and_compilation()
    read = {k: v for k, v in original["plan"]["steps"][0].items() if k != "kind"}
    raw = {"mode": "read_prefix", "intent_summary": original["plan"]["intent_summary"], "reads": [read], "boundaries": []}
    compiled = author.compile_proposal(packet, original["suppliedPages"], raw)
    assert [n["kind"] for n in compiled["flow"]["nodes"]] == ["strict_region", "reason"]
    assert compiled["flow"]["nodes"][-1]["depends_on"] == ["n0"]
    assert compiled["flow"]["nodes"][-1]["inputs"]["fields"]["original_task"]["value"] == packet["task"]
    assert author.compile_proposal(packet, compiled["suppliedPages"], compiled["plan"]) == compiled


def test_compilation_roundtrip_does_not_depend_on_json_object_order():
    packet, compiled, _ = packet_and_compilation()
    raw = copy.deepcopy(compiled["plan"])
    raw["boundaries"] = [{"evidence": ["task"], "kind": "uncertain_semantics", "explanation": "Open diagnostic duty needs independent semantic review."}]
    first = author.compile_proposal(packet, compiled["suppliedPages"], raw)
    saved_packet = json.loads(json.dumps(packet, sort_keys=True))
    saved = json.loads(json.dumps(first, sort_keys=True))
    assert author.compile_proposal(saved_packet, saved["suppliedPages"], saved["plan"]) == first
