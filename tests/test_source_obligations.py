"""Typed review wiring, not a model-generalization score."""

import copy
import json

import pytest

from evaluation import flow_checkpoint, source_ledger as ledger, source_obligations as review
from evaluation.source_retrieval import decision_phase, requests_view
from evaluation.structured_flow_demo import fixture


@pytest.fixture
def packet():
    bundle, tree, reads, _ = fixture()
    return {"bundle": bundle, "task": "Draft the read region for an explicit device, preserving source conditions.",
        "taskOrigin": "developer_authored_evaluation_request", "inputSchema": tree.input_schema,
        "catalog": {"tools": [json.loads(next(s.text for s in c.spec.sources if s.role == "tool")) for c in reads.values()]},
        "reads": {n: c.model_dump(by_alias=True, mode="json") for n, c in reads.items()}}


def inspection(wire):
    content = json.loads(wire["messages"][1]["content"])
    block = next(b for b in content["sourceBlocks"] if "Host-bound contract" in b["text"])
    return {"mode": "inspect_obligations", "obligations": [{"source": {"block_id": block["id"]},
        "requirement": "Execution requires the original host binding and explicit identity.", "category": "authorization",
        "phases": ["execution"], "handling": "host_gate", "hostGate": next(iter(content["hostExecutionGates"])),
        "reason": "A source requirement is not an observation that live identity is missing."}]}


def candidate(wire):
    content = json.loads(wire["messages"][1]["content"])
    _, tree, _, _ = fixture()
    raw = tree.model_dump(mode="json")
    first = raw["steps"][0]
    block = next(b for b in content["sourceBlocks"] if first["source"]["quote"] in b["text"])
    first["source"] = {"block_id": block["id"]}
    raw["steps"] = [first, {"kind": "end", "source": first["source"], "outcome": "needs_l1",
                           "explanation": "Untranslated branches still require review."}]
    return {"mode": "candidate", "tree": raw, "remaining": []}


@pytest.fixture
def transport(monkeypatch):
    queue, calls = [], []
    monkeypatch.setattr(ledger.prior.OllamaAnchoredAuthorAdapter, "preflight", lambda self: {"model": ledger.QWEN_MODEL, "digest": "fixture"})
    def send(arm, wire):
        calls.append(wire)
        choice = queue.pop(0)(wire)
        return {"httpStatus": 200, "latencyMs": 10, "body": json.dumps({"model": ledger.QWEN_MODEL,
            "done": True, "done_reason": "stop", "prompt_eval_count": 100, "eval_count": 40,
            "message": {"role": "assistant", "content": json.dumps(choice)}})}
    monkeypatch.setattr(flow_checkpoint, "send", send)
    return queue, calls


def test_inspect_then_construct_preserves_execution_requirements_without_granting_them(packet, tmp_path, transport, monkeypatch):
    queue, calls = transport
    queue.extend([inspection, candidate])
    folder = tmp_path / "run"
    ledger.freeze(packet, folder, profile="obligation_first")
    result = ledger.run(folder, max_new_calls=2)
    assert result["modelCallsRecorded"] == len(calls) == 2 and result["compiledReadNodes"] == 1
    assert result["sourceObligationCount"] == 1 and not result["obligationInspectionVerified"]
    assert not result["runtimeAuthorityGranted"] and result["providerCalls"] == 0
    assert result["semanticAccuracy"] is None
    assert calls[0]["format"]["properties"]["mode"]["const"] == "inspect_obligations"
    content = json.loads(calls[1]["messages"][1]["content"])
    item = content["obligationReview"]["obligations"][0]
    assert item["phases"] == ["execution"] and item["handling"] == "host_gate"
    assert "Execution requires" in content["ledgerNavigation"][item["noteIndex"]]["interpretation"]
    assert not content["hostExecutionGates"][item["hostGate"]]["satisfied"]
    stored = json.loads((folder / "round-000/obligation-review.json").read_text())
    source = stored["obligations"][0]["source"]
    original = next(d["content"] for d in packet["bundle"]["documents"] if d["path"] == source["path"])
    assert original[source["start"]:source["end"]] == source["quote"]
    monkeypatch.setattr(flow_checkpoint, "send", lambda *a: pytest.fail("offline replay called model"))
    assert ledger.run(folder) == result


@pytest.mark.parametrize("issue", ["phase", "category", "invented_gate", "unused_gate", "invented_source", "early_candidate"])
def test_inconsistent_classification_or_stage_skip_fails_without_repair(packet, tmp_path, transport, issue):
    queue, calls = transport
    def broken(wire):
        if issue == "early_candidate":
            return candidate(wire)
        value = inspection(wire)
        item = value["obligations"][0]
        if issue == "phase":
            item["phases"] = ["authoring"]
        elif issue == "category":
            item["category"] = "output_policy"
        elif issue == "invented_gate":
            item["hostGate"] = "invented_gate"
        elif issue == "unused_gate":
            item["handling"] = "flow_proposal"
        else:
            item["source"]["block_id"] = "not_a_source"
        return value
    queue.append(broken)
    folder = tmp_path / "run"
    ledger.freeze(packet, folder, profile="obligation_first")
    result = ledger.run(folder, max_new_calls=6)
    assert result["status"] == "candidate_invalid_or_unresolved" and len(calls) == 1
    assert not result["compiled"] and result["sourceObligationCount"] == 0
    assert (folder / "round-000/response.json").exists()
    assert ledger.run(folder, max_new_calls=6) == result


def test_direct_profile_is_explicit_comparison_not_inspection_evidence(packet, tmp_path, transport):
    queue, _ = transport
    queue.append(candidate)
    folder = tmp_path / "run"
    ledger.freeze(packet, folder, profile="direct")
    result = ledger.run(folder, max_new_calls=1)
    assert result["compiled"] and result["profile"] == "direct" and result["sourceObligationCount"] == 0


@pytest.mark.parametrize("option,expected", [([], "direct"), (["--profile", "obligation_first"], "obligation_first")])
def test_cli_does_not_silently_promote_unverified_review(packet, tmp_path, transport, monkeypatch, option, expected):
    source, root = tmp_path / "input.json", tmp_path / "run"
    source.write_text(json.dumps(packet))
    monkeypatch.setattr("sys.argv", ["source_ledger", "freeze", str(root), "--inputs", str(source), *option])
    ledger.main()
    manifest = ledger.load_manifest(root)
    assert manifest["profile"] == expected
    assert ("obligationReview" in manifest["initialState"]) == (expected == "obligation_first")
    assert transport[1] == []


def test_inspection_counts_toward_total_budget_and_page_delivery_round(packet):
    state = ledger.initial_state(packet, "obligation_first")
    wire, _ = ledger.make_request(packet, state)
    blocks = ledger.citation_blocks(ledger.frame(packet, state))
    before = copy.deepcopy(state)
    result = review.retain(packet, state, inspection(wire), blocks)
    assert state == before and result["inspectionCalls"] == 1
    result["requests"] = [{"fromPages": [], "requestedPages": [], "reason": "Test an explicit subsequent delivery.", "deliveryRound": 2}]
    assert requests_view(result, recorded_rounds=2)[0]["textDelivery"] == "pending"
    assert requests_view(result, recorded_rounds=3)[0]["deliveryRound"] == 2
    result["requests"] *= 4
    assert decision_phase(result, 6) == "last_round_requires_candidate_or_gap"


def test_profile_rejects_unknown_and_is_frozen(packet, tmp_path, transport):
    with pytest.raises(ValueError, match="profile"):
        ledger.freeze(packet, tmp_path / "wrong", profile="skip_all_safety")
    root = tmp_path / "run"
    ledger.freeze(packet, root, profile="obligation_first")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["profile"] = "direct"
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="drift"):
        ledger.run(root)


def test_inspection_excludes_task_parameters_and_host_limitations_but_construction_keeps_them(packet):
    packet["task"] = "PRIVATE_TASK_MARKER_8847 do not copy task values into source rules."
    state = ledger.initial_state(packet, "obligation_first")
    wire, _ = ledger.make_request(packet, state)
    content = json.loads(wire["messages"][1]["content"])
    assert not {"task", "taskOrigin", "inputSchema", "hostCatalog", "hostBindings", "authoringBoundary"} & set(content)
    assert "PRIVATE_TASK_MARKER_8847" not in wire["messages"][1]["content"]
    state = review.retain(packet, state, inspection(wire), ledger.citation_blocks(ledger.frame(packet, state)))
    next_wire, _ = ledger.make_request(packet, state)
    assert json.loads(next_wire["messages"][1]["content"])["task"] == packet["task"]


def test_cross_cutting_rule_retains_multiple_phases_without_claiming_gate_coverage(packet):
    state = ledger.initial_state(packet, "obligation_first")
    wire, _ = ledger.make_request(packet, state)
    choice = inspection(wire)
    item = choice["obligations"][0]
    item.update(phases=["execution", "completion"], category="output_policy", handling="unresolved", hostGate="none")
    result = review.retain(packet, state, choice, ledger.citation_blocks(ledger.frame(packet, state)))
    assert result["obligationReview"]["obligations"][0]["phases"] == ["execution", "completion"]
    assert not result["obligationReview"]["runtimeAuthorityGranted"]


def test_provisional_gap_fetches_unread_inert_source_once_then_preserves_unresolved_result(packet, tmp_path, transport):
    from evaluation.translation_intake import _bundle, _document
    queue, calls = transport
    b = packet["bundle"]
    packet["bundle"] = _bundle({**{k: v for k, v in b.items() if k != "bundleDigest"}, "documents": [*b["documents"],
        _document("reference.md", b"parameter_rule is defined here; live state remains unobserved.", mode="100644", origin="fixture")]})
    def gap(wire):
        content = json.loads(wire["messages"][1]["content"])
        block = next(b for b in content["sourceBlocks"] if len(b["text"]) >= 8)
        return {"mode": "gap_report", "gaps": [{"source": {"block_id": block["id"]}, "category": "source_context",
            "missing": "The `parameter_rule` definition needs to be inspected before deciding.",
            "nextAction": "Inspect retained source definitions; never guess live facts."}]}
    queue.extend([inspection, gap, gap])
    folder = tmp_path / "run"
    ledger.freeze(packet, folder, profile="obligation_first")
    result = ledger.run(folder, max_new_calls=6)
    assert len(calls) == 3 and result["status"] == "source_grounded_gap_requires_review"
    assert result["retrievalRequests"][0]["deliveryRound"] == 2
    assert (folder / "round-001/gap-report.json").exists()
    search = json.loads((folder / "round-001/gap-source-search.json").read_text())
    assert search["selectedPages"] and search["sourceScriptCalls"] == 0
    assert not result["compiled"] and not result["runtimeAuthorityGranted"]
    assert ledger.run(folder) == result
