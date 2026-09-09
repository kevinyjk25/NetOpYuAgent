"""Progressive source/candidate mechanics, not a translated-Skill accuracy score."""

import copy
import json

import pytest

from evaluation import flow_checkpoint as checkpoint
from evaluation import structured_authoring as author
from evaluation.structured_flow_demo import fixture
from evaluation.translation_intake import _bundle, _document


@pytest.fixture
def packet():
    bundle, tree, reads, _ = fixture()
    raw = {k: v for k, v in bundle.items() if k != "bundleDigest"}
    raw["documents"].append(_document("references/extra.md", b"Additional exact source text for review.\n",
                                       mode="100644", origin="fixture"))
    bundle = _bundle(raw)
    return {"bundle": bundle, "task": "Read the current interfaces; retain uncertain duties for review.",
            "taskOrigin": "developer_authored_evaluation_request", "inputSchema": tree.input_schema,
            "catalog": {"tools": [json.loads(next(s.text for s in c.spec.sources if s.role == "tool"))
                                  for c in reads.values()]},
            "reads": {name: c.model_dump(by_alias=True, mode="json") for name, c in reads.items()}}


def view(packet):
    pages = author.pages_for(packet)
    visible = {k for k, p in pages.items() if p["path"] == packet["bundle"]["entryPath"]}
    return pages, visible, author.make_request(packet, pages, visible)


def candidate(packet):
    _, tree, _, _ = fixture()
    pages, visible, _ = view(packet)
    step = tree.model_dump(mode="json")["steps"][0]
    step["source"] = {"page_id": next(iter(visible)), "quote": step["source"]["quote"]}
    return {"mode": "candidate", "remaining": [], "tree": {
        "api_version": tree.api_version, "source_digest": packet["bundle"]["bundleDigest"],
        "purpose": "Synthetic isolated interface read region only", "input_schema": packet["inputSchema"],
        "max_read_age_seconds": 5, "unresolved": [], "steps": [step, {
            "kind": "end", "source": copy.deepcopy(step["source"]), "outcome": "needs_l1",
            "explanation": "Further source duties require L1 review; no whole Skill completion."}]}}


def envelope(value):
    return {"httpStatus": 200, "latencyMs": 10, "body": json.dumps({
        "model": author.QWEN_MODEL, "done": True, "done_reason": "stop",
        "prompt_eval_count": 100, "eval_count": 50,
        "message": {"role": "assistant", "content": json.dumps(value)}})}


@pytest.fixture
def transport(monkeypatch):
    calls, outputs = [], []
    monkeypatch.setattr(author.OllamaAnchoredAuthorAdapter, "preflight", lambda self: {"model": author.QWEN_MODEL, "artifact": "fixture"})
    def send(arm, wire):
        calls.append((arm, wire))
        return envelope(outputs.pop(0))
    monkeypatch.setattr(checkpoint, "send", send)
    return calls, outputs


def test_source_request_then_candidate_compilation_and_zero_call_replay(packet, tmp_path, transport, monkeypatch):
    calls, outputs = transport
    pages, visible, _ = view(packet)
    outputs.extend([{"mode": "request_pages", "pages": sorted(set(pages) - visible),
                     "reason": "Read original reference text before making a candidate."}, candidate(packet)])
    root = tmp_path / "run"
    manifest = author.freeze(packet, root)
    before = copy.deepcopy(packet)
    result = author.run(root, max_new_calls=2)
    assert result["status"] == "compiled_region_requires_semantic_review"
    assert len(calls) == result["modelCallsRecorded"] == 2
    assert result["submittedPages"] == list(pages) and not result["notSubmittedPages"]
    assert result["providerCalls"] == result["sourceScriptCalls"] == 0
    assert result["semanticAccuracy"] is None and not result["runtimeAuthorityGranted"]
    assert result["inputTokens"] == 200 and result["outputTokens"] == 100
    assert result["requestTotalMs"] == 20
    assert packet == before == manifest["inputs"]
    monkeypatch.setattr(checkpoint, "send", lambda *a: pytest.fail("offline replay must not contact a model"))
    assert author.run(root) == result
    lowered = json.loads((root / "round-001/tree.json").read_text())
    source = lowered["steps"][0]["source"]
    original = next(d["content"] for d in packet["bundle"]["documents"] if d["path"] == source["path"])
    assert original[source["start"]:source["end"]] == source["quote"]


@pytest.mark.parametrize("mutation", ["unresolved", "unknown_tool", "unseen_source", "wrong_quote", "wrong_digest", "wrong_schema", "extra", "plain_arguments", "effect"])
def test_bad_candidate_retained_without_retry_or_execution(packet, tmp_path, transport, mutation):
    calls, outputs = transport
    choice = candidate(packet)
    step = choice["tree"]["steps"][0]
    if mutation == "unresolved":
        choice["tree"]["unresolved"] = ["Missing original prerequisite semantics"]
    elif mutation == "unknown_tool":
        step["tool"] = "invented_read"
    elif mutation == "unseen_source":
        pages, visible, _ = view(packet)
        step["source"] = {"page_id": next(iter(set(pages) - visible)), "quote": "Additional exact source text for review."}
    elif mutation == "wrong_quote":
        step["source"]["quote"] = "An invented quotation unsupported by the source."
    elif mutation == "wrong_digest":
        choice["tree"]["source_digest"] = "sha256:" + "f" * 64
    elif mutation == "wrong_schema":
        choice["tree"]["input_schema"] = {"type": "string"}
    elif mutation == "extra":
        choice["semanticAccuracy"] = 1
    elif mutation == "effect":
        choice["tree"]["steps"] = [{"kind": "effect_candidate", "source": step["source"],
                                      "binding_id": "invented", "arguments": step["arguments"]}]
    else:
        step["arguments"] = {"device": {"id": "lab-sw1"}}
    outputs.append(choice)
    root = tmp_path / "run"
    author.freeze(packet, root)
    result = author.run(root, max_new_calls=4)
    assert result["status"] == "candidate_invalid_or_unresolved" and len(calls) == 1
    assert author.run(root, max_new_calls=4) == result and len(calls) == 1
    assert (root / "round-000/response.json").exists()
    assert not (root / "round-000/compilation.json").exists()
    if mutation == "unresolved":
        assert (root / "round-000/review-input.json").exists()


def test_zero_call_budget_then_resume_and_no_progress_page_request(packet, tmp_path, transport):
    calls, outputs = transport
    root = tmp_path / "run"
    author.freeze(packet, root)
    assert author.run(root)["status"] == "new_call_budget_exhausted" and not calls
    _, visible, _ = view(packet)
    outputs.append({"mode": "request_pages", "pages": sorted(visible), "reason": "Repeated request must not trigger another retry."})
    result = author.run(root, max_new_calls=4)
    assert result["status"] == "candidate_invalid_or_unresolved" and len(calls) == 1


@pytest.mark.parametrize("budget", [-1, 5, True, 1.5])
def test_call_budget_checked_before_io(tmp_path, budget):
    with pytest.raises(ValueError, match="budget"):
        author.run(tmp_path / "missing", max_new_calls=budget)


def test_context_budget_never_silently_discards_requested_pages(packet, tmp_path, transport, monkeypatch):
    calls, outputs = transport
    pages, visible, wire = view(packet)
    expanded = author.make_request(packet, pages, set(pages))
    budget = len(json.dumps(wire, ensure_ascii=False).encode())
    assert len(json.dumps(expanded, ensure_ascii=False).encode()) > budget
    monkeypatch.setattr(author, "MAX_WIRE_BYTES", budget)
    root = tmp_path / "run"
    author.freeze(packet, root)
    outputs.append({"mode": "request_pages", "pages": sorted(set(pages) - visible), "reason": "Request the remaining source reference page."})
    result = author.run(root, max_new_calls=4)
    assert result["status"] == "context_budget_stopped_no_truncation" and len(calls) == 1
    assert set(result["notSubmittedPages"]) == set(pages) - visible
    assert not (root / "round-001").exists()


@pytest.mark.parametrize("mutation", ["review_answers", "description", "missing_read", "source", "not_object"])
def test_packet_rejects_review_leakage_and_host_source_drift(packet, mutation):
    if mutation == "review_answers":
        packet["findings"] = []
    elif mutation == "description":
        packet["catalog"]["tools"][0]["description"] = "New semantics"
    elif mutation == "missing_read":
        packet["reads"].pop(next(iter(packet["reads"])))
    elif mutation == "source":
        packet["bundle"]["documents"][0]["content"] += " changed"
    else:
        packet = []
    with pytest.raises(ValueError):
        author.validate_inputs(packet)


@pytest.mark.parametrize("mutation", ["implementation", "initial_pages", "authority", "partial", "model"])
def test_frozen_drift_rejected_without_model_calls(packet, tmp_path, transport, monkeypatch, mutation):
    calls, _ = transport
    root = tmp_path / "run"
    manifest = author.freeze(packet, root)
    if mutation == "model":
        monkeypatch.setattr(author.OllamaAnchoredAuthorAdapter, "preflight", lambda self: {"model": "changed"})
    elif mutation == "partial":
        (root / "round-000").mkdir()
    else:
        if mutation == "implementation":
            manifest["implementation"] = {}
        elif mutation == "initial_pages":
            manifest["initialPages"] = []
        else:
            manifest["runtimeAuthorityGranted"] = True
        manifest = author.seal({k: v for k, v in manifest.items() if k != "reportDigest"})
        (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        author.run(root, max_new_calls=4)
    assert not calls


def test_duplicate_json_keys_and_nonunique_quote_are_not_repaired(packet):
    pages, visible, wire = view(packet)
    raw = envelope({})
    response = json.loads(raw["body"])
    response["message"]["content"] = '{"mode":"candidate","mode":"request_pages"}'
    raw["body"] = json.dumps(response)
    _, result = author.derive(packet, pages, visible, wire, raw)
    assert result["candidateStatus"] == "candidate_invalid_or_unresolved"
    key = next(iter(visible))
    pages[key]["text"] = "duplicate exact source\nduplicate exact source"
    with pytest.raises(ValueError, match="exactly once"):
        author.source_mark({"page_id": key, "quote": "duplicate exact source"}, pages, visible)


def test_literal_source_keys_are_not_lowered_as_statement_citations(packet):
    pages, visible, wire = view(packet)
    choice = candidate(packet)
    payload = {"source": {"page_id": "not-a-page", "quote": "literal value"}}
    choice["tree"]["steps"][0]["arguments"] = {"kind": "literal", "value": payload}
    files, result = author.derive(packet, pages, visible, wire, envelope(choice))
    assert files["tree.json"]["steps"][0]["arguments"]["value"] == payload
    assert result["candidateStatus"] == "candidate_invalid_or_unresolved"  # tool schema, not citation


def test_existing_manifest_rejected_before_preflight(packet, tmp_path, monkeypatch):
    monkeypatch.setattr(author.OllamaAnchoredAuthorAdapter, "preflight", lambda self: pytest.fail("no preflight"))
    with pytest.raises(FileExistsError):
        author.freeze(packet, tmp_path)
