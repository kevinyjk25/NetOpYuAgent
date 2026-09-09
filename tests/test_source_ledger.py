"""Source-window mechanics and failure preservation; not public-Skill accuracy."""

import copy
import json

import pytest

from evaluation import flow_checkpoint as checkpoint
from evaluation import source_ledger as ledger
from evaluation.structured_flow_demo import fixture
from evaluation.translation_intake import _bundle, _document


@pytest.fixture
def packet():
    bundle, tree, reads, _ = fixture()
    body = {k: v for k, v in bundle.items() if k != "bundleDigest"}
    body["documents"] += [_document(f"references/{name}.md", (f"Exact original {name} prerequisite.\n" + "Untrusted source context.\n" * 1200).encode(),
                                      mode="100644", origin="fixture") for name in ("alpha", "beta")]
    return {"bundle": _bundle(body), "task": "Read the interfaces with original prerequisites; retain unknown duties.",
            "taskOrigin": "developer_authored_evaluation_request", "inputSchema": tree.input_schema,
            "catalog": {"tools": [json.loads(next(s.text for s in c.spec.sources if s.role == "tool")) for c in reads.values()]},
            "reads": {name: c.model_dump(by_alias=True, mode="json") for name, c in reads.items()}}


def page_for(packet, suffix):
    return next(k for k, p in ledger.pages_for(packet).items() if p["path"].endswith(suffix))


def note(page_id, quote):
    return {"source": {"page_id": page_id, "quote": quote}, "kind": "constraint",
            "interpretation": "Retained original prerequisite for later semantic review."}


def request(packet, suffix, notes=()):
    return {"mode": "request_pages", "pages": [page_for(packet, suffix)], "notes": list(notes),
            "reason": "Inspect a referenced original page; dependency closure is not proven."}


def candidate(packet, source_id):
    _, tree, _, _ = fixture()
    step = tree.model_dump(mode="json")["steps"][0]
    step["source"] = {"page_id": source_id, "quote": step["source"]["quote"]}
    return {"mode": "candidate", "remaining": [], "tree": {
        "api_version": tree.api_version, "source_digest": packet["bundle"]["bundleDigest"],
        "purpose": "One synthetic read region, never whole Skill success", "input_schema": packet["inputSchema"],
        "max_read_age_seconds": 5, "unresolved": [], "steps": [step, {
            "kind": "end", "source": copy.deepcopy(step["source"]), "outcome": "needs_l1", "explanation": "All remaining duties need review."}]}}


def envelope(choice):
    return {"httpStatus": 200, "latencyMs": 10, "body": json.dumps({"model": ledger.QWEN_MODEL,
        "done": True, "done_reason": "stop", "prompt_eval_count": 100, "eval_count": 40,
        "message": {"role": "assistant", "content": json.dumps(choice)}})}


@pytest.fixture
def transport(monkeypatch):
    calls, outputs = [], []
    monkeypatch.setattr(ledger.prior.OllamaAnchoredAuthorAdapter, "preflight", lambda self: {"model": ledger.QWEN_MODEL, "digest": "fixture"})
    def send(arm, wire):
        calls.append(wire)
        # Test fixtures are authored as quoted source selectors for readability;
        # the fake model emits the current block ID, never a repaired model output.
        choice = copy.deepcopy(outputs.pop(0))
        blocks = json.loads(wire["messages"][1]["content"])["sourceBlocks"]
        def source(mark):
            return {"block_id": next((b["id"] for b in blocks if b["page_id"] == mark["page_id"]
                                      and mark["quote"] in b["text"]), "not-a-current-block")}
        def steps(items):
            for item in items:
                item["source"] = source(item["source"])
                if item["kind"] == "if_equal":
                    steps(item["when_equal"])
                    steps(item["otherwise"])
        if choice["mode"] == "request_pages":
            for n in choice["notes"]:
                n["source"] = source(n["source"])
        elif choice["mode"] == "gap_report":
            for g in choice["gaps"]:
                g["source"] = source(g["source"])
        else:
            steps(choice["tree"]["steps"])
            for duty in choice["remaining"]:
                duty["source"] = source(duty["source"])
        return envelope(choice)
    monkeypatch.setattr(checkpoint, "send", send)
    return calls, outputs


def test_windows_replace_pages_rehydrate_exact_sources_and_replay_offline(packet, tmp_path, transport, monkeypatch):
    calls, outputs = transport
    initial = ledger.initial_state(packet)
    quote = candidate(packet, initial["window"][0])["tree"]["steps"][0]["source"]["quote"]
    outputs.extend([request(packet, "alpha.md", [note(initial["window"][0], quote)]),
                    request(packet, "beta.md", [note(page_for(packet, "alpha.md"), "Exact original alpha prerequisite.")]),
                    candidate(packet, "e000")])
    root = tmp_path / "run"
    before = copy.deepcopy(packet)
    ledger.freeze(packet, root)
    result = ledger.run(root, max_new_calls=3)
    assert len(calls) == result["modelCallsRecorded"] == 3
    assert result["candidateProduced"] and result["compiledReadNodes"] == 1
    assert result["terminalOutcomes"] == ["needs_l1"]
    assert not result["runtimeAuthorityGranted"] and not result["sourceCoverageProven"]
    assert result["providerCalls"] == result["sourceScriptCalls"] == 0 and result["semanticAccuracy"] is None
    assert result["retainedNotes"] == 2 and result["inputTokens"] == 300
    assert not any(r["semanticDependencyResolved"] for r in result["dependencyRequests"])
    last = json.loads(calls[-1]["messages"][1]["content"])
    assert last["currentFullPages"] == [page_for(packet, "beta.md")]
    assert {p["id"] for p in last["sourcePages"]} == {"e000", "e001", page_for(packet, "beta.md")}
    assert initial["window"][0] not in {p["id"] for p in last["sourcePages"]}
    span = json.loads((root / "round-002/tree.json").read_text())["steps"][0]["source"]
    text = next(d["content"] for d in packet["bundle"]["documents"] if d["path"] == span["path"])
    assert text[span["start"]:span["end"]] == span["quote"] and quote in span["quote"]
    assert packet == before
    monkeypatch.setattr(checkpoint, "send", lambda *a: pytest.fail("offline replay"))
    assert ledger.run(root) == result


@pytest.mark.parametrize("issue", ["wrong_quote", "invented_page", "same_window", "summary_as_quote", "ledger_exhausted"])
def test_invalid_notes_or_requests_stop_without_retry(packet, tmp_path, transport, monkeypatch, issue):
    calls, outputs = transport
    initial = ledger.initial_state(packet)
    quote = candidate(packet, initial["window"][0])["tree"]["steps"][0]["source"]["quote"]
    n = note(initial["window"][0], quote)
    choice = request(packet, "alpha.md", [n])
    if issue == "wrong_quote":
        n["source"]["quote"] = "Unsupported invented source phrase."
    elif issue == "invented_page":
        choice["pages"] = ["https://evil.invalid/script.py"]
    elif issue == "same_window":
        choice["pages"] = initial["window"]
    elif issue == "summary_as_quote":
        n["source"]["quote"] = n["interpretation"]
    else:
        monkeypatch.setattr(ledger, "MAX_NOTES", 0)
    root = tmp_path / "run"
    ledger.freeze(packet, root)
    outputs.append(choice)
    result = ledger.run(root, max_new_calls=6)
    assert result["status"] == "candidate_invalid_or_unresolved" and len(calls) == 1
    assert ledger.run(root, max_new_calls=6) == result and len(calls) == 1
    assert (root / "round-000/response.json").exists()


def test_old_page_id_cannot_be_cited_after_eviction_even_when_seen(packet, tmp_path, transport):
    calls, outputs = transport
    root_id = ledger.initial_state(packet)["window"][0]
    quote = candidate(packet, root_id)["tree"]["steps"][0]["source"]["quote"]
    outputs.extend([request(packet, "alpha.md", [note(root_id, quote)]), candidate(packet, root_id)])
    root = tmp_path / "run"
    ledger.freeze(packet, root)
    result = ledger.run(root, max_new_calls=6)
    assert len(calls) == 2 and result["status"] == "candidate_invalid_or_unresolved"
    assert not result["compiled"]


def test_source_page_revisit_is_explicit_and_allowed_not_an_automatic_retry(packet, tmp_path, transport):
    calls, outputs = transport
    root_id = ledger.initial_state(packet)["window"][0]
    outputs.extend([request(packet, "alpha.md"), {"mode": "request_pages", "pages": [root_id],
                    "reason": "The original root context is needed again; no summary is evidence.", "notes": []}, candidate(packet, root_id)])
    root = tmp_path / "run"
    ledger.freeze(packet, root)
    result = ledger.run(root, max_new_calls=3)
    assert len(calls) == 3 and result["compiled"] and result["retainedNotes"] == 0
    assert not result["semanticDependencyClosureProven"]


def test_long_entry_is_paged_and_unread_root_portions_remain_visible(packet):
    body = {k: v for k, v in packet["bundle"].items() if k != "bundleDigest"}
    root_path = body["entryPath"]
    text = "Original long source.\n" * 4000
    body["documents"] = [d for d in body["documents"] if d["path"] != root_path]
    body["documents"].append(_document(root_path, text.encode(), mode="100644", origin="fixture"))
    packet["bundle"] = _bundle(body)
    wire, measured = ledger.make_request(packet, ledger.initial_state(packet))
    content = json.loads(wire["messages"][1]["content"])
    assert len(content["sourcePages"]) == 1 and content["unreadEntryPages"]
    pages = ledger.pages_for(packet)
    assert "".join(p["text"] for p in pages.values() if p["path"] == root_path) == text
    assert measured["accepted"] and measured["actualInputTokens"] is None


def test_utf8_budget_is_not_character_count_or_a_token_attestation():
    base = {"messages": [{"content": "a" * 20000}], "format": {}}
    ascii_value = ledger.budget(base)
    base["messages"][0]["content"] = "中" * 20000
    unicode_value = ledger.budget(base)
    assert ascii_value["accepted"] and not unicode_value["accepted"]
    assert unicode_value["messageUtf8Bytes"] == 60000
    assert not unicode_value["tokenizerAttested"] and unicode_value["actualInputTokens"] is None


def test_long_unicode_pages_reconstruct_original_offsets_without_token_assumptions(packet):
    body = {k: v for k, v in packet["bundle"].items() if k != "bundleDigest"}
    root_path = body["entryPath"]
    text = "中文🙂\r\n" * 6000
    body["documents"] = [d for d in body["documents"] if d["path"] != root_path]
    body["documents"].append(_document(root_path, text.encode(), mode="100644", origin="fixture"))
    packet["bundle"] = _bundle(body)
    pages = [p for p in ledger.pages_for(packet).values() if p["path"] == root_path]
    assert "".join(p["text"] for p in pages) == text
    assert all(p["text"] == text[p["start"]:p["end"]] and len(p["text"].encode()) <= 12000 for p in pages)
    _, measured = ledger.make_request(packet, ledger.initial_state(packet))
    assert measured["accepted"]


def test_zero_budget_and_policy_stop_do_not_send_or_drop_sources(packet, tmp_path, transport, monkeypatch):
    calls, outputs = transport
    state = ledger.initial_state(packet)
    _, first = ledger.make_request(packet, state)
    future = copy.deepcopy(state)
    future["window"] = [page_for(packet, "alpha.md"), page_for(packet, "beta.md")]
    _, later = ledger.make_request(packet, future)
    assert first["inputByteProxy"] < later["inputByteProxy"]
    monkeypatch.setattr(ledger, "TEMPLATE_RESERVE", ledger.CONTEXT_TOKENS - ledger.OUTPUT_TOKENS - first["inputByteProxy"] - 1000)
    root = tmp_path / "run"
    ledger.freeze(packet, root)
    assert ledger.run(root)["status"] == "new_call_budget_exhausted" and not calls
    outputs.append({"mode": "request_pages", "pages": future["window"], "notes": [], "reason": "Both source references require review before proceeding."})
    result = ledger.run(root, max_new_calls=6)
    assert len(calls) == 1 and result["status"] == "source_window_resource_budget_exhausted"
    assert result["submittedPages"] == state["window"] and result["blockedBudget"]["accepted"] is False
    assert result["retrievalRequests"][-1]["textDelivery"] == "pending"
    assert not (root / "round-001").exists()


@pytest.mark.parametrize("issue", ["policy", "implementation", "initial_state", "model", "second_preflight", "partial"])
def test_frozen_drift_is_never_retried(packet, tmp_path, transport, monkeypatch, issue):
    calls, outputs = transport
    root = tmp_path / "run"
    manifest = ledger.freeze(packet, root)
    if issue == "model":
        monkeypatch.setattr(ledger.prior.OllamaAnchoredAuthorAdapter, "preflight", lambda self: {"model": "other"})
    elif issue == "second_preflight":
        models = iter([manifest["model"], {"model": "other"}])
        monkeypatch.setattr(ledger.prior.OllamaAnchoredAuthorAdapter, "preflight", lambda self: next(models))
        outputs.append(candidate(packet, manifest["initialState"]["window"][0]))
    elif issue == "partial":
        (root / "round-000").mkdir()
    else:
        key = {"policy": "policy", "implementation": "implementation", "initial_state": "initialState"}[issue]
        manifest[key] = {}
        manifest = ledger.prior.seal({k: v for k, v in manifest.items() if k != "reportDigest"})
        (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        ledger.run(root, max_new_calls=6)
    assert len(calls) == (1 if issue == "second_preflight" else 0)


@pytest.mark.parametrize("budget", [-1, 7, True, 1.5])
def test_invalid_call_budget_rejected_before_io(tmp_path, budget):
    with pytest.raises(ValueError, match="budget"):
        ledger.run(tmp_path / "absent", max_new_calls=budget)


def test_candidate_with_only_stop_is_not_counted_as_executable_read(packet, tmp_path, transport):
    calls, outputs = transport
    root = tmp_path / "run"
    manifest = ledger.freeze(packet, root)
    choice = candidate(packet, manifest["initialState"]["window"][0])
    choice["tree"]["steps"] = [choice["tree"]["steps"][-1]]
    outputs.append(choice)
    result = ledger.run(root, max_new_calls=1)
    assert len(calls) == 1 and result["compiled"] and result["candidateProduced"]
    assert result["compiledReadNodes"] == 0 and not result["wholeSkillTranslationProven"]


def test_existing_freeze_never_preflights_or_overwrites(packet, tmp_path, monkeypatch):
    monkeypatch.setattr(ledger.prior.OllamaAnchoredAuthorAdapter, "preflight", lambda self: pytest.fail("must not call"))
    with pytest.raises(FileExistsError):
        ledger.freeze(packet, tmp_path)


@pytest.mark.parametrize("focus", ["root", "alpha"])
def test_interval_union_preserves_all_notes_and_exact_source_coverage(packet, focus):
    pages = ledger.pages_for(packet)
    state = ledger.initial_state(packet)
    root_id = state["window"][0]
    alpha = page_for(packet, "alpha.md")
    spans = [(root_id, 0, 180), (root_id, 90, 210), (root_id, 90, 210), (alpha, 0, 220), (alpha, 300, 450)]
    documents = {d["path"]: d for d in packet["bundle"]["documents"]}
    for i, (page_id, start, end) in enumerate(spans):
        page = pages[page_id]
        end = min(end, len(page["text"]))
        absolute = page["start"] + start
        state["notes"].append({"source": {"path": page["path"], "start": absolute, "end": page["start"] + end,
                                          "quote": page["text"][start:end]},
                               "documentDigest": documents[page["path"]]["sha256"], "kind": "constraint",
                               "interpretation": f"Distinct unverified interpretation number {i}; preserve all notes."})
    state["window"] = [root_id if focus == "root" else alpha]
    before = copy.deepcopy(state)
    old_ranges = [(pages[k]["path"], pages[k]["start"], pages[k]["end"]) for k in state["window"]]
    for n in state["notes"]:
        s = n["source"]
        old_ranges.append((s["path"], max(0, s["start"] - 160), min(len(documents[s["path"]]["content"]), s["end"] + 160)))
    expected = {(p, i) for p, start, end in old_ranges for i in range(start, end)}
    current, anchors = ledger.source_layout(packet, state)
    actual = {(p["path"], i) for p in current.values() for i in range(p["start"], p["end"])}
    assert actual == expected
    assert sum(len(p["text"]) for p in current.values()) == len(expected)
    assert state == before and len(anchors) == len(state["notes"])
    for n, key in zip(state["notes"], anchors, strict=True):
        assert n["source"]["quote"] in current[key]["text"]
    wire, _ = ledger.make_request(packet, state)
    navigation = json.loads(wire["messages"][1]["content"])["ledgerNavigation"]
    assert [n["interpretation"] for n in navigation] == [n["interpretation"] for n in state["notes"]]


def test_delivery_facts_never_turn_into_semantic_approval(packet):
    from evaluation.source_retrieval import delivery_index, requests_view
    pages = ledger.pages_for(packet)
    state = ledger.initial_state(packet)
    root = state["window"][0]
    other = page_for(packet, "alpha.md")
    state["submitted"] = [root]
    state["window"] = [other]
    state["requests"] = [{"fromPages": [root], "requestedPages": [other], "reason": "Need original prerequisites.",
                          "semanticDependencyResolved": False}]
    current = ledger.frame(packet, state)
    current["e000"] = {**pages[root], "start": 10, "end": 25, "text": pages[root]["text"][10:25]}
    rows = {r["id"]: r for r in delivery_index(pages, current, state)}
    assert rows[root]["submittedBefore"] and rows[root]["visibility"] == "partial"
    assert rows[root]["currentIntervals"] == [[10, 25]]
    assert not rows[other]["submittedBefore"] and rows[other]["visibility"] == "full"
    view = requests_view(state)
    assert view[0]["textDelivery"] == "supplied" and "semanticDependencyResolved" not in view[0]
    wire, _ = ledger.make_request(packet, state)
    content = json.loads(wire["messages"][1]["content"])
    assert content["semanticReviewStatus"] == "not_performed"
    for row in content["sourceIndex"]:
        assert content["sourceDocumentPaths"][row["document"]] == pages[row["id"]]["path"]
        assert (row["start"], row["end"]) == (pages[row["id"]]["start"], pages[row["id"]]["end"])
        ranges = row.get("currentIntervals", [[row["start"], row["end"]]] if row["visibility"] == "full" else [])
        assert ranges == next(r for r in delivery_index(pages, ledger.frame(packet, state), state) if r["id"] == row["id"])["currentIntervals"]
    assert {r["id"] for r in content["sourceIndex"]} == set(pages)
    assert state["requests"][0]["semanticDependencyResolved"] is False
    state["requests"] *= 2
    assert requests_view(state, recorded_rounds=2)[-1]["textDelivery"] == "pending"
    assert requests_view(state, recorded_rounds=3)[-1]["deliveryRound"] == 2


def gap_choice(packet, root):
    mark = candidate(packet, root)["tree"]["steps"][0]["source"]
    return {"mode": "gap_report", "gaps": [{"source": mark, "category": "permission",
        "missing": "The supplied sources do not establish host permission for this request.",
        "nextAction": "Request explicit host consent; do not execute the proposed read."}]}


def test_repeated_reads_joint_context_then_gap_is_not_translation_success(packet, tmp_path, transport, monkeypatch):
    calls, outputs = transport
    root = ledger.initial_state(packet)["window"][0]
    other = page_for(packet, "alpha.md")
    outputs.extend([request(packet, "alpha.md"), {"mode": "request_pages", "pages": [root], "notes": [],
                    "reason": "Revisit broader original root context before proposing a candidate."},
                    request(packet, "alpha.md"), gap_choice(packet, root)])
    folder = tmp_path / "run"
    ledger.freeze(packet, folder)
    result = ledger.run(folder, max_new_calls=6)
    assert len(calls) == 4 and result["status"] == "source_grounded_gap_requires_review"
    assert result["gapReportProduced"] and result["gapCount"] == 1
    assert result["uniqueGapDiagnoses"] == 1
    assert not result["candidateProduced"] and not result["compiled"]
    assert result["semanticAccuracy"] is None and not result["runtimeAuthorityGranted"]
    content = json.loads(calls[-1]["messages"][1]["content"])
    assert set(content["currentFullPages"]) == {root, other}
    assert content["authoringPhase"] == "decision"
    assert {b["properties"]["mode"]["const"] for b in calls[-1]["format"]["oneOf"]} == {"candidate", "gap_report"}
    for key in (root, other):
        page = ledger.pages_for(packet)[key]
        assert "".join(b["text"] for b in content["sourceBlocks"] if b["page_id"] == key) == page["text"]
    gap = json.loads((folder / "round-003/gap-report.json").read_text())
    assert gap["translationSucceeded"] is False and gap["semanticEntailmentProven"] is False
    monkeypatch.setattr(checkpoint, "send", lambda *a: pytest.fail("offline replay sent a request"))
    assert ledger.run(folder) == result


def test_last_round_requires_candidate_or_gap_but_does_not_fake_success(packet):
    state = ledger.initial_state(packet)
    state["requests"] = [{"fromPages": [], "requestedPages": state["window"], "reason": "Exact request retained."}] * 5
    wire, _ = ledger.make_request(packet, state)
    assert json.loads(wire["messages"][1]["content"])["decisionReason"] == "last_round_requires_candidate_or_gap"
    files, result = ledger.derive(packet, state, wire, envelope(request(packet, "alpha.md")))
    assert not files and result["candidateStatus"] == "candidate_invalid_or_unresolved"


def test_gap_with_invented_source_is_rejected_and_preserved(packet, tmp_path, transport):
    calls, outputs = transport
    root = ledger.initial_state(packet)["window"][0]
    choice = gap_choice(packet, root)
    choice["gaps"][0]["source"]["quote"] = "A fabricated missing prerequisite never in the source."
    outputs.append(choice)
    folder = tmp_path / "run"
    ledger.freeze(packet, folder)
    result = ledger.run(folder, max_new_calls=6)
    assert len(calls) == 1 and not result["gapReportProduced"] and not result["candidateProduced"]
    assert result["status"] == "candidate_invalid_or_unresolved"
    assert (folder / "round-000/response.json").exists()
