"""Source-only phase integrity and lexical constraints; never model scores."""
import copy
import json

import pytest

from evaluation import source_ledger as ledger, source_program_anchors as anchors, source_program_lines as lines
from evaluation.source_blocks import citation_blocks
from evaluation.stage1_cases import packet as stage_packet
from tests.test_source_semantic_plan import prepared as prepare_inline
from tests.test_source_obligations import packet as packet_fixture

packet = packet_fixture


def prepared(packet):
    # Historical separate binding remains mechanically testable, but the new
    # semantic frontend carries sources inline and does not invoke this phase.
    return prepare_inline(packet, inline=False)


def draft(packet):
    state, wire, choice = prepared(packet)
    blocks = citation_blocks(ledger.frame(packet, state))
    frozen = anchors.draft(choice, packet, blocks)
    source = choice["procedure"][0]["source"]["block_id"]
    response = {"mode": "program_sources", "draftDigest": frozen["reportDigest"],
                "sources": {slot["id"]: {"basis": "Test mapping, not a semantic proof.", "evidence_id": source}
                            for slot in frozen["slots"]}}
    return frozen, response, blocks


@pytest.mark.parametrize("mutation", ["missing", "extra", "extra_metadata", "wrong_digest", "unknown_source"])
def test_source_assignment_cannot_change_slots_or_bind_another_draft(packet, mutation):
    frozen, response, blocks = draft(packet)
    response = copy.deepcopy(response)
    first = next(iter(response["sources"]))
    if mutation == "missing":
        response["sources"].pop(first)
    elif mutation == "extra":
        response["sources"]["unexpected"] = response["sources"][first]
    elif mutation == "extra_metadata":
        response["sources"][first] = {"source": response["sources"][first], "program": "replace it"}
    elif mutation == "wrong_digest":
        response["draftDigest"] = "sha256:" + "0" * 64
    else:
        response["sources"][first] = "nonexistent"
    with pytest.raises(ValueError):
        anchors.bind(frozen, response, packet, blocks)


def test_draft_rederivation_rejects_changed_program(packet):
    frozen, response, blocks = draft(packet)
    frozen["choice"]["program"][1]["equals"] = 1
    with pytest.raises(ValueError, match="drift"):
        anchors.bind(frozen, response, packet, blocks)


def test_source_scan_cannot_skip_or_add_presented_fragments(packet):
    _, _, choice = prepared(packet)
    state = ledger.initial_state(packet, "plan_first", semantic_plan=True)
    blocks = citation_blocks(ledger.frame(packet, state))
    valid = anchors.draft(choice, packet, blocks)
    assert len(valid["sourceScan"]) == len(anchors.scan_fragments(blocks))
    assert "not_verified" in valid["sourceScanMeaning"]
    for row in valid["sourceScan"]:
        original = anchors.scan_fragments(blocks)[row["id"]]["block"]
        assert row["source"]["quote"] == original["text"]
    missing = copy.deepcopy(choice)
    missing["source_scan"].pop(next(iter(missing["source_scan"])))
    with pytest.raises(ValueError, match="every original fragment"):
        anchors.draft(missing, packet, blocks)
    extra = copy.deepcopy(choice)
    extra["source_scan"]["invented"] = next(iter(extra["source_scan"].values()))
    with pytest.raises(ValueError, match="every original fragment"):
        anchors.draft(extra, packet, blocks)


def test_selected_source_quote_is_exact_and_original_block_is_retained(packet):
    frozen, response, blocks = draft(packet)
    line_id = next(k for k in anchors.evidence_choices(blocks) if ":L" in k)
    response["sources"] = {k: {"basis": "Test original line mapping.", "evidence_id": line_id} for k in response["sources"]}
    _, _, audit = anchors.bind(frozen, response, packet, blocks)
    for row in audit["anchors"]:
        mark = anchors.evidence_choices(blocks)[response["sources"][row["id"]]["evidence_id"]]
        quote = mark["block"]["text"]
        assert row["source"]["quote"] == quote
        assert row["originalBlock"]["quote"] == blocks[mark["block_id"]]["text"]
        offset = row["source"]["start"] - row["originalBlock"]["start"]
        assert row["originalBlock"]["quote"][offset:offset + len(quote)] == quote
    first = next(iter(response["sources"]))
    response["sources"][first] = "fabricated support for an absent rule"
    with pytest.raises(ValueError, match="current evidence choice"):
        anchors.bind(frozen, response, packet, blocks)


def test_paged_source_ids_never_overwrite_planning_metadata(packet):
    frozen, response, blocks = draft(packet)
    later = copy.deepcopy(blocks)
    # Another page may reuse a block ID for different original offsets/text.
    keys = list(later)
    later[keys[0]], later[keys[-1]] = later[keys[-1]], later[keys[0]]
    choice, combined, audit = anchors.bind(frozen, response, packet, later)
    assert all(s["source"]["block_id"].startswith("plan_") for s in choice["procedure"])
    for key, value in blocks.items():
        assert combined["plan_" + key] == value
        assert combined["anchor_" + key] == later[key]
    assert not audit["programStructureChanged"] and not audit["runtimeAuthorityGranted"]
    assert len(audit["anchors"]) == len(frozen["slots"])
    assert "UNBOUND_" not in choice["program"]


def test_evidence_choices_preserve_unicode_crlf_duplicates_and_whole_blocks():
    text = "先读取原始设备数据。\r\nRepeat exact original rule.\r\nRepeat exact original rule.\r\n"
    block = {"page_id": "p9", "path": "references/check.md", "start": 41,
             "end": 41 + len(text), "text": text}
    choices = anchors.evidence_choices({"b9": block})
    assert choices["b9"]["block"] == block  # multi-line evidence stays available
    assert set(choices) == {"b9", "b9:L1", "b9:L2", "b9:L3"}
    for value in choices.values():
        selected = value["block"]
        assert text[selected["start"] - 41:selected["end"] - 41] == selected["text"]
    assert choices["b9:L2"]["block"]["text"] == choices["b9:L3"]["block"]["text"]
    assert choices["b9:L2"]["block"]["start"] != choices["b9:L3"]["block"]["start"]


def test_evidence_selection_is_not_a_semantic_oracle(packet):
    frozen, response, blocks = draft(packet)
    # Selecting unrelated but real original text still satisfies provenance.
    # The binder does not pretend to infer entailment or silently correct it.
    response["sources"] = {key: {"basis": "A confident but unverified and unrelated claim.",
                                "evidence_id": next(iter(anchors.evidence_choices(blocks)))} for key in response["sources"]}
    _, _, audit = anchors.bind(frozen, response, packet, blocks)
    assert audit["semanticEntailmentProven"] is False
    assert audit["runtimeAuthorityGranted"] is False
    assert audit["anchors"][0]["unverifiedBasis"] == "A confident but unverified and unrelated claim."


@pytest.mark.parametrize("basis", [None, True, "", " " * 10, "x" * 401])
def test_source_mapping_explanation_is_bounded_inert_metadata(packet, basis):
    frozen, response, blocks = draft(packet)
    response["sources"][next(iter(response["sources"]))]["basis"] = basis
    with pytest.raises(ValueError):
        anchors.bind(frozen, response, packet, blocks)


def test_source_slot_schema_keeps_target_role_and_all_original_choices(packet):
    frozen, _, blocks = draft(packet)
    schema = anchors.schema(frozen, blocks, {}, {})["oneOf"][-1]["properties"]["sources"]["properties"]
    for slot in frozen["slots"]:
        value = schema[slot["id"]]
        assert value["description"] == slot["role"] + ": " + slot["programText"]
        assert value["properties"]["evidence_id"]["enum"] == list(anchors.evidence_choices(blocks))
        assert list(value["properties"]) == ["basis", "evidence_id"]


def test_evidence_budget_does_not_silently_drop_sources():
    blocks = {f"b{i}": {"page_id": "p0", "path": "SKILL.md", "start": i * 16,
              "end": i * 16 + 16, "text": "Original rule.  "} for i in range(513)}
    with pytest.raises(ValueError, match="budget exceeded"):
        anchors.evidence_choices(blocks)


@pytest.mark.parametrize("field,value", [("parameters", {"device": "guess"}), ("tool", "unknown"),
                                       ("operationMode", "undeclared"), ("op", "exec")])
def test_typed_decoding_rejects_parameters_and_unsupported_forms(packet, field, value):
    _, _, choice = prepared(packet)
    choice["program"][0][field] = value
    with pytest.raises(ValueError, match="typed program"):
        lines.render(choice["program"], packet["catalog"])


def test_typed_constraint_does_not_choose_semantics(packet):
    _, _, choice = prepared(packet)
    assert "== 0" in lines.render(choice["program"], packet["catalog"])
    # Wrong but grammatical logic still fits: syntax is not a semantic Oracle.
    choice["program"][1]["equals"] = 999
    assert "== 999" in lines.render(choice["program"], packet["catalog"])


def test_short_sealed_reference_is_eagerly_disclosed_without_review_or_execution():
    value = stage_packet("reference")
    legacy = ledger.initial_state(value, "plan_first")
    state = ledger.initial_state(value, "plan_first", semantic_plan=True)
    assert len(legacy["window"]) == 1 and len(state["window"]) == 2
    assert len(state["seededReferences"]) == 1
    assert state["seededReferences"][0]["semanticReviewPerformed"] is False
    wire, _ = ledger.make_request(value, state)
    text = json.loads(wire["messages"][1]["content"])
    assert any("health.alarmId" in b["text"] for b in text["sourceBlocks"])


@pytest.mark.parametrize("field,value", [
    ("contextRole", "script_execution_candidate"),
    ("availability", "missing"),
    ("presentCandidatePaths", ["one", "two"]),
])
def test_unavailable_ambiguous_or_script_reference_is_not_implicitly_loaded(monkeypatch, field, value):
    packet = stage_packet("reference")
    pages = ledger.pages_for(packet)
    # Isolate the navigation filter. Production also rejects any tampered bundle
    # before reaching this selection; this mutation is not a valid input bundle.
    monkeypatch.setattr(ledger, "pages_for", lambda packet: pages)
    packet["bundle"]["references"][0][field] = value
    state = ledger.initial_state(packet, "plan_first", semantic_plan=True)
    assert state["seededReferences"] == [] and len(state["window"]) == 1


def test_reference_budget_rejection_keeps_whole_page_out(monkeypatch):
    packet = stage_packet("reference")
    monkeypatch.setattr(ledger, "make_request", lambda *args: ({}, {"accepted": False}))
    state = ledger.initial_state(packet, "plan_first", semantic_plan=True)
    assert state["seededReferences"] == [] and len(state["window"]) == 1
