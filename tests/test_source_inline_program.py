"""Inline provenance is carried exactly; source truth is still a review duty."""
import copy
import json

import pytest
from jsonschema import Draft202012Validator

from evaluation import source_inline_program as inline, source_ledger as ledger, source_program_lines as lines
from evaluation.source_program_anchors import evidence_choices
from evaluation.source_blocks import citation_blocks
from tests.test_source_semantic_plan import prepared, unmarked, wire_choice
from tests.test_source_obligations import packet as packet_fixture

packet = packet_fixture


def test_inline_sources_survive_lowering_without_second_model_binding(packet):
    state, wire, choice = prepared(packet)
    blocks = citation_blocks(ledger.frame(packet, state))
    original = copy.deepcopy(choice)
    anchored, combined, draft, audit = inline.prepare(choice, packet, blocks)
    assert original == choice
    assert Draft202012Validator(wire["format"]).is_valid(wire_choice(choice))
    assert draft["choice"] == choice
    assert audit["additionalSourceBindingModelCalls"] == 0
    assert not any(audit[k] for k in ("programStructureChanged", "sourceSelectionsInferred", "semanticEntailmentProven", "runtimeAuthorityGranted"))
    assert anchored["program"] == draft["renderedProgram"]
    assert len(draft["inlineSourceMap"]) == 5  # read, condition, two exits and one duty
    for row in draft["inlineSourceMap"]:
        assert combined[row["selectedEvidenceId"]]["text"] == row["source"]["quote"]
    assert draft["parsedStructure"]["steps"][1]["left"]["kind"] == "array_length"


@pytest.mark.parametrize("mutation", ["missing_node", "missing_duty", "unknown", "fake_operation", "empty_scan"])
def test_inline_unknown_missing_or_forged_source_does_not_compile(packet, mutation):
    state, _, choice = prepared(packet)
    blocks = citation_blocks(ledger.frame(packet, state))
    if mutation == "missing_node":
        del choice["program"][0]["source_id"]
    elif mutation == "missing_duty":
        del choice["program"][1]["otherwise"][0]["duties"][0]["source_id"]
    elif mutation == "unknown":
        choice["program"][0]["source_id"] = "not-delivered"
    elif mutation == "fake_operation":
        choice["program"][0]["op"] = "exec"
    else:
        choice["source_scan"] = {}
    with pytest.raises(ValueError):
        inline.prepare(choice, packet, blocks)


def test_real_but_unrelated_source_is_not_repaired_or_certified(packet):
    state, _, choice = prepared(packet)
    blocks = citation_blocks(ledger.frame(packet, state))
    wrong = next(key for key, b in blocks.items() if "# " in b["text"] and len(b["text"]) >= 8)
    choice["program"][0]["source_id"] = wrong
    _, _, draft, audit = inline.prepare(choice, packet, blocks)
    assert draft["inlineSourceMap"][0]["selectedEvidenceId"] == wrong
    assert not audit["semanticEntailmentProven"]


def test_source_marked_and_unmarked_programs_have_same_control_structure(packet):
    _, _, choice = prepared(packet)
    clean = unmarked(choice["program"])
    assert "source_id" not in str(clean)
    assert lines.canonicalize(choice["program"])["statementOrigins"] == lines.canonicalize(clean)["statementOrigins"]
    # The inline annotations do not create an additional source-planning call,
    # operation, implicit completion or business guard.
    assert len(lines.canonicalize(choice["program"])["statements"]) == len(clean)


def test_shared_source_navigation_keeps_every_original_choice_and_text(packet):
    state, wire, _ = prepared(packet)
    body = json.loads(wire["messages"][1]["content"])
    choices = evidence_choices(citation_blocks(ledger.frame(packet, state)))
    assert body["programEvidenceChoices"]["ids"] == list(choices)
    disclosed = {b["id"]: b["text"] for b in body["sourceBlocks"] + body["sourceScanFragments"]}
    for key, value in choices.items():
        assert disclosed[key] == value["block"]["text"]
    for row in body["sourceScanFragments"]:
        assert set(row) == {"id", "block_id", "text"}
        parent = next(b for b in body["sourceBlocks"] if b["id"] == row["block_id"])
        assert row["text"] in parent["text"]  # full original location/context remains


@pytest.mark.parametrize("family", ["wiring", "reference", "approval"])
def test_all_stage1_inputs_fit_unchanged_model_budget(family):
    from evaluation.stage1_cases import packet
    value = packet(family)
    state = ledger.initial_state(value, "plan_first", semantic_plan=True)
    _, limits = ledger.make_request(value, state)
    assert limits["accepted"]
    assert limits["proxyLimit"] == 40960 and limits["outputTokenReserve"] == 4096
    if family == "reference":
        assert len(state["window"]) == 2  # not dropping the original procedure to fit


def test_continuation_moves_keep_each_original_statement_and_duty_origin(packet):
    state, _, choice = prepared(packet)
    blocks = citation_blocks(ledger.frame(packet, state))
    pending = choice["program"][1]["otherwise"][0]
    source_id = next(key for key, b in blocks.items() if "Read interfaces for" in b["text"])
    pending["source_id"] = source_id
    pending["duties"][0]["source_id"] = source_id
    second = {**choice["program"][0], "name": "second_observation"}
    choice["program"][1]["otherwise"] = [second]
    choice["program"].append(pending)
    original = copy.deepcopy(choice)
    _, _, draft, audit = inline.prepare(choice, packet, blocks)
    assert choice == original and not audit["sourceSelectionsInferred"]
    assert draft["controlFlowNormalization"]["moves"]
    moved = next(row for row in draft["inlineSourceMap"] if row["originalPointer"] == "/program/2")
    duty = next(row for row in draft["inlineSourceMap"] if row["originalPointer"] == "/program/2/duties/0")
    assert moved["normalizedPointer"] == "/program/1/otherwise/1"
    assert duty["normalizedPointer"] == "/program/1/otherwise/1/duties/0"
    assert moved["selectedEvidenceId"] == duty["selectedEvidenceId"] == source_id


def test_predicate_provenance_is_adjacent_to_condition_not_child_action():
    schema = lines.schema({"tools": []}, evidence_ids=["b9", "b4:L1"])
    predicates = [r for r in schema["$defs"]["ProgramStatement"]["anyOf"]
                  if r["properties"]["op"]["const"] in lines.BRANCHES]
    for row in predicates:
        keys = list(row["properties"])
        assert keys.index("equals") < keys.index("source_id") < keys.index("when_equal")
        assert row["properties"]["source_id"]["$ref"] == "#/$defs/ProgramPredicateSourceId"
    assert "predicate" in schema["$defs"]["ProgramPredicateSourceId"]["description"]
    assert schema["$defs"]["ProgramPredicateSourceId"]["$ref"] == "#/$defs/ProgramSourceId"
    assert schema["$defs"]["ProgramSourceId"]["enum"] == ["b9", "b4:L1"]  # no correct origin is supplied


@pytest.mark.parametrize("source", ["b0003", "available", "get_interfaces", "obs8"])
def test_model_data_references_cannot_be_confused_with_evidence_or_field_ids(packet, source):
    _, wire, value = prepared(packet)
    value["program"][1]["value"]["source"] = source
    assert not Draft202012Validator(wire["format"]).is_valid(wire_choice(value))


def test_allowed_but_undefined_slot_still_fails_original_dominance_check(packet):
    state, wire, value = prepared(packet)
    value["program"][1]["value"]["source"] = "obs7"
    assert Draft202012Validator(wire["format"]).is_valid(wire_choice(value))  # no dependency answer supplied
    with pytest.raises(ValueError, match="dominate"):
        inline.prepare(value, packet, citation_blocks(ledger.frame(packet, state)))


def test_future_requirements_are_declared_before_program_without_changing_meaning(packet):
    _, wire, _ = prepared(packet)
    plan = next(r for r in wire["format"]["oneOf"] if r.get("properties", {}).get("mode", {}).get("const") == "operation_plan")
    keys = list(plan["properties"])
    assert all(keys.index(k) < keys.index("program") for k in ("execution_requirements", "business_gaps", "outside_task_duties"))


def test_closed_model_tree_keeps_original_node_and_duty_pointers(packet):
    state, wire, value = prepared(packet)
    raw = wire_choice(value)
    original = copy.deepcopy(raw)
    assert Draft202012Validator(wire["format"]).is_valid(raw)
    _, _, draft, audit = inline.prepare(raw, packet, citation_blocks(ledger.frame(packet, state)))
    assert raw == original == draft["choice"]
    assert draft["controlSyntaxLowering"]["statementsDiscarded"] == 0
    assert not draft["controlFlowNormalization"]["moves"]
    assert not draft["controlFlowNormalization"]["redundantTerminals"]
    duty = next(row for row in draft["inlineSourceMap"] if row["role"] == "duties")
    assert duty["originalPointer"] == "/program/next/otherwise/duties/0"
    assert duty["normalizedPointer"] == "/program/1/otherwise/0/duties/0"
    assert not audit["sourceSelectionsInferred"]
