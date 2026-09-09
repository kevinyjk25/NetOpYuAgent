"""Host-owned source IDs do not prove the selected meaning."""

import json

import pytest

from evaluation.flow_source_selection import SelectedDraft, expand, selected_request, spans
from evaluation.flow_translation import local_sources
from evaluation.flow_grounded_translation import project


def fixture():
    source = local_sources().model_copy(update={"source_text":
        "Read input device_id once with read_inventory_device, then finish.\n"
        "This is planned inventory, not live health, and grants no permission.\n"})
    selected = SelectedDraft.model_validate({"business_source_ids": ["s0001", "s0002"],
        "entry": 0, "issues": [], "steps": [
            {"kind": "read", "tool": "read_inventory_device", "arguments": {
                "device_id": {"kind": "reference", "source": "input", "field": "device_id"}},
             "next": 1, "requires": [], "source_id": "s0001"},
            {"kind": "end", "outcome": "read_path_completed", "requires": [0], "source_id": "s0001"}]})
    return source, selected


def test_compiler_selects_source_without_model_copying_or_paraphrasing():
    source, selected = fixture()
    cited = expand(source, selected)
    assert cited.purpose_quotes == tuple(spans(source).values())
    draft = project(source, cited)
    assert draft.nodes[0].next == "node-1"
    assert draft.nodes[1].explanation == spans(source)["s0001"]


@pytest.mark.parametrize("field", ["purpose", "node", "branch", "issue"])
def test_host_or_unknown_source_ids_rejected(field):
    source, selected = fixture()
    raw = selected.model_dump(mode="json")
    if field == "purpose":
        raw["business_source_ids"] = ["host-0001"]
    elif field == "node":
        raw["steps"][0]["source_id"] = "s9999"
    elif field == "branch":
        raw["steps"][0] = {"kind": "branch", "source_id": "s0001", "requires": [],
            "left": {"kind": "reference", "source": "input", "field": "device_id"},
            "equals": {"kind": "constant", "value": "x"}, "on_true": 1, "on_false": 1,
            "true_source_id": "s0001", "false_source_id": "host-0001"}
    else:
        raw["issues"] = [{"kind": "source_ambiguity", "source_id": "host-0001", "question": "Missing source fact for this test?"}]
    with pytest.raises(ValueError, match="absent from target"):
        expand(source, SelectedDraft.model_validate(raw))


def test_branch_schema_requires_both_polarity_source_ids():
    schema = SelectedDraft.model_json_schema()["$defs"]["SelectedBranch"]
    assert {"true_source_id", "false_source_id"} <= set(schema["required"])
    assert schema["properties"]["true_source_id"]["type"] == "string"


def test_new_request_has_no_legacy_question_field_and_no_answer_metadata():
    source, _ = fixture()
    wire = selected_request(source)
    payload = json.loads(wire["messages"][1]["content"])
    assert payload["targetSkillSpans"] == spans(source)
    assert "sourceSkill" not in payload
    assert "unresolved_questions" not in wire["messages"][0]["content"]
    assert "purpose_quotes" not in wire["messages"][0]["content"]
    assert wire["format"]["$defs"]["SelectedBranch"]["properties"]["false_source_id"]["enum"] == list(spans(source))
    assert selected_request(source.model_copy(update={"source_path": "hidden_label"})) == wire


def test_selecting_wrong_real_source_is_not_automatically_semantic_truth():
    source, selected = fixture()
    raw = selected.model_dump(mode="json")
    raw["steps"][0]["source_id"] = "s0002"
    cited = expand(source, SelectedDraft.model_validate(raw))
    assert project(source, cited).nodes[0].tool == "read_inventory_device"
    assert cited.steps[0].source_quote.startswith("This is planned")
    # Existence passes, entailment remains a required review; no automatic repair.
