"""Generic compiler tests, not public-skill or model accuracy."""
import copy
import json

import pytest

from evaluation.hybrid_authoring import compile_proposal, pages_for, make_request, author_response_schema
from evaluation.structured_flow_demo import fixture


def inputs():
    bundle, tree, reads, _ = fixture()
    return {"bundle": bundle, "task": "Inspect the caller device interfaces and draft a diagnosis; do not make changes.",
        "taskOrigin": "developer_authored_evaluation_request", "inputSchema": tree.input_schema,
        "catalog": {"tools": []}, "reads": {}}, reads


def test_tool_free_reasoning_preserves_source_task_origins_without_fake_l0():
    packet, _ = inputs()
    visible = list(pages_for(packet))
    choice = {"mode": "proposal", "intent_summary": "Compile future caller inputs into a governed reasoning task without pretending absent device tools are available.", "steps": [{"kind": "reason", "id": "n0", "after": [], "evidence": [visible[0], "task"],
        "assignment": "Explain available information, explicitly retain unavailable actual device observations."}],
        "outputs": ["n0"], "boundaries": []}
    before = copy.deepcopy(packet)
    result = compile_proposal(packet, visible, choice)
    assert packet == before and result["sourceScriptCalls"] == 0
    mapping = result["sourceTaskMappings"][0]["origins"]
    assert mapping["task"]["origin"] == "caller_task" and mapping[visible[0]]["origin"] == "skill_source"
    assert not result["l0WholeSkill"] and not result["qualification"]["wholeGraphDeterministic"]
    assert not result["runtimeAuthorityGranted"]


def test_task_citation_cannot_impersonate_an_unseen_source_page():
    packet, _ = inputs()
    visible = list(pages_for(packet))
    choice = {"mode": "proposal", "intent_summary": "Compile future caller inputs into a governed reasoning task without pretending absent device tools are available.", "steps": [{"kind": "reason", "id": "n0", "after": [], "evidence": ["p999"],
        "assignment": "Explain only caller information and clearly note missing device evidence."}], "outputs": ["n0"], "boundaries": []}
    with pytest.raises(ValueError):
        compile_proposal(packet, visible, choice)


def test_exact_format_contract_is_visible_to_model_and_all_read_pages_cannot_be_requested_again():
    packet, _ = inputs()
    visible = list(pages_for(packet))
    request = make_request(packet, visible)
    payload = json.loads(request["messages"][-1]["content"])
    assert payload["requiredOutputSchema"] == author_response_schema(packet, visible)
    assert request["format"] == "json"
    assert "request_pages" not in str(payload["requiredOutputSchema"])
    assert all(payload["sourcePages"][p] == pages_for(packet)[p]["text"] for p in visible)


def test_retained_task_profile_does_not_rewrite_open_duties_or_count_pure_l1_as_l0():
    packet, _ = inputs()
    visible = list(pages_for(packet))
    choice = {"mode": "read_prefix", "intent_summary": "Retain the caller request as an open reasoning task; no host is available to retrieve actual device observations.",
        "reads": [], "boundaries": [{"evidence": ["task"], "kind": "missing_host",
            "explanation": "No declared primitive can retrieve the requested device observations."}]}
    before = copy.deepcopy(choice)
    compiled = compile_proposal(packet, visible, choice)
    assert compiled["plan"] == choice == before and compiled["retainedOriginalTask"]
    node = compiled["flow"]["nodes"][0]
    assert node["inputs"]["fields"]["original_task"]["value"] == packet["task"]
    assert json.loads(node["inputs"]["fields"]["authoring_boundaries"]["value"]) == choice["boundaries"]
    assert compiled["sourceTaskMappings"][0]["interpretation"] == "original_task_retained_not_rewritten"
    assert not compiled["l0WholeSkill"] and node["kind"] == "reason"


def test_prefix_cannot_invent_tool_or_reserved_reason_node():
    packet, _ = inputs()
    choice = {"mode": "read_prefix", "intent_summary": "Attempt to invent a tool outside the declared host catalog; this must never become an executable prefix.",
        "reads": [{"id": "n7", "after": [], "evidence": ["task"], "tool": "shell", "arguments": {}}], "boundaries": []}
    with pytest.raises(ValueError, match="schema mismatch"):
        compile_proposal(packet, list(pages_for(packet)), choice)
