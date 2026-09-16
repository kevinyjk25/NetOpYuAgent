"""Freeze checks for the diagnostic, without any network/model calls."""
import copy
import json

import pytest

from evaluation.task_delivery_ablation import pair
from skill_authoring import compiler, delivery, reasoning_transport


def captured():
    task = "Compare the supplied evidence, preserve uncertainty and identify one next observation."
    contract = delivery.compile_selection({"requirements": [{"kind": "analysis", "language": "",
        "source_ref": "task:0"}], "unrepresented": []}, {"task": task}, single_choice=True)
    req = {"instructions": "Keep host policy unchanged.\n" + delivery.generation(contract),
        "inputs": {"original_task": task, "source_material": "Inert reference example.",
            "caller": {"path": "/snapshot"}, "n0": {"text": "Actual observation"},
            "authoring_boundaries": ["Unverified annotation"], "delivery_contract": json.dumps(contract)},
        "tools": [], "runtimeAuthorityGranted": False, "outputSchema": delivery.response_schema(contract),
        "evidencePolicy": {}, "observationAgesAtStartMs": {}, "maxOutputTokens": 2048}
    wire = {"messages": reasoning_transport.messages(req), "format": req["outputSchema"],
        "model": compiler.MODEL, "stream": False, "think": False,
        "options": {k: v for k, v in compiler.MODEL_CONFIG.items() if k != "think"}}
    return {"governedRequest": req, "wireRequest": wire}


def test_pair_changes_only_delivery_mediation_and_preserves_original_input():
    data = captured()
    before = copy.deepcopy(data)
    arms = pair(data)
    assert data == before and arms["choice"]["wire"] == data["wireRequest"]
    old, new = [json.loads(arms[a]["wire"]["messages"][-1]["content"])["actual_task_caller_and_observations"]
                for a in ("choice", "task")]
    assert {k: v for k, v in old.items() if k != "delivery_contract"} == {
        k: v for k, v in new.items() if k != "delivery_contract"}
    assert arms["task"]["wire"]["messages"][0]["content"].startswith("Keep host policy unchanged.\n")
    assert arms["choice"]["wire"]["options"] == arms["task"]["wire"]["options"]


@pytest.mark.parametrize("field", ["model", "messages", "options"])
def test_changed_captured_settings_or_input_fail_before_any_call(field):
    data = captured()
    data["wireRequest"][field] = None
    with pytest.raises(ValueError):
        pair(data)
