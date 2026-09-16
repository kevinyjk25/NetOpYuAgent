import copy
import json
import pytest

from evaluation.hybrid_reasoning_transport import messages


def test_reference_reordering_preserves_every_input_and_keeps_observation_last():
    request = {"inputs": {"source_material": "template example: old", "authoring_boundaries": "untrusted annotation",
        "caller": {"key": "new"}, "original_task": "Inspect actual request", "n0": {"observations": {"read": "current"}},
        "unknown_future_field": {"retain": True}}, "instructions": "Host-bound instructions",
        "evidencePolicy": "historical", "observationAgesAtStartMs": {"n0": 5}, "outputSchema": {"type": "object"}, "maxOutputTokens": 100}
    before = copy.deepcopy(request)
    rendered = messages(request)
    refs = json.loads(rendered[1]["content"])["reference_only_not_current_observations"]
    current = json.loads(rendered[2]["content"])["actual_task_caller_and_observations"]
    assert {**refs, **current} == request["inputs"] == before["inputs"]
    assert request == before and current["n0"]["observations"]["read"] == "current"
    assert rendered[0]["content"] == request["instructions"]
    assert all("tools" not in message for message in rendered)


def test_completed_read_index_keeps_old_plan_out_of_current_evidence():
    request = {"inputs": {"original_task": "Update the requested artifact",
        "previousRemainingActions": ["Read the detail export"],
        "previousCandidate": {"draft": "Prior unverified artifact"},
        "priorReadResults": [{"tool": "read_export", "arguments": {"read_export": {"path": "index"}},
            "result": {"read_export": {"text": "detail exists"}}, "receiptDigest": "historical"}],
        "selection": {"decision": "read_export", "requests": {"read_export": {"path": "detail"}}},
        "currentReadNodeTools": {"read-0": "read_export"},
        "currentObservation": {"outcome": "read_path_completed", "observations": {"read-0": {"text": "actual detail"}}}},
        "instructions": "Host instructions", "evidencePolicy": "no future authority", "observationAgesAtStartMs": {"read": 1},
        "outputSchema": {"type": "object"}, "maxOutputTokens": 100}
    before = copy.deepcopy(request)
    rendered = messages(request)
    reference = json.loads(rendered[1]["content"])["reference_only_not_current_observations"]
    last = json.loads(rendered[-1]["content"])
    current = last["actual_task_caller_and_observations"]
    assert "previousRemainingActions" in reference and "previousCandidate" in reference
    assert "previousRemainingActions" not in current and "previousCandidate" not in current
    assert {**reference, **current} == before["inputs"] and request == before
    indexed = last["readStatusIndex"]
    assert [e["state"] for e in indexed["entries"]] == ["completed_in_recorded_prior_execution", "completed_in_this_execution"]
    assert indexed["entries"][1]["arguments"] == {"path": "detail"}
    assert indexed["entries"][1]["payloadAt"] == "/currentObservation/observations/read-0"
    assert not indexed["authorityGranted"] and not indexed["semanticTaskCompletion"]
    request["inputs"]["selection"]["decision"] = "invented"
    with pytest.raises(ValueError, match="host-bound"):
        messages(request)


def test_no_current_read_is_not_indexed_as_completed():
    from evaluation.hybrid_reasoning_transport import read_status_index
    value = {"currentReadNodeTools": {"read-0": "read_export"}, "selection": {"decision": "clarify", "requests": {}},
             "currentObservation": {"outcome": "needs_l1", "observations": {}}}
    assert read_status_index(value)["entries"] == []
    value["currentObservation"]["observations"]["read-0"] = {}
    with pytest.raises(ValueError, match="non-read"):
        read_status_index(value)
