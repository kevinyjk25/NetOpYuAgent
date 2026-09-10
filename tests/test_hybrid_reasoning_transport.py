import copy
import json

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
