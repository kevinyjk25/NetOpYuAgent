"""One representation per delivery item; no model semantics are certified."""
import copy

import pytest

from dsh_adapter import hybrid_session
from network_runtime.l0.structured_schema import DataBindingError, checked_schema
from skill_authoring import delivery


def contract(kind="analysis", language=""):
    return delivery.compile_selection({"requirements": [{"kind": kind, "language": language,
        "source_ref": "task:0"}], "unrepresented": []}, {"task": "Provide the requested result."}, single_choice=True)


@pytest.mark.parametrize("kind,language,content", [
    ("artifact", "sql", {"body": "select 1"}), ("analysis", "", {"text": "Observed facts only."}),
    ("decision", "", {"conclusion": "Not ready", "basis": "Missing evidence"}),
    ("next_steps", "", {"items": ["Collect authorized evidence."]}),
])
def test_one_value_preserves_content_and_has_no_cross_field_state(kind, language, content):
    frozen = contract(kind, language)
    schema = checked_schema(delivery.response_schema(frozen))
    assert set(schema["properties"]) == {"delivery", "uncertainties"}
    candidate = {"delivery": {"d0": content}, "uncertainties": []}
    before = copy.deepcopy(candidate)
    result = delivery.render(frozen, candidate)
    assert result["shapeComplete"] and not result["semanticApproval"] and result["taskSuccess"] is None
    assert candidate == before
    assert delivery._canonical_response(frozen, candidate)["delivery"]["d0"]["content"] == content


@pytest.mark.parametrize("reason", ["Evidence unavailable", "None", "用户未提供必要数据", "false"])
def test_every_string_is_unresolved_never_a_success_sentinel(reason):
    result = delivery.render(contract(), {"delivery": {"d0": reason}, "uncertainties": []})
    assert not result["shapeComplete"] and result["checks"][0]["status"] == "unresolved"
    assert reason in result["rendered"]


@pytest.mark.parametrize("value", [None, [], True, 9, "", "  ", {"text": 4},
    {"text": "Provided", "gap": "None"}, {"state": "provided", "content": {"text": "Old"}, "gap": ""}])
def test_invalid_choice_rejected_without_repair(value):
    candidate = {"delivery": {"d0": value}, "uncertainties": []}
    before = copy.deepcopy(candidate)
    with pytest.raises(DataBindingError):
        delivery.render(contract(), candidate)
    assert candidate == before


def test_duplicate_state_and_missing_or_extra_ids_fail_closed():
    for candidate in [
        {"delivery": {"d0": {"text": "Provided"}}, "unresolved": {"d0": "None"}, "uncertainties": []},
        {"delivery": {}, "uncertainties": []},
        {"delivery": {"d0": {"text": "Provided"}, "d1": "Unbound"}, "uncertainties": []},
    ]:
        with pytest.raises(DataBindingError):
            delivery.render(contract(), candidate)


def test_v3_keeps_original_conflict_and_never_autoupgrades():
    frozen = delivery.compile_selection(contract()["proposal"], {"task": "Provide the requested result."})
    assert frozen["profile"] == delivery.COMPACT_PROFILE
    with pytest.raises(DataBindingError, match="provided content cannot also be unresolved"):
        delivery.render(frozen, {"delivery": {"d0": {"text": "Provided"}},
                                "unresolved": {"d0": "None"}, "uncertainties": []})


def test_host_outcomes_are_never_semantic_completion():
    checks = {"status": "no_failed_checks"}
    assert hybrid_session._host_result(None, None)["state"] == "rejected"
    unresolved = delivery.render(contract(), {"delivery": {"d0": "Missing evidence"}, "uncertainties": []})
    assert hybrid_session._host_result(unresolved, checks)["state"] == "needs_revision"
    candidate = delivery.render(contract(), {"delivery": {"d0": {"text": "Unverified"}}, "uncertainties": []})
    outcome = hybrid_session._host_result(candidate, checks)
    assert outcome["state"] == "candidate_unverified" and outcome["taskSuccess"] is None
    assert not outcome["semanticApproval"] and outcome["deliveryDigest"] == candidate["reportDigest"]
    assert hybrid_session._host_result(candidate, {"status": "failed_checks"})["state"] == "needs_revision"


def test_complete_slots_do_not_claim_semantic_coverage_of_a_misinterpreted_task():
    frozen = delivery.compile_selection({"requirements": [{"kind": "analysis", "language": "",
        "source_ref": "task:0"}], "unrepresented": []}, {"task": "Provide an executable query, not prose."}, single_choice=True)
    result = delivery.render(frozen, {"delivery": {"d0": {"text": "Only an explanation, no query."}}, "uncertainties": []})
    assert result["shapeComplete"] and result["declaredCoverageComplete"]
    assert result["semanticCoverage"] == {"status": "not_assessed", "selectedKinds": "model_proposed_presentation",
                                           "originalTaskRemainsAuthoritative": True}
    assert "not_proof_of_task_coverage" in result["declaredCoverageCompleteMeaning"]
    assert result["taskSuccess"] is None and not result["semanticApproval"]
