"""Unknown feasibility never becomes success or permission to run a candidate."""

import itertools

import pytest

from evaluation.flow_behavior_probe import evaluate
from evaluation.flow_guard_binding import slots_for
from evaluation.flow_guard_necessity import bind_necessity
from tests.test_flow_guard_binding import access_without_guards
from tests.test_flow_guard_counterfactual import answer


@pytest.mark.parametrize("no,yes", list(itertools.product(("possible", "forbidden", "unknown"), repeat=2)))
def test_necessity_and_feasibility_are_separate_for_all_truth_pairs(no, yes):
    _, sources, tree = access_without_guards()
    answers = {s["id"]: dict(if_false=answer(no), if_true=answer(yes)) for s in slots_for(sources, tree)}
    result = bind_necessity(sources, tree, answers)
    expected = {("forbidden", "possible"): "require_true", ("forbidden", "unknown"): "require_true",
        ("possible", "forbidden"): "require_false", ("unknown", "forbidden"): "require_false",
        ("possible", "possible"): "not_individually_required"}.get((no, yes), "unresolved")
    assert all(r["decision"]["decision"] == expected for r in result["derivations"])
    assert result["guardCandidateGenerated"] == (expected != "unresolved")
    assert not result["pathFeasibilityProven"] and not result["sufficiencyProven"]
    assert not result["runtimeAuthorityGranted"] and result["activationEligibility"] == "not_established"
    if "unknown" in (no, yes):
        assert len(result["retainedUncertainty"]) == 3
        assert all(row["originalAnswers"] == answers[row["slotId"]] for row in result["retainedUncertainty"])


def test_partial_source_judgment_can_form_inactive_guards_without_filling_answers():
    case, sources, tree = access_without_guards()
    answers = {s["id"]: dict(if_false=answer("forbidden"), if_true=answer("unknown")) for s in slots_for(sources, tree)}
    result = bind_necessity(sources, tree, answers)
    assert all(r["originalAnswers"]["if_true"]["status"] == "unknown" for r in result["derivations"])
    assert result["status"] == "inactive_necessary_guard_candidate"
    # Explicit inert host consent is confined to the existing private test API;
    # successful finite behavior does not clear retained uncertainty.
    assert evaluate(case, result["binding"]["tree"])["passed"] == 11
    assert result["fullSourceReview"] == "required_not_run"
    assert all(r["sourceQuotes"]["if_false"] in sources.source_text for r in result["derivations"])


def test_authoring_entry_keeps_unknowns_and_replays_without_new_calls(tmp_path, monkeypatch):
    from evaluation import flow_guard_counterfactual as cf
    from evaluation.flow_guard_necessity import author_necessity
    _, sources, tree = access_without_guards()
    answers = {s["id"]: dict(if_false=answer("forbidden"), if_true=answer("unknown")) for s in slots_for(sources, tree)}
    generation = dict(result=dict(status="text_received"), **{"answers.json": answers})
    monkeypatch.setattr(cf, "author", lambda *a, **kw: generation)
    root = tmp_path / "necessary-guards"
    with pytest.raises(ValueError, match="budget"):
        author_necessity(sources, tree, root)
    result = author_necessity(sources, tree, root, max_new_calls=1)
    assert result == author_necessity(sources, tree, root)
    assert result["synthesis"]["retainedUncertainty"]
    assert not result["synthesis"]["runtimeAuthorityGranted"]
