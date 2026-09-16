from copy import deepcopy
import pytest

from evaluation import hybrid_duty_contract as d
from evaluation.hybrid_draft_review import build_review_input
from tests.test_hybrid_draft_review import inputs


def fixture():
    raw = inputs()
    payload = build_review_input(raw)
    source = d.source_input(payload)
    proposal = {"duties": [{"task_quote": raw["original_task"], "description": "Draft supported documentation."}],
                "overflow": False, "unrepresented_duties": ""}
    plan = d.bind_plan(source, proposal)
    return raw, payload, source, proposal, plan, d.context(payload, plan)


def test_candidate_and_prior_duty_changes_do_not_leak_to_planner():
    raw, _, source, _, _, _ = fixture()
    raw["candidate"] = {"values": {"canary": "SECRET"}, "draft": "CANARY", "notes": ["secret note"]}
    raw["open_duties"] = "PRIOR REVIEW CANARY"
    assert source == d.source_input(build_review_input(raw))


@pytest.mark.parametrize("mutation", ["quote", "capacity", "undisclosed", "invented_overflow", "source_seal"])
def test_plan_cannot_hide_capacity_or_invent_task_anchors(mutation):
    _, _, source, raw, _, _ = fixture()
    if mutation == "quote":
        raw["duties"][0]["task_quote"] = "invented"
    elif mutation == "capacity":
        raw["duties"] *= 7
    elif mutation == "undisclosed":
        raw["overflow"] = True
    elif mutation == "invented_overflow":
        raw["unrepresented_duties"] = "some duties"
    else:
        source["originalTask"] += "drift"
    with pytest.raises(ValueError):
        d.bind_plan(source, raw)


def test_new_candidate_can_reuse_plan_but_new_source_cannot():
    raw, _, _, _, plan, _ = fixture()
    raw["candidate"]["draft"] = "A different answer"
    assert d.context(build_review_input(raw), plan)["candidateLocations"]
    raw["original_task"] += "do another thing"
    with pytest.raises(ValueError, match="drift"):
        d.context(build_review_input(raw), plan)


def test_positive_opinion_without_witness_stays_unknown():
    _, _, _, _, plan, ctx = fixture()
    result = d.bind_check(ctx, plan["duties"][0], {"outcome": "met", "artifact_location": "", "artifact_quote": "",
        "explanation": "An ungrounded positive judgment.", "issues": []})
    assert result["outcome"] == "unknown" and result["bindingIssues"]
    assert not result["semanticApproval"]


def test_complete_note_inventory_is_required_and_predicate_coverage_not_claimed():
    _, _, _, _, _, ctx = fixture()
    with pytest.raises(ValueError):
        d.bind_notes(ctx, {})
    src = ctx["sourceContext"]["sourceSpans"][0]
    raw = {"n000": {"atoms": [{"quote": ctx["candidateLocations"]["n000"], "status": "supported",
        "evidence": [{"source_id": src["source_span_id"], "quote": src["exactQuote"]}], "explanation": "Fixture, not entailment."}]}}
    result = d.bind_notes(ctx, raw)
    assert not result["notes"]["n000"]["needsInspection"]
    assert not result["notes"]["n000"]["allPredicatesEnumeratedProven"]
    raw["n000"]["atoms"][0]["quote"] = "missing"
    result = d.bind_notes(ctx, raw)
    assert result["notes"]["n000"]["needsInspection"] and result["bindingIssues"]


def test_note_atoms_keep_true_and_unknown_predicates_separate():
    _, _, _, _, _, ctx = fixture()
    ctx = deepcopy(ctx)
    ctx["candidateLocations"]["n000"] = "Case is not reviewed or approved."
    src = ctx["sourceContext"]["sourceSpans"][0]
    atoms = [{"quote": q, "status": s, "evidence": [{"source_id": src["source_span_id"], "quote": src["exactQuote"]}],
              "explanation": "Transport fixture, no semantic judgment."} for q, s in [("reviewed", "unknown"), ("approved", "supported")]]
    result = d.bind_notes(ctx, {"n000": {"atoms": atoms}})
    assert [a["status"] for a in result["notes"]["n000"]["atoms"]] == ["unknown", "supported"]
    assert result["notes"]["n000"]["needsInspection"]
