import copy

import pytest

from evaluation.hybrid_draft_review import build_review_input
from evaluation import hybrid_semantic_witness as w
from tests.test_hybrid_draft_review import inputs


def fixture():
    payload = build_review_input(inputs())
    source = w.source_input(payload)
    raw = {"source_digest": source["reportDigest"], "requirements": [{
        "task_quote": source["originalTask"], "required_content_or_boundary": "Draft scoped documentation without execution."}],
        "observations": {s["source_span_id"]: {"usable_content": "Mechanical fixture", "limits": "No semantic assertion"}
                         for s in source["sourceSpans"] if s["kind"] == "observation"}}
    plan = w.bind_plan(source, raw)
    supplied = w.check_input(payload, plan)
    check = {"input_digest": supplied["reportDigest"], "requirements": {"r000": {
        "artifact_location": "", "artifact_quote": "", "explanation": "No fulfillment established.", "outcome": "unknown", "correction": ""}},
        "observation_reconciliation": {key: {"candidate_location": "", "candidate_quote": "", "source_quote": "",
            "explanation": "No semantic comparison established.", "outcome": "unknown", "correction": ""}
            for key in raw["observations"]}, "other_problems": [], "scope_note": "Fixture only, no meaning proved."}
    return payload, source, raw, plan, supplied, check


def test_planner_is_candidate_blind_even_when_draft_and_prior_duties_change():
    supplied = inputs()
    first = w.source_input(build_review_input(supplied))
    supplied["candidate"].update(draft="PLAN CANARY", notes=["another canary"], values={"x": 9})
    supplied["open_duties"] = "candidate-derived action canary"
    assert w.source_input(build_review_input(supplied)) == first
    assert "candidate" not in w.plan_input(first)
    assert not first["interpretationsAreEvidence"]


def test_complete_original_evidence_and_candidate_remain_in_comparison():
    payload, source, _, plan, supplied, _ = fixture()
    assert supplied["originalSourceContext"] == source
    assert supplied["candidate"] == payload["candidate"]
    assert supplied["unverifiedPlan"] == plan
    assert not plan["requirementCompletenessProven"]
    assert not supplied["semanticApproval"]


@pytest.mark.parametrize("mutation", ["foreign_quote", "blank_quote", "missing_observation", "extra_observation", "wrong_digest"])
def test_invalid_plan_does_not_silently_drop_or_rewrite_task_evidence(mutation):
    _, source, raw, _, _, _ = fixture()
    if mutation == "foreign_quote":
        raw["requirements"][0]["task_quote"] = "invented task duty"
    elif mutation == "blank_quote":
        raw["requirements"][0]["task_quote"] = " "
    elif mutation == "missing_observation":
        raw["observations"].pop(next(iter(raw["observations"])))
    elif mutation == "extra_observation":
        raw["observations"]["s999"] = {"usable_content": "", "limits": ""}
    else:
        raw["source_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError):
        w.bind_plan(source, raw)


def test_source_drift_rejects_plan_but_new_candidate_same_sources_is_allowed():
    payload, _, _, plan, _, _ = fixture()
    changed = inputs()
    changed["candidate"]["draft"] = "A newly revised candidate."
    assert w.check_input(build_review_input(changed), plan)["candidate"] != payload["candidate"]
    changed["original_task"] += " Also deploy it."
    with pytest.raises(ValueError, match="different source"):
        w.check_input(build_review_input(changed), plan)


@pytest.mark.parametrize("location,quote", [("", ""), ("d000", "task was completed"), ("d000", " ")])
def test_positive_fulfillment_without_exact_artifact_is_withheld(location, quote):
    _, _, _, _, supplied, raw = fixture()
    raw["requirements"]["r000"].update(outcome="met", artifact_location=location, artifact_quote=quote)
    before = copy.deepcopy(raw)
    result = w.bind_check(supplied, raw)
    assert raw == before
    assert result["requirements"]["r000"]["outcome"] == "unknown"
    assert result["bindingIssues"][0]["code"] == "fulfillment_without_exact_artifact_witness"


def test_exact_quote_is_location_not_semantic_fulfillment_proof():
    _, _, _, _, supplied, raw = fixture()
    raw["requirements"]["r000"].update(outcome="met", artifact_location="d000",
                                        artifact_quote=supplied["candidateLocations"]["d000"]["text"])
    result = w.bind_check(supplied, raw)
    assert result["requirements"]["r000"]["outcome"] == "met"
    assert not result["requirements"]["r000"]["semanticFulfillmentProven"]
    assert not result["completeAnswerApproved"] and not result["noFindingsMeansSuccess"]


def test_conflict_requires_two_exact_locations_and_preserves_other_gap():
    _, source, _, _, supplied, raw = fixture()
    key = next(iter(raw["observation_reconciliation"]))
    row = raw["observation_reconciliation"][key]
    row.update(outcome="conflict", candidate_location="n000", candidate_quote=supplied["candidateLocations"]["n000"]["text"],
               source_quote="invented quote")
    raw["requirements"]["r000"]["outcome"] = "gap"
    failed = w.bind_check(supplied, raw)
    assert failed["observationReconciliation"][key]["outcome"] == "unknown"
    assert failed["findings"][0]["kind"] == "requirement_gap"
    row["source_quote"] = next(s["exactQuote"] for s in source["sourceSpans"] if s["source_span_id"] == key)
    bound = w.bind_check(supplied, raw)
    assert bound["observationReconciliation"][key]["twoSidedWitnessLocated"]
    assert not bound["observationReconciliation"][key]["semanticConflictProven"]


@pytest.mark.parametrize("field", ["requirements", "observation_reconciliation"])
def test_check_cannot_drop_mandatory_task_or_observation_cell(field):
    _, _, _, _, supplied, raw = fixture()
    raw[field].pop(next(iter(raw[field])))
    with pytest.raises(ValueError):
        w.bind_check(supplied, raw)


def test_unbound_problem_does_not_become_repair_instruction():
    _, _, _, _, supplied, raw = fixture()
    raw["other_problems"] = [{"candidate_location": "n000", "candidate_quote": "missing candidate text",
        "source_span_ids": ["s000"], "explanation": "Potential semantic problem.", "correction": "Do something."}]
    result = w.bind_check(supplied, raw)
    assert not result["findings"] and result["bindingIssues"]


def test_capacity_never_truncates_original_records():
    supplied = inputs()
    supplied["observations"]["n0"]["observations"] = {"read": [f"entry {i}" for i in range(33)]}
    with pytest.raises(ValueError, match="never truncate"):
        w.source_input(build_review_input(supplied))


def test_mutated_bound_report_is_rejected():
    payload, _, _, plan, supplied, raw = fixture()
    plan["requirements"][0]["required_content_or_boundary"] = "perform write"
    with pytest.raises(ValueError, match="digest drift"):
        w.check_input(payload, plan)
    supplied["candidate"]["draft"] = "replaced"
    with pytest.raises(ValueError, match="digest drift"):
        w.bind_check(supplied, raw)
