from copy import deepcopy
import json
import pytest

from evaluation.hybrid_draft_review import build_review_input
from evaluation.hybrid_draft_slots import editing_slots
from evaluation.hybrid_repair_plan import plan_repairs
from evaluation import hybrid_repair_cells as cells, hybrid_draft_loop
from tests.test_hybrid_draft_review import inputs, raw_review
from tests.test_hybrid_review_roles import role_response


def fixture():
    original = inputs()
    original["candidate"]["draft"] = "# Report\n\nOnly a candidate.\n\n## Limits\n\nNot approved.\n"
    payload = build_review_input(original)
    review = raw_review(payload)
    for row in review["claims"]:
        row.update(verdict="supported", source_span_ids=[payload["sourceSpans"][-1]["source_span_id"]],
                   draft_span_id=payload["draftSpans"][0]["draft_span_id"], suggested_revision="")
    return original, payload, editing_slots(payload, complete_sections=True)["slots"], review


def test_positive_opinions_or_lexical_difference_do_not_schedule_regeneration():
    _, payload, slots, review = fixture()
    plan = plan_repairs(payload, slots, review)
    assert not any(plan["assignments"].values())
    assert not plan["lexicalDifferenceAloneTriggersEdit"] and not plan["authorityGranted"]
    assert plan["noFindingMeans"] == "retain_unverified_candidate_not_semantic_approval"
    assert not plan_repairs(payload, slots, None)["priorReviewAvailable"]


def test_negative_statement_uses_host_address_not_model_echo():
    _, payload, slots, review = fixture()
    unit = next(c for c in payload["claims"] if c["facet"] == "all_claims_in_text_block")
    row = next(c for c in review["claims"] if c["claim_id"] == unit["claimId"])
    row.update(verdict="insufficient_evidence", rationale="A specific claim needs inspection.", draft_span_id="foreign")
    plan = plan_repairs(payload, slots, review)
    assert len(plan["assignments"]["e00"]) == 1 and not plan["assignments"]["e01"]
    assert plan["assignments"]["e00"][0]["draftSpanId"] == unit["declaredValue"]["draftSpanId"]


def test_unlocated_omission_is_not_broadcast_or_erased():
    _, payload, slots, review = fixture()
    unit = next(c for c in payload["claims"] if c["facet"] == "observation_to_draft")
    row = next(c for c in review["claims"] if c["claim_id"] == unit["claimId"])
    row.update(verdict="insufficient_evidence", rationale="Responsibility missing; location not established.", draft_span_id="")
    plan = plan_repairs(payload, slots, review)
    assert not any(plan["assignments"].values())
    assert plan["unlocatedFindings"][0]["claimId"] == unit["claimId"]
    assert not plan["unlocatedFindings"][0]["semanticProblemProven"]


def test_whole_draft_owner_can_inspect_missing_content_without_inventing_span():
    original, _, _, _ = fixture()
    original["candidate"]["draft"] = "A compact answer with no separate editable sections."
    payload = build_review_input(original)
    slots = editing_slots(payload, complete_sections=True)["slots"]
    review = raw_review(payload)
    unit = next(c for c in payload["claims"] if c["facet"] == "observation_to_draft")
    for row in review["claims"]:
        row.update(verdict="supported", draft_span_id="")
    row = next(c for c in review["claims"] if c["claim_id"] == unit["claimId"])
    row.update(verdict="insufficient_evidence", rationale="An observed responsibility is omitted.")
    plan = plan_repairs(payload, slots, review)
    finding = plan["assignments"][slots[0]["id"]][0]
    assert finding["locationBasis"] == "unique_whole_draft_owner_not_semantic_location"
    assert finding["draftSpanId"] == "" and not finding["semanticProblemProven"]
    assert not plan["unlocatedFindings"]


def test_selecting_one_of_multiple_cells_does_not_make_it_whole_draft_owner():
    _, payload, slots, review = fixture()
    for row in review["claims"]:
        row.update(verdict="insufficient_evidence", draft_span_id="")
    plan = plan_repairs(payload, slots[:1], review)
    assert plan["unlocatedFindings"]
    assert all(f["locationBasis"] != "unique_whole_draft_owner_not_semantic_location"
               for findings in plan["assignments"].values() for f in findings)


@pytest.mark.parametrize("whole_draft", [False, True])
@pytest.mark.parametrize("echo_body", [False, True])
def test_note_problem_never_targets_body_even_when_model_echoes_a_valid_body_id(whole_draft, echo_body):
    original, _, _, _ = fixture()
    if whole_draft:
        original["candidate"]["draft"] = "One complete owned draft."
    payload = build_review_input(original)
    review = raw_review(payload)
    for row in review["claims"]:
        row.update(verdict="supported", draft_span_id="")
    unit = next(c for c in payload["claims"] if c["pointer"].startswith("/candidate/notes/"))
    row = next(r for r in review["claims"] if r["claim_id"] == unit["claimId"])
    row.update(verdict="contradicted", source_span_ids=["s000"],
               draft_span_id=payload["draftSpans"][0]["draft_span_id"] if echo_body else "")
    planned = plan_repairs(payload, editing_slots(payload, complete_sections=True)["slots"], review)
    assert not any(planned["assignments"].values())
    assert planned["unlocatedFindings"][0]["candidatePointer"] == unit["pointer"]
    assert planned["unlocatedFindings"][0]["locationBasis"] == "host_note_pointer"
    assert planned["unlocatedFindings"][0]["draftSpanId"] == ""


def test_default_retains_candidate_without_editor_or_repair_credit(tmp_path, monkeypatch):
    original, payload, _, review = fixture()
    before = deepcopy(original)
    monkeypatch.setattr(cells, "source_inputs", lambda _: (original, {"fixture": True}, review))
    monkeypatch.setattr(cells, "author_once", lambda *a, **k: (_ for _ in ()).throw(AssertionError("unlocated editor invoked")))
    calls = []
    def review_once(folder, request, derive, **kwargs):
        supplied = request["governedRequest"]["inputs"]
        calls.append(supplied)
        material = build_review_input(supplied)
        value = role_response(material, raw_review(material), wire=True)
        envelope = {"httpStatus": 200, "latencyMs": 1, "body": json.dumps({"model": "qwen3.5:9b", "done": True,
            "done_reason": "stop", "message": {"content": json.dumps(value)}, "prompt_eval_count": 10, "eval_count": 10})}
        files, cost = derive(envelope)
        return {**files, "result": cost}
    monkeypatch.setattr(hybrid_draft_loop, "author_once", review_once)
    report = cells.run(tmp_path / "prior", tmp_path / "result", edit_mode="grounded_patch")
    assert report["execution"]["modelCallsReserved"] == 0
    assert all(t.get("modelInvoked") is False for t in report["execution"]["trace"])
    candidate = json.loads((tmp_path / "result/materialized/candidate.json").read_text())
    assert candidate == before["candidate"]
    assert original == before and len(calls) == 1
    assert report["semanticSuccess"] is None and not report["completeAnswerApproved"]
