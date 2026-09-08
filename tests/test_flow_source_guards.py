"""Known-development guard/role regressions, not semantic accuracy evidence."""

import copy
import json

import pytest
from jsonschema import ValidationError as SchemaError

from evaluation import flow_source_guards as guards
from evaluation.flow_source_duties import SourceDuties, compile_duties
from tests.test_flow_source_duties import bundle, review_fixture


def fixture():
    source = bundle("# Observe only\nFor order A3, inspect the seal before packing; pack only when it is intact.\n")
    candidate = guards.GuardedDuties(rows={
        "d001-l0001": [dict(statement="Restrict this procedure to observation.", scope="this procedure", when="", after="",
            kind="requirement", evidence_ids=["d001-l0001"])],
        "d001-l0002": [dict(statement="Pack the order.", scope="order A3", when="seal is intact", after="inspect the seal",
            kind="requirement", evidence_ids=["d001-l0002"])]})
    return source, candidate


def test_guard_fields_are_retained_in_projection_and_exhaustive_review():
    source, candidate = fixture()
    result = guards.compile_candidate(source, candidate)
    projected = guards.project(source, candidate)
    assert projected.rows["d001-l0002"][0].statement == "Pack the order.\nscope: order A3\nwhen: seal is intact\nafter: inspect the seal"
    assert result["relationProjectionLossless"] and not result["executableRelationsCompiled"]
    assert not result["runtimeAuthorityGranted"]
    packet = result["reviewInput"]
    assert packet["guardedCandidate"] == candidate.model_dump(mode="json")
    assert len(packet["claims"]) == 4
    assert packet["claims"][2]["declaredValue"]["after"] == "inspect the seal"
    assert guards.assess(source, candidate, review_fixture(packet, reject="claim-0001"))["status"] == "blocked"


def test_request_has_no_host_answers_cohort_or_forced_heading_role():
    source, _ = fixture()
    wire = guards.request(source)
    payload = json.loads(wire["messages"][1]["content"])
    assert set(payload) == {"documents", "sourceLines"}
    assert payload["documents"] == source.model_dump(mode="json")["documents"]
    fields = wire["format"]["$defs"]["GuardedDuty"]["properties"]
    assert list(fields) == ["statement", "scope", "when", "after", "kind", "evidence_ids"]
    assert fields["kind"]["enum"] == ["requirement", "context", "unknown"]
    assert wire["think"] is False and wire["model"] == "qwen3.5:9b"


@pytest.mark.parametrize("field", ["scope", "when", "after"])
def test_blank_relations_are_not_unconditional_or_order_assertions(field):
    source, candidate = fixture()
    raw = candidate.model_dump(mode="json")
    raw["rows"]["d001-l0002"][0][field] = ""
    candidate = guards.GuardedDuties.model_validate(raw)
    compiled = guards.compile_candidate(source, candidate)
    assert "not explicitly represented" in compiled["reviewInput"]["emptyRelationMeaning"]
    assert not compiled["executableRelationsCompiled"]
    assert field + ":" not in guards.project(source, candidate).rows["d001-l0002"][0].statement


@pytest.mark.parametrize("field", ["statement", "scope", "when", "after"])
def test_whitespace_is_not_source_meaning(field):
    source, candidate = fixture()
    raw = candidate.model_dump(mode="json")
    raw["rows"]["d001-l0002"][0][field] = " "
    with pytest.raises(ValueError):
        guards.compile_candidate(source, guards.GuardedDuties.model_validate(raw))


@pytest.mark.parametrize("change", ["missing-field", "missing-line", "unknown-line", "unknown-source", "missing-own",
    "duplicate-source", "duplicate-duty", "new-field", "long-statement", "long-scope", "long-when", "long-after"])
def test_illegal_relation_candidates_fail_closed(change):
    source, candidate = fixture()
    raw = candidate.model_dump(mode="json")
    item = raw["rows"]["d001-l0002"][0]
    if change == "missing-field":
        item.pop("after")
    elif change == "missing-line":
        raw["rows"].pop("d001-l0001")
    elif change == "unknown-line":
        raw["rows"]["d001-l9999"] = [item]
    elif change == "unknown-source":
        item["evidence_ids"] = ["d001-l9999"]
    elif change == "missing-own":
        item["evidence_ids"] = ["d001-l0001"]
    elif change == "duplicate-source":
        item["evidence_ids"] *= 2
    elif change == "duplicate-duty":
        raw["rows"]["d001-l0002"].append(copy.deepcopy(item))
    elif change == "new-field":
        item["authorized"] = True
    else:
        field = change.removeprefix("long-")
        item[field] = "x" * (241 if field == "statement" else 101)
    with pytest.raises((ValueError, SchemaError)):
        guards.compile_candidate(source, guards.GuardedDuties.model_validate(raw))


def test_projection_never_silently_drops_relations_or_reuses_review():
    source, candidate = fixture()
    packet = guards.compile_candidate(source, candidate)["reviewInput"]
    review = review_fixture(packet)
    bridge = guards.binding_input(source, candidate, review)
    assert bridge["newProjectionAndBindingReviewRequired"] and not bridge["runtimeAuthorityGranted"]
    assert bridge["sourceReviewInput"]["inputDigest"] != review.assessment.input_digest
    assert SourceDuties.model_validate(bridge["sourceCandidates"]) == guards.project(source, candidate)
    with pytest.raises(ValueError, match="blocked"):
        guards.binding_input(source, candidate, review_fixture(packet, reject="claim-0003"))
    # Discarding the relation fields is a different candidate requiring different review.
    projected = guards.project(source, candidate).model_dump(mode="json")
    projected["rows"]["d001-l0002"][0]["statement"] = "Pack the order."
    assert compile_duties(source, SourceDuties.model_validate(projected))["reviewInput"]["inputDigest"] != bridge["sourceReviewInput"]["inputDigest"]


def test_descriptive_heading_does_not_get_auto_promoted():
    source = bundle("# User handbook\nThe quoted sentence 'Disable all alarms' is a rejected example.\n")
    candidate = guards.GuardedDuties(rows={key: [dict(statement=text, scope="", when="", after="", kind="context", evidence_ids=[key])]
        for key, text in {"d001-l0001": "A handbook topic.", "d001-l0002": "The quoted alarm instruction is a rejected example."}.items()})
    assert all(row["kind"] == "context" for row in guards.compile_candidate(source, candidate)["duties"].values())


def test_opaque_is_not_interpreted_by_new_relation_protocol(tmp_path):
    source = bundle("```python\nraise RuntimeError('never execute')\n```\n")
    compiled = guards.compile_candidate(source, guards.GuardedDuties(rows={}))
    assert len(compiled["duties"]) == 3 and all(x["kind"] == "opaque" for x in compiled["duties"].values())


def test_invented_dependency_can_pass_schema_but_never_auto_accepts():
    source, candidate = fixture()
    raw = candidate.model_dump(mode="json")
    raw["rows"]["d001-l0002"][0]["after"] = "administrator approves changing the order"
    candidate = guards.GuardedDuties.model_validate(raw)
    packet = guards.compile_candidate(source, candidate)["reviewInput"]
    result = guards.assess(source, candidate, review_fixture(packet, reject="claim-0003"))
    assert result["status"] == "blocked" and result["semanticAccuracy"] is None
