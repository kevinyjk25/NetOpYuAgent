import copy
import json

import pytest

from evaluation.hybrid_draft_review import build_review_input, REVIEW_SCHEMA
from evaluation.hybrid_review_roles import groups, materialize, model_input, output_schema
from network_runtime.l0.structured_schema import validate_data
from tests.test_hybrid_draft_review import inputs


def role_response(payload, legacy=None, *, wire=False):
    """Mechanical provider fixture, not a production legacy-response fallback."""
    lookup = {r["claim_id"]: r for r in legacy["claims"]} if legacy else {}
    result = {"input_digest": payload["inputDigest"], "scope_note": "Synthetic structural fixture, not a semantic judgment."}
    for key, claims in groups(payload).items():
        result[key] = []
        for claim in claims:
            prior = lookup.get(claim["claimId"], {})
            row = {"claim_id": claim["claimId"], "rationale": prior.get("rationale", "No semantic support established by this mechanical fixture."),
                   "suggested_revision": prior.get("suggested_revision", "Inspect the original task and observations.")}
            verdict = prior.get("verdict", "insufficient_evidence")
            if key == "coverage_checks":
                row.update(draft_span_id=prior.get("draft_span_id", ""),
                    coverage="preserved" if verdict == "supported" and prior.get("draft_span_id") else "missing")
            else:
                row["source_span_ids"] = prior.get("source_span_ids", [])
                if key == "statement_checks":
                    row["judgment"] = {"supported": "grounded", "contradicted": "conflicts", "insufficient_evidence": "unknown"}[verdict]
                else:
                    row["outcome"] = "satisfied" if verdict == "supported" else "gap"
                    row["draft_span_id"] = prior.get("draft_span_id", "")
                    row["artifact_quote"] = next((s["exactQuote"] for s in payload["draftSpans"]
                                                   if s["draft_span_id"] == row["draft_span_id"]), "")
            result[key].append(row)
    if legacy:
        # Let deliberate missing-ID fixtures remain invalid rather than filling them.
        for key in groups(payload):
            result[key] = [r for r in result[key] if r["claim_id"] in lookup]
    if wire:
        for key in groups(payload):
            rows = result[key]
            assert len({r["claim_id"] for r in rows}) == len(rows)
            result[key] = {r["claim_id"]: {k: v for k, v in r.items() if k != "claim_id"} for r in rows}
    return result


def test_role_groups_are_disjoint_complete_and_preserve_all_original_text():
    payload = build_review_input(inputs())
    before = copy.deepcopy(payload)
    view = model_input(payload)
    assert payload == before
    assert view["reviewInput"]["candidate"] == payload["candidate"]
    assert view["reviewInput"]["originalTask"] == payload["originalTask"]
    assert view["reviewInput"]["hostOpenDuties"] == payload["hostOpenDuties"]
    ids = [c["claimId"] for rows in groups(payload).values() for c in rows]
    assert len(ids) == len(set(ids)) == len(payload["claims"])
    assert set(ids) == {c["claimId"] for c in payload["claims"]}
    encoded = json.dumps(view, ensure_ascii=False)
    for source in payload["sourceSpans"]:
        assert json.dumps(source["exactQuote"], ensure_ascii=False)[1:-1] in encoded


def test_materialization_binds_statement_location_without_model_reattribution():
    payload = build_review_input(inputs())
    raw = role_response(payload)
    converted = materialize(payload, raw)
    validate_data(REVIEW_SCHEMA, converted)
    assert converted["input_digest"] == payload["inputDigest"]
    assert all(r["verdict"] == "insufficient_evidence" for r in converted["claims"])
    for claim, row in zip(payload["claims"], converted["claims"]):
        if claim["facet"] == "all_claims_in_text_block":
            assert row["draft_span_id"] == claim["declaredValue"]["draftSpanId"]


@pytest.mark.parametrize("mutation", ["wrong_group", "duplicate", "missing_id", "invented_source", "no_positive_citation"])
def test_role_decisions_cannot_bypass_coverage_ids_locations_or_citations(mutation):
    payload = build_review_input(inputs())
    raw = role_response(payload)
    if mutation == "wrong_group":
        raw["statement_checks"][0]["claim_id"] = raw["task_checks"][0]["claim_id"]
    elif mutation == "duplicate":
        assert len(raw["task_checks"]) > 1
        raw["task_checks"][1]["claim_id"] = raw["task_checks"][0]["claim_id"]
    elif mutation == "missing_id":
        raw["task_checks"].pop()
    elif mutation == "invented_source":
        raw["statement_checks"][0]["source_span_ids"] = ["not-a-source"]
    else:
        raw["task_checks"][0].update(outcome="satisfied", source_span_ids=[])
    with pytest.raises(ValueError):
        materialize(payload, raw)


def test_unlocated_positive_is_withheld_without_discarding_other_findings():
    from evaluation.hybrid_review_roles import binding_issues
    payload = build_review_input(inputs())
    raw = role_response(payload)
    raw["coverage_checks"][0].update(coverage="preserved", draft_span_id="")
    before = copy.deepcopy(raw)
    result = materialize(payload, raw)
    key = raw["coverage_checks"][0]["claim_id"]
    row = next(row for row in result["claims"] if row["claim_id"] == key)
    assert row["verdict"] == "insufficient_evidence" and row["draft_span_id"] == ""
    assert row["rationale"].startswith("Host withheld")
    assert len(result["claims"]) == len(payload["claims"])
    assert binding_issues(raw, payload)[0]["disposition"] == "withheld_as_insufficient_evidence_not_supported"
    assert raw == before
    validate_data(REVIEW_SCHEMA, result)


def test_preserved_and_explicit_irrelevance_are_different_raw_decisions_not_host_truth():
    payload = build_review_input(inputs())
    raw = role_response(payload)
    raw["coverage_checks"][0].update(coverage="preserved", draft_span_id=payload["draftSpans"][0]["draft_span_id"])
    before = copy.deepcopy(raw)
    result = materialize(payload, raw)
    assert raw == before
    key = raw["coverage_checks"][0]["claim_id"]
    assert next(r for r in result["claims"] if r["claim_id"] == key)["verdict"] == "supported"
    raw["coverage_checks"][0].update(coverage="not_required", draft_span_id="")
    assert materialize(payload, raw) != result
    # Neither result has an authority/approval field; raw decisions remain separately archived.
    assert set(result) == {"input_digest", "claims", "scope_note"}


def test_empty_statement_group_is_legal_without_an_empty_enum():
    supplied = inputs()
    supplied["candidate"].update(draft="", notes=[])
    payload = build_review_input(supplied)
    raw = role_response(payload)
    assert raw["statement_checks"] == []
    validate_data(output_schema(payload), raw)


def test_keyed_wire_owns_ids_and_normalizes_without_semantic_changes():
    from evaluation.hybrid_review_roles import wire_schema, materialize_wire
    payload = build_review_input(inputs())
    legacy = role_response(payload)
    keyed = role_response(payload, wire=True)
    before = copy.deepcopy(keyed)
    schema = wire_schema(payload)
    for group, claims in groups(payload).items():
        assert set(schema["properties"][group]["properties"]) == {c["claimId"] for c in claims}
        assert set(schema["properties"][group]["required"]) == set(keyed[group])
        assert all(row == {"$ref": "#/$defs/" + group} for row in schema["properties"][group]["properties"].values())
        assert all("claim_id" not in row for row in keyed[group].values())
    assert materialize_wire(payload, keyed) == materialize(payload, legacy)
    assert keyed == before


@pytest.mark.parametrize("mutation", ["missing", "extra", "wrong_group", "array", "nested_id"])
def test_keyed_generation_cannot_hide_missing_or_foreign_checks(mutation):
    from evaluation.hybrid_review_roles import materialize_wire
    payload = build_review_input(inputs())
    raw = role_response(payload, wire=True)
    key = next(iter(raw["task_checks"]))
    if mutation == "missing":
        raw["task_checks"].pop(key)
    elif mutation == "extra":
        raw["task_checks"]["invented"] = copy.deepcopy(raw["task_checks"][key])
    elif mutation == "wrong_group":
        raw["statement_checks"][key] = raw["task_checks"].pop(key)
    elif mutation == "array":
        raw = role_response(payload)
    else:
        raw["task_checks"][key]["claim_id"] = "invented"
    with pytest.raises(ValueError):
        materialize_wire(payload, raw)


def test_keyed_generation_still_withholds_locationless_positive():
    from evaluation.hybrid_review_roles import materialize_wire, wire_binding_issues
    payload = build_review_input(inputs())
    raw = role_response(payload, wire=True)
    key = next(iter(raw["coverage_checks"]))
    raw["coverage_checks"][key].update(coverage="preserved", draft_span_id="")
    converted = materialize_wire(payload, raw)
    assert next(r for r in converted["claims"] if r["claim_id"] == key)["verdict"] == "insufficient_evidence"
    assert wire_binding_issues(raw, payload)[0]["claimId"] == key


@pytest.mark.parametrize("quote", ["", " ", "The required work has been completed.", "Read the project"])
def test_task_citation_alone_cannot_establish_delivered_fulfillment(quote):
    from evaluation.hybrid_review_roles import materialize_wire, wire_binding_issues
    payload = build_review_input(inputs())
    raw = role_response(payload, wire=True)
    key = next(iter(raw["task_checks"]))
    raw["task_checks"][key].update(outcome="satisfied", source_span_ids=["s000"],
        draft_span_id=payload["draftSpans"][0]["draft_span_id"], artifact_quote=quote)
    before = copy.deepcopy(raw)
    converted = materialize_wire(payload, raw)
    row = next(r for r in converted["claims"] if r["claim_id"] == key)
    assert row["verdict"] == "insufficient_evidence" and row["draft_span_id"] == ""
    assert wire_binding_issues(raw, payload)[0]["code"] == "task_fulfillment_without_exact_artifact_witness"
    assert raw == before


def test_exact_artifact_witness_is_not_semantic_proof_even_for_a_heading():
    from evaluation.hybrid_draft_review import assess_review
    payload = build_review_input(inputs())
    raw = role_response(payload)
    span = payload["draftSpans"][0]
    row = raw["task_checks"][0]
    row.update(outcome="satisfied", source_span_ids=["s000"],
               draft_span_id=span["draft_span_id"], artifact_quote=span["exactQuote"])
    normalized = materialize(payload, raw)
    assert next(r for r in normalized["claims"] if r["claim_id"] == row["claim_id"])["verdict"] == "supported"
    # The host locates a real heading; it cannot prove fulfillment by quotation.
    assessed = assess_review(payload, normalized)
    assert not assessed["completeAnswerApproved"] and not assessed["semanticClaimCoverageProven"]


def test_gap_with_wrong_quoted_target_stays_open_not_assigned_to_arbitrary_body():
    payload = build_review_input(inputs())
    raw = role_response(payload)
    row = raw["task_checks"][0]
    row.update(outcome="gap", draft_span_id=payload["draftSpans"][0]["draft_span_id"], artifact_quote="not in this artifact")
    result = materialize(payload, raw)
    bound = next(r for r in result["claims"] if r["claim_id"] == row["claim_id"])
    assert bound["verdict"] == "insufficient_evidence" and bound["draft_span_id"] == ""
