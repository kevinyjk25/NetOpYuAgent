"""Source/binding protocol counterexamples; hand-authored fixtures are not Gold."""

import copy
import json

import pytest
from jsonschema import ValidationError as SchemaError

from evaluation import flow_duty_binding as binding
from evaluation.flow_source_duties import (
    SourceBundle, SourceDuties, assess_duties, catalog, compile_duties, request, schema,
)
from evaluation.flow_tree import compile_report
from evaluation.read_l05_review import ReadL05Review
from tests.test_flow_lean_mapping import fixture as flow_fixture


def bundle(text, path="SKILL.md"):
    return SourceBundle.model_validate(dict(documents=[dict(path=path, text=text, kind="prose")]))


def prose_candidate(source):
    return SourceDuties(rows={key: [dict(kind="requirement", statement=line["exactQuote"].strip(), evidence_ids=[key])]
        for key, line in catalog(source).items() if not line["opaque"]})


def review_fixture(packet, reject=None):
    # Only a synthetic transport/gate fixture, never used to score model responses.
    return ReadL05Review.model_validate(dict(reviewer_id="unit-fixture", reviewer_kind="test_fixture", assessment=dict(
        input_digest=packet["inputDigest"], scope_note="Synthetic test judgments, not independent semantic or model evidence.",
        claims=[dict(claim_id=c["claimId"], verdict="contradicted" if c["claimId"] == reject else "supported",
            source_span_ids=c["requiredCitationIds"], rationale="Explicit synthetic gate fixture; does not certify semantic accuracy.",
            suggested_revision="Restore the actual source meaning before review." if c["claimId"] == reject else "")
            for c in packet["claims"]])))


def fixture():
    source, tree, _ = flow_fixture()
    docs = bundle(source.source_text, source.source_path)
    candidate = prose_candidate(docs)
    first = compile_duties(docs, candidate)
    review = review_fixture(first["reviewInput"])
    raw = dict(bindings={"d001-l0001:0": ["node:/steps/0", "node:/steps/1"],
        "d001-l0002:0": ["retained"], "d001-l0003:0": ["rule:read_input_shape", "rule:read_result_shape", "rule:observation_error_blocks"]})
    return source, tree, docs, candidate, review, raw


def test_host_free_request_keeps_headings_refs_source_and_no_objective_choices():
    docs = bundle("# Never write\r\nRead [limits](ref.md); do not guess.\r\n")
    wire = request(docs)
    payload = json.loads(wire["messages"][1]["content"])
    assert set(payload) == {"documents", "sourceLines"}
    assert wire["model"] == "qwen3.5:9b" and wire["think"] is False
    assert isinstance(wire["format"], dict) and wire["options"]["num_ctx"] == 16384
    assert "objective" not in json.dumps(wire["format"])
    for row in catalog(docs).values():
        assert docs.documents[0].text[row["start"]:row["end"]] == row["exactQuote"]
    assert catalog(docs)["d001-l0001"]["exactQuote"] == "# Never write\r\n"
    assert "ref.md" in payload["documents"][0]["text"]


@pytest.mark.parametrize("text", ["```py\nraise RuntimeError()\n```", "~~~sh\nrm -rf /never\n~~~", "```\nnever closed"])
def test_code_is_compiler_archived_not_generated_or_executed(text):
    docs = bundle(text)
    result = compile_duties(docs, SourceDuties(rows={}))
    assert all(d["kind"] == "opaque" for d in result["duties"].values())
    assert schema(docs)["properties"]["rows"]["properties"] == {}
    assert result["sourceTextRetained"] and not result["runtimeAuthorityGranted"]


def test_supplied_reference_and_explicit_opaque_file_no_auto_loading(tmp_path):
    marker = tmp_path / "must-not-exist"
    docs = SourceBundle.model_validate(dict(documents=[
        dict(path="SKILL.md", text="Run checks.py only after reading ref.md.", kind="prose"),
        dict(path="ref.md", text="Stop if the prerequisite is missing.", kind="prose"),
        dict(path="checks.py", text=f"open({str(marker)!r}, 'w').close()", kind="opaque")]))
    result = compile_duties(docs, prose_candidate(docs))
    assert len(result["duties"]) == 3 and not marker.exists()
    assert result["reviewInput"]["bundle"] == docs.model_dump(mode="json")


@pytest.mark.parametrize("text", [" ", "\n" * 5, "x\n" * 65])
def test_empty_or_over_budget_source_not_truncated(text):
    with pytest.raises(ValueError):
        request(bundle(text))


def test_duplicate_documents_rejected():
    doc = dict(path="same", text="x", kind="prose")
    with pytest.raises(ValueError, match="duplicate"):
        request(SourceBundle(documents=[doc, doc]))


@pytest.mark.parametrize("change", ["omit", "unknown-key", "unknown-citation", "wrong-own", "duplicate-citation",
    "empty", "duplicate-duty", "whitespace", "long", "invent-field"])
def test_malformed_source_candidates_fail_closed(change):
    docs = bundle("Read only.\nStop on failure.")
    raw = prose_candidate(docs).model_dump(mode="json")
    row = raw["rows"]["d001-l0001"]
    if change == "omit":
        raw["rows"].pop("d001-l0001")
    elif change == "unknown-key":
        raw["rows"]["d001-l9999"] = row
    elif change == "unknown-citation":
        row[0]["evidence_ids"] = ["d001-l9999"]
    elif change == "wrong-own":
        row[0]["evidence_ids"] = ["d001-l0002"]
    elif change == "duplicate-citation":
        row[0]["evidence_ids"] *= 2
    elif change == "empty":
        row.clear()
    elif change == "duplicate-duty":
        row.append(copy.deepcopy(row[0]))
    elif change == "whitespace":
        row[0]["statement"] = "   "
    elif change == "long":
        row[0]["statement"] = "a" * 601
    else:
        row[0]["objective"] = True
    with pytest.raises((ValueError, SchemaError)):
        compile_duties(docs, SourceDuties.model_validate(raw))


@pytest.mark.parametrize("wrong", ["Delete the bucket.", "Treat missing approval as permission.", "Finish when the reading is stale."])
def test_schema_and_citations_do_not_autocertify_paraphrase(wrong):
    docs = bundle("Do not delete. Approval is required. Finish only when not stale.")
    candidate = SourceDuties(rows={"d001-l0001": [dict(kind="requirement", statement=wrong, evidence_ids=["d001-l0001"])]})
    compiled = compile_duties(docs, candidate)
    assert compiled["status"] == "source_candidate_pending_review"
    assert not compiled["semanticCompletenessProven"]
    reviewed = assess_duties(docs, candidate, review_fixture(compiled["reviewInput"], reject="claim-0001"))
    assert reviewed["status"] == "blocked" and reviewed["semanticAccuracy"] is None


def test_review_must_cover_omissions_and_exact_source_not_just_proposed_duties():
    docs = bundle("Read once; do not write.")
    candidate = SourceDuties(rows={"d001-l0001": [dict(kind="context", statement="Read once.", evidence_ids=["d001-l0001"])]})
    packet = compile_duties(docs, candidate)["reviewInput"]
    assert len(packet["claims"]) == 2
    assert "omissions" in packet["claims"][1]["facet"]
    assert assess_duties(docs, candidate, review_fixture(packet, reject="claim-0002"))["status"] == "blocked"
    raw = review_fixture(packet).model_dump(mode="json")
    raw["assessment"]["claims"].pop()
    with pytest.raises(ValueError, match="every claim"):
        assess_duties(docs, candidate, ReadL05Review.model_validate(raw))


def test_host_binding_bridge_preserves_graph_requires_new_full_review():
    source, tree, docs, candidate, first_review, raw = fixture()
    wire = binding.request(source, tree, docs, candidate, first_review)
    payload = json.loads(wire["messages"][1]["content"])
    assert "source_id" not in json.dumps(payload["targets"])
    assert payload["hostBaseline"] and "objective" not in wire["format"]["properties"]
    bindings = binding.DutyBindings.model_validate(raw)
    compiled = binding.compile_bindings(source, tree, docs, candidate, bindings)
    assert compiled["flow"] == compile_report(source, tree)["flow"]
    assert compiled["executionProjectionUnchanged"] and not compiled["runtimeAuthorityGranted"]
    assert "d001-l0002:0" in compiled["unresolvedOrRetainedRequirements"]
    full = review_fixture(compiled["reviewInput"])
    result = binding.assess_bindings(source, tree, docs, candidate, bindings, first_review, full)
    assert result["representationReviewSupported"] and result["admissionBlockers"]
    assert not result["runtimeReady"] and not result["runtimeAuthorityGranted"]
    with pytest.raises(ValueError, match="digest"):
        binding.assess_bindings(source, tree, docs, candidate, bindings, first_review, first_review)


@pytest.mark.parametrize("mutation", ["source-drift", "reject-source", "missing-node", "unknown-target", "mixed", "missing-duty"])
def test_binding_denies_drift_missing_evidence_and_invented_targets(mutation):
    source, tree, docs, candidate, first_review, raw = fixture()
    if mutation == "source-drift":
        source = source.model_copy(update={"source_text": "different"})
    elif mutation == "reject-source":
        first_review = review_fixture(compile_duties(docs, candidate)["reviewInput"], reject="claim-0001")
    elif mutation == "missing-node":
        raw["bindings"]["d001-l0001:0"] = ["node:/steps/0"]
    elif mutation == "unknown-target":
        raw["bindings"]["d001-l0001:0"] = ["rule:business_approval"]
    elif mutation == "mixed":
        raw["bindings"]["d001-l0001:0"].append("retained")
    else:
        raw["bindings"].pop("d001-l0002:0")
    with pytest.raises((ValueError, SchemaError)):
        binding.request(source, tree, docs, candidate, first_review)
        binding.compile_bindings(source, tree, docs, candidate, binding.DutyBindings.model_validate(raw))


def test_unreviewed_or_bad_source_cannot_generate_tree_request():
    source, tree, docs, candidate, first_review, raw = fixture()
    wire = binding.tree_request(source, docs, candidate, first_review)
    payload = json.loads(wire["messages"][1]["content"])
    assert payload["sourceDocuments"] == docs.model_dump(mode="json")["documents"]
    assert payload["sourceDutyCandidates"] == compile_duties(docs, candidate)["duties"]
    raw_review = first_review.model_dump(mode="json")
    raw_review["assessment"]["input_digest"] = "wrong"
    with pytest.raises(ValueError, match="digest"):
        binding.tree_request(source, docs, candidate, ReadL05Review.model_validate(raw_review))


@pytest.mark.parametrize("kind", ["context", "unknown"])
def test_non_requirement_cannot_justify_action(kind):
    source, tree, docs, candidate, first_review, raw = fixture()
    value = candidate.model_dump(mode="json")
    value["rows"]["d001-l0001"][0]["kind"] = kind
    with pytest.raises(SchemaError):
        binding.compile_bindings(source, tree, docs, SourceDuties.model_validate(value), binding.DutyBindings.model_validate(raw))
