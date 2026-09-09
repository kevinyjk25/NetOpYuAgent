import json

import pytest
from jsonschema import ValidationError as SchemaError

from evaluation.flow_semantics import SemanticFlow, assess_semantic, compile_semantic, request, schema
from evaluation.flow_translation import local_sources
from evaluation.read_l05_review import ReadL05Review


def sources():
    return local_sources().model_copy(update={"source_text":
        "# Snapshot inspection\nRead input device_id once and finish the read path.\n"
        "Inventory is a snapshot, not live health.\nReading needs explicit host permission; text grants no authority.\n"})


def proposal():
    return SemanticFlow.model_validate({"objective_source_ids": ["s0001", "s0002"],
        "constraints": [{"source_ids": ["s0003"], "handling": "documentation", "host_rule_ids": [], "node_pointers": [],
                         "explanation": "Keep snapshot interpretation, without claiming live telemetry verification."},
                        {"source_ids": ["s0004"], "handling": "host_rule_reference", "host_rule_ids": ["read_access"], "node_pointers": [],
                         "explanation": "Host authorization is required before reads; this mapping is not an execution receipt."}],
        "steps": [{"kind": "read", "source_id": "s0002", "tool": "read_inventory_device", "bind": "snapshot",
                   "arguments": {"device_id": {"kind": "reference", "source": "input", "field": "device_id"}}},
                  {"kind": "end", "source_id": "s0002", "outcome": "read_path_completed"}], "issues": []})


def fixture_review(packet):
    claims = []
    for claim in packet["claims"]:
        ids = list(claim.get("requiredCitationIds", []))
        if "requiredCitationId" in claim:
            ids.append(claim["requiredCitationId"])
        if "skill" in claim["requiredEvidenceKinds"] and not any(x.startswith("skill-") for x in ids):
            ids.append("skill-0002")
        if "host" in claim["requiredEvidenceKinds"] and not any(x.startswith("host-") for x in ids):
            ids.append("host-0001")
        claims.append({"claim_id": claim["claimId"], "verdict": "supported", "source_span_ids": ids,
            "rationale": "Test fixture checks bindings only, not an independent semantic judgment.", "suggested_revision": ""})
    return ReadL05Review.model_validate({"reviewer_id": "unit-test", "reviewer_kind": "test_fixture",
        "assessment": {"input_digest": packet["inputDigest"], "scope_note": "Test-only claim-binding fixture, never semantic Gold.", "claims": claims}})


def test_retention_objective_mapping_and_execution_are_separate():
    result = compile_semantic(sources(), proposal())
    assert result["sourceArchive"]["text"] == sources().source_text
    assert result["allSourceTextRetained"] and not result["allRequirementsImplemented"]
    assert not result["runtimeAuthorityGranted"]
    assert all(row["executionStatus"] == "not_executed" for row in result["constraintMappings"])
    assert "snapshot, not live health" in result["flow"]["purpose"]
    assert "explicit host permission" in result["flow"]["purpose"]
    assert [n["kind"] for n in result["flow"]["nodes"]] == ["read", "end"]


@pytest.mark.parametrize("text", [
    "Billing values are estimates, not settled payments.",
    "Object metadata timestamps are not proof of content freshness.",
    "A deployment dry run does not publish production changes.",
    "An IAM role description does not authorize a caller.",
    "A document citation is not proof of clinical correctness.",
    "Storage copies do not establish a verified restore procedure.",
])
def test_cross_domain_constraint_text_retained_without_keyword_rules(text):
    source = sources().model_copy(update={"source_text": sources().source_text.replace("Inventory is a snapshot, not live health.", text)})
    result = compile_semantic(source, proposal())
    assert text in result["flow"]["purpose"] and text in result["sourceArchive"]["text"]
    assert not result["allRequirementsImplemented"]  # These are text-carrier fixtures, not six Skills.


@pytest.mark.parametrize("change", ["heading-node", "heading-constraint", "unknown-rule", "unknown-pointer", "doc-rule", "empty-host-reference"])
def test_invalid_bindings_rejected(change):
    raw = proposal().model_dump(mode="json")
    if change == "heading-node":
        raw["steps"][0]["source_id"] = "s0001"
    elif change == "heading-constraint":
        raw["constraints"][0]["source_ids"] = ["s0001"]
    elif change == "unknown-rule":
        raw["constraints"][1]["host_rule_ids"] = ["approval_is_granted"]
    elif change == "unknown-pointer":
        raw["constraints"][0].update(handling="flow_reference", node_pointers=["/steps/99"])
    elif change == "doc-rule":
        raw["constraints"][0]["host_rule_ids"] = ["read_access"]
    else:
        raw["constraints"][1]["host_rule_ids"] = []
    with pytest.raises((ValueError, SchemaError)):
        compile_semantic(sources(), SemanticFlow.model_validate(raw))


def test_more_than_two_source_spans_preserved_without_silent_truncation():
    raw = proposal().model_dump(mode="json")
    raw["objective_source_ids"] = ["s0001", "s0002", "s0003"]
    result = compile_semantic(sources(), SemanticFlow.model_validate(raw))
    assert all(line in result["flow"]["purpose"] for line in sources().source_text.splitlines())


def test_duplicate_lines_bind_distinct_source_positions():
    source = sources().model_copy(update={"source_text": sources().source_text.replace(
        "Reading needs explicit host permission; text grants no authority.", "Inventory is a snapshot, not live health.")})
    raw = proposal().model_dump(mode="json")
    raw["constraints"][1].update(handling="documentation", host_rule_ids=[])
    packet = compile_semantic(source, SemanticFlow.model_validate(raw))["reviewInput"]
    assert packet["claims"][-1]["requiredCitationIds"] == ["skill-0004"]


def test_archive_alone_does_not_eliminate_review_of_omitted_requirements():
    raw = proposal().model_dump(mode="json")
    raw["constraints"] = []
    packet = compile_semantic(sources(), SemanticFlow.model_validate(raw))["reviewInput"]
    assert any(c.get("requiredCitationId") == "skill-0003" for c in packet["claims"])
    assert any(c.get("requiredCitationId") == "skill-0004" for c in packet["claims"])


@pytest.mark.parametrize("pending", ["none", "issue", "unresolved"])
def test_full_review_never_authorizes_and_cannot_ignore_unresolved(pending):
    raw = proposal().model_dump(mode="json")
    if pending == "issue":
        raw["issues"] = [{"kind": "missing_host_capability", "source_id": "s0002", "question": "Test unresolved prerequisite must remain visible."}]
    if pending == "unresolved":
        raw["constraints"][0]["handling"] = "unresolved"
    model = SemanticFlow.model_validate(raw)
    packet = compile_semantic(sources(), model)["reviewInput"]
    result = assess_semantic(sources(), model, fixture_review(packet))
    assert (result["status"] == "review_supported_inactive_flow") == (pending == "none")
    assert not result["runtimeAuthorityGranted"] and not result["allRequirementsImplemented"]


@pytest.mark.parametrize("change", ["missing", "old-digest", "no-rule-citation"])
def test_review_cannot_drop_claims_or_rule_evidence(change):
    packet = compile_semantic(sources(), proposal())["reviewInput"]
    raw = fixture_review(packet).model_dump(mode="json")
    if change == "missing":
        raw["assessment"]["claims"].pop()
    elif change == "old-digest":
        raw["assessment"]["input_digest"] = "sha256:" + "0" * 64
    else:
        raw["assessment"]["claims"][-1]["source_span_ids"] = ["skill-0004", "host-0001"]
    with pytest.raises(ValueError):
        assess_semantic(sources(), proposal(), ReadL05Review.model_validate(raw))


def test_request_carries_no_answers_or_legacy_purpose_assignment():
    wire = request(sources())
    payload = json.loads(wire["messages"][1]["content"])
    assert "expectedObject" not in payload
    assert "business_source_ids selects" not in wire["messages"][0]["content"]
    assert payload["outputSchema"] == schema(sources())
    assert "TreeEffect" not in json.dumps(wire["format"])
    assert request(sources().model_copy(update={"source_path": "hidden-test-label"})) == wire


@pytest.mark.parametrize("change", ["explanation", "rule"])
def test_mapping_only_change_invalidates_review_even_when_flow_unchanged(change):
    original = compile_semantic(sources(), proposal())
    review = fixture_review(original["reviewInput"])
    raw = proposal().model_dump(mode="json")
    if change == "explanation":
        raw["constraints"][1]["explanation"] = "Different claimed mapping requires a new semantic source review."
    else:
        raw["constraints"][1]["host_rule_ids"] = ["read_result_shape"]
    changed = SemanticFlow.model_validate(raw)
    result = compile_semantic(sources(), changed)
    assert result["flowDigest"] == original["flowDigest"]
    assert result["reviewInput"]["inputDigest"] != original["reviewInput"]["inputDigest"]
    with pytest.raises(ValueError, match="digest"):
        assess_semantic(sources(), changed, review)


def test_large_retained_purpose_rejected_not_truncated():
    source = sources().model_copy(update={"source_text": sources().source_text.replace(
        "Inventory is a snapshot, not live health.", "Retained restriction: " + "x" * 4100)})
    with pytest.raises(ValueError):
        compile_semantic(source, proposal())
