import json

import pytest
from jsonschema import ValidationError as SchemaError

from evaluation.flow_source_ledger import LedgerFlow, assess_ledger, compile_ledger, request
from evaluation.flow_translation import local_sources
from evaluation.read_l05_review import ReadL05Review


def source():
    return local_sources().model_copy(update={"source_text":
        "# Read snapshot\nRead input device_id once and finish.\n"
        "Snapshot data is not live health.\nExplicit host read authorization is required.\n"})


def proposal():
    return LedgerFlow.model_validate({"objective_source_ids": ["s0001", "s0002"], "source_dispositions": {
        "s0002": [{"kind": "operation", "node_pointers": ["/steps/0", "/steps/1"], "explanation": "Read the original input once and then end the read path."}],
        "s0003": [{"kind": "documentation", "explanation": "Preserve snapshot interpretation without claiming live telemetry."}],
        "s0004": [{"kind": "host_rule_reference", "host_rule_ids": ["read_access"], "explanation": "Reference actual host authorization; it has not been executed in this proposal."}]},
        "steps": [{"kind": "read", "source_id": "s0002", "tool": "read_inventory_device", "bind": "snapshot",
                   "arguments": {"device_id": {"kind": "reference", "source": "input", "field": "device_id"}}},
                  {"kind": "end", "source_id": "s0002", "outcome": "read_path_completed"}], "issues": []})


def test_total_ledger_preserves_text_but_claims_no_truth_or_execution():
    result = compile_ledger(source(), proposal())
    assert result["sourceDispositionComplete"] and not result["sourceDispositionTruthProven"]
    assert result["sourceArchive"]["text"] == source().source_text
    assert not result["runtimeAuthorityGranted"] and not result["allRequirementsImplemented"]
    claims = result["reviewInput"]["claims"]
    assert len([c for c in claims if c["facet"] == "source_disposition_is_faithful_not_merely_present"]) == 3


@pytest.mark.parametrize("mutation", ["missing", "extra", "empty", "doc-with-rules", "unknown-node", "unknown-rule"])
def test_omission_and_contradictory_mapping_shapes_rejected(mutation):
    raw = proposal().model_dump(mode="json")
    if mutation == "missing":
        raw["source_dispositions"].pop("s0003")
    elif mutation == "extra":
        raw["source_dispositions"]["s9999"] = raw["source_dispositions"]["s0003"]
    elif mutation == "empty":
        raw["source_dispositions"]["s0003"] = []
    elif mutation == "doc-with-rules":
        raw["source_dispositions"]["s0003"][0]["host_rule_ids"] = ["read_access"]
    elif mutation == "unknown-node":
        raw["source_dispositions"]["s0002"][0]["node_pointers"] = ["/steps/99"]
    else:
        raw["source_dispositions"]["s0004"][0]["host_rule_ids"] = ["guarantee_live_health"]
    with pytest.raises((ValueError, SchemaError)):
        compile_ledger(source(), LedgerFlow.model_validate(raw))


def test_full_source_requirement_review_remains_even_for_wrong_classification():
    raw = proposal().model_dump(mode="json")
    raw["source_dispositions"]["s0003"] = [{"kind": "operation", "node_pointers": ["/steps/0"],
        "explanation": "Deliberately incorrect test classification requiring source review."}]
    result = compile_ledger(source(), LedgerFlow.model_validate(raw))
    claims = result["reviewInput"]["claims"]
    assert any(c.get("requiredCitationId") == "skill-0003" for c in claims)
    assert any(c["facet"] == "source_disposition_is_faithful_not_merely_present" and
               c["requiredCitationIds"] == ["skill-0003"] for c in claims)
    assert not result["sourceDispositionTruthProven"]


def test_request_requires_every_body_key_without_answer_nodes():
    wire = request(source())
    schema = wire["format"]["properties"]["source_dispositions"]
    assert schema["required"] == ["s0002", "s0003", "s0004"]
    assert not schema["additionalProperties"]
    payload = json.loads(wire["messages"][1]["content"])
    assert "expectedObject" not in payload
    assert "constraints" not in wire["format"]["properties"]


@pytest.mark.parametrize("unresolved", [False, True])
def test_review_cannot_promote_unresolved_or_authorize(unresolved):
    raw = proposal().model_dump(mode="json")
    if unresolved:
        raw["source_dispositions"]["s0003"][0]["kind"] = "unresolved"
    model = LedgerFlow.model_validate(raw)
    packet = compile_ledger(source(), model)["reviewInput"]
    claims = []
    for c in packet["claims"]:
        ids = list(c.get("requiredCitationIds", []))
        if c.get("requiredCitationId"):
            ids.append(c["requiredCitationId"])
        if "skill" in c["requiredEvidenceKinds"] and not any(x.startswith("skill-") for x in ids):
            ids.append("skill-0002")
        if "host" in c["requiredEvidenceKinds"] and not any(x.startswith("host-") for x in ids):
            ids.append("host-0001")
        claims.append({"claim_id": c["claimId"], "verdict": "supported", "source_span_ids": ids,
            "rationale": "Test-only plumbing fixture, not semantic evidence.", "suggested_revision": ""})
    review = ReadL05Review.model_validate({"reviewer_id": "test", "reviewer_kind": "test_fixture", "assessment": {
        "input_digest": packet["inputDigest"], "claims": claims, "scope_note": "Fixture exercises review binding, not actual entailment."}})
    result = assess_ledger(source(), model, review)
    assert (result["status"] == "review_supported_inactive_flow") == (not unresolved)
    assert not result["runtimeAuthorityGranted"]
