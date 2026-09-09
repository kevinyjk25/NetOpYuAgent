import json

import pytest
from jsonschema import ValidationError as SchemaError

from evaluation.flow_mapping import MappingProposal, assess_mapping, compile_mapping, execution_projection, request
from evaluation.flow_translation import local_sources
from evaluation.flow_tree import FlowTree
from evaluation.read_l05_review import ReadL05Review


def fixture():
    source = local_sources().model_copy(update={"source_text": "# Snapshot\nRead input device_id and finish.\nSnapshot data is not live health.\n"})
    tree = FlowTree.model_validate({"business_source_ids": ["s0001"], "steps": [
        {"kind": "read", "source_id": "s0001", "tool": "read_inventory_device", "bind": "result",
         "arguments": {"device_id": {"kind": "reference", "source": "input", "field": "device_id"}}},
        {"kind": "end", "source_id": "s0002", "outcome": "read_path_completed"}], "issues": []})
    mapping = {"objective_source_ids": ["s0001", "s0002"], "source_dispositions": {
        "s0002": [{"kind": "operation", "node_pointers": ["/steps/0", "/steps/1"], "explanation": "The source read and end operations match these existing nodes."}],
        "s0003": [{"kind": "documentation", "explanation": "Preserve snapshot interpretation without claiming live health verification."}]},
        "node_source_ids": {"/steps/0": "s0002", "/steps/1": "s0002"}}
    return source, tree, mapping


def test_fixed_catalog_and_metadata_only_schema():
    source, tree, _ = fixture()
    wire = request(source, tree)
    assert set(wire["format"]["properties"]) == {"objective_source_ids", "source_dispositions", "node_source_ids"}
    assert wire["format"]["$defs"]["OperationUse"]["properties"]["node_pointers"]["items"]["enum"] == ["/steps/0", "/steps/1"]
    assert set(json.loads(wire["messages"][1]["content"])["compilerNodeCatalog"]) == {"/steps/0", "/steps/1"}


def test_citation_repair_cannot_change_execution_or_parent():
    source, tree, raw = fixture()
    before = tree.model_dump(mode="json")
    result = compile_mapping(source, tree, MappingProposal.model_validate(raw))
    assert tree.model_dump(mode="json") == before
    assert result["executionProjectionUnchanged"] and not result["runtimeAuthorityGranted"]
    assert result["flow"]["nodes"][0]["arguments"]["device_id"]["source"] == "input"
    assert execution_projection(tree)["steps"][0]["kind"] == "read"


@pytest.mark.parametrize("change", ["operations", "tool-name", "missing-node", "wrong-source", "missing-paragraph"])
def test_model_cannot_supply_execution_edits_or_unbound_references(change):
    source, tree, raw = fixture()
    if change == "operations":
        raw["steps"] = []
    elif change == "tool-name":
        raw["source_dispositions"]["s0002"][0]["node_pointers"] = ["read_inventory_device"]
    elif change == "missing-node":
        raw["node_source_ids"].pop("/steps/0")
    elif change == "wrong-source":
        raw["node_source_ids"]["/steps/0"] = "s0001"
    else:
        raw["source_dispositions"].pop("s0003")
    with pytest.raises((ValueError, SchemaError)):
        compile_mapping(source, tree, MappingProposal.model_validate(raw))


def plumbing_review(packet):
    claims = []
    for claim in packet['claims']:
        ids = list(claim.get('requiredCitationIds', []))
        if claim.get('requiredCitationId'):
            ids.append(claim['requiredCitationId'])
        for kind, fallback in [('skill', 'skill-0002'), ('host', 'host-0001')]:
            if kind in claim['requiredEvidenceKinds'] and not any(s.startswith(kind + '-') for s in ids):
                ids.append(fallback)
        claims.append(dict(claim_id=claim['claimId'], verdict='supported', source_span_ids=ids,
                           rationale='Test-only review plumbing, not semantic evidence.', suggested_revision=''))
    return ReadL05Review.model_validate(dict(reviewer_id='test', reviewer_kind='test_fixture', assessment=dict(
        input_digest=packet['inputDigest'], scope_note='Test fixture checks binding only, not meaning.', claims=claims)))


@pytest.mark.parametrize('blocker', ['none', 'parent-issue', 'unresolved'])
def test_supported_metadata_never_authorizes_or_removes_real_gaps(blocker):
    source, parent, raw = fixture()
    if blocker == 'parent-issue':
        tree = parent.model_dump(mode='json')
        tree['issues'] = [dict(kind='source_ambiguity', source_id='s0002', question='An unresolved source fact remains in this test fixture.')]
        parent = FlowTree.model_validate(tree)
    elif blocker == 'unresolved':
        raw['source_dispositions']['s0003'][0]['kind'] = 'unresolved'
    proposal = MappingProposal.model_validate(raw)
    review = plumbing_review(compile_mapping(source, parent, proposal)['reviewInput'])
    result = assess_mapping(source, parent, proposal, review)
    assert (result['status'] == 'review_supported_inactive_flow') == (blocker == 'none')
    assert not result['runtimeAuthorityGranted'] and not result['semanticAlignmentProven']


@pytest.mark.parametrize('change', ['parent-alias', 'mapping-note'])
def test_old_review_cannot_be_reused_even_when_compiled_graph_is_unchanged(change):
    source, parent, raw = fixture()
    proposal = MappingProposal.model_validate(raw)
    original = compile_mapping(source, parent, proposal)
    review = plumbing_review(original['reviewInput'])
    if change == 'parent-alias':
        tree = parent.model_dump(mode='json')
        tree['steps'][0]['bind'] = 'another_alias'
        parent = FlowTree.model_validate(tree)
    else:
        raw['source_dispositions']['s0003'][0]['explanation'] = 'A different metadata note must invalidate the previous source review.'
        proposal = MappingProposal.model_validate(raw)
    revised = compile_mapping(source, parent, proposal)
    assert original['flow'] == revised['flow']
    with pytest.raises(ValueError, match='digest'):
        assess_mapping(source, parent, proposal, review)


def test_wrong_exact_source_citation_rejected_despite_supported_verdict():
    source, parent, raw = fixture()
    proposal = MappingProposal.model_validate(raw)
    packet = compile_mapping(source, parent, proposal)['reviewInput']
    review = plumbing_review(packet).model_dump(mode='json')
    target = next(c['claimId'] for c in packet['claims'] if c.get('requiredCitationIds') == ['skill-0003'])
    next(c for c in review['assessment']['claims'] if c['claim_id'] == target)['source_span_ids'] = ['skill-0002']
    with pytest.raises(ValueError, match='own source'):
        assess_mapping(source, parent, proposal, ReadL05Review.model_validate(review))
