import json

import pytest
from jsonschema import ValidationError as SchemaError

from evaluation.flow_clause_mapping import clauses
from evaluation.flow_lean_mapping import LeanMapping, assess_lean, compile_lean, expand, request
from evaluation.flow_translation import local_sources
from evaluation.flow_tree import FlowTree
from evaluation.read_l05_review import ReadL05Review


def fixture():
    source = local_sources().model_copy(update={'source_text': 'Read input device_id once, then finish.\nSnapshot is not live health.\nInvalid input or results must block.\n'})
    tree = FlowTree.model_validate(dict(business_source_ids=['s0001'], steps=[
        dict(kind='read', source_id='s0001', tool='read_inventory_device', bind='snapshot',
             arguments=dict(device_id=dict(kind='reference', source='input', field='device_id'))),
        dict(kind='end', source_id='s0001', outcome='read_path_completed')], issues=[]))
    raw = dict(objective=['clause-0001'], selections={
        'clause-0001': dict(targets=['operation:/steps/0', 'operation:/steps/1'], extra_sources=[]),
        'clause-0002': dict(targets=['documentation'], extra_sources=[]),
        'clause-0003': dict(targets=['rule:read_input_shape', 'rule:read_result_shape', 'rule:observation_error_blocks'], extra_sources=[])})
    return source, tree, raw


def review_fixture(packet):
    claims = []
    for c in packet['claims']:
        ids = list(c.get('requiredCitationIds', [])) + ([c['requiredCitationId']] if 'requiredCitationId' in c else [])
        if 'skill' in c['requiredEvidenceKinds'] and not any(x.startswith(('skill-', 'clause-')) for x in ids):
            ids.append('skill-0001')
        if 'host' in c['requiredEvidenceKinds'] and not any(x.startswith('host-') for x in ids):
            ids.append('host-0001')
        claims.append(dict(claim_id=c['claimId'], verdict='supported', source_span_ids=ids,
            rationale='Fixture for review binding only, not semantic evidence.', suggested_revision=''))
    return ReadL05Review.model_validate(dict(reviewer_id='test', reviewer_kind='test_fixture', assessment=dict(
        input_digest=packet['inputDigest'], scope_note='Synthetic plumbing fixture, not independent semantic review.', claims=claims)))


def test_compiler_owns_all_anchors_and_renders_no_new_business_prose():
    source, parent, raw = fixture()
    proposal = LeanMapping.model_validate(raw)
    wire = request(source, parent)
    assert set(wire['format']['properties']) == {'objective', 'selections'}
    assert 'explanation' not in json.dumps(wire['format'])
    projected = expand(source, parent, proposal)
    assert all(key in use.evidence_clause_ids for key, uses in projected.clause_dispositions.items() for use in uses)
    assert projected.node_clause_ids == {'/steps/0': ('clause-0001',), '/steps/1': ('clause-0001',)}
    host = projected.clause_dispositions['clause-0003'][0].disposition
    assert set(host.host_rule_ids) == {'read_input_shape', 'read_result_shape', 'observation_error_blocks'}
    result = compile_lean(source, parent, proposal)
    assert result['executionProjectionUnchanged'] and not result['runtimeAuthorityGranted']
    assert result['reviewInput']['anchorOwner'] == 'compiler'
    assert any(c['facet'].startswith('clause_fidelity') for c in result['reviewInput']['claims'])


@pytest.mark.parametrize('change', ['omit', 'own-extra', 'unknown-extra', 'unknown-target', 'empty', 'duplicate', 'prose', 'edit-flow', 'missing-node'])
def test_illegal_fields_references_and_missing_coverage_fail_closed(change):
    source, parent, raw = fixture()
    if change == 'omit':
        raw['selections'].pop('clause-0002')
    elif change == 'own-extra':
        raw['selections']['clause-0002']['extra_sources'] = ['clause-0002']
    elif change == 'unknown-extra':
        raw['selections']['clause-0002']['extra_sources'] = ['clause-9999']
    elif change == 'unknown-target':
        raw['selections']['clause-0001']['targets'] = ['operation:read_inventory_device']
    elif change == 'empty':
        raw['selections']['clause-0003']['targets'] = []
    elif change == 'duplicate':
        raw['selections']['clause-0002']['targets'] = ['documentation', 'documentation']
    elif change == 'prose':
        raw['selections']['clause-0002']['explanation'] = 'Enforce all permissions.'
    elif change == 'edit-flow':
        raw['steps'] = []
    else:
        raw['selections']['clause-0001']['targets'] = ['operation:/steps/0']
    with pytest.raises((ValueError, SchemaError)):
        compile_lean(source, parent, LeanMapping.model_validate(raw))


def test_single_clause_has_valid_empty_extra_source_schema():
    source, parent, raw = fixture()
    source = source.model_copy(update={'source_text': 'Read input device_id once, then finish.'})
    raw['selections'] = {'clause-0001': raw['selections']['clause-0001']}
    assert len(clauses(source)) == 1
    assert compile_lean(source, parent, LeanMapping.model_validate(raw))['clauseAccountingComplete']


@pytest.mark.parametrize('kind', ['unresolved', 'parent-issue', 'supported'])
def test_no_review_grants_authority_and_real_issues_remain(kind):
    source, parent, raw = fixture()
    if kind == 'unresolved':
        raw['selections']['clause-0003']['targets'] = ['unresolved']
    elif kind == 'parent-issue':
        tree = parent.model_dump(mode='json')
        tree['issues'] = [dict(kind='source_ambiguity', source_id='s0001', question='A real unresolved source question remains in this test.')]
        parent = FlowTree.model_validate(tree)
    p = LeanMapping.model_validate(raw)
    packet = compile_lean(source, parent, p)['reviewInput']
    r = assess_lean(source, parent, p, review_fixture(packet))
    assert (r['status'] == 'review_supported_inactive_flow') == (kind == 'supported')
    assert not r['runtimeAuthorityGranted'] and not r['semanticAlignmentProven']


@pytest.mark.parametrize('wrong_target', ['documentation', 'flow:/steps/1', 'rule:read_access'])
def test_removing_free_prose_does_not_remove_wrong_check_semantics_review(wrong_target):
    source, parent, raw = fixture()
    raw['selections']['clause-0003']['targets'] = [wrong_target]
    p = LeanMapping.model_validate(raw)
    packet = compile_lean(source, parent, p)['reviewInput']
    claim = next(c for c in packet['claims'] if c['l05Pointer'] == '/clause_dispositions/clause-0003/0')
    assert claim['declaredValue']['disposition']['kind'] in ['documentation', 'flow_reference', 'host_rule_reference']
    rv = review_fixture(packet).model_dump(mode='json')
    row = next(c for c in rv['assessment']['claims'] if c['claim_id'] == claim['claimId'])
    row.update(verdict='insufficient_evidence', rationale='The selected target does not implement required input and result validation.',
               suggested_revision='Select actual input/result checks or mark the requirement unresolved.')
    assert assess_lean(source, parent, p, ReadL05Review.model_validate(rv))['status'] == 'blocked'


def test_adding_extra_evidence_changes_digest_without_changing_execution():
    source, parent, raw = fixture()
    first = LeanMapping.model_validate(raw)
    original = compile_lean(source, parent, first)
    review = review_fixture(original['reviewInput'])
    raw['selections']['clause-0002']['extra_sources'] = ['clause-0003']
    second = LeanMapping.model_validate(raw)
    assert compile_lean(source, parent, second)['flow'] == original['flow']
    with pytest.raises(ValueError, match='digest'):
        assess_lean(source, parent, second, review)
