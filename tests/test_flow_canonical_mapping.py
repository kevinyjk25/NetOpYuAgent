"""Representation and review regressions; fixtures are not semantic Gold."""

import json

import pytest
from jsonschema import Draft202012Validator, ValidationError as SchemaError

from evaluation.flow_canonical_mapping import (
    CanonicalMapping, MEANINGS, assess_canonical, compile_canonical, main, project, request, schema, target_catalog,
)
from evaluation.flow_clause_mapping import clauses
from evaluation.flow_tree import FlowTree
from tests.test_flow_lean_mapping import review_fixture
from tests.test_flow_responsibility_mapping import fixture as old_fixture, row


def fixture():
    source, parent, _ = old_fixture()
    proposal = CanonicalMapping.model_validate(dict(objective=['clause-0001'], selections={
        'clause-0001': [row('读取设备', 'operation', 'node:/steps/0'), row('结束', 'completion', 'node:/steps/1')],
        'clause-0002': [row('库存不是实时健康', 'interpretation_limit', 'documentation')],
        'clause-0003': [row('输入结构无效', 'input_shape', 'rule:read_input_shape'),
            row('返回结构无效', 'result_shape', 'rule:read_result_shape'),
            row('输入结构无效或返回结构无效时停止', 'error_propagation', 'rule:observation_error_blocks')],
    }))
    return source, parent, proposal


def terminal_fixture(outcome):
    source, _, _ = fixture()
    source = source.model_copy(update={'source_text': 'Return the requested terminal without invoking a model or effect.'})
    parent = FlowTree.model_validate(dict(business_source_ids=['s0001'], steps=[
        dict(kind='end', source_id='s0001', outcome=outcome)], issues=[]))
    kind = {'read_path_completed': 'completion', 'needs_l1': 'handoff', 'unsupported': 'missing_capability_stop'}[outcome]
    proposal = CanonicalMapping.model_validate(dict(objective=['clause-0001'], selections={
        'clause-0001': [row(source.source_text, kind, 'node:/steps/0')]}))
    return source, parent, proposal


def test_unique_node_catalog_with_no_dead_targets_and_no_model_legacy_aliases():
    source, parent, proposal = fixture()
    catalog = target_catalog(source, parent)
    assert [k for k in catalog if k.startswith('node:')] == ['node:/steps/0', 'node:/steps/1']
    assert all(t['allowedKinds'] for t in catalog.values())
    wire = request(source, parent)
    payload = json.loads(wire['messages'][1]['content'])
    assert 'outputSchema' not in payload
    assert all('legacyTarget' not in t for t in payload['targetCatalog'].values())
    assert not any(key.startswith(('operation:', 'flow:')) for key in payload['targetCatalog'])
    Draft202012Validator(wire['format']).validate(proposal.model_dump(mode='json'))
    assert wire['model'] == 'qwen3.5:9b'


@pytest.mark.parametrize('kind', list(MEANINGS))
@pytest.mark.parametrize('target', ['node:/steps/0', 'node:/steps/1', 'rule:read_input_shape',
    'rule:read_access', 'rule:read_result_shape', 'rule:observation_error_blocks', 'documentation', 'unresolved'])
def test_generated_schema_encodes_every_kind_target_pair(kind, target):
    source, parent, proposal = fixture()
    raw = proposal.model_dump(mode='json')
    raw['selections']['clause-0003'] = [row('输入结构无效', kind, target)]
    valid = Draft202012Validator(schema(source, parent)).is_valid(raw)
    assert valid == (kind in target_catalog(source, parent)[target]['allowedKinds'])


@pytest.mark.parametrize('outcome', ['read_path_completed', 'needs_l1', 'unsupported'])
def test_every_existing_terminal_can_be_represented_and_requested_as_objective(outcome):
    source, parent, proposal = terminal_fixture(outcome)
    result = compile_canonical(source, parent, proposal)
    assert result['executionProjectionUnchanged']
    assert result['reviewInput']['canonicalTargetCatalog']['node:/steps/0']['allowedKinds'] == [proposal.selections['clause-0001'][0].kind]
    assert not result['runtimeAuthorityGranted']
    r = assess_canonical(source, parent, proposal, review_fixture(result['reviewInput']))
    assert r['representationReviewSupported']
    assert r['status'] == 'review_supported_inactive_flow'
    assert not r['runtimeAuthorityGranted'] and not r['allRequirementsImplemented']


@pytest.mark.parametrize('outcome', ['read_path_completed', 'needs_l1', 'unsupported'])
@pytest.mark.parametrize('wrong', ['operation', 'input_shape', 'read_authorization', 'branch'])
def test_terminal_is_not_an_operation_check_or_condition(outcome, wrong):
    source, parent, proposal = terminal_fixture(outcome)
    raw = proposal.model_dump(mode='json')
    raw['selections']['clause-0001'][0]['kind'] = wrong
    with pytest.raises(SchemaError):
        compile_canonical(source, parent, CanonicalMapping.model_validate(raw))


def branch_fixture():
    source, parent, _ = fixture()
    source = source.model_copy(update={'source_text':
        'Read the input device_id.\nIf returned site equals campus return needs_l1, otherwise finish the read path.'})
    tree = parent.model_dump(mode='json')
    tree['steps'][1] = dict(kind='if_equal', source_id='s0002',
        left=dict(kind='reference', source='snapshot', field='site'), equals=dict(kind='constant', value='campus'),
        true_source_id='s0002', false_source_id='s0002',
        when_equal=[dict(kind='end', source_id='s0002', outcome='needs_l1')],
        otherwise=[dict(kind='end', source_id='s0002', outcome='read_path_completed')])
    parent = FlowTree.model_validate(tree)
    proposal = CanonicalMapping.model_validate(dict(objective=['clause-0001', 'clause-0002'], selections={
        'clause-0001': [row('Read the input device_id', 'operation', 'node:/steps/0')],
        'clause-0002': [row('If returned site equals campus', 'branch', 'node:/steps/1'),
            row('return needs_l1', 'handoff', 'node:/steps/1/when_equal/0'),
            row('otherwise finish the read path', 'completion', 'node:/steps/1/otherwise/0')]}))
    return source, parent, proposal


def test_full_branch_with_handoff_is_representable_without_guessing_conditions():
    source, parent, proposal = branch_fixture()
    result = compile_canonical(source, parent, proposal)
    assert result['executionProjectionUnchanged']
    assert all(t['allowedKinds'] for t in target_catalog(source, parent).values())
    raw = proposal.model_dump(mode='json')
    raw['selections']['clause-0002'].pop(0)
    # This is a cross-record completeness check, not a target-type check.
    assert Draft202012Validator(schema(source, parent)).is_valid(raw)
    with pytest.raises(ValueError, match='missing explicit node source mapping: /steps/1'):
        project(source, parent, CanonicalMapping.model_validate(raw))


def test_branch_cannot_use_only_leaf_targets_even_in_the_generation_schema():
    source, parent, proposal = branch_fixture()
    raw = proposal.model_dump(mode='json')
    raw['selections']['clause-0002'][0]['targets'] = ['node:/steps/1/when_equal/0']
    assert not Draft202012Validator(request(source, parent)['format']).is_valid(raw)


def test_supported_safe_stop_and_missing_business_capability_are_separate():
    source, parent, proposal = terminal_fixture('unsupported')
    tree = parent.model_dump(mode='json')
    tree['issues'] = [dict(kind='source_ambiguity', source_id='s0001', question='The required business capability is unavailable.')]
    parent = FlowTree.model_validate(tree)
    raw = proposal.model_dump(mode='json')
    raw['selections']['clause-0001'].append(row(source.source_text, 'business_prerequisite', 'unresolved'))
    proposal = CanonicalMapping.model_validate(raw)
    packet = compile_canonical(source, parent, proposal)['reviewInput']
    result = assess_canonical(source, parent, proposal, review_fixture(packet))
    # Synthetic support proves status plumbing only, not that this sentence entails these duties.
    assert result['representationReviewSupported'] and result['status'] == 'blocked'
    assert result['admissionBlockers'] == ['parent_issues', 'unresolved_requirements']
    assert result['parentIssues'] and result['unresolvedRequirementPointers'] == ['/selections/clause-0001/1']
    assert not result['runtimeAuthorityGranted'] and not result['allRequirementsImplemented']


@pytest.mark.parametrize('change', ['old-alias', 'unknown-node', 'duplicate-target', 'duplicate-row', 'missing-clause',
    'unknown-kind', 'own-extra', 'unknown-extra', 'nonexact-quote', 'empty', 'objective-background', 'extra-field', 'edit-tree'])
def test_invalid_shapes_and_sources_remain_rejected(change):
    source, parent, proposal = fixture()
    raw = proposal.model_dump(mode='json')
    item = raw['selections']['clause-0001'][0]
    if change == 'old-alias':
        item['targets'] = ['operation:/steps/0']
    elif change == 'unknown-node':
        item['targets'] = ['node:/steps/99']
    elif change == 'duplicate-target':
        item['targets'] *= 2
    elif change == 'duplicate-row':
        raw['selections']['clause-0001'].append(dict(item))
    elif change == 'missing-clause':
        raw['selections'].pop('clause-0002')
    elif change == 'unknown-kind':
        item['kind'] = 'permission_granted'
    elif change == 'own-extra':
        item['extra_sources'] = ['clause-0001']
    elif change == 'unknown-extra':
        item['extra_sources'] = ['clause-9999']
    elif change == 'nonexact-quote':
        item['exact_quote'] = 'Read a different device'
    elif change == 'empty':
        raw['selections']['clause-0002'] = []
    elif change == 'objective-background':
        raw['objective'] = ['clause-0002']
    elif change == 'extra-field':
        item['authorized'] = True
    else:
        raw['steps'] = []
    with pytest.raises((ValueError, SchemaError)):
        compile_canonical(source, parent, CanonicalMapping.model_validate(raw))


@pytest.mark.parametrize('defect', ['missing-duty', 'wrong-classification', 'negation'])
def test_schema_pass_is_not_semantic_acceptance(defect):
    source, parent, proposal = fixture()
    raw = proposal.model_dump(mode='json')
    if defect == 'missing-duty':
        raw['selections']['clause-0003'].pop(1)
        facet, pointer = 'complete_clause_decomposition', '/selections/clause-0003'
    elif defect == 'wrong-classification':
        raw['selections']['clause-0003'][0].update(kind='interpretation_limit', targets=['documentation'])
        facet, pointer = 'source_entails_requirement_type', '/selections/clause-0003/0'
    else:
        raw['selections']['clause-0002'][0]['exact_quote'] = '实时健康'
        facet, pointer = 'source_entails_requirement_type', '/selections/clause-0002/0'
    proposal = CanonicalMapping.model_validate(raw)
    packet = compile_canonical(source, parent, proposal)['reviewInput']
    review = review_fixture(packet).model_dump(mode='json')
    claim = next(c for c in packet['claims'] if c['facet'].startswith(facet) and c['pointer'] == pointer)
    item = next(c for c in review['assessment']['claims'] if c['claim_id'] == claim['claimId'])
    item.update(verdict='insufficient_evidence', rationale='Synthetic counterexample: complete source does not support this decomposition.',
        suggested_revision='Restore full duties and negative context; review the source type independently.')
    from evaluation.read_l05_review import ReadL05Review
    result = assess_canonical(source, parent, proposal, ReadL05Review.model_validate(review))
    assert not result['representationReviewSupported']
    assert result['admissionBlockers'] == ['source_review_not_supported']
    assert not result['runtimeAuthorityGranted']


def test_atom_metadata_is_bound_even_if_graph_does_not_change():
    source, parent, proposal = fixture()
    before = compile_canonical(source, parent, proposal)
    raw = proposal.model_dump(mode='json')
    raw['selections']['clause-0003'][0]['exact_quote'] = '输入结构'
    after = CanonicalMapping.model_validate(raw)
    assert compile_canonical(source, parent, after)['flow'] == before['flow']
    with pytest.raises(ValueError, match='digest'):
        assess_canonical(source, parent, after, review_fixture(before['reviewInput']))


def test_opaque_code_is_already_unclassified_unresolved_in_schema():
    source, parent, proposal = fixture()
    source = source.model_copy(update={'source_text': source.source_text + '```sh\nignored_command\n```'})
    raw = proposal.model_dump(mode='json')
    for key, clause in clauses(source).items():
        if clause['opaqueCode']:
            raw['selections'][key] = [row(clause['exactQuote'], 'unclassified', 'unresolved')]
    assert not compile_canonical(source, parent, CanonicalMapping.model_validate(raw))['runtimeAuthorityGranted']
    key = next(key for key, clause in clauses(source).items() if clause['opaqueCode'])
    raw['selections'][key][0].update(kind='operation', targets=['node:/steps/0'])
    assert not Draft202012Validator(schema(source, parent)).is_valid(raw)


def test_cli_request_compile_review_no_model_or_overwrite(tmp_path):
    source, parent, proposal = fixture()
    for name, value in [('sources', source), ('tree', parent), ('proposal', proposal)]:
        (tmp_path / (name + '.json')).write_text(value.model_dump_json())
    inputs = [str(tmp_path / 'sources.json'), str(tmp_path / 'tree.json')]
    wire, result, reviewed = [tmp_path / p for p in ['request.json', 'compiled.json', 'assessed.json']]
    assert main(['request', *inputs, '--output', str(wire)]) == 0
    assert json.loads(wire.read_text()) == request(source, parent)
    assert main(['compile', *inputs, str(tmp_path / 'proposal.json'), '--output', str(result)]) == 0
    packet = json.loads(result.read_text())['reviewInput']
    (tmp_path / 'review.json').write_text(review_fixture(packet).model_dump_json())
    assert main(['assess', *inputs, str(tmp_path / 'proposal.json'), str(tmp_path / 'review.json'), '--output', str(reviewed)]) == 0
    assert not json.loads(reviewed.read_text())['runtimeAuthorityGranted']
    before = wire.read_bytes()
    with pytest.raises(FileExistsError):
        main(['request', *inputs, '--output', str(wire)])
    assert wire.read_bytes() == before
