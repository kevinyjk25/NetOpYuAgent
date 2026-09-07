"""Development counterexamples, not independent semantic Gold or model accuracy."""

import copy
import json

import pytest
from jsonschema import Draft202012Validator, ValidationError as SchemaError

from evaluation.flow_clause_mapping import clauses
from evaluation.flow_responsibility_mapping import (
    ResponsibilityMapping, assess_responsibilities, compatible, compile_responsibilities,
    main, project, request,
)
from evaluation.flow_lean_mapping import target_catalog
from evaluation.flow_translation import local_sources
from evaluation.flow_tree import FlowTree
from evaluation.read_l05_review import ReadL05Review
from tests.test_flow_lean_mapping import review_fixture


def row(quote, kind, *targets):
    return dict(exact_quote=quote, kind=kind, targets=list(targets), extra_sources=[])


def fixture():
    source = local_sources().model_copy(update={'source_text':
        '读取设备并结束。\n库存不是实时健康。\n输入结构无效或返回结构无效时停止。\n'})
    parent = FlowTree.model_validate(dict(business_source_ids=['s0001'], steps=[
        dict(kind='read', source_id='s0001', tool='read_inventory_device', bind='snapshot',
             arguments=dict(device_id=dict(kind='reference', source='input', field='device_id'))),
        dict(kind='end', source_id='s0001', outcome='read_path_completed')], issues=[]))
    proposal = ResponsibilityMapping.model_validate(dict(objective=['clause-0001'], selections={
        'clause-0001': [row('读取设备并结束', 'operation', 'operation:/steps/0', 'operation:/steps/1')],
        'clause-0002': [row('库存不是实时健康', 'interpretation_limit', 'documentation')],
        'clause-0003': [row('输入结构无效', 'input_shape', 'rule:read_input_shape'),
                        row('返回结构无效', 'result_shape', 'rule:read_result_shape'),
                        row('输入结构无效或返回结构无效时停止', 'error_propagation', 'rule:observation_error_blocks')],
    }))
    return source, parent, proposal


def test_exact_atoms_and_complete_review_preserve_unchanged_execution():
    source, parent, proposal = fixture()
    result = compile_responsibilities(source, parent, proposal)
    assert result['executionProjectionUnchanged'] and not result['runtimeAuthorityGranted']
    packet = result['reviewInput']
    assert packet['compatibilityOnly'] and not packet['semanticDecompositionProven']
    assert len(result['requirementAtoms']) == 5
    for atom in result['requirementAtoms']:
        assert source.source_text[atom['start']:atom['end']] == atom['exact_quote']
    assert sum(c['facet'].startswith('complete_clause_decomposition') for c in packet['claims']) == 3
    assert sum(c['facet'].startswith('source_entails_requirement_type') for c in packet['claims']) == 5
    assert any(c['facet'] == 'clause_fidelity_no_added_requirement_or_retention_as_enforcement' for c in packet['claims'])
    r = assess_responsibilities(source, parent, proposal, review_fixture(packet))
    assert r['status'] == 'review_supported_inactive_flow'
    assert not r['runtimeAuthorityGranted'] and not r['semanticAlignmentProven']


@pytest.mark.parametrize('kind,allowed', [
    ('input_shape', 'read_input_shape'), ('result_shape', 'read_result_shape'),
    ('read_authorization', 'read_access'), ('error_propagation', 'observation_error_blocks'),
])
@pytest.mark.parametrize('rule', ['read_input_shape', 'read_result_shape', 'read_access', 'observation_error_blocks'])
def test_checks_are_not_interchangeable(kind, allowed, rule):
    source, parent, _ = fixture()
    key = 'rule:' + rule
    assert compatible(kind, key, target_catalog(source, parent)[key]) == (allowed == rule)


@pytest.mark.parametrize('kind', ['business_prerequisite', 'interpretation_limit', 'authority_limit', 'missing_capability_stop'])
@pytest.mark.parametrize('rule', ['read_input_shape', 'read_result_shape', 'read_access', 'observation_error_blocks'])
def test_business_facts_and_dependencies_cannot_be_established_by_read_guards(kind, rule):
    source, parent, _ = fixture()
    key = 'rule:' + rule
    assert not compatible(kind, key, target_catalog(source, parent)[key])


@pytest.mark.parametrize('kind', ['input_shape', 'result_shape', 'read_authorization', 'error_propagation', 'business_prerequisite', 'operation', 'branch'])
def test_mandatory_declared_duties_cannot_use_documentation(kind):
    source, parent, _ = fixture()
    assert not compatible(kind, 'documentation', target_catalog(source, parent)['documentation'])


@pytest.mark.parametrize('mutation', ['wrong-rule', 'unknown-type', 'unknown-target', 'missing-clause',
    'empty-atoms', 'duplicate-atom', 'duplicate-target', 'unknown-extra', 'own-extra', 'nonexact', 'blank',
    'omit-node', 'limit-objective', 'free-prose', 'modify-flow', 'too-many-atoms'])
def test_malformed_and_duty_mismatched_proposals_reject_without_repair(mutation):
    source, parent, proposal = fixture()
    raw = proposal.model_dump(mode='json')
    selected = raw['selections']
    atom = selected['clause-0003'][0]
    if mutation == 'wrong-rule':
        atom['targets'] = ['rule:read_access']
    elif mutation == 'unknown-type':
        atom['kind'] = 'approved'
    elif mutation == 'unknown-target':
        atom['targets'] = ['rule:approve_everything']
    elif mutation == 'missing-clause':
        selected.pop('clause-0002')
    elif mutation == 'empty-atoms':
        selected['clause-0002'] = []
    elif mutation == 'duplicate-atom':
        selected['clause-0003'].append(copy.deepcopy(atom))
    elif mutation == 'duplicate-target':
        atom['targets'] *= 2
    elif mutation == 'unknown-extra':
        atom['extra_sources'] = ['clause-9999']
    elif mutation == 'own-extra':
        atom['extra_sources'] = ['clause-0003']
    elif mutation == 'nonexact':
        atom['exact_quote'] = 'Input is invalid'
    elif mutation == 'blank':
        atom['exact_quote'] = ' '
    elif mutation == 'omit-node':
        selected['clause-0001'][0]['targets'] = ['operation:/steps/0']
    elif mutation == 'limit-objective':
        raw['objective'] = ['clause-0002']
    elif mutation == 'free-prose':
        atom['explanation'] = 'All requirements executed.'
    elif mutation == 'modify-flow':
        raw['steps'] = []
    else:
        selected['clause-0003'] *= 3
    with pytest.raises((ValueError, SchemaError)):
        compile_responsibilities(source, parent, ResponsibilityMapping.model_validate(raw))


@pytest.mark.parametrize('text,quote', [('Read read read.', 'read'), ('Read aaa once.', 'aa')])
def test_repeated_including_overlapping_quotes_do_not_guess_offsets(text, quote):
    source, parent, proposal = fixture()
    source = source.model_copy(update={'source_text': text})
    raw = dict(objective=['clause-0001'], selections={'clause-0001': [row(quote, 'operation', 'operation:/steps/0', 'operation:/steps/1')]})
    with pytest.raises(ValueError, match='unique nonblank exact substring'):
        project(source, parent, ResponsibilityMapping.model_validate(raw))


def branch_fixture():
    source, parent, _ = fixture()
    source = source.model_copy(update={'source_text': 'Read the device.\nIf it is down stop as unsupported, otherwise finish.\n'})
    tree = parent.model_dump(mode='json')
    tree['steps'][1] = dict(kind='if_equal', source_id='s0002',
        left=dict(kind='reference', source='snapshot', field='status'), equals=dict(kind='constant', value='down'),
        true_source_id='s0002', false_source_id='s0002',
        when_equal=[dict(kind='end', source_id='s0002', outcome='unsupported')],
        otherwise=[dict(kind='end', source_id='s0002', outcome='read_path_completed')])
    parent = FlowTree.model_validate(tree)
    raw = dict(objective=['clause-0001'], selections={
        'clause-0001': [row('Read the device', 'operation', 'operation:/steps/0')],
        'clause-0002': [row('If it is down', 'branch', 'flow:/steps/1'),
            row('stop as unsupported', 'missing_capability_stop', 'flow:/steps/1/when_equal/0'),
            row('otherwise finish', 'operation', 'operation:/steps/1/otherwise/0')]})
    return source, parent, raw


def test_condition_requires_its_own_node_mapping_not_only_leaves():
    source, parent, raw = branch_fixture()
    lean, _ = project(source, parent, ResponsibilityMapping.model_validate(raw))
    assert 'flow:/steps/1' in lean.selections['clause-0002'].targets
    raw['selections']['clause-0002'].pop(0)
    with pytest.raises(ValueError, match='missing explicit node source mapping: /steps/1'):
        project(source, parent, ResponsibilityMapping.model_validate(raw))


@pytest.mark.parametrize('kind,target,expected', [
    ('branch', 'flow:/steps/1', True), ('branch', 'flow:/steps/1/otherwise/0', False),
    ('business_prerequisite', 'flow:/steps/1', True),
    ('missing_capability_stop', 'flow:/steps/1/when_equal/0', True),
    ('input_shape', 'flow:/steps/1/when_equal/0', False),
    ('operation', 'operation:/steps/1/when_equal/0', False),
    ('missing_capability_stop', 'flow:/steps/1/otherwise/0', False),
])
def test_flow_node_kind_is_necessary_but_not_predicate_entailment(kind, target, expected):
    source, parent, _ = branch_fixture()
    assert compatible(kind, target, target_catalog(source, parent)[target]) == expected


@pytest.mark.parametrize('defect', ['omitted-duty', 'wrong-type', 'cherry-pick'])
def test_compatible_but_semantically_wrong_mapping_retains_review_and_blocks(defect):
    source, parent, proposal = fixture()
    raw = proposal.model_dump(mode='json')
    if defect == 'omitted-duty':
        raw['selections']['clause-0003'].pop(1)
        facet = 'complete_clause_decomposition'
    elif defect == 'wrong-type':
        raw['selections']['clause-0003'][0].update(kind='interpretation_limit', targets=['documentation'])
        facet = 'source_entails_requirement_type'
    else:
        raw['selections']['clause-0002'][0]['exact_quote'] = '实时健康'
        facet = 'source_entails_requirement_type'
    proposal = ResponsibilityMapping.model_validate(raw)
    packet = compile_responsibilities(source, parent, proposal)['reviewInput']
    review = review_fixture(packet).model_dump(mode='json')
    claim = next(c for c in packet['claims'] if c['facet'].startswith(facet) and c['pointer'].startswith('/selections/clause-000' + ('2' if defect == 'cherry-pick' else '3')))
    judgment = next(c for c in review['assessment']['claims'] if c['claim_id'] == claim['claimId'])
    judgment.update(verdict='insufficient_evidence', rationale='Development counterexample: missing duty or wrong classification/negation.',
                    suggested_revision='Review the complete owning clause; retain missing duties and negative context.')
    assert assess_responsibilities(source, parent, proposal, ReadL05Review.model_validate(review))['status'] == 'blocked'


def test_atom_edits_invalidate_old_review_even_with_identical_flow_and_lean_projection():
    source, parent, first = fixture()
    a = compile_responsibilities(source, parent, first)
    raw = first.model_dump(mode='json')
    raw['selections']['clause-0003'][0]['exact_quote'] = '输入结构'
    second = ResponsibilityMapping.model_validate(raw)
    b = compile_responsibilities(source, parent, second)
    assert a['flow'] == b['flow'] and a['leanSelections'] == b['leanSelections']
    with pytest.raises(ValueError, match='digest'):
        assess_responsibilities(source, parent, second, review_fixture(a['reviewInput']))


def test_exact_atom_and_owning_context_are_mandatory_review_citations():
    source, parent, proposal = fixture()
    packet = compile_responsibilities(source, parent, proposal)['reviewInput']
    raw = review_fixture(packet).model_dump(mode='json')
    claim = next(c for c in packet['claims'] if c['facet'].startswith('source_entails_requirement_type'))
    judgment = next(c for c in raw['assessment']['claims'] if c['claim_id'] == claim['claimId'])
    judgment['source_span_ids'].remove('clause-0001')
    with pytest.raises(ValueError, match='exact atom'):
        assess_responsibilities(source, parent, proposal, ReadL05Review.model_validate(raw))


@pytest.mark.parametrize('reason', ['unresolved', 'parent-issue'])
def test_supported_review_cannot_erase_unresolved_or_parent_issues(reason):
    source, parent, proposal = fixture()
    if reason == 'unresolved':
        raw = proposal.model_dump(mode='json')
        raw['selections']['clause-0003'][0]['targets'].append('unresolved')
        proposal = ResponsibilityMapping.model_validate(raw)
    else:
        tree = parent.model_dump(mode='json')
        tree['issues'] = [dict(kind='source_ambiguity', source_id='s0001', question='A business dependency is still unresolved.')]
        parent = FlowTree.model_validate(tree)
    packet = compile_responsibilities(source, parent, proposal)['reviewInput']
    result = assess_responsibilities(source, parent, proposal, review_fixture(packet))
    assert result['status'] == 'blocked' and not result['runtimeAuthorityGranted']


def test_shared_schema_without_duplicate_user_schema_and_inert_code():
    source, parent, proposal = fixture()
    wire = request(source, parent)
    Draft202012Validator(wire['format']).validate(proposal.model_dump(mode='json'))
    payload = json.loads(wire['messages'][1]['content'])
    assert 'outputSchema' not in payload
    assert wire['model'] == 'qwen3.5:9b'
    assert payload['compatibleTargetIds']['business_prerequisite'] == ['unresolved']
    source = source.model_copy(update={'source_text': source.source_text + '```sh\nrun_dangerous_code\n```\n'})
    raw = proposal.model_dump(mode='json')
    for key, clause in clauses(source).items():
        if clause['opaqueCode']:
            raw['selections'][key] = [row(clause['exactQuote'], 'unclassified', 'unresolved')]
    result = compile_responsibilities(source, parent, ResponsibilityMapping.model_validate(raw))
    assert not result['runtimeAuthorityGranted']
    key = next(key for key, clause in clauses(source).items() if clause['opaqueCode'])
    raw['selections'][key][0].update(kind='operation', targets=['operation:/steps/0'])
    with pytest.raises(ValueError, match='opaque code'):
        project(source, parent, ResponsibilityMapping.model_validate(raw))


def test_cli_offline_roundtrip_and_no_overwrite(tmp_path):
    source, parent, proposal = fixture()
    for name, value in [('sources', source), ('tree', parent), ('proposal', proposal)]:
        (tmp_path / (name + '.json')).write_text(value.model_dump_json())
    inputs = [str(tmp_path / 'sources.json'), str(tmp_path / 'tree.json')]
    wire_path, compile_path, assess_path = [tmp_path / name for name in ['wire.json', 'compiled.json', 'assessed.json']]
    assert main(['request', *inputs, '--output', str(wire_path)]) == 0
    assert json.loads(wire_path.read_text()) == request(source, parent)
    assert main(['compile', *inputs, str(tmp_path / 'proposal.json'), '--output', str(compile_path)]) == 0
    compiled = json.loads(compile_path.read_text())
    (tmp_path / 'review.json').write_text(review_fixture(compiled['reviewInput']).model_dump_json())
    assert main(['assess', *inputs, str(tmp_path / 'proposal.json'), str(tmp_path / 'review.json'), '--output', str(assess_path)]) == 0
    assert not json.loads(assess_path.read_text())['runtimeAuthorityGranted']
    before = wire_path.read_bytes()
    with pytest.raises(FileExistsError, match='preserve previous evidence'):
        main(['request', *inputs, '--output', str(wire_path)])
    assert wire_path.read_bytes() == before
