"""Development counterexamples, not independently authored semantic Gold."""

import copy
import json

import pytest
from jsonschema import Draft202012Validator

from evaluation.flow_clause_mapping import clauses
from evaluation.flow_node_evidence import (
    HANDLING, EvidenceMapping, assess_evidence, compile_evidence, main, projection, request, schema,
)
from evaluation.flow_tree import FlowTree, compile_report
from evaluation.read_l05_review import ReadL05Review
from tests.test_flow_canonical_mapping import branch_fixture, fixture as canonical_fixture, terminal_fixture
from tests.test_flow_lean_mapping import review_fixture


def fixture(builder=canonical_fixture):
    source, tree, canonical = builder()
    nodes, residuals = {}, {key: [] for key in canonical.selections}
    for anchor, rows in canonical.selections.items():
        for row in rows:
            for target in row.targets:
                if target.startswith('node:'):
                    slot = nodes.setdefault(target[5:], dict(status='evidence_candidate', citations=[], reason=''))
                    slot['citations'].append(dict(clause=anchor, exact_quote=row.exact_quote, objective=anchor in canonical.objective))
                else:
                    handling = next(k for k, v in HANDLING.items() if v == (row.kind, target))
                    residuals[anchor].append(dict(exact_quote=row.exact_quote, handling=handling, objective=False))
    return source, tree, EvidenceMapping.model_validate(dict(node_evidence=nodes, residuals=residuals))


def test_mandatory_slots_and_compiler_roles_remove_independent_kind_and_objective_choice():
    source, tree, proposal = fixture()
    wire = request(source, tree)
    assert set(wire['format']['properties']) == {'node_evidence', 'residuals'}
    assert 'objective' not in wire['format']['properties']
    payload = json.loads(wire['messages'][1]['content'])
    assert payload['fixedNodes']['/steps/0']['compilerRole'] == 'operation'
    assert payload['fixedNodes']['/steps/1']['compilerRole'] == 'completion'
    canonical, diagnostic, trace = projection(source, tree, proposal)
    assert not diagnostic['errors'] and len(trace) == 6
    assert canonical.objective == ('clause-0001',)
    result = compile_evidence(source, tree, proposal)
    assert result['executionProjectionUnchanged'] and not result['runtimeAuthorityGranted']
    assert result['flow']['nodes'] == compile_report(source, tree)['flow']['nodes']
    packet = result['reviewInput']
    assert packet['rolesOwnedByCompiler'] and not packet['completeSourceDecompositionProven']
    assert sum(c['facet'].startswith('node_evidence_entails') for c in packet['claims']) == 2
    r = assess_evidence(source, tree, proposal, review_fixture(packet))
    assert r['representationReviewSupported'] and not r['runtimeAuthorityGranted']


@pytest.mark.parametrize('status', ['insufficient_evidence', 'contradicted'])
def test_required_slot_can_disagree_without_manufacturing_support(status):
    source, tree, proposal = fixture(branch_fixture)
    raw = proposal.model_dump(mode='json')
    raw['node_evidence']['/steps/1'] = dict(status=status, citations=[], reason='The source does not establish this predicate or its polarity.')
    Draft202012Validator(schema(source, tree)).validate(raw)
    proposal = EvidenceMapping.model_validate(raw)
    canonical, diagnostic, _ = projection(source, tree, proposal)
    assert canonical is None and diagnostic['explicitNodeGaps'] == 1
    assert 'node_' + status in [e['code'] for e in diagnostic['errors']]
    with pytest.raises(ValueError, match='node_' + status):
        compile_evidence(source, tree, proposal)


@pytest.mark.parametrize('mutation', ['missing-node', 'empty-support', 'fake-kind', 'independent-objective',
    'wrong-residual-objective', 'unknown-citation', 'unknown-node', 'no-gap-reason', 'gap-objective', 'missing-clause'])
def test_generation_schema_prevents_old_cross_record_and_forced_support_errors(mutation):
    source, tree, proposal = fixture()
    raw = proposal.model_dump(mode='json')
    slot = raw['node_evidence']['/steps/0']
    if mutation == 'missing-node':
        del raw['node_evidence']['/steps/1']
    elif mutation == 'empty-support':
        slot['citations'] = []
    elif mutation == 'fake-kind':
        slot['kind'] = 'input_shape'
    elif mutation == 'independent-objective':
        raw['objective'] = ['clause-0002']
    elif mutation == 'wrong-residual-objective':
        raw['residuals']['clause-0002'][0]['objective'] = True
    elif mutation == 'unknown-citation':
        slot['citations'][0]['clause'] = 'clause-9999'
    elif mutation == 'unknown-node':
        raw['node_evidence']['/steps/99'] = copy.deepcopy(slot)
    elif mutation == 'no-gap-reason':
        slot.update(status='insufficient_evidence', citations=[], reason='')
    elif mutation == 'gap-objective':
        slot.update(status='contradicted', reason='The positive evidence is absent.')
    else:
        del raw['residuals']['clause-0002']
    assert not Draft202012Validator(schema(source, tree)).is_valid(raw)


@pytest.mark.parametrize('outcome', ['read_path_completed', 'needs_l1', 'unsupported'])
def test_each_terminal_is_expressible_not_an_effect(outcome):
    source, tree, proposal = fixture(lambda: terminal_fixture(outcome))
    r = compile_evidence(source, tree, proposal)
    assert r['flow']['nodes'][0]['outcome'] == outcome
    assert not r['runtimeAuthorityGranted']


@pytest.mark.parametrize('defect,code', [('fake-quote', 'quote_not_unique_exact'), ('duplicate', 'duplicate_requirement'),
    ('no-objective', 'objective_evidence_missing_or_budget'), ('uncited-empty', 'source_clause_unaccounted')])
def test_projection_still_checks_sources_not_just_required_json_keys(defect, code):
    source, tree, proposal = fixture()
    raw = proposal.model_dump(mode='json')
    if defect == 'fake-quote':
        raw['node_evidence']['/steps/0']['citations'][0]['exact_quote'] = 'made up'
    elif defect == 'duplicate':
        raw['residuals']['clause-0002'] *= 2
    elif defect == 'no-objective':
        for slot in raw['node_evidence'].values():
            for citation in slot['citations']:
                citation['objective'] = False
    else:
        raw['residuals']['clause-0002'] = []
    canonical, diagnostics, _ = projection(source, tree, EvidenceMapping.model_validate(raw))
    assert canonical is None and code in [e['code'] for e in diagnostics['errors']]


@pytest.mark.parametrize('defect', ['negation', 'missing-residual', 'wrong-rule', 'invented-node-evidence'])
def test_format_valid_semantic_counterexamples_require_full_source_review(defect):
    source, tree, proposal = fixture()
    raw = proposal.model_dump(mode='json')
    if defect == 'negation':
        raw['residuals']['clause-0002'][0]['exact_quote'] = '实时健康'
    elif defect == 'missing-residual':
        raw['residuals']['clause-0003'].pop(1)
    elif defect == 'wrong-rule':
        raw['residuals']['clause-0003'][0]['handling'] = 'read_access'
    else:
        raw['node_evidence']['/steps/0']['citations'][0].update(clause='clause-0002', exact_quote='库存不是实时健康')
    proposal = EvidenceMapping.model_validate(raw)
    packet = compile_evidence(source, tree, proposal)['reviewInput']
    review = review_fixture(packet).model_dump(mode='json')
    claim = next(c for c in packet['claims'] if c['facet'].startswith(
        'node_evidence_entails' if defect == 'invented-node-evidence' else 'complete_clause_decomposition'))
    verdict = next(j for j in review['assessment']['claims'] if j['claim_id'] == claim['claimId'])
    verdict.update(verdict='insufficient_evidence', rationale='Synthetic source-fidelity counterexample.',
        suggested_revision='Review the original complete clause, not only the legal target label.')
    r = assess_evidence(source, tree, proposal, ReadL05Review.model_validate(review))
    assert not r['representationReviewSupported'] and r['status'] == 'blocked'


def test_real_gap_can_coexist_with_evidenced_stop_without_becoming_implemented():
    source, tree, proposal = fixture(lambda: terminal_fixture('unsupported'))
    tree = FlowTree.model_validate({**tree.model_dump(mode='json'), 'issues': [dict(kind='missing_host_capability', source_id='s0001', question='Required approval API is absent from host context.')]})
    raw = proposal.model_dump(mode='json')
    raw['residuals']['clause-0001'] = [dict(exact_quote=source.source_text, handling='unresolved', objective=True)]
    proposal = EvidenceMapping.model_validate(raw)
    packet = compile_evidence(source, tree, proposal)['reviewInput']
    r = assess_evidence(source, tree, proposal, review_fixture(packet))
    assert r['representationReviewSupported']
    assert r['admissionBlockers'] == ['parent_issues', 'unresolved_requirements']


def test_review_binding_includes_new_protocol_and_objective_flags():
    source, tree, proposal = fixture()
    review = review_fixture(compile_evidence(source, tree, proposal)['reviewInput'])
    raw = proposal.model_dump(mode='json')
    # Both cite same clause; canonical objective stays equal but the raw claim changes.
    raw['node_evidence']['/steps/1']['citations'][0]['objective'] = False
    with pytest.raises(ValueError, match='digest'):
        assess_evidence(source, tree, EvidenceMapping.model_validate(raw), review)


def test_source_code_only_still_has_valid_schema_and_no_forced_positive_answer():
    source, tree, _ = fixture(lambda: terminal_fixture('unsupported'))
    source = source.model_copy(update={'source_text': '```sh\nnever_run_me\n```'})
    raw_tree = tree.model_dump(mode='json')
    raw_tree['business_source_ids'] = ['s0002']
    raw_tree['steps'][0]['source_id'] = 's0002'
    tree = FlowTree.model_validate(raw_tree)
    s = schema(source, tree)
    Draft202012Validator.check_schema(s)
    assert len(s['$defs']['NodeEvidence']['anyOf']) == 1
    raw = dict(node_evidence={'/steps/0': dict(status='insufficient_evidence', citations=[], reason='Opaque source cannot establish this stop.')},
        residuals={k: [dict(exact_quote=c['exactQuote'], handling='unresolved', objective=True)]
            for k, c in clauses(source).items()})
    Draft202012Validator(s).validate(raw)
    assert projection(source, tree, EvidenceMapping.model_validate(raw))[0] is None


def test_cli_preserves_artifacts(tmp_path):
    source, tree, proposal = fixture()
    files = []
    for name, obj in [('source', source), ('tree', tree), ('proposal', proposal)]:
        path = tmp_path / (name + '.json')
        path.write_text(obj.model_dump_json())
        files.append(str(path))
    out = tmp_path / 'compiled.json'
    args = ['compile', *files[:2], '--proposal', files[2], '--output', str(out)]
    assert main(args) == 0
    with pytest.raises(FileExistsError):
        main(args)
