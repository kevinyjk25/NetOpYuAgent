"""Fault-localization fixtures, not semantic Gold or translator success cases."""

import copy
import json

import httpx
import pytest

from evaluation import flow_diagnostics as diagnostics
from evaluation.flow_canonical_mapping import compile_canonical
from evaluation.flow_tree import tree_review_input
from network_runtime.contracts import sha256_json
from tests.test_flow_canonical_mapping import branch_fixture, fixture, terminal_fixture
from tests.test_flow_lean_mapping import review_fixture


def inputs(builder=fixture):
    return tuple(x.model_dump(mode='json') for x in builder())


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail('Offline diagnosis attempted network access')
    monkeypatch.setattr(httpx.Client, 'request', refuse)


def test_valid_trace_is_not_semantic_accuracy_or_runtime_authority():
    source, tree, mapping = inputs()
    before = copy.deepcopy((source, tree, mapping))
    r = diagnostics.diagnose(source, tree, mapping)
    assert (source, tree, mapping) == before
    assert r['earliestFailedStage'] is None
    assert r['stages']['mapping']['status'] == 'compiled_pending_review'
    assert r['stages']['lowering_consistency']['status'] == 'execution_projection_equal'
    assert r['stages']['semantic_review']['status'] == r['stages']['runtime']['status'] == 'not_evaluated'
    assert r['semanticAccuracy'] is r['semanticLossRate'] is r['confidenceProbability'] is None
    assert r['independentObligationCount'] is None
    assert r['modelCalls'] == r['runtimeExecutions'] == r['writes'] == 0
    assert not r['runtimeAuthorityGranted']
    assert r['reportDigest'] == sha256_json({k: v for k, v in r.items() if k != 'reportDigest'})
    assert r == diagnostics.diagnose(source, tree, mapping)
    for row in r['candidateRequirements']:
        span = row['sourceSpan']
        assert source['source_text'][span['start']:span['end']] == span['exactQuote']
    assert all(n['eligibleCandidatePointers'] and n['l0Node'] for n in r['nodeTrace'])


def test_collects_conflict_and_missing_condition_beyond_first_failure():
    source, tree, mapping = inputs(branch_fixture)
    mapping['objective'] = ['clause-0001']
    mapping['selections']['clause-0001'][0].update(kind='input_shape', targets=['rule:read_input_shape'])
    mapping['selections']['clause-0002'].pop(0)
    r = diagnostics.diagnose(source, tree, mapping)
    assert r['earliestFailedStage'] == 'mapping'
    assert r['counts']['byCode'] == {'node_evidence_missing': 2, 'objective_kind_conflict': 1}
    condition = next(n for n in r['nodeTrace'] if n['treePointer'] == '/steps/1')
    assert condition['node']['kind'] == 'if_equal' and condition['l0Node']['kind'] == 'branch'
    assert condition['l0Pointer'] == '/nodes/1'
    assert not condition['eligibleCandidatePointers']
    assert r['stages']['semantic_review']['status'] == 'not_evaluated'


@pytest.mark.parametrize('defect,code', [
    ('quote', 'quote_not_unique_exact'), ('target', 'unknown_target'),
    ('kind', 'incompatible_target_kind'), ('duplicate', 'duplicate_requirement'),
    ('extra', 'own_source_repeated'), ('clause', 'source_clause_unmapped'),
])
def test_actionable_local_defects(defect, code):
    source, tree, mapping = inputs()
    row = mapping['selections']['clause-0001'][0]
    if defect == 'quote':
        row['exact_quote'] = 'a fabricated source'
    elif defect == 'target':
        row['targets'] = ['node:/steps/999']
    elif defect == 'kind':
        row['kind'] = 'completion'
    elif defect == 'duplicate':
        mapping['selections']['clause-0001'].append(copy.deepcopy(row))
    elif defect == 'extra':
        row['extra_sources'] = ['clause-0001']
    else:
        del mapping['selections']['clause-0002']
    r = diagnostics.diagnose(source, tree, mapping)
    assert code in r['counts']['byCode']
    assert r['stages']['mapping']['status'] == 'failed'
    assert all(f['pointer'] and f['suggestedAction'] for f in r['findings'] if f['code'] == code)


@pytest.mark.parametrize('value', [None, [], 'bad', {}, {'selections': []},
    {'selections': {'clause-0001': [None]}},
    {'objective': ['clause-0001'], 'selections': {'clause-0001': [{'kind': [], 'targets': [{}], 'exact_quote': 3}]}}])
def test_malformed_json_shapes_are_diagnosed_without_crashing(value):
    source, tree, _ = inputs()
    r = diagnostics.diagnose(source, tree, value)
    assert r['earliestFailedStage'] == 'mapping'
    assert r['counts']['byCode']['mapping_schema_invalid'] >= 1


def test_ambiguous_or_negation_shortened_quote_does_not_prove_entailment():
    source, tree, mapping = inputs()
    mapping['selections']['clause-0002'][0]['exact_quote'] = '实时健康'
    r = diagnostics.diagnose(source, tree, mapping)
    assert r['stages']['mapping']['status'] == 'compiled_pending_review'
    assert r['stages']['semantic_review']['status'] == 'not_evaluated'
    assert all(c['semanticSupport'] == 'not_established' for c in r['candidateRequirements'])
    source['source_text'] = source['source_text'].replace('库存不是实时健康', '库存不是实时健康也不是实时健康')
    r = diagnostics.diagnose(source, tree, mapping)
    assert r['counts']['byCode']['quote_not_unique_exact'] == 1


def test_source_review_counterexample_is_localized_and_bound():
    source, tree, proposal = fixture()
    packet = compile_canonical(source, tree, proposal)['reviewInput']
    review = review_fixture(packet).model_dump(mode='json')
    claim = next(c for c in packet['claims'] if c['facet'].startswith('source_entails_requirement_type'))
    verdict = next(c for c in review['assessment']['claims'] if c['claim_id'] == claim['claimId'])
    verdict.update(verdict='contradicted', rationale='Synthetic wrong-role counterexample.', suggested_revision='Review the quoted action and preserve its role.')
    args = [x.model_dump(mode='json') for x in (source, tree, proposal)]
    r = diagnostics.diagnose(*args, review=review)
    assert r['stages']['semantic_review']['status'] == 'failed'
    finding = next(f for f in r['findings'] if f['code'] == 'source_review_contradicted')
    assert finding['pointer'] == claim['pointer']
    assert finding['evidenceBasis'] == 'supplied_reviewer_judgment'
    review['assessment']['input_digest'] = 'sha256:' + '0' * 64
    r = diagnostics.diagnose(*args, review=review)
    assert r['stages']['semantic_review']['status'] == 'failed' and r['sourceReview'] is None


def test_positive_fixture_review_does_not_create_accuracy_denominator():
    source, tree, proposal = fixture()
    review = review_fixture(compile_canonical(source, tree, proposal)['reviewInput']).model_dump(mode='json')
    r = diagnostics.diagnose(*inputs(), review=review)
    assert r['stages']['semantic_review']['status'] == 'review_supported_not_proof'
    assert r['semanticAccuracy'] is r['independentObligationCount'] is None
    assert r['sourceReview']['reviewerKind'] == 'test_fixture'


def test_reported_host_gap_and_mapping_omission_are_separate():
    source, tree, mapping = inputs(lambda: terminal_fixture('unsupported'))
    tree['issues'] = [dict(kind='missing_host_capability', source_id='s0001', question='The required approval API is not in the supplied host contract.')]
    r = diagnostics.diagnose(source, tree, mapping)
    assert r['stages']['mapping']['status'] == 'compiled_pending_review'
    assert r['counts']['byCode'] == {'declared_missing_host_capability': 1}
    assert r['findings'][0]['evidenceBasis'] == 'model_reported_not_independently_verified'
    mapping['selections']['clause-0001'][0]['targets'] = ['unresolved']
    r = diagnostics.diagnose(source, tree, mapping)
    assert r['counts']['byCode']['node_evidence_missing'] == 1
    assert r['counts']['byCode']['unresolved_requirement'] == 1


def test_opaque_script_is_inert_and_not_executed(tmp_path):
    source, tree, mapping = inputs()
    marker = tmp_path / 'must-not-exist'
    source['source_text'] += '\n```sh\ntouch ' + str(marker) + '\n```\n'
    r = diagnostics.diagnose(source, tree, mapping)
    assert any(c['opaqueCode'] for c in r['sourceClauses'])
    assert not marker.exists() and r['writes'] == 0


@pytest.mark.parametrize('field,value', [('source', {}), ('tree', {})])
def test_upstream_failure_does_not_turn_downstream_into_zero_percent(field, value):
    source, tree, mapping = inputs()
    r = diagnostics.diagnose(value if field == 'source' else source, value if field == 'tree' else tree, mapping)
    assert r['earliestFailedStage'] == ('source' if field == 'source' else 'representation')
    assert r['stages']['mapping']['status'] == 'not_evaluated'
    assert r['semanticLossRate'] is None


def test_lowering_drift_in_branch_polarity_is_not_a_mapping_omission(monkeypatch):
    original = diagnostics.compile_canonical
    def bad_compile(*args):
        r = copy.deepcopy(original(*args))
        branch = r['flow']['nodes'][1]
        branch['on_true'], branch['on_false'] = branch['on_false'], branch['on_true']
        return r
    monkeypatch.setattr(diagnostics, 'compile_canonical', bad_compile)
    r = diagnostics.diagnose(*inputs(branch_fixture))
    assert r['earliestFailedStage'] == 'lowering_consistency'
    assert r['counts']['byCode'] == {'lowering_execution_changed': 1}


def test_same_tree_assisted_witness_is_not_an_answer_repair_or_success():
    source, tree, mapping = inputs(branch_fixture)
    witness = dict(sourcesDigest=sha256_json(source), tree=tree, mapping=copy.deepcopy(mapping))
    mapping['selections']['clause-0002'].pop(0)
    r = diagnostics.diagnose(source, tree, mapping, witness=witness)
    assert r['earliestFailedStage'] == 'mapping'
    assert r['witness']['sameExecutionTree'] and r['witness']['mechanicallyConstructible']
    assert not r['witness']['replacesOriginal'] and not r['witness']['countsAsFirstPassSuccess']
    witness['sourcesDigest'] = 'bad'
    assert diagnostics.diagnose(source, tree, mapping, witness=witness)['witness']['status'] == 'rejected'


def test_witness_can_localize_an_invalid_first_pass_without_replacing_it():
    source, tree, mapping = inputs()
    witness = dict(sourcesDigest=sha256_json(source), tree=tree, mapping=mapping)
    r = diagnostics.diagnose(source, {}, mapping, witness=witness)
    assert r['earliestFailedStage'] == 'representation'
    assert r['witness']['mechanicallyConstructible'] and not r['witness']['sameExecutionTree']


def test_first_pass_semantic_review_can_run_while_mapping_is_blocked():
    source, tree, _ = fixture()
    packet = tree_review_input(source, tree)
    review = review_fixture(packet).model_dump(mode='json')
    claim = packet['claims'][0]
    item = next(j for j in review['assessment']['claims'] if j['claim_id'] == claim['claimId'])
    item.update(verdict='contradicted', rationale='Synthetic source intent mismatch.', suggested_revision='Restore the original intention.')
    r = diagnostics.diagnose(source.model_dump(mode='json'), tree.model_dump(mode='json'), {}, tree_review=review)
    assert r['earliestFailedStage'] == 'parent_semantic_review'
    assert r['stages']['mapping']['status'] == 'failed'
    assert r['stages']['semantic_review']['status'] == 'not_evaluated'
    assert r['parentSourceReview']['reviewerKind'] == 'test_fixture'


def test_reordering_mapping_keys_preserves_findings_and_source_span_identity():
    source, tree, mapping = inputs()
    first = diagnostics.diagnose(source, tree, mapping)
    mapping['selections'] = dict(reversed(list(mapping['selections'].items())))
    second = diagnostics.diagnose(source, tree, mapping)
    for key in ('findings', 'candidateRequirements', 'nodeTrace', 'counts', 'inputDigests'):
        assert first[key] == second[key]
    # Existing compiler claim ordering depends on iteration order; do not rewrite its receipt.
    assert first['mappedCompilationDigest'] != second['mappedCompilationDigest']


def test_cli_never_overwrites_or_places_report_in_frozen_batch(tmp_path):
    paths = []
    for name, value in zip(('sources', 'tree', 'mapping'), inputs(), strict=True):
        path = tmp_path / (name + '.json')
        path.write_text(json.dumps(value))
        paths.append(str(path))
    out = tmp_path / 'report.json'
    assert diagnostics.main(['case', *paths, '--output', str(out)]) == 0
    before = out.read_bytes()
    with pytest.raises(FileExistsError):
        diagnostics.main(['case', *paths, '--output', str(out)])
    assert out.read_bytes() == before
    with pytest.raises(ValueError, match='outside frozen'):
        diagnostics.main(['batch', str(tmp_path), '--output', str(tmp_path / 'new.json')])


def test_batch_requires_frozen_replay_and_does_not_rescore(tmp_path, monkeypatch):
    from evaluation import flow_canonical_pilot as pilot
    source, tree, mapping = inputs()
    for phase, name, value in [('flow', 'tree', tree), ('mapping', 'mapping', mapping)]:
        folder = tmp_path / 'case-a' / phase
        folder.mkdir(parents=True)
        (folder / (name + '.json')).write_text(json.dumps(value))
    calls = []
    def replay(root):
        calls.append('validated-replay')
        return dict(reportDigest='frozen', flowQualified=1, mappingQualified=1)
    monkeypatch.setattr(pilot, 'report', replay)
    monkeypatch.setattr(pilot, 'load', lambda root: dict(manifestDigest='manifest', cases=[dict(id='case-a', sources=source)]))
    r = diagnostics.diagnose_batch(tmp_path)
    assert calls == ['validated-replay'] and r['originalReportDigest'] == 'frozen'
    assert r['originalMappingQualified'] == 1 and r['modelCalls'] == 0
    def reject(root):
        raise ValueError('frozen implementation drift')
    monkeypatch.setattr(pilot, 'report', reject)
    with pytest.raises(ValueError, match='drift'):
        diagnostics.diagnose_batch(tmp_path)
