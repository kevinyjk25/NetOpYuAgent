import json

import pytest
from jsonschema import ValidationError as SchemaError

from evaluation.flow_clause_mapping import ClauseMapping, assess_clauses, clauses, compile_clauses, request
from evaluation.flow_translation import local_sources
from evaluation.flow_tree import FlowTree
from evaluation.read_l05_review import ReadL05Review


def fixture():
    source = local_sources().model_copy(update={'source_text': '# Snapshot\nRead input device_id; then finish.\nSnapshot is not live health.\n'})
    tree = FlowTree.model_validate({'business_source_ids': ['s0002'], 'steps': [
        {'kind': 'read', 'source_id': 's0002', 'tool': 'read_inventory_device', 'bind': 'snapshot',
         'arguments': {'device_id': {'kind': 'reference', 'source': 'input', 'field': 'device_id'}}},
        {'kind': 'end', 'source_id': 's0002', 'outcome': 'read_path_completed'}], 'issues': []})
    raw = {'objective_clause_ids': ['clause-0001', 'clause-0002'], 'clause_dispositions': {
        'clause-0001': [{'evidence_clause_ids': ['clause-0001'], 'disposition': {
            'kind': 'operation', 'node_pointers': ['/steps/0'], 'explanation': 'Read the exact input device identifier.'}}],
        'clause-0002': [{'evidence_clause_ids': ['clause-0001', 'clause-0002'], 'disposition': {
            'kind': 'operation', 'node_pointers': ['/steps/1'], 'explanation': 'Finish after the read described by the preceding clause.'}}],
        'clause-0003': [{'evidence_clause_ids': ['clause-0003'], 'disposition': {
            'kind': 'documentation', 'explanation': 'Retain the snapshot interpretation without claiming live checks.'}}]},
        'node_clause_ids': {'/steps/0': ['clause-0001'], '/steps/1': ['clause-0001', 'clause-0002']}}
    return source, tree, raw


@pytest.mark.parametrize('text', [
    '读取设备；完成。不要写入。\n', 'Read input; finish. No writes.\n',
    'Use `scripts/a.py; not a separator` once. Stop.\n',
    'Use ``a;`b`` once; stop.\n', 'Same.\n\nSame.\n',
    '```python\n# inert comment\nprint("x;y")\n```\nStop.\n',
])
def test_exact_offsets_and_no_nonspace_body_loss(text):
    source = local_sources().model_copy(update={'source_text': text})
    rows = list(clauses(source).values())
    assert all(text[r['start']:r['end']] == r['exactQuote'] for r in rows)
    assert ''.join(''.join(r['exactQuote'].split()) for r in rows) == ''.join(text.split())
    assert all(a['end'] <= b['start'] for a, b in zip(rows, rows[1:]))
    if 'scripts/a.py' in text:
        assert any('scripts/a.py; not a separator' in r['exactQuote'] for r in rows)
    if '```' in text:
        assert len([r for r in rows if r['opaqueCode']]) == 4


def test_exact_multi_source_metadata_not_automatic_enforcement():
    source, tree, raw = fixture()
    before = tree.model_dump(mode='json')
    result = compile_clauses(source, tree, ClauseMapping.model_validate(raw))
    packet = result['reviewInput']
    assert tree.model_dump(mode='json') == before
    assert result['executionProjectionUnchanged'] and result['clauseAccountingComplete']
    assert not result['runtimeAuthorityGranted'] and not result['semanticAlignmentProven']
    assert any(c.get('requiredCitationIds') == ['clause-0001', 'clause-0002'] for c in packet['claims'])
    assert packet['paragraphProjectionRole'] == 'compiler_retention_only_not_model_enforcement'
    assert len([c for c in packet['claims'] if c['facet'].startswith('source_requirement')]) == 2


@pytest.mark.parametrize('change', ['missing', 'empty', 'unbound-anchor', 'unknown-clause', 'tool-name', 'inject-operation', 'mixed-type', 'duplicate'])
def test_exact_coverage_binding_and_metadata_only_shapes(change):
    source, parent, raw = fixture()
    if change == 'missing':
        raw['clause_dispositions'].pop('clause-0003')
    elif change == 'empty':
        raw['clause_dispositions']['clause-0003'] = []
    elif change == 'unbound-anchor':
        raw['clause_dispositions']['clause-0003'][0]['evidence_clause_ids'] = ['clause-0001']
    elif change == 'unknown-clause':
        raw['node_clause_ids']['/steps/0'] = ['clause-9999']
    elif change == 'tool-name':
        raw['clause_dispositions']['clause-0001'][0]['disposition']['node_pointers'] = ['read_inventory_device']
    elif change == 'inject-operation':
        raw['steps'] = []
    elif change == 'mixed-type':
        raw['clause_dispositions']['clause-0003'][0]['disposition']['host_rule_ids'] = ['read_access']
    else:
        raw['objective_clause_ids'] = ['clause-0001', 'clause-0001']
    with pytest.raises((ValueError, SchemaError)):
        compile_clauses(source, parent, ClauseMapping.model_validate(raw))


def review_fixture(packet):
    claims = []
    for c in packet['claims']:
        ids = list(c.get('requiredCitationIds', [])) + ([c['requiredCitationId']] if 'requiredCitationId' in c else [])
        for kind, fallback in [('skill', 'skill-0002'), ('host', 'host-0001')]:
            if kind in c['requiredEvidenceKinds'] and not any(s.startswith(kind + '-') or (kind == 'skill' and s.startswith('clause-')) for s in ids):
                ids.append(fallback)
        claims.append(dict(claim_id=c['claimId'], verdict='supported', source_span_ids=ids,
            rationale='Test fixture exercises binding, not actual semantic entailment.', suggested_revision=''))
    return ReadL05Review.model_validate(dict(reviewer_id='test', reviewer_kind='test_fixture', assessment=dict(
        input_digest=packet['inputDigest'], scope_note='Not semantic evidence; test plumbing only.', claims=claims)))


@pytest.mark.parametrize('unresolved', [False, True])
def test_unresolved_stays_blocked_and_no_review_grants_authority(unresolved):
    source, parent, raw = fixture()
    if unresolved:
        raw['clause_dispositions']['clause-0003'][0]['disposition']['kind'] = 'unresolved'
    proposal = ClauseMapping.model_validate(raw)
    result = assess_clauses(source, parent, proposal, review_fixture(compile_clauses(source, parent, proposal)['reviewInput']))
    assert (result['status'] == 'review_supported_inactive_flow') == (not unresolved)
    assert not result['runtimeAuthorityGranted']


def test_wrong_semantics_and_host_report_leakage_remain_review_claims():
    source, parent, raw = fixture()
    raw['clause_dispositions']['clause-0001'][0]['disposition']['explanation'] = 'Read and report every returned field even though the source only requests a read.'
    proposal = ClauseMapping.model_validate(raw)
    packet = compile_clauses(source, parent, proposal)['reviewInput']
    claim = next(c for c in packet['claims'] if c['l05Pointer'] == '/clause_dispositions/clause-0001/0')
    review = review_fixture(packet).model_dump(mode='json')
    row = next(c for c in review['assessment']['claims'] if c['claim_id'] == claim['claimId'])
    row.update(verdict='insufficient_evidence', rationale='The exact source requires reading, not a new report operation.',
               suggested_revision='Remove the invented report or supply actual source intent and an appropriate flow node.')
    assert assess_clauses(source, parent, proposal, ReadL05Review.model_validate(review))['status'] == 'blocked'
    wire = request(source, parent)
    assert 'not permission to add a user operation' in wire['messages'][0]['content']
    assert 'sourceClauseCatalog' in json.loads(wire['messages'][1]['content'])


def test_same_graph_but_clause_note_edit_invalidates_review():
    source, parent, raw = fixture()
    proposal = ClauseMapping.model_validate(raw)
    original = compile_clauses(source, parent, proposal)
    review = review_fixture(original['reviewInput'])
    raw['clause_dispositions']['clause-0003'][0]['disposition']['explanation'] = 'Another interpretation must receive a new source review before acceptance.'
    changed = ClauseMapping.model_validate(raw)
    assert compile_clauses(source, parent, changed)['flow'] == original['flow']
    with pytest.raises(ValueError, match='digest'):
        assess_clauses(source, parent, changed, review)


@pytest.mark.parametrize('text', ['# Heading only\n', 'Stop.\n' * 97])
def test_empty_or_oversized_clause_catalog_refused_without_truncation(text):
    source = local_sources().model_copy(update={'source_text': text})
    with pytest.raises(ValueError, match='1..96'):
        clauses(source)


def test_missing_exact_clause_citation_cannot_be_laundered_with_paragraph():
    source, parent, raw = fixture()
    proposal = ClauseMapping.model_validate(raw)
    packet = compile_clauses(source, parent, proposal)['reviewInput']
    review = review_fixture(packet).model_dump(mode='json')
    target = next(c for c in packet['claims'] if c['l05Pointer'] == '/clause_dispositions/clause-0003/0')
    next(c for c in review['assessment']['claims'] if c['claim_id'] == target['claimId'])['source_span_ids'] = ['skill-0003']
    with pytest.raises(ValueError, match='exact clause'):
        assess_clauses(source, parent, proposal, ReadL05Review.model_validate(review))


def test_supported_metadata_cannot_clear_parent_issue():
    source, parent, raw = fixture()
    tree = parent.model_dump(mode='json')
    tree['issues'] = [dict(kind='source_ambiguity', source_id='s0002', question='A source ambiguity remains in this synthetic test fixture.')]
    parent = FlowTree.model_validate(tree)
    proposal = ClauseMapping.model_validate(raw)
    packet = compile_clauses(source, parent, proposal)['reviewInput']
    assert assess_clauses(source, parent, proposal, review_fixture(packet))['status'] == 'blocked'


def test_opaque_code_comments_remain_clause_evidence_without_execution():
    source = local_sources().model_copy(update={'source_text': '```python\n# inert comment\nprint("never run")\n```\nStop unsupported.\n'})
    parent = FlowTree.model_validate(dict(business_source_ids=['s0005'], steps=[
        dict(kind='end', source_id='s0005', outcome='unsupported')], issues=[
            dict(kind='missing_host_capability', source_id='s0005', question='Host has no script contract; this test must not execute code.')]))
    catalog = clauses(source)
    raw = dict(objective_clause_ids=['clause-0005'], node_clause_ids={'/steps/0': ['clause-0005']}, clause_dispositions={
        key: [dict(evidence_clause_ids=[key], disposition=dict(kind='unresolved', explanation=
            'Inert source is preserved with the unavailable script capability explicitly unresolved.'))] for key in catalog})
    packet = compile_clauses(source, parent, ClauseMapping.model_validate(raw))['reviewInput']
    assert any(row['exactQuote'] == '# inert comment' for row in packet['sourceSpans'])
    assert not packet['runtimeAuthorityGranted']
