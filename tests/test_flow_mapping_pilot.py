import json

import httpx
import pytest

from evaluation import flow_mapping_pilot as pilot
from evaluation.flow_translation import _write, local_sources
from network_runtime.contracts import sha256_json


@pytest.mark.parametrize('valid', [True, False])
def test_metadata_first_attempt_replay_preserves_parent_and_never_retries(tmp_path, monkeypatch, valid):
    source = local_sources().model_copy(update={'source_text': 'Read input device_id once and finish.\nSnapshot is not live health.\n'})
    tree = {'business_source_ids': ['s0001'], 'steps': [
        {'kind': 'read', 'source_id': 's0001', 'bind': 'snapshot', 'tool': 'read_inventory_device',
         'arguments': {'device_id': {'kind': 'reference', 'source': 'input', 'field': 'device_id'}}},
        {'kind': 'end', 'source_id': 's0001', 'outcome': 'read_path_completed'}], 'issues': []}
    model = {'model': 'qwen3.5:9b', 'digest': 'unit-test'}
    monkeypatch.setattr(pilot.OllamaAnchoredAuthorAdapter, 'preflight', lambda self: model)
    # Parent provenance is covered by the bounded-pilot replay tests, not fabricated as real evidence here.
    parents_checked = []
    monkeypatch.setattr(pilot, 'parent_report', lambda p: parents_checked.append(p) or {'reportDigest': 'test-only'})
    parent = tmp_path / 'parent'
    (parent / 'fixture').mkdir(parents=True)
    body = {'cases': [{'id': 'fixture', 'sources': source.model_dump(mode='json')}]}
    _write(parent / 'manifest.json', {**body, 'manifestDigest': sha256_json(body)})
    _write(parent / 'fixture/tree.json', tree)
    parent_bytes = (parent / 'fixture/tree.json').read_bytes()
    root = tmp_path / 'run'
    manifest = pilot.freeze(parent, root)
    assert parents_checked == [parent]
    assert manifest['evidenceRole'] == 'auxiliary_mapping_over_preserved_9b_proposal_not_fresh_translation'
    raw = {'objective_source_ids': ['s0001'], 'source_dispositions': {
        's0001': [{'kind': 'operation', 'node_pointers': ['/steps/0', '/steps/1'],
                    'explanation': 'Read the input once and then finish this test path.'}],
        's0002': [{'kind': 'documentation', 'explanation': 'Retain the snapshot limitation without claiming live telemetry.'}]},
        'node_source_ids': {'/steps/0': 's0001', '/steps/1': 's0001'}}
    if not valid:
        raw['source_dispositions']['s0001'][0]['node_pointers'] = ['read_inventory_device']
    calls = []

    def post(self, url, **kwargs):
        calls.append(kwargs)
        return httpx.Response(200, request=httpx.Request('POST', url), json={
            'message': {'content': json.dumps(raw)}, 'eval_count': 20, 'prompt_eval_count': 10})

    monkeypatch.setattr(httpx.Client, 'post', post)
    pilot.run(root)
    pilot.run(root)
    assert len(calls) == 1
    report = pilot.report(root)
    assert report['structurallyQualified'] == int(valid)
    assert report['runtimeExecutions'] == report['sourceReviewed'] == 0
    assert report['rows'][0]['outputTokens'] == 20
    assert (parent / 'fixture/tree.json').read_bytes() == parent_bytes
    (root / 'fixture/status.json').write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        pilot.report(root)
