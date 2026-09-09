import json

import httpx
import pytest

from evaluation import flow_two_pass_pilot as pilot
from evaluation.flow_translation import _write, local_sources
from network_runtime.contracts import sha256_json


@pytest.mark.parametrize('mode', ['valid', 'bad-flow', 'bad-mapping', 'transport', 'resume'])
def test_fresh_two_phase_costs_checkpoints_and_failed_first_pass(tmp_path, monkeypatch, mode):
    source = local_sources().model_copy(update={'source_text': 'Read input device_id and finish.\nSnapshot is not live health.\n'})
    tree = dict(business_source_ids=['s0001'], steps=[
        dict(kind='read', source_id='s0001', bind='snapshot', tool='read_inventory_device',
             arguments=dict(device_id=dict(kind='reference', source='input', field='device_id'))),
        dict(kind='end', source_id='s0001', outcome='read_path_completed')], issues=[])
    mapping = dict(objective_clause_ids=['clause-0001'], clause_dispositions={
        'clause-0001': [dict(evidence_clause_ids=['clause-0001'], disposition=dict(kind='operation',
            node_pointers=['/steps/0', '/steps/1'], explanation='Read the input identifier once, then finish.'))],
        'clause-0002': [dict(evidence_clause_ids=['clause-0002'], disposition=dict(kind='documentation',
            explanation='Preserve snapshot limitations without claiming live health checks.'))]},
        node_clause_ids={'/steps/0': ['clause-0001'], '/steps/1': ['clause-0001']})
    body = dict(cases=[dict(id='fixture', sources=source.model_dump(mode='json'))])
    src = tmp_path / 'sources.json'
    _write(src, {**body, 'manifestDigest': sha256_json(body)})
    model = dict(model='qwen3.5:9b', digest='test-only')
    monkeypatch.setattr(pilot.OllamaAnchoredAuthorAdapter, 'preflight', lambda self: model)
    root = tmp_path / 'run'
    pilot.freeze(src, root)
    calls = []

    def post(self, url, **kwargs):
        wire = kwargs['json']
        phase = 'mapping' if 'node_clause_ids' in wire['format']['properties'] else 'flow'
        calls.append(phase)
        if mode == 'transport':
            raise httpx.ConnectError('test unavailable')
        value = mapping if phase == 'mapping' else tree
        if (mode == 'bad-flow' and phase == 'flow') or (mode == 'bad-mapping' and phase == 'mapping'):
            value = {}
        return httpx.Response(200, request=httpx.Request('POST', url), json=dict(
            message=dict(content=json.dumps(value)), prompt_eval_count=11, eval_count=22))

    monkeypatch.setattr(httpx.Client, 'post', post)
    if mode == 'resume':
        m = pilot.load(root)
        pilot.phase(root / 'fixture/flow', source, None, m['cases'][0]['flowRequest'], model)
    pilot.run(root)
    pilot.run(root)
    r = pilot.report(root)
    expected = 1 if mode in ['bad-flow', 'transport'] else 2
    assert len(calls) == r['modelCalls'] == expected
    assert r['inputTokens'] == (0 if mode == 'transport' else expected * 11)
    assert r['outputTokens'] == (0 if mode == 'transport' else expected * 22)
    assert r['tokenCountsComplete'] == (mode != 'transport')
    assert r['mappingQualified'] == int(mode in ['valid', 'resume'])
    assert r['sourceReviewed'] == r['reviewSupportedInactiveFlows'] == r['runtimeExecutions'] == 0
    if mode in ['bad-flow', 'transport']:
        assert not (root / 'fixture/mapping').exists()
    (root / 'fixture/flow/status.json').write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        pilot.run(root)
    assert len(calls) == expected
