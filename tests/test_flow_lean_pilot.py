import json

import httpx
import pytest

from evaluation import flow_lean_pilot as pilot
from evaluation.flow_translation import _write, local_sources
from network_runtime.contracts import sha256_json


@pytest.mark.parametrize('mode', ['valid', 'bad-flow', 'bad-mapping', 'transport', 'resume'])
def test_two_new_calls_no_historical_tree_and_resume_without_duplicates(tmp_path, monkeypatch, mode):
    source = local_sources().model_copy(update={'source_text': 'Read input device_id and finish.\nSnapshot is not live health.\n'})
    tree = dict(business_source_ids=['s0001'], steps=[
        dict(kind='read', source_id='s0001', bind='snapshot', tool='read_inventory_device',
             arguments=dict(device_id=dict(kind='reference', source='input', field='device_id'))),
        dict(kind='end', source_id='s0001', outcome='read_path_completed')], issues=[])
    mapping = dict(objective=['clause-0001'], selections={
        'clause-0001': dict(targets=['operation:/steps/0', 'operation:/steps/1'], extra_sources=[]),
        'clause-0002': dict(targets=['documentation'], extra_sources=[])})
    body = dict(cases=[dict(id='fixture', sources=source.model_dump(mode='json'), parentTree={'not': 'an answer'})])
    src = tmp_path / 'sources.json'
    _write(src, {**body, 'manifestDigest': sha256_json(body)})
    model = dict(model='qwen3.5:9b', digest='test-only')
    monkeypatch.setattr(pilot.OllamaAnchoredAuthorAdapter, 'preflight', lambda self: model)
    root = tmp_path / 'run'
    manifest = pilot.freeze(src, root)
    assert 'parentTree' not in manifest['cases'][0]
    calls = []

    def post(self, url, **kwargs):
        wire = kwargs['json']
        phase = 'mapping' if 'selections' in wire['format']['properties'] else 'flow'
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
        pilot.phase(root / 'fixture/flow', source, None, manifest['cases'][0]['flowRequest'], model)
    pilot.run(root)
    pilot.run(root)
    r = pilot.report(root)
    count = 1 if mode in ['bad-flow', 'transport'] else 2
    assert len(calls) == r['modelCalls'] == count
    assert r['inputTokens'] == (0 if mode == 'transport' else count * 11)
    assert r['tokenCountsComplete'] == (mode != 'transport')
    assert r['mappingQualified'] == int(mode in ['valid', 'resume'])
    assert r['sourceReviewed'] == r['reviewSupportedInactiveFlows'] == r['runtimeExecutions'] == 0
    (root / 'fixture/flow/status.json').write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        pilot.run(root)
    assert len(calls) == count
