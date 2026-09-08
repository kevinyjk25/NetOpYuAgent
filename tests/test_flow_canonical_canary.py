import json

import httpx
import pytest

from evaluation import flow_canonical_canary as canary


@pytest.mark.parametrize('mode', ['valid', 'wrong-kind', 'transport', 'partial'])
def test_terminal_canary_preserves_failed_calls_and_never_repeats_checkpoints(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(canary.OllamaAnchoredAuthorAdapter, 'preflight', lambda self: dict(model='qwen3.5:9b', digest='fixture'))
    root = tmp_path / 'canary'
    manifest = canary.freeze(root)
    calls = []

    def post(self, url, **kwargs):
        wire = kwargs['json']
        calls.append(wire)
        if mode == 'transport':
            raise httpx.ConnectError('test unavailable')
        p = json.loads(wire['messages'][1]['content'])
        outcome = p['fixedModelFlow']['steps'][0]['outcome']
        kind = {'read_path_completed': 'completion', 'needs_l1': 'handoff', 'unsupported': 'missing_capability_stop'}[outcome]
        answer = dict(objective=['clause-0001'], selections={'clause-0001': [dict(
            exact_quote=p['clauses']['clause-0001']['exactQuote'], kind='input_shape' if mode == 'wrong-kind' else kind,
            targets=['node:/steps/0'], extra_sources=[])]})
        return httpx.Response(200, json=dict(message=dict(content=json.dumps(answer)), prompt_eval_count=11, eval_count=22))

    monkeypatch.setattr(httpx.Client, 'post', post)
    if mode == 'partial':
        (root / manifest['cases'][0]['id']).mkdir()
        with pytest.raises(ValueError):
            canary.run(root)
        assert not calls
        return
    canary.run(root)
    canary.run(root)
    report = canary.report(root)
    assert len(calls) == report['modelCalls'] == 3
    assert report['qualified'] == (3 if mode == 'valid' else 0)
    assert report['inputTokens'] == (0 if mode == 'transport' else 33)
    assert report['outputTokens'] == (0 if mode == 'transport' else 66)
    assert report['tokenCountsComplete'] == (mode != 'transport')
    assert report['sourceReviewed'] == report['runtimeExecutions'] == report['publicSkills'] == 0
    (root / manifest['cases'][0]['id'] / 'status.json').write_text('{}')
    with pytest.raises(ValueError):
        canary.run(root)
    assert len(calls) == 3
