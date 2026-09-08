"""Fixed-parent terminal mapping probes, not fresh two-pass Skill translation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx
from jsonschema import ValidationError as SchemaError

from evaluation.flow_canonical_mapping import CanonicalMapping, compile_canonical, request
from evaluation.flow_responsibility_pilot import implementation as prior_implementation
from evaluation.flow_translation import FlowSources, _write, local_sources
from evaluation.flow_tree import FlowTree
from evaluation.flow_tree_authoring import ROOT, digest_file, receipt, verify_receipt
from evaluation.flow_tree_bounded_pilot import environment
from evaluation.translation_case_authoring import OllamaAnchoredAuthorAdapter
from network_runtime.contracts import sha256_json

PROTOCOL = 'fixed-parent-canonical-terminal-canary/v1'


def cases():
    rows = []
    for outcome in ['read_path_completed', 'needs_l1', 'unsupported']:
        source = local_sources().model_copy(update={'source_text':
            f'Return {outcome} without invoking any model, tool, script or write.',
            'source_path': 'synthetic-terminal-protocol-canary/' + outcome})
        tree = FlowTree.model_validate(dict(business_source_ids=['s0001'],
            steps=[dict(kind='end', source_id='s0001', outcome=outcome)], issues=[]))
        rows.append(dict(id=outcome, sources=source.model_dump(mode='json'), tree=tree.model_dump(mode='json'), wire=request(source, tree)))
    return rows


def implementation():
    return {name: digest_file(ROOT / name) for name in sorted(set(prior_implementation()) |
        {'evaluation/flow_canonical_mapping.py', 'evaluation/flow_canonical_canary.py'})}


def freeze(root):
    body = dict(protocol=PROTOCOL, cases=cases(), implementation=implementation(), environment=environment(),
        model=OllamaAnchoredAuthorAdapter().preflight(), attemptsPerCase=1,
        evidenceRole='fixed_hand_authored_parent_decoder_canary_not_fresh_translation')
    root.mkdir(parents=True, exist_ok=False)
    manifest = {**body, 'manifestDigest': sha256_json(body)}
    _write(root / 'manifest.json', manifest)
    return manifest


def load(root):
    m = json.loads((root / 'manifest.json').read_text())
    if (m['manifestDigest'] != sha256_json({k: v for k, v in m.items() if k != 'manifestDigest'}) or
        m['protocol'] != PROTOCOL or m['implementation'] != implementation() or m['environment'] != environment() or m['cases'] != cases()):
        raise ValueError('frozen canary drift')
    return m


def derive(case, envelope):
    cost = dict(latencyMs=envelope['latencyMs'], inputTokens=None, outputTokens=None)
    if envelope['httpStatus'] is None:
        return {}, dict(status='transport_error', error=envelope['body'], **cost)
    files = {}
    try:
        if envelope['httpStatus'] != 200:
            raise ValueError('non-200 model response')
        body = json.loads(envelope['body'])
        cost.update(inputTokens=body.get('prompt_eval_count'), outputTokens=body.get('eval_count'), doneReason=body.get('done_reason'))
        proposal = CanonicalMapping.model_validate_json(body['message']['content'])
        files['mapping.json'] = proposal.model_dump(mode='json')
        compiled = compile_canonical(FlowSources.model_validate(case['sources']), FlowTree.model_validate(case['tree']), proposal)
        files['compilation.json'] = compiled
        status = dict(status='compiled_pending_source_review', inputDigest=compiled['reviewInput']['inputDigest'])
    except (ValueError, KeyError, TypeError, SchemaError) as e:
        status = dict(status='blocked', errorType=type(e).__name__, error=str(e))
    return files, {**status, **cost}


def replay(folder, case, model):
    verify_receipt(folder)
    if json.loads((folder / 'request.json').read_text()) != dict(wireRequest=case['wire'], model=model):
        raise ValueError('canary request drift')
    files, status = derive(case, json.loads((folder / 'response.json').read_text()))
    if (set(receipt(folder)) != {'request.json', 'response.json', 'status.json', *files} or
        json.loads((folder / 'status.json').read_text()) != status or
        any(json.loads((folder / name).read_text()) != content for name, content in files.items())):
        raise ValueError('canary derivation drift')
    return files, status


def run(root):
    m = load(root)
    for case in m['cases']:
        folder = root / case['id']
        if folder.exists():
            replay(folder, case, m['model'])
            continue
        if OllamaAnchoredAuthorAdapter().preflight() != m['model']:
            raise ValueError('canary model drift')
        folder.mkdir()
        _write(folder / 'request.json', dict(wireRequest=case['wire'], model=m['model']))
        start = time.monotonic()
        try:
            with httpx.Client(timeout=360, trust_env=False) as client:
                response = client.post('http://127.0.0.1:11434/api/chat', json=case['wire'])
            envelope = dict(httpStatus=response.status_code, body=response.text, latencyMs=(time.monotonic() - start) * 1000)
        except httpx.HTTPError as e:
            envelope = dict(httpStatus=None, body=f'{type(e).__name__}: {e}', latencyMs=(time.monotonic() - start) * 1000)
        _write(folder / 'response.json', envelope)
        files, status = derive(case, envelope)
        for name, content in files.items():
            _write(folder / name, content)
        _write(folder / 'status.json', status)
        _write(folder / 'receipt.json', receipt(folder))
        print(json.dumps(dict(case=case['id'], **status)), flush=True)


def report(root):
    m = load(root)
    rows = []
    for case in m['cases']:
        files, status = replay(root / case['id'], case, m['model'])
        rows.append(dict(case=case['id'], **status, proposal=files.get('mapping.json'), files=receipt(root / case['id'])))
    body = dict(protocol=PROTOCOL, manifestDigest=m['manifestDigest'], model=m['model'], rows=rows,
        modelCalls=len(rows), qualified=sum(r['status'] == 'compiled_pending_source_review' for r in rows),
        latencyMs=sum(r['latencyMs'] for r in rows), inputTokens=sum(r['inputTokens'] or 0 for r in rows),
        outputTokens=sum(r['outputTokens'] or 0 for r in rows), tokenCountsComplete=all(r['inputTokens'] is not None and r['outputTokens'] is not None for r in rows),
        sourceReviewed=0, runtimeExecutions=0, publicSkills=0, evidenceRole=m['evidenceRole'],
        boundary='Fixed model-visible hand-authored terminal trees, no mapping answer supplied. Decoder/representation probes only, not source fidelity or fresh two-pass accuracy. No retries or execution. Failures included in cost.')
    return {**body, 'reportDigest': sha256_json(body)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['freeze', 'run', 'report'])
    p.add_argument('root', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.command == 'freeze':
        print(freeze(args.root)['manifestDigest'])
    elif args.command == 'run':
        run(args.root)
    elif args.output:
        _write(args.output, report(args.root))
    else:
        p.error('report requires --output')


if __name__ == '__main__':
    main()
