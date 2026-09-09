"""Frozen fresh source->flow->mandatory node evidence pilot; no answer reuse."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx
from jsonschema import ValidationError as SchemaError

from evaluation.flow_node_evidence import EvidenceMapping as ClauseMapping, assess_evidence as assess_clauses, compile_evidence as compile_clauses, projection, request as clause_request
from evaluation.flow_canonical_pilot import implementation as parent_implementation
from evaluation.flow_translation import FlowSources, _write
from evaluation.flow_tree import FlowTree
from evaluation.flow_tree_authoring import ROOT, digest_file, receipt, verify_receipt
from evaluation.flow_tree_bounded_pilot import derive as derive_tree, environment
from evaluation.flow_tree_capabilities import bounded_request
from evaluation.read_l05_review import ReadL05Review
from evaluation.translation_case_authoring import OllamaAnchoredAuthorAdapter
from network_runtime.contracts import sha256_json

PROTOCOL = 'fresh-node-evidence-paired/v1'


def implementation():
    paths = set(parent_implementation()) | {'evaluation/flow_node_evidence.py', 'evaluation/flow_node_evidence_pilot.py'}
    return {path: digest_file(ROOT / path) for path in sorted(paths)}


def freeze(source_manifest: Path, root: Path):
    parent = json.loads(source_manifest.read_text())
    if parent['manifestDigest'] != sha256_json({k: v for k, v in parent.items() if k != 'manifestDigest'}):
        raise ValueError('source manifest digest mismatch')
    rows = []
    for c in parent['cases']:
        key = c['id']
        if Path(key).name != key or key in {'.', '..'} or key in [r['id'] for r in rows]:
            raise ValueError('invalid/duplicate case ID')
        source = FlowSources.model_validate(c['sources'])
        rows.append(dict(id=key, sources=source.model_dump(mode='json'), flowRequest=bounded_request(source)))
    if not rows:
        raise ValueError('empty batch')
    body = dict(protocol=PROTOCOL, sourceManifestDigest=parent['manifestDigest'], cases=rows,
        model=OllamaAnchoredAuthorAdapter().preflight(), implementation=implementation(), environment=environment(),
        attemptsPerPhase=1, evidenceRole='fresh_node_evidence_two_pass_known_development_not_holdout',
        secondPass='compile actual first-pass nodes, then require evidence-or-disagreement slots and residual source duties; no previous trees or answers')
    manifest = {**body, 'manifestDigest': sha256_json(body)}
    root.mkdir(parents=True, exist_ok=False)
    _write(root / 'manifest.json', manifest)
    return manifest


def load(root):
    m = json.loads((root / 'manifest.json').read_text())
    if (m['manifestDigest'] != sha256_json({k: v for k, v in m.items() if k != 'manifestDigest'}) or
            m['protocol'] != PROTOCOL or m['implementation'] != implementation() or m['environment'] != environment()):
        raise ValueError('frozen manifest/implementation/environment drift')
    for c in m['cases']:
        if c['flowRequest'] != bounded_request(FlowSources.model_validate(c['sources'])):
            raise ValueError('frozen flow request drift')
    return m


def derive(source, parent, envelope):
    if envelope['httpStatus'] is None:
        return {}, dict(status='transport_error', latencyMs=envelope['latencyMs'], error=envelope['body'])
    if parent is None:
        return derive_tree(source, envelope)
    files, cost = {}, dict(latencyMs=envelope['latencyMs'])
    try:
        if envelope['httpStatus'] != 200:
            raise ValueError('non-200 model response')
        raw = json.loads(envelope['body'])
        cost.update(inputTokens=raw.get('prompt_eval_count'), outputTokens=raw.get('eval_count'), doneReason=raw.get('done_reason'))
        proposal = ClauseMapping.model_validate_json(raw['message']['content'])
        files['mapping.json'] = proposal.model_dump(mode='json')
        files['diagnostics.json'] = projection(source, parent, proposal)[1]
        compiled = compile_clauses(source, parent, proposal)
        files.update({'compilation.json': compiled, 'review-input.json': compiled['reviewInput']})
        status = dict(status='awaiting_source_review', inputDigest=compiled['reviewInput']['inputDigest'])
    except (ValueError, KeyError, TypeError, SchemaError) as e:
        status = dict(status='blocked', errorType=type(e).__name__, error=str(e))
    return files, {**status, **cost}


def replay(folder, source, parent, wire, model):
    verify_receipt(folder)
    if json.loads((folder / 'request.json').read_text()) != dict(wireRequest=wire, model=model):
        raise ValueError('phase request/model mismatch')
    envelope = json.loads((folder / 'response.json').read_text())
    files, status = derive(source, parent, envelope)
    if (set(receipt(folder)) != {'request.json', 'response.json', 'status.json', *files} or
            json.loads((folder / 'status.json').read_text()) != status or
            any(json.loads((folder / k).read_text()) != v for k, v in files.items())):
        raise ValueError('phase raw derivation mismatch')
    return files, status


def phase(folder, source, parent, wire, model):
    if folder.exists():
        return replay(folder, source, parent, wire, model)
    if OllamaAnchoredAuthorAdapter().preflight() != model:
        raise ValueError('model identity drift')
    folder.mkdir(parents=True)
    _write(folder / 'request.json', dict(wireRequest=wire, model=model))
    start = time.monotonic()
    try:
        with httpx.Client(timeout=360, trust_env=False) as client:
            r = client.post('http://127.0.0.1:11434/api/chat', json=wire)
        envelope = dict(httpStatus=r.status_code, body=r.text, latencyMs=(time.monotonic() - start) * 1000)
    except httpx.HTTPError as e:
        envelope = dict(httpStatus=None, body=f'{type(e).__name__}: {e}', latencyMs=(time.monotonic() - start) * 1000)
    _write(folder / 'response.json', envelope)
    files, status = derive(source, parent, envelope)
    for k, v in files.items():
        _write(folder / k, v)
    _write(folder / 'status.json', status)
    _write(folder / 'receipt.json', receipt(folder))
    print(json.dumps(dict(phase=str(folder), **status)), flush=True)
    return files, status


def run(root):
    m = load(root)
    for c in m['cases']:
        source = FlowSources.model_validate(c['sources'])
        files, status = phase(root / c['id'] / 'flow', source, None, c['flowRequest'], m['model'])
        if status['status'] == 'awaiting_source_review':
            parent = FlowTree.model_validate(files['tree.json'])
            phase(root / c['id'] / 'mapping', source, parent, clause_request(source, parent), m['model'])


def report(root, reviews=None):
    m = load(root)
    rows = []
    for c in m['cases']:
        source = FlowSources.model_validate(c['sources'])
        folder = root / c['id']
        files, status = replay(folder / 'flow', source, None, c['flowRequest'], m['model'])
        row = dict(case=c['id'], flow=status, flowFiles=receipt(folder / 'flow'))
        if status['status'] == 'awaiting_source_review':
            parent = FlowTree.model_validate(files['tree.json'])
            files, status = replay(folder / 'mapping', source, parent, clause_request(source, parent), m['model'])
            row.update(mapping=status, mappingFiles=receipt(folder / 'mapping'), mappingDiagnostics=files.get('diagnostics.json'))
            if status['status'] == 'awaiting_source_review' and reviews:
                review = ReadL05Review.model_validate_json((reviews / (c['id'] + '.json')).read_text())
                row['review'] = assess_clauses(source, parent, ClauseMapping.model_validate(files['mapping.json']), review)
        elif (folder / 'mapping').exists():
            raise ValueError('mapping exists after failed first pass')
        rows.append(row)
    phases = [r[k] for r in rows for k in ['flow', 'mapping'] if k in r]
    body = dict(protocol=PROTOCOL, manifestDigest=m['manifestDigest'], model=m['model'], rows=rows,
        knownFlows=len(rows), publicSkills=0, modelCalls=len(phases),
        flowQualified=sum(r['flow']['status'] == 'awaiting_source_review' for r in rows),
        mappingQualified=sum(r.get('mapping', {}).get('status') == 'awaiting_source_review' for r in rows),
        schemaQualifiedMappings=sum(bool((r.get('mappingDiagnostics') or {}).get('schemaQualified')) for r in rows),
        explicitNodeGaps=sum((r.get('mappingDiagnostics') or {}).get('explicitNodeGaps', 0) for r in rows),
        sourceReviewed=sum('review' in r for r in rows),
        representationReviewSupported=sum(r.get('review', {}).get('representationReviewSupported', False) for r in rows),
        reviewSupportedInactiveFlows=sum(r.get('review', {}).get('status') == 'review_supported_inactive_flow' for r in rows),
        latencyMs=sum(p['latencyMs'] for p in phases), inputTokens=sum(p.get('inputTokens') or 0 for p in phases),
        outputTokens=sum(p.get('outputTokens') or 0 for p in phases),
        tokenCountsComplete=all(p.get('inputTokens') is not None and p.get('outputTokens') is not None for p in phases),
        runtimeExecutions=0, writes=0, boundary='Fresh two-pass known development. Structure/accounting is not source fidelity or activation. Reviews, if supplied, are not automatically independent Gold. Costs include failures/waiting, not preflight/review/tests.')
    return {**body, 'reportDigest': sha256_json(body)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['freeze', 'run', 'report'])
    p.add_argument('root', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--reviews', type=Path)
    a = p.parse_args()
    if a.command != 'run' and a.output is None:
        p.error('freeze/report requires --output')
    if a.command == 'freeze':
        print(freeze(a.root, a.output)['manifestDigest'])
    elif a.command == 'run':
        run(a.root)
    else:
        _write(a.output, report(a.root, a.reviews))


if __name__ == '__main__':
    main()
