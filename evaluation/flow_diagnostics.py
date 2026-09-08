"""Offline layered diagnostics beside frozen translators; never repairs or executes."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from jsonschema import Draft202012Validator
from pydantic import ValidationError

from evaluation.flow_canonical_mapping import (
    CanonicalMapping, OBJECTIVE_KINDS, assess_canonical, compile_canonical, schema, target_catalog,
)
from evaluation.flow_clause_mapping import clauses
from evaluation.flow_mapping import execution_projection
from evaluation.flow_translation import FlowSources
from evaluation.flow_tree import FlowTree, assess_tree, compile_report
from evaluation.flow_tree_authoring import digest_file
from evaluation.read_l05_review import ReadL05Review
from network_runtime.contracts import sha256_json

PROTOCOL = 'offline-flow-diagnostics/v1'
STAGES = ('source', 'representation', 'parent_semantic_review', 'mapping', 'lowering_consistency', 'semantic_review', 'runtime')


def pointer(parts):
    return ''.join('/' + str(p).replace('~', '~0').replace('/', '~1') for p in parts)


def signed(body):
    return {**body, 'reportDigest': sha256_json(body)}


def _execution(flow):
    # Exclude only known provenance prose, never arguments, outcomes, guards or edges.
    raw = {k: v for k, v in flow.items() if k != 'purpose'}
    raw['nodes'] = [{k: v for k, v in n.items() if not (n['kind'] == 'end' and k == 'explanation')}
                    for n in flow['nodes']]
    return raw


def diagnose(source_raw, tree_raw, mapping_raw, *, review=None, tree_review=None, witness=None):
    """Aggregate independently checkable defects without rescore or semantic auto-accept.

    Inputs are inert JSON. Candidate atoms are NOT independent source obligations.
    A supplied alternative witness is assisted evidence, never a replacement answer.
    """
    findings, candidates, nodes, catalog = [], [], [], {}
    stages = {s: dict(status='not_evaluated', reason='Prerequisite unavailable or evidence not supplied.') for s in STAGES}
    stages['runtime']['reason'] = 'Offline diagnostics never invokes Runtime, tools, scripts or models.'
    body = dict(protocol=PROTOCOL,
        diagnosticImplementationDigest=digest_file(Path(__file__)),
        inputDigests={k: sha256_json(v) for k, v in dict(sources=source_raw, tree=tree_raw, mapping=mapping_raw).items()},
        stages=stages, findings=findings, candidateRequirements=candidates, nodeTrace=nodes,
        sourceClauses=[], sourceLines=[], declaredIssues=[], sourceReview=None, parentSourceReview=None,
        semanticAccuracy=None, semanticLossRate=None, confidenceProbability=None,
        independentObligationCount=None, runtimeAuthorityGranted=False,
        modelCalls=0, runtimeExecutions=0, writes=0,
        boundary='Mechanical defects and supplied review evidence only; not automatic causal attribution, Gold, admission or a success probability.')

    def add(code, stage, location, message, *, related=(), sources=(), severity='error',
            basis='mechanically_observed', action='Inspect the indicated input; do not change frozen evidence.'):
        item = dict(code=code, stage=stage, pointer=location, relatedPointers=list(related),
            sourceClauseIds=list(sources), severity=severity, evidenceBasis=basis,
            explanation=message, suggestedAction=action)
        item['findingId'] = sha256_json(item)
        findings.append(item)
        return item['findingId']

    def fail(stage, error):
        stages[stage] = dict(status='failed', reason=str(error))
        if isinstance(error, ValidationError):
            for row in error.errors(include_url=False, include_input=False):
                add(stage + '_schema_invalid', stage, pointer(row['loc']), row['msg'])
        else:
            add(stage + '_qualification_rejected', stage, '', str(error),
                action='Use a separately source-reviewed witness to distinguish unsupported representation from a bad proposal; this error alone is not a schema-gap proof.')

    def finish():
        if witness is not None:
            body['witness'] = _witness(source_raw, tree_raw, witness)
        body['earliestFailedStage'] = next((s for s in STAGES if stages[s]['status'] == 'failed'), None)
        body['counts'] = dict(byCode=dict(sorted(Counter(f['code'] for f in findings).items())),
            candidateRequirements=len(candidates), sourceClauses=len(catalog), nodes=len(nodes),
            nodesWithEligibleEvidence=sum(bool(n['eligibleCandidatePointers']) for n in nodes),
            unresolvedCandidates=sum('unresolved' in c['targets'] for c in candidates))
        body['sourceClauses'] = [dict(clauseId=k, **v,
            candidatePointers=[c['pointer'] for c in candidates if c['clauseId'] == k],
            semanticCoverage='not_established') for k, v in catalog.items()]
        return signed(body)

    def review_findings(packet, supplied, stage):
        claims = {c['claimId']: c for c in packet['claims']}
        for judgment in supplied['assessment']['claims']:
            if judgment['verdict'] != 'supported':
                claim = claims[judgment['claim_id']]
                add('source_review_' + judgment['verdict'], stage, claim.get('treePointer', claim['pointer']), judgment['rationale'],
                    related=[p for p in (claim.get('l05Pointer'), claim.get('l0Pointer')) if p],
                    sources=judgment['source_span_ids'], basis='supplied_reviewer_judgment', action=judgment['suggested_revision'])

    try:
        source = FlowSources.model_validate(source_raw)
        catalog = clauses(source)
    except ValueError as error:
        fail('source', error)
        return finish()
    stages['source'] = dict(status='mechanically_valid', reason='Exact source retained; decomposition completeness is unreviewed.')
    offset = 0
    for index, line in enumerate(source.source_text.splitlines(keepends=True), 1):
        body['sourceLines'].append(dict(sourceId=f's{index:04d}', start=offset, end=offset + len(line), exactQuote=line))
        offset += len(line)
    try:
        tree = FlowTree.model_validate(tree_raw)
        parent = compile_report(source, tree)
        targets = target_catalog(source, tree)
    except ValueError as error:
        fail('representation', error)
        return finish()
    stages['representation'] = dict(status='structurally_qualified', reason='This candidate compiles under the actual host contract; fidelity and general expressiveness remain unproven.')
    body['parentCompilationDigest'] = parent['reportDigest']
    body['parentFlowDigest'] = parent['flowDigest']
    if tree_review is not None:
        body['suppliedTreeReviewDigest'] = sha256_json(tree_review)
        try:
            checked = ReadL05Review.model_validate(tree_review)
            body['parentSourceReview'] = assess_tree(source, tree, checked)
            stages['parent_semantic_review'] = dict(
                status='review_supported_not_proof' if all(j.verdict == 'supported' for j in checked.assessment.claims) else 'failed',
                reason='Optional digest-bound first-pass review is independent of second-pass compilation. Reviewer judgment, not semantic Gold.')
            review_findings(parent['reviewInput'], tree_review, 'parent_semantic_review')
        except ValueError as error:
            fail('parent_semantic_review', error)
    for index, issue in enumerate(tree.issues):
        stage = {'source_ambiguity': 'source', 'unsupported_control_flow': 'representation',
                 'missing_host_capability': 'host_context'}[issue.kind]
        body['declaredIssues'].append(issue.model_dump(mode='json'))
        add('declared_' + issue.kind, stage, f'/issues/{index}', issue.question,
            sources=[k for k, c in catalog.items() if c['sourceId'] == issue.source_id],
            severity='warning', basis='model_reported_not_independently_verified',
            action='Inspect source and host contract independently. A correctly represented stop does not implement the unavailable business capability.')

    validator = Draft202012Validator(schema(source, tree))
    errors = sorted(validator.iter_errors(mapping_raw), key=lambda e: (pointer(e.absolute_path), e.message))
    for error in errors:
        add('mapping_schema_invalid', 'mapping', pointer(error.absolute_path), error.message,
            related=[pointer(error.absolute_schema_path)])
    # Keep inspecting valid-shaped fragments after unrelated malformed fragments.
    mapping = mapping_raw if isinstance(mapping_raw, dict) else {}
    selections = mapping.get('selections', {})
    selections = selections if isinstance(selections, dict) else {}
    objective = mapping.get('objective', [])
    objective = objective if isinstance(objective, list) else []
    eligible, declared = {}, {}
    for anchor, clause in catalog.items():
        rows = selections.get(anchor, [])
        if not isinstance(rows, list):
            rows = []
        if not rows:
            add('source_clause_unmapped', 'mapping', pointer(['selections', anchor]),
                'No candidate requirement for this clause; not a quantified semantic-loss rate.', sources=[anchor])
        seen = set()
        for index, item in enumerate(rows):
            path = pointer(['selections', anchor, index])
            if not isinstance(item, dict):
                continue
            quote = item.get('exact_quote')
            quote = quote if isinstance(quote, str) else ''
            start = clause['exactQuote'].find(quote)
            exact = bool(quote.strip()) and start >= 0 and clause['exactQuote'].find(quote, start + 1) < 0
            if not exact:
                add('quote_not_unique_exact', 'mapping', path + '/exact_quote',
                    'Quote must be a unique nonblank substring of its owning clause.', sources=[anchor])
            refs = item.get('targets', [])
            refs = refs if isinstance(refs, list) else []
            refs = [r for r in refs if isinstance(r, str)]
            extras = item.get('extra_sources', [])
            extras = extras if isinstance(extras, list) else []
            own_extra = anchor in extras
            if own_extra:
                add('own_source_repeated', 'mapping', path + '/extra_sources', 'Owning clause is compiler supplied.', sources=[anchor])
            duplicate = sha256_json(item) in seen
            if duplicate:
                add('duplicate_requirement', 'mapping', path, 'Duplicate candidate requirement.', sources=[anchor])
            seen.add(sha256_json(item))
            # Validate this row in isolation so unrelated bad rows cannot hide good evidence.
            row_schema = dict(validator.schema['properties']['selections']['properties'][anchor]['items'],
                              **{'$defs': validator.schema['$defs']})
            valid_row = Draft202012Validator(row_schema).is_valid(item) and exact and not own_extra and not duplicate
            for ref in refs:
                if ref in targets and 'pointer' in targets[ref]:
                    declared.setdefault(ref, []).append(path)
                    if valid_row:
                        eligible.setdefault(ref, []).append(path)
                if ref not in targets:
                    add('unknown_target', 'mapping', path + '/targets', f'Unknown target {ref}.', sources=[anchor])
                elif item.get('kind') not in targets[ref]['allowedKinds']:
                    add('incompatible_target_kind', 'mapping', path + '/kind',
                        'Declared role is incompatible with this actual target.', related=[ref], sources=[anchor])
            if 'unresolved' in refs:
                add('unresolved_requirement', 'mapping', path, 'Candidate explicitly unresolved; cause needs source/host review.',
                    sources=[anchor], severity='warning', basis='model_reported_not_independently_verified',
                    action='Distinguish genuine unavailable capability, absent representation, and avoidable mapping omission; never automatically fill or discard it.')
            span = None if not exact else dict(start=clause['start'] + start, end=clause['start'] + start + len(quote), exactQuote=quote)
            candidates.append(dict(candidateId=sha256_json(dict(input=body['inputDigests']['mapping'], pointer=path)),
                sourceSpanId=None if span is None else sha256_json(dict(source=body['inputDigests']['sources'], **span)),
                pointer=path, clauseId=anchor, sourceSpan=span, kind=item.get('kind'), targets=refs,
                mechanicalEvidenceEligible=valid_row, semanticSupport='not_established', executionStatus='not_evaluated'))
        if anchor in objective and rows and not any(isinstance(r, dict) and isinstance(r.get('kind'), str) and r['kind'] in OBJECTIVE_KINDS for r in rows):
            add('objective_kind_conflict', 'mapping', '/objective',
                'Objective clause has no compatible declared intent/stop/handoff type; this establishes internal inconsistency, not which classification is semantically correct.',
                related=[pointer(['selections', anchor])], sources=[anchor],
                action='Review the source intent once, then link objective to reviewed evidence; do not guess whether the objective or the classification should change.')
    for origin in parent['origins']:
        ref = 'node:' + origin['treePointer']
        node = targets[ref]['node']
        evidence = eligible.get(ref, [])
        if not evidence:
            add('node_evidence_missing', 'mapping', origin['treePointer'],
                'Node exists and lowers, but has no mechanically eligible source mapping. This is not proof that its operation was lost.',
                related=[origin['l0Pointer'], *declared.get(ref, [])],
                sources=[k for k, c in catalog.items() if c['sourceId'] == node['source_id']],
                action='Inspect the original clause and existing node. Supply faithful evidence or flag an unsupported/invented node; never auto-justify the node.')
        nodes.append(dict(**origin, target=ref, node=node, l0Node=parent['flow']['nodes'][origin['nodeIndex']],
            allowedKinds=targets[ref]['allowedKinds'], parentSourceId=node['source_id'], parentCitationTrusted=False,
            declaredCandidatePointers=declared.get(ref, []), eligibleCandidatePointers=evidence,
            mappingStatus='mechanically_linked_semantics_unreviewed' if evidence else 'missing_eligible_evidence',
            executionStatus='not_evaluated'))
    mapping_errors = [f['findingId'] for f in findings if f['stage'] == 'mapping' and f['severity'] == 'error']
    compiled = None
    if mapping_errors:
        stages['mapping'] = dict(status='failed', findingIds=mapping_errors, reason='Independent checks aggregate defects; frozen compiler remains unchanged.')
    else:
        try:
            compiled = compile_canonical(source, tree, CanonicalMapping.model_validate(mapping_raw))
            stages['mapping'] = dict(status='compiled_pending_review', reason='Mechanical compilation is not semantic acceptance.')
        except ValueError as error:
            fail('mapping', error)
    if compiled:
        consistent = _execution(parent['flow']) == _execution(compiled['flow'])
        body['mappedCompilationDigest'] = compiled['reportDigest']
        stages['lowering_consistency'] = dict(status='execution_projection_equal' if consistent else 'failed',
            reason='Compares first-pass and mapped L0, excluding only purpose and terminal explanation. Shared compiler defects and source fidelity require separate tests.')
        if not consistent:
            add('lowering_execution_changed', 'lowering_consistency', '/flow', 'Mapping changed executable projection of the compiled parent.')
        if review is not None:
            try:
                result = assess_canonical(source, tree, CanonicalMapping.model_validate(mapping_raw), ReadL05Review.model_validate(review))
                body['sourceReview'] = result
                stages['semantic_review'] = dict(status='review_supported_not_proof' if result['representationReviewSupported'] else 'failed',
                    reason='Digest-bound supplied reviewer judgments; test fixtures/AI simulation are not independent Gold.')
                review_findings(compiled['reviewInput'], review, 'semantic_review')
            except ValueError as error:
                fail('semantic_review', error)
    if review is not None:
        body['suppliedReviewDigest'] = sha256_json(review)
    return finish()


def _witness(source_raw, tree_raw, witness):
    """Test a supplied alternative, keeping its assisted role and semantics limitations."""
    if not isinstance(witness, dict) or witness.get('sourcesDigest') != sha256_json(source_raw):
        return dict(status='rejected', reason='Witness sources/host context digest mismatch.', witnessDigest=sha256_json(witness))
    alternate = diagnose(source_raw, witness.get('tree'), witness.get('mapping'), review=witness.get('review'), tree_review=witness.get('tree_review'))
    same_tree = False
    try:
        same_tree = execution_projection(FlowTree.model_validate(tree_raw)) == execution_projection(FlowTree.model_validate(witness.get('tree')))
    except ValueError:
        pass
    return dict(status='assisted_diagnostic_only', witnessDigest=sha256_json(witness),
        sameExecutionTree=same_tree, mechanicallyConstructible=alternate['stages']['mapping']['status'] == 'compiled_pending_review',
        diagnostic=alternate, replacesOriginal=False, countsAsFirstPassSuccess=False,
        conclusion='Constructibility is bounded structural evidence, not proof of source fidelity or automatic root cause. Review the witness independently.')


def diagnose_batch(root):
    # Replay validates frozen implementation, environment, request and raw checkpoints.
    # Importing report never calls run/preflight or executes source content.
    from evaluation.flow_canonical_pilot import load, report
    original = report(root)
    manifest = load(root)
    results = []
    for case in manifest['cases']:
        folder = root / case['id']
        def read(name):
            path = folder / name
            return json.loads(path.read_text()) if path.exists() else None
        results.append(dict(case=case['id'], diagnosis=diagnose(case['sources'], read('flow/tree.json'), read('mapping/mapping.json'))))
    body = dict(protocol=PROTOCOL, evidenceRole='posthoc_mechanical_diagnostics_known_development_not_rescore',
        originalManifestDigest=manifest['manifestDigest'], originalReportDigest=original['reportDigest'],
        originalFlowQualified=original['flowQualified'], originalMappingQualified=original['mappingQualified'],
        cases=results, modelCalls=0, runtimeExecutions=0, writes=0, runtimeAuthorityGranted=False,
        independentObligationCount=None, semanticAccuracy=None, semanticLossRate=None,
        findingsByCode=dict(sorted(Counter(f['code'] for r in results for f in r['diagnosis']['findings']).items())))
    return signed(body)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    one = sub.add_parser('case')
    for name in ('sources', 'tree', 'mapping'):
        one.add_argument(name, type=Path)
    one.add_argument('--review', type=Path)
    one.add_argument('--tree-review', type=Path)
    one.add_argument('--witness', type=Path)
    batch = sub.add_parser('batch')
    batch.add_argument('root', type=Path)
    for command in (one, batch):
        command.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('output exists; preserve previous evidence')
    if args.command == 'batch' and args.output.resolve().is_relative_to(args.root.resolve()):
        raise ValueError('diagnostic output must be outside frozen batch')
    if args.command == 'batch':
        result = diagnose_batch(args.root)
    else:
        def read(path):
            return json.loads(path.read_text()) if path else None
        result = diagnose(read(args.sources), read(args.tree), read(args.mapping),
                          review=read(args.review), tree_review=read(args.tree_review), witness=read(args.witness))
    with args.output.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
