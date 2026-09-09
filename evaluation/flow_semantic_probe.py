"""Bounded source->constructor->condition-expression probe; no production run.

Freeze at most 12 explicit, manually reviewed cases. The generation APIs accept
only FlowSources and generated trees, never private references or observations.
This is not independent Gold, a sealed public cohort or an admission gate.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from evaluation.flow_behavior import BehaviorSuite, _seal, check_behavior, validate_suite
from evaluation.flow_checkpoint import environment, implementation
from evaluation.flow_contract_authoring import author_candidate
from evaluation.flow_condition_expression import author as author_expression
from evaluation.flow_joint_conditions import region
from evaluation.flow_translation import FlowSources, _write
from evaluation.flow_tree import FlowTree
from evaluation.translation_case_authoring import OllamaAnchoredAuthorAdapter

PROTOCOL = "source-semantic-development-probe/v2"


def fingerprint():
    return implementation("evaluation/flow_semantic_probe.py", "evaluation/flow_condition_expression.py",
        "evaluation/flow_joint_conditions.py", "evaluation/flow_joint_lowering.py")


def validate_cases(cases):
    if not isinstance(cases, list) or not 1 <= len(cases) <= 12:
        raise ValueError("small probe requires 1..12 explicit cases; bulk validation needs a separate decision")
    seen = set()
    for case in cases:
        key = case["id"]
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", key) or key in seen:
            raise ValueError("invalid/duplicate case ID")
        seen.add(key)
        source = FlowSources.model_validate(case["sources"])
        suite = BehaviorSuite.model_validate(case["suite"])
        validate_suite(source, suite)
        witness = check_behavior(source, FlowTree.model_validate(case["reference"]), suite)
        if witness["behavior"] != "matched_finite_oracle":
            raise ValueError("reference is not feasible; fix the fixture before freezing or calling a model")


def freeze(cases_file, root, *, evidence_role="fresh_source_development"):
    cases = json.loads(cases_file.read_text())
    validate_cases(cases)
    if evidence_role not in ("fresh_source_development", "known_case_development_revision"):
        raise ValueError("probe roles cannot claim independent/heldout evidence")
    manifest = _seal(dict(protocol=PROTOCOL, cases=cases, implementation=fingerprint(), environment=environment(),
        model=OllamaAnchoredAuthorAdapter().preflight(), guardMode="source_expression_explicit_inactive_revision_not_acceptance",
        attemptsPerPhase=1, evidenceRole=evidence_role + "_same_assistant_not_holdout",
        publicSkills=0, runtimeAuthorityGranted=False))
    root.mkdir(parents=True, exist_ok=False)
    _write(root / "manifest.json", manifest)
    return manifest


def load(root):
    m = json.loads((root / "manifest.json").read_text())
    if (m != _seal({k: v for k, v in m.items() if k != "reportDigest"}) or m["protocol"] != PROTOCOL
            or m["implementation"] != fingerprint() or m["environment"] != environment()):
        raise ValueError("frozen semantic probe drift")
    validate_cases(m["cases"])
    return m


def run(root, max_new_calls):
    if type(max_new_calls) is not int or max_new_calls < 0:
        raise ValueError("invalid explicit call budget")
    m = load(root)
    remaining = max_new_calls
    for case in m["cases"]:
        source = FlowSources.model_validate(case["sources"])
        generation_path = root / case["id"] / "generation"
        is_new = not generation_path.exists()
        if is_new and not remaining:
            return
        if is_new and OllamaAnchoredAuthorAdapter().preflight() != m["model"]:
            raise ValueError("model artifact drift")
        generation = author_candidate(source, generation_path, max_new_calls=int(is_new))
        remaining -= int(is_new)
        if generation["result"]["status"] != "text_received":
            raise ValueError("recorded transport/envelope failure; no retry or automatic continuation")
        if "candidate.json" in generation:
            tree = FlowTree.model_validate(generation["candidate.json"])
            try:
                region(source, tree)
            except ValueError as error:
                print(json.dumps(dict(case=case["id"], conditionScope="not_evaluated", reason=str(error))), flush=True)
            else:
                path = root / case["id"] / "condition"
                is_new = not path.exists()
                if is_new and not remaining:
                    return
                if is_new and OllamaAnchoredAuthorAdapter().preflight() != m["model"]:
                    raise ValueError("model artifact drift")
                condition = author_expression(source, tree, path, max_new_calls=int(is_new))
                remaining -= int(is_new)
                if condition["result"]["status"] != "text_received":
                    raise ValueError("recorded condition transport failure; do not retry")
        print(json.dumps(dict(case=case["id"], generation=generation["result"], remainingNewCalls=remaining)), flush=True)


def report(root):
    m, rows, costs = load(root), [], []
    for case in m["cases"]:
        source, suite = FlowSources.model_validate(case["sources"]), BehaviorSuite.model_validate(case["suite"])
        row = dict(id=case["id"], domain=case.get("domain"), scope=suite.scope, totalScenarios=len(suite.scenarios), status="not_run")
        path = root / case["id"] / "generation"
        if path.exists():
            generated = author_candidate(source, path, max_new_calls=0)
            costs.append(generated["result"])
            row.update(status=generated["result"]["candidateStatus"] if generated["result"]["status"] == "text_received"
                else generated["result"]["status"], generation=generated["result"])
            if "candidate.json" in generated:
                tree = FlowTree.model_validate(generated["candidate.json"])
                row["firstBehavior"] = check_behavior(source, tree, suite)
                final = None
                try:
                    region(source, tree)
                except ValueError as error:
                    row.update(status="outside_condition_expression_scope", conditionScopeReason=str(error))
                    if (len(tree.steps) == 1 and tree.steps[0].kind == "end"
                            and tree.steps[0].outcome == "unsupported" and tree.issues):
                        final = tree  # Declared capability stop, never useful read completion.
                else:
                    condition_path = root / case["id"] / "condition"
                    row["status"] = "condition_not_run"
                    if condition_path.exists():
                        condition = author_expression(source, tree, condition_path, max_new_calls=0)
                        costs.append(condition["result"])
                        row["condition"] = condition
                        synthesis = condition.get("derivation.json") or {}
                        row["status"] = synthesis.get("status", "condition_generation_failed")
                        if synthesis.get("status") == "unchanged_finite_agreement":
                            final = tree
                        elif "tree" in synthesis.get("revision", {}):
                            final = FlowTree.model_validate(synthesis["revision"]["tree"])
                if final is not None:
                    row["finalBehavior"] = check_behavior(source, final, suite)
                    row["status"] = "completed_inactive_research_candidate"
        rows.append(row)
    metrics = {}
    for phase in ("firstBehavior", "finalBehavior"):
        metrics[phase] = dict(matchedCases=sum((r.get(phase) or {}).get("behavior") == "matched_finite_oracle" for r in rows),
            passedScenarios=sum((r.get(phase) or {}).get("passed", 0) for r in rows),
            executableMatched=sum(r["scope"] == "executable_fragment" and
                (r.get(phase) or {}).get("behavior") == "matched_finite_oracle" for r in rows),
            safeStopMatched=sum(r["scope"] == "safe_partial_stop" and
                (r.get(phase) or {}).get("behavior") == "matched_finite_oracle" for r in rows))
    times = sorted(c["latencyMs"] for c in costs)
    return _seal(dict(protocol=PROTOCOL, manifestDigest=m["reportDigest"], rows=rows, metrics=metrics,
        totalCases=len(rows), totalScenarios=sum(r["totalScenarios"] for r in rows), modelCalls=len(costs),
        inputTokens=sum(c.get("inputTokens") or 0 for c in costs), outputTokens=sum(c.get("outputTokens") or 0 for c in costs),
        postTotalMs=sum(times), requestP50Ms=times[math.ceil(len(times) * .5) - 1] if times else None,
        requestP95Ms=times[math.ceil(len(times) * .95) - 1] if times else None,
        semanticAccuracy=None, wholeSkillTranslations=0, publicSkills=0, runtimeAuthorityGranted=False,
        fullSourceReview="required_not_run", evidenceRole=m["evidenceRole"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("freeze", "run", "report"))
    p.add_argument("root", type=Path)
    p.add_argument("--cases", type=Path)
    p.add_argument("--max-new-calls", type=int, default=0)
    p.add_argument("--evidence-role", choices=("fresh_source_development", "known_case_development_revision"), default="fresh_source_development")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.command == "freeze" and a.cases:
        print(freeze(a.cases, a.root, evidence_role=a.evidence_role)["reportDigest"])
    elif a.command == "run":
        run(a.root, a.max_new_calls)
    elif a.command == "report" and a.output:
        _write(a.output, report(a.root))
    else:
        p.error("freeze needs --cases; report needs --output")


if __name__ == "__main__":
    main()
