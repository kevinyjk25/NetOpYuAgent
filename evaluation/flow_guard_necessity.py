"""Separate necessary guards from feasibility of the surviving execution path.

If execution with F=false is forbidden, any permitted execution implies F=true.
This implication does not require proving that a permitted execution exists.
The premise is still an unverified model/source judgment, never host authority.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from evaluation.flow_behavior import _seal
from evaluation.flow_guard_binding import bind_guards, slots_for
from evaluation.flow_guard_counterfactual import answer_schema
from evaluation.flow_source_selection import spans
from evaluation.flow_translation import FlowSources, _write
from evaluation.flow_tree import FlowTree
from evaluation.flow_tree_authoring import digest_file
from network_runtime.contracts import sha256_json

PROTOCOL = "necessary-guards-with-separate-feasibility/v1"


def bind_necessity(sources: FlowSources, tree: FlowTree, answers: dict) -> dict:
    """Produce an INACTIVE candidate, retaining all uncertainty and both answers."""
    slots = slots_for(sources, tree)
    source_spans = spans(sources)
    Draft202012Validator(answer_schema(sources, slots)).validate(answers)
    decisions, rows = {}, []
    for slot in slots:
        row = answers[slot["id"]]
        no, yes = row["if_false"], row["if_true"]
        pair = (no["status"], yes["status"])
        if pair == ("forbidden", "forbidden"):
            decision = dict(decision="unresolved", source_id=no["source_id"])
            rule, feasibility = "both_values_forbidden_do_not_construct_a_live_path", "source_judged_unreachable"
        elif no["status"] == "forbidden":
            decision = dict(decision="require_true", source_id=no["source_id"], on_failure=no["on_failure"])
            rule = "forbidden_false_implies_execution_requires_true"
            feasibility = "model_reports_possible_not_proven" if yes["status"] == "possible" else "unknown"
        elif yes["status"] == "forbidden":
            decision = dict(decision="require_false", source_id=yes["source_id"], on_failure=yes["on_failure"])
            rule = "forbidden_true_implies_execution_requires_false"
            feasibility = "model_reports_possible_not_proven" if no["status"] == "possible" else "unknown"
        elif pair == ("possible", "possible"):
            decision = dict(decision="not_individually_required", source_id=no["source_id"])
            rule, feasibility = "both_values_possible_no_individual_requirement", "model_reports_possible_not_proven"
        else:
            uncertain = no if no["status"] == "unknown" else yes
            decision = dict(decision="unresolved", source_id=uncertain["source_id"])
            rule, feasibility = "no_forbidden_value_and_incomplete_judgment", "unknown"
        decisions[slot["id"]] = decision
        rows.append(dict(slot=slot, originalAnswers=row, decision=decision, inferenceRule=rule,
            pathFeasibility=feasibility, sourceQuotes={key: source_spans[sample["source_id"]] for key, sample in row.items()}))
    binding = bind_guards(sources, tree, decisions)
    unknowns = [dict(slotId=r["slot"]["id"], fact=r["slot"]["reference"],
        targetPointer=r["slot"]["targetPointer"], pathFeasibility=r["pathFeasibility"],
        originalAnswers=r["originalAnswers"]) for r in rows if r["pathFeasibility"] != "model_reports_possible_not_proven"]
    return _seal(dict(protocol=PROTOCOL, sourcesDigest=sha256_json(sources.model_dump(mode="json")),
        parentTreeDigest=sha256_json(tree.model_dump(mode="json")), answersDigest=sha256_json(answers),
        derivations=rows, binding=binding, guardCandidateGenerated="tree" in binding,
        retainedUncertainty=unknowns, pathFeasibilityProven=False, sufficiencyProven=False,
        semanticAlignmentProven=False, runtimeAuthorityGranted=False, fullSourceReview="required_not_run",
        activationEligibility="not_established", status="inactive_necessary_guard_candidate" if "tree" in binding else "unresolved_not_executable",
        boundary="Necessary-condition synthesis only; neither positive-path unknowns nor model premise uncertainty is removed."))


def author_necessity(sources: FlowSources, tree: FlowTree, root: Path, *, max_new_calls: int = 0) -> dict:
    """Reusable source -> model questions -> inactive guard synthesis, no Oracle."""
    from evaluation.flow_guard_counterfactual import author

    inputs = dict(sourcesDigest=sha256_json(sources.model_dump(mode="json")),
        treeDigest=sha256_json(tree.model_dump(mode="json")), implementationDigest=digest_file(Path(__file__)))
    if root.exists():
        if json.loads((root / "inputs.json").read_text()) != inputs:
            raise ValueError("necessary-guard source/tree/implementation drift")
    else:
        if type(max_new_calls) is not int or max_new_calls < 1:
            raise ValueError("explicit one-call budget required")
        root.mkdir(parents=True)
        _write(root / "inputs.json", inputs)
    generation = author(sources, tree, root / "generation", max_new_calls=max_new_calls)
    result = dict(generationResult=generation["result"],
        synthesis=bind_necessity(sources, tree, generation["answers.json"]) if "answers.json" in generation else None)
    target = root / "synthesis.json"
    if target.exists():
        if json.loads(target.read_text()) != result:
            raise ValueError("necessary-guard derivation drift")
    else:
        _write(target, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bind", "author"))
    parser.add_argument("sources", type=Path)
    parser.add_argument("tree", type=Path)
    parser.add_argument("--answers", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-calls", type=int, default=0)
    args = parser.parse_args()
    sources, tree = FlowSources.model_validate_json(args.sources.read_text()), FlowTree.model_validate_json(args.tree.read_text())
    if args.command == "author":
        result = author_necessity(sources, tree, args.output, max_new_calls=args.max_new_calls)
        print(json.dumps(dict(generationResult=result["generationResult"],
            synthesisStatus=(result["synthesis"] or {}).get("status"))))
    elif args.answers:
        _write(args.output, bind_necessity(sources, tree, json.loads(args.answers.read_text())))
    else:
        parser.error("bind requires --answers")


if __name__ == "__main__":
    main()
