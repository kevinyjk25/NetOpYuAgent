"""Explicit, inactive Boolean-region revision from a saved source decision table.

Never automatic acceptance or appended unary guards. Original candidates and
answers remain immutable; every new branch retains the table/source lineage.
Parameters, read identities and completion citations are preserved, not proven.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from evaluation.flow_behavior import _seal
from evaluation.flow_joint_conditions import compare, region
from evaluation.flow_translation import FlowSources, _write
from evaluation.flow_tree import FlowTree, compile_report
from network_runtime.contracts import sha256_json

PROTOCOL = "explicit-joint-condition-revision/v1"


def propose(sources: FlowSources, tree: FlowTree, answers: dict) -> dict:
    """Reduce a complete finite decision table into EXISTING branch semantics.

    Refuse any unknown and all-stop tables; do not count agreement as review or
    alter an already-agreeing tree. This consumes no private oracle or model.
    """
    diagnosis = compare(sources, tree, answers)
    body = dict(protocol=PROTOCOL, parentTreeDigest=diagnosis["treeDigest"],
        sourcesDigest=diagnosis["sourcesDigest"], answersDigest=sha256_json(answers),
        diagnosisDigest=diagnosis["reportDigest"], runtimeAuthorityGranted=False,
        semanticAlignmentProven=False, fullSourceReview="required_not_run", wholeSkillTranslations=0,
        preservedUnverified=["producer/target identity", "argument role and value semantics", "completion citation",
            "all source meaning outside this Boolean region", "source judgment correctness and citation entailment"])
    if diagnosis["unknowns"]:
        return _seal(dict(**body, status="unresolved_no_revision"))
    if not diagnosis["disagreements"]:
        return _seal(dict(**body, status="unchanged_finite_agreement"))
    if not any(r["outcome"] == "read" for r in answers.values()):
        return _seal(dict(**body, status="no_read_witness_requires_source_resolution"))
    context = region(sources, tree)
    raw = tree.model_dump(mode="json")

    def at(pointer):
        value = raw
        for part in pointer.lstrip("/").split("/"):
            value = value[int(part)] if isinstance(value, list) else value[part]
        return copy.deepcopy(value)

    target_index = context["targetIndices"][0]
    target = at(context["targetPointers"][0])
    completion_index = context["nodes"][target_index]["next"]
    completion = at(context["origins"][completion_index]["treePointer"])
    origin_rows = []

    def build(keys, depth, pointer):
        source_ids = sorted({s for key in keys for s in answers[key]["source_ids"]})
        outcomes = {answers[key]["outcome"] for key in keys}
        if len(outcomes) == 1:
            outcome = next(iter(outcomes))
            if outcome == "read":
                return []  # Share the single original target and completion.
            origin_rows.append(dict(treePointer=pointer + "/0", tableRows=keys, sourceIds=source_ids))
            return [dict(kind="end", source_id=source_ids[0], outcome=outcome)]
        field = context["facts"][depth]
        no = [k for k in keys if not context["assignments"][k][field]]
        yes = [k for k in keys if context["assignments"][k][field]]
        origin_rows.append(dict(treePointer=pointer + "/0", tableRows=keys, sourceIds=source_ids))
        return [dict(kind="if_equal", source_id=source_ids[0],
            left=dict(kind="reference", source=raw["steps"][0]["bind"], field=field),
            equals=dict(kind="constant", value=True), true_source_id=answers[yes[0]]["source_ids"][0],
            false_source_id=answers[no[0]]["source_ids"][0],
            when_equal=build(yes, depth + 1, pointer + "/0/when_equal"),
            otherwise=build(no, depth + 1, pointer + "/0/otherwise"))]

    # Build in a block at /steps/1, then correct the root mapping only. Child
    # pointers already refer to nested paths inside that root branch.
    guards = build(list(context["assignments"]), 0, "/guard")
    for row in origin_rows:
        row["treePointer"] = row["treePointer"].replace("/guard/0", "/steps/1", 1)
    candidate = FlowTree.model_validate({**raw, "steps": [raw["steps"][0], *guards, target, completion]})
    compiled = compile_report(sources, candidate)
    verification = compare(sources, candidate, answers)
    if verification["disagreements"] or verification["unknowns"]:
        raise ValueError("compiler invariant: lowered region differs from the supplied finite table")
    return _seal(dict(**body, status="revised_inactive_pending_source_review", tree=candidate.model_dump(mode="json"),
        compilation=compiled, tableOrigins=origin_rows, finiteVerification=verification,
        boundary="An explicit development revision, not a fresh translation, independent verification or Runtime admission."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path)
    parser.add_argument("tree", type=Path)
    parser.add_argument("answers", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _write(args.output, propose(FlowSources.model_validate_json(args.sources.read_text()),
        FlowTree.model_validate_json(args.tree.read_text()), json.loads(args.answers.read_text())))


if __name__ == "__main__":
    main()
