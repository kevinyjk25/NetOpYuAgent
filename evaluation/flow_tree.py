"""Bounded authoring AST -> existing flow graph; never a second executor."""

from __future__ import annotations

import argparse
from typing import Annotated, Literal
from pathlib import Path

from pydantic import Field

from evaluation.flow_grounded_translation import cited_review_input, project
from evaluation.flow_source_selection import SelectedDraft, SelectedIssue, expand
from evaluation.flow_translation import FlowSources, _write, lower
from evaluation.read_l05_review import ReadL05Review
from evaluation.translation_source_alignment import evaluate_source_assessment
from network_runtime.contracts import sha256_json
from network_runtime.l0.flow import Constant, Reference, Value
from network_runtime.l0.models import StrictModel

PROTOCOL = "hierarchical-flow/v1"
Alias = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]


class TreeRead(StrictModel):
    kind: Literal["read"]
    source_id: str
    tool: str
    bind: Alias
    arguments: dict[str, Value]


class TreeIf(StrictModel):
    kind: Literal["if_equal"]
    source_id: str
    left: Reference
    equals: Constant
    true_source_id: str
    false_source_id: str
    when_equal: tuple["Statement", ...] = Field(max_length=64)
    otherwise: tuple["Statement", ...] = Field(max_length=64)


class TreeEnd(StrictModel):
    kind: Literal["end"]
    source_id: str
    outcome: Literal["read_path_completed", "needs_l1", "unsupported"]


class TreeEffect(StrictModel):
    kind: Literal["effect_candidate"]
    source_id: str
    binding_id: str
    arguments: dict[str, Value]


Statement = Annotated[TreeRead | TreeIf | TreeEnd | TreeEffect, Field(discriminator="kind")]


class FlowTree(StrictModel):
    business_source_ids: tuple[str, ...] = Field(min_length=1, max_length=2)
    steps: tuple[Statement, ...] = Field(min_length=1, max_length=64)
    issues: tuple[SelectedIssue, ...] = Field(max_length=16)


TreeIf.model_rebuild()
FlowTree.model_rebuild()


def _bounded(tree: FlowTree) -> None:
    count, names = 0, set()

    def visit(block, depth):
        nonlocal count
        if depth > 16:
            raise ValueError("tree nesting exceeds 16 levels")
        for statement in block:
            count += 1
            if count > 64:
                raise ValueError("tree exceeds 64 graph nodes")
            if isinstance(statement, TreeRead):
                if statement.bind == "input" or statement.bind in names:
                    raise ValueError("read aliases must be globally unique and cannot shadow input")
                names.add(statement.bind)
            elif isinstance(statement, TreeIf):
                visit(statement.when_equal, depth + 1)
                visit(statement.otherwise, depth + 1)
    visit(tree.steps, 1)


def compile_tree(sources: FlowSources, tree: FlowTree) -> tuple[SelectedDraft, list[dict]]:
    """Preserve AST order/polarity; synthesize forward edges and lexical read bindings."""
    tree = FlowTree.model_validate(tree.model_dump())
    _bounded(tree)
    nodes, origins = [], []

    def value(reference, environment):
        result = reference.model_dump(mode="json")
        if isinstance(reference, Reference) and reference.source != "input":
            if reference.source not in environment:
                raise ValueError("read alias unavailable in lexical scope")
            result["source"] = environment[reference.source]
        return result

    def block(statements, inherited, pointer):
        environment = dict(inherited)
        entry, exits = None, []
        for offset, statement in enumerate(statements):
            if entry is not None and not exits:
                raise ValueError("statement after terminal flow is unreachable; never drop it")
            index = len(nodes)
            path = f"{pointer}/{offset}"
            if entry is None:
                entry = index
            for predecessor, edge in exits:
                nodes[predecessor][edge] = index
            raw = {"source_id": statement.source_id, "requires": []}
            origins.append({"treePointer": path, "nodeIndex": index, "l0Pointer": f"/nodes/{index}"})
            if isinstance(statement, TreeRead):
                raw.update(kind="read", tool=statement.tool,
                    arguments={key: value(item, environment) for key, item in statement.arguments.items()}, next=None)
                environment[statement.bind] = index  # Binding is unavailable in its own arguments.
                nodes.append(raw)
                exits = [(index, "next")]
            elif isinstance(statement, TreeIf):
                raw.update(kind="branch", left=value(statement.left, environment),
                    equals=statement.equals.model_dump(mode="json"),
                    true_source_id=statement.true_source_id, false_source_id=statement.false_source_id,
                    on_true=None, on_false=None)
                nodes.append(raw)
                exits = []
                for field, statements in (("when_equal", statement.when_equal), ("otherwise", statement.otherwise)):
                    child, child_exits = block(statements, environment, path + "/" + field)
                    edge = "on_true" if field == "when_equal" else "on_false"
                    if child is None:
                        exits.append((index, edge))  # Empty branch falls through to common continuation.
                    else:
                        raw[edge] = child
                        exits.extend(child_exits)
                # Branch-local bindings never escape, even if one branch terminates.
            elif isinstance(statement, TreeEffect):
                raw.update(kind="effect_candidate", binding_id=statement.binding_id,
                    arguments={key: value(item, environment) for key, item in statement.arguments.items()})
                nodes.append(raw)
                exits = []
            else:
                raw.update(kind="end", outcome=statement.outcome)
                nodes.append(raw)
                exits = []
        return entry, exits

    entry, exits = block(tree.steps, {}, "/steps")
    if exits:
        raise ValueError("every root path requires an explicit terminal; never infer success")
    parents = {index: set() for index in range(len(nodes))}
    for index, node in enumerate(nodes):
        for field in ("next", "on_true", "on_false"):
            if field in node:
                successor = node[field]
                if successor is None or successor <= index:
                    raise ValueError("compiler invariant: generated edges must be strictly forward")
                parents[successor].add(index)
    dominators = {}
    for index, node in enumerate(nodes):
        before = set.intersection(*(dominators[p] | {p} for p in parents[index])) if parents[index] else set()
        dominators[index] = before
        node["requires"] = sorted(before)
    selected = SelectedDraft(business_source_ids=tree.business_source_ids, entry=entry, steps=nodes, issues=tree.issues)
    project(sources, expand(sources, selected))  # Existing source/type/DAG/contract gate remains mandatory.
    return selected, origins


def tree_review_input(sources: FlowSources, tree: FlowTree) -> dict:
    selected, origins = compile_tree(sources, tree)
    packet = cited_review_input(sources, expand(sources, selected))
    packet.pop("inputDigest")
    packet.update(treeProtocol=PROTOCOL, treeProposal=tree.model_dump(mode="json"), origins=origins)
    for claim in packet["claims"]:
        path = claim.get("l0Pointer") or ""
        if path.startswith("/nodes/"):
            index = int(path.split("/")[2])
            claim["treePointer"] = origins[index]["treePointer"]
    return {**packet, "inputDigest": sha256_json(packet)}


def assess_tree(sources: FlowSources, tree: FlowTree, review: ReadL05Review) -> dict:
    review = ReadL05Review.model_validate(review.model_dump())
    packet = tree_review_input(sources, tree)
    assessment = evaluate_source_assessment(packet, review.assessment)
    judgments = {row.claim_id: row for row in review.assessment.claims}
    for claim in packet["claims"]:
        judgment = judgments[claim["claimId"]]
        if judgment.verdict == "supported" and claim.get("requiredCitationId") and claim["requiredCitationId"] not in judgment.source_span_ids:
            raise ValueError("source coverage must cite its exact requirement")
    body = {"inputDigest": packet["inputDigest"], "reviewDigest": sha256_json(review.model_dump(mode="json")),
        "reviewerId": review.reviewer_id, "reviewerKind": review.reviewer_kind,
        "status": "review_supported_inactive_flow" if not tree.issues and all(row.verdict == "supported" for row in review.assessment.claims) else "blocked",
        "assessment": assessment, "runtimeAuthorityGranted": False, "semanticAlignmentProven": False}
    return {**body, "reportDigest": sha256_json(body)}


def compile_report(sources: FlowSources, tree: FlowTree) -> dict:
    selected, origins = compile_tree(sources, tree)
    cited = expand(sources, selected)
    draft = project(sources, cited)
    flow, packet = lower(sources, draft)
    body = {"treeProtocol": PROTOCOL, "treeDigest": sha256_json(tree.model_dump(mode="json")),
        "sourcesDigest": sha256_json(sources.model_dump(mode="json")),
        "selectedProposal": selected.model_dump(mode="json"), "flow": flow.model_dump(mode="json"),
        "flowDigest": packet["flowDigest"], "origins": origins,
        "reviewInput": tree_review_input(sources, tree), "runtimeAuthorityGranted": False,
        "status": "compiled_pending_source_review_not_executable"}
    return {**body, "reportDigest": sha256_json(body)}


def example_report() -> dict:
    from evaluation.flow_translation import local_sources
    source = Path(__file__).resolve().parents[1] / "examples/read-flow/flow-tree.json"
    result = compile_report(local_sources(), FlowTree.model_validate_json(source.read_text()))
    body = {"evidenceRole": "hand_authored_compiler_example_not_model_translation",
        "treePath": "examples/read-flow/flow-tree.json", "sourcePath": "examples/read-flow/flow-source.md",
        **{key: result[key] for key in ("treeDigest", "sourcesDigest", "flowDigest", "flow", "origins", "status")},
        "compilationDigest": result["reportDigest"], "reviewInputDigest": result["reviewInput"]["inputDigest"],
        "modelCalls": 0, "runtimeExecutions": 0, "runtimeAuthorityGranted": False}
    return {**body, "reportDigest": sha256_json(body)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("schema", "example", "compile", "assess"))
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--tree", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "schema":
        result = FlowTree.model_json_schema()
    elif args.command == "example":
        result = example_report()
    else:
        if not args.sources or not args.tree or (args.command == "assess" and not args.review):
            parser.error("compile/assess require --sources and --tree; assess also requires --review")
        sources = FlowSources.model_validate_json(args.sources.read_text())
        tree = FlowTree.model_validate_json(args.tree.read_text())
        result = compile_report(sources, tree) if args.command == "compile" else assess_tree(
            sources, tree, ReadL05Review.model_validate_json(args.review.read_text()))
    _write(args.output, result)


if __name__ == "__main__":
    main()
