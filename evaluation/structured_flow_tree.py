"""Source-anchored structured FlowTree v2 lowering into the existing flow engine.

This deterministic authoring bridge neither calls a model nor certifies semantic
entailment. It never loads source scripts or invokes providers during compilation.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field

from evaluation.translation_intake import validate_bundle
from network_runtime.contracts import sha256_json
from network_runtime.l0.flow import StructuredFlowProposal, qualify_flow
from network_runtime.l0.models import StrictModel
from network_runtime.l0.structured_schema import snapshot_json


class SourceSpan(StrictModel):
    path: str
    start: int = Field(ge=0, strict=True)
    end: int = Field(gt=0, strict=True)
    quote: str = Field(min_length=8, max_length=1600)


class StructuredTreeRead(StrictModel):
    kind: Literal["read"]
    source: SourceSpan
    tool: str
    bind: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    arguments: dict[str, Any]


class StructuredTreeIf(StrictModel):
    kind: Literal["if_equal"]
    source: SourceSpan
    left: dict[str, Any]
    equals: Any
    when_equal: tuple["Statement", ...] = Field(max_length=64)
    otherwise: tuple["Statement", ...] = Field(max_length=64)


class StructuredTreeEffect(StrictModel):
    kind: Literal["effect_candidate"]
    source: SourceSpan
    binding_id: str
    arguments: dict[str, Any]


class StructuredTreeEnd(StrictModel):
    kind: Literal["end"]
    source: SourceSpan
    outcome: Literal["read_path_completed", "needs_l1", "unsupported"]
    explanation: str = Field(min_length=1)


Statement = Annotated[StructuredTreeRead | StructuredTreeIf | StructuredTreeEffect | StructuredTreeEnd,
                      Field(discriminator="kind")]


class StructuredFlowTree(StrictModel):
    api_version: Literal["netopyu.io/structured-flow-tree/v1"]
    source_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    purpose: str = Field(min_length=1)
    input_schema: dict[str, Any]
    max_read_age_seconds: float = Field(gt=0, le=300, allow_inf_nan=False)
    steps: tuple[Statement, ...] = Field(min_length=1, max_length=64)
    unresolved: tuple[str, ...] = Field(default=(), max_length=64)


StructuredTreeIf.model_rebuild()
StructuredFlowTree.model_rebuild()


def compile_structured_tree(bundle: dict, tree: StructuredFlowTree, reads: dict, effects: dict) -> dict:
    """Check exact source spans/lexical scope, lower edges, then use the shared qualifier."""
    validate_bundle(bundle)
    tree = StructuredFlowTree.model_validate(snapshot_json(tree.model_dump(mode="json")))
    if tree.source_digest != bundle["bundleDigest"] or tree.unresolved:
        raise ValueError("source digest mismatch or unresolved authoring obligations")
    documents = {d["path"]: d for d in bundle["documents"] if d["representation"] == "inert_utf8_text"}
    nodes, origins, aliases, used_reads = [], [], set(), {}

    def anchor(span, pointer, key):
        doc = documents.get(span.path)
        if not doc or doc["content"][span.start:span.end] != span.quote or span.end - span.start != len(span.quote):
            raise ValueError("source span must match the exact inert file and offsets")
        return {"treePointer": pointer, "nodeId": key, "sourcePath": span.path,
                "sourceDigest": doc["sha256"], "start": span.start, "end": span.end, "quote": span.quote,
                "semanticEntailmentProven": False}

    def expression(raw, environment):
        copied = snapshot_json(raw)
        pending, count = [copied], 0
        while pending:
            item = pending.pop()
            count += 1
            if count > 512:
                raise ValueError("binding expression exceeds node budget")
            if not isinstance(item, dict):
                continue
            if item.get("kind") in ("reference", "column_rows"):
                name = item.get("source")
                if not isinstance(name, str) or (name != "input" and name not in environment):
                    raise ValueError("source alias is not available in lexical scope")
                if name != "input":
                    item["source"] = environment[name]
            elif item.get("kind") == "object" and isinstance(item.get("fields"), dict):
                pending.extend(item["fields"].values())
            elif item.get("kind") == "array" and isinstance(item.get("items"), list):
                pending.extend(item["items"])
        return copied

    def block(statements, inherited, pointer, depth):
        if depth > 16:
            raise ValueError("structured tree nesting exceeds 16")
        environment, entry, exits = dict(inherited), None, []
        for index, statement in enumerate(statements):
            if len(nodes) >= 64 or (entry is not None and not exits):
                raise ValueError("tree node budget or unreachable continuation")
            key, at = f"node-{len(nodes)}", f"{pointer}/{index}"
            if entry is None:
                entry = key
            for parent, edge in exits:
                parent[edge] = key
            node = {"id": key}
            nodes.append(node)
            origins.append(anchor(statement.source, at, key))
            if isinstance(statement, StructuredTreeRead):
                if statement.bind == "input" or statement.bind in aliases:
                    raise ValueError("read aliases must be globally unique and cannot shadow input")
                if statement.tool not in reads or reads[statement.tool].spec.tool != statement.tool:
                    raise ValueError("tool must match an exact supplied read contract")
                contract = reads[statement.tool]
                node.update(kind="read", contract_hash=contract.contract_hash,
                            arguments=expression(statement.arguments, environment), next=None)
                used_reads[contract.contract_hash] = contract
                aliases.add(statement.bind)
                environment[statement.bind] = key
                exits = [(node, "next")]
            elif isinstance(statement, StructuredTreeIf):
                node.update(kind="branch", left=expression(statement.left, environment),
                            equals={"kind": "constant", "value": statement.equals}, on_true=None, on_false=None)
                exits = []
                for field, edge in (("when_equal", "on_true"), ("otherwise", "on_false")):
                    child, tails = block(getattr(statement, field), environment, at + "/" + field, depth + 1)
                    if child is None:
                        exits.append((node, edge))
                    else:
                        node[edge] = child
                        exits.extend(tails)
                # Even a branch that terminates cannot leak a local alias.
            elif isinstance(statement, StructuredTreeEffect):
                node.update(kind="effect_candidate", binding_id=statement.binding_id,
                            arguments=expression(statement.arguments, environment))
                exits = []
            else:
                node.update(kind="end", outcome=statement.outcome, explanation=statement.explanation)
                exits = []
        return entry, exits

    entry, exits = block(tree.steps, {}, "/steps", 1)
    if exits:
        raise ValueError("all root paths need explicit terminals; never infer success")
    tree_digest = sha256_json(tree.model_dump(mode="json"))
    flow = StructuredFlowProposal(api_version="netopyu.io/l0-flow-proposal/v2", source_digest=tree.source_digest,
                                  authoring_digest=tree_digest, purpose=tree.purpose, input_schema=tree.input_schema,
                                  entry=entry, nodes=nodes, max_read_age_seconds=tree.max_read_age_seconds)
    packet = qualify_flow(flow, used_reads, effects)
    body = {"treeDigest": tree_digest, "sourceBundleDigest": bundle["bundleDigest"], "flow": flow.model_dump(mode="json"),
            "qualification": packet, "origins": origins, "status": "compiled_pending_semantic_and_host_review",
            "modelCalls": 0, "wholeSkillTranslationProven": False, "runtimeAuthorityGranted": False}
    return {**body, "reportDigest": sha256_json(body)}
