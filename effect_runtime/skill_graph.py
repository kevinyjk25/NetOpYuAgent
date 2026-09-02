"""Direction rules for mixing natural-language L1 and deterministic L0 Skills."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
import re
from typing import Iterable


class SkillLevel(StrEnum):
    L1 = "L1"
    L0 = "L0"


@dataclass(frozen=True)
class SkillNode:
    skill_id: str
    level: SkillLevel
    active: bool = False
    version: str | None = None
    artifact_digest: str | None = None


@dataclass(frozen=True)
class SkillEdge:
    source: str
    target: str


def validate_skill_graph(
    nodes: Iterable[SkillNode], edges: Iterable[SkillEdge],
) -> dict[str, object]:
    """Validate the one-way determinization boundary.

    L1 may orchestrate L1 or call an active L0 contract. L0 may compose only
    active L0 contracts. L0 -> L1 is forbidden because it would widen model
    discretion in the middle of an effect transaction. Replanning exits the
    transaction and explicitly returns control to L1 instead.
    """
    node_list = list(nodes)
    edge_list = list(edges)
    index = {node.skill_id: node for node in node_list}
    findings: list[dict[str, str]] = []
    if len(index) != len(node_list):
        findings.append({"code": "DUPLICATE_SKILL_ID", "message": "Skill ids must be unique."})
    adjacency: dict[str, list[str]] = {key: [] for key in index}
    for edge in edge_list:
        source = index.get(edge.source)
        target = index.get(edge.target)
        if source is None or target is None:
            findings.append({
                "code": "UNKNOWN_SKILL_REFERENCE",
                "message": f"Unknown edge {edge.source!r} -> {edge.target!r}.",
            })
            continue
        adjacency[source.skill_id].append(target.skill_id)
        if source.level == SkillLevel.L0 and target.level == SkillLevel.L1:
            findings.append({
                "code": "L0_TO_L1_FORBIDDEN",
                "message": f"{source.skill_id} cannot call L1 {target.skill_id} inside a transaction.",
            })
        if target.level == SkillLevel.L0 and not target.active:
            findings.append({
                "code": "INACTIVE_L0_REFERENCE",
                "message": f"Referenced L0 {target.skill_id} is not active and digest-bound.",
            })
        if target.level == SkillLevel.L0 and target.active and (
            not target.version
            or not target.artifact_digest
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", target.artifact_digest)
        ):
            findings.append({
                "code": "L0_ARTIFACT_BINDING_MISSING",
                "message": f"Referenced L0 {target.skill_id} lacks an exact version/sha256 binding.",
            })

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str, trail: tuple[str, ...]) -> None:
        if node_id in visiting:
            cycle = " -> ".join((*trail, node_id))
            findings.append({"code": "SKILL_GRAPH_CYCLE", "message": f"Cycle detected: {cycle}."})
            return
        if node_id in visited:
            return
        visiting.add(node_id)
        for target_id in adjacency.get(node_id, []):
            visit(target_id, (*trail, node_id))
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in sorted(index):
        visit(node_id, ())
    return {
        "schema": "effect-runtime.io/skill-graph-report/v1",
        "gate": "passed" if not findings else "blocked",
        "nodes": [
            {**asdict(node), "level": node.level.value}
            for node in node_list
        ],
        "edges": [asdict(edge) for edge in edge_list],
        "findings": findings,
        "replanBoundary": "An L0 stop/replan result ends the effect transaction before control returns to L1.",
    }


__all__ = ["SkillEdge", "SkillLevel", "SkillNode", "validate_skill_graph"]
