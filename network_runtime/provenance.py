"""Privacy-minimized cross-step Evidence provenance projection."""

from __future__ import annotations

from typing import Any, Iterable

from .contracts import PreparedPlan, sha256_json


def _evidence_dicts(
    plan: PreparedPlan,
    record: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    values = [item.to_dict() for item in plan.preflight]
    result = record.get("result")
    if isinstance(result, dict):
        values.extend(
            item for item in result.get("evidence") or ()
            if isinstance(item, dict)
        )
    unique: dict[str, dict[str, Any]] = {}
    for value in values:
        evidence_id = str(value.get("evidence_id") or sha256_json({
            "type": value.get("evidence_type"),
            "source": value.get("source"),
            "target": value.get("target"),
            "observed_at": value.get("observed_at"),
            "value_digest": sha256_json(value.get("value")),
        }))
        unique[evidence_id] = {**value, "evidence_id": evidence_id}
    return tuple(unique[key] for key in sorted(unique))


def build_provenance_dag(
    plan: PreparedPlan,
    events: Iterable[dict[str, Any]],
    record: dict[str, Any],
) -> dict[str, Any]:
    """Link Evidence → Observation → Capability/Collector → Object."""
    nodes: dict[str, dict[str, Any]] = {}
    edges: set[tuple[str, str, str]] = set()

    def node(node_id: str, kind: str, **attributes: Any) -> None:
        candidate = {
            "id": node_id,
            "kind": kind,
            **{
                key: value for key, value in attributes.items()
                if value not in (None, "", [])
            },
        }
        current = nodes.get(node_id)
        if current is None or (
            current.get("kind") == "evidence_reference" and kind != "evidence_reference"
        ):
            nodes[node_id] = candidate
            return
        # Multiple graph steps may refer to the same object.  Preserve the
        # strongest node type while filling any non-conflicting metadata.
        for key, value in candidate.items():
            current.setdefault(key, value)

    def edge(source: str, relation: str, target: str) -> None:
        edges.add((source, relation, target))

    for value in _evidence_dicts(plan, record):
        evidence_id = str(value["evidence_id"])
        observation_id = "observation:" + sha256_json({
            "evidence_id": evidence_id,
            "observed_at": value.get("observed_at"),
            "value_digest": sha256_json(value.get("value")),
        }).removeprefix("sha256:")
        capability = str(
            value.get("source_capability") or value.get("source") or "unknown"
        )
        capability_id = "capability:" + capability
        collector = str(value.get("collector_identity") or "unknown")
        collector_id = "collector:" + sha256_json(collector).removeprefix("sha256:")
        objects = tuple(str(item) for item in value.get("scope") or ()) or (
            str(value.get("target") or "unknown"),
        )
        node(
            evidence_id,
            "evidence",
            semantic_type=value.get("semantic_type") or value.get("evidence_type"),
            passed=value.get("passed"),
            value_digest=sha256_json(value.get("value")),
        )
        node(
            observation_id,
            "observation",
            observed_at=value.get("observed_at"),
        )
        node(capability_id, "capability", capability=capability)
        node(
            collector_id,
            "collector",
            identity_digest=(
                value.get("collector_digest") or sha256_json(collector)
            ),
        )
        edge(evidence_id, "derived_from", observation_id)
        edge(observation_id, "collected_via", capability_id)
        edge(observation_id, "collected_by", collector_id)
        for object_value in objects:
            object_id = "object:" + sha256_json(object_value).removeprefix("sha256:")
            node(
                object_id,
                "network_object",
                object_digest=sha256_json(object_value),
                object_kind=(
                    object_value.split(":", 1)[0]
                    if ":" in object_value else "opaque"
                ),
            )
            edge(observation_id, "describes", object_id)
        for parent in value.get("parent_evidence_ids") or ():
            parent_id = str(parent)
            node(parent_id, "evidence_reference")
            edge(evidence_id, "depends_on", parent_id)

    for event in events:
        if event.get("event_type") != "graph_node_finished":
            continue
        payload = event.get("payload") or {}
        step_id = "step:" + str(payload.get("node_id") or "unknown")
        node(
            step_id,
            "graph_step",
            phase=payload.get("phase"),
            outcome=payload.get("outcome"),
            duration_ms=payload.get("duration_ms"),
        )
        for evidence_id in payload.get("input_evidence_ids") or ():
            evidence_id = str(evidence_id)
            node(evidence_id, "evidence_reference")
            edge(step_id, "consumes", evidence_id)
        for evidence_id in payload.get("output_evidence_ids") or ():
            evidence_id = str(evidence_id)
            node(evidence_id, "evidence_reference")
            edge(step_id, "produces", evidence_id)

    rendered_nodes = [nodes[key] for key in sorted(nodes)]
    rendered_edges = [
        {"from": source, "relation": relation, "to": target}
        for source, relation, target in sorted(edges)
    ]
    indegree = {node_id: 0 for node_id in nodes}
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in nodes}
    dangling_edges = 0
    for source, _, target in edges:
        if source not in nodes or target not in nodes:
            dangling_edges += 1
            continue
        if target not in adjacency[source]:
            adjacency[source].add(target)
            indegree[target] += 1
    ready = sorted(node_id for node_id, degree in indegree.items() if degree == 0)
    visited = 0
    while ready:
        source = ready.pop(0)
        visited += 1
        for target in sorted(adjacency[source]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    acyclic = visited == len(nodes)
    evidence_nodes = [item for item in rendered_nodes if item["kind"] == "evidence"]
    complete = sum(
        any(
            edge_value[0] == item["id"] and edge_value[1] == "derived_from"
            for edge_value in edges
        )
        for item in evidence_nodes
    )
    return {
        "schema": "netopyu.io/evidence-provenance-dag/v1",
        "plan_id": plan.plan_id,
        "plan_hash": plan.plan_hash,
        "nodes": rendered_nodes,
        "edges": rendered_edges,
        "coverage": {
            "evidence_nodes": len(evidence_nodes),
            "traceable_observations": complete,
            "traceability_rate": (
                round(complete / len(evidence_nodes), 6) if evidence_nodes else 1.0
            ),
        },
        "integrity": {
            "acyclic": acyclic,
            "dangling_edges": dangling_edges,
        },
        "dag_digest": sha256_json({
            "nodes": rendered_nodes,
            "edges": rendered_edges,
        }),
        "claim_boundary": (
            "Traceability proves recorded lineage, not truth of the observed payload."
        ),
    }


__all__ = ["build_provenance_dag"]
