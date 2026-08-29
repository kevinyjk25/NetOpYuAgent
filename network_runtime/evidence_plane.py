"""P2.2 read-only evidence projection, operations metrics, and incident views.

Every source is opened read-only and projected to digests plus bounded scalar
metadata.  Raw prompts, argument values, approval identities, provider payloads,
and filesystem paths are never emitted.  This module cannot execute, approve,
activate, register, or mutate a Runtime/Provider/Catalog source.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Literal

from network_runtime.contracts import sha256_json
from network_runtime.l0.promotion import PromotionError
from network_runtime.l0.workbench import inspect_workbench


EVIDENCE_PLANE_SCHEMA = "netopyu.io/evidence-plane-snapshot/v1"
EVIDENCE_EVENT_SCHEMA = "netopyu.io/evidence-plane-event/v1"
EVIDENCE_INCIDENT_SCHEMA = "netopyu.io/evidence-plane-incident/v1"
EVIDENCE_TREND_SCHEMA = "netopyu.io/evidence-plane-trend/v1"
_MAX_DATABASE_BYTES = 256_000_000
_MAX_EVENTS = 20_000
_MAX_PROPOSALS = 500
_SOURCE_KINDS = {"runtime", "decision", "saga", "provider_release", "promotion"}


class EvidencePlaneError(RuntimeError):
    pass


def _digest(label: str, value: Any) -> str:
    return sha256_json({label: value})


def _safe_path(path: str | Path, *, database: bool = False) -> Path:
    supplied = Path(path).expanduser()
    if supplied.is_symlink() or not supplied.is_file():
        raise EvidencePlaneError("evidence source is missing or unsafe")
    resolved = supplied.resolve()
    if database and resolved.stat().st_size > _MAX_DATABASE_BYTES:
        raise EvidencePlaneError("evidence database exceeds 256 MB")
    return resolved


def _read_only_database(path: str | Path) -> sqlite3.Connection:
    source = _safe_path(path, database=True)
    connection = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True, timeout=5)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA busy_timeout=5000")
    return connection


def _tables(database: sqlite3.Connection) -> set[str]:
    return {
        str(row[0]) for row in database.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        parsed = datetime.fromisoformat(normalized)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _duration_ms(start: str | None, end: str | None) -> float | None:
    left, right = _parse_time(start), _parse_time(end)
    if left is None or right is None or right < left:
        return None
    return (right - left).total_seconds() * 1000


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    ordered = sorted(float(item) for item in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] * (1 - fraction) + ordered[upper] * fraction, 3)


def _source(kind: str, path: Path, integrity: str, records: int, *, truncated: bool) -> dict[str, Any]:
    if kind not in _SOURCE_KINDS:
        raise EvidencePlaneError("unsupported evidence source kind")
    return {
        "source_id": _digest("source", {"kind": kind, "path": str(path)}),
        "kind": kind,
        "integrity": integrity,
        "records": records,
        "truncated": truncated,
        "path_exposed": False,
    }


def _base_event(
    *,
    source_id: str,
    source_kind: str,
    ordinal: int,
    occurred_at: str | None,
    correlation_digest: str,
    category: str,
    event_type: str,
    state: str | None,
    outcome: str | None,
    source_integrity: Literal["verified", "invalid", "unverified"],
    evidence_digest: str,
    attributes: dict[str, str | int | float | bool | None] | None = None,
) -> dict[str, Any]:
    body = {
        "apiVersion": EVIDENCE_EVENT_SCHEMA,
        "source_id": source_id,
        "source_kind": source_kind,
        "ordinal": ordinal,
        "occurred_at": occurred_at,
        "correlation_digest": correlation_digest,
        "category": category,
        "event_type": event_type[:128],
        "state": None if state is None else state[:128],
        "outcome": None if outcome is None else outcome[:128],
        "source_integrity": source_integrity,
        "evidence_digest": evidence_digest,
        "attributes": attributes or {},
    }
    return {**body, "event_id": sha256_json(body)}


def _incident(
    *,
    source_id: str,
    correlation_digest: str,
    severity: Literal["low", "medium", "high", "critical"],
    code: str,
    occurred_at: str | None,
    evidence_digest: str,
) -> dict[str, Any]:
    body = {
        "apiVersion": EVIDENCE_INCIDENT_SCHEMA,
        "source_id": source_id,
        "correlation_digest": correlation_digest,
        "severity": severity,
        "code": code,
        "occurred_at": occurred_at,
        "evidence_digest": evidence_digest,
        "contains_raw_payload": False,
    }
    return {**body, "incident_id": sha256_json(body)}


def _category(event_type: str) -> str:
    lowered = event_type.lower()
    if "approval" in lowered:
        return "approval"
    if any(value in lowered for value in ("verify", "verification", "postcondition")):
        return "verification"
    if any(value in lowered for value in ("rollback", "compensat", "restore")):
        return "compensation"
    if any(value in lowered for value in ("execute", "execution", "write", "effect")):
        return "execution"
    if "release" in lowered or "deploy" in lowered:
        return "publication"
    if "decision" in lowered or "route" in lowered:
        return "decision"
    if "promotion" in lowered or "proposal" in lowered or "review" in lowered:
        return "promotion"
    return "lifecycle"


def _collect_runtime(path: str | Path, limit: int) -> dict[str, Any]:
    resolved = _safe_path(path, database=True)
    source_id = _digest("source", {"kind": "runtime", "path": str(resolved)})
    with _read_only_database(resolved) as database:
        tables = _tables(database)
        if not {"plans", "plan_events"}.issubset(tables):
            raise EvidencePlaneError("runtime evidence database schema is unsupported")
        event_columns = {
            str(row["name"])
            for row in database.execute("PRAGMA table_info(plan_events)").fetchall()
        }
        has_event_chain = {"prev_event_hash", "event_hash"}.issubset(event_columns)
        plan_rows = database.execute(
            "SELECT plan_id, plan_hash, state, created_at, updated_at FROM plans "
            "ORDER BY created_at LIMIT ?", (limit + 1,),
        ).fetchall()
        event_rows = database.execute(
            (
                "SELECT event_id, plan_id, from_state, to_state, event_type, payload_json, "
                "prev_event_hash, event_hash, created_at FROM plan_events "
                if has_event_chain else
                "SELECT event_id, plan_id, from_state, to_state, event_type, payload_json, "
                "NULL AS prev_event_hash, NULL AS event_hash, created_at FROM plan_events "
            ) + "ORDER BY event_id LIMIT ?", (limit + 1,),
        ).fetchall()
        workflow_rows = []
        if "workflow_runs" in tables:
            workflow_rows = database.execute(
                "SELECT run_id, session_id, profile, status, started_at, updated_at "
                "FROM workflow_runs ORDER BY started_at LIMIT ?", (limit + 1,),
            ).fetchall()
    truncated = any(len(rows) > limit for rows in (plan_rows, event_rows, workflow_rows))
    plan_rows, event_rows, workflow_rows = plan_rows[:limit], event_rows[:limit], workflow_rows[:limit]
    chain_errors: list[str] = []
    previous_by_plan: dict[str, str] = {}
    events: list[dict[str, Any]] = []
    rollback_correlations: set[str] = set()
    for ordinal, row in enumerate(event_rows, 1):
        plan_id = str(row["plan_id"])
        previous = previous_by_plan.get(plan_id, "GENESIS")
        expected = sha256_json({
            "plan_id": plan_id,
            "from_state": row["from_state"],
            "to_state": row["to_state"],
            "event_type": row["event_type"],
            "payload_json": row["payload_json"],
            "created_at": row["created_at"],
            "prev_event_hash": previous,
        })
        if has_event_chain and (
            row["prev_event_hash"] != previous or row["event_hash"] != expected
        ):
            chain_errors.append(_digest("runtime_event", int(row["event_id"])))
        source_event_digest = (
            str(row["event_hash"])
            if has_event_chain else sha256_json({
                "plan_id": plan_id, "from_state": row["from_state"],
                "to_state": row["to_state"], "event_type": row["event_type"],
                "payload_json": row["payload_json"], "created_at": row["created_at"],
            })
        )
        previous_by_plan[plan_id] = str(row["event_hash"] or "")
        correlation = _digest("runtime_plan", plan_id)
        if "rollback" in str(row["event_type"]) or "compensat" in str(row["event_type"]):
            rollback_correlations.add(correlation)
        events.append(_base_event(
            source_id=source_id, source_kind="runtime", ordinal=ordinal,
            occurred_at=str(row["created_at"]), correlation_digest=correlation,
            category=_category(str(row["event_type"])), event_type=str(row["event_type"]),
            state=str(row["to_state"]), outcome=None,
            source_integrity="unverified" if truncated or not has_event_chain else (
                "invalid" if chain_errors else "verified"
            ),
            evidence_digest=source_event_digest,
            attributes={
                "from_state": None if row["from_state"] is None else str(row["from_state"]),
                "to_state": str(row["to_state"]),
            },
        ))
    state_counts = Counter(str(row["state"]) for row in plan_rows)
    durations = [
        value for value in (
            _duration_ms(str(row["created_at"]), str(row["updated_at"])) for row in plan_rows
        ) if value is not None
    ]
    incidents: list[dict[str, Any]] = []
    for row in plan_rows:
        state = str(row["state"])
        if state in {
            "manual_intervention_required", "outcome_indeterminate", "execution_failed",
        }:
            severity: Literal["high", "critical"] = (
                "critical" if state == "manual_intervention_required" else "high"
            )
            incidents.append(_incident(
                source_id=source_id,
                correlation_digest=_digest("runtime_plan", str(row["plan_id"])),
                severity=severity, code=f"RUNTIME_{state.upper()}",
                occurred_at=str(row["updated_at"]), evidence_digest=str(row["plan_hash"]),
            ))
    if chain_errors:
        incidents.append(_incident(
            source_id=source_id, correlation_digest=_digest("runtime_source", source_id),
            severity="critical", code="RUNTIME_EVENT_CHAIN_INVALID", occurred_at=None,
            evidence_digest=sha256_json(chain_errors),
        ))
    for ordinal, row in enumerate(workflow_rows, len(events) + 1):
        events.append(_base_event(
            source_id=source_id, source_kind="runtime", ordinal=ordinal,
            occurred_at=str(row["updated_at"]),
            correlation_digest=_digest("workflow_session", str(row["session_id"])),
            category="workflow", event_type="workflow_snapshot", state=str(row["status"]),
            outcome=str(row["status"]), source_integrity="unverified",
            evidence_digest=sha256_json({
                "run_id": row["run_id"], "profile": row["profile"],
                "status": row["status"], "started_at": row["started_at"],
                "updated_at": row["updated_at"],
            }),
            attributes={"profile": str(row["profile"])},
        ))
    integrity = (
        "unverified" if truncated or not has_event_chain
        else ("invalid" if chain_errors else "verified")
    )
    terminal = sum(state_counts[value] for value in (
        "verified_success", "rollback_verified", "precondition_changed",
        "manual_intervention_required", "rejected", "expired",
    ))
    metrics = {
        "plans": len(plan_rows),
        "state_counts": dict(sorted(state_counts.items())),
        "verified_success_rate": (
            round(state_counts["verified_success"] / terminal, 6) if terminal else None
        ),
        "rollback_attempts": len(rollback_correlations),
        "rollback_verified": state_counts["rollback_verified"],
        "rollback_success_rate": (
            round(state_counts["rollback_verified"] / len(rollback_correlations), 6)
            if rollback_correlations else None
        ),
        "manual_intervention": state_counts["manual_intervention_required"],
        "duration_p50_ms": _percentile(durations, 0.5),
        "duration_p95_ms": _percentile(durations, 0.95),
        "event_chain_present": has_event_chain,
    }
    return {
        "source": _source("runtime", resolved, integrity, len(events), truncated=truncated),
        "events": events, "incidents": incidents, "metrics": metrics,
    }


def _collect_decisions(path: str | Path, limit: int) -> dict[str, Any]:
    resolved = _safe_path(path, database=True)
    source_id = _digest("source", {"kind": "decision", "path": str(resolved)})
    with _read_only_database(resolved) as database:
        tables = _tables(database)
        if "decisions" not in tables:
            raise EvidencePlaneError("decision evidence database schema is unsupported")
        rows = database.execute(
            "SELECT sequence, created_at, decision_id, session_id, harness, status, action, "
            "target, evidence_digest, duration_ms, lifecycle_status, lifecycle_reason, "
            "envelope_json FROM decisions ORDER BY sequence LIMIT ?", (limit + 1,),
        ).fetchall()
        observations = []
        if "observations" in tables:
            observations = database.execute(
                "SELECT sequence, created_at, decision_id, session_id, target_match, "
                "arguments_exact, safety_escape, outcome FROM observations "
                "ORDER BY sequence LIMIT ?", (limit + 1,),
            ).fetchall()
    truncated = len(rows) > limit or len(observations) > limit
    rows, observations = rows[:limit], observations[:limit]
    invalid: list[str] = []
    events: list[dict[str, Any]] = []
    durations: list[float] = []
    status_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    lifecycle_counts: Counter[str] = Counter()
    for ordinal, row in enumerate(rows, 1):
        try:
            envelope = json.loads(str(row["envelope_json"]))
            evidence = envelope["evidence"]
            if sha256_json(evidence) != str(row["evidence_digest"]):
                raise ValueError("digest mismatch")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            invalid.append(_digest("decision_sequence", int(row["sequence"])))
        status = str(row["status"])
        action = None if row["action"] is None else str(row["action"])
        lifecycle = str(row["lifecycle_status"])
        status_counts[status] += 1
        lifecycle_counts[lifecycle] += 1
        if action:
            action_counts[action] += 1
        durations.append(float(row["duration_ms"]))
        events.append(_base_event(
            source_id=source_id, source_kind="decision", ordinal=ordinal,
            occurred_at=str(row["created_at"]),
            correlation_digest=_digest("decision_session", str(row["session_id"])),
            category="decision", event_type="l1_decision", state=lifecycle,
            outcome=action or status,
            source_integrity="unverified" if truncated else ("invalid" if invalid else "verified"),
            evidence_digest=str(row["evidence_digest"]),
            attributes={"harness": str(row["harness"]), "status": status},
        ))
    target_matches = 0
    argument_exact = 0
    argument_observed = 0
    safety_escapes = 0
    incidents: list[dict[str, Any]] = []
    for ordinal, row in enumerate(observations, len(events) + 1):
        target_matches += int(bool(row["target_match"]))
        if row["arguments_exact"] is not None:
            argument_observed += 1
            argument_exact += int(bool(row["arguments_exact"]))
        safety = bool(row["safety_escape"])
        safety_escapes += int(safety)
        correlation = _digest("decision_session", str(row["session_id"]))
        evidence_digest = sha256_json({
            "decision_digest": _digest("decision_id", str(row["decision_id"])),
            "target_match": bool(row["target_match"]),
            "arguments_exact": (
                None if row["arguments_exact"] is None else bool(row["arguments_exact"])
            ),
            "safety_escape": safety, "outcome": row["outcome"],
        })
        events.append(_base_event(
            source_id=source_id, source_kind="decision", ordinal=ordinal,
            occurred_at=str(row["created_at"]), correlation_digest=correlation,
            category="decision", event_type="l1_route_observed", state="observed",
            outcome=str(row["outcome"]), source_integrity="unverified",
            evidence_digest=evidence_digest,
            attributes={
                "target_match": bool(row["target_match"]),
                "arguments_exact": (
                    None if row["arguments_exact"] is None else bool(row["arguments_exact"])
                ),
                "safety_escape": safety,
            },
        ))
        if safety:
            incidents.append(_incident(
                source_id=source_id, correlation_digest=correlation, severity="critical",
                code="L1_SAFETY_ESCAPE_OBSERVED", occurred_at=str(row["created_at"]),
                evidence_digest=evidence_digest,
            ))
    if invalid:
        incidents.append(_incident(
            source_id=source_id, correlation_digest=_digest("decision_source", source_id),
            severity="critical", code="L1_EVIDENCE_DIGEST_INVALID", occurred_at=None,
            evidence_digest=sha256_json(invalid),
        ))
    integrity = "unverified" if truncated else ("invalid" if invalid else "verified")
    metrics = {
        "decisions": len(rows),
        "observations": len(observations),
        "status_counts": dict(sorted(status_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "lifecycle_counts": dict(sorted(lifecycle_counts.items())),
        "target_match_rate": round(target_matches / len(observations), 6) if observations else None,
        "target_matches": target_matches,
        "argument_exact_rate": (
            round(argument_exact / argument_observed, 6) if argument_observed else None
        ),
        "argument_observed": argument_observed,
        "argument_exact": argument_exact,
        "safety_escapes": safety_escapes,
        "model_p50_ms": _percentile(durations, 0.5),
        "model_p95_ms": _percentile(durations, 0.95),
    }
    return {
        "source": _source("decision", resolved, integrity, len(events), truncated=truncated),
        "events": events, "incidents": incidents, "metrics": metrics,
    }


def _collect_sagas(path: str | Path, limit: int) -> dict[str, Any]:
    resolved = _safe_path(path, database=True)
    source_id = _digest("source", {"kind": "saga", "path": str(resolved)})
    with _read_only_database(resolved) as database:
        tables = _tables(database)
        if not {"effect_sagas", "effect_saga_events"}.issubset(tables):
            raise EvidencePlaneError("Saga evidence database schema is unsupported")
        sagas = database.execute(
            "SELECT saga_id, correlation_id, definition_hash, state, created_at, updated_at "
            "FROM effect_sagas ORDER BY created_at LIMIT ?", (limit + 1,),
        ).fetchall()
        rows = database.execute(
            "SELECT event_id, saga_id, event_type, payload_json, created_at, previous_hash, "
            "event_hash FROM effect_saga_events ORDER BY event_id LIMIT ?", (limit + 1,),
        ).fetchall()
    truncated = len(sagas) > limit or len(rows) > limit
    sagas, rows = sagas[:limit], rows[:limit]
    previous: dict[str, str | None] = {}
    invalid: list[str] = []
    events: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows, 1):
        saga_id = str(row["saga_id"])
        prior = previous.get(saga_id)
        try:
            payload = json.loads(str(row["payload_json"]))
        except json.JSONDecodeError:
            payload = {"invalid": True}
            invalid.append(_digest("saga_event", int(row["event_id"])))
        expected = sha256_json({
            "saga_id": saga_id, "event_type": row["event_type"], "payload": payload,
            "created_at": row["created_at"], "previous_hash": prior,
        })
        if row["previous_hash"] != prior or row["event_hash"] != expected:
            invalid.append(_digest("saga_event", int(row["event_id"])))
        previous[saga_id] = str(row["event_hash"])
        events.append(_base_event(
            source_id=source_id, source_kind="saga", ordinal=ordinal,
            occurred_at=str(row["created_at"]),
            correlation_digest=_digest("saga", saga_id), category=_category(str(row["event_type"])),
            event_type=str(row["event_type"]), state=None, outcome=None,
            source_integrity="unverified" if truncated else ("invalid" if invalid else "verified"),
            evidence_digest=str(row["event_hash"]),
        ))
    states = Counter(str(row["state"]) for row in sagas)
    incidents: list[dict[str, Any]] = []
    for row in sagas:
        state = str(row["state"])
        if state in {"failed", "manual_intervention_required"}:
            incidents.append(_incident(
                source_id=source_id, correlation_digest=_digest("saga", str(row["saga_id"])),
                severity="critical" if state == "manual_intervention_required" else "high",
                code=f"SAGA_{state.upper()}", occurred_at=str(row["updated_at"]),
                evidence_digest=str(row["definition_hash"]),
            ))
    if invalid:
        incidents.append(_incident(
            source_id=source_id, correlation_digest=_digest("saga_source", source_id),
            severity="critical", code="SAGA_EVENT_CHAIN_INVALID", occurred_at=None,
            evidence_digest=sha256_json(invalid),
        ))
    integrity = "unverified" if truncated else ("invalid" if invalid else "verified")
    metrics = {
        "sagas": len(sagas), "state_counts": dict(sorted(states.items())),
        "verified_success_rate": (
            round(states["verified_success"] / len(sagas), 6) if sagas else None
        ),
        "compensated": states["compensated"],
        "manual_intervention": states["manual_intervention_required"],
    }
    return {
        "source": _source("saga", resolved, integrity, len(events), truncated=truncated),
        "events": events, "incidents": incidents, "metrics": metrics,
    }


def _collect_provider_releases(path: str | Path, limit: int) -> dict[str, Any]:
    resolved = _safe_path(path, database=True)
    source_id = _digest("source", {"kind": "provider_release", "path": str(resolved)})
    with _read_only_database(resolved) as database:
        if "release_events" not in _tables(database):
            raise EvidencePlaneError("Provider release evidence schema is unsupported")
        rows = database.execute(
            "SELECT event_id, event_type, release_digest, payload_json, prev_event_hash, "
            "event_hash, created_at FROM release_events ORDER BY event_id LIMIT ?", (limit + 1,),
        ).fetchall()
    truncated = len(rows) > limit
    rows = rows[:limit]
    prior = "GENESIS"
    invalid: list[str] = []
    events: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    incidents: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows, 1):
        expected = sha256_json({
            "event_type": row["event_type"], "release_digest": row["release_digest"],
            "payload_json": row["payload_json"], "prev_event_hash": prior,
            "created_at": row["created_at"],
        })
        if row["prev_event_hash"] != prior or row["event_hash"] != expected:
            invalid.append(_digest("provider_release_event", int(row["event_id"])))
        prior = str(row["event_hash"])
        event_type = str(row["event_type"])
        counts[event_type] += 1
        release = str(row["release_digest"] or "none")
        correlation = _digest("provider_release", release)
        events.append(_base_event(
            source_id=source_id, source_kind="provider_release", ordinal=ordinal,
            occurred_at=str(row["created_at"]), correlation_digest=correlation,
            category="publication", event_type=event_type, state=None, outcome=event_type,
            source_integrity="unverified" if truncated else ("invalid" if invalid else "verified"),
            evidence_digest=str(row["event_hash"]),
        ))
        if event_type == "release_rolled_back":
            incidents.append(_incident(
                source_id=source_id, correlation_digest=correlation, severity="medium",
                code="PROVIDER_RELEASE_ROLLBACK", occurred_at=str(row["created_at"]),
                evidence_digest=str(row["event_hash"]),
            ))
    if invalid:
        incidents.append(_incident(
            source_id=source_id, correlation_digest=_digest("provider_source", source_id),
            severity="critical", code="PROVIDER_EVENT_CHAIN_INVALID", occurred_at=None,
            evidence_digest=sha256_json(invalid),
        ))
    integrity = "unverified" if truncated else ("invalid" if invalid else "verified")
    return {
        "source": _source("provider_release", resolved, integrity, len(events), truncated=truncated),
        "events": events, "incidents": incidents,
        "metrics": {"events": len(rows), "event_counts": dict(sorted(counts.items()))},
    }


def _collect_promotions(path: str | Path, limit: int) -> dict[str, Any]:
    supplied = Path(path).expanduser()
    if supplied.is_symlink() or not supplied.is_dir():
        raise EvidencePlaneError("Promotion evidence root is missing or unsafe")
    resolved = supplied.resolve()
    source_id = _digest("source", {"kind": "promotion", "path": str(resolved)})
    children = [
        item for item in sorted(resolved.iterdir(), key=lambda value: value.name)
        if item.is_dir() and not item.is_symlink()
    ]
    truncated = len(children) > min(limit, _MAX_PROPOSALS)
    children = children[:min(limit, _MAX_PROPOSALS)]
    events: list[dict[str, Any]] = []
    incidents: list[dict[str, Any]] = []
    states: Counter[str] = Counter()
    invalid = 0
    for ordinal, child in enumerate(children, 1):
        try:
            view = inspect_workbench(child)
            status = str(view["status"])
            proposal_hash = str(view["proposal"]["proposal_hash"])
            occurred_at = None
            review_path = child / "review.json"
            if review_path.is_file() and not review_path.is_symlink():
                try:
                    occurred_at = json.loads(review_path.read_text(encoding="utf-8")).get("reviewedAt")
                except (OSError, TypeError, json.JSONDecodeError):
                    occurred_at = None
            states[status] += 1
            events.append(_base_event(
                source_id=source_id, source_kind="promotion", ordinal=ordinal,
                occurred_at=occurred_at,
                correlation_digest=_digest("promotion_proposal", proposal_hash),
                category="promotion", event_type="promotion_snapshot", state=status,
                outcome=status, source_integrity="verified", evidence_digest=proposal_hash,
                attributes={"activation_available": False},
            ))
        except (OSError, PromotionError, TypeError, ValueError):
            invalid += 1
            correlation = _digest("promotion_directory", child.name)
            evidence = _digest("invalid_promotion", child.name)
            events.append(_base_event(
                source_id=source_id, source_kind="promotion", ordinal=ordinal,
                occurred_at=None, correlation_digest=correlation, category="promotion",
                event_type="promotion_invalid", state="invalid", outcome="invalid",
                source_integrity="invalid", evidence_digest=evidence,
                attributes={"activation_available": False},
            ))
            incidents.append(_incident(
                source_id=source_id, correlation_digest=correlation, severity="high",
                code="PROMOTION_PACKAGE_INVALID", occurred_at=None, evidence_digest=evidence,
            ))
    integrity = "unverified" if truncated else ("invalid" if invalid else "verified")
    return {
        "source": _source("promotion", resolved, integrity, len(events), truncated=truncated),
        "events": events, "incidents": incidents,
        "metrics": {
            "proposals": len(children), "status_counts": dict(sorted(states.items())),
            "invalid": invalid, "activation_available": False,
        },
    }


def collect_evidence_snapshot(
    *,
    runtime_journals: Iterable[str | Path] = (),
    decision_stores: Iterable[str | Path] = (),
    saga_stores: Iterable[str | Path] = (),
    provider_registries: Iterable[str | Path] = (),
    proposal_roots: Iterable[str | Path] = (),
    limit_per_source: int = 5_000,
    generated_at: str | None = None,
) -> dict[str, Any]:
    if not 1 <= limit_per_source <= _MAX_EVENTS:
        raise EvidencePlaneError(f"evidence limit_per_source must be 1..{_MAX_EVENTS}")
    collectors: list[tuple[str, str | Path]] = [
        *(("runtime", item) for item in runtime_journals),
        *(("decision", item) for item in decision_stores),
        *(("saga", item) for item in saga_stores),
        *(("provider_release", item) for item in provider_registries),
        *(("promotion", item) for item in proposal_roots),
    ]
    if not collectors:
        raise EvidencePlaneError("at least one evidence source is required")
    handlers = {
        "runtime": _collect_runtime,
        "decision": _collect_decisions,
        "saga": _collect_sagas,
        "provider_release": _collect_provider_releases,
        "promotion": _collect_promotions,
    }
    projections = [handlers[kind](path, limit_per_source) for kind, path in collectors]
    sources = [item["source"] for item in projections]
    if len({item["source_id"] for item in sources}) != len(sources):
        raise EvidencePlaneError("duplicate evidence source")
    raw_events = [event for item in projections for event in item["events"]]
    raw_events.sort(key=lambda item: (
        item["occurred_at"] is None, item["occurred_at"] or "", item["source_id"], item["ordinal"],
    ))
    chained_events: list[dict[str, Any]] = []
    previous = "GENESIS"
    for event in raw_events:
        value = {**event, "previous_projection_digest": previous}
        value["projection_digest"] = sha256_json(value)
        previous = value["projection_digest"]
        chained_events.append(value)
    incidents = sorted(
        [incident for item in projections for incident in item["incidents"]],
        key=lambda item: (
            {"critical": 0, "high": 1, "medium": 2, "low": 3}[item["severity"]],
            item["occurred_at"] is None, item["occurred_at"] or "", item["incident_id"],
        ),
    )
    source_metrics = {
        item["source"]["source_id"]: {
            "kind": item["source"]["kind"], **item["metrics"],
        }
        for item in projections
    }
    integrity_counts = Counter(item["integrity"] for item in sources)
    incident_counts = Counter(item["severity"] for item in incidents)
    source_kind = {item["source_id"]: item["kind"] for item in sources}
    incident_clusters = Counter(
        (source_kind[item["source_id"]], item["severity"], item["code"])
        for item in incidents
    )
    decision_metrics = [
        item["metrics"] for item in projections if item["source"]["kind"] == "decision"
    ]
    promotion_metrics = [
        item["metrics"] for item in projections if item["source"]["kind"] == "promotion"
    ]
    drift_signals = {
        "unverified_or_invalid_sources": (
            integrity_counts["unverified"] + integrity_counts["invalid"]
        ),
        "l1_target_mismatches": sum(
            item["observations"] - item["target_matches"] for item in decision_metrics
        ),
        "l1_argument_mismatches": sum(
            item["argument_observed"] - item["argument_exact"] for item in decision_metrics
        ),
        "l1_safety_escapes": sum(item["safety_escapes"] for item in decision_metrics),
        "invalid_promotions": sum(item["invalid"] for item in promotion_metrics),
    }
    body = {
        "apiVersion": EVIDENCE_PLANE_SCHEMA,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "status": (
            "degraded"
            if integrity_counts["invalid"] or integrity_counts["unverified"]
            else "valid"
        ),
        "sources": sources,
        "events": chained_events,
        "incidents": incidents,
        "metrics": {
            "source_count": len(sources), "event_count": len(chained_events),
            "incident_count": len(incidents),
            "integrity_counts": dict(sorted(integrity_counts.items())),
            "incident_counts": dict(sorted(incident_counts.items())),
            "incident_clusters": [
                {"source_kind": key[0], "severity": key[1], "code": key[2], "count": count}
                for key, count in sorted(incident_clusters.items())
            ],
            "drift_signals": drift_signals,
            "by_source": source_metrics,
        },
        "projection_head": previous,
        "privacy": {
            "raw_prompts": False, "argument_values": False, "approval_identities": False,
            "provider_payloads": False, "filesystem_paths": False,
        },
        "authority": {
            "observation_only": True, "approval": False, "execution": False,
            "registration": False, "activation": False,
        },
        "claim_boundary": (
            "Digest-bound local evidence projection and operational indicators only; "
            "not a production SLO, immutable external audit store, or success probability."
        ),
    }
    return {**body, "snapshot_digest": sha256_json(body)}


def verify_evidence_snapshot(snapshot: dict[str, Any]) -> None:
    if snapshot.get("apiVersion") != EVIDENCE_PLANE_SCHEMA:
        raise EvidencePlaneError("evidence snapshot schema is unsupported")
    body = dict(snapshot)
    declared = body.pop("snapshot_digest", None)
    if declared != sha256_json(body):
        raise EvidencePlaneError("evidence snapshot digest is invalid")


def _trend_point(snapshot: dict[str, Any]) -> dict[str, Any]:
    metrics = snapshot["metrics"]
    by_source = list(metrics["by_source"].values())
    runtime = [item for item in by_source if item["kind"] == "runtime"]
    decisions = [item for item in by_source if item["kind"] == "decision"]
    runtime_states: Counter[str] = Counter()
    rollback_attempts = 0
    rollback_verified = 0
    for item in runtime:
        runtime_states.update(item["state_counts"])
        rollback_attempts += int(item["rollback_attempts"])
        rollback_verified += int(item["rollback_verified"])
    terminal = sum(runtime_states[value] for value in (
        "verified_success", "rollback_verified", "precondition_changed",
        "manual_intervention_required", "rejected", "expired",
    ))
    observation_count = sum(int(item["observations"]) for item in decisions)
    target_matches = sum(int(item["target_matches"]) for item in decisions)
    argument_observed = sum(int(item["argument_observed"]) for item in decisions)
    argument_exact = sum(int(item["argument_exact"]) for item in decisions)

    def maximum(items: list[dict[str, Any]], field: str) -> float | None:
        values = [float(item[field]) for item in items if item.get(field) is not None]
        return round(max(values), 3) if values else None

    return {
        "snapshot_digest": snapshot["snapshot_digest"],
        "generated_at": snapshot["generated_at"],
        "status": snapshot["status"],
        "incidents": int(metrics["incident_count"]),
        "critical_incidents": int(metrics["incident_counts"].get("critical", 0)),
        "degraded_sources": int(
            metrics["integrity_counts"].get("invalid", 0)
            + metrics["integrity_counts"].get("unverified", 0)
        ),
        "l1_target_mismatches": int(metrics["drift_signals"]["l1_target_mismatches"]),
        "l1_argument_mismatches": int(metrics["drift_signals"]["l1_argument_mismatches"]),
        "invalid_promotions": int(metrics["drift_signals"]["invalid_promotions"]),
        "runtime_verified_success_rate": (
            round(runtime_states["verified_success"] / terminal, 6) if terminal else None
        ),
        "runtime_rollback_success_rate": (
            round(rollback_verified / rollback_attempts, 6) if rollback_attempts else None
        ),
        "runtime_p50_max_ms": maximum(runtime, "duration_p50_ms"),
        "runtime_p95_max_ms": maximum(runtime, "duration_p95_ms"),
        "l1_target_match_rate": (
            round(target_matches / observation_count, 6) if observation_count else None
        ),
        "l1_argument_exact_rate": (
            round(argument_exact / argument_observed, 6) if argument_observed else None
        ),
        "l1_safety_escapes": sum(int(item["safety_escapes"]) for item in decisions),
        "l1_model_p50_max_ms": maximum(decisions, "model_p50_ms"),
        "l1_model_p95_max_ms": maximum(decisions, "model_p95_ms"),
    }


def analyze_evidence_trend(snapshots: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(snapshots)
    if len(values) < 2:
        raise EvidencePlaneError("evidence trend requires at least two snapshots")
    for snapshot in values:
        verify_evidence_snapshot(snapshot)
    if len({item["snapshot_digest"] for item in values}) != len(values):
        raise EvidencePlaneError("evidence trend snapshots must be unique")
    try:
        timestamps = [_parse_time(str(item["generated_at"])) for item in values]
    except (KeyError, TypeError) as error:
        raise EvidencePlaneError("evidence trend timestamp is invalid") from error
    if any(item is None for item in timestamps):
        raise EvidencePlaneError("evidence trend timestamp is invalid")
    values = [
        item for _, item in sorted(
            zip(timestamps, values), key=lambda pair: pair[0],  # type: ignore[arg-type]
        )
    ]
    points = [_trend_point(item) for item in values]
    first, last = points[0], points[-1]
    regression_reasons: list[str] = []
    improvement_reasons: list[str] = []
    for field in (
        "incidents", "critical_incidents", "degraded_sources",
        "l1_target_mismatches", "l1_argument_mismatches",
        "l1_safety_escapes", "invalid_promotions",
    ):
        if last[field] > first[field]:
            regression_reasons.append(f"{field}_increased")
        elif last[field] < first[field]:
            improvement_reasons.append(f"{field}_decreased")
    if first["status"] == "valid" and last["status"] != "valid":
        regression_reasons.append("snapshot_status_degraded")
    elif first["status"] != "valid" and last["status"] == "valid":
        improvement_reasons.append("snapshot_status_recovered")
    status = (
        "regressed" if regression_reasons
        else ("improved" if improvement_reasons else "stable")
    )
    numeric_fields = [
        key for key, value in first.items()
        if key not in {"snapshot_digest", "generated_at", "status"}
        and isinstance(value, (int, float)) and isinstance(last.get(key), (int, float))
    ]
    body = {
        "apiVersion": EVIDENCE_TREND_SCHEMA,
        "status": status,
        "points": points,
        "deltas": {
            key: round(float(last[key]) - float(first[key]), 6) for key in numeric_fields
        },
        "regression_reasons": sorted(set(regression_reasons)),
        "improvement_reasons": sorted(set(improvement_reasons)),
        "authority": "observation_only",
        "activation_available": False,
        "production_slo_claim_available": False,
        "claim_boundary": (
            "Local digest-bound indicator trend only; latency changes have no automatic "
            "SLO classification and require controlled-environment review."
        ),
    }
    return {**body, "trend_digest": sha256_json(body)}


_HTML = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="referrer" content="no-referrer"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data:"><title>NetOpYu Evidence Plane</title><style>
:root{color-scheme:dark;--bg:#071019;--panel:#101d2a;--line:#274158;--text:#e8f1f8;--muted:#91a9ba;--ok:#4ade80;--warn:#fbbf24;--bad:#fb7185;--accent:#38bdf8}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 10% 0,#12304a,#071019 48%);color:var(--text);font:14px/1.5 system-ui;padding:26px}.wrap{max-width:1280px;margin:auto}.panel{background:#101d2aed;border:1px solid var(--line);border-radius:14px;padding:18px;margin-bottom:15px;box-shadow:0 18px 45px #0004}h1,h2{margin:0 0 10px}h1{font-size:25px}h2{font-size:17px}.muted{color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}.card{background:#091522;border:1px solid var(--line);border-radius:11px;padding:14px}.value{font-size:25px;font-weight:700}.ok{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--warn)}table{border-collapse:collapse;width:100%;font-size:12px}th,td{border-bottom:1px solid var(--line);padding:8px;text-align:left;vertical-align:top}code{color:#bae6fd}input{width:100%;background:#071019;color:var(--text);border:1px solid var(--line);padding:9px;border-radius:8px;margin:6px 0 12px}.pill{border:1px solid var(--line);border-radius:99px;padding:3px 8px;display:inline-block}.notice{border-left:4px solid var(--warn)}.scroll{overflow:auto;max-height:520px}
</style></head><body><main class="wrap"><section class="panel"><h1>NetOpYu Evidence Plane</h1><p class="muted">P2.2 本地只读证据面 / local read-only evidence plane</p><div id="cards" class="cards"></div></section><section class="panel notice"><strong>Observation only.</strong> 本页面没有审批、执行、注册或激活 API；指标不是生产 SLO 或成功概率。</section><section class="panel"><h2>Sources</h2><div class="scroll"><table><thead><tr><th>Kind</th><th>Integrity</th><th>Records</th><th>Truncated</th></tr></thead><tbody id="sources"></tbody></table></div></section><section class="panel"><h2>Incidents</h2><div class="scroll"><table><thead><tr><th>Severity</th><th>Code</th><th>Time</th><th>Correlation digest</th></tr></thead><tbody id="incidents"></tbody></table></div></section><section class="panel"><h2>Timeline</h2><input id="filter" placeholder="Filter event type, category, state, outcome"><div class="scroll"><table><thead><tr><th>Time</th><th>Source</th><th>Category</th><th>Event</th><th>State / outcome</th><th>Correlation</th></tr></thead><tbody id="timeline"></tbody></table></div></section></main><script id="evidence-data" type="application/json">__DATA__</script><script>
'use strict';const d=JSON.parse(document.getElementById('evidence-data').textContent);const el=id=>document.getElementById(id);const node=(tag,text,cls)=>{const n=document.createElement(tag);n.textContent=String(text);if(cls)n.className=cls;return n};const cards=[['Status',d.status],['Sources',d.metrics.source_count],['Events',d.metrics.event_count],['Incidents',d.metrics.incident_count],['Projection head',d.projection_head.slice(0,18)+'…']];for(const [k,v] of cards){const c=node('div','', 'card');c.append(node('div',k,'muted'));c.append(node('div',v,'value '+(k==='Status'?(v==='valid'?'ok':'bad'):'')));el('cards').append(c)}for(const s of d.sources){const tr=node('tr','');for(const v of [s.kind,s.integrity,s.records,s.truncated])tr.append(node('td',v));el('sources').append(tr)}for(const i of d.incidents){const tr=node('tr','');for(const v of [i.severity,i.code,i.occurred_at??'n/a',i.correlation_digest.slice(0,22)+'…'])tr.append(node('td',v));el('incidents').append(tr)}const render=q=>{el('timeline').replaceChildren();for(const e of d.events){const hay=JSON.stringify([e.event_type,e.category,e.state,e.outcome]).toLowerCase();if(q&&!hay.includes(q))continue;const tr=node('tr','');for(const v of [e.occurred_at??'n/a',e.source_kind,e.category,e.event_type,[e.state,e.outcome].filter(Boolean).join(' / '),e.correlation_digest.slice(0,22)+'…'])tr.append(node('td',v));el('timeline').append(tr)}};render('');el('filter').addEventListener('input',event=>render(event.target.value.trim().toLowerCase()));
</script></body></html>"""


def render_evidence_html(snapshot: dict[str, Any]) -> str:
    verify_evidence_snapshot(snapshot)
    encoded = json.dumps(
        snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    return _HTML.replace("__DATA__", encoded)


def export_evidence_html(snapshot: dict[str, Any], output: str | Path) -> dict[str, Any]:
    supplied = Path(output).expanduser()
    if supplied.is_symlink():
        raise EvidencePlaneError("evidence HTML output target is unsafe")
    destination = supplied.resolve()
    if destination.exists() and not destination.is_file():
        raise EvidencePlaneError("evidence HTML output target is unsafe")
    rendered = render_evidence_html(snapshot)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered, encoding="utf-8")
    return {
        "ok": True, "output": str(destination),
        "snapshot_digest": snapshot["snapshot_digest"],
        "html_sha256": "sha256:" + sha256(rendered.encode("utf-8")).hexdigest(),
        "authority": "observation_only", "activation_available": False,
    }


__all__ = [
    "EVIDENCE_EVENT_SCHEMA",
    "EVIDENCE_INCIDENT_SCHEMA",
    "EVIDENCE_PLANE_SCHEMA",
    "EVIDENCE_TREND_SCHEMA",
    "EvidencePlaneError",
    "analyze_evidence_trend",
    "collect_evidence_snapshot",
    "export_evidence_html",
    "render_evidence_html",
    "verify_evidence_snapshot",
]
