"""Local CLI for P2.1 Catalog governance and P2.2 Evidence Plane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from network_runtime.catalog_control import (
    CatalogGovernanceError,
    bootstrap_runtime_governance_catalog,
    catalog_compatibility_report,
    dump_governance_catalog,
    evaluate_catalog_access,
    load_governance_catalog,
    validate_runtime_catalog_binding,
)
from network_runtime.contracts import sha256_json
from network_runtime.evidence_plane import (
    EVIDENCE_PLANE_SCHEMA,
    EvidencePlaneError,
    analyze_evidence_trend,
    collect_evidence_snapshot,
    export_evidence_html,
)


def _safe_write(path: str | Path, content: str) -> Path:
    supplied = Path(path).expanduser()
    if supplied.is_symlink():
        raise ValueError("output target is unsafe")
    destination = supplied.resolve()
    if destination.exists() and not destination.is_file():
        raise ValueError("output target is unsafe")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    return destination


def _sources(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runtime-journal", action="append", default=[])
    parser.add_argument("--decision-store", action="append", default=[])
    parser.add_argument("--saga-store", action="append", default=[])
    parser.add_argument("--provider-registry", action="append", default=[])
    parser.add_argument("--proposal-root", action="append", default=[])
    parser.add_argument("--limit-per-source", type=int, default=5000)


def _collect(args: argparse.Namespace) -> dict[str, Any]:
    return collect_evidence_snapshot(
        runtime_journals=args.runtime_journal,
        decision_stores=args.decision_store,
        saga_stores=args.saga_store,
        provider_registries=args.provider_registry,
        proposal_roots=args.proposal_root,
        limit_per_source=args.limit_per_source,
    )


def _load_snapshot(path: str | Path) -> dict[str, Any]:
    supplied = Path(path).expanduser()
    if supplied.is_symlink() or not supplied.is_file() or supplied.stat().st_size > 64_000_000:
        raise EvidencePlaneError("evidence snapshot is missing, unsafe, or oversized")
    try:
        value = json.loads(supplied.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidencePlaneError("evidence snapshot JSON is invalid") from error
    if not isinstance(value, dict) or value.get("apiVersion") != EVIDENCE_PLANE_SCHEMA:
        raise EvidencePlaneError("evidence snapshot schema is unsupported")
    body = dict(value)
    declared = body.pop("snapshot_digest", None)
    if declared != sha256_json(body):
        raise EvidencePlaneError("evidence snapshot digest is invalid")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="NetOpYu P2.1 Catalog governance and P2.2 read-only Evidence Plane",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    bootstrap = sub.add_parser(
        "catalog-bootstrap", help="project all activated L0 contracts into a governed catalog",
    )
    bootstrap.add_argument("--output", required=True)
    validate = sub.add_parser("catalog-validate", help="validate catalog and Runtime coverage")
    validate.add_argument("--catalog", required=True)
    authorize = sub.add_parser(
        "catalog-authorize", help="evaluate catalog workflow authority only (never Runtime authority)",
    )
    authorize.add_argument("--catalog", required=True)
    authorize.add_argument("--team", required=True)
    authorize.add_argument(
        "--action", required=True,
        choices=("discover", "bind_read", "propose_write", "review", "publish", "deprecate"),
    )
    authorize.add_argument("--capability", required=True)
    authorize.add_argument("--version", required=True)
    authorize.add_argument("--tenant", required=True)
    authorize.add_argument("--environment", required=True)
    diff = sub.add_parser("catalog-diff", help="report compatibility and consumer impact")
    diff.add_argument("--previous", required=True)
    diff.add_argument("--candidate", required=True)

    collect = sub.add_parser("evidence-collect", help="build a privacy-minimized JSON snapshot")
    _sources(collect)
    collect.add_argument("--output")
    export = sub.add_parser("evidence-export", help="export an offline Evidence Plane HTML page")
    _sources(export)
    export.add_argument("--output", required=True)
    export.add_argument("--snapshot-output")
    incident = sub.add_parser("evidence-incident", help="read one digest-only incident projection")
    incident.add_argument("--snapshot", required=True)
    incident.add_argument("--incident-id", required=True)
    trend = sub.add_parser(
        "evidence-trend", help="compare two or more unique digest-bound snapshots",
    )
    trend.add_argument("--snapshot", action="append", required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "catalog-bootstrap":
            catalog = bootstrap_runtime_governance_catalog()
            destination = _safe_write(args.output, dump_governance_catalog(catalog))
            print(json.dumps({
                "ok": True, "output": str(destination),
                "catalog_hash": catalog.catalog_hash,
                "capabilities": len(catalog.capabilities),
                "delegations": len(catalog.delegations),
                "activation_available": False,
            }, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.command == "catalog-validate":
            report = validate_runtime_catalog_binding(load_governance_catalog(args.catalog))
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
            return 0 if report["ok"] else 1
        if args.command == "catalog-authorize":
            decision = evaluate_catalog_access(
                load_governance_catalog(args.catalog), team_id=args.team,
                action=args.action, capability_id=args.capability, version=args.version,
                tenant=args.tenant, environment=args.environment,
            )
            print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))
            return 0 if decision["allowed"] else 1
        if args.command == "catalog-diff":
            report = catalog_compatibility_report(
                load_governance_catalog(args.previous),
                load_governance_catalog(args.candidate),
            )
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
            return 0 if report["compatible"] else 1
        if args.command == "evidence-collect":
            snapshot = _collect(args)
            rendered = json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            if args.output:
                destination = _safe_write(args.output, rendered)
                print(json.dumps({
                    "ok": True, "output": str(destination),
                    "snapshot_digest": snapshot["snapshot_digest"],
                    "status": snapshot["status"],
                }, ensure_ascii=False, indent=2, sort_keys=True))
            else:
                print(rendered, end="")
            return 0 if snapshot["status"] == "valid" else 1
        if args.command == "evidence-export":
            snapshot = _collect(args)
            result = export_evidence_html(snapshot, args.output)
            if args.snapshot_output:
                destination = _safe_write(
                    args.snapshot_output,
                    json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                )
                result["snapshot_output"] = str(destination)
            result["status"] = snapshot["status"]
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0 if snapshot["status"] == "valid" else 1
        if args.command == "evidence-incident":
            snapshot = _load_snapshot(args.snapshot)
            match = next(
                (item for item in snapshot["incidents"] if item["incident_id"] == args.incident_id),
                None,
            )
            if match is None:
                raise EvidencePlaneError("unknown evidence incident id")
            print(json.dumps(match, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.command == "evidence-trend":
            report = analyze_evidence_trend(_load_snapshot(item) for item in args.snapshot)
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
            return 1 if report["status"] == "regressed" else 0
    except (CatalogGovernanceError, EvidencePlaneError, KeyError, TypeError, ValueError) as error:
        print(json.dumps({
            "ok": False, "error": type(error).__name__, "message": str(error),
        }, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
