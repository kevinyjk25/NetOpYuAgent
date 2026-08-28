#!/usr/bin/env python3
"""Verify and fault-test the complete local small-production network."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MANIFEST = ROOT / "labs" / "p075-b-small-production" / "lab.yaml"


async def _reset(provider: Any) -> dict[str, Any]:
    for user_id, admitted in {
        "erin": False, "bob": True, "carol": True, "guest": True,
    }.items():
        await provider.set_user_admission(user_id, admitted=admitted)

    reviewed_policy = {
        ("erin", "crm"): False,
        ("bob", "crm"): True,
        ("guest", "crm"): False,
        ("guest", "wiki"): False,
        ("carol", "wiki"): True,
        ("guest", "portal"): True,
        ("erin", "monitoring"): False,
        ("bob", "monitoring"): False,
        ("carol", "monitoring"): False,
        ("guest", "monitoring"): False,
    }
    for (user_id, app_id), allowed in reviewed_policy.items():
        await provider.set_application_access(user_id, app_id, allowed=allowed)
    for fault_id in ("primary-internet-uplink", "backup-internet-uplink"):
        await provider.set_fault(fault_id, kind="link_up")
    return {
        "ok": True,
        "lab": provider.manifest.name,
        "erin_admitted": await provider.user_admitted("erin"),
        "bob_admitted": await provider.user_admitted("bob"),
        "guest_crm_blocked": await provider.application_access_blocked("guest", "crm"),
    }


async def _verify(provider: Any) -> dict[str, Any]:
    from network_lab.cli import _verify as verify_manifest

    manifest_report = await verify_manifest(provider)
    if not manifest_report["ok"]:
        return {"ok": False, "phase": "manifest-baseline", "manifest": manifest_report}

    app_probes = {
        "bob-to-crm-http": await provider.application_probe("bob", "crm"),
        "carol-to-wiki-http": await provider.application_probe("carol", "wiki"),
        "guest-to-crm-http": await provider.application_probe("guest", "crm"),
        "guest-to-portal-http": await provider.application_probe("guest", "portal"),
        "erin-to-crm-http": await provider.application_probe("erin", "crm"),
    }
    expected = {
        "bob-to-crm-http": True,
        "carol-to-wiki-http": True,
        "guest-to-crm-http": False,
        "guest-to-portal-http": True,
        "erin-to-crm-http": False,
    }
    app_checks = {
        probe_id: bool(result["ok"]) is expected[probe_id]
        for probe_id, result in app_probes.items()
    }
    internet_route = await provider.show(
        "campus-core-1", "show ip route 198.51.100.0/24",
    )
    return_route = await provider.show("isp-1", "show bgp ipv4 unicast 10.0.0.0/8")
    route_checks = {
        "primary_internet_path_selected": "10.0.0.22" in internet_route,
        "enterprise_aggregate_advertised": "192.0.2.2" in return_route,
    }
    checks = {**manifest_report["checks"], **app_checks, **route_checks}
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "manifest": manifest_report,
        "http": app_probes,
        "route_evidence": {
            "campus_to_internet": internet_route,
            "internet_to_enterprise": return_route,
        },
    }


async def _wait_for_path(
    provider: Any, *, next_hop: str, attempts: int = 30,
) -> tuple[bool, dict[str, Any], str]:
    probe: dict[str, Any] = {"ok": False}
    route = ""
    for _ in range(attempts):
        await asyncio.sleep(1)
        probe = await provider.probe("guest-to-internet")
        route = await provider.show("campus-core-1", "show ip route 198.51.100.0/24")
        if probe["ok"] and next_hop in route:
            return True, probe, route
    return False, probe, route


async def _exercise_failover(provider: Any) -> dict[str, Any]:
    baseline = await _verify(provider)
    if not baseline["ok"]:
        return {"ok": False, "phase": "baseline", "baseline": baseline}

    started = time.monotonic()
    await provider.set_fault("primary-internet-uplink", kind="link_down")
    backup_ok = False
    backup_probe: dict[str, Any] = {"ok": False}
    backup_route = ""
    backup_edge_route = ""
    try:
        backup_ok, backup_probe, backup_route = await _wait_for_path(
            provider, next_hop="10.0.0.18",
        )
        backup_edge_route = await provider.show(
            "campus-core-2", "show ip route 198.51.100.0/24",
        )
    finally:
        await provider.set_fault("primary-internet-uplink", kind="link_up")

    recovery_ok, recovery_probe, recovery_route = await _wait_for_path(
        provider, next_hop="10.0.0.22",
    )
    checks = {
        "baseline_passed": bool(baseline["ok"]),
        "backup_data_plane_passed": bool(backup_probe["ok"]),
        "backup_core_selected": backup_ok and "10.0.0.18" in backup_route,
        "backup_edge_selected": "10.0.0.34" in backup_edge_route,
        "primary_data_plane_recovered": bool(recovery_probe["ok"]),
        "primary_edge_restored": recovery_ok and "10.0.0.22" in recovery_route,
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "convergence_and_recovery_seconds": round(time.monotonic() - started, 3),
        "backup_probe": backup_probe,
        "recovery_probe": recovery_probe,
        "backup_route_evidence": backup_route,
        "backup_edge_route_evidence": backup_edge_route,
        "recovery_route_evidence": recovery_route,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    from network_lab import ContainerlabProvider, load_manifest

    provider = ContainerlabProvider(load_manifest(args.manifest), command_timeout=args.timeout)
    if args.command == "reset":
        if not args.approve_local_lab:
            raise PermissionError("reset requires --approve-local-lab")
        return await _reset(provider)
    if args.command == "verify":
        return await _verify(provider)
    if args.command == "exercise-failover":
        if not args.approve_local_lab:
            raise PermissionError("fault injection requires --approve-local-lab")
        return await _exercise_failover(provider)
    raise AssertionError(f"unhandled command: {args.command}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--timeout", type=float, default=30.0)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify")
    reset = commands.add_parser("reset")
    reset.add_argument("--approve-local-lab", action="store_true")
    failover = commands.add_parser("exercise-failover")
    failover.add_argument("--approve-local-lab", action="store_true")
    args = parser.parse_args()
    try:
        report = asyncio.run(run(args))
    except (PermissionError, RuntimeError, ValueError, OSError) as error:
        report = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
