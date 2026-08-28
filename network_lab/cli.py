"""Operator CLI for a bounded, manifest-owned Containerlab lifecycle."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from .containerlab import ContainerlabProvider, LabCommandError
from .manifest import load_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "labs" / "p075-a-frr" / "lab.yaml"


def _provider(path: str, timeout: float) -> ContainerlabProvider:
    return ContainerlabProvider(load_manifest(path), command_timeout=timeout)


def _approval(args: argparse.Namespace) -> None:
    if not args.approve_local_lab:
        raise PermissionError(
            "this command changes the disposable local lab; pass --approve-local-lab explicitly"
        )


async def _verify(provider: ContainerlabProvider) -> dict[str, Any]:
    status = await provider.topology_status()
    if not status["ok"]:
        return {"ok": False, "status": status, "errors": ["not all lab nodes are running"]}
    bgp_devices = {
        device_id: device
        for device_id, device in provider.manifest.devices.items()
        if device.expected_bgp_neighbors > 0
    }
    neighbors: dict[str, str] = {}
    neighbor_checks: dict[str, bool] = {}
    bgp: dict[str, Any] = {}
    bgp_checks: dict[str, bool] = {}
    # A freshly deployed multi-node topology needs time for both protocols to
    # converge. Verify the control plane before spending time on data probes.
    for attempt in range(31):
        neighbors = {
            device_id: await provider.show(device_id, "show ip ospf neighbor")
            for device_id in sorted(provider.manifest.devices)
        }
        neighbor_checks = {
            device_id: output.lower().count("full") >= provider.manifest.devices[
                device_id
            ].expected_ospf_neighbors
            for device_id, output in neighbors.items()
        }
        if provider.manifest.fabric:
            bgp = {
                device_id: await provider.fabric_bgp_evpn_summary(device_id)
                for device_id in sorted(bgp_devices)
            }
            bgp_checks = {
                device_id: bool(value["ok"])
                for device_id, value in bgp.items()
            }
        else:
            bgp = {
                device_id: await provider.show(device_id, "show bgp summary")
                for device_id in sorted(bgp_devices)
            }
            bgp_checks = {
                device_id: _established_bgp_neighbors(output)
                >= bgp_devices[device_id].expected_bgp_neighbors
                for device_id, output in bgp.items()
            }
        if all(neighbor_checks.values()) and all(bgp_checks.values()):
            break
        if attempt < 30:
            await asyncio.sleep(1)
    probes = {
        probe_id: await provider.probe(probe_id)
        for probe_id in sorted(provider.manifest.probes)
    }
    route = ""
    primary_selected = True
    if "branch-r1" in provider.manifest.devices:
        route = await provider.show("branch-r1", "show ip route 10.20.20.0/24")
        primary_selected = "10.0.0.2" in route and "eth2" in route
    fabric = await provider.fabric_state() if provider.manifest.fabric else None
    checks = {
        "all_nodes_running": bool(status["ok"]),
        "ospf_expected_neighbors": all(neighbor_checks.values()),
        "bgp_expected_neighbors": all(bgp_checks.values()),
        "all_manifest_probe_expectations": all(
            bool(item["ok"]) is provider.manifest.probes[probe_id].expected
            for probe_id, item in probes.items()
        ),
        "primary_wan_selected": primary_selected,
        "fabric_contract_and_state": fabric is None or bool(fabric["ok"]),
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "status": status,
        "ospf": neighbor_checks,
        "bgp": bgp_checks,
        "probes": probes,
        "route_evidence": route,
        "fabric": fabric,
    }


def _established_bgp_neighbors(output: str) -> int:
    """Count only BGP summary rows whose final field is a numeric prefix count."""
    established = 0
    for raw in output.splitlines():
        fields = raw.split()
        if len(fields) >= 10 and fields[0][:1].isdigit() and fields[9].isdigit():
            established += 1
    return established


async def _exercise_failover(provider: ContainerlabProvider) -> dict[str, Any]:
    baseline = await _verify(provider)
    if not baseline["ok"]:
        return {"ok": False, "phase": "baseline", "baseline": baseline}
    started = time.monotonic()
    await provider.set_fault("primary-wan-branch", kind="link_down")
    failover_probe: dict[str, Any] = {"ok": False}
    route = ""
    try:
        for _ in range(12):
            await asyncio.sleep(1)
            failover_probe = await provider.probe("branch-to-dc")
            route = await provider.show("branch-r1", "show ip route 10.20.20.0/24")
            if failover_probe["ok"] and "10.0.0.6" in route and "eth3" in route:
                break
    finally:
        await provider.set_fault("primary-wan-branch", kind="link_up")
    recovery: dict[str, Any] = {"ok": False}
    recovered_route = ""
    for _ in range(12):
        await asyncio.sleep(1)
        recovery = await provider.probe("branch-to-dc")
        recovered_route = await provider.show("branch-r1", "show ip route 10.20.20.0/24")
        if recovery["ok"] and "10.0.0.2" in recovered_route and "eth2" in recovered_route:
            break
    checks = {
        "baseline_passed": bool(baseline["ok"]),
        "backup_probe_passed": bool(failover_probe["ok"]),
        "backup_route_selected": "10.0.0.6" in route and "eth3" in route,
        "primary_link_recovered": bool(recovery["ok"]),
        "primary_route_restored": "10.0.0.2" in recovered_route and "eth2" in recovered_route,
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "convergence_seconds": round(time.monotonic() - started, 3),
        "failover_probe": failover_probe,
        "recovery_probe": recovery,
        "failover_route_evidence": route,
        "recovered_route_evidence": recovered_route,
    }


async def _exercise_fabric_failover(provider: ContainerlabProvider) -> dict[str, Any]:
    """Remove one leaf-to-spine path and prove EVPN L2 service plus recovery."""
    fabric = provider.manifest.fabric
    if fabric is None:
        raise LabCommandError("exercise-fabric-failover requires a fabric manifest")
    required = {"leaf1-spine1", "leaf1-spine2"}
    if not required.issubset(provider.manifest.fault_targets):
        raise LabCommandError("fabric failover targets leaf1-spine1/leaf1-spine2 are not declared")
    baseline = await _verify(provider)
    if not baseline["ok"]:
        return {"ok": False, "phase": "baseline", "baseline": baseline}
    started = time.monotonic()
    await provider.set_fault("leaf1-spine1", kind="link_down")
    degraded_summary: dict[str, Any] = {}
    degraded_probes: dict[str, Any] = {}
    try:
        for _ in range(20):
            await asyncio.sleep(1)
            degraded_summary = await provider.fabric_bgp_evpn_summary("leaf-1")
            degraded_probes = {
                probe_id: await provider.probe(probe_id)
                for probe_id in ("tenant-a-l2vpn", "tenant-b-l2vpn")
                if probe_id in provider.manifest.probes
            }
            if (
                int(degraded_summary.get("established_neighbors", 0)) >= 1
                and all(item.get("ok") is True for item in degraded_probes.values())
            ):
                break
    finally:
        await provider.set_fault("leaf1-spine1", kind="link_up")
    recovered_summary: dict[str, Any] = {}
    recovery_probes: dict[str, Any] = {}
    for _ in range(20):
        await asyncio.sleep(1)
        recovered_summary = await provider.fabric_bgp_evpn_summary("leaf-1")
        recovery_probes = {
            probe_id: await provider.probe(probe_id)
            for probe_id in ("tenant-a-l2vpn", "tenant-b-l2vpn")
            if probe_id in provider.manifest.probes
        }
        if (
            recovered_summary.get("ok") is True
            and all(item.get("ok") is True for item in recovery_probes.values())
        ):
            break
    checks = {
        "baseline_passed": bool(baseline["ok"]),
        "one_evpn_path_remained": int(degraded_summary.get("established_neighbors", 0)) >= 1,
        "l2vpn_survived_single_spine_loss": (
            bool(degraded_probes) and all(item.get("ok") is True for item in degraded_probes.values())
        ),
        "all_evpn_peers_recovered": recovered_summary.get("ok") is True,
        "l2vpn_recovered": (
            bool(recovery_probes) and all(item.get("ok") is True for item in recovery_probes.values())
        ),
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "convergence_seconds": round(time.monotonic() - started, 3),
        "degraded_bgp": degraded_summary,
        "degraded_probes": degraded_probes,
        "recovered_bgp": recovered_summary,
        "recovery_probes": recovery_probes,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    provider = _provider(args.manifest, args.timeout)
    if args.command == "preflight":
        return await provider.preflight()
    if args.command == "deploy":
        _approval(args)
        return {"ok": True, "output": await provider.deploy(reconfigure=args.reconfigure)}
    if args.command == "status":
        result = await provider.topology_status()
        return result
    if args.command == "verify":
        return await _verify(provider)
    if args.command == "exercise-failover":
        _approval(args)
        return await _exercise_failover(provider)
    if args.command == "exercise-fabric-failover":
        _approval(args)
        return await _exercise_fabric_failover(provider)
    if args.command == "fault":
        _approval(args)
        output = await provider.set_fault(args.fault_id, kind=args.kind, value=args.value)
        return {"ok": True, "fault_id": args.fault_id, "kind": args.kind, "output": output}
    if args.command == "destroy":
        _approval(args)
        return {"ok": True, "output": await provider.destroy()}
    raise AssertionError(f"unhandled command {args.command}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    result.add_argument("--timeout", type=float, default=30.0)
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight")
    commands.add_parser("status")
    commands.add_parser("verify")
    deploy = commands.add_parser("deploy")
    deploy.add_argument("--approve-local-lab", action="store_true")
    deploy.add_argument("--reconfigure", action="store_true")
    exercise = commands.add_parser("exercise-failover")
    exercise.add_argument("--approve-local-lab", action="store_true")
    fabric_exercise = commands.add_parser("exercise-fabric-failover")
    fabric_exercise.add_argument("--approve-local-lab", action="store_true")
    fault = commands.add_parser("fault")
    fault.add_argument("fault_id")
    fault.add_argument(
        "kind", choices=("link_down", "link_up", "delay_ms", "loss_pct", "clear_netem"),
    )
    fault.add_argument("--value", type=int)
    fault.add_argument("--approve-local-lab", action="store_true")
    destroy = commands.add_parser("destroy")
    destroy.add_argument("--approve-local-lab", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        report = asyncio.run(run(args))
    except (PermissionError, LabCommandError, ValueError, OSError) as error:
        report = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1
