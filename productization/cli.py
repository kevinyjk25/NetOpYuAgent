"""Single user-facing entry point for learning, integrating, and evaluating."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from evaluation.cockpit import export_convergence_html
from evaluation.convergence import (
    ConvergenceReportError,
    build_convergence_report,
    load_convergence_snapshot,
    load_l1_report,
    load_runtime_report,
)
from network_runtime.catalog_control import load_governance_catalog
from .integration import (
    IntegrationPackError,
    assess_integration_pack,
    integration_pack_json_schema,
    load_integration_pack,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = PROJECT_ROOT / "data" / "capability_governance_catalog.yaml"
DEFAULT_CONVERGENCE = PROJECT_ROOT / "data" / "convergence_baseline.json"
DEFAULT_EVALUATION_OUTPUT = PROJECT_ROOT / "artifacts" / "convergence"


def _discover_command(name: str) -> str | None:
    discovered = shutil.which(name)
    if discovered:
        return discovered
    bundled = Path.home() / ".cache" / "codex-runtimes" / "codex-primary-runtime" / "dependencies"
    candidates = (
        bundled / "node" / "bin" / name,
        bundled / "bin" / "fallback" / name,
    )
    return next((str(path) for path in candidates if path.is_file()), None)


def _safe_write(path: str | Path, text: str) -> Path:
    supplied = Path(path).expanduser()
    if supplied.is_symlink():
        raise ValueError("output target is unsafe")
    destination = supplied.resolve()
    if destination.exists() and not destination.is_file():
        raise ValueError("output target is unsafe")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return destination


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _journeys() -> dict[str, Any]:
    journeys = [
        {
            "id": "understand",
            "goalZh": "先理解系统解决了什么，以及哪些仍未解决",
            "goalEn": "Understand what is controlled and what remains probabilistic",
            "prerequisites": ["Python dependencies installed"],
            "commands": [
                "scripts/netopyu doctor",
                "scripts/netopyu evaluate",
                "open artifacts/convergence/cockpit.html",
            ],
            "evidence": [
                "DSH-only versus Runtime control Oracles",
                "Per-model qualification, latency, and first-failure layers",
                "Prompt-free and argument-value-free case traces",
            ],
            "writes": "Only generated reports under artifacts/; no network/service effects",
        },
        {
            "id": "local-demo",
            "goalZh": "在临时 mock 状态中查看 L1 + L0 的审批、验证和审计闭环",
            "goalEn": "Walk through L1 + L0 approval, verification, and audit locally",
            "prerequisites": ["Core Python dependencies", "Explicit local-simulation approval"],
            "commands": [
                "scripts/netopyu demo --scenario l1-l0 --approve-local-simulation",
            ],
            "evidence": ["Plan and contract hashes", "Independent postconditions", "Hash-chain audit"],
            "writes": "Two temporary in-memory mock effects; state is restored and temp files are removed",
        },
        {
            "id": "integrate",
            "goalZh": "把自己的 MCP/REST/NETCONF/SSH/Controller 能力映射成受控 read/write",
            "goalEn": "Map external interfaces into controlled read/write capabilities",
            "prerequisites": ["Interface schemas", "Provider owner", "Verifier and compensation design"],
            "commands": [
                "scripts/netopyu integration-check --pack examples/integration-rest-mcp/pack.yaml",
                "scripts/netopyu capabilities",
                "scripts/netopyu-l0 promote ...",
                "scripts/netopyu-dsh retirement",
            ],
            "evidence": ["Strict Integration Pack assessment", "L1→L0.5→L0 trace", "Provider and Runtime gates"],
            "writes": "The check is proposal-only and cannot connect, register, activate, approve, or execute",
        },
    ]
    return {
        "apiVersion": "netopyu.io/golden-journeys/v1",
        "recommendedOrder": [item["id"] for item in journeys],
        "journeys": journeys,
    }


def _doctor() -> tuple[dict[str, Any], bool]:
    checks: list[dict[str, Any]] = []

    def add(check_id: str, ok: bool, detail: str, required_for: list[str]) -> None:
        checks.append({"id": check_id, "ok": ok, "detail": detail, "requiredFor": required_for})

    add("python", sys.version_info >= (3, 11), sys.version.split()[0], ["evaluation", "demo", "integration"])
    for module in ("pydantic", "yaml"):
        add(
            f"python-module-{module}", importlib.util.find_spec(module) is not None,
            "installed" if importlib.util.find_spec(module) else "missing",
            ["evaluation", "demo", "integration"],
        )
    add("netopyu-cli", (PROJECT_ROOT / "scripts" / "netopyu").is_file(), "launcher present", ["all"])
    add("dsh-launcher", (PROJECT_ROOT / "scripts" / "netopyu-dsh").is_file(), "launcher present; process not probed", ["dsh-ui"])
    node = _discover_command("node")
    pnpm = _discover_command("pnpm")
    ollama = _discover_command("ollama")
    containerlab = _discover_command("containerlab")
    add("node", node is not None, node or "not found", ["dsh-ui"])
    add("pnpm", pnpm is not None, pnpm or "not found", ["dsh-install"])
    add("ollama", ollama is not None, "binary present; server/model not probed" if ollama else "not found", ["live-model"])
    add("containerlab", containerlab is not None, containerlab or "not found", ["network-lab"])
    try:
        catalog = load_governance_catalog(DEFAULT_CATALOG)
        add("catalog", bool(catalog.capabilities), f"{len(catalog.capabilities)} governed capabilities", ["integration"])
    except Exception as error:  # diagnostic boundary
        add("catalog", False, f"{type(error).__name__}: {error}", ["integration"])
    try:
        snapshot = load_convergence_snapshot(DEFAULT_CONVERGENCE)
        add("evaluation-baseline", bool(snapshot.get("models")), f"{len(snapshot.get('caseEvidence', []))} redacted cases", ["evaluation"])
    except Exception as error:  # diagnostic boundary
        add("evaluation-baseline", False, f"{type(error).__name__}: {error}", ["evaluation"])
    by_id = {item["id"]: item["ok"] for item in checks}
    modes = {
        "evaluationReady": all(by_id.get(item, False) for item in ("python", "python-module-pydantic", "python-module-yaml", "evaluation-baseline")),
        "integrationReviewReady": all(by_id.get(item, False) for item in ("python", "python-module-pydantic", "python-module-yaml", "catalog")),
        "dshSourceReady": bool(by_id.get("dsh-launcher") and by_id.get("node")),
        "liveModelReady": bool(by_id.get("ollama")),
        "networkLabReady": bool(by_id.get("containerlab")),
    }
    required = modes["evaluationReady"] and modes["integrationReviewReady"]
    return {
        "apiVersion": "netopyu.io/product-doctor/v1",
        "ok": required,
        "modes": modes,
        "checks": checks,
        "boundaries": [
            "Read-only inspection: no endpoint, model server, DSH process, or lab was contacted.",
            "liveModelReady means the binary is discoverable, not that a model is qualified.",
            "networkLabReady means the CLI is discoverable, not that a topology is deployed.",
        ],
    }, required


def _capabilities(catalog_path: str | Path, profile: str | None) -> dict[str, Any]:
    catalog = load_governance_catalog(catalog_path)
    selected = [
        capability for capability in catalog.capabilities
        if profile is None or profile in capability.profiles
    ]
    return {
        "apiVersion": "netopyu.io/capability-view/v1",
        "catalogHash": catalog.catalog_hash,
        "profile": profile,
        "count": len(selected),
        "capabilities": [{
            "id": item.id,
            "version": item.version,
            "kind": item.kind,
            "domain": item.domain,
            "profiles": list(item.profiles),
            "lifecycle": item.lifecycle,
            "ownerTeam": item.owner_team,
            "stewardTeam": item.steward_team,
        } for item in selected],
        "authority": "catalog discovery only; no read/write, approval, publication, or activation authority",
    }


def _evaluate(args: argparse.Namespace) -> dict[str, Any]:
    custom = bool(args.runtime_report or args.l1_report)
    if custom and (not args.runtime_report or not args.l1_report):
        raise ValueError("custom evaluation requires one runtime report and at least one L1 report")
    if custom:
        report = build_convergence_report(
            load_runtime_report(args.runtime_report),
            [load_l1_report(path) for path in args.l1_report],
        )
        source = "custom-full-reports"
    else:
        report = load_convergence_snapshot(args.baseline)
        source = "source-controlled-redacted-baseline"
    output = Path(args.output_dir).expanduser().resolve()
    json_path = _safe_write(
        output / "convergence.json",
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    html_path = export_convergence_html(report, output / "cockpit.html")
    return {
        "ok": True,
        "source": source,
        "json": str(json_path),
        "html": str(html_path),
        "snapshotDigest": report["snapshotDigest"],
        "runtime": report["answer"]["deterministicExecutionControls"],
        "semantics": report["answer"]["semanticIntentConvergence"],
        "productionGeneralization": report["answer"]["productionGeneralization"],
        "models": [{
            "model": item["model"], "qualified": item["qualified"],
            "e2e": item["metrics"]["endToEndAccuracy"],
            "p50Ms": item["metrics"]["p50Ms"], "p95Ms": item["metrics"]["p95Ms"],
        } for item in report["models"]],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="NetOpYu product front door: learn, evaluate, integrate, and demo safely",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("journeys", help="show the three supported Golden Paths")
    sub.add_parser("doctor", help="read-only local readiness inspection")
    capabilities = sub.add_parser("capabilities", help="discover governed capabilities")
    capabilities.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    capabilities.add_argument("--profile", choices=("lan", "dc", "wan"))
    integration = sub.add_parser("integration-check", help="validate a proposal-only Integration Pack")
    integration.add_argument("--pack", required=True)
    integration.add_argument("--schema-output")
    evaluate = sub.add_parser("evaluate", help="export the unified convergence report and cockpit")
    evaluate.add_argument("--baseline", default=str(DEFAULT_CONVERGENCE))
    evaluate.add_argument("--runtime-report")
    evaluate.add_argument("--l1-report", action="append")
    evaluate.add_argument("--output-dir", default=str(DEFAULT_EVALUATION_OUTPUT))
    demo = sub.add_parser("demo", help="run an explicitly approved local walkthrough")
    demo.add_argument("--scenario", choices=("l1-l0",), default="l1-l0")
    demo.add_argument("--approve-local-simulation", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "journeys":
            _print(_journeys())
            return 0
        if args.command == "doctor":
            report, ok = _doctor()
            _print(report)
            return 0 if ok else 1
        if args.command == "capabilities":
            _print(_capabilities(args.catalog, args.profile))
            return 0
        if args.command == "integration-check":
            pack = load_integration_pack(args.pack)
            if args.schema_output:
                _safe_write(
                    args.schema_output,
                    json.dumps(integration_pack_json_schema(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                )
            _print(assess_integration_pack(pack))
            return 0
        if args.command == "evaluate":
            _print(_evaluate(args))
            return 0
        if args.command == "demo":
            if not args.approve_local_simulation:
                _print({
                    "ok": False,
                    "error": "explicit --approve-local-simulation is required",
                    "effects": "two temporary mock writes; no real endpoint is contacted",
                })
                return 2
            completed = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "scripts" / "l1_l0_demo.py"), "--approve-local-simulation"],
                cwd=PROJECT_ROOT,
                check=False,
            )
            return completed.returncode
    except (ConvergenceReportError, IntegrationPackError, ValueError, OSError) as error:
        _print({"ok": False, "error": f"{type(error).__name__}: {error}"})
        return 1
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
