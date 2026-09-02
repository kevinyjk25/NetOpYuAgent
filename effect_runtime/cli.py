"""Domain-neutral CLI for Skill package and progressive-routing gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .progressive import (
    EffectSemantics,
    ModelConfidence,
    RiskTier,
    decide_progressive_execution,
)
from .skill_graph import SkillEdge, SkillLevel, SkillNode, validate_skill_graph
from .skill_package import inspect_skill_package


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect and route progressively determinized Effect Skills",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    package = sub.add_parser(
        "inspect-package", help="hash and inspect one Anthropic Skill folder without executing scripts",
    )
    package.add_argument("--skill", required=True)
    package.add_argument(
        "--bound-script", action="append", default=[], metavar="PATH=CAPABILITY_ID",
    )

    decide = sub.add_parser(
        "decide", help="apply hard gates and risk-tier confidence routing",
    )
    decide.add_argument("--assessment", required=True)
    package_input = decide.add_mutually_exclusive_group(required=True)
    package_input.add_argument("--skill")
    package_input.add_argument("--package-report")
    decide.add_argument(
        "--bound-script", action="append", default=[], metavar="PATH=CAPABILITY_ID",
    )
    decide.add_argument("--risk", choices=[item.value for item in RiskTier], required=True)
    decide.add_argument(
        "--effect-semantics",
        choices=[item.value for item in EffectSemantics],
        required=True,
    )
    decide.add_argument("--l0-active", action="store_true")
    decide.add_argument("--l0-artifact-digest")
    decide.add_argument("--l0-ref", action="append", default=[])
    decide.add_argument("--repeat-stability", type=float)
    decide.add_argument("--simulation-pass-rate", type=float)
    decide.add_argument("--activation-reviewed", action="store_true")
    decide.add_argument("--approval-control-available", action="store_true")
    decide.add_argument("--model-confidence", type=float)
    decide.add_argument("--model-calibrated", action="store_true")
    decide.add_argument("--model-artifact-digest")
    decide.add_argument("--calibration-digest")

    graph = sub.add_parser(
        "validate-graph", help="validate L1/L0 composition direction from a JSON graph",
    )
    graph.add_argument("graph")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "inspect-package":
        value = inspect_skill_package(args.skill, bound_scripts=args.bound_script)
    elif args.command == "validate-graph":
        raw = _read_json(args.graph)
        nodes = [
            SkillNode(
                skill_id=str(item["skill_id"]),
                level=SkillLevel(str(item["level"])),
                active=bool(item.get("active", False)),
                version=(str(item["version"]) if item.get("version") else None),
                artifact_digest=(
                    str(item["artifact_digest"])
                    if item.get("artifact_digest") else None
                ),
            )
            for item in raw.get("nodes", [])
        ]
        edges = [
            SkillEdge(source=str(item["source"]), target=str(item["target"]))
            for item in raw.get("edges", [])
        ]
        value = validate_skill_graph(nodes, edges)
    else:
        assessment = _read_json(args.assessment)
        package_report = (
            inspect_skill_package(args.skill, bound_scripts=args.bound_script)
            if args.skill else _read_json(args.package_report)
        )
        model = None
        if args.model_confidence is not None:
            model = ModelConfidence(
                score=args.model_confidence,
                calibrated=args.model_calibrated,
                model_artifact_digest=args.model_artifact_digest,
                calibration_digest=args.calibration_digest,
            )
        value = decide_progressive_execution(
            assessment=assessment,
            package_report=package_report,
            risk=args.risk,
            effect_semantics=args.effect_semantics,
            l0_active=args.l0_active,
            l0_artifact_digest=args.l0_artifact_digest,
            referenced_l0=args.l0_ref,
            repeat_stability=args.repeat_stability,
            simulation_pass_rate=args.simulation_pass_rate,
            activation_reviewed=args.activation_reviewed,
            approval_control_available=args.approval_control_available,
            model_confidence=model,
        )
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    gate = value.get("gate") or (value.get("hardGate") or {}).get("status")
    return 0 if gate != "blocked" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
