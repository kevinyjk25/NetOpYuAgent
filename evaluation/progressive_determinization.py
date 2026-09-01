"""Build a reproducible report for the generic progressive-determinization layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from effect_runtime import decide_progressive_execution, inspect_skill_package
from effect_runtime.promotion import assess_promotion
from evaluation.progressive_skill_suite import evaluate_progressive_skill_suite


ROOT = Path(__file__).resolve().parent.parent
ANCHOR = ROOT / "network_runtime" / "l0" / "promotion_examples" / "url1-network-access"
ANCHOR_CANDIDATE = ROOT / "network_runtime" / "l0" / "examples" / "s1-network-access-grant.yaml"


def _model_evidence(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    source = Path(path).expanduser()
    if not source.is_file():
        return {"available": False, "path": str(source)}
    raw = json.loads(source.read_text(encoding="utf-8"))
    return {
        "available": True,
        "schema": raw.get("schema"),
        "model": raw.get("model"),
        "modelArtifactDigest": raw.get("model_artifact_digest"),
        "status": raw.get("status"),
        "qualified": raw.get("qualified"),
        "dataset": raw.get("dataset"),
        "metrics": raw.get("metrics"),
        "latency": raw.get("latency"),
        "claimBoundary": raw.get("claimBoundary"),
    }


def build_progressive_report(
    *, model_report_path: str | Path | None = None,
) -> dict[str, Any]:
    package_suite = evaluate_progressive_skill_suite()
    assessment = assess_promotion(
        skill_path=ANCHOR / "SKILL.md",
        candidate_path=ANCHOR_CANDIDATE,
        capability_catalog_path=ANCHOR / "capabilities.yaml",
        l05_path=ANCHOR / "L0.5.yaml",
    ).report
    package = inspect_skill_package(ANCHOR)
    decision = decide_progressive_execution(
        assessment=assessment,
        package_report=package,
        risk="medium",
        effect_semantics="reversible",
        l0_active=True,
        l0_artifact_digest=str(assessment["candidate"]["compiledHash"]),
        repeat_stability=1.0,
        simulation_pass_rate=1.0,
    )
    semantic_summary = assessment["semanticCoverage"]["summary"]
    return {
        "schema": "effect-runtime.io/progressive-determinization-report/v1",
        "architecture": {
            "runtime": "Effect Runtime",
            "referenceProfile": "network",
            "domainBoundary": (
                "Promotion, package trust, routing and transaction controls are domain-neutral; "
                "network Capability contracts remain the first reference profile."
            ),
            "writeBoundary": "L1 direct writes are forbidden; writes require active L0 plus Effect Runtime.",
        },
        "heterogeneousSkillSuite": {
            "summary": package_suite["summary"],
            "domains": package_suite["domains"],
            "features": package_suite["features"],
            "artifact": "artifacts/progressive-determinization/skill-package-fixtures.json",
            "claimBoundary": package_suite["claimBoundary"],
        },
        "networkAnchor": {
            "promotionStatus": assessment["status"],
            "semanticGate": assessment["semanticCoverage"]["gate"],
            "semanticSummary": semantic_summary,
            "packageGate": package["gate"],
            "packageDigest": package["packageDigest"],
            "decision": decision,
        },
        "modelEvidence": _model_evidence(model_report_path),
        "claimBoundary": (
            "Fixture Oracle coverage, public reverse-bootstrap model results and routing scores "
            "are regression evidence, not production success probabilities."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-report")
    parser.add_argument(
        "--output", default="artifacts/progressive-determinization/report.json",
    )
    args = parser.parse_args(argv)
    value = build_progressive_report(model_report_path=args.model_report)
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")
    print(text, end="")
    decision = value["networkAnchor"]["decision"]
    return 0 if decision["route"] == "l0_runtime" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_progressive_report"]
