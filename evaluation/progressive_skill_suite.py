"""Structural qualification suite for heterogeneous Anthropic Skill packages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from effect_runtime import inspect_skill_package


DEFAULT_SUITE = Path(__file__).parent / "fixtures" / "progressive-skills" / "cases.yaml"


def evaluate_progressive_skill_suite(path: str | Path = DEFAULT_SUITE) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    cases = raw.get("cases") or []
    results: list[dict[str, Any]] = []
    features: set[str] = set()
    domains: set[str] = set()
    passed = 0
    for case in cases:
        skill_path = manifest_path.parent / str(case["skillPath"])
        report = inspect_skill_package(
            skill_path, bound_scripts=tuple(case.get("boundScripts") or ()),
        )
        expected_gate = str(case["expectedPackageGate"])
        expected_finding = case.get("expectedFinding")
        observed_findings = {item["code"] for item in report["findings"]}
        oracle_pass = report["gate"] == expected_gate and (
            not expected_finding or expected_finding in observed_findings
        )
        passed += int(oracle_pass)
        features.update(str(value) for value in case.get("features") or ())
        domains.add(str(case["domain"]))
        results.append({
            "caseId": case["id"],
            "domain": case["domain"],
            "features": case.get("features") or [],
            "risk": case["risk"],
            "effectSemantics": case["effectSemantics"],
            "expectedPackageGate": expected_gate,
            "observedPackageGate": report["gate"],
            "expectedFinding": expected_finding,
            "observedFindings": sorted(observed_findings),
            "packageDigest": report["packageDigest"],
            "oraclePass": oracle_pass,
        })
    total = len(results)
    return {
        "schema": "effect-runtime.io/anthropic-skill-fixture-report/v1",
        "suiteSchema": raw.get("schema"),
        "modelBaseline": raw.get("modelBaseline"),
        "summary": {
            "cases": total,
            "passed": passed,
            "failed": total - passed,
            "oracleCoveragePercent": round(100.0 * passed / total, 2) if total else 0.0,
            "domains": len(domains),
            "featureTypes": len(features),
        },
        "domains": sorted(domains),
        "features": sorted(features),
        "cases": results,
        "claimBoundary": raw.get("claimBoundary"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default=str(DEFAULT_SUITE))
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    value = evaluate_progressive_skill_suite(args.suite)
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if value["summary"]["failed"] == 0 else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["DEFAULT_SUITE", "evaluate_progressive_skill_suite"]
