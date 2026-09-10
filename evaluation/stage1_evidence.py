"""Bind developer review and local Stage-1 evidence. Never infer semantic truth."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from evaluation.structured_authoring import seal
from evaluation.structured_binding_probe import read_json, write_artifacts


def digest(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * p
    index = int(position)
    return round(values[index] + (values[min(index + 1, len(values) - 1)] - values[index]) * (position - index), 3)


def collect(root, review_path, validation_paths):
    root = Path(root).resolve()
    review_path = Path(review_path).resolve()
    review = read_json(review_path)
    if review["reviewKind"] != "developer_ai_not_independent_gold":
        raise ValueError("Stage 1 cannot claim independent semantic review")
    rows, accepted, fingerprints, configurations, costs, all_costs = [], [], [], [], [], []
    decisions = {r["run"]: r for r in review["reviews"]}
    if len(decisions) != len(review["reviews"]):
        raise ValueError("duplicate review runs")
    for manifest_path in sorted(root.glob("*-v*/manifest.json")):
        name = manifest_path.parent.name
        if name.split("-", 1)[0] not in {"approval", "reference", "wiring"}:
            continue
        manifest = read_json(manifest_path)
        results = [read_json(p) for p in sorted(manifest_path.parent.glob("round-*/result.json"))]
        all_costs.extend(results)
        compiles = list(manifest_path.parent.glob("round-*/compilation.json"))
        requests = list(manifest_path.parent.glob("round-*/request.json"))
        row = {"run": name, "protocol": manifest["protocol"], "manifestDigest": manifest["reportDigest"],
               "recordedCalls": len(results), "unreceiptedRequests": len(requests) - len(results),
               "candidateStatus": results[-1]["candidateStatus"] if results else "interrupted_without_response",
               "compiledCandidates": len(compiles), "diagnostics": [r["diagnostic"] for r in results if "diagnostic" in r]}
        if manifest["protocol"] == review["protocol"]:
            decision = decisions.get(name)
            if not decision or decision["status"] != "accepted_read_region" or len(compiles) != 1:
                raise ValueError("all final-version runs require explicit successful region review")
            compiled = read_json(compiles[0])
            if decision["treeDigest"] != compiled["treeDigest"] or not decision["findings"] or not decision["limitations"]:
                raise ValueError("review is not bound to this generated candidate")
            behavior = read_json(root / ("behavior-" + name) / "report.json")
            if (behavior["treeDigest"] != decision["treeDigest"] or behavior["manifestDigest"] != manifest["reportDigest"]
                    or behavior["failed"] or behavior["passed"] < 8 or any(behavior[k] for k in ("sourceScriptCalls", "effectCalls", "networkCalls"))):
                raise ValueError("local generated-region behavior validation is incomplete or unsafe")
            if row["unreceiptedRequests"] or row["candidateStatus"] != "compiled_region_requires_semantic_review":
                raise ValueError("final construction has an incomplete/failed checkpoint")
            row.update(treeDigest=decision["treeDigest"], behaviorPassed=behavior["passed"],
                       developerReviewStatus=decision["status"])
            accepted.append(row)
            fingerprints.append(manifest["implementation"])
            configurations.append({"model": manifest["model"], "policy": manifest["policy"],
                "profile": manifest["profile"], "reasoning": manifest["initialState"]["modelReasoning"],
                "argumentReasoning": manifest["initialState"].get("modelArgumentReasoning", False),
                "semanticPlan": manifest["initialState"]["semanticPlan"]})
            for path, result in zip(sorted(manifest_path.parent.glob("round-*/result.json")), results, strict=True):
                wire = read_json(path.with_name("request.json")).get("wireRequest", {})
                messages = wire.get("messages", [])
                payload = json.loads(messages[-1]["content"]) if messages else {}
                phase = payload.get("authoringPhase", "operation_plan_without_arguments" if "programLanguage" in payload else "unreported")
                costs.append({**result, "phase": phase, "reasoningRequested": wire.get("think")})
        # Complete historical reports must replay byte-for-byte with their own
        # implementation. An interrupted request cannot be replayed as a result.
        original_report = root / ("report-" + name) / "report.json"
        if original_report.exists():
            replay_report = root / ("replay-" + name) / "report.json"
            if not replay_report.exists() or digest(original_report) != digest(replay_report):
                raise ValueError("missing or changed original-version replay: " + name)
            row["originalVersionReplayIdentical"] = True
        rows.append(row)
    counts = Counter(r["run"].split("-", 1)[0] for r in accepted)
    if counts != {"approval": 2, "reference": 2, "wiring": 2} or set(decisions) != {r["run"] for r in accepted}:
        raise ValueError("require all six final runs, two per known source family")
    if not all(f == fingerprints[0] for f in fingerprints):
        raise ValueError("final version is not frozen across all runs")
    if not all(c == configurations[0] for c in configurations):
        raise ValueError("final model configuration is not frozen across all runs")
    full = Path(validation_paths["full"]).read_text()
    targeted = Path(validation_paths["targeted"]).read_text()
    for text in (full, targeted):
        if not re.search(r"\d+ passed.* in [\d.]+s", text) or re.search(r"\d+ (failed|error)", text):
            raise ValueError("missing successful validation result")
    for name in ("ruff", "docs", "diff"):
        if "PASS" not in Path(validation_paths[name]).read_text():
            raise ValueError("validation command did not record success: " + name)
    paths = [p for p in root.rglob("*") if p.is_file() and "evidence-summary" not in p.parts]
    paths.extend([review_path, *(Path(p).resolve() for p in validation_paths.values())])
    latencies = [r["latencyMs"] for r in costs if r.get("latencyMs") is not None]
    def totals(items):
        return {key: sum(r.get(key) or 0 for r in items) for key in ("inputTokens", "outputTokens", "latencyMs")}
    phase_costs = {}
    for phase in sorted({r["phase"] for r in costs}):
        selected = [r for r in costs if r["phase"] == phase]
        times = [r["latencyMs"] for r in selected if r.get("latencyMs") is not None]
        phase_costs[phase] = {"calls": len(selected), **totals(selected),
            "p50Ms": percentile(times, .5), "p95Ms": percentile(times, .95),
            "reasoningRequested": sorted({r["reasoningRequested"] for r in selected}, key=str)}
    return seal({"stage": 1, "status": "ready_for_stage_2_known_public_development_batch",
        "protocol": review["protocol"], "implementation": fingerprints[0], "configuration": configurations[0],
        "reviewKind": review["reviewKind"], "uniqueKnownSourceFamilies": 3, "newPublicSkills": 0,
        "finalFirstConstructionAccepted": len(accepted), "finalFirstConstructionTotal": 6,
        "generatedRegionBehaviorPassed": sum(r["behaviorPassed"] for r in accepted),
        "finalModelCalls": len(costs), "finalCost": totals(costs),
        "finalPhaseCosts": phase_costs, "evidenceCollectorDigest": digest(__file__),
        "finalRequestLatencyMs": {"p50": percentile(latencies, .5), "p95": percentile(latencies, .95)},
        "allRuns": rows, "allRecordedCalls": sum(r["recordedCalls"] for r in rows),
        "allRecordedCost": totals(all_costs),
        "receiptsWithUnknownTokenUsage": sum(r.get("inputTokens") is None or r.get("outputTokens") is None for r in all_costs),
        "tokenAccounting": "provider_reported_counts_only; unknown receipts/requests and unreported reasoning are not zero cost",
        "unreceiptedRequests": sum(r["unreceiptedRequests"] for r in rows),
        "wholeSkillAccuracy": None, "unseenGeneralizationProven": False, "productionSuccessProbability": None,
        "largeRuntimeABUnlocked": False, "stage2Executed": False, "sourceScriptsExecuted": False,
        "limitations": review["limitations"], "validationPaths": validation_paths,
        "artifactDigests": {str(p): digest(p) for p in sorted(set(paths))}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--review", required=True)
    parser.add_argument("--validation", required=True, help="JSON mapping full/targeted/ruff/docs/diff to saved logs")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = collect(args.root, args.review, read_json(args.validation))
    write_artifacts(args.output, {"report.json": report})
    print(report["status"], report["reportDigest"])


if __name__ == "__main__":
    main()
