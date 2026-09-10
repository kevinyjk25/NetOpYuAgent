"""Evidence aggregation verifies bindings, not semantic review truth."""
import json

import pytest

from evaluation import stage1_evidence as evidence


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def bundle(tmp_path):
    root = tmp_path / "runs"
    review = {"protocol": "windowed-source-ledger/v37", "reviewKind": "developer_ai_not_independent_gold",
              "reviews": [], "limitations": ["Synthetic test records, not a real translation result."]}
    for case in ("wiring", "approval", "reference"):
        for repeat in ("a", "b"):
            name = f"{case}-v37-{repeat}"
            manifest = {"protocol": review["protocol"], "reportDigest": "manifest-" + name, "implementation": {"fixed": True},
                "model": {"model": "test-not-live", "digest": "test-digest"}, "policy": {"fixed": True},
                "profile": "plan_first", "initialState": {"modelReasoning": False, "semanticPlan": True}}
            save(root / name / "manifest.json", manifest)
            folder = root / name / "round-000"
            save(folder / "request.json", {})
            save(folder / "result.json", {"candidateStatus": "compiled_region_requires_semantic_review", "latencyMs": 10})
            save(folder / "compilation.json", {"treeDigest": "tree-" + name})
            save(root / ("behavior-" + name) / "report.json", {"treeDigest": "tree-" + name, "manifestDigest": manifest["reportDigest"],
                "passed": 8, "failed": 0, "sourceScriptCalls": 0, "effectCalls": 0, "networkCalls": 0})
            review["reviews"].append({"run": name, "treeDigest": "tree-" + name, "status": "accepted_read_region",
                "findings": ["synthetic explicit review marker"], "limitations": ["not real semantic evidence"]})
    review_path = tmp_path / "review.json"
    save(review_path, review)
    logs = {}
    for name in ("full", "targeted", "ruff", "docs", "diff"):
        path = tmp_path / (name + ".log")
        path.write_text("10 passed, 2 subtests passed in 1.23s" if name in {"full", "targeted"} else "PASS")
        logs[name] = str(path)
    return root, review_path, logs


def test_evidence_requires_all_cases_and_keeps_limits(tmp_path):
    root, review, logs = bundle(tmp_path)
    report = evidence.collect(root, review, logs)
    assert report["finalFirstConstructionAccepted"] == 6
    assert report["generatedRegionBehaviorPassed"] == 48
    assert report["wholeSkillAccuracy"] is None and not report["largeRuntimeABUnlocked"]
    assert report["receiptsWithUnknownTokenUsage"] == 6  # synthetic receipts deliberately omit token counts
    assert report["configuration"]["reasoning"] is False
    assert all(evidence.digest(path) == digest for path, digest in report["artifactDigests"].items())


@pytest.mark.parametrize("mutation", ["unreviewed", "wrong_tree", "failed_behavior", "changed_code", "changed_config", "tests_failed", "missing_repeat"])
def test_incomplete_or_unbound_success_is_rejected(tmp_path, mutation):
    root, review_path, logs = bundle(tmp_path)
    review = json.loads(review_path.read_text())
    name = review["reviews"][0]["run"]
    if mutation in {"unreviewed", "wrong_tree"}:
        review["reviews"][0]["status" if mutation == "unreviewed" else "treeDigest"] = "invalid"
        save(review_path, review)
    elif mutation == "failed_behavior":
        path = root / ("behavior-" + name) / "report.json"
        value = json.loads(path.read_text())
        value["failed"] = 1
        save(path, value)
    elif mutation in {"changed_code", "changed_config", "missing_repeat"}:
        path = root / name / "manifest.json"
        value = json.loads(path.read_text())
        if mutation == "changed_config":
            value["initialState"]["modelReasoning"] = True
        else:
            value["implementation" if mutation == "changed_code" else "protocol"] = "different"
        save(path, value)
    else:
        from pathlib import Path
        Path(logs["full"]).write_text("1 failed, 9 passed in 1.23s")
    with pytest.raises(ValueError):
        evidence.collect(root, review_path, logs)


def test_latency_interpolation_has_explicit_empty_behavior():
    assert evidence.percentile([], .5) is None
    assert evidence.percentile([10, 20], .5) == 15
    assert evidence.percentile([10, 20], .95) == 19.5
