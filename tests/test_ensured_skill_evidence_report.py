import json
from pathlib import Path

from evaluation.ensured_skill_evidence_report import build_report


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _harness(model: str) -> dict:
    metrics = {
        "unsafeExecutionRate": 0.0,
        "falseCommitRate": 0.0,
        "taskCompletionRate": 90.0,
    }
    return {
        "schema": "effect-runtime.io/real-harness-auto-runtime-ab/v1",
        "dataset": {"repetitions": 3},
        "causalProtocol": {"sameModel": model},
        "routing": {"falseAccepts": 0},
        "metrics": {
            "dsh_l1_native": {**metrics, "unsafeExecutionRate": 10.0},
            "dsh_l0_auto_runtime": metrics,
        },
        "coverageEffectCurve": [{"runtimeRouted": 1}],
    }


def test_evidence_index_requires_mechanisms_repetition_and_two_models(
    tmp_path: Path,
) -> None:
    scenario = _write(tmp_path / "scenario.json", {
        "schema": "ensuredskill.io/es-p0-execution-report/v1",
        "summary": {
            "taskCompletionRate": 100.0, "unsafeExecutionRate": 0.0,
            "falseCommitRate": 0.0, "compensationSuccessRate": 100.0,
        },
    })
    full = {
        "taskCompletionRate": 100.0, "unsafeExecutionRate": 0.0,
        "invalidActionRate": 0.0, "compensationSuccessRate": 100.0,
    }
    ablation = _write(tmp_path / "ablation.json", {
        "schema": "ensuredskill.io/es-p0-ablation-report/v1",
        "variants": {
            "full": {"summary": full},
            "without_contract": {"summary": {**full, "unsafeExecutionRate": 20.0}},
            "without_evidence": {"summary": {**full, "unsafeExecutionRate": 20.0}},
            "without_guard": {"summary": {**full, "unsafeExecutionRate": 20.0}},
            "without_transaction": {"summary": {**full, "invalidActionRate": 20.0}},
            "without_compensation": {"summary": {**full, "compensationSuccessRate": 0.0}},
        },
    })
    main = _write(tmp_path / "main.json", _harness("qwen3.5:9b"))
    weak = _write(tmp_path / "weak.json", _harness("qwen2.5:7b"))

    report = build_report(
        scenario_report=scenario, ablation_report=ablation,
        main_harness_report=main, weak_harness_report=weak,
    )

    assert report["evidenceComplete"]
    assert report["safetyHypothesisSupported"]
    assert report["status"] == "local_hypothesis_supported"
    assert report["checks"]["sealedOrPrivateGeneralizationSet"] is False
    assert report["checks"]["realVendorDeviceQualification"] is False
