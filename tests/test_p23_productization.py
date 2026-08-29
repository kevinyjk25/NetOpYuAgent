from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from evaluation.cockpit import render_convergence_html
from evaluation.convergence import ConvergenceReportError, load_convergence_snapshot
from productization.cli import main
from productization.integration import (
    IntegrationPackError,
    assess_integration_pack,
    load_integration_pack,
)


ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "examples" / "integration-rest-mcp" / "pack.yaml"
BASELINE = ROOT / "data" / "convergence_baseline.json"


def test_integration_pack_is_strict_proposal_only() -> None:
    pack = load_integration_pack(PACK)
    report = assess_integration_pack(pack)
    assert pack.purpose == "proposal_only"
    assert pack.activation_available is False
    assert report["interfaceContractValid"] is True
    assert report["runtimeBindingComplete"] is False
    assert report["status"] == "ready_for_l0_authoring"
    assert report["activationAvailable"] is False
    assert report["counts"] == {
        "providers": 2,
        "readCapabilities": 2,
        "writeCapabilities": 2,
        "writeCapabilitiesWithIndependentVerifier": 2,
        "reversibleWritesWithCompensation": 1,
    }


@pytest.mark.parametrize("mutation", ["secret", "missing-verifier", "missing-compensation"])
def test_integration_pack_rejects_unsafe_effect_contracts(tmp_path: Path, mutation: str) -> None:
    value = yaml.safe_load(PACK.read_text(encoding="utf-8"))
    if mutation == "secret":
        value["providers"][0]["authentication"]["token"] = "do-not-accept-values"
    elif mutation == "missing-verifier":
        value["capabilities"][2]["controls"]["verifierRef"] = "missing.read"
    else:
        value["capabilities"][2]["controls"].pop("compensationRef")
    path = tmp_path / "pack.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    with pytest.raises(IntegrationPackError):
        load_integration_pack(path)


def test_convergence_baseline_answers_the_three_product_questions() -> None:
    report = load_convergence_snapshot(BASELINE)
    assert report["answer"]["deterministicExecutionControls"] == "verified_on_core72"
    assert report["answer"]["semanticIntentConvergence"] == "qualified_on_fixed_set"
    assert report["answer"]["qualifiedModels"] == ["qwen2.5:7b"]
    assert report["answer"]["productionGeneralization"] == "not_proven"
    assert report["runtimeComparison"]["runtimeControlEffectiveness"]["passed"] == 64
    assert report["runtimeComparison"]["dshOnlyControlEffectiveness"]["passed"] == 5
    assert len(report["caseEvidence"]) == 368
    models = {item["model"]: item for item in report["models"]}
    assert models["qwen2.5:7b"]["qualified"] is True
    assert models["qwen3.6:27b"]["qualified"] is False
    assert models["qwen2.5:7b"]["failureLayers"] == {
        "parameter_grounding": 8, "passed": 168, "semantic_selection": 8,
    }
    assert models["qwen3.6:27b"]["failureLayers"]["protocol"] == 2


def test_convergence_case_projection_contains_no_prompts_or_argument_values() -> None:
    report = load_convergence_snapshot(BASELINE)
    forbidden = {"prompt", "query", "arguments", "expected_arguments", "raw_output", "error"}
    for case in report["caseEvidence"]:
        assert forbidden.isdisjoint({key.lower() for key in case})
        assert "inputTokens" not in case
        assert "outputDigest" not in case


def test_convergence_digest_and_html_are_fail_closed(tmp_path: Path) -> None:
    report = load_convergence_snapshot(BASELINE)
    html = render_convergence_html(report)
    assert "Content-Security-Policy" in html
    assert "connect-src 'none'" in html
    assert "Runtime 逐 Oracle 证据" in html
    assert "runtime-cases" in html
    assert "<button" not in html
    assert "http://" not in html and "https://" not in html
    assert report["snapshotDigest"] in html

    tampered = copy.deepcopy(report)
    tampered["answer"]["productionGeneralization"] = "proven"
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ConvergenceReportError):
        load_convergence_snapshot(path)


def test_product_cli_golden_paths_and_evaluation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["journeys"]) == 0
    journeys = json.loads(capsys.readouterr().out)
    assert journeys["recommendedOrder"] == ["understand", "local-demo", "integrate"]

    assert main(["integration-check", "--pack", str(PACK)]) == 0
    assessment = json.loads(capsys.readouterr().out)
    assert assessment["activationAvailable"] is False

    assert main(["evaluate", "--baseline", str(BASELINE), "--output-dir", str(tmp_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["productionGeneralization"] == "not_proven"
    assert (tmp_path / "convergence.json").is_file()
    assert (tmp_path / "cockpit.html").is_file()


def test_demo_requires_explicit_local_approval(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["demo", "--scenario", "l1-l0"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False
    assert "explicit" in report["error"]
