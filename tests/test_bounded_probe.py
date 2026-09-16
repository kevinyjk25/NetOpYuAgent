"""Probe artifacts are synthetic mechanism evidence, never model scores."""
import json

import pytest

from evaluation.bounded_probe import FAMILIES, main, run, scorer_probes
from evaluation.bounded_scoring import digest


def test_twelve_scorer_families_have_correct_controls_and_counterexamples():
    fixtures, rows = scorer_probes()
    assert len(fixtures) == len(rows) == 24
    assert {row["id"].split(":")[0] for row in rows} == set(FAMILIES)
    assert all(row["matched"] for row in rows), [row for row in rows if not row["matched"]]
    assert any(row["actual"].get("status") == "pass" for row in rows)
    assert any(row["actual"].get("status") == "measurement_invalid" for row in rows)


def test_cli_seals_artifacts_and_keeps_fixture_budget_outside_official_registry(tmp_path, capsys):
    target = tmp_path / "probe"
    assert main([str(target), "--skip-gateway"]) == 0
    report = json.loads((target / "report.json").read_text())
    fixtures = json.loads((target / "scorer-fixtures.json").read_text())
    assert report.pop("reportDigest") == digest(report)
    assert report["fixturesDigest"] == digest(fixtures)
    assert report["all_expected_decisions"] is True
    assert report["actualModelCalls"] == 0
    assert report["evidenceRole"] == "synthetic_measurement_probe_not_agent_benchmark"
    assert not any(report[key] for key in ("researchEvidenceEligible", "pilotQualified", "liveAdapterReady"))
    assert report["automaticEffectBridge"] == "not_tested"
    assert report["runtime36ProbeGate"] == "not_assessed"
    assert report["gateway"]["status"] == "not_tested"
    assert (target / "fixture-budget.sqlite3").is_file()
    assert report["budget"]["snapshot"]["study"]["status"] == "halted"
    assert report["budget"]["snapshot"]["usage"]["model_requests"] == 2
    assert all(check["matched"] for check in report["budget"]["checks"])
    assert json.loads(capsys.readouterr().out)["pilotQualified"] is False
    before = (target / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        run(target, include_gateway=False)
    assert (target / "report.json").read_bytes() == before


def test_existing_local_gateway_is_only_synthetic_mechanism_evidence(tmp_path):
    report = run(tmp_path / "with-gateway")
    assert report["gateway"]["matched"] is True
    assert report["gateway"]["status"] == "synthetic_mechanisms_only"
    assert report["gateway"]["automaticAuthoring"] == "not_implemented"
    assert report["gateway"]["report"]["productionCorrectnessProven"] is False
    assert report["actualModelCalls"] == 0 and report["all_expected_decisions"]
    assert report["pilotQualified"] is False
