"""A fixed small final batch cannot be recast as unseen-Skill evidence."""
from pathlib import Path

from evaluation.convergence_session_cases import export
from evaluation.hybrid_session_acceptance import audit_expectations
from evaluation.snapshot_session_cases import export as snapshot_export
from skill_authoring.artifacts import read_json, write_artifacts
from skill_authoring.contracts import seal


def test_export_is_new_only_private_audit_and_known_skill_accounting(tmp_path):
    # Exercise fixture construction with inert unit Skill data, not real LLMs.
    snapshot_export(tmp_path / "snapshot")
    spec = read_json(tmp_path / "snapshot/specifications/circuit-recovery/specification.json")
    packet = read_json(tmp_path / "snapshot/cases/circuit-recovery/inputs/packet.json")
    spec.update(inputSchema=packet["inputSchema"], tools=packet["catalog"]["tools"])
    spec["task"] = (
        "distinguish implementation, effectiveness verification and closure; identify owners and the next outstanding evidence. "
        "Use IRQL to draft a bounded authentication-failure hunt from the captured function inventory; "
        "count failures per user, enrich with employee identity, show the ten largest counts. "
        "Do not execute the query or open an application. quantify error rate and p99 latency, scope affected traffic; "
        "distinguish symptom evidence from root cause; identify a useful next evidence source. at most three sections")
    packet["task"] = spec["task"]
    names = ["capa-status", "irql-draft", "mesh-incident"]
    write_artifacts(tmp_path / "old", {"freeze.json": seal({"inputsAndExpectations": {n: [spec, packet] for n in names}})})
    output = tmp_path / "new"
    export(tmp_path / "old/freeze.json", output)
    manifest = read_json(output / "manifest.json")
    assert manifest["taskCount"] == 6 and manifest["freshSkills"] == 0
    assert manifest["retries"] == 0 and not manifest["postRunTuningAllowed"]
    for name in manifest["cases"]:
        s = read_json(output / "specifications" / name / "specification.json")
        p = read_json(output / "cases" / name / "inputs/packet.json")
        assert len(s["expectations"]) == 4
        assert p["task"] == s["task"]
        assert "expectationAudit" not in p and "expectations" not in p
        assert audit_expectations(s)["calibrated"] is (name != "capa-status")
    before = (output / "manifest.json").read_bytes()
    import pytest
    with pytest.raises(FileExistsError):
        export(tmp_path / "old/freeze.json", output)
    assert (output / "manifest.json").read_bytes() == before


def test_fixture_generator_is_not_a_product_authoring_special_case():
    root = Path(__file__).resolve().parents[1]
    for folder in ["skill_authoring", "dsh_adapter", "network_runtime"]:
        assert not any("convergence_session_cases" in p.read_text() for p in (root / folder).rglob("*.py"))
