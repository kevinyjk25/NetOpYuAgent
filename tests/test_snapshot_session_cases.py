"""Synthetic protocol fixtures remain distinct from independent Skill evidence."""
from evaluation.snapshot_session_cases import export, specs
from skill_authoring.artifacts import read_json
from skill_authoring.compiler import validate_packet


def test_export_keeps_expected_answers_out_of_model_packet(tmp_path):
    output = tmp_path / "cases"
    export(output)
    manifest = read_json(output / "manifest.json")
    assert manifest["skills"] == 1 and len(manifest["tasks"]) == 2
    assert manifest["fixtureVersion"] == 2
    assert "quantify packet loss" in specs()["circuit-recovery"]["task"]
    assert not manifest["formalStageExit"] and manifest["kind"] == "developer_synthetic_not_holdout"
    for name, spec in specs().items():
        packet = read_json(output / "cases" / name / "inputs/packet.json")
        validate_packet(packet)
        assert packet["task"] == spec["task"] and "expectations" not in packet
        assert list(packet["reads"]) == ["read_export"]


def test_distinct_resources_are_not_model_supplied_permissions():
    data = specs()["stock-difference"]
    resources = data["fixture"]["resources"]["read_export"]["resources"]
    assert len({r[0]["path"] for r in resources}) == 3
    assert all(set(r[0]) == {"path"} for r in resources)
    assert data["fixture"]["arguments"]["exportPath"] == resources[0][0]["path"]
