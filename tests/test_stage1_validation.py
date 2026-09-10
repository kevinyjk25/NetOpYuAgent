"""The developer behavior oracle stays outside model input and cannot bless drift."""
import copy

import pytest

from evaluation import source_ledger as ledger, stage1_cases, stage1_validation
from evaluation.structured_binding_probe import write_artifacts
from tests.test_source_semantic_plan import anchored, prepared, slot_arguments, envelope, packet as packet_fixture

packet = packet_fixture


@pytest.mark.parametrize("name,tools", [("approval", 2), ("reference", 3)])
def test_development_packets_supply_sources_not_expected_programs(name, tools):
    value = stage1_cases.packet(name)
    assert set(value) == {"bundle", "task", "taskOrigin", "inputSchema", "catalog", "reads"}
    assert len(value["catalog"]["tools"]) == tools
    wire, budget = ledger.make_request(value, ledger.initial_state(value, "plan_first", semantic_plan=True))
    assert budget["accepted"] and "expectedCalls" not in str(wire)
    assert not any(d.get("executed", False) for d in value["bundle"]["documents"])


def generated(packet, tmp_path):
    state, wire, value = prepared(packet)
    files, _ = ledger.derive(packet, state, wire, envelope(value))
    files = anchored(packet, state, files)
    state = files["next-state.json"]
    wire, _ = ledger.make_request(packet, state)
    files, result = ledger.derive(packet, state, wire, envelope(slot_arguments(wire)))
    assert result["candidateStatus"] == "compiled_region_requires_semantic_review", result
    run = tmp_path / "run"
    write_artifacts(run, {"manifest.json": {"inputs": packet, "reportDigest": "test-only-manifest"}})
    write_artifacts(run / "round-001", files)
    return run, files


def test_explicit_review_digest_and_original_compilation_are_required(packet, tmp_path):
    run, files = generated(packet, tmp_path)
    digest = files["compilation.json"]["treeDigest"]
    with pytest.raises(ValueError, match="reviewed Tree digest"):
        stage1_validation.validate("wiring", run, "unreviewed")
    # A compilable prefix deliberately omits the source counter branch. Mechanical
    # validity must not turn that omission into a passing behavior report.
    report = stage1_validation.validate("wiring", run, digest)
    assert report["failed"] > 0 and report["effectCalls"] == 0
    tampered = copy.deepcopy(files)
    tampered["compilation.json"]["flow"]["purpose"] = "tampered compiled flow"
    other = tmp_path / "tampered"
    write_artifacts(other, {"manifest.json": {"inputs": packet, "reportDigest": "test-only-manifest"}})
    write_artifacts(other / "round-001", tampered)
    with pytest.raises(ValueError, match="differs from reviewed Tree"):
        stage1_validation.validate("wiring", other, digest)
