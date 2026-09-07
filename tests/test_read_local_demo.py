"""Local file/provider and explicit host-policy wiring, never production evidence."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from evaluation.read_local_demo import (
    DATASET, ReadIntentDraft, build_forward_proposal, decoder_schema, host_binding, run_reviewed, source_bundle,
)
from network_provider.local_inventory import LocalInventoryReader
from network_runtime.access import ObservationAccessContext
from network_runtime.capabilities import DataSensitivity
from network_runtime.l0.compiler import compile_documents
from network_runtime.l0.read_execution import execute_host_read


def _contract():
    proposal = build_forward_proposal(source_bundle(), ReadIntentDraft(
        purpose="Read a device from local snapshot data, not live device health.",
        tool="read_inventory_device", action_type="read_only", unresolved_questions=(),
    ))
    return compile_documents([proposal.to_manifest()])[0]


def _context():
    return ObservationAccessContext(
        subject_id="local-test", roles=frozenset({"network-reader"}),
        scopes=frozenset({"inventory:read", "device_id:campus-sw1"}), purpose="read test",
        clearance=DataSensitivity.INTERNAL,
    )


def test_real_file_read_preserves_bytes_and_filters_undeclared_fields() -> None:
    before = DATASET.read_bytes()
    contract = _contract()
    result = execute_host_read(contract, {"device_id": "campus-sw1"}, _context(),
                               host_binding(contract, LocalInventoryReader(DATASET)))
    assert result["payload"] == {"device_id": "campus-sw1", "site": "campus", "status": "planned-lab"}
    assert result["providerEvidence"]["sourceKind"] == "local_inventory_snapshot_not_live_telemetry"
    assert result["shapeCheck"]["shapeValid"]
    assert not result["contractActivated"] and not result["productionIdentityVerified"]
    assert DATASET.read_bytes() == before


@pytest.mark.parametrize("change", [
    {"authenticated": False}, {"authenticated": "true"}, {"subject_id": ""},
    {"roles": frozenset()}, {"roles": frozenset({"system"})},
    {"scopes": frozenset({"*"})}, {"scopes": frozenset({"inventory:read"})},
    {"scopes": frozenset({"device_id:campus-sw1"})},
    {"clearance": DataSensitivity.PUBLIC}, {"implicit_local_context": True},
])
def test_denied_context_never_calls_provider(change: dict) -> None:
    calls = []
    contract = _contract()
    binding = replace(host_binding(contract, LocalInventoryReader(DATASET)),
                      observe=lambda args: calls.append(args))
    with pytest.raises(PermissionError):
        execute_host_read(contract, {"device_id": "campus-sw1"}, replace(_context(), **change), binding)
    assert not calls


def test_different_resource_is_denied_before_provider() -> None:
    contract = _contract()
    with pytest.raises(PermissionError, match="scope_denied"):
        execute_host_read(contract, {"device_id": "idc-sw1"}, _context(),
                          host_binding(contract, LocalInventoryReader(DATASET)))


@pytest.mark.parametrize("field,value", [("approved_contract_hash", "bad"), ("required_scopes", frozenset())])
def test_host_binding_is_exact(field: str, value: object) -> None:
    contract = _contract()
    with pytest.raises(ValueError, match="host read binding"):
        execute_host_read(contract, {"device_id": "campus-sw1"}, _context(),
                          replace(host_binding(contract, LocalInventoryReader(DATASET)), **{field: value}))


@pytest.mark.parametrize("arguments", [{}, {"device_id": "campus-sw1", "path": "/etc/passwd"},
                                       {"device_id": "${target}"}, {"device_id": 12}])
def test_bad_parameters_never_call_provider(arguments: dict) -> None:
    contract = _contract()
    calls = []
    binding = replace(host_binding(contract, LocalInventoryReader(DATASET)), observe=lambda args: calls.append(args))
    with pytest.raises(ValueError):
        execute_host_read(contract, arguments, _context(), binding)
    assert not calls


def test_output_schema_failure_does_not_create_success_receipt() -> None:
    contract = _contract()
    binding = replace(host_binding(contract, LocalInventoryReader(DATASET)),
                      observe=lambda args: ({"status": 1}, {}))
    with pytest.raises(ValueError):
        execute_host_read(contract, {"device_id": "campus-sw1"}, _context(), binding)


@pytest.mark.parametrize("content", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}', '[]'])
def test_bad_inventory_content_is_rejected(tmp_path: Path, content: str) -> None:
    path = tmp_path / "inventory.json"
    path.write_text(content)
    with pytest.raises(ValueError):
        LocalInventoryReader(path).observe({"device_id": "campus-sw1"})


def test_symlink_dataset_rejected(tmp_path: Path) -> None:
    alias = tmp_path / "inventory.json"
    alias.symlink_to(DATASET)
    with pytest.raises(OSError):
        LocalInventoryReader(alias).observe({"device_id": "campus-sw1"})


def test_provider_reads_actual_file_updates_not_canned_reply(tmp_path: Path) -> None:
    path = tmp_path / "inventory.json"
    reader = LocalInventoryReader(path)
    for site in ("first-site", "second-site"):
        path.write_text(json.dumps({"campus-sw1": {"site": site, "status": "fixture"}}))
        payload, _ = reader.observe({"device_id": "campus-sw1"})
        assert payload["site"] == site


@pytest.mark.parametrize("field,value", [("tool", "run_shell"), ("action_type", "unknown")])
def test_unknown_forward_draft_cannot_admit_contract(field: str, value: str) -> None:
    draft = ReadIntentDraft(purpose="Observe inventory", tool="read_inventory_device",
                            action_type="read_only", unresolved_questions=())
    with pytest.raises(ValueError):
        build_forward_proposal(source_bundle(), draft.model_copy(update={field: value}))


def test_decoder_compatibility_does_not_weaken_output_validation() -> None:
    assert "maxLength" not in decoder_schema()["properties"]["purpose"]
    with pytest.raises(ValueError):
        ReadIntentDraft(purpose="x" * 2001, tool="read_inventory_device",
                        action_type="read_only", unresolved_questions=())


@pytest.mark.parametrize("command_text,allowed", [
    ("read_inventory_device device_id=campus-sw1", True),
    ("read_inventory_device device_id=idc-sw1", False),
    ("do not read_inventory_device device_id=campus-sw1", False),
    ("read_inventory_device device_id=campus-sw1 device_id=idc-sw1", False),
])
def test_full_local_wiring_requires_host_approval_and_literal_request(tmp_path, command_text, allowed) -> None:
    from evaluation.read_l05_review import build_read_review_input
    from tests.test_read_l05_review import _review

    draft = ReadIntentDraft(purpose="Read local inventory snapshot, not live state.",
                            tool="read_inventory_device", action_type="read_only", unresolved_questions=())
    bundle = source_bundle()
    proposal = build_forward_proposal(bundle, draft)
    # Explicit protocol fixture, not an actual model response or semantic evidence.
    files = {
        "inputs.json": {"bundle": bundle},
        "response.json": {"raw": {"message": {"content": draft.model_dump_json()}}},
        "l05.json": proposal.model_dump(by_alias=True, mode="json"),
        "review-input.json": build_read_review_input(proposal),
        "review.json": _review(proposal).model_dump(mode="json"),
    }
    for name, value in files.items():
        (tmp_path / name).write_text(json.dumps(value))
    with pytest.raises(PermissionError, match="host approval"):
        run_reviewed(tmp_path, tmp_path / "review.json", command_text, host_approved=False)
    if allowed:
        result = run_reviewed(tmp_path, tmp_path / "review.json", command_text, host_approved=True)
        assert result["receipt"]["payload"]["device_id"] == "campus-sw1"
        assert not result["dshOrAgentLoopExecuted"]
    else:
        with pytest.raises((PermissionError, ValueError)):
            run_reviewed(tmp_path, tmp_path / "review.json", command_text, host_approved=True)
