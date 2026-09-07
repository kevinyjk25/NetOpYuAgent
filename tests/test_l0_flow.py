"""Bounded flow mechanics and real local reads, not LLM translation evidence."""

from dataclasses import replace
import hashlib
import json

import pytest

from evaluation.read_local_demo import DATASET, host_binding
from network_provider.local_inventory import LocalInventoryReader
from network_runtime.contracts import sha256_json
from network_runtime.l0.flow import EffectTarget, FlowProposal, HostFlowConsent, qualify_flow, run_read_flow
from network_runtime.l0.read_contracts import compile_read
from network_runtime.l0.models import AtomicReadManifest
from tests.test_read_local_demo import _context, _contract


def _ref(source, field):
    return {"kind": "reference", "source": source, "field": field}


def _constant(value):
    return {"kind": "constant", "value": value}


def _fixture():
    contract = _contract()
    proposal = FlowProposal.model_validate({
        "api_version": "netopyu.io/l0-flow-proposal/v1", "source_digest": sha256_json("synthetic flow fixture"),
        "purpose": "Local snapshot branch, not live-health diagnosis", "entry": "lookup",
        "input_schema": contract.spec.input_schema.model_dump(by_alias=True), "max_read_age_seconds": 30,
        "nodes": [
            {"kind": "read", "id": "lookup", "contract_hash": contract.contract_hash,
             "arguments": {"device_id": _ref("input", "device_id")}, "next": "select"},
            {"kind": "branch", "id": "select", "left": _ref("lookup", "site"),
             "equals": _constant("campus"), "on_true": "reread", "on_false": "reason"},
            {"kind": "read", "id": "reread", "contract_hash": contract.contract_hash,
             "arguments": {"device_id": _ref("lookup", "device_id")}, "next": "done"},
            {"kind": "end", "id": "done", "outcome": "read_path_completed", "explanation": "Two typed local reads complete."},
            {"kind": "end", "id": "reason", "outcome": "needs_l1", "explanation": "Ask L1; no effect authorization."},
        ],
    })
    return proposal, {contract.contract_hash: contract}


def _run(proposal, reads, *, context=None, bindings=None, effects=None, consent=None, arguments=None):
    effects = effects or {}
    arguments = arguments or {"device_id": "campus-sw1"}
    packet = qualify_flow(proposal, reads, effects)
    bindings = bindings if bindings is not None else {key: host_binding(contract, LocalInventoryReader(DATASET)) for key, contract in reads.items()}
    return run_read_flow(proposal, arguments, reads=reads, effects=effects, bindings=bindings,
                         context=context or _context(),
                         consent=consent or HostFlowConsent(packet["flowDigest"], sha256_json(arguments)))


def test_sequence_branch_output_binding_really_reads_local_file():
    proposal, reads = _fixture()
    before = DATASET.read_bytes()
    report = _run(proposal, reads)
    assert report["status"] == "read_path_completed"
    assert [row["node"] for row in report["trace"]] == ["lookup", "select", "reread", "done"]
    assert report["trace"][2]["receipt"]["payload"]["device_id"] == "campus-sw1"
    assert not report["effectExecuted"] and not report["wholeSkillCorrectnessProven"]
    assert report["reportDigest"] == sha256_json({k: v for k, v in report.items() if k != "reportDigest"})
    assert DATASET.read_bytes() == before


def test_false_branch_keeps_l1_handoff_not_success():
    proposal, reads = _fixture()
    raw = proposal.model_dump()
    raw["nodes"][1]["equals"] = _constant("elsewhere")
    report = _run(FlowProposal.model_validate(raw), reads)
    assert report["status"] == "needs_l1"
    assert [row["node"] for row in report["trace"]] == ["lookup", "select", "reason"]


@pytest.mark.parametrize("mutation", ["cycle", "unknown", "unreachable", "duplicate", "reserved",
                                      "forward_ref", "unknown_field", "wrong_type", "bool_number",
                                      "missing_arg", "extra_arg", "nonfinite", "object", "array"])
def test_malformed_or_unsupported_flow_never_qualifies(mutation):
    proposal, reads = _fixture()
    raw = proposal.model_dump()
    first, branch = raw["nodes"][:2]
    raw["nodes"] = list(raw["nodes"])
    if mutation == "cycle":
        first["next"] = "lookup"
    elif mutation == "unknown":
        first["next"] = "missing"
    elif mutation == "unreachable":
        branch["on_false"] = "reread"
    elif mutation == "duplicate":
        raw["nodes"].append(first)
    elif mutation == "reserved":
        first["id"] = "input"
        raw["entry"] = "input"
    elif mutation == "forward_ref":
        branch["left"] = _ref("reread", "site")
    elif mutation == "unknown_field":
        branch["left"] = _ref("lookup", "password")
    elif mutation in {"wrong_type", "bool_number", "nonfinite", "object", "array"}:
        branch["equals"] = _constant({"wrong_type": 42, "bool_number": True, "nonfinite": float("nan"), "object": {}, "array": []}[mutation])
    elif mutation == "missing_arg":
        first["arguments"] = {}
    else:
        first["arguments"]["path"] = _constant("/etc/passwd")
    with pytest.raises(ValueError):
        qualify_flow(FlowProposal.model_validate(raw), reads, {})


def test_join_rejects_value_from_only_one_branch():
    proposal, reads = _fixture()
    raw = proposal.model_dump()
    raw["nodes"] = list(raw["nodes"])
    raw["nodes"][1]["on_false"] = "done"
    raw["nodes"][3] = {"kind": "branch", "id": "done", "left": _ref("reread", "site"),
                       "equals": _constant("campus"), "on_true": "reason", "on_false": "reason"}
    with pytest.raises(ValueError, match="every incoming path"):
        qualify_flow(FlowProposal.model_validate(raw), reads, {})


@pytest.mark.parametrize("digest", ["flow", "arguments"])
def test_stale_host_consent_denies_before_read(digest):
    proposal, reads = _fixture()
    packet = qualify_flow(proposal, reads, {})
    consent = HostFlowConsent("bad" if digest == "flow" else packet["flowDigest"],
                              "bad" if digest == "arguments" else sha256_json({"device_id": "campus-sw1"}))
    with pytest.raises(PermissionError):
        _run(proposal, reads, consent=consent)


def test_missing_host_environment_denies_before_read():
    proposal, reads = _fixture()
    with pytest.raises(ValueError, match="missing"):
        qualify_flow(proposal, {}, {})
    with pytest.raises(PermissionError):
        _run(proposal, reads, bindings={})


def test_provider_error_blocks_and_redacts_sensitive_exception():
    proposal, reads = _fixture()
    calls = []

    def fail(args):
        calls.append(args)
        raise RuntimeError("secret-password")

    bindings = {key: replace(host_binding(contract, LocalInventoryReader(DATASET)), observe=fail) for key, contract in reads.items()}
    report = _run(proposal, reads, bindings=bindings)
    assert report["status"] == "blocked" and report["blockedAt"] == "lookup" and len(calls) == 1
    assert "secret-password" not in str(report)


def test_read_access_denial_does_not_take_false_branch():
    proposal, reads = _fixture()
    report = _run(proposal, reads, context=replace(_context(), authenticated=False))
    assert report["status"] == "blocked" and len(report["trace"]) == 1


@pytest.mark.parametrize("times", [[0, 31], [30, 0]])
def test_expired_or_clock_regressed_read_is_unknown_not_false(monkeypatch, times):
    proposal, reads = _fixture()
    clock = iter(times)
    monkeypatch.setattr("network_runtime.l0.flow.time.monotonic", lambda: next(clock))
    report = _run(proposal, reads)
    assert report["status"] == "blocked" and report["blockedAt"] == "select"


def test_effect_leaf_is_non_executable_even_with_flow_consent():
    proposal, reads = _fixture()
    raw = proposal.model_dump()
    raw["nodes"] = list(raw["nodes"])
    raw["nodes"][3] = {"kind": "effect_candidate", "id": "done", "binding_id": "reviewed-target",
                       "arguments": {"device_id": _ref("reread", "device_id")}}
    target = EffectTarget(profile="test", tool="test_tool", skill_id="test.skill", contract_hash="fixture",
                          input_schema=proposal.input_schema)
    report = _run(FlowProposal.model_validate(raw), reads, effects={"reviewed-target": target})
    assert report["status"] == "awaiting_effect_admission" and not report["effectExecuted"]
    assert report["candidate"]["arguments"] == {"device_id": "campus-sw1"}


def test_effect_cannot_continue_into_uncoordinated_second_write():
    proposal, _ = _fixture()
    raw = proposal.model_dump()
    raw["nodes"] = list(raw["nodes"])
    raw["nodes"][3] = {"kind": "effect_candidate", "id": "done", "binding_id": "target", "arguments": {}, "next": "reason"}
    with pytest.raises(ValueError):
        FlowProposal.model_validate(raw)


def test_missing_optional_output_is_unknown_not_false():
    proposal, reads = _fixture()
    contract = next(iter(reads.values()))
    spec = contract.spec.model_dump(by_alias=True, mode="json")
    spec["outputSchema"]["required"].remove("site")
    for source in spec["sources"]:
        if source["role"] == "tool":
            tool = json.loads(source["text"])
            tool["outputSchema"] = spec["outputSchema"]
            source["text"] = json.dumps(tool)
            source["sha256"] = "sha256:" + hashlib.sha256(source["text"].encode()).hexdigest()
    contract = compile_read(AtomicReadManifest(apiVersion="netopyu.io/l0-effect/v2", kind="AtomicRead",
                                               metadata=contract.metadata, spec=spec))
    raw = proposal.model_dump()
    for node in raw["nodes"]:
        if node["kind"] == "read":
            node["contract_hash"] = contract.contract_hash
    binding = replace(host_binding(contract, LocalInventoryReader(DATASET)),
                      observe=lambda args: ({"device_id": "campus-sw1", "status": "planned-lab"}, {}))
    report = _run(FlowProposal.model_validate(raw), {contract.contract_hash: contract},
                  bindings={contract.contract_hash: binding})
    assert report["status"] == "blocked" and report["blockedAt"] == "select"
    assert not any(row.get("matched") is False for row in report["trace"])


def test_demo_covers_both_real_read_paths_without_model_or_writes():
    from evaluation.flow_local_demo import run_demo
    with pytest.raises(PermissionError):
        run_demo(allow_local_read=False)
    result = run_demo(allow_local_read=True)
    assert result["datasetUnchanged"] and result["modelCalls"] == 0
    assert [row["status"] for row in result["outcomes"]] == ["read_path_completed", "needs_l1"]
    assert sum(step["kind"] == "read" for row in result["outcomes"] for step in row["trace"]) == 3


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), 301])
def test_invalid_age_limit_cannot_disable_freshness(value):
    proposal, _ = _fixture()
    with pytest.raises(ValueError):
        FlowProposal.model_validate({**proposal.model_dump(), "max_read_age_seconds": value})
