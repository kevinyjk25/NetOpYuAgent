"""Binding integrity is not semantic equivalence or independent review."""

import copy
import json

import pytest

from evaluation import source_host_binding as binding
from evaluation.structured_flow_demo import fixture
from evaluation.translation_intake import _bundle, _document


@pytest.fixture
def binding_packet():
    bundle, tree, reads, _ = fixture()
    quote = "legacy_read --target DEVICE returns original interface observations."
    path = "references/synthetic-adapter.md"
    doc = _document(path, quote.encode(), mode="100644", origin="fixture")
    bundle = _bundle({**{k: v for k, v in bundle.items() if k != "bundleDigest"}, "documents": [*bundle["documents"], doc]})
    tools = [json.loads(next(s.text for s in c.spec.sources if s.role == "tool")) for c in reads.values()]
    packet = {"bundle": bundle, "task": "Read the explicitly selected device; no guessed prerequisites.",
        "taskOrigin": "developer_authored_evaluation_request", "inputSchema": tree.input_schema,
        "catalog": {"tools": tools}, "reads": {n: c.model_dump(by_alias=True, mode="json") for n, c in reads.items()}}
    tool = next(t for t in tools if t["name"] == "get_interfaces")
    mapping = {"id": "fixture-read", "sourceBundleDigest": bundle["bundleDigest"],
        "source": {"path": path, "start": 0, "end": len(quote), "quote": quote},
        "sourceOperation": "legacy_read", "hostTool": tool["name"], "contractHash": reads[tool["name"]].contract_hash,
        "inputSchemaDigest": binding.sha256_json(tool["inputSchema"]), "outputSchemaDigest": binding.sha256_json(tool["outputSchema"]),
        "parameterMap": [{"sourceArgument": "--target", "hostParameter": "device"}],
        "scope": "Synthetic read adapter declaration only; no whole-wrapper equivalence.",
        "limitations": ["Authorization and nested argument construction remain unresolved."],
        "reviewKind": "developer_reviewed_adapter_declaration_not_independent_gold"}
    return packet, mapping


def test_explicit_binding_is_lossless_and_not_execution_authority(binding_packet):
    packet, mapping = binding_packet
    before = copy.deepcopy((packet, mapping))
    assert binding.validate_bindings(packet, [mapping]) == [mapping]
    view = binding.binding_view([mapping])[0]
    assert not view["semanticEquivalenceProven"] and not view["runtimeAuthorityGranted"]
    assert (packet, mapping) == before


@pytest.mark.parametrize("issue", ["bundle", "quote", "operation", "tool", "contract", "input", "output", "parameter",
                                 "source_argument", "duplicate", "review", "extra", "empty_limits", "repeated_target", "end"])
def test_binding_rejects_drift_and_ambiguous_correspondences(binding_packet, issue):
    packet, mapping = binding_packet
    if issue in {"bundle", "contract", "input", "output"}:
        key = {"bundle": "sourceBundleDigest", "contract": "contractHash", "input": "inputSchemaDigest", "output": "outputSchemaDigest"}[issue]
        mapping[key] = "sha256:wrong"
    elif issue == "quote":
        mapping["source"]["quote"] += " invented"
    elif issue == "operation":
        mapping["sourceOperation"] = "invented_operation"
    elif issue == "tool":
        mapping["hostTool"] = "unlisted_write"
    elif issue == "parameter":
        mapping["parameterMap"][0]["hostParameter"] = "unknown"
    elif issue == "source_argument":
        mapping["parameterMap"][0]["sourceArgument"] = "--invented"
    elif issue == "review":
        mapping["reviewKind"] = "model_guessed"
    elif issue == "extra":
        mapping["runtimeAuthorityGranted"] = True
    elif issue == "empty_limits":
        mapping["limitations"] = []
    elif issue == "repeated_target":
        mapping["parameterMap"] *= 2
    elif issue == "end":
        mapping["source"]["end"] += 50
    with pytest.raises(ValueError):
        binding.validate_bindings(packet, [mapping] * (2 if issue == "duplicate" else 1))


def test_optional_binding_does_not_silently_infer_from_tool_names(binding_packet):
    packet, _ = binding_packet
    assert binding.validate_bindings(packet, []) == []


def test_frozen_binding_reaches_request_without_replacing_original_packet(binding_packet, tmp_path, monkeypatch):
    from evaluation import source_ledger as ledger
    from evaluation.structured_binding_probe import read_json
    packet, mapping = binding_packet
    monkeypatch.setattr(ledger.prior.OllamaAnchoredAuthorAdapter, "preflight", lambda self: {"model": "fixture"})
    folder = tmp_path / "run"
    ledger.freeze(packet, folder, bindings=[mapping])
    manifest = ledger.load_manifest(folder)
    assert manifest["inputs"] == packet and manifest["hostBindings"] == [mapping]
    wire, _ = ledger.make_request(packet, manifest["initialState"], manifest["hostBindings"])
    content = json.loads(wire["messages"][1]["content"])
    assert content["hostBindings"][0]["contractHash"] == mapping["contractHash"]
    assert content["hostCatalog"] == packet["catalog"]
    boundary = content["authoringBoundary"]
    assert not boundary["liveCredentialsRequiredForAuthoring"] and not boundary["executionPermissionSatisfied"]
    assert not boundary["providerExecution"] and not boundary["sourceSpecificPreconditionsAutomaticallyCovered"]
    for item in boundary["readPrerequisites"]:
        original = packet["reads"][item["tool"]]
        assert item["contractHash"] == original["contractHash"]
        assert item["access"] == original["spec"]["access"]
        assert item["resourceScopes"] == original["spec"]["resourceScopes"]
    report = ledger.run(folder)
    assert report["hostBindingCount"] == 1 and report["modelCallsRecorded"] == 0 and not report["candidateProduced"]
    tampered = read_json(folder / "manifest.json")
    tampered["hostBindings"][0]["scope"] = "changed declared scope"
    (folder / "manifest.json").write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="drift"):
        ledger.load_manifest(folder)


def test_cli_reads_versioned_object_not_raw_json_array(binding_packet, tmp_path, monkeypatch, capsys):
    from evaluation import source_ledger as ledger
    packet, mapping = binding_packet
    monkeypatch.setattr(ledger.prior.OllamaAnchoredAuthorAdapter, "preflight", lambda self: {"model": "fixture"})
    source, bindings, run = tmp_path / "input.json", tmp_path / "bindings.json", tmp_path / "run"
    source.write_text(json.dumps(packet))
    bindings.write_text(json.dumps({"apiVersion": "netopyu.io/source-host-bindings/v1", "bindings": [mapping]}))
    monkeypatch.setattr("sys.argv", ["source_ledger", "freeze", str(run), "--inputs", str(source), "--bindings", str(bindings)])
    ledger.main()
    assert "sha256:" in capsys.readouterr().out
    assert ledger.load_manifest(run)["hostBindings"] == [mapping]


@pytest.mark.parametrize("packet", [[], {}, {"apiVersion": "wrong", "bindings": []},
                                   {"apiVersion": "netopyu.io/source-host-bindings/v1", "bindings": [], "authority": True}])
def test_versioned_binding_envelope_rejects_ambiguous_input(packet):
    with pytest.raises(ValueError, match="versioned"):
        binding.binding_packet_declarations(packet)
