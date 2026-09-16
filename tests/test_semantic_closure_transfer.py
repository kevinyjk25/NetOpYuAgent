import copy

import pytest

from evaluation import semantic_closure_transfer as transfer
from evaluation.structured_binding_probe import read_json
from tests.test_hybrid_live_demo import packet_and_compilation


def spec_and_bundle():
    packet, _, _ = packet_and_compilation()
    return {"snapshot": "not-read", "candidateId": "fixture-id", "domain": "bounded-test",
        "task": packet["task"], "inputSchema": packet["inputSchema"], "tools": packet["catalog"]["tools"],
        "fixture": {"arguments": {"device": {"id": "lab-sw1"}}, "resources": {}},
        "expectations": ["GOLD_SENTINEL_MUST_NOT_REACH_MODEL"]}, packet["bundle"]


def test_expected_answers_never_enter_packet_or_host_graph():
    spec, bundle = spec_and_bundle()
    packet = transfer.packet_for(bundle, spec)
    assert "GOLD_SENTINEL" not in str(packet)
    assert packet["task"] == spec["task"] and "flow" not in packet
    changed = copy.deepcopy(spec)
    changed["tools"][0]["annotations"]["readOnlyHint"] = False
    with pytest.raises(ValueError, match="read-only"):
        transfer.packet_for(bundle, changed)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/b", "", "UPPER"])
def test_case_paths_are_confined(tmp_path, name):
    with pytest.raises(ValueError, match="confined"):
        transfer.case_folder(tmp_path, name)


def test_freeze_precedes_input_selection_and_detects_tampering(tmp_path, monkeypatch):
    one = "evaluation/semantic_closure_transfer.py"
    pin = {one: transfer.fingerprint()[one]}
    monkeypatch.setattr(transfer, "fingerprint", lambda: dict(pin))
    monkeypatch.setattr(transfer.OllamaAnchoredAuthorAdapter, "preflight", lambda _: {"model": "qwen3.5:9b", "modelArtifactDigest": "local-test"})
    root = tmp_path / "cohort"
    frozen = transfer.freeze(root)
    assert not frozen["sourceSelectionHasOccurred"] and (root / "freeze/source-snapshot.tar.gz").is_file()
    spec, bundle = spec_and_bundle()
    monkeypatch.setattr(transfer, "bundle_from_snapshot", lambda *args: bundle)
    transfer.prepare(root, "sample", spec)
    folder, packet = transfer.verify_case(root, "sample")
    assert "GOLD_SENTINEL" not in str(packet)
    assert read_json(folder / "inputs/expectations.json")["neverModelInput"]
    with pytest.raises(FileExistsError):
        transfer.prepare(root, "sample", spec)
    pin[one] = "drift"
    with pytest.raises(ValueError, match="implementation drift"):
        transfer.verify_case(root, "sample")


@pytest.mark.parametrize("limit,expected", [(10, ["p0", "p1", "small"]), (4, ["p0"]), (1, ["p0"])])
def test_entry_budget_is_not_displaced_by_large_references(monkeypatch, limit, expected):
    packet = {"bundle": {"entryPath": "SKILL.md"}}
    pages = {"p0": {"path": "SKILL.md", "size": 3}, "p1": {"path": "SKILL.md", "size": 3},
             "large": {"path": "references/large.md", "size": 20},
             "small": {"path": "references/small.md", "size": 2},
             "script": {"path": "scripts/run.py", "size": 1}}
    monkeypatch.setattr(transfer.author, "pages_for", lambda _: pages)
    monkeypatch.setattr(transfer.author, "make_request", lambda _, keys: sum(pages[k]["size"] for k in keys))
    monkeypatch.setattr(transfer, "budget", lambda size: {"accepted": size <= limit})
    assert transfer.initial_source_pages(packet) == expected
    assert "script" not in expected and pages["large"]["size"] == 20


def test_reference_document_is_not_silently_partially_supplied(monkeypatch):
    packet = {"bundle": {"entryPath": "SKILL.md"}}
    pages = {"entry": {"path": "SKILL.md"}, "r1": {"path": "ref.md"}, "r2": {"path": "ref.md"}}
    monkeypatch.setattr(transfer.author, "pages_for", lambda _: pages)
    monkeypatch.setattr(transfer.author, "make_request", lambda _, keys: keys)
    monkeypatch.setattr(transfer, "budget", lambda keys: {"accepted": len(keys) <= 2})
    assert transfer.initial_source_pages(packet) == ["entry"]
