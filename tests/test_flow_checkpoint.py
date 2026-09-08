"""Consolidation regression only, not additional Skill/generalization evidence."""

import json
import subprocess
import sys

import pytest

from evaluation import flow_checkpoint as checkpoint


def derive(envelope):
    return {"candidate.json": envelope}, {"status": envelope["status"]}


@pytest.fixture
def transport(monkeypatch):
    calls = []
    monkeypatch.setattr(checkpoint.OllamaAnchoredAuthorAdapter, "preflight", lambda self: {"model": "fixture"})

    def send(arm, wire):
        calls.append((arm, wire))
        return {"status": wire["status"]}

    monkeypatch.setattr(checkpoint, "send", send)
    return calls


@pytest.mark.parametrize("status", ["text_received", "transport_error", "invalid_candidate"])
def test_one_attempt_then_offline_replay_even_for_recorded_failure(tmp_path, transport, status):
    root = tmp_path / "run"
    inputs = {"wireRequest": {"status": status}, "implementation": {"source": "v1"}}
    result = checkpoint.author_once(root, inputs, derive, max_new_calls=1, label="test")
    assert checkpoint.author_once(root, inputs, derive, max_new_calls=0, label="test") == result
    assert len(transport) == 1


@pytest.mark.parametrize("budget", [0, -1, True, 1.0])
def test_missing_explicit_budget_never_creates_checkpoint(tmp_path, transport, budget):
    root = tmp_path / "run"
    with pytest.raises(ValueError, match="budget"):
        checkpoint.author_once(root, {"wireRequest": {}}, derive, max_new_calls=budget, label="test")
    assert not root.exists() and not transport


@pytest.mark.parametrize("mutation", ["partial", "receipt", "derivation", "request", "extra"])
def test_drift_never_retries_or_overwrites_evidence(tmp_path, transport, mutation):
    root = tmp_path / "run"
    inputs = {"wireRequest": {"status": "text_received"}}
    checkpoint.author_once(root, inputs, derive, max_new_calls=1, label="test")
    if mutation == "partial":
        (root / "receipt.json").unlink()
    elif mutation == "request":
        inputs = {"wireRequest": {"status": "changed"}}
    else:
        target = root / ("extra.json" if mutation == "extra" else "candidate.json")
        target.write_text(json.dumps({"status": "changed"}))
        if mutation != "receipt":
            (root / "receipt.json").write_text(json.dumps(checkpoint.receipt(root)))
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    with pytest.raises(ValueError, match="drift|checkpoint"):
        checkpoint.author_once(root, inputs, derive, max_new_calls=1, label="test")
    assert len(transport) == 1
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}


def test_active_fingerprint_covers_common_logic_and_runtime_not_historical_pilots():
    files = checkpoint.implementation()
    assert {"evaluation/flow_checkpoint.py", "evaluation/flow_guard_necessity.py",
        "network_runtime/l0/flow.py", "network_runtime/contracts.py"} <= files.keys()
    assert not any(name.endswith("_pilot.py") or "source_dut" in name for name in files)
    assert all(value.startswith("sha256:") for value in files.values())


def test_recommended_authoring_import_does_not_load_historical_probe_or_oracles():
    script = (
        "import sys; from evaluation import flow_guard_necessity; "
        "assert 'evaluation.flow_behavior_examples' not in sys.modules; "
        "from evaluation import flow_behavior_probe; "
        "assert 'evaluation.flow_source_duty_pilot' not in sys.modules; "
        "assert 'evaluation.flow_node_evidence_pilot' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True)
