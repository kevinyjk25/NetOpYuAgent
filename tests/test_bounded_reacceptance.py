"""Temporary SQLite/evidence gate fixtures only; no controller or model runs."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import json
import sqlite3

import pytest

from evaluation import bounded_reacceptance as module
from evaluation.bounded_budget import BudgetLedger
from evaluation.bounded_pilot import seal
from network_runtime.contracts import sha256_json


AUTHORIZATION = {"source": "unit:test_not_user_authority", "statement": "explicit unit fixture",
                 "approved_at": "2026-09-18T00:00:00Z", "scope": "one_zero_inference_engineering_reacceptance"}


def parent_fixture(tmp_path, *, protocol=None, review=None, dialogues=None):
    """Structurally valid fictional failure, not the real preserved batch."""
    directory, registry = tmp_path / "parent", tmp_path / "budget.sqlite3"
    directory.mkdir()
    (directory / "controller").mkdir()
    protocol = protocol if protocol is not None else seal({"role": "unit protocol"})
    review = review if review is not None else seal({"role": "unit review"})
    dialogues = dialogues if dialogues is not None else {"unit": []}
    dependencies = seal({"role": "unit dependencies"})
    frozen = seal({"protocol_digest": protocol["digest"], "dialogues": dialogues, "role": "unit freeze"})
    ledger = BudgetLedger(registry, clock=lambda: 1000)
    ledger.register_study(module.PARENT_STUDY_ID, frozen["digest"])
    candidate = ledger.register_candidate(module.PARENT_STUDY_ID, sha256_json("unit candidate"))
    arm = ledger.start_arm(module.PARENT_STUDY_ID, candidate, "unit-case", 1, "B", sha256_json("unit context"))
    ledger.pause_study(module.PARENT_STUDY_ID, "unit_measurement_invalid")
    ledger.finish_arm(arm, "measurement_invalid")
    controller = seal({"study_id": module.PARENT_STUDY_ID, "assigned_arms": 24,
        "controllerMechanicsPassed": False, "freeze_digest": frozen["digest"],
        "rows": [{"terminal": "measurement_invalid", "arm_id": arm}], "unrun": [{}] * 23,
        "budget": ledger.snapshot(module.PARENT_STUDY_ID)})
    before = seal({"protocol_digest": protocol["digest"], "review_digest": review["digest"],
                   "dialogues_digest": sha256_json(dialogues), "dependency_digest": dependencies["digest"]})
    report = seal({"mechanical_acceptance_passed": False, "actualModelCalls": 0, "liveGenerationEnabled": False,
        "protocol_digest": protocol["digest"], "review_digest": review["digest"],
        "dependency_digest": dependencies["digest"], "controller_report_digest": controller["digest"]})
    documents = {"report.json": report, "pre-run.json": before, "protocol.json": protocol,
        "reference-freeze.json": review, "dependencies.json": dependencies,
        "controller/report.json": controller, "controller/freeze.json": frozen}
    for name, value in documents.items():
        (directory / name).write_text(json.dumps(value))
    amendment = module.prepare_amendment(directory, registry, AUTHORIZATION)
    path = tmp_path / "amendment.json"
    path.write_text(json.dumps(amendment))
    return directory, registry, path, amendment


def reseal(value, **changes):
    return seal({**{key: item for key, item in value.items() if key != "digest"}, **changes})


def test_prepare_and_load_are_readonly_and_bind_original_rows(tmp_path):
    directory, registry, path, value = parent_fixture(tmp_path)
    raw_db = registry.read_bytes()
    raw_files = {name: (directory / name).read_bytes() for name in module.PARENT_FILES}
    assert module.load_amendment(path, registry) == value
    assert module.prepare_amendment(directory, registry, AUTHORIZATION) == value
    assert registry.read_bytes() == raw_db
    assert raw_files == {name: (directory / name).read_bytes() for name in module.PARENT_FILES}
    assert value["parent_ledger_snapshot"] == module.parent_snapshot(registry)


@pytest.mark.parametrize("change", ["missing", "empty", "scope", "second_child", "new_window", "new_registry"])
def test_explicit_authorization_and_fixed_scope_required(tmp_path, change):
    _, registry, path, value = parent_fixture(tmp_path)
    changed = copy.deepcopy(value)
    if change == "missing":
        changed.pop("authorization")
    elif change == "empty":
        changed["authorization"]["statement"] = ""
    elif change == "scope":
        changed["authorization"]["scope"] = "research_retry"
    elif change == "second_child":
        changed["child_study_id"] += "-2"
    elif change == "new_window":
        changed["limits"]["original_r0_window_restarted"] = True
    else:
        changed["registry_path"] = str(tmp_path / "different.sqlite3")
    path.write_text(json.dumps(reseal(changed)))
    with pytest.raises(ValueError, match="authorization"):
        module.load_amendment(path, registry)


def test_missing_manifest_is_not_authorization(tmp_path):
    with pytest.raises(ValueError, match="amendment file"):
        module.load_amendment(tmp_path / "absent.json", tmp_path / "absent.sqlite3")


@pytest.mark.parametrize("name", module.PARENT_FILES)
def test_old_evidence_drift_rejected_even_when_json_semantics_unchanged(tmp_path, name):
    directory, registry, path, _ = parent_fixture(tmp_path)
    with (directory / name).open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="evidence drift"):
        module.load_amendment(path, registry)


def test_parent_report_must_match_live_readonly_snapshot(tmp_path):
    _, registry, path, _ = parent_fixture(tmp_path)
    with sqlite3.connect(registry) as db:
        db.execute("UPDATE studies SET last_clock=last_clock+1")
    with pytest.raises(ValueError, match="snapshot differ"):
        module.load_amendment(path, registry)


@pytest.mark.parametrize("field", ["protocol_digest", "review_digest", "dialogues_digest"])
def test_no_task_label_or_dialogue_changes(tmp_path, field):
    _, registry, path, _ = parent_fixture(tmp_path)
    with pytest.raises(ValueError, match="cannot change"):
        module.load_amendment(path, registry, bindings={field: sha256_json("changed")})


def test_claim_consumes_once_and_child_preserves_all_parent_rows(tmp_path):
    directory, registry, path, value = parent_fixture(tmp_path)
    old = module.parent_snapshot(registry)
    originals = {name: (directory / name).read_bytes() for name in module.PARENT_FILES}
    output = tmp_path / "child"
    receipt = module.claim(registry, value, output)
    assert module.load_amendment(path, registry) == value
    with pytest.raises(ValueError, match="context differs"):
        module.consume_claim(registry, receipt, tmp_path / "new-output", value["protocol_digest"], value["dialogues_digest"])
    child = module.consume_claim(registry, receipt, output, value["protocol_digest"], value["dialogues_digest"])
    assert child == module.CHILD_STUDY_ID
    with pytest.raises(ValueError, match="spent"):
        module.consume_claim(registry, receipt, output, value["protocol_digest"], value["dialogues_digest"])
    ledger = BudgetLedger(registry, clock=lambda: 2000)
    ledger.register_study(child, sha256_json("child engineering freeze"))
    candidate = ledger.register_candidate(child, sha256_json("child implementation"))
    arm = ledger.start_arm(child, candidate, "unit-case", 1, "B", sha256_json("unit context"))
    ledger.finish_arm(arm, "completed")
    module.verify_parent(registry, receipt)
    assert module.load_amendment(path, registry) == value
    assert module.parent_snapshot(registry) == old
    assert originals == {name: (directory / name).read_bytes() for name in module.PARENT_FILES}
    assert module.linked_counts(receipt, 24)["total_assigned_arms"] == 48
    assert module.linked_counts(receipt, 24)["total_started_arms"] == 25


def test_new_directory_or_another_manifest_cannot_reclaim_parent(tmp_path):
    _, registry, _, value = parent_fixture(tmp_path)
    module.claim(registry, value, tmp_path / "first")
    amended = reseal(value, authorization={**AUTHORIZATION, "statement": "a different fixture"})
    with pytest.raises(ValueError, match="already claimed"):
        module.claim(registry, amended, tmp_path / "different")


def test_concurrent_connections_only_one_claim_wins(tmp_path):
    _, registry, _, value = parent_fixture(tmp_path)
    def attempt(index):
        try:
            return module.claim(registry, value, tmp_path / str(index))
        except ValueError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(attempt, range(2)))
    assert sum(item is not None for item in outcomes) == 1


def test_parent_drift_after_claim_denies_consume_and_verification(tmp_path):
    _, registry, _, value = parent_fixture(tmp_path)
    output = tmp_path / "child"
    receipt = module.claim(registry, value, output)
    with sqlite3.connect(registry) as db:
        db.execute("UPDATE arms SET elapsed_ms=elapsed_ms+1")
    with pytest.raises(ValueError, match="halted parent changed"):
        module.consume_claim(registry, receipt, output, value["protocol_digest"], value["dialogues_digest"])
    with pytest.raises(ValueError, match="halted parent changed"):
        module.verify_parent(registry, receipt)


def test_controller_without_claim_still_cannot_restart_parent(monkeypatch, tmp_path):
    from evaluation import bounded_controller as controller
    from tests.test_bounded_controller import sample
    protocol, references, dialogues = sample()
    _, registry, _, _ = parent_fixture(tmp_path, protocol=protocol, dialogues=dialogues)
    before = module.parent_snapshot(registry)
    binary = tmp_path / "inert-dsh-entry"
    binary.write_text("never executed")
    monkeypatch.setattr(controller, "implementation_fingerprint", lambda: {"unit": "no execution"})
    with pytest.raises(ValueError):
        controller.run(protocol, references, tmp_path / "new-output", dialogues=dialogues,
                       registry=registry, provider_root=tmp_path / "providers", dsh_binary=binary)
    assert module.parent_snapshot(registry) == before
    assert not (tmp_path / "new-output").exists()


def test_controller_consumes_claim_before_ledger_or_new_arm(monkeypatch, tmp_path):
    from evaluation import bounded_controller as controller
    from tests.test_bounded_controller import sample
    protocol, references, dialogues = sample()
    _, registry, _, value = parent_fixture(tmp_path, protocol=protocol, dialogues=dialogues)
    output = tmp_path / "child"
    receipt = module.claim(registry, value, output)
    binary = tmp_path / "inert-dsh-entry"
    binary.write_text("never executed")
    monkeypatch.setattr(controller, "implementation_fingerprint", lambda: {"unit": "no execution"})
    def stop_before_ledger(*args):
        with sqlite3.connect(registry) as db:
            assert db.execute("SELECT consumed FROM engineering_reacceptance_claims").fetchone()[0] == 1
            assert db.execute("SELECT COUNT(*) FROM arms").fetchone()[0] == 1
        raise RuntimeError("unit deliberate stop before any child study/arm")
    monkeypatch.setattr(controller, "BudgetLedger", stop_before_ledger)
    with pytest.raises(RuntimeError, match="unit deliberate stop"):
        controller.run(protocol, references, output, dialogues=dialogues, registry=registry,
                       provider_root=tmp_path / "providers", dsh_binary=binary, reacceptance_claim=receipt)
    assert not output.exists()
