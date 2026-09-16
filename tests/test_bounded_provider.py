"""Local state/receipt integrity only; no inference or product Effect authority."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import sqlite3

import pytest

from evaluation.bounded_provider import FIXTURE_SCHEMA, LocalProviderPool, ProviderError, validate_fixture
from network_runtime.contracts import sha256_json


def fixture():
    tools = []
    for name, operation, kind in (("read_state", "read", "read"), ("verify_state", "read", "verify"),
                                  ("set_state", "set", "effect"), ("delete_state", "delete", "effect")):
        properties = {"target": {"type": "string"}}
        if operation == "set":
            properties["enabled"] = {"type": "boolean"}
        tools.append(dict(name=name, description=f"Simulated {operation} enabled property", contract_id=name,
            input_schema=dict(type="object", properties=properties, required=list(properties), additionalProperties=False),
            operation=operation, kind=kind, target={"argument": "target"}, property="enabled",
            value={"argument": "enabled"} if operation == "set" else None, requires_approval=operation != "read"))
    return dict(schema=FIXTURE_SCHEMA, state={"router-1": {"enabled": False}, "router-2": {"enabled": False}}, tools=tools)


def provider(tmp_path):
    return LocalProviderPool(tmp_path / "pool").create_arm("arm-B", "treatment", fixture())


def test_clones_are_independent_and_both_fixture_and_actual_state_are_bound(tmp_path):
    pool, source = LocalProviderPool(tmp_path / "pool"), fixture()
    a, b = pool.create_arm("A", "control", source), pool.create_arm("B", "treatment", source)
    assert a.path != b.path and a.binding["isolation_id"] != b.binding["isolation_id"]
    assert a.binding["fixture_digest"] == b.binding["fixture_digest"] == sha256_json(source)
    assert a.binding["initial_state_digest"] == b.binding["initial_state_digest"] == sha256_json(source["state"])
    source["state"]["router-1"]["enabled"] = "caller mutation"
    result = a.invoke("set_state", {"target": "router-1", "enabled": True}, request_id="write", approved=True)
    assert result["result"]["applied"] and result["receipt"]["call"]["independent"] is False
    assert result["receipt"]["call"]["result_value"] is None
    assert a.snapshot()["state"]["router-1"]["enabled"] is True
    assert b.snapshot()["state"]["router-1"]["enabled"] is False
    assert set(a.visible_tools()[0]) == {"name", "description", "input_schema", "output_schema"}
    assert a.binding["simulation"] and not a.binding["productEffectAuthority"]


def test_verify_is_new_sqlite_read_and_receipts_form_sealed_ordered_chain(tmp_path):
    p = provider(tmp_path)
    p.invoke("set_state", {"target": "router-1", "enabled": True}, request_id="write", approved=True)
    read = p.invoke("verify_state", {"target": "router-1"}, request_id="verify")
    assert read["result"]["value"] is True and read["result"]["revision"] == 2
    assert read["receipt"]["call"]["independent"] and read["receipt"]["call"]["kind"] == "verify"
    head = sha256_json(p.binding)
    for row in p.receipts():
        receipt = deepcopy(row["receipt"])
        assert receipt.pop("receipt_digest") == sha256_json(receipt)
        assert receipt["previous_receipt_digest"] == head
        head = row["receipt"]["receipt_digest"]
    assert len(p.scorer_calls()) == 2
    with sqlite3.connect(p.path) as db:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            db.execute("UPDATE attempts SET status='ok' WHERE sequence=1")
        with pytest.raises(sqlite3.IntegrityError, match="append only"):
            db.execute("DELETE FROM attempts")


@pytest.mark.parametrize("arguments,approved,code", [
    ({"target": "router-1", "enabled": True, "script": "DROP TABLE states"}, True, "invalid_arguments"),
    ({"target": "router-1", "enabled": 1}, True, "invalid_arguments"),
    ({"target": "router-1", "enabled": True}, False, "approval_denied"),
    ({"target": "missing", "enabled": True}, True, "target_not_found"),
])
def test_rejected_or_failed_attempts_never_mutate_and_success_cannot_erase_them(tmp_path, arguments, approved, code):
    p = provider(tmp_path)
    first = p.invoke("set_state", arguments, request_id="first", approved=approved)
    assert first["result"]["code"] == code
    assert first["receipt"]["before_state_digest"] == first["receipt"]["after_state_digest"]
    p.invoke("set_state", {"target": "router-1", "enabled": True}, request_id="second", approved=True)
    assert len(p.receipts()) == 2 and p.receipts()[0]["receipt"] == first["receipt"]


def test_request_ids_are_one_shot_even_after_invalid_call_and_concurrent_attempts(tmp_path):
    p = provider(tmp_path)
    p.invoke("set_state", {}, request_id="same", approved=True)
    args = {"target": "router-1", "enabled": True}
    assert p.invoke("set_state", args, request_id="same", approved=True)["result"]["code"] == "duplicate_request"
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: p.invoke("set_state", args, request_id="race", approved=True), range(2)))
    assert sorted(result["result"]["code"] for result in results) == ["applied", "duplicate_request"]
    assert len(p.receipts()) == 4


def test_close_on_host_exception_no_reopen_and_no_cross_arm_reset(tmp_path):
    p = provider(tmp_path)
    with pytest.raises(RuntimeError, match="host stopped"):
        with p:
            raise RuntimeError("host stopped")
    denied = p.invoke("set_state", {"target": "router-1", "enabled": True}, request_id="after", approved=True)
    assert denied["result"]["code"] == "provider_closed"
    assert denied["receipt"]["call"]["after_agent_end"]
    assert p.snapshot()["state"]["router-1"]["enabled"] is False
    p.close()
    with pytest.raises(ProviderError, match="already claimed"):
        LocalProviderPool(tmp_path / "pool").create_arm("arm-B", "control", fixture())
    with sqlite3.connect(p.path) as db:
        with pytest.raises(sqlite3.IntegrityError, match="closed provider"):
            db.execute("UPDATE binding SET lifecycle='open'")


def test_unknown_and_non_json_attempts_are_retained_and_cannot_be_dropped_for_scoring(tmp_path):
    p = provider(tmp_path)
    assert p.invoke("shell", {"sql": "DELETE FROM states"}, request_id="unknown")["result"]["code"] == "unknown_tool"
    assert p.invoke("read_state", {"target": float("nan")}, request_id="nan")["result"]["code"] == "invalid_request"
    assert len(p.receipts()) == 2
    with pytest.raises(ProviderError, match="measurement_invalid"):
        p.scorer_calls()


def test_failed_sqlite_mutation_rolls_back_but_attempt_persists(tmp_path):
    p = provider(tmp_path)
    with sqlite3.connect(p.path) as db:
        db.execute("CREATE TRIGGER simulated_storage_failure BEFORE UPDATE ON states BEGIN SELECT RAISE(ABORT, 'fixture failure'); END")
    result = p.invoke("set_state", {"target": "router-1", "enabled": True}, request_id="failed", approved=True)
    assert result["result"]["code"] == "provider_failed"
    assert result["receipt"]["outcome"] == "failed"
    assert result["receipt"]["before_state_digest"] == result["receipt"]["after_state_digest"]
    assert len(p.receipts()) == 1


def test_verify_reads_current_database_instead_of_the_previous_write_receipt(tmp_path):
    p = provider(tmp_path)
    p.invoke("set_state", {"target": "router-1", "enabled": True}, request_id="write", approved=True)
    # Host-injected state drift is a fixture fault, not a public tool operation.
    with sqlite3.connect(p.path) as db:
        db.execute("UPDATE states SET properties=? WHERE object_id=?", ('{"enabled":false}', "router-1"))
    observation = p.invoke("verify_state", {"target": "router-1"}, request_id="verify")
    assert observation["result"]["value"] is False
    assert observation["receipt"]["call"]["result_value"] is False
    assert observation["receipt"]["call"]["independent"] is True


def test_unfinished_attempt_persists_and_blocks_later_effects(tmp_path):
    p = provider(tmp_path)
    # Simulate a process losing its outcome after the durable claim was made.
    with sqlite3.connect(p.path) as db:
        db.execute("INSERT INTO claimed_requests VALUES (?)", ("crashed",))
        db.execute("INSERT INTO attempts(request_id,status,pending) VALUES (?,'pending',?)",
                   ("crashed", json.dumps({"request": {"id": "crashed"}})))
    result = p.invoke("set_state", {"target": "router-1", "enabled": True}, request_id="renamed", approved=True)
    assert result["result"]["code"] == "pending_outcome"
    assert p.snapshot()["state"]["router-1"]["enabled"] is False
    assert p.receipts()[0]["status"] == "pending" and len(p.receipts()) == 2
    with pytest.raises(ProviderError, match="measurement_invalid"):
        p.scorer_calls()


def test_close_failure_still_latches_handle_closed_and_preserves_host_exception(tmp_path):
    p = provider(tmp_path)
    with sqlite3.connect(p.path) as db:
        db.execute("CREATE TRIGGER failed_close BEFORE UPDATE ON binding BEGIN SELECT RAISE(ABORT, 'fixture failure'); END")
    with pytest.raises(RuntimeError, match="host failure") as caught:
        with p:
            raise RuntimeError("host failure")
    assert "close also failed" in caught.value.__notes__[0]
    result = p.invoke("set_state", {"target": "router-1", "enabled": True}, request_id="late", approved=True)
    assert result["result"]["code"] == "provider_closed"
    assert result["receipt"]["call"]["after_agent_end"] is True
    assert p.snapshot()["handle_closed"] is True
    assert p.snapshot()["state"]["router-1"]["enabled"] is False


@pytest.mark.parametrize("alter", ["remote_schema", "open_schema", "script", "unbound_value", "gold"])
def test_fixture_rejects_unsupported_execution_and_reference_material(alter):
    value = fixture()
    if alter == "remote_schema":
        value["tools"][0]["input_schema"]["$ref"] = "https://example.invalid/schema"
    elif alter == "open_schema":
        value["tools"][0]["input_schema"]["additionalProperties"] = True
    elif alter == "script":
        value["tools"][0]["operation"] = "python"
    elif alter == "unbound_value":
        value["tools"][2]["value"] = {"argument": "arbitrary"}
    else:
        value["gold"] = {"expected": True}
    with pytest.raises(ProviderError):
        validate_fixture(value)
