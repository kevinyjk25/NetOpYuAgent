from __future__ import annotations

import asyncio
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.public_skill_fixture_mcp import (
    FixtureBackendFactory,
    FixtureToolEngine,
    _handler,
    build_server,
    validate_fixture_catalog,
    validate_fixture_state,
)
from integrations.clients.mcp_client import MCPClient
from network_runtime.contracts import OutcomeIndeterminateError


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _schema(*required: str, **properties):  # type: ignore[no-untyped-def]
    return {
        "type": "object", "additionalProperties": False,
        "required": list(required), "properties": properties,
    }


def _catalog() -> dict[str, object]:
    return {
        "apiVersion": "effect-runtime.io/public-skill-tool-catalog/v2",
        "assignmentId": "wild-assignment-001",
        "capabilities": [
            {
                "capabilityId": "directory.user.read", "toolName": "get_user",
                "description": "Read one user record", "actionType": "read_only",
                "inputSchema": _schema("user_id", user_id={"type": "string", "minLength": 1}),
                "operation": {"kind": "read_record", "collection": "users", "keyArgument": "user_id"},
            },
            {
                "capabilityId": "directory.user.validate", "toolName": "validate_user_role",
                "description": "Validate a proposed role", "actionType": "read_only",
                "inputSchema": _schema(
                    "user_id", "role", user_id={"type": "string"}, role={"type": "string"},
                ),
                "operation": {
                    "kind": "validate_record", "collection": "users", "keyArgument": "user_id",
                    "valueArguments": {"role": "role"},
                },
            },
            {
                "capabilityId": "directory.user.role.set", "toolName": "set_user_role",
                "description": "Set one approved user role", "actionType": "reversible",
                "inputSchema": _schema(
                    "user_id", "role", "expected_revision",
                    user_id={"type": "string"}, role={"type": "string"},
                    expected_revision={"type": "integer", "minimum": 1},
                ),
                "operation": {
                    "kind": "upsert_record", "collection": "users", "keyArgument": "user_id",
                    "valueArguments": {"role": "role"}, "revisionArgument": "expected_revision",
                },
            },
            {
                "capabilityId": "directory.user.restore", "toolName": "restore_user",
                "description": "Restore an approved user snapshot", "actionType": "reversible",
                "inputSchema": _schema(
                    "user_id", "approved_preflight", user_id={"type": "string"},
                    approved_preflight={"type": "object"},
                ),
                "operation": {
                    "kind": "restore_record", "collection": "users", "keyArgument": "user_id",
                    "snapshotArgument": "approved_preflight",
                },
            },
            {
                "capabilityId": "policy.user.read", "toolName": "get_policy",
                "description": "Return the static role policy", "actionType": "read_only",
                "inputSchema": _schema(),
                "operation": {"kind": "static", "resultKey": "policy"},
            },
            {
                "capabilityId": "directory.user.delete", "toolName": "delete_user",
                "description": "Delete one user", "actionType": "irreversible",
                "inputSchema": _schema(
                    "user_id", "expected_revision", user_id={"type": "string"},
                    expected_revision={"type": "integer"},
                ),
                "operation": {
                    "kind": "delete_record", "collection": "users", "keyArgument": "user_id",
                    "revisionArgument": "expected_revision",
                },
            },
        ],
    }


def _fixture(*, approved: bool = True, fault: str = "none") -> dict[str, object]:
    return {
        "apiVersion": "effect-runtime.io/public-skill-fixture-state/v1",
        "caseId": "wild-assignment-001-01",
        "approval": {"approved": approved, "actor": "reviewer-01"},
        "fault": fault,
        "collections": {"users": {"alice": {"role": "viewer", "revision": 1}}},
        "staticResults": {"policy": {"allowedRoles": ["viewer", "editor"]}},
        "verificationMismatchPatch": (
            {"users": {"alice": {"role": "corrupted"}}}
            if fault == "verification_mismatch" else {}
        ),
    }


def _set(engine: FixtureToolEngine, *, runtime_authorized: bool = False):  # type: ignore[no-untyped-def]
    return engine.invoke(
        "set_user_role",
        {"user_id": "alice", "role": "editor", "expected_revision": 1},
        runtime_authorized=runtime_authorized,
    )


def test_fixture_tool_catalog_and_operations_are_declarative(tmp_path: Path) -> None:
    capabilities = validate_fixture_catalog(_catalog())
    assert len(capabilities) == 6
    engine = FixtureToolEngine(_catalog(), _fixture(), tmp_path / "native.sqlite", mode="l1_native")
    assert engine.invoke("get_policy", {})["result"]["allowedRoles"] == ["viewer", "editor"]
    before = engine.invoke("get_user", {"user_id": "alice"})
    assert before["record"] == {"role": "viewer", "revision": 1}
    assert engine.invoke("validate_user_role", {"user_id": "alice", "role": "editor"})["ok"]
    changed = _set(engine)
    assert changed["code"] == "effect_accepted"
    after = engine.invoke("get_user", {"user_id": "alice"})
    assert after["record"] == {"role": "editor", "revision": 2}
    restored = engine.invoke(
        "restore_user", {"user_id": "alice", "approved_preflight": before["snapshot"]},
    )
    assert restored["code"] == "restored"
    assert engine.invoke("get_user", {"user_id": "alice"})["record"]["role"] == "viewer"
    assert engine.store.audit()["callCount"] == 7


def test_fixture_modes_enforce_runtime_and_safe_stop_boundaries(tmp_path: Path) -> None:
    runtime = FixtureToolEngine(_catalog(), _fixture(), tmp_path / "runtime.sqlite", mode="l0_runtime")
    denied = _set(runtime)
    assert denied["code"] == "runtime_transaction_required"
    assert runtime.invoke("get_user", {"user_id": "alice"})["record"]["role"] == "viewer"
    accepted = _set(runtime, runtime_authorized=True)
    assert accepted["code"] == "effect_accepted"

    safe = FixtureToolEngine(_catalog(), _fixture(), tmp_path / "safe.sqlite", mode="safe_stop")
    assert _set(safe)["code"] == "translation_unqualified_safe_stop"
    assert safe.invoke("get_user", {"user_id": "alice"})["ok"] is True

    denied_approval = FixtureToolEngine(
        _catalog(), _fixture(approved=False), tmp_path / "denied.sqlite", mode="l0_runtime",
    )
    assert _set(denied_approval, runtime_authorized=True)["code"] == "approval_denied"
    native_control = FixtureToolEngine(
        _catalog(), _fixture(approved=False), tmp_path / "control.sqlite", mode="l1_native",
    )
    assert _set(native_control)["ok"] is True


def test_fixture_faults_are_deterministic_and_audited(tmp_path: Path) -> None:
    before = FixtureToolEngine(
        _catalog(), _fixture(fault="provider_error_before_send"),
        tmp_path / "before.sqlite", mode="l1_native",
    )
    with pytest.raises(RuntimeError, match="before send"):
        _set(before)
    assert before.store.get("users", "alice")["role"] == "viewer"  # type: ignore[index]

    unknown = FixtureToolEngine(
        _catalog(), _fixture(fault="after_send_unknown"),
        tmp_path / "unknown.sqlite", mode="l1_native",
    )
    with pytest.raises(OutcomeIndeterminateError):
        _set(unknown)
    assert unknown.store.get("users", "alice")["role"] == "editor"  # type: ignore[index]
    assert unknown.store.audit()["counts"]["effect:outcome_indeterminate"] == 1

    mismatch = FixtureToolEngine(
        _catalog(), _fixture(fault="verification_mismatch"),
        tmp_path / "mismatch.sqlite", mode="l1_native",
    )
    assert _set(mismatch)["record"]["role"] == "editor"
    assert mismatch.store.get("users", "alice")["role"] == "corrupted"  # type: ignore[index]

    compensation = FixtureToolEngine(
        _catalog(), _fixture(fault="compensation_failure"),
        tmp_path / "compensate.sqlite", mode="l1_native",
    )
    with pytest.raises(RuntimeError, match="compensation"):
        compensation.invoke(
            "restore_user",
            {"user_id": "alice", "approved_preflight": {"role": "viewer", "revision": 1}},
        )


def test_fixture_rejects_unknown_arguments_and_malformed_contracts(tmp_path: Path) -> None:
    engine = FixtureToolEngine(_catalog(), _fixture(), tmp_path / "invalid.sqlite", mode="l1_native")
    rejected = engine.invoke("get_user", {"user_id": "alice", "scope": "all"})
    assert rejected["code"] == "unknown_parameters:scope"
    malformed = _catalog()
    malformed["capabilities"][0]["operation"]["keyArgument"] = "undeclared"  # type: ignore[index]
    with pytest.raises(ValueError, match="undeclared arguments"):
        validate_fixture_catalog(malformed)
    invalid_fixture = _fixture(fault="verification_mismatch")
    invalid_fixture["verificationMismatchPatch"] = {}
    with pytest.raises(ValueError, match="requires a patch"):
        validate_fixture_state(invalid_fixture)


def test_fixture_backend_injects_runtime_authority_and_mcp_signature(tmp_path: Path) -> None:
    factory = FixtureBackendFactory(
        _catalog(), _fixture(), tmp_path / "backend.sqlite", mode="l0_runtime",
    )
    session = asyncio.run(factory("public-skill-fixture"))
    plan = SimpleNamespace(
        plan_id="plan-1", plan_hash="sha256:plan", intent_hash="sha256:intent",
        preflight=[SimpleNamespace(value={"record": {"role": "viewer", "revision": 1}})],
    )
    result = json.loads(asyncio.run(session.invoke_effect(
        "set_user_role",
        {"user_id": "alice", "role": "editor", "expected_revision": 1},
        plan=plan, phase="execute",
    )))
    assert result["code"] == "effect_accepted"
    capability = validate_fixture_catalog(_catalog())[0]
    signature = inspect.signature(_handler(factory.engine, capability))
    assert list(signature.parameters) == ["user_id"]
    server = build_server(_catalog(), _fixture(), tmp_path / "server.sqlite", mode="l1_native")
    assert server is not None


def test_fixture_official_mcp_stdio_transport(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    fixture = tmp_path / "fixture.json"
    catalog.write_text(json.dumps(_catalog()), encoding="utf-8")
    fixture.write_text(json.dumps(_fixture()), encoding="utf-8")

    async def exercise() -> None:
        client = MCPClient.from_config({
            "public-fixture": {
                "transport": "stdio",
                "command": [
                    sys.executable, "-m", "evaluation.public_skill_fixture_mcp",
                    "--catalog", str(catalog), "--fixture", str(fixture),
                    "--store", str(tmp_path / "mcp.sqlite"), "--mode", "l1_native",
                ],
                "cwd": str(Path(__file__).resolve().parent.parent),
                "domain": "public-skill", "trusted_for_writes": False,
                "expected_server_name": "effect-runtime.public-skill-fixture",
                "expected_server_version": "1.0.0",
            },
        })
        await client.connect_all()
        try:
            assert len(client.list_tools()) == 6
            result = await client.call_tool("get_user", {"user_id": "alice"})
            assert result.is_error is False
            assert result.structured_content["record"]["role"] == "viewer"
        finally:
            await client.disconnect_all()

    asyncio.run(exercise())


def test_versioned_fixture_examples_are_valid() -> None:
    root = PROJECT_ROOT / "evaluation/fixtures/public-skill-fixture"
    catalog = json.loads((root / "catalog-v2.json").read_text(encoding="utf-8"))
    fixture = json.loads((root / "case-state-v1.json").read_text(encoding="utf-8"))
    assert len(validate_fixture_catalog(catalog)) == 4
    assert validate_fixture_state(
        fixture, expected_case_id="wild-assignment-example-01",
    )["approval"]["approved"] is True
