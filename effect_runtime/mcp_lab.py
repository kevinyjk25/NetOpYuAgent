"""Cross-domain MCP simulator and trusted Effect Runtime adapter.

The lab exposes six independent domains.  Each domain owns four capabilities:
read current state, validate a proposed change, apply the change, and restore
the approved snapshot.  The in-process adapter and the official MCP server use
the same store and contracts, so tests can be fast without inventing a second
execution semantics.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dsh_adapter.backend import BackendSession
from network_runtime.contracts import OutcomeIndeterminateError, sha256_json
from network_runtime.l0.compiler import compile_documents, parse_document
from network_runtime.l0.models import CompiledAtomicEffect
from network_runtime.l0_skills import L0SkillContract, REGISTRY as L0_SKILLS
from network_runtime.policies import (
    ToolContract, register_reviewed_contract, unregister_reviewed_contract,
)


DOMAINS = ("network", "iam", "cloud", "service_desk", "data", "platform")
DEFAULT_ENTITIES = {
    "network": "edge-sw-01",
    "iam": "user-alice",
    "cloud": "vm-orders-01",
    "service_desk": "inc-2026-001",
    "data": "dataset-sales",
    "platform": "checkout-api",
}
INITIAL_VALUES = {
    "network": "vlan-10",
    "iam": "viewer",
    "cloud": "small",
    "service_desk": "p3",
    "data": "internal",
    "platform": "replicas-2",
}


@dataclass(frozen=True)
class MCPToolSpec:
    name: str
    domain: str
    role: str
    capability_id: str
    action_type: str
    parameters: dict[str, dict[str, Any]]
    required: tuple[str, ...]
    contract_id: str | None = None
    internal_parameters: tuple[str, ...] = ()

    def metadata(self) -> dict[str, Any]:
        source = f"mcp:effect-{self.domain}"
        return {
            "description": f"Effect Lab {self.domain} {self.role} capability",
            "parameters": self.parameters,
            "required": list(self.required),
            "output_schema": {"type": "object", "additionalProperties": True},
            "action_type": self.action_type,
            "hitl": self.action_type != "read_only",
            "tags": ["mcp", "effect-lab", self.domain, self.role],
            "domain": self.domain,
            "sensitivity": "confidential" if self.domain in {"iam", "data"} else "internal",
            "required_roles": ["operations-reader"] if self.action_type == "read_only" else [],
            "scope_fields": ["entity_id"],
            "freshness_limit_seconds": 60,
            "capability_id": self.capability_id,
            "capability_version": "1.0.0",
            "provider_role": "observer" if self.action_type == "read_only" else (
                "compensator" if self.role == "restore" else "actor"
            ),
            "provider_kind": "effect-simulator-mcp",
            "provider_identity": f"{source}:effect-{self.domain}@1.0.0",
            "input_schema_digest": sha256_json(self.parameters),
            "output_schema_digest": sha256_json({"type": "object"}),
            "declared_contract_id": self.contract_id,
            "result_contract": "structured-content-required-v1",
            "trusted_for_writes": True,
            "internal_parameters": list(self.internal_parameters),
        }


def tool_catalog() -> tuple[MCPToolSpec, ...]:
    values: list[MCPToolSpec] = []
    for domain in DOMAINS:
        prefix = domain
        values.extend((
            MCPToolSpec(
                f"{prefix}_get_state", domain, "observation",
                f"effect.{domain}.state.get", "read_only",
                {"entity_id": {"type": "string"}}, ("entity_id",),
            ),
            MCPToolSpec(
                f"{prefix}_validate_change", domain, "preflight",
                f"effect.{domain}.change.validate", "read_only",
                {
                    "entity_id": {"type": "string"},
                    "desired_value": {"type": "string"},
                }, ("entity_id", "desired_value"),
            ),
            MCPToolSpec(
                f"{prefix}_apply_change", domain, "effect",
                f"effect.{domain}.state.apply", "reversible",
                {
                    "entity_id": {"type": "string"},
                    "desired_value": {"type": "string"},
                    "expected_revision": {"type": "integer", "minimum": 1},
                    "change_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                ("entity_id", "desired_value", "expected_revision", "change_id", "reason"),
                f"effect-{domain}-state-change-v1",
            ),
            MCPToolSpec(
                f"{prefix}_restore_state", domain, "restore",
                f"effect.{domain}.state.restore", "reversible",
                {
                    "entity_id": {"type": "string"},
                    "approved_preflight": {"type": "object"},
                }, ("entity_id", "approved_preflight"),
                f"effect-{domain}-state-restore-v1", ("approved_preflight",),
            ),
        ))
    return tuple(values)


TOOLS = tool_catalog()
TOOLS_BY_NAME = {item.name: item for item in TOOLS}


class EffectLabStore:
    """Small deterministic state system shared by MCP and Runtime tests."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.fault = "none"
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS states (
                    domain TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    value TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    PRIMARY KEY(domain, entity_id)
                );
                CREATE TABLE IF NOT EXISTS calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain TEXT NOT NULL,
                    tool TEXT NOT NULL,
                    phase TEXT NOT NULL
                );
                """
            )
            for domain in DOMAINS:
                db.execute(
                    "INSERT OR IGNORE INTO states(domain, entity_id, value, revision) VALUES(?,?,?,1)",
                    (domain, DEFAULT_ENTITIES[domain], INITIAL_VALUES[domain]),
                )

    def reset(self) -> None:
        with self._lock, self._connect() as db:
            db.execute("DELETE FROM states")
            db.execute("DELETE FROM calls")
            for domain in DOMAINS:
                db.execute(
                    "INSERT INTO states(domain, entity_id, value, revision) VALUES(?,?,?,1)",
                    (domain, DEFAULT_ENTITIES[domain], INITIAL_VALUES[domain]),
                )
        self.fault = "none"

    def _record(self, db: sqlite3.Connection, domain: str, tool: str, phase: str) -> None:
        db.execute(
            "INSERT INTO calls(domain, tool, phase) VALUES(?,?,?)",
            (domain, tool, phase),
        )

    def calls(self, *, domain: str | None = None) -> int:
        with self._connect() as db:
            if domain:
                row = db.execute("SELECT COUNT(*) AS count FROM calls WHERE domain=?", (domain,)).fetchone()
            else:
                row = db.execute("SELECT COUNT(*) AS count FROM calls").fetchone()
        return int(row["count"])

    def effect_calls(self, *, domain: str | None = None) -> int:
        with self._connect() as db:
            if domain:
                row = db.execute(
                    "SELECT COUNT(*) AS count FROM calls WHERE domain=? AND phase IN ('effect','compensate')",
                    (domain,),
                ).fetchone()
            else:
                row = db.execute(
                    "SELECT COUNT(*) AS count FROM calls WHERE phase IN ('effect','compensate')"
                ).fetchone()
        return int(row["count"])

    def phase_counts(self, *, domain: str | None = None) -> dict[str, int]:
        """Return observable provider calls grouped by execution phase."""
        with self._connect() as db:
            if domain:
                rows = db.execute(
                    "SELECT phase, COUNT(*) AS count FROM calls "
                    "WHERE domain=? GROUP BY phase",
                    (domain,),
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT phase, COUNT(*) AS count FROM calls GROUP BY phase"
                ).fetchall()
        return {str(item["phase"]): int(item["count"]) for item in rows}

    def _has_fault(self, name: str) -> bool:
        return name in {item.strip() for item in self.fault.split("+") if item.strip()}

    def state(self, domain: str, entity_id: str) -> dict[str, Any]:
        with self._lock, self._connect() as db:
            self._record(db, domain, f"{domain}_get_state", "observe")
            row = db.execute(
                "SELECT value, revision FROM states WHERE domain=? AND entity_id=?",
                (domain, entity_id),
            ).fetchone()
        if row is None:
            return {"ok": False, "code": "entity_not_found", "domain": domain, "entity_id": entity_id}
        return {
            "ok": True, "domain": domain, "entity_id": entity_id,
            "value": str(row["value"]), "revision": int(row["revision"]),
            "maintenance_window_open": not self._has_fault("guard_violation"),
            "simulation": True,
        }

    def validate(self, domain: str, entity_id: str, desired_value: str) -> dict[str, Any]:
        current = self.state(domain, entity_id)
        with self._connect() as db:
            self._record(db, domain, f"{domain}_validate_change", "preflight")
        return {
            **current,
            "valid": current.get("ok") is True and bool(desired_value.strip()),
            "desired_value": desired_value,
        }

    def apply(
        self, domain: str, entity_id: str, desired_value: str,
        expected_revision: int, change_id: str, reason: str,
    ) -> dict[str, Any]:
        if self._has_fault("provider_error_before_send"):
            raise RuntimeError("injected provider error before send")
        with self._lock, self._connect() as db:
            self._record(db, domain, f"{domain}_apply_change", "effect")
            row = db.execute(
                "SELECT revision FROM states WHERE domain=? AND entity_id=?",
                (domain, entity_id),
            ).fetchone()
            if row is None:
                return {"ok": False, "code": "entity_not_found"}
            if int(row["revision"]) != expected_revision:
                return {
                    "ok": False, "code": "revision_conflict",
                    "expected_revision": expected_revision,
                    "actual_revision": int(row["revision"]),
                }
            stored = "__verification_mismatch__" if self._has_fault("verification_mismatch") else desired_value
            revision = expected_revision + 1
            db.execute(
                "UPDATE states SET value=?, revision=? WHERE domain=? AND entity_id=?",
                (stored, revision, domain, entity_id),
            )
        if self._has_fault("after_send_unknown"):
            raise OutcomeIndeterminateError("injected disconnect after provider accepted write")
        return {
            "ok": True, "domain": domain, "entity_id": entity_id,
            "value": desired_value, "revision": revision,
            "change_id": change_id, "reason": reason, "simulation": True,
        }

    def restore(self, domain: str, entity_id: str, approved_preflight: dict[str, Any]) -> dict[str, Any]:
        if self._has_fault("compensation_failure"):
            raise RuntimeError("injected compensation failure")
        facts = approved_preflight.get("facts") if isinstance(approved_preflight, dict) else None
        if not isinstance(facts, dict) or facts.get("entity_id") != entity_id:
            return {"ok": False, "code": "approved_snapshot_invalid"}
        with self._lock, self._connect() as db:
            self._record(db, domain, f"{domain}_restore_state", "compensate")
            db.execute(
                "UPDATE states SET value=?, revision=? WHERE domain=? AND entity_id=?",
                (str(facts["value"]), int(facts["revision"]), domain, entity_id),
            )
        return {
            "ok": True, "domain": domain, "entity_id": entity_id,
            "value": facts["value"], "revision": facts["revision"], "simulation": True,
        }


def _json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class EffectLabBackendFactory:
    """Backend factory that preserves the same MCP identity and store semantics."""

    def __init__(self, store: EffectLabStore) -> None:
        self.store = store

    async def __call__(self, profile_id: str) -> BackendSession:
        expected_prefix = "effect-"
        if not profile_id.startswith(expected_prefix):
            raise ValueError(f"Effect Lab profile must start with {expected_prefix!r}")
        domain = profile_id.removeprefix(expected_prefix)
        if domain not in DOMAINS:
            raise ValueError(f"unknown Effect Lab domain {domain!r}")
        specs = [item for item in TOOLS if item.domain == domain]
        callables: dict[str, Any] = {}
        for spec in specs:
            if spec.role == "observation":
                async def get_state(arguments: dict[str, Any], *, _domain: str = domain) -> str:
                    return _json(self.store.state(_domain, str(arguments["entity_id"])))
                callables[spec.name] = get_state
            elif spec.role == "preflight":
                async def validate(arguments: dict[str, Any], *, _domain: str = domain) -> str:
                    return _json(self.store.validate(
                        _domain, str(arguments["entity_id"]), str(arguments["desired_value"]),
                    ))
                callables[spec.name] = validate
            elif spec.role == "effect":
                async def apply(arguments: dict[str, Any], *, _domain: str = domain) -> str:
                    return _json(self.store.apply(
                        _domain, str(arguments["entity_id"]), str(arguments["desired_value"]),
                        int(arguments["expected_revision"]), str(arguments["change_id"]),
                        str(arguments["reason"]),
                    ))
                callables[spec.name] = apply
            else:
                async def restore(arguments: dict[str, Any], *, _domain: str = domain) -> str:
                    return _json(self.store.restore(
                        _domain, str(arguments["entity_id"]),
                        dict(arguments["approved_preflight"]),
                    ))
                callables[spec.name] = restore
        metadata = {item.name: item.metadata() for item in specs}
        sources = {item.name: f"mcp:effect-{domain}" for item in specs}
        from runtime import ToolResultStore

        tool_store = ToolResultStore(
            db_path=str(self.store.path.with_name(self.store.path.name + ".results.sqlite")),
        )
        return BackendSession(
            mode="pragmatic", profile_id=profile_id, callables=callables,
            metadata=metadata, sources=sources,
            report={
                "mode": "pragmatic", "ready": True, "profile": profile_id,
                "transport_contract": "mcp", "simulation": True,
            },
            _tool_store=tool_store,
        )


def _authoring_document(domain: str, contract: ToolContract) -> dict[str, Any]:
    tool = f"{domain}_apply_change"
    argument = lambda name: "${arguments." + name + "}"
    parameters = {
        "entity_id": {"type": "string", "required": True, "minLength": 1, "maxLength": 128},
        "desired_value": {"type": "string", "required": True, "minLength": 1, "maxLength": 256},
        "expected_revision": {"type": "integer", "required": True, "minimum": 1},
        "change_id": {"type": "string", "required": True, "minLength": 1, "maxLength": 128},
        "reason": {"type": "string", "required": True, "minLength": 1, "maxLength": 1024},
    }
    request = {name: argument(name) for name in parameters}
    return {
        "apiVersion": "netopyu.io/l0-effect/v2",
        "kind": "AtomicEffect",
        "metadata": {
            "id": f"effect.{domain}.state.apply", "version": "1.0.0",
            "owner": "effect-runtime",
            "description": f"Reviewed cross-domain state transition for {domain}",
            "labels": {
                "runtime-tool-contract": contract.contract_id,
                "runtime-verifier": contract.verifier,
                "runtime-preflight-fields": "entity_id",
                "source": "anthropic-skill-progressive-determinization",
            },
        },
        "spec": {
            "template": "effect-runtime-cross-domain-v1",
            "profiles": [f"effect-{domain}"],
            "effect": {
                "capability": f"effect.{domain}.state.apply",
                "tool": tool, "request": request,
            },
            "intent": {
                "kind": f"set_{domain}_state", "targetFields": ["entity_id"],
                "desiredState": {"value": argument("desired_value")},
            },
            "parameters": parameters,
            "preflight": [{
                "id": "approved-state", "capability": f"effect.{domain}.state.get",
                "arguments": {"entity_id": argument("entity_id")},
                "snapshotFields": ["value", "revision"],
                "predicates": [
                    {"field": "ok", "operator": "equals", "expected": True},
                    {
                        "field": "maintenance_window_open",
                        "operator": "equals",
                        "expected": True,
                    },
                    {
                        "field": "revision",
                        "operator": "equals",
                        "expected": argument("expected_revision"),
                    },
                ],
            }],
            "verification": {
                "capability": f"effect.{domain}.state.get",
                "arguments": {"entity_id": argument("entity_id")},
                "predicates": [{
                    "field": "value", "operator": "equals",
                    "expected": argument("desired_value"),
                }],
            },
            "compensation": {
                "capability": f"effect.{domain}.state.restore",
                "tool": f"{domain}_restore_state",
                "arguments": {"entity_id": argument("entity_id")},
                "verification": {
                    "capability": f"effect.{domain}.state.get",
                    "arguments": {"entity_id": argument("entity_id")},
                    "predicates": [{"field": "restored", "operator": "equals", "expected": True}],
                },
            },
            "approval": {"required": True, "risk": "medium", "mode": "single"},
            "failurePolicy": {
                "beforeSend": "abort", "afterSendUnknown": "reconcile_read_only",
                "verificationFailed": "compensate",
                "compensationFailed": "manual_intervention",
            },
        },
    }


def _runtime_contract(domain: str) -> ToolContract:
    return ToolContract(
        contract_id=f"effect-{domain}-state-change-v1",
        preflight_tool=f"{domain}_get_state", preflight_fields=("entity_id",),
        verifier="structured-state-v1",
        rollback_tool=f"{domain}_restore_state", rollback_fields=("entity_id",),
        compensator="structured-snapshot-v1",
        capability_id=f"effect.{domain}.state.apply",
        allowed_provider_kinds=("effect-simulator-mcp",),
        verification_tool=f"{domain}_get_state", verification_fields=("entity_id",),
    )


def register_effect_lab_runtime() -> dict[str, str]:
    """Install six trusted adapter contracts and their compiled L0 artifacts."""
    digests: dict[str, str] = {}
    for domain in DOMAINS:
        tool = f"{domain}_apply_change"
        contract = _runtime_contract(domain)
        register_reviewed_contract(tool, contract)
        compiled_values = compile_documents((parse_document(
            _authoring_document(domain, contract), source=f"effect-lab:{domain}",
        ),))
        compiled = compiled_values[0]
        if not isinstance(compiled, CompiledAtomicEffect):
            raise RuntimeError("Effect Lab L0 compiler returned a non-atomic contract")
        existing = L0_SKILLS.get(compiled.metadata.id, compiled.metadata.version)
        candidate = L0SkillContract.from_compiled(compiled, type("Binding", (), {
            "tool_name": tool, "tool_contract_id": contract.contract_id,
        })())
        if existing is None:
            L0_SKILLS.register(candidate)
        elif existing.contract_hash != candidate.contract_hash:
            raise RuntimeError(f"conflicting Effect Lab L0 registration: {compiled.metadata.id}")
        digests[compiled.metadata.id] = compiled.contract_hash
    return digests


def unregister_effect_lab_runtime() -> None:
    for domain in DOMAINS:
        L0_SKILLS.unregister(f"effect.{domain}.state.apply", "1.0.0")
        unregister_reviewed_contract(f"{domain}_apply_change", _runtime_contract(domain))


@contextmanager
def effect_lab_runtime_registration():
    """Scope temporary lab contracts so production catalog counts never drift."""
    digests = register_effect_lab_runtime()
    try:
        yield digests
    finally:
        unregister_effect_lab_runtime()


def build_server(domain: str, store_path: str | Path):
    """Build one official-SDK MCP server for a single simulated domain."""
    if domain not in DOMAINS:
        raise ValueError(f"unknown Effect Lab domain {domain!r}")
    from mcp.server.mcpserver import MCPServer

    store = EffectLabStore(store_path)
    store.fault = os.environ.get("NETOPYU_EFFECT_LAB_FAULT", "none")
    server = MCPServer(
        f"effect-runtime.{domain}", version="1.0.0",
        description=f"Deterministic {domain} integration simulator",
    )
    spec_by_role = {item.role: item for item in TOOLS if item.domain == domain}

    def meta(role: str) -> dict[str, Any]:
        return {"netopyu": spec_by_role[role].metadata()}

    def get_state(entity_id: str) -> dict[str, Any]:
        return store.state(domain, entity_id)

    def validate_change(entity_id: str, desired_value: str) -> dict[str, Any]:
        return store.validate(domain, entity_id, desired_value)

    def apply_change(
        entity_id: str, desired_value: str, expected_revision: int,
        change_id: str, reason: str,
    ) -> dict[str, Any]:
        return store.apply(
            domain, entity_id, desired_value, expected_revision, change_id, reason,
        )

    def restore_state(entity_id: str, approved_preflight: dict[str, Any]) -> dict[str, Any]:
        return store.restore(domain, entity_id, approved_preflight)

    server.tool(name=spec_by_role["observation"].name, meta=meta("observation"), structured_output=True)(get_state)
    server.tool(name=spec_by_role["preflight"].name, meta=meta("preflight"), structured_output=True)(validate_change)
    server.tool(name=spec_by_role["effect"].name, meta=meta("effect"), structured_output=True)(apply_change)
    server.tool(name=spec_by_role["restore"].name, meta=meta("restore"), structured_output=True)(restore_state)
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=DOMAINS, required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8190)
    args = parser.parse_args(argv)
    server = build_server(args.domain, args.store)
    if args.transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", host=args.host, port=args.port, stateless_http=True, json_response=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ENTITIES", "DOMAINS", "EffectLabBackendFactory", "EffectLabStore",
    "INITIAL_VALUES", "MCPToolSpec", "TOOLS", "build_server",
    "effect_lab_runtime_registration", "register_effect_lab_runtime",
    "tool_catalog", "unregister_effect_lab_runtime",
]
