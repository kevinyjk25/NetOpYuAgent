"""Deterministic declarative Tool/MCP simulator for ES-P1-Wild studies.

Public Skill packages remain inert.  Human-authored Tool Catalog v2 documents
select from six reviewed operations over a fixture-backed record store; they
cannot provide code, shell commands, imports, templates, or arbitrary callables.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from dsh_adapter.backend import BackendSession
from network_runtime.contracts import OutcomeIndeterminateError, sha256_json


CATALOG_SCHEMA = "effect-runtime.io/public-skill-tool-catalog/v2"
FIXTURE_SCHEMA = "effect-runtime.io/public-skill-fixture-state/v1"
RESULT_SCHEMA = "effect-runtime.io/public-skill-fixture-tool-result/v1"
Mode = Literal["l1_native", "l0_runtime", "safe_stop"]
_IDENTIFIER = re.compile(r"^[a-zA-Z][a-zA-Z0-9._-]{1,127}$")
_OPERATIONS = {
    "static", "read_record", "validate_record", "upsert_record", "restore_record",
    "delete_record",
}
_FAULTS = {
    "none", "provider_error_before_send", "after_send_unknown",
    "verification_mismatch", "compensation_failure",
}
_JSON_TYPES: dict[str, type[Any] | tuple[type[Any], ...]] = {
    "string": str, "integer": int, "number": (int, float), "boolean": bool,
    "object": dict, "array": list,
}


@dataclass(frozen=True)
class FixtureCapability:
    capability_id: str
    tool_name: str
    description: str
    action_type: str
    input_schema: dict[str, Any]
    operation: dict[str, Any]

    @property
    def phase(self) -> str:
        if self.action_type == "read_only":
            return "observe"
        if self.operation["kind"] == "restore_record":
            return "compensate"
        return "effect"

    def metadata(self) -> dict[str, Any]:
        provider_role = "observer" if self.action_type == "read_only" else (
            "compensator" if self.operation["kind"] == "restore_record" else "actor"
        )
        parameters = dict(self.input_schema["properties"])
        internal_parameters = [] if self.action_type == "read_only" else ["effect_phase"]
        if self.operation["kind"] == "restore_record":
            internal_parameters.append(self.operation["snapshotArgument"])
        return {
            "description": self.description, "parameters": parameters,
            "required": list(self.input_schema["required"]),
            "output_schema": {"type": "object", "additionalProperties": True},
            "action_type": self.action_type, "hitl": self.action_type != "read_only",
            "tags": ["mcp", "public-skill-fixture", self.operation["kind"]],
            "domain": "public-skill", "sensitivity": "synthetic",
            "required_roles": [], "scope_fields": [], "freshness_limit_seconds": 300,
            "capability_id": self.capability_id, "capability_version": "1.0.0",
            "provider_role": provider_role, "provider_kind": "public-skill-fixture-mcp",
            "provider_identity": "mcp:public-skill-fixture@1.0.0",
            "input_schema_digest": sha256_json(self.input_schema),
            "output_schema_digest": sha256_json({"type": "object"}),
            "declared_contract_id": None, "result_contract": "structured-content-required-v1",
            "trusted_for_writes": False,
            "internal_parameters": internal_parameters,
        }


def _argument_names(operation: dict[str, Any]) -> set[str]:
    names = {
        str(operation[key]) for key in (
            "keyArgument", "revisionArgument", "snapshotArgument",
        )
        if key in operation
    }
    names.update(str(item) for item in operation.get("valueArguments", {}).values())
    return names


def _validate_operation(operation: Any, action_type: str, properties: dict[str, Any]) -> None:
    if not isinstance(operation, dict) or operation.get("kind") not in _OPERATIONS:
        raise ValueError("fixture MCP operation kind is invalid")
    kind = operation["kind"]
    required = {
        "static": {"kind", "resultKey"},
        "read_record": {"kind", "collection", "keyArgument"},
        "validate_record": {"kind", "collection", "keyArgument", "valueArguments"},
        "upsert_record": {"kind", "collection", "keyArgument", "valueArguments"},
        "restore_record": {"kind", "collection", "keyArgument", "snapshotArgument"},
        "delete_record": {"kind", "collection", "keyArgument"},
    }[kind]
    optional = {"revisionArgument"} if kind in {"upsert_record", "delete_record"} else set()
    if not required.issubset(operation) or not set(operation).issubset(required | optional):
        raise ValueError("fixture MCP operation fields mismatch")
    for key in ("collection", "keyArgument", "revisionArgument", "snapshotArgument", "resultKey"):
        if key in operation and (
            not isinstance(operation[key], str) or not operation[key].strip()
        ):
            raise ValueError("fixture MCP operation identifier is invalid")
    value_arguments = operation.get("valueArguments")
    if value_arguments is not None and (
        not isinstance(value_arguments, dict)
        or not value_arguments
        or not all(
            isinstance(field, str) and field and isinstance(argument, str) and argument
            for field, argument in value_arguments.items()
        )
    ):
        raise ValueError("fixture MCP valueArguments are invalid")
    unknown_arguments = _argument_names(operation) - set(properties)
    if unknown_arguments:
        raise ValueError("fixture MCP operation references undeclared arguments")
    read_kind = kind in {"static", "read_record", "validate_record"}
    if read_kind != (action_type == "read_only"):
        raise ValueError("fixture MCP actionType does not match its operation")
    if kind in {"upsert_record", "restore_record"} and action_type != "reversible":
        raise ValueError("fixture MCP state-changing operation must be reversible")
    if kind == "delete_record" and action_type not in {"reversible", "irreversible"}:
        raise ValueError("fixture MCP delete actionType is invalid")


def validate_fixture_catalog(value: Any) -> tuple[FixtureCapability, ...]:
    if not isinstance(value, dict) or set(value) != {
        "apiVersion", "assignmentId", "capabilities",
    }:
        raise ValueError("fixture MCP Tool Catalog fields mismatch")
    if value["apiVersion"] != CATALOG_SCHEMA:
        raise ValueError("fixture MCP Tool Catalog Schema mismatch")
    if not isinstance(value["assignmentId"], str) or not _IDENTIFIER.fullmatch(value["assignmentId"]):
        raise ValueError("fixture MCP assignmentId is invalid")
    if not isinstance(value["capabilities"], list):
        raise ValueError("fixture MCP capabilities must be a list")
    capabilities: list[FixtureCapability] = []
    identifiers: set[str] = set()
    names: set[str] = set()
    for raw in value["capabilities"]:
        if not isinstance(raw, dict) or set(raw) != {
            "capabilityId", "toolName", "description", "actionType", "inputSchema", "operation",
        }:
            raise ValueError("fixture MCP capability fields mismatch")
        capability_id = raw["capabilityId"]
        tool_name = raw["toolName"]
        if (
            not isinstance(capability_id, str) or not _IDENTIFIER.fullmatch(capability_id)
            or not isinstance(tool_name, str) or not _IDENTIFIER.fullmatch(tool_name)
            or capability_id in identifiers or tool_name in names
        ):
            raise ValueError("fixture MCP capability identity is invalid or duplicate")
        if not isinstance(raw["description"], str) or not raw["description"].strip():
            raise ValueError("fixture MCP capability description is empty")
        if raw["actionType"] not in {"read_only", "reversible", "irreversible"}:
            raise ValueError("fixture MCP capability actionType is invalid")
        schema = raw["inputSchema"]
        if (
            not isinstance(schema, dict) or schema.get("type") != "object"
            or schema.get("additionalProperties") is not False
            or not isinstance(schema.get("properties"), dict)
            or not isinstance(schema.get("required"), list)
            or not all(isinstance(item, str) for item in schema["required"])
            or not set(schema["required"]).issubset(schema["properties"])
        ):
            raise ValueError("fixture MCP inputSchema must be a closed object")
        _validate_operation(raw["operation"], raw["actionType"], schema["properties"])
        identifiers.add(capability_id)
        names.add(tool_name)
        capabilities.append(FixtureCapability(
            capability_id=capability_id, tool_name=tool_name,
            description=raw["description"], action_type=raw["actionType"],
            input_schema=copy.deepcopy(schema), operation=copy.deepcopy(raw["operation"]),
        ))
    return tuple(capabilities)


def validate_fixture_state(value: Any, *, expected_case_id: str | None = None) -> dict[str, Any]:
    required = {
        "apiVersion", "caseId", "approval", "fault", "collections", "staticResults",
        "verificationMismatchPatch",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("fixture MCP state fields mismatch")
    if value["apiVersion"] != FIXTURE_SCHEMA:
        raise ValueError("fixture MCP state Schema mismatch")
    if (
        not isinstance(value["caseId"], str) or not _IDENTIFIER.fullmatch(value["caseId"])
        or expected_case_id is not None and value["caseId"] != expected_case_id
    ):
        raise ValueError("fixture MCP state case binding mismatch")
    if (
        not isinstance(value["approval"], dict)
        or set(value["approval"]) != {"approved", "actor"}
        or not isinstance(value["approval"]["approved"], bool)
        or not isinstance(value["approval"]["actor"], str)
    ):
        raise ValueError("fixture MCP approval input is invalid")
    if value["fault"] not in _FAULTS:
        raise ValueError("fixture MCP fault is invalid")
    if not isinstance(value["collections"], dict) or not isinstance(value["staticResults"], dict):
        raise ValueError("fixture MCP state collections are invalid")
    for collection, records in value["collections"].items():
        if not isinstance(collection, str) or not collection or not isinstance(records, dict):
            raise ValueError("fixture MCP collection is invalid")
        for key, record in records.items():
            if (
                not isinstance(key, str) or not key or not isinstance(record, dict)
                or not isinstance(record.get("revision"), int)
                or isinstance(record.get("revision"), bool) or record["revision"] < 1
            ):
                raise ValueError("fixture MCP record must have a positive revision")
    patch = value["verificationMismatchPatch"]
    if not isinstance(patch, dict):
        raise ValueError("fixture MCP verification mismatch patch is invalid")
    if value["fault"] == "verification_mismatch" and not patch:
        raise ValueError("fixture MCP verification mismatch fault requires a patch")
    return copy.deepcopy(value)


def load_fixture_inputs(
    catalog_path: str | Path, fixture_path: str | Path, *, expected_case_id: str | None = None,
) -> tuple[dict[str, Any], tuple[FixtureCapability, ...], dict[str, Any]]:
    catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    capabilities = validate_fixture_catalog(catalog)
    fixture = validate_fixture_state(
        json.loads(Path(fixture_path).read_text(encoding="utf-8")),
        expected_case_id=expected_case_id,
    )
    return catalog, capabilities, fixture


def _validate_arguments(capability: FixtureCapability, arguments: dict[str, Any]) -> str | None:
    if not isinstance(arguments, dict):
        return "arguments_not_object"
    schema = capability.input_schema
    properties = schema["properties"]
    unknown = sorted(set(arguments) - set(properties) - {"effect_phase"})
    if unknown:
        return "unknown_parameters:" + ",".join(unknown)
    missing = sorted(
        key for key in schema["required"] if key not in arguments or arguments[key] in (None, "")
    )
    if missing:
        return "missing_parameters:" + ",".join(missing)
    for name, item in arguments.items():
        if name == "effect_phase":
            continue
        declared = properties[name]
        expected = _JSON_TYPES.get(str(declared.get("type") or ""))
        if expected is not None and (
            not isinstance(item, expected)
            or declared.get("type") in {"integer", "number"} and isinstance(item, bool)
        ):
            return f"invalid_type:{name}"
        if "enum" in declared and item not in declared["enum"]:
            return f"invalid_enum:{name}"
        if isinstance(item, str):
            if len(item) < int(declared.get("minLength", 0)):
                return f"below_min_length:{name}"
            if "maxLength" in declared and len(item) > int(declared["maxLength"]):
                return f"above_max_length:{name}"
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            if "minimum" in declared and item < declared["minimum"]:
                return f"below_minimum:{name}"
            if "maximum" in declared and item > declared["maximum"]:
                return f"above_maximum:{name}"
    return None


def _merge_patch(document: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(document)
    for key, value in patch.items():
        if value is None:
            result.pop(key, None)
        elif isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_patch(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class FixtureStateStore:
    """SQLite-backed deterministic state and observable call journal."""

    def __init__(self, path: str | Path, fixture: dict[str, Any]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fixture = validate_fixture_state(fixture)
        self.fixture_digest = sha256_json(self.fixture)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as database:
            database.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS records (
                    collection TEXT NOT NULL, record_key TEXT NOT NULL, document TEXT NOT NULL,
                    PRIMARY KEY(collection, record_key)
                );
                CREATE TABLE IF NOT EXISTS calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, tool_name TEXT NOT NULL,
                    capability_id TEXT NOT NULL, phase TEXT NOT NULL,
                    arguments_digest TEXT NOT NULL, outcome TEXT NOT NULL,
                    result_digest TEXT
                );
                """
            )
            row = database.execute(
                "SELECT value FROM metadata WHERE key='fixture_digest'"
            ).fetchone()
            if row is not None and str(row["value"]) != self.fixture_digest:
                raise ValueError("fixture MCP store is already bound to a different fixture")
            if row is None:
                database.execute(
                    "INSERT INTO metadata(key,value) VALUES('fixture_digest',?)",
                    (self.fixture_digest,),
                )
                database.execute(
                    "INSERT INTO metadata(key,value) VALUES('case_id',?)",
                    (self.fixture["caseId"],),
                )
                self._seed(database)

    def _seed(self, database: sqlite3.Connection) -> None:
        for collection, records in self.fixture["collections"].items():
            for key, document in records.items():
                database.execute(
                    "INSERT INTO records(collection,record_key,document) VALUES(?,?,?)",
                    (collection, key, json.dumps(document, ensure_ascii=False, sort_keys=True)),
                )

    def reset(self) -> None:
        with self._lock, self._connect() as database:
            database.execute("DELETE FROM records")
            database.execute("DELETE FROM calls")
            self._seed(database)

    def get(self, collection: str, key: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as database:
            row = database.execute(
                "SELECT document FROM records WHERE collection=? AND record_key=?",
                (collection, key),
            ).fetchone()
        return None if row is None else json.loads(str(row["document"]))

    def put(self, collection: str, key: str, document: dict[str, Any]) -> None:
        with self._lock, self._connect() as database:
            database.execute(
                "INSERT INTO records(collection,record_key,document) VALUES(?,?,?) "
                "ON CONFLICT(collection,record_key) DO UPDATE SET document=excluded.document",
                (collection, key, json.dumps(document, ensure_ascii=False, sort_keys=True)),
            )

    def delete(self, collection: str, key: str) -> bool:
        with self._lock, self._connect() as database:
            cursor = database.execute(
                "DELETE FROM records WHERE collection=? AND record_key=?", (collection, key),
            )
        return cursor.rowcount > 0

    def record_call(
        self, capability: FixtureCapability, arguments: dict[str, Any], outcome: str,
        result: dict[str, Any] | None,
    ) -> None:
        with self._lock, self._connect() as database:
            database.execute(
                "INSERT INTO calls(tool_name,capability_id,phase,arguments_digest,outcome,result_digest) "
                "VALUES(?,?,?,?,?,?)",
                (
                    capability.tool_name, capability.capability_id, capability.phase,
                    sha256_json(arguments), outcome, None if result is None else sha256_json(result),
                ),
            )

    def audit(self) -> dict[str, Any]:
        with self._connect() as database:
            rows = database.execute(
                "SELECT phase,outcome,COUNT(*) AS count FROM calls GROUP BY phase,outcome"
            ).fetchall()
            total = database.execute("SELECT COUNT(*) AS count FROM calls").fetchone()
        counts = {f"{row['phase']}:{row['outcome']}": int(row["count"]) for row in rows}
        return {
            "caseId": self.fixture["caseId"], "fixtureDigest": self.fixture_digest,
            "callCount": int(total["count"]), "counts": counts,
            "auditDigest": sha256_json({"caseId": self.fixture["caseId"], "counts": counts}),
        }


class FixtureToolEngine:
    def __init__(
        self, catalog: dict[str, Any], fixture: dict[str, Any], store_path: str | Path,
        *, mode: Mode,
    ) -> None:
        if mode not in {"l1_native", "l0_runtime", "safe_stop"}:
            raise ValueError("fixture MCP execution mode is invalid")
        self.catalog = copy.deepcopy(catalog)
        self.capabilities = validate_fixture_catalog(catalog)
        self.by_name = {item.tool_name: item for item in self.capabilities}
        self.fixture = validate_fixture_state(fixture)
        self.store = FixtureStateStore(store_path, self.fixture)
        self.mode = mode

    @property
    def approval(self) -> dict[str, Any]:
        return dict(self.fixture["approval"])

    def _result(self, ok: bool, code: str, **values: Any) -> dict[str, Any]:
        return {
            "apiVersion": RESULT_SCHEMA, "ok": ok, "code": code,
            "mode": self.mode, "simulation": True, **values,
        }

    def _operation(
        self, capability: FixtureCapability, arguments: dict[str, Any], *, runtime_authorized: bool,
    ) -> dict[str, Any]:
        operation = capability.operation
        kind = operation["kind"]
        if kind == "static":
            key = operation["resultKey"]
            if key not in self.fixture["staticResults"]:
                return self._result(False, "static_result_missing")
            return self._result(True, "static_result", result=copy.deepcopy(self.fixture["staticResults"][key]))
        collection = operation["collection"]
        key = str(arguments[operation["keyArgument"]])
        current = self.store.get(collection, key)
        if kind == "read_record":
            if current is None:
                return self._result(False, "record_not_found", collection=collection, key=key)
            return self._result(
                True, "record_found", collection=collection, key=key,
                record=current, snapshot=copy.deepcopy(current),
            )
        if kind == "validate_record":
            desired = {
                field: copy.deepcopy(arguments[argument])
                for field, argument in operation["valueArguments"].items()
            }
            return self._result(
                current is not None, "preflight_valid" if current is not None else "record_not_found",
                collection=collection, key=key, current=current, desired=desired,
            )
        if not runtime_authorized and self.mode == "l0_runtime":
            return self._result(False, "runtime_transaction_required")
        if self.mode == "safe_stop":
            return self._result(False, "translation_unqualified_safe_stop")
        if runtime_authorized and not self.approval["approved"]:
            return self._result(False, "approval_denied")
        fault = self.fixture["fault"]
        if capability.phase == "effect" and fault == "provider_error_before_send":
            raise RuntimeError("injected fixture provider error before send")
        if kind == "restore_record":
            if fault == "compensation_failure":
                raise RuntimeError("injected fixture compensation failure")
            snapshot = arguments[operation["snapshotArgument"]]
            if not isinstance(snapshot, dict):
                return self._result(False, "snapshot_invalid")
            if isinstance(snapshot.get("facts"), dict):
                snapshot = snapshot["facts"]
            if isinstance(snapshot.get("record"), dict):
                snapshot = snapshot["record"]
            self.store.put(collection, key, copy.deepcopy(snapshot))
            return self._result(True, "restored", collection=collection, key=key, record=snapshot)
        if current is None:
            return self._result(False, "record_not_found", collection=collection, key=key)
        revision_argument = operation.get("revisionArgument")
        if revision_argument is not None and arguments[revision_argument] != current.get("revision"):
            return self._result(
                False, "revision_conflict", expectedRevision=arguments[revision_argument],
                actualRevision=current.get("revision"),
            )
        if kind == "delete_record":
            self.store.delete(collection, key)
            result = self._result(True, "deleted", collection=collection, key=key)
        else:
            desired = {
                field: copy.deepcopy(arguments[argument])
                for field, argument in operation["valueArguments"].items()
            }
            updated = {**current, **desired, "revision": int(current["revision"]) + 1}
            if fault == "verification_mismatch":
                patch = (
                    self.fixture["verificationMismatchPatch"]
                    .get(collection, {}).get(key, {})
                )
                updated = _merge_patch(updated, patch)
            self.store.put(collection, key, updated)
            result = self._result(True, "effect_accepted", collection=collection, key=key, record={
                **current, **desired, "revision": int(current["revision"]) + 1,
            })
        if fault == "after_send_unknown":
            raise OutcomeIndeterminateError("injected fixture disconnect after accepted effect")
        return result

    def invoke(
        self, tool_name: str, arguments: dict[str, Any], *, runtime_authorized: bool = False,
    ) -> dict[str, Any]:
        capability = self.by_name.get(tool_name)
        if capability is None:
            return self._result(False, "tool_not_found")
        values = dict(arguments)
        injected_phase = values.pop("effect_phase", None)
        if injected_phase is not None:
            if injected_phase not in {"execute", "compensate"}:
                return self._result(False, "runtime_phase_invalid")
            runtime_authorized = True
        error = _validate_arguments(capability, values)
        if error:
            result = self._result(False, error)
            self.store.record_call(capability, values, "rejected", result)
            return result
        try:
            result = self._operation(capability, values, runtime_authorized=runtime_authorized)
        except OutcomeIndeterminateError:
            self.store.record_call(capability, values, "outcome_indeterminate", None)
            raise
        except Exception:
            self.store.record_call(capability, values, "provider_error", None)
            raise
        self.store.record_call(
            capability, values, "accepted" if result["ok"] else "rejected", result,
        )
        return result


class FixtureBackendFactory:
    """Build BackendSession objects without exposing the fixture operations to the Agent."""

    def __init__(
        self, catalog: dict[str, Any], fixture: dict[str, Any], store_path: str | Path,
        *, mode: Mode,
    ) -> None:
        self.engine = FixtureToolEngine(catalog, fixture, store_path, mode=mode)

    async def __call__(self, profile_id: str) -> BackendSession:
        if profile_id != "public-skill-fixture":
            raise ValueError("fixture MCP profile must be 'public-skill-fixture'")
        callables: dict[str, Any] = {}
        for capability in self.engine.capabilities:
            async def invoke(arguments: dict[str, Any], *, _tool: str = capability.tool_name) -> str:
                return json.dumps(
                    self.engine.invoke(_tool, dict(arguments)), ensure_ascii=False, sort_keys=True,
                )
            callables[capability.tool_name] = invoke
        return BackendSession(
            mode="pragmatic", profile_id=profile_id, callables=callables,
            metadata={item.tool_name: item.metadata() for item in self.engine.capabilities},
            sources={item.tool_name: "mcp:public-skill-fixture" for item in self.engine.capabilities},
            report={
                "mode": "pragmatic", "ready": True, "profile": profile_id,
                "transport_contract": "mcp", "simulation": True,
                "fixtureDigest": self.engine.store.fixture_digest,
            },
        )


def _python_type(schema: dict[str, Any]) -> type[Any]:
    declared = str(schema.get("type") or "")
    value = _JSON_TYPES.get(declared, Any)
    return value if isinstance(value, type) else Any


def _handler(engine: FixtureToolEngine, capability: FixtureCapability):
    async def invoke(**arguments: Any) -> dict[str, Any]:
        return engine.invoke(capability.tool_name, arguments)

    invoke.__name__ = capability.tool_name
    invoke.__doc__ = capability.description
    parameters = []
    required = set(capability.input_schema["required"])
    for name, schema in capability.input_schema["properties"].items():
        parameters.append(inspect.Parameter(
            name, inspect.Parameter.KEYWORD_ONLY,
            default=inspect.Parameter.empty if name in required else None,
            annotation=_python_type(schema),
        ))
    invoke.__signature__ = inspect.Signature(parameters, return_annotation=dict[str, Any])  # type: ignore[attr-defined]
    return invoke


def build_server(
    catalog: dict[str, Any], fixture: dict[str, Any], store_path: str | Path, *, mode: Mode,
):
    from mcp.server.mcpserver import MCPServer

    engine = FixtureToolEngine(catalog, fixture, store_path, mode=mode)
    server = MCPServer(
        "effect-runtime.public-skill-fixture", version="1.0.0",
        description="Deterministic declarative ES-P1-Wild Tool simulator",
    )
    for capability in engine.capabilities:
        server.tool(
            name=capability.tool_name,
            description=capability.description,
            meta={"netopyu": capability.metadata()}, structured_output=True,
        )(_handler(engine, capability))
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--mode", choices=("l1_native", "l0_runtime", "safe_stop"), required=True)
    parser.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8191)
    args = parser.parse_args(argv)
    catalog, _, fixture = load_fixture_inputs(args.catalog, args.fixture)
    server = build_server(catalog, fixture, args.store, mode=args.mode)
    if args.transport == "stdio":
        server.run("stdio")
    else:
        server.run(
            "streamable-http", host=args.host, port=args.port,
            stateless_http=True, json_response=True,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CATALOG_SCHEMA", "FIXTURE_SCHEMA", "FixtureBackendFactory", "FixtureCapability",
    "FixtureStateStore", "FixtureToolEngine", "build_server", "load_fixture_inputs",
    "validate_fixture_catalog", "validate_fixture_state",
]
