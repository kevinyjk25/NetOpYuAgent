"""Host-owned local SQLite simulator and receipt source for bounded R0 work.

No Gold, model, network, source script, arbitrary SQL or product Effect gateway
is accepted. A fixed pool claims each arm once; changing/deleting its registry
is outside the cooperative-operator threat model. There is no execution-reopen
API. ``fixture_digest`` hashes the full host fixture; ``initial_state_digest``
hashes the cloned state read back from SQLite (excluding revision counters).

Tools declare fixed read/set/delete operations on one object property, with
target/value bound to a constant or a named input argument. ``kind`` and
``contract_id`` are host tool identities, not evaluator reference answers.
Only a separate read invocation produces independent observation evidence.
Approval/origin are host assertions for this simulator, never product authority.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import sqlite3
import threading
import uuid

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from network_runtime.contracts import sha256_json


FIXTURE_SCHEMA = "netopyu.io/bounded-provider-fixture/v1"
RECEIPT_SCHEMA = "netopyu.io/bounded-provider-receipt/v1"
_TOOL_FIELDS = {"name", "description", "input_schema", "contract_id", "kind", "operation",
                "target", "property", "value", "requires_approval"}
_SCHEMA_KEYS = {"type", "properties", "required", "additionalProperties", "items", "enum", "const",
                "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "minLength", "maxLength",
                "pattern", "minItems", "maxItems", "uniqueItems", "minProperties", "maxProperties",
                "description", "title", "anyOf", "oneOf", "allOf", "not"}
OUTPUT_SCHEMA = {"type": "object", "additionalProperties": False,
    "properties": {"ok": {"type": "boolean"}, "code": {"type": "string"}, "simulation": {"const": True},
        "object_id": {"type": "string"}, "property": {"type": "string"}, "exists": {"type": "boolean"},
        "value": {}, "revision": {"type": "integer", "minimum": 1}, "applied": {"type": "boolean"}},
    "required": ["ok", "code", "simulation"]}


class ProviderError(ValueError):
    """Rejected simulator lifecycle or host configuration, without authority."""


def _json(value):
    def check(item):
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for element in item:
                check(element)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for element in item.values():
                check(element)
            return
        raise ProviderError("finite JSON with string keys required")
    check(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _copy(value):
    return json.loads(_json(value))


def _text(value):
    return type(value) is str and bool(value.strip()) and len(value) <= 256


def _schema(schema):
    if type(schema) is not dict or set(schema) - _SCHEMA_KEYS:
        raise ProviderError("unsupported schema keyword; references and remote schemas are forbidden")
    if schema.get("type") == "object" or "properties" in schema:
        if schema.get("additionalProperties") is not False:
            raise ProviderError("object argument schemas must be closed")
        for nested in schema.get("properties", {}).values():
            _schema(nested)
    for key in ("items", "not"):
        if key in schema:
            _schema(schema[key])
    for key in ("anyOf", "oneOf", "allOf"):
        for nested in schema.get(key, []):
            _schema(nested)
    Draft202012Validator.check_schema(schema)


def validate_fixture(value):
    """Validate host facts and operations, never an evaluator's expected calls."""
    value = _copy(value)
    if type(value) is not dict or set(value) != {"schema", "state", "tools"} or value["schema"] != FIXTURE_SCHEMA:
        raise ProviderError("exact provider fixture schema/state/tools required")
    state = value["state"]
    if type(state) is not dict or not state or any(not _text(k) or type(v) is not dict for k, v in state.items()):
        raise ProviderError("state must map object identifiers to property objects")
    if any(not _text(key) for row in state.values() for key in row):
        raise ProviderError("state property names must be nonempty strings")
    if type(value["tools"]) is not list or not 1 <= len(value["tools"]) <= 32:
        raise ProviderError("one to 32 fixed host tools required")
    names, contracts = set(), set()
    for tool in value["tools"]:
        if type(tool) is not dict or set(tool) != _TOOL_FIELDS:
            raise ProviderError("exact host tool fields required")
        if any(not _text(tool[key]) for key in ("name", "description", "contract_id", "property")):
            raise ProviderError("nonempty bounded tool identities required")
        if tool["name"] in names or tool["contract_id"] in contracts:
            raise ProviderError("duplicate host tool or contract identity")
        names.add(tool["name"])
        contracts.add(tool["contract_id"])
        if tool["operation"] not in {"read", "set", "delete"} or type(tool["requires_approval"]) is not bool:
            raise ProviderError("fixed operation and explicit approval flag required")
        kinds = {"read", "verify"} if tool["operation"] == "read" else {"effect", "compensate"}
        if tool["kind"] not in kinds:
            raise ProviderError("host receipt kind differs from operation")
        schema = tool["input_schema"]
        _schema(schema)
        if schema.get("type") != "object":
            raise ProviderError("object input schema required")
        for key in ("target", "value"):
            binding = tool[key]
            if key == "value" and tool["operation"] != "set":
                if binding is not None:
                    raise ProviderError("only set operations bind a value")
                continue
            if type(binding) is not dict or set(binding) not in ({"argument"}, {"constant"}):
                raise ProviderError("bindings are one named argument or host constant")
            if "argument" in binding and (binding["argument"] not in schema.get("properties", {})
                                          or binding["argument"] not in schema.get("required", [])):
                raise ProviderError("binding argument must be declared and required")
            if key == "target" and "constant" in binding and not _text(binding["constant"]):
                raise ProviderError("constant target must be an object identifier")
    return value


@contextmanager
def _connection(path):
    db = sqlite3.connect(path.as_uri() + "?mode=rw", uri=True, timeout=30, isolation_level=None)
    db.row_factory = sqlite3.Row
    try:
        yield db
    finally:
        db.close()


def _new_file(path):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)


class LocalProviderPool:
    """One fixed host directory and durable one-shot arm assignment registry."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry = self.root / "provider-registry.sqlite3"
        if self.registry.is_symlink():
            raise ProviderError("provider registry cannot be a symlink")
        try:
            _new_file(self.registry)
        except FileExistsError:
            pass
        with _connection(self.registry) as db:
            db.execute("CREATE TABLE IF NOT EXISTS arms (arm_id TEXT PRIMARY KEY, arm TEXT NOT NULL, "
                       "isolation_id TEXT NOT NULL UNIQUE, db_name TEXT NOT NULL UNIQUE, fixture_digest TEXT NOT NULL)")

    def create_arm(self, arm_id, arm, fixture):
        fixture = validate_fixture(fixture)
        if not _text(arm_id) or arm not in {"control", "treatment"}:
            raise ProviderError("explicit bounded arm id and control/treatment required")
        isolation_id = "simulator-" + uuid.uuid4().hex
        db_name = sha256_json(arm_id).removeprefix("sha256:") + ".sqlite3"
        with _connection(self.registry) as db:
            try:
                db.execute("INSERT INTO arms VALUES (?, ?, ?, ?, ?)",
                           (arm_id, arm, isolation_id, db_name, sha256_json(fixture)))
            except sqlite3.IntegrityError as exc:
                raise ProviderError("arm identity already claimed; no reset, reopen or replay") from exc
        # A failed initialization keeps its registry claim; never erase evidence.
        path = self.root / db_name
        _new_file(path)
        return ArmProvider._create(path, fixture, arm_id, arm, isolation_id)


class ArmProvider:
    """Host handle; expose only visible_tools() and invoke's result to an Agent."""
    @classmethod
    def _create(cls, path, fixture, arm_id, arm, isolation_id):
        self = cls()
        self.path, self._pid, self._lock = path, os.getpid(), threading.RLock()
        self._closed = False
        self._tools = {tool["name"]: tool for tool in fixture["tools"]}
        with _connection(path) as db:
            db.executescript("""
                CREATE TABLE binding (value TEXT NOT NULL, lifecycle TEXT NOT NULL);
                CREATE TABLE states (object_id TEXT PRIMARY KEY, properties TEXT NOT NULL, revision INTEGER NOT NULL);
                CREATE TABLE attempts (sequence INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT,
                    status TEXT NOT NULL, pending TEXT NOT NULL, receipt TEXT);
                CREATE TABLE claimed_requests (request_id TEXT PRIMARY KEY);
                CREATE TRIGGER preserve_attempts BEFORE DELETE ON attempts BEGIN SELECT RAISE(ABORT, 'append only'); END;
                CREATE TRIGGER immutable_receipts BEFORE UPDATE ON attempts WHEN OLD.status != 'pending'
                    BEGIN SELECT RAISE(ABORT, 'immutable receipt'); END;
                CREATE TRIGGER no_reopen BEFORE UPDATE OF lifecycle ON binding WHEN OLD.lifecycle = 'closed'
                    BEGIN SELECT RAISE(ABORT, 'closed provider'); END;
            """)
            db.execute("BEGIN IMMEDIATE")
            db.executemany("INSERT INTO states VALUES (?, ?, 1)", [(k, _json(v)) for k, v in fixture["state"].items()])
            binding = dict(arm_id=arm_id, arm=arm, isolation_id=isolation_id, fixture_digest=sha256_json(fixture),
                           initial_state_digest=sha256_json(self._state(db)), simulation=True, productEffectAuthority=False)
            db.execute("INSERT INTO binding VALUES (?, 'open')", (_json(binding),))
            db.commit()
        self._binding = binding
        return self

    @staticmethod
    def _state(db):
        return {row["object_id"]: json.loads(row["properties"]) for row in db.execute("SELECT * FROM states ORDER BY object_id")}

    @property
    def binding(self):
        return _copy(self._binding)

    def visible_tools(self):
        return [{"name": tool["name"], "description": tool["description"],
                 "input_schema": _copy(tool["input_schema"]), "output_schema": _copy(OUTPUT_SCHEMA)}
                for tool in self._tools.values()]

    def invoke(self, tool_name, arguments, *, request_id, approved=False, origin="agent"):
        """Append every attempt. Duplicate/closed/invalid attempts cannot mutate.

        request_id, approved and origin are supplied by the host transport, not
        tool arguments. A database failure leaves a pending claim that blocks
        subsequent dispatch. This method never retries a failed operation.
        """
        if os.getpid() != self._pid:
            raise ProviderError("provider handles cannot reopen execution in another process")
        try:
            request = _copy({"tool": tool_name, "arguments": arguments, "request_id": request_id,
                             "approved": approved, "origin": origin})
            json_valid = True
        except (ProviderError, ValueError):
            request, json_valid = {"invalid_non_json_request": True}, False
        with self._lock, _connection(self.path) as db:
            db.execute("BEGIN IMMEDIATE")
            state = self._state(db)
            current = db.execute("SELECT * FROM binding").fetchone()
            if json.loads(current["value"]) != self._binding:
                db.rollback()
                raise ProviderError("provider arm binding drift")
            pending_exists = db.execute("SELECT 1 FROM attempts WHERE status='pending'").fetchone() is not None
            duplicate = _text(request_id) and db.execute("SELECT 1 FROM claimed_requests WHERE request_id=?", (request_id,)).fetchone() is not None
            if _text(request_id) and not duplicate:
                db.execute("INSERT INTO claimed_requests VALUES (?)", (request_id,))
            pending = dict(request=request, request_digest=sha256_json(request), before_state_digest=sha256_json(state))
            sequence = db.execute("INSERT INTO attempts(request_id,status,pending) VALUES (?,'pending',?)",
                                  (request_id if _text(request_id) else None, _json(pending))).lastrowid
            db.commit()
            closed = self._closed or current["lifecycle"] != "open"
            code = ("provider_closed" if closed else "pending_outcome" if pending_exists
                    else "duplicate_request" if duplicate else "invalid_request" if not json_valid or not _text(request_id)
                    or not _text(tool_name) or type(approved) is not bool or type(origin) is not str
                    or origin not in {"agent", "runtime", "evaluator"} else None)
            tool = self._tools.get(tool_name) if type(tool_name) is str else None
            result, target, independent, outcome = {"ok": False, "code": code or "unknown_tool", "simulation": True}, None, False, "denied"
            try:
                db.execute("BEGIN IMMEDIATE")
                if code is None and tool is not None:
                    try:
                        Draft202012Validator(tool["input_schema"]).validate(arguments)
                        target = self._resolve(tool["target"], arguments)
                        if not _text(target):
                            raise ProviderError("invalid target")
                    except (ValidationError, KeyError, TypeError, ProviderError):
                        result["code"] = "invalid_arguments"
                    else:
                        if tool["operation"] != "read" and tool["requires_approval"] and not approved:
                            result["code"] = "approval_denied"
                        else:
                            result, independent = self._operate(db, tool, arguments, target)
                            outcome = "ok" if result["ok"] else "failed"
                envelope = self._finish(db, sequence, pending, request_id, tool_name, arguments, tool, target,
                                        result, outcome, independent, approved, origin, closed)
                db.commit()
                return envelope
            except Exception:
                db.rollback()
                # The initial attempt is durable. If storage is unavailable, it
                # remains pending and no later request can dispatch an effect.
                db.execute("BEGIN IMMEDIATE")
                result = {"ok": False, "code": "provider_failed", "simulation": True}
                envelope = self._finish(db, sequence, pending, request_id, tool_name, arguments, tool, target,
                                        result, "failed", False, approved, origin, closed)
                db.commit()
                return envelope

    @staticmethod
    def _resolve(binding, arguments):
        return arguments[binding["argument"]] if "argument" in binding else _copy(binding["constant"])

    def _operate(self, db, tool, arguments, target):
        row = db.execute("SELECT * FROM states WHERE object_id=?", (target,)).fetchone()
        if row is None:
            return {"ok": False, "code": "target_not_found", "simulation": True}, tool["operation"] == "read"
        properties, prop, revision = json.loads(row["properties"]), tool["property"], row["revision"]
        result = dict(ok=True, simulation=True, object_id=target, property=prop, revision=revision)
        if tool["operation"] == "read":
            return {**result, "code": "observed", "exists": prop in properties, "value": properties.get(prop)}, True
        if tool["operation"] == "set":
            properties[prop] = self._resolve(tool["value"], arguments)
        else:
            properties.pop(prop, None)
        db.execute("UPDATE states SET properties=?, revision=revision+1 WHERE object_id=?", (_json(properties), target))
        return {**result, "code": "applied", "applied": True, "revision": revision + 1}, False

    def _finish(self, db, sequence, pending, request_id, name, arguments, tool, target,
                result, outcome, independent, approved, origin, after_end):
        prior = db.execute("SELECT receipt FROM attempts WHERE sequence < ? AND receipt IS NOT NULL ORDER BY sequence DESC LIMIT 1",
                           (sequence,)).fetchone()
        previous = json.loads(prior[0])["receipt_digest"] if prior else sha256_json(self._binding)
        scorable = (tool is not None and type(arguments) is dict and type(approved) is bool
                    and type(origin) is str and origin in {"agent", "runtime", "evaluator"}
                    and "invalid_non_json_request" not in pending["request"])
        call = dict(id=f"{self._binding['isolation_id']}:{sequence}", contract_id=tool["contract_id"], sequence=sequence,
                    tool=tool["name"], kind=tool["kind"], object_id=target if _text(target) else "<unbound>", arguments=_copy(arguments),
                    property=tool["property"], result_value=result.get("value") if independent else None,
                    outcome=outcome, independent=independent, approved=approved, origin=origin,
                    after_agent_end=after_end) if scorable else None
        body = {"schema": RECEIPT_SCHEMA, **self._binding, **pending, "sequence": sequence,
                "result": result, "result_digest": sha256_json(result), "outcome": outcome,
                "after_state_digest": sha256_json(self._state(db)), "previous_receipt_digest": previous,
                "scorable": scorable, "call": call, "call_digest": sha256_json(call)}
        receipt = {**body, "receipt_digest": sha256_json(body)}
        db.execute("UPDATE attempts SET status=?, receipt=? WHERE sequence=?", (outcome, _json(receipt), sequence))
        return _copy({"result": result, "receipt": receipt})

    def receipts(self):
        with self._lock, _connection(self.path) as db:
            return [{"sequence": row["sequence"], "status": row["status"], "pending": json.loads(row["pending"]),
                     "receipt": json.loads(row["receipt"]) if row["receipt"] else None}
                    for row in db.execute("SELECT * FROM attempts ORDER BY sequence")]

    def scorer_calls(self):
        """No attempt is dropped to obtain a valid scorer input."""
        receipts = self.receipts()
        if any(row["receipt"] is None or not row["receipt"]["scorable"] for row in receipts):
            raise ProviderError("unprojectable attempt retained; mark observation measurement_invalid")
        return [row["receipt"]["call"] for row in receipts]

    def snapshot(self):
        with self._lock, _connection(self.path) as db:
            state = self._state(db)
            return {**self.binding, "state": state, "state_digest": sha256_json(state),
                    "lifecycle": db.execute("SELECT lifecycle FROM binding").fetchone()[0], "handle_closed": self._closed}

    def close(self):
        with self._lock:
            self._closed = True
            with _connection(self.path) as db:
                db.execute("UPDATE binding SET lifecycle='closed' WHERE lifecycle='open'")
        return self.snapshot()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except Exception as close_error:
            if exc is None:
                raise
            exc.add_note(f"provider close also failed: {type(close_error).__name__}")
        return False
