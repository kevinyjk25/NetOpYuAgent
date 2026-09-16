"""Installed DSH + scripted HTTP model + isolated SQLite + original read Runtime.

An integration probe, NOT the twelve-task pilot, semantic evidence, a real 9B
run, or a writing capability. The same inert source/tool/task reaches each DSH.
The compiler returns an online scripted proposal, admitted by existing code.
Treatment routes exact read prefixes through the original graph while DSH keeps
the open L1 work. Deterministic rejection falls back before any Runtime read.
Nothing runs after DSH to complete its task. All model traffic is scripted.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import shutil
import sqlite3
import subprocess
import threading
import time

from dsh_adapter.settings import sync_settings
from evaluation.bounded_budget import BudgetLedger
from evaluation.bounded_pilot import seal, write_new
from evaluation.bounded_provider import FIXTURE_SCHEMA, LocalProviderPool
from evaluation.bounded_transport import MODEL, ModelBroker, ScriptedModel, strict_json
from evaluation.dsh_support import _default_dsh_binary, _node_path, parse_dumped_config, REQUIRED_DISABLED_IDS, SAFE_ACTIVE_IDS
from evaluation.local_read_fixture import packet_for
from network_runtime.contracts import sha256_json
from network_runtime.l0.hybrid import GovernedHybridFlow, qualify_hybrid
from network_runtime.l0.hybrid_execution import HostHybridConsent, context_digest, run_hybrid
from skill_authoring import compiler, isolated_compiler, local_execution
from skill_authoring.model_endpoint import bind_model_routes
from skill_authoring.source import _bundle, _document

ROOT = Path(__file__).resolve().parents[1]
SOURCE = """---
name: bounded-snapshot-fixture
description: Read one explicitly supplied local snapshot and explain uncertainty.
---
Use read_export with the caller's path to read the snapshot. Explain the observed
text without claiming approval or permission. Never mutate state or run scripts.
This is a synthetic transport fixture, not a collected public Skill.
"""
TASK = "Read the supplied snapshot at /sandbox/snapshot and explain what is observed; do not change anything."
INPUT_SCHEMA = compiler.obj({"path": {"type": "string"}})
OUTPUT_SCHEMA = compiler.obj({"text": {"type": "string"}})
CATALOG = [{"name": "read_export", "description": "Read the exact scoped local snapshot (simulation)",
            "input_schema": INPUT_SCHEMA, "output_schema": OUTPUT_SCHEMA}]
ARGUMENTS = {"path": "/sandbox/snapshot"}


def fixture():
    return {"schema": FIXTURE_SCHEMA, "state": {"/sandbox/snapshot": {
        "text": "Local simulated campus snapshot. Approval is not present."}}, "tools": [{
        "name": "read_export", "description": CATALOG[0]["description"], "input_schema": INPUT_SCHEMA,
        "contract_id": "read_export", "kind": "read", "operation": "read", "target": {"argument": "path"},
        "property": "text", "value": None, "requires_approval": False}]}


def packet():
    path = "SKILL.md"
    bundle = _bundle({"apiVersion": "effect-runtime.io/translation-intake/v1", "candidateId": "bounded-dsh-fixture",
        "repository": "local/synthetic-transport-fixture", "commitSha": "0" * 40,
        "snapshotDigest": sha256_json(SOURCE), "entryPath": path,
        "documents": [_document(path, SOURCE.encode(), mode="100644", origin="local_development_fixture")],
        "supplementAttempts": [], "parentBundleDigest": None,
        "evidenceRole": "synthetic_mechanical_wiring_not_public_skill"})
    return packet_for(bundle, {"task": TASK, "inputSchema": INPUT_SCHEMA, "tools": [{
        "name": "read_export", "inputSchema": INPUT_SCHEMA, "outputSchema": OUTPUT_SCHEMA,
        "annotations": {"readOnlyHint": True}}]})


def scripted_proposal():
    return {"mode": "read_prefix", "intent_summary": "Read the exact caller snapshot under the host binding and retain the complete original task for native DSH reasoning.",
        "reads": [{"id": "n0", "after": [], "evidence": ["task"], "tool": "read_export",
                   "arguments": {"path": {"caller": "input#/path"}}}], "boundaries": []}


def responder(*, reject_compiler=False):
    def respond(stage, request):
        if stage == "compiler":
            value = {"invalid": "scripted schema rejection"} if reject_compiler else scripted_proposal()
            return {"content": json.dumps(value)}
        if stage not in {"agent", "fallback"}:
            raise ValueError("unexpected probe model role")
        results = [message for message in request["messages"] if message.get("role") == "tool"]
        if not results:
            return {"content": "", "tool_calls": [{"id": "snapshot-call",
                "function": {"name": "read_export", "arguments": ARGUMENTS}}]}
        value = strict_json(results[-1]["content"])
        if not isinstance(value, dict) or set(value) != {"text"} or not isinstance(value["text"], str):
            raise ValueError("DSH did not receive a successful scoped read result")
        return {"content": "Scripted integration probe ended after the scoped read. This is not an LLM or semantic success result."}
    return respond


class ToolHost:
    """A fixed host mapping; no oracle-driven dispatch or post-agent completion."""
    def __init__(self, provider, broker, directory, *, treatment, model_routes=None):
        self.provider, self.broker, self.directory = provider, broker, Path(directory)
        self.treatment = treatment
        self.model_routes = Path(model_routes) if model_routes is not None else None
        if self.model_routes is not None and not self.model_routes.is_absolute():
            raise ValueError("absolute host-owned model routes required")
        self.token = secrets.token_hex(24)
        # Lifecycle checks never wait for compilation, a Provider call, or HTTP.
        self.lock = threading.RLock()
        self.dispatch_lock = threading.Lock()
        self.journal_lock = threading.Lock()
        self.idle = threading.Event()
        self.idle.set()
        self.close_done = threading.Event()
        self.inflight = 0
        self.claimed = False
        self.closed = False
        self.halted = False
        self.server = None
        self.route = "native" if not treatment else None
        self.graph = None
        self.result = None
        self.errors = []
        self.study_id = broker.ledger.inspect_arm(broker.arm_id)["arm"]["study_id"]
        self.journal_path = self.directory / "host-attempts.sqlite3"
        descriptor = os.open(self.journal_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        with sqlite3.connect(self.journal_path, timeout=1) as db:
            db.executescript("""CREATE TABLE events(sequence INTEGER PRIMARY KEY AUTOINCREMENT, value TEXT NOT NULL);
                CREATE TRIGGER immutable_update BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT, 'append only'); END;
                CREATE TRIGGER immutable_delete BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT, 'append only'); END;""")
        self._record("host_created", binding=provider.binding, treatment=treatment)

    def _record(self, event, *, attempt_id=None, **details):
        with self.journal_lock, sqlite3.connect(self.journal_path, timeout=1) as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute("SELECT value FROM events ORDER BY sequence DESC LIMIT 1").fetchone()
            value = seal({"event": event, "attempt_id": attempt_id, "arm_id": self.broker.arm_id,
                          "at": time.time(), "details": details,
                          "previous_digest": json.loads(previous[0])["digest"] if previous else sha256_json(self.provider.binding)})
            db.execute("INSERT INTO events(value) VALUES (?)", (json.dumps(value, allow_nan=False),))

    def attempts(self):
        """Durable append-only events, including rejected requests and delivery."""
        with self.journal_lock, sqlite3.connect(self.journal_path, timeout=1) as db:
            return [json.loads(row[0]) for row in db.execute("SELECT value FROM events ORDER BY sequence")]

    def _halt(self, reason):
        with self.lock:
            self.halted = True
        self.broker.ledger.pause_study(self.study_id, reason)

    def _guard(self):
        with self.lock:
            if self.closed or self.halted or self.broker.closed:
                raise PermissionError("tool host terminated or halted")
        remaining = self.broker.ledger.check_arm(self.broker.arm_id)
        with self.lock:
            if self.closed or self.halted or self.broker.closed:
                raise PermissionError("tool host terminated or halted")
        return remaining

    def _enter(self):
        with self.lock:
            if self.closed or self.halted:
                raise PermissionError("tool host terminated or halted")
            self.inflight += 1
            self.idle.clear()

    def _leave(self):
        with self.lock:
            self.inflight -= 1
            if not self.inflight:
                self.idle.set()

    def call(self, body, *, _attempt_id=None):
        attempt_id = _attempt_id or secrets.token_hex(16)
        entered = False
        try:
            try:
                frozen = strict_json(json.dumps(body, allow_nan=False))
            except (TypeError, ValueError):
                frozen = {"invalid_non_json_request": True}
            self._record("tool_requested", attempt_id=attempt_id, request=frozen)
            body = frozen
            self._enter()
            entered = True
            self._guard()
            if (not isinstance(body, dict) or set(body) != {"tool", "arguments", "request_id"}
                    or body["tool"] != "read_export" or not isinstance(body["arguments"], dict)
                    or not isinstance(body["request_id"], str) or not body["request_id"]):
                raise ValueError("host-bound request required")
            with self.lock:
                if self.closed or self.halted or self.claimed:
                    raise PermissionError("one-shot probe call unavailable or already claimed")
                self.claimed = True
            with self.dispatch_lock:
                value = self._dispatch(body)
            self._guard()
            self._record("tool_completed", attempt_id=attempt_id, result_digest=sha256_json(value), route=self.route)
            return value
        except BaseException as exc:
            self.errors.append(type(exc).__name__)
            self._halt("tool_host_call_failed_or_unknown")
            self._record("tool_rejected_or_unknown", attempt_id=attempt_id, error_type=type(exc).__name__, route=self.route)
            raise
        finally:
            if entered:
                self._leave()

    def _dispatch(self, body):
        self._guard()
        if self.treatment:
            if self.model_routes is None:
                raise PermissionError("treatment requires explicit host model routes")
            p = packet()
            try:
                # The immutable context remains bound until even a late compiler
                # exits; closing the broker cannot restore a legacy endpoint.
                with bind_model_routes(self.model_routes, self.broker.arm_id):
                    self._guard()
                    choice = isolated_compiler.invoke(p, list(compiler.pages_for(p)), self.directory / "compiler")
                    self._guard()
            except isolated_compiler.ResponseRejected:
                self._guard()
                self.route = "fallback"
                self.broker.set_agent_stage("fallback")
            else:
                self._guard()
                compiled = compiler.compile_proposal(p, list(compiler.pages_for(p)), choice)
                self._guard()
                write_new(self.directory / "compilation.json", compiled)
                if compiled.get("status") != "compiled_mixed_candidate_requires_review":
                    raise ValueError("scripted proposal was not admitted")
                raw = copy.deepcopy(compiled["flow"])
                raw["nodes"] = [n for n in raw["nodes"] if n["kind"] == "strict_region"]
                raw["outputs"] = [n["id"] for n in raw["nodes"]]
                raw["max_model_calls"] = 0
                flow = GovernedHybridFlow.model_validate(raw)
                reads, bindings = local_execution.bindings_for(p, {}, [])
                def observe(arguments):
                    value = self.read(arguments, body["request_id"], origin="runtime")
                    return value, {"source": "actual_isolated_sqlite_fixture", "simulation": True,
                                   "fixtureDigest": self.provider.binding["fixture_digest"]}
                bindings = {key: replace(binding, observe=observe) for key, binding in bindings.items()}
                qualification = qualify_hybrid(flow, reads)
                context = local_execution.local_context()
                self._guard()
                self.graph = run_hybrid(flow, body["arguments"], reads=reads, read_bindings=bindings,
                    reasoners={}, gates={}, context=context, consent=HostHybridConsent(
                        qualification["graphDigest"], sha256_json(body["arguments"]), context_digest(context)))
                self._guard()
                write_new(self.directory / "graph.json", self.graph)
                if self.graph["status"] != "governed_graph_completed" or self.result is None:
                    raise ValueError("original Runtime did not complete the scoped read")
                self.route = "runtime_read_prefix"
        if self.route in {"native", "fallback"}:
            self.read(body["arguments"], body["request_id"], origin="agent")
        self._guard()
        return copy.deepcopy(self.result)

    def read(self, arguments, request_id, *, origin):
        self._guard()
        envelope = self.provider.invoke("read_export", arguments, request_id=request_id, origin=origin)
        self._guard()
        value = envelope["result"]
        if not value["ok"] or not value.get("exists") or not isinstance(value.get("value"), str):
            raise ValueError("scoped snapshot not observed")
        # Same fixed projection on both arms; never synthesize a value or reuse a
        # private fixture directly. Full receipt stays in the host-only journal.
        self.result = {"text": value["value"]}
        return copy.deepcopy(self.result)

    def start(self):
        with self.lock:
            if self.closed or self.server is not None:
                raise PermissionError("tool host cannot restart")
        host = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass
            def do_GET(self):  # noqa: N802
                self.respond(False)
            def do_POST(self):  # noqa: N802
                self.respond(True)
            def respond(self, post):
                attempt_id = secrets.token_hex(16)
                entered = False
                try:
                    try:
                        host._record("http_received", attempt_id=attempt_id, method="POST" if post else "GET")
                        host._enter()
                        entered = True
                        remaining = host._guard()["remaining_seconds"]
                        self.connection.settimeout(min(2.0, remaining))
                        expected = "/" + host.token + ("/invoke" if post else "/catalog")
                        if self.path != expected:
                            raise PermissionError("unknown host capability")
                        if post:
                            size = int(self.headers.get("Content-Length", "0"))
                            if not 0 < size <= 65536:
                                raise ValueError("bounded JSON required")
                            raw = self.rfile.read(size)
                            host._record("http_body", attempt_id=attempt_id, size=len(raw),
                                         body_digest="sha256:" + hashlib.sha256(raw).hexdigest())
                            if len(raw) != size:
                                raise ValueError("truncated request body")
                            value = host.call(strict_json(raw), _attempt_id=attempt_id)
                        else:
                            value = CATALOG
                        status = 200
                    except Exception as exc:
                        host.errors.append(type(exc).__name__)
                        host._halt("tool_host_http_failed_or_unknown")
                        host._record("http_rejected_or_unknown", attempt_id=attempt_id, error_type=type(exc).__name__)
                        value, status = {"error": type(exc).__name__}, 409
                    # Headers, body and flush are all part of uncertain delivery.
                    try:
                        if status == 200:
                            host._guard()
                        data = json.dumps(value, allow_nan=False).encode()
                        self.connection.settimeout(2.0)
                        self.send_response(status)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(data)))
                        self.end_headers()
                        self.wfile.write(data)
                        self.wfile.flush()
                        if status == 200:
                            host._guard()
                        host._record("http_delivered", attempt_id=attempt_id, status=status,
                                     result_digest=sha256_json(value))
                    except Exception as exc:
                        self.close_connection = True
                        host.errors.append(type(exc).__name__)
                        host._halt("tool_host_delivery_unknown")
                        host._record("http_delivery_unknown", attempt_id=attempt_id, error_type=type(exc).__name__)
                finally:
                    if entered:
                        host._leave()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=0.05), daemon=True)
        self.thread.start()
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}/{self.token}"
        return self

    def close(self, timeout=2.0):
        """Revoke immediately; bounded drain never waits on the dispatch lock.

        Python callbacks cannot be killed. A timed-out cleanup remains a daemon
        and is reported as undrained, never as a successful arm observation.
        """
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 <= timeout <= 10:
            raise ValueError("close timeout must be between zero and ten seconds")
        deadline = time.monotonic() + timeout
        with self.lock:
            first = not self.closed
            self.closed = True
            active = self.inflight
            if active:
                self.halted = True
        if first:
            def cleanup():
                try:
                    if active:
                        self._halt("tool_host_closed_with_inflight_request")
                    self._record("host_closed", inflight=active)
                    if self.server is not None:
                        self.server.shutdown()
                        self.server.server_close()
                        self.thread.join(timeout=max(0, deadline - time.monotonic()))
                    self.idle.wait(timeout=max(0, deadline - time.monotonic()))
                    self.provider.close()
                except Exception as exc:
                    self.errors.append(type(exc).__name__)
                    self._halt("tool_host_cleanup_failed")
                    self._record("host_cleanup_failed", error_type=type(exc).__name__)
                finally:
                    self.close_done.set()
            threading.Thread(target=cleanup, daemon=True).start()
        self.close_done.wait(timeout=max(0, deadline - time.monotonic()))
        with self.lock:
            drained = self.inflight == 0 and self.close_done.is_set()
            if not drained:
                self.halted = True
            return {"closed": True, "drained": drained, "inflight": self.inflight,
                    "cleanup_finished": self.close_done.is_set()}


def implementation_fingerprint():
    paths = {path for folder in ("skill_authoring", "evaluation", "network_runtime/l0")
             for path in (ROOT / folder).glob("*.py")}
    paths.update(ROOT / name for name in ("evaluation/bounded_dsh_tools.mjs", "evaluation/dsh_shadow.patch.yml",
                 "dsh_adapter/settings.py", "network_runtime/contracts.py", "requirements.txt"))
    return {str(path.relative_to(ROOT)): "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def delivered_result(directory, expected):
    """Check actual DSH feedback, not merely a successful private Provider read."""
    found = []
    for path in (directory / "transport").glob("*/request.json"):
        request = strict_json(path.read_bytes())
        if request["stage"] not in {"agent", "fallback"}:
            continue
        messages = request["source_request"]["messages"]
        for message in messages:
            if message.get("role") != "tool":
                continue
            content = message.get("content")
            if isinstance(content, list):
                if any(set(part) != {"type", "text"} or part["type"] != "text" for part in content):
                    return {"matched": False, "reason": "non_text_feedback"}
                content = "".join(part["text"] for part in content)
            try:
                value = strict_json(content)
            except (ValueError, TypeError):
                return {"matched": False, "reason": "invalid_tool_feedback"}
            prior = [call for item in messages[:messages.index(message)]
                     for call in item.get("tool_calls", []) if call.get("id") == message.get("tool_call_id")]
            found.append(value == expected and message.get("tool_call_id") == "snapshot-call"
                         and len(prior) == 1 and prior[0]["function"]["name"] == "read_export")
    return {"matched": len(found) == 1 and all(found), "feedbackCount": len(found),
            "expectedDigest": sha256_json(expected)}


def run(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    implementation = implementation_fingerprint()
    binary = _default_dsh_binary()
    node = Path(shutil.which("node", path=_node_path()))
    versions = {"dsh": subprocess.run([str(binary), "--version"], env={"PATH": _node_path(), "HOME": str(output)},
                                     capture_output=True, text=True, timeout=30, check=True).stdout.strip(),
                "node": subprocess.run([str(node), "--version"], capture_output=True, text=True,
                                       timeout=10, check=True).stdout.strip()}
    binaries = {"dsh_entry": "sha256:" + hashlib.sha256(binary.resolve().read_bytes()).hexdigest(),
                "node": "sha256:" + hashlib.sha256(node.resolve().read_bytes()).hexdigest()}
    common = {"task": TASK, "skill_text": SOURCE, "tools": CATALOG, "arguments": ARGUMENTS}
    write_new(output / "freeze.json", seal({"common": common, "common_digest": sha256_json(common),
        "provider_fixture_digest": sha256_json(fixture()), "implementation": implementation,
        "versions": versions, "entryBinaryDigests": binaries, "completeDependencySnapshot": False,
        "evidenceRole": "scripted_real_DSH_plumbing_not_pilot", "actualModelCalls": 0,
        "trustedTokenPreflight": False, "sourceScriptsExecutable": False}))
    ledger = BudgetLedger(output / "fixture-budget.sqlite")
    ledger.register_study("scripted-dsh-plumbing", sha256_json(common))
    candidate = ledger.register_candidate("scripted-dsh-plumbing", sha256_json(implementation))
    pool = LocalProviderPool(output / "provider-private")
    rows = []
    for case, arm in (("native", "A"), ("compiled", "B"), ("fallback", "B")):
        directory = output / case
        directory.mkdir()
        arm_id = ledger.start_arm("scripted-dsh-plumbing", candidate, "fixture-" + case, 1, arm, sha256_json(common))
        provider = pool.create_arm(arm_id, "control" if arm == "A" else "treatment", fixture())
        backend = ScriptedModel(responder(reject_compiler=case == "fallback"))
        broker = ModelBroker(ledger, arm_id, backend, directory / "transport").start()
        routes = {"schema": "netopyu.local-model-routes/v1", "model": MODEL,
            "model_digest": backend.digest, "arm_id": arm_id, "routes": {
                "compile": {"base_url": broker.route("compiler")}, "runtime": {"base_url": broker.route("runtime")}}}
        route_file = directory / "model-routes.json"
        write_new(route_file, routes)
        route_file.chmod(0o600)
        host = ToolHost(provider, broker, directory, treatment=arm == "B", model_routes=route_file).start()
        result = None
        process_error = None
        terminal = "incomplete"
        try:
            home = directory / "dsh-home"
            sync_settings(home / "settings.yaml", base_url=broker.route("agent"), primary_model=MODEL,
                          fast_model=MODEL, default_model=MODEL)
            patch = (ROOT / "evaluation/dsh_shadow.patch.yml").read_text()
            patch += "\n- insert:\n    - id: bounded-probe-tools\n      name: " + json.dumps(str(ROOT / "evaluation/bounded_dsh_tools.mjs")) + "\n"
            (directory / "probe.patch.yml").write_text(patch)
            # Deliberate minimal subprocess env: no inherited model keys,
            # unrelated plugin settings or user profile. DSH's own home is new.
            env = {"PATH": _node_path(), "HOME": str(home), "DSH_HOME": str(home),
                "NETOPYU_OLLAMA_API_KEY": "scripted-not-a-secret", "DSH_OTEL_ENABLED": "false",
                "NETOPYU_BOUNDED_TOOL_ENDPOINT": host.endpoint,
                "NETOPYU_L1_SHADOW_SYSTEM_PROMPT": SOURCE + "\nTask and tools are local synthetic fixtures; no source scripts."}
            argv = [str(binary), "--profile", "headless", "--patch", str(directory / "probe.patch.yml")]
            config = subprocess.run([*argv, "--dump-config"], cwd=directory, env=env, capture_output=True, text=True, timeout=30)
            (directory / "composed-config.txt").write_text(config.stdout)
            entries = {entry.entry_id: entry for entry in parse_dumped_config(config.stdout)}
            active = {key for key, entry in entries.items() if not entry.disabled}
            if (config.returncode or active != (SAFE_ACTIVE_IDS | {"bounded-probe-tools"})
                    or any(key in active for key in REQUIRED_DISABLED_IDS)
                    or "bounded-probe-tools" not in active):
                raise ValueError("DSH active plugin isolation failed")
            remaining = ledger.check_arm(arm_id)["remaining_seconds"]
            result = subprocess.run([*argv, TASK], cwd=directory, env=env, capture_output=True, text=True, timeout=min(90, remaining))
            (directory / "dsh-stdout.txt").write_text(result.stdout)
            (directory / "dsh-stderr.txt").write_text(result.stderr)
            terminal = "completed" if result.returncode == 0 else "incomplete"
        except Exception as exc:
            terminal = "timeout" if isinstance(exc, subprocess.TimeoutExpired) else "measurement_invalid"
            process_error = {"type": type(exc).__name__, "terminal": terminal}
            if isinstance(exc, subprocess.TimeoutExpired):
                for name in ("stdout", "stderr"):
                    partial = getattr(exc, name, None) or ""
                    if isinstance(partial, bytes):
                        partial = partial.decode("utf-8", errors="replace")
                    (directory / ("dsh-" + name + ".txt")).write_text(partial)
            write_new(directory / "process-error.json", process_error)
            ledger.pause_study("scripted-dsh-plumbing", "probe_process_failure")
        finally:
            broker.close()
            close_state = host.close()
            # Only collection here. Never execute Runtime/provider after DSH.
            write_new(directory / "provider-receipts.json", provider.receipts())
            write_new(directory / "provider-state.json", provider.snapshot())
            write_new(directory / "host-attempts.json", host.attempts())
            ledger.finish_arm(arm_id, terminal)
            write_new(directory / "meter.json", ledger.inspect_arm(arm_id))
        calls = ledger.inspect_arm(arm_id)["calls"]
        expected = ["agent", "agent"] if case == "native" else ["agent", "compiler", "fallback" if case == "fallback" else "agent"]
        # SQLite uses timestamps; order is evidence, not sorted by expected role.
        stages = [row["stage"] for row in calls]
        try:
            receipts = provider.scorer_calls()
        except ValueError as exc:
            receipts = []
            process_error = {"type": type(exc).__name__, "terminal": "measurement_invalid"}
        delivery = delivered_result(directory, host.result)
        code_unchanged = implementation == implementation_fingerprint()
        passed = (result is not None and result.returncode == 0 and not process_error
                  and code_unchanged and delivery["matched"] and close_state["drained"]
                  and stages == expected and len(receipts) == 1
                  and receipts[0]["outcome"] == "ok" and not host.errors and not broker.errors
                  and host.route == {"native": "native", "compiled": "runtime_read_prefix", "fallback": "fallback"}[case]
                  and provider.snapshot()["state_digest"] == provider.binding["initial_state_digest"])
        row = {"case": case, "passed": passed, "actualModelCalls": 0, "realDSH": True,
            "scriptedRequests": len(calls), "stages": stages, "route": host.route,
            "providerCalls": len(receipts), "providerBinding": provider.binding,
            "graphStatus": host.graph["status"] if host.graph else None, "processExit": result.returncode if result else None,
            "processError": process_error, "delivery": delivery, "implementationUnchanged": code_unchanged,
            "hostClose": close_state, "hostAttemptEvents": len(host.attempts()),
            "hostToolAttempts": sum(event["event"] == "tool_requested" for event in host.attempts()),
            "commonInputDigest": sha256_json(common), "hostErrors": host.errors, "brokerErrors": broker.errors}
        write_new(directory / "observation.json", seal(row))
        rows.append(row)
        print(json.dumps(row), flush=True)
        if not passed:
            ledger.pause_study("scripted-dsh-plumbing", "probe_failure")
            break
    report = seal({"schema": "netopyu.io/bounded-dsh-probe/v1", "cases": rows,
        "allPassed": len(rows) == 3 and all(row["passed"] for row in rows), "actualModelCalls": 0,
        "liveAdapterReady": False, "pilotQualified": False, "researchEvidenceEligible": False,
        "trustedTokenPreflight": False, "automaticEffectBridge": "not_tested",
        "runtimeReasonerInThisProbe": "not_invoked_native_DSH_retains_L1",
        "evidenceRole": "scripted_real_DSH_plumbing_not_pilot",
        "limitations": ["one synthetic source, not the frozen twelve-task sample",
            "source text preloaded; no Skill retrieval assessment", "fixture counts, not actual Qwen tokenization",
            "capability-scoped tools and separate DBs, not an adversarial process sandbox"],
        "versions": versions, "budget": ledger.snapshot("scripted-dsh-plumbing")})
    write_new(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output)["allPassed"] else 1)
