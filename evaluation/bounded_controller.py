"""Fixed R0 engineering batch: installed DSH, inert dialogues, original Runtime.

This is not a live-model pilot or a second executor. A host-written, predeclared
dialogue only exercises transport. It never reads references or fixture values.
Treatment compiles at most two per-tool read prefixes online inside DSH's tool
turn; the original Runtime executes each admitted read. Remaining read calls
explicitly fall back to DSH's native tool route. Writes stop: automatic Effect
admission is not implemented here. Semantic judgments default to unknown.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import subprocess
import threading
import time

from jsonschema import Draft202012Validator

from dsh_adapter.settings import sync_settings
from evaluation.bounded_budget import BudgetLedger
from evaluation.bounded_dsh_probe import ToolHost, implementation_fingerprint
from evaluation.bounded_pilot import (
    assess, case_initial_state_digest, schedule, seal, validate_agent_input,
    validate_protocol, validate_references, write_new,
)
from evaluation.bounded_provider import LocalProviderPool, validate_fixture
from evaluation.bounded_reacceptance import consume_claim, linked_counts, verify_parent
from evaluation.bounded_scoring import bind_review, seal_observation
from evaluation.bounded_transport import MODEL, ModelBroker, ScriptedModel, strict_json
from evaluation.dsh_support import (
    REQUIRED_DISABLED_IDS, SAFE_ACTIVE_IDS, _default_dsh_binary, _node_path, parse_dumped_config,
)
from evaluation.local_read_fixture import packet_for
from network_runtime.contracts import sha256_json
from network_runtime.l0.hybrid import GovernedHybridFlow, qualify_hybrid
from network_runtime.l0.hybrid_execution import HostHybridConsent, context_digest, run_hybrid
from skill_authoring import compiler, isolated_compiler, local_execution
from skill_authoring.contracts import budget as author_budget
from skill_authoring.model_endpoint import bind_model_routes
from skill_authoring.source import _bundle, _document

ROOT = Path(__file__).resolve().parents[1]
STUDY_ID = "r0-controller-engineering-v1"
REGISTRY = ROOT / "artifacts/bounded-r0-engineering/budget.sqlite3"
PROVIDER_ROOT = ROOT / "artifacts/bounded-r0-engineering/providers"
ENDING = "Mechanical R0 dialogue ended. This is not a semantic success claim or a live model response."


def validate_dialogues(cases, dialogues):
    """Freeze finite mechanical instructions, not an oracle or adaptive policy."""
    frozen = strict_json(json.dumps(dialogues, allow_nan=False))
    if not isinstance(frozen, dict) or set(frozen) != {c["case_id"] for c in cases}:
        raise ValueError("one predeclared dialogue for every assigned case required")
    for case in cases:
        tools = {t["name"]: t for t in case["agent_input"]["tools"]}
        kinds = {t["name"]: t["kind"] for t in validate_fixture(case["provider_fixture"])["tools"]}
        steps = frozen[case["case_id"]]
        if not isinstance(steps, list) or not 1 <= len(steps) <= 6:
            raise ValueError("one to six predeclared tool turns required")
        for step in steps:
            if (not isinstance(step, dict) or set(step) != {"tool", "arguments"}
                    or step["tool"] not in tools or not isinstance(step["arguments"], dict)):
                raise ValueError("dialogue must use exact public tool and arguments")
            if kinds.get(step["tool"]) not in {"read", "verify"}:
                raise ValueError("R0 mechanical dialogues permit read/verify only, never writes")
            Draft202012Validator(tools[step["tool"]]["input_schema"]).validate(step["arguments"])
    return frozen


def mechanical_responder(steps):
    """No Gold, state, task criterion or expected answer enters this closure."""
    steps = copy.deepcopy(steps)

    def respond(stage, request):
        if stage == "compiler":
            payload = strict_json(request["messages"][-1]["content"])
            tools = payload["hostTools"]
            if len(tools) != 1:
                raise ValueError("mechanical compiler supports exactly one declared read tool")
            return {"content": json.dumps({"mode": "read_prefix", "intent_summary":
                "Read exactly the caller supplied arguments through the declared read contract, retaining the original source and task for native DSH reasoning.",
                "reads": [{"id": "n0", "after": [], "evidence": ["task"],
                           "tool": tools[0]["name"], "arguments": {"caller": "input#"}}], "boundaries": []})}
        if stage not in {"agent", "fallback"}:
            raise ValueError("this R0 dialogue does not invoke a Runtime reasoner")
        # Progress depends only on receipt count, never receipt values or Gold.
        count = sum(m.get("role") == "tool" for m in request["messages"])
        if count > len(steps):
            raise ValueError("unexpected extra tool feedback")
        if count == len(steps):
            return {"content": ENDING}
        step = steps[count]
        return {"content": "", "tool_calls": [{"id": f"mechanical-{count}", "function": {
            "name": step["tool"], "arguments": copy.deepcopy(step["arguments"])}}]}
    return respond


def read_packet(inputs, tool):
    """Original author input from complete inert source and public tool schema."""
    documents = [_document("SKILL.md", inputs["skill_text"].encode(), mode="100644", origin="r0_inert_source")]
    documents.extend(_document(r["path"], r["text"].encode(), mode="100644", origin="r0_inert_reference")
                     for r in inputs["references"])
    bundle = _bundle({"apiVersion": "effect-runtime.io/translation-intake/v1", "candidateId": "r0-read-prefix",
        "repository": "r0/frozen-public-input", "commitSha": "0" * 40,
        "snapshotDigest": sha256_json(documents), "entryPath": "SKILL.md", "documents": documents,
        "supplementAttempts": [], "parentBundleDigest": None,
        "evidenceRole": "engineering_per_tool_read_prefix_not_whole_skill_translation"})
    return packet_for(bundle, {"task": inputs["task"], "inputSchema": tool["input_schema"], "tools": [{
        "name": tool["name"], "inputSchema": tool["input_schema"], "outputSchema": tool["output_schema"],
        "annotations": {"readOnlyHint": True}}]})


class DynamicToolHost(ToolHost):
    """Generic multi-call host; lifecycle/journal inherited from the closed probe."""
    def __init__(self, provider, broker, directory, *, inputs, fixture, treatment, model_routes=None):
        super().__init__(provider, broker, directory, treatment=treatment, model_routes=model_routes)
        self.inputs = validate_agent_input(inputs)
        fixture = validate_fixture(fixture)
        self.catalog = copy.deepcopy(self.inputs["tools"])
        self.tools = {t["name"]: t for t in self.catalog}
        self.kinds = {t["name"]: t["kind"] for t in fixture["tools"]}
        if set(self.tools) != set(self.kinds):
            raise ValueError("public and Provider tool identities differ")
        for tool in fixture["tools"]:
            if tool["input_schema"] != self.tools[tool["name"]]["input_schema"]:
                raise ValueError("public and Provider argument schemas differ")
        self.claims, self.prefixes, self.proposals = set(), {}, []
        self.routes, self.completed = [], []
        self.compile_attempts = 0

    def call(self, body, *, _attempt_id=None):
        attempt = _attempt_id or secrets.token_hex(16)
        entered = False
        try:
            body = strict_json(json.dumps(body, allow_nan=False))
            self._record("tool_requested", attempt_id=attempt, request=body)
            self._enter()
            entered = True
            self._guard()
            if (not isinstance(body, dict) or set(body) != {"tool", "arguments", "request_id"}
                    or body["tool"] not in self.tools or not isinstance(body["arguments"], dict)
                    or not isinstance(body["request_id"], str) or not body["request_id"]):
                raise ValueError("host-bound public tool request required")
            Draft202012Validator(self.tools[body["tool"]]["input_schema"]).validate(body["arguments"])
            with self.lock:
                if body["request_id"] in self.claims:
                    raise PermissionError("tool request already claimed; no replay")
                self.claims.add(body["request_id"])
            with self.dispatch_lock:
                self._guard()
                value, route = self._dispatch_dynamic(body)
            self._guard()
            self.routes.append(route)
            self.route = ("safe_stop" if "safe_stop" in self.routes else "native" if not self.treatment
                          else "fallback" if "fallback" in self.routes else "runtime")
            self.completed.append({"request_id": body["request_id"], "tool": body["tool"],
                                   "result_digest": sha256_json(value), "route": route})
            self._record("tool_completed", attempt_id=attempt, **self.completed[-1])
            return copy.deepcopy(value)
        except BaseException as exc:
            self.errors.append(type(exc).__name__)
            self._halt("controller_tool_failed_or_unknown")
            self._record("tool_rejected_or_unknown", attempt_id=attempt, error_type=type(exc).__name__)
            raise
        finally:
            if entered:
                self._leave()

    def _read(self, body, origin):
        self._guard()
        envelope = self.provider.invoke(body["tool"], body["arguments"],
                                        request_id=body["request_id"], origin=origin)
        self._guard()
        value = envelope["result"]
        Draft202012Validator(self.tools[body["tool"]]["output_schema"]).validate(value)
        return copy.deepcopy(value)

    def _compile(self, tool):
        self._guard()
        if self.model_routes is None:
            raise PermissionError("explicit arm-bound compiler route required")
        if self.compile_attempts >= 2:
            return None
        packet = read_packet(self.inputs, tool)
        visible = list(compiler.pages_for(packet))
        wire = isolated_compiler.make_request(packet, visible)
        preflight = author_budget(wire)
        self._guard()
        if not preflight["accepted"]:
            # The unchanged author rejects these bytes before any HTTP request.
            # Do not truncate sources, invent an online proposal, or call this
            # uncertain model usage. Native DSH retains the full original input.
            self._record("known_compiler_input_budget_rejection", tool=tool["name"],
                         request_digest=sha256_json(wire), author_budget=preflight)
            return None
        self.compile_attempts += 1
        directory = self.directory / f"compile-{self.compile_attempts}"
        directory.mkdir()
        try:
            with bind_model_routes(self.model_routes, self.broker.arm_id):
                self._guard()
                choice = isolated_compiler.invoke(packet, visible, directory / "author")
                self._guard()
        except isolated_compiler.ResponseRejected:
            self._guard()
            self._record("known_compiler_response_rejection", tool=tool["name"])
            return None
        # This mechanical interceptor supports only the exact current read.
        reads = choice.get("reads")
        if (choice.get("mode") != "read_prefix" or not isinstance(reads, list) or len(reads) != 1
                or reads[0].get("tool") != tool["name"] or reads[0].get("after") != []
                or reads[0].get("arguments") != {"caller": "input#"}):
            self._record("known_unsupported_read_prefix", proposal_digest=sha256_json(choice))
            return None
        compiled = compiler.compile_proposal(packet, visible, choice)
        self._guard()
        write_new(directory / "compilation.json", compiled)
        if compiled.get("status") != "compiled_mixed_candidate_requires_review":
            raise ValueError("original author did not admit the online proposal")
        raw = copy.deepcopy(compiled["flow"])
        raw["nodes"] = [n for n in raw["nodes"] if n["kind"] == "strict_region"]
        if len(raw["nodes"]) != 1:
            raise ValueError("exactly one original strict read region required")
        raw["outputs"], raw["max_model_calls"] = [raw["nodes"][0]["id"]], 0
        flow = GovernedHybridFlow.model_validate(raw)
        contracts, bindings = local_execution.bindings_for(packet, {}, [])
        qualification = qualify_hybrid(flow, contracts)
        self._guard()
        self.proposals.append({"tool": tool["name"], "compiled_digest": compiled["reportDigest"],
                               "graph_digest": qualification["graphDigest"]})
        return flow, contracts, bindings, qualification

    def _dispatch_dynamic(self, body):
        name = body["tool"]
        if self.kinds[name] not in {"read", "verify"}:
            return {"ok": False, "code": "automatic_effect_bridge_not_tested", "simulation": True}, "safe_stop"
        if not self.treatment:
            return self._read(body, "agent"), "native"
        if name not in self.prefixes:
            self.prefixes[name] = self._compile(self.tools[name])
        prefix = self.prefixes[name]
        if prefix is None:
            self.broker.set_agent_stage("fallback")
            return self._read(body, "agent"), "fallback"
        flow, contracts, bindings, qualification = prefix
        observed = []

        def observe(arguments):
            if arguments != body["arguments"] or observed:
                raise PermissionError("Runtime changed exact arguments or repeated the scoped read")
            value = self._read(body, "runtime")
            observed.append(value)
            return value, {"source": "actual_isolated_sqlite_fixture", "simulation": True,
                           "fixtureDigest": self.provider.binding["fixture_digest"]}

        bound = {key: replace(binding, observe=observe) for key, binding in bindings.items()}
        context = local_execution.local_context()
        self._guard()
        graph = run_hybrid(flow, body["arguments"], reads=contracts, read_bindings=bound,
            reasoners={}, gates={}, context=context, consent=HostHybridConsent(
                qualification["graphDigest"], sha256_json(body["arguments"]), context_digest(context)))
        self._guard()
        write_new(self.directory / f"graph-{len(self.completed)}.json", graph)
        if graph["status"] != "governed_graph_completed" or len(observed) != 1:
            raise ValueError("original Runtime did not finish one exact read")
        return observed[0], "runtime"

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
                attempt, entered = secrets.token_hex(16), False
                try:
                    try:
                        host._enter()
                        entered = True
                        remaining = host._guard()["remaining_seconds"]
                        self.connection.settimeout(min(2, remaining))
                        if self.path != "/" + host.token + ("/invoke" if post else "/catalog"):
                            raise PermissionError("unknown tool capability")
                        host._record("http_received", attempt_id=attempt, method="POST" if post else "GET")
                        if post:
                            size = int(self.headers.get("Content-Length", "0"))
                            if not 0 < size <= 65536:
                                raise ValueError("bounded JSON required")
                            raw = self.rfile.read(size)
                            if len(raw) != size:
                                raise ValueError("truncated request")
                            value = host.call(strict_json(raw), _attempt_id=attempt)
                        else:
                            value = host.catalog
                        status = 200
                    except Exception as exc:
                        host.errors.append(type(exc).__name__)
                        host._halt("controller_tool_http_unknown")
                        host._record("http_rejected_or_unknown", attempt_id=attempt, error_type=type(exc).__name__)
                        value, status = {"error": type(exc).__name__}, 409
                    try:
                        if status == 200:
                            host._guard()
                        data = json.dumps(value, allow_nan=False).encode()
                        self.connection.settimeout(2)
                        self.send_response(status)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(data)))
                        self.end_headers()
                        self.wfile.write(data)
                        self.wfile.flush()
                        if status == 200:
                            host._guard()
                        host._record("http_delivered", attempt_id=attempt, status=status)
                    except Exception as exc:
                        self.close_connection = True
                        host.errors.append(type(exc).__name__)
                        host._halt("controller_tool_delivery_unknown")
                        host._record("http_delivery_unknown", attempt_id=attempt, error_type=type(exc).__name__)
                finally:
                    if entered:
                        host._leave()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=0.05), daemon=True)
        self.thread.start()
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}/{self.token}"
        return self


def scripted_broker(ledger, arm_id, responder, transport_dir):
    return ModelBroker(ledger, arm_id, ScriptedModel(responder), transport_dir)


def _idle_before_revoke(host, broker, ledger, arm_id):
    """Bounded bookkeeping drain before close; close itself revokes immediately."""
    deadline = time.monotonic() + min(2, ledger.check_arm(arm_id)["remaining_seconds"])
    while time.monotonic() < deadline:
        ledger.check_arm(arm_id)
        with broker.lock:
            clear = not broker.inflight and not broker.workers and not broker.connections
        if clear and host.idle.is_set():
            return True
        threading.Event().wait(min(0.01, max(0, deadline - time.monotonic())))
    return False


def _literal_source_patch(patch):
    """Only the controller uses opaque source insertion, never template prose.

    DSH interpolates persona templates but does not rescan variable values. The
    host plugin registers this one variable; source bytes are not escaped or
    rewritten. Preserve the legacy shadow overlay for the earlier probes.
    """
    original = "    persona: !!js process.env.NETOPYU_L1_SHADOW_SYSTEM_PROMPT"
    if patch.count(original) != 1:
        raise ValueError("expected exactly one original shadow persona binding")
    return patch.replace(original, '    persona: "{{netopyu_inert_source}}"', 1)


def _run_dsh(inputs, directory, broker, host, binary):
    home = directory / "dsh-home"
    broker.ledger.check_arm(broker.arm_id)
    sync_settings(home / "settings.yaml", base_url=broker.route("agent"), primary_model=MODEL,
                  fast_model=MODEL, default_model=MODEL)
    patch = _literal_source_patch((ROOT / "evaluation/dsh_shadow.patch.yml").read_text())
    patch += "\n- insert:\n    - id: bounded-probe-tools\n      name: " + json.dumps(str(ROOT / "evaluation/bounded_dsh_tools.mjs")) + "\n"
    (directory / "controller.patch.yml").write_text(patch)
    env = {"PATH": _node_path(), "HOME": str(home), "DSH_HOME": str(home),
        "NETOPYU_OLLAMA_API_KEY": "local-fixture-not-a-secret", "DSH_OTEL_ENABLED": "false",
        "NETOPYU_BOUNDED_TOOL_ENDPOINT": host.endpoint,
        "NETOPYU_BOUNDED_LITERAL_SOURCE": "1",
        "NETOPYU_L1_SHADOW_SYSTEM_PROMPT": "Inert source material; no source scripts may execute.\n" +
            json.dumps({k: inputs[k] for k in ("skill_text", "references")}, ensure_ascii=False)}
    argv = [str(binary), "--profile", "headless", "--patch", str(directory / "controller.patch.yml")]
    remaining = broker.ledger.check_arm(broker.arm_id)["remaining_seconds"]
    config = subprocess.run([*argv, "--dump-config"], cwd=directory, env=env, capture_output=True,
                            text=True, timeout=min(30, remaining))
    (directory / "composed-config.txt").write_text(config.stdout)
    active = {e.entry_id for e in parse_dumped_config(config.stdout) if not e.disabled}
    if config.returncode or active != SAFE_ACTIVE_IDS | {"bounded-probe-tools"} or active & REQUIRED_DISABLED_IDS:
        raise ValueError("DSH active plugin isolation failed")
    remaining = broker.ledger.check_arm(broker.arm_id)["remaining_seconds"]
    request = json.dumps({k: inputs[k] for k in ("task", "arguments", "tools")}, ensure_ascii=False)
    result = subprocess.run([*argv, request], cwd=directory, env=env, capture_output=True, text=True, timeout=remaining)
    broker.ledger.check_arm(broker.arm_id)
    (directory / "dsh-stdout.txt").write_text(result.stdout)
    (directory / "dsh-stderr.txt").write_text(result.stderr)
    return result


def delivered_feedback(transport_dir, completed):
    """Compare actual DSH feedback to actual host results, not private Gold."""
    expected = {r["request_id"]: r for r in completed}
    found = {}
    try:
        for path in sorted(Path(transport_dir).glob("*/request.json")):
            request = strict_json(path.read_bytes())
            if request["stage"] not in {"agent", "fallback"}:
                continue
            messages = request["source_request"]["messages"]
            for index, message in enumerate(messages):
                if message.get("role") != "tool":
                    continue
                call_id, content = message.get("tool_call_id"), message.get("content")
                if isinstance(content, list):
                    if any(set(p) != {"type", "text"} or p["type"] != "text" for p in content):
                        raise ValueError("non-text tool feedback")
                    content = "".join(p["text"] for p in content)
                value_digest = sha256_json(strict_json(content))
                prior = [call for m in messages[:index] for call in m.get("tool_calls", []) if call.get("id") == call_id]
                if (call_id not in expected or len(prior) != 1
                        or prior[0]["function"]["name"] != expected[call_id]["tool"]
                        or value_digest != expected[call_id]["result_digest"]):
                    raise ValueError("tool feedback binding mismatch")
                found[call_id] = value_digest
        return {"matched": bool(expected) and set(found) == set(expected), "feedback_count": len(found)}
    except (ValueError, KeyError, TypeError):
        return {"matched": False, "feedback_count": len(found), "reason": "invalid_feedback_binding"}


def _observation(reference, arm, provider, host, meter, artifact, valid, terminal, reviewer):
    usage = [{"id": c["request_id"], "stage": {"agent": "native", "compiler": "compile"}.get(c["stage"], c["stage"]),
              "input_tokens": c["actual_input"], "output_tokens": c["actual_output"],
              "wall_ms": c["elapsed_ms"]}
             for c in meter["calls"]]
    proposals = host.proposals if host else []
    observation = {"case_id": reference["case_id"], "repetition": 1, "arm": arm,
        "initial_state_digest": provider.binding["initial_state_digest"], "isolation_id": provider.binding["isolation_id"],
        "route": (host.route if host and host.route else "native" if arm == "control" else "safe_stop"),
        "outcome": "completed" if valid and terminal == "completed" else "outcome_unknown",
        "protocol_completed": valid, "timed_out": terminal == "timeout", "sandbox_escape": False,
        "artifact_digest": artifact, "initial_artifact_digest": artifact, "revision_count": 0,
        "calls": provider.scorer_calls(), "model_usage": usage, "wall_ms": meter["arm"]["elapsed_ms"],
        "online_proposal": None if arm == "control" else {"source": "online_session",
            "proposal_digest": sha256_json(proposals) if proposals else None, "loadable": bool(proposals)}}
    judgments = {"review_kind": "external_developer_review", "measurement_valid": valid,
        "criteria": {c["id"]: "unknown" for c in reference["criteria"]}, "false_completion": None,
        "duties": {d["id"]: {"fidelity": "unknown", "representation": "unknown", "strict_correct": None}
                   for d in reference["duties"]} if arm == "treatment" else {}}
    if reviewer is not None:
        judgments = reviewer(copy.deepcopy(reference), copy.deepcopy(observation), copy.deepcopy(judgments))
        # A semantic reviewer cannot turn failed collection/delivery into valid
        # measurement, even when providing otherwise affirmative judgments.
        judgments["measurement_valid"] = bool(valid and judgments["measurement_valid"])
    observation["review"] = bind_review(reference, observation, judgments)
    return seal_observation(observation)


def run(protocol, references, output, *, dialogues, registry=REGISTRY, provider_root=PROVIDER_ROOT,
        broker_factory=scripted_broker, reviewer=None, dsh_binary=None, reacceptance_claim=None):
    """Consume one immutable 12-pair engineering schedule; never reset for output."""
    protocol = validate_protocol(protocol)
    references = validate_references(protocol, references)
    cases = protocol["development_cases"]
    dialogues = validate_dialogues(cases, dialogues)
    output, registry, provider_root = Path(output).resolve(), Path(registry).resolve(), Path(provider_root).resolve()
    if registry.is_relative_to(output) or provider_root.is_relative_to(output):
        raise ValueError("engineering registry and Provider pool must be fixed outside the run output")
    binary = Path(dsh_binary) if dsh_binary is not None else _default_dsh_binary()
    implementation = implementation_fingerprint()
    assignments = [{**row, "arm": arm} for row in schedule(cases, phase="development", seed=protocol["seed"])
                   for arm in row["arms"]]
    study_id = STUDY_ID
    if reacceptance_claim is not None:
        study_id = consume_claim(registry, reacceptance_claim, output, protocol["digest"], sha256_json(dialogues))
    frozen = seal({"protocol_digest": protocol["digest"], "dialogues": dialogues, "schedule": assignments,
        "implementation": implementation, "dsh_entry_digest": "sha256:" + hashlib.sha256(binary.resolve().read_bytes()).hexdigest(),
        "study_id": study_id, "purpose": "R0_engineering_not_formal_pilot", "wholeSkillTranslation": False,
        **({"reacceptance": linked_counts(reacceptance_claim, 0)} if reacceptance_claim is not None else {})})
    ledger = BudgetLedger(registry)
    ledger.register_study(study_id, frozen["digest"])
    candidate = ledger.register_candidate(study_id, sha256_json(implementation))
    if ledger.snapshot(study_id)["arm_count"]:
        raise ValueError("engineering batch already claimed; no replay or output-directory reset")
    output.mkdir(parents=True, exist_ok=False)
    output.chmod(0o700)
    write_new(output / "freeze.json", frozen)
    pool = LocalProviderPool(provider_root)
    cases_by_id, refs = {c["case_id"]: c for c in cases}, {r["case_id"]: r for r in references}
    rows, observations = [], []
    for assignment in assignments:
        case, arm = cases_by_id[assignment["case_id"]], assignment["arm"]
        provider = broker = host = result = None
        error, terminal, drained, delivery = None, "incomplete", False, {"matched": False}
        broker_close, host_close = {"drained": False}, {"drained": False}
        collector_errors, collected_attempts, meter = [], [], None
        if reacceptance_claim is not None:
            try:
                verify_parent(registry, reacceptance_claim)
            except Exception:
                ledger.pause_study(study_id, "reacceptance_parent_drift")
                raise
        arm_id = ledger.start_arm(study_id, candidate, case["case_id"], 1, "A" if arm == "control" else "B", case["input_digest"])
        directory = output / f"{len(rows):02}-{arm}"
        directory.mkdir()

        def collector_failure(stage, exc):
            nonlocal error
            collector_errors.append({"stage": stage, "type": type(exc).__name__})
            error = {"type": type(exc).__name__, "terminal": "collector_measurement_invalid"}
            try:
                ledger.pause_study(study_id, "controller_collector_unknown")
            except Exception as pause_exc:
                # Still attempt arm finalization; failed persistence is not an
                # excuse to leave a claim active or to fabricate successful IO.
                collector_errors.append({"stage": "pause_study", "type": type(pause_exc).__name__})

        def collect_write(path, value):
            try:
                write_new(path, value)
                return True
            except Exception as exc:
                collector_failure(path.name, exc)
                return False

        try:
            # Arm time already runs: preparation, token preflight and close count.
            ledger.check_arm(arm_id)
            provider = pool.create_arm(arm_id, arm, case["provider_fixture"])
            if (provider.binding["initial_state_digest"] != case_initial_state_digest(case)
                    or provider.snapshot()["state_digest"] != case_initial_state_digest(case)
                    or provider.binding["fixture_digest"] != sha256_json(case["provider_fixture"])):
                raise ValueError("Provider initial state DB readback differs from frozen case")
            inputs = validate_agent_input(case["agent_input"])
            write_new(directory / "agent-input.json", inputs)
            broker = broker_factory(ledger, arm_id, mechanical_responder(dialogues[case["case_id"]]), directory / "transport").start()
            ledger.check_arm(arm_id)
            routes = {"schema": "netopyu.local-model-routes/v1", "model": MODEL, "model_digest": broker.backend.digest,
                "arm_id": arm_id, "routes": {"compile": {"base_url": broker.route("compiler")},
                                             "runtime": {"base_url": broker.route("runtime")}}}
            route_file = directory / "model-routes.json"
            write_new(route_file, routes)
            route_file.chmod(0o600)
            host = DynamicToolHost(provider, broker, directory, inputs=inputs, fixture=case["provider_fixture"],
                                   treatment=arm == "treatment", model_routes=route_file).start()
            result = _run_dsh(inputs, directory, broker, host, binary)
            if result.returncode:
                raise ValueError("DSH exited without completing the mechanical dialogue")
            drained = _idle_before_revoke(host, broker, ledger, arm_id)
            if not drained:
                raise TimeoutError("delivery bookkeeping did not drain")
            ledger.check_arm(arm_id)
            terminal = "completed"
        except Exception as exc:
            terminal = "timeout" if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)) else "measurement_invalid"
            error = {"type": type(exc).__name__, "terminal": terminal}
            collect_write(directory / "process-error.json", error)
            ledger.pause_study(study_id, "controller_failure_or_unknown")
        finally:
            if broker is not None:
                try:
                    broker_close = broker.close(timeout=2)
                except Exception as exc:
                    broker_close = {"drained": False, "error_type": type(exc).__name__}
                    ledger.pause_study(study_id, "controller_broker_close_unknown")
            if host is not None:
                try:
                    host_close = host.close(timeout=2)
                except Exception as exc:
                    host_close = {"drained": False, "error_type": type(exc).__name__}
                    ledger.pause_study(study_id, "controller_host_close_unknown")
            elif provider is not None:
                try:
                    provider.close()
                except Exception as exc:
                    collector_failure("provider_close", exc)
            # Collection only: never invoke a provider/runtime after DSH exit.
            if provider is not None:
                for name, read in (("provider-receipts.json", provider.receipts), ("provider-state.json", provider.snapshot)):
                    try:
                        collect_write(directory / name, read())
                    except Exception as exc:
                        collector_failure(name, exc)
            if host is not None:
                try:
                    collected_attempts = host.attempts()
                    collect_write(directory / "host-attempts.json", collected_attempts)
                    delivery = delivered_feedback(directory / "transport", host.completed)
                except Exception as exc:
                    collector_failure("host_attempts_or_delivery", exc)
            try:
                if reacceptance_claim is not None:
                    verify_parent(registry, reacceptance_claim)
                valid = bool(not collector_errors and terminal == "completed" and drained and broker_close.get("drained")
                    and host_close.get("drained") and delivery["matched"] and host and not host.errors
                    and broker and not broker.errors and len(host.completed) == len(dialogues[case["case_id"]])
                    and implementation_fingerprint() == implementation)
            except Exception as exc:
                collector_failure("implementation_fingerprint", exc)
                valid = False
            if not valid:
                ledger.pause_study(study_id, "controller_measurement_not_valid")
                terminal = terminal if terminal != "completed" else "measurement_invalid"
            try:
                finished = ledger.finish_arm(arm_id, terminal)
                valid = valid and finished["status"] == "completed"
                terminal = finished["status"]
            except Exception as exc:
                collector_failure("finish_arm", exc)
                valid, terminal = False, "measurement_invalid"
            try:
                meter = ledger.inspect_arm(arm_id)
                collect_write(directory / "meter.json", meter)
            except Exception as exc:
                collector_failure("inspect_arm", exc)
            valid = valid and not collector_errors
        artifact_path = directory / "dsh-stdout.txt"
        try:
            artifact = "sha256:" + hashlib.sha256(artifact_path.read_bytes()).hexdigest() if artifact_path.exists() else None
        except Exception as exc:
            artifact, valid = None, False
            collector_failure("artifact_read", exc)
        if provider is not None and meter is not None and not collector_errors:
            try:
                observation = _observation(refs[case["case_id"]], arm, provider, host, meter, artifact, valid, terminal, reviewer)
                if collect_write(directory / "observation.json", observation):
                    observations.append(observation)
                else:
                    valid = False
            except Exception as exc:
                # Keep the original ledger terminal unchanged; record collector
                # failure separately and leave this observation missing in the
                # full assigned denominator rather than synthesizing receipts.
                valid = False
                collector_failure("observation", exc)
                collect_write(directory / "collector-error.json", error)
        row = {"case_id": case["case_id"], "arm": arm, "arm_id": arm_id, "mechanics_passed": valid,
               "terminal": terminal, "error": error, "host_close": host_close, "broker_close": broker_close,
               "delivery": delivery, "provider_binding": provider.binding if provider else None,
               "routes": host.routes if host else [], "compiler_requests": host.compile_attempts if host else 0,
               "collector_errors": collector_errors,
               "known_author_input_rejections": [e["details"] for e in collected_attempts
                   if e["event"] == "known_compiler_input_budget_rejection"]}
        rows.append(row)
        if not collect_write(directory / "controller-arm.json", seal(row)):
            valid = False
            row["mechanics_passed"], row["error"] = False, error
        if not valid:
            break
    scorecard = assess(protocol, references, observations)
    report = seal({"schema": "netopyu.io/bounded-r0-controller/v1", "study_id": study_id,
        "freeze_digest": frozen["digest"], "assigned_arms": 24, "rows": rows,
        "unrun": assignments[len(rows):], "observations": observations, "scorecard": scorecard,
        "controllerMechanicsPassed": len(rows) == 24 and all(r["mechanics_passed"] for r in rows),
        "r0Complete": False, "pilotQualified": False, "researchEvidenceEligible": False,
        "runtimeLargeEvaluationAllowed": False, "actualModelCalls": 0, "automaticEffectBridge": "not_tested",
        "semanticReview": "not_supplied_all_unknown" if reviewer is None else "external_judgments_supplied_not_automatically_proven",
        "translationScope": "online_per_tool_read_prefix_not_whole_skill_translation_not_R1_default",
        "dialogueRole": "predeclared_mechanical_engineering_surrogate_not_Gold_or_natural_model_policy",
        "runtimeReasoner": "not_invoked_native_DSH_retains_L1", "sourceScriptsExecutable": False,
        "authorBudgetPolicy": "unchanged_full_source_pre_request_rejection_is_explicit_read_fallback_not_translation",
        "completeDependencySnapshot": False, "budget": ledger.snapshot(study_id),
        **({"reacceptance": linked_counts(reacceptance_claim, len(rows))} if reacceptance_claim is not None else {})})
    try:
        write_new(output / "report.json", report)
    except Exception as exc:
        ledger.pause_study(study_id, "controller_report_persistence_failed")
        report = seal({**{k: v for k, v in report.items() if k != "digest"},
                       "controllerMechanicsPassed": False, "reportPersistenceError": type(exc).__name__})
    return report
