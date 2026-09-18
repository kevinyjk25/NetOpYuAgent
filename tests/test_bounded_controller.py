"""Explicit local R0 fixtures: no real model or full controller batch."""
import copy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from evaluation import bounded_controller as module
from evaluation.bounded_budget import BudgetLedger
from evaluation.bounded_pilot import FAMILIES, PRIMITIVES, case_input_digest, make_protocol
from evaluation.bounded_provider import ArmProvider, FIXTURE_SCHEMA, OUTPUT_SCHEMA, LocalProviderPool
from evaluation.bounded_scoring import seal_reference
from evaluation.bounded_transport import ModelBroker, ScriptedModel
from network_runtime.contracts import sha256_json


def test_literal_source_patch_preserves_legacy_overlay_and_requires_exact_binding():
    original = (module.ROOT / "evaluation/dsh_shadow.patch.yml").read_text()
    patched = module._literal_source_patch(original)
    assert 'persona: "{{netopyu_inert_source}}"' in patched
    assert "persona: !!js process.env.NETOPYU_L1_SHADOW_SYSTEM_PROMPT" in original
    assert patched.replace('    persona: "{{netopyu_inert_source}}"',
        "    persona: !!js process.env.NETOPYU_L1_SHADOW_SYSTEM_PROMPT") == original
    for invalid in ("", original + original):
        with pytest.raises(ValueError, match="exactly one"):
            module._literal_source_patch(invalid)


def native_literal_probe(fixtures):
    """Installed pure renderPrompt + actual plugin; mocked fetch, no DSH turn."""
    node = shutil.which("node", path=module._node_path())
    binary = module._default_dsh_binary()
    modules = next((p for p in binary.parents if p.name == "node_modules"), None)
    package = modules / "@deepseek-ai/dsh/package.json" if modules else None
    if not node or not package or not package.is_file():
        pytest.skip("installed DSH and Node needed for native literal codec check")
    script = r'''
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import { readFileSync, realpathSync } from 'node:fs';
import { createHash } from 'node:crypto';
const input = JSON.parse(readFileSync(0, 'utf8'));
const require = createRequire(realpathSync(input.package));
const rendererPath = require.resolve('@deepseek-ai/dsh-system-prompt');
const { renderPrompt } = await import(pathToFileURL(rendererPath).href);
const plugin = await import(pathToFileURL(input.plugin).href);
const hash = text => createHash('sha256').update(text).digest('hex');
let fetches = 0;
globalThis.fetch = async url => {
  if (url !== 'http://127.0.0.1:1/literal-fixture/catalog') throw new Error('unexpected fixture fetch');
  fetches++;
  return {ok: true, json: async () => []};
};
process.env.NETOPYU_BOUNDED_TOOL_ENDPOINT = 'http://127.0.0.1:1/literal-fixture';
const render = (text, variables) => renderPrompt({sections:[{name:'deployment:persona',text}],
  contexts:[],tools:[],variables});
const rows = [];
for (const [name, source] of input.fixtures) {
  const variables = {model:'MUST_NOT_EXPAND',cwd:'MUST_NOT_EXPAND',provider:'MUST_NOT_EXPAND'};
  process.env.NETOPYU_BOUNDED_LITERAL_SOURCE = '1';
  process.env.NETOPYU_L1_SHADOW_SYSTEM_PROMPT = source;
  await plugin.apply({tools:{register(){throw new Error('empty fixture catalog');}},
    systemPrompt:{variable(name, provider){variables[name] = provider();}}});
  const actual = render('{{netopyu_inert_source}}', variables);
  if (!Buffer.from(actual).equals(Buffer.from(source))) throw new Error('literal bytes changed: '+name);
  rows.push({name,bytes:Buffer.byteLength(source),byte_identical:true,sha256:hash(source)});
}
delete process.env.NETOPYU_BOUNDED_LITERAL_SOURCE;
process.env.NETOPYU_L1_SHADOW_SYSTEM_PROMPT = 'Legacy persona unchanged.';
let registrations = 0;
await plugin.apply({tools:{register(){}},systemPrompt:{variable(){registrations++;}}});
const legacyUnchanged = registrations === 0 && render('Legacy persona unchanged.',{}) === 'Legacy persona unchanged.';
if (!legacyUnchanged) throw new Error('legacy probe behavior changed');
let directRejected = false, backslashRejected = false;
try {render('{{destination_service_name}}',{});} catch {directRejected = true;}
try {render('\\{{destination_service_name}}',{});} catch {backslashRejected = true;}
let missingSourceRejected = false;
process.env.NETOPYU_BOUNDED_LITERAL_SOURCE = '1';
delete process.env.NETOPYU_L1_SHADOW_SYSTEM_PROMPT;
try {await plugin.apply({systemPrompt:{variable(){}}});} catch {missingSourceRejected = true;}
console.log(JSON.stringify({evidence_role:'installed_DSH_pure_renderPrompt_and_plugin_not_controller_or_model',
  renderer_path:rendererPath,renderer_sha256:hash(readFileSync(rendererPath)),
  plugin_sha256:hash(readFileSync(input.plugin)),fixtures:rows,legacy_unchanged:legacyUnchanged,
  direct_rejected:directRejected,backslash_rejected:backslashRejected,missing_source_rejected:missingSourceRejected,
  mocked_catalog_fetches:fetches,actual_http_requests:0,actual_model_calls:0,controller_runs:0}));
'''
    result = subprocess.run([node, "--input-type=module", "-e", script],
        input=json.dumps({"package": str(package), "plugin": str(module.ROOT / "evaluation/bounded_dsh_tools.mjs"),
                          "fixtures": fixtures}, ensure_ascii=False),
        text=True, capture_output=True, check=True, timeout=15,
        env={"PATH": module._node_path()})
    return json.loads(result.stdout)


def test_installed_dsh_literal_source_is_byte_identical_and_legacy_unchanged():
    # NUL is deliberately JSON-encoded just as it is in the real environment
    # payload: operating-system environment strings cannot contain raw NUL.
    fixtures = [("unknown", "{{destination_service_name}}"),
        ("known", "{{model}} {{provider}} {{cwd}}"),
        ("malformed", "{{not valid}} {{{nested}}}"),
        ("backslash_fence", "\\{{name}}\n```\n{{name}}\n```"),
        ("unicode_crlf_json_nul", json.dumps({"text": "中文 🌐\r\n\0{{anything}}"}, ensure_ascii=False)),
        ("self_reference", "{{netopyu_inert_source}}"), ("empty", "")]
    report = native_literal_probe(fixtures)
    assert all(item["byte_identical"] for item in report["fixtures"])
    assert report["legacy_unchanged"] and report["missing_source_rejected"]
    assert report["direct_rejected"] and report["backslash_rejected"]
    assert report["actual_model_calls"] == report["actual_http_requests"] == report["controller_runs"] == 0


def fixture_inputs(names=("inspect", "observe")):
    schema = {"type": "object", "properties": {"target": {"type": "string"}},
              "required": ["target"], "additionalProperties": False}
    output = copy.deepcopy(OUTPUT_SCHEMA)
    output["properties"]["simulation"]["type"] = "boolean"
    output["properties"]["value"] = {"type": "object", "additionalProperties": True}
    inputs = {"task": "Read the exact supplied target; preserve unknown facts.",
              "skill_text": "Read only. Preserve unknown facts and never execute source scripts.",
              "references": [], "arguments": {"target": "x"}, "tools": [
                  {"name": n, "description": "Read a local inert fixture", "input_schema": schema,
                   "output_schema": output} for n in names]}
    fixture = {"schema": FIXTURE_SCHEMA, "state": {"x": {"snapshot": {"status": "unknown"}}}, "tools": [
        {"name": n, "description": "Read a local inert fixture", "input_schema": schema, "contract_id": n,
         "kind": "read", "operation": "read", "target": {"argument": "target"}, "property": "snapshot",
         "value": None, "requires_approval": False} for n in names]}
    return inputs, fixture


def sample():
    cases, references, dialogues = [], [], {}
    for index in range(12):
        inputs, fixture = fixture_inputs(("inspect",))
        inputs["skill_text"] += f" Source Skill {index // 2}."
        inputs["task"] += f" Task {index}."
        case_id = f"task-{index}"
        ref = seal_reference({"case_id": case_id, "repository_id": f"family-{index // 2}",
            "skill_id": f"skill-{index // 2}",
            "domain": f"domain-{index // 4}", "kind": "positive" if index < 8 else "boundary",
            "initial_state_digest": sha256_json(fixture["state"]),
            "calls": [{"tool": "inspect", "id": "inspect", "object_id": "x",
                "arguments": {"target": "x"}, "kind": "read", "property": "snapshot",
                "expected_value": {"status": "unknown"}, "min_calls": 1, "max_calls": 1,
                "approval_required": False, "verify_with": None}],
            "allowed_outcomes": ["completed"] if index < 8 else ["rejected"],
            "criteria": [{"id": "facts", "critical": True, "statement": "Preserve unknown facts.",
                          "source_quote": "preserve unknown facts"}],
            "duties": [{"id": "read", "critical": True, "strict_eligible": True,
                        "statement": "Read the exact target.", "source_quote": "Read the exact supplied target"}]})
        case = {"case_id": case_id, "skill_id": f"skill-{index // 2}", "repository_id": f"repo-{index // 2}",
            "repository_family": ref["repository_id"], "domain": ref["domain"], "kind": ref["kind"],
            "families": [sorted(FAMILIES)[index // 2]], "source_revision": "fixed-test-fixture",
            "source_kind": "synthetic_development", "agent_input": inputs, "provider_fixture": fixture,
            "reference_digest": ref["reference_digest"]}
        case["input_digest"] = case_input_digest(case)
        cases.append(case)
        references.append(ref)
        dialogues[case_id] = [{"tool": "inspect", "arguments": {"target": "x"}}]
    protocol = make_protocol("unit-protocol", cases, model_digest=sha256_json("model"),
        harness_digest=sha256_json("harness"), support={"primitives": sorted(PRIMITIVES),
        "unsupported": ["automatic Effect bridge"], "source_scripts": "inert",
        "effect_gateway": "existing_active_contracts_local_simulator"})
    return protocol, references, dialogues


def host_fixture(tmp_path, *, treatment=True, names=("inspect", "observe"), operation="read"):
    inputs, fixture = fixture_inputs(names)
    if operation != "read":
        for tool in fixture["tools"]:
            tool.update(operation=operation, kind="effect", value={"constant": {"changed": True}} if operation == "set" else None)
    ledger = BudgetLedger(tmp_path / "ledger.sqlite")
    ledger.register_study("test", "protocol")
    candidate = ledger.register_candidate("test", "candidate")
    arm = ledger.start_arm("test", candidate, "case", 1, "B" if treatment else "A", "context")
    provider = LocalProviderPool(tmp_path / "providers").create_arm(arm, "treatment" if treatment else "control", fixture)
    broker = ModelBroker(ledger, arm, ScriptedModel(module.mechanical_responder([])), tmp_path / "transport")
    directory = tmp_path / "host"
    directory.mkdir()
    routes = directory / "routes.json"
    routes.write_text(json.dumps({"schema": "netopyu.local-model-routes/v1", "model": module.MODEL,
        "model_digest": broker.backend.digest, "arm_id": arm, "routes": {
        "compile": {"base_url": "http://127.0.0.1:1/r/compiler"},
        "runtime": {"base_url": "http://127.0.0.1:1/r/runtime"}}}))
    routes.chmod(0o600)
    host = module.DynamicToolHost(provider, broker, directory, inputs=inputs, fixture=fixture,
                                  treatment=treatment, model_routes=routes)
    return host, broker, provider


def fake_compile(packet, visible, folder):
    wire = module.isolated_compiler.make_request(packet, visible)
    return json.loads(module.mechanical_responder([])("compiler", wire)["content"])


def body(name="inspect", request_id="one"):
    return {"tool": name, "arguments": {"target": "x"}, "request_id": request_id}


def test_mechanical_dialogue_ignores_values_and_has_no_semantic_success():
    respond = module.mechanical_responder([{"tool": "inspect", "arguments": {"target": "x"}}])
    first = respond("agent", {"messages": []})
    assert first["tool_calls"][0]["function"]["arguments"] == {"target": "x"}
    a = respond("agent", {"messages": [{"role": "tool", "content": "private answer A"}]})
    b = respond("fallback", {"messages": [{"role": "tool", "content": "failure and private answer B"}]})
    assert a == b == {"content": module.ENDING}
    with pytest.raises(ValueError, match="Runtime reasoner"):
        respond("runtime", {"messages": []})


def test_multiple_dynamic_tools_original_runtime_and_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(module.isolated_compiler, "invoke", fake_compile)
    host, broker, provider = host_fixture(tmp_path)
    try:
        first = host.call(body())
        second = host.call(body("observe", "two"))
        third = host.call(body("inspect", "three"))
        assert first == second == third
        assert host.compile_attempts == 2 and len(host.proposals) == 2
        assert host.routes == ["runtime"] * 3
        assert [r["origin"] for r in provider.scorer_calls()] == ["runtime"] * 3
        assert [r["tool"] for r in provider.scorer_calls()] == ["inspect", "observe", "inspect"]
        assert len(list(host.directory.glob("graph-*.json"))) == 3
    finally:
        broker.close(timeout=0)
        host.close()


def test_third_tool_falls_back_without_extra_compile(monkeypatch, tmp_path):
    monkeypatch.setattr(module.isolated_compiler, "invoke", fake_compile)
    host, broker, provider = host_fixture(tmp_path, names=("inspect", "observe", "third"))
    try:
        for index, tool in enumerate(("inspect", "observe", "third")):
            host.call(body(tool, str(index)))
        assert host.compile_attempts == 2
        assert host.routes == ["runtime", "runtime", "fallback"]
        assert provider.scorer_calls()[-1]["origin"] == "agent"
        assert broker.agent_stage == "fallback"
    finally:
        broker.close(timeout=0)
        host.close()


def test_control_never_compiles_and_replay_is_not_executed(monkeypatch, tmp_path):
    monkeypatch.setattr(module.isolated_compiler, "invoke", lambda *a: pytest.fail("Control compiler"))
    host, broker, provider = host_fixture(tmp_path, treatment=False)
    try:
        host.call(body())
        with pytest.raises(PermissionError, match="replay"):
            host.call(body())
        assert len(provider.scorer_calls()) == 1
        assert host.routes == ["native"] and host.compile_attempts == 0
        assert broker.ledger.inspect_arm(broker.arm_id)["study_status"] == "halted"
    finally:
        broker.close(timeout=0)
        host.close()


def test_unknown_compile_halts_without_provider_fallback(monkeypatch, tmp_path):
    def unknown(*args):
        raise OSError("unknown completion")
    monkeypatch.setattr(module.isolated_compiler, "invoke", unknown)
    host, broker, provider = host_fixture(tmp_path)
    try:
        with pytest.raises(OSError):
            host.call(body())
        assert not provider.scorer_calls() and not host.completed
        assert broker.ledger.inspect_arm(broker.arm_id)["study_status"] == "halted"
    finally:
        broker.close(timeout=0)
        host.close()


def test_known_author_input_budget_rejection_falls_back_without_proposal(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "author_budget", lambda wire: {"accepted": False, "actualInputTokens": None})
    monkeypatch.setattr(module.isolated_compiler, "invoke", lambda *args: pytest.fail("rejected wire must not dispatch"))
    host, broker, provider = host_fixture(tmp_path)
    try:
        assert host.call(body())["ok"]
        assert host.routes == ["fallback"] and host.compile_attempts == 0 and not host.proposals
        assert provider.scorer_calls()[0]["origin"] == "agent"
        assert broker.ledger.inspect_arm(broker.arm_id)["study_status"] == "active"
        assert any(e["event"] == "known_compiler_input_budget_rejection" for e in host.attempts())
    finally:
        broker.close(timeout=0)
        host.close()


@pytest.mark.parametrize("treatment", [True, False])
@pytest.mark.parametrize("operation", ["set", "delete"])
def test_treatment_write_is_explicit_stop_not_native_effect(tmp_path, treatment, operation):
    host, broker, provider = host_fixture(tmp_path, treatment=treatment, operation=operation)
    initial = provider.snapshot()["state"]
    try:
        result = host.call(body())
        assert result["code"] == "automatic_effect_bridge_not_tested"
        assert host.route == "safe_stop" and not provider.scorer_calls()
        assert provider.snapshot()["state"] == initial
    finally:
        broker.close(timeout=0)
        host.close()


def test_exact_arguments_and_late_delivery_are_guarded(monkeypatch, tmp_path):
    host, broker, provider = host_fixture(tmp_path, treatment=False)
    original = provider.invoke
    def closes_after_observation(*args, **kwargs):
        result = original(*args, **kwargs)
        broker.closed = True
        return result
    monkeypatch.setattr(provider, "invoke", closes_after_observation)
    try:
        with pytest.raises(PermissionError, match="terminated"):
            host.call(body())
        assert len(provider.scorer_calls()) == 1 and not host.completed
    finally:
        broker.closed = False
        broker.close(timeout=0)
        host.close()


def test_dialogues_reject_missing_and_extra_or_invalid_arguments():
    protocol, _, dialogues = sample()
    module.validate_dialogues(protocol["development_cases"], dialogues)
    missing = copy.deepcopy(dialogues)
    missing.pop("task-0")
    with pytest.raises(ValueError, match="every assigned"):
        module.validate_dialogues(protocol["development_cases"], missing)
    dialogues["task-0"][0]["answer"] = "do not expose"
    with pytest.raises(ValueError, match="exact public"):
        module.validate_dialogues(protocol["development_cases"], dialogues)


def test_dialogues_reject_write_plan_before_any_execution():
    protocol, _, dialogues = sample()
    tool = protocol["development_cases"][0]["provider_fixture"]["tools"][0]
    tool.update(kind="effect", operation="set", value={"constant": {"changed": True}})
    with pytest.raises(ValueError, match="read/verify only"):
        module.validate_dialogues(protocol["development_cases"], dialogues)


def unit_driver(monkeypatch, tmp_path, *, fail_first=False, undrained=False):
    monkeypatch.setattr(module, "implementation_fingerprint", lambda: {"unit-fixture": "sealed"})
    binary = tmp_path / "dsh-unit-not-executable"
    binary.write_text("not a DSH execution")
    monkeypatch.setattr(module.DynamicToolHost, "start", lambda self: self)
    brokers = []

    def factory(ledger, arm_id, responder, transport_dir):
        broker = ModelBroker(ledger, arm_id, ScriptedModel(responder), transport_dir)
        broker.start = lambda: broker
        broker.route = lambda role: "http://127.0.0.1:1/r/" + role
        brokers.append(broker)
        if undrained:
            broker.close = lambda **kwargs: {"drained": False}
        return broker

    def invoke(packet, visible, folder):
        broker = brokers[-1]
        response = broker.dispatch(broker.tokens["compiler"], "native", module.isolated_compiler.make_request(packet, visible))
        return json.loads(response["message"]["content"])
    monkeypatch.setattr(module.isolated_compiler, "invoke", invoke)

    def drive(inputs, directory, broker, host, binary):
        if fail_first:
            raise TimeoutError("fixed unit timeout")
        messages = [{"role": "user", "content": inputs["task"]}]
        while True:
            payload = {"model": module.MODEL, "messages": messages, "stream": False,
                       "max_tokens": 100, "parallel_tool_calls": False}
            envelope = broker.dispatch(broker.tokens["agent"], "openai", payload)
            message = envelope["message"]
            if not message.get("tool_calls"):
                break
            calls = copy.deepcopy(message["tool_calls"])
            # DSH's native/OpenAI adapter supplies a stable tool-call id.
            for call in calls:
                call.setdefault("id", f"mechanical-{len(host.completed)}")
                call["function"]["arguments"] = json.dumps(call["function"]["arguments"])
            messages.append({"role": "assistant", "content": "", "tool_calls": calls})
            for call in calls:
                value = host.call({"tool": call["function"]["name"],
                    "arguments": json.loads(call["function"]["arguments"]), "request_id": call["id"]})
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(value)})
        (directory / "dsh-stdout.txt").write_text(message["content"])
        return subprocess.CompletedProcess([], 0, message["content"], "")
    monkeypatch.setattr(module, "_run_dsh", drive)
    return binary, factory


def test_fixed_24_arm_controller_keeps_semantics_unknown_and_no_replay(monkeypatch, tmp_path):
    protocol, references, dialogues = sample()
    binary, factory = unit_driver(monkeypatch, tmp_path)
    kwargs = {"dialogues": dialogues, "registry": tmp_path / "fixed/ledger.sqlite",
              "provider_root": tmp_path / "fixed/providers", "dsh_binary": binary, "broker_factory": factory}
    report = module.run(protocol, references, tmp_path / "run", **kwargs)
    assert report["controllerMechanicsPassed"] is True, report["rows"]
    assert len(report["observations"]) == 24 and report["unrun"] == []
    assert report["semanticReview"] == "not_supplied_all_unknown"
    assert not report["r0Complete"] and not report["pilotQualified"] and report["actualModelCalls"] == 0
    assert all(set(o["review"]["criteria"].values()) == {"unknown"} for o in report["observations"])
    by_case = {}
    for row in report["rows"]:
        by_case.setdefault(row["case_id"], []).append(row["provider_binding"])
    assert all(a["initial_state_digest"] == b["initial_state_digest"] and a["isolation_id"] != b["isolation_id"]
               for a, b in by_case.values())
    with pytest.raises(ValueError, match="no replay"):
        module.run(protocol, references, tmp_path / "another-output", **kwargs)
    assert not (tmp_path / "another-output").exists()


@pytest.mark.parametrize("failure", ["timeout", "undrained"])
def test_unknown_stops_with_unrun_assignments_in_denominator(monkeypatch, tmp_path, failure):
    protocol, references, dialogues = sample()
    binary, factory = unit_driver(monkeypatch, tmp_path, fail_first=failure == "timeout", undrained=failure == "undrained")
    report = module.run(protocol, references, tmp_path / "run", dialogues=dialogues,
        registry=tmp_path / "fixed/ledger.sqlite", provider_root=tmp_path / "fixed/providers",
        dsh_binary=binary, broker_factory=factory)
    assert not report["controllerMechanicsPassed"] and len(report["rows"]) == 1
    assert len(report["unrun"]) == 23 and report["assigned_arms"] == 24
    assert report["budget"]["study"]["status"] == "halted"
    assert report["observations"][0]["review"]["measurement_valid"] is False


def test_registry_cannot_move_inside_output(tmp_path):
    protocol, references, dialogues = sample()
    with pytest.raises(ValueError, match="fixed outside"):
        module.run(protocol, references, tmp_path / "run", dialogues=dialogues,
                   registry=tmp_path / "run/ledger.sqlite", provider_root=tmp_path / "fixed/providers")


def test_delivery_is_bound_to_actual_results_not_success_words(tmp_path):
    directory = tmp_path / "transport/request"
    directory.mkdir(parents=True)
    messages = [{"role": "assistant", "tool_calls": [{"id": "one", "function": {"name": "inspect"}}]},
                {"role": "tool", "tool_call_id": "one", "content": '{"ok":true}'}]
    (directory / "request.json").write_text(json.dumps({"stage": "agent", "source_request": {"messages": messages}}))
    rows = [{"request_id": "one", "tool": "inspect", "result_digest": sha256_json({"ok": False})}]
    assert module.delivered_feedback(directory.parent, rows)["matched"] is False


def test_inert_source_is_retained_without_script_execution():
    inputs, _ = fixture_inputs(("inspect",))
    inputs["references"] = [{"path": "scripts/never.py", "text": "raise RuntimeError('inert')"}]
    packet = module.read_packet(inputs, inputs["tools"][0])
    assert packet["bundle"]["documents"][1]["content"] == inputs["references"][0]["text"]
    assert not Path("never.py").exists()


def test_actual_loopback_compiler_and_original_runtime_share_arm(tmp_path):
    """Actual HTTP compiler, host scripted response; never a tokenizer or LLM."""
    host, broker, provider = host_fixture(tmp_path, names=("inspect",))
    broker.start()
    routes = json.loads(host.model_routes.read_text())
    routes["routes"] = {"compile": {"base_url": broker.route("compiler")},
                        "runtime": {"base_url": broker.route("runtime")}}
    host.model_routes.write_text(json.dumps(routes))
    try:
        result = host.call(body())
        assert result["value"] == {"status": "unknown"}
        assert host.route == "runtime" and host.compile_attempts == 1
        calls = broker.ledger.inspect_arm(broker.arm_id)["calls"]
        assert len(calls) == 1 and calls[0]["stage"] == "compiler"
        assert calls[0]["arm_id"] == broker.arm_id and calls[0]["actual_input"] == 64
        assert provider.scorer_calls()[0]["origin"] == "runtime"
        assert module._idle_before_revoke(host, broker, broker.ledger, broker.arm_id)
    finally:
        assert broker.close(timeout=2)["drained"]
        assert host.close(timeout=2)["drained"]


def test_collector_failure_stops_without_fabricating_receipts(monkeypatch, tmp_path):
    protocol, references, dialogues = sample()
    binary, factory = unit_driver(monkeypatch, tmp_path)
    monkeypatch.setattr(module, "_observation", lambda *args: (_ for _ in ()).throw(ValueError("unknown receipt")))
    report = module.run(protocol, references, tmp_path / "run", dialogues=dialogues,
        registry=tmp_path / "fixed/ledger.sqlite", provider_root=tmp_path / "fixed/providers",
        dsh_binary=binary, broker_factory=factory)
    assert len(report["rows"]) == 1 and len(report["unrun"]) == 23
    assert not report["observations"] and not report["controllerMechanicsPassed"]
    assert report["rows"][0]["error"]["terminal"] == "collector_measurement_invalid"


@pytest.mark.parametrize("failure", ["receipts", "snapshot", "attempts", "write"])
def test_finally_collection_failure_still_finishes_arm_and_keeps_denominator(monkeypatch, tmp_path, failure):
    protocol, references, dialogues = sample()
    binary, factory = unit_driver(monkeypatch, tmp_path)

    def failed(*args, **kwargs):
        raise OSError("unit collector failure")

    if failure == "receipts":
        monkeypatch.setattr(ArmProvider, "receipts", failed)
    elif failure == "snapshot":
        original = ArmProvider.snapshot
        def late_snapshot(provider):
            if provider._closed:
                raise OSError("unit closed-state collector failure")
            return original(provider)
        monkeypatch.setattr(ArmProvider, "snapshot", late_snapshot)
    elif failure == "attempts":
        monkeypatch.setattr(module.DynamicToolHost, "attempts", failed)
    else:
        original = module.write_new
        def late_write(path, value):
            if path.name == "provider-receipts.json":
                raise OSError("unit receipt persistence failure")
            return original(path, value)
        monkeypatch.setattr(module, "write_new", late_write)

    report = module.run(protocol, references, tmp_path / "run", dialogues=dialogues,
        registry=tmp_path / "fixed/ledger.sqlite", provider_root=tmp_path / "fixed/providers",
        dsh_binary=binary, broker_factory=factory)
    assert len(report["rows"]) == 1 and len(report["unrun"]) == 23 and report["assigned_arms"] == 24
    assert not report["controllerMechanicsPassed"] and not report["observations"]
    assert report["rows"][0]["collector_errors"]
    assert report["rows"][0]["error"]["terminal"] == "collector_measurement_invalid"
    assert report["budget"]["arms"][0]["finished_at"] is not None
    assert report["budget"]["arms"][0]["status"] != "active"
    assert report["budget"]["study"]["status"] == "halted"
    assert (tmp_path / "run/report.json").exists()
