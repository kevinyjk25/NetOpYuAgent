"""One source-anchored wiring fixture and adversarial variants; not Skill accuracy."""

import copy
import hashlib
import json
from dataclasses import replace

import pytest

from evaluation.structured_flow_demo import context, fixture, host_bindings, run_demo
from evaluation.structured_flow_tree import StructuredFlowTree, compile_structured_tree
from network_runtime.capabilities import CapabilityKind, EffectSemantics
from network_runtime.contracts import sha256_json
from network_runtime.l0 import flow as engine
from network_runtime.l0.read_execution import execute_host_read
from network_runtime.l0.structured_reads import StructuredReadManifest, compile_structured_read, verify_read_contract
from network_runtime.l0.structured_schema import DataBindingError


def setup():
    bundle, tree, reads, effects = fixture()
    compilation = compile_structured_tree(bundle, tree, reads, effects)
    proposal = engine.parse_flow(compilation["flow"])
    return bundle, tree, reads, effects, compilation, proposal


def execute(proposal, reads, effects, bindings, ctx=None, arguments=None, consent=None):
    arguments = {"device": {"id": "lab-sw1"}} if arguments is None else arguments
    reads = {c.contract_hash: c for c in reads.values()}
    packet = engine.qualify_flow(proposal, reads, effects)
    consent = consent or engine.HostFlowConsent(packet["flowDigest"], sha256_json(arguments))
    return engine.run_read_flow(proposal, arguments, reads=reads, effects=effects,
                               bindings=bindings, context=ctx or context(), consent=consent)


@pytest.mark.parametrize("up,errors,expected,count", [
    (True, 7, "read_path_completed", 1), (False, 0, "read_path_completed", 2),
    (False, 7, "awaiting_effect_admission", 2),
])
def test_nested_branch_paths_use_the_shared_executor_without_effects(up, errors, expected, count):
    _, _, reads, effects, compilation, proposal = setup()
    calls = []
    result = execute(proposal, reads, effects, host_bindings(reads, calls, interfaces=[{"name": "eth0", "adminUp": up}], errors=errors))
    assert result["status"] == expected and len(calls) == count
    assert not result["effectExecuted"] and not result["wholeSkillCorrectnessProven"]
    assert compilation["flow"]["authoring_digest"] == compilation["treeDigest"]
    if count == 2:
        assert calls[1]["arguments"] == {"device": {"id": "lab-sw1"}, "interface": {"name": "eth0"}}
    if expected == "awaiting_effect_admission":
        assert result["candidate"]["arguments"] == {"device": {"id": "lab-sw1"}, "interface": {"name": "eth0"}, "enabled": True}
        assert result["trace"][-1]["argumentBinding"]["runtimeAuthorityGranted"] is False


@pytest.mark.parametrize("patch", [
    {"authenticated": False}, {"implicit_local_context": True}, {"roles": frozenset({"system"})},
    {"scopes": frozenset({"*"})}, {"scopes": frozenset({"device_id:lab-sw1"})},
    {"scopes": frozenset({"network:read", "device_id:other"})}, {"roles": frozenset({"viewer"})},
])
def test_explicit_host_identity_and_nested_resource_scopes_before_calls(patch):
    _, _, reads, effects, _, proposal = setup()
    calls = []
    result = execute(proposal, reads, effects, host_bindings(reads, calls), replace(context(), **patch))
    assert result["status"] == "blocked" and calls == []
    assert not result["effectExecuted"]


def test_second_nested_resource_scope_is_checked_before_second_call():
    _, _, reads, effects, _, proposal = setup()
    calls = []
    result = execute(proposal, reads, effects, host_bindings(reads, calls, interfaces=[{"name": "eth9", "adminUp": False}]))
    assert result["status"] == "blocked" and len(calls) == 1
    assert result["blockedAt"] == "node-2"


@pytest.mark.parametrize("changed", ["input_schema_digest", "output_schema_digest", "capability_id", "scope_fields", "kind", "effect_semantics"])
def test_drifted_host_capability_never_invokes(changed):
    _, _, reads, effects, _, proposal = setup()
    calls, bindings = [], None
    bindings = host_bindings(reads, calls)
    key = reads["get_interfaces"].contract_hash
    values = {"input_schema_digest": "bad", "output_schema_digest": "bad", "capability_id": "other",
              "scope_fields": (), "kind": CapabilityKind.EFFECT, "effect_semantics": EffectSemantics.DESTRUCTIVE}
    bindings[key] = replace(bindings[key], capability=replace(bindings[key].capability, **{changed: values[changed]}))
    result = execute(proposal, reads, effects, bindings)
    assert result["status"] == "blocked" and calls == []


def test_exact_host_consent_binds_both_request_and_graph_before_any_read():
    _, _, reads, effects, compilation, proposal = setup()
    calls = []
    bindings = host_bindings(reads, calls)
    for consent in (engine.HostFlowConsent("bad", sha256_json({"device": {"id": "lab-sw1"}})),
                    engine.HostFlowConsent(compilation["qualification"]["flowDigest"], "bad")):
        with pytest.raises(PermissionError, match="consent"):
            execute(proposal, reads, effects, bindings, consent=consent)
    assert calls == []


@pytest.mark.parametrize("interfaces", [[], [{"name": "eth0", "adminUp": "false"}]])
def test_absent_or_invalid_nested_output_cannot_be_used(interfaces):
    _, _, reads, effects, _, proposal = setup()
    calls = []
    result = execute(proposal, reads, effects, host_bindings(reads, calls, interfaces=interfaces))
    assert result["status"] == "blocked" and len(calls) == 1
    assert result["trace"][-1]["diagnostic"]["code"] in {"missing_source_value", "value_constraint"}


@pytest.mark.parametrize("later", [6.0, -1.0])
def test_branch_control_evidence_revalidated_after_slow_read_even_when_not_an_argument(monkeypatch, later):
    _, _, reads, effects, _, proposal = setup()
    now, calls = [0.0], []
    monkeypatch.setattr(engine.time, "monotonic", lambda: now[0])
    bindings = host_bindings(reads, calls)
    key = reads["get_interface_counters"].contract_hash
    original = bindings[key].observe
    def delayed(args):
        result = original(args)
        now[0] = later
        return result
    bindings[key] = replace(bindings[key], observe=delayed)
    result = execute(proposal, reads, effects, bindings)
    assert result["status"] == "blocked" and len(calls) == 2
    assert result["trace"][-1]["diagnostic"]["code"] == "read_evidence_expired"
    assert "candidate" not in result and not result["effectExecuted"]


def test_capability_age_limit_is_stricter_than_flow_age_limit(monkeypatch):
    _, _, reads, effects, _, proposal = setup()
    now, calls = [0.0], []
    monkeypatch.setattr(engine.time, "monotonic", lambda: now[0])
    bindings = host_bindings(reads, calls)
    first = reads["get_interfaces"].contract_hash
    bindings[first] = replace(bindings[first], capability=replace(bindings[first].capability, freshness_limit_seconds=1))
    second = reads["get_interface_counters"].contract_hash
    original = bindings[second].observe
    def delayed(args):
        now[0] = 2.0
        return original(args)
    bindings[second] = replace(bindings[second], observe=delayed)
    result = execute(proposal, reads, effects, bindings)
    assert result["status"] == "blocked" and len(calls) == 2


def test_provider_mutation_cannot_change_inputs_or_previous_receipts():
    _, _, reads, effects, _, proposal = setup()
    arguments = {"device": {"id": "lab-sw1"}}
    rows, calls = [{"name": "eth0", "adminUp": False}], []
    bindings = host_bindings(reads, calls, interfaces=rows)
    second = reads["get_interface_counters"].contract_hash
    original = bindings[second].observe
    def mutating(args):
        result = original(copy.deepcopy(args))
        args["device"]["id"] = "mutated"
        rows[0]["name"] = "mutated"
        return result
    bindings[second] = replace(bindings[second], observe=mutating)
    result = execute(proposal, reads, effects, bindings, arguments=arguments)
    assert result["status"] == "awaiting_effect_admission"
    assert arguments == {"device": {"id": "lab-sw1"}}
    assert result["candidate"]["arguments"]["interface"]["name"] == "eth0"
    assert result["trace"][0]["receipt"]["payload"]["interfaces"][0]["name"] == "eth0"


@pytest.mark.parametrize("error", [RuntimeError("secret-provider-token"),
                                  DataBindingError("provider", "/secret-path", "secret-provider-token")])
def test_provider_error_is_redacted(error):
    _, _, reads, effects, _, proposal = setup()
    bindings = host_bindings(reads, [])
    key = reads["get_interfaces"].contract_hash
    def fail(_):
        raise error
    bindings[key] = replace(bindings[key], observe=fail)
    result = execute(proposal, reads, effects, bindings)
    assert result["status"] == "blocked" and "secret-provider-token" not in json.dumps(result)
    assert "secret-path" not in json.dumps(result)


@pytest.mark.parametrize("value,constant,matched", [
    (None, None, True), (0, None, False), (True, 1, False), (False, 0, False),
    (1, True, False), (1, 1.0, True),
])
def test_nullable_mixed_scalar_branches_follow_json_types(value, constant, matched):
    proposal = engine.parse_flow({
        "api_version": "netopyu.io/l0-flow-proposal/v2", "source_digest": "sha256:" + "a" * 64,
        "purpose": "Scalar type regression only", "entry": "check", "max_read_age_seconds": 5,
        "input_schema": {"type": "object", "properties": {"choice": {"type": ["null", "boolean", "number"]}},
                         "required": ["choice"], "additionalProperties": False},
        "nodes": [
            {"kind": "branch", "id": "check", "left": {"kind": "reference", "source": "input", "pointer": "/choice"},
             "equals": {"kind": "constant", "value": constant}, "on_true": "yes", "on_false": "no"},
            {"kind": "end", "id": "yes", "outcome": "read_path_completed", "explanation": "Matched"},
            {"kind": "end", "id": "no", "outcome": "needs_l1", "explanation": "Not matched"},
        ],
    })
    result = execute(proposal, {}, {}, {}, arguments={"choice": value})
    assert result["trace"][0]["matched"] is matched
    assert result["status"] == ("read_path_completed" if matched else "needs_l1")


def test_existing_demo_output_is_rejected_before_even_fixture_calls(tmp_path, monkeypatch):
    import evaluation.structured_flow_demo as demo

    def unexpected():
        raise AssertionError("no fixture or callback work before overwrite rejection")

    monkeypatch.setattr(demo, "fixture", unexpected)
    with pytest.raises(FileExistsError):
        demo.run_demo(tmp_path)


def test_no_branch_local_alias_can_escape_or_self_reference():
    bundle, tree, reads, effects, _, _ = setup()
    raw = tree.model_dump(mode="json")
    raw["steps"][0]["arguments"] = {"kind": "reference", "source": "interfaces", "pointer": ""}
    with pytest.raises(ValueError, match="lexical"):
        compile_structured_tree(bundle, StructuredFlowTree.model_validate(raw), reads, effects)
    raw = tree.model_dump(mode="json")
    # Both branches fall through, but only the true branch has counters.
    raw["steps"][1]["when_equal"] = [raw["steps"][1]["when_equal"][0]]
    raw["steps"][1]["otherwise"] = []
    raw["steps"].append({"kind": "effect_candidate", "source": raw["steps"][1]["source"], "binding_id": "enable",
                         "arguments": {"kind": "reference", "source": "counters", "pointer": ""}})
    with pytest.raises(ValueError, match="lexical"):
        compile_structured_tree(bundle, StructuredFlowTree.model_validate(raw), reads, effects)


@pytest.mark.parametrize("issue", ["cycle", "not_dominating", "unreachable"])
def test_direct_graph_cannot_bypass_tree_checks(issue):
    _, _, reads, effects, _, proposal = setup()
    raw = proposal.model_dump(mode="json")
    if issue == "cycle":
        raw["nodes"][0]["next"] = "node-0"
    elif issue == "not_dominating":
        raw["nodes"][0]["arguments"] = {"kind": "reference", "source": "node-2", "pointer": ""}
    else:
        raw["nodes"].append({"kind": "end", "id": "unused", "outcome": "unsupported", "explanation": "never silently drop"})
    with pytest.raises(ValueError):
        engine.qualify_flow(engine.parse_flow(raw), {r.contract_hash: r for r in reads.values()}, effects)


def test_source_spans_and_tree_digest_are_bound_but_not_semantic_proof():
    bundle, tree, reads, effects, compilation, _ = setup()
    assert not compilation["wholeSkillTranslationProven"]
    raw = tree.model_dump()
    raw["steps"][0]["source"]["start"] += 1
    with pytest.raises(ValueError, match="source span"):
        compile_structured_tree(bundle, StructuredFlowTree.model_validate(raw), reads, effects)
    raw = tree.model_dump()
    raw["purpose"] += " Revised explicit purpose."
    changed = compile_structured_tree(bundle, StructuredFlowTree.model_validate(raw), reads, effects)
    assert changed["qualification"]["flowDigest"] != compilation["qualification"]["flowDigest"]


def test_declared_contract_drift_or_authority_mutation_is_rejected():
    _, _, reads, _, _, _ = setup()
    contract = copy.deepcopy(reads["get_interfaces"])
    contract.spec.input_schema["properties"]["device"]["properties"]["id"]["minLength"] = 2
    with pytest.raises(ValueError, match="differs"):
        verify_read_contract(contract)
    raw = reads["get_interfaces"].model_dump(by_alias=True)
    raw["spec"]["resourceScopes"]["device_id"] = "/missing"
    with pytest.raises(ValueError):
        StructuredReadManifest.model_validate({"apiVersion": "netopyu.io/l0-structured-read/v1", "kind": "StructuredRead",
                                              "metadata": raw["metadata"], "spec": raw["spec"]})
    raw = reads["get_interfaces"].model_dump(by_alias=True)
    # Re-sealing an altered tool source cannot make a false readOnlyHint safe.
    source = next(s for s in raw["spec"]["sources"] if s["role"] == "tool")
    tool = json.loads(source["text"])
    tool["annotations"]["readOnlyHint"] = False
    source["text"] = json.dumps(tool)
    source["sha256"] = "sha256:" + hashlib.sha256(source["text"].encode()).hexdigest()
    with pytest.raises(ValueError, match="read-only"):
        compile_structured_read(StructuredReadManifest.model_validate({"apiVersion": "netopyu.io/l0-structured-read/v1", "kind": "StructuredRead",
                                                                      "metadata": raw["metadata"], "spec": raw["spec"]}))


def test_structured_contract_uses_shared_gateway_directly():
    _, _, reads, _, _, _ = setup()
    calls = []
    contract = reads["get_interfaces"]
    binding = host_bindings(reads, calls)[contract.contract_hash]
    receipt = execute_host_read(contract, {"device": {"id": "lab-sw1"}}, context(), binding)
    assert receipt["status"] == "local_read_completed_shape_valid" and len(calls) == 1
    assert not receipt["contractActivated"]


def test_local_demo_seals_compile_calls_and_candidate_only_result(tmp_path):
    report = run_demo(tmp_path / "new")
    assert report["readCalls"] == 2 and report["effectCalls"] == report["modelCalls"] == 0
    assert report["status"] == "awaiting_effect_admission" and report["translationMetrics"] is None
    for name, digest in report["artifactDigests"].items():
        assert sha256_json(json.loads((tmp_path / "new" / name).read_text())) == digest
