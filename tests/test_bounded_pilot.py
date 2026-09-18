"""R0 input/cost/lifecycle probes, not Skill or real-model results."""
import copy
import json

import pytest

from evaluation.bounded_budget import BudgetError, BudgetLedger
from evaluation.bounded_execution import MeasuredCalls, agent_context
from evaluation.bounded_pilot import (
    FAMILIES, PRIMITIVES, case_initial_state_digest, case_input_digest, confirmation_plan, make_protocol, prepare, read_json,
    schedule, seal, validate_agent_input, validate_cases, validate_protocol, validate_references,
)
from network_runtime.contracts import sha256_json


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def case_set(prefix="dev", public=False):
    cases = []
    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    for i in range(12):
        inputs = {"task": f"Read the scoped value for task {i}.",
                  "skill_text": f"{prefix} source Skill {i // 2}. Read only; preserve unknown facts.",
                  "references": [{"path": "scripts/never-run.sh", "text": "exit 99"}],
                  "tools": [{"name": "observe", "description": "Read authorized local fixture",
                             "input_schema": schema, "output_schema": schema}],
                  "arguments": {}}
        cases.append({"case_id": f"{prefix}-{i}", "skill_id": f"{prefix}-skill-{i // 2}",
                      "repository_id": f"{prefix}-repo-{i // 2}",
                      "repository_family": f"{prefix}-upstream-{i // 2}",
                      "domain": f"domain-{i // 4}", "kind": "positive" if i < 8 else "boundary",
                      "families": [sorted(FAMILIES)[i // 2]], "source_revision": "pinned-synthetic-revision",
                      "source_kind": "pinned_public" if public else "synthetic_development",
                      "agent_input": inputs, "provider_fixture": {"state": {"value": i}},
                      "input_digest": "pending",
                      "reference_digest": sha256_json({"reference": f"{prefix}-{i}"})})
        cases[-1]["input_digest"] = case_input_digest(cases[-1])
    return cases


def protocol():
    return make_protocol("measurement-probe", case_set(), model_digest=sha256_json("model-fixture"),
                         harness_digest=sha256_json("harness-fixture"),
                         support={"primitives": sorted(PRIMITIVES), "unsupported": ["arbitrary program semantics"],
                                  "source_scripts": "inert", "effect_gateway": "existing_active_contracts_local_simulator"})


def test_protocol_closed_fields_thresholds_source_and_model_are_frozen():
    frozen = protocol()
    assert validate_protocol(frozen) == frozen
    for name, value in (("limits", {}), ("thresholds", {}), ("authority", "runtimeLargeEvaluationAllowed")):
        changed = {k: v for k, v in frozen.items() if k != "digest"}
        changed[name] = value
        with pytest.raises(ValueError):
            validate_protocol(seal(changed))
    changed = copy.deepcopy(frozen)
    changed["model"]["name"] = "other-model"
    with pytest.raises(ValueError, match="drift"):
        validate_protocol(changed)


@pytest.mark.parametrize("mutation", ["count", "mixture", "duplicate", "copy", "families", "gold", "drift"])
def test_case_manifest_rejects_scope_inflation_and_extra_model_context(mutation):
    cases = case_set()
    if mutation == "count":
        cases.pop()
    elif mutation == "mixture":
        cases[0]["kind"] = "boundary"
    elif mutation == "duplicate":
        cases[1]["case_id"] = cases[0]["case_id"]
    elif mutation == "copy":
        for c in cases[2:4]:
            c["agent_input"]["skill_text"] = cases[0]["agent_input"]["skill_text"]
            c["input_digest"] = case_input_digest(c)
    elif mutation == "families":
        cases[0]["families"] = ["unknown"]
    elif mutation == "gold":
        cases[0]["agent_input"]["expected_parameters"] = {"answer": "hidden"}
        cases[0]["input_digest"] = case_input_digest(cases[0])
    else:
        cases[0]["agent_input"]["task"] += " altered"
    with pytest.raises(ValueError):
        validate_cases(cases)


def test_counterbalance_fixed_seed_pair_identity_and_unique_assignments():
    cases = case_set()
    first = schedule(cases, phase="development")
    assert first == schedule(list(reversed(cases)), phase="development")
    assert len(first) == 12
    assert sum(r["arms"] == ["control", "treatment"] for r in first) == 6
    later = schedule(case_set("new", public=True), phase="confirmation")
    assert len(later) == 36 and sum(r["arms"][0] == "control" for r in later) == 18
    assert len({(r["case_id"], r["repetition"]) for r in later}) == 36


def test_confirmation_disjointness_not_relabeling_or_just_new_case_ids():
    known = {"skill_ids": [], "repository_families": [], "source_digests": []}
    new = case_set("unseen", public=True)
    plan = confirmation_plan(protocol(), new, known, candidate_digest=sha256_json("v1"))
    assert plan["researchEvidenceEligible"] is False
    for kind in ("repo", "skill", "text"):
        inventory = copy.deepcopy(known)
        if kind == "repo":
            inventory["repository_families"] = [new[0]["repository_family"]]
        elif kind == "skill":
            inventory["skill_ids"] = [new[0]["skill_id"]]
        else:
            inventory["source_digests"] = [sha256_json({k: new[0]["agent_input"][k]
                                                       for k in ("skill_text", "references")})]
        with pytest.raises(ValueError, match="overlaps"):
            confirmation_plan(protocol(), new, inventory, candidate_digest=sha256_json("v1"))
    with pytest.raises(ValueError, match="synthetic"):
        confirmation_plan(protocol(), case_set("new"), known, candidate_digest=sha256_json("v1"))


def test_preparation_agent_projection_has_no_labels_or_executable_source(tmp_path, monkeypatch):
    import evaluation.bounded_pilot as module
    monkeypatch.setattr(module, "source_fingerprint", lambda: {"digest": sha256_json("source"), "files": {}})
    p = protocol()
    registry = tmp_path / "fixed" / "ledger.sqlite"
    report = prepare(p, tmp_path / "one", registry=registry)
    assert report["realModelCalls"] == 0 and report["liveAdapterReady"] is False
    assert not report["runtimeLargeEvaluationAllowed"]
    exposed = read_json(tmp_path / "one/agent/dev-0.json")
    assert set(exposed) == {"task", "skill_text", "references", "tools", "arguments"}
    assert not list((tmp_path / "one").rglob("*.sh"))
    assert set(exposed).isdisjoint({"reference_digest", "kind", "expected_parameters", "families"})
    assert "state" not in exposed and "provider_fixture" not in exposed
    assert read_json(tmp_path / "one/provider-private/dev-0.json") == {"state": {"value": 0}}
    second = prepare(p, tmp_path / "two", registry=registry)
    assert report["protocol_digest"] == second["protocol_digest"]
    assert BudgetLedger(registry).snapshot(p["study_id"])["arm_count"] == 0
    changed = copy.deepcopy({k: v for k, v in p.items() if k != "digest"})
    changed["development_cases"][0]["reference_digest"] = sha256_json("changed label")
    with pytest.raises(BudgetError, match="fingerprint"):
        prepare(seal(changed), tmp_path / "three", registry=registry)
    with pytest.raises(ValueError, match="new output"):
        prepare(p, tmp_path / "one", registry=registry)


def test_common_input_is_deep_copied_and_source_scripts_are_only_text():
    case = case_set()[0]
    control = agent_context(case, arm="control", isolation_id="a")
    treatment = agent_context(case, arm="treatment", isolation_id="b")
    assert control.input_digest == treatment.input_digest
    assert control.initial_state_digest == treatment.initial_state_digest
    assert control.initial_state_digest == sha256_json(case["provider_fixture"]["state"])
    assert control.fixture_digest == treatment.fixture_digest == sha256_json(case["provider_fixture"])
    assert control.fixture_digest != control.initial_state_digest
    control.inputs["arguments"]["new"] = 900
    assert treatment.inputs["arguments"] == {}
    assert case["agent_input"]["arguments"] == {}
    assert "fixture" not in control.inputs and "provider_fixture" not in control.inputs
    assert case["provider_fixture"] == {"state": {"value": 0}}


@pytest.mark.parametrize("payload", ['{"x":1,"x":2}', '{"x":NaN}'])
def test_json_source_does_not_coerce_duplicate_keys_or_nan(tmp_path, payload):
    path = tmp_path / "input.json"
    path.write_text(payload)
    with pytest.raises(ValueError):
        read_json(path)


def meter(tmp_path):
    clock = Clock()
    ledger = BudgetLedger(tmp_path / "fixed.sqlite", clock=clock)
    ledger.register_study("probe", "protocol")
    candidate = ledger.register_candidate("probe", "code")
    arm = ledger.start_arm("probe", candidate, "case", 1, "B", "same-input")
    return MeasuredCalls(ledger, arm, clock=clock), ledger, clock


def test_full_arm_cost_includes_compile_qualification_native_fallback_and_tail(tmp_path):
    calls, ledger, clock = meter(tmp_path)
    def response(_):
        clock.now += 2
        return {"response": "ok", "input_tokens": 10, "output_tokens": 5}
    calls.invoke("compile", "compile", {"source": "inert"}, input_tokens=20, max_output_tokens=10, invoke=response)
    calls.timed_stage("qualification", lambda: setattr(clock, "now", clock.now + 3))
    calls.invoke("fallback", "fallback", {}, input_tokens=20, max_output_tokens=10, invoke=response)
    calls.timed_stage("delivery", lambda: setattr(clock, "now", clock.now + 1))
    result = calls.close()
    assert result["wall_ms"] == 8000
    assert result["ledger"]["elapsed_ms"] == 8000
    assert sum(c["input_tokens"] for c in result["model_usage"]) == 20
    assert result["liveAdapterReady"] is False
    assert ledger.snapshot("probe")["usage"]["model_requests"] == 2
    with pytest.raises(BudgetError):
        calls.invoke("post-agent", "runtime", {}, input_tokens=20, max_output_tokens=10, invoke=response)
    with pytest.raises(BudgetError):
        calls.guard_effect()


def test_reservation_precedes_callback_and_unknown_cannot_fallback(tmp_path):
    calls, ledger, _ = meter(tmp_path)
    def interrupted(_):
        assert ledger.snapshot("probe")["pending_outcome"]
        raise TimeoutError("transport unknown")
    with pytest.raises(TimeoutError):
        calls.invoke("first", "compile", {}, input_tokens=20, max_output_tokens=10, invoke=interrupted)
    dispatched = []
    with pytest.raises(BudgetError, match="halted"):
        calls.invoke("fallback", "fallback", {}, input_tokens=20, max_output_tokens=10, invoke=dispatched.append)
    assert not dispatched
    assert calls.close()["ledger"]["status"] == "outcome_unknown"


def test_meter_reservation_exceeded_usage_is_not_a_successful_response(tmp_path):
    calls, ledger, _ = meter(tmp_path)
    with pytest.raises(BudgetError):
        calls.invoke("bad-usage", "native", {}, input_tokens=20, max_output_tokens=10,
                     invoke=lambda _: {"response": "ignored", "input_tokens": 21, "output_tokens": 1})
    assert ledger.snapshot("probe")["study"]["status"] == "halted"


def test_fixture_meter_cannot_claim_to_be_live_tokenizer_attested(tmp_path):
    calls, ledger, _ = meter(tmp_path)
    with pytest.raises(ValueError, match="not implemented"):
        MeasuredCalls(ledger, calls.arm_id, counting_mode="tokenizer_attested")


def test_reconstructed_meter_retains_all_persisted_costs_and_elapsed_time(tmp_path):
    calls, ledger, clock = meter(tmp_path)
    calls.invoke("first", "compile", {}, input_tokens=20, max_output_tokens=10,
                 invoke=lambda _: {"response": "candidate", "input_tokens": 8, "output_tokens": 3})
    clock.now += 5
    rebuilt = MeasuredCalls(BudgetLedger(ledger.path, clock=clock), calls.arm_id, clock=clock)
    clock.now += 2
    result = rebuilt.close()
    assert result["wall_ms"] == 7000
    assert result["model_usage"][0]["input_tokens"] == 8
    assert result["model_usage"][0]["id"] == "first"


def test_local_stage_cannot_execute_after_unknown_model_outcome(tmp_path):
    calls, _, _ = meter(tmp_path)
    with pytest.raises(BudgetError):
        calls.invoke("unknown", "compile", {}, input_tokens=20, max_output_tokens=10,
                     invoke=lambda _: {"response": "untrusted", "input_tokens": None, "output_tokens": None})
    observed = []
    with pytest.raises(BudgetError):
        calls.timed_stage("provider", lambda: observed.append("should not execute"))
    assert observed == []


def references_for(cases):
    from evaluation.bounded_scoring import seal_reference
    references = []
    for case in cases:
        reference = seal_reference({"case_id": case["case_id"], "skill_id": case["skill_id"],
            "repository_id": case["repository_family"], "domain": case["domain"], "kind": case["kind"],
            "initial_state_digest": case_initial_state_digest(case),
            "criteria": [{"id": "scope", "critical": True, "statement": "Do not mutate state",
                          "source_quote": "Read only"}],
            "duties": [{"id": "read", "critical": False, "strict_eligible": True,
                        "statement": "Read scoped state", "source_quote": "Read the scoped value"}],
            "calls": [], "allowed_outcomes": ["completed"] if case["kind"] == "positive" else ["rejected"]})
        case["reference_digest"] = reference["reference_digest"]
        references.append(reference)
    return references


def test_references_bind_scope_text_identity_and_canonical_repository_family():
    from evaluation.bounded_pilot import assess
    p = protocol()
    body = {k: v for k, v in p.items() if k != "digest"}
    references = references_for(body["development_cases"])
    p = seal(body)
    assert validate_references(p, references) == references
    result = assess(p, references, [])
    assert result["scorecard"]["status"] == "inconclusive"
    assert result["pilotQualified"] is False
    from evaluation.bounded_scoring import seal_reference
    changed = copy.deepcopy(references)
    changed[0]["criteria"][0]["source_quote"] = "PRIVATE expected correct answer"
    changed[0] = seal_reference(changed[0])
    body["development_cases"][0]["reference_digest"] = changed[0]["reference_digest"]
    with pytest.raises(ValueError, match="visible source"):
        validate_references(seal(body), changed)


def test_empty_tool_schema_and_untrusted_foreign_fields_rejected():
    value = case_set()[0]["agent_input"]
    value["tools"][0]["input_schema"] = False
    with pytest.raises(ValueError):
        validate_agent_input(value)


@pytest.mark.parametrize("mutation", ["valid", "unknown_tool", "invalid_arguments"])
def test_reference_calls_must_be_possible_with_frozen_host_catalog(mutation):
    from evaluation.bounded_scoring import seal_reference
    body = {k: v for k, v in protocol().items() if k != "digest"}
    references = references_for(body["development_cases"])
    references[0]["calls"] = [{"id": "read", "tool": "observe", "kind": "read",
        "object_id": "scoped", "arguments": {}, "property": "value", "expected_value": 0,
        "min_calls": 1, "max_calls": 1, "approval_required": False, "verify_with": None}]
    if mutation == "unknown_tool":
        references[0]["calls"][0]["tool"] = "nonexistent"
    elif mutation == "invalid_arguments":
        references[0]["calls"][0]["arguments"] = {"not_in_schema": True}
    references[0] = seal_reference(references[0])
    body["development_cases"][0]["reference_digest"] = references[0]["reference_digest"]
    if mutation == "valid":
        assert validate_references(seal(body), references) == references
    else:
        with pytest.raises(ValueError, match="frozen"):
            validate_references(seal(body), references)


def test_reference_and_agent_context_match_real_provider_database_state(tmp_path):
    from evaluation.bounded_provider import FIXTURE_SCHEMA, LocalProviderPool
    body = {key: value for key, value in protocol().items() if key != "digest"}
    case = body["development_cases"][0]
    case["provider_fixture"] = {"schema": FIXTURE_SCHEMA, "state": {"scoped": {"value": 0}}, "tools": [{
        "name": "observe", "description": case["agent_input"]["tools"][0]["description"],
        "input_schema": case["agent_input"]["tools"][0]["input_schema"],
        "contract_id": "observe", "operation": "read", "kind": "read",
        "target": {"constant": "scoped"}, "property": "value", "value": None, "requires_approval": False}]}
    case["input_digest"] = case_input_digest(case)
    references = references_for(body["development_cases"])
    assert validate_references(seal(body), references) == references
    pool = LocalProviderPool(tmp_path / "fixed-provider-pool")
    providers = [pool.create_arm("arm-A", "control", case["provider_fixture"]),
                 pool.create_arm("arm-B", "treatment", case["provider_fixture"])]
    assert providers[0].path != providers[1].path
    try:
        for provider, arm in zip(providers, ("control", "treatment"), strict=True):
            context = agent_context(case, arm=arm, isolation_id=provider.binding["isolation_id"])
            snapshot = provider.snapshot()
            receipt = provider.invoke("observe", {}, request_id="initial-read")["receipt"]
            assert snapshot["state"] == case["provider_fixture"]["state"]
            assert context.initial_state_digest == snapshot["state_digest"] == references[0]["initial_state_digest"]
            assert receipt["initial_state_digest"] == snapshot["state_digest"]
            assert context.fixture_digest == receipt["fixture_digest"] == sha256_json(case["provider_fixture"])
            assert context.fixture_digest != context.initial_state_digest
            assert receipt["result"]["value"] == 0 and receipt["call"]["independent"] is True
            assert receipt["call"]["origin"] == "agent" and receipt["call"]["after_agent_end"] is False
            assert receipt["productEffectAuthority"] is False
            assert "state" not in context.inputs and "provider_fixture" not in context.inputs
    finally:
        for provider in providers:
            provider.close()


def test_old_full_fixture_reference_is_rejected_without_rewriting_label_or_case():
    from evaluation.bounded_scoring import seal_reference
    body = {key: value for key, value in protocol().items() if key != "digest"}
    references = references_for(body["development_cases"])
    references[0]["initial_state_digest"] = sha256_json(body["development_cases"][0]["provider_fixture"])
    references[0] = seal_reference(references[0])
    body["development_cases"][0]["reference_digest"] = references[0]["reference_digest"]
    frozen = seal(body)
    before = copy.deepcopy((frozen, references))
    with pytest.raises(ValueError, match="Provider state, not the full fixture"):
        validate_references(frozen, references)
    assert (frozen, references) == before


def test_fixture_policy_changes_keep_state_digest_but_change_full_input_binding():
    case = case_set()[0]
    original = agent_context(case, arm="control", isolation_id="A")
    changed = copy.deepcopy(case)
    changed["provider_fixture"]["tools"] = [{"requires_approval": True, "description": "changed host policy"}]
    with pytest.raises(ValueError, match="input drift"):
        agent_context(changed, arm="treatment", isolation_id="B")
    changed["input_digest"] = case_input_digest(changed)
    updated = agent_context(changed, arm="treatment", isolation_id="B")
    assert original.initial_state_digest == updated.initial_state_digest
    assert original.fixture_digest != updated.fixture_digest
    assert original.input_digest != updated.input_digest
    changed["provider_fixture"]["state"]["value"] = 1
    changed["input_digest"] = case_input_digest(changed)
    changed_state = agent_context(changed, arm="treatment", isolation_id="C")
    assert changed_state.initial_state_digest != original.initial_state_digest


@pytest.mark.parametrize("fixture", [{}, {"state": None}, {"state": []}, {"state": "not an object"}])
def test_missing_or_invalid_state_cannot_fall_back_to_hashing_whole_fixture(fixture):
    cases = case_set()
    cases[0]["provider_fixture"] = fixture
    cases[0]["input_digest"] = case_input_digest(cases[0])
    with pytest.raises(ValueError, match="explicit state object"):
        validate_cases(cases)
    with pytest.raises(ValueError, match="explicit state object"):
        agent_context(cases[0], arm="control", isolation_id="A")


def test_frozen_schema_validation_never_fetches_external_references():
    value = case_set()[0]["agent_input"]
    value["tools"][0]["input_schema"]["$ref"] = "https://invalid.example/schema.json"
    with pytest.raises(ValueError, match="local-only"):
        validate_agent_input(value)


def test_cli_help_is_zero_inference_and_has_no_live_run_flag():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-m", "evaluation.bounded_pilot", "--help"],
                            text=True, capture_output=True, check=True)
    assert "{check,prepare,inspect,score}" in result.stdout
    assert "--run" not in result.stdout
    assert json.loads(json.dumps(protocol()))["schema"].endswith("/v1")
