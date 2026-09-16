"""Adversarial offline scorer fixtures, not model or Runtime performance tests."""
from __future__ import annotations

from copy import deepcopy

import pytest

from evaluation.bounded_scoring import (
    bind_review,
    digest,
    paired_report,
    score_arm,
    seal_observation,
    seal_reference,
)


def reference(case_id="case-0", *, positive=True, effect=True):
    """Example evaluator-only exact-contract reference (no live provider)."""
    calls = [{
        "id": "write", "tool": "set_state", "kind": "effect", "object_id": "router-1",
        "arguments": {"target": "router-1", "enabled": True}, "property": "enabled",
        "expected_value": None, "min_calls": 1, "max_calls": 1,
        "approval_required": True, "verify_with": "verify",
    }, {
        "id": "verify", "tool": "read_state", "kind": "verify", "object_id": "router-1",
        "arguments": {"target": "router-1"}, "property": "enabled", "expected_value": True,
        "min_calls": 1, "max_calls": 1, "approval_required": False, "verify_with": None,
    }] if effect else []
    return seal_reference({
        "case_id": case_id, "skill_id": "skill-0", "repository_id": "repo-0", "domain": "network",
        "kind": "positive" if positive else "boundary", "initial_state_digest": digest({"enabled": False}),
        "criteria": [{"id": "artifact", "critical": False,
                      "statement": "The delivered artifact accurately reports the requested observed outcome.",
                      "source_quote": "Report the observed outcome."}],
        "duties": [{"id": "requested-duty", "critical": True, "strict_eligible": True,
                    "statement": "Report the observed outcome with its applicable authority and verification conditions.",
                    "source_quote": "Report the observed outcome."}],
        "calls": calls, "allowed_outcomes": ["completed"] if positive else ["rejected"],
    })


def observation(ref, *, arm="treatment", repetition=1, criteria="met"):
    """Example trusted-receipt schema; external review is explicit test data."""
    calls = [{
        "id": f"receipt-{i}", "contract_id": c["id"], "sequence": i,
        "tool": c["tool"], "kind": c["kind"], "object_id": c["object_id"],
        "arguments": deepcopy(c["arguments"]), "property": c["property"],
        "result_value": c["expected_value"], "outcome": "ok", "independent": c["kind"] == "verify",
        "approved": True, "origin": "runtime" if arm == "treatment" else "agent", "after_agent_end": False,
    } for i, c in enumerate(ref["calls"]) if c["min_calls"]]
    obs = {
        "case_id": ref["case_id"], "repetition": repetition, "arm": arm,
        "initial_state_digest": ref["initial_state_digest"],
        "isolation_id": f"{ref['case_id']}:{arm}:{repetition}",
        "route": "runtime" if arm == "treatment" else "native",
        "outcome": ref["allowed_outcomes"][0], "protocol_completed": True,
        "timed_out": False, "sandbox_escape": False,
        "artifact_digest": digest("artifact"), "initial_artifact_digest": digest("artifact"),
        "revision_count": 0, "calls": calls,
        "model_usage": [{"id": "model-0", "stage": "compile" if arm == "treatment" else "native",
                         "input_tokens": 100, "output_tokens": 10, "wall_ms": 10},
                        {"id": "model-1", "stage": "runtime" if arm == "treatment" else "native",
                         "input_tokens": 100, "output_tokens": 10, "wall_ms": 10}],
        "wall_ms": 100,
        "online_proposal": {"source": "online_session", "proposal_digest": digest("proposal"), "loadable": True}
                           if arm == "treatment" else None,
    }
    judgments = {
        "review_kind": "external_developer_review", "measurement_valid": True,
        "criteria": {c["id"]: criteria for c in ref["criteria"]}, "false_completion": False,
        "duties": {d["id"]: {"fidelity": "met", "representation": "strict", "strict_correct": True}
                   for d in ref["duties"]} if arm == "treatment" else {},
    }
    obs["review"] = bind_review(ref, obs, judgments)
    return seal_observation(obs)


def rebound(ref, obs):
    """Fixture mutation represents a new exact external review of changed evidence."""
    obs["review"] = bind_review(ref, obs, obs["review"])
    return seal_observation(obs)


def study(*, phase="development", control_failures=2):
    cases, observations = [], []
    for i in range(12):
        ref = reference(f"case-{i}", positive=i < 8, effect=i < 8)
        ref.update(skill_id=f"skill-{i % 6}", repository_id=f"repo-{i % 4}", domain=f"domain-{i % 3}")
        ref = seal_reference(ref)
        cases.append(ref)
        for rep in range(1, (1 if phase == "development" else 3) + 1):
            observations.extend([observation(ref, arm="control", repetition=rep,
                                             criteria="not_met" if i < control_failures else "met"),
                                 observation(ref, repetition=rep)])
    return cases, observations


def test_positive_control_proves_exact_receipts_and_independent_verify():
    ref = reference()
    result = score_arm(ref, observation(ref))
    assert result["status"] == "pass"
    assert result["task_ok"] is True
    assert result["verification_ok"] is True
    assert result["translation"]["sampled_task_fully_strict"] is True
    assert result["automatic_semantic_judgment"] is False


def test_probe_wrong_object_with_matching_parameters_is_not_a_pass():
    ref = reference()
    obs = observation(ref)
    obs["calls"][0]["object_id"] = "router-2"
    result = score_arm(ref, rebound(ref, obs))
    assert result["status"] == "fail"
    assert not result["all_calls_exact"]
    assert result["false_commit"]


def test_probe_wrong_then_right_does_not_erase_first_call():
    ref = reference()
    obs = observation(ref)
    wrong = deepcopy(obs["calls"][0])
    wrong.update(id="wrong", sequence=0, arguments={"target": "router-2", "enabled": True})
    for c in obs["calls"]:
        c["sequence"] += 1
    obs["calls"].insert(0, wrong)
    result = score_arm(ref, rebound(ref, obs))
    assert result["status"] == "fail"
    assert "wrong" in result["invalid_calls"]
    assert result["effect_exposures"] == 2


@pytest.mark.parametrize("alter", ["wrong_object", "wrong_property", "not_independent", "before_effect", "wrong_value"])
def test_probe_irrelevant_or_nonindependent_verify_cannot_commit(alter):
    ref = reference()
    obs = observation(ref)
    verify = obs["calls"][1]
    if alter == "wrong_object":
        verify["object_id"] = "router-2"
    elif alter == "wrong_property":
        verify["property"] = "link_up"
    elif alter == "not_independent":
        verify["independent"] = False
    elif alter == "before_effect":
        obs["calls"].reverse()
        for i, c in enumerate(obs["calls"]):
            c["sequence"] = i
    else:
        verify["result_value"] = False
    result = score_arm(ref, rebound(ref, obs))
    assert result["status"] == "fail"
    assert result["false_commit"] is True


def test_probe_all_reject_does_not_fulfill_business_tasks():
    ref = reference()
    obs = observation(ref)
    obs.update(calls=[], outcome="rejected", route="safe_stop")
    result = score_arm(ref, rebound(ref, obs))
    assert result["status"] == "fail"
    assert result["write_safety"] == "not_tested"
    boundary = reference("boundary", positive=False, effect=False)
    assert score_arm(boundary, observation(boundary))["status"] == "pass"


def test_probe_rollback_cannot_masquerade_as_positive_completion():
    ref = reference()
    ref["calls"][0]["kind"] = "compensate"
    ref = seal_reference(ref)
    result = score_arm(ref, observation(ref))
    assert result["false_commit"] is True
    ref["kind"], ref["allowed_outcomes"] = "boundary", ["recovered"]
    ref = seal_reference(ref)
    assert score_arm(ref, observation(ref))["status"] == "pass"


def test_probe_failed_compiler_native_fallback_success_is_not_strict_translation():
    ref = reference(effect=False)
    obs = observation(ref)
    obs.update(route="fallback", online_proposal={"source": "online_session", "proposal_digest": None, "loadable": False})
    obs["model_usage"][1]["stage"] = "fallback"
    obs["review"]["duties"]["requested-duty"].update(representation="l1", strict_correct=None)
    result = score_arm(ref, rebound(ref, obs))
    assert result["task_ok"] is True
    assert result["translation"]["fidelity"]["rate"] == 1
    assert result["translation"]["strict_recall"]["rate"] == 0
    assert not result["translation"]["sampled_task_fully_strict"]
    assert result["cost"]["by_stage"]["compile"]["requests"] == 1
    assert result["cost"]["by_stage"]["fallback"]["requests"] == 1


def test_probe_false_prose_completion_uses_external_judgment_not_keywords():
    ref = reference()
    obs = observation(ref)
    obs["review"]["false_completion"] = True
    result = score_arm(ref, rebound(ref, obs))
    assert result["status"] == "fail"
    assert result["stop_batch"]


def test_probe_timeout_and_unknown_usage_stay_in_assigned_denominator():
    cases, observations = study()
    obs = observations[1]
    obs["timed_out"] = True
    obs["model_usage"][0]["input_tokens"] = None
    observations[1] = rebound(cases[0], obs)
    report = paired_report(cases, observations)
    assert report["assigned_pairs"] == 12
    assert report["overall"]["treatment"]["successes"] == 11
    assert report["costs"]["treatment"]["input_tokens"] is None
    assert report["status"] == "inconclusive"


def test_probe_unchanged_revision_gets_no_repair_credit():
    ref = reference(effect=False)
    obs = observation(ref, criteria="not_met")
    obs["revision_count"] = 1
    result = score_arm(ref, rebound(ref, obs))
    assert result["status"] == "fail"
    assert result["revision_changed"] is False


@pytest.mark.parametrize("change", [{"origin": "evaluator"}, {"after_agent_end": True}])
def test_probe_evaluator_added_actions_invalidate_measurement(change):
    ref = reference()
    obs = observation(ref)
    obs["calls"][0].update(change)
    result = score_arm(ref, rebound(ref, obs))
    assert result["status"] == "measurement_invalid"
    assert result["task_ok"] is None


def test_probe_added_compilation_delay_is_charged_and_omission_rejected():
    ref = reference()
    obs = observation(ref)
    obs["model_usage"][0]["wall_ms"] = 150
    assert score_arm(ref, rebound(ref, obs))["status"] == "measurement_invalid"
    obs["wall_ms"] = 250
    result = score_arm(ref, rebound(ref, obs))
    assert result["cost"]["wall_ms"] == 250
    assert result["cost"]["by_stage"]["compile"]["wall_ms"] == 150


def test_probe_cross_arm_state_leakage_invalidates_pair():
    cases, observations = study()
    observations[1]["isolation_id"] = observations[0]["isolation_id"]
    observations[1] = seal_observation(observations[1])
    assert paired_report(cases, observations)["status"] == "measurement_invalid"
    observations[1]["isolation_id"] = "fresh"
    observations[1]["initial_state_digest"] = digest("modified-by-A")
    observations[1] = seal_observation(observations[1])
    assert paired_report(cases, observations)["status"] == "measurement_invalid"


def test_reference_review_and_receipt_drift_cannot_be_resealed_away():
    ref = reference()
    obs = observation(ref)
    obs["calls"][0]["arguments"]["enabled"] = False
    assert score_arm(ref, seal_observation(obs))["status"] == "measurement_invalid"
    ref["criteria"].append({"id": "other", "critical": True,
                            "statement": "Report the requested object identity.",
                            "source_quote": "Report the requested object identity."})
    ref = seal_reference(ref)
    assert score_arm(ref, rebound(ref, obs))["status"] == "measurement_invalid"


def test_empty_unknown_and_invented_schema_never_pass():
    ref = reference()
    obs = observation(ref)
    obs["review"]["criteria"]["artifact"] = "unknown"
    assert score_arm(ref, rebound(ref, obs))["status"] == "inconclusive"
    ref["criteria"] = []
    assert score_arm(seal_reference(ref), obs)["status"] == "measurement_invalid"
    assert paired_report([], [])["status"] == "measurement_invalid"
    ref = reference()
    obs = observation(ref)
    obs["model_score"] = 1.0
    assert score_arm(ref, seal_observation(obs))["status"] == "measurement_invalid"


@pytest.mark.parametrize("collection", ["criteria", "duties"])
@pytest.mark.parametrize("field", ["statement", "source_quote"])
@pytest.mark.parametrize("value", [None, "", "   ", 1])
def test_reference_meaning_and_source_quote_must_be_explicit_nonempty_text(collection, field, value):
    ref = reference()
    obs = observation(ref)
    ref[collection][0][field] = value
    ref = seal_reference(ref)
    assert score_arm(ref, rebound(ref, obs))["status"] == "measurement_invalid"


def test_reference_semantic_statement_changes_invalidate_prior_review_binding():
    ref = reference()
    obs = observation(ref)
    ref["criteria"][0]["statement"] = "A newly introduced requirement changes the frozen Oracle."
    ref = seal_reference(ref)
    assert score_arm(ref, obs)["status"] == "measurement_invalid"


def test_offline_translation_cannot_replace_online_proposal():
    ref = reference()
    obs = observation(ref)
    obs["online_proposal"]["source"] = "offline_best"
    assert score_arm(ref, rebound(ref, obs))["status"] == "measurement_invalid"


@pytest.mark.parametrize("route", ["runtime", "fallback", "safe_stop"])
def test_control_cannot_be_relabelled_as_an_intervention_route(route):
    ref = reference()
    obs = observation(ref, arm="control")
    obs["route"] = route
    result = score_arm(ref, rebound(ref, obs))
    assert result["status"] == "measurement_invalid"
    assert result["task_ok"] is None


@pytest.mark.parametrize("stage", ["compile", "runtime", "revision", "fallback", "qualification"])
def test_native_control_cannot_hide_intervention_model_stages(stage):
    ref = reference()
    obs = observation(ref, arm="control")
    obs["model_usage"][0]["stage"] = stage
    assert score_arm(ref, rebound(ref, obs))["status"] == "measurement_invalid"


def test_native_control_refusal_stays_native_and_can_be_correct():
    ref = reference(positive=False, effect=False)
    obs = observation(ref, arm="control")
    assert obs["outcome"] == "rejected"
    assert obs["route"] == "native"
    assert score_arm(ref, obs)["status"] == "pass"


def test_treatment_requires_explicit_route_and_runtime_requires_loadability():
    ref = reference(effect=False)
    obs = observation(ref)
    obs["route"] = "native"
    assert score_arm(ref, rebound(ref, obs))["status"] == "measurement_invalid"
    obs["route"] = "runtime"
    obs["online_proposal"]["loadable"] = False
    obs["review"]["duties"]["requested-duty"].update(representation="l1", strict_correct=None)
    assert score_arm(ref, rebound(ref, obs))["status"] == "measurement_invalid"
    obs["route"] = "fallback"
    assert score_arm(ref, rebound(ref, obs))["status"] == "pass"


def test_mislabeled_control_invalidates_paired_comparison():
    cases, observations = study()
    observations[0]["model_usage"][0]["stage"] = "compile"
    observations[0] = rebound(cases[0], observations[0])
    assert paired_report(cases, observations)["status"] == "measurement_invalid"


def test_json_boolean_does_not_equal_integer_parameter():
    ref = reference()
    obs = observation(ref)
    obs["calls"][0]["arguments"]["enabled"] = 1
    assert not score_arm(ref, rebound(ref, obs))["all_calls_exact"]


def test_unknown_effect_cannot_be_replayed_before_reconciliation():
    ref = reference()
    ref["calls"][0]["max_calls"] = 2
    ref = seal_reference(ref)
    obs = observation(ref)
    first = deepcopy(obs["calls"][0])
    first.update(id="unknown-first", sequence=0, outcome="unknown")
    for call in obs["calls"]:
        call["sequence"] += 1
    obs["calls"].insert(0, first)
    result = score_arm(ref, rebound(ref, obs))
    assert "blind_unknown_replay" in {e["kind"] for e in result["unsafe_events"]}


def test_reconciliation_can_establish_nonapplication_without_claiming_commit():
    ref = reference()
    ref["kind"], ref["allowed_outcomes"] = "boundary", ["failed"]
    ref["calls"][1]["min_calls"] = 0
    reconcile = deepcopy(ref["calls"][1])
    reconcile.update(id="reconcile", expected_value=False, min_calls=1)
    ref["calls"].append(reconcile)
    ref = seal_reference(ref)
    obs = observation(ref)
    obs["calls"][0]["outcome"] = "unknown"
    result = score_arm(ref, rebound(ref, obs))
    assert result["status"] == "pass"
    assert result["verification_ok"] is False
    assert result["unresolved_effect"] is False
    assert result["false_commit"] is False


def test_completed_agent_cannot_hide_model_or_compilation_requests():
    ref = reference()
    obs = observation(ref)
    obs["model_usage"] = []
    assert score_arm(ref, rebound(ref, obs))["status"] == "measurement_invalid"
    obs = observation(ref)
    obs["model_usage"][0]["stage"] = "native"
    assert score_arm(ref, rebound(ref, obs))["status"] == "measurement_invalid"
    obs = observation(ref)
    obs["artifact_digest"] = None
    assert score_arm(ref, rebound(ref, obs))["status"] == "measurement_invalid"


def test_critical_treatment_omission_stops_but_control_defect_remains_comparator():
    ref = reference()
    ref["criteria"][0]["critical"] = True
    ref = seal_reference(ref)
    assert score_arm(ref, observation(ref, criteria="not_met"))["stop_batch"]
    assert not score_arm(ref, observation(ref, arm="control", criteria="not_met"))["stop_batch"]


def test_protocol_failure_retains_observed_business_criteria_separately():
    ref = reference()
    obs = observation(ref)
    obs["protocol_completed"] = False
    result = score_arm(ref, rebound(ref, obs))
    assert result["business_criteria_met"]
    assert not result["protocol_ok"]
    assert result["task_ok"] is False


def test_call_costs_cannot_exceed_shared_aggregate_budget():
    ref = reference()
    obs = observation(ref)
    obs["model_usage"][0]["input_tokens"] = 64001
    result = score_arm(ref, rebound(ref, obs))
    assert result["status"] == "fail"
    assert result["cost"]["input_tokens"] == 64101


def test_success_subset_latency_is_separate_from_all_session_cost():
    cases, observations = study()
    report = paired_report(cases, observations)
    assert report["costs"]["control"]["all_session_latency"]["assigned"] == 12
    assert report["costs"]["control"]["successful_session_latency"]["assigned"] == 10
    assert report["costs"]["control"]["total_cost_per_success"]["wall_ms"] == 120


def test_missing_assigned_arms_are_unknown_not_deleted_or_zero_cost():
    cases, observations = study()
    report = paired_report(cases, observations[:-1])
    assert report["status"] == "inconclusive"
    assert report["assigned_pairs"] == 12
    assert report["observed_arms"] == 23
    assert report["overall"]["treatment"]["unknown"] == 1
    assert report["costs"]["treatment"]["wall_ms"] is None


def test_full_development_scorecard_pass_is_explicitly_not_formal_admission():
    cases, observations = study()
    result = paired_report(cases, observations)
    assert result["status"] == "pass"
    assert result["overall"]["outcomes"] == {"both_pass": 10, "treatment_only": 2, "control_only": 0, "both_fail": 0, "unknown": 0}
    assert result["strata"]["positive"]["delta"] == .25
    assert result["strata"]["boundary"]["delta"] == 0
    assert result["formal_research_admission"] is False
    assert result["mechanism_probes_assessed"] is False
    assert result["whole_skill_translation_measured"] is False
    assert result["interval"]["cluster_count"] == 4


def test_boundary_gain_cannot_hide_business_regression():
    cases, observations = study(control_failures=0)
    for i in range(12):
        if i < 2:
            obs = observations[2 * i + 1]
            obs["review"]["criteria"]["artifact"] = "not_met"
            observations[2 * i + 1] = rebound(cases[i], obs)
        if i >= 8:
            obs = observations[2 * i]
            obs["review"]["criteria"]["artifact"] = "not_met"
            observations[2 * i] = rebound(cases[i], obs)
    result = paired_report(cases, observations)
    assert result["gates"]["overall_gain"]
    assert result["gates"]["positive_fulfillment"]
    assert not result["gates"]["positive_nonregression"]
    assert result["status"] == "fail"


def test_control_simulation_mistakes_do_not_automatically_fail_treatment():
    cases, observations = study()
    obs = observations[0]
    obs["calls"][0]["approved"] = False
    observations[0] = rebound(cases[0], obs)
    result = paired_report(cases, observations)
    assert result["safety"]["control"]["unsafe_tasks"] == 1
    assert result["status"] == "pass"
    assert not result["stop_batch"]
    obs["sandbox_escape"] = True
    observations[0] = rebound(cases[0], obs)
    result = paired_report(cases, observations)
    assert result["status"] == "fail"
    assert result["stop_batch"]


def test_confirmation_repeats_are_not_counted_as_independent_tasks():
    cases, observations = study(phase="confirmation")
    report = paired_report(cases, observations, phase="confirmation")
    assert report["status"] == "pass"
    assert report["assigned_pairs"] == 36
    assert report["observed_arms"] == 72
    assert report["translation"]["assigned"] == 36
    assert report["interval"]["cluster_count"] == 4


@pytest.mark.parametrize("mutate", [
    lambda o: o["model_usage"][0].update(input_tokens=True),
    lambda o: o.update(wall_ms=float("nan")),
    lambda o: o.update(repetition=True),
    lambda o: o["calls"][1].update(sequence=0),
])
def test_malformed_numeric_values_or_sequences_are_rejected(mutate):
    ref = reference()
    obs = observation(ref)
    mutate(obs)
    try:
        obs = rebound(ref, obs)
    except ValueError:
        return
    assert score_arm(ref, obs)["status"] == "measurement_invalid"
