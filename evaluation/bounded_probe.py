"""Zero-model R0 measurement fixtures; never Agent or research qualification.

All scorer receipts, reviews and model telemetry below are invented fixture
data. The budget database is an output-scoped mechanism fixture, not the fixed
official study registry. Only the optional gateway demo executes real code
against isolated local mock state; it performs no automatic authoring.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

from evaluation import bounded_budget, bounded_scoring
from evaluation.bounded_budget import BudgetError, BudgetLedger
from evaluation.bounded_scoring import bind_review, digest, paired_report, score_arm, seal_observation, seal_reference


FAMILIES = (
    "wrong_object", "wrong_then_right", "unrelated_verification", "all_reject",
    "rollback_completion", "compiler_failure_fallback", "false_prose_completion",
    "timeout_unknown_usage", "unchanged_revision", "evaluator_added_action",
    "compilation_delay", "cross_arm_state_leakage",
)


def _fixture():
    contracts = [dict(id="write", tool="set_state", kind="effect", object_id="router-1",
        arguments={"target": "router-1", "enabled": True}, property="enabled", expected_value=None,
        min_calls=1, max_calls=1, approval_required=True, verify_with="verify"),
        dict(id="verify", tool="read_state", kind="verify", object_id="router-1",
        arguments={"target": "router-1"}, property="enabled", expected_value=True,
        min_calls=1, max_calls=1, approval_required=False, verify_with=None)]
    ref = seal_reference(dict(case_id="synthetic-case", skill_id="synthetic-skill", repository_id="synthetic-repo",
        domain="network", kind="positive", initial_state_digest=digest({"enabled": False}),
        criteria=[{"id": "artifact", "critical": False, "statement": "Deliver the fixture's declared terminal result.",
                   "source_quote": "Report the declared terminal result."}],
        duties=[{"id": "duty", "critical": True, "strict_eligible": True,
                 "statement": "Preserve exact target, approval and independent verification obligations.",
                 "source_quote": "Bind the exact target, obtain approval and verify independently."}],
        calls=contracts, allowed_outcomes=["completed"]))
    obs = dict(case_id=ref["case_id"], repetition=1, arm="treatment", initial_state_digest=ref["initial_state_digest"],
        isolation_id="synthetic-B", route="runtime", outcome="completed", protocol_completed=True,
        timed_out=False, sandbox_escape=False, artifact_digest=digest("artifact"), initial_artifact_digest=digest("artifact"),
        revision_count=0, calls=[{**{k: deepcopy(c[k]) for k in ("tool", "kind", "object_id", "arguments", "property")},
            "id": f"receipt-{i}", "contract_id": c["id"], "sequence": i, "result_value": c["expected_value"],
            "outcome": "ok", "independent": c["kind"] == "verify", "approved": True,
            "origin": "runtime", "after_agent_end": False} for i, c in enumerate(contracts)],
        model_usage=[dict(id="fixture-model", stage="compile", input_tokens=100, output_tokens=10, wall_ms=10)],
        wall_ms=100, online_proposal=dict(source="online_session", proposal_digest=digest("fixture-proposal"), loadable=True),
        review=dict(review_kind="external_developer_review", measurement_valid=True, criteria={"artifact": "met"},
            false_completion=False, duties={"duty": dict(fidelity="met", representation="strict", strict_correct=True)}))
    return ref, obs


def _seal(ref, obs):
    ref = seal_reference(ref)
    obs["review"] = bind_review(ref, obs, obs["review"])
    return ref, seal_observation(obs)


def _pair(ref, obs):
    ref, treatment = _seal(ref, obs)
    control = deepcopy(obs)
    control.update(arm="control", isolation_id="synthetic-A", route="native", online_proposal=None)
    control["review"]["duties"] = {}
    control["model_usage"][0]["stage"] = "native"
    _, control = _seal(ref, control)
    return [ref], [control, treatment]


def _select(value, key):
    for part in key.split("."):
        value = value[part]
    return value


def scorer_probes():
    """Produce explicit counterexamples and controls using only scorer APIs."""
    fixtures, rows = [], []
    for family in FAMILIES:
        for variant in ("control", "counterexample"):
            ref, obs = _fixture()
            bad = variant == "counterexample"
            expected = {"status": "pass", "task_ok": True}
            paired = family in {"timeout_unknown_usage", "cross_arm_state_leakage"}
            if family == "wrong_object" and bad:
                obs["calls"][0]["object_id"] = "router-2"
                expected = {"status": "fail", "all_calls_exact": False, "false_commit": True}
            elif family == "wrong_then_right" and bad:
                wrong = deepcopy(obs["calls"][0])
                wrong.update(id="wrong-first", arguments={"target": "router-2", "enabled": True})
                obs["calls"].insert(0, wrong)
                for i, call in enumerate(obs["calls"]):
                    call["sequence"] = i
                expected = {"status": "fail", "effect_exposures": 2}
            elif family == "unrelated_verification" and bad:
                obs["calls"][1]["object_id"] = "router-2"
                expected = {"status": "fail", "verification_ok": False, "false_commit": True}
            elif family == "all_reject":
                obs.update(calls=[], outcome="rejected", route="safe_stop")
                if not bad:
                    ref.update(kind="boundary", allowed_outcomes=["rejected"], calls=[])
                expected = {"status": "fail" if bad else "pass", "task_ok": not bad, "write_safety": "not_tested"}
            elif family == "rollback_completion":
                ref["calls"][0]["kind"] = obs["calls"][0]["kind"] = "compensate"
                if not bad:
                    ref.update(kind="boundary", allowed_outcomes=["recovered"])
                    obs["outcome"] = "recovered"
                expected = {"status": "fail" if bad else "pass", "false_commit": bad}
            elif family == "compiler_failure_fallback" and bad:
                ref["calls"], obs["calls"] = [], []
                obs.update(route="fallback", online_proposal=dict(source="online_session", proposal_digest=None, loadable=False))
                obs["model_usage"].append(dict(id="fixture-fallback", stage="fallback", input_tokens=10, output_tokens=10, wall_ms=10))
                obs["review"]["duties"]["duty"].update(representation="l1", strict_correct=None)
                expected.update({"translation.strict_recall.rate": 0, "translation.sampled_task_fully_strict": False})
            elif family == "false_prose_completion" and bad:
                obs["review"]["false_completion"] = True
                expected = {"status": "fail", "stop_batch": True}
            elif family == "timeout_unknown_usage":
                if bad:
                    obs["timed_out"], obs["model_usage"][0]["input_tokens"] = True, None
                expected = {"assigned_pairs": 1, "overall.treatment.successes": 0 if bad else 1,
                            "costs.treatment.input_tokens": None if bad else 100}
            elif family == "unchanged_revision":
                obs["revision_count"] = 1
                obs["artifact_digest"] = digest("artifact" if bad else "revised-artifact")
                if bad:
                    obs["review"]["criteria"]["artifact"] = "not_met"
                expected = {"status": "fail" if bad else "pass", "revision_changed": not bad}
            elif family == "evaluator_added_action" and bad:
                obs["calls"][0].update(origin="evaluator", after_agent_end=True)
                expected = {"status": "measurement_invalid", "task_ok": None}
            elif family == "compilation_delay":
                obs["model_usage"][0]["wall_ms"] = 150
                obs["wall_ms"] = 100 if bad else 250
                expected = {"status": "measurement_invalid"} if bad else {"status": "pass", "cost.wall_ms": 250,
                                                                              "cost.by_stage.compile.wall_ms": 150}
            elif family == "cross_arm_state_leakage":
                expected = {"status": "measurement_invalid"} if bad else {"assigned_pairs": 1, "overall.treatment.successes": 1}
            if paired:
                refs, observations = _pair(ref, obs)
                if family == "cross_arm_state_leakage" and bad:
                    observations[1]["isolation_id"] = observations[0]["isolation_id"]
                    observations[1] = seal_observation(observations[1])
                inputs = {"cases": refs, "observations": observations}
                result = paired_report(**inputs)
            else:
                ref, obs = _seal(ref, obs)
                inputs = {"reference": ref, "observation": obs}
                result = score_arm(ref, obs)
            fixtures.append(dict(id=f"{family}:{variant}", evaluator="paired_report" if paired else "score_arm", inputs=inputs))
            actual = {key: _select(result, key) for key in expected}
            rows.append(dict(id=fixtures[-1]["id"], expected=expected, actual=actual,
                             matched=actual == expected, fixture_digest=digest(fixtures[-1])))
    return fixtures, rows


def budget_probes(path):
    """Exercise persistent gates with invented reservations and scripted time."""
    now, checks = [1000.0], []
    ledger = BudgetLedger(path, clock=lambda: now[0])
    study = "synthetic-budget-fixture-not-official-study"
    ledger.register_study(study, digest("fixture-protocol"))
    candidate = ledger.register_candidate(study, digest("fixture-candidate"))
    arm = ledger.start_arm(study, candidate, "fixture-case", 1, "A", digest("fixture-context"))

    def rejected(name, action, message):
        try:
            action()
        except BudgetError as exc:
            checks.append(dict(id=name, matched=message in str(exc), rejection=str(exc)))
        else:
            checks.append(dict(id=name, matched=False, rejection=None))

    checks.append(dict(id="legal_recovery_reserve", matched=ledger.guard_effect(arm)["effect_authority_granted"] is False))
    ledger.reserve_model_call(arm, "fixture-known", "agent", 100, 20)
    now[0] += 2
    checks.append(dict(id="known_usage_settles", matched=ledger.settle_call(arm, "fixture-known", 10, 5)["status"] == "settled"))
    ledger = BudgetLedger(path, clock=lambda: now[0])
    rejected("settled_request_cannot_replay", lambda: ledger.reserve_model_call(arm, "fixture-known", "agent", 1, 1), "claimed")
    now[0] = 1361.0
    rejected("effect_recovery_reserve_enforced", lambda: ledger.guard_effect(arm), "recovery reserve")
    ledger.reserve_model_call(arm, "fixture-unknown", "compiler", 200, 300)
    ledger = BudgetLedger(path, clock=lambda: now[0])
    rejected("pending_survives_reopen", lambda: ledger.reserve_model_call(arm, "renamed", "agent", 1, 1), "unsettled")
    rejected("pending_blocks_effect", lambda: ledger.guard_effect(arm), "unsettled")
    unknown = ledger.settle_call(arm, "fixture-unknown")
    checks.append(dict(id="unknown_retains_reservation", matched=unknown["status"] == "unknown" and unknown["charged_output"] == 300))
    rejected("unknown_halts_new_work", lambda: ledger.reserve_model_call(arm, "another-name", "agent", 1, 1), "halted")
    terminal = ledger.finish_arm(arm, "completed")
    checks.append(dict(id="unknown_cannot_be_completion", matched=terminal["status"] == "outcome_unknown"))
    rejected("terminal_cannot_be_replaced", lambda: ledger.finish_arm(arm, "completed"), "terminal")
    return {"scope": "output_scoped_fixture_not_official_registry", "clock": "scripted_epoch_seconds",
            "actualModelCalls": 0, "checks": checks, "snapshot": ledger.snapshot(study)}


def run(output: str | Path, *, include_gateway: bool = True) -> dict:
    """Write a new artifact directory; existing paths are never reused."""
    if include_gateway and os.environ.get("NETOPYU_BACKEND", "mock").strip().lower() != "mock":
        raise ValueError("gateway fixture requires NETOPYU_BACKEND unset or mock")
    target = Path(output)
    target.mkdir(parents=True, exist_ok=False)
    fixtures, scorer = scorer_probes()
    budget = budget_probes(target / "fixture-budget.sqlite3")
    gateway = {"status": "not_tested", "automaticAuthoring": "not_implemented"}
    if include_gateway:
        from evaluation.flow_effect_demo import run_demo
        evidence = asyncio.run(run_demo(approve_local_simulation=True))
        states = [row["state"] for row in evidence["outcomes"]]
        gateway.update(status="synthetic_mechanisms_only", report=evidence,
                       matched=states == ["verified_success", "precondition_changed", "rollback_verified"]
                       and evidence["modelCalls"] == 0 and not evidence["outcomes"][1]["mockChanges"])
    sources = {Path(module.__file__).name: "sha256:" + hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
               for module in (bounded_budget, bounded_scoring)}
    sources[Path(__file__).name] = "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    body = dict(schema="netopyu.io/bounded-probe/v1", actualModelCalls=0,
        evidenceRole="synthetic_measurement_probe_not_agent_benchmark", researchEvidenceEligible=False,
        pilotQualified=False, liveAdapterReady=False, sourceDigests=sources,
        fixtureNotice="All scorer receipts, semantic reviews and model usage are invented fixtures; budget reservations never call a model.",
        scorerFamilyCount=len(FAMILIES), scorerProbeCount=len(scorer), scorer=scorer, budget=budget, gateway=gateway,
        fixturesDigest=digest(fixtures), all_expected_decisions=all(row["matched"] for row in scorer + budget["checks"])
            and gateway.get("matched", True), runtime36ProbeGate="not_assessed", automaticEffectBridge="not_tested")
    report = {**body, "reportDigest": digest(body)}
    for name, value in (("scorer-fixtures.json", fixtures), ("report.json", report)):
        with (target / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--skip-gateway", action="store_true")
    args = parser.parse_args(argv)
    report = run(args.output, include_gateway=not args.skip_gateway)
    print(json.dumps({"report": str(args.output / "report.json"), "actualModelCalls": 0,
                      "all_expected_decisions": report["all_expected_decisions"], "pilotQualified": False}))
    return 0 if report["all_expected_decisions"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
