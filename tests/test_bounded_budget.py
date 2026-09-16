"""Budget mechanisms only: no model calls or task-quality claims."""
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from evaluation.bounded_budget import BudgetError, BudgetLedger, CAPS


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def ledger(tmp_path):
    clock = Clock()
    value = BudgetLedger(tmp_path / "fixed-study.sqlite3", clock=clock)
    value.register_study("study", "protocol")
    candidate = value.register_candidate("study", "v1")
    return value, clock, candidate


def start(ledger, candidate, *, case="case", arm="A", repetition=1, context="context"):
    return ledger.start_arm("study", candidate, case, repetition, arm, context)


def complete_call(ledger, arm, request="r", *, stage="agent", input_tokens=10, output_tokens=10):
    ledger.reserve_model_call(arm, request, stage, input_tokens, output_tokens)
    return ledger.settle_call(arm, request, input_tokens, output_tokens)


def test_fingerprints_identity_and_terminal_survive_reopen(ledger):
    value, clock, candidate = ledger
    assert value.register_candidate("study", "v1") == candidate
    with pytest.raises(BudgetError, match="fingerprint"):
        value.register_study("study", "other")
    arm = start(value, candidate)
    complete_call(value, arm)
    clock.now += 7
    result = value.finish_arm(arm, "completed", elapsed_ms=1)
    assert result["elapsed_ms"] == 7000
    reopened = BudgetLedger(value.path, clock=clock)
    with pytest.raises(BudgetError, match="already claimed"):
        start(reopened, candidate)
    with pytest.raises(BudgetError, match="terminal"):
        reopened.reserve_model_call(arm, "new-output-folder", "agent", 1, 1)
    with pytest.raises(BudgetError, match="terminal"):
        reopened.guard_effect(arm)
    with pytest.raises(BudgetError, match="terminal"):
        reopened.finish_arm(arm, "failed")
    assert reopened.snapshot("study")["usage"]["model_requests"] == 1


def test_pending_reservation_blocks_reopen_and_retains_charge(ledger):
    value, clock, candidate = ledger
    arm = start(value, candidate)
    value.reserve_model_call(arm, "physical-1", "compiler", 200, 300)
    reopened = BudgetLedger(value.path, clock=clock)
    with pytest.raises(BudgetError, match="unsettled"):
        reopened.reserve_model_call(arm, "renamed", "agent", 1, 1)
    with pytest.raises(BudgetError, match="unsettled"):
        start(reopened, candidate, arm="B")
    snapshot = reopened.snapshot("study")
    assert snapshot["pending_outcome"]
    assert snapshot["usage"]["charged_output_tokens"] == 300
    unknown = reopened.settle_call(arm, "physical-1")
    assert unknown["status"] == "unknown" and unknown["charged_input"] == 200
    with pytest.raises(BudgetError, match="halted"):
        reopened.reserve_model_call(arm, "third-name", "agent", 1, 1)
    assert reopened.finish_arm(arm, "completed")["status"] == "outcome_unknown"


def test_inspect_arm_projects_all_previous_process_calls_and_original_clock(ledger):
    value, clock, candidate = ledger
    arm = start(value, candidate)
    complete_call(value, arm, "first-process", input_tokens=10, output_tokens=20)
    clock.now += 7
    reopened = BudgetLedger(value.path, clock=clock)
    complete_call(reopened, arm, "second-process", input_tokens=30, output_tokens=40)
    active = reopened.inspect_arm(arm)
    assert active["arm"]["observed_elapsed_ms"] == 7000
    assert {call["request_id"] for call in active["calls"]} == {"first-process", "second-process"}
    assert active["arm"]["usage"]["known_input_tokens"] == 40
    assert active["arm"]["usage"]["known_output_tokens"] == 60
    reopened.finish_arm(arm, "completed", elapsed_ms=1)
    clock.now += 100
    terminal = value.inspect_arm(arm)
    assert terminal["arm"]["elapsed_ms"] == terminal["arm"]["observed_elapsed_ms"] == 7000
    assert terminal["arm"]["remaining_seconds"] == 0
    with pytest.raises(BudgetError, match="unknown arm_id"):
        value.inspect_arm("not-an-arm")


@pytest.mark.parametrize("inputs,outputs,status", [(-1, 2, "invalid_usage"), (1, -1, "invalid_usage"),
                                                   (True, 1, "invalid_usage"), (11, 2, "budget_exceeded"),
                                                   (1, 21, "budget_exceeded"), (None, 1, "unknown")])
def test_bad_usage_records_attempt_and_halts_without_refund(ledger, inputs, outputs, status):
    value, _, candidate = ledger
    arm = start(value, candidate)
    value.reserve_model_call(arm, "r", "agent", 10, 20)
    result = value.settle_call(arm, "r", inputs, outputs)
    assert result["status"] == status
    assert result["charged_input"] >= 10 and result["charged_output"] >= 20
    snapshot = value.snapshot("study")
    assert snapshot["study"]["status"] == "halted"
    assert snapshot["usage"]["model_requests"] == 1
    with pytest.raises(BudgetError, match="already settled"):
        value.settle_call(arm, "r", 1, 1)


def test_actual_usage_releases_only_unused_tokens_not_request_identity(ledger):
    value, _, candidate = ledger
    arm = start(value, candidate)
    value.reserve_model_call(arm, "r", "agent", 64000, 6000)
    value.settle_call(arm, "r", 20, 30)
    with pytest.raises(BudgetError, match="already claimed"):
        value.reserve_model_call(arm, "r", "agent", 1, 1)
    value.reserve_model_call(arm, "next", "runtime", 63980, 5970)
    assert value.snapshot("study")["usage"]["charged_input_tokens"] == 64000


@pytest.mark.parametrize("arm_name", ["A", "B"])
def test_both_arms_have_equal_request_caps(ledger, arm_name):
    value, _, candidate = ledger
    arm = start(value, candidate, arm=arm_name)
    for i in range(10):
        complete_call(value, arm, str(i))
    with pytest.raises(BudgetError, match="model-request budget"):
        value.reserve_model_call(arm, "11", "agent", 1, 1)
    assert value.snapshot("study")["usage"]["model_requests"] == 10


@pytest.mark.parametrize("stage,maximum", [("compiler", 2), ("revision", 1)])
def test_stage_caps_include_all_attempts(ledger, stage, maximum):
    value, _, candidate = ledger
    arm = start(value, candidate)
    for i in range(maximum):
        complete_call(value, arm, str(i), stage=stage)
    with pytest.raises(BudgetError, match=stage + " request budget"):
        value.reserve_model_call(arm, "extra", stage, 1, 1)
    with pytest.raises(BudgetError, match="unknown model stage"):
        value.reserve_model_call(arm, "renamed-stage", "compiler_retry", 1, 1)


@pytest.mark.parametrize("input_tokens,output_tokens,next_input,next_output,match", [
    (63999, 1, 2, 1, "input-token"), (1, 5999, 1, 2, "output-token")])
def test_aggregate_token_caps(ledger, input_tokens, output_tokens, next_input, next_output, match):
    value, _, candidate = ledger
    arm = start(value, candidate)
    complete_call(value, arm, input_tokens=input_tokens, output_tokens=output_tokens)
    with pytest.raises(BudgetError, match=match):
        value.reserve_model_call(arm, "extra", "fallback", next_input, next_output)


def test_restart_deadline_and_recovery_reserve(ledger):
    value, clock, candidate = ledger
    arm = start(value, candidate)
    clock.now += 360
    reopened = BudgetLedger(value.path, clock=clock)
    assert reopened.guard_effect(arm)["remaining_seconds"] == 60
    assert reopened.guard_effect(arm)["effect_authority_granted"] is False
    with pytest.raises(BudgetError, match="insufficient"):
        reopened.guard_effect(arm, 90)
    clock.now += 0.01
    with pytest.raises(BudgetError, match="insufficient"):
        reopened.guard_effect(arm)
    with pytest.raises(BudgetError, match="at least 60"):
        reopened.guard_effect(arm, 59)
    clock.now += 60
    with pytest.raises(BudgetError, match="wall-clock"):
        reopened.reserve_model_call(arm, "late", "agent", 1, 1)
    result = reopened.finish_arm(arm, "completed", elapsed_ms=1)
    assert result["status"] == "timeout" and result["elapsed_ms"] > 420000


@pytest.mark.parametrize("terminal", ["timeout", "cancelled"])
def test_explicit_infrastructure_termination_stops_the_whole_batch(ledger, terminal):
    value, _, candidate = ledger
    arm = start(value, candidate)
    value.finish_arm(arm, terminal)
    assert value.snapshot("study")["study"]["status"] == "halted"
    with pytest.raises(BudgetError, match="halted"):
        start(value, candidate, arm="B")


@pytest.mark.parametrize("terminal", ["completed", "failed", "incomplete"])
def test_known_business_terminal_does_not_stop_the_next_assigned_arm(ledger, terminal):
    value, _, candidate = ledger
    arm = start(value, candidate)
    value.finish_arm(arm, terminal)
    assert value.snapshot("study")["study"]["status"] == "active"
    start(value, candidate, arm="B")


def test_public_arm_check_refuses_pending_halted_and_expired_work(ledger):
    value, clock, candidate = ledger
    arm = start(value, candidate)
    assert value.check_arm(arm) == {"arm_id": arm, "deadline": 1420,
                                    "remaining_seconds": 420, "authority_granted": False}
    value.reserve_model_call(arm, "r", "agent", 10, 10)
    with pytest.raises(BudgetError, match="unsettled"):
        value.check_arm(arm)
    value.settle_call(arm, "r", 5, 5)
    clock.now = 1420
    with pytest.raises(BudgetError, match="wall-clock"):
        value.check_arm(arm)
    value.finish_arm(arm, "timeout")
    with pytest.raises(BudgetError, match="halted"):
        value.check_arm(arm)


def test_late_model_settlement_retains_usage_and_stops_batch(ledger):
    value, clock, candidate = ledger
    arm = start(value, candidate)
    value.reserve_model_call(arm, "r", "agent", 20, 20)
    clock.now += 421
    result = value.settle_call(arm, "r", 5, 5)
    assert result["status"] == "budget_exceeded"
    assert result["charged_input"] == 20
    with pytest.raises(BudgetError, match="halted"):
        start(value, candidate, arm="B")


def test_clock_rollback_cannot_reset_remaining_time(ledger):
    value, clock, candidate = ledger
    arm = start(value, candidate)
    clock.now -= 1
    with pytest.raises(BudgetError, match="clock rollback"):
        value.reserve_model_call(arm, "r", "agent", 1, 1)
    assert value.snapshot("study")["clock_rollback"]
    assert value.finish_arm(arm, "completed")["status"] == "measurement_invalid"


def test_rejected_deadline_check_persists_clock_high_water_mark(ledger):
    value, clock, candidate = ledger
    arm = start(value, candidate)
    clock.now += 421
    with pytest.raises(BudgetError, match="wall-clock"):
        value.guard_effect(arm)
    clock.now -= 100
    reopened = BudgetLedger(value.path, clock=clock)
    with pytest.raises(BudgetError, match="clock rollback"):
        reopened.reserve_model_call(arm, "time-reset", "agent", 1, 1)


def test_settlement_after_terminal_never_reopens_terminal_or_reclassifies_it(ledger):
    value, _, candidate = ledger
    arm = start(value, candidate)
    value.reserve_model_call(arm, "r", "agent", 10, 20)
    assert value.finish_arm(arm, "completed")["status"] == "outcome_unknown"
    value.settle_call(arm, "r", -1, 1)
    assert value.snapshot("study")["arms"][0]["status"] == "outcome_unknown"


def test_pause_is_irreversible_and_completion_is_accounting_only(ledger):
    value, _, candidate = ledger
    arm = start(value, candidate)
    value.reserve_model_call(arm, "r", "agent", 10, 10)
    value.pause_study("study", "user pause")
    assert value.settle_call(arm, "r", 5, 5)["status"] == "settled"
    value.finish_arm(arm, "cancelled")
    assert value.register_study("study", "protocol")["status"] == "halted"
    with pytest.raises(BudgetError, match="halted"):
        value.register_candidate("study", "v2")
    with pytest.raises(BudgetError, match="halted"):
        start(value, candidate, arm="B")


def test_candidate_and_case_freezes(ledger):
    value, _, candidate = ledger
    arm = start(value, candidate)
    with pytest.raises(BudgetError, match="active arm"):
        value.register_candidate("study", "v2")
    value.finish_arm(arm, "completed")
    with pytest.raises(BudgetError, match="context changed"):
        start(value, candidate, arm="B", context="altered")
    second = value.register_candidate("study", "v2")
    with pytest.raises(BudgetError, match="superseded"):
        start(value, candidate, arm="B")
    with pytest.raises(BudgetError, match="cases cannot change"):
        start(value, second, case="new")
    with pytest.raises(BudgetError, match="context changed"):
        start(value, second, context="altered")
    arm = start(value, second)
    value.finish_arm(arm, "completed")
    with pytest.raises(BudgetError, match="version budget"):
        value.register_candidate("study", "v3")
    with pytest.raises(BudgetError, match="last development digest"):
        value.register_candidate("study", "v1", "confirmation")
    confirmed = value.register_candidate("study", "v2", "confirmation")
    assert value.register_candidate("study", "v2", "confirmation") == confirmed
    with pytest.raises(BudgetError, match="overlaps"):
        start(value, confirmed)
    arm = start(value, confirmed, case="unseen", repetition=3)
    value.finish_arm(arm, "completed")
    with pytest.raises(BudgetError, match="confirmation candidate"):
        value.register_candidate("study", "v3", "confirmation")
    with pytest.raises(BudgetError, match="repetition"):
        start(value, confirmed, case="unseen", repetition=4)


def test_concurrent_independent_connections_claim_one_request(ledger):
    value, clock, candidate = ledger
    arm = start(value, candidate)
    barrier = threading.Barrier(2)

    def reserve(request):
        independent = BudgetLedger(value.path, clock=clock)
        barrier.wait()
        try:
            return independent.reserve_model_call(arm, request, "agent", 10, 20)["status"]
        except BudgetError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(reserve, ["physical-a", "physical-b"]))
    assert sorted(results) == ["rejected", "reserved"]
    assert value.snapshot("study")["usage"]["model_requests"] == 1


def test_offline_accounting_is_separate_bounded_and_persistent(ledger):
    value, clock, candidate = ledger
    for i in range(2):
        result = value.reserve_offline_call("study", candidate, "case", str(i), 24000, 3000)
        assert result["deadline"] == clock.now + 180
        value.settle_offline_call(candidate, "case", str(i), 100, 200)
    with pytest.raises(BudgetError, match="offline case"):
        value.reserve_offline_call("study", candidate, "case", "third", 1, 1)
    with pytest.raises(BudgetError, match="per-request"):
        value.reserve_offline_call("study", candidate, "another", "first", 24001, 1)
    snapshot = value.snapshot("study")
    assert snapshot["usage"]["model_requests"] == 0
    assert snapshot["offline_usage"]["model_requests"] == 2
    assert snapshot["offline_usage"]["known_output_tokens"] == 400
    assert snapshot["offline_usage"]["usage_complete"]


def test_full_protocol_caps_accumulate_across_versions_and_phases(ledger):
    value, _, candidate = ledger
    for version in range(2):
        if version:
            candidate = value.register_candidate("study", "v2")
        for case in range(12):
            for arm_name in ("A", "B"):
                arm = start(value, candidate, case=f"dev-{case}", arm=arm_name)
                for request in range(10):
                    complete_call(value, arm, str(request), input_tokens=1, output_tokens=1)
                value.finish_arm(arm, "completed")
        with pytest.raises(BudgetError, match="case budget"):
            start(value, candidate, case="thirteenth")
    candidate = value.register_candidate("study", "v2", "confirmation")
    for case in range(12):
        for repetition in range(1, 4):
            for arm_name in ("A", "B"):
                arm = start(value, candidate, case=f"new-{case}", repetition=repetition, arm=arm_name)
                for request in range(10):
                    complete_call(value, arm, str(request), input_tokens=1, output_tokens=1)
                value.finish_arm(arm, "completed")
    snapshot = value.snapshot("study")
    assert snapshot["arm_count"] == CAPS["study_arms"] == 120
    assert snapshot["usage"]["model_requests"] == CAPS["study_model_requests"] == 1200
    with pytest.raises(BudgetError, match="case budget"):
        start(value, candidate, case="fresh-directory-cannot-reset")
