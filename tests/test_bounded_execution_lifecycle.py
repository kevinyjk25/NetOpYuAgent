"""Synchronous scripted lifecycle checks; no models, transport, or task evidence."""
from concurrent.futures import Future
import threading

import pytest

from evaluation.bounded_budget import BudgetError, BudgetLedger, CAPS
from evaluation.bounded_execution import MeasuredCalls, QualificationRejected


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def measured(tmp_path):
    clock = Clock()
    ledger = BudgetLedger(tmp_path / "lifecycle.sqlite", clock=clock)
    ledger.register_study("lifecycle", "frozen-protocol")
    candidate = ledger.register_candidate("lifecycle", "frozen-code")
    arm = ledger.start_arm("lifecycle", candidate, "case", 1, "B", "common-input")
    return MeasuredCalls(ledger, arm, clock=clock), ledger, clock


def envelope(response="scripted"):
    return {"response": response, "input_tokens": 3, "output_tokens": 2}


def model_call(calls, callback, request="request", stage="native"):
    return calls.invoke(request, stage, {"prompt": "fixture"}, input_tokens=8,
                        max_output_tokens=4, invoke=callback)


@pytest.mark.parametrize("stop", ["pause", "close", "external_finish"])
def test_known_model_usage_after_stop_is_accounting_not_delivery(measured, stop):
    calls, ledger, _ = measured

    def response(_):
        if stop == "pause":
            ledger.pause_study("lifecycle", "operator stopped")
        elif stop == "close":
            assert calls.close()["ledger"]["status"] == "outcome_unknown"
        else:
            assert ledger.finish_arm(calls.arm_id, "completed")["status"] == "outcome_unknown"
        return envelope()

    with pytest.raises(BudgetError):
        model_call(calls, response)
    recorded = ledger.inspect_arm(calls.arm_id)
    assert recorded["calls"][0]["status"] == "settled"
    assert recorded["calls"][0]["actual_input"] == 3
    assert recorded["study_status"] == "halted"
    assert calls.usage[0]["outcome"] == "outcome_unknown"
    invoked = []
    with pytest.raises(BudgetError):
        model_call(calls, lambda _: invoked.append(True), request="fallback", stage="fallback")
    assert invoked == []


@pytest.mark.parametrize("stop", ["pause", "close"])
def test_local_result_after_stop_is_not_delivered(measured, stop):
    calls, ledger, _ = measured

    def provider():
        if stop == "close":
            assert calls.close()["ledger"]["status"] == "outcome_unknown"
        else:
            ledger.pause_study("lifecycle", "operator stopped")
        return "must not be delivered"

    with pytest.raises(BudgetError):
        calls.timed_stage("provider", provider)
    assert calls.stages[0]["outcome"] == "outcome_unknown"
    assert calls.stages[0]["error_type"] == "BudgetError"
    assert ledger.inspect_arm(calls.arm_id)["study_status"] == "halted"


@pytest.mark.parametrize("stage", ["provider", "verification", "reconciliation", "delivery", "qualification"])
@pytest.mark.parametrize("error_type", [ValueError, TimeoutError])
def test_unknown_local_error_halts_and_remains_in_close_receipt(measured, stage, error_type):
    calls, ledger, clock = measured

    def failure():
        clock.now += 2
        raise error_type("unclassified fixture failure")

    with pytest.raises(error_type):
        calls.timed_stage(stage, failure)
    assert ledger.inspect_arm(calls.arm_id)["study_status"] == "halted"
    invoked = []
    with pytest.raises(BudgetError):
        model_call(calls, lambda _: invoked.append(True))
    assert invoked == []
    report = calls.close()
    assert report["ledger"]["status"] == "outcome_unknown"
    assert report["local_stages"] == [{"stage": stage, "wall_ms": 2000,
                                        "outcome": "outcome_unknown", "error_type": error_type.__name__}]
    assert report["model_usage"] == []


def test_only_explicit_qualification_rejection_is_recoverable(measured):
    calls, ledger, _ = measured

    def rejected():
        raise QualificationRejected("host schema validator rejected proposal")

    with pytest.raises(QualificationRejected):
        calls.timed_stage("qualification", rejected)
    assert calls.stages[0]["outcome"] == "rejected"
    assert ledger.inspect_arm(calls.arm_id)["study_status"] == "active"
    assert model_call(calls, lambda _: envelope(), stage="fallback") == "scripted"
    assert calls.close()["ledger"]["status"] == "completed"


def test_qualification_exception_in_provider_is_not_safe_rejection(measured):
    calls, ledger, _ = measured

    def wrong_stage():
        raise QualificationRejected("not a qualification result")

    with pytest.raises(QualificationRejected):
        calls.timed_stage("provider", wrong_stage)
    assert calls.stages[0]["outcome"] == "outcome_unknown"
    assert ledger.inspect_arm(calls.arm_id)["study_status"] == "halted"


@pytest.mark.parametrize("api", ["model", "local"])
@pytest.mark.parametrize("callable_object", [False, True])
def test_async_callback_is_rejected_before_execution_or_reservation(measured, api, callable_object):
    calls, ledger, _ = measured
    invoked = []

    async def asynchronous(*args):
        invoked.append(True)
        return envelope()

    class AsyncCallable:
        async def __call__(self, *args):
            return await asynchronous(*args)

    callback = AsyncCallable() if callable_object else asynchronous
    with pytest.raises(TypeError, match="synchronous callback"):
        if api == "model":
            model_call(calls, callback)
        else:
            calls.timed_stage("provider", callback)
    assert invoked == []
    assert ledger.inspect_arm(calls.arm_id)["calls"] == []
    assert calls.stages == []
    assert calls.close()["ledger"]["status"] == "completed"


@pytest.mark.parametrize("api", ["model", "local", "model_nested_response"])
@pytest.mark.parametrize("kind", ["coroutine", "future", "async_generator"])
def test_deferred_return_is_unknown_and_never_awaited(measured, api, kind):
    calls, ledger, _ = measured
    invoked = []

    async def deferred():
        invoked.append(True)
        return "must not execute"

    async def stream():
        invoked.append(True)
        yield "must not execute"

    value = deferred() if kind == "coroutine" else Future() if kind == "future" else stream()
    with pytest.raises(TypeError, match="deferred execution"):
        if api == "local":
            calls.timed_stage("verification", lambda: value)
        elif api == "model_nested_response":
            model_call(calls, lambda _: envelope({"nested": [value]}))
        else:
            model_call(calls, lambda _: value)
    assert invoked == []
    recorded = ledger.inspect_arm(calls.arm_id)
    assert recorded["study_status"] == "halted"
    if api != "local":
        assert recorded["calls"][0]["status"] == "unknown"
        assert recorded["calls"][0]["charged_input"] == 8
    assert calls.close()["ledger"]["status"] == "outcome_unknown"


@pytest.mark.parametrize("stop", ["pause", "deadline", "close"])
def test_reserved_call_rechecks_lifecycle_before_physical_dispatch(measured, monkeypatch, stop):
    calls, ledger, clock = measured
    original = ledger.reserve_model_call

    def reserve_then_stop(*args, **kwargs):
        result = original(*args, **kwargs)
        if stop == "pause":
            ledger.pause_study("lifecycle", "stopped after reservation")
        elif stop == "deadline":
            clock.now += CAPS["arm_seconds"]
        else:
            calls.close()
        return result

    monkeypatch.setattr(ledger, "reserve_model_call", reserve_then_stop)
    invoked = []
    with pytest.raises(BudgetError):
        model_call(calls, lambda _: invoked.append(True))
    assert invoked == []
    assert calls.usage[0]["physical_call_attempted"] is False
    assert ledger.inspect_arm(calls.arm_id)["calls"][0]["status"] != "reserved"
    assert ledger.inspect_arm(calls.arm_id)["study_status"] == "halted"


def test_close_during_other_thread_local_callback_withholds_late_result(measured):
    calls, ledger, _ = measured
    entered, release = threading.Event(), threading.Event()
    delivered, errors = [], []

    def provider():
        entered.set()
        assert release.wait(5)
        return "late result"

    def worker():
        try:
            delivered.append(calls.timed_stage("provider", provider))
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        assert entered.wait(5)
        report = calls.close()
        assert report["ledger"]["status"] == "outcome_unknown"
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert delivered == []
    assert len(errors) == 1 and isinstance(errors[0], BudgetError)
    assert ledger.inspect_arm(calls.arm_id)["study_status"] == "halted"
    assert calls.stages[0]["outcome"] == "outcome_unknown"


def test_rebuilt_wrapper_cannot_finalize_halted_local_work_as_completed(measured):
    calls, ledger, clock = measured

    def failure():
        raise RuntimeError("provider state unknown")

    with pytest.raises(RuntimeError):
        calls.timed_stage("provider", failure)
    rebuilt = MeasuredCalls(ledger, calls.arm_id, clock=clock)
    assert rebuilt.close()["ledger"]["status"] == "outcome_unknown"


def test_synchronous_positive_control_keeps_costs_and_classifies_success(measured):
    calls, ledger, clock = measured

    def response(payload):
        assert payload == {"prompt": "fixture"}
        clock.now += 2
        return envelope({"answer": "scripted"})

    assert model_call(calls, response) == {"answer": "scripted"}
    assert calls.timed_stage("verification", lambda: {"fixture_check": True}) == {"fixture_check": True}
    report = calls.close()
    assert report["ledger"]["status"] == "completed"
    assert report["wall_ms"] == 2000
    assert report["model_usage"][0]["input_tokens"] == 3
    assert report["request_receipts"][0]["outcome"] == "completed"
    assert report["local_stages"][0]["outcome"] == "completed"
    assert ledger.inspect_arm(calls.arm_id)["study_status"] == "active"
    assert report["liveAdapterReady"] is False and report["realModelCalls"] is None
