"""Measured call boundary for R0 scripted adapter checks, not a live harness.

Calls are claimed before invocation; errors/unknown usage retain their claim.
The controller never supplies Gold or completes an action for a backend after
it returns. An adapter must route *all* physical model requests through this
boundary. R0 tests use deterministic callbacks; no network transport is
provided here, and their timings/tokens are not 9B performance evidence.
"""
from __future__ import annotations

import copy
import inspect
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass

from evaluation.bounded_budget import BudgetError, BudgetLedger
from evaluation.bounded_pilot import case_initial_state_digest, case_input_digest, validate_agent_input
from network_runtime.contracts import sha256_json

STAGE_NAMES = {"native": "agent", "compile": "compiler", "runtime": "runtime",
               "revision": "revision", "fallback": "fallback"}


class QualificationRejected(ValueError):
    """A host validator's known rejection, not an unknown execution failure.

    Only the qualification stage recognizes this classification. Ordinary
    exceptions (including ValueError) never imply a safe rejection or retry.
    """


def _synchronous_callback(callback):
    target = getattr(callback, "__call__", None)
    if (not callable(callback) or inspect.iscoroutinefunction(callback)
            or inspect.isasyncgenfunction(callback) or inspect.isgeneratorfunction(callback)
            or inspect.iscoroutinefunction(target) or inspect.isasyncgenfunction(target)
            or inspect.isgeneratorfunction(target)):
        raise TypeError("measured synchronous API requires a synchronous callback")


def _synchronous_result(value, seen=None):
    """Reject deferred execution, including inside the model response envelope.

    This never waits for a Future or drives a coroutine. A Future may already
    be running: rejection means outcome unknown, not proof of cancellation.
    """
    if inspect.isawaitable(value) or isinstance(value, Future) or inspect.isasyncgen(value) or inspect.isgenerator(value):
        if inspect.iscoroutine(value) and inspect.getcoroutinestate(value) == inspect.CORO_CREATED:
            value.close()
        raise TypeError("measured synchronous API cannot return deferred execution")
    if isinstance(value, (dict, list, tuple, set, frozenset)):
        seen = set() if seen is None else seen
        if id(value) in seen:
            return
        seen.add(id(value))
        for item in (tuple(value.keys()) + tuple(value.values()) if isinstance(value, dict) else value):
            _synchronous_result(item, seen)


@dataclass(frozen=True)
class AgentContext:
    """Only common visible inputs plus intervention identity, never labels."""
    arm: str
    input_digest: str
    fixture_digest: str
    initial_state_digest: str
    isolation_id: str
    inputs: dict


def agent_context(case, *, arm, isolation_id):
    if arm not in {"control", "treatment"} or not isinstance(isolation_id, str) or not isolation_id:
        raise ValueError("explicit arm/isolation required")
    inputs = validate_agent_input(case["agent_input"])
    if case_input_digest(case) != case["input_digest"]:
        raise ValueError("input drift")
    return AgentContext(arm=arm, input_digest=case["input_digest"],
                        fixture_digest=sha256_json(case["provider_fixture"]),
                        initial_state_digest=case_initial_state_digest(case),
                        isolation_id=isolation_id, inputs=copy.deepcopy(inputs))


class MeasuredCalls:
    """One arm lifecycle; the authoritative persistent budget lives in SQLite.

    Input reservations must come from a trusted tokenizer/bound, not the LLM.
    The only supported R0 counting mode is fixture_exact. A live adapter must
    implement and verify its counting method before this permits real calls.
    """
    def __init__(self, ledger: BudgetLedger, arm_id: str, *, clock=time.time,
                 counting_mode="fixture_exact"):
        if counting_mode != "fixture_exact":
            raise ValueError("live tokenizer-attested metering is not implemented")
        self.ledger = ledger
        self.arm_id = arm_id
        self.clock = clock
        self.started = clock()
        self.closed = False
        self.usage = []
        self.stages = []
        self._lock = threading.RLock()
        self._inflight = 0
        self._unknown_outcome = False
        self._study_id = ledger.inspect_arm(arm_id)["arm"]["study_id"]

    def _halt(self, row, error):
        with self._lock:
            self._unknown_outcome = True
            row.update(outcome="outcome_unknown", error_type=type(error).__name__)
        self.ledger.pause_study(self._study_id, "measured callback outcome unknown")

    def _delivery_guard(self):
        # Call with _lock held. Usage settlement is accounting, not delivery
        # permission; another owner may have halted or finalized this arm.
        if self.closed:
            raise BudgetError("cannot deliver after arm close")
        self.ledger.check_arm(self.arm_id)

    def invoke(self, request_id, stage, payload, *, input_tokens, max_output_tokens, invoke):
        _synchronous_callback(invoke)
        payload_copy = copy.deepcopy(payload)
        row = {"id": request_id, "stage": stage, "input_tokens": None,
               "output_tokens": None, "wall_ms": None, "request_digest": sha256_json(payload_copy),
               "outcome": "in_progress", "error_type": None, "physical_call_attempted": False}
        with self._lock:
            if self.closed or stage not in STAGE_NAMES:
                raise BudgetError("no model call after close or with unmetered stage")
            self.ledger.reserve_model_call(self.arm_id, request_id, STAGE_NAMES[stage],
                                           input_tokens, max_output_tokens)
            began = self.clock()
            self.usage.append(row)
            self._inflight += 1
        try:
            try:
                with self._lock:
                    if self.closed:
                        raise BudgetError("cannot dispatch after arm close")
                    self.ledger.guard_reserved_call(self.arm_id, request_id)
                    row["physical_call_attempted"] = True
                result = invoke(payload_copy)
                _synchronous_result(result)
                if not isinstance(result, dict) or set(result) != {"response", "input_tokens", "output_tokens"}:
                    raise ValueError("explicit response and usage envelope required")
            except BaseException as error:
                self._halt(row, error)
                self.ledger.settle_call(self.arm_id, request_id,
                                        actual_input_tokens=None, actual_output_tokens=None)
                raise
            row.update(input_tokens=result["input_tokens"], output_tokens=result["output_tokens"])
            try:
                settled = self.ledger.settle_call(self.arm_id, request_id,
                                                  actual_input_tokens=result["input_tokens"],
                                                  actual_output_tokens=result["output_tokens"])
                if settled["status"] != "settled":
                    raise BudgetError("unknown or invalid physical usage: do not retry or enter fallback")
                with self._lock:
                    self._delivery_guard()
                    row["outcome"] = "completed"
                    return result["response"]
            except BaseException as error:
                self._halt(row, error)
                raise
        finally:
            with self._lock:
                row["wall_ms"] = max(0.0, (self.clock() - began) * 1000)
                self._inflight -= 1

    def timed_stage(self, name, invoke):
        """Measure local qualification/verification work inside total arm time."""
        _synchronous_callback(invoke)
        with self._lock:
            if self.closed or name not in {"qualification", "provider", "verification", "reconciliation", "delivery"}:
                raise BudgetError("unknown local stage or arm already closed")
            self.ledger.check_arm(self.arm_id)
            began = self.clock()
            row = {"stage": name, "wall_ms": None, "outcome": "in_progress", "error_type": None}
            self.stages.append(row)
            self._inflight += 1
        try:
            result = invoke()
            _synchronous_result(result)
            with self._lock:
                self._delivery_guard()
                row["outcome"] = "completed"
                return result
        except QualificationRejected as error:
            if name != "qualification":
                self._halt(row, error)
            else:
                try:
                    with self._lock:
                        self._delivery_guard()
                        row.update(outcome="rejected", error_type=type(error).__name__)
                except BaseException as guard_error:
                    self._halt(row, guard_error)
                    raise
            raise
        except BaseException as error:
            self._halt(row, error)
            raise
        finally:
            with self._lock:
                row["wall_ms"] = max(0.0, (self.clock() - began) * 1000)
                self._inflight -= 1

    def guard_effect(self, *, reserve_seconds=60):
        with self._lock:
            if self.closed:
                raise BudgetError("cannot dispatch after agent termination")
            return self.ledger.guard_effect(self.arm_id, reserve_seconds=reserve_seconds)

    def close(self, status="completed"):
        with self._lock:
            if self.closed:
                raise BudgetError("arm already finalized")
            self.closed = True
            if self._inflight or self._unknown_outcome:
                status = "outcome_unknown"
            elif status == "completed":
                try:
                    self.ledger.check_arm(self.arm_id)
                except BudgetError:
                    status = "outcome_unknown"
        elapsed = max(0.0, (self.clock() - self.started) * 1000)
        result = self.ledger.finish_arm(self.arm_id, status, elapsed_ms=elapsed)
        recorded = self.ledger.inspect_arm(self.arm_id)["calls"]
        reverse = {value: key for key, value in STAGE_NAMES.items()}
        # Costs survive reconstructing this Python wrapper or reopening its DB.
        return {"ledger": result, "wall_ms": result["elapsed_ms"], "model_usage": [
            {"id": row["request_id"], "stage": reverse[row["stage"]],
             "input_tokens": row["actual_input"], "output_tokens": row["actual_output"],
             "wall_ms": row["elapsed_ms"]} for row in recorded],
            "request_receipts": copy.deepcopy(self.usage), "local_stages": copy.deepcopy(self.stages),
            "measurement_kind": "scripted_meter_contract_not_live_DSH",
            "realModelCalls": None, "liveAdapterReady": False}
