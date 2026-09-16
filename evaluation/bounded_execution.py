"""Measured call boundary for R0 scripted adapter checks, not a live harness.

Calls are claimed before invocation; errors/unknown usage retain their claim.
The controller never supplies Gold or completes an action for a backend after
it returns. An adapter must route *all* physical model requests through this
boundary. R0 tests use deterministic callbacks; no network transport is
provided here, and their timings/tokens are not 9B performance evidence.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

from evaluation.bounded_budget import BudgetError, BudgetLedger
from evaluation.bounded_pilot import case_input_digest, validate_agent_input
from network_runtime.contracts import sha256_json

STAGE_NAMES = {"native": "agent", "compile": "compiler", "runtime": "runtime",
               "revision": "revision", "fallback": "fallback"}


@dataclass(frozen=True)
class AgentContext:
    """Only common visible inputs plus intervention identity, never labels."""
    arm: str
    input_digest: str
    initial_state_digest: str
    isolation_id: str
    inputs: dict


def agent_context(case, *, arm, isolation_id):
    if arm not in {"control", "treatment"} or not isinstance(isolation_id, str) or not isolation_id:
        raise ValueError("explicit arm/isolation required")
    inputs = validate_agent_input(case["agent_input"])
    if case_input_digest(case) != case["input_digest"]:
        raise ValueError("input drift")
    return AgentContext(arm, case["input_digest"], sha256_json(case["provider_fixture"]),
                        isolation_id, copy.deepcopy(inputs))


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

    def invoke(self, request_id, stage, payload, *, input_tokens, max_output_tokens, invoke):
        if self.closed or stage not in STAGE_NAMES:
            raise BudgetError("no model call after close or with unmetered stage")
        self.ledger.reserve_model_call(self.arm_id, request_id, STAGE_NAMES[stage],
                                       input_tokens, max_output_tokens)
        began = self.clock()
        row = {"id": request_id, "stage": stage, "input_tokens": None,
               "output_tokens": None, "wall_ms": None, "request_digest": sha256_json(payload)}
        self.usage.append(row)
        try:
            result = invoke(copy.deepcopy(payload))
            if not isinstance(result, dict) or set(result) != {"response", "input_tokens", "output_tokens"}:
                raise ValueError("explicit response and usage envelope required")
        except BaseException:
            row["wall_ms"] = max(0.0, (self.clock() - began) * 1000)
            self.ledger.settle_call(self.arm_id, request_id,
                                    actual_input_tokens=None, actual_output_tokens=None)
            raise
        row.update(input_tokens=result["input_tokens"], output_tokens=result["output_tokens"],
                   wall_ms=max(0.0, (self.clock() - began) * 1000))
        settled = self.ledger.settle_call(self.arm_id, request_id,
                                          actual_input_tokens=result["input_tokens"],
                                          actual_output_tokens=result["output_tokens"])
        if settled["status"] != "settled":
            raise BudgetError("unknown or invalid physical usage: do not retry or enter fallback")
        return result["response"]

    def timed_stage(self, name, invoke):
        """Measure local qualification/verification work inside total arm time."""
        if self.closed or name not in {"qualification", "provider", "verification", "reconciliation", "delivery"}:
            raise BudgetError("unknown local stage or arm already closed")
        self.ledger.check_arm(self.arm_id)
        began = self.clock()
        try:
            result = invoke()
            self.ledger.check_arm(self.arm_id)
            return result
        finally:
            self.stages.append({"stage": name, "wall_ms": max(0.0, (self.clock() - began) * 1000)})

    def guard_effect(self, *, reserve_seconds=60):
        if self.closed:
            raise BudgetError("cannot dispatch after agent termination")
        return self.ledger.guard_effect(self.arm_id, reserve_seconds=reserve_seconds)

    def close(self, status="completed"):
        if self.closed:
            raise BudgetError("arm already finalized")
        self.closed = True
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
