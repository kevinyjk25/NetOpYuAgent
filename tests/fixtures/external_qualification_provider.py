"""Repository fixture copied outside the repository for JSONL qualification tests."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


WIRE = "netopyu.io/provider-qualification-wire/v1"
STATE_PATH = Path(sys.argv[1]).resolve()
CONTRACT_PATH = Path(sys.argv[2]).resolve()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def digest(value):
    return "sha256:" + hashlib.sha256(canonical(value).encode()).hexdigest()


def read_state():
    if not STATE_PATH.exists():
        return {"current": "baseline", "operations": {}, "escalations": {}}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def write_state(value):
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(canonical(value), encoding="utf-8")
    temporary.replace(STATE_PATH)


def success(request_id, result):
    return {"apiVersion": WIRE, "requestId": request_id, "ok": True, "result": result}


def failure(request_id, code, message):
    return {
        "apiVersion": WIRE,
        "requestId": request_id,
        "ok": False,
        "error": {"code": code, "message": message},
    }


def handle(request):
    request_id = request.get("requestId")
    if request.get("apiVersion") != WIRE or not isinstance(request_id, str):
        return failure(str(request_id), "bad_request", "wire schema or request id mismatch")
    action = request.get("action")
    payload = request.get("payload")
    if not isinstance(payload, dict):
        return failure(request_id, "bad_request", "payload must be an object")
    state = read_state()
    if action == "describe":
        return success(request_id, json.loads(CONTRACT_PATH.read_text(encoding="utf-8")))
    if action == "reset":
        state = {"current": "baseline", "operations": {}, "escalations": {}}
        write_state(state)
        return success(request_id, state["current"])
    if action == "snapshot":
        return success(request_id, state["current"])
    if action == "apply":
        operation_id = payload.get("operation_id")
        if payload.get("sequence") != 1:
            return failure(request_id, "out_of_order", "only sequence one is valid")
        if payload.get("fault") == "timeout_before_send":
            return failure(request_id, "TimeoutError", "fault before effect dispatch")
        if operation_id in state["operations"]:
            return success(request_id, state["operations"][operation_id])
        operation = {"operation_id": operation_id, "state": "applied", "apply_attempts": 1}
        if payload.get("fault") == "unknown_terminal":
            operation["state"] = "unknown"
            state["operations"][operation_id] = operation
            write_state(state)
            return failure(request_id, "TimeoutError", "terminal state is unknown")
        state["current"] = digest(payload.get("arguments"))
        state["operations"][operation_id] = operation
        write_state(state)
        if payload.get("fault") == "after_commit_before_response":
            return failure(request_id, "TimeoutError", "effect committed before response loss")
        return success(request_id, operation)
    if action == "reconcile":
        operation_id = payload.get("operation_id")
        return success(request_id, state["operations"].get(operation_id, {
            "operation_id": operation_id, "state": "unknown", "apply_attempts": 0,
        }))
    if action == "compensate":
        operation_id = payload.get("operation_id")
        if operation_id not in state["operations"]:
            return failure(request_id, "unknown_operation", "operation is not durable")
        if payload.get("fault") == "compensation_failure":
            state["escalations"][operation_id] = "manual_intervention_required"
            write_state(state)
            return failure(request_id, "compensation_failed", "manual recovery required")
        state["current"] = "baseline"
        state["operations"][operation_id]["state"] = "compensated"
        write_state(state)
        return success(request_id, state["operations"][operation_id])
    if action == "escalation":
        return success(
            request_id,
            state["escalations"].get(payload.get("operation_id"), "none"),
        )
    return failure(request_id, "unknown_action", f"unsupported action {action!r}")


for raw_line in sys.stdin:
    try:
        request = json.loads(raw_line)
        response = handle(request)
    except Exception as error:
        response = failure("unknown", type(error).__name__, str(error))
    sys.stdout.write(canonical(response) + "\n")
    sys.stdout.flush()
