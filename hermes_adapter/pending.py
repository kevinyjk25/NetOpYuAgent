"""Process-local, fail-closed approval bindings for the Hermes adapter."""

from __future__ import annotations

import hmac
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _expired(expires_at: str) -> bool:
    try:
        parsed = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return True
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed <= datetime.now(timezone.utc)


@dataclass(frozen=True)
class PendingPlan:
    plan_id: str
    plan_hash: str
    tool_name: str
    profile: str
    expires_at: str
    plan: dict[str, Any]
    execution_nonce: str = field(repr=False)

    def public(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "plan_hash": self.plan_hash,
            "tool_name": self.tool_name,
            "profile": self.profile,
            "expires_at": self.expires_at,
        }


@dataclass(frozen=True)
class PendingRemoteApproval:
    continuation_id: str
    peer: str
    plan_hash: str
    request: dict[str, Any]
    interrupt_id: str
    approval: dict[str, Any]

    def public(self) -> dict[str, Any]:
        return {
            "continuation_id": self.continuation_id,
            "peer": self.peer,
            "plan_hash": self.plan_hash,
            "approval": self.approval,
        }


class PendingActions:
    """Keep bearer nonces outside model-visible results and filesystem state.

    Restarting Hermes discards every pending approval.  This deliberately
    trades recovery convenience for a safe failure mode: the operator must ask
    the model to prepare a fresh plan after a restart.
    """

    def __init__(self) -> None:
        self._plans: dict[str, PendingPlan] = {}
        self._remote: dict[str, PendingRemoteApproval] = {}
        self._lock = threading.RLock()

    def add_plan(self, prepared: dict[str, Any]) -> PendingPlan:
        plan = prepared.get("plan")
        nonce = prepared.get("execution_nonce")
        if not isinstance(plan, dict) or not isinstance(nonce, str) or not nonce:
            raise ValueError("prepared Network Runtime response has no plan-bound nonce")
        item = PendingPlan(
            plan_id=str(plan.get("plan_id", "")),
            plan_hash=str(plan.get("plan_hash", "")),
            tool_name=str(plan.get("tool_name", "")),
            profile=str(plan.get("profile", "")),
            expires_at=str(plan.get("expires_at", "")),
            plan=dict(plan),
            execution_nonce=nonce,
        )
        if not all((item.plan_id, item.plan_hash, item.tool_name, item.profile, item.expires_at)):
            raise ValueError("prepared Network Runtime plan is incomplete")
        with self._lock:
            self._plans[item.plan_id] = item
        return item

    def claim_plan(self, plan_id: str, plan_hash: str) -> PendingPlan:
        with self._lock:
            item = self._plans.get(plan_id)
            if item is None:
                raise LookupError("pending Network L0 plan not found in this Hermes process")
            if not hmac.compare_digest(item.plan_hash, plan_hash):
                raise PermissionError("plan hash does not match the pending Network L0 plan")
            if _expired(item.expires_at):
                self._plans.pop(plan_id, None)
                raise TimeoutError("pending Network L0 plan expired; prepare a fresh plan")
            return self._plans.pop(plan_id)

    def add_remote(self, result: dict[str, Any], request: dict[str, Any]) -> PendingRemoteApproval:
        approval = result.get("approval")
        if not isinstance(approval, dict) or not str(approval.get("plan_hash", "")):
            raise ValueError("remote A2A approval has no structured Network L0 plan hash")
        interrupt_id = str(result.get("interrupt_id", ""))
        if not interrupt_id:
            raise ValueError("remote A2A approval has no interrupt id")
        item = PendingRemoteApproval(
            continuation_id=str(uuid.uuid4()),
            peer=str(result.get("peer", "")),
            plan_hash=str(approval["plan_hash"]),
            request=dict(request),
            interrupt_id=interrupt_id,
            approval=dict(approval),
        )
        with self._lock:
            self._remote[item.continuation_id] = item
        return item

    def claim_remote(self, continuation_id: str, plan_hash: str) -> PendingRemoteApproval:
        with self._lock:
            item = self._remote.get(continuation_id)
            if item is None:
                raise LookupError("pending remote Network L0 approval not found")
            if not hmac.compare_digest(item.plan_hash, plan_hash):
                raise PermissionError("remote plan hash does not match the pending approval")
            return self._remote.pop(continuation_id)

    def list_public(self) -> dict[str, Any]:
        with self._lock:
            expired_ids = [key for key, value in self._plans.items() if _expired(value.expires_at)]
            for key in expired_ids:
                self._plans.pop(key, None)
            return {
                "plans": [item.public() for item in self._plans.values()],
                "remote_approvals": [item.public() for item in self._remote.values()],
            }
