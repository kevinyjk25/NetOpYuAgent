"""
hitl/schemas.py
---------------
All Pydantic models for the Human-in-the-Loop module.

Covers:
  - HitlPayload       : what the agent sends to the operator for review
  - HitlDecision      : what the operator sends back
  - HitlInterruptEvent: emitted into the A2A EventQueue
  - HitlAuditRecord   : written to persistent audit store
  - HitlState         : typed LangGraph state dict
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TriggerKind(str, Enum):
    SEVERITY          = "severity"
    LOW_CONFIDENCE    = "low_confidence"
    DESTRUCTIVE       = "destructive_action"
    AMBIGUOUS_INTENT  = "ambiguous_intent"
    DESTRUCTIVE_OP    = "destructive_op"


class DecisionKind(str, Enum):
    APPROVE   = "approve"
    REJECT    = "reject"
    EDIT      = "edit"       # approve with modified params
    ESCALATE  = "escalate"
    TIMEOUT   = "timeout"    # set automatically by timeout watchdog


class InterruptState(str, Enum):
    PENDING   = "pending"
    RESOLVED  = "resolved"
    TIMED_OUT = "timed_out"
    ESCALATED = "escalated"


class RiskLevel(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Proposed action (what the agent wants to do)
# ---------------------------------------------------------------------------

class ProposedAction(BaseModel):
    """Describes the action the agent intends to execute."""
    action_type: str                       # e.g. "restart_service", "rollback_deploy"
    target: str                            # e.g. "payments-service / prod"
    parameters: dict[str, Any] = Field(default_factory=dict)
    estimated_impact: Optional[str] = None
    reversible: bool = True
    risk_level: RiskLevel = RiskLevel.LOW


# ---------------------------------------------------------------------------
# HITL Payload  (agent → operator)
# ---------------------------------------------------------------------------

class HitlPayload(BaseModel):
    """
    Everything the operator needs to make an informed decision.
    Stored in the LangGraph checkpoint and sent via every notification channel.
    """
    interrupt_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    thread_id: str                          # LangGraph thread / conversation ID
    context_id: str                         # A2A context_id
    task_id: str                            # A2A task_id

    trigger_kind: TriggerKind
    risk_level: RiskLevel

    # What happened so far
    user_query: str
    intent_summary: str                     # one-sentence LLM-generated summary
    confidence_score: float                 # 0.0 – 1.0

    # What the agent proposes to do
    proposed_action: ProposedAction

    # Supporting evidence
    context_snapshot: dict[str, Any] = Field(default_factory=dict)
    recent_alerts: list[dict[str, Any]] = Field(default_factory=list)

    # SLA
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    sla_seconds: int = 900                  # 15 min default; override per trigger

    # Status (updated in-place by the decision router)
    status: InterruptState = InterruptState.PENDING
    resolved_at: Optional[str] = None
    resolved_by: Optional[str] = None      # operator ID or "timeout_watchdog"


# ---------------------------------------------------------------------------
# HITL Decision  (operator → system)
# ---------------------------------------------------------------------------

class HitlDecision(BaseModel):
    """Operator response, POSTed to /hitl/decisions/{interrupt_id}."""
    interrupt_id: str
    thread_id: str
    decision: DecisionKind
    operator_id: str
    comment: Optional[str] = None

    # For DecisionKind.EDIT — partial override of ProposedAction.parameters
    parameter_patch: Optional[dict[str, Any]] = None

    # For DecisionKind.ESCALATE
    escalation_target: Optional[str] = None   # e.g. "on-call-sre"

    decided_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# HITL Interrupt event  (emitted into A2A EventQueue as a special artifact)
# ---------------------------------------------------------------------------

class HitlInterruptEvent(BaseModel):
    """
    Surfaced to the A2A caller as a TaskArtifactUpdateEvent with
    artifact.name == "hitl_interrupt". The caller can render a review UI.
    """
    kind: Literal["hitl_interrupt"] = "hitl_interrupt"
    interrupt_id: str
    thread_id: str
    trigger_kind: TriggerKind
    risk_level: RiskLevel
    summary: str
    proposed_action: ProposedAction
    review_url: Optional[str] = None       # deep link into web dashboard


# ---------------------------------------------------------------------------
# Audit record  (written to PostgreSQL / append-only log)
# ---------------------------------------------------------------------------

class AuditEventKind(str, Enum):
    INTERRUPT_FIRED   = "interrupt_fired"
    NOTIFY_SENT       = "notify_sent"
    DECISION_RECEIVED = "decision_received"
    GRAPH_RESUMED     = "graph_resumed"
    GRAPH_ABORTED     = "graph_aborted"
    TIMEOUT_TRIGGERED = "timeout_triggered"
    ESCALATION_SENT   = "escalation_sent"


class HitlAuditRecord(BaseModel):
    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    interrupt_id: str
    thread_id: str
    event_kind: AuditEventKind
    actor: str                              # agent | operator_id | timeout_watchdog
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# LangGraph state dict
# ---------------------------------------------------------------------------

class HitlState(BaseModel):
    """
    Typed representation of the LangGraph StateGraph state.
    All fields must be serialisable (no non-JSON types).
    """
    # Inputs
    query: str = ""
    thread_id: str = ""
    context_id: str = ""
    task_id: str = ""
    user_metadata: dict[str, Any] = Field(default_factory=dict)

    # Intent classification
    intent_type: str = ""
    intent_confidence: float = 1.0
    intent_summary: str = ""
    intent_candidates: list[dict[str, Any]] = Field(default_factory=list)

    # Risk assessment
    risk_level: RiskLevel = RiskLevel.LOW
    risk_reasons: list[str] = Field(default_factory=list)
    is_destructive: bool = False

    # Planner output
    proposed_action: Optional[dict[str, Any]] = None   # ProposedAction.model_dump()
    plan_steps: list[str] = Field(default_factory=list)

    # HITL
    hitl_required: bool = False
    hitl_payload: Optional[dict[str, Any]] = None      # HitlPayload.model_dump()
    hitl_decision: Optional[dict[str, Any]] = None     # HitlDecision.model_dump()

    # Execution results
    execution_results: list[dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None

    # Streaming / A2A
    emitted_chunks: list[dict[str, Any]] = Field(default_factory=list)
