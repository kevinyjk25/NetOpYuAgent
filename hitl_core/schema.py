"""
hitl_core.schema — Domain-neutral data models for human-in-the-loop pipelines.

Design goals:
  • Zero langchain / langgraph dependency.
  • All fields generic enough to fit IT-ops, LLM safety, content moderation,
    finance compliance, etc. — domain-specific extensions go in metadata dicts
    rather than hardcoded fields.
  • Pydantic v2 throughout. Every model is JSON-serialisable so checkpointing
    to Redis / SQLite / file is a one-line model_dump_json().

Public surface (re-exported from hitl_core.__init__):
  TriggerKind, DecisionKind, RiskLevel, InterruptState, AuditEventKind
  ChoiceOption, ClarificationField, ProposedAction
  HitlPayload, HitlDecision, HitlAuditRecord
  ResumeHandle, CheckpointEntry
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TriggerKind(str, Enum):
    """Why a HITL interrupt was raised. Drivers can extend at runtime by
    passing a string that doesn't match an enum member; the pipeline
    treats unknown kinds as opaque labels."""
    SEVERITY          = "severity"            # severity threshold exceeded
    LOW_CONFIDENCE    = "low_confidence"      # classifier uncertain
    DESTRUCTIVE       = "destructive_action"  # action mutates state
    AMBIGUOUS_INTENT  = "ambiguous_intent"    # multiple plausible plans
    DESTRUCTIVE_OP    = "destructive_op"      # alias kept for compat
    USER_CHOICE       = "user_choice"         # operator picks from N
    CLARIFICATION     = "clarification"       # operator answers Qs
    POLICY_VIOLATION  = "policy_violation"    # policy engine blocked
    # ── Async delegation (2026-05) ──────────────────────────────────────
    # Used for H2-style fire-and-forget HITLs (e.g. "push RADIUS check
    # request to ops queue, agent continues with assumed value").
    # See hitl_core/DESIGN.md §async HITL for the merge-back semantics.
    EXTERNAL_DELEGATION = "external_delegation"

    @classmethod
    def coerce(cls, value: Any) -> "TriggerKind":
        """Accept either enum, str, or unknown string (for forward-compat).
        Unknown strings raise ValueError so callers can decide to fall
        back to LOW_CONFIDENCE or POLICY_VIOLATION."""
        if isinstance(value, cls):
            return value
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Unknown trigger_kind: {value!r}")


class InterruptMode(str, Enum):
    """How the pipeline / caller should treat an interrupt.

    Added 2026-05 to support 3-mode HITL (sync追问 / 异步委托 / 同步高危).
    This is ORTHOGONAL to TriggerKind — same trigger can be raised in
    different modes by different skills (e.g. RADIUS check can be SYNC if
    operator is on-call, ASYNC if not).

    SYNC_BLOCKING
        Default. Pipeline awaits operator decision before continuing.
        Maps to existing `request_approval()` API.

    ASYNC_NONBLOCKING
        Pipeline does NOT await. Returns immediately with the
        caller-supplied `default_value`. Real decision (if any) arrives
        later via `on_resolved` callback; caller decides how to merge
        back into agent state (typically writes a `confirmed_fact` so
        subsequent turns see it). Used for H2: "push to external
        approval system, continue with optimistic default".

    MFA_BLOCKING
        Like SYNC_BLOCKING but requires a second-factor confirmation.
        NOT IMPLEMENTED (2026-05) — deferred to a separate sprint
        because the MFA backend choice (TOTP / WebAuthn / SMS) is a
        product decision, not a technical one.
    """
    SYNC_BLOCKING     = "sync_blocking"
    ASYNC_NONBLOCKING = "async_nonblocking"
    MFA_BLOCKING      = "mfa_blocking"   # reserved, not wired


class DecisionKind(str, Enum):
    """What the operator chose. Pipeline branches on this."""
    APPROVE   = "approve"     # proceed as proposed
    REJECT    = "reject"      # do not proceed
    EDIT      = "edit"        # approve with parameter overrides
    CHOOSE    = "choose"      # operator picked one of payload.choices
    ANSWER    = "answer"      # operator supplied missing info
    ESCALATE  = "escalate"    # bump to a higher reviewer
    TIMEOUT   = "timeout"     # set automatically by SLA watchdog


class RiskLevel(str, Enum):
    """Visual / sorting hint for the UI; not enforced semantically."""
    LOW       = "low"
    MEDIUM    = "medium"
    HIGH      = "high"
    CRITICAL  = "critical"


class InterruptState(str, Enum):
    """Lifecycle of a checkpoint entry."""
    PENDING   = "pending"     # awaiting operator decision
    RESOLVED  = "resolved"    # operator decided, callback ran
    EXPIRED   = "expired"     # SLA timeout, no decision
    CANCELLED = "cancelled"   # explicitly aborted
    # ── Async-only states (2026-05) ─────────────────────────────────────
    # ASYNC interrupts have additional substates because the caller has
    # already proceeded with a default; we still want UI visibility into
    # whether the ack ever arrived.
    ACKED     = "acked"       # async: external system confirmed receipt
    WORKING   = "working"     # async: external system processing


class AuditEventKind(str, Enum):
    """Granular audit log event types."""
    INTERRUPT_RAISED  = "interrupt_raised"
    DECISION_MADE     = "decision_made"
    GRAPH_RESUMED     = "graph_resumed"
    EXECUTION_DONE    = "execution_done"
    EXECUTION_FAILED  = "execution_failed"
    EXPIRED           = "expired"
    # ── Async HITL audit events (2026-05) ───────────────────────────────
    ASYNC_DELEGATED   = "async_delegated"   # H2 fired, agent proceeded
    ASYNC_RESOLVED    = "async_resolved"    # H2 ack arrived (may diverge)
    ASYNC_TIMEOUT     = "async_timeout"     # H2 SLA passed, no decision


# ---------------------------------------------------------------------------
# Reusable building blocks
# ---------------------------------------------------------------------------

class ChoiceOption(BaseModel):
    """One selectable item for trigger_kind == USER_CHOICE.

    Operator clicks one and the id flows back as decision.selected_choice_id.
    `metadata` is opaque — UI can render whatever the host injected.
    """
    id: str = Field(..., description="Stable id returned in the decision")
    label: str = Field(..., description="Short display name (one line)")
    description: Optional[str] = Field(None, description="Optional second-line detail")
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClarificationField(BaseModel):
    """One question for trigger_kind == CLARIFICATION.

    Operator types a free-text answer; it flows back as
    decision.clarification_answers[key].
    """
    key: str = Field(..., description="Field name returned in the decision")
    prompt: str = Field(..., description="Question shown to the operator")
    placeholder: Optional[str] = Field(None, description="Hint shown in input")
    required: bool = True


class ActionTypePrefix:
    """Canonical prefixes for ProposedAction.action_type (debt #10, 2026-05).

    action_type stays a free-form str (domain-neutral), but these constants +
    the ProposedAction builders below give callers ONE place to construct the
    common shapes so we stop hand-writing `"tool_call:" + name` at every site.
    The convention is `<prefix>:<name>`; callers may still pass raw strings.
    """
    TOOL_CALL  = "tool_call"     # tool_call:<tool_name>
    BATCH      = "batch"         # batch:<tool_name>
    DIAGNOSTIC = "diagnostic"    # diagnostic:<what>
    DELEGATE   = "delegate"      # delegate:<agent_id>  (Phase 2B)


class ProposedAction(BaseModel):
    """What the agent intends to do.

    Domain-neutral by design. action_type and target are free-form strings;
    parameters is an opaque dict whose keys may be exposed for editing via
    HitlPayload.editable_param_keys. Concrete domains layer their own
    typed models on top by validating after model_dump().

    Prefer the builders (tool_call / batch / diagnostic / delegate) over
    constructing action_type strings by hand — see ActionTypePrefix (debt #10).
    """
    action_type: str = Field(..., description="Free-form e.g. 'restart_service' or 'tool_call:edit_config'")
    target: str = Field(..., description="What the action affects, e.g. 'payments-svc' or 'ap-01'")
    parameters: dict[str, Any] = Field(default_factory=dict)
    estimated_impact: Optional[str] = None
    reversible: bool = True
    risk_level: RiskLevel = RiskLevel.LOW

    # ── Builders (debt #10) — one place to construct the common shapes ──
    @classmethod
    def tool_call(cls, tool_name: str, target: str, parameters: dict[str, Any] | None = None,
                  **kw) -> "ProposedAction":
        return cls(action_type=f"{ActionTypePrefix.TOOL_CALL}:{tool_name}",
                   target=target, parameters=parameters or {}, **kw)

    @classmethod
    def batch(cls, tool_name: str, target: str, parameters: dict[str, Any] | None = None,
              **kw) -> "ProposedAction":
        return cls(action_type=f"{ActionTypePrefix.BATCH}:{tool_name}",
                   target=target, parameters=parameters or {}, **kw)

    @classmethod
    def diagnostic(cls, what: str, target: str, parameters: dict[str, Any] | None = None,
                   **kw) -> "ProposedAction":
        return cls(action_type=f"{ActionTypePrefix.DIAGNOSTIC}:{what}",
                   target=target, parameters=parameters or {}, **kw)

    @classmethod
    def delegate(cls, agent_id: str, target: str, parameters: dict[str, Any] | None = None,
                 **kw) -> "ProposedAction":
        """Phase 2B: a delegation to a peer agent (target = subtask description)."""
        return cls(action_type=f"{ActionTypePrefix.DELEGATE}:{agent_id}",
                   target=target, parameters=parameters or {}, **kw)


# ---------------------------------------------------------------------------
# Interrupt payload — the "card" the operator sees
# ---------------------------------------------------------------------------

class HitlPayload(BaseModel):
    """Full description of an interrupt — sent to the UI, persisted in store.

    All multi-mode fields (choices / clarification_fields / editable_param_keys)
    are optional and unused for plain APPROVE/REJECT cards. The pipeline
    inspects trigger_kind to decide how to render.
    """
    interrupt_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    thread_id: str = ""               # opaque session/thread id (host-defined)
    context_id: str = ""              # alias for thread_id; some hosts split them
    task_id: Optional[str] = None     # outer task id if pipeline runs in one

    # Why
    trigger_kind: TriggerKind = TriggerKind.LOW_CONFIDENCE
    risk_level: RiskLevel = RiskLevel.MEDIUM
    # How the pipeline / caller treats this interrupt (added 2026-05).
    # See InterruptMode docstring for semantics. SYNC_BLOCKING preserves
    # legacy behaviour (every existing call site).
    interrupt_mode: InterruptMode = InterruptMode.SYNC_BLOCKING

    # What — request context
    user_query: str = ""
    intent_summary: str = ""
    confidence_score: float = 0.0

    # Proposed action (may be a placeholder for diagnostic-only payloads)
    proposed_action: ProposedAction = Field(
        default_factory=lambda: ProposedAction(
            action_type="diagnostic", target="-", parameters={},
        )
    )

    # ── Multi-mode extensions ────────────────────────────────────────
    choices: list[ChoiceOption] = Field(default_factory=list)
    clarification_fields: list[ClarificationField] = Field(default_factory=list)
    editable_param_keys: list[str] = Field(default_factory=list)

    # Supporting evidence (free-form so domains can stuff whatever)
    context_snapshot: dict[str, Any] = Field(default_factory=dict)
    recent_alerts: list[dict[str, Any]] = Field(default_factory=list)

    # Lifecycle
    state: InterruptState = InterruptState.PENDING
    sla_seconds: int = 600
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Operator's response
# ---------------------------------------------------------------------------

class HitlDecision(BaseModel):
    """Operator's response, posted back to the pipeline."""
    interrupt_id: str
    thread_id: str = ""
    decision: DecisionKind
    operator_id: str = "unknown"
    comment: Optional[str] = None

    # For DecisionKind.EDIT — keys must be a subset of
    # payload.editable_param_keys (validated by router)
    parameter_patch: Optional[dict[str, Any]] = None

    # For DecisionKind.CHOOSE — must match one of payload.choices[].id
    selected_choice_id: Optional[str] = None

    # For DecisionKind.ANSWER — keys must cover payload.clarification_fields
    # (where required=True)
    clarification_answers: Optional[dict[str, str]] = None

    # For DecisionKind.ESCALATE
    escalation_target: Optional[str] = None

    decided_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Resume handle — serialisable spec for callback re-invocation
# ---------------------------------------------------------------------------
# This is the key architectural primitive that makes Redis checkpointing
# possible: instead of pickling Python functions (which can't survive
# process restarts or cross-host deployment), we store a *spec* describing
# how to reconstruct the call. The host registers a small set of named
# resumers at startup; resuming an interrupt looks up the resumer by name
# and calls it with the spec's payload.

class ResumeHandle(BaseModel):
    """Serialisable description of how to resume an interrupted pipeline.

    `resumer_name` is a string the host registered at startup; the router
    looks it up at decision time. `state` carries any data the resumer
    needs, e.g. the original query, session_id, partial pipeline state.
    """
    resumer_name: str = Field(..., description="Host-registered handler name")
    state: dict[str, Any] = Field(default_factory=dict)
    # Optional pipeline checkpoint — for fully self-resumable pipelines
    pipeline_state: Optional[dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Checkpoint entry — what the store persists per interrupt
# ---------------------------------------------------------------------------

class CheckpointEntry(BaseModel):
    """A single record in the checkpoint store."""
    interrupt_id: str
    payload: HitlPayload
    resume_handle: ResumeHandle
    state: InterruptState = InterruptState.PENDING
    registered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    decided_at: Optional[datetime] = None
    decision: Optional[HitlDecision] = None


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

class HitlAuditRecord(BaseModel):
    """One audit log entry — append-only, JSON-serialisable."""
    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    interrupt_id: str
    thread_id: str = ""
    event_kind: AuditEventKind
    actor: str = "system"               # operator id, or "system" for auto events
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Batch approval — multiple independent HITLs grouped for one operator action
# ---------------------------------------------------------------------------
# Many real workflows need to ask the operator about several things at once
# (restart 5 APs; push the same config to 8 switches). Forcing those into a
# linear "approve A then approve B then ..." sequence wastes operator time
# and misses the natural batching opportunity. A Batch groups N independent
# interrupts so the UI can render one card with N rows; operators decide
# each row, then submit once.
#
# Critically: the constituent interrupts are *real* HitlPayload entries in
# the store. The Batch is just an index over them. This keeps single and
# batched workflows on the same plumbing — the router doesn't care whether
# a decision arrived alone or as part of a batch.

class BatchPolicy(str, Enum):
    """How partial decisions affect the overall batch outcome."""
    BEST_EFFORT     = "best_effort"      # each child decision independent;
                                          # pipeline gets all decisions, decides
                                          # what to do with each
    ALL_OR_NOTHING  = "all_or_nothing"   # any reject → whole batch fails;
                                          # pipeline aborts the work-flow


class BatchWaitMode(str, Enum):
    """When the pipeline considers the batch resolved."""
    ALL        = "all"          # wait for every child interrupt to be decided
    THRESHOLD  = "threshold"    # resume once N children reach a target state
                                # (reserved for future implementation)
    STREAMING  = "streaming"    # yield each decision as it arrives
                                # (reserved for future implementation)


class BatchState(str, Enum):
    """Lifecycle of a batch envelope."""
    PENDING        = "pending"          # at least one child still pending
    RESOLVED       = "resolved"         # all children decided / resolved
    PARTIAL        = "partial"          # some decided, some expired/cancelled
    EXPIRED        = "expired"          # batch SLA elapsed before resolution
    CANCELLED      = "cancelled"        # producer aborted the batch


class HitlBatch(BaseModel):
    """Envelope binding N interrupts as one operator action.

    The children themselves live in the store as ordinary CheckpointEntry
    rows; this object is the index. Storing batch_id back-pointers on
    each CheckpointEntry (HitlPayload.context_snapshot["batch_id"]) lets
    the router answer "is this decision part of a batch?" cheaply.
    """
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    thread_id: str = ""
    task_id: Optional[str] = None

    # Children — interrupt_ids in submission order. Order matters for UI
    # rendering (operators expect predictable left-to-right / top-to-bottom).
    interrupt_ids: list[str] = Field(default_factory=list)

    # Policy & wait mode set by the producer at creation time. Cannot
    # change after submission (would invite race conditions).
    policy: BatchPolicy = BatchPolicy.BEST_EFFORT
    wait_mode: BatchWaitMode = BatchWaitMode.ALL
    threshold_count: Optional[int] = None  # for WaitMode.THRESHOLD

    # Display
    title: str = ""                # short label e.g. "Restart 3 APs"
    description: str = ""          # multiline operator-facing summary

    # Lifecycle
    state: BatchState = BatchState.PENDING
    sla_seconds: int = 1800        # batch-level SLA; children inherit unless overridden
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None

    # Free-form host metadata (resumer name, custom UI hints, etc.)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchSubmission(BaseModel):
    """Operator's response when submitting a batch decision.

    The transport layer accepts this as the body of POST /hitl/batch/{id}.
    The router fans it out into individual HitlDecision rows internally;
    each decision flows through the same validation as an unbatched one.
    """
    batch_id: str
    operator_id: str = "unknown"
    comment: Optional[str] = None
    # Each entry is a partial HitlDecision — the router fills in
    # interrupt_id from the batch's child list and validates the
    # decision shape against each payload.
    decisions: list[HitlDecision] = Field(default_factory=list)


class BatchSnapshot(BaseModel):
    """Read-model the UI fetches to render a batch card. Comprises the
    envelope + all child payloads + any decisions already submitted.

    Returned by HitlRouter.load_batch() and the HTTP transport layer.
    Strictly read-only — operators submit decisions via BatchSubmission.
    """
    batch: HitlBatch
    children: list[CheckpointEntry] = Field(default_factory=list)
    decided_count: int = 0
    pending_count: int = 0