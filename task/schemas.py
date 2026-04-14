"""
task/schemas.py
---------------
All Pydantic models and enums for the Task module.
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

class TaskState(str, Enum):
    PENDING       = "pending"
    RUNNING       = "running"
    WAITING_HITL  = "waiting_hitl"
    WAITING_DEP   = "waiting_dep"      # waiting on dependency
    COMPLETED     = "completed"
    FAILED        = "failed"
    CANCELLED     = "cancelled"
    RETRYING      = "retrying"


class TaskPriority(int, Enum):
    CRITICAL = 1
    HIGH     = 2
    MEDIUM   = 3
    LOW      = 4


class TaskScope(str, Enum):
    INTRA  = "intra"   # single-agent, local
    INTER  = "inter"   # cross-agent via A2A


class AgentAssignment(BaseModel):
    agent_id:  str
    agent_url: str
    skill_id:  str


# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------

class TaskDefinition(BaseModel):
    task_id:      str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_id:    Optional[str] = None          # set for subtasks
    session_id:   str
    context_id:   str                            # A2A context_id join key
    scope:        TaskScope = TaskScope.INTRA
    description:  str
    priority:     TaskPriority = TaskPriority.MEDIUM
    state:        TaskState    = TaskState.PENDING
    dependencies: list[str]    = Field(default_factory=list)   # task_ids
    assignment:   Optional[AgentAssignment] = None
    parameters:   dict[str, Any] = Field(default_factory=dict)
    result:       Optional[dict[str, Any]] = None
    error:        Optional[str] = None
    retry_count:  int  = 0
    max_retries:  int  = 3
    created_at:   str  = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    started_at:   Optional[str] = None
    completed_at: Optional[str] = None
    metadata:     dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class MultiRoundContext(BaseModel):
    """Carries state across multiple A2A turns in one session."""
    open_questions:      list[str] = Field(default_factory=list)
    deferred_task_ids:   list[str] = Field(default_factory=list)
    pending_hitl_ids:    list[str] = Field(default_factory=list)
    confirmed_facts:     list[str] = Field(default_factory=list)
    last_agent_response: Optional[str] = None


class SessionRecord(BaseModel):
    session_id:    str = Field(default_factory=lambda: str(uuid.uuid4()))
    context_id:    str                           # A2A context_id
    user_id:       Optional[str] = None
    turn_count:    int = 0
    memory_session_id: str = ""                  # join key for MemoryRouter
    multi_round:   MultiRoundContext = Field(default_factory=MultiRoundContext)
    active_task_ids: list[str] = Field(default_factory=list)
    created_at:    str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_active_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata:      dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

class TaskEventKind(str, Enum):
    CREATED    = "created"
    DISPATCHED = "dispatched"
    STARTED    = "started"
    COMPLETED  = "completed"
    FAILED     = "failed"
    CANCELLED  = "cancelled"
    RETRIED    = "retried"
    HITL_PAUSE = "hitl_pause"
    HITL_RESUME= "hitl_resume"


class TaskAuditRecord(BaseModel):
    audit_id:   str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id:    str
    session_id: str
    event_kind: TaskEventKind
    actor:      str
    payload:    dict[str, Any] = Field(default_factory=dict)
    timestamp:  str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Subtask assignment result (from A2A dispatcher)
# ---------------------------------------------------------------------------

class SubtaskAssignmentResult(BaseModel):
    task_id:    str
    agent_id:   str
    agent_url:  str
    a2a_task_id: str               # task_id returned by remote A2A agent
    dispatched_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
