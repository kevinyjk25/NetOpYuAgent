"""
a2a/schemas.py
--------------
Pydantic models for A2A protocol objects:
  Task, TaskStatus, Message, Part, Artifact, and all event types.

Keeps the rest of the codebase type-safe and schema-validated.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TaskState(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class PartKind(str, Enum):
    TEXT = "text"
    FILE = "file"
    DATA = "data"


class EventKind(str, Enum):
    TASK = "task"
    TASK_STATUS_UPDATE = "taskStatusUpdate"
    TASK_ARTIFACT_UPDATE = "taskArtifactUpdate"
    MESSAGE = "message"


# ---------------------------------------------------------------------------
# Message / Part primitives
# ---------------------------------------------------------------------------

class TextPart(BaseModel):
    kind: Literal["text"] = "text"
    text: str


class FilePart(BaseModel):
    kind: Literal["file"] = "file"
    name: str
    mime_type: str
    data: str  # base64-encoded


class DataPart(BaseModel):
    kind: Literal["data"] = "data"
    data: dict[str, Any]


Part = Union[TextPart, FilePart, DataPart]


class Message(BaseModel):
    kind: Literal["message"] = "message"
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: Literal["user", "assistant"] = "user"
    parts: list[Part]
    metadata: Optional[dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Task & status
# ---------------------------------------------------------------------------

class TaskStatus(BaseModel):
    state: TaskState
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    message: Optional[str] = None


class Task(BaseModel):
    id: str
    context_id: Optional[str] = None
    status: TaskStatus
    artifacts: list["Artifact"] = Field(default_factory=list)
    metadata: Optional[dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Artifact
# ---------------------------------------------------------------------------

class Artifact(BaseModel):
    artifact_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    parts: list[Part]
    metadata: Optional[dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Events emitted to the EventQueue
# ---------------------------------------------------------------------------

class TaskStatusUpdateEvent(BaseModel):
    kind: Literal["taskStatusUpdate"] = "taskStatusUpdate"
    task_id: str
    context_id: str
    status: TaskStatus


class TaskArtifactUpdateEvent(BaseModel):
    kind: Literal["taskArtifactUpdate"] = "taskArtifactUpdate"
    task_id: str
    context_id: str
    artifact: Artifact


class MessageEvent(BaseModel):
    """Final message – sending this will terminate the EventConsumer."""
    kind: Literal["message"] = "message"
    task_id: str
    context_id: str
    message: Message


A2AEvent = Union[TaskStatusUpdateEvent, TaskArtifactUpdateEvent, MessageEvent]


# ---------------------------------------------------------------------------
# JSON-RPC envelope helpers
# ---------------------------------------------------------------------------

class JsonRpcRequest(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    method: str
    params: dict[str, Any]
    id: Union[int, str, None] = None


class JsonRpcResponse(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    result: Optional[Any] = None
    error: Optional[dict[str, Any]] = None
    id: Union[int, str, None] = None


# ---------------------------------------------------------------------------
# Push-notification webhook payload
# ---------------------------------------------------------------------------

class PushNotificationPayload(BaseModel):
    task_id: str
    context_id: str
    state: TaskState
    artifact: Optional[Artifact] = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
