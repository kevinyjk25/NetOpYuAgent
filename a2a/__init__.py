"""
a2a – IT Ops Agent A2A Module
==============================

Public surface::

    from a2a import (
        create_a2a_app,          # FastAPI sub-application factory
        ITOpsAgentExecutor,      # Concrete executor – subclass to plug in your LLM
        AgentExecutor,           # Abstract base
        EventQueue,              # Async event queue
        RequestContext,          # Wraps incoming request params
        InMemoryTaskStore,       # Task store
        get_agent_card,          # AgentCard builder
        # --- Schemas ---
        Task, TaskStatus, TaskState,
        Message, TextPart, DataPart, FilePart,
        Artifact,
        TaskStatusUpdateEvent, TaskArtifactUpdateEvent, MessageEvent,
        PushNotificationPayload,
        # --- Processors (extend the strategy chain) ---
        A2AEventProcessor,
        A2ATokenProcessor, A2ABatchTokenProcessor,
        A2AMessageProcessor, A2ANodeResultProcessor, A2ANodeStepProcessor,
    )
"""

from a2a.agent_card import get_agent_card
from a2a.agent_executor import (
    AgentExecutor,
    ITOpsAgentExecutor,
    A2AEventProcessor,
    A2ATokenProcessor,
    A2ABatchTokenProcessor,
    A2AMessageProcessor,
    A2ANodeResultProcessor,
    A2ANodeStepProcessor,
    DEFAULT_PROCESSORS,
)
from a2a.event_queue import EventQueue, RequestContext
from a2a.push_notifications import PushNotificationService
from a2a.request_handler import DefaultRequestHandler
from a2a.schemas import (
    Artifact,
    DataPart,
    FilePart,
    Message,
    MessageEvent,
    Part,
    PartKind,
    PushNotificationPayload,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)
from a2a.server import create_a2a_app
from a2a.task_store import InMemoryTaskStore, TaskStore

__all__ = [
    # App factory
    "create_a2a_app",
    # Executors
    "AgentExecutor",
    "ITOpsAgentExecutor",
    # Event processors
    "A2AEventProcessor",
    "A2ATokenProcessor",
    "A2ABatchTokenProcessor",
    "A2AMessageProcessor",
    "A2ANodeResultProcessor",
    "A2ANodeStepProcessor",
    "DEFAULT_PROCESSORS",
    # Queue / context
    "EventQueue",
    "RequestContext",
    # Handler
    "DefaultRequestHandler",
    # Task store
    "TaskStore",
    "InMemoryTaskStore",
    # Push notifications
    "PushNotificationService",
    # Agent card
    "get_agent_card",
    # Schemas
    "Task",
    "TaskStatus",
    "TaskState",
    "Message",
    "Part",
    "PartKind",
    "TextPart",
    "DataPart",
    "FilePart",
    "Artifact",
    "TaskStatusUpdateEvent",
    "TaskArtifactUpdateEvent",
    "MessageEvent",
    "PushNotificationPayload",
]
