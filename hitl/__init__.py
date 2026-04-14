"""
hitl — Human-in-the-Loop module for IT Ops Monitoring Agent
============================================================

Public surface::

    from hitl import (
        # Schemas
        HitlPayload, HitlDecision, HitlState,
        TriggerKind, DecisionKind, RiskLevel, InterruptState,
        ProposedAction, HitlInterruptEvent, HitlAuditRecord,

        # L1 Triggers
        HitlConfig, HitlTrigger,
        DestructiveActionTrigger, SeverityTrigger,
        ConfidenceTrigger, AmbiguityTrigger,
        build_trigger_chain, evaluate_triggers,

        # L2 Graph
        build_hitl_graph, run_with_hitl,

        # L3 Review
        HitlReviewService, ReviewChannelConfig,
        A2APushNotificationChannel, SlackWebhookChannel,
        PagerDutyChannel, WebDashboardSSEChannel,

        # L4 Decision
        HitlDecisionRouter, HitlTimeoutWatchdog, DecisionResult,

        # L5 Audit
        HitlAuditService, InMemoryAuditBackend, PostgresAuditBackend,

        # A2A integration
        ITOpsHitlAgentExecutor, HitlA2AEventProcessor,
        build_hitl_processors,

        # FastAPI router
        create_hitl_router,
    )
"""

from .a2a_integration import (
    HitlA2AEventProcessor,
    ITOpsHitlAgentExecutor,
    build_hitl_processors,
)
from .audit import (
    HitlAuditService,
    InMemoryAuditBackend,
    PostgresAuditBackend,
)
from .decision import (
    DecisionResult,
    HitlDecisionRouter,
    HitlTimeoutWatchdog,
)
from .graph import build_hitl_graph, run_with_hitl
from .review import (
    A2APushNotificationChannel,
    HitlReviewService,
    PagerDutyChannel,
    ReviewChannelConfig,
    SlackWebhookChannel,
    WebDashboardSSEChannel,
    WebSocketHitlChannel,
    WebSocketHitlManager,
    get_sse_channel,
    get_ws_manager,
)
from .router import create_hitl_router
from .schemas import (
    AuditEventKind,
    DecisionKind,
    HitlAuditRecord,
    HitlDecision,
    HitlInterruptEvent,
    HitlPayload,
    HitlState,
    InterruptState,
    ProposedAction,
    RiskLevel,
    TriggerKind,
)
from .triggers import (
    AmbiguityTrigger,
    ConfidenceTrigger,
    DestructiveActionTrigger,
    HitlConfig,
    HitlTrigger,
    SeverityTrigger,
    build_trigger_chain,
    evaluate_triggers,
)

__all__ = [
    # Schemas
    "HitlPayload", "HitlDecision", "HitlState",
    "TriggerKind", "DecisionKind", "RiskLevel", "InterruptState",
    "ProposedAction", "HitlInterruptEvent", "HitlAuditRecord", "AuditEventKind",
    # Triggers
    "HitlConfig", "HitlTrigger",
    "DestructiveActionTrigger", "SeverityTrigger",
    "ConfidenceTrigger", "AmbiguityTrigger",
    "build_trigger_chain", "evaluate_triggers",
    # Graph
    "build_hitl_graph", "run_with_hitl",
    # Review
    "HitlReviewService", "ReviewChannelConfig",
    "A2APushNotificationChannel", "SlackWebhookChannel",
    "PagerDutyChannel", "WebDashboardSSEChannel", "get_sse_channel",
    "WebSocketHitlManager", "WebSocketHitlChannel", "get_ws_manager",
    # Decision
    "HitlDecisionRouter", "HitlTimeoutWatchdog", "DecisionResult",
    # Audit
    "HitlAuditService", "InMemoryAuditBackend", "PostgresAuditBackend",
    # A2A integration
    "ITOpsHitlAgentExecutor", "HitlA2AEventProcessor", "build_hitl_processors",
    # Router
    "create_hitl_router",
]