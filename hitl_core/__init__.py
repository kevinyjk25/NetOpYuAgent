"""
hitl_core — Portable, langchain-free human-in-the-loop pipeline.

Quick start:

    from hitl_core import (
        HitlPipeline, PipelineState, PipelineContext,
        InMemoryCheckpointStore,
        HitlPayload, HitlDecision, TriggerKind, DecisionKind, RiskLevel,
    )

    store = InMemoryCheckpointStore()
    pipeline = HitlPipeline(store=store)

    async def my_plan_step(ctx: PipelineContext):
        if needs_approval:
            decision = await ctx.request_approval(HitlPayload(...))
            if decision.decision != DecisionKind.APPROVE:
                raise PipelineAborted("Operator rejected")

    pipeline.add_step("plan", my_plan_step)

    async for event in pipeline.run(PipelineState(user_query="...")):
        if event["type"] == "interrupt":
            decision = await get_operator_decision(event["payload"])
            await pipeline.resume_with(decision)
        elif event["type"] == "batch_interrupt":
            # Render N-child batch card to operator
            submission = await get_operator_batch_decision(event["batch"])
            await router.deliver_batch(submission)
        elif event["type"] == "done":
            print(event["state"])

Batch approval — for parallel-independent HITLs:

    async def batch_step(ctx):
        resolution = await ctx.request_batch_approval(
            payloads=[payload_for_ap1, payload_for_ap2, payload_for_ap3],
            title="Restart 3 APs",
            policy=BatchPolicy.BEST_EFFORT,
        )
        for d in resolution.decisions:
            ...

Redis-backed checkpointing (production):

    from hitl_core import RedisCheckpointStore
    store = RedisCheckpointStore(redis_url="redis://prod-redis:6379/0")
"""

from .audit import (
    AuditLogger,
    AuditSink,
    FileAuditSink,
    InMemoryAuditSink,
    RedisAuditSink,
    build_sink_from_config,
)
from .batch import BATCH_ID_KEY, BatchCoordinator, BatchResolution, get_batch_id
from .coreference import (
    Coreferencer,
    CoreferenceResult,
    DEFAULT_DEVICE_PATTERN,
    DEFAULT_DEVICE_TOOL_SIGNATURES,
    build_default_device_coreferencer,
)
from .pipeline import (
    HitlPipeline,
    PipelineAborted,
    PipelineContext,
    PipelineState,
    PipelineStep,
)
from .router import (
    DecisionValidationError,
    HitlRouter,
    Resumer,
    ResumeError,
)
from .schema import (
    AuditEventKind,
    BatchPolicy,
    BatchSnapshot,
    BatchState,
    BatchSubmission,
    BatchWaitMode,
    CheckpointEntry,
    ChoiceOption,
    ClarificationField,
    DecisionKind,
    HitlAuditRecord,
    HitlBatch,
    HitlDecision,
    HitlPayload,
    InterruptState,
    ProposedAction,
    ResumeHandle,
    RiskLevel,
    TriggerKind,
)
from .store import (
    BaseCheckpointStore,
    InMemoryCheckpointStore,
    RedisCheckpointStore,
    SqliteCheckpointStore,
    build_store_from_config,
)
from .triggers import (
    DestructiveTrigger,
    LowConfidenceTrigger,
    PolicyViolationTrigger,
    SeverityTrigger,
    Trigger,
    TriggerContext,
    TriggerEngine,
    TriggerOutcome,
)

__all__ = [
    # Pipeline
    "HitlPipeline",
    "PipelineAborted",
    "PipelineContext",
    "PipelineState",
    "PipelineStep",
    # Router
    "HitlRouter",
    "Resumer",
    "DecisionValidationError",
    "ResumeError",
    # Batch
    "BatchCoordinator",
    "BatchResolution",
    "BATCH_ID_KEY",
    "get_batch_id",
    # Triggers
    "Trigger",
    "TriggerContext",
    "TriggerEngine",
    "TriggerOutcome",
    "DestructiveTrigger",
    "LowConfidenceTrigger",
    "PolicyViolationTrigger",
    "SeverityTrigger",
    # Audit
    "AuditLogger",
    "AuditSink",
    "FileAuditSink",
    "InMemoryAuditSink",
    "RedisAuditSink",
    "build_sink_from_config",
    # Coreference
    "Coreferencer",
    "CoreferenceResult",
    "DEFAULT_DEVICE_PATTERN",
    "DEFAULT_DEVICE_TOOL_SIGNATURES",
    "build_default_device_coreferencer",
    # Schema
    "AuditEventKind",
    "BatchPolicy",
    "BatchSnapshot",
    "BatchState",
    "BatchSubmission",
    "BatchWaitMode",
    "CheckpointEntry",
    "ChoiceOption",
    "ClarificationField",
    "DecisionKind",
    "HitlAuditRecord",
    "HitlBatch",
    "HitlDecision",
    "HitlPayload",
    "InterruptState",
    "ProposedAction",
    "ResumeHandle",
    "RiskLevel",
    "TriggerKind",
    # Store
    "BaseCheckpointStore",
    "InMemoryCheckpointStore",
    "RedisCheckpointStore",
    "SqliteCheckpointStore",
    "build_store_from_config",
]

__version__ = "0.3.0"