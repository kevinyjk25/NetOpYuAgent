"""
task/__init__.py  [MODIFIED — v2]
----------------------------------
Changes from v1
---------------
  1. create_task_system() accepts optional ``registry`` parameter.
  2. TaskSystem dataclass gains ``registry`` field.
  3. TaskPlanner is now constructed with the registry reference.
  4. All other components unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

from .inter.coordinator import A2ATaskDispatcher, MultiRoundCoordinator, ResultAggregator
from .inter.hitl_bridge import HitlTaskBridge
from .inter.session import SessionManager
from .intra.planner import TaskExecutor, TaskPlanner, TaskScheduler
from .intra.store import RetryManager, TaskStore
from task.schemas import (
    AgentAssignment,
    MultiRoundContext,
    SessionRecord,
    SubtaskAssignmentResult,
    TaskAuditRecord,
    TaskDefinition,
    TaskEventKind,
    TaskPriority,
    TaskScope,
    TaskState,
)

if TYPE_CHECKING:
    from registry.registry import AgentRegistry


@dataclass
class TaskSystem:
    store:       TaskStore
    planner:     TaskPlanner
    scheduler:   TaskScheduler
    executor:    TaskExecutor
    retry:       RetryManager
    session_mgr: SessionManager
    dispatcher:  A2ATaskDispatcher
    coordinator: MultiRoundCoordinator
    aggregator:  ResultAggregator
    hitl_bridge: Optional[HitlTaskBridge]    = None
    registry:    Optional["AgentRegistry"]   = None   # ← NEW


async def create_task_system(
    redis_client:    Any | None = None,
    pg_pool:         Any | None = None,
    hitl_router:     Any | None = None,
    review_svc:      Any | None = None,
    max_concurrency: int = 5,
    registry:        Optional["AgentRegistry"] = None,  # ← NEW
) -> TaskSystem:
    """
    Factory: wire up and return a fully-initialised TaskSystem.

    New in v2: pass ``registry`` to enable automatic AgentAssignment
    resolution in TaskPlanner.decompose().
    """
    store       = TaskStore(redis_client, pg_pool)
    retry       = RetryManager(store)
    executor    = TaskExecutor(store, retry)
    planner     = TaskPlanner(store, registry=registry)   # ← passes registry
    scheduler   = TaskScheduler(store, max_concurrency)
    session_mgr = SessionManager(redis_client, pg_pool)
    dispatcher  = A2ATaskDispatcher()
    coordinator = MultiRoundCoordinator(session_mgr)
    aggregator  = ResultAggregator()

    hitl_bridge = None
    if hitl_router and review_svc:
        hitl_bridge = HitlTaskBridge(hitl_router, review_svc, store, session_mgr)

    return TaskSystem(
        store=store, planner=planner, scheduler=scheduler,
        executor=executor, retry=retry, session_mgr=session_mgr,
        dispatcher=dispatcher, coordinator=coordinator,
        aggregator=aggregator, hitl_bridge=hitl_bridge,
        registry=registry,
    )


__all__ = [
    "TaskSystem", "create_task_system",
    "TaskStore", "RetryManager",
    "TaskPlanner", "TaskScheduler", "TaskExecutor",
    "SessionManager",
    "A2ATaskDispatcher", "MultiRoundCoordinator", "ResultAggregator",
    "HitlTaskBridge",
    "TaskDefinition", "SessionRecord", "TaskAuditRecord",
    "TaskState", "TaskPriority", "TaskScope",
    "AgentAssignment", "MultiRoundContext",
]
