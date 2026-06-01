"""scheduler — in-memory periodic task scheduler (Phase 4, 2026-05).

Registers `schedule_create` / `schedule_list` / `schedule_cancel` as agent
tools. A background tick loop fires due jobs in one of two modes:
  - tool:  invoke a local tool (via injected tool_invoker)
  - query: send a query to this agent's own loop (via injected query_runner)

In-memory only (jobs + run history lost on restart — prototype scope).
Results are NOT pushed to the user; they accumulate in run history for the
SCHEDULE tab to read (GET /webui/schedule).

L0/L1 decoupling: the service imports nothing from runtime/ or webui/ — it
depends only on two injected callables (tool_invoker, query_runner), mirroring
the delegate_fn / batch_resolver_fn injection pattern.
"""
from scheduler.service import (
    SchedulerService, ScheduledJob, build_scheduler_tools,
    SCHEDULER_TOOL_METADATA,
)

__all__ = [
    "SchedulerService", "ScheduledJob", "build_scheduler_tools",
    "SCHEDULER_TOOL_METADATA",
]
