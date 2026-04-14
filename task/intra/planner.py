"""
task/intra/planner.py  [MODIFIED — v2]
--------------------------------------
Changes from v1
---------------
  1. TaskPlanner.__init__() accepts optional ``registry`` (AgentRegistry).
  2. TaskPlanner.decompose() gains a ``skill_role_map`` parameter:
       dict[agent_role_name → skill_id]
     When provided alongside a registry, the planner resolves
     AgentAssignment automatically from the registry instead of
     requiring the caller to build the dict manually.
  3. Falls back to the old ``agent_assignments`` dict if registry is None
     (fully backward compatible).
  4. Logs which resolution path was taken per subtask.

TaskScheduler and TaskExecutor are unchanged.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Optional, TYPE_CHECKING

from ..schemas import (
    AgentAssignment,
    TaskAuditRecord,
    TaskDefinition,
    TaskEventKind,
    TaskPriority,
    TaskScope,
    TaskState,
)
from task.intra.store import RetryManager, TaskStore

if TYPE_CHECKING:
    from registry.registry import AgentRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TaskPlanner  (v2 — registry-aware)
# ---------------------------------------------------------------------------

class TaskPlanner:
    """
    Decomposes a goal into a DAG of TaskDefinition objects.

    Registry-aware resolution (v2)
    --------------------------------
    Pass ``registry`` at construction time and ``skill_role_map`` to
    ``decompose()`` to enable automatic AgentAssignment resolution:

        planner = TaskPlanner(store, registry=registry)
        tasks   = await planner.decompose(
            goal="Analyse P1 alerts and predict next week",
            session_id=..., context_id=...,
            skill_role_map={
                "alert_analyst": "alert_analysis",
                "predictor":     "trend_prediction",
            },
        )

    Manual fallback (v1 compatible)
    --------------------------------
        planner = TaskPlanner(store)
        tasks   = await planner.decompose(
            goal=..., session_id=..., context_id=...,
            agent_assignments={"alert_analyst": AgentAssignment(...)},
        )
    """

    def __init__(
        self,
        store:    TaskStore,
        registry: Optional["AgentRegistry"] = None,
    ) -> None:
        self._store    = store
        self._registry = registry

    async def decompose(
        self,
        goal: str,
        session_id: str,
        context_id: str,
        priority: TaskPriority = TaskPriority.MEDIUM,
        # ── NEW: registry-based resolution ───────────────────────────
        skill_role_map: Optional[dict[str, str]] = None,
        # ── OLD: manual dict (backward compat) ───────────────────────
        agent_assignments: Optional[dict[str, AgentAssignment]] = None,
    ) -> list[TaskDefinition]:
        """
        Decompose goal into subtasks. Returns the full DAG list.

        Parameters
        ----------
        skill_role_map:
            Maps the planner's logical role names to registry skill_ids.
            Only used when ``registry`` was provided at construction time.
            Example: {"alert_analyst": "alert_analysis"}

        agent_assignments:
            v1-style manual dict. Used as fallback when registry is None
            or a role is not in skill_role_map.
        """
        subtask_specs = await self._llm_decompose(goal)

        tasks:  list[TaskDefinition] = []
        id_map: dict[str, str]       = {}   # logical name → task_id

        for spec in subtask_specs:
            deps  = [id_map[d] for d in spec.get("depends_on", []) if d in id_map]
            role  = spec.get("agent_role", "")

            assignment = await self._resolve_assignment(
                role=role,
                skill_role_map=skill_role_map,
                agent_assignments=agent_assignments,
            )

            task = TaskDefinition(
                session_id=session_id,
                context_id=context_id,
                scope=TaskScope.INTER if assignment else TaskScope.INTRA,
                description=spec["description"],
                priority=priority,
                dependencies=deps,
                assignment=assignment,
                parameters=spec.get("parameters", {}),
            )
            id_map[spec["name"]] = task.task_id
            tasks.append(task)

            await self._store.save(task)
            await self._store.write_audit(TaskAuditRecord(
                task_id=task.task_id,
                session_id=session_id,
                event_kind=TaskEventKind.CREATED,
                actor="task_planner",
                payload={
                    "description": task.description,
                    "dependencies": deps,
                    "scope": task.scope.value,
                    "agent_url": assignment.agent_url if assignment else None,
                },
            ))

        logger.info(
            "TaskPlanner: decomposed '%s' into %d task(s) "
            "(inter=%d, intra=%d)",
            goal[:60], len(tasks),
            sum(1 for t in tasks if t.scope == TaskScope.INTER),
            sum(1 for t in tasks if t.scope == TaskScope.INTRA),
        )
        return tasks

    # ------------------------------------------------------------------
    # Assignment resolution (registry-first, manual fallback)
    # ------------------------------------------------------------------

    async def _resolve_assignment(
        self,
        role: str,
        skill_role_map: Optional[dict[str, str]],
        agent_assignments: Optional[dict[str, AgentAssignment]],
    ) -> Optional[AgentAssignment]:
        """
        Resolve an AgentAssignment for the given role.

        Priority order:
          1. Registry lookup (if registry + skill_role_map provided)
          2. Manual agent_assignments dict (v1 fallback)
          3. None → task runs locally (intra)
        """
        if not role:
            return None

        # ── 1. Registry resolution ────────────────────────────────────
        if self._registry is not None and skill_role_map and role in skill_role_map:
            skill_id = skill_role_map[role]
            result   = await self._registry.resolve(skill_id)
            if result is not None:
                logger.debug(
                    "TaskPlanner: registry resolved role=%s skill=%s → %s (%s)",
                    role, skill_id, result.agent_name, result.agent_url,
                )
                return AgentAssignment(
                    agent_id  = result.agent_id,
                    agent_url = result.agent_url,
                    skill_id  = result.skill_id,
                )
            logger.warning(
                "TaskPlanner: no healthy agent for skill=%s (role=%s), "
                "falling back to intra",
                skill_id, role,
            )
            return None

        # ── 2. Manual dict fallback ───────────────────────────────────
        if agent_assignments and role in agent_assignments:
            logger.debug(
                "TaskPlanner: using manual AgentAssignment for role=%s", role
            )
            return agent_assignments[role]

        return None

    # ------------------------------------------------------------------
    # LLM decomposition stub (unchanged from v1)
    # ------------------------------------------------------------------

    async def _llm_decompose(self, goal: str) -> list[dict[str, Any]]:
        """
        Stub — replace with a LangChain structured output chain.

        Each returned dict must contain:
          name        : unique logical name for this subtask
          description : human-readable task description
          agent_role  : optional; matched against skill_role_map
          depends_on  : list of other task names this depends on
          parameters  : dict of extra params passed to the executor
        """
        goal_lower = goal.lower()

        if "alert" in goal_lower and "predict" in goal_lower:
            return [
                {
                    "name": "fetch",
                    "description": f"Fetch alert data for: {goal}",
                    "agent_role": "alert_analyst",
                    "depends_on": [],
                    "parameters": {},
                },
                {
                    "name": "analyse",
                    "description": "Analyse alert severity distribution",
                    "agent_role": "alert_analyst",
                    "depends_on": ["fetch"],
                    "parameters": {},
                },
                {
                    "name": "predict",
                    "description": "Predict next-week alert trend",
                    "agent_role": "predictor",
                    "depends_on": ["fetch"],
                    "parameters": {},
                },
                {
                    "name": "summarise",
                    "description": "Summarise analysis and prediction",
                    "agent_role": "",        # no remote agent needed
                    "depends_on": ["analyse", "predict"],
                    "parameters": {},
                },
            ]

        if "alert" in goal_lower:
            return [
                {
                    "name": "fetch",
                    "description": f"Fetch alert data for: {goal}",
                    "agent_role": "alert_analyst",
                    "depends_on": [],
                    "parameters": {},
                },
                {
                    "name": "analyse",
                    "description": "Analyse and report alerts",
                    "agent_role": "alert_analyst",
                    "depends_on": ["fetch"],
                    "parameters": {},
                },
            ]

        return [
            {
                "name": "main",
                "description": goal,
                "agent_role": "",
                "depends_on": [],
                "parameters": {},
            }
        ]


# ---------------------------------------------------------------------------
# TaskScheduler  (unchanged from v1)
# ---------------------------------------------------------------------------

class TaskScheduler:
    """Priority queue scheduler with dependency resolution and concurrency cap."""

    def __init__(self, store: TaskStore, max_concurrency: int = 5) -> None:
        self._store   = store
        self._max_con = max_concurrency

    async def get_ready_tasks(self, tasks: list[TaskDefinition]) -> list[TaskDefinition]:
        completed_ids = {t.task_id for t in tasks if t.state == TaskState.COMPLETED}
        ready = [
            t for t in tasks
            if t.state == TaskState.PENDING
            and all(dep in completed_ids for dep in t.dependencies)
        ]
        ready.sort(key=lambda t: t.priority.value)
        return ready

    async def run_dag(
        self,
        tasks: list[TaskDefinition],
        executor: "TaskExecutor",
    ) -> list[TaskDefinition]:
        remaining = {t.task_id: t for t in tasks}
        active: dict[str, asyncio.Task] = {}

        while remaining or active:
            ready = await self.get_ready_tasks(list(remaining.values()))
            for task in ready:
                if len(active) >= self._max_con:
                    break
                if task.task_id in active:
                    continue
                remaining.pop(task.task_id)
                active[task.task_id] = asyncio.create_task(executor.run(task))

            if not active:
                break

            done, _ = await asyncio.wait(
                active.values(), return_when=asyncio.FIRST_COMPLETED
            )
            for fut in done:
                for tid, t in list(active.items()):
                    if t is fut:
                        del active[tid]
                        break

        return tasks


# ---------------------------------------------------------------------------
# TaskExecutor  (unchanged from v1)
# ---------------------------------------------------------------------------

class TaskExecutor:
    """Runs a single TaskDefinition."""

    def __init__(
        self,
        store:         TaskStore,
        retry_manager: RetryManager,
        tool_registry: Optional[dict[str, Callable]] = None,
    ) -> None:
        self._store  = store
        self._retry  = retry_manager
        self._tools  = tool_registry or {}

    async def run(self, task: TaskDefinition) -> TaskDefinition:
        task.state      = TaskState.RUNNING
        task.started_at = datetime.now(timezone.utc).isoformat()
        await self._store.save(task)
        await self._store.write_audit(TaskAuditRecord(
            task_id=task.task_id, session_id=task.session_id,
            event_kind=TaskEventKind.STARTED, actor="task_executor",
        ))

        try:
            result = await self._execute_task(task)
            task.state        = TaskState.COMPLETED
            task.result       = result
            task.completed_at = datetime.now(timezone.utc).isoformat()
            await self._store.save(task)
            await self._store.write_audit(TaskAuditRecord(
                task_id=task.task_id, session_id=task.session_id,
                event_kind=TaskEventKind.COMPLETED, actor="task_executor",
                payload={"result_keys": list(result.keys())},
            ))
        except Exception as exc:
            logger.exception("TaskExecutor: task_id=%s failed: %s", task.task_id, exc)
            retry = await self._retry.handle_failure(task, str(exc))
            if retry:
                return await self.run(task)
        return task

    async def stream(self, task: TaskDefinition) -> AsyncIterator[dict[str, Any]]:
        yield {"node_step": f"Starting: {task.description}", "node": "task_executor"}
        result_task = await self.run(task)
        if result_task.state == TaskState.COMPLETED:
            yield {"node_result": result_task.result or {}, "node": "task_result"}
        else:
            yield {"message": f"Task failed: {result_task.error}", "node": "task_executor"}

    async def _execute_task(self, task: TaskDefinition) -> dict[str, Any]:
        await asyncio.sleep(0)
        tool = self._tools.get(task.parameters.get("tool", ""))
        if tool:
            return await tool(task.parameters)
        return {
            "status":      "completed",
            "description": task.description,
            "output":      f"Result for: {task.description}",
        }
