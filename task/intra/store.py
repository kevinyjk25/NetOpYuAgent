"""
task/intra/store.py
-------------------
TaskStore (Redis hot + Postgres cold) and RetryManager.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from ..schemas import (
    TaskAuditRecord,
    TaskDefinition,
    TaskEventKind,
    TaskState,
)

logger = logging.getLogger(__name__)

_RETRY_BASE_DELAY = 2.0   # seconds; doubles each attempt


class TaskStore:
    """
    Hot path: task state in Redis (fast lookup by task_id).
    Cold path: completed task record written to PostgreSQL.

    Dev mode: both backends are replaced with in-process dicts.
    """

    def __init__(
        self,
        redis_client: Any | None = None,
        pg_pool:      Any | None = None,
    ) -> None:
        self._redis = redis_client
        self._pg    = pg_pool
        self._local: dict[str, TaskDefinition] = {}   # fallback

    # ------------------------------------------------------------------
    # Task CRUD
    # ------------------------------------------------------------------

    async def save(self, task: TaskDefinition) -> None:
        if self._redis:
            key = f"task:{task.task_id}"
            await self._redis.set(key, task.model_dump_json(), ex=86_400)
        else:
            self._local[task.task_id] = task

        # Persist terminal states to Postgres
        if self._pg and task.state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED):
            await self._pg_write(task)

        logger.debug("TaskStore.save task_id=%s state=%s", task.task_id, task.state)

    async def get(self, task_id: str) -> Optional[TaskDefinition]:
        if self._redis:
            raw = await self._redis.get(f"task:{task_id}")
            if raw:
                return TaskDefinition.model_validate_json(raw)
        return self._local.get(task_id)

    async def get_by_session(self, session_id: str) -> list[TaskDefinition]:
        if self._redis:
            # In production: maintain a session→task_ids index in Redis
            logger.warning("get_by_session: Redis index not implemented; using local fallback")
        return [t for t in self._local.values() if t.session_id == session_id]

    async def delete(self, task_id: str) -> None:
        if self._redis:
            await self._redis.delete(f"task:{task_id}")
        self._local.pop(task_id, None)

    async def list_pending(self) -> list[TaskDefinition]:
        return [
            t for t in self._local.values()
            if t.state == TaskState.PENDING
        ]

    # ------------------------------------------------------------------
    # Audit log
    # ------------------------------------------------------------------

    async def write_audit(self, record: TaskAuditRecord) -> None:
        if self._pg:
            await self._pg_write_audit(record)
        logger.debug(
            "TaskAudit: task_id=%s event=%s actor=%s",
            record.task_id, record.event_kind, record.actor,
        )

    # ------------------------------------------------------------------
    # PostgreSQL helpers (stubs — wire to real asyncpg pool)
    # ------------------------------------------------------------------

    async def _pg_write(self, task: TaskDefinition) -> None:
        try:
            async with self._pg.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO task_records
                      (task_id, session_id, context_id, scope, description,
                       priority, state, result, error, retry_count,
                       created_at, completed_at, metadata)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb,$9,$10,$11,$12,$13::jsonb)
                    ON CONFLICT (task_id) DO UPDATE
                      SET state=EXCLUDED.state, result=EXCLUDED.result,
                          error=EXCLUDED.error, completed_at=EXCLUDED.completed_at
                    """,
                    task.task_id, task.session_id, task.context_id,
                    task.scope.value, task.description,
                    task.priority.value, task.state.value,
                    json.dumps(task.result), task.error, task.retry_count,
                    datetime.fromisoformat(task.created_at),
                    datetime.fromisoformat(task.completed_at) if task.completed_at else None,
                    json.dumps(task.metadata),
                )
        except Exception as exc:
            logger.error("TaskStore Postgres write failed: %s", exc)

    async def _pg_write_audit(self, record: TaskAuditRecord) -> None:
        try:
            async with self._pg.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO task_audit
                      (audit_id, task_id, session_id, event_kind, actor, payload, timestamp)
                    VALUES ($1,$2,$3,$4,$5,$6::jsonb,$7)
                    """,
                    record.audit_id, record.task_id, record.session_id,
                    record.event_kind.value, record.actor,
                    json.dumps(record.payload),
                    datetime.fromisoformat(record.timestamp),
                )
        except Exception as exc:
            logger.error("TaskAudit Postgres write failed: %s", exc)


# ---------------------------------------------------------------------------
# RetryManager
# ---------------------------------------------------------------------------

class RetryManager:
    """
    Exponential back-off retry with dead-letter queue (DLQ) support.

    Usage
    -----
        manager = RetryManager(store)
        should_retry = await manager.handle_failure(task, error)
    """

    def __init__(
        self,
        store: TaskStore,
        dlq_callback: Any | None = None,   # async fn(task) for DLQ
    ) -> None:
        self._store = store
        self._dlq   = dlq_callback

    async def handle_failure(
        self,
        task: TaskDefinition,
        error: str,
    ) -> bool:
        """
        Decide whether to retry or send to DLQ.
        Returns True if task will be retried, False if sent to DLQ.
        """
        task.retry_count += 1
        task.error        = error

        if task.retry_count <= task.max_retries:
            delay = _RETRY_BASE_DELAY ** task.retry_count
            logger.warning(
                "RetryManager: task_id=%s attempt=%d/%d delay=%.1fs",
                task.task_id, task.retry_count, task.max_retries, delay,
            )
            task.state = TaskState.RETRYING
            await self._store.save(task)
            await self._store.write_audit(TaskAuditRecord(
                task_id=task.task_id,
                session_id=task.session_id,
                event_kind=TaskEventKind.RETRIED,
                actor="retry_manager",
                payload={"attempt": task.retry_count, "error": error, "delay": delay},
            ))
            await asyncio.sleep(delay)
            task.state = TaskState.PENDING
            await self._store.save(task)
            return True

        # Exceeded max retries → DLQ
        task.state = TaskState.FAILED
        await self._store.save(task)
        logger.error(
            "RetryManager: task_id=%s permanently failed after %d attempts",
            task.task_id, task.retry_count,
        )
        if self._dlq:
            await self._dlq(task)
        return False
