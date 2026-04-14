"""
a2a/task_store.py
-----------------
In-memory TaskStore with basic TTL eviction.

In production, swap ``InMemoryTaskStore`` for a Redis or PostgreSQL-backed
implementation by subclassing ``TaskStore``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

from .schemas import Task

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 3600  # 1 hour


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class TaskStore(ABC):
    @abstractmethod
    async def save(self, task: Task) -> None: ...

    @abstractmethod
    async def get(self, task_id: str) -> Optional[Task]: ...

    @abstractmethod
    async def delete(self, task_id: str) -> None: ...

    @abstractmethod
    async def list_all(self) -> list[Task]: ...


# ---------------------------------------------------------------------------
# In-memory implementation
# ---------------------------------------------------------------------------

class InMemoryTaskStore(TaskStore):
    """
    Thread-safe in-memory task store with optional TTL eviction.

    Parameters
    ----------
    ttl_seconds:
        How long to keep completed/failed/canceled tasks before evicting them.
        Active (submitted / working) tasks are never evicted.
    """

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        self._store: dict[str, Task] = {}
        self._timestamps: dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._ttl = ttl_seconds

    async def save(self, task: Task) -> None:
        async with self._lock:
            self._store[task.id] = task
            self._timestamps[task.id] = time.monotonic()
            logger.debug("TaskStore.save task_id=%s state=%s", task.id, task.status.state)

    async def get(self, task_id: str) -> Optional[Task]:
        async with self._lock:
            return self._store.get(task_id)

    async def delete(self, task_id: str) -> None:
        async with self._lock:
            self._store.pop(task_id, None)
            self._timestamps.pop(task_id, None)
            logger.debug("TaskStore.delete task_id=%s", task_id)

    async def list_all(self) -> list[Task]:
        async with self._lock:
            return list(self._store.values())

    async def evict_expired(self) -> int:
        """
        Remove tasks that have exceeded the TTL and are in a terminal state.
        Returns the number of tasks evicted.
        """
        terminal = {"completed", "failed", "canceled"}
        now = time.monotonic()
        to_delete: list[str] = []

        async with self._lock:
            for task_id, task in self._store.items():
                if task.status.state.value in terminal:
                    age = now - self._timestamps.get(task_id, now)
                    if age > self._ttl:
                        to_delete.append(task_id)

            for task_id in to_delete:
                del self._store[task_id]
                del self._timestamps[task_id]

        if to_delete:
            logger.info("TaskStore evicted %d expired task(s)", len(to_delete))

        return len(to_delete)
