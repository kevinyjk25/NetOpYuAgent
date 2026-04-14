"""
registry/store.py
-----------------
Storage backends for the Agent Registry.

InMemoryRegistryStore  — dev / testing, process-scoped
RedisRegistryStore     — production, shared across replicas

Both implement the RegistryStore protocol.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional

from registry.schemas import AgentEntry, AgentHealthState, RegistryConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class RegistryStore(ABC):
    """Protocol every store backend must satisfy."""

    @abstractmethod
    async def save(self, entry: AgentEntry) -> None: ...

    @abstractmethod
    async def get(self, agent_id: str) -> Optional[AgentEntry]: ...

    @abstractmethod
    async def get_by_url(self, url: str) -> Optional[AgentEntry]: ...

    @abstractmethod
    async def list_all(self) -> list[AgentEntry]: ...

    @abstractmethod
    async def delete(self, agent_id: str) -> None: ...

    @abstractmethod
    async def update_health(
        self,
        agent_id: str,
        health: AgentHealthState,
        failures: int = 0,
    ) -> None: ...


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------

class InMemoryRegistryStore(RegistryStore):
    """
    Thread-safe in-memory store.
    Suitable for single-process deployments and tests.
    """

    def __init__(self) -> None:
        self._store: dict[str, AgentEntry] = {}
        self._url_idx: dict[str, str] = {}     # url → agent_id
        self._lock = asyncio.Lock()

    async def save(self, entry: AgentEntry) -> None:
        async with self._lock:
            # Remove old URL index if URL changed
            old = self._store.get(entry.agent_id)
            if old and old.base_url in self._url_idx:
                del self._url_idx[old.base_url]
            self._store[entry.agent_id] = entry
            self._url_idx[entry.base_url] = entry.agent_id
        logger.debug("RegistryStore.save agent_id=%s url=%s", entry.agent_id, entry.base_url)

    async def get(self, agent_id: str) -> Optional[AgentEntry]:
        async with self._lock:
            return self._store.get(agent_id)

    async def get_by_url(self, url: str) -> Optional[AgentEntry]:
        async with self._lock:
            aid = self._url_idx.get(url.rstrip("/"))
            return self._store.get(aid) if aid else None

    async def list_all(self) -> list[AgentEntry]:
        async with self._lock:
            return list(self._store.values())

    async def delete(self, agent_id: str) -> None:
        async with self._lock:
            entry = self._store.pop(agent_id, None)
            if entry:
                self._url_idx.pop(entry.base_url, None)

    async def update_health(
        self,
        agent_id: str,
        health: AgentHealthState,
        failures: int = 0,
    ) -> None:
        async with self._lock:
            entry = self._store.get(agent_id)
            if entry:
                entry.health = health
                entry.consecutive_failures = failures
                entry.last_checked_at = datetime.now(timezone.utc).isoformat()
                if health == AgentHealthState.HEALTHY:
                    entry.last_seen_at = entry.last_checked_at


# ---------------------------------------------------------------------------
# Redis store
# ---------------------------------------------------------------------------

class RedisRegistryStore(RegistryStore):
    """
    Redis-backed store — shares registry state across multiple app replicas.

    Keys
    ----
      {prefix}:agent:{agent_id}  →  JSON-serialised AgentEntry
      {prefix}:url:{base_url}    →  agent_id  (lookup index)
      {prefix}:all               →  Redis Set of agent_ids
    """

    def __init__(
        self,
        redis_client: Any,          # redis.asyncio.Redis
        config: RegistryConfig | None = None,
    ) -> None:
        self._r      = redis_client
        self._cfg    = config or RegistryConfig()
        self._prefix = self._cfg.redis_prefix
        self._ttl    = self._cfg.agent_ttl_seconds * 2   # generous TTL

    def _key(self, agent_id: str) -> str:
        return f"{self._prefix}:agent:{agent_id}"

    def _url_key(self, url: str) -> str:
        return f"{self._prefix}:url:{url.rstrip('/')}"

    def _all_key(self) -> str:
        return f"{self._prefix}:all"

    async def save(self, entry: AgentEntry) -> None:
        key  = self._key(entry.agent_id)
        data = entry.model_dump_json()
        pipe = self._r.pipeline()
        pipe.set(key, data, ex=self._ttl)
        pipe.set(self._url_key(entry.base_url), entry.agent_id, ex=self._ttl)
        pipe.sadd(self._all_key(), entry.agent_id)
        await pipe.execute()
        logger.debug("RedisRegistryStore.save agent_id=%s", entry.agent_id)

    async def get(self, agent_id: str) -> Optional[AgentEntry]:
        raw = await self._r.get(self._key(agent_id))
        return AgentEntry.model_validate_json(raw) if raw else None

    async def get_by_url(self, url: str) -> Optional[AgentEntry]:
        agent_id = await self._r.get(self._url_key(url))
        if not agent_id:
            return None
        agent_id_str = agent_id.decode() if isinstance(agent_id, bytes) else agent_id
        return await self.get(agent_id_str)

    async def list_all(self) -> list[AgentEntry]:
        ids = await self._r.smembers(self._all_key())
        entries = []
        for aid in ids:
            aid_str = aid.decode() if isinstance(aid, bytes) else aid
            entry = await self.get(aid_str)
            if entry:
                entries.append(entry)
        return entries

    async def delete(self, agent_id: str) -> None:
        entry = await self.get(agent_id)
        pipe  = self._r.pipeline()
        pipe.delete(self._key(agent_id))
        if entry:
            pipe.delete(self._url_key(entry.base_url))
        pipe.srem(self._all_key(), agent_id)
        await pipe.execute()

    async def update_health(
        self,
        agent_id: str,
        health: AgentHealthState,
        failures: int = 0,
    ) -> None:
        entry = await self.get(agent_id)
        if not entry:
            return
        entry.health = health
        entry.consecutive_failures = failures
        entry.last_checked_at = datetime.now(timezone.utc).isoformat()
        if health == AgentHealthState.HEALTHY:
            entry.last_seen_at = entry.last_checked_at
        await self.save(entry)
