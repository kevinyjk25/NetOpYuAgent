"""
registry/registry.py
--------------------
AgentRegistry — the single entry-point for all registry operations.

Key responsibilities
--------------------
  register()      register an agent (static or dynamic)
  resolve()       look up the best agent for a given skill_id
  resolve_all()   list every agent that supports a skill
  deregister()    remove an agent
  self_register() register THIS process's own AgentCard

Background tasks (start with start() / stop with stop())
  _health_watcher  : periodic health checks on all registered agents
  _refresh_watcher : periodic re-fetch of dynamic agents' AgentCards

Load balancing strategies (configurable via RegistryConfig.lb_strategy)
  round_robin     : cycle through healthy agents in registration order
  random          : pick uniformly at random
  least_loaded    : pick the agent with the fewest active tasks
                    (requires callers to call record_task_start/end)
"""

from __future__ import annotations

import asyncio
import logging
import random as _random
from typing import Any, Optional

from registry.discovery import AgentDiscovery
from .schemas import (
    AgentEntry,
    AgentHealthState,
    RegistrationSource,
    RegistryConfig,
    ResolutionResult,
)
from .store import InMemoryRegistryStore, RegistryStore

logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Central agent registry.

    Parameters
    ----------
    store   : RegistryStore backend (in-memory or Redis)
    config  : RegistryConfig thresholds
    discovery : AgentDiscovery for fetching remote cards

    Usage
    -----
        registry = AgentRegistry()
        await registry.start()                     # launch background tasks

        # Register agents from config
        await registry.register_from_urls([
            "http://agent-b.internal/api/v1/a2a",
            "http://predictor.internal/api/v1/a2a",
        ])

        # Resolve in TaskPlanner
        result = await registry.resolve("alert_analysis")
        assignment = AgentAssignment(
            agent_id  = result.agent_id,
            agent_url = result.agent_url,
            skill_id  = result.skill_id,
        )

        await registry.stop()
    """

    def __init__(
        self,
        store:     Optional[RegistryStore]   = None,
        config:    Optional[RegistryConfig]  = None,
        discovery: Optional[AgentDiscovery]  = None,
    ) -> None:
        self._store     = store     or InMemoryRegistryStore()
        self._cfg       = config    or RegistryConfig()
        self._discovery = discovery or AgentDiscovery(self._cfg)

        # Round-robin cursors: skill_id → cycle iterator
        self._rr_cursors: dict[str, Any] = {}
        # Active task counts for least-loaded strategy: agent_id → int
        self._task_counts: dict[str, int] = {}

        self._running       = False
        self._health_task:  Optional[asyncio.Task] = None
        self._refresh_task: Optional[asyncio.Task] = None

        # URLs to keep refreshing (dynamic agents)
        self._dynamic_urls: list[str] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Launch background health-check and refresh tasks."""
        self._running    = True
        self._health_task  = asyncio.create_task(self._health_watcher())
        self._refresh_task = asyncio.create_task(self._refresh_watcher())
        logger.info("AgentRegistry started (strategy=%s)", self._cfg.lb_strategy)

    async def stop(self) -> None:
        """Cancel background tasks gracefully."""
        self._running = False
        for task in (self._health_task, self._refresh_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        logger.info("AgentRegistry stopped")

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    async def register(self, entry: AgentEntry) -> None:
        """
        Register a pre-built AgentEntry.
        Indices are built automatically if not already present.
        """
        if not entry.skill_index:
            entry.build_indices()
        await self._store.save(entry)
        logger.info(
            "AgentRegistry.register agent_id=%s name=%s skills=%s",
            entry.agent_id, entry.card.name,
            list(entry.skill_index.keys()),
        )

    async def register_from_url(
        self,
        url: str,
        source: RegistrationSource = RegistrationSource.DYNAMIC,
    ) -> Optional[AgentEntry]:
        """
        Fetch AgentCard from *url* and register the agent.
        Returns the entry on success, None on failure.
        """
        entry = await self._discovery.fetch(url, source=source)
        if entry is None:
            logger.error("AgentRegistry: could not discover agent at %s", url)
            return None
        await self.register(entry)
        if source == RegistrationSource.DYNAMIC:
            if url not in self._dynamic_urls:
                self._dynamic_urls.append(url)
        return entry

    async def register_from_urls(
        self,
        urls: list[str],
        source: RegistrationSource = RegistrationSource.STATIC,
    ) -> list[AgentEntry]:
        """Fetch and register multiple agents concurrently."""
        entries = await self._discovery.fetch_many(urls, source=source)
        for entry in entries:
            await self.register(entry)
        if source == RegistrationSource.DYNAMIC:
            for url in urls:
                if url not in self._dynamic_urls:
                    self._dynamic_urls.append(url)
        return entries

    async def register_static(
        self,
        agent_id: str,
        agent_url: str,
        skill_ids: list[str],
        agent_name: str = "",
        skill_descriptions: Optional[dict[str, str]] = None,
    ) -> AgentEntry:
        """
        Register an agent from a static config without fetching its card.
        Useful for agents that don't serve an AgentCard endpoint.
        """
        from .schemas import AgentSkill, RawAgentCard
        descs = skill_descriptions or {}
        skills = [
            AgentSkill(id=sid, name=sid, description=descs.get(sid, ""))
            for sid in skill_ids
        ]
        card  = RawAgentCard(name=agent_name or agent_id, url=agent_url, skills=skills)
        entry = AgentEntry(
            agent_id=agent_id,
            card=card,
            source=RegistrationSource.STATIC,
            health=AgentHealthState.HEALTHY,  # assume healthy for static
        )
        entry.build_indices()
        await self.register(entry)
        return entry

    async def self_register(self, card_dict: dict) -> AgentEntry:
        """
        Register this process's own AgentCard (called from main.py at startup).
        The card is sourced locally, so no HTTP fetch is needed.
        """
        entry = AgentDiscovery._parse(
            card_dict,
            base_url=card_dict.get("url", "http://localhost:8000/api/v1/a2a"),
            source=RegistrationSource.SELF,
        )
        entry.health = AgentHealthState.HEALTHY
        await self.register(entry)
        logger.info("AgentRegistry: self-registered as %s", entry.card.name)
        return entry

    async def deregister(self, agent_id: str) -> None:
        await self._store.delete(agent_id)
        self._task_counts.pop(agent_id, None)
        logger.info("AgentRegistry.deregister agent_id=%s", agent_id)

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    async def resolve(
        self,
        skill_id: str,
        exclude_agent_ids: Optional[list[str]] = None,
    ) -> Optional[ResolutionResult]:
        """
        Return the best available agent for *skill_id* using the configured
        load-balancing strategy.

        Returns None if no healthy agent supports the skill.
        """
        candidates = await self._candidates_for_skill(
            skill_id, exclude=exclude_agent_ids or []
        )
        if not candidates:
            logger.warning("AgentRegistry.resolve: no agent for skill=%s", skill_id)
            return None

        agent = self._pick(candidates, skill_id)
        skill = agent.get_skill(skill_id)

        return ResolutionResult(
            agent_id=agent.agent_id,
            agent_url=agent.base_url,
            skill_id=skill_id,
            skill=skill,
            agent_name=agent.card.name,
            health=agent.health,
        )

    async def resolve_all(self, skill_id: str) -> list[ResolutionResult]:
        """Return all healthy agents that support *skill_id*."""
        candidates = await self._candidates_for_skill(skill_id)
        results = []
        for agent in candidates:
            skill = agent.get_skill(skill_id)
            if skill:
                results.append(ResolutionResult(
                    agent_id=agent.agent_id,
                    agent_url=agent.base_url,
                    skill_id=skill_id,
                    skill=skill,
                    agent_name=agent.card.name,
                    health=agent.health,
                ))
        return results

    async def resolve_by_tag(self, tag: str) -> list[ResolutionResult]:
        """Return one result per (agent, skill) pair matching *tag*."""
        all_agents = await self._store.list_all()
        results = []
        for agent in all_agents:
            if not agent.is_available:
                continue
            for skill in agent.skills_for_tag(tag):
                results.append(ResolutionResult(
                    agent_id=agent.agent_id,
                    agent_url=agent.base_url,
                    skill_id=skill.id,
                    skill=skill,
                    agent_name=agent.card.name,
                    health=agent.health,
                ))
        return results

    async def list_agents(self) -> list[AgentEntry]:
        return await self._store.list_all()

    async def get_agent(self, agent_id: str) -> Optional[AgentEntry]:
        return await self._store.get(agent_id)

    # ------------------------------------------------------------------
    # Task-load tracking (for least_loaded strategy)
    # ------------------------------------------------------------------

    def record_task_start(self, agent_id: str) -> None:
        self._task_counts[agent_id] = self._task_counts.get(agent_id, 0) + 1

    def record_task_end(self, agent_id: str) -> None:
        if agent_id in self._task_counts:
            self._task_counts[agent_id] = max(0, self._task_counts[agent_id] - 1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _candidates_for_skill(
        self,
        skill_id: str,
        exclude: Optional[list[str]] = None,
    ) -> list[AgentEntry]:
        all_agents = await self._store.list_all()
        excl = set(exclude or [])
        return [
            a for a in all_agents
            if a.is_available
            and a.supports_skill(skill_id)
            and a.agent_id not in excl
        ]

    def _pick(self, candidates: list[AgentEntry], skill_id: str) -> AgentEntry:
        strategy = self._cfg.lb_strategy

        if strategy == "random" or len(candidates) == 1:
            return _random.choice(candidates)

        if strategy == "least_loaded":
            return min(
                candidates,
                key=lambda a: self._task_counts.get(a.agent_id, 0),
            )

        # Default: round_robin
        cursor = self._rr_cursors.get(skill_id)
        if cursor is None or True:   # rebuild each time (candidates may change)
            ids = [a.agent_id for a in candidates]
            # Find where we left off
            last = getattr(self, f"_rr_last_{skill_id}", None)
            if last and last in ids:
                idx = (ids.index(last) + 1) % len(ids)
            else:
                idx = 0
            chosen = candidates[idx]
            setattr(self, f"_rr_last_{skill_id}", chosen.agent_id)
            return chosen

        return candidates[0]

    # ------------------------------------------------------------------
    # Background tasks
    # ------------------------------------------------------------------

    async def _health_watcher(self) -> None:
        """Periodically health-check all registered agents."""
        interval = self._cfg.health_check_interval_seconds
        logger.info("AgentRegistry health watcher started (interval=%ds)", interval)
        while self._running:
            await asyncio.sleep(interval)
            try:
                await self._run_health_checks()
            except Exception as exc:
                logger.exception("Health watcher error: %s", exc)

    async def _run_health_checks(self) -> None:
        agents = await self._store.list_all()
        # Skip self-registered agents (health known locally)
        remote = [a for a in agents if a.source != RegistrationSource.SELF]
        if not remote:
            return

        results = await asyncio.gather(
            *[self._discovery.health_check(a) for a in remote],
            return_exceptions=True,
        )
        for agent, result in zip(remote, results):
            if isinstance(result, Exception):
                new_health = AgentHealthState.UNHEALTHY
                failures   = agent.consecutive_failures + 1
            else:
                new_health = result
                failures   = 0 if result == AgentHealthState.HEALTHY else agent.consecutive_failures + 1

            if failures >= self._cfg.max_consecutive_failures:
                new_health = AgentHealthState.UNHEALTHY

            if new_health != agent.health:
                logger.info(
                    "AgentRegistry: agent %s health %s → %s",
                    agent.agent_id, agent.health.value, new_health.value,
                )
            await self._store.update_health(agent.agent_id, new_health, failures)

    async def _refresh_watcher(self) -> None:
        """Periodically re-fetch AgentCards for dynamic agents."""
        interval = self._cfg.agent_ttl_seconds
        logger.info("AgentRegistry refresh watcher started (interval=%ds)", interval)
        while self._running:
            await asyncio.sleep(interval)
            if not self._dynamic_urls:
                continue
            try:
                await self._refresh_dynamic()
            except Exception as exc:
                logger.exception("Refresh watcher error: %s", exc)

    async def _refresh_dynamic(self) -> None:
        entries = await self._discovery.fetch_many(
            self._dynamic_urls, source=RegistrationSource.DYNAMIC
        )
        for entry in entries:
            existing = await self._store.get_by_url(entry.base_url)
            if existing:
                entry.agent_id = existing.agent_id   # preserve identity
                entry.health   = existing.health      # preserve last known health
            await self._store.save(entry)
        logger.debug("AgentRegistry: refreshed %d dynamic agents", len(entries))
