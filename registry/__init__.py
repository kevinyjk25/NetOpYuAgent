"""
registry/__init__.py
--------------------
Public surface of the Agent Registry module.

Quick start
-----------
    from registry import create_registry, AgentRegistry, RegistryConfig

    registry = await create_registry(
        static_urls=[
            "http://agent-b.internal/api/v1/a2a",
            "http://predictor.internal/api/v1/a2a",
        ],
        redis_url="redis://localhost:6379",   # optional
    )
    await registry.start()

    # In TaskPlanner:
    result = await registry.resolve("alert_analysis")
    from task.schemas import AgentAssignment
    assignment = AgentAssignment(
        agent_id  = result.agent_id,
        agent_url = result.agent_url,
        skill_id  = result.skill_id,
    )
"""

from __future__ import annotations

from typing import Any, Optional

from .discovery import AgentDiscovery
from .registry import AgentRegistry
from registry.router import create_registry_router
from registry.schemas import (
    AgentEntry,
    AgentHealthState,
    AgentSkill,
    RawAgentCard,
    RegistrationSource,
    RegistryConfig,
    ResolutionResult,
)
from registry.store import InMemoryRegistryStore, RedisRegistryStore, RegistryStore


async def create_registry(
    static_urls: Optional[list[str]] = None,
    static_agents: Optional[list[dict]] = None,
    redis_url: Optional[str] = None,
    config: Optional[RegistryConfig] = None,
    own_card: Optional[dict] = None,
) -> AgentRegistry:
    """
    Factory: build and return a ready-to-use AgentRegistry.

    Parameters
    ----------
    static_urls:
        List of A2A base URLs to fetch AgentCards from at startup.
        Example: ["http://agent-b.svc/api/v1/a2a"]

    static_agents:
        List of static agent dicts for agents that don't serve AgentCard:
        [{"agent_id": "x", "agent_url": "...", "skill_ids": ["s1"]}]

    redis_url:
        If provided, uses RedisRegistryStore (shared across replicas).
        Falls back to InMemoryRegistryStore if None.

    config:
        Custom RegistryConfig. Defaults apply if None.

    own_card:
        If provided, self-registers this process's AgentCard immediately.
        Pass the dict returned by a2a.agent_card.get_agent_card(url).
    """
    cfg = config or RegistryConfig()

    # Build store
    if redis_url:
        try:
            import redis.asyncio as aioredis
            rc    = aioredis.from_url(redis_url)
            store = RedisRegistryStore(rc, cfg)
        except ImportError:
            import warnings
            warnings.warn("redis package not installed; using in-memory store")
            store = InMemoryRegistryStore()
    else:
        store = InMemoryRegistryStore()

    discovery = AgentDiscovery(cfg)
    registry  = AgentRegistry(store=store, config=cfg, discovery=discovery)

    # Self-register own card
    if own_card:
        await registry.self_register(own_card)

    # Fetch and register static URL-based agents
    if static_urls:
        await registry.register_from_urls(
            static_urls, source=RegistrationSource.STATIC
        )

    # Register purely static (no HTTP fetch) agents
    if static_agents:
        for spec in static_agents:
            await registry.register_static(
                agent_id=spec.get("agent_id", spec["agent_url"]),
                agent_url=spec["agent_url"],
                skill_ids=spec.get("skill_ids", []),
                agent_name=spec.get("agent_name", ""),
                skill_descriptions=spec.get("skill_descriptions", {}),
            )

    return registry


__all__ = [
    # Factory
    "create_registry",
    # Core
    "AgentRegistry",
    # Router
    "create_registry_router",
    # Schemas
    "AgentEntry", "AgentSkill", "RawAgentCard",
    "AgentHealthState", "RegistrationSource",
    "RegistryConfig", "ResolutionResult",
    # Store
    "RegistryStore", "InMemoryRegistryStore", "RedisRegistryStore",
    # Discovery
    "AgentDiscovery",
]
