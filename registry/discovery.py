"""
registry/discovery.py
---------------------
AgentDiscovery — fetches and parses AgentCards from remote agents.

Two discovery modes
-------------------
  Static   : pre-configured list of URLs loaded at startup
  Dynamic  : periodic sweep of known URLs to refresh cards

The discovery service produces AgentEntry objects ready for the store.
It does NOT write to the store — that is the registry's job.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from .schemas import (
    AgentEntry,
    AgentHealthState,
    AgentSkill,
    RawAgentCard,
    RegistrationSource,
    RegistryConfig,
)

logger = logging.getLogger(__name__)

_WELL_KNOWN_PATHS = [
    "/.well-known/agent-card.json",
    "/agent-card",
    "/api/v1/a2a/.well-known/agent-card.json",
    "/api/v1/a2a/agent-card",
]


class AgentDiscovery:
    """
    Fetches and parses A2A AgentCards from remote URLs.

    Usage
    -----
        discovery = AgentDiscovery(config)
        entry = await discovery.fetch("http://agent-b.internal/api/v1/a2a")
        if entry:
            await store.save(entry)
    """

    def __init__(self, config: Optional[RegistryConfig] = None) -> None:
        self._cfg     = config or RegistryConfig()
        self._timeout = self._cfg.discovery_timeout_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch(
        self,
        base_url: str,
        source: RegistrationSource = RegistrationSource.DYNAMIC,
    ) -> Optional[AgentEntry]:
        """
        Attempt to fetch an AgentCard from *base_url*.

        Tries each of _WELL_KNOWN_PATHS in order and returns the first
        successful parse, or None on complete failure.
        """
        base = base_url.rstrip("/")
        card_dict: Optional[dict] = None

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for path in _WELL_KNOWN_PATHS:
                url = base + path
                try:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        card_dict = resp.json()
                        logger.info(
                            "AgentDiscovery: fetched AgentCard from %s", url
                        )
                        break
                except Exception as exc:
                    logger.debug("AgentDiscovery: %s failed: %s", url, exc)

        if card_dict is None:
            logger.warning(
                "AgentDiscovery: no AgentCard found at %s (tried %d paths)",
                base, len(_WELL_KNOWN_PATHS),
            )
            return None

        return self._parse(card_dict, base_url=base, source=source)

    async def fetch_many(
        self,
        urls: list[str],
        source: RegistrationSource = RegistrationSource.STATIC,
    ) -> list[AgentEntry]:
        """Fetch multiple agents concurrently."""
        import asyncio
        results = await asyncio.gather(
            *[self.fetch(url, source) for url in urls],
            return_exceptions=True,
        )
        entries = []
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                logger.error("AgentDiscovery: fetch failed for %s: %s", url, result)
            elif result is not None:
                entries.append(result)
        return entries

    async def health_check(self, entry: AgentEntry) -> AgentHealthState:
        """
        Perform a lightweight health check on a registered agent.

        Tries GET /health first; falls back to GET /agent-card (proves reachable).
        """
        base = entry.base_url
        health_urls = [
            base + "/health",
            base + "/api/v1/a2a/health",
            base + _WELL_KNOWN_PATHS[0],
        ]
        async with httpx.AsyncClient(timeout=self._cfg.health_check_timeout_seconds) as client:
            for url in health_urls:
                try:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        return AgentHealthState.HEALTHY
                    if resp.status_code in (429, 503):
                        return AgentHealthState.DEGRADED
                except Exception:
                    continue
        return AgentHealthState.UNHEALTHY

    # ------------------------------------------------------------------
    # Internal parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse(
        raw: dict[str, Any],
        base_url: str,
        source: RegistrationSource,
    ) -> AgentEntry:
        """Convert a raw AgentCard dict into a typed AgentEntry."""

        # Normalise skill objects
        skills: list[AgentSkill] = []
        for s in raw.get("skills", []):
            skills.append(AgentSkill(
                id=s.get("id", s.get("name", "unknown")),
                name=s.get("name", s.get("id", "unknown")),
                description=s.get("description", ""),
                tags=s.get("tags", []),
                examples=s.get("examples", []),
            ))

        # Use card's url if present; else fall back to base_url we fetched from
        card_url = raw.get("url", base_url)

        card = RawAgentCard(
            name=raw.get("name", "unknown-agent"),
            description=raw.get("description", ""),
            version=raw.get("version", "0.0.0"),
            protocolVersion=raw.get("protocolVersion", "0.3.0"),
            url=card_url,
            skills=skills,
            capabilities=raw.get("capabilities", {}),
            authentication=raw.get("authentication", {}),
            metadata=raw.get("metadata", {}),
        )

        entry = AgentEntry(card=card, source=source, health=AgentHealthState.UNKNOWN)
        entry.build_indices()
        return entry
