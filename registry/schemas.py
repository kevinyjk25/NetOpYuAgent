"""
registry/schemas.py
-------------------
All Pydantic models for the Agent Registry module.

Key types
---------
AgentSkill       — one skill declared by an agent (mirrors A2A AgentCard skill)
AgentEntry       — a fully-registered agent: card + derived assignments + health
AgentHealthState — live health status of a registered agent
RegistryConfig   — tunable thresholds (TTL, poll interval, etc.)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class AgentHealthState(str, Enum):
    UNKNOWN   = "unknown"
    HEALTHY   = "healthy"
    DEGRADED  = "degraded"
    UNHEALTHY = "unhealthy"


class RegistrationSource(str, Enum):
    STATIC   = "static"    # loaded from config file / env
    DYNAMIC  = "dynamic"   # auto-discovered via well-known URL
    SELF     = "self"      # this process registered itself


# ---------------------------------------------------------------------------
# Skill (mirrors A2A AgentCard skill object)
# ---------------------------------------------------------------------------

class AgentSkill(BaseModel):
    id:          str
    name:        str
    description: str = ""
    tags:        list[str] = Field(default_factory=list)
    examples:    list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Raw AgentCard (as returned by the remote /.well-known/agent-card.json)
# ---------------------------------------------------------------------------

class RawAgentCard(BaseModel):
    name:              str
    description:       str = ""
    version:           str = "0.0.0"
    protocol_version:  str = Field("0.3.0", alias="protocolVersion")
    url:               str
    skills:            list[AgentSkill] = Field(default_factory=list)
    capabilities:      dict[str, Any]   = Field(default_factory=dict)
    authentication:    dict[str, Any]   = Field(default_factory=dict)
    metadata:          dict[str, Any]   = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Registry entry (enriched, stored internally)
# ---------------------------------------------------------------------------

class AgentEntry(BaseModel):
    """
    One registered agent, enriched beyond the raw AgentCard.

    skill_index maps skill_id → AgentSkill for O(1) lookups.
    tag_index   maps tag     → list[skill_id] for tag-based queries.
    """
    agent_id:   str = Field(default_factory=lambda: str(uuid.uuid4()))
    card:       RawAgentCard
    source:     RegistrationSource = RegistrationSource.DYNAMIC
    health:     AgentHealthState   = AgentHealthState.UNKNOWN

    # Derived indices (computed on registration)
    skill_index: dict[str, AgentSkill]      = Field(default_factory=dict)
    tag_index:   dict[str, list[str]]       = Field(default_factory=dict)

    # Health tracking
    last_seen_at:     Optional[str] = None
    last_checked_at:  Optional[str] = None
    consecutive_failures: int = 0

    # Lifecycle
    registered_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    ttl_seconds: int = 300   # re-check interval

    def build_indices(self) -> None:
        """(Re)build skill_index and tag_index from card.skills."""
        self.skill_index = {s.id: s for s in self.card.skills}
        tag_idx: dict[str, list[str]] = {}
        for skill in self.card.skills:
            for tag in skill.tags:
                tag_idx.setdefault(tag, []).append(skill.id)
        self.tag_index = tag_idx

    def get_skill(self, skill_id: str) -> Optional[AgentSkill]:
        return self.skill_index.get(skill_id)

    def supports_skill(self, skill_id: str) -> bool:
        return skill_id in self.skill_index

    def skills_for_tag(self, tag: str) -> list[AgentSkill]:
        ids = self.tag_index.get(tag, [])
        return [self.skill_index[i] for i in ids if i in self.skill_index]

    @property
    def base_url(self) -> str:
        return self.card.url.rstrip("/")

    @property
    def is_available(self) -> bool:
        return self.health in (AgentHealthState.HEALTHY, AgentHealthState.DEGRADED)


# ---------------------------------------------------------------------------
# Resolution result (returned by registry.resolve())
# ---------------------------------------------------------------------------

class ResolutionResult(BaseModel):
    """
    What the registry gives back when a caller resolves a skill_id.
    Directly usable to build a task.schemas.AgentAssignment.
    """
    agent_id:  str
    agent_url: str
    skill_id:  str
    skill:     AgentSkill
    agent_name: str
    health:    AgentHealthState


# ---------------------------------------------------------------------------
# Registry config
# ---------------------------------------------------------------------------

class RegistryConfig(BaseModel):
    # Health check
    health_check_interval_seconds: int   = 60
    health_check_timeout_seconds:  float = 5.0
    max_consecutive_failures:      int   = 3     # mark UNHEALTHY after this

    # TTL
    agent_ttl_seconds: int = 300    # evict if not refreshed

    # Discovery
    discovery_timeout_seconds: float = 10.0

    # Load balancing strategy: "round_robin" | "least_loaded" | "random"
    lb_strategy: str = "round_robin"

    # Redis key prefix (when using Redis backend)
    redis_prefix: str = "registry"
