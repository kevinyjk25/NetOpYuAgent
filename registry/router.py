"""
registry/router.py
------------------
FastAPI router exposing the Agent Registry over HTTP.

Mount in main.py:

    from registry.router import create_registry_router
    app.include_router(create_registry_router(registry), prefix="/registry")

Endpoints
---------
  GET  /registry/agents                     list all agents
  GET  /registry/agents/{agent_id}          get one agent
  POST /registry/agents                     manually register (static config)
  DELETE /registry/agents/{agent_id}        deregister
  GET  /registry/skills/{skill_id}/resolve  resolve best agent for skill
  GET  /registry/skills/{skill_id}/all      all agents for skill
  GET  /registry/tags/{tag}                 all agents for tag
  POST /registry/agents/{agent_id}/refresh  force re-fetch card
  GET  /registry/health                     registry health check
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .registry import AgentRegistry
from registry.schemas import AgentEntry, AgentHealthState, RegistrationSource

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class StaticRegisterRequest(BaseModel):
    agent_id:   Optional[str] = None
    agent_url:  str
    skill_ids:  list[str]
    agent_name: str = ""
    skill_descriptions: dict[str, str] = {}


class DynamicRegisterRequest(BaseModel):
    agent_url: str


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------

def create_registry_router(registry: AgentRegistry) -> APIRouter:
    api = APIRouter(tags=["Registry"])

    # ------------------------------------------------------------------
    # List / inspect
    # ------------------------------------------------------------------

    @api.get("/agents", summary="List all registered agents")
    async def list_agents() -> JSONResponse:
        agents = await registry.list_agents()
        return JSONResponse([_entry_summary(a) for a in agents])

    @api.get("/agents/{agent_id}", summary="Get one agent")
    async def get_agent(agent_id: str) -> JSONResponse:
        entry = await registry.get_agent(agent_id)
        if not entry:
            raise HTTPException(404, f"Agent {agent_id!r} not found")
        return JSONResponse(_entry_detail(entry))

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    @api.post("/agents", summary="Register agent (static or dynamic)")
    async def register_agent(body: DynamicRegisterRequest) -> JSONResponse:
        """
        Fetch the AgentCard from agent_url and register the agent.
        Returns the registered entry on success.
        """
        entry = await registry.register_from_url(
            body.agent_url, source=RegistrationSource.DYNAMIC
        )
        if entry is None:
            raise HTTPException(
                422, f"Could not discover AgentCard at {body.agent_url}"
            )
        return JSONResponse(_entry_summary(entry), status_code=201)

    @api.post("/agents/static", summary="Register agent from static config")
    async def register_static(body: StaticRegisterRequest) -> JSONResponse:
        entry = await registry.register_static(
            agent_id=body.agent_id or body.agent_url,
            agent_url=body.agent_url,
            skill_ids=body.skill_ids,
            agent_name=body.agent_name,
            skill_descriptions=body.skill_descriptions,
        )
        return JSONResponse(_entry_summary(entry), status_code=201)

    @api.delete("/agents/{agent_id}", summary="Deregister agent")
    async def deregister(agent_id: str) -> JSONResponse:
        entry = await registry.get_agent(agent_id)
        if not entry:
            raise HTTPException(404, f"Agent {agent_id!r} not found")
        await registry.deregister(agent_id)
        return JSONResponse({"deregistered": agent_id})

    @api.post("/agents/{agent_id}/refresh", summary="Force AgentCard refresh")
    async def refresh_agent(agent_id: str) -> JSONResponse:
        entry = await registry.get_agent(agent_id)
        if not entry:
            raise HTTPException(404, f"Agent {agent_id!r} not found")
        refreshed = await registry.register_from_url(
            entry.base_url, source=entry.source
        )
        if not refreshed:
            raise HTTPException(502, f"Failed to refresh AgentCard from {entry.base_url}")
        return JSONResponse(_entry_summary(refreshed))

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    @api.get("/skills/{skill_id}/resolve", summary="Resolve best agent for skill")
    async def resolve_skill(
        skill_id: str,
        exclude: Optional[str] = None,
    ) -> JSONResponse:
        """
        Returns one ResolutionResult using the configured LB strategy.
        Optional ?exclude=agent_id1,agent_id2 to skip specific agents.
        """
        excl = exclude.split(",") if exclude else []
        result = await registry.resolve(skill_id, exclude_agent_ids=excl)
        if result is None:
            raise HTTPException(
                503, f"No healthy agent available for skill {skill_id!r}"
            )
        return JSONResponse(result.model_dump())

    @api.get("/skills/{skill_id}/all", summary="All agents for skill")
    async def all_for_skill(skill_id: str) -> JSONResponse:
        results = await registry.resolve_all(skill_id)
        return JSONResponse([r.model_dump() for r in results])

    @api.get("/tags/{tag}", summary="All agents for tag")
    async def all_for_tag(tag: str) -> JSONResponse:
        results = await registry.resolve_by_tag(tag)
        return JSONResponse([r.model_dump() for r in results])

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @api.get("/health", summary="Registry health check")
    async def health() -> JSONResponse:
        agents = await registry.list_agents()
        total   = len(agents)
        healthy = sum(1 for a in agents if a.health == AgentHealthState.HEALTHY)
        degraded = sum(1 for a in agents if a.health == AgentHealthState.DEGRADED)
        unhealthy = total - healthy - degraded
        return JSONResponse({
            "status": "ok",
            "agents": {"total": total, "healthy": healthy,
                       "degraded": degraded, "unhealthy": unhealthy},
        })

    return api


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _entry_summary(entry: AgentEntry) -> dict:
    return {
        "agent_id":  entry.agent_id,
        "name":      entry.card.name,
        "url":       entry.base_url,
        "health":    entry.health.value,
        "source":    entry.source.value,
        "skills":    list(entry.skill_index.keys()),
        "tags":      list(entry.tag_index.keys()),
        "registered_at": entry.registered_at,
    }


def _entry_detail(entry: AgentEntry) -> dict:
    summary = _entry_summary(entry)
    summary["version"]          = entry.card.version
    summary["description"]      = entry.card.description
    summary["capabilities"]     = entry.card.capabilities
    summary["last_seen_at"]     = entry.last_seen_at
    summary["last_checked_at"]  = entry.last_checked_at
    summary["consecutive_failures"] = entry.consecutive_failures
    summary["skill_details"]    = [s.model_dump() for s in entry.card.skills]
    return summary
