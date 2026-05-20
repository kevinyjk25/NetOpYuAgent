"""
a2a/agent_card.py
-----------------
AgentCard definition for the IT Ops / Monitoring Agent.
Exposes GET /.well-known/agent-card.json as required by the A2A spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Skill definitions
# ---------------------------------------------------------------------------

SKILLS: list[dict[str, Any]] = [
    {
        "id": "alert_analysis",
        "name": "alert_analysis",
        "description": "Analyse IT alerts by count, severity, source, and time range",
        "tags": ["alert", "monitoring", "ops"],
        "examples": [
            "Which service triggered the most alerts in the last 7 days?",
            "Show critical alerts grouped by severity for this month",
        ],
    },
    {
        "id": "incident_management",
        "name": "incident_management",
        "description": "Query and summarise incident / ticket data, SLA status, and MTTR",
        "tags": ["incident", "ticket", "sla", "ops"],
        "examples": [
            "How many P1 incidents occurred last week?",
            "What is the average resolution time for database incidents?",
        ],
    },
    {
        "id": "trend_prediction",
        "name": "trend_prediction",
        "description": "Predict future alert or incident trends using historical monitoring data",
        "tags": ["prediction", "trend", "forecast", "monitoring"],
        "examples": [
            "Predict alert volume for next week based on the last 3 months",
            "Is the disk usage trend likely to cause issues in 30 days?",
        ],
    },
    {
        "id": "multi_dataset_analysis",
        "name": "multi_dataset_analysis",
        "description": "Cross-correlate alerts, metrics, logs, and ticket data for root-cause analysis",
        "tags": ["rca", "correlation", "multi-source", "ops"],
        "examples": [
            "Correlate network latency spikes with alert surges last Tuesday",
            "Which applications have both high error rates and open incidents?",
        ],
    },
]


# ---------------------------------------------------------------------------
# AgentCard builder
# ---------------------------------------------------------------------------

def get_agent_card(
    base_url: str,
    identity: "Optional[AgentIdentityConfigLike]" = None,
) -> dict[str, Any]:
    """
    Return an A2A-compliant AgentCard dict.

    Args:
        base_url: The public URL at which this agent is reachable,
                  e.g. ``http://localhost:8000/api/v1/a2a``
        identity: Optional AgentIdentityConfig. When provided, the card
                  uses identity.agent_id / display_name / description /
                  capabilities. When None (legacy path), the static
                  fields below + the module-level SKILLS list are used —
                  preserves the original single-agent behaviour.

    Why optional: every existing caller still works without passing
    identity (backwards compat). New code (Phase 1 multi-agent) passes
    cfg.agent so each process advertises its own id and capabilities.
    """
    # Resolve identity-driven fields with fallbacks to legacy defaults.
    if identity is not None and getattr(identity, "agent_id", None):
        # Use the configured identity. We still keep a stable "name" for
        # the AgentCard — A2A spec uses `name` as a display field — and
        # carry agent_id separately in metadata so peers can resolve it.
        card_name        = identity.display_name or identity.agent_id
        card_description = identity.description or _LEGACY_DESCRIPTION
        agent_id         = identity.agent_id

        # Build skills list from identity.capabilities if any, else fall
        # back to legacy SKILLS so historic deployments don't lose them.
        if identity.capabilities:
            skills = [
                {
                    "id":          cap.skill_id,
                    "name":        cap.name or cap.skill_id,
                    "description": cap.description,
                    "tags":        list(cap.tags),
                    "examples":    list(cap.examples),
                }
                for cap in identity.capabilities
            ]
        else:
            skills = SKILLS
    else:
        # Legacy single-agent path.
        card_name        = "IT Ops Monitoring Agent"
        card_description = _LEGACY_DESCRIPTION
        agent_id         = "default-agent"
        skills           = SKILLS

    return {
        "name":             card_name,
        "description":      card_description,
        "version":          "1.0.0",
        "protocolVersion":  "0.3.0",
        "preferredTransport": "JSONRPC",
        "url":              base_url,
        "defaultInputModes":  ["text"],
        "defaultOutputModes": ["text"],
        "capabilities": {
            "streaming": True,
            "pushNotifications": True,
            "stateTransitionHistory": True,
        },
        "authentication": {
            "schemes": ["Bearer"],
            "description": "Pass a JWT obtained from your OAuth2 / OIDC provider.",
        },
        # `agent_id` is OUR addition (not in the A2A spec) — peers use it
        # as the registry key so two LAN-agent replicas don't collide.
        # Lives in the top-level so it's easy to grep in network captures.
        "agent_id":         agent_id,
        "skills":           skills,
    }


_LEGACY_DESCRIPTION = (
    "Intelligent IT operations agent specialising in alert analysis, "
    "incident management, trend prediction, and multi-dataset correlation."
)


# `AgentIdentityConfigLike` is a structural type hint — accept anything
# with the right attributes so we don't have to import config.py here
# (a2a/ stays as a self-contained protocol module).
try:
    from typing import Protocol

    class AgentIdentityConfigLike(Protocol):
        agent_id:     str
        display_name: str
        description:  str
        capabilities: list
except ImportError:
    AgentIdentityConfigLike = object  # type: ignore
