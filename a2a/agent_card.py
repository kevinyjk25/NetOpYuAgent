"""
a2a/agent_card.py
-----------------
AgentCard definition for the IT Ops / Monitoring Agent.
Exposes GET /.well-known/agent-card.json as required by the A2A spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


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

def get_agent_card(base_url: str) -> dict[str, Any]:
    """
    Return an A2A-compliant AgentCard dict.

    Args:
        base_url: The public URL at which this agent is reachable,
                  e.g. ``http://localhost:8000/api/v1/a2a``
    """
    return {
        "name": "IT Ops Monitoring Agent",
        "description": (
            "Intelligent IT operations agent specialising in alert analysis, "
            "incident management, trend prediction, and multi-dataset correlation."
        ),
        "version": "1.0.0",
        "protocolVersion": "0.3.0",
        "preferredTransport": "JSONRPC",
        "url": base_url,
        "defaultInputModes": ["text"],
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
        "skills": SKILLS,
    }
