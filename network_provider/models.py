"""Structured cross-process evidence returned by Network Observer MCP."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class NetworkEvidenceEnvelope(BaseModel):
    """Authenticated semantic payload plus freshness and integrity metadata."""

    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    code: str = "ok"
    correlation_id: str
    observed_at: str
    simulation: bool
    provider_identity: str
    capability_id: str
    capability_version: str
    payload_digest: str
    content_type: str
    payload: Any
