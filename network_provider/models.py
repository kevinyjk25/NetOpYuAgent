"""Structured cross-process evidence returned by Network Observer MCP."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class NetworkEvidenceEnvelope(BaseModel):
    """Identity-bound semantic payload plus freshness and integrity metadata."""

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


class ActorEffectResult(BaseModel):
    """Durable Network Actor result; Runtime still decides terminal success."""

    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    code: str = "ok"
    operation_id: str
    plan_hash: str
    capability_id: str
    capability_version: str
    target_key: str
    fence_token: int
    state: str
    replayed: bool
    reconciled: bool
    snapshot_digest: str
    audit_chain_valid: bool
    simulation: bool
    payload: Any = None


class ActorFinalizeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    operation_id: str
    state: str
    runtime_terminal_state: str
    audit_chain_valid: bool
