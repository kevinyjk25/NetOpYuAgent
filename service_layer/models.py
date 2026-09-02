"""Strict structured results shared by the service MCP servers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StrictResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    code: str = "ok"
    correlation_id: str
    observed_at: str
    simulation: bool = True


class UserRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str
    name: str
    department: str
    status: str


class UsersResult(StrictResult):
    users: list[UserRecord]


class UserResult(StrictResult):
    user: UserRecord


class ApplicationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app_id: str
    name: str
    address: str
    port: int
    path: str
    owner: str
    tier: str
    valid_roles: list[str]


class ApplicationsResult(StrictResult):
    applications: list[ApplicationRecord]


class ApplicationResult(StrictResult):
    application: ApplicationRecord


class EntitlementResult(StrictResult):
    user_id: str
    app_id: str
    roles: list[str]
    allowed: bool
    revision: int = Field(ge=0)


class PolicyEvaluationResult(StrictResult):
    user_id: str
    app_id: str
    eligible: bool
    reasons: list[str]
    current_roles: list[str]
    recommended_role: str | None


class EntitlementMutationResult(EntitlementResult):
    operation: str
    change_id: str
    idempotency_key: str
    replayed: bool = False


class ChangeResult(StrictResult):
    change_id: str
    status: str
    window_open: bool
    risk: str
    approved_by: str | None


class ChangeValidationResult(ChangeResult):
    permitted: bool
    reasons: list[str]


class EndpointBindingResult(StrictResult):
    subject_type: str
    subject_id: str
    endpoint: str | None
    address: str | None
    network_target_resolved: bool


class ServiceHealthResult(StrictResult):
    service: str
    environment: str
    status: str
    version: str
    replicas_ready: int = Field(ge=0)
    replicas_desired: int = Field(ge=0)
    revision: int = Field(ge=0)


class ServiceMutationResult(ServiceHealthResult):
    operation: str
    change_id: str
    idempotency_key: str
    replayed: bool = False


class OperationResult(StrictResult):
    operation: str
    target_id: str
    state: dict[str, Any]
    revision: int = Field(ge=0)
    change_id: str | None = None
    idempotency_key: str | None = None
    replayed: bool = False
