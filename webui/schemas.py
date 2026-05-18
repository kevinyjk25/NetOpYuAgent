"""
webui/schemas.py — Request / response pydantic models for the WebUI.

EXTRACTED FROM webui/backend.py during the audit hotfix. Lives in its own
dependency-free module so that route extraction files (routes_hitl.py,
routes_system.py, etc.) can import these at module level WITHOUT creating
a circular import via webui/backend.py.

Why module-level matters: FastAPI resolves `req: HitlDecisionRequest`
annotations by looking up the name in the *enclosing module's globals*.
If the model is imported only inside a function body (late import for
circular-import-avoidance), FastAPI can't find it, falls back to treating
`req` as a query parameter, and every POST returns 422
    {"loc": ("query", "req"), "msg": "Field required"}.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    query:           str            = Field(..., min_length=1, max_length=8_000,
                                           description="User query — max 8 000 chars")
    session_id:      Optional[str]  = Field(None, pattern=r"^[a-zA-Z0-9_-]{1,128}$",
                                           description="Session ID — 1-128 chars of [a-zA-Z0-9_-]")
    confirmed_facts: list[str]      = Field(default_factory=list, max_length=60,
                                           description="Carry-forward facts — max 60 items")
    working_set:     list[dict]     = Field(default_factory=list, max_length=20)
    env_context:     dict           = Field(default_factory=dict)
    delegation_mode: str            = Field("fresh", pattern=r"^(fresh|forked)$")

    @field_validator("confirmed_facts")
    @classmethod
    def cap_fact_length(cls, v: list[str]) -> list[str]:
        """Prevent individual facts from inflating the LLM context."""
        return [f[:500] for f in v]

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        return v.strip()


class ToolCallRequest(BaseModel):
    args: dict[str, Any] = {}


class HitlDecisionRequest(BaseModel):
    operator_id:     str = "webui-operator"
    comment:         Optional[str] = None
    parameter_patch: Optional[dict] = None
    # For DecisionKind.CHOOSE — id of the operator-picked option
    selected_choice_id: Optional[str] = None
    # For DecisionKind.ANSWER — operator's answers to clarification fields
    clarification_answers: Optional[dict[str, str]] = None


__all__ = [
    "ChatRequest",
    "ToolCallRequest",
    "HitlDecisionRequest",
]
