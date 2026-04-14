"""
memory/schemas.py
-----------------
All Pydantic models and enums for the Memory module.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class MemoryTier(str, Enum):
    REALTIME   = "realtime"    # LLM context window
    SHORT_TERM = "short_term"  # Redis, session-scoped
    MID_TERM   = "mid_term"    # Vector store, days-weeks
    LONG_TERM  = "long_term"   # PostgreSQL, permanent


class MemoryRecordType(str, Enum):
    TURN          = "turn"           # raw chat turn
    TOOL_CALL     = "tool_call"      # tool invocation + result
    SUMMARY       = "summary"        # LLM-generated summary
    ENTITY        = "entity"         # named entity / fact
    INCIDENT      = "incident"       # resolved IT incident
    USER_PREF     = "user_pref"      # operator preference
    TASK_RESULT   = "task_result"    # completed task output


class ConsolidationStatus(str, Enum):
    PENDING    = "pending"
    RUNNING    = "running"
    DONE       = "done"
    FAILED     = "failed"


# ---------------------------------------------------------------------------
# Core record
# ---------------------------------------------------------------------------

class MemoryRecord(BaseModel):
    record_id:    str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id:   str
    record_type:  MemoryRecordType
    tier:         MemoryTier
    content:      str                        # raw text / serialised JSON
    metadata:     dict[str, Any] = Field(default_factory=dict)
    embedding:    Optional[list[float]] = None
    importance:   float = 0.5               # 0.0 – 1.0
    access_count: int   = 0
    created_at:   str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    expires_at:   Optional[str] = None


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class RetrievalResult(BaseModel):
    record:    MemoryRecord
    score:     float          # blended recency × relevance
    tier:      MemoryTier
    retrieved_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class RetrievalQuery(BaseModel):
    query_text:  str
    session_id:  str
    top_k:       int   = 10
    tiers:       list[MemoryTier] = Field(
        default_factory=lambda: list(MemoryTier)
    )
    recency_weight:   float = 0.3
    relevance_weight: float = 0.7
    max_tokens:       int   = 2048


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------

class ConsolidationJob(BaseModel):
    job_id:     str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    status:     ConsolidationStatus = ConsolidationStatus.PENDING
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: Optional[str] = None
    records_processed: int = 0
    entities_extracted: int = 0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class MemoryConfig(BaseModel):
    # Short-term
    short_term_ttl_seconds:   int   = 86_400     # 24 h
    short_term_max_turns:     int   = 50

    # Mid-term
    mid_term_ttl_days:        int   = 30
    mid_term_top_k:           int   = 10
    mid_term_decay_factor:    float = 0.05        # score *= e^(-λ·days)
    mid_term_similarity_threshold: float = 0.72

    # Long-term
    long_term_min_importance: float = 0.75        # only promote above this

    # Retrieval blending
    recency_weight:   float = 0.3
    relevance_weight: float = 0.7

    # Token budget
    max_context_tokens: int = 3_000
    embedding_model:    str = "text-embedding-3-small"

    # Consolidation
    consolidation_min_turns: int = 5
