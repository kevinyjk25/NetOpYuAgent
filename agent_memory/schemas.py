"""
agent_memory/schemas.py
Core data models. Pure dataclasses — no external dependencies.
"""
from __future__ import annotations
import uuid
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MAX_TEXT_LEN = 1_000_000   # 1 MB hard cap on any single text field
_MAX_FACT_LEN = 2_000
_MAX_TOOL_NAME_LEN = 128
_MAX_USER_ID_LEN = 256
_MAX_SESSION_ID_LEN = 256


def _now() -> float:
    return time.time()


def _uid() -> str:
    return uuid.uuid4().hex


def _validate_str(value: Any, name: str, max_len: int, allow_empty: bool = False) -> str:
    if value is None:
        raise ValueError(f"{name} must not be None")
    s = str(value).strip()
    if not allow_empty and not s:
        raise ValueError(f"{name} must not be empty")
    if len(s) > max_len:
        logger.warning("%s truncated from %d to %d chars", name, len(s), max_len)
        s = s[:max_len]
    return s


@dataclass
class MemoryChunk:
    """Raw text chunk — long-term memory (Claw-style)."""
    chunk_id: str = field(default_factory=_uid)
    user_id: str = ""
    session_id: str = ""
    text: str = ""
    source: str = "conversation"
    created_at: float = field(default_factory=_now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.user_id = _validate_str(self.user_id, "user_id", _MAX_USER_ID_LEN)
        self.session_id = _validate_str(self.session_id, "session_id", _MAX_SESSION_ID_LEN)
        self.text = _validate_str(self.text, "text", _MAX_TEXT_LEN)
        self.source = _validate_str(self.source, "source", 64, allow_empty=False)
        if not isinstance(self.metadata, dict):
            self.metadata = {}

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryChunk":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class MemoryFact:
    """Distilled fact — mid-term memory (Hermes-style)."""
    fact_id: str = field(default_factory=_uid)
    user_id: str = ""
    session_id: str = ""
    fact: str = ""
    fact_type: str = "general"
    confidence: float = 1.0
    created_at: float = field(default_factory=_now)
    source_chunk_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    _VALID_TYPES = frozenset(
        ["preference", "entity", "procedure", "lesson", "config", "env", "general"]
    )

    def __post_init__(self) -> None:
        self.user_id = _validate_str(self.user_id, "user_id", _MAX_USER_ID_LEN)
        self.session_id = _validate_str(self.session_id, "session_id", _MAX_SESSION_ID_LEN)
        self.fact = _validate_str(self.fact, "fact", _MAX_FACT_LEN)
        if self.fact_type not in self._VALID_TYPES:
            logger.warning("Unknown fact_type %r, defaulting to 'general'", self.fact_type)
            self.fact_type = "general"
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        if not isinstance(self.source_chunk_ids, list):
            self.source_chunk_ids = []
        if not isinstance(self.metadata, dict):
            self.metadata = {}

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryFact":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ToolResultEntry:
    """Cached tool output — short-term memory (P0-style)."""
    ref_id: str = field(default_factory=_uid)
    user_id: str = ""
    session_id: str = ""
    tool_name: str = ""
    content: str = ""          # empty after persisted; use ShortTermStore.read()
    total_length: int = 0
    created_at: float = field(default_factory=_now)
    expires_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.user_id = _validate_str(self.user_id, "user_id", _MAX_USER_ID_LEN)
        self.session_id = _validate_str(self.session_id, "session_id", _MAX_SESSION_ID_LEN)
        self.tool_name = _validate_str(self.tool_name, "tool_name", _MAX_TOOL_NAME_LEN)
        if not isinstance(self.metadata, dict):
            self.metadata = {}

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "ToolResultEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class RetrievalResult:
    """Unified retrieval result from any memory layer."""
    items: List[Any] = field(default_factory=list)
    layer: str = ""
    query: str = ""
    total_found: int = 0
    elapsed_ms: float = 0.0
