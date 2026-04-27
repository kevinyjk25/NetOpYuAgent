"""
agent_memory/session_state.py
==============================
SessionState — Dual-Track Memory for a single agent session.

Inspired by NetOpYuAgent's ContextBudgetManager + working_set design.

Two tracks, fundamentally different semantics:

  HOT TRACK (in-memory, ephemeral, this session only):
    confirmed_facts  — truths verified by tool calls; highest budget priority
    working_set      — entities/devices actively being discussed this turn
    recent_tool_results — small inline tool outputs from recent calls

  COLD TRACK (persistent, DB-backed, cross-session):
    long_term_chunks — retrieved from LongTermStore (FTS5 + TF-IDF)
    mid_term_facts   — retrieved from MidTermStore (FTS5 + TF-IDF)
    user_profile     — from UserModelEngine (behavioral model)

Why this split matters:
  Hot track items are ALWAYS included first (they're small, always relevant).
  Cold track items fill remaining budget via similarity search.
  Without this split, every turn pays DB retrieval cost even for obvious facts
  that were just confirmed moments ago in the same conversation.

MMR Retrieval (Maximal Marginal Relevance):
  Cold track uses MMR to balance relevance vs diversity.
  Formula: score = λ · sim(q, d) − (1−λ) · max_sim(d, selected)
  Prevents returning 5 near-identical chunks about the same event.
  λ=1.0 → pure relevance (like current TF-IDF); λ=0.0 → pure diversity.
"""
from __future__ import annotations

import logging
import math
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class ConfirmedFact:
    """A fact that has been verified by a tool call or explicit agent confirmation."""
    text:       str
    source:     str   = "manual"      # "tool:<name>", "manual", "llm_verified"
    confidence: float = 1.0
    added_at:   float = field(default_factory=time.time)

    def format(self) -> str:
        return f"[{self.source}] {self.text}"


@dataclass
class WorkingSetEntry:
    """An entity/device/service currently being actively debugged or discussed."""
    entity_id:   str
    label:       str
    entity_type: str  = "device"     # "device", "service", "user", "config", etc.
    metadata:    Dict[str, Any] = field(default_factory=dict)
    added_at:    float = field(default_factory=time.time)

    def format(self) -> str:
        meta = f" ({', '.join(f'{k}={v}' for k, v in self.metadata.items())})" \
               if self.metadata else ""
        return f"{self.label} [{self.entity_type}]{meta}"


@dataclass
class RecentToolResult:
    """A small tool result kept inline (large ones go to ShortTermStore)."""
    tool_name: str
    content:   str     # only kept if len(content) <= inline_threshold
    ref_id:    Optional[str] = None   # set if content is in ShortTermStore
    added_at:  float = field(default_factory=time.time)

    def format(self, max_chars: int = 500) -> str:
        if self.ref_id:
            return f"[tool:{self.tool_name}] [STORED:{self.ref_id}]"
        return f"[tool:{self.tool_name}] {self.content[:max_chars]}"


# ── SessionState ──────────────────────────────────────────────────────────────

class SessionState:
    """
    Per-session dual-track memory state.

    One instance per active (user_id, session_id). Lives in memory only.
    Thread-safe (a session can be accessed from multiple threads in async agents).

    Usage:
        state = SessionState(user_id="alice", session_id="s1")

        # Hot track writes
        state.confirm_fact("R1 is reachable", source="tool:ping")
        state.add_to_working_set("R1", "Router R1", "device", {"ip": "10.0.0.1"})
        state.add_tool_result("ping", "PING OK 10.0.0.1 rtt=2ms")

        # Context building (used by MemoryManager.build_context_budgeted)
        hot = state.build_hot_context()

        # Access for budget manager
        state.confirmed_facts  → List[ConfirmedFact]
        state.working_set      → List[WorkingSetEntry]
        state.recent_results   → List[RecentToolResult]
    """

    def __init__(
        self,
        user_id:    str,
        session_id: str,
        max_confirmed_facts:  int = 50,
        max_working_set:      int = 20,
        max_recent_results:   int = 10,
        inline_threshold:     int = 800,   # chars; larger → ShortTermStore
    ) -> None:
        self.user_id    = user_id
        self.session_id = session_id
        self._max_cf    = max_confirmed_facts
        self._max_ws    = max_working_set
        self._max_tr    = max_recent_results
        self._inline    = inline_threshold
        self._lock      = threading.Lock()

        self._confirmed_facts: List[ConfirmedFact]   = []
        self._working_set:     List[WorkingSetEntry] = []
        self._recent_results:  List[RecentToolResult] = []
        self._turn_count: int = 0

    # ── hot track writes ──────────────────────────────────────────────────────

    def confirm_fact(
        self,
        text:       str,
        source:     str   = "manual",
        confidence: float = 1.0,
    ) -> ConfirmedFact:
        """
        Record a verified fact. Highest budget priority — always injected.
        Call after tool verification confirms something is true/false.
        Example: state.confirm_fact("R1 ping OK", source="tool:ping")
        """
        cf = ConfirmedFact(text=text, source=source, confidence=confidence)
        with self._lock:
            # Deduplicate: skip if same text already confirmed
            if not any(f.text == text for f in self._confirmed_facts):
                self._confirmed_facts.append(cf)
                # Trim to cap (keep most recent)
                if len(self._confirmed_facts) > self._max_cf:
                    self._confirmed_facts = self._confirmed_facts[-self._max_cf:]
        return cf

    def retract_fact(self, text: str) -> bool:
        """Remove a confirmed fact (e.g., after contradicting evidence)."""
        with self._lock:
            before = len(self._confirmed_facts)
            self._confirmed_facts = [f for f in self._confirmed_facts if f.text != text]
            return len(self._confirmed_facts) < before

    def add_to_working_set(
        self,
        entity_id:   str,
        label:       str,
        entity_type: str = "device",
        metadata:    Optional[Dict[str, Any]] = None,
    ) -> WorkingSetEntry:
        """
        Add an entity to the active working set.
        Working set = things being currently debugged / discussed.
        Idempotent: adding the same entity_id updates its metadata.
        """
        entry = WorkingSetEntry(
            entity_id=entity_id, label=label,
            entity_type=entity_type, metadata=metadata or {}
        )
        with self._lock:
            # Update if exists
            for i, ws in enumerate(self._working_set):
                if ws.entity_id == entity_id:
                    self._working_set[i] = entry
                    return entry
            self._working_set.append(entry)
            if len(self._working_set) > self._max_ws:
                self._working_set = self._working_set[-self._max_ws:]
        return entry

    def remove_from_working_set(self, entity_id: str) -> bool:
        with self._lock:
            before = len(self._working_set)
            self._working_set = [e for e in self._working_set if e.entity_id != entity_id]
            return len(self._working_set) < before

    def add_tool_result(
        self,
        tool_name: str,
        content:   str,
        ref_id:    Optional[str] = None,
    ) -> RecentToolResult:
        """
        Track a tool result for hot-track injection.
        Large results: pass ref_id (from ShortTermStore) so we inject the reference token.
        Small results: pass content directly for inline inclusion.
        """
        tr = RecentToolResult(
            tool_name=tool_name,
            content=content if len(content) <= self._inline else "",
            ref_id=ref_id,
        )
        with self._lock:
            self._recent_results.append(tr)
            if len(self._recent_results) > self._max_tr:
                self._recent_results = self._recent_results[-self._max_tr:]
        return tr

    def increment_turn(self) -> int:
        with self._lock:
            self._turn_count += 1
            return self._turn_count

    # ── hot track reads ───────────────────────────────────────────────────────

    @property
    def confirmed_facts(self) -> List[ConfirmedFact]:
        with self._lock:
            return list(self._confirmed_facts)

    @property
    def working_set(self) -> List[WorkingSetEntry]:
        with self._lock:
            return list(self._working_set)

    @property
    def recent_results(self) -> List[RecentToolResult]:
        with self._lock:
            return list(self._recent_results)

    @property
    def turn_count(self) -> int:
        with self._lock:
            return self._turn_count

    def build_hot_context(self, max_chars: int = 1_200) -> Dict[str, str]:
        """
        Format each hot-track section as a string, respecting max_chars per section.
        Returns dict: section_name → formatted_string
        """
        result: Dict[str, str] = {}

        cfs = self.confirmed_facts
        if cfs:
            lines = ["## Confirmed Facts (this session)"]
            for cf in cfs:
                lines.append(f"  • {cf.format()}")
            result["confirmed_facts"] = "\n".join(lines)

        ws = self.working_set
        if ws:
            lines = ["## Working Set (active entities)"]
            for entry in ws:
                lines.append(f"  • {entry.format()}")
            result["working_set"] = "\n".join(lines)

        tr = self.recent_results
        if tr:
            lines = ["## Recent Tool Results"]
            used = len(lines[0])
            for r in reversed(tr):   # most recent first
                line = f"  • {r.format()}"
                if used + len(line) > max_chars:
                    break
                lines.append(line)
                used += len(line)
            if len(lines) > 1:
                result["recent_tools"] = "\n".join(lines)

        return result

    def clear(self) -> None:
        """Reset all hot-track state (call at session end)."""
        with self._lock:
            self._confirmed_facts.clear()
            self._working_set.clear()
            self._recent_results.clear()
            self._turn_count = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id":         self.user_id,
            "session_id":      self.session_id,
            "turn_count":      self._turn_count,
            "confirmed_facts": [{"text": f.text, "source": f.source,
                                  "confidence": f.confidence}
                                 for f in self._confirmed_facts],
            "working_set":     [{"entity_id": e.entity_id, "label": e.label,
                                  "entity_type": e.entity_type}
                                 for e in self._working_set],
            "recent_results":  [{"tool": r.tool_name, "has_content": bool(r.content),
                                  "ref_id": r.ref_id}
                                 for r in self._recent_results],
        }


# ── MMR Retrieval helper ──────────────────────────────────────────────────────

def mmr_rerank(
    candidates: List[Tuple[str, float]],   # (text, relevance_score)
    top_k:      int   = 5,
    lambda_:    float = 0.6,               # 1.0 = pure relevance, 0.0 = pure diversity
) -> List[Tuple[str, float]]:
    """
    Maximal Marginal Relevance reranking.

    Balances relevance to the query against diversity within the result set.
    Formula per step: argmax_d [ λ·rel(d,q) − (1−λ)·max_{s∈S} sim(d,s) ]

    Args:
        candidates: list of (text, relevance_score) sorted by relevance desc
        top_k:      number of results to return
        lambda_:    balance parameter (0.6 = slight relevance bias)

    Returns:
        Reranked list of (text, mmr_score)
    """
    if not candidates or lambda_ >= 1.0:
        return candidates[:top_k]

    # Precompute simple token-overlap similarity matrix
    def _tok_sim(a: str, b: str) -> float:
        ta = set(_tokenize(a))
        tb = set(_tokenize(b))
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)   # Jaccard

    selected: List[Tuple[str, float]] = []
    remaining = list(candidates)

    while remaining and len(selected) < top_k:
        best_score = -float("inf")
        best_item  = remaining[0]

        for text, rel in remaining:
            # Diversity penalty: similarity to already selected items
            max_sim_to_selected = max(
                (_tok_sim(text, s_text) for s_text, _ in selected),
                default=0.0,
            )
            mmr_score = lambda_ * rel - (1 - lambda_) * max_sim_to_selected
            if mmr_score > best_score:
                best_score = mmr_score
                best_item  = (text, mmr_score)

        selected.append(best_item)
        remaining = [(t, r) for t, r in remaining if t != best_item[0]]

    return selected


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9\u4e00-\u9fff]+", text.lower())


# ── SessionStateRegistry ──────────────────────────────────────────────────────

class SessionStateRegistry:
    """
    Process-level registry of active SessionState objects.
    MemoryManager holds one of these; users look up their session by (user_id, session_id).
    """

    def __init__(self) -> None:
        self._states: Dict[Tuple[str, str], SessionState] = {}
        self._lock = threading.Lock()

    def get_or_create(
        self,
        user_id:    str,
        session_id: str,
        **kwargs: Any,
    ) -> SessionState:
        key = (user_id, session_id)
        with self._lock:
            if key not in self._states:
                self._states[key] = SessionState(
                    user_id=user_id, session_id=session_id, **kwargs
                )
            return self._states[key]

    def get(self, user_id: str, session_id: str) -> Optional[SessionState]:
        return self._states.get((user_id, session_id))

    def drop(self, user_id: str, session_id: str) -> None:
        """Remove and clear a session state (call at session end)."""
        key = (user_id, session_id)
        with self._lock:
            state = self._states.pop(key, None)
        if state:
            state.clear()

    def active_sessions(self, user_id: str) -> List[str]:
        with self._lock:
            return [sid for uid, sid in self._states if uid == user_id]

    @property
    def total_active(self) -> int:
        return len(self._states)
