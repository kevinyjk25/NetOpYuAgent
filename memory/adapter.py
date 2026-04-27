"""
memory/adapter.py
─────────────────
Adapter that wraps the new agent_memory.MemoryManager (sync, multi-user)
and exposes the async interface the rest of the codebase already calls
(dtm.recall, dtm.after_turn, etc.).

Why an adapter?
  - agent_memory is a clean, well-tested, sync, multi-user module.
  - The runtime expects async methods on a 'memory router' object.
  - Wrapping rather than rewriting keeps the new module tested as-is.

Multi-user safety:
  - The new module is per-user-isolated by design (every method takes user_id).
  - This adapter resolves user_id from the operator JWT claims via a setter
    set by backend.py; falls back to "system" if no operator context is
    available (background tasks, watchdogs, internal callers).

Threading:
  - MemoryManager uses a SQLite WAL connection pool (thread-local connections).
  - Async wrappers run sync calls via asyncio.to_thread to avoid blocking
    the event loop.
"""
from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from agent_memory import MemoryManager
from agent_memory.schemas import MemoryFact, MemoryChunk

logger = logging.getLogger(__name__)


# ── Per-request operator context ──────────────────────────────────────────
# Set by backend.py at the start of each request from the verified JWT.
# Read by adapter methods to scope memory operations to the calling user.
_current_operator: ContextVar[Optional[str]] = ContextVar("current_operator", default=None)


def set_current_operator(operator_id: str) -> None:
    """Bind the calling operator's identity to this async task. Called by
    backend.py inside each authenticated route handler before invoking the
    runtime loop or memory writes."""
    _current_operator.set(operator_id)


def get_current_operator() -> str:
    """Return the calling operator's user_id; defaults to 'system' for
    background tasks (HITL watchdog, registry health checks, etc.)."""
    return _current_operator.get() or "system"


@dataclass
class RecallResult:
    """Mimics the previous DTM RecallResult shape — used by backend.py and
    hitl/a2a_integration.py to inject memory into LLM context.

    Track A = long-term raw chunks (FTS5 + TF-IDF retrieval)
    Track B = mid-term distilled facts (curated knowledge)
    """
    prompt_context: str
    fact_count:     int
    chunk_count:    int
    # Memory tab fields:
    results:        list = field(default_factory=list)  # serialized items
    track_a_count:  int  = 0     # = chunk_count, kept for FE clarity
    track_b_count:  int  = 0     # = fact_count, kept for FE clarity
    winner:         str  = ""    # "A" | "B" | "tie" | ""


class MemoryAdapter:
    """
    Async facade over agent_memory.MemoryManager. Provides the methods that
    runtime/loop.py, hitl/a2a_integration.py, and webui/backend.py expect:

        await adapter.recall(query, session_id, max_chars=1200)
        await adapter.after_turn(session_id, user_text, assistant_text, tool_calls)
        await adapter.cache_tool_result(session_id, tool_name, content)
        await adapter.read_cached(ref_id, offset, length)

    Each call resolves user_id from set_current_operator() and forwards to
    the underlying MemoryManager. SQL execution happens in a thread pool
    via asyncio.to_thread.
    """

    def __init__(
        self,
        data_dir:           str = "./data/memory",
        llm_fn:             Optional[Callable[[str], str]] = None,
        user_model_llm_fn:  Optional[Callable[[str, str], str]] = None,
        inline_threshold:   int = 4_000,
        session_ttl:        int = 86_400,
        enable_user_model:  bool = True,
    ) -> None:
        self._mgr = MemoryManager(
            data_dir          = data_dir,
            llm_fn            = llm_fn,
            user_model_llm_fn = user_model_llm_fn,
            inline_threshold  = inline_threshold,
            session_ttl       = session_ttl,
            enable_user_model = enable_user_model,
        )
        logger.info("MemoryAdapter ready — backend=%s", data_dir)

    # ── Recall (read path) ────────────────────────────────────────────────

    async def recall(
        self,
        query:      str,
        session_id: str,
        max_chars:  int = 1200,
    ) -> RecallResult:
        """Build memory context for a given query within the current
        operator's scope. Returns prompt-ready text plus counts."""
        user_id = get_current_operator()

        def _do() -> RecallResult:
            ctx_str = self._mgr.build_context(
                user_id    = user_id,
                query      = query,
                session_id = session_id,
                max_chars  = max_chars,
                include_user_profile = True,
                include_facts        = True,
                include_chunks       = True,
                include_skills       = True,
            )
            # search() returns {"long_term": RetrievalResult, "mid_term": RetrievalResult}
            search_out = self._mgr.search(
                user_id    = user_id,
                query      = query,
                session_id = session_id,
                top_k      = 5,
            )
            chunks_rr = search_out.get("long_term")
            facts_rr  = search_out.get("mid_term")
            chunk_items = chunks_rr.items if chunks_rr and hasattr(chunks_rr, "items") else []
            fact_items  = facts_rr.items  if facts_rr  and hasattr(facts_rr,  "items") else []

            # Serialize for the Memory tab. Each item is either a MemoryChunk
            # (long_term/Track A) or MemoryFact (mid_term/Track B).
            serialized = []
            for it in chunk_items:
                serialized.append({
                    "track":       "A",
                    "score":       round(getattr(it, "score", 0.0), 3),
                    "source":      getattr(it, "source", "conversation"),
                    "memory_type": "chunk",
                    "content":     (getattr(it, "text", "") or "")[:500],
                    "recency_ts":  getattr(it, "created_at", 0),
                    "tags":        list(getattr(it, "metadata", {}).get("tags", []))[:6],
                })
            for it in fact_items:
                serialized.append({
                    "track":       "B",
                    "score":       round(getattr(it, "score", getattr(it, "confidence", 1.0)), 3),
                    "source":      "facts",
                    "memory_type": getattr(it, "fact_type", "general"),
                    "content":     (getattr(it, "fact", "") or "")[:500],
                    "recency_ts":  getattr(it, "created_at", 0),
                    "tags":        [],
                })

            chunk_count = len(chunk_items)
            fact_count  = len(fact_items)
            if chunk_count > fact_count:
                winner = "A"
            elif fact_count > chunk_count:
                winner = "B"
            elif chunk_count > 0:
                winner = "tie"
            else:
                winner = ""

            return RecallResult(
                prompt_context = ctx_str,
                fact_count     = fact_count,
                chunk_count    = chunk_count,
                results        = serialized,
                track_a_count  = chunk_count,
                track_b_count  = fact_count,
                winner         = winner,
            )

        try:
            return await asyncio.to_thread(_do)
        except Exception as exc:
            logger.warning("MemoryAdapter.recall failed: %s", exc)
            return RecallResult(
                prompt_context="", fact_count=0, chunk_count=0,
                results=[], track_a_count=0, track_b_count=0, winner="",
            )

    # ── Write (after every completed turn) ────────────────────────────────

    async def after_turn(
        self,
        session_id:     str,
        user_text:      str,
        assistant_text: str,
        tool_calls:     Optional[list[dict]] = None,
        importance:     float = 0.5,
    ) -> list:
        """Persist the turn into all memory layers:
          - long-term chunk (raw turn text, hybrid search)
          - mid-term facts (LLM/rule-based extraction)
          - user profile update (behaviour modelling)

        Returns the list of new MemoryFact objects (subset of what was
        previously returned by the curator)."""
        user_id = get_current_operator()
        tool_calls = tool_calls or []

        def _do() -> list:
            new_facts: list = []
            try:
                # 1. Raw chunk → long-term FTS5+TF-IDF
                turn_text = f"User: {user_text}\nAssistant: {assistant_text}"
                self._mgr.remember(
                    user_id    = user_id,
                    session_id = session_id,
                    text       = turn_text,
                    source     = "conversation",
                    importance = importance,
                )
            except Exception as exc:
                logger.warning("MemoryAdapter.after_turn remember failed: %s", exc)

            try:
                # 2. Distill facts → mid-term
                new_facts = self._mgr.distill(
                    user_id    = user_id,
                    session_id = session_id,
                    text       = f"{user_text}\n{assistant_text}",
                )
            except Exception as exc:
                logger.warning("MemoryAdapter.after_turn distill failed: %s", exc)

            try:
                # 3. Update behaviour profile
                self._mgr.update_user_profile(
                    user_id        = user_id,
                    session_id     = session_id,
                    user_text      = user_text,
                    assistant_text = assistant_text,
                    tool_calls     = tool_calls,
                )
            except Exception as exc:
                logger.warning("MemoryAdapter.after_turn profile update failed: %s", exc)

            return new_facts

        return await asyncio.to_thread(_do)

    # ── Tool result cache (drop-in for ToolResultStore on the memory layer) ──

    async def cache_tool_result(
        self,
        session_id: str,
        tool_name:  str,
        content:    str,
    ) -> dict:
        """Cache a large tool output. Returns a dict with ref_id and preview
        for prompt injection."""
        user_id = get_current_operator()

        def _do() -> dict:
            entry = self._mgr.cache_tool_result(
                user_id    = user_id,
                session_id = session_id,
                tool_name  = tool_name,
                content    = content,
            )
            return {
                "ref_id":     entry.ref_id,
                "tool_name":  tool_name,
                "preview":    self._mgr.get_cache_preview(user_id, entry.ref_id),
                "total_size": entry.total_length,
            }

        return await asyncio.to_thread(_do)

    async def read_cached(
        self,
        ref_id: str,
        offset: int = 0,
        length: int = 2_000,
    ) -> dict:
        """Read a slice of a cached tool result by byte offset."""
        user_id = get_current_operator()

        def _do() -> dict:
            return self._mgr.read_cached(
                user_id = user_id,
                ref_id  = ref_id,
                offset  = offset,
                length  = length,
            )

        return await asyncio.to_thread(_do)

    # ── Stats / health ────────────────────────────────────────────────────

    async def stats(self) -> dict:
        user_id = get_current_operator()
        return await asyncio.to_thread(self._mgr.stats, user_id)


    # ── Backward-compatibility shims (so old curator/fts/dtm callers keep working) ──

    async def recall_for_session(self, query: str, session_id: str) -> str:
        """Old curator API — returns plain text context. Maps to recall().prompt_context."""
        result = await self.recall(query, session_id, max_chars=1200)
        return result.prompt_context

    async def get_stats(self) -> dict:
        """Old fts API — returns memory stats."""
        try:
            return await self.stats()
        except Exception:
            return {}

    def set_llm_fn(self, llm_fn: Callable[[str], str]) -> None:
        """
        Wire an LLM into the FactExtractor (and ReflectionEngine) AFTER the
        adapter is constructed. This enables LLM-driven fact distillation —
        without it, extraction falls back to English-only regex patterns and
        returns no facts for non-English conversations.

        Called by main.py once the LLM engine is built.
        """
        try:
            if hasattr(self._mgr, "extractor"):
                self._mgr.extractor._llm_fn = llm_fn
            if hasattr(self._mgr, "_reflector"):
                self._mgr._reflector._llm_fn = llm_fn
            logger.info("MemoryAdapter: LLM-driven fact extraction enabled")
        except Exception as exc:
            logger.warning("MemoryAdapter.set_llm_fn failed: %s", exc)

    async def list_sessions(self, limit: int = 50) -> list[dict]:
        """List sessions with metadata for the current operator."""
        user_id = get_current_operator()
        def _do() -> list[dict]:
            return self._mgr.long_term.list_sessions_with_meta(user_id, limit=limit)
        return await asyncio.to_thread(_do)

    async def get_session_history(self, session_id: str) -> list[dict]:
        """Return chronological chunks of a session for UI replay."""
        user_id = get_current_operator()
        def _do() -> list[dict]:
            return self._mgr.long_term.get_chunks_by_session(user_id, session_id)
        return await asyncio.to_thread(_do)

    # Old curator's _turn_counter / _shallow_n / _deep_n are no longer meaningful;
    # consolidation is handled internally by agent_memory.ConsolidationWorker.
    # Provide stub attributes so any leftover stats/health endpoints don't crash.
    @property
    def _turn_counter(self) -> dict:
        return {}

    @property
    def _shallow_n(self) -> int:
        return 5

    @property
    def _deep_n(self) -> int:
        return 20

    def close(self) -> None:
        """Close SQLite connections + WAL checkpoint. Called by lifespan
        shutdown."""
        try:
            self._mgr.close()
        except Exception as exc:
            logger.warning("MemoryAdapter close failed: %s", exc)
