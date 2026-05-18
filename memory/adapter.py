"""
memory/adapter.py
─────────────────
Thin async adapter over agent_memory.MemoryManager.

Responsibilities (kept here):
  - ContextVar for per-request operator binding
  - asyncio.to_thread wrapping for sync MemoryManager methods
  - Importance-based write fanout (chunk-only / +distill / +profile)
  - Backward-compat shims for old curator/dtm/fts call sites

Algorithms that USED to live here have moved into agent_memory:
  - Dual-track recall + MMR + type_boost  →  agent_memory.retrieval.recall_orchestrator.recall
  - Periodic nudge + contradiction routing →  agent_memory.retrieval.recall_orchestrator.run_nudge
  - RecallResult dataclass                  →  agent_memory.retrieval.recall_orchestrator.RecallResult

This split keeps the algorithm layer self-contained inside agent_memory
(callable from any caller, with no adapter dependency) and leaves this
file as protocol/IO glue only.

Multi-user safety:
  Operator id is resolved from a ContextVar set by backend.py on each
  authenticated request. Background tasks that don't bind one use the
  "system" default. Every MemoryManager call takes user_id explicitly.

Threading:
  MemoryManager uses a SQLite WAL connection pool. Sync calls run via
  asyncio.to_thread to avoid blocking the event loop.
"""
from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar
from typing import Any, Callable, Optional

from agent_memory import MemoryManager
from agent_memory.retrieval.recall_orchestrator import (
    RecallResult,
    recall as _orch_recall,
    run_nudge as _orch_run_nudge,
    should_nudge as _orch_should_nudge,
)

logger = logging.getLogger(__name__)


# ── Per-request operator context ──────────────────────────────────────────
# Set by backend.py at the start of each request from the verified JWT.
# Read by adapter methods to scope memory operations to the calling user.
_current_operator: ContextVar[Optional[str]] = ContextVar(
    "current_operator", default=None
)


def set_current_operator(operator_id: str) -> None:
    """Bind the calling operator's identity to this async task. Called by
    backend.py inside each authenticated route handler before invoking the
    runtime loop or memory writes."""
    _current_operator.set(operator_id)


def get_current_operator() -> str:
    """Return the operator id bound to the current async task, or the
    'system' default for background tasks / unauthenticated callers."""
    return _current_operator.get() or "system"


# Re-export RecallResult for callers that import it from memory.adapter
__all__ = [
    "MemoryAdapter",
    "RecallResult",
    "set_current_operator",
    "get_current_operator",
]


class MemoryAdapter:
    """
    Async facade over agent_memory.MemoryManager.

    All public methods are async; they wrap sync MemoryManager calls in
    asyncio.to_thread so the event loop stays responsive while SQLite
    queries run.
    """

    def __init__(
        self,
        data_dir:           str = "./data/memory",
        llm_fn:             Optional[Callable[..., str]] = None,
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
        # Per-session turn counter for nudge scheduling. Lives on the adapter
        # instance (one per process); on restart we lose count and start
        # nudging from turn 5 again, which is acceptable.
        self._turn_counter: dict[str, int] = {}

        # ── Deferred wiring slots ─────────────────────────────────────────
        # Optional cross-module hooks. Both are set AFTER construction (via
        # set_conflict_detector / set_consolidator) because their config
        # gates and dependencies (LLM engine for reconcile, etc.) aren't
        # ready when MemoryAdapter is first instantiated in build_services.
        # If still unset at write-time, add_fact / after_turn fall back to
        # the legacy direct-write path — i.e. the feature is purely opt-in.
        self._conflict_detector: Any = None
        self._consolidator: Any = None
        # Throttle: track per-session last-consolidate turn count so we
        # don't fire consolidation N times in a row on a long session.
        self._last_consolidate_turn: dict[str, int] = {}

        logger.info("MemoryAdapter ready — backend=%s", data_dir)

    # ── Deferred hooks (cross-module wiring) ─────────────────────────────
    #
    # These setters are called by main.py once the cross-module collaborators
    # are constructed. Either remaining unset is a valid runtime state —
    # the relevant code path falls back to the legacy direct-write.

    def set_conflict_detector(self, detector: Any) -> None:
        """Inject a FactConflictDetector instance.

        Once set, add_fact() routes inserts through detector.insert_with_reconcile()
        instead of the direct mid_term.add_fact path, so semantic duplicates and
        contradictions are reconciled rather than blindly stored. Set to None
        to revert.

        Why a setter and not a constructor arg: the detector takes a memory
        protocol (i.e. this very adapter) and an LLM function, both of which
        are built in build_services *after* this adapter. Circular dependency
        if we tried construction-time wiring.
        """
        self._conflict_detector = detector
        logger.info(
            "MemoryAdapter: conflict detector %s",
            "WIRED" if detector is not None else "CLEARED",
        )

    def set_consolidator(self, *, threshold_turns: int = 30) -> None:
        """Enable per-session auto-consolidation.

        After every after_turn() call we increment the session's turn counter.
        When it crosses `threshold_turns` (and is above the previous high-water
        mark) we fire MemoryManager.consolidate_session() as a background task.
        Counters are tracked per-session in _last_consolidate_turn so a long
        session triggers exactly once per N turns, not every turn forever.

        Pass threshold_turns=0 to disable.
        """
        # We don't store a separate consolidator instance — MemoryManager
        # owns one internally and exposes consolidate_session(). We just
        # store the threshold here as the gate.
        self._consolidate_threshold = max(0, int(threshold_turns))
        if self._consolidate_threshold > 0:
            self._consolidator = self._mgr  # marker: enabled
            logger.info(
                "MemoryAdapter: auto-consolidate ENABLED (every %d turns per session)",
                self._consolidate_threshold,
            )
        else:
            self._consolidator = None
            logger.info("MemoryAdapter: auto-consolidate DISABLED")

    # ── Recall (read path) ────────────────────────────────────────────────

    async def recall(
        self,
        query:        str,
        session_id:   str,
        max_chars:    int = 1200,
        recent_turns: int = 4,
    ) -> RecallResult:
        """Build prompt-ready memory context for a query.

        Delegates the entire dual-track + MMR + type-boost pipeline to
        agent_memory.retrieval.recall_orchestrator. This adapter's only
        job is to bind the operator id and run the sync algorithm in a
        thread pool.
        """
        user_id = get_current_operator()
        try:
            return await asyncio.to_thread(
                _orch_recall,
                self._mgr, user_id, query, session_id, max_chars, recent_turns,
            )
        except Exception as exc:
            logger.warning("MemoryAdapter.recall failed: %s", exc)
            return RecallResult(
                prompt_context="", fact_count=0, chunk_count=0,
                results=[], track_a_count=0, track_b_count=0, winner="",
            )

    # ── Public fact-writing (cross-module integrations use this) ─────────
    #
    # External adapters (JournalToFactsAdapter, FactConflictDetector) need a
    # stable surface for writing facts WITHOUT importing agent_memory internals.
    # Keep this signature stable; it's part of the module's public API.

    async def add_fact(
        self,
        session_id: str,
        user_id:    str,
        fact_text:  str,
        *,
        fact_type:  str   = "general",
        confidence: float = 1.0,
        metadata:   Optional[dict] = None,
        ttl_days:   Optional[float] = None,
    ) -> str:
        """Insert a fact into the mid-term store.

        Returns the fact_id. Duplicates (same user+session+text) are silently
        deduped and return the existing id.

        If a FactConflictDetector has been wired via set_conflict_detector(),
        the insert is routed through detector.insert_with_reconcile() which
        finds semantically similar existing facts and applies an action
        (equivalent → boost existing, refinement → boost + insert,
        contradiction → LLM reconcile, unrelated → straight insert).
        Falls back to the direct path on detector failure or if not wired,
        so the cross-module hook is purely opt-in and never breaks writes.
        """
        # ── Routed path: conflict-aware insertion ────────────────────
        if self._conflict_detector is not None:
            try:
                result = await self._conflict_detector.insert_with_reconcile(
                    session_id=session_id,
                    user_id=user_id,
                    fact_text=fact_text,
                    fact_type=fact_type,
                    confidence=confidence,
                    metadata=metadata,
                    ttl_days=ttl_days,
                )
                # ReconcileResult.inserted_fact_id is None when verdict ==
                # equivalent (existing fact's confidence boosted, no new
                # row). Return the related_fact_id in that case so callers
                # always get a valid id for citation/chunk linkage.
                fid = (
                    getattr(result, "inserted_fact_id", None)
                    or getattr(result, "related_fact_id", None)
                    or ""
                )
                if fid:
                    return fid
                # Detector returned nothing usable — fall through to direct
                # write so the caller doesn't get an empty string back.
                logger.debug(
                    "add_fact: detector returned no fact_id (verdict=%s), "
                    "falling through to direct write",
                    getattr(result, "verdict", "?"),
                )
            except Exception as exc:
                # Detector failures must never block a write. Log and fall
                # through. Symmetric to how all the other cross-module
                # hooks degrade (SkillEvolver, JournalToFacts).
                logger.warning(
                    "add_fact: conflict detector raised %s — falling back to direct write",
                    exc,
                )

        # ── Direct path: legacy behaviour (exact-hash dedup only) ────
        from agent_memory.schemas import MemoryFact
        fact = MemoryFact(
            user_id=user_id, session_id=session_id, fact=fact_text,
            fact_type=fact_type, confidence=confidence,
            metadata=metadata or {},
        )
        return await asyncio.to_thread(
            self._mgr.mid_term.add_fact, fact, ttl_days,
        )

    async def find_similar_facts(
        self,
        user_id:    str,
        query_text: str,
        *,
        session_id: Optional[str] = None,
        fact_type:  Optional[str] = None,
        top_k:      int = 5,
    ) -> list[dict]:
        """Find facts semantically similar to `query_text`.

        Used by FactConflictDetector to find candidate conflicts before
        inserting a new fact. Returns a list of {fact_id, fact, confidence,
        score, metadata} dicts.
        """
        def _go():
            res = self._mgr.mid_term.search(
                user_id=user_id, query=query_text, session_id=session_id,
                fact_type=fact_type, top_k=top_k,
            )
            return [
                {
                    "fact_id":    f.fact_id,
                    "fact":       f.fact,
                    "fact_type":  f.fact_type,
                    "confidence": f.confidence,
                    "score":      sc,
                    "metadata":   getattr(f, "metadata", {}) or {},
                }
                for f, sc in zip(res.facts, res.scores)
            ]
        try:
            return await asyncio.to_thread(_go)
        except Exception as exc:
            logger.warning("MemoryAdapter.find_similar_facts failed: %s", exc)
            return []

    async def update_fact_confidence(
        self,
        fact_id:    str,
        new_confidence: float,
        *,
        reason: str = "",
    ) -> bool:
        """Adjust a fact's confidence (used by conflict reconciliation).
        Returns True on success."""
        def _go():
            try:
                self._mgr.mid_term.update_fact(
                    fact_id=fact_id,
                    new_confidence=max(0.0, min(1.0, new_confidence)),
                )
                return True
            except Exception as exc:
                logger.warning("update_fact_confidence(%s) failed: %s", fact_id, exc)
                return False
        return await asyncio.to_thread(_go)

    # ── Write (after every completed turn) ────────────────────────────────

    async def after_turn(
        self,
        session_id:     str,
        user_text:      str,
        assistant_text: str,
        tool_calls:     Optional[list[dict]] = None,
        importance:     float = 0.5,
    ) -> list:
        """Persist a completed turn into the memory layers.

        Importance-based fanout (ported from legacy memory/router.py):
          importance < 0.30  → SKIP entirely (greetings, filler)
          0.30 ≤ imp < 0.50  → long-term chunk only (raw text searchable)
          0.50 ≤ imp < 0.75  → chunk + distill facts
          imp ≥ 0.75         → chunk + distill + user-profile update

        After write, schedules a periodic nudge as a background task if
        the turn counter hits the shallow (5) or deep (20) interval.
        """
        user_id = get_current_operator()
        tool_calls = tool_calls or []

        # Per-session turn counter (synchronous bump; counter dict is
        # initialised in __init__ so concurrent first-turn calls don't race).
        self._turn_counter[session_id] = self._turn_counter.get(session_id, 0) + 1
        turn_n = self._turn_counter[session_id]

        def _do() -> list:
            new_facts: list = []

            if importance < 0.30:
                logger.debug(
                    "after_turn: skipping low-importance turn (imp=%.2f) for session=%s",
                    importance, session_id[:12],
                )
                return new_facts

            try:
                turn_text = f"User: {user_text}\nAssistant: {assistant_text}"
                self._mgr.remember(
                    user_id    = user_id,
                    session_id = session_id,
                    text       = turn_text,
                    source     = "conversation",
                    importance = importance,
                )
            except Exception as exc:
                logger.warning("after_turn: remember failed: %s", exc)

            if importance < 0.50:
                return new_facts

            try:
                new_facts = self._mgr.distill(
                    user_id    = user_id,
                    session_id = session_id,
                    text       = f"{user_text}\n{assistant_text}",
                )
            except Exception as exc:
                logger.warning("after_turn: distill failed: %s", exc)

            if importance < 0.75:
                return new_facts

            try:
                self._mgr.update_user_profile(
                    user_id        = user_id,
                    session_id     = session_id,
                    user_text      = user_text,
                    assistant_text = assistant_text,
                    tool_calls     = tool_calls,
                )
            except Exception as exc:
                logger.warning("after_turn: profile update failed: %s", exc)

            return new_facts

        new_facts = await asyncio.to_thread(_do)

        # Schedule nudge as a background task. The orchestrator owns the
        # actual algorithm (deep_review, contradiction routing, etc.) —
        # this adapter just decides timing and binds operator context.
        nudge_kind = _orch_should_nudge(turn_n)
        if nudge_kind is not None:
            asyncio.create_task(
                self._nudge_async(user_id, session_id, deep=nudge_kind)
            )

        # ── Auto-consolidation gate ────────────────────────────────
        # Schedule consolidate_session() as a background task when this
        # session has accumulated enough turns since the last compression
        # pass. The consolidator merges old chunks into LLM-summarised
        # rollups, keeping the long-term context window bounded as
        # sessions grow. Without this, long sessions monotonically
        # increase chunk count → FTS5/embedding search latency rises.
        #
        # Per-session "high water mark" semantics: fire when
        #   turn_n >= last_consolidate + threshold
        # so a long session at turn 120 with threshold=30 fires at
        # turns 30, 60, 90, 120 — once per N turns, never reentrant.
        # Background task means hot path isn't blocked by the LLM
        # summarisation call.
        if (
            self._consolidator is not None
            and self._consolidate_threshold > 0
        ):
            last_at = self._last_consolidate_turn.get(session_id, 0)
            if turn_n - last_at >= self._consolidate_threshold:
                self._last_consolidate_turn[session_id] = turn_n
                asyncio.create_task(
                    self._consolidate_async(user_id, session_id, turn_n)
                )

        return new_facts

    async def _consolidate_async(
        self, user_id: str, session_id: str, turn_n: int,
    ) -> None:
        """Background wrapper for MemoryManager.consolidate_session.

        Runs in a fire-and-forget task so the hot after_turn path doesn't
        wait for the LLM summarisation. Failures are logged but never
        propagate — symmetric to _nudge_async.
        """
        try:
            stats = await asyncio.to_thread(
                self._mgr.consolidate_session, user_id, session_id,
            )
            logger.info(
                "MemoryAdapter consolidate: session=%s turn=%d → %s",
                session_id[:12], turn_n,
                {k: stats.get(k) for k in ("merged", "kept", "compression_ratio")
                 if k in (stats or {})} or stats,
            )
        except Exception as exc:
            logger.debug(
                "MemoryAdapter consolidate failed (non-fatal): %s", exc,
            )

    async def _nudge_async(self, user_id: str, session_id: str, deep: bool) -> None:
        """Background wrapper around the sync orchestrator nudge."""
        try:
            stats = await asyncio.to_thread(
                _orch_run_nudge, self._mgr, user_id, session_id, deep,
            )
            kind = "deep" if deep else "shallow"
            logger.info(
                "MemoryAdapter nudge[%s]: session=%s reviewed=%d turns → "
                "%d new facts, %d contradictions",
                kind, session_id[:12],
                stats.get("reviewed_turns", 0),
                stats.get("new_facts", 0),
                stats.get("contradictions", 0),
            )
        except Exception as exc:
            logger.debug("MemoryAdapter nudge failed (non-fatal): %s", exc)

    # ── Tool result cache (drop-in for ToolResultStore on the memory layer) ──

    async def cache_tool_result(
        self,
        session_id: str,
        tool_name:  str,
        content:    str,
    ) -> dict:
        """Cache a large tool output. Returns ref_id + preview for prompt injection."""
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
                user_id=user_id, ref_id=ref_id, offset=offset, length=length,
            )
        return await asyncio.to_thread(_do)

    # ── Stats / health ────────────────────────────────────────────────────

    async def stats(self) -> dict:
        user_id = get_current_operator()
        return await asyncio.to_thread(self._mgr.stats, user_id)

    # ── Backward-compatibility shims (old curator/fts/dtm call sites) ─────

    async def recall_for_session(self, query: str, session_id: str) -> str:
        """Old curator API — returns plain text context."""
        result = await self.recall(query, session_id, max_chars=1200)
        return result.prompt_context

    async def get_stats(self) -> dict:
        """Old fts API — returns memory stats."""
        try:
            return await self.stats()
        except Exception:
            return {}

    def set_llm_fn(self, llm_fn: Callable[..., str]) -> None:
        """Wire an LLM into the FactExtractor (and ReflectionEngine) AFTER
        the adapter is constructed. Called by main.py once the LLM engine
        is built. Without this, extraction falls back to English-only
        regex patterns and returns no facts for non-English conversations.

        Both single-arg `llm_fn(prompt) -> str` and two-arg
        `llm_fn(system, user) -> str` signatures are supported — the
        FactExtractor auto-detects via inspect.signature.
        """
        try:
            if hasattr(self._mgr, "extractor"):
                ext = self._mgr.extractor
                ext._llm_fn = llm_fn
                # Re-detect signature in case it changed
                if hasattr(ext, "_detect_two_arg"):
                    ext._llm_takes_system = ext._detect_two_arg(llm_fn)
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

    # Stats stubs for the old curator interface (some healthchecks read these)
    @property
    def _shallow_n(self) -> int:
        from agent_memory.retrieval.recall_orchestrator import SHALLOW_NUDGE_INTERVAL
        return SHALLOW_NUDGE_INTERVAL

    @property
    def _deep_n(self) -> int:
        from agent_memory.retrieval.recall_orchestrator import DEEP_NUDGE_INTERVAL
        return DEEP_NUDGE_INTERVAL

    def close(self) -> None:
        """Close SQLite connections + WAL checkpoint. Called by lifespan shutdown."""
        try:
            self._mgr.close()
        except Exception as exc:
            logger.warning("MemoryAdapter close failed: %s", exc)