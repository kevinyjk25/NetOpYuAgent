"""
agent_memory/retrieval/recall_orchestrator.py
---------------------------------------------
Sync retrieval/recall algorithm layer for agent_memory.

This module owns the multi-track recall pipeline that arbitrates between
Track A (long-term chunks) and Track B (mid-term distilled facts), plus
the periodic nudge loop and contradiction routing. It is intentionally
sync and stateless — the async wrapper in memory/adapter.py wraps each
public function with asyncio.to_thread.

Why pulled out of memory/adapter.py?
  - adapter.py was carrying both protocol/IO concerns (ContextVar, async,
    importance-based write fanout) AND the entire recall algorithm. That
    coupled the agent_memory algorithm details to the adapter abstraction.
  - With this split, agent_memory becomes algorithmically self-sufficient:
    you can use MemoryManager + RecallOrchestrator with no adapter and
    still get all the dual-track + MMR + nudge + contradiction features.
  - adapter.py shrinks to ~350 lines of thin async wrapping.

Public API
----------
    RecallResult                      — dataclass returned by recall()
    recall(mgr, user_id, query, …)    — full dual-track recall pipeline
    run_nudge(mgr, user_id, …)        — periodic shallow/deep nudge
    mmr_select(candidates, k, lambda_)— exposed for tests/reuse

Design constraints
------------------
  - Sync only (caller wraps with asyncio.to_thread).
  - No global state — everything passed as args.
  - No knowledge of operators / ContextVar / FastAPI / async — those live
    in the adapter.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Tuning constants (ported from legacy memory/dual_track.py + curator.py) ──

# Track B (curated facts) gets a relevance boost vs Track A (raw chunks)
# because facts are pre-abstracted signal, denser per token.
TRACK_B_WEIGHT = 1.5

# Per-fact-type retrieval boost — higher = more durable / reusable signal.
# Maps the agent_memory 7-class taxonomy to a multiplier; missing → 1.0.
TYPE_BOOST = {
    "lesson":     1.30,    # incident lessons — highest reuse value
    "procedure":  1.20,    # tool patterns / runbooks
    "config":     1.10,    # device/service configurations
    "entity":     1.05,    # named devices, services
    "env":        1.00,    # environment facts (baseline)
    "general":    0.95,    # uncategorised
    "preference": 0.85,    # user habits — relevant but local
}

# MMR diversity selection: lambda=0.7 means relevance still dominates but
# we penalise candidates that look near-duplicate to already-selected ones.
MMR_LAMBDA = 0.7

# Same-session recent-turns budget: ~40% of total recall budget is reserved
# for "what just happened in this conversation" so pronoun queries
# ("this device", "它") can resolve against prior turns even when the
# query has no keyword overlap with those turns.
RECENT_TURNS_BUDGET_FRAC = 0.40

# Nudge schedule (see legacy memory/curator.py):
#   shallow every 5 turns: scan recent text for missed cross-turn patterns
#   deep    every 20 turns: full review + contradiction detection vs existing facts
SHALLOW_NUDGE_INTERVAL = 5
DEEP_NUDGE_INTERVAL    = 20


def _is_stale_hitl_placeholder(text: str) -> bool:
    """Detect HITL-pending placeholder turns left by older chat_stream code.

    These are turns whose assistant_text is just the "⚠ HITL interrupt —
    awaiting approval" notice that was written BEFORE the operator made
    a decision. Once the operator approves/rejects/escalates, the proper
    completed turn is written in a separate turn (with a [HITL …] tag),
    leaving the placeholder as stale data that would mislead future
    recalls into thinking the action is still pending.

    Used by recall() in two places:
      1. Recent-turns prepend  — filter so pronoun resolution doesn't
         see the placeholder.
      2. Cross-session search — filter so query-driven retrieval
         doesn't surface the placeholder either.
    """
    if not text:
        return False
    t = text
    # Markers from chat_stream's old HITL placeholder writes:
    if "human approval required" in t and "Interrupt ID" in t:
        return True
    if "HITL interrupt" in t and "Click Approve or Reject to continue" in t:
        return True
    if "Approval card is now in the HITL tab" in t:
        return True
    return False


# ── Public dataclass — kept stable, used by callers (backend.py, etc.) ──

@dataclass
class RecallResult:
    """Result of a dual-track recall, ready for prompt injection.

    Track A = long-term raw chunks (FTS5 + TF-IDF retrieval)
    Track B = mid-term distilled facts (curated knowledge)

    Fields:
      prompt_context : combined text to inject before the user query
      results        : serialized item dicts for the Memory UI tab
      *_count        : counts AFTER MMR selection (what the LLM actually sees)
      winner         : "A" | "B" | "tie" | ""  — for the recall step indicator
    """
    prompt_context: str
    fact_count:     int = 0
    chunk_count:    int = 0
    results:        list = field(default_factory=list)
    track_a_count:  int  = 0
    track_b_count:  int  = 0
    winner:         str  = ""


# ── Algorithm: MMR diversity selection ────────────────────────────────────

def mmr_select(
    candidates: list[dict],
    top_k: int = 10,
    lambda_: float = MMR_LAMBDA,
) -> list[dict]:
    """Maximal Marginal Relevance selection — diversity-aware top-k.

    Without MMR the top-k recall is dominated by near-duplicates (same
    incident re-asked across turns), which starves diverse facts from
    the prompt budget.

    score(c) = lambda * c.score - (1 - lambda) * max_jaccard_to_selected(c)
      lambda=1.0 → pure relevance ranking
      lambda=0.0 → pure diversity
      lambda=0.7 → relevance-weighted diversity

    Each candidate is a dict with at least "content" and "score" keys.
    """
    if not candidates:
        return []
    pool = sorted(
        candidates,
        key=lambda x: float(x.get("score", 0.0)),
        reverse=True,
    )
    selected: list[dict] = []

    def _tokens(s: str) -> set[str]:
        return set(s.lower().split()) if s else set()

    while pool and len(selected) < top_k:
        if not selected:
            selected.append(pool.pop(0))
            continue
        sel_tokens = [_tokens(s.get("content", "")) for s in selected]
        best_idx = 0
        best_score = -1e9
        for i, c in enumerate(pool):
            c_tokens = _tokens(c.get("content", ""))
            max_sim = 0.0
            for st in sel_tokens:
                union = len(c_tokens | st)
                if union == 0:
                    continue
                sim = len(c_tokens & st) / union
                if sim > max_sim:
                    max_sim = sim
            mmr = lambda_ * float(c.get("score", 0.0)) - (1.0 - lambda_) * max_sim
            if mmr > best_score:
                best_score = mmr
                best_idx = i
        selected.append(pool.pop(best_idx))
    return selected


# ── Algorithm: dual-track recall ──────────────────────────────────────────

def recall(
    mgr,                       # MemoryManager (no type hint to avoid circular import at module load)
    user_id: str,
    query: str,
    session_id: str,
    max_chars: int = 1200,
    recent_turns: int = 4,
    top_k: int = 10,
) -> RecallResult:
    """Build prompt-ready memory context for a query within one user's scope.

    Pipeline:
      1. Same-session recent N turns prepended UNCONDITIONALLY (no query
         match required) — fixes pronoun coreference in multi-turn dialogue.
      2. Query-driven retrieval from long_term + mid_term in parallel.
      3. Score each item: chunks use search-returned score; facts use
         confidence × TRACK_B_WEIGHT × TYPE_BOOST[fact_type].
      4. MMR diversity selection over the merged pool.
      5. Build prompt_context: recent_turns header + arbitrated body.

    Returns RecallResult; on internal failure, returns empty result (the
    adapter logs the error and proceeds without context).
    """
    # ── 1. Same-session recent turns (unconditional prepend) ────────────
    recent_section = ""
    recent_chunks: list = []
    recent_budget = int(max_chars * RECENT_TURNS_BUDGET_FRAC)
    if session_id and recent_turns > 0 and recent_budget > 0:
        try:
            # Use the LIMIT-aware fast path so very long sessions don't
            # scan thousands of rows just to grab the last N turns.
            # Pull a few extra rows (×2) so we can drop legacy "⚠ HITL
            # interrupt — awaiting approval" placeholder turns that
            # versions before the chat_stream/_submit_hitl_decision split
            # may have written. After filtering we still keep `recent_turns`
            # genuinely-actionable turns.
            fetch_limit = recent_turns * 2
            if hasattr(mgr.long_term, "get_recent_chunks_by_session"):
                rows = mgr.long_term.get_recent_chunks_by_session(
                    user_id, session_id, limit=fetch_limit,
                )
            else:
                # Fallback for older long_term implementations
                rows = mgr.long_term.get_chunks_by_session(user_id, session_id)
                rows = rows[-fetch_limit:] if rows else []

            # Filter: any turn whose assistant_text is just a "HITL pending"
            # placeholder is stale by definition (see _is_stale_hitl_placeholder
            # docstring at module top). Drop them from recent_turns so the LLM
            # doesn't see "still awaiting approval" for actions that have since
            # completed in a later turn.
            filtered_rows = [
                r for r in rows
                if not _is_stale_hitl_placeholder(r.get("text") or "")
            ][-recent_turns:]   # keep last N actionable turns

            if filtered_rows:
                lines = ["## Recent turns (this session)"]
                used = len(lines[0]) + 1
                for r in filtered_rows:
                    text = (r.get("text") or "").replace("\n", " ")
                    line = f"- {text[:400]}"
                    if used + len(line) + 1 > recent_budget:
                        break
                    lines.append(line)
                    used += len(line) + 1
                    recent_chunks.append(r)
                if len(lines) > 1:
                    recent_section = "\n".join(lines)

            # Diagnostic: how many stale placeholders did we drop?
            n_dropped = len(rows) - len(filtered_rows) if rows else 0
            if n_dropped > 0:
                logger.info(
                    "recall: filtered %d stale HITL-pending placeholder(s) "
                    "from session=%s recent turns",
                    n_dropped, session_id[:12],
                )
        except Exception as exc:
            logger.warning("recall: recent_turns gather failed: %s", exc)

    # ── 2. Single-pass retrieval — pull facts AND chunks ONCE.
    #      Old code ran build_context (which queries both layers) AND then
    #      mgr.search (which queries them again), doubling the SQL load on
    #      every recall. Now we run search once, then build prompt_context
    #      from the same items used for the Memory tab.
    chunk_items: list = []
    fact_items: list  = []
    try:
        search_out = mgr.search(
            user_id=user_id, query=query, session_id=session_id, top_k=top_k,
        )
        chunks_rr = search_out.get("long_term")
        facts_rr  = search_out.get("mid_term")
        if chunks_rr and hasattr(chunks_rr, "items"):
            # Filter stale HITL-pending placeholders from cross-session
            # retrieval too — same rationale as in the recent-turns filter.
            # Without this, a query like "did we fix ap-02" hits the legacy
            # placeholder via FTS5 and surfaces it as "Relevant Memory",
            # confusing the LLM into reporting the action still pending.
            chunk_items = [
                it for it in chunks_rr.items
                if not _is_stale_hitl_placeholder(getattr(it, "text", "") or "")
            ]
            n_dropped_xs = len(chunks_rr.items) - len(chunk_items)
            if n_dropped_xs > 0:
                logger.info(
                    "recall: filtered %d stale HITL placeholder(s) from "
                    "cross-session chunk results", n_dropped_xs,
                )
        if facts_rr and hasattr(facts_rr, "items"):
            fact_items = facts_rr.items
    except Exception as exc:
        logger.warning("recall: search failed: %s", exc)

    # User profile + skills (cheap, in-memory ops — no SQL on the hot path)
    profile_section = ""
    skills_section  = ""
    try:
        if getattr(mgr, "user_model", None):
            profile_section = mgr.user_model.get_prompt_section(
                user_id, max_chars=int(max_chars * 0.15),
            ) or ""
    except Exception:
        pass
    try:
        if getattr(mgr, "skill_store", None):
            skills_ctx = mgr.skill_store.build_skills_context(
                user_id, query, top_k=2,
            ) or ""
            skills_section = skills_ctx[:int(max_chars * 0.20)]
    except Exception:
        pass

    # Format facts + chunks sections from the SAME items the Memory tab uses
    fact_section  = ""
    chunk_section = ""
    remaining = max(200, max_chars - len(recent_section) - len(profile_section) - len(skills_section))
    fact_budget  = int(remaining * 0.55)
    chunk_budget = int(remaining * 0.45)
    if fact_items:
        try:
            fact_section = mgr._format_facts(fact_items, max_chars=fact_budget) or ""
        except Exception:
            pass
    if chunk_items:
        try:
            chunk_section = mgr._format_chunks(chunk_items, max_chars=chunk_budget) or ""
        except Exception:
            pass

    # Assemble prompt_context: recent_turns first (resolves pronouns),
    # then profile, skills, facts, chunks.
    parts = [s for s in [recent_section, profile_section, skills_section,
                          fact_section, chunk_section] if s]
    final_ctx = "\n\n".join(parts)

    # ── 3. Serialize items with scoring (Track B weight + type boost) ───
    serialized: list[dict] = []

    # Recent-session chunks first — flag them so the UI shows why they're top.
    for r in recent_chunks:
        serialized.append({
            "track":       "A",
            "score":       1.0,
            "source":      "recent_turn",
            "memory_type": "chunk",
            "content":     (r.get("text") or "")[:500],
            "recency_ts":  r.get("created_at", 0),
            "tags":        ["same-session"],
        })

    # Cross-session chunks
    for it in chunk_items:
        serialized.append({
            "track":       "A",
            "score":       round(float(getattr(it, "score", 0.0)), 3),
            "source":      getattr(it, "source", "conversation"),
            "memory_type": "chunk",
            "content":     (getattr(it, "text", "") or "")[:500],
            "recency_ts":  getattr(it, "created_at", 0),
            "tags":        list(getattr(it, "metadata", {}).get("tags", []))[:6],
        })

    # Mid-term facts with full Track B + type-boost scoring
    for it in fact_items:
        ftype = getattr(it, "fact_type", "general")
        conf  = float(getattr(it, "confidence", 1.0))
        boost = TYPE_BOOST.get(ftype, 1.0)
        score = conf * TRACK_B_WEIGHT * boost
        serialized.append({
            "track":       "B",
            "score":       round(score, 3),
            "source":      "facts",
            "memory_type": ftype,
            "content":     (getattr(it, "fact", "") or "")[:500],
            "recency_ts":  getattr(it, "created_at", 0),
            "tags":        list(getattr(it, "metadata", {}).get("tags", []))[:6],
        })

    # ── 4. MMR diversity dedup over the combined pool ────────────────────
    serialized = mmr_select(serialized, top_k=top_k, lambda_=MMR_LAMBDA)

    # Counts reflect what survived MMR (not the raw retrieval set)
    track_a = sum(1 for s in serialized if s.get("track") == "A")
    track_b = sum(1 for s in serialized if s.get("track") == "B")
    if track_a > track_b:
        winner = "A"
    elif track_b > track_a:
        winner = "B"
    elif track_a > 0:
        winner = "tie"
    else:
        winner = ""

    return RecallResult(
        prompt_context = final_ctx,
        fact_count     = track_b,
        chunk_count    = track_a,
        results        = serialized,
        track_a_count  = track_a,
        track_b_count  = track_b,
        winner         = winner,
    )


# ── Algorithm: periodic nudge (shallow / deep) ────────────────────────────

def run_nudge(
    mgr,                # MemoryManager
    user_id: str,
    session_id: str,
    deep: bool,
) -> dict:
    """Periodic re-review of recent turns.

    Shallow (every SHALLOW_NUDGE_INTERVAL turns):
      Re-distill the combined recent text. Catches cross-turn patterns the
      per-turn distiller missed.

    Deep (every DEEP_NUDGE_INTERVAL turns):
      Use FactExtractor.deep_review() to find both new cross-turn facts AND
      contradictions vs existing facts. Contradictions are persisted as
      low-confidence "lesson" facts with metadata.contradiction=True so a
      reviewer can find them later — but never auto-applied.

    Returns a stats dict so the caller can log:
      {"reviewed_turns": int, "new_facts": int, "contradictions": int}
    """
    n_turns = DEEP_NUDGE_INTERVAL if deep else SHALLOW_NUDGE_INTERVAL
    stats = {"reviewed_turns": 0, "new_facts": 0, "contradictions": 0}

    try:
        if hasattr(mgr.long_term, "get_recent_chunks_by_session"):
            recent_rows = mgr.long_term.get_recent_chunks_by_session(
                user_id, session_id, limit=n_turns,
            )
        else:
            rows = mgr.long_term.get_chunks_by_session(user_id, session_id)
            recent_rows = rows[-n_turns:] if rows else []
        if not recent_rows:
            return stats
        stats["reviewed_turns"] = len(recent_rows)

        combined = "\n\n---\n\n".join(
            f"Turn {i+1}: {(r.get('text') or '')[:400]}"
            for i, r in enumerate(recent_rows)
        )

        if not deep:
            # Shallow: just re-distill the combined text
            new_facts = mgr.distill(
                user_id=user_id, session_id=session_id, text=combined,
            )
            stats["new_facts"] = len(new_facts)
            return stats

        # Deep: pull existing facts as "do not duplicate" + contradiction signal
        existing_facts: list = []
        try:
            existing_rr = mgr.search_facts(
                user_id    = user_id,
                query      = "preferences habits patterns config",
                session_id = session_id,
                top_k      = 20,
            )
            if existing_rr and hasattr(existing_rr, "items"):
                existing_facts = existing_rr.items
        except Exception as exc:
            logger.debug("deep nudge: existing facts fetch failed: %s", exc)

        existing_strs = [
            getattr(f, "fact", "") for f in existing_facts[:20]
            if getattr(f, "fact", "")
        ]

        extractor = mgr.extractor
        if not hasattr(extractor, "deep_review"):
            # Older extractor — fall back to shallow distill
            new_facts = mgr.distill(
                user_id=user_id, session_id=session_id, text=combined,
            )
            stats["new_facts"] = len(new_facts)
            return stats

        new_facts_list, contradictions = extractor.deep_review(
            recent_text    = combined,
            existing_facts = existing_strs,
            user_id        = user_id,
            session_id     = session_id,
        )

        # Persist new facts
        if new_facts_list:
            try:
                mgr.mid_term.add_facts_batch(new_facts_list)
                stats["new_facts"] = len(new_facts_list)
            except Exception as exc:
                logger.warning("deep nudge: persist new facts failed: %s", exc)

        # Persist contradictions as low-confidence lesson-type facts with
        # metadata.contradiction=True. NOT auto-applied — they exist as
        # surface for human review only.
        if contradictions:
            for c in contradictions[:10]:
                if not isinstance(c, dict):
                    continue
                old = str(c.get("old_fact", ""))[:200]
                obs = str(c.get("new_observation", ""))[:200]
                rsn = str(c.get("reason", ""))[:160]
                if not old or not obs:
                    continue
                try:
                    mgr.add_fact(
                        user_id    = user_id,
                        session_id = session_id,
                        fact       = f"[CONTRADICTION] previously: {old} | recently: {obs} ({rsn})",
                        fact_type  = "lesson",
                        confidence = 0.55,
                        metadata   = {
                            "contradiction":   True,
                            "old_fact":        old,
                            "new_observation": obs,
                            "reason":          rsn,
                        },
                    )
                    stats["contradictions"] += 1
                except Exception as exc:
                    logger.debug("deep nudge: contradiction persist skipped: %s", exc)

    except Exception as exc:
        logger.debug("run_nudge failed (non-fatal): %s", exc)

    return stats


def should_nudge(turn_count: int) -> Optional[bool]:
    """Decide whether the current turn triggers a nudge and which kind.

    Returns:
      True   → run a deep nudge
      False  → run a shallow nudge
      None   → no nudge this turn
    """
    if turn_count <= 0:
        return None
    if turn_count % DEEP_NUDGE_INTERVAL == 0:
        return True
    if turn_count % SHALLOW_NUDGE_INTERVAL == 0:
        return False
    return None