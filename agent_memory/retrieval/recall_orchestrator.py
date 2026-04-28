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
            rows = mgr.long_term.get_chunks_by_session(user_id, session_id)
            rows = rows[-recent_turns:] if rows else []
            if rows:
                lines = ["## Recent turns (this session)"]
                used = len(lines[0]) + 1
                for r in rows:
                    text = (r.get("text") or "").replace("\n", " ")
                    line = f"- {text[:400]}"
                    if used + len(line) + 1 > recent_budget:
                        break
                    lines.append(line)
                    used += len(line) + 1
                    recent_chunks.append(r)
                if len(lines) > 1:
                    recent_section = "\n".join(lines)
        except Exception as exc:
            logger.warning("recall: recent_turns gather failed: %s", exc)

    # ── 2. Query-driven retrieval (existing build_context handles facts +
    #      cross-session chunks + user profile + skills) ──────────────────
    remaining_budget = max(200, max_chars - len(recent_section))
    try:
        ctx_str = mgr.build_context(
            user_id    = user_id,
            query      = query,
            session_id = session_id,
            max_chars  = remaining_budget,
            include_user_profile = True,
            include_facts        = True,
            include_chunks       = True,
            include_skills       = True,
        )
    except Exception as exc:
        logger.warning("recall: build_context failed: %s", exc)
        ctx_str = ""

    final_ctx = (
        recent_section + "\n\n" + ctx_str if (recent_section and ctx_str)
        else (recent_section or ctx_str)
    )

    # ── 3. Get raw items for the Memory tab + MMR selection ─────────────
    chunk_items: list = []
    fact_items: list  = []
    try:
        search_out = mgr.search(
            user_id=user_id, query=query, session_id=session_id, top_k=5,
        )
        chunks_rr = search_out.get("long_term")
        facts_rr  = search_out.get("mid_term")
        if chunks_rr and hasattr(chunks_rr, "items"):
            chunk_items = chunks_rr.items
        if facts_rr and hasattr(facts_rr, "items"):
            fact_items = facts_rr.items
    except Exception as exc:
        logger.warning("recall: search failed: %s", exc)

    # ── 4. Serialize items with scoring (Track B weight + type boost) ───
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

    # ── 5. MMR diversity dedup over the combined pool ────────────────────
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
