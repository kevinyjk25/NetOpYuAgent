"""
memory/dual_track.py
--------------------
Dual-Track Memory (DTM) — converged design of OpenClaw's static file-chunk
model and Hermes's dynamic LLM-nudge model.

Design philosophy
-----------------
OpenClaw's insight:  the most durable, inspectable, portable memory is plain
                     text files — chunked, hybrid-searched (BM25 + vector),
                     flushed proactively before compaction, with MMR dedup.

Hermes's insight:    LLM-driven curation (curator, user model, skill evolver)
                     extracts *structured* knowledge that raw chunks miss —
                     facts, traits, lessons, contradictions.

Neither alone is sufficient:
  • Static chunks without curation repeat noise. Searching "why did auth fail"
    returns 50 raw log turns, not the extracted lesson "cert expires 2d before
    each quarter-end — pre-renew by week 3."
  • Dynamic curation without static chunks loses verbatim evidence. The curator
    abstracts facts, but the raw turns they came from are gone.

Dual-Track stores and searches BOTH, then picks the better answer:

    Track A — Static Chunks (OpenClaw-style)
        • Every turn → chunked markdown → FTS5 + sqlite-vec (or BM25 fallback)
        • Daily compaction: day's turns → one dated .md file
        • Pre-compaction flush: before context window fills, force a write
        • Retrieval: hybrid BM25 + vector, MMR dedup, temporal decay

    Track B — Dynamic Facts (Hermes-style)
        • Every turn → MemoryCurator extracts structured CuratedMemory records
        • Periodic nudge (every N turns): deep review, contradiction check
        • UserModelEngine tracks operator expertise and preferences
        • Retrieval: fact-type filter + semantic similarity on content

    Retrieval arbitration
        • Run both tracks in parallel for every query
        • Score each result: relevance × confidence × recency
        • Pick the winner per-slot: if facts exist they get higher weight
          (they're pre-abstracted); if no facts match, fall back to raw chunks
        • Return a single ranked list with source tags (A or B)

Integration into existing system
---------------------------------
The DTM is a thin orchestration layer. It does NOT replace any existing module:

    BEFORE:
        MemoryCurator.recall_for_session(query)    → FTS5 raw-turn search only
        MemoryCurator.after_turn(...)              → curator extracts facts only

    AFTER:
        DualTrackMemory.recall(query, session_id)  → runs both tracks, arbitrates
        DualTrackMemory.after_turn(...)            → writes both tracks

    main.py replaces the two curator calls with one DualTrackMemory call.
    Everything else (FTS5SessionStore, MemoryCurator, UserModelEngine) is
    unchanged and re-used by DualTrackMemory internally.

File layout on disk (under HERMES_DATA_DIR)
--------------------------------------------
    data/
    ├── state.db                     # FTS5 turns (Track A raw)
    ├── daily/
    │   ├── 2026-04-15.md            # compacted day file (OpenClaw style)
    │   └── 2026-04-16.md
    ├── facts/
    │   └── facts.jsonl              # structured CuratedMemory records (Track B)
    └── dtm_index.sqlite             # DTM scoring metadata (hit counts, decay)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import pathlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scored result returned by retrieval arbitration
# ---------------------------------------------------------------------------

@dataclass
class DTMResult:
    """A single memory result after dual-track retrieval and arbitration."""
    content:     str          # the text to inject into the prompt
    track:       str          # "A" (raw chunk) or "B" (curated fact)
    score:       float        # arbitrated final score 0.0–1.0
    source:      str          # e.g. "daily/2026-04-15.md" or "facts.jsonl"
    memory_type: str          # "raw_chunk" | curated type name
    recency_ts:  float        # unix timestamp of the original turn
    tags:        list[str]    = field(default_factory=list)


@dataclass
class DTMRecallResult:
    """Full result of a dual-track recall call."""
    results:        list[DTMResult]
    track_a_count:  int           # raw chunks found
    track_b_count:  int           # curated facts found
    winner:         str           # "A", "B", or "tie"
    prompt_context: str           # ready-to-inject context string


# ---------------------------------------------------------------------------
# DualTrackMemory
# ---------------------------------------------------------------------------

class DualTrackMemory:
    """
    Orchestrates Track A (static chunks, OpenClaw-style) and Track B
    (dynamic facts, Hermes-style) for write and retrieval.

    Parameters
    ----------
    fts_store:
        Existing FTS5SessionStore — Track A raw storage.
    curator:
        Existing MemoryCurator — Track B LLM-driven extraction.
    user_model:
        Existing UserModelEngine — user profile enrichment.
    data_dir:
        Root directory for all DTM files (daily/*.md, facts/facts.jsonl,
        dtm_index.sqlite).
    llm_fn:
        async (system: str, user: str) -> str — same signature as all
        other Hermes modules.
    compaction_turns:
        How many turns accumulate before daily compaction runs.
        Default: 20 (roughly one working session).
    nudge_turns:
        How many turns between deep Hermes nudge reviews.
        Default: 10.
    track_b_weight:
        Relative weight given to Track B (curated facts) over Track A
        during arbitration. 1.0 = equal weight.  >1 = facts preferred.
        Default: 1.5 (facts win when they exist).
    temporal_half_life_days:
        Track A temporal decay — score halves every N days.
        Default: 7 (recent context matters for IT ops).
    """

    def __init__(
        self,
        fts_store,                          # FTS5SessionStore
        curator,                            # MemoryCurator
        user_model=None,                    # UserModelEngine (optional)
        data_dir: str = "./data",
        llm_fn: Optional[Callable] = None,
        compaction_turns: int = 20,
        nudge_turns: int = 10,
        track_b_weight: float = 1.5,
        temporal_half_life_days: float = 7.0,
    ) -> None:
        self._fts     = fts_store
        self._curator = curator
        self._umodel  = user_model
        self._llm_fn  = llm_fn

        self._data_dir = pathlib.Path(data_dir)
        self._daily_dir = self._data_dir / "daily"
        self._facts_path = self._data_dir / "facts" / "facts.jsonl"
        self._index_path = self._data_dir / "dtm_index.sqlite"

        self._compaction_turns = compaction_turns
        self._nudge_turns      = nudge_turns
        self._track_b_weight   = track_b_weight
        self._half_life_secs   = temporal_half_life_days * 86400.0

        # Turn counters per session
        self._turn_counters: dict[str, int] = {}

        # In-memory cache of today's raw turns (for compaction)
        self._today_turns: list[dict] = []

        self._ensure_dirs()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def after_turn(
        self,
        session_id:     str,
        user_text:      str,
        assistant_text: str,
        tool_calls:     Optional[list[dict]] = None,
        importance:     float = 0.5,
    ) -> list:
        """
        Call this after every completed turn — replaces the two separate
        calls to curator.after_turn() and fts.write_turn() in main.py.

        Track A write:  FTS5 raw turn + append to today buffer
        Track B write:  MemoryCurator LLM extraction
        Compaction:     if turn counter hits threshold, flush today's buffer
                        to daily/<date>.md (OpenClaw-style)
        Nudge:          if turn counter hits nudge_turns, run deep review
        """
        tool_calls = tool_calls or []

        # ── Track A: raw turn to FTS5 ────────────────────────────────
        try:
            await self._fts.write_turn(
                session_id     = session_id,
                user_text      = user_text,
                assistant_text = assistant_text,
                tool_calls     = tool_calls,
                importance     = importance,
            )
        except Exception as exc:
            logger.warning("DTM Track A write failed: %s", exc)

        # ── Track A: append to today's in-memory buffer ───────────────
        self._today_turns.append({
            "ts":        time.time(),
            "session":   session_id,
            "user":      user_text,
            "assistant": assistant_text,
            "tools":     [t.get("tool", "") for t in tool_calls],
        })

        # ── Turn counter ──────────────────────────────────────────────
        self._turn_counters[session_id] = self._turn_counters.get(session_id, 0) + 1
        turn_n = self._turn_counters[session_id]

        # ── Track B: LLM extraction ───────────────────────────────────
        memories: list = []
        try:
            memories = await self._curator.after_turn(
                session_id     = session_id,
                user_text      = user_text,
                assistant_text = assistant_text,
                tool_calls     = tool_calls,
            )
            if memories:
                await self._append_facts(memories)
        except Exception as exc:
            logger.warning("DTM Track B write failed: %s", exc)

        # ── Periodic deep nudge (Hermes-style) ────────────────────────
        if turn_n % self._nudge_turns == 0:
            asyncio.create_task(self._deep_nudge(session_id, user_text))

        # ── Daily compaction (OpenClaw-style) ─────────────────────────
        if len(self._today_turns) >= self._compaction_turns:
            asyncio.create_task(self._compact_today())

        return memories

    async def recall(
        self,
        query:      str,
        session_id: str,
        max_chars:  int = 1500,
        top_k:      int = 6,
    ) -> DTMRecallResult:
        """
        Dual-track retrieval — replaces curator.recall_for_session().

        Runs Track A and Track B retrieval in parallel, arbitrates scores,
        MMR-deduplicates, and returns a ranked DTMRecallResult.
        """
        # ── Parallel retrieval ────────────────────────────────────────
        track_a_task = asyncio.create_task(
            self._retrieve_track_a(query, session_id, top_k * 2)
        )
        track_b_task = asyncio.create_task(
            self._retrieve_track_b(query, top_k * 2)
        )
        raw_a, raw_b = await asyncio.gather(track_a_task, track_b_task)

        # ── Arbitration and scoring ───────────────────────────────────
        scored: list[DTMResult] = []

        now = time.time()
        for r in raw_a:
            # Temporal decay: score halves every half_life_secs
            age_secs  = max(0.0, now - r.recency_ts)
            decay     = math.exp(-0.693 * age_secs / self._half_life_secs)
            final     = r.score * decay
            r.score   = round(final, 4)
            scored.append(r)

        for r in raw_b:
            # Facts get a weight boost — they're pre-abstracted signal
            r.score = round(r.score * self._track_b_weight, 4)
            scored.append(r)

        # ── MMR deduplication (OpenClaw-style) ────────────────────────
        selected = self._mmr_select(scored, top_k, lambda_=0.7)

        track_a_count = sum(1 for r in selected if r.track == "A")
        track_b_count = sum(1 for r in selected if r.track == "B")
        winner = (
            "B" if track_b_count > track_a_count
            else "A" if track_a_count > track_b_count
            else "tie"
        )

        # ── Build prompt context ───────────────────────────────────────
        prompt_context = self._build_prompt(selected, max_chars, query)

        logger.info(
            "DTM recall: query=%r A=%d B=%d winner=%s chars=%d",
            query[:40], track_a_count, track_b_count, winner, len(prompt_context),
        )

        return DTMRecallResult(
            results       = selected,
            track_a_count = track_a_count,
            track_b_count = track_b_count,
            winner        = winner,
            prompt_context= prompt_context,
        )

    async def pre_compaction_flush(self, session_id: str, recent_text: str) -> None:
        """
        OpenClaw's 'memory flush' — call this when context window is ~80% full.
        Forces immediate compaction of today's buffer so nothing is lost.
        Also triggers a Track B deep nudge to extract facts before context shrinks.
        """
        logger.info("DTM pre-compaction flush for session %s", session_id[:12])
        await self._compact_today()
        await self._deep_nudge(session_id, recent_text)

    # ------------------------------------------------------------------
    # Track A retrieval (static chunks, FTS5 + temporal decay)
    # ------------------------------------------------------------------

    async def _retrieve_track_a(
        self, query: str, session_id: str, limit: int
    ) -> list[DTMResult]:
        results: list[DTMResult] = []

        # 1. FTS5 raw turns (existing Hermes store)
        try:
            fts_results = await self._fts.search(
                query          = query,
                limit          = limit,
                session_exclude= session_id,
            )
            for r in fts_results:
                content = (
                    f"[Turn — {datetime.fromtimestamp(r.ts, tz=timezone.utc).strftime('%Y-%m-%d')}]\n"
                    f"Q: {r.user_text[:200]}\n"
                    f"A: {r.snippet[:300]}"
                )
                results.append(DTMResult(
                    content    = content,
                    track      = "A",
                    score      = max(0.01, 1.0 / (1.0 + abs(r.rank))),
                    source     = f"fts5:{r.session_id[:8]}",
                    memory_type= "raw_chunk",
                    recency_ts = r.ts,
                    tags       = [t.get("tool", "") for t in (r.tool_calls or [])],
                ))
        except Exception as exc:
            logger.debug("DTM Track A FTS5 search failed: %s", exc)

        # 2. Daily compacted .md files (OpenClaw-style chunked search)
        try:
            daily_results = await self._search_daily_files(query, limit // 2)
            results.extend(daily_results)
        except Exception as exc:
            logger.debug("DTM Track A daily file search failed: %s", exc)

        return results

    async def _search_daily_files(self, query: str, limit: int) -> list[DTMResult]:
        """
        Simple BM25-style keyword search over daily/*.md files.
        Falls back gracefully when no embeddings are available.
        """
        results: list[DTMResult] = []
        if not self._daily_dir.exists():
            return results

        query_words = set(query.lower().split())
        # Most recent files first
        daily_files = sorted(self._daily_dir.glob("*.md"), reverse=True)[:14]

        for path in daily_files:
            try:
                content = path.read_text(encoding="utf-8")
                date_str = path.stem  # "2026-04-15"
                ts = datetime.strptime(date_str, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                ).timestamp()

                # Chunk the file (OpenClaw-style ~400 token chunks with overlap)
                chunks = self._chunk_markdown(content, chunk_chars=1600, overlap_chars=200)
                for chunk_text, line_start in chunks:
                    chunk_lower = chunk_text.lower()
                    hits = sum(1 for w in query_words if w in chunk_lower and len(w) > 2)
                    if hits == 0:
                        continue
                    score = hits / max(len(query_words), 1)
                    results.append(DTMResult(
                        content    = f"[Daily {date_str} L{line_start}]\n{chunk_text[:500]}",
                        track      = "A",
                        score      = score,
                        source     = f"daily/{path.name}",
                        memory_type= "daily_chunk",
                        recency_ts = ts,
                    ))
            except Exception:
                continue

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    # ------------------------------------------------------------------
    # Track B retrieval (dynamic curated facts)
    # ------------------------------------------------------------------

    async def _retrieve_track_b(self, query: str, limit: int) -> list[DTMResult]:
        """
        Load curated facts from facts.jsonl and score against query.
        Simple keyword + memory_type relevance scoring.
        """
        results: list[DTMResult] = []
        if not self._facts_path.exists():
            return results

        query_words = set(query.lower().split())
        # Memory types most relevant for IT ops queries
        type_boost = {
            "incident_lesson":  1.3,
            "tool_pattern":     1.2,
            "operational_fact": 1.1,
            "environment_fact": 1.0,
            "user_preference":  0.8,
        }

        try:
            lines = self._facts_path.read_text(encoding="utf-8").splitlines()
            for line in lines[-500:]:  # last 500 facts max
                if not line.strip():
                    continue
                try:
                    fact = json.loads(line)
                except json.JSONDecodeError:
                    continue

                content    = fact.get("content", "")
                mtype      = fact.get("memory_type", "")
                confidence = float(fact.get("confidence", 0.5))
                tags       = fact.get("tags", [])
                ts         = float(fact.get("created_at", time.time()))

                # Keyword relevance
                content_lower = content.lower()
                tag_text      = " ".join(tags).lower()
                searchable    = content_lower + " " + tag_text + " " + mtype
                hits = sum(1 for w in query_words if w in searchable and len(w) > 2)
                if hits == 0:
                    continue

                kw_score   = hits / max(len(query_words), 1)
                boost      = type_boost.get(mtype, 1.0)
                final      = kw_score * confidence * boost

                results.append(DTMResult(
                    content    = f"[{mtype}] {content}",
                    track      = "B",
                    score      = round(final, 4),
                    source     = "facts.jsonl",
                    memory_type= mtype,
                    recency_ts = ts,
                    tags       = tags,
                ))
        except Exception as exc:
            logger.warning("DTM Track B retrieval failed: %s", exc)

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    # ------------------------------------------------------------------
    # MMR deduplication (OpenClaw-style)
    # ------------------------------------------------------------------

    @staticmethod
    def _mmr_select(
        candidates: list[DTMResult],
        k: int,
        lambda_: float = 0.7,
    ) -> list[DTMResult]:
        """
        Maximal Marginal Relevance selection.
        Balances relevance (score) with diversity (penalise similar content).
        lambda_=1.0 → pure relevance ranking. lambda_=0.0 → pure diversity.
        """
        if not candidates:
            return []
        candidates = sorted(candidates, key=lambda r: r.score, reverse=True)
        selected: list[DTMResult] = []

        while len(selected) < k and candidates:
            if not selected:
                best = candidates.pop(0)
            else:
                # MMR score = λ·relevance − (1−λ)·max_similarity_to_selected
                def mmr_score(c: DTMResult) -> float:
                    sel_words = set(" ".join(s.content.lower().split()) for s in selected)
                    c_words   = set(c.content.lower().split())
                    # Jaccard similarity
                    sim = len(c_words & sel_words) / max(len(c_words | sel_words), 1)
                    return lambda_ * c.score - (1 - lambda_) * sim

                best = max(candidates, key=mmr_score)
                candidates.remove(best)
            selected.append(best)

        return selected

    # ------------------------------------------------------------------
    # Daily compaction (OpenClaw-style)
    # ------------------------------------------------------------------

    async def _compact_today(self) -> None:
        """
        Flush today's in-memory turn buffer to daily/<YYYY-MM-DD>.md.
        This is the OpenClaw MEMORY.md / dated-file pattern applied to our
        IT ops context. The file is human-readable, git-versionable, and
        searchable by _search_daily_files.
        """
        if not self._today_turns:
            return

        turns = list(self._today_turns)
        self._today_turns.clear()

        date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        path = self._daily_dir / f"{date_str}.md"

        lines = []
        if path.exists():
            # Append to existing file for today
            lines.append("")  # blank line separator

        lines.append(f"## Session snapshot — {datetime.now(tz=timezone.utc).strftime('%H:%M UTC')}")
        lines.append("")

        for t in turns:
            session_short = t["session"][:8]
            ts_str = datetime.fromtimestamp(t["ts"], tz=timezone.utc).strftime("%H:%M")
            tools  = ", ".join(t["tools"]) if t["tools"] else "—"
            lines.append(f"**[{ts_str}]** `{session_short}` tools={tools}")
            lines.append(f"Q: {t['user'][:300]}")
            ans = t["assistant"][:400]
            lines.append(f"A: {ans}")
            lines.append("")

        content = "\n".join(lines)
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(content)
            logger.info("DTM compaction: %d turns → %s (%d chars)", len(turns), path.name, len(content))
        except Exception as exc:
            logger.warning("DTM compaction write failed: %s", exc)

    # ------------------------------------------------------------------
    # Deep nudge (Hermes-style periodic review)
    # ------------------------------------------------------------------

    async def _deep_nudge(self, session_id: str, recent_text: str) -> None:
        """
        Hermes periodic nudge — deeper LLM review beyond per-turn extraction.
        Looks at the last N turns together and checks for:
        - Cross-turn patterns (multiple similar incidents → generalised lesson)
        - Contradictions with existing facts
        - High-value operational facts that per-turn curator missed

        This is the Hermes 'shallow/deep nudge' mechanism from MemoryCurator,
        triggered periodically rather than every turn.
        """
        if not self._llm_fn:
            return
        try:
            result = await self._curator.periodic_nudge(session_id, recent_text)
            if result and result.memories_curated:
                await self._append_facts(result.memories_curated)
                logger.info(
                    "DTM deep nudge: session=%s curated=%d",
                    session_id[:12], len(result.memories_curated),
                )
        except AttributeError:
            # periodic_nudge may not exist in current curator version
            pass
        except Exception as exc:
            logger.debug("DTM deep nudge failed: %s", exc)

    # ------------------------------------------------------------------
    # Facts persistence (Track B)
    # ------------------------------------------------------------------

    async def _append_facts(self, memories: list) -> None:
        """Append curated memories to facts.jsonl."""
        self._facts_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._facts_path.open("a", encoding="utf-8") as f:
                for m in memories:
                    try:
                        record = {
                            "content":     getattr(m, "content", str(m)),
                            "memory_type": getattr(m, "memory_type", ""),
                            "confidence":  getattr(m, "confidence", 0.5),
                            "tags":        getattr(m, "tags", []),
                            "session_id":  getattr(m, "session_id", ""),
                            "created_at":  getattr(m, "created_at", time.time()),
                            "memory_id":   getattr(m, "memory_id", ""),
                        }
                        # Normalise enum values
                        if hasattr(record["memory_type"], "value"):
                            record["memory_type"] = record["memory_type"].value
                        f.write(json.dumps(record) + "\n")
                    except Exception:
                        pass
        except Exception as exc:
            logger.warning("DTM facts append failed: %s", exc)

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(results: list[DTMResult], max_chars: int, query: str) -> str:
        """
        Build the context string to inject before the LLM call.
        Track B facts first (more signal-dense), then Track A chunks.
        Deduplicates by content so the same fact isn't repeated.
        """
        if not results:
            return ""

        seen: set[str] = set()
        facts:  list[DTMResult] = []
        chunks: list[DTMResult] = []

        for r in results:
            key = r.content.strip()[:120]   # deduplicate by content prefix
            if key in seen:
                continue
            seen.add(key)
            if r.track == "B":
                facts.append(r)
            else:
                chunks.append(r)

        parts = []
        if facts:
            parts.append("── Relevant operational facts ──")
            for r in facts:
                parts.append(f"• {r.content}")
        if chunks:
            parts.append("── Related past context ──")
            for r in chunks:
                parts.append(r.content)

        text = "\n".join(parts)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n… [truncated]"
        return text

    # ------------------------------------------------------------------
    # Markdown chunker (OpenClaw-style sliding window)
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_markdown(
        content: str,
        chunk_chars: int = 1600,
        overlap_chars: int = 200,
    ) -> list[tuple[str, int]]:
        """
        Split markdown content into overlapping chunks.
        Returns list of (chunk_text, start_line_number).
        Mirrors OpenClaw's chunkMarkdown() logic.
        """
        lines = content.splitlines()
        chunks: list[tuple[str, int]] = []
        current: list[str] = []
        current_chars = 0
        start_line = 0

        for i, line in enumerate(lines):
            current.append(line)
            current_chars += len(line) + 1

            if current_chars >= chunk_chars:
                chunks.append(("\n".join(current), start_line))
                # Keep overlap from the end
                overlap_text = "\n".join(current)[-overlap_chars:]
                overlap_lines = overlap_text.splitlines()
                current = overlap_lines
                current_chars = sum(len(l) + 1 for l in overlap_lines)
                start_line = max(0, i - len(overlap_lines) + 1)

        if current:
            chunks.append(("\n".join(current), start_line))
        return chunks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_dirs(self) -> None:
        self._daily_dir.mkdir(parents=True, exist_ok=True)
        self._facts_path.parent.mkdir(parents=True, exist_ok=True)

    def stats(self) -> dict:
        """Return DTM statistics for the WebUI stats panel."""
        daily_count = len(list(self._daily_dir.glob("*.md"))) if self._daily_dir.exists() else 0
        facts_count = 0
        if self._facts_path.exists():
            try:
                facts_count = sum(1 for _ in self._facts_path.open())
            except Exception:
                pass
        return {
            "daily_files":      daily_count,
            "facts_count":      facts_count,
            "pending_turns":    len(self._today_turns),
            "track_b_weight":   self._track_b_weight,
            "compaction_turns": self._compaction_turns,
            "nudge_turns":      self._nudge_turns,
        }