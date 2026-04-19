"""
memory/curator.py
------------------
MemoryCurator — Hermes-style active memory curation.

Hermes innovation this implements
-----------------------------------
§03 环节一：策划记忆 (Curated Memory)
  "每轮对话结束后，Hermes会主动决定哪些信息值得记住。
   是主动决定，不是被动存储。"

§03 nudge机制
  "系统还有一个周期性的nudge机制，定时提醒Agent回顾最近的交互。"

How this differs from your current IngestionPipeline
------------------------------------------------------
Current system:  every turn is stored with heuristic importance scoring.
                 All turns go in; filtering happens at retrieval time.

MemoryCurator:   after EACH turn, asks the LLM:
                  "What from this conversation is worth remembering?
                   Extract only specific, reusable facts — not the chat."
                 Results are stored as high-importance ENTITY / USER_PREF records.
                 Ephemeral chat turns are also stored (FTS5) but clearly typed.

                 Periodic nudge: every N turns, reviews recent interactions
                 and asks: "Did anything new emerge that I haven't captured yet?"

Three types of curated memories
---------------------------------
  OPERATIONAL_FACT   — device states, confirmed diagnoses, tool results
  USER_PREFERENCE    — inferred operator habits, tool choices, style
  INCIDENT_LESSON    — what worked/failed in a specific incident response

Nudge schedule
--------------
  After every 5 turns:  shallow review of last 5 turns
  After every 20 turns: deep review + re-evaluation of existing USER_PREF records

LLM prompts (stub)
-------------------
Both prompts are designed to return structured JSON so they work with
any LLM including Ollama. Replace _call_llm() with your real engine.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Curated memory types
# ---------------------------------------------------------------------------

class CuratedMemoryType(str, Enum):
    OPERATIONAL_FACT  = "operational_fact"   # device state, confirmed finding
    USER_PREFERENCE   = "user_preference"    # operator style / tool preference
    INCIDENT_LESSON   = "incident_lesson"    # what worked/failed in an incident
    TOOL_PATTERN      = "tool_pattern"       # which tool was effective for what
    ENVIRONMENT_FACT  = "environment_fact"   # site config, network topology


@dataclass
class CuratedMemory:
    content:     str
    memory_type: CuratedMemoryType
    confidence:  float            # 0.0–1.0 how confident the LLM is
    session_id:  str
    source_turn: int              # turn number within the session
    tags:        list[str] = field(default_factory=list)
    created_at:  float     = field(default_factory=time.time)
    memory_id:   str       = field(default_factory=lambda: __import__("uuid").uuid4().hex[:12])


@dataclass
class NudgeResult:
    session_id:        str
    turns_reviewed:    int
    memories_curated:  list[CuratedMemory]
    shallow_nudge:     bool   # True = quick scan, False = deep review
    elapsed_ms:        float
# ---------------------------------------------------------------------------
# Prompt system instructions (stable LLM system messages)
# ---------------------------------------------------------------------------

_CURATION_SYSTEM = """You are a memory curator for an IT operations AI assistant.
A conversation turn just completed. Extract ONLY facts worth storing for future sessions.

Do NOT store:
- Greetings or pleasantries
- Questions without answers
- Temporary states (unless significant)
- Information already obvious from context

DO store:
- Confirmed device states, network facts, topology details
- Operator preferences revealed by choices (e.g. "prefers httpx over requests")
- Effective tool patterns ("syslog_search with lines=100 worked for this incident type")
- Incident lessons ("restarting the RADIUS service restored auth in INC-1291")
- Environment facts ("Site-A uses VLAN 10 for management, VLAN 20 for users")

Respond with ONLY a JSON array. No explanation, no preamble, no markdown fences.
If nothing is worth storing, respond with exactly: []
Each item: {"content": "...", "type": "operational_fact|user_preference|incident_lesson|tool_pattern|environment_fact", "confidence": 0.0-1.0, "tags": ["tag1","tag2"]}"""

_NUDGE_SHALLOW_SYSTEM = """You are reviewing recent IT operations conversation turns.
Extract any important facts, preferences, or lessons NOT yet captured in memory.

Respond with ONLY a JSON array. No explanation. If nothing new, respond with: []
Each item: {"content": "...", "type": "operational_fact|user_preference|incident_lesson|tool_pattern|environment_fact", "confidence": 0.0-1.0, "tags": ["tag1","tag2"]}"""

_NUDGE_DEEP_SYSTEM = """You are doing a deep review of an IT operator's working patterns.
Identify contradictions, refinements, or patterns to update in memory.

Respond with ONLY a JSON object. No explanation, no markdown fences.
Format: {"new_memories": [...], "corrections": [{"old_content": "...", "new_content": "...", "reason": "..."}]}"""

# Legacy aliases (kept for backward compat — replaced by SYSTEM variants above)
_NUDGE_SHALLOW_PROMPT = ""
_NUDGE_DEEP_PROMPT = ""




# ---------------------------------------------------------------------------
# MemoryCurator
# ---------------------------------------------------------------------------

class MemoryCurator:
    """
    Active memory curation engine — Hermes §03 learning loop, node 1.

    Integrates with:
      - FTS5SessionStore: reads recent turns for nudge review
      - MemoryRouter: writes curated records as high-importance entities
      - LLMEngine: uses an auxiliary lightweight LLM call for curation

    Nudge schedule:
      shallow_interval  = 5 turns  → quick scan of last 5 turns
      deep_interval     = 20 turns → full review + preference re-evaluation
    """

    def __init__(
        self,
        fts_store:          Any,         # FTS5SessionStore
        memory_router:      Any,         # MemoryRouter
        llm_fn:             Optional[Callable] = None,  # async(prompt) -> str
        shallow_interval:   int = 5,
        deep_interval:      int = 20,
        min_confidence:     float = 0.6,
    ) -> None:
        self._fts          = fts_store
        self._router       = memory_router
        self._llm_fn       = llm_fn  # None → _call_llm uses _stub_llm directly
        self._shallow_n    = shallow_interval
        self._deep_n       = deep_interval
        self._min_conf     = min_confidence
        self._turn_counter: dict[str, int] = {}   # session_id → turn count

    # ------------------------------------------------------------------
    # Called after every turn
    # ------------------------------------------------------------------

    async def after_turn(
        self,
        session_id:     str,
        user_text:      str,
        assistant_text: str,
        tool_calls:     Optional[list[dict]] = None,
    ) -> list[CuratedMemory]:
        """
        Primary curation hook — call immediately after each turn completes.
        Asks the LLM: "What from this turn is worth remembering?"
        Returns curated memories (already written to MemoryRouter).
        """
        self._turn_counter[session_id] = self._turn_counter.get(session_id, 0) + 1
        turn_num = self._turn_counter[session_id]

        # Step 1: per-turn curation
        curated = await self._curate_turn(
            session_id, user_text, assistant_text,
            tool_calls or [], turn_num,
        )

        # Step 2: nudge (if interval hit)
        nudge_curated: list[CuratedMemory] = []
        if turn_num % self._deep_n == 0:
            nudge = await self.nudge(session_id, deep=True)
            nudge_curated = nudge.memories_curated
        elif turn_num % self._shallow_n == 0:
            nudge = await self.nudge(session_id, deep=False)
            nudge_curated = nudge.memories_curated

        return curated + nudge_curated

    # ------------------------------------------------------------------
    # Per-turn curation
    # ------------------------------------------------------------------

    async def _curate_turn(
        self,
        session_id:     str,
        user_text:      str,
        assistant_text: str,
        tool_calls:     list[dict],
        turn_num:       int,
    ) -> list[CuratedMemory]:
        tool_summary = json.dumps(
            [{"tool": tc.get("tool", ""), "result_preview": str(tc.get("result", ""))[:100]}
             for tc in tool_calls[:3]]
        )
        user_content = (
            f"User: {user_text[:600]}\n"
            f"Assistant: {assistant_text[:600]}\n"
            f"Tool calls made: {tool_summary}"
        )
        raw = await self._call_llm(_CURATION_SYSTEM, user_content)
        memories = self._parse_memory_list(raw, session_id, turn_num)
        await self._persist_memories(memories)
        logger.debug(
            "MemoryCurator.after_turn: session=%s turn=%d → %d memories",
            session_id, turn_num, len(memories),
        )
        return memories

    # ------------------------------------------------------------------
    # Nudge mechanism (Hermes periodic review)
    # ------------------------------------------------------------------

    async def nudge(
        self,
        session_id: str,
        deep:       bool = False,
    ) -> NudgeResult:
        """
        Periodic review of recent turns for missed memories.

        shallow (every 5 turns):  scan last 5 turns, find anything missed
        deep (every 20 turns):    full history + re-evaluate USER_PREF records,
                                   detect contradictions
        """
        start = time.monotonic()
        n_turns = self._deep_n if deep else self._shallow_n

        # Fetch recent turns from FTS store
        recent = await self._fts.get_session_turns(
            session_id, limit=n_turns, min_importance=0.0
        )
        if not recent:
            return NudgeResult(session_id, 0, [], not deep, 0.0)

        turns_text = "\n---\n".join(
            f"Turn {i+1}: User: {t.user_text[:200]}\nAssistant: {t.assistant_text[:200]}"
            for i, t in enumerate(reversed(recent))
        )

        # Existing memories for dedup / deep comparison
        existing = await self._get_existing_memories(session_id)
        existing_text = "\n".join(f"- {m}" for m in existing[:20])

        if deep:
            existing_prefs = [m for m in existing if "prefer" in m.lower() or "habit" in m.lower()]
            existing_prefs_text = "\n".join(f"- {p}" for p in existing_prefs[:10])
            user_content = (
                f"Recent history ({len(recent)} turns):\n{turns_text[:1500]}\n\n"
                f"Existing user preferences:\n{existing_prefs_text}"
            )
            raw = await self._call_llm(_NUDGE_DEEP_SYSTEM, user_content)
            memories = self._parse_deep_nudge(raw, session_id, len(recent))
        else:
            user_content = (
                f"Turns to review ({len(recent)} turns):\n{turns_text[:1200]}\n\n"
                f"Already in memory (do not duplicate):\n{existing_text}"
            )
            raw = await self._call_llm(_NUDGE_SHALLOW_SYSTEM, user_content)
            memories = self._parse_memory_list(raw, session_id, len(recent))

        await self._persist_memories(memories)
        await self._fts.write_nudge_log(session_id, len(recent), len(memories))

        elapsed = (time.monotonic() - start) * 1000
        logger.info(
            "MemoryCurator.nudge: session=%s deep=%s turns=%d → %d new memories (%.0fms)",
            session_id, deep, len(recent), len(memories), elapsed,
        )
        return NudgeResult(session_id, len(recent), memories, not deep, elapsed)

    # ------------------------------------------------------------------
    # Cross-session recall hook (called before new session)
    # ------------------------------------------------------------------

    async def recall_for_session(
        self,
        new_query:      str,
        session_id:     str,
        max_chars:      int = 1200,
    ) -> str:
        """
        Before starting a new session / new turn, search FTS5 for related
        past context and return a summarized string for prompt injection.

        This is the core Hermes FTS5 recall:
          "新对话开始前，根据当前话题搜索历史记忆"
        """
        results = await self._fts.search(
            query=new_query,
            limit=6,
            session_exclude=session_id,
        )
        if not results:
            return ""
        summary = await self._fts.summarize_results(results, new_query, max_chars)
        return summary

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    async def _persist_memories(self, memories: list[CuratedMemory]) -> None:
        for mem in memories:
            if mem.confidence < self._min_conf:
                continue
            try:
                await self._router.ingest_entity(
                    session_id=mem.session_id,
                    entity_text=mem.content,
                    entity_type=mem.memory_type.value,
                )
            except Exception as exc:
                logger.warning("MemoryCurator: persist failed for %r: %s", mem.content[:60], exc)

    async def _get_existing_memories(self, session_id: str) -> list[str]:
        """Get already-curated memory contents for deduplication."""
        try:
            from memory.schemas import MemoryRecordType, RetrievalQuery, MemoryTier
            query = RetrievalQuery(
                query_text="preferences habits patterns",
                session_id=session_id,
                top_k=20,
                tiers=[MemoryTier.LONG_TERM, MemoryTier.MID_TERM],
            )
            results = await self._router.retrieve(query)
            return [r.record.content for r in results if r.record.record_type == MemoryRecordType.ENTITY]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # LLM caller — wraps whatever fn was injected at init time
    # ------------------------------------------------------------------

    async def _call_llm(self, system: str, user: str) -> str:
        """
        Call the LLM with explicit system (instruction) + user (data) separation.
        Always returns valid JSON (array or object) for safe parsing.
        Strips <think> reasoning blocks from thinking models.
        """
        import re as _re
        if self._llm_fn is None:
            return await self._stub_llm(user)
        try:
            raw = await self._llm_fn(system, user)
            raw = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL | _re.IGNORECASE).strip()
            # Reject non-JSON — model echoed prompt, returned prose, or errored
            first_char = raw.lstrip()[:1]
            if not raw or first_char not in ("[", "{"):
                logger.debug("MemoryCurator: non-JSON response (%r...) — returning []", raw[:60])
                return "[]"
            return raw
        except Exception as exc:
            logger.warning("MemoryCurator: llm_fn failed (%s) — returning []", exc)
            return "[]"

    @staticmethod
    async def _stub_llm(text: str) -> str:
        """
        Deterministic stub — always returns a valid JSON array string.
        Guards against being called with raw LLM output instead of a curation prompt.
        """
        await asyncio.sleep(0)
        import re as _re
        # Only extract memories from structured curation prompt format
        if not ("User:" in text or "Assistant:" in text or "Tool calls" in text):
            return "[]"
        memories = []

        # Detect device mentions
        devices = _re.findall(r"\b(ap-\d+|sw-[\w-]+|router-\d+)\b", text, _re.IGNORECASE)
        if devices:
            dev = devices[0]
            memories.append({
                "content": f"Device {dev} was involved in this session's diagnostic",
                "type": "operational_fact", "confidence": 0.72, "tags": ["device", dev],
            })

        # Detect tool preferences
        if "httpx" in text.lower():
            memories.append({
                "content": "Operator prefers httpx library for HTTP calls",
                "type": "user_preference", "confidence": 0.78, "tags": ["tools", "python"],
            })

        # Detect incident lessons
        if any(w in text.lower() for w in ["radius", "auth failure", "authentication"]):
            memories.append({
                "content": "RADIUS authentication failures investigated — check RADIUS timeout and certificate expiry",
                "type": "incident_lesson", "confidence": 0.82, "tags": ["auth", "radius"],
            })

        if "syslog" in text.lower() and "error" in text.lower():
            memories.append({
                "content": "syslog_search with severity=error effective for AP diagnostic queries",
                "type": "tool_pattern", "confidence": 0.70, "tags": ["syslog", "diagnostic"],
            })

        return json.dumps(memories)

    # ------------------------------------------------------------------
    # JSON parsers
    # ------------------------------------------------------------------

    def _parse_memory_list(
        self,
        raw: str,
        session_id: str,
        turn_num: int,
    ) -> list[CuratedMemory]:
        """Parse LLM output into CuratedMemory objects."""
        try:
            text = raw.strip()
            # Strip markdown fences if present
            text = re.sub(r"^```json?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            # Find JSON array
            m = re.search(r"\[.*\]", text, re.DOTALL)
            if not m:
                return []
            items = json.loads(m.group(0))
            memories = []
            for item in items:
                if not isinstance(item, dict) or "content" not in item:
                    continue
                try:
                    mem_type = CuratedMemoryType(item.get("type", "operational_fact"))
                except ValueError:
                    mem_type = CuratedMemoryType.OPERATIONAL_FACT
                memories.append(CuratedMemory(
                    content=item["content"][:500],
                    memory_type=mem_type,
                    confidence=float(item.get("confidence", 0.7)),
                    session_id=session_id,
                    source_turn=turn_num,
                    tags=item.get("tags", []),
                ))
            return memories
        except Exception as exc:
            logger.warning("MemoryCurator: parse failed: %s — raw=%r", exc, raw[:200])
            return []

    def _parse_deep_nudge(
        self,
        raw: str,
        session_id: str,
        turn_num: int,
    ) -> list[CuratedMemory]:
        """Parse deep nudge response which may include corrections."""
        try:
            text = raw.strip()
            text = re.sub(r"^```json?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            data = json.loads(text)
            new_items = data.get("new_memories", [])
            # Build CuratedMemory from new items
            memories = []
            for item in new_items:
                if not isinstance(item, dict):
                    continue
                try:
                    mem_type = CuratedMemoryType(item.get("type", "user_preference"))
                except ValueError:
                    mem_type = CuratedMemoryType.USER_PREFERENCE
                memories.append(CuratedMemory(
                    content=item.get("content", "")[:500],
                    memory_type=mem_type,
                    confidence=float(item.get("confidence", 0.7)),
                    session_id=session_id,
                    source_turn=turn_num,
                    tags=item.get("tags", []),
                ))
            # Log corrections (don't process automatically — they need HITL)
            corrections = data.get("corrections", [])
            if corrections:
                logger.info(
                    "MemoryCurator deep nudge: %d corrections suggested (require review)",
                    len(corrections),
                )
            return memories
        except Exception:
            # Fall back to parsing as plain list
            return self._parse_memory_list(raw, session_id, turn_num)