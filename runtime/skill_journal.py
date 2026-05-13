"""
runtime/skill_journal.py
------------------------
SkillJournal — per-session, per-turn observability for skill orchestration.

Purpose
-------
The runtime loop currently picks top-K skills from a catalog and injects
their summaries into the LLM's prompt. Whether the LLM follows a skill,
mid-flight abandons it, or improvises has NO observable signal. This makes:

  - Failure-mode analysis impossible ("which skills are weak?")
  - SkillEvolver feedback noisy (it can't tell which skill caused success/failure)
  - Operator confidence low ("did the agent even use the skill it loaded?")

This module is a passive observer. It does NOT change control flow:
  - No retries
  - No fallbacks
  - No prompts injected based on journal state

It just records what happened so we can:
  1. Look at it via WebUI / API
  2. Feed it to evolver as training signal
  3. Detect patterns that justify *future* control-flow changes (Plan B/C)

Once we have 1-2 weeks of journal data, we'll have evidence to decide if
fallback chains or auto-HITL escalation are actually warranted.

Public API
----------
    journal = SkillJournal(session_id=..., query=...)

    journal.record_selection(top_k_skills=[(id, score), ...], ambiguous=False)
    journal.record_skill_load(skill_id, turn=N, position=P)
    journal.record_tool_call(turn, tool_name, args, ok, error=None)
    journal.record_completion(outcome=..., final_response_preview=..., turns=N)

    journal.to_dict()   →  JSON-serialisable summary for storage / display
    journal.attributed_skills() → which skills appear to have driven progress

Storage
-------
SkillJournalStore is a thin process-wide append-only buffer; entries beyond
cfg.skill_orchestration.journal_max_entries are LRU-evicted. Optionally
persisted to disk (cfg.skill_orchestration.journal_persist_path).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from typing      import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

@dataclass
class _Event:
    """One recorded event in the journal. type field discriminates payload."""
    type:      str          # "selection" | "load" | "tool_call" | "completion" | "note"
    turn:      int
    ts:        float        # monotonic seconds since journal start
    payload:   dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillJournal:
    """Per-task journal. One instance per stream() invocation.

    Lifecycle:
      __init__ → record_*() during stream → to_dict() at end → store
    """
    session_id:   str
    query:        str
    started_at:   float = field(default_factory=time.time)
    _t0:          float = field(default_factory=time.monotonic)
    events:       list[_Event] = field(default_factory=list)

    # Convenience accumulators
    top_k:        list[tuple[str, float]] = field(default_factory=list)
    ambiguous:    bool = False
    loaded_skills: list[str] = field(default_factory=list)
    tool_calls:   list[dict[str, Any]] = field(default_factory=list)
    outcome:      Optional[str] = None
    total_turns:  int = 0

    # ── Recording (thread-safe enough — runtime loop is single-task per journal) ──

    def record_selection(
        self,
        top_k_skills:  list[tuple[str, float]],
        ambiguous:     bool,
        turn:          int = 1,
    ) -> None:
        self.top_k     = list(top_k_skills)
        self.ambiguous = bool(ambiguous)
        self._emit("selection", turn, {
            "top_k":     [(sid, round(float(sc), 4)) for sid, sc in top_k_skills],
            "ambiguous": ambiguous,
        })

    def record_skill_load(
        self,
        skill_id: str,
        turn:     int,
        position: Optional[int] = None,    # index in top_k (0-based) when known
        score:    Optional[float] = None,
    ) -> None:
        self.loaded_skills.append(skill_id)
        self._emit("load", turn, {
            "skill_id": skill_id,
            "position": position,
            "score":    (round(float(score), 4) if score is not None else None),
        })

    def record_tool_call(
        self,
        turn:      int,
        tool_name: str,
        args:      Optional[dict[str, Any]] = None,
        ok:        bool = True,
        error:     Optional[str] = None,
        elapsed_ms: Optional[float] = None,
    ) -> None:
        # Don't log args verbatim — could contain secrets. Just keys + types.
        arg_shape: dict[str, str] = {}
        for k, v in (args or {}).items():
            arg_shape[k] = type(v).__name__
        entry = {
            "tool_name":  tool_name,
            "arg_shape":  arg_shape,
            "ok":         ok,
            "error":      error[:200] if error else None,
            "elapsed_ms": round(elapsed_ms, 1) if elapsed_ms else None,
        }
        self.tool_calls.append({"turn": turn, **entry})
        self._emit("tool_call", turn, entry)

    def record_completion(
        self,
        outcome:                str,   # "answer" | "interrupted" | "error" | "max_turns"
        final_response_preview: str = "",
        total_turns:            int = 0,
    ) -> None:
        self.outcome     = outcome
        self.total_turns = total_turns
        self._emit("completion", total_turns, {
            "outcome": outcome,
            "preview": (final_response_preview or "")[:300].replace("\n", " "),
        })

    def note(self, turn: int, message: str) -> None:
        """Free-form annotation, e.g. 'skill_X marked stuck' for future Plan B."""
        self._emit("note", turn, {"message": message[:300]})

    # ── Aggregation / introspection ──────────────────────────────────

    def attributed_skills(self) -> list[dict[str, Any]]:
        """Best-effort attribution of work to skills.

        For each loaded skill, count the tool calls that happened on the
        turns BETWEEN its load and the next skill load (or end). This isn't
        perfect (the LLM might do work for skill A on turn 3 even after
        loading skill B on turn 4), but it's a strong-enough signal for the
        evolver to use as training data, and it surfaces "loaded but no
        work done" as a red flag.
        """
        if not self.loaded_skills:
            return []

        # Build load-turn map by walking events
        load_turns: list[tuple[str, int]] = []
        for ev in self.events:
            if ev.type == "load":
                load_turns.append((ev.payload["skill_id"], ev.turn))

        out = []
        for i, (sid, load_turn) in enumerate(load_turns):
            next_turn = load_turns[i + 1][1] if i + 1 < len(load_turns) else self.total_turns + 1
            tools_in_window = [
                tc for tc in self.tool_calls
                if load_turn <= tc["turn"] < next_turn
            ]
            out.append({
                "skill_id":           sid,
                "loaded_at_turn":     load_turn,
                "active_until_turn":  next_turn - 1,
                "tool_calls_in_window": len(tools_in_window),
                "tool_calls_failed":    sum(1 for tc in tools_in_window if not tc.get("ok", True)),
                "appeared_dormant":     (len(tools_in_window) == 0),  # red flag
            })
        return out

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary for storage / WebUI display."""
        return {
            "session_id":     self.session_id,
            "query":          self.query[:300],
            "started_at":     self.started_at,
            "duration_s":     round(time.monotonic() - self._t0, 2),
            "top_k":          [{"id": sid, "score": round(float(sc), 4)} for sid, sc in self.top_k],
            "ambiguous":      self.ambiguous,
            "loaded_skills":  list(self.loaded_skills),
            "tool_calls":     list(self.tool_calls),
            "outcome":        self.outcome,
            "total_turns":    self.total_turns,
            "attribution":    self.attributed_skills(),
            "events":         [
                {"type": e.type, "turn": e.turn, "ts": round(e.ts, 3), "payload": e.payload}
                for e in self.events
            ],
        }

    # ── Internal ─────────────────────────────────────────────────────

    def _emit(self, ev_type: str, turn: int, payload: dict[str, Any]) -> None:
        self.events.append(_Event(
            type=ev_type, turn=turn,
            ts=time.monotonic() - self._t0,
            payload=payload,
        ))


# ---------------------------------------------------------------------------
# Process-wide store
# ---------------------------------------------------------------------------

class SkillJournalStore:
    """LRU-bounded in-memory buffer of completed journals.

    Optionally persists each completed journal to disk as JSONL.
    """

    def __init__(self, max_entries: int = 200, persist_path: Optional[str] = None):
        self._lock      = threading.RLock()
        self._max       = int(max_entries)
        self._persist   = persist_path
        self._entries: list[dict[str, Any]] = []

    def append(self, journal_dict: dict[str, Any]) -> None:
        with self._lock:
            self._entries.append(journal_dict)
            while len(self._entries) > self._max:
                self._entries.pop(0)
            # Persist under the same lock — without this, two threads
            # writing JSONL concurrently can interleave bytes mid-line and
            # produce un-parseable corrupt records (POSIX guarantees
            # atomic writes only up to PIPE_BUF ~4KB; a journal entry
            # often exceeds that).
            if self._persist:
                try:
                    with open(self._persist, "a", encoding="utf-8") as fp:
                        fp.write(json.dumps(journal_dict, ensure_ascii=False) + "\n")
                except Exception as exc:
                    logger.warning(
                        "SkillJournalStore: persist failed (%s) — continuing in-memory only",
                        exc,
                    )

    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            return list(reversed(self._entries[-limit:]))

    def filter(self, *, skill_id: Optional[str] = None, outcome: Optional[str] = None,
               ambiguous: Optional[bool] = None, limit: int = 50) -> list[dict[str, Any]]:
        out = []
        with self._lock:
            for entry in reversed(self._entries):
                if skill_id and skill_id not in entry.get("loaded_skills", []):
                    continue
                if outcome and entry.get("outcome") != outcome:
                    continue
                if ambiguous is not None and entry.get("ambiguous") != ambiguous:
                    continue
                out.append(entry)
                if len(out) >= limit:
                    break
        return out

    def stats(self) -> dict[str, Any]:
        """Aggregate counters for dashboarding."""
        with self._lock:
            n = len(self._entries)
            if n == 0:
                return {"count": 0}

            outcomes:   dict[str, int] = {}
            skill_use:  dict[str, int] = {}
            skill_dormant: dict[str, int] = {}
            ambiguous_count = 0

            for e in self._entries:
                outcomes[e.get("outcome") or "unknown"] = outcomes.get(e.get("outcome") or "unknown", 0) + 1
                if e.get("ambiguous"):
                    ambiguous_count += 1
                for attr in e.get("attribution", []):
                    sid = attr["skill_id"]
                    skill_use[sid] = skill_use.get(sid, 0) + 1
                    if attr.get("appeared_dormant"):
                        skill_dormant[sid] = skill_dormant.get(sid, 0) + 1

            return {
                "count":            n,
                "outcomes":         outcomes,
                "ambiguous_rate":   round(ambiguous_count / n, 3),
                "skill_use_count":  skill_use,
                "skill_dormant_count": skill_dormant,
            }


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_GLOBAL_STORE: Optional[SkillJournalStore] = None


def get_journal_store(
    *,
    max_entries:   int = 200,
    persist_path:  Optional[str] = None,
) -> SkillJournalStore:
    """Process-wide journal store. First caller's params win."""
    global _GLOBAL_STORE
    if _GLOBAL_STORE is None:
        _GLOBAL_STORE = SkillJournalStore(max_entries=max_entries, persist_path=persist_path)
    return _GLOBAL_STORE