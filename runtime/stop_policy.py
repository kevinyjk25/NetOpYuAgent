"""
runtime/stop_policy.py
-----------------------
StopPolicy — explicit stop conditions for every agent loop turn.

Why this exists
---------------
Without stop conditions, an agent can spin indefinitely: calling tools that
return inconclusive results, reformulating the same query, or getting stuck in
low-confidence loops while consuming tokens and user time.

StopPolicy evaluates a set of counters and thresholds after every turn and
returns a StopDecision that tells the runtime what to do next.

Stop conditions (all configurable)
------------------------------------
  max_turns               Hard ceiling on loop iterations
  max_tool_calls          Total tool invocations across all turns
  max_no_progress_turns   Consecutive turns with no new confirmed fact or
                          meaningful output change → graceful stop
  token_budget            Cumulative input tokens consumed
  confidence_floor        If intent confidence stays below this after N turns,
                          stop rather than guess
  low_confidence_turns    Number of turns allowed below confidence_floor

Stop outcomes
-------------
  CONTINUE      Normal, keep looping
  STOP_GRACEFUL Loop has run its course; emit best-effort summary
  STOP_HITL     Uncertain enough that a human should decide
  STOP_BUDGET   Token or turn hard limit hit

Usage
-----
    policy = StopPolicy()
    state  = LoopState()

    for turn in loop:
        state.turns += 1
        state.tool_calls += n_tools_called_this_turn

        decision = policy.evaluate(state)
        if decision.should_stop:
            emit_summary(decision.reason)
            break
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class StopOutcome(str, Enum):
    CONTINUE      = "continue"
    STOP_GRACEFUL = "stop_graceful"   # natural end, emit summary
    STOP_HITL     = "stop_hitl"       # escalate to human
    STOP_BUDGET   = "stop_budget"     # hard limit hit
    USER_CANCELLED = "user_cancelled" # operator hit Stop mid-stream


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class StopPolicyConfig:
    # Turn-based limits
    max_turns:              int   = 30
    max_tool_calls:         int   = 50
    max_no_progress_turns:  int   = 10

    # Token budget (soft; stop before hitting model hard limit)
    token_budget:           int   = 50_000

    # Confidence
    confidence_floor:       float = 0.45   # below this for too long → stop
    low_confidence_turns:   int   = 2      # allowed turns below floor

    # Clarification gate
    # When the agent's confidence is below `clarification_threshold` AND it
    # has not yet exhausted the per-session budget, the loop emits a
    # CLARIFICATION HITL interrupt instead of guessing. The threshold sits
    # above confidence_floor so we ASK rather than escalate to a generic
    # HITL stop. Set max_clarifications=0 to disable entirely.
    clarification_threshold:    float = 0.50
    max_clarifications:         int   = 2

    # Parallel delegation guard
    max_parallel_delegations: int = 5


# ---------------------------------------------------------------------------
# FactsLedger — typed, category-aware replacement for the raw confirmed_facts
# ---------------------------------------------------------------------------

class FactCategory:
    """Category prefixes for FactsLedger entries."""
    FACT          = "[FACT]"           # real confirmed network fact
    TOOL_EXEC     = "TOOL_EXEC:"       # tool execution record
    PREV_ANALYSIS = "PREV_ANALYSIS:"   # synthesis summary from previous turn
    NUDGE         = "_NUDGE:"          # internal LLM control instruction
    LESSON        = "[LESSON]"         # reflection lesson


class FactsLedger:
    """
    DESIGN-06 fix: typed multi-bucket store replacing the mixed confirmed_facts list.

    Each category has an independent entry list and a configurable max-entries cap
    (loaded from AppConfig if available, otherwise from class defaults).
    The ledger serialises to list[str] for backwards compatibility with all
    existing code that reads state.confirmed_facts.

    Priority for context injection (highest first):
      1. FACT        — real confirmed facts, highest value
      2. PREV_ANALYSIS — useful cross-turn context
      3. TOOL_EXEC   — tool call records (for deduplication)
      4. LESSON      — reflection lessons
      5. NUDGE       — ephemeral control, excluded from serialisation by default
    """

    # Default per-category caps (overridden by config if available)
    _DEFAULT_CAPS: dict[str, int] = {
        FactCategory.FACT:          200,
        FactCategory.TOOL_EXEC:     100,
        FactCategory.PREV_ANALYSIS:  20,
        FactCategory.LESSON:         50,
        FactCategory.NUDGE:           5,
    }

    def __init__(self, caps: Optional[dict[str, int]] = None):
        self._caps: dict[str, int] = caps or dict(self._DEFAULT_CAPS)
        self._buckets: dict[str, list[str]] = {cat: [] for cat in self._caps}

    # ── Write ────────────────────────────────────────────────────────

    def add(self, text: str, category: str = FactCategory.FACT) -> None:
        """Add an entry.  Oldest entries are dropped when the cap is reached."""
        if category not in self._buckets:
            self._buckets[category] = []
        bucket = self._buckets[category]
        cap    = self._caps.get(category, 200)
        if len(bucket) >= cap:
            # Evict oldest half to avoid O(N) churn on every add
            evict = max(1, cap // 4)
            del bucket[:evict]
        bucket.append(text)

    def add_fact(self, text: str)          -> None: self.add(text, FactCategory.FACT)
    def add_tool_exec(self, text: str)     -> None: self.add(text, FactCategory.TOOL_EXEC)
    def add_prev_analysis(self, text: str) -> None: self.add(text, FactCategory.PREV_ANALYSIS)
    def add_nudge(self, text: str)         -> None: self.add(text, FactCategory.NUDGE)
    def add_lesson(self, text: str)        -> None: self.add(text, FactCategory.LESSON)

    def clear_nudges(self) -> None:
        self._buckets[FactCategory.NUDGE] = []

    # ── Read ─────────────────────────────────────────────────────────

    def to_list(self, include_nudges: bool = False) -> list[str]:
        """Serialise to list[str] in priority order for backwards compat."""
        out: list[str] = []
        order = [
            FactCategory.FACT,
            FactCategory.PREV_ANALYSIS,
            FactCategory.TOOL_EXEC,
            FactCategory.LESSON,
        ]
        if include_nudges:
            order.append(FactCategory.NUDGE)
        for cat in order:
            out.extend(self._buckets.get(cat, []))
        return out

    def facts_only(self) -> list[str]:
        return list(self._buckets.get(FactCategory.FACT, []))

    def tool_execs(self) -> list[str]:
        return list(self._buckets.get(FactCategory.TOOL_EXEC, []))

    def nudges(self) -> list[str]:
        return list(self._buckets.get(FactCategory.NUDGE, []))

    # ── Backwards-compat: full list emulation for legacy readers ────
    # FactsLedger duck-types as list[str] so any code that treats
    # confirmed_facts as a list (slicing, contains, iteration, append/extend)
    # works without isinstance checks.

    def __iter__(self):
        return iter(self.to_list())

    def __len__(self) -> int:
        return sum(len(b) for b in self._buckets.values())

    def __bool__(self) -> bool:
        # Explicit because some callers do `if confirmed_facts:` and we
        # want truthy when ANY bucket has items, not just FACT.
        return self.__len__() > 0

    def __getitem__(self, key):
        """Slicing/indexing returns from the priority-ordered serialised view.
        Supports the most common patterns we observed in the codebase:
          facts[:20]   → ContextBudgetManager._format_confirmed_facts
          facts[-5:]   → StopPolicy._build_summary
          facts[-3:]   → coordinator.py shared_facts
          facts[i]     → element access
        """
        return self.to_list()[key]

    def __contains__(self, item) -> bool:
        """Supports `f in confirmed_facts` checks across all buckets."""
        for bucket in self._buckets.values():
            if item in bucket:
                return True
        return False

    def append(self, item: str) -> None:
        """List-compatible append. Routes by prefix to the correct bucket so
        legacy code doing `state.confirmed_facts.append("TOOL_EXEC: …")`
        still ends up in the right place."""
        if not isinstance(item, str):
            item = str(item)
        if   item.startswith(FactCategory.TOOL_EXEC):     self.add_tool_exec(item)
        elif item.startswith(FactCategory.PREV_ANALYSIS): self.add_prev_analysis(item)
        elif item.startswith(FactCategory.NUDGE):         self.add_nudge(item)
        elif item.startswith(FactCategory.LESSON):        self.add_lesson(item)
        else:                                             self.add_fact(item)

    def extend(self, items) -> None:
        """List-compatible extend with per-item prefix routing."""
        for it in items or []:
            self.append(it)

    def __repr__(self) -> str:
        sizes = {cat: len(b) for cat, b in self._buckets.items() if b}
        return f"FactsLedger({sizes})"

    @classmethod
    def from_list(cls, items: list[str]) -> "FactsLedger":
        """Reconstruct a FactsLedger from a serialised list[str] (e.g. from
        a previous session's confirmed_facts).  Uses prefix detection to
        route each item into the correct bucket."""
        ledger = cls()
        for item in (items or []):
            if item.startswith(FactCategory.TOOL_EXEC):
                ledger.add_tool_exec(item)
            elif item.startswith(FactCategory.PREV_ANALYSIS):
                ledger.add_prev_analysis(item)
            elif item.startswith(FactCategory.NUDGE):
                ledger.add_nudge(item)
            elif item.startswith(FactCategory.LESSON):
                ledger.add_lesson(item)
            else:
                ledger.add_fact(item)
        return ledger


# ---------------------------------------------------------------------------
# Mutable loop state (caller maintains this across turns)
# ---------------------------------------------------------------------------

@dataclass
class LoopState:
    # Counters
    turns:               int   = 0
    tool_calls:          int   = 0
    parallel_delegations: int  = 0
    tokens_consumed:     int   = 0

    # Progress tracking
    no_progress_turns:   int   = 0
    last_response_hash:  Optional[str] = None    # detect repeated outputs

    # Clarification budget — caps active "ask the operator" interrupts
    # so a hopelessly under-specified query can't trap the loop in a
    # rapid-fire interrogation. Soft cap; the loop falls back to default
    # assumptions or stop_graceful when exceeded.
    clarifications_asked: int = 0

    # Confidence tracking
    current_confidence:  float = 1.0
    low_confidence_turns_count: int = 0

    # Accumulated outputs
    # DESIGN-06 fix: confirmed_facts is now a FactsLedger that separates
    # real facts, tool exec records, analysis summaries, and nudges into
    # independent capped buckets. It serialises to list[str] for compat.
    confirmed_facts:     FactsLedger = field(default_factory=FactsLedger)
    unresolved_points:   list[str] = field(default_factory=list)
    tool_summaries:      list[str] = field(default_factory=list)

    def record_tool_call(self, tool_name: str, summary: str = "") -> None:
        self.tool_calls += 1
        if summary:
            self.tool_summaries.append(f"{tool_name}: {summary}")

    def record_new_fact(self, fact: str) -> None:
        """Call when the agent confirms a new structured fact."""
        if isinstance(self.confirmed_facts, FactsLedger):
            self.confirmed_facts.add_fact(fact)
        else:
            self.confirmed_facts.append(fact)  # type: ignore[union-attr]
        self.no_progress_turns = 0   # reset stall counter

    def record_no_progress(self) -> None:
        self.no_progress_turns += 1

    def record_response(self, response_text: str) -> bool:
        """
        Hash the response to detect repetition.
        Returns True if this response is meaningfully different from the last.
        """
        import hashlib
        h = hashlib.md5(response_text.encode(), usedforsecurity=False).hexdigest()[:8]
        if h == self.last_response_hash:
            self.record_no_progress()
            return False
        self.last_response_hash = h
        return True

    def update_confidence(self, confidence: float) -> None:
        self.current_confidence = confidence


# ---------------------------------------------------------------------------
# Stop decision
# ---------------------------------------------------------------------------

@dataclass
class StopDecision:
    outcome: StopOutcome
    reason:  str
    summary: str = ""   # best-effort summary to emit before stopping

    @property
    def should_stop(self) -> bool:
        return self.outcome != StopOutcome.CONTINUE


# ---------------------------------------------------------------------------
# StopPolicy
# ---------------------------------------------------------------------------

class StopPolicy:
    """
    Evaluates whether the agent loop should continue after each turn.

    Call evaluate(state) at the end of every turn.  The returned StopDecision
    tells the runtime whether to continue, stop gracefully, escalate to HITL,
    or halt due to budget exhaustion.

    All thresholds are configurable via StopPolicyConfig.
    """

    def __init__(self, config: Optional[StopPolicyConfig] = None) -> None:
        self._cfg = config or StopPolicyConfig()

    def evaluate(self, state: LoopState) -> StopDecision:
        """
        Check all stop conditions in priority order.
        First matching condition wins.
        """
        cfg = self._cfg

        # ── 1. Hard budget limits ─────────────────────────────────────
        if state.turns >= cfg.max_turns:
            summary = self._build_summary(state, "Turn limit reached")
            logger.warning(
                "StopPolicy: max_turns=%d reached", cfg.max_turns
            )
            return StopDecision(
                outcome=StopOutcome.STOP_BUDGET,
                reason=f"Maximum turns ({cfg.max_turns}) reached",
                summary=summary,
            )

        if state.tool_calls >= cfg.max_tool_calls:
            summary = self._build_summary(state, "Tool call limit reached")
            logger.warning(
                "StopPolicy: max_tool_calls=%d reached", cfg.max_tool_calls
            )
            return StopDecision(
                outcome=StopOutcome.STOP_BUDGET,
                reason=f"Maximum tool calls ({cfg.max_tool_calls}) reached",
                summary=summary,
            )

        if state.tokens_consumed >= cfg.token_budget:
            summary = self._build_summary(state, "Token budget exhausted")
            logger.warning(
                "StopPolicy: token_budget=%d reached", cfg.token_budget
            )
            return StopDecision(
                outcome=StopOutcome.STOP_BUDGET,
                reason=f"Token budget ({cfg.token_budget:,}) exhausted",
                summary=summary,
            )

        if state.parallel_delegations >= cfg.max_parallel_delegations:
            logger.warning(
                "StopPolicy: max_parallel_delegations=%d reached",
                cfg.max_parallel_delegations,
            )
            return StopDecision(
                outcome=StopOutcome.STOP_BUDGET,
                reason=f"Max parallel delegations ({cfg.max_parallel_delegations}) reached",
                summary=self._build_summary(state, "Delegation limit"),
            )

        # ── 2. Low-progress stall ─────────────────────────────────────
        if state.no_progress_turns >= cfg.max_no_progress_turns:
            summary = self._build_summary(state, "No progress detected")
            logger.info(
                "StopPolicy: %d consecutive no-progress turns, stopping gracefully",
                state.no_progress_turns,
            )
            return StopDecision(
                outcome=StopOutcome.STOP_GRACEFUL,
                reason=(
                    f"No meaningful progress for {state.no_progress_turns} "
                    f"consecutive turns"
                ),
                summary=summary,
            )

        # ── 3. Persistent low confidence → HITL ──────────────────────
        if state.current_confidence < cfg.confidence_floor:
            state.low_confidence_turns_count += 1
        else:
            state.low_confidence_turns_count = 0

        if state.low_confidence_turns_count >= cfg.low_confidence_turns:
            summary = self._build_summary(state, "Insufficient confidence")
            logger.info(
                "StopPolicy: confidence=%.2f below floor=%.2f for %d turns → HITL",
                state.current_confidence, cfg.confidence_floor,
                state.low_confidence_turns_count,
            )
            return StopDecision(
                outcome=StopOutcome.STOP_HITL,
                reason=(
                    f"Confidence {state.current_confidence:.0%} remained below "
                    f"{cfg.confidence_floor:.0%} for {state.low_confidence_turns_count} turns"
                ),
                summary=summary,
            )

        return StopDecision(outcome=StopOutcome.CONTINUE, reason="")

    # ------------------------------------------------------------------
    # Summary builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary(state: LoopState, trigger: str) -> str:
        lines = [f"[Stop reason: {trigger}]"]
        if state.confirmed_facts:
            lines.append("Confirmed:")
            lines.extend(f"  • {f}" for f in state.confirmed_facts[-5:])
        if state.unresolved_points:
            lines.append("Still unresolved:")
            lines.extend(f"  ? {p}" for p in state.unresolved_points[-3:])
        if state.tool_summaries:
            lines.append("Tools used:")
            lines.extend(f"  – {s}" for s in state.tool_summaries[-5:])
        lines.append(
            f"(Turns: {state.turns}, Tool calls: {state.tool_calls})"
        )
        return "\n".join(lines)