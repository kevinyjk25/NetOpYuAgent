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

    # Parallel delegation guard
    max_parallel_delegations: int = 5


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

    # Confidence tracking
    current_confidence:  float = 1.0
    low_confidence_turns_count: int = 0

    # Accumulated outputs
    confirmed_facts:     list[str] = field(default_factory=list)
    unresolved_points:   list[str] = field(default_factory=list)
    tool_summaries:      list[str] = field(default_factory=list)

    def record_tool_call(self, tool_name: str, summary: str = "") -> None:
        self.tool_calls += 1
        if summary:
            self.tool_summaries.append(f"{tool_name}: {summary}")

    def record_new_fact(self, fact: str) -> None:
        """Call when the agent confirms a new structured fact."""
        self.confirmed_facts.append(fact)
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