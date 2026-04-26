"""
agent_memory/context_budget.py
==============================
ContextBudgetManager — inspired by NetOpYuAgent/runtime/context_budget.py

Treats the LLM prompt window as a finite resource.
Each memory section competes for tokens with an explicit priority ordering.
When the total exceeds the budget, lowest-priority sections are evicted first.

Priority tiers (eviction order: P6 → P1):
  P1  user_profile      — behavioral model of the user (never evicted)
  P2  confirmed_facts   — verified truths from tool calls (never evicted)
  P3  working_set       — current entities/devices being discussed
  P4  mid_term_facts    — distilled facts from past turns
  P5  long_term_chunks  — relevant historical chunks
  P6  environment       — system/env state (evicted first)

Token counting:
  Default: chars / CHARS_PER_TOKEN heuristic (4 for ASCII, 2 for CJK)
  Optional: plug in tiktoken or any fn(text: str) -> int

Why this matters vs crude % splits:
  - A % split runs even when a section is empty (wastes nothing, but lies)
  - A budget manager stops adding items the moment the cap is hit
  - Priorities mean confirmed_facts ALWAYS appear; environment may not
  - Token counting is honest: 1 CJK char ≠ 1 ASCII char ≠ 1 token
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Token counting ────────────────────────────────────────────────────────────

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]")

# How many characters (on average) map to one token for each script.
_CHARS_PER_TOKEN_ASCII = 4.0
_CHARS_PER_TOKEN_CJK   = 1.5   # CJK chars are ~2x denser than ASCII per token


def estimate_tokens(text: str) -> int:
    """
    Estimate token count without a tokenizer.
    Uses a mixed-script heuristic:
      ASCII/Latin:  ~4 chars / token
      CJK ideograms: ~1.5 chars / token
    Falls back to len(text) // 4 for empty strings.
    If tiktoken is installed and a tiktoken_fn is passed to ContextBudgetManager,
    that function is used instead.
    """
    if not text:
        return 0
    cjk_chars = len(_CJK_RE.findall(text))
    other_chars = len(text) - cjk_chars
    return max(1, int(cjk_chars / _CHARS_PER_TOKEN_CJK
                      + other_chars / _CHARS_PER_TOKEN_ASCII))


# ── Priority tier constants ───────────────────────────────────────────────────

class Priority:
    USER_PROFILE     = 1   # always included
    CONFIRMED_FACTS  = 2   # always included
    WORKING_SET      = 3
    MID_TERM_FACTS   = 4
    LONG_TERM_CHUNKS = 5
    ENVIRONMENT      = 6   # evicted first


# ── Section item ──────────────────────────────────────────────────────────────

@dataclass
class BudgetSection:
    """One named section that will appear in the context block."""
    name:     str
    content:  str
    priority: int          # lower number = higher priority = evicted last
    tokens:   int = 0      # filled by ContextBudgetManager

    def __post_init__(self) -> None:
        # tokens is computed externally; this avoids circular dependency
        pass


# ── ContextBudgetManager ──────────────────────────────────────────────────────

class ContextBudgetManager:
    """
    Assembles the LLM context block from memory sections under a token budget.

    Usage:
        mgr = ContextBudgetManager(token_budget=3200)
        mgr.add(Priority.USER_PROFILE,     "User profile text…")
        mgr.add(Priority.CONFIRMED_FACTS,  "Facts confirmed by tools…")
        mgr.add(Priority.WORKING_SET,      "Current working devices…")
        mgr.add(Priority.MID_TERM_FACTS,   "Distilled session facts…")
        mgr.add(Priority.LONG_TERM_CHUNKS, "Historical context…")
        context = mgr.build()

    Or use the convenience method on MemoryManager:
        ctx = mem.build_context_budgeted(session_state, query, budget=3200)
    """

    def __init__(
        self,
        token_budget: int = 3_200,
        tiktoken_fn: Optional[Callable[[str], int]] = None,
        separator: str = "\n\n",
    ) -> None:
        self._budget    = token_budget
        self._count_fn  = tiktoken_fn or estimate_tokens
        self._separator = separator
        self._sections: List[BudgetSection] = []

    def add(
        self,
        priority: int,
        content: str,
        name: str = "",
    ) -> "ContextBudgetManager":
        """Add a section. Sections with lower priority number survive eviction longer."""
        if not content or not content.strip():
            return self
        tokens = self._count_fn(content)
        self._sections.append(BudgetSection(
            name=name or f"p{priority}",
            content=content,
            priority=priority,
            tokens=tokens,
        ))
        return self

    def build(self) -> Tuple[str, "BudgetReport"]:
        """
        Assemble sections within the token budget.

        Returns:
            (context_string, BudgetReport)

        Algorithm:
          1. Sort sections by priority (lowest number first → highest priority).
          2. Greedily add sections; if adding one exceeds budget, try fitting
             individual items if the section is multi-line.
          3. Sections that don't fit are recorded in the report.
        """
        # Sort: P1 first (highest priority), P6 last
        ordered = sorted(self._sections, key=lambda s: s.priority)

        used_tokens  = 0
        kept:    List[BudgetSection] = []
        dropped: List[BudgetSection] = []

        for section in ordered:
            remaining = self._budget - used_tokens
            if section.tokens <= remaining:
                kept.append(section)
                used_tokens += section.tokens
            elif section.priority <= Priority.CONFIRMED_FACTS:
                # P1/P2 are never evicted — include even if over budget
                kept.append(section)
                used_tokens += section.tokens
                logger.warning(
                    "Budget exceeded by non-evictable section '%s' (%d tokens)",
                    section.name, section.tokens,
                )
            else:
                # Try to fit as many lines as possible
                partial = self._trim_to_budget(section.content, remaining)
                if partial:
                    trimmed = BudgetSection(
                        name=section.name + " [trimmed]",
                        content=partial,
                        priority=section.priority,
                        tokens=self._count_fn(partial),
                    )
                    kept.append(trimmed)
                    used_tokens += trimmed.tokens
                    logger.debug(
                        "Section '%s' trimmed: %d→%d tokens",
                        section.name, section.tokens, trimmed.tokens,
                    )
                else:
                    dropped.append(section)
                    logger.debug(
                        "Section '%s' dropped (%d tokens, %d remaining)",
                        section.name, section.tokens, remaining,
                    )

        context = self._separator.join(s.content for s in kept)
        report = BudgetReport(
            total_budget=self._budget,
            used_tokens=used_tokens,
            kept_sections=[(s.name, s.tokens) for s in kept],
            dropped_sections=[(s.name, s.tokens) for s in dropped],
        )
        return context, report

    def remaining_tokens(self) -> int:
        """How many tokens are still available (before build())."""
        used = sum(s.tokens for s in self._sections)
        return max(0, self._budget - used)

    def reset(self) -> None:
        self._sections.clear()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _trim_to_budget(self, content: str, token_limit: int) -> str:
        """Return as many complete lines as fit within token_limit."""
        if token_limit <= 0:
            return ""
        lines = content.splitlines()
        kept_lines: List[str] = []
        used = 0
        for line in lines:
            lt = self._count_fn(line)
            if used + lt > token_limit:
                break
            kept_lines.append(line)
            used += lt
        return "\n".join(kept_lines)


@dataclass
class BudgetReport:
    """Summary of how the token budget was spent."""
    total_budget:     int
    used_tokens:      int
    kept_sections:    List[Tuple[str, int]] = field(default_factory=list)
    dropped_sections: List[Tuple[str, int]] = field(default_factory=list)

    @property
    def utilization(self) -> float:
        return self.used_tokens / self.total_budget if self.total_budget else 0.0

    @property
    def over_budget(self) -> bool:
        return self.used_tokens > self.total_budget

    def summary(self) -> str:
        lines = [
            f"Budget: {self.used_tokens}/{self.total_budget} tokens "
            f"({self.utilization:.0%}){' ⚠ OVER' if self.over_budget else ''}",
        ]
        for name, toks in self.kept_sections:
            lines.append(f"  ✓ {name}: {toks} tokens")
        for name, toks in self.dropped_sections:
            lines.append(f"  ✗ {name}: {toks} tokens (dropped)")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_budget":     self.total_budget,
            "used_tokens":      self.used_tokens,
            "utilization":      round(self.utilization, 3),
            "over_budget":      self.over_budget,
            "kept_sections":    self.kept_sections,
            "dropped_sections": self.dropped_sections,
        }
