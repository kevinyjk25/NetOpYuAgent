"""
runtime/context_budget_v2.py
----------------------------
Priority-based context budget — an alternative strategy alongside the
legacy length-based compress_paged_outputs in runtime/context_budget.py.

Framework principle:
  This is a NEW module, not a replacement. The legacy strategy is preserved
  intact at runtime/context_budget.py and remains the default.

  Switch via cfg.context_budget.strategy = "legacy" | "priority".

Why a new strategy:
  The legacy strategy is implicit: hit a length, truncate the tail. This
  works for tool output streams but doesn't help with the system prompt's
  global token budget (where multiple sections — facts, skills, retrieved
  memory, recent turns — compete for space).

  This module manages a token budget like a priority-based eviction policy:

      budget = TokenBudget(total=64000)
      budget.reserve("system_core",  4000,  priority=P0_FIXED)
      budget.reserve("user_profile", 500,   priority=P0_FIXED)
      budget.reserve("recent_turns", 20000, priority=P1_HIGH,    evictable=True)
      budget.reserve("tool_results", 30000, priority=P1_HIGH,    evictable=True)
      budget.reserve("retrieved_mem",10000, priority=P2_MEDIUM,  evictable=True)
      budget.reserve("skills",       5000,  priority=P2_MEDIUM,  evictable=True)
      budget.reserve("older_summary",5000,  priority=P3_LOW,     evictable=True)
      report = budget.commit()

  When the sum-of-reservations exceeds total, sections are TRIMMED in
  reverse priority order until everything fits. Each section gets a
  trim_callback that knows how to shrink itself (drop oldest entries,
  summarise, etc.) and reports how many chars/tokens it actually freed.

Independence guarantees:
  - Standalone module; no imports from runtime/loop.py or integrations/
  - Pure data plumbing — no LLM calls
  - All sizes counted in characters (not tokens) — token counting is the
    caller's concern. ~4 chars/token is a fine approximation for English;
    bilingual content runs ~2-3 chars/token, callers can adjust.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing      import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Priority enum (strings for YAML-friendliness)
# ---------------------------------------------------------------------------

P0_FIXED    = "P0"      # never trimmed (system prompt, identity, safety)
P1_HIGH     = "P1"      # last-resort (recent turns, in-flight tool results)
P2_MEDIUM   = "P2"      # trim before P1 (retrieved memory, skill summary)
P3_LOW      = "P3"      # trim first (older summaries, fallback hints)

_PRIORITY_ORDER = (P3_LOW, P2_MEDIUM, P1_HIGH, P0_FIXED)  # eviction order: P3 first


# ---------------------------------------------------------------------------
# Reservation
# ---------------------------------------------------------------------------

@dataclass
class _Reservation:
    name:           str
    requested:      int                       # chars asked for
    priority:       str
    evictable:      bool   = True
    payload:        Any    = None
    # Optional callback to trim payload. Receives (current_content, target_size)
    # and returns (new_content, new_size). If None, trimming = simple right-truncate.
    trim_callback:  Optional[Callable[[Any, int], tuple[Any, int]]] = None

    # Set after commit
    granted:        int    = 0
    final_payload:  Any    = None


@dataclass
class BudgetReport:
    """Returned by TokenBudget.commit() — caller uses this to build the final prompt."""
    total:        int                                # total budget in chars
    used:         int                                # final used chars
    sections:     dict[str, "BudgetSection"] = field(default_factory=dict)
    trim_log:     list[str]                  = field(default_factory=list)

    def get(self, section_name: str) -> Any:
        """Convenience: get the (possibly trimmed) payload for a section."""
        s = self.sections.get(section_name)
        return s.payload if s else None

    @property
    def utilisation(self) -> float:
        return self.used / self.total if self.total else 0.0


@dataclass
class BudgetSection:
    name:     str
    granted:  int
    payload:  Any
    priority: str


# ---------------------------------------------------------------------------
# TokenBudget
# ---------------------------------------------------------------------------

class TokenBudget:
    """Priority-aware budget that trims low-priority sections to fit a cap.

    Trimming policy:
      - First, sum all `requested` sizes.
      - If <= total, every section gets its full request.
      - If >, walk sections in PRIORITY_ORDER (P3 first) and trim each
        EVICTABLE section to its proportional share, until total fits.
      - Non-evictable sections always get their full request even if the
        budget is blown (logs a warning).

    Default trim:
      If a section has no trim_callback, payload is treated as str and
      right-truncated with a '…[truncated]' marker. Callers should provide
      callbacks for structured payloads (lists, dicts) that can be sliced
      more meaningfully (e.g. "keep newest N entries").
    """

    def __init__(self, total: int):
        self._total = max(0, int(total))
        self._reservations: list[_Reservation] = []

    # ── Reservation API ──────────────────────────────────────────────

    def reserve(
        self,
        name:           str,
        size:           int,
        *,
        priority:       str = P2_MEDIUM,
        evictable:      bool = True,
        payload:        Any = None,
        trim_callback:  Optional[Callable[[Any, int], tuple[Any, int]]] = None,
    ) -> None:
        """Declare a section's space request.

        `size` should be the natural / "ideal" size for the payload.
        It's the caller's responsibility to provide a `trim_callback` if
        the payload is anything other than a plain string.
        """
        if priority not in _PRIORITY_ORDER:
            raise ValueError(f"invalid priority {priority!r}; use {_PRIORITY_ORDER}")
        self._reservations.append(_Reservation(
            name=name, requested=max(0, int(size)), priority=priority,
            evictable=evictable, payload=payload, trim_callback=trim_callback,
        ))

    # ── Commit ───────────────────────────────────────────────────────

    def commit(self) -> BudgetReport:
        """Run trim policy and produce final per-section grants + payloads."""
        report = BudgetReport(total=self._total, used=0)

        # Stable order by reservation index for log readability
        original_order = list(enumerate(self._reservations))

        # Step 1: assume everyone gets their request
        sum_requested = sum(r.requested for r in self._reservations)
        if sum_requested <= self._total:
            # Easy path: enough room for everyone
            for r in self._reservations:
                r.granted = r.requested
                r.final_payload = r.payload
            for i, r in original_order:
                report.sections[r.name] = BudgetSection(
                    name=r.name, granted=r.granted,
                    payload=r.final_payload, priority=r.priority,
                )
                report.used += r.granted
            report.trim_log.append(
                f"no trimming needed: {sum_requested}/{self._total} chars"
            )
            return report

        # Step 2: over-budget. Start everyone at their request, then trim low → high
        over = sum_requested - self._total
        report.trim_log.append(
            f"over budget by {over} chars (requested {sum_requested}, total {self._total}); "
            f"trimming low-priority sections first"
        )

        # Trim within each priority level, lowest first
        for priority in _PRIORITY_ORDER:   # P3 → P2 → P1 → P0
            if over <= 0:
                break
            # All evictable reservations at this level
            evictables = [r for r in self._reservations
                          if r.priority == priority and r.evictable]
            if not evictables:
                continue
            total_in_level = sum(r.requested for r in evictables)
            if total_in_level == 0:
                continue

            # Take proportional bites from each evictable section in this level
            # until either over=0 or this level is exhausted
            for r in evictables:
                if over <= 0:
                    break
                # Max we can take from this section = its full requested size
                # (yes, P3 can go to 0).
                take = min(r.requested, over)
                r.requested -= take
                over -= take
                report.trim_log.append(
                    f"  trimmed {take} chars from {r.name!r} (priority={priority})"
                )

        if over > 0:
            # Non-evictable sections push us over budget — log but allow
            report.trim_log.append(
                f"  WARNING: still {over} chars over budget (non-evictable sections); "
                f"prompt may be too long"
            )

        # Step 3: apply each trim_callback (or default str truncate)
        for r in self._reservations:
            target = r.requested
            if r.payload is None:
                r.granted = 0
                r.final_payload = None
                continue

            if r.trim_callback is not None:
                try:
                    new_payload, new_size = r.trim_callback(r.payload, target)
                    r.final_payload = new_payload
                    r.granted = int(new_size)
                except Exception as exc:
                    logger.warning(
                        "TokenBudget: trim_callback for %r failed (%s); using untrimmed",
                        r.name, exc,
                    )
                    r.final_payload = r.payload
                    r.granted = len(str(r.payload))
            else:
                # Default: str truncation
                s = r.payload if isinstance(r.payload, str) else str(r.payload)
                if len(s) <= target:
                    r.final_payload = s
                    r.granted = len(s)
                elif target <= 0:
                    r.final_payload = ""
                    r.granted = 0
                else:
                    suffix = "…[truncated]"
                    keep = max(0, target - len(suffix))
                    r.final_payload = s[:keep] + suffix
                    r.granted = len(r.final_payload)

        # Step 4: build report
        for i, r in original_order:
            report.sections[r.name] = BudgetSection(
                name=r.name, granted=r.granted,
                payload=r.final_payload, priority=r.priority,
            )
            report.used += r.granted

        return report


# ---------------------------------------------------------------------------
# Sane built-in trim callbacks for common payload shapes
# ---------------------------------------------------------------------------

def trim_list_keep_newest(items: list, target_chars: int) -> tuple[list, int]:
    """Trim a list of stringifiable items, keeping the newest (tail).

    Useful for: turn history, recent tool calls, recent memory facts.
    """
    if not items:
        return [], 0
    out: list = []
    size = 0
    for item in reversed(items):
        s = str(item)
        if size + len(s) + 1 > target_chars:
            break
        out.insert(0, item)
        size += len(s) + 1
    return out, size


def trim_dict_keep_keys(target_keys: list[str]):
    """Factory for trim_callback that keeps only listed keys (in given order).

    Used when payload is a dict where only certain keys are essential.
    """
    def _trim(d: dict, target_chars: int) -> tuple[dict, int]:
        if not isinstance(d, dict):
            return d, len(str(d))
        out: dict = {}
        size = 0
        for k in target_keys:
            if k not in d:
                continue
            v_str = str(d[k])
            if size + len(v_str) + len(k) + 4 > target_chars:
                break
            out[k] = d[k]
            size += len(v_str) + len(k) + 4
        return out, size
    return _trim


def trim_str_head_tail(content: str, target_chars: int) -> tuple[str, int]:
    """Keep the first 40% and last 40% of a long string, summary in middle.
    Useful for tool outputs where head (metadata) and tail (latest events) matter.
    """
    if not content or target_chars <= 0:
        return "", 0
    if len(content) <= target_chars:
        return content, len(content)
    marker = "\n…[middle elided to fit budget]…\n"
    keep = max(0, target_chars - len(marker))
    head = int(keep * 0.5)
    tail = keep - head
    out = content[:head] + marker + content[-tail:]
    return out, len(out)
