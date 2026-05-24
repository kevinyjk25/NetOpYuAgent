"""
runtime/loop_context.py
-----------------------
Internal per-turn state for AgentRuntimeLoop._stream_impl (Item 4 step 4a,
2026-05).

_stream_impl threads a dozen-plus local variables through a ~1400-line
`while True` loop. To decompose that loop into phase methods (refresh / assemble
/ clarify / tools), those phases need a shared mutable state object instead of
a forest of closure locals. _LoopContext IS that object: read-only inputs +
mutable cross-turn state in one place, passed as `(self, ctx)` to each phase.

This is INTERNAL (underscore-prefixed, not re-exported). Public loop types live
in loop_types.py. Module-independence: imports only sibling runtime modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

from .loop_types import DelegationMode

if TYPE_CHECKING:
    from .context_budget import DeviceRef
    from .stop_policy import LoopState


@dataclass
class _LoopContext:
    """Mutable per-turn state shared across _stream_impl phase methods.

    Field groups:
      - inputs   : set once at construction, treated read-only by phases
      - state    : the LoopState (turns, confirmed_facts, working_set, ...)
      - tool_*   : tool results + dedup guard, accumulated across turns
      - clarify  : one-shot clarification gate flags
      - cache_*  : memoised recall/skill computations (query is turn-invariant)
      - last_*   : refresh-cadence bookkeeping for the memoisation above
    """
    # ── read-only inputs ──────────────────────────────────────────────
    query:           str
    session_id:      str
    env_ctx:         dict[str, Any]
    tool_reg:        dict[str, Any]
    delegation_mode: DelegationMode
    parent_state:    Optional["LoopState"]

    # ── mutable cross-turn state ──────────────────────────────────────
    state:           "LoopState"
    tool_outputs:    dict[str, str] = field(default_factory=dict)   # persists across turns
    called_tools:    set[str]       = field(default_factory=set)    # dedup guard

    # ── clarification gate (one-shot) ─────────────────────────────────
    clarification_done: bool          = False
    initial_confidence: Optional[float] = None

    # ── memoised per-query computations (refreshed on a cadence) ──────
    cached_memory_results:   list = field(default_factory=list)
    cached_skill_section:    str  = ""
    cached_selected_skills:  list = field(default_factory=list)
    cached_skill_count:      int  = 0
    cached_skill_ambiguous:  bool = False
    last_recall_turn:        int  = -1
    last_skill_turn:         int  = -1
    last_facts_count:        int  = -1
    last_emitted_skill_sig:  str  = ""
