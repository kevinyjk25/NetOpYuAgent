"""
runtime/loop_types.py
---------------------
Public type definitions for the agent runtime loop (Item 4 refactor, 2026-05).

These enums + dataclasses were defined inline in loop.py. They are pure data
declarations with no dependency on the loop's logic, so moving them here
separates "the shapes the loop speaks in" from "the loop's behaviour" and
shrinks loop.py. loop.py re-imports every name below, and runtime/__init__.py
continues to re-export them from runtime.loop, so all existing imports
(`from runtime.loop import RuntimeConfig`, etc.) are unchanged.

Module-independence: imports only sibling runtime modules (context_budget,
stop_policy) + stdlib. Never task/ or registry/.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .context_budget import BudgetConfig, ResourceRef
from .stop_policy import StopOutcome, StopPolicyConfig


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class QueryComplexity(str, Enum):
    SIMPLE  = "simple"
    COMPLEX = "complex"


class DelegationMode(str, Enum):
    """
    fresh  — start a sub-agent with only the explicitly passed context
    forked — inherit parent confirmed_facts + working_set (P1)
    """
    FRESH  = "fresh"
    FORKED = "forked"


class ForkContextPolicy(str, Enum):
    """How much parent context a forked sub-agent inherits."""
    FULL         = "full"           # everything
    FACTS_ONLY   = "facts_only"     # only confirmed_facts
    WORKING_SET  = "working_set"    # facts + working_set, not raw memory


# ---------------------------------------------------------------------------
# Verification result (P1)
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    passed:   bool
    reason:   str
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def ok(
        cls,
        reason: str = "Verification passed",
        warnings: Optional[list[str]] = None,
    ) -> "VerificationResult":
        """Construct a passing result.  warnings is accepted (and stored)
        so callers that detect non-fatal anomalies can surface them even
        on a successful verification — the BUG-09 post_verify() rewrite
        relies on this symmetric signature with fail()."""
        return cls(passed=True, reason=reason, warnings=warnings or [])

    @classmethod
    def fail(cls, reason: str, warnings: Optional[list[str]] = None) -> "VerificationResult":
        return cls(passed=False, reason=reason, warnings=warnings or [])


# ---------------------------------------------------------------------------
# Complexity decision
# ---------------------------------------------------------------------------

@dataclass
class ComplexityDecision:
    complexity: QueryComplexity
    reason:     str
    confidence: float = 1.0
    model_tier: str   = "full_model"   # P2: fast_model | full_model


# ---------------------------------------------------------------------------
# Runtime config
# ---------------------------------------------------------------------------

@dataclass
class RuntimeConfig:
    budget:      BudgetConfig      = field(default_factory=BudgetConfig)
    stop_policy: StopPolicyConfig  = field(default_factory=StopPolicyConfig)

    # Complexity thresholds
    simple_confidence_floor:  float = 0.70
    simple_max_tool_calls:    int   = 4

    # P1: delegation
    default_delegation_mode:   DelegationMode   = DelegationMode.FRESH
    default_fork_context:      ForkContextPolicy = ForkContextPolicy.FACTS_ONLY

    # Pre-verification REMOVED — replaced by tool-level HITL gate.
    # The flag is kept for one release for backward compatibility but
    # has no effect; the new pre_verify() stub returns ok unconditionally.
    enable_pre_verification:   bool = False   # DEPRECATED, no-op (kept for compat)
    enable_post_verification:  bool = True

    # Model tiering — flag retained for caller compatibility but unconsumed
    # in the active runtime path. Tier hint travels via ComplexityDecision.
    # Wire a real consumer in integrations/llm_engine.py if you want to act on it.
    enable_model_tiering:      bool = False   # DEPRECATED, unconsumed (kept for compat)

    # Tool result inline limit
    tool_result_inline_limit:  int  = 4_000

    # CAP 5: tools that force HITL before execution even on SIMPLE path
    # Populated from HITL_TOOL_NAMES env var in main.py
    hitl_tool_names: frozenset = field(default_factory=frozenset)

    # ── Type #2 EDIT-flavoured HITL ────────────────────────────────────
    # Tools where the operator should be allowed to edit the proposed
    # parameters before approving (e.g. edit the config_lines list).
    # Subset of hitl_tool_names — for any tool in BOTH sets, the executor
    # raises trigger_edit_approval (with editable_param_keys) instead of
    # the bare approve/reject panel.
    #
    # L0 default is EMPTY — this is a per-business mapping. The active profile
    # (L1) injects its own via cfg.tools.editable_hitl_tools, e.g. the network
    # profile maps edit_device_config → [config_lines, reason]. Keeping L0 free
    # of concrete tool names is part of the L0/L1 separation (Stage A, 2026-05).
    editable_hitl_tools: dict[str, list[str]] = field(default_factory=dict)

    # ── Type #3 CLARIFICATION ───────────────────────────────────────────
    # Auto-clarify when the agent's confidence in its plan is below this
    # threshold AND the operator hasn't exceeded their per-session budget.
    # 0 disables auto-clarify entirely.
    clarification_confidence_floor: float = 0.45
    clarification_max_per_session:  int   = 2


# ---------------------------------------------------------------------------
# Loop result
# ---------------------------------------------------------------------------

@dataclass
class LoopResult:
    outcome:          StopOutcome
    final_response:   str
    confirmed_facts:  list[str]   = field(default_factory=list)
    working_set:      list[ResourceRef] = field(default_factory=list)
    unresolved:       list[str]   = field(default_factory=list)
    tool_summaries:   list[str]   = field(default_factory=list)
    turns_taken:      int         = 0
    escalated_to_dag: bool        = False
    verification:     Optional[VerificationResult] = None
