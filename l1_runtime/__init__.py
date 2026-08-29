"""Production L1 decision plane for harness-to-Runtime intent narrowing.

This package only creates bounded proposals.  It has no authority to invoke a
tool, approve a plan, or execute an effect; those remain L0 Runtime concerns.
"""

from .contracts import L1Decision, L1DecisionAction, L1DecisionEnvelope
from .canary_policy import CanaryPolicyResult, CanaryRoute, evaluate_canary_policy
from .service import (
    L1DecisionPlane,
    decide_shadow,
    decision_metrics,
    observe_decision,
    recent_decisions,
)

__all__ = [
    "L1Decision",
    "L1DecisionAction",
    "L1DecisionEnvelope",
    "L1DecisionPlane",
    "CanaryPolicyResult",
    "CanaryRoute",
    "decide_shadow",
    "decision_metrics",
    "evaluate_canary_policy",
    "observe_decision",
    "recent_decisions",
]
