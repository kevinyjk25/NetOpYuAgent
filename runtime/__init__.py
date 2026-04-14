"""
runtime — Agent Runtime module
================================
Core exports (no fastapi/pydantic dependency at import time):
  ToolResultStore, ContextBudgetManager, BudgetConfig, DeviceRef
  AgentRuntimeLoop, RuntimeConfig, QueryComplexity, ComplexityDecision,
  DelegationMode, ForkContextPolicy, LoopResult, VerificationResult
  StopPolicy, StopPolicyConfig, StopDecision, StopOutcome, LoopState

Optional submodules (imported lazily inside their factory functions):
  tool_cache.py      — requires fastapi (create_cache_router lazy-imports it)
  skill_catalog.py   — standalone, no fastapi dep
  delegation.py      — standalone
  model_tier.py      — standalone
"""
from .context_budget import BudgetConfig, ContextBudgetManager, DeviceRef, ToolResultStore
from .loop import (
    AgentRuntimeLoop, ComplexityDecision, DelegationMode, ForkContextPolicy,
    LoopResult, QueryComplexity, RuntimeConfig, VerificationResult,
)
from .stop_policy import LoopState, StopDecision, StopOutcome, StopPolicy, StopPolicyConfig

__all__ = [
    # Context budget
    "ContextBudgetManager", "BudgetConfig", "ToolResultStore", "DeviceRef",
    # Loop
    "AgentRuntimeLoop", "RuntimeConfig", "QueryComplexity", "ComplexityDecision",
    "DelegationMode", "ForkContextPolicy", "LoopResult", "VerificationResult",
    # Stop policy
    "StopPolicy", "StopPolicyConfig", "StopDecision", "StopOutcome", "LoopState",
]