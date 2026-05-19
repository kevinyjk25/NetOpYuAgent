"""
runtime — Agent Runtime module
================================
Core exports (no fastapi/pydantic dependency at import time):
  ToolResultStore, ContextBudgetManager, BudgetConfig, DeviceRef
  AgentRuntimeLoop, RuntimeConfig, QueryComplexity, ComplexityDecision,
  DelegationMode, ForkContextPolicy, LoopResult, VerificationResult
  StopPolicy, StopPolicyConfig, StopDecision, StopOutcome, LoopState
  HookEvent, HookRegistry, get_hook_registry        # Sprint 2 (2026-05)

Optional submodules (imported lazily inside their factory functions):
  skill_catalog.py   — standalone, no fastapi dep
  delegation.py      — standalone
  model_tier.py      — standalone
"""
from .context_budget import BudgetConfig, ContextBudgetManager, DeviceRef, ToolResultStore
from runtime.loop import (
    AgentRuntimeLoop, ComplexityDecision, DelegationMode, ForkContextPolicy,
    LoopResult, QueryComplexity, RuntimeConfig, VerificationResult,
)
from .stop_policy import LoopState, StopDecision, StopOutcome, StopPolicy, StopPolicyConfig
from runtime.hooks import HookEvent, HookRegistry, get_hook_registry

__all__ = [
    # Context budget
    "ContextBudgetManager", "BudgetConfig", "ToolResultStore", "DeviceRef",
    # Loop
    "AgentRuntimeLoop", "RuntimeConfig", "QueryComplexity", "ComplexityDecision",
    "DelegationMode", "ForkContextPolicy", "LoopResult", "VerificationResult",
    # Stop policy
    "StopPolicy", "StopPolicyConfig", "StopDecision", "StopOutcome", "LoopState",
    # Hooks (Sprint 2)
    "HookEvent", "HookRegistry", "get_hook_registry",
]