"""EnsuredSkill Reliability Runtime public API.

The reliability primitives are intentionally importable without loading the
network profile or a Harness. ``EffectRuntime`` is resolved lazily as the
current compatibility façade while the network engine is migrated onto the
domain-neutral kernel.
"""
from .progressive import (
    EffectSemantics,
    ModelConfidence,
    ProgressivePolicy,
    RiskTier,
    Route,
    decide_progressive_execution,
)
from .saga import SagaCoordinator, SagaDefinition, SagaState, SagaStepSpec
from .skill_graph import SkillEdge, SkillLevel, SkillNode, validate_skill_graph
from .skill_package import build_skill_disclosure_packet, inspect_skill_package
from .graph_scheduler import (
    GraphScheduleError,
    NodeOutcome,
    NodeResult,
    TypedGraphScheduler,
    graph_from_step_contract,
)
from .reliability import (
    AutonomyDecision,
    EvidenceRecord,
    EvidenceRequirement,
    ExecutionPhase,
    GuardResult,
    ReliabilityContract,
    Reversibility,
    RiskFactors,
    RiskPolicy,
    TransactionStateMachine,
    TypedExecutionGraph,
    build_transaction_graph,
    contract_from_compiled_l0,
    evaluate_evidence,
    evaluate_guards,
)

def __getattr__(name: str):
    if name != "EffectRuntime":
        raise AttributeError(name)
    from network_runtime.engine import NetworkRuntime

    class EffectRuntime(NetworkRuntime):
        """Compatibility façade over the network-first reference engine."""

    EffectRuntime.__name__ = "EffectRuntime"
    globals()[name] = EffectRuntime
    return EffectRuntime


__all__ = [
    "EffectRuntime", "EffectSemantics", "ModelConfidence", "ProgressivePolicy",
    "RiskTier", "Route", "SagaCoordinator", "SagaDefinition", "SagaState",
    "SagaStepSpec", "SkillEdge", "SkillLevel", "SkillNode",
    "build_skill_disclosure_packet", "decide_progressive_execution",
    "inspect_skill_package", "GraphScheduleError", "NodeOutcome", "NodeResult",
    "TypedGraphScheduler", "graph_from_step_contract",
    "validate_skill_graph",
    "AutonomyDecision", "EvidenceRecord", "EvidenceRequirement",
    "ExecutionPhase", "GuardResult", "ReliabilityContract", "Reversibility", "RiskFactors",
    "RiskPolicy", "TransactionStateMachine", "TypedExecutionGraph",
    "build_transaction_graph", "contract_from_compiled_l0", "evaluate_evidence",
    "evaluate_guards",
]
