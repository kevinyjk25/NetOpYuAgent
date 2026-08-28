"""Shared side-effect safety kernel for Network and Service L0 contracts.

``NetworkRuntime`` remains a compatibility name. New harness code imports
``EffectRuntime`` to make the domain-neutral ownership boundary explicit.
"""

from network_runtime.engine import NetworkRuntime
from .saga import SagaCoordinator, SagaDefinition, SagaState, SagaStepSpec


class EffectRuntime(NetworkRuntime):
    """Domain-neutral façade over the proven plan/approval/evidence engine."""


__all__ = [
    "EffectRuntime", "SagaCoordinator", "SagaDefinition", "SagaState", "SagaStepSpec",
]
