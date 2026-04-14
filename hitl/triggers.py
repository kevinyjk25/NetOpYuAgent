"""
hitl/triggers.py
----------------
Layer 1 — Interrupt triggers.

Each trigger is a stateless class implementing two methods:

    should_interrupt(state: HitlState) -> bool
    build_payload(state: HitlState, thread_id, context_id, task_id) -> HitlPayload

The graph evaluates all registered triggers before every ``executor`` node.
The first trigger that fires wins; multiple matches are logged but only the
highest-priority one produces the interrupt payload.

Trigger priority (highest → lowest):
    1. DestructiveActionTrigger  (always blocks — no override)
    2. SeverityTrigger
    3. ConfidenceTrigger
    4. AmbiguityTrigger

Configuration is read from the ``HitlConfig`` dataclass so thresholds can be
changed per deployment without code changes.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from .schemas import (
    HitlPayload,
    HitlState,
    ProposedAction,
    RiskLevel,
    TriggerKind,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HitlConfig:
    """
    Central configuration for all triggers and the timeout watchdog.
    Override defaults via environment variables or dependency injection.
    """
    # ConfidenceTrigger
    confidence_threshold: float = 0.75

    # SeverityTrigger
    high_severity_levels: tuple[str, ...] = ("P0", "P1", "critical")
    max_auto_host_count: int = 5           # block if action affects > N hosts

    # DestructiveActionTrigger
    destructive_action_types: tuple[str, ...] = (
        "restart_service",
        "rollback_deploy",
        "delete_resource",
        "drain_node",
        "force_failover",
        "flush_cache",
    )

    # AmbiguityTrigger
    max_confidence_gap: float = 0.15       # fire if top-2 differ by < this

    # SLA seconds per risk level
    sla_map: dict[str, int] = field(default_factory=lambda: {
        RiskLevel.CRITICAL: 300,   #  5 min
        RiskLevel.HIGH:     600,   # 10 min
        RiskLevel.MEDIUM:   900,   # 15 min
        RiskLevel.LOW:      1800,  # 30 min
    })

    # Dashboard deep-link template; {interrupt_id} is substituted
    review_url_template: str = "http://localhost:3000/hitl/review/{interrupt_id}"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class HitlTrigger(ABC):
    """Base class for all interrupt triggers."""

    priority: int = 99                     # lower = higher priority
    kind: TriggerKind

    def __init__(self, config: HitlConfig) -> None:
        self.config = config

    @abstractmethod
    def should_interrupt(self, state: HitlState) -> bool:
        """Return True if this trigger requires a human interrupt."""

    @abstractmethod
    def _derive_risk(self, state: HitlState) -> RiskLevel:
        """Derive the risk level to embed in the payload."""

    def build_payload(
        self,
        state: HitlState,
        thread_id: str,
        context_id: str,
        task_id: str,
    ) -> HitlPayload:
        """Build a fully populated HitlPayload from the current graph state."""
        risk = self._derive_risk(state)
        action = self._extract_action(state, risk)

        payload = HitlPayload(
            thread_id=thread_id,
            context_id=context_id,
            task_id=task_id,
            trigger_kind=self.kind,
            risk_level=risk,
            user_query=state.query,
            intent_summary=state.intent_summary or state.query,
            confidence_score=state.intent_confidence,
            proposed_action=action,
            context_snapshot={
                "intent_type": state.intent_type,
                "risk_reasons": state.risk_reasons,
                "plan_steps": state.plan_steps,
            },
            recent_alerts=state.user_metadata.get("recent_alerts", []),
            sla_seconds=self.config.sla_map.get(risk, 900),
        )
        logger.info(
            "Trigger %s fired | interrupt_id=%s risk=%s",
            self.kind.value,
            payload.interrupt_id,
            risk.value,
        )
        return payload

    # ------------------------------------------------------------------

    @staticmethod
    def _extract_action(state: HitlState, risk: RiskLevel) -> ProposedAction:
        """Convert state.proposed_action dict → ProposedAction model."""
        raw = state.proposed_action or {}
        return ProposedAction(
            action_type=raw.get("action_type", "unknown"),
            target=raw.get("target", "unknown"),
            parameters=raw.get("parameters", {}),
            estimated_impact=raw.get("estimated_impact"),
            reversible=raw.get("reversible", True),
            risk_level=risk,
        )


# ---------------------------------------------------------------------------
# Trigger 1: Destructive action  (highest priority — always blocks)
# ---------------------------------------------------------------------------

class DestructiveActionTrigger(HitlTrigger):
    """
    Fires unconditionally when the planner proposes an action whose type is
    on the destructive action allowlist (restart, rollback, delete, drain…).

    This trigger cannot be bypassed — it always results in a CRITICAL interrupt.
    """
    priority = 1
    kind = TriggerKind.DESTRUCTIVE

    def should_interrupt(self, state: HitlState) -> bool:
        action_type = (state.proposed_action or {}).get("action_type", "")
        hit = action_type in self.config.destructive_action_types
        if hit:
            logger.debug("DestructiveActionTrigger: action_type=%s", action_type)
        return hit

    def _derive_risk(self, state: HitlState) -> RiskLevel:
        # Destructive actions affecting many hosts are CRITICAL; others HIGH
        params = (state.proposed_action or {}).get("parameters", {})
        affected = params.get("host_count", 1)
        return RiskLevel.CRITICAL if affected > self.config.max_auto_host_count else RiskLevel.HIGH


# ---------------------------------------------------------------------------
# Trigger 2: High-severity alert
# ---------------------------------------------------------------------------

class SeverityTrigger(HitlTrigger):
    """
    Fires when the incoming alert payload carries a P0 / P1 / critical label,
    or when the proposed action's blast radius exceeds the safe host threshold.
    """
    priority = 2
    kind = TriggerKind.SEVERITY

    def should_interrupt(self, state: HitlState) -> bool:
        alerts = state.user_metadata.get("recent_alerts", [])
        for alert in alerts:
            if alert.get("severity", "").upper() in [
                s.upper() for s in self.config.high_severity_levels
            ]:
                return True

        # Also fire if intent_type is explicitly flagged critical
        if state.intent_type in ("critical_incident", "outage_response"):
            return True

        return False

    def _derive_risk(self, state: HitlState) -> RiskLevel:
        alerts = state.user_metadata.get("recent_alerts", [])
        for alert in alerts:
            if alert.get("severity", "").upper() == "P0":
                return RiskLevel.CRITICAL
        return RiskLevel.HIGH


# ---------------------------------------------------------------------------
# Trigger 3: Low confidence
# ---------------------------------------------------------------------------

class ConfidenceTrigger(HitlTrigger):
    """
    Fires when the intent classifier's top-class confidence score is below
    ``config.confidence_threshold``.  Signals the agent is uncertain about
    what the user wants — better to ask than act on a wrong interpretation.
    """
    priority = 3
    kind = TriggerKind.LOW_CONFIDENCE

    def should_interrupt(self, state: HitlState) -> bool:
        below = state.intent_confidence < self.config.confidence_threshold
        if below:
            logger.debug(
                "ConfidenceTrigger: score=%.3f threshold=%.3f",
                state.intent_confidence,
                self.config.confidence_threshold,
            )
        return below

    def _derive_risk(self, state: HitlState) -> RiskLevel:
        score = state.intent_confidence
        if score < 0.40:
            return RiskLevel.HIGH
        if score < 0.60:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW


# ---------------------------------------------------------------------------
# Trigger 4: Ambiguous intent
# ---------------------------------------------------------------------------

class AmbiguityTrigger(HitlTrigger):
    """
    Fires when the intent classifier returns two or more competing
    interpretations whose confidence scores are within ``max_confidence_gap``
    of each other — the agent cannot distinguish the user's true intent.
    """
    priority = 4
    kind = TriggerKind.AMBIGUOUS_INTENT

    def should_interrupt(self, state: HitlState) -> bool:
        candidates = state.intent_candidates
        if len(candidates) < 2:
            return False
        scores = sorted(
            [c.get("confidence", 0.0) for c in candidates], reverse=True
        )
        gap = scores[0] - scores[1]
        ambiguous = gap < self.config.max_confidence_gap
        if ambiguous:
            logger.debug(
                "AmbiguityTrigger: top=%.3f second=%.3f gap=%.3f",
                scores[0], scores[1], gap,
            )
        return ambiguous

    def _derive_risk(self, state: HitlState) -> RiskLevel:
        return RiskLevel.MEDIUM


# ---------------------------------------------------------------------------
# Trigger registry
# ---------------------------------------------------------------------------

def build_trigger_chain(config: Optional[HitlConfig] = None) -> list[HitlTrigger]:
    """
    Return all triggers sorted by priority (lowest number = evaluated first).
    Pass a custom ``HitlConfig`` to override thresholds.
    """
    cfg = config or HitlConfig()
    triggers: list[HitlTrigger] = [
        DestructiveActionTrigger(cfg),
        SeverityTrigger(cfg),
        ConfidenceTrigger(cfg),
        AmbiguityTrigger(cfg),
    ]
    return sorted(triggers, key=lambda t: t.priority)


def evaluate_triggers(
    state: HitlState,
    triggers: list[HitlTrigger],
    thread_id: str,
    context_id: str,
    task_id: str,
) -> Optional[HitlPayload]:
    """
    Evaluate triggers in priority order. Return the first matching payload,
    or None if no interrupt is needed.
    """
    for trigger in triggers:
        if trigger.should_interrupt(state):
            return trigger.build_payload(state, thread_id, context_id, task_id)
    return None
