"""
hitl_core.triggers — Pluggable HITL trigger evaluators.

A "trigger" is a small async callable that examines a candidate action
+ context and decides whether to escalate to a human. The TriggerEngine
runs registered triggers in priority order; the first match wins.

Design decisions:

  • Domain-neutral. Triggers receive a generic TriggerContext (free-form
    dict) plus the candidate ProposedAction. Domains layer their own
    context fields without changing this module.

  • Injection-first. The host registers triggers at startup. The
    built-in triggers (severity, low confidence, destructive verb,
    policy violation) are *optional* — opt-in by import + register.
    No domain assumption baked into the engine itself.

  • Composition. Multiple triggers can vote; first-match-wins is the
    default but the engine exposes evaluate_all() for hosts that want
    "raise the strictest trigger" semantics.

  • Cheap by default. Triggers are pure async functions; no heavy DI,
    no middleware. Built-ins are regex/threshold checks (sub-millisecond);
    LLM-based triggers are valid but should be rate-limited by the host.

Typical usage:

    engine = TriggerEngine()
    engine.register("severity",       SeverityTrigger(critical_threshold=0.8))
    engine.register("destructive",    DestructiveTrigger())
    engine.register("low_confidence", LowConfidenceTrigger(floor=0.45))

    decision = await engine.evaluate(action, context)
    if decision is not None:
        # decision is a TriggerOutcome; build a HitlPayload from it
        payload = build_payload_from_outcome(decision, action)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Protocol

from .schema import ProposedAction, RiskLevel, TriggerKind

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trigger context — passed to every trigger
# ---------------------------------------------------------------------------

@dataclass
class TriggerContext:
    """Inputs the engine offers triggers. Free-form metadata lives in
    `extras` so domains can pass whatever they need without modifying
    this dataclass."""
    user_query:        str = ""
    thread_id:         str = ""
    confidence_score:  float = 1.0
    classifier_output: dict[str, Any] = field(default_factory=dict)
    history_snapshot:  list[dict[str, Any]] = field(default_factory=list)
    # Free-form host extension point. IT-ops puts confirmed_facts here;
    # finance puts portfolio state; content-mod puts user history; etc.
    extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Trigger outcome — what a trigger returns when it matches
# ---------------------------------------------------------------------------

@dataclass
class TriggerOutcome:
    """A trigger emits this when it wants to escalate. The engine
    surfaces it to the producer who builds a HitlPayload from it."""
    kind:        TriggerKind
    risk_level:  RiskLevel
    reason:      str                           # human-readable why
    name:        str = ""                      # the registered trigger's name
    # Optional details merged into payload.context_snapshot by the
    # producer. Triggers use this to surface evidence (e.g. matched
    # regex, threshold breached, etc.) so the operator can audit.
    evidence:    dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Trigger protocol
# ---------------------------------------------------------------------------

class Trigger(Protocol):
    """Async callable: (action, context) -> TriggerOutcome | None.
    Return None means "this trigger doesn't fire on this input"."""
    async def __call__(
        self, action: ProposedAction, context: TriggerContext,
    ) -> Optional[TriggerOutcome]: ...


# ---------------------------------------------------------------------------
# Built-in triggers — optional, import + register if you want them
# ---------------------------------------------------------------------------

class LowConfidenceTrigger:
    """Fires when classifier confidence falls below `floor`.

    Useful as a generic "we're not sure, ask the human" signal. Hosts
    typically pair this with a clarification UI by mapping
    TriggerKind.LOW_CONFIDENCE to a clarification card.
    """
    def __init__(self, *, floor: float = 0.45,
                 risk_level: RiskLevel = RiskLevel.LOW):
        self.floor = floor
        self.risk_level = risk_level

    async def __call__(
        self, action: ProposedAction, ctx: TriggerContext,
    ) -> Optional[TriggerOutcome]:
        if ctx.confidence_score >= self.floor:
            return None
        return TriggerOutcome(
            kind=TriggerKind.LOW_CONFIDENCE,
            risk_level=self.risk_level,
            reason=(
                f"Confidence {ctx.confidence_score:.0%} below floor "
                f"{self.floor:.0%}"
            ),
            evidence={"confidence": ctx.confidence_score, "floor": self.floor},
        )


class DestructiveTrigger:
    """Fires when the candidate action looks destructive.

    Two signals:
      1. action.reversible == False  → strong destructive signal
      2. user_query contains a destructive verb (configurable)

    Defaults to common English + Chinese verbs that signal mutation.
    """
    DEFAULT_VERBS = (
        # English
        "delete", "destroy", "remove", "wipe", "drop", "purge",
        "restart", "reboot", "shutdown", "kill",
        "rollback", "revert", "fix", "repair",
        "push", "deploy", "release", "publish",
        "drain", "failover", "evict",
        # Chinese
        "修复", "修改", "删除", "重启", "重置", "推送",
        "下发", "回滚", "迁移", "调整",
    )

    def __init__(self, *, verbs: tuple[str, ...] = DEFAULT_VERBS,
                 reversible_is_safe: bool = True):
        self.verbs = verbs
        self.reversible_is_safe = reversible_is_safe
        # Build a single regex covering all verbs for cheap lookup.
        # Use word boundaries for ASCII; Chinese has no word boundary
        # so substring match is acceptable (no \w-letter problem since
        # destructive Chinese verbs are 2-3 chars and unambiguous).
        ascii_verbs = [v for v in verbs if v.isascii()]
        cjk_verbs   = [v for v in verbs if not v.isascii()]
        parts = []
        if ascii_verbs:
            parts.append(r"\b(?:" + "|".join(re.escape(v) for v in ascii_verbs) + r")\b")
        if cjk_verbs:
            parts.append("(?:" + "|".join(re.escape(v) for v in cjk_verbs) + ")")
        self._verb_re = re.compile("|".join(parts), re.IGNORECASE) if parts else None

    async def __call__(
        self, action: ProposedAction, ctx: TriggerContext,
    ) -> Optional[TriggerOutcome]:
        # Strong signal: producer marked it irreversible
        if not self.reversible_is_safe and not action.reversible:
            return TriggerOutcome(
                kind=TriggerKind.DESTRUCTIVE,
                risk_level=RiskLevel.HIGH,
                reason="Action is marked irreversible",
                evidence={"action_type": action.action_type, "target": action.target},
            )

        # Weak signal: query contains a destructive verb
        if self._verb_re and ctx.user_query:
            m = self._verb_re.search(ctx.user_query)
            if m:
                return TriggerOutcome(
                    kind=TriggerKind.DESTRUCTIVE,
                    risk_level=action.risk_level if action.risk_level != RiskLevel.LOW
                                else RiskLevel.MEDIUM,
                    reason=f"Destructive verb in query: {m.group(0)!r}",
                    evidence={"matched_verb": m.group(0),
                              "action_type":  action.action_type},
                )
        return None


class SeverityTrigger:
    """Fires when an alert severity exceeds threshold.

    Reads from `ctx.extras["max_severity"]` if present (a float on
    [0, 1] scale that the host has normalised). Falls back to
    `ctx.classifier_output["severity"]` for compat with classifier
    pipelines that emit severity directly.
    """
    def __init__(self, *, critical_threshold: float = 0.8):
        self.threshold = critical_threshold

    async def __call__(
        self, action: ProposedAction, ctx: TriggerContext,
    ) -> Optional[TriggerOutcome]:
        sev = ctx.extras.get("max_severity")
        if sev is None:
            sev = ctx.classifier_output.get("severity")
        if sev is None:
            return None
        try:
            sev = float(sev)
        except (TypeError, ValueError):
            return None
        if sev < self.threshold:
            return None
        return TriggerOutcome(
            kind=TriggerKind.SEVERITY,
            risk_level=RiskLevel.HIGH if sev >= 0.9 else RiskLevel.MEDIUM,
            reason=f"Severity {sev:.2f} ≥ threshold {self.threshold:.2f}",
            evidence={"severity": sev, "threshold": self.threshold},
        )


class PolicyViolationTrigger:
    """Fires when a host-supplied policy check returns False.

    The policy check is itself an injected callable, so different hosts
    can use static rule lists, LLM-based judgement, or external APIs.
    Triggers don't care which.
    """
    def __init__(
        self,
        check: Callable[[ProposedAction, TriggerContext], Awaitable[tuple[bool, str]]],
    ):
        # check returns (passes: bool, reason: str). When passes=False the
        # trigger fires with the reason as the human-readable explanation.
        self._check = check

    async def __call__(
        self, action: ProposedAction, ctx: TriggerContext,
    ) -> Optional[TriggerOutcome]:
        passes, reason = await self._check(action, ctx)
        if passes:
            return None
        return TriggerOutcome(
            kind=TriggerKind.POLICY_VIOLATION,
            risk_level=RiskLevel.HIGH,
            reason=reason,
            evidence={"action_type": action.action_type, "target": action.target},
        )


# ---------------------------------------------------------------------------
# TriggerEngine
# ---------------------------------------------------------------------------

class TriggerEngine:
    """Run a chain of triggers against a candidate action.

    Two evaluation modes:

      evaluate(action, ctx)         → first-match-wins (None if no match)
      evaluate_all(action, ctx)     → list of every match (for "strictest"
                                      policies or just diagnostics)

    Triggers run in registration order. Concurrency is not used by
    default — most triggers are sub-millisecond and starting tasks
    isn't worth the overhead. Hosts that have slow triggers (LLM calls,
    external APIs) can call evaluate_concurrent() for asyncio.gather
    semantics.
    """

    def __init__(self) -> None:
        self._triggers: list[tuple[str, Trigger]] = []

    def register(self, name: str, trigger: Trigger) -> None:
        """Add a trigger to the chain. Order matters for evaluate()."""
        self._triggers.append((name, trigger))

    def unregister(self, name: str) -> None:
        self._triggers = [(n, t) for (n, t) in self._triggers if n != name]

    @property
    def registered(self) -> list[str]:
        return [n for n, _ in self._triggers]

    async def evaluate(
        self,
        action: ProposedAction,
        ctx: TriggerContext,
    ) -> Optional[TriggerOutcome]:
        """First-match-wins evaluation. Returns the matching outcome,
        or None when no trigger fires."""
        for name, trigger in self._triggers:
            try:
                outcome = await trigger(action, ctx)
            except Exception as exc:
                # Bad triggers must not block the pipeline — log and
                # continue. Failed triggers behave like "did not fire".
                logger.warning("Trigger %s raised %s — skipping", name, exc)
                continue
            if outcome is not None:
                outcome.name = name
                return outcome
        return None

    async def evaluate_all(
        self,
        action: ProposedAction,
        ctx: TriggerContext,
    ) -> list[TriggerOutcome]:
        """Run every registered trigger; return all outcomes (in order).
        Useful for "strictest signal wins" policies — the host can pick
        the highest risk_level from the list."""
        results = []
        for name, trigger in self._triggers:
            try:
                outcome = await trigger(action, ctx)
            except Exception as exc:
                logger.warning("Trigger %s raised %s — skipping", name, exc)
                continue
            if outcome is not None:
                outcome.name = name
                results.append(outcome)
        return results

    async def evaluate_concurrent(
        self,
        action: ProposedAction,
        ctx: TriggerContext,
    ) -> list[TriggerOutcome]:
        """Like evaluate_all but runs triggers concurrently. Use when
        triggers are I/O-bound (LLM calls, external APIs) — order is
        no longer guaranteed in the result."""
        import asyncio

        async def _safe(name: str, t: Trigger) -> Optional[TriggerOutcome]:
            try:
                out = await t(action, ctx)
                if out is not None:
                    out.name = name
                return out
            except Exception as exc:
                logger.warning("Trigger %s raised %s — skipping", name, exc)
                return None

        outcomes = await asyncio.gather(
            *(_safe(n, t) for n, t in self._triggers),
            return_exceptions=False,
        )
        return [o for o in outcomes if o is not None]