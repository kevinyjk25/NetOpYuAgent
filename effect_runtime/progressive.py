"""Confidence-gated progressive determinization policy.

Hard evidence gates are evaluated before confidence. Confidence selects a
safe route; it never grants a natural-language Skill direct write access.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Mapping, Sequence


_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_L0_REFERENCE_RE = re.compile(r"^.+@[^@#]+#sha256:[0-9a-f]{64}$")


class RiskTier(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EffectSemantics(StrEnum):
    READ_ONLY = "read_only"
    REVERSIBLE = "reversible"
    DESTRUCTIVE = "destructive"
    IRREVERSIBLE = "irreversible"


class Route(StrEnum):
    L0_RUNTIME = "l0_runtime"
    HYBRID_L1_L0 = "hybrid_l1_l0"
    L1_READ_ONLY = "l1_read_only"
    CLARIFICATION_REQUIRED = "clarification_required"
    PROPOSAL_ONLY = "proposal_only"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class ModelConfidence:
    score: float
    calibrated: bool
    model_artifact_digest: str | None = None
    calibration_digest: str | None = None


@dataclass(frozen=True)
class ProgressivePolicy:
    read_only_threshold: float = 0.70
    low_threshold: float = 0.80
    medium_threshold: float = 0.85
    high_threshold: float = 0.92
    critical_threshold: float = 0.96
    hybrid_threshold: float = 0.70

    def threshold(self, risk: RiskTier, semantics: EffectSemantics) -> float:
        if semantics == EffectSemantics.READ_ONLY:
            return self.read_only_threshold
        return {
            RiskTier.LOW: self.low_threshold,
            RiskTier.MEDIUM: self.medium_threshold,
            RiskTier.HIGH: self.high_threshold,
            RiskTier.CRITICAL: self.critical_threshold,
        }[risk]


def _bounded(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if number > 1.0:
        number /= 100.0
    return min(1.0, max(0.0, number))


def _geometric(signals: Sequence[tuple[str, float, float]]) -> float:
    total_weight = sum(weight for _, _, weight in signals)
    if not signals or total_weight <= 0:
        return 0.0
    floor = 1e-9
    return math.exp(
        sum(weight * math.log(max(floor, score)) for _, score, weight in signals)
        / total_weight
    )


def decide_progressive_execution(
    *,
    assessment: Mapping[str, Any],
    package_report: Mapping[str, Any],
    risk: RiskTier | str,
    effect_semantics: EffectSemantics | str,
    l0_active: bool,
    l0_artifact_digest: str | None = None,
    referenced_l0: Sequence[str] = (),
    repeat_stability: float | None = None,
    simulation_pass_rate: float | None = None,
    activation_reviewed: bool = False,
    approval_control_available: bool = False,
    model_confidence: ModelConfidence | None = None,
    policy: ProgressivePolicy | None = None,
) -> dict[str, Any]:
    """Choose a fail-closed execution route from immutable evidence.

    ``l0_active`` means the exact L0 artifact has already completed review and
    activation. A high score can never activate an artifact by itself.
    """
    risk = RiskTier(risk)
    semantics = EffectSemantics(effect_semantics)
    policy = policy or ProgressivePolicy()
    findings: list[dict[str, str]] = []
    hard_failures: list[str] = []

    semantic = assessment.get("semanticCoverage") or {}
    promotion_findings = assessment.get("findings") or []
    if assessment.get("status") == "blocked" or semantic.get("gate") != "passed":
        hard_failures.append("PROMOTION_GATE_BLOCKED")
    if any(str(item.get("severity", "")).lower() == "error" for item in promotion_findings):
        hard_failures.append("PROMOTION_ERROR_PRESENT")
    if package_report.get("gate") != "passed":
        hard_failures.append("PACKAGE_GATE_BLOCKED")
    if any(not _L0_REFERENCE_RE.fullmatch(value) for value in referenced_l0):
        hard_failures.append("L0_REFERENCE_NOT_VERSION_DIGEST_BOUND")

    summary = semantic.get("summary") or {}
    ambiguous = int(summary.get("ambiguous", 0) or 0)
    missing = int(summary.get("missing", 0) or 0)
    blocking = int(summary.get("blockingRequirements", 0) or 0)
    if blocking:
        hard_failures.append("BLOCKING_SEMANTIC_REQUIREMENT")

    signals: list[tuple[str, float, float]] = []
    inputs = (
        ("semantic_mapping", summary.get("averageMappingConfidence"), 0.30),
        ("l1_to_l05", summary.get("averageL1ToL05Confidence"), 0.15),
        ("l05_to_l0", summary.get("averageL05ToL0Confidence"), 0.15),
        (
            "package_traceability",
            (package_report.get("summary") or {}).get("referenceCoveragePercent"),
            0.10,
        ),
        ("repeat_stability", repeat_stability, 0.15),
        ("simulation_pass_rate", simulation_pass_rate, 0.15),
    )
    for name, raw, weight in inputs:
        score = _bounded(raw)
        if score is not None:
            signals.append((name, score, weight))
        elif name in {"repeat_stability", "simulation_pass_rate"}:
            findings.append({
                "severity": "warning", "code": f"{name.upper()}_MISSING",
                "message": f"{name} is absent and therefore contributes no confidence.",
            })

    if model_confidence is not None:
        digest_bound = bool(
            model_confidence.model_artifact_digest
            and model_confidence.calibration_digest
        )
        if model_confidence.calibrated and digest_bound:
            score = _bounded(model_confidence.score)
            assert score is not None
            signals.append(("calibrated_model_judge", score, 0.10))
        else:
            findings.append({
                "severity": "warning", "code": "MODEL_SIGNAL_EXCLUDED",
                "message": "Uncalibrated or non-digest-bound model confidence cannot affect routing.",
            })

    confidence = _geometric(signals)
    threshold = policy.threshold(risk, semantics)
    write_evidence_complete = (
        semantics == EffectSemantics.READ_ONLY
        or (repeat_stability is not None and simulation_pass_rate is not None)
    )
    active_l0_bound = bool(
        l0_active and l0_artifact_digest and _DIGEST_RE.fullmatch(l0_artifact_digest)
    )
    if l0_active and not active_l0_bound:
        findings.append({
            "severity": "error", "code": "ACTIVE_L0_DIGEST_MISSING",
            "message": "An active L0 route requires the exact sha256 artifact digest.",
        })
    route = Route.BLOCKED
    reason = "One or more deterministic hard gates failed."

    if not hard_failures:
        if missing or ambiguous:
            route = Route.CLARIFICATION_REQUIRED
            reason = "The semantic map still contains missing or ambiguous requirements."
        elif semantics != EffectSemantics.READ_ONLY and not write_evidence_complete:
            route = Route.PROPOSAL_ONLY
            reason = "Write routing requires both repeat-stability and simulation evidence."
        elif l0_active and not active_l0_bound:
            route = Route.PROPOSAL_ONLY
            reason = "The active L0 artifact is not bound to an exact sha256 digest."
        elif active_l0_bound and confidence >= threshold:
            privileged = risk in {RiskTier.HIGH, RiskTier.CRITICAL} or semantics in {
                EffectSemantics.DESTRUCTIVE, EffectSemantics.IRREVERSIBLE,
            }
            if privileged and not (activation_reviewed and approval_control_available):
                route = Route.PROPOSAL_ONLY
                reason = "Privileged effects require reviewed activation and an approval control."
            else:
                route = Route.L0_RUNTIME
                reason = "All hard gates passed and the active L0 meets its risk-tier threshold."
        elif referenced_l0 and confidence >= policy.hybrid_threshold:
            route = Route.HYBRID_L1_L0
            reason = "L1 may orchestrate only the listed active L0 contracts; the candidate itself is not executable."
        elif semantics == EffectSemantics.READ_ONLY:
            route = Route.L1_READ_ONLY
            reason = "Confidence is insufficient for L0, but the operation is constrained to read-only interaction."
        else:
            route = Route.PROPOSAL_ONLY
            reason = "Confidence is insufficient or L0 is inactive; direct L1 writes remain forbidden."

    return {
        "schema": "effect-runtime.io/progressive-decision/v1",
        "route": route.value,
        "reason": reason,
        "risk": risk.value,
        "effectSemantics": semantics.value,
        "hardGate": {
            "status": "passed" if not hard_failures else "blocked",
            "failures": hard_failures,
        },
        "confidence": {
            "score": round(confidence, 4),
            "scorePercent": round(confidence * 100.0, 2),
            "threshold": threshold,
            "thresholdPercent": round(threshold * 100.0, 2),
            "method": "weighted_geometric_evidence_v1",
            "signals": [
                {"name": name, "score": round(score, 4), "weight": weight}
                for name, score, weight in signals
            ],
            "claimBoundary": "This routing score is not a production success probability.",
        },
        "controls": {
            "candidateL0Active": l0_active,
            "candidateL0ArtifactDigest": l0_artifact_digest,
            "candidateL0DigestBound": active_l0_bound,
            "activationReviewed": activation_reviewed,
            "approvalControlAvailable": approval_control_available,
            "referencedActiveL0": list(referenced_l0),
            "l1DirectWriteAllowed": False,
            "runtimeRequiredForWrite": True,
            "autoActivationAllowed": False,
            "autoExecutionAllowed": False,
        },
        "findings": findings,
        "policy": asdict(policy),
    }


__all__ = [
    "EffectSemantics", "ModelConfidence", "ProgressivePolicy", "RiskTier",
    "Route", "decide_progressive_execution",
]
