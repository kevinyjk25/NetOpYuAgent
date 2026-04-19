"""
runtime/policy_engine.py
─────────────────────────
Prompt-based policy evaluation engine.

Replaces hard-coded keyword lists in classify(), pre_verify(), and HITL triggers
with natural-language policies defined in config.yaml under the ``policies:`` key.

Each policy is evaluated by sending a focused LLM prompt:
  [system: policy description + examples]
  [user: the query being evaluated]

The LLM returns JSON: {"match": true|false, "reason": "brief explanation"}

Operators tune policies in config.yaml — no code changes needed.
Results are cached per-turn to avoid duplicate LLM calls for the same query.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PolicyDefinition:
    """Loaded from config.yaml policies[] entries."""
    name:        str
    description: str
    prompt:      str
    confidence:  float = 0.85
    examples:    list  = field(default_factory=list)   # [{query, match}, ...]


@dataclass
class PolicyResult:
    match:         bool
    reason:        str
    confidence:    float
    policy:        str
    latency_ms:    float = 0.0
    from_cache:    bool  = False
    from_fallback: bool  = False   # True when LLM unavailable, used keyword heuristic


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class PolicyEngine:
    """
    Evaluate named policies against user queries using LLM.

    Parameters
    ----------
    policies    : list of PolicyDefinition loaded from config
    llm_call    : async fn(system: str, user: str) -> str
    cache_ttl_s : how long to cache results per query (default 60s)
    """

    # Keyword fallbacks — only used when LLM is unavailable
    _FALLBACK_DESTRUCTIVE = frozenset({
        "restart", "rollback", "delete", "drain", "failover", "flush",
        "reboot", "terminate", "shutdown", "wipe", "reset",
        "重启", "回滚", "删除", "终止", "关机", "重置", "下发配置", "推送配置",
    })
    _FALLBACK_INCIDENT = frozenset({
        "p0 incident", "p1 incident", "sev0", "sev1", "major incident",
        "critical outage", "production down",
    })

    def __init__(
        self,
        policies:    list[PolicyDefinition],
        llm_call:    Callable[[str, str], Awaitable[str]],
        cache_ttl_s: int = 60,
    ) -> None:
        self._policies   = {p.name: p for p in policies}
        self._llm_call   = llm_call
        self._cache:     dict[str, tuple[PolicyResult, float]] = {}
        self._cache_ttl  = cache_ttl_s

    # ── Public API ────────────────────────────────────────────────────────────

    async def evaluate(self, policy_name: str, query: str) -> PolicyResult:
        """
        Evaluate ``policy_name`` against ``query``.
        Returns a PolicyResult. Falls back to keyword heuristics if LLM fails.
        """
        cache_key = f"{policy_name}|{query}"
        cached = self._cache.get(cache_key)
        if cached:
            result, ts = cached
            if time.monotonic() - ts < self._cache_ttl:
                return PolicyResult(
                    match=result.match, reason=result.reason,
                    confidence=result.confidence, policy=policy_name,
                    from_cache=True,
                )

        policy = self._policies.get(policy_name)
        if policy is None:
            logger.warning("PolicyEngine: unknown policy %r — using fallback", policy_name)
            return self._fallback(policy_name, query)

        t0 = time.monotonic()
        try:
            result = await self._evaluate_with_llm(policy, query)
        except Exception as exc:
            logger.warning("PolicyEngine: LLM eval failed (%s) — using fallback", exc)
            result = self._fallback(policy_name, query)

        result.latency_ms = round((time.monotonic() - t0) * 1000, 1)
        self._cache[cache_key] = (result, time.monotonic())

        logger.info(
            "PolicyEngine: policy=%r match=%s conf=%.2f reason=%s latency=%.0fms%s",
            policy_name, result.match, result.confidence,
            result.reason[:60], result.latency_ms,
            " [fallback]" if result.from_fallback else "",
        )
        return result

    async def evaluate_any(
        self, policy_names: list[str], query: str
    ) -> dict[str, PolicyResult]:
        """Evaluate multiple policies concurrently. Returns {name: result}."""
        tasks   = [self.evaluate(name, query) for name in policy_names]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return {
            name: (self._fallback(name, query) if isinstance(r, Exception) else r)
            for name, r in zip(policy_names, results)
        }

    # ── LLM evaluation ────────────────────────────────────────────────────────

    async def _evaluate_with_llm(
        self, policy: PolicyDefinition, query: str
    ) -> PolicyResult:
        parts = [
            f"Policy: {policy.description}\n",
            policy.prompt.strip() + "\n",
        ]
        if policy.examples:
            parts.append("\nExamples:")
            for ex in policy.examples:
                m = "true" if ex.get("match") else "false"
                parts.append(f'  Query: "{ex["query"]}" → match: {m}')

        parts.append(
            '\n\nEvaluate the user query below. Return ONLY valid JSON:\n'
            '{"match": true or false, "reason": "one sentence explanation"}'
        )
        system = "\n".join(parts)
        raw    = await self._llm_call(system, query)
        return self._parse_result(raw, policy)

    @staticmethod
    def _parse_result(raw: str, policy: PolicyDefinition) -> PolicyResult:
        raw = raw.strip()
        raw = re.sub(r"^```json?\s*", "", raw)
        raw = re.sub(r"\s*```$",      "", raw)
        try:
            data = json.loads(raw)
            return PolicyResult(
                match      = bool(data.get("match", False)),
                reason     = str(data.get("reason", ""))[:200],
                confidence = policy.confidence,
                policy     = policy.name,
            )
        except Exception:
            lower = raw.lower()
            matched = "true" in lower[:20]
            return PolicyResult(matched, raw[:100], policy.confidence, policy.name)

    # ── Keyword fallback ──────────────────────────────────────────────────────

    def _fallback(self, policy_name: str, query: str) -> PolicyResult:
        """Keyword heuristic when LLM is unavailable — safety net only."""
        q = query.lower()

        def _wm(kw: str) -> bool:
            if " " in kw or not kw.isascii():
                return kw in q
            return bool(re.search(
                r"(?<![a-z0-9])" + re.escape(kw) + r"(?![a-z0-9])", q
            ))

        if policy_name in ("classify_destructive", "hitl_high_risk"):
            match = any(_wm(kw) for kw in self._FALLBACK_DESTRUCTIVE)
            return PolicyResult(
                match=match,
                reason=("keyword fallback: destructive keyword detected"
                        if match else "keyword fallback: no destructive keyword"),
                confidence=0.75 if match else 0.80,
                policy=policy_name, from_fallback=True,
            )
        if policy_name == "classify_incident_severity":
            match = any(kw in q for kw in self._FALLBACK_INCIDENT)
            return PolicyResult(
                match=match,
                reason="keyword fallback: incident keyword" if match else "no incident",
                confidence=0.70, policy=policy_name, from_fallback=True,
            )
        return PolicyResult(
            match=False, reason="unknown policy — safe default",
            confidence=0.5, policy=policy_name, from_fallback=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Factory helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_policies_from_config(cfg_policies: list[dict]) -> list[PolicyDefinition]:
    """Parse config.yaml policies[] list into PolicyDefinition objects."""
    out = []
    for p in (cfg_policies or []):
        try:
            out.append(PolicyDefinition(
                name        = p["name"],
                description = p.get("description", p["name"]),
                prompt      = p["prompt"],
                confidence  = float(p.get("confidence", 0.85)),
                examples    = p.get("examples", []),
            ))
        except (KeyError, TypeError) as e:
            logger.warning("PolicyEngine: skipping malformed policy entry: %s", e)
    return out


_GLOBAL_ENGINE: Optional[PolicyEngine] = None


def get_policy_engine() -> Optional[PolicyEngine]:
    """Return the globally registered PolicyEngine, or None if not yet set."""
    return _GLOBAL_ENGINE


def set_policy_engine(engine: PolicyEngine) -> None:
    """Register the PolicyEngine singleton — called during app startup."""
    global _GLOBAL_ENGINE
    _GLOBAL_ENGINE = engine
