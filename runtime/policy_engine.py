"""
runtime/policy_engine.py
─────────────────────────
Prompt-based policy evaluation engine.

Policy definitions live in config.yaml under ``policies:`` — operators tune them
without touching source code. Each policy is evaluated with a focused LLM call.

Design
------
- evaluate()      : async, uses real LLM, cached 120 s per (policy, query)
- evaluate_sync() : synchronous wrapper via asyncio; falls back to keyword
                    heuristics when called from non-async context or on error
- _fallback()     : keyword-heuristic safety net — used ONLY when LLM unavailable
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


@dataclass
class PolicyDefinition:
    name:        str
    description: str
    prompt:      str
    confidence:  float = 0.85
    examples:    list  = field(default_factory=list)
    # When True, the policy's own prompt already specifies the LLM output
    # schema (e.g. structured JSON beyond {match, reason}). The engine will
    # NOT inject the default `{"match": ..., "reason": ...}` template, and
    # will store the raw LLM response in `result.reason` for the caller
    # to parse. `result.match` defaults to True (informational policies
    # don't gate flow on a single bool — the caller decides based on the
    # parsed JSON content).
    raw_output:  bool = False


@dataclass
class PolicyResult:
    match:         bool
    reason:        str
    confidence:    float
    policy:        str
    latency_ms:    float = 0.0
    from_cache:    bool  = False
    from_fallback: bool  = False


class PolicyEngine:
    """
    Evaluate named policies against user queries using LLM.
    Falls back to keyword heuristics when LLM is unavailable.
    """

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
        cache_ttl_s: int = 120,
    ) -> None:
        self._policies   = {p.name: p for p in policies}
        self._llm_call   = llm_call
        self._cache:     dict[str, tuple[PolicyResult, float]] = {}
        self._cache_ttl  = cache_ttl_s

    # ── Async API (primary path) ──────────────────────────────────────────────

    async def evaluate(
        self,
        policy_name: str,
        query: str,
        context: str = "",
    ) -> PolicyResult:
        """Evaluate one policy against query using real LLM. Cached per TTL.

        `context` is optional supplementary information (e.g. recalled session
        history, confirmed facts) that the policy LLM can use to resolve
        references. Cache key includes a short hash of context so different
        contexts produce different cached results.
        """
        # Include context in the cache key — same query under different prior
        # session histories may legitimately get different policy decisions.
        if context:
            import hashlib as _hash
            _ck = _hash.md5(context.encode("utf-8", "replace")).hexdigest()[:12]
            cache_key = f"{policy_name}|{query}|ctx:{_ck}"
        else:
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
            logger.warning("PolicyEngine: unknown policy %r — fallback", policy_name)
            return self._fallback(policy_name, query)

        t0 = time.monotonic()
        try:
            result = await self._evaluate_with_llm(policy, query, context)
        except Exception as exc:
            logger.warning("PolicyEngine: LLM eval failed (%s) — fallback", exc)
            result = self._fallback(policy_name, query)

        result.latency_ms = round((time.monotonic() - t0) * 1000, 1)
        self._cache[cache_key] = (result, time.monotonic())
        logger.info(
            "PolicyEngine: policy=%r match=%s conf=%.2f latency=%.0fms reason=%s%s",
            policy_name, result.match, result.confidence, result.latency_ms,
            result.reason[:60], " [fallback]" if result.from_fallback else "",
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

    # ── Sync wrapper (for classify() called from FastAPI sync context) ────────

    def evaluate_sync(self, policy_name: str, query: str) -> PolicyResult:
        """
        Synchronous wrapper around evaluate().
        Uses the running event loop if available; falls back to keyword heuristic
        when blocking is not possible (i.e. inside an active async task).
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
        try:
            # 3.10+: get_event_loop() emits DeprecationWarning when called
            # outside a running loop. Use get_running_loop() to detect "are
            # we already inside a loop?" and asyncio.run() to bootstrap one
            # otherwise — that's the documented replacement pattern.
            try:
                _running = asyncio.get_running_loop()
            except RuntimeError:
                _running = None
            if _running is not None:
                # Inside async context — cannot block. Return cached fallback.
                # The async path (evaluate_any) should be used instead.
                return self._fallback(policy_name, query)
            return asyncio.run(self.evaluate(policy_name, query))
        except Exception:
            return self._fallback(policy_name, query)

    def evaluate_any_sync(
        self, policy_names: list[str], query: str
    ) -> dict[str, PolicyResult]:
        """Synchronous multi-policy evaluation."""
        try:
            try:
                _running = asyncio.get_running_loop()
            except RuntimeError:
                _running = None
            if _running is not None:
                return {n: self._fallback(n, query) for n in policy_names}
            return asyncio.run(self.evaluate_any(policy_names, query))
        except Exception:
            return {n: self._fallback(n, query) for n in policy_names}

    # ── LLM path ─────────────────────────────────────────────────────────────

    async def _evaluate_with_llm(
        self, policy: PolicyDefinition, query: str, context: str = ""
    ) -> PolicyResult:
        parts = [
            f"Policy: {policy.description}\n",
            policy.prompt.strip() + "\n",
        ]
        if policy.examples:
            parts.append("\nExamples:")
            for ex in policy.examples:
                m = "true" if ex.get("match") else "false"
                parts.append(f'  Query: "{ex["query"]}" -> match: {m}')
        # Inject recalled session context so policies can resolve pronouns
        # ("this device", "it") against prior turns instead of treating
        # every reference-bearing query as ambiguous.
        if context:
            parts.append(
                "\n\nPrior conversation context (use this to resolve references "
                "like 'this device', 'it', 'the AP' — entities mentioned earlier "
                "in the session count as known):\n"
                f"-----\n{context[:1500]}\n-----"
            )
        if policy.raw_output:
            # Policy defines its own JSON schema — caller will parse the
            # raw output. Just hand the LLM the user query.
            parts.append("\n\nEvaluate the user query below. Return ONLY the JSON specified above.")
        else:
            parts.append(
                '\n\nEvaluate the user query below.'
                '\nReturn ONLY valid JSON, nothing else:'
                '\n{"match": true or false, "reason": "one sentence"}'
            )
        system = "\n".join(parts)
        raw    = await self._llm_call(system, query)
        if policy.raw_output:
            # Stash raw LLM response as reason; match=True by default since
            # raw_output policies are informational, not gating booleans.
            cleaned = raw.strip()
            cleaned = re.sub(r"^```json?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$",      "", cleaned)
            return PolicyResult(
                match      = True,
                reason     = cleaned[:2000],
                confidence = policy.confidence,
                policy     = policy.name,
            )
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
            matched = "true" in raw.lower()[:20]
            return PolicyResult(matched, raw[:100], policy.confidence, policy.name)

    # ── Keyword fallback (safety net only) ───────────────────────────────────

    def _fallback(self, policy_name: str, query: str) -> PolicyResult:
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
                reason=("keyword: destructive" if match else "keyword: safe"),
                confidence=0.75 if match else 0.80,
                policy=policy_name, from_fallback=True,
            )
        if policy_name == "classify_incident_severity":
            match = any(kw in q for kw in self._FALLBACK_INCIDENT)
            return PolicyResult(
                match=match,
                reason="keyword: incident" if match else "keyword: no incident",
                confidence=0.70, policy=policy_name, from_fallback=True,
            )
        return PolicyResult(
            match=False, reason="unknown policy — safe default",
            confidence=0.5, policy=policy_name, from_fallback=True,
        )


# ── Factory ───────────────────────────────────────────────────────────────────

def load_policies_from_config(cfg_policies: list[dict]) -> list[PolicyDefinition]:
    out = []
    for p in (cfg_policies or []):
        try:
            out.append(PolicyDefinition(
                name        = p["name"],
                description = p.get("description", p["name"]),
                prompt      = p["prompt"],
                confidence  = float(p.get("confidence", 0.85)),
                examples    = p.get("examples", []),
                raw_output  = bool(p.get("raw_output", False)),
            ))
        except (KeyError, TypeError) as e:
            logger.warning("PolicyEngine: skipping malformed policy: %s", e)
    return out


_GLOBAL_ENGINE: Optional[PolicyEngine] = None


def get_policy_engine() -> Optional[PolicyEngine]:
    return _GLOBAL_ENGINE


def set_policy_engine(engine: PolicyEngine) -> None:
    global _GLOBAL_ENGINE
    _GLOBAL_ENGINE = engine