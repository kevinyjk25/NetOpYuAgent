"""
integrations/fact_conflict_detector.py
--------------------------------------
Cross-module adapter — detects + reconciles contradicting MemoryFacts.

Framework principle:
  - Memory module exposes `find_similar_facts` and `update_fact_confidence`
    via MemoryAdapter; it does NOT know about LLMs or reconciliation logic.
  - This module orchestrates the conflict-detection workflow externally.
  - Config-gated; disable cfg.cross_module.fact_conflict_detection.enabled to
    fall back to "blind insert" (deduped only by exact text hash).

Why this exists:
  Today MidTermStore dedups by EXACT text hash. Semantically equivalent
  facts ("Uses Cisco IOS 15.4" vs "Cisco IOS version 15.4") both get
  stored. Worse, semantically CONTRADICTING facts ("uses IOS 15.4" vs
  "uses IOS 16.2") also both get stored, then both surface during recall
  and confuse the LLM.

  This detector intercepts NEW facts before they're written, finds top-K
  similar existing facts, classifies the relationship (equivalent /
  refinement / contradiction / unrelated), and applies the right action:

      equivalent      → boost old fact's confidence + skip insert
      refinement      → boost old + insert new with link
      contradiction   → LLM reconcile → keep one, demote/expire other
      unrelated       → just insert

  Reconciliation classification can be:
    - Cheap (lexical heuristic): when similarity is very high and texts
      differ only in tense/wording → mark equivalent. 0 LLM calls.
    - LLM-driven: when similarity is high but content differs → ask LLM
      to classify and pick winner. Optional, opt-in via config.

Independence guarantees:
  - Reads + writes through MemoryAdapter only
  - No dependency on skill, hitl, runtime modules
  - Disabling removes the whole feature with zero code change
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public verdict enum + result type
# ---------------------------------------------------------------------------

VERDICT_EQUIVALENT    = "equivalent"
VERDICT_REFINEMENT    = "refinement"
VERDICT_CONTRADICTION = "contradiction"
VERDICT_UNRELATED     = "unrelated"


@dataclass
class ReconcileResult:
    """Outcome of one conflict-reconciliation pass for a single new fact."""
    action:           str                       # "inserted" | "skipped" | "demoted_other" | "demoted_new"
    inserted_fact_id: Optional[str]   = None
    verdict:          str             = VERDICT_UNRELATED
    related_fact_id:  Optional[str]   = None
    related_text:     str             = ""
    similarity:       float           = 0.0
    notes:            list[str]       = field(default_factory=list)


# ---------------------------------------------------------------------------
# Protocols — duck-typed integration points
# ---------------------------------------------------------------------------

class _MemoryProtocol(Protocol):
    async def add_fact(self, session_id: str, user_id: str, fact_text: str,
                       *, fact_type: str = "general", confidence: float = 1.0,
                       metadata: Optional[dict] = None,
                       ttl_days: Optional[float] = None) -> str: ...
    async def find_similar_facts(self, user_id: str, query_text: str,
                                 *, session_id: Optional[str] = None,
                                 fact_type: Optional[str] = None,
                                 top_k: int = 5) -> list[dict]: ...
    async def update_fact_confidence(self, fact_id: str, new_confidence: float,
                                     *, reason: str = "") -> bool: ...


# Async LLM signature: (system, user) → str
_LLMFn = Callable[[str, str], Awaitable[str]]


# ---------------------------------------------------------------------------
# Heuristic similarity helpers (no external deps)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)

def _jaccard(a: str, b: str) -> float:
    """Token-set Jaccard similarity. Used as a cheap first-pass filter."""
    ta = set(_TOKEN_RE.findall((a or "").lower()))
    tb = set(_TOKEN_RE.findall((b or "").lower()))
    if not ta or not tb:
        return 0.0
    inter = ta & tb
    union = ta | tb
    return len(inter) / len(union) if union else 0.0


def _structural_jaccard(a: str, b: str) -> float:
    """Jaccard after replacing numbers with a placeholder.

    "X uses Y 15.4" and "X uses Y 16.2" should have high structural similarity
    even though the literal jaccard is lower. This is the second signal we use
    to detect contradictions (high structure + different numbers).
    """
    a_norm = re.sub(r"\d+(?:\.\d+)?", "<N>", a or "")
    b_norm = re.sub(r"\d+(?:\.\d+)?", "<N>", b or "")
    return _jaccard(a_norm, b_norm)


def _likely_equivalent(text_a: str, text_b: str, *, threshold: float = 0.85) -> bool:
    """High-jaccard texts with same numeric content → equivalent."""
    if _jaccard(text_a, text_b) < threshold:
        return False
    # If both contain numbers, the numbers must match
    nums_a = re.findall(r"\d+(?:\.\d+)?", text_a)
    nums_b = re.findall(r"\d+(?:\.\d+)?", text_b)
    if nums_a and nums_b and set(nums_a) != set(nums_b):
        return False
    return True


def _likely_contradiction(text_a: str, text_b: str) -> bool:
    """Cheap heuristic: high-Jaccard texts with DIFFERENT numbers → contradiction."""
    if _jaccard(text_a, text_b) < 0.5:
        return False
    nums_a = re.findall(r"\d+(?:\.\d+)?", text_a)
    nums_b = re.findall(r"\d+(?:\.\d+)?", text_b)
    if nums_a and nums_b and set(nums_a) != set(nums_b):
        return True
    # Also flag opposite-polarity tokens
    neg_a = bool(re.search(r"\b(not|no|never|none|disabled|off)\b", text_a, re.I))
    neg_b = bool(re.search(r"\b(not|no|never|none|disabled|off)\b", text_b, re.I))
    if neg_a != neg_b and _jaccard(text_a, text_b) >= 0.7:
        return True
    return False


# ---------------------------------------------------------------------------
# LLM-driven classifier (optional, gated on cfg.llm_reconcile_enabled)
# ---------------------------------------------------------------------------

_RECONCILE_SYSTEM = """You compare two factual statements and classify their relationship.

Return ONLY a JSON object with this exact shape:
{"verdict": "<equivalent|refinement|contradiction|unrelated>", "winner": "<a|b|either>", "rationale": "<short reason>"}

Definitions:
- equivalent   : same meaning, different wording. Either can be discarded.
- refinement   : one adds detail or precision the other lacks; keep both, prefer the more specific.
- contradiction: statements cannot both be true.
- unrelated    : different topics; keep both independently.

When verdict is contradiction, `winner` is the more credible statement (a or b).
For other verdicts use "either" unless one is more specific."""


_LLM_USER_TPL = """Statement A: {a}
Statement B: {b}

Classify their relationship."""


async def _classify_with_llm(
    llm_fn:      _LLMFn,
    text_a:      str,
    text_b:      str,
    *,
    timeout_s:   float = 8.0,
) -> Optional[dict[str, Any]]:
    """Ask the LLM to classify the relationship. Returns parsed dict or None
    on failure / timeout (caller falls back to "unrelated" so memory is never
    corrupted by a flaky classifier)."""
    try:
        user_prompt = _LLM_USER_TPL.format(a=text_a, b=text_b)
        raw = await asyncio.wait_for(
            llm_fn(_RECONCILE_SYSTEM, user_prompt),
            timeout=timeout_s,
        )
    except (asyncio.TimeoutError, Exception) as exc:
        logger.warning("FactConflictDetector: LLM classify failed/timeout: %s", exc)
        return None

    if not raw:
        return None
    s = raw.strip()
    # Strip markdown fences
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    # Find first JSON object
    m = re.search(r"\{[^{}]*\}", s, re.DOTALL)
    if m:
        s = m.group(0)
    try:
        obj = json.loads(s)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    verdict = obj.get("verdict", "").lower().strip()
    if verdict not in (VERDICT_EQUIVALENT, VERDICT_REFINEMENT,
                       VERDICT_CONTRADICTION, VERDICT_UNRELATED):
        return None
    return {
        "verdict":  verdict,
        "winner":   obj.get("winner", "either").lower(),
        "rationale": obj.get("rationale", ""),
    }


# ---------------------------------------------------------------------------
# Main class — used as middleware before memory.add_fact()
# ---------------------------------------------------------------------------

class FactConflictDetector:
    """Wraps fact insertion with conflict detection + reconciliation.

    Usage (config-gated by caller):
        detector = FactConflictDetector(memory_adapter, llm_fn=...)
        result = await detector.insert_with_reconcile(
            session_id="...", user_id="...",
            fact_text="user uses IOS 15.4", fact_type="config",
        )

    Returns ReconcileResult so callers can audit / observe.
    """

    def __init__(
        self,
        memory:        _MemoryProtocol,
        *,
        llm_fn:                Optional[_LLMFn] = None,
        similarity_threshold:  float = 0.70,
        equivalence_threshold: float = 0.85,
        llm_reconcile_enabled: bool  = False,
        llm_timeout_s:         float = 8.0,
        top_k_candidates:      int   = 5,
        confidence_boost:      float = 0.05,
        contradiction_demote:  float = 0.4,
        scope_user_id_for_search: bool = True,
    ):
        self._memory = memory
        self._llm = llm_fn
        self._sim_thr = float(similarity_threshold)
        self._eq_thr  = float(equivalence_threshold)
        self._llm_enabled = bool(llm_reconcile_enabled) and llm_fn is not None
        self._llm_timeout = float(llm_timeout_s)
        self._top_k = max(1, int(top_k_candidates))
        self._boost = float(confidence_boost)
        self._demote = float(contradiction_demote)
        self._scope_user = bool(scope_user_id_for_search)

    # ── Public entrypoint ─────────────────────────────────────────────

    async def insert_with_reconcile(
        self,
        *,
        session_id: str,
        user_id:    str,
        fact_text:  str,
        fact_type:  str   = "general",
        confidence: float = 1.0,
        metadata:   Optional[dict] = None,
        ttl_days:   Optional[float] = None,
    ) -> ReconcileResult:
        """Insert `fact_text` after checking for conflicts with existing facts."""
        # 1. Find candidate similar facts
        search_user = user_id if self._scope_user else "_any"
        candidates = await self._memory.find_similar_facts(
            user_id=search_user, query_text=fact_text,
            fact_type=fact_type, top_k=self._top_k,
        )

        if not candidates:
            # Nothing similar — straight insert
            fid = await self._memory.add_fact(
                session_id=session_id, user_id=user_id, fact_text=fact_text,
                fact_type=fact_type, confidence=confidence,
                metadata=metadata, ttl_days=ttl_days,
            )
            return ReconcileResult(
                action="inserted", inserted_fact_id=fid,
                verdict=VERDICT_UNRELATED,
            )

        # 2. Find the best similarity match
        best = max(candidates, key=lambda c: max(c.get("score", 0.0),
                                                  _jaccard(c.get("fact", ""), fact_text)))
        best_text = best.get("fact", "")
        best_id   = best.get("fact_id", "")
        # Use the LARGEST of: search score, literal Jaccard, structural Jaccard.
        # Structural Jaccard normalises numbers, catching "X uses Y 15.4" vs
        # "X uses Y 16.2" which look similar after normalising → potential
        # contradiction, NOT unrelated.
        sim_literal    = _jaccard(best_text, fact_text)
        sim_structural = _structural_jaccard(best_text, fact_text)
        sim = max(float(best.get("score", 0.0)), sim_literal, sim_structural)

        if sim < self._sim_thr:
            # Not similar enough to warrant reconciliation
            fid = await self._memory.add_fact(
                session_id=session_id, user_id=user_id, fact_text=fact_text,
                fact_type=fact_type, confidence=confidence,
                metadata=metadata, ttl_days=ttl_days,
            )
            return ReconcileResult(
                action="inserted", inserted_fact_id=fid,
                verdict=VERDICT_UNRELATED, related_fact_id=best_id,
                related_text=best_text, similarity=sim,
            )

        # 3. Apply heuristic classification first
        verdict = None
        if _likely_equivalent(fact_text, best_text, threshold=self._eq_thr):
            verdict = VERDICT_EQUIVALENT
        elif _likely_contradiction(fact_text, best_text):
            verdict = VERDICT_CONTRADICTION

        # 4. If still unsure, optionally ask the LLM
        llm_result = None
        if verdict is None and self._llm_enabled:
            llm_result = await _classify_with_llm(
                self._llm, fact_text, best_text, timeout_s=self._llm_timeout,
            )
            if llm_result:
                verdict = llm_result["verdict"]

        # If LLM not available + heuristic unsure → treat as refinement (safest:
        # keep both, prefer new)
        if verdict is None:
            verdict = VERDICT_REFINEMENT

        # 5. Apply the verdict
        if verdict == VERDICT_EQUIVALENT:
            # Skip insert, boost existing
            await self._memory.update_fact_confidence(
                best_id,
                min(1.0, float(best.get("confidence", 0.5)) + self._boost),
                reason="equivalent_reinsertion",
            )
            return ReconcileResult(
                action="skipped", verdict=verdict,
                related_fact_id=best_id, related_text=best_text,
                similarity=sim,
                notes=["equivalent existing fact boosted, new fact dropped"],
            )

        elif verdict == VERDICT_CONTRADICTION:
            # Decide which wins
            winner = (llm_result or {}).get("winner", "a")  # 'a' = new fact
            if winner == "b":
                # Old wins; demote new (insert with low confidence + flag in metadata)
                lower_conf = max(0.0, confidence * (1 - self._demote))
                md = dict(metadata or {})
                md.update({
                    "contradicts":         best_id,
                    "loser_in_reconcile":  True,
                })
                fid = await self._memory.add_fact(
                    session_id=session_id, user_id=user_id, fact_text=fact_text,
                    fact_type=fact_type, confidence=lower_conf,
                    metadata=md, ttl_days=ttl_days,
                )
                return ReconcileResult(
                    action="demoted_new", inserted_fact_id=fid, verdict=verdict,
                    related_fact_id=best_id, related_text=best_text,
                    similarity=sim,
                    notes=["contradicts existing; new fact demoted"],
                )
            else:
                # New wins; demote old
                await self._memory.update_fact_confidence(
                    best_id,
                    max(0.0, float(best.get("confidence", 0.5)) * (1 - self._demote)),
                    reason="contradicted_by_new",
                )
                md = dict(metadata or {})
                md["supersedes"] = best_id
                fid = await self._memory.add_fact(
                    session_id=session_id, user_id=user_id, fact_text=fact_text,
                    fact_type=fact_type, confidence=confidence,
                    metadata=md, ttl_days=ttl_days,
                )
                return ReconcileResult(
                    action="demoted_other", inserted_fact_id=fid, verdict=verdict,
                    related_fact_id=best_id, related_text=best_text,
                    similarity=sim,
                    notes=["contradicts existing; old fact demoted"],
                )

        else:  # refinement / unrelated → just insert
            md = dict(metadata or {})
            if verdict == VERDICT_REFINEMENT:
                md["refines"] = best_id
            fid = await self._memory.add_fact(
                session_id=session_id, user_id=user_id, fact_text=fact_text,
                fact_type=fact_type, confidence=confidence,
                metadata=md, ttl_days=ttl_days,
            )
            return ReconcileResult(
                action="inserted", inserted_fact_id=fid, verdict=verdict,
                related_fact_id=best_id, related_text=best_text,
                similarity=sim,
            )
