"""skills/skill_preference.py — learn which skill a user picks for a kind of
request, then progressively recommend → auto-select it.

Closes the loop the design calls for:
  choice → preference fact → recall + weight future selection → stage
  (learn / recommend / auto) by confidence.

Built entirely on the existing memory adapter:
  - add_fact(fact_type="skill_preference", ...) — conflict detector auto-BOOSTS
    confidence when the same choice recurs (this IS the progressive mechanism).
  - find_similar_facts(..., fact_type="skill_preference") — embedding recall
    (B1a: no intent extraction, similarity over the stored query sample).
  - update_fact_confidence(...) — downgrade on a wrong auto-select (safety valve).

Per-user: every call passes user_id, so preferences are per operator (B5).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

FACT_TYPE = "skill_preference"


@dataclass
class PreferenceConfig:
    enabled: bool = True
    recommend_floor: float = 0.50      # >= → recommend (pin in HITL)
    auto_threshold: float = 0.85       # >= → auto-select, no HITL
    initial_confidence: float = 0.60   # written on first choice
    ttl_days: float = 90.0
    auto_exclude_hitl: bool = True     # high-risk skills never auto
    recall_top_k: int = 3
    base_boost: float = 0.20           # max score boost at conf=1, sim=1

    @classmethod
    def from_cfg(cls) -> "PreferenceConfig":
        try:
            from config import cfg as _c
            so = getattr(_c, "skill_orchestration", None)
            if so is None:
                return cls()
            return cls(
                enabled=bool(getattr(so, "preference_learning_enabled", True)),
                recommend_floor=float(getattr(so, "preference_recommend_floor", 0.50)),
                auto_threshold=float(getattr(so, "preference_auto_threshold", 0.85)),
                initial_confidence=float(getattr(so, "preference_initial_confidence", 0.60)),
                ttl_days=float(getattr(so, "preference_ttl_days", 90.0)),
                auto_exclude_hitl=bool(getattr(so, "preference_auto_exclude_hitl", True)),
            )
        except Exception:
            return cls()


@dataclass
class PreferenceHit:
    skill_id: str
    confidence: float
    similarity: float
    fact_id: str

    @property
    def boost(self) -> float:
        # Weight by both how sure we are (confidence) and how close the
        # current query is to the remembered one (similarity).
        return self.confidence * self.similarity


class SkillPreferenceService:
    """Records and recalls skill-choice preferences. Stateless beyond the
    injected memory adapter; safe to construct per request."""

    def __init__(self, memory_adapter, cfg: Optional[PreferenceConfig] = None):
        self._mem = memory_adapter
        self.cfg = cfg or PreferenceConfig.from_cfg()

    # ── B1: record a choice ──────────────────────────────────────────────
    async def record_choice(
        self, *, user_id: str, session_id: str, query: str,
        chosen_skill_id: Optional[str], candidates: list[str],
    ) -> Optional[str]:
        """Write (or boost, via conflict detector) a preference fact.
        chosen_skill_id=None means the operator chose "no skill" — recorded so
        we don't keep pestering for that query kind."""
        if not self.cfg.enabled or self._mem is None:
            return None
        label = chosen_skill_id or "__none__"
        fact_text = f"对于类似「{query[:120]}」的请求,用户选择使用 skill: {label}"
        try:
            return await self._mem.add_fact(
                session_id=session_id, user_id=user_id, fact_text=fact_text,
                fact_type=FACT_TYPE,
                confidence=self.cfg.initial_confidence,
                ttl_days=self.cfg.ttl_days,
                metadata={
                    "chosen_skill_id": chosen_skill_id,   # None for __none__
                    "query_sample": query[:200],
                    "candidates": candidates[:5],
                },
            )
        except Exception as exc:
            logger.warning("record_choice failed: %s", exc)
            return None

    # ── B2: recall preferences for a query ───────────────────────────────
    async def recall(self, *, user_id: str, query: str) -> list[PreferenceHit]:
        if not self.cfg.enabled or self._mem is None:
            return []
        try:
            rows = await self._mem.find_similar_facts(
                user_id=user_id, query_text=query,
                fact_type=FACT_TYPE, top_k=self.cfg.recall_top_k,
            )
        except Exception as exc:
            logger.warning("preference recall failed: %s", exc)
            return []
        hits: list[PreferenceHit] = []
        for r in rows:
            md = r.get("metadata") or {}
            sid = md.get("chosen_skill_id")
            if not sid:   # __none__ preference — no skill to boost
                continue
            hits.append(PreferenceHit(
                skill_id=sid,
                confidence=float(r.get("confidence", 0.0)),
                similarity=float(r.get("score", 0.0)),
                fact_id=r.get("fact_id", ""),
            ))
        return hits

    # ── B2: weight selection scores by preferences ───────────────────────
    def apply_boost(
        self, selected: list[tuple], hits: list[PreferenceHit],
    ) -> list[tuple]:
        """Return selected (skill_id, score) re-weighted + re-sorted by
        preference. Pure function — no I/O."""
        if not hits:
            return selected
        boost_by_skill: dict[str, float] = {}
        for h in hits:
            b = self.cfg.base_boost * h.boost
            boost_by_skill[h.skill_id] = max(boost_by_skill.get(h.skill_id, 0.0), b)
        out = [
            (sid, score + boost_by_skill.get(sid, 0.0))
            for sid, score in selected
        ]
        # Include preferred skills that weren't in the candidate list at all.
        present = {sid for sid, _ in selected}
        for sid, b in boost_by_skill.items():
            if sid not in present:
                out.append((sid, b))
        out.sort(key=lambda t: t[1], reverse=True)
        return out

    # ── B3: stage decision (learn / recommend / auto) ────────────────────
    def stage_for(
        self, hits: list[PreferenceHit], *, skill_requires_hitl=None,
    ) -> tuple[str, Optional[str]]:
        """Return (stage, preferred_skill_id).
        stage ∈ {"learn", "recommend", "auto"}.
        skill_requires_hitl: optional callable(skill_id)->bool for the
        auto-exclude-high-risk rule."""
        if not hits:
            return "learn", None
        top = max(hits, key=lambda h: h.confidence)
        sid = top.skill_id
        if top.confidence >= self.cfg.auto_threshold:
            if self.cfg.auto_exclude_hitl and skill_requires_hitl and skill_requires_hitl(sid):
                return "recommend", sid   # high-risk → never auto, cap at recommend
            return "auto", sid
        if top.confidence >= self.cfg.recommend_floor:
            return "recommend", sid
        return "learn", sid

    # ── Safety valve: wrong auto-select → downgrade ──────────────────────
    async def demote(self, fact_id: str, *, to: float = 0.4) -> bool:
        """Lower a preference's confidence (after a wrong auto-select), so it
        falls back from auto → recommend."""
        if not fact_id or self._mem is None:
            return False
        try:
            return await self._mem.update_fact_confidence(
                fact_id, to, reason="wrong auto-select feedback",
            )
        except Exception as exc:
            logger.warning("preference demote failed: %s", exc)
            return False
