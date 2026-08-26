"""skills/append_merger.py — P3: append intent → targeted merge into the in-use skill.

Closes进化3: skill 过时 → 用户加载执行后追加诉求 → 生成 delta 并 merge 进存量 skill.

The evolver's existing merge fires when a NEW task happens to be jaccard-similar
to some stored skill — it doesn't know "the user just loaded & ran skill X and is
now appending to it". P3 supplies that missing context:

  active skill in this session (loaded + executed)  +  the append text
        → attribute the append to the right skill (CSI.nearest_skill, targeted)
        → merge the delta into THAT skill (not a fuzzy match, not a new skill)

Decoupling: depends only on injected CSI, a merge callback (evolver), and the
session's active-skill signal (passed in / read from journal). No reverse imports.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class MergeConfig:
    # Min CSI similarity for the append to be attributed to the active skill.
    attribution_floor: float = 0.45
    # If the active skill is known from session state, trust it over CSI when
    # CSI is uncertain (the session signal is ground truth).
    prefer_session_active: bool = True

    @classmethod
    def from_cfg(cls) -> "MergeConfig":
        try:
            from config import cfg as _c
            so = getattr(_c, "skill_orchestration", None)
            if so is None:
                return cls()
            return cls(
                attribution_floor=float(getattr(so, "append_attribution_floor", 0.45)),
            )
        except Exception:
            return cls()


@dataclass
class MergeResult:
    merged:      bool
    skill_id:    Optional[str]
    reason:      str
    similarity:  float = 0.0


class AppendMerger:
    """Attributes an append to the in-use skill and merges the delta into it.

    csi:        CapabilitySemanticIndex (nearest_skill / nearest_skill_async)
    merge_cb:   async callable(skill_id, append_text, session_id, tools) -> bool
                (wraps evolver._merge_into_existing_skill or apply_feedback)
    """

    def __init__(self, csi, merge_cb, cfg: Optional[MergeConfig] = None):
        self._csi = csi
        self._merge = merge_cb
        self.cfg = cfg or MergeConfig.from_cfg()

    def _attribute(self, *, active_skill, session_tools, append_text):
        """Decide which skill the append belongs to.
        Returns (skill_id, similarity, reason)."""
        # 1. Session signal is ground truth: a skill was loaded + executed.
        if self.cfg.prefer_session_active and active_skill:
            return active_skill, 1.0, "session active skill (loaded+executed)"
        # 2. Otherwise attribute via CSI nearest_skill (targeted, not fuzzy).
        try:
            hit = self._csi.nearest_skill(set(session_tools or []), append_text or "")
        except Exception as exc:
            return None, 0.0, f"CSI attribution error: {exc}"
        if not hit:
            return None, 0.0, "no skill matched the append"
        sid, sim = hit
        if sim.score < self.cfg.attribution_floor:
            return None, sim.score, (
                f"best match '{sid}' below attribution floor "
                f"({sim.score:.2f} < {self.cfg.attribution_floor})")
        return sid, sim.score, f"CSI attributed to '{sid}' ({', '.join(sim.reasons)})"

    async def maybe_merge(
        self, *, append_text: str, session_id: str,
        active_skill: Optional[str] = None,
        session_tools: Optional[list[str]] = None,
    ) -> MergeResult:
        """If the append can be attributed to an existing skill, merge the delta
        into it. Returns a MergeResult describing what happened."""
        if not append_text or not append_text.strip():
            return MergeResult(False, None, "empty append text")
        sid, sim, reason = self._attribute(
            active_skill=active_skill, session_tools=session_tools,
            append_text=append_text)
        if not sid:
            return MergeResult(False, None, reason, sim)
        try:
            ok = await self._merge(
                skill_id=sid, append_text=append_text,
                session_id=session_id, tools=list(session_tools or []))
        except Exception as exc:
            logger.warning("P3 merge failed for skill '%s': %s", sid, exc)
            return MergeResult(False, sid, f"merge error: {exc}", sim)
        if ok:
            logger.info(
                "P3: merged append delta into existing skill '%s' (%s)", sid, reason)
            return MergeResult(True, sid, reason, sim)
        return MergeResult(False, sid, "merge callback returned False", sim)
