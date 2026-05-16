"""
skills/catalog.py
------------------
SkillCatalogService — progressive disclosure of tool/skill capabilities.

Problem it solves (from the PDF review, Section VI)
-----------------------------------------------------
The current Registry exposes AgentCard skill metadata for *service discovery*,
but all descriptions are injected upfront into the prompt, wasting tokens
on skills the model will never use in this turn.

Solution: Two-level disclosure
  Level 1  (always in prompt)  — skill_id, one-line purpose, risk_level
  Level 2  (loaded on demand)  — full description, parameter schema,
                                  return format, examples, constraints

The model sees only Level 1 at the start of each turn.  When it decides to
call a skill it emits [SKILL_LOAD:skill_id], and the runtime fetches Level 2
and appends it to the context before the actual tool call.

This reduces per-turn token overhead by ~60% in a system with 20+ skills.

Usage
-----
    catalog = SkillCatalogService()
    catalog.register_all(SKILL_DEFINITIONS)

    # In prompt assembly:
    summary = catalog.format_summary()     # Level 1 — always inject

    # When model requests a skill:
    detail  = catalog.load_detail("syslog_search")   # Level 2 — on demand

    # Check if a skill is safe to auto-execute:
    if catalog.requires_hitl("restart_service"):
        ...
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SkillSummary:
    """Level 1: injected into every prompt."""
    skill_id:    str
    name:        str
    purpose:     str           # one sentence
    risk_level:  str           # low | medium | high | critical
    requires_hitl: bool = False
    tags:        list[str] = field(default_factory=list)


@dataclass
class SkillDetail:
    """Level 2: loaded only when the model decides to call this skill."""
    skill_id:       str
    description:    str                   # full paragraph
    parameters:     dict[str, str]        # param_name → description
    returns:        str                   # description of return format
    examples:       list[dict[str, Any]]  # list of {args, expected_output}
    constraints:    list[str]             # preconditions / safety rules
    estimated_size: str = "small"         # small | medium | large
    returns_large:  bool = False


@dataclass
class Skill:
    summary: SkillSummary
    detail:  SkillDetail


# ---------------------------------------------------------------------------
# SkillCatalogService
# ---------------------------------------------------------------------------

class SkillCatalogService:
    """
    Manages the two-level skill catalog.

    register()         — register a single Skill
    register_all()     — bulk register from a dict
    format_summary()   — build Level 1 prompt string
    load_detail()      — return Level 2 detail for one skill
    requires_hitl()    — check if a skill needs human approval
    """

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}
        # Optional Retriever for upgraded scoring (BM25 + embedding fusion).
        # When None, falls back to the legacy keyword overlap path.
        self._retriever = None

    def attach_retriever(self, retriever) -> None:
        """Attach a retrieval/.Retriever instance for upgraded scoring.

        The retriever must be already indexed with this catalog's skills
        (use retrieval.build_skill_retriever / skills_to_corpus).
        Calling attach_retriever(None) reverts to legacy keyword scoring.
        """
        self._retriever = retriever

    def register(self, skill: Skill) -> None:
        self._skills[skill.summary.skill_id] = skill
        logger.debug("SkillCatalog: registered %s", skill.summary.skill_id)

    def register_all(self, definitions: dict[str, dict[str, Any]]) -> None:
        for skill_id, defn in definitions.items():
            summary = SkillSummary(
                skill_id=skill_id,
                name=defn["name"],
                purpose=defn["purpose"],
                risk_level=defn.get("risk_level", "low"),
                requires_hitl=defn.get("requires_hitl", False),
                tags=defn.get("tags", []),
            )
            detail = SkillDetail(
                skill_id=skill_id,
                description=defn.get("description", defn["purpose"]),
                parameters=defn.get("parameters", {}),
                returns=defn.get("returns", "string"),
                examples=defn.get("examples", []),
                constraints=defn.get("constraints", []),
                estimated_size=defn.get("estimated_size", "small"),
                returns_large=defn.get("returns_large", False),
            )
            self.register(Skill(summary=summary, detail=detail))

    def filter_to_registry(self, tool_registry: dict, *, strict: bool = False) -> int:
        """
        Remove skills whose skill_id is not present in tool_registry.

        A skill named 'syslog_search' is only useful if tool_registry contains
        a callable named 'syslog_search'. Without it the LLM loads the skill,
        reads steps that call [TOOL:syslog_search], and gets a silent error.

        Args:
            tool_registry: dict of available tool names → callables
            strict: if True, also remove skills whose description mentions
                    [TOOL:xxx] where xxx is not in the registry

        Returns: number of skills removed
        """
        import re as _re
        to_remove = []
        for skill_id, skill in list(self._skills.items()):
            # Primary check: skill_id matches a tool name
            if skill_id not in tool_registry:
                to_remove.append(skill_id)
                continue
            if strict:
                # Secondary check: any [TOOL:xxx] in skill detail not in registry
                # Use centralized parser for whitespace tolerance
                from runtime.directive_parser import find_tool_names as _ftn
                tool_refs = _ftn(skill.detail.description or "")
                for ref in tool_refs:
                    if ref not in tool_registry:
                        to_remove.append(skill_id)
                        break
        for skill_id in to_remove:
            del self._skills[skill_id]
            logger.info("SkillCatalog: removed skill %r (tool not in registry)", skill_id)
        if to_remove:
            logger.info(
                "SkillCatalog: filtered %d skill(s) — %d remain",
                len(to_remove), len(self._skills),
            )
        return len(to_remove)

    def format_summary(self) -> str:
        """
        Build the Level 1 prompt section.
        Injected at the top of every turn — compact by design.
        """
        if not self._skills:
            return ""
        lines = ["[AVAILABLE SKILLS — call [SKILL_LOAD:skill_id] for full details]"]
        for s in self._skills.values():
            hitl_tag = " ⚠ HITL" if s.summary.requires_hitl else ""
            lines.append(
                f"  {s.summary.skill_id:<25} [{s.summary.risk_level:>8}]{hitl_tag}"
                f"  {s.summary.purpose}"
            )
        return "\n".join(lines)

    def load_detail(self, skill_id: str) -> Optional[str]:
        """
        Return the Level 2 detail block for injection into the prompt.
        Returns None if the skill_id is not registered.
        """
        skill = self._skills.get(skill_id)
        if skill is None:
            logger.warning("SkillCatalog: unknown skill_id=%s", skill_id)
            return None

        d = skill.detail
        lines = [
            f"[SKILL DETAIL: {skill_id}]",
            f"Description: {d.description}",
            f"Returns: {d.returns}  (size: {d.estimated_size})",
        ]
        if d.returns_large:
            lines.append(
                "NOTE: This skill returns large output. "
                "Use read_stored_result(ref_id) to page through the result."
            )
        if d.parameters:
            lines.append("Parameters:")
            for p, desc in d.parameters.items():
                lines.append(f"  {p}: {desc}")
        if d.constraints:
            lines.append("Constraints:")
            for c in d.constraints:
                lines.append(f"  • {c}")
        if d.examples:
            ex = d.examples[0]
            lines.append(f"Example: args={ex.get('args', {})}  →  {ex.get('note', '')}")
        return "\n".join(lines)

    def requires_hitl(self, skill_id: str) -> bool:
        skill = self._skills.get(skill_id)
        return skill.summary.requires_hitl if skill else False

    def get_summary(self, skill_id: str) -> Optional[SkillSummary]:
        skill = self._skills.get(skill_id)
        return skill.summary if skill else None

    def list_skills(self) -> list[SkillSummary]:
        return [s.summary for s in self._skills.values()]

    @property
    def skill_count(self) -> int:
        return len(self._skills)

    def as_markdown(self, skill_id: str) -> Optional[str]:
        """
        Return the full markdown content of a skill in IT-ops skill format.
        Works for both built-in skills (synthesised from catalog detail)
        and evolved/uploaded skills whose .md source is stored in detail.description.
        """
        skill = self._skills.get(skill_id)
        if skill is None:
            return None
        s, d = skill.summary, skill.detail

        # If the description already looks like markdown (starts with #), return it directly
        if d.description.strip().startswith("#"):
            return d.description

        # Synthesise clean markdown from the structured detail
        hitl_str = "yes" if s.requires_hitl else "no"
        lines = [
            f"# {s.name}",
            f"**Purpose:** {s.purpose}",
            f"**Tags:** [{', '.join(s.tags)}]",
            f"**Risk:** {s.risk_level}",
            f"**HITL:** {hitl_str}",
            "",
        ]
        if d.description and d.description != s.purpose:
            lines += ["## Description", d.description, ""]
        if d.parameters:
            lines.append("## Parameters")
            for pname, pdesc in d.parameters.items():
                lines.append(f"- `{pname}`: {pdesc}")
            lines.append("")
        if d.examples:
            lines.append("## Examples")
            for ex in d.examples:
                lines.append(f"    {ex}")
            lines.append("")
        if d.constraints:
            lines.append("## Constraints")
            for c in d.constraints:
                lines.append(f"- {c}")
            lines.append("")
        return "\n".join(lines)

    def _legacy_score(
        self,
        query_words: set,
        query:       str,
    ) -> list[tuple[float, str]]:
        r"""Legacy multi-field weighted keyword scorer.

        Used when no retriever is attached (or retriever errors). Compared
        to the original single-concat keyword scorer, this version:
          - Scores each skill field separately, then takes a weighted sum,
            so a strong match in `purpose` isn't diluted by unrelated text
            in `description`.
          - Tag matches use whole-word boundary on the query too, so
            short tags like "ap" don't accidentally match "happen".
          - For CJK queries (where \b\w{3,}\b returns nothing), falls
            back to per-character substring containment so the scorer
            doesn't silently return all zeros.
        """
        import re as _re

        # Detect "did the regex actually tokenise anything?"
        # (CJK chars are word-y but the {3,} length filter cuts them all.)
        weak_token_signal = (len(query_words) == 0 and len(query) > 0)

        # Per-field weights, loaded from cfg.skill_orchestration.scoring.
        # The defaults match the original hardcoded values; tune via YAML or env
        # to bias scoring toward particular skill metadata fields.
        try:
            from config import cfg as _app_cfg
            _sc_cfg = getattr(getattr(_app_cfg, "skill_orchestration", None), "scoring", None)
            W_PURPOSE     = float(getattr(_sc_cfg, "purpose_weight",     0.40)) if _sc_cfg else 0.40
            W_DESCRIPTION = float(getattr(_sc_cfg, "description_weight", 0.20)) if _sc_cfg else 0.20
            W_TAGS        = float(getattr(_sc_cfg, "tags_weight",        0.20)) if _sc_cfg else 0.20
            W_PARAMS      = float(getattr(_sc_cfg, "params_weight",      0.10)) if _sc_cfg else 0.10
            W_NAME_ID     = float(getattr(_sc_cfg, "name_id_weight",     0.10)) if _sc_cfg else 0.10
        except Exception:
            W_PURPOSE, W_DESCRIPTION, W_TAGS, W_PARAMS, W_NAME_ID = 0.40, 0.20, 0.20, 0.10, 0.10

        # Auto-normalise so any weight choice produces scores in roughly [0, 1].
        # This makes the ambiguity_floor/gap_threshold thresholds reusable
        # across different weight profiles without manual rescaling.
        _w_sum = W_PURPOSE + W_DESCRIPTION + W_TAGS + W_PARAMS + W_NAME_ID
        if _w_sum > 0 and abs(_w_sum - 1.0) > 0.01:
            W_PURPOSE     /= _w_sum
            W_DESCRIPTION /= _w_sum
            W_TAGS        /= _w_sum
            W_PARAMS      /= _w_sum
            W_NAME_ID     /= _w_sum

        q_lower = query.lower()
        scored: list[tuple[float, str]] = []

        for skill_id, skill in self._skills.items():
            s = skill.summary
            d = skill.detail

            def field_score(text: str) -> float:
                t = (text or "").lower()
                if not t:
                    return 0.0
                if weak_token_signal:
                    # CJK fallback — count overlapping substrings of length 2+
                    hits = sum(1 for i in range(len(q_lower) - 1)
                               if q_lower[i:i + 2] in t)
                    return min(1.0, hits / max(len(q_lower) - 1, 1))
                # ASCII path — word-set overlap normalised by query size
                words = set(_re.findall(r"\b\w{2,}\b", t))
                if not words:
                    return 0.0
                return len(query_words & words) / max(len(query_words), 1)

            purpose_s     = field_score(s.purpose)
            description_s = field_score(d.description)
            params_s      = field_score(" ".join(d.parameters.keys()))
            name_id_s     = field_score(f"{s.name} {skill_id.replace('_', ' ')}")

            # Tags — exact whole-word match for ASCII; substring for CJK.
            if weak_token_signal:
                tag_hits = sum(1 for t in s.tags if t.lower() in q_lower)
            else:
                tag_pattern = _re.compile(
                    r"\b(" + "|".join(_re.escape(t.lower()) for t in s.tags) + r")\b"
                ) if s.tags else None
                tag_hits = len(tag_pattern.findall(q_lower)) if tag_pattern else 0
            tags_s = tag_hits / max(len(s.tags), 1)

            score = (
                W_PURPOSE     * purpose_s
                + W_DESCRIPTION * description_s
                + W_TAGS        * tags_s
                + W_PARAMS      * params_s
                + W_NAME_ID     * name_id_s
            )
            scored.append((round(score, 4), skill_id))

        return scored

    def select_skills_for_query(
        self,
        query: str,
        top_k: int = 5,
        ambiguity_threshold: Optional[float] = None,
        ambiguity_floor:     Optional[float] = None,
    ) -> "SkillSelectionResult":
        """
        Score all registered skills against the query and return the top-K.

        Scoring strategy:
          - If a Retriever has been attached via attach_retriever() (the
            production path), delegate to it. The retriever is typically
            Hybrid (BM25 + embedding fusion + cache) and gives much better
            results on CJK queries, paraphrase, and rare-word queries.
          - Otherwise, use the legacy multi-field weighted keyword scorer
            (see _legacy_score for details). This path also handles CJK by
            falling back to per-character substring containment.

        Multiple skills can match — the agent receives all of them in the
        prompt (Level 1 summary). Level 2 detail is loaded on demand via
        [SKILL_LOAD:skill_id] if the LLM needs it.

        Ambiguity detection:
          ambiguous=True when top-1 score >= ambiguity_floor AND the
          top-2 score gap is below ambiguity_threshold AND at least 2
          candidates exist. Caller chooses how to react (HITL, auto-pick…).

          When ambiguity_threshold/floor are None, defaults come from
          cfg.skill_orchestration (ambiguity_gap_threshold / ambiguity_floor).
        """
        # Load thresholds from config when not explicitly supplied
        if ambiguity_threshold is None or ambiguity_floor is None:
            try:
                from config import cfg as _app_cfg
                _so_cfg = getattr(_app_cfg, "skill_orchestration", None)
                if ambiguity_threshold is None:
                    ambiguity_threshold = float(getattr(_so_cfg, "ambiguity_gap_threshold", 0.08))
                if ambiguity_floor is None:
                    ambiguity_floor = float(getattr(_so_cfg, "ambiguity_floor", 0.40))
            except Exception:
                if ambiguity_threshold is None:
                    ambiguity_threshold = 0.08
                if ambiguity_floor is None:
                    ambiguity_floor = 0.40

        import re as _re
        query_words = set(_re.findall(r'\b\w{2,}\b', query.lower()))

        # ── Scoring path A: retriever-driven (preferred when wired) ──
        if self._retriever is not None:
            try:
                # The retriever is already indexed with this catalog.
                # corpus item dicts include id/text/tags from skills_to_corpus().
                # Oversample so we have enough candidates for re-weighting below.
                res = self._retriever.retrieve(
                    query, top_k=max(top_k * 2, 6),
                )
                scored: list[tuple[float, str]] = []
                seen: set[str] = set()
                for m in res.matches:
                    if m.id in self._skills:
                        # Lightly boost when the retriever reports BM25 strong-match
                        # (lexical overlap is high-signal for exact skill IDs).
                        bm25_part = float(m.breakdown.get("bm25", 0.0) or 0.0)
                        boost = min(0.05, bm25_part * 0.1)
                        scored.append((round(float(m.score) + boost, 4), m.id))
                        seen.add(m.id)
                # Add zero-scored skills not in retrieved set so top_k always
                # has 5 candidates even if the retriever returned fewer.
                for sid in self._skills:
                    if sid not in seen:
                        scored.append((0.0, sid))
            except Exception as _exc:
                logger.warning(
                    "SkillCatalog: retriever scoring failed (%s) — falling back to keyword",
                    _exc,
                )
                scored = self._legacy_score(query_words, query)
        else:
            # ── Scoring path B: legacy keyword (fallback when no retriever) ──
            scored = self._legacy_score(query_words, query)

        scored.sort(reverse=True)
        top = scored[:top_k]

        # ambiguous fires only when:
        #   1. top score is HIGH ENOUGH to be worth loading (>=
        #      ambiguity_floor; otherwise nothing's a real match — just
        #      let the LLM read the catalog summary), AND
        #   2. top-2 scores are within ambiguity_threshold (real tie).
        # This prevents weak matches (e.g. top=0.22, second=0.16) from
        # being flagged as "ambiguous" — when no skill is a strong fit,
        # the LLM picks from the prompt context without operator help.
        ambiguous = (
            len(top) >= 2
            and top[0][0] >= ambiguity_floor
            and abs(top[0][0] - top[1][0]) < ambiguity_threshold
        )

        meaningful = [(sc, sid) for sc, sid in top if sc >= 0.01]
        if not meaningful:
            summary  = self.format_summary()
            selected = [(sid, sc) for sc, sid in top[:top_k]]
        else:
            lines = [f"[RELEVANT SKILLS — top {len(meaningful)} matched for this query]"]
            for score, skill_id in meaningful:
                sk = self._skills[skill_id]
                hitl_tag = " ⚠ HITL" if sk.summary.requires_hitl else ""
                lines.append(
                    f"  {skill_id:<25} [{sk.summary.risk_level:>8}]{hitl_tag}"
                    f"  {sk.summary.purpose}"
                    f"  (score={score:.2f})"
                )
            summary  = "\n".join(lines)
            selected = [(sid, sc) for sc, sid in meaningful]

        return SkillSelectionResult(
            selected=selected,
            ambiguous=ambiguous,
            summary=summary,
            top_score=top[0][0] if top else 0.0,
            second_score=top[1][0] if len(top) > 1 else 0.0,
        )


# ---------------------------------------------------------------------------
# Skill selection result
# ---------------------------------------------------------------------------

from dataclasses import dataclass as _dc

@_dc
class SkillSelectionResult:
    selected:     list   # [(skill_id, score), ...] sorted descending
    ambiguous:    bool   # top-2 scores within ambiguity_threshold of each other
    summary:      str    # Level-1 prompt string containing only matched skills
    top_score:    float
    second_score: float


# ---------------------------------------------------------------------------
# Default IT-ops skill definitions (registered at startup)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Query-matched skill selection (Q4)
# --------------------------------------------------------------------------
# DEFAULT_SKILL_DEFINITIONS removed — use ToolLoader.skill_definitions() instead.
# Skills are now in:
#   skills/builtin/registry.py   (always-available)
#   skills/mock/registry.py      (mock mode)
#   skills/pragmatic/registry.py (pragmatic mode)