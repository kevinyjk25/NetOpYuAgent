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
from typing import Any, Optional

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
                tool_refs = _re.findall(r"\[TOOL:(\w+)\]", skill.detail.description or "")
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

    def select_skills_for_query(
        self,
        query: str,
        top_k: int = 5,
        ambiguity_threshold: float = 0.15,
    ) -> "SkillSelectionResult":
        """
        Score all registered skills against the query and return the top-K.

        Scoring (keyword + tag overlap, no embedding needed):
          keyword_score = |query_words ∩ skill_words| / |query_words|
          tag_score     = tags appearing in query / total tags
          composite     = keyword_score * 0.7 + tag_score * 0.3

        Multiple skills can match — the agent receives all of them in the
        prompt (Level 1 summary). Level 2 detail is loaded on demand via
        [SKILL_LOAD:skill_id] if the LLM needs it.

        ambiguous is True when the top-2 scores differ by less than
        ambiguity_threshold AND both are non-trivial (> 0.05).
        The caller decides whether to trigger HITL on ambiguity.
        """
        import re as _re
        query_words = set(_re.findall(r'\b\w{3,}\b', query.lower()))

        scored: list[tuple[float, str]] = []
        for skill_id, skill in self._skills.items():
            s = skill.summary
            d = skill.detail
            skill_text = " ".join([
                skill_id, s.name, s.purpose, d.description,
                " ".join(s.tags),
                " ".join(d.parameters.keys()),
            ]).lower()
            skill_words = set(_re.findall(r'\b\w{3,}\b', skill_text))

            kw_score  = len(query_words & skill_words) / max(len(query_words), 1)
            tag_score = sum(1 for t in s.tags if t.lower() in query.lower()) / max(len(s.tags), 1)
            score     = round(kw_score * 0.7 + tag_score * 0.3, 4)
            scored.append((score, skill_id))

        scored.sort(reverse=True)
        top = scored[:top_k]

        ambiguous = (
            len(top) >= 2
            and top[0][0] > 0.05
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
