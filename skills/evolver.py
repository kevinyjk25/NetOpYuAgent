"""
skills/evolver.py
------------------
SkillEvolver — Hermes-style self-evolving skill system.

Hermes innovations this implements
-------------------------------------
§03 环节二：自主创建Skill (Autonomous Skill Creation)
  "当Hermes完成了一个相对复杂的任务，它会问自己一个问题：
   这个解决方案以后还会用到吗？如果答案是yes，它就把解决方案
   提炼成一个Skill文件。"

§03 环节三：Skill自改进 (Skill Self-Improvement)
  "Skill创建出来不是终点。每次使用的过程中，如果你给了反馈，
   Hermes会拿着这些反馈修改Skill本身。"

§05 Skill系统：会自我进化的能力
  "Hermes的Skill是活的。它跑在学习循环里，根据实际反馈自动优化。"

What this builds on top of your SkillCatalogService
------------------------------------------------------
Current system:  static register() / register_all() with no write path,
                 no version history, no feedback mechanism.

SkillEvolver adds:
  1. Auto-creation:  after a complex task completes, asks LLM
                     "Should this be a reusable skill? If so, write it."
                     → produces a SkillVersion (markdown content)
                     → registers it in SkillCatalogService

  2. Feedback loop:  after skill is used and user gives feedback,
                     calls LLM to patch the specific steps/constraints
                     → creates SkillVersion(version+1) with diff
                     → rolls back if quality drops

  3. Version history: SkillVersion chain with rollback support
                      Every change tracked with reason + timestamp

  4. Eligibility scoring: not every task becomes a skill
                          Threshold: complexity + reuse_potential score

Skill file format (agentskills.io compatible markdown)
-------------------------------------------------------
    # <skill_name>
    **Purpose:** <one sentence>
    **Tags:** [tag1, tag2]
    **Risk:** low|medium|high
    **HITL:** yes|no

    ## Parameters
    - `param1` (type): description
    - `param2` (type): description

    ## Steps
    1. Step one
    2. Step two
    3. Step three

    ## Constraints
    - Constraint one
    - Constraint two

    ## Notes
    Free-form notes and lessons learned.
"""
from __future__ import annotations

import asyncio
import json
import logging
import pathlib
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Skill version model
# ---------------------------------------------------------------------------



def _trunc_cfg():
    """Lazy load AppConfig.truncation; returns None if config not yet loaded."""
    try:
        from config import cfg as _app_cfg
        return getattr(_app_cfg, "truncation", None)
    except Exception:
        return None


def _trunc(value: str, key: str, default: int) -> str:
    """Truncate value to cfg.truncation.<key> chars, or default if config missing."""
    cfg = _trunc_cfg()
    n = int(getattr(cfg, key, default)) if cfg else default
    return value[:n] if isinstance(value, str) else value

class SkillChangeReason(str, Enum):
    AUTO_CREATED    = "auto_created"      # LLM created after task completion
    FEEDBACK_PATCH  = "feedback_patch"    # user feedback improved a step
    MANUAL_EDIT     = "manual_edit"       # operator directly edited
    CONTRADICTION   = "contradiction"     # previous version had errors
    MERGE           = "merge"             # merged two similar skills


@dataclass
class SkillVersion:
    skill_id:    str
    version:     int
    content:     str          # full markdown content
    reason:      SkillChangeReason
    created_at:  float = field(default_factory=time.time)
    author:      str   = "agent"    # "agent" | "operator" | "system"
    diff_summary: str  = ""         # human-readable change summary
    quality_score: float = 0.0      # 0.0–1.0, estimated by LLM after feedback


@dataclass
class SkillCreationProposal:
    """LLM proposal to create a new skill after task completion."""
    should_create:    bool
    skill_id:         str        # snake_case identifier
    reuse_potential:  float      # 0.0–1.0
    complexity_score: float      # 0.0–1.0
    markdown_content: str
    rationale:        str


@dataclass
class FeedbackApplication:
    """Result of applying feedback to an existing skill."""
    skill_id:     str
    old_version:  int
    new_version:  int
    changes:      list[str]
    quality_delta: float      # positive = improved


# ---------------------------------------------------------------------------
# LLM Prompts
# ---------------------------------------------------------------------------

_ELIGIBILITY_SYSTEM = """You are deciding whether an IT operations task solution is worth capturing as a reusable skill.

A skill is worth creating if:
  - It will likely appear again in similar form (reuse_potential >= 0.6)
  - It involves more than 2 steps or specific parameter choices
  - It encodes non-obvious domain knowledge

Respond with ONLY a JSON object. No explanation, no markdown.
{"should_create": true|false, "reuse_potential": 0.0-1.0, "rationale": "one sentence reason"}"""

_SKILL_WRITE_SYSTEM = """You are writing an IT operations skill file in agentskills.io markdown format.
The skill will be loaded by an AI agent when a similar task appears in future.

Write ONLY the markdown content. No code fences, no explanation, no preamble.
Use this structure exactly:
# <descriptive_skill_name>
**Purpose:** <one clear sentence>
**Tags:** [tag1, tag2, tag3]
**Risk:** low|medium|high
**HITL:** yes|no

## Parameters
- `param_name` (type): description

## Steps
1. Concrete, actionable step
2. Next step

## Constraints
- Important guard or precondition

## Notes
Lessons learned, edge cases, warnings."""

_FEEDBACK_PATCH_SYSTEM = """You are improving an IT operations skill based on operator feedback.
Preserve everything that worked. Only change what the feedback identifies as wrong or suboptimal.

Respond with ONLY a JSON object. No explanation, no markdown.
{"updated_content": "full updated markdown", "changes": ["change 1", "change 2"], "quality_delta": -1.0 to +1.0}"""


_SKILL_MERGE_SYSTEM = """You are merging a new IT operations solution into an existing skill.

The existing skill already covers a similar problem. A new instance of that
problem was solved with a slightly different approach. Your job: produce a
SINGLE updated skill file that captures the best of both, without
duplicating content or contradicting the existing version.

Rules:
  - Preserve the existing skill's structure (Purpose / Tags / Risk / HITL /
    Parameters / Steps / Constraints / Notes).
  - Add new tools to the Tags list if they widen coverage.
  - Add new steps to the Steps list ONLY if they're genuinely new
    (not paraphrases of existing steps).
  - Append a `## Notes` entry if the new solution found a useful edge case
    or alternative path.
  - Never grow the file past 1500 chars — trim verbose Notes if needed.

Output ONLY the merged markdown. No code fences, no preamble, no explanation.
"""


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _tokenize_for_similarity(text: str) -> set[str]:
    """Tokenise text for Jaccard similarity computation.

    Strategy:
      - Lowercase
      - Extract ASCII tokens (English words, device IDs, tool names) directly
      - For runs of CJK characters: emit BOTH unigrams (single chars) and
        bigrams (adjacent pairs) so 'RADIUS 认证' matches '诊断 RADIUS 认证'
        even though Chinese has no word boundaries
      - Drop common stop words and very short ASCII tokens

    Returns an empty set when nothing usable — caller treats that as
    'no signal, skip similarity check'.
    """
    if not text:
        return set()
    lower = text.lower()
    tokens: set[str] = set()

    # 1. ASCII tokens (English words + device IDs like ap-01, sw-3).
    #    For device ids, also emit the bare prefix ('ap-01' → both 'ap_01'
    #    AND 'ap') so a description mentioning a specific device can still
    #    match an existing skill tagged with the device family.
    for tok in re.findall(r"[a-z][a-z0-9_\-]+", lower):
        if len(tok) < 2:
            continue
        unified = tok.replace("-", "_")
        tokens.add(unified)
        # Emit base prefix too, for device-id ↔ family bridging
        m = re.match(r"^([a-z]{2,})[\-_]?\d", tok)
        if m:
            tokens.add(m.group(1))

    # 2. CJK runs: unigrams + bigrams. This catches the common case where
    #    a Chinese description shares technical terms with an English-tagged
    #    skill — the ASCII terms (radius, ap, etc.) carry through, and CJK
    #    n-grams add a bit more signal between two Chinese descriptions.
    cjk_runs = re.findall(r"[\u4e00-\u9fff]+", text)
    for run in cjk_runs:
        for ch in run:
            tokens.add(ch)
        for i in range(len(run) - 1):
            tokens.add(run[i:i+2])

    # Drop noise
    stop = {
        "the","a","an","for","to","of","in","on","with","and","or","is","are",
        "was","were","be","been","this","that","it","as","by","at","from","can",
        "do","does","not","skill","task","help","check","please",
        "用","的","了","是","有","和","在","或","可以","对","我","你","他","它",
        "请","帮","如何","怎么","什么","这个","那个",
    }
    return {t for t in tokens if t not in stop and len(t) >= 1}




# ---------------------------------------------------------------------------
# SkillEvolver
# ---------------------------------------------------------------------------

class SkillEvolver:
    """
    Self-evolving skill management — Hermes §03 §05 learning loop nodes 2 & 3.

    Integrates with:
      SkillCatalogService  — reads/writes skill definitions
      LLMEngine           — creates and patches skills via LLM
      FTS5SessionStore    — searches for similar existing skills before creating

    Thresholds
    -----------
      min_complexity       = 3.0   (out of 10) to trigger creation evaluation
      min_reuse_potential  = 0.60  to actually create the skill
      max_skills           = 200   cap before oldest low-quality skills are pruned
    """

    def __init__(
        self,
        catalog:             Any,              # SkillCatalogService
        llm_fn:              Optional[Callable] = None,
        fts_store:           Optional[Any] = None,
        min_complexity:      float = 3.0,
        min_reuse_potential: float = 0.60,
        max_skills:          int   = 200,
        skills_dir:          Optional[str] = None,  # directory to persist skill .md files
        apply_changes:       bool = True,  # False = suggest-only (no live catalog mutation)
    ) -> None:
        self._catalog      = catalog
        self._llm_fn       = llm_fn
        self._fts          = fts_store
        self._min_complex  = min_complexity
        self._min_reuse    = min_reuse_potential
        self._max_skills   = max_skills
        self._skills_dir   = pathlib.Path(skills_dir) if skills_dir else None
        # When False, apply_feedback / evaluate_skill_creation compute and
        # record the proposed change (version history, suggestion logs) but do
        # NOT mutate the live skill catalog. Lets an operator run the
        # self-improvement loop in "observe" mode and review suggestions
        # before trusting auto-application. Wired from
        # cfg.skill_orchestration.auto_evolve_apply.
        self._apply_changes = apply_changes

        # On startup, load any previously-persisted skill files
        if self._skills_dir:
            self._skills_dir.mkdir(parents=True, exist_ok=True)
            self._load_skills_from_disk()

        # Version history: skill_id → list[SkillVersion]
        self._versions: dict[str, list[SkillVersion]] = {}

        # Pending feedback queue: (skill_id, feedback, success, problem_step)
        self._feedback_queue: list[tuple] = []

        # ── A/B safety net (Sprint-3-pre, 2026-05) ─────────────────────────
        # Optional compliance-bench wrapper to gate feedback patches. Set via
        # set_bench_runner() after startup. When set, apply_feedback() runs
        # the baseline (old content) and candidate (new content) through a
        # compliance subset; if args_ok would DROP, the patch is rolled back
        # instead of saved. When unset, falls back to the legacy unchecked
        # path (zero regression for unmigrated deployments).
        #
        # Signature contract:
        #   bench_runner(skill_id: str, candidate_content: str)
        #       → Awaitable[ToolComplianceReport | None]
        # Returning None means "couldn't bench" — apply_feedback treats this
        # as "skip safety net" rather than blocking the patch (so a benign
        # bench failure doesn't trap legitimate improvements).
        self._bench_runner: Optional[Callable[[str, str], Awaitable[Any]]] = None

    # ------------------------------------------------------------------
    # 环节二: Auto-creation after complex task
    # ------------------------------------------------------------------

    async def after_task(
        self,
        task_description:  str,
        solution_summary:  str,
        tools_used:        list[str],
        solution_steps:    list[str],
        key_observations:  list[str],
        complexity:        float = 5.0,
        operator_prefs:    str   = "",
        session_id:        str   = "default",
    ) -> Optional[SkillCreationProposal]:
        """
        Called after a complex task completes.
        Evaluates whether to create a new skill and does so if eligible.

        Returns the creation proposal (or None if skill was not created).
        """
        # Step 1: Check eligibility
        if complexity < self._min_complex:
            logger.debug("SkillEvolver: task below complexity threshold (%.1f < %.1f)", complexity, self._min_complex)
            return None

        proposal = await self._evaluate_creation_eligibility(
            task_description, solution_summary, tools_used, complexity
        )

        if not proposal.should_create or proposal.reuse_potential < self._min_reuse:
            logger.debug(
                "SkillEvolver: skill creation skipped — should_create=%s reuse=%.2f",
                proposal.should_create, proposal.reuse_potential,
            )
            return None

        # Step 2: Check for similar existing skill — if found, MERGE the
        # new solution's delta into it instead of creating a duplicate.
        similar = await self._find_similar_skill(task_description)
        if similar:
            existing_id, jaccard = similar
            logger.info(
                "SkillEvolver: similar skill exists (%s, jaccard=%.2f) → merging delta",
                existing_id, jaccard,
            )
            merged = await self._merge_into_existing_skill(
                existing_id      = existing_id,
                task_description = task_description,
                solution_steps   = solution_steps,
                tools_used       = tools_used,
                key_observations = key_observations,
                operator_prefs   = operator_prefs,
            )
            if merged and merged.new_version > merged.old_version:
                # Return a proposal-shaped object pointing at the merged skill
                # so the UI can show "skill updated" instead of "skill created".
                return SkillCreationProposal(
                    should_create   = False,    # nothing new — but updated existing
                    skill_id        = existing_id,
                    reuse_potential = proposal.reuse_potential,
                    complexity_score= proposal.complexity_score,
                    markdown_content= "",
                    rationale       = (
                        f"Merged into existing skill '{existing_id}' "
                        f"(jaccard={jaccard:.2f}, version→{merged.new_version})"
                    ),
                )
            # Merge failed — fall through to a fresh creation as last resort
            logger.debug("SkillEvolver: merge failed, falling through to fresh creation")

        # Step 3: Write the skill content via LLM
        markdown = await self._write_skill_content(
            task_description, solution_steps, tools_used,
            key_observations, operator_prefs,
        )
        proposal.markdown_content = markdown

        # Step 4: Register in catalog
        await self._register_skill(proposal, session_id)

        logger.info(
            "SkillEvolver: created skill '%s' (reuse=%.2f)",
            proposal.skill_id, proposal.reuse_potential,
        )
        return proposal

    # ------------------------------------------------------------------
    # 环节三: Skill self-improvement via feedback
    # ------------------------------------------------------------------

    def set_bench_runner(
        self,
        runner: Optional[Callable[[str, str], Awaitable[Any]]],
    ) -> None:
        """Inject the compliance-bench wrapper used by apply_feedback's
        A/B safety net.

        See __init__ docstring for the signature contract. Pass None to
        disable the safety net (back to legacy unchecked path). main.py
        wires this post-construction, after SkillEvolver, OllamaEngine,
        and the compliance golden set are all loaded.
        """
        self._bench_runner = runner
        if runner is not None:
            logger.info("SkillEvolver: A/B safety-net bench runner wired")
        else:
            logger.info("SkillEvolver: A/B safety-net bench runner cleared")

    async def apply_feedback(
        self,
        skill_id:     str,
        feedback:     str,
        success:      bool = True,
        problem_step: Optional[str] = None,
    ) -> Optional[FeedbackApplication]:
        """
        Apply operator feedback to improve an existing skill.

        This is the Hermes "Skill自改进" mechanism:
        "每次使用的过程中，如果你给了反馈，Hermes会拿着这些反馈
         修改Skill本身。"
        """
        # Get current skill content
        current_detail = self._catalog.load_detail(skill_id)
        if current_detail is None:
            logger.warning("SkillEvolver.apply_feedback: skill %r not found", skill_id)
            return None

        current_versions = self._versions.get(skill_id, [])
        current_version  = len(current_versions)

        # Ask LLM to patch the skill
        user_content = (
            f"Current skill:\n{_trunc(current_detail, 'skill_detail_chars', 2000)}\n\n"
            f"Operator feedback: {_trunc(feedback, 'operator_feedback_chars', 500)}\n"
            f"Was the skill successful overall? {'yes' if success else 'no'}\n"
            f"Specific step with issues: {problem_step or 'not specified'}"
        )
        raw = await self._call_llm(_FEEDBACK_PATCH_SYSTEM, user_content)

        # Parse response
        try:
            data = self._parse_json_response(raw)
            updated_content  = data.get("updated_content", "")
            changes          = data.get("changes", [])
            quality_delta    = float(data.get("quality_delta", 0.0))
        except Exception as exc:
            logger.warning("SkillEvolver.apply_feedback: parse failed: %s", exc)
            return None

        if not updated_content:
            return None

        # ── A/B safety net ──────────────────────────────────────────────
        # If a bench runner is wired, run baseline vs candidate before
        # touching the catalog. A drop in args_ok rejects the patch.
        # This MUST run AFTER updated_content is known but BEFORE any
        # catalog mutation, so a rejected patch leaves no trace.
        if self._bench_runner is not None:
            try:
                baseline_report  = await self._bench_runner(skill_id, current_detail)
                candidate_report = await self._bench_runner(skill_id, updated_content)
            except Exception as bench_exc:
                # Bench failures are non-fatal — log and proceed with the
                # patch. This matches the hooks "observer not gatekeeper"
                # stance: a flaky bench should not block legitimate skill
                # improvements indefinitely.
                logger.warning(
                    "SkillEvolver.apply_feedback: bench runner failed (%s) "
                    "— proceeding without safety net for skill %s",
                    bench_exc, skill_id,
                )
                baseline_report = None
                candidate_report = None

            if baseline_report is not None and candidate_report is not None:
                # Both reports must expose .args_rate (ToolComplianceReport)
                # — duck-typed for testability. A strict drop is rejected;
                # equal/better is allowed (we don't require strict gains
                # since LLM noise can sit within ±2% even with same prompt).
                base_score = float(getattr(baseline_report, "args_rate", 0.0) or 0.0)
                cand_score = float(getattr(candidate_report, "args_rate", 0.0) or 0.0)
                if cand_score < base_score:
                    logger.warning(
                        "SkillEvolver: rollback for skill %r — args_ok would "
                        "DROP %.2f → %.2f (n=%d). Patch rejected, old version kept.",
                        skill_id, base_score, cand_score,
                        int(getattr(baseline_report, "total", 0) or 0),
                    )
                    return None
                logger.info(
                    "SkillEvolver: A/B bench OK for skill %r — args_ok "
                    "%.2f → %.2f (n=%d)",
                    skill_id, base_score, cand_score,
                    int(getattr(baseline_report, "total", 0) or 0),
                )

        # Create new version
        new_ver = SkillVersion(
            skill_id=skill_id,
            version=current_version + 1,
            content=updated_content,
            reason=SkillChangeReason.FEEDBACK_PATCH,
            diff_summary="; ".join(changes[:3]),
            quality_score=max(0.0, min(1.0, 0.5 + quality_delta)),
        )
        if skill_id not in self._versions:
            self._versions[skill_id] = []
        self._versions[skill_id].append(new_ver)

        # Update the catalog with improved content — UNLESS we're in
        # suggest-only mode, in which case we record the proposed version
        # (above) and surface it via the return value / logs, but leave the
        # live catalog untouched for an operator to review.
        if self._apply_changes:
            await self._update_catalog_from_markdown(skill_id, updated_content)
            _applied_note = "applied"
        else:
            _applied_note = "suggested (suggest-only mode; catalog NOT mutated)"

        result = FeedbackApplication(
            skill_id=skill_id,
            old_version=current_version,
            new_version=current_version + 1,
            changes=changes,
            quality_delta=quality_delta,
        )
        logger.info(
            "SkillEvolver: %s skill '%s' v%d→v%d quality_delta=%.2f changes=%d",
            _applied_note, skill_id, current_version, current_version + 1,
            quality_delta, len(changes),
        )
        return result

    def rollback(self, skill_id: str, to_version: Optional[int] = None) -> bool:
        """
        Roll back a skill to a previous version.
        If to_version is None, rolls back to the second-to-last version.

        Note: this is a sync wrapper that schedules the async update.
        Call from async context with: await evolver.rollback_async(skill_id, version)
        """
        versions = self._versions.get(skill_id, [])
        if not versions:
            return False

        target_idx = (to_version - 1) if to_version is not None else len(versions) - 2
        if target_idx < 0 or target_idx >= len(versions):
            logger.warning("SkillEvolver.rollback: version %d not found for skill %r",
                           target_idx + 1, skill_id)
            return False

        target_ver = versions[target_idx]
        # Apply synchronously via the catalog's register_all (which is sync)
        parsed = self._parse_markdown_to_definition(skill_id, target_ver.content)
        try:
            self._catalog.register_all({skill_id: parsed})
        except Exception as exc:
            logger.warning("SkillEvolver.rollback: catalog update failed: %s", exc)
            return False

        logger.info("SkillEvolver: rolled back skill '%s' to v%d", skill_id, target_ver.version)
        return True

    # ------------------------------------------------------------------
    # Version history API
    # ------------------------------------------------------------------

    def get_version_history(self, skill_id: str) -> list[dict]:
        versions = self._versions.get(skill_id, [])
        return [
            {
                "version":     v.version,
                "reason":      v.reason.value,
                "author":      v.author,
                "created_at":  v.created_at,
                "diff_summary": v.diff_summary,
                "quality_score": v.quality_score,
            }
            for v in versions
        ]

    def get_all_skill_stats(self) -> list[dict]:
        return [
            {
                "skill_id":    sid,
                "versions":    len(vlist),
                "latest_reason": vlist[-1].reason.value if vlist else "unknown",
                "quality_score": vlist[-1].quality_score if vlist else 0.0,
            }
            for sid, vlist in self._versions.items()
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _evaluate_creation_eligibility(
        self,
        task_description: str,
        solution_summary: str,
        tools_used:       list[str],
        complexity:       float,
    ) -> SkillCreationProposal:
        user_content = (
            f"Task description: {task_description[:400]}\n"
            f"Solution summary: {solution_summary[:400]}\n"
            f"Tools used: {', '.join(tools_used[:5])}\n"
            f"Complexity (1-10): {int(complexity)}"
        )
        raw = await self._call_llm(_ELIGIBILITY_SYSTEM, user_content)
        try:
            data = self._parse_json_response(raw)
            return SkillCreationProposal(
                should_create=bool(data.get("should_create", False)),
                skill_id=self._generate_skill_id(task_description),
                reuse_potential=float(data.get("reuse_potential", 0.0)),
                complexity_score=complexity / 10.0,
                markdown_content="",
                rationale=data.get("rationale", ""),
            )
        except Exception:
            return SkillCreationProposal(
                should_create=False, skill_id="", reuse_potential=0.0,
                complexity_score=0.0, markdown_content="", rationale="parse_failed",
            )

    async def _write_skill_content(
        self,
        task_description: str,
        solution_steps:   list[str],
        tools_used:       list[str],
        key_observations: list[str],
        operator_prefs:   str,
    ) -> str:
        steps_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(solution_steps[:8]))
        obs_text   = "\n".join(f"- {o}" for o in key_observations[:4])
        user_content = (
            f"Task: {task_description[:400]}\n"
            f"Solution steps taken:\n{steps_text}\n"
            f"Tools that proved effective: {', '.join(tools_used[:6])}\n"
            f"Key observations:\n{obs_text}\n"
            f"Operator preferences: {(_trunc(operator_prefs, 'operator_prefs_chars', 200) or 'not specified')}"
        )
        raw = await self._call_llm(_SKILL_WRITE_SYSTEM, user_content)
        # Strip any accidental code fences from the markdown
        return re.sub(r"^```(?:markdown)?\s*\n?", "", raw.strip()).rstrip("```").strip()

    async def _find_similar_skill(self, task_description: str) -> Optional[tuple[str, float]]:
        """Find an existing skill semantically similar to the proposed task.

        Returns (skill_id, similarity_score) for the best match above the
        threshold, or None if nothing close enough exists.

        Algorithm: token-Jaccard over (skill name + purpose + tags) vs the
        task description. This is intentionally cheap — for a pool of <100
        skills it runs in <1ms and avoids an LLM round-trip.

        Threshold is dynamic:
          - 0.35 by default (English-heavy or mixed descriptions)
          - 0.20 when the description is CJK-dominant (CJK n-grams inflate
            the token-set denominator, lowering Jaccard scores even for
            obvious paraphrases — compensate by lowering the bar)
        """
        if not self._catalog:
            return None
        try:
            task_tokens = _tokenize_for_similarity(task_description)
            if not task_tokens:
                return None

            # Heuristic: if any meaningful chunk of the description is CJK,
            # treat as CJK-influenced and use a much lower threshold. CJK
            # n-grams inflate the token-set denominator (each character
            # becomes ≥ 2 tokens: the unigram + a bigram), and they don't
            # overlap with ASCII-tagged English skills at all — so the
            # signal-to-noise ratio in jaccard collapses.
            cjk_chars = sum(1 for c in task_description if "\u4e00" <= c <= "\u9fff")
            ascii_chars = sum(1 for c in task_description if c.isascii() and c.isalpha())
            total_letters = cjk_chars + ascii_chars
            cjk_share = (cjk_chars / total_letters) if total_letters else 0.0
            if cjk_share > 0.50:
                threshold = 0.08    # CJK-dominant — almost no n-gram overlap with EN tags
            elif cjk_share > 0.20:
                threshold = 0.15    # mixed — partial ASCII overlap
            else:
                threshold = 0.35    # English-dominant — strict

            best_id    = None
            best_score = 0.0
            for summary in self._catalog.list_skills():
                signature = " ".join([
                    summary.name or "",
                    summary.purpose or "",
                    " ".join(summary.tags or []),
                ])
                sk_tokens = _tokenize_for_similarity(signature)
                if not sk_tokens:
                    continue
                union = len(task_tokens | sk_tokens)
                if union == 0:
                    continue
                jaccard = len(task_tokens & sk_tokens) / union
                if jaccard > best_score:
                    best_score = jaccard
                    best_id    = summary.skill_id
            if best_id and best_score >= threshold:
                logger.info(
                    "SkillEvolver: similar skill found — id=%s jaccard=%.2f "
                    "(threshold=%.2f, cjk_share=%.2f)",
                    best_id, best_score, threshold, cjk_share,
                )
                return (best_id, best_score)
        except Exception as exc:
            logger.debug("_find_similar_skill failed: %s", exc)
        return None

    async def _merge_into_existing_skill(
        self,
        existing_id:      str,
        task_description: str,
        solution_steps:   list[str],
        tools_used:       list[str],
        key_observations: list[str],
        operator_prefs:   str,
    ) -> Optional[FeedbackApplication]:
        """When a similar skill already exists, merge the new solution's
        delta (extra tools / new steps / new observations) into it instead
        of creating a duplicate. Produces a new version of the existing
        skill — same id, version++.

        Algorithm:
          1. Load existing skill markdown
          2. Send (existing_md + new_solution_signals) to the LLM with a
             merge prompt that says "ADD what's missing, KEEP what works"
          3. Persist as new version via apply_feedback path

        Returns a FeedbackApplication (same shape as apply_feedback) so
        callers can log/audit consistently.
        """
        existing_md = self._catalog.as_markdown(existing_id) if self._catalog else None
        if not existing_md:
            logger.debug("merge_skill: existing %s has no markdown — skip merge", existing_id)
            return None

        steps_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(solution_steps[:8]))
        obs_text   = "\n".join(f"- {o}" for o in key_observations[:4])
        user_content = (
            f"=== Existing skill ===\n{existing_md}\n\n"
            f"=== New solution observed for similar task ===\n"
            f"Task: {task_description[:300]}\n"
            f"Steps taken:\n{steps_text}\n"
            f"Tools used: {', '.join(tools_used[:6])}\n"
            f"Key observations:\n{obs_text}\n"
            f"Operator preferences: {(_trunc(operator_prefs, 'operator_prefs_chars', 200) or 'not specified')}\n\n"
            f"Merge instructions:\n"
            f"  - KEEP everything in the existing skill that still applies.\n"
            f"  - ADD any new steps / tools / observations that genuinely "
            f"    extend the skill's coverage.\n"
            f"  - DEDUPE: don't repeat what's already there.\n"
            f"  - If the new solution contradicts the existing skill, keep "
            f"    the existing version unchanged and note the alternative in "
            f"    a `## Notes` section.\n"
            f"  - Stay under 1500 chars total."
        )
        try:
            raw = await self._call_llm(_SKILL_MERGE_SYSTEM, user_content)
            updated = re.sub(r"^```(?:markdown)?\s*\n?", "", raw.strip()).rstrip("```").strip()
            if not updated or len(updated) < 50:
                logger.debug("merge_skill: LLM produced empty/tiny content — skip")
                return None
        except Exception as exc:
            logger.warning("merge_skill: LLM call failed (%s) — skip merge", exc)
            return None

        # Reuse the apply_feedback persistence path: it already handles
        # version++, catalog re-registration, disk save.
        return await self._persist_merged_version(
            skill_id    = existing_id,
            new_content = updated,
            note        = f"Merged delta from task: {task_description[:80]}",
        )

    async def _persist_merged_version(
        self, skill_id: str, new_content: str, note: str,
    ) -> FeedbackApplication:
        """Write a new version of an existing skill's markdown.
        Mirrors the persistence half of apply_feedback()."""
        if self._catalog:
            try:
                parsed = self._parse_markdown_to_definition(skill_id, new_content)
                self._catalog.register_all({skill_id: parsed})
            except Exception as exc:
                logger.warning("merge_skill: catalog re-register failed: %s", exc)

        self._save_skill_to_disk(skill_id, new_content)

        history = self._versions.setdefault(skill_id, [])
        next_v  = (history[-1].version + 1) if history else 2
        v = SkillVersion(
            skill_id=skill_id, version=next_v,
            content=new_content,
            reason=SkillChangeReason.MERGE,
            author="agent",
            diff_summary=note,
            quality_score=0.0,    # quality re-evaluated on next use
        )
        history.append(v)
        logger.info(
            "SkillEvolver: merged into %s (version %d) — %s",
            skill_id, next_v, note[:60],
        )
        return FeedbackApplication(
            skill_id      = skill_id,
            old_version   = next_v - 1,
            new_version   = next_v,
            changes       = [note],
            quality_delta = 0.0,
        )

    async def _register_skill(
        self, proposal: SkillCreationProposal, session_id: str
    ) -> None:
        """Register a newly created skill in SkillCatalogService and persist to disk."""
        parsed = self._parse_markdown_to_definition(
            proposal.skill_id, proposal.markdown_content
        )
        # Always record the proposed version (cheap, in-memory, reviewable).
        v = SkillVersion(
            skill_id=proposal.skill_id,
            version=1,
            content=proposal.markdown_content,
            reason=SkillChangeReason.AUTO_CREATED,
            author="agent",
            diff_summary=f"Auto-created from task: {_trunc(proposal.rationale, 'rationale_chars', 100)}",
            quality_score=proposal.reuse_potential,
        )
        self._versions[proposal.skill_id] = [v]

        # In suggest-only mode, stop here: the proposal is recorded for review
        # but NOT registered in the live catalog or written to disk.
        if not self._apply_changes:
            logger.info(
                "SkillEvolver: suggested new skill '%s' (suggest-only mode; "
                "NOT registered in catalog)", proposal.skill_id,
            )
            return

        try:
            self._catalog.register_all({proposal.skill_id: parsed})
        except Exception as exc:
            logger.warning("SkillEvolver: catalog registration failed: %s", exc)

        # Persist markdown to disk so skill survives restarts
        self._save_skill_to_disk(proposal.skill_id, proposal.markdown_content)

    def _save_skill_to_disk(self, skill_id: str, content: str) -> None:
        """Write skill markdown to HERMES_DATA_DIR/skills/<skill_id>.md"""
        if not self._skills_dir:
            return
        try:
            path = self._skills_dir / f"{skill_id}.md"
            path.write_text(content, encoding="utf-8")
            logger.info("SkillEvolver: saved skill to %s", path)
        except Exception as exc:
            logger.warning("SkillEvolver: disk save failed for %s: %s", skill_id, exc)

    def _load_skills_from_disk(self) -> None:
        """Load all .md files from skills_dir into the catalog on startup."""
        if not self._skills_dir or not self._skills_dir.exists():
            return
        loaded = 0
        for path in sorted(self._skills_dir.glob("*.md")):
            skill_id = path.stem
            try:
                content = path.read_text(encoding="utf-8")
                parsed  = self._parse_markdown_to_definition(skill_id, content)
                self._catalog.register_all({skill_id: parsed})
                loaded += 1
            except Exception as exc:
                logger.warning("SkillEvolver: failed to load %s: %s", path, exc)
        if loaded:
            logger.info("SkillEvolver: loaded %d persisted skill(s) from %s", loaded, self._skills_dir)

    async def _update_catalog_from_markdown(
        self, skill_id: str, markdown: str
    ) -> None:
        """Re-parse updated markdown, update catalog in place, and persist to disk."""
        parsed = self._parse_markdown_to_definition(skill_id, markdown)
        try:
            self._catalog.register_all({skill_id: parsed})
        except Exception as exc:
            logger.warning("SkillEvolver: catalog update failed: %s", exc)
        self._save_skill_to_disk(skill_id, markdown)

    # ------------------------------------------------------------------
    # Markdown ↔ definition converters
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json_response(raw: str) -> dict:
        """Extract a JSON object from an LLM response.

        Used by apply_feedback() and evaluate_skill_creation() to read
        the LLM's structured response. The LLM is *prompted* to emit
        strict JSON, but in practice models wrap the JSON in:
          - markdown code fences (```json … ```)
          - prose preamble ("Here is the JSON:\\n{...}")
          - trailing chatter
          - <think>…</think> blocks (some models)

        We try strict parsing first, then fall back to extracting the
        first balanced { … } substring. Returns a dict (possibly empty
        if everything failed). Never raises — callers decide what to
        do with an empty result.

        FIXED 2026-05: this method was referenced from two call-sites
        but had never been implemented; every LLM-driven path silently
        hit the `except Exception` branch and returned None, breaking
        skill feedback patches and auto-creation evaluation entirely.
        """
        if not raw:
            return {}

        text = raw.strip()

        # 1. Strip <think>…</think> blocks (qwen-3, deepseek-r1, etc).
        #    These can contain braces that confuse the balanced scan.
        text = re.sub(
            r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE,
        ).strip()

        # 2. Strip markdown code fence wrapping. ```json … ``` and ``` … ```
        #    are both common. Match conservatively — only if the FIRST line
        #    is a fence, so we don't eat braces inside legitimate content.
        if text.startswith("```"):
            # Find the closing fence; take what's between.
            first_nl = text.find("\n")
            if first_nl != -1:
                close = text.rfind("```")
                if close > first_nl:
                    text = text[first_nl + 1 : close].strip()

        # 3. Strict parse first — common path when the LLM behaved.
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else {}
        except (json.JSONDecodeError, ValueError):
            pass

        # 4. Fall back to balanced-brace extraction. Find the first '{'
        #    and walk forward tracking depth + string state. Returns the
        #    first complete top-level object; ignores anything after.
        first_brace = text.find("{")
        if first_brace < 0:
            logger.debug("SkillEvolver._parse_json_response: no '{' found")
            return {}

        depth = 0
        in_str = False
        esc = False
        end = -1
        for i in range(first_brace, len(text)):
            ch = text[i]
            if esc:
                esc = False
                continue
            if ch == "\\" and in_str:
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break

        if end < 0:
            logger.debug(
                "SkillEvolver._parse_json_response: unbalanced braces (depth=%d)",
                depth,
            )
            return {}

        candidate = text[first_brace : end + 1]
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else {}
        except (json.JSONDecodeError, ValueError) as exc:
            logger.debug(
                "SkillEvolver._parse_json_response: JSON decode failed: %s "
                "(snippet=%r)", exc, candidate[:80],
            )
            return {}

    @staticmethod
    def _parse_markdown_to_definition(skill_id: str, content: str) -> dict:
        """
        Parse agentskills.io-format markdown into a SkillCatalogService definition dict.
        """
        lines = content.splitlines()
        defn: dict[str, Any] = {
            "name":          skill_id.replace("_", " ").title(),
            "purpose":       "",
            "risk_level":    "low",
            "requires_hitl": False,
            "tags":          [],
            "description":   content[:600],
            "parameters":    {},
            "returns":       "string",
            "examples":      [],
            "constraints":   [],
            "estimated_size": "small",
            "returns_large":  False,
        }
        current_section = ""
        steps: list[str] = []

        for line in lines:
            stripped = line.strip()

            # Title (purpose)
            if stripped.startswith("# ") and not defn["purpose"]:
                defn["name"] = stripped[2:].strip()

            # Purpose from **Purpose:** line
            elif "**Purpose:**" in stripped or stripped.startswith("Purpose:"):
                defn["purpose"] = re.sub(r"\*?\*?Purpose:\*?\*?\s*", "", stripped).strip()

            # Tags
            elif "**Tags:**" in stripped or stripped.startswith("Tags:"):
                tags_str = re.sub(r"\*?\*?Tags:\*?\*?\s*", "", stripped)
                defn["tags"] = [t.strip().strip("[]") for t in tags_str.split(",") if t.strip()]

            # Risk
            elif "**Risk:**" in stripped or stripped.startswith("Risk:"):
                risk = re.sub(r"\*?\*?Risk:\*?\*?\s*", "", stripped).strip().lower()
                if risk in ("low", "medium", "high", "critical"):
                    defn["risk_level"] = risk

            # HITL
            elif "**HITL:**" in stripped:
                defn["requires_hitl"] = "yes" in stripped.lower()

            # Section headers
            elif stripped.startswith("## "):
                current_section = stripped[3:].lower()

            # Parameters section
            elif current_section == "parameters" and stripped.startswith("-"):
                # - `param` (type): description
                m = re.match(r"-\s+`?(\w+)`?\s*(?:\(([^)]*)\))?:?\s*(.*)", stripped)
                if m:
                    defn["parameters"][m.group(1)] = m.group(3) or m.group(1)

            # Steps section
            elif current_section == "steps" and re.match(r"\d+\.", stripped):
                steps.append(re.sub(r"^\d+\.\s*", "", stripped))

            # Constraints section
            elif current_section == "constraints" and stripped.startswith("-"):
                defn["constraints"].append(stripped[1:].strip())

        if not defn["purpose"] and defn["name"]:
            defn["purpose"] = f"Execute {defn['name'].lower()} procedure"

        if steps:
            defn["description"] = defn.get("description", "") + "\n\nSteps:\n" + "\n".join(
                f"{i+1}. {s}" for i, s in enumerate(steps)
            )
        return defn

    async def _call_llm(self, system: str, user: str) -> str:
        """
        DESIGN-04 fix: wraps all LLM calls in asyncio.wait_for() so a hung
        Ollama/OpenAI backend cannot block the Hermes pipeline indefinitely.

        Timeout is read from AppConfig.hermes.skill_evolver_llm_timeout_seconds
        (default 30s, configurable in config.yaml).  On timeout, returns an
        empty string so callers can gracefully skip the current Hermes step.
        """
        # Load timeout from config (read once per call; cheap dict lookup)
        try:
            from config import cfg as _app_cfg
            _hermes = getattr(_app_cfg, "hermes", None)
            timeout = float(getattr(_hermes, "skill_evolver_llm_timeout_seconds", 30.0))
        except Exception:
            timeout = 30.0

        if self._llm_fn is None:
            return await self._stub_llm(system + "\n\n" + user)

        try:
            coro = self._llm_fn(system, user)
            raw  = await asyncio.wait_for(coro, timeout=timeout)
            return (raw or "").strip()
        except asyncio.TimeoutError:
            logger.warning(
                "SkillEvolver._call_llm: timed out after %.1fs (system=%s…)",
                timeout, system[:80],
            )
            return ""
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("SkillEvolver._call_llm failed: %s", exc)
            try:
                return await self._stub_llm(system + "\n\n" + user)
            except Exception:
                return ""


    async def _stub_llm(text: str) -> str:
        await asyncio.sleep(0)
        p = text.lower()

        if "should_create" in p:
            # Creation eligibility check
            return json.dumps({
                "should_create":   True,
                "reuse_potential": 0.78,
                "rationale":       "Multi-step diagnostic with specific tool sequence — high reuse",
            })

        if "updated_content" in p:
            # Feedback patch
            return json.dumps({
                "updated_content": "# Updated Skill\n**Purpose:** Updated via feedback\n\n## Steps\n1. Check device status\n2. Review syslogs\n3. Apply fix\n\n## Notes\nUpdated based on operator feedback.",
                "changes":         ["Added pre-check step", "Clarified verification step"],
                "quality_delta":   0.15,
            })

        if "are_similar" in p:
            # Similarity check
            return json.dumps({"are_similar": False, "similarity_score": 0.3, "reason": "different domains"})

        # Default: write skill content
        return """# Network Diagnostic Procedure
**Purpose:** Diagnose network connectivity issues for IT operations
**Tags:** [network, diagnostic, syslog]
**Risk:** low
**HITL:** no

## Parameters
- `device_id` (string): Target device identifier
- `severity` (string): Log severity filter (error|warn|info)

## Steps
1. Query device status using get_device_status tool
2. Search syslogs for error patterns using syslog_search
3. Check interface metrics for utilization spikes
4. Review BGP/routing table if connectivity issues persist
5. Document findings and recommended actions

## Constraints
- Always verify device ID before running tools
- Do not modify device configuration without HITL approval

## Notes
This procedure works well for AP-related connectivity complaints.
Use lines=100 for syslog_search as a starting point."""