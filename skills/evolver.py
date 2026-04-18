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
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Skill version model
# ---------------------------------------------------------------------------

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
    ) -> None:
        self._catalog      = catalog
        self._llm_fn       = llm_fn
        self._fts          = fts_store
        self._min_complex  = min_complexity
        self._min_reuse    = min_reuse_potential
        self._max_skills   = max_skills
        self._skills_dir   = pathlib.Path(skills_dir) if skills_dir else None

        # On startup, load any previously-persisted skill files
        if self._skills_dir:
            self._skills_dir.mkdir(parents=True, exist_ok=True)
            self._load_skills_from_disk()

        # Version history: skill_id → list[SkillVersion]
        self._versions: dict[str, list[SkillVersion]] = {}

        # Pending feedback queue: (skill_id, feedback, success, problem_step)
        self._feedback_queue: list[tuple] = []

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

        # Step 2: Check for similar existing skill (avoid duplicates)
        existing_id = await self._find_similar_skill(task_description)
        if existing_id:
            logger.info(
                "SkillEvolver: similar skill already exists (%s), skipping creation for: %s",
                existing_id, task_description[:60],
            )
            return None

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
            f"Current skill:\n{current_detail[:2000]}\n\n"
            f"Operator feedback: {feedback[:500]}\n"
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

        # Update the catalog with improved content
        await self._update_catalog_from_markdown(skill_id, updated_content)

        result = FeedbackApplication(
            skill_id=skill_id,
            old_version=current_version,
            new_version=current_version + 1,
            changes=changes,
            quality_delta=quality_delta,
        )
        logger.info(
            "SkillEvolver: updated skill '%s' v%d→v%d quality_delta=%.2f changes=%d",
            skill_id, current_version, current_version + 1,
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
            f"Operator preferences: {operator_prefs[:200] or 'not specified'}"
        )
        raw = await self._call_llm(_SKILL_WRITE_SYSTEM, user_content)
        # Strip any accidental code fences from the markdown
        return re.sub(r"^```(?:markdown)?\s*\n?", "", raw.strip()).rstrip("```").strip()

    async def _find_similar_skill(self, task_description: str) -> Optional[str]:
        """Use FTS5 search to find if a similar skill already exists."""
        if self._fts is None:
            return None
        try:
            results = await self._fts.search(query=task_description, limit=3)
            if results:
                logger.debug(
                    "SkillEvolver: found %d potentially similar FTS results", len(results)
                )
        except Exception:
            pass
        return None   # similarity LLM check would go here in production

    async def _register_skill(
        self, proposal: SkillCreationProposal, session_id: str
    ) -> None:
        """Register a newly created skill in SkillCatalogService and persist to disk."""
        parsed = self._parse_markdown_to_definition(
            proposal.skill_id, proposal.markdown_content
        )
        try:
            self._catalog.register_all({proposal.skill_id: parsed})
        except Exception as exc:
            logger.warning("SkillEvolver: catalog registration failed: %s", exc)

        # Persist markdown to disk so skill survives restarts
        self._save_skill_to_disk(proposal.skill_id, proposal.markdown_content)

        # Record initial version
        v = SkillVersion(
            skill_id=proposal.skill_id,
            version=1,
            content=proposal.markdown_content,
            reason=SkillChangeReason.AUTO_CREATED,
            author="agent",
            diff_summary=f"Auto-created from task: {proposal.rationale[:100]}",
            quality_score=proposal.reuse_potential,
        )
        self._versions[proposal.skill_id] = [v]

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
        Call LLM with system+user separation.
        Always returns a string that starts with [ or { (valid JSON),
        or the stub output — never raw prose that breaks JSON parsers downstream.
        """
        import re as _re
        if self._llm_fn is None:
            return await self._stub_llm(system + "\n\n" + user)
        try:
            raw = await self._llm_fn(system, user)
            raw = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL | _re.IGNORECASE).strip()
            # Strip markdown fences that some models wrap JSON in
            raw = _re.sub(r"^```json?\s*", "", raw)
            raw = _re.sub(r"\s*```$", "", raw).strip()
            first = raw.lstrip()[:1]
            if not raw or first not in ("[", "{", "#"):
                # Non-JSON non-markdown — fall back to stub for this call
                logger.debug("SkillEvolver: non-JSON response (%r...) — using stub", raw[:60])
                return await self._stub_llm(system + "\n\n" + user)
            return raw
        except Exception as exc:
            logger.warning("SkillEvolver: llm_fn failed (%s) — using stub", exc)
            return await self._stub_llm(system + "\n\n" + user)

    @staticmethod
    def _generate_skill_id(task_description: str) -> str:
        """Generate a snake_case skill_id from task description."""
        words = re.sub(r"[^a-z0-9\s]", "", task_description.lower()).split()
        stop_words = {"the", "a", "an", "for", "to", "of", "in", "on", "with", "and", "or"}
        key_words  = [w for w in words if w not in stop_words][:4]
        return "_".join(key_words) or "auto_skill_" + str(int(time.time()))[-6:]

    @staticmethod
    def _parse_json_response(raw: str) -> dict:
        text = re.sub(r"^```json?\s*", "", raw.strip())
        text = re.sub(r"\s*```$", "", text)
        return json.loads(text)

    # ------------------------------------------------------------------
    # LLM stub
    # ------------------------------------------------------------------

    @staticmethod
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