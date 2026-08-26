"""
webui/routes_skills.py — Skills + skill_journal endpoints.

EXTRACTED FROM webui/backend.py during audit refactor D-4 (see
AUDIT_REPORT.md). Routes here cover the skill catalog, upload, generation
from text, and skill journal (recent/stats/filter).

Public API:
    register_skills_routes(app, services)
"""
from __future__ import annotations

import json
import logging
import pathlib
import re
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def register_skills_routes(app: FastAPI, services: dict[str, Any]) -> None:
    """Attach /skills/* and /skill_journal/* endpoints to `app`."""
    # Late import to avoid circular dependency on backend.py at module load
    from webui.backend import _identity   # noqa: F401 (used in some routes)

    @app.post("/evolution/sweep")
    async def evolution_sweep(request: Request):
        """Manually trigger P1 (recurring-trajectory mining) and optionally P3
        (append merge), for verification — bypasses the every-N-tasks counter
        and the append-marker gate.

        Body (all optional):
          { "limit": 200, "session_id": "...", "append_text": "...",
            "dry_run": true }
        Returns what fired + why, so you can see the mechanism, not just results.
        """
        try:
            body = {}
            try:
                body = await request.json()
            except Exception:
                pass
            limit       = int(body.get("limit", 200))
            session_id  = body.get("session_id")
            append_text = body.get("append_text")
            dry_run     = bool(body.get("dry_run", False))

            evolver = services.get("skill_evolver")
            if evolver is None:
                raise HTTPException(503, "skill_evolver not available")

            from webui.backend import build_csi_from_profile
            from skills.trajectory_miner import TrajectoryMiner
            from runtime.skill_journal import get_journal_store
            jstore = get_journal_store()
            csi = await build_csi_from_profile(services)

            result: dict[str, Any] = {"embedder": (
                "real" if services.get("embedder") else "none(tag+tool_set only)")}

            async def _evolve_cb(**kw):
                return await evolver.after_task(
                    task_description=kw["task_description"],
                    solution_summary=kw["task_description"][:200],
                    tools_used=kw["tools_used"], solution_steps=kw["solution_steps"],
                    key_observations=kw["key_observations"],
                    complexity=kw["complexity"], session_id=kw["session_id"])
            miner = TrajectoryMiner(jstore, csi, _evolve_cb)
            clusters = miner.find_recurring(limit=limit)
            result["p1"] = {
                "recurring_clusters": [
                    {"size": c.size, "tools": sorted(c.rep_tools),
                     "sample_query": c.sample_query,
                     "covered_by_skill": c.covered_by_skill}
                    for c in clusters
                ],
                "threshold": miner.cfg.recurrence_threshold,
            }
            if not dry_run:
                solidified = []
                for c in clusters:
                    p = await miner.solidify(c)
                    if p:
                        solidified.append(getattr(p, "skill_id", str(p)))
                result["p1"]["solidified"] = solidified

            if session_id and append_text:
                from skills.append_merger import AppendMerger
                traj = jstore.extract_trajectory(session_id)
                active = (traj.get("loaded_skills") or [None])[0]
                async def _merge_cb(*, skill_id, append_text, session_id, tools):
                    fb = await evolver._merge_into_existing_skill(
                        existing_id=skill_id, task_description=append_text,
                        solution_steps=traj.get("steps", []), tools_used=tools,
                        key_observations=traj.get("observations", []), operator_prefs="")
                    return fb is not None
                merger = AppendMerger(csi, _merge_cb)
                mres = await merger.maybe_merge(
                    append_text=append_text, session_id=session_id,
                    active_skill=active, session_tools=traj.get("tools", []))
                result["p3"] = {
                    "active_skill": active, "merged": mres.merged,
                    "skill_id": mres.skill_id, "reason": mres.reason,
                    "similarity": round(mres.similarity, 3),
                }
            return JSONResponse(content=result)
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("/evolution/sweep failed: %s", exc)
            raise HTTPException(500, str(exc))

    @app.get("/evolution/space")
    async def evolution_space():
        """Dump the CSI capability space (clusters + per-capability domain/
        tool_set) for auditing — the interpretable index, not a black box."""
        try:
            from webui.backend import build_csi_from_profile
            csi = await build_csi_from_profile(services)
            return JSONResponse(content=csi.export_space())
        except Exception as exc:
            logger.warning("/evolution/space failed: %s", exc)
            raise HTTPException(500, str(exc))

    @app.get("/skill_journal/recent")
    async def skill_journal_recent(limit: int = 20):
        """Return the most recent SkillJournal entries (newest first)."""
        try:
            from config import cfg as _app_cfg
            _so = getattr(_app_cfg, "skill_orchestration", None)
            if not _so or not getattr(_so, "journal_api_enabled", True):
                raise HTTPException(503, "Skill journal API disabled")
            from runtime.skill_journal import get_journal_store
            store = get_journal_store()
            return JSONResponse(content={"entries": store.list_recent(limit=min(max(1, limit), 100))})
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("/skill_journal/recent failed: %s", exc)
            raise HTTPException(500, str(exc))

    @app.get("/evolution/gaps")
    async def evolution_gaps(limit: int = 200):
        """Aggregate declared CAPABILITY_GAP events from recent journal
        entries — the 'missing capability ledger'. The inverse of P1's
        solidify: P1 captures what the agent DOES repeatedly and worth
        codifying; this captures what it CANNOT do and is worth adding."""
        try:
            from runtime.skill_journal import get_journal_store
            store = get_journal_store()
            counts: dict[str, int] = {}
            samples: dict[str, str] = {}
            total = 0
            for e in store.list_recent(limit=limit):
                for g in (e.get("capability_gaps") or []):
                    total += 1
                    counts[g] = counts.get(g, 0) + 1
                    samples.setdefault(g, e.get("query", ""))
            ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
            return JSONResponse(content={
                "total_gap_events": total,
                "distinct_gaps": len(counts),
                "gaps": [{"detail": d, "count": c, "sample_query": samples.get(d, "")}
                         for d, c in ranked],
            })
        except Exception as exc:
            logger.warning("/evolution/gaps failed: %s", exc)
            raise HTTPException(500, str(exc))

    @app.get("/skill_journal/stats")
    async def skill_journal_stats():
        """Aggregate stats: outcomes, per-skill use count, dormancy."""
        try:
            from config import cfg as _app_cfg
            _so = getattr(_app_cfg, "skill_orchestration", None)
            if not _so or not getattr(_so, "journal_api_enabled", True):
                raise HTTPException(503, "Skill journal API disabled")
            from runtime.skill_journal import get_journal_store
            return JSONResponse(content=get_journal_store().stats())
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("/skill_journal/stats failed: %s", exc)
            raise HTTPException(500, str(exc))

    @app.get("/skill_journal/filter")
    async def skill_journal_filter(
        skill_id:   Optional[str]  = None,
        outcome:    Optional[str]  = None,
        ambiguous:  Optional[bool] = None,
        limit:      int            = 50,
    ):
        """Filter journal entries by skill, outcome, or ambiguity flag."""
        try:
            from config import cfg as _app_cfg
            _so = getattr(_app_cfg, "skill_orchestration", None)
            if not _so or not getattr(_so, "journal_api_enabled", True):
                raise HTTPException(503, "Skill journal API disabled")
            from runtime.skill_journal import get_journal_store
            return JSONResponse(content={
                "entries": get_journal_store().filter(
                    skill_id=skill_id, outcome=outcome,
                    ambiguous=ambiguous, limit=min(max(1, limit), 200),
                )
            })
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("/skill_journal/filter failed: %s", exc)
            raise HTTPException(500, str(exc))

    @app.get("/skills")
    async def list_skills() -> JSONResponse:
        catalog    = services["skill_catalog"]
        skill_evol = services.get("skill_evolver")
        import pathlib as _pl

        # Which skills live on disk (evolved / uploaded — not just built-in).
        # Standard layout: <kebab-name>/SKILL.md (metadata.skill_id is the id);
        # legacy flat layout: <skill_id>.md.
        evolved_ids: set = set()
        if skill_evol and getattr(skill_evol, "_skills_dir", None):
            skills_dir = _pl.Path(skill_evol._skills_dir)
            if skills_dir.exists():
                for md in skills_dir.glob("*/SKILL.md"):
                    evolved_ids.add(md.parent.name.replace("-", "_"))
                # Legacy flat files
                evolved_ids |= {p.stem for p in skills_dir.glob("*.md")}

        return JSONResponse(content=[
            {
                "skill_id":      s.skill_id,
                "name":          s.name,
                "purpose":       s.purpose,
                "risk_level":    s.risk_level,
                "requires_hitl": s.requires_hitl,
                "tags":          s.tags,
                "is_evolved":    s.skill_id in evolved_ids,
            }
            for s in catalog.list_skills()
        ])

    @app.get("/skills/{skill_id}")
    async def get_skill_detail(skill_id: str) -> JSONResponse:
        """Load skill Level 2 detail on demand — progressive disclosure."""
        catalog = services["skill_catalog"]
        detail  = catalog.load_detail(skill_id)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"Skill {skill_id!r} not found")
        summary = catalog.get_summary(skill_id)
        return JSONResponse(content={
            "skill_id":      skill_id,
            "requires_hitl": catalog.requires_hitl(skill_id),
            "detail":        detail,
            "risk_level":    summary.risk_level if summary else "unknown",
        })

    @app.post("/skills/upload")
    async def upload_skill(request: Request,
    ) -> JSONResponse:
        """
        Upload a skill markdown file (.md) or JSON definition (.json).
        The skill is registered in the catalog and persisted to HERMES_DATA_DIR/skills/.
        Uses Request directly (not File()) to work correctly in mounted sub-apps.
        Gated behind 'admin' role.
        """
        ident = await _identity()
        if not ident.has_role("admin"):
            raise HTTPException(
                status_code=403,
                detail="Skill upload requires the 'admin' role",
            )

        catalog    = services.get("skill_catalog")
        skill_evol = services.get("skill_evolver")
        if not catalog:
            raise HTTPException(status_code=503, detail="Skill catalog not available")

        try:
            form = await request.form()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to parse form data: {exc}")

        upload = form.get("file")
        if upload is None:
            raise HTTPException(status_code=400, detail="No file field in form data — field name must be 'file'")

        try:
            content_bytes = await upload.read()
            content = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File must be UTF-8 encoded text")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to read file: {exc}")

        filename  = getattr(upload, "filename", None) or "uploaded_skill"
        skill_id  = filename.removesuffix(".md").removesuffix(".json")
        # SKILL.md uploads carry the id in metadata, not the filename.
        if skill_id.upper() == "SKILL":
            skill_id = "uploaded_skill"
        # Sanitise: only alphanumeric + underscore
        import re as _re
        skill_id  = _re.sub(r"[^a-zA-Z0-9_]", "_", skill_id).strip("_") or "uploaded_skill"

        if filename.endswith(".json"):
            import json as _json
            try:
                defn = _json.loads(content)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}")
            skill_id = defn.get("skill_id", skill_id)
            try:
                catalog.register_all({skill_id: defn})
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Registration failed: {exc}")
        else:
            # Markdown upload. Standard SKILL.md (with YAML frontmatter) is the
            # preferred form; bare-body markdown is still accepted. Both route
            # through skill_format / the evolver parser so the resulting
            # definition is identical to a freshly-loaded standard skill.
            from skills.skill_format import (
                SkillFormatError,
                has_frontmatter as _has_fm,
                load_skill_md as _load_md,
            )
            try:
                if _has_fm(content):
                    # Standard format — metadata.skill_id is authoritative.
                    skill_id, defn = _load_md(content, skill_id_hint=skill_id)
                elif skill_evol and hasattr(skill_evol, "_parse_markdown_to_definition"):
                    defn = skill_evol._parse_markdown_to_definition(skill_id, content)
                else:
                    # Minimal fallback: register with raw content as description.
                    defn = {
                        "name":          skill_id.replace("_", " ").title(),
                        "purpose":       content.split("\n")[0].lstrip("# ").strip()[:200],
                        "description":   content,
                        "risk_level":    "low",
                        "requires_hitl": False,
                        "tags":          [],
                        "parameters":    {},
                        "returns":       "string",
                        "examples":      [],
                        "constraints":   [],
                        "estimated_size": "small",
                        "returns_large":  False,
                    }
            except SkillFormatError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid SKILL.md format: {exc}",
                )
            try:
                catalog.register_all({skill_id: defn})
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Registration failed: {exc}")

        # Persist to disk via SkillEvolver if available
        if skill_evol and hasattr(skill_evol, "_save_skill_to_disk"):
            skill_evol._save_skill_to_disk(skill_id, content)

        logger.info("Skill uploaded and registered: %s (persisted=%s)", skill_id,
                     bool(skill_evol and getattr(skill_evol, "_skills_dir", None)))
        return JSONResponse(content={
            "status":   "registered",
            "skill_id": skill_id,
            "chars":    len(content),
            "persisted": bool(skill_evol and getattr(skill_evol, "_skills_dir", None)),
        })

    @app.post("/skills/generate")
    async def generate_skill_from_text(request: Request) -> JSONResponse:
        """
        Generate a skill markdown draft from a free-form conversation snippet.

        The user pastes a chat excerpt (or any prose describing a procedure);
        the LLM converts it to the standard skill format. The draft is
        RETURNED but NOT registered — the user reviews + edits it in the UI,
        then POSTs to /skills/upload to actually register it.

        Body (JSON):
          {
            "text":       "<conversation excerpt or procedure description>",
            "hint_name":  "<optional desired skill name>",
            "hint_tags":  ["optional", "tags"]
          }

        Returns:
          {
            "skill_id":   "<auto-generated stable id>",
            "markdown":   "<draft markdown content>",
            "similar_to": "<existing_id>" | null,   // if Jaccard ≥ 0.35
            "similarity": 0.42                      // Jaccard score
          }

        The frontend can then:
          - Show the draft in an editable textarea
          - If `similar_to` is set, offer "Merge into <existing>" instead of new
          - On confirm, send the (possibly edited) markdown to /skills/upload
        """
        ident = await _identity()
        if not ident.has_role("admin"):
            raise HTTPException(
                status_code=403,
                detail="Skill generation requires the 'admin' role",
            )

        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Body must be JSON")

        text = (body or {}).get("text", "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="`text` field is required")
        if len(text) > 10_000:
            raise HTTPException(status_code=400, detail="`text` exceeds 10000 chars")

        hint_name = (body or {}).get("hint_name", "").strip()
        hint_tags = (body or {}).get("hint_tags", []) or []

        skill_evol = services.get("skill_evolver")
        catalog    = services.get("skill_catalog")
        if skill_evol is None:
            raise HTTPException(status_code=503, detail="SkillEvolver not configured")

        # Diagnostic: warn early if SkillEvolver has no LLM wired — without
        # this the response would be the static stub regardless of input.
        if getattr(skill_evol, "_llm_fn", None) is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "SkillEvolver has no LLM configured — generation would "
                    "produce hardcoded boilerplate. Check server logs for "
                    "'SkillEvolver: NO llm_engine in services' and ensure "
                    "llm_engine is registered in services."
                ),
            )

        # 1. Generate the skill markdown FIRST. Doing this before we compute
        #    the skill_id and similarity lets us derive the id from the
        #    actual generated title (semantically meaningful) instead of
        #    from arbitrary tokens in the user's raw input (which may be
        #    JSON keys, prose, or just "故障诊断").
        from skills.evolver import _SKILL_WRITE_SYSTEM
        user_content = (
            f"Source text (operator-supplied conversation/procedure):\n"
            f"-----\n{text[:6000]}\n-----\n\n"
            f"Desired skill name hint: {hint_name or '(infer from text)'}\n"
            f"Desired tags hint: {', '.join(hint_tags) if hint_tags else '(infer)'}\n\n"
            f"Convert the source text above into a standard skill markdown file. "
            f"Capture the actionable steps, identify which tools/parameters are used, "
            f"and infer a reasonable Risk and HITL level. The Tags line MUST contain "
            f"3-5 short English/lowercase keywords describing the skill domain "
            f"(e.g. [network, dns, troubleshooting]). Keep total length under 1500 chars."
        )
        try:
            raw = await skill_evol._call_llm(_SKILL_WRITE_SYSTEM, user_content)
            import re as _re_local
            markdown = _re_local.sub(r"^```(?:markdown)?\s*\n?", "", raw.strip()).rstrip("```").strip()
        except Exception as exc:
            logger.warning("/skills/generate LLM call failed: %s", exc)
            raise HTTPException(status_code=502, detail=f"LLM generation failed: {exc}")

        if not markdown or len(markdown) < 30:
            raise HTTPException(status_code=502, detail="LLM produced empty/too-short content")

        # Detect stub fallback: if the response equals the well-known stub
        # output, refuse instead of returning misleading boilerplate.
        if "Network Diagnostic Procedure" in markdown[:80] and "get_device_status" in markdown:
            # Cross-check: was the input actually about generic network diagnostic?
            text_lower = text.lower()
            looks_legitimately_about_topic = any(
                k in text_lower for k in ("network diagnostic", "diagnose network", "get_device_status")
            )
            if not looks_legitimately_about_topic:
                logger.error(
                    "/skills/generate: LLM appears to have returned the stub "
                    "fallback (Network Diagnostic Procedure) — input did NOT "
                    "request that. LLM call likely failed silently."
                )
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "LLM returned stub-fallback content (hardcoded "
                        "'Network Diagnostic Procedure'). Your input was about "
                        "something else. Check llm_engine connectivity in server logs."
                    ),
                )

        # 2. Derive a stable, meaningful skill_id from the generated markdown.
        #    Priority: hint_name → H1 title → text fallback.
        title_source = hint_name
        if not title_source:
            m = _re_local.match(r"^\s*#\s+(.+)$", markdown, flags=_re_local.MULTILINE)
            if m:
                title_source = m.group(1).strip()
        skill_id = skill_evol._generate_skill_id(title_source or text[:200])

        # 3. Run similarity check on the GENERATED skill signature (H1 + tags
        #    parsed from the markdown). This is far more accurate than running
        #    similarity on raw user input — generated skills always include
        #    standardised English tag keywords that match catalog entries.
        signature_for_sim = title_source or text[:200]
        # Augment with parsed tags from the generated markdown
        tag_match = _re_local.search(
            r"\*\*Tags:\*\*\s*\[([^\]]*)\]", markdown, flags=_re_local.IGNORECASE,
        )
        if tag_match:
            signature_for_sim += " " + tag_match.group(1)

        similar = await skill_evol._find_similar_skill(signature_for_sim)
        similar_id      = similar[0] if similar else None
        similar_score   = similar[1] if similar else 0.0
        similar_summary = None
        if similar_id and catalog:
            sm = catalog.get_summary(similar_id)
            if sm:
                similar_summary = {
                    "name":    sm.name,
                    "purpose": sm.purpose,
                    "tags":    sm.tags,
                }

        # 4. Detect explicit id collision (same skill_id already registered)
        #    so the UI can warn even when similarity is below threshold.
        id_collides = False
        if catalog and skill_id:
            try:
                id_collides = catalog.get_summary(skill_id) is not None
            except Exception:
                id_collides = False

        logger.info(
            "/skills/generate: id=%s draft_chars=%d similar_to=%s (j=%.2f) id_collides=%s",
            skill_id, len(markdown), similar_id, similar_score, id_collides,
        )
        return JSONResponse(content={
            "skill_id":         skill_id,
            "markdown":         markdown,
            "similar_to":       similar_id,
            "similarity":       round(similar_score, 3),
            "similar_summary":  similar_summary,    # name/purpose/tags of conflict, for UI
            "id_collides":      id_collides,        # exact id already exists
        })

    @app.get("/skills/{skill_id}/content")
    async def get_skill_raw_content(skill_id: str) -> JSONResponse:
        """
        Return the human-readable markdown content of a skill.
        Priority:
          1. Disk file (HERMES_DATA_DIR/skills/<id>.md) — evolved/uploaded skills
          2. catalog.as_markdown()                       — built-in skills synthesised as markdown
          3. 404 if not registered at all
        """
        skill_evol = services.get("skill_evolver")
        raw_content = None
        source = "unknown"

        # 1. Try disk file first (evolved / uploaded skills). Prefer the
        #    standard layout <kebab-name>/SKILL.md, fall back to legacy <id>.md.
        if skill_evol and getattr(skill_evol, "_skills_dir", None):
            import pathlib as _pl
            from skills.skill_format import skill_id_to_name as _to_name
            base = _pl.Path(skill_evol._skills_dir)
            std_path = base / _to_name(skill_id) / "SKILL.md"
            legacy_path = base / f"{skill_id}.md"
            if std_path.exists():
                raw_content = std_path.read_text(encoding="utf-8")
                source = "disk"
            elif legacy_path.exists():
                raw_content = legacy_path.read_text(encoding="utf-8")
                source = "disk"

        # 2. Fall back to catalog.as_markdown() — works for built-in skills too
        if raw_content is None:
            catalog = services.get("skill_catalog")
            if catalog and hasattr(catalog, "as_markdown"):
                raw_content = catalog.as_markdown(skill_id)
                if raw_content:
                    source = "catalog"

        if raw_content is None:
            raise HTTPException(status_code=404, detail=f"Skill {skill_id!r} not found")

        return JSONResponse(content={
            "skill_id": skill_id,
            "content":  raw_content,
            "source":   source,
        })