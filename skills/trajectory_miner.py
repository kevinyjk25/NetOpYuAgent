"""skills/trajectory_miner.py — P1: repeated trajectory → skill.

Closes关注点#3: "重复性的单元操作轨迹可以流程化为 skill".

The evolver used to fire on a SINGLE complex task (LLM-judged reuse_potential).
P1 changes that to: observe many real trajectories (from the journal, fed real
data by P0), cluster them by similarity (via CSI — the shared capability space,
NOT a fourth bespoke similarity), and only when the SAME kind of trajectory has
recurred >= N times do we ask the evolver to solidify it into a skill.

"Single run → maybe generate" becomes "repeated → solidify".

Decoupling: depends only on an injected journal store, a CSI instance, and an
evolver callback. No reverse imports. Pure orchestration over existing parts.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryCluster:
    members:    list[str]              # session_ids in this cluster
    rep_tools:  set[str]               # union of tools across the cluster
    size:       int
    sample_query: str = ""             # a representative query
    sample_steps: list[str] = field(default_factory=list)
    covered_by_skill: Optional[str] = None   # if an existing skill already covers it


@dataclass
class MinerConfig:
    recurrence_threshold: int   = 3      # cluster must recur >= this to solidify
    similarity_threshold: float = 0.5    # CSI trajectory-cluster link threshold
    only_successful:      bool  = True   # mine only completed runs
    skip_if_skill_loaded: bool  = True   # don't re-mine runs that already used a skill

    @classmethod
    def from_cfg(cls) -> "MinerConfig":
        try:
            from config import cfg as _c
            so = getattr(_c, "skill_orchestration", None)
            if so is None:
                return cls()
            return cls(
                recurrence_threshold=int(getattr(so, "trajectory_recurrence_threshold", 3)),
                similarity_threshold=float(getattr(so, "trajectory_similarity_threshold", 0.5)),
            )
        except Exception:
            return cls()


class TrajectoryMiner:
    """Mines the journal for recurring trajectories and solidifies them.

    journal_store: object with list_recent(limit) + extract_trajectory(session_id)
    csi:           CapabilitySemanticIndex (cluster_trajectories + nearest_skill)
    evolve_cb:     async callable(task_description, solution_steps, tools_used,
                   key_observations, complexity, session_id) -> proposal|None
    """

    def __init__(self, journal_store, csi, evolve_cb, cfg: Optional[MinerConfig] = None):
        self._journal = journal_store
        self._csi = csi
        self._evolve = evolve_cb
        self.cfg = cfg or MinerConfig.from_cfg()

    # ── gather trajectories from journal history ─────────────────────────
    def _gather(self, limit: int = 200) -> list[dict]:
        out = []
        try:
            entries = self._journal.list_recent(limit=limit)
        except Exception as exc:
            logger.warning("trajectory gather failed: %s", exc)
            return out
        for e in entries:
            if self.cfg.only_successful and e.get("outcome") not in ("completed", "success", None):
                continue
            sid = e.get("session_id")
            if not sid:
                continue
            # journal tool_calls are dicts {turn, tool_name, ...}; normalize to
            # tool-name strings (tolerate plain strings too).
            tools = []
            for tc in (e.get("tool_calls", []) or []):
                if isinstance(tc, dict):
                    nm = tc.get("tool_name") or tc.get("tool")
                    if nm:
                        tools.append(nm)
                elif isinstance(tc, str):
                    tools.append(tc)
            if not tools:
                # fall back to extracting from events
                try:
                    traj = self._journal.extract_trajectory(sid)
                    tools = traj.get("tools", [])
                except Exception:
                    tools = []
            if not tools:
                continue
            out.append({
                "id": sid,
                "tools": tools,
                "query": e.get("query", ""),
                "loaded_skills": list(e.get("loaded_skills", []) or []),
            })
        return out

    # ── find recurring clusters ──────────────────────────────────────────
    def find_recurring(self, limit: int = 200) -> list[TrajectoryCluster]:
        trajs = self._gather(limit=limit)
        if not trajs:
            return []
        raw = self._csi.cluster_trajectories(
            trajs, threshold=self.cfg.similarity_threshold)
        # map session_id -> traj for sample enrichment
        by_id = {t["id"]: t for t in trajs}
        clusters: list[TrajectoryCluster] = []
        for c in raw:
            if c["size"] < self.cfg.recurrence_threshold:
                continue
            members = c["members"]
            sample = by_id.get(members[0], {})
            # does an existing skill already cover this trajectory?
            covered = None
            try:
                hit = self._csi.nearest_skill(c["rep_tools"], sample.get("query", ""))
                if hit and hit[1].score >= 0.6:
                    covered = hit[0]
            except Exception:
                pass
            clusters.append(TrajectoryCluster(
                members=members, rep_tools=set(c["rep_tools"]), size=c["size"],
                sample_query=sample.get("query", ""),
                covered_by_skill=covered,
            ))
        return clusters

    # ── solidify a recurring cluster into a skill ────────────────────────
    async def solidify(self, cluster: TrajectoryCluster) -> Optional[Any]:
        """Ask the evolver to generate a skill from a recurring trajectory.
        Skips clusters already covered by an existing skill (that's P3's job —
        merge an append, not mint a duplicate)."""
        if cluster.covered_by_skill:
            logger.info(
                "P1: cluster (size=%d) already covered by skill '%s' — skipping "
                "solidify (P3 territory)", cluster.size, cluster.covered_by_skill)
            return None
        # Reconstruct a real trajectory for the representative member.
        rep_sid = cluster.members[0]
        try:
            traj = self._journal.extract_trajectory(rep_sid)
        except Exception:
            traj = {"steps": [], "tools": list(cluster.rep_tools), "observations": []}
        steps = traj.get("steps", [])
        # complexity scaled by recurrence + trajectory length (recurring +
        # multi-step = worth solidifying).
        complexity = 5.0 + min(cluster.size, 5) * 0.5 + min(len(steps), 8) * 0.3
        try:
            proposal = await self._evolve(
                task_description=cluster.sample_query or "recurring operation",
                solution_steps=steps,
                tools_used=list(cluster.rep_tools),
                key_observations=[
                    f"该操作轨迹在历史中重复出现 {cluster.size} 次，已达流程化阈值",
                    *traj.get("observations", []),
                ],
                complexity=complexity,
                session_id=rep_sid,
            )
            if proposal:
                logger.info(
                    "P1: solidified recurring trajectory (size=%d, tools=%s) into "
                    "skill proposal", cluster.size, sorted(cluster.rep_tools)[:5])
            return proposal
        except Exception as exc:
            logger.warning("P1 solidify failed: %s", exc)
            return None

    # ── one-shot sweep: find + solidify all eligible ─────────────────────
    async def sweep(self, limit: int = 200) -> list[Any]:
        proposals = []
        for cluster in self.find_recurring(limit=limit):
            p = await self.solidify(cluster)
            if p:
                proposals.append(p)
        return proposals
