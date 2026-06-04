"""skills/capability_index.py — Capability Semantic Index (CSI v1).

Explicit, interpretable capability space for tools (atomic) and skills
(composite). One unified similarity / attribution used by P1 (trajectory
clustering), P3 (targeted merge), and skill selection — replacing the four
inconsistent similarities (retriever embedding, evolver jaccard, preference
embedding, ad-hoc).

Design (see CAPABILITY_SEMANTIC_INDEX_DESIGN.md):
  - Absorbs MoE's cluster + top-k routing SHAPE, but attribution is readable,
    adjustable, auditable — NOT a learned black-box gating.
  - Every attribution returns (target, score, reasons[]).

Three evaluation-driven corrections baked in:
  1. Two-level clusters: (primary_domain, secondary_tag) so DC splits into
     dc/fabric and dc/application instead of one coarse dc bucket.
  2. Hybrid similarity (embedding + tag + tool_set + action) — never tag-only,
     which is brittle for CN/cross-language/cross-domain skills.
  3. tool_set is layered ground-truth-first: declared allowed-tools, then real
     execution trajectory (P0), then tag inference as fallback.

Decoupling: this module only READS tool/skill metadata + an OPTIONAL injected
embed function. It does not import evolver/loop/preference. Callers depend on
it one-way.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ── similarity primitives ────────────────────────────────────────────────
def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 0.0
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class SimResult:
    score: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class CapabilityVector:
    cap_id: str
    kind: str                      # "tool" | "skill"
    tags: set[str] = field(default_factory=set)
    action_type: str = "read_only"   # tool: read_only|reversible|destructive ; skill: risk
    domain: str = "other"
    secondary: str = ""            # secondary domain key (fabric/application/...)
    tool_set: set[str] = field(default_factory=set)   # ground-truth-first
    embedding: list[float] = field(default_factory=list)
    text: str = ""


@dataclass
class Cluster:
    name: str                      # "lan/access", "dc/fabric", ...
    members: list[str] = field(default_factory=list)
    dominant_tags: set[str] = field(default_factory=set)


@dataclass
class RouteHit:
    target: str                    # cap_id or cluster name
    score: float
    reasons: list[str] = field(default_factory=list)


# ── domain inference (two-level) ─────────────────────────────────────────
_PRIMARY_RULES = [
    ("dc",       lambda n, t: n.startswith("dc_") or "dc" in t),
    ("access",   lambda n, t: t & {"user", "access", "nac", "auth", "identity", "dot1x"}),
    ("config",   lambda n, t: t & {"config", "deploy"}),
    ("service",  lambda n, t: t & {"services", "health"}),
    ("observ",   lambda n, t: t & {"monitoring", "metrics", "alerts", "logs", "diagnostics"}),
    ("traffic",  lambda n, t: t & {"traffic", "security"}),
    ("inventory", lambda n, t: t & {"inventory", "discovery", "hardware"}),
]
# DC secondary split (correction #1)
_DC_FABRIC = {"fabric", "evpn", "vxlan", "bgp", "vni", "path", "overlay", "route", "control-plane"}
_DC_APP = {"application", "app", "acl", "rbac", "access"}


def _infer_domain(name: str, tags: set[str]) -> tuple[str, str]:
    primary = "other"
    for dom, pred in _PRIMARY_RULES:
        if pred(name, tags):
            primary = dom
            break
    secondary = ""
    if primary == "dc":
        if tags & _DC_APP:
            secondary = "application"
        elif tags & _DC_FABRIC:
            secondary = "fabric"
        else:
            secondary = "core"
    return primary, secondary


class CapabilitySemanticIndex:
    """Build once at startup; query for similarity / routing / attribution.

    embed_fn: optional ``(text) -> list[float]``. If None, similarity falls
    back to tag+tool_set+action only (still useful, just no semantic axis).
    Inject the existing embedder to enable the embedding axis.
    """

    def __init__(self, embed_fn: Optional[Callable[[str], list[float]]] = None,
                 weights: Optional[dict] = None):
        self._embed = embed_fn
        # correction #2: hybrid weights; tool_set raised (best skill→tool signal)
        self.w = weights or {"emb": 0.45, "tag": 0.25, "tool": 0.25, "act": 0.05}
        self._caps: dict[str, CapabilityVector] = {}
        self._clusters: dict[str, Cluster] = {}

    # ── build ────────────────────────────────────────────────────────────
    def build(self, tool_defs: dict[str, dict], skill_defs: dict[str, dict]) -> None:
        self._caps.clear()
        self._clusters.clear()
        for name, m in (tool_defs or {}).items():
            tags = set(m.get("tags", []))
            dom, sec = _infer_domain(name, tags)
            self._caps[name] = CapabilityVector(
                cap_id=name, kind="tool", tags=tags,
                action_type=m.get("action_type", "read_only"),
                domain=dom, secondary=sec,
                text=f"{name} {m.get('description','')} {' '.join(tags)}",
            )
        for sid, d in (skill_defs or {}).items():
            if sid == "read_stored_result":
                continue
            tags = set(d.get("tags", []))
            dom, sec = _infer_domain(sid, tags)
            # correction #3: tool_set ground-truth-first
            tset = set(d.get("allowed_tools", []) or [])
            if not tset:
                tset = set(d.get("tool_deps", []) or [])
            self._caps[sid] = CapabilityVector(
                cap_id=sid, kind="skill", tags=tags,
                action_type=d.get("risk_level", "low"),
                domain=dom, secondary=sec, tool_set=tset,
                text=f"{sid} {d.get('purpose','')} {d.get('description','')[:200]} {' '.join(tags)}",
            )
        # embeddings (optional)
        if self._embed is not None:
            for cap in self._caps.values():
                try:
                    cap.embedding = self._embed(cap.text) or []
                except Exception:
                    cap.embedding = []
        # two-level clustering
        for cap in self._caps.values():
            key = cap.domain + (f"/{cap.secondary}" if cap.secondary else "")
            cl = self._clusters.setdefault(key, Cluster(name=key))
            cl.members.append(cap.cap_id)
            cl.dominant_tags |= cap.tags
        logger.info("CSI: built %d capabilities in %d clusters (%s)",
                    len(self._caps), len(self._clusters), ", ".join(sorted(self._clusters)))

    # ── similarity (interpretable hybrid) ────────────────────────────────
    def _sim(self, a: CapabilityVector, b: CapabilityVector) -> SimResult:
        reasons = []
        emb = cosine(a.embedding, b.embedding) if (a.embedding and b.embedding) else 0.0
        tag = jaccard(a.tags, b.tags)
        act = 1.0 if a.action_type == b.action_type else 0.0
        if a.tool_set and b.tool_set:
            tool = jaccard(a.tool_set, b.tool_set)
        else:
            tool = None
        w = dict(self.w)
        if tool is None:               # redistribute tool weight when N/A
            _tw = w["tool"]
            w["tool"] = 0.0
            w["emb"] += _tw * 0.6
            w["tag"] += _tw * 0.4
            tool = 0.0
        score = w["emb"] * emb + w["tag"] * tag + w["tool"] * tool + w["act"] * act
        reasons = [f"emb={emb:.2f}", f"tag={tag:.2f}", f"tool_set={tool:.2f}", f"act={act:.0f}"]
        return SimResult(round(score, 4), reasons)

    def similarity(self, a_id: str, b_id: str) -> Optional[SimResult]:
        a, b = self._caps.get(a_id), self._caps.get(b_id)
        if a is None or b is None:
            return None
        return self._sim(a, b)

    # ── routing: text / tool_set → top-k capabilities ────────────────────
    def route(self, *, text: Optional[str] = None, tool_set=None,
              kind: Optional[str] = None, top_k: int = 3) -> list[RouteHit]:
        probe = CapabilityVector(
            cap_id="__probe__", kind="probe",
            tool_set=set(tool_set or []),
            embedding=(self._embed(text) if (self._embed and text) else []),
            text=text or "",
        )
        hits = []
        for cap in self._caps.values():
            if kind and cap.kind != kind:
                continue
            r = self._sim(probe, cap)
            if r.score > 0:
                hits.append(RouteHit(cap.cap_id, r.score, r.reasons))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    # ── P3: which existing skill does a trajectory belong to ─────────────
    def nearest_skill(self, tool_set, text: str = "") -> Optional[tuple[str, SimResult]]:
        hits = self.route(text=text, tool_set=tool_set, kind="skill", top_k=1)
        if not hits:
            return None
        h = hits[0]
        return h.target, SimResult(h.score, h.reasons)

    # ── P1: cluster a batch of trajectories by similarity ────────────────
    def cluster_trajectories(self, trajectories: list[dict], *,
                             threshold: float = 0.5) -> list[dict]:
        """Greedy single-link clustering of trajectories.
        Each trajectory: {id, tools:[...], text:str}. Returns list of
        {members:[id], rep_tools:set, size:int}."""
        clusters: list[dict] = []
        for tr in trajectories:
            tset = set(tr.get("tools", []))
            txt = tr.get("text", "")
            placed = False
            for cl in clusters:
                tool_sim = jaccard(tset, cl["rep_tools"])
                if tool_sim >= threshold:
                    cl["members"].append(tr.get("id"))
                    cl["rep_tools"] |= tset
                    cl["size"] += 1
                    placed = True
                    break
            if not placed:
                clusters.append({"members": [tr.get("id")], "rep_tools": set(tset), "size": 1})
        return clusters

    # ── diagnostics ──────────────────────────────────────────────────────
    def export_space(self) -> dict:
        return {
            "clusters": {
                k: {"members": cl.members, "dominant_tags": sorted(cl.dominant_tags)}
                for k, cl in sorted(self._clusters.items())
            },
            "capabilities": {
                cid: {"kind": c.kind, "domain": c.domain, "secondary": c.secondary,
                      "tool_set": sorted(c.tool_set)}
                for cid, c in self._caps.items()
            },
        }
