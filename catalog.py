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

DEFAULT_SKILL_DEFINITIONS: dict[str, dict[str, Any]] = {
    "syslog_search": {
        "name":           "Syslog Search",
        "purpose":        "Search syslog entries across network devices",
        "risk_level":     "low",
        "requires_hitl":  False,
        "tags":           ["logs", "diagnostics"],
        "description":    (
            "Queries the centralised syslog aggregator for matching entries. "
            "Returns structured log lines with timestamp, host, process, severity, and message. "
            "Supports glob patterns for host filtering."
        ),
        "parameters":     {
            "host":    "Device name or glob (e.g. 'ap-*', 'sw-core-01')",
            "keyword": "Text to search for in the log message",
            "lines":   "Maximum number of lines to return (default: 300)",
        },
        "returns":        "Newline-separated log entries, one per line",
        "estimated_size": "large",
        "returns_large":  True,
        "examples":       [{"args": {"host": "ap-01", "keyword": "error", "lines": 100},
                            "note": "Returns recent error logs from ap-01"}],
        "constraints":    ["Rate-limited to 10 queries/minute per session"],
    },
    "prometheus_query": {
        "name":           "Prometheus Query",
        "purpose":        "Query Prometheus metrics for network devices",
        "risk_level":     "low",
        "requires_hitl":  False,
        "tags":           ["metrics", "monitoring"],
        "description":    (
            "Executes a Prometheus range query and returns time-series data in JSON format. "
            "Useful for trend analysis, anomaly detection, and SLA verification."
        ),
        "parameters":     {
            "metric":         "Prometheus metric name (e.g. 'up', 'interface_errors_total')",
            "job":            "Job label filter (e.g. 'network_devices')",
            "range_minutes":  "Look-back window in minutes (default: 60)",
        },
        "returns":        "JSON object with status, data.result array, and metadata",
        "estimated_size": "large",
        "returns_large":  True,
        "examples":       [{"args": {"metric": "up", "job": "network_devices", "range_minutes": 30},
                            "note": "Returns 30-minute uptime time series for all devices"}],
        "constraints":    [],
    },
    "netflow_dump": {
        "name":           "NetFlow Dump",
        "purpose":        "Dump NetFlow/IPFIX flow records for a site",
        "risk_level":     "low",
        "requires_hitl":  False,
        "tags":           ["network", "flows", "diagnostics"],
        "description":    (
            "Exports raw NetFlow records from the collector for a given site. "
            "Very large output — always stored externally. "
            "Use read_stored_result to drill into specific flows."
        ),
        "parameters":     {
            "site":  "Site name (e.g. 'site-a')",
            "flows": "Number of flow records to return (default: 500)",
        },
        "returns":        "Tab-separated flow records with src/dst/proto/bytes/packets",
        "estimated_size": "large",
        "returns_large":  True,
        "examples":       [{"args": {"site": "site-a", "flows": 100},
                            "note": "Returns 100 recent flows from site-a"}],
        "constraints":    ["Maximum 1000 flows per query"],
    },
    "dns_lookup": {
        "name":           "DNS Lookup",
        "purpose":        "Resolve a hostname and return DNS records",
        "risk_level":     "low",
        "requires_hitl":  False,
        "tags":           ["dns", "diagnostics"],
        "description":    "Performs a forward DNS lookup and returns A, AAAA, NS records and TTLs.",
        "parameters":     {"hostname": "Fully qualified domain name"},
        "returns":        "DNS records with TTL values and resolver metadata",
        "estimated_size": "small",
        "returns_large":  False,
        "examples":       [{"args": {"hostname": "payments.internal"}, "note": "Resolve internal hostname"}],
        "constraints":    [],
    },
    "device_info": {
        "name":           "Device Info",
        "purpose":        "Get hardware and firmware details for a network device",
        "risk_level":     "low",
        "requires_hitl":  False,
        "tags":           ["inventory", "diagnostics"],
        "description":    "Returns model, firmware version, uptime, client counts, and channel info.",
        "parameters":     {"device_id": "Device identifier (e.g. 'ap-01', 'sw-core-01')"},
        "returns":        "Structured device attributes as key-value pairs",
        "estimated_size": "small",
        "returns_large":  False,
        "examples":       [{"args": {"device_id": "ap-01"}, "note": "Get details for access point ap-01"}],
        "constraints":    [],
    },
    "alert_summary": {
        "name":           "Alert Summary",
        "purpose":        "Return current alert counts by severity",
        "risk_level":     "low",
        "requires_hitl":  False,
        "tags":           ["alerts", "monitoring"],
        "description":    "Fetches active alert counts from PagerDuty/OpsGenie grouped by severity.",
        "parameters":     {"severity": "Filter: all | P0 | P1 | P2 | P3 (default: all)"},
        "returns":        "Alert counts with recent incident IDs and MTTR",
        "estimated_size": "small",
        "returns_large":  False,
        "examples":       [{"args": {"severity": "P1"}, "note": "Get all P1 alerts"}],
        "constraints":    [],
    },
    "service_health": {
        "name":           "Service Health",
        "purpose":        "Check health status of a backend service",
        "risk_level":     "low",
        "requires_hitl":  False,
        "tags":           ["health", "monitoring"],
        "description":    "Calls the /health endpoint of a service and returns status, uptime, error rate.",
        "parameters":     {"service": "Service name (e.g. 'payments-service', 'auth-service')"},
        "returns":        "Health status with uptime, error rate, and last deployment time",
        "estimated_size": "small",
        "returns_large":  False,
        "examples":       [{"args": {"service": "payments-service"}, "note": "Check payments health"}],
        "constraints":    [],
    },
    "restart_service": {
        "name":           "Restart Service",
        "purpose":        "Trigger a rolling restart of a production service",
        "risk_level":     "high",
        "requires_hitl":  True,
        "tags":           ["ops", "destructive"],
        "description":    (
            "Issues a rolling restart to the specified service in the target environment. "
            "Always requires HITL approval. Post-action health verification is automatic."
        ),
        "parameters":     {
            "service":     "Service name",
            "environment": "Target environment: prod | staging | dev",
            "rolling":     "Whether to use rolling restart (default: true)",
        },
        "returns":        "Restart status with pod counts and rollout progress",
        "estimated_size": "small",
        "returns_large":  False,
        "examples":       [{"args": {"service": "payments-service", "environment": "prod"},
                            "note": "Rolling restart payments in prod — HITL required"}],
        "constraints":    [
            "Requires HITL approval before execution",
            "Only available in maintenance window unless P0/P1 incident",
            "Post-action health check is mandatory",
        ],
    },
    "read_stored_result": {
        "name":           "Read Stored Result",
        "purpose":        "Page through a large tool result by reference ID",
        "risk_level":     "low",
        "requires_hitl":  False,
        "tags":           ["cache", "retrieval"],
        "description":    (
            "Retrieves a slice of a previously cached tool result. "
            "Use this when a tool returned [STORED:tool:ref_id] to access the full data."
        ),
        "parameters":     {
            "ref_id":  "The reference ID from the [STORED:...] label",
            "offset":  "Byte offset to start reading from (default: 0)",
            "length":  "Number of bytes to read (default: 2000)",
        },
        "returns":        "A slice of the stored text with metadata showing total size and next offset",
        "estimated_size": "small",
        "returns_large":  False,
        "examples":       [{"args": {"ref_id": "a3f9c12b", "offset": 0, "length": 2000},
                            "note": "Read first 2KB of cached syslog result"}],
        "constraints":    [],
    },
}


# ---------------------------------------------------------------------------
# Query-matched skill selection (Q4)
# ---------------------------------------------------------------------------