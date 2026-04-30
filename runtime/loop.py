"""
runtime/loop.py  [v2 — P1/P2 features integrated]
---------------------------------------------------
P1 additions:
  - Skill progressive disclosure via SkillCatalogService
  - Forked delegation: delegation_mode = fresh | forked
  - Working Set and Confirmed Facts as first-class LoopState fields
  - Pre/Post verification hooks (pre_verify, post_verify)

P2 additions:
  - Prompt-cache-friendly context assembly (stable prefix first)
  - Model tiering hint (classify as fast_model vs full_model)
  - Lightweight verification step after tool execution

Backward compatible: all new parameters are Optional.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Optional, TYPE_CHECKING

from .context_budget import BudgetConfig, ContextBudgetManager, DeviceRef, ToolResultStore
from .stop_policy import LoopState, StopDecision, StopOutcome, StopPolicy, StopPolicyConfig

if TYPE_CHECKING:
    from memory.adapter import MemoryAdapter as MemoryRouter
    from skills.catalog import SkillCatalogService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class QueryComplexity(str, Enum):
    SIMPLE  = "simple"
    COMPLEX = "complex"


class DelegationMode(str, Enum):
    """
    fresh  — start a sub-agent with only the explicitly passed context
    forked — inherit parent confirmed_facts + working_set (P1)
    """
    FRESH  = "fresh"
    FORKED = "forked"


class ForkContextPolicy(str, Enum):
    """How much parent context a forked sub-agent inherits."""
    FULL         = "full"           # everything
    FACTS_ONLY   = "facts_only"     # only confirmed_facts
    WORKING_SET  = "working_set"    # facts + working_set, not raw memory


# ---------------------------------------------------------------------------
# Verification result (P1)
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    passed:   bool
    reason:   str
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def ok(cls, reason: str = "Verification passed") -> "VerificationResult":
        return cls(passed=True, reason=reason)

    @classmethod
    def fail(cls, reason: str, warnings: list[str] = None) -> "VerificationResult":
        return cls(passed=False, reason=reason, warnings=warnings or [])


# ---------------------------------------------------------------------------
# Complexity decision
# ---------------------------------------------------------------------------

@dataclass
class ComplexityDecision:
    complexity: QueryComplexity
    reason:     str
    confidence: float = 1.0
    model_tier: str   = "full_model"   # P2: fast_model | full_model


_DESTRUCTIVE_KEYWORDS = frozenset({
    # English
    "restart", "rollback", "delete", "drain", "failover", "flush",
    "reboot", "terminate", "shutdown", "wipe", "reset",
    # Chinese equivalents (substring match is fine for CJK — no word boundaries needed)
    "重启", "回滚", "删除", "终止", "关机", "重置", "下发配置", "推送配置",
})
_P0P1_KEYWORDS = frozenset({
    "p0", "p1", "critical", "outage", "down", "emergency",
    "sev0", "sev1", "major incident",
})
_PARALLEL_KEYWORDS = frozenset({
    "all sites", "all devices", "across regions", "compare", "correlate",
    "multiple", "batch", "bulk", "foreach",
})
_FAST_MODEL_KEYWORDS = frozenset({
    "dns", "ping", "status", "check", "what is", "show me", "list",
})


# ---------------------------------------------------------------------------
# Runtime config
# ---------------------------------------------------------------------------

@dataclass
class RuntimeConfig:
    budget:      BudgetConfig      = field(default_factory=BudgetConfig)
    stop_policy: StopPolicyConfig  = field(default_factory=StopPolicyConfig)

    # Complexity thresholds
    simple_confidence_floor:  float = 0.70
    simple_max_tool_calls:    int   = 4

    # P1: delegation
    default_delegation_mode:   DelegationMode   = DelegationMode.FRESH
    default_fork_context:      ForkContextPolicy = ForkContextPolicy.FACTS_ONLY

    # P1: verification
    enable_pre_verification:   bool = True
    enable_post_verification:  bool = True

    # P2: model tiering
    enable_model_tiering:      bool = False   # set True when real LLM is wired

    # Tool result inline limit
    tool_result_inline_limit:  int  = 4_000

    # CAP 5: tools that force HITL before execution even on SIMPLE path
    # Populated from HITL_TOOL_NAMES env var in main.py
    hitl_tool_names: frozenset = field(default_factory=frozenset)

    # ── Type #2 EDIT-flavoured HITL ────────────────────────────────────
    # Tools where the operator should be allowed to edit the proposed
    # parameters before approving (e.g. edit the config_lines list).
    # Subset of hitl_tool_names — for any tool in BOTH sets, the executor
    # raises trigger_edit_approval (with editable_param_keys) instead of
    # the bare approve/reject panel.
    editable_hitl_tools: dict[str, list[str]] = field(default_factory=lambda: {
        "edit_device_config":  ["config_lines", "reason"],
        "rollback_deploy":     ["snapshot_id", "reason"],
        "restart_service":     ["service_name", "graceful"],
    })

    # ── Type #3 CLARIFICATION ───────────────────────────────────────────
    # Auto-clarify when the agent's confidence in its plan is below this
    # threshold AND the operator hasn't exceeded their per-session budget.
    # 0 disables auto-clarify entirely.
    clarification_confidence_floor: float = 0.45
    clarification_max_per_session:  int   = 2


# ---------------------------------------------------------------------------
# Loop result
# ---------------------------------------------------------------------------

@dataclass
class LoopResult:
    outcome:          StopOutcome
    final_response:   str
    confirmed_facts:  list[str]   = field(default_factory=list)
    working_set:      list[DeviceRef] = field(default_factory=list)
    unresolved:       list[str]   = field(default_factory=list)
    tool_summaries:   list[str]   = field(default_factory=list)
    turns_taken:      int         = 0
    escalated_to_dag: bool        = False
    verification:     Optional[VerificationResult] = None


# ---------------------------------------------------------------------------
# AgentRuntimeLoop
# ---------------------------------------------------------------------------


def _heuristic_missing_fields(query: str) -> list[dict]:
    """Best-effort extraction of likely missing parameters from a user query.

    Used by the clarification gate when the caller didn't supply explicit
    `_clarification_fields`. Pattern matches on common IT-ops verbs:

      - "查日志 / show logs / syslog"   → device_id, time_range, severity
      - "查配置 / get config"            → device_id
      - "修复 / fix / repair / 修改"      → device_id, what_to_change
      - "重启 / restart / reboot"        → device_id, scope
      - generic short query              → device_id alone

    Returns a list of {key, prompt, placeholder?, required?} dicts.
    Empty list when query is specific enough that we'd just be asking
    redundant questions.
    """
    q = (query or "").lower()
    fields: list[dict] = []

    has_device_id = bool(__import__("re").search(
        # Matches ap-01, sw-3, router1, sw-core-01, sw-acc-02, ap-floor3 etc.
        r"(?<![a-z0-9])(ap|sw|router|switch)[-_]?[a-z0-9]*[-_]?\d+(?![a-z0-9])", q
    ))

    asks_logs = any(kw in q for kw in (
        "syslog", "log", "日志", "事件", "show logs",
    ))
    asks_config = any(kw in q for kw in (
        "config", "running-config", "配置",
    )) and not asks_logs
    asks_change = any(kw in q for kw in (
        "修复", "fix", "repair", "修改", "改", "调整",
        "restart", "reboot", "重启", "重置", "rollback",
    ))

    if asks_logs:
        if not has_device_id:
            fields.append({"key": "device_id",
                           "prompt": "要查哪台设备的日志？",
                           "placeholder": "ap-01 / sw-core-01 / router-01",
                           "required": True})
        fields.append({"key": "time_range",
                       "prompt": "时间范围？",
                       "placeholder": "last 1h / last 24h / 2026-04-29",
                       "required": False})
        fields.append({"key": "severity",
                       "prompt": "日志级别？",
                       "placeholder": "error / warn / info",
                       "required": False})
        return fields

    if asks_change and not has_device_id:
        fields.append({"key": "device_id",
                       "prompt": "要操作哪台设备？",
                       "placeholder": "ap-01 / sw-core-01",
                       "required": True})
        fields.append({"key": "what_to_change",
                       "prompt": "具体要修改什么配置项？",
                       "placeholder": "RADIUS timeout / NTP server / VLAN ACL",
                       "required": True})
        return fields

    if asks_config and not has_device_id:
        fields.append({"key": "device_id",
                       "prompt": "要查哪台设备的配置？",
                       "placeholder": "ap-01 / sw-core-01",
                       "required": True})
        return fields

    # Generic ambiguous: ask for a target device only if the query is short
    # and devicey-sounding. Avoid over-asking for casual queries.
    # Note: use regex word boundary so "ap" doesn't match "approval".
    _has_devicey_term = bool(__import__("re").search(
        r"(?<![a-z])(ap|switch|device|router|sw)(?![a-z])|设备",
        q,
    ))
    if len(q) < 20 and not has_device_id and _has_devicey_term:
        fields.append({"key": "device_id",
                       "prompt": "请问您指的是哪台设备？",
                       "placeholder": "ap-01 / sw-core-01 / router-01",
                       "required": True})

    return fields


def _call_key(tool_name: str, tool_args: dict) -> str:
    """
    Deduplicate tool calls by name+args fingerprint, not just name.
    This allows calling the same tool with different args (e.g. different
    device_ids) in one session without the second being blocked as a duplicate.
    Only blocks genuinely identical calls (same tool, same arguments).
    """
    import json as _json
    try:
        args_sig = _json.dumps(tool_args, sort_keys=True)
    except Exception:
        args_sig = str(tool_args)
    return f"{tool_name}|{args_sig}"


def _build_tool_ledger(
    tool_outputs: dict,
    tool_reg: dict,
    raw_outputs: dict,
) -> list[str]:
    """
    Build confirmed_facts ledger entries from the current session's tool_outputs.
    Written at end of stream() so next HTTP request can seed called_tools and
    surface existing ref_ids without re-fetching.

    Collapsing rules:
    - Multiple read_stored_result pages for the same ref_id → single summary entry
    - [STORED:] labels annotated with total_size from their read results
    - Inline results recorded with size
    """
    import json as _j, re as _re

    # Pass 1: collect total_size and page count per ref_id from read_stored_result results
    ref_info: dict = {}   # ref_id → {total_size, pages, last_offset}
    for key, val in raw_outputs.items():
        tool = key.split("|")[0] if "|" in key else key
        if tool != "read_stored_result" or "_summary" in key:
            continue
        try:
            args   = _j.loads(key.split("|", 1)[1]) if "|" in key else {}
            ref    = args.get("ref_id", "").strip("[]")
            ref    = ref.rsplit(":", 1)[-1].strip() if ":" in ref else ref
            offset = int(args.get("offset", 0))
            if not ref:
                continue
            total_m = _re.search(r"Total size:\s*([\d,]+)", val)
            total   = int(total_m.group(1).replace(",", "")) if total_m else 0
            if ref not in ref_info:
                ref_info[ref] = {"total_size": total, "last_offset": offset, "pages": 0}
            else:
                if offset > ref_info[ref]["last_offset"]:
                    ref_info[ref]["last_offset"] = offset
                    ref_info[ref]["total_size"]   = total
            ref_info[ref]["pages"] += 1
        except Exception:
            pass

    ledger: list[str] = []
    seen:   set[str]  = set()
    seen_read_refs:  set[str] = set()   # track which refs already have a read entry

    for key, stored in tool_outputs.items():
        if key in seen or "_summary" in key:
            continue
        seen.add(key)
        tool_name = key.split("|")[0] if "|" in key else key

        # Collapse all read_stored_result pages for same ref_id into ONE entry
        if tool_name == "read_stored_result":
            try:
                args   = _j.loads(key.split("|", 1)[1]) if "|" in key else {}
                ref    = args.get("ref_id", "").strip("[]")
                ref    = ref.rsplit(":", 1)[-1].strip() if ":" in ref else ref
            except Exception:
                ref = ""
            if ref and ref in seen_read_refs:
                continue   # already emitted a summary for this ref_id
            if ref:
                seen_read_refs.add(ref)
                info    = ref_info.get(ref, {})
                pages   = info.get("pages", 1)
                total   = info.get("total_size", 0)
                covered = info.get("last_offset", 0) + 2000
                ledger.append(
                    f"TOOL_EXEC: read_stored_result|ref={ref} pages_read={pages} "
                    f"bytes_covered=0-{min(covered, total)} total={total}"
                )
            continue

        raw  = raw_outputs.get(key, stored)
        ref_m = _re.search(r"\[STORED:\w+:(\w+)\]", stored)
        if ref_m:
            ref_id = ref_m.group(1)
            total  = ref_info.get(ref_id, {}).get("total_size", len(raw))
            ledger.append(
                f"TOOL_EXEC: {key} → ref={ref_id} total_size={total} "
                f"[reuse: read_stored_result ref_id={ref_id}]"
            )
        else:
            ledger.append(f"TOOL_EXEC: {key} → inline size={len(raw)}")

    return ledger


class AgentRuntimeLoop:
    """
    Thin default execution path.

    v2 additions (P1/P2):
      - SkillCatalogService for progressive skill disclosure
      - DelegationMode: fresh | forked (context inheritance)
      - Confirmed facts and working set as first-class state
      - Pre/post verification hooks
      - P2: prompt-cache-friendly ordering, model tier hint
    """

    def __init__(
        self,
        memory_router:   Optional["MemoryRouter"] = None,
        config:          Optional[RuntimeConfig]  = None,
        tool_store:      Optional[ToolResultStore] = None,
        skill_catalog:   Optional["SkillCatalogService"] = None,
    ) -> None:
        self._memory       = memory_router
        self._cfg          = config or RuntimeConfig()
        self._store        = tool_store or ToolResultStore()
        self._budget       = ContextBudgetManager(self._cfg.budget, self._store)
        self._policy       = StopPolicy(self._cfg.stop_policy)
        self._skill_catalog = skill_catalog
        # Per-session clarification counter so the agent can't loop on
        # "ask user → low confidence → ask again". Capped by config.
        self._clarification_counts: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Clarification gate — Type #3 multi-mode HITL
    # ------------------------------------------------------------------

    # Action verbs that indicate the user wants a write/destructive op.
    # If one of these appears in the query but no concrete target does,
    # the agent is about to hallucinate — better to ask.
    _ACTION_VERBS = frozenset({
        # English
        "fix", "repair", "modify", "edit", "change", "update", "set",
        "restart", "reboot", "rollback", "deploy", "delete", "remove",
        # Chinese
        "修复", "修改", "更改", "下发", "推送", "重启", "回滚", "删除",
        "调整", "优化", "配置",
    })

    # Read/diagnostic verbs — don't NEED a target (could be "查所有设备")
    # but DO need a scope to avoid scanning everything.
    _READ_VERBS = frozenset({
        "查", "查询", "看", "检查", "诊断", "show", "list", "check",
        "look", "find", "search", "view", "diagnose",
    })

    @staticmethod
    def _query_has_action_verb(q_lower: str) -> bool:
        return any(v in q_lower for v in AgentRuntimeLoop._ACTION_VERBS)

    @staticmethod
    def _query_has_read_verb(q_lower: str) -> bool:
        return any(v in q_lower for v in AgentRuntimeLoop._READ_VERBS)

    @staticmethod
    def _query_mentions_concrete_target(q: str) -> bool:
        """Heuristic: does the query name a specific device/service?
        Looks for tokens like ap-NN, sw-core-NN, router-NN, IPs, hostnames.
        """
        import re as _re
        # Common device-id patterns: ap-01, sw-core-01, router-02, radius-01
        if _re.search(r"\b[a-z]{2,}[-_]\w{2,}\b", q.lower()):
            return True
        # IPv4
        if _re.search(r"\b\d+\.\d+\.\d+\.\d+\b", q):
            return True
        return False

    def _maybe_clarification_fields(
        self,
        *,
        query: str,
        top_skill_score: float,
        asked_count: int,
        recent_context: str = "",
    ) -> list[dict]:
        """Return a list of clarification fields if the query is too vague
        to act on safely; empty list otherwise.

        Heuristics (cheap, no LLM call):
          0. Prior turn already asked clarification → treat current query
             as the answer, don't re-ask
          1. Already asked at-budget → don't ask again, let agent guess
          2. Top skill score >= floor → confident enough, proceed
          3. Action verb + no concrete target → ask which device
          4. Read verb + no scope → ask which device & time range
          5. Otherwise: proceed (no clarification needed)
        """
        # Prior-turn check — if recent recall context contains the agent's
        # own clarification preamble, this query IS the answer the
        # operator typed in response. Re-asking would loop forever.
        if recent_context and (
            "为了准确处理这个请求，我需要您补充" in recent_context
            or "Clarification asked via chat turn" in recent_context
        ):
            return []

        if asked_count >= self._cfg.clarification_max_per_session:
            return []
        if top_skill_score >= self._cfg.clarification_confidence_floor:
            return []

        q_lower = query.lower().strip()
        has_target = self._query_mentions_concrete_target(query)
        has_action = self._query_has_action_verb(q_lower)
        has_read   = self._query_has_read_verb(q_lower)

        # Case A: action verb + no concrete target → ask for device
        if has_action and not has_target:
            return [{
                "key":         "device_id",
                "prompt":      "Which device should this action target?",
                "placeholder": "e.g. ap-01, sw-core-01",
                "required":    True,
            }]

        # Case B: read/diagnostic + no concrete target — ask scope
        if has_read and not has_target and len(q_lower) < 30:
            return [
                {
                    "key":         "device_id",
                    "prompt":      "Which device(s) do you want to query?",
                    "placeholder": "e.g. ap-01, or 'all aps'",
                    "required":    True,
                },
                {
                    "key":         "time_range",
                    "prompt":      "Time range (only for log/event queries)?",
                    "placeholder": "e.g. last 1h, last 24h",
                    "required":    False,
                },
            ]

        # No clarification needed
        return []

    # ------------------------------------------------------------------
    # Classify
    # ------------------------------------------------------------------

    def classify(self, query: str) -> ComplexityDecision:
        """
        Classify query complexity using prompt-based PolicyEngine when available.
        Falls back to keyword heuristics if PolicyEngine is not wired.
        Policy definitions live in config.yaml — operators tune them without code changes.
        """
        from runtime.policy_engine import get_policy_engine as _get_pe
        _engine = _get_pe()
        if _engine is not None:
            # Use synchronous fallback evaluation (keyword heuristic via engine._fallback)
            # The async path is used when classify() is called from async context in backend
            _dr = _engine._fallback("classify_destructive", query)
            _ir = _engine._fallback("classify_incident_severity", query)
            if _dr.match:
                return ComplexityDecision(
                    complexity=QueryComplexity.COMPLEX,
                    reason=f"Policy[classify_destructive]: {_dr.reason}",
                    confidence=_dr.confidence, model_tier="full_model",
                )
            if _ir.match:
                return ComplexityDecision(
                    complexity=QueryComplexity.COMPLEX,
                    reason=f"Policy[classify_incident_severity]: {_ir.reason}",
                    confidence=_ir.confidence, model_tier="full_model",
                )
            return ComplexityDecision(
                complexity=QueryComplexity.SIMPLE,
                reason="Policy: non-destructive query",
                confidence=0.85, model_tier="full_model",
            )

        # ── Keyword heuristic (fallback when PolicyEngine not yet wired) ──────
        import re as _re
        q = query.lower()

        def _word_match(kw: str, text: str) -> bool:
            if " " in kw or not kw.isascii():
                return kw in text
            return bool(_re.search(r"(?<![a-z0-9])" + _re.escape(kw) + r"(?![a-z0-9])", text))

        if any(_word_match(kw, q) for kw in _DESTRUCTIVE_KEYWORDS):
            return ComplexityDecision(
                complexity=QueryComplexity.COMPLEX,
                reason="Destructive action detected (keyword heuristic)",
                confidence=0.90, model_tier="full_model",
            )
        if any(_word_match(kw, q) for kw in _P0P1_KEYWORDS):
            return ComplexityDecision(
                complexity=QueryComplexity.COMPLEX,
                reason="P0/P1 severity (keyword heuristic)",
                confidence=0.85, model_tier="full_model",
            )
        tier = "fast_model" if any(_word_match(kw, q) for kw in _FAST_MODEL_KEYWORDS) else "full_model"
        return ComplexityDecision(
            complexity=QueryComplexity.SIMPLE,
            reason="Single-intent diagnostic query (keyword heuristic)",
            confidence=0.80, model_tier=tier,
        )

    # ------------------------------------------------------------------
    # Pre-verification (P1)
    # ------------------------------------------------------------------

    async def classify_async(self, query: str) -> "ComplexityDecision":
        """
        Async classify — uses PolicyEngine LLM evaluation when wired.
        Called from backend.py (async context). Falls back to synchronous
        keyword heuristic if engine is unavailable or LLM fails.
        """
        from runtime.policy_engine import get_policy_engine as _get_pe
        _engine = _get_pe()
        if _engine is not None:
            try:
                results = await _engine.evaluate_any(
                    ["classify_destructive", "classify_incident_severity"], query
                )
                destructive = results.get("classify_destructive")
                incident    = results.get("classify_incident_severity")
                if destructive and destructive.match:
                    return ComplexityDecision(
                        complexity=QueryComplexity.COMPLEX,
                        reason=f"Policy[classify_destructive]: {destructive.reason}",
                        confidence=destructive.confidence, model_tier="full_model",
                    )
                if incident and incident.match:
                    return ComplexityDecision(
                        complexity=QueryComplexity.COMPLEX,
                        reason=f"Policy[classify_incident_severity]: {incident.reason}",
                        confidence=incident.confidence, model_tier="full_model",
                    )
                return ComplexityDecision(
                    complexity=QueryComplexity.SIMPLE,
                    reason="Policy: non-destructive diagnostic query",
                    confidence=0.85, model_tier="full_model",
                )
            except Exception as _e:
                import logging as _log
                _log.getLogger(__name__).warning(
                    "classify_async: PolicyEngine failed (%s) — keyword fallback", _e
                )
        # Fallback to synchronous keyword heuristic
        return self.classify(query)


    async def pre_verify(
        self,
        query: str,
        confirmed_facts: list[str],
        env_context: dict[str, Any],
    ) -> VerificationResult:
        """
        Pre-action verification using PolicyEngine when available.
        Falls back to keyword check if engine is not wired.

        We pass any recalled session context (env_context["_fts_context"]) and
        confirmed_facts to the policy so that follow-up queries with pronouns
        ("this device", "it", "the AP") can be resolved against prior turns.
        Without this, multi-turn queries get BLOCKed for "missing device ID".
        """
        from runtime.policy_engine import get_policy_engine as _get_pe
        _engine = _get_pe()
        if _engine is not None:
            try:
                # Build a context block the policy LLM can read alongside the query.
                _ctx_parts: list[str] = []
                _fts = (env_context or {}).get("_fts_context")
                if _fts:
                    _ctx_parts.append(str(_fts)[:1500])
                if confirmed_facts:
                    _ctx_parts.append(
                        "Confirmed facts:\n" +
                        "\n".join(f"- {f}" for f in list(confirmed_facts)[:10])
                    )
                _ctx = "\n\n".join(_ctx_parts)
                result = await _engine.evaluate(
                    "preverify_safe_to_proceed", query, context=_ctx
                )
                if not result.match:
                    return VerificationResult.fail(
                        f"Policy[preverify_safe_to_proceed]: {result.reason}"
                    )
                return VerificationResult.ok("Pre-verification passed")
            except Exception as _e:
                import logging as _log
                _log.getLogger(__name__).warning("pre_verify PolicyEngine error: %s", _e)

        # Keyword fallback
        q = query.lower()
        if any(kw in q for kw in _DESTRUCTIVE_KEYWORDS):
            return VerificationResult.fail(
                "Destructive action reached pre_verify — escalate to HITL"
            )
        return VerificationResult.ok("Pre-verification passed")


    async def post_verify(
        self,
        action_type: str,
        result: str,
        confirmed_facts: list[str],
    ) -> VerificationResult:
        """
        Post-action verification: check the action achieved its intended outcome.
        Replace with real health-check calls in production.
        """
        warnings = []
        result_lower = result.lower()

        if "error" in result_lower or "fail" in result_lower:
            warnings.append("Result contains 'error'/'fail' — manual check recommended")

        if action_type == "restart_service":
            # TODO: post-action health probe — verify the restarted service via
            # whatever health-check tool is registered for it.
            if "healthy" not in result_lower:
                return VerificationResult.fail(
                    f"Post-restart health check inconclusive for {action_type}",
                    warnings=warnings,
                )

        return VerificationResult.ok(
            f"Post-verification passed for {action_type}"
            + (f" (warnings: {len(warnings)})" if warnings else "")
        )

    # ------------------------------------------------------------------
    # Build forked context (P1)
    # ------------------------------------------------------------------

    def build_fork_context(
        self,
        parent_state: LoopState,
        policy: ForkContextPolicy = ForkContextPolicy.FACTS_ONLY,
    ) -> dict[str, Any]:
        """
        Build the context dict to pass to a forked sub-agent.
        The sub-agent inherits part of the parent's accumulated state.
        """
        if policy == ForkContextPolicy.FULL:
            return {
                "confirmed_facts": list(parent_state.confirmed_facts),
                "working_set":     list(getattr(parent_state, "working_set", [])),
                "tool_summaries":  list(parent_state.tool_summaries),
            }
        if policy == ForkContextPolicy.WORKING_SET:
            return {
                "confirmed_facts": list(parent_state.confirmed_facts),
                "working_set":     list(getattr(parent_state, "working_set", [])),
            }
        # FACTS_ONLY (default)
        return {
            "confirmed_facts": list(parent_state.confirmed_facts),
        }

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    async def run(
        self,
        query:           str,
        session_id:      str,
        env_context:     Optional[dict[str, Any]] = None,
        confirmed_facts: Optional[list[str]] = None,
        working_set:     Optional[list[DeviceRef]] = None,
        tool_registry:   Optional[dict[str, Any]] = None,
        delegation_mode: DelegationMode = DelegationMode.FRESH,
        parent_state:    Optional[LoopState] = None,
    ) -> LoopResult:
        env_ctx  = env_context or {}
        tool_reg = tool_registry or {}

        # P1: forked delegation — inherit parent context
        if delegation_mode == DelegationMode.FORKED and parent_state is not None:
            fork_ctx      = self.build_fork_context(parent_state, self._cfg.default_fork_context)
            confirmed_facts = fork_ctx.get("confirmed_facts", confirmed_facts or [])
            if not working_set:
                working_set = fork_ctx.get("working_set", [])
            logger.info(
                "RuntimeLoop: forked delegation — inheriting %d facts from parent",
                len(confirmed_facts),
            )

        state = LoopState()
        state.confirmed_facts = list(confirmed_facts or [])
        setattr(state, "working_set", list(working_set or []))

        # Pre-verification deferred until AFTER first memory recall (see below)
        # so PolicyEngine can see prior session context. See identical fix in
        # stream() for the rationale.

        chunks: list[str] = []
        last_tool_result  = ""
        tool_outputs: dict[str, str] = {}   # persists across turns — tool results feed next LLM call
        _pre_verified = False                # run pre_verify exactly once, after first recall
        # Seed called_tools from any prior tool calls visible in memory context.
        # This prevents the LLM from re-calling the same tool+args when memory
        # already has the result from a previous stream() invocation.
        # called_tools uses _call_key(name, args) fingerprints — not bare names.
        # Also seeded from TOOL_EXEC ledger in confirmed_facts (from prior HTTP requests)
        # so tools that ran in previous rounds are not re-executed.
        called_tools: set[str] = set()
        _known_stores: dict[str, str] = {}  # ref_id → tool_name (for context injection)
        import json as _j2, re as _re2
        for _fact in (confirmed_facts or []):
            if _fact.startswith("TOOL_EXEC: "):
                # Parse: "TOOL_EXEC: tool_name|{args} → ref=abc size=N"
                _body = _fact[len("TOOL_EXEC: "):]
                _arrow = _body.find(" → ")
                if _arrow > 0:
                    _call_part = _body[:_arrow].strip()
                    _info_part = _body[_arrow+3:].strip()
                    # Seed called_tools with the _call_key fingerprint
                    called_tools.add(_call_part)
                    # Extract ref_id if present
                    _ref_m = _re2.search(r"ref=(\w+)", _info_part)
                    if _ref_m:
                        _tool_n = _call_part.split("|")[0]
                        _known_stores[_ref_m.group(1)] = _tool_n
        if called_tools:
            logger.info("stream: seeded %d prior tool calls from ledger; %d known stores",
                        len(called_tools), len(_known_stores))

        while True:
            state.turns += 1
            memory_results = await self._retrieve_memory(query, session_id)

            # P1: pre-verification (deferred until first recall; runs once)
            if not _pre_verified and self._cfg.enable_pre_verification:
                _pre_env = dict(env_ctx)
                if "_fts_context" not in _pre_env and memory_results:
                    try:
                        _pre_env["_fts_context"] = "\n".join(
                            str(m) for m in memory_results
                        )[:1500]
                    except Exception:
                        pass
                pre = await self.pre_verify(query, state.confirmed_facts, _pre_env)
                _pre_verified = True
                if not pre.passed:
                    return LoopResult(
                        outcome=StopOutcome.STOP_HITL,
                        final_response=f"Pre-verification failed: {pre.reason}",
                        confirmed_facts=state.confirmed_facts,
                        verification=pre,
                    )

            # NOTE: We intentionally do NOT seed called_tools from memory context.
            # Memory may mention a prior tool call on device X, but we still need
            # to allow that tool to run for devices Y and Z in the current turn.
            # Deduplication is by _call_key (name+args), so the same tool with
            # different args is allowed; same call with same args is blocked.

            # P2: skill catalog summary always prepended (cache-stable prefix)
            skill_section = ""
            if self._skill_catalog:
                skill_section = self._skill_catalog.format_summary()

            # Compress paged results before assembly to prevent context overflow
            from runtime.context_budget import compress_paged_outputs as _compress
            _to_assemble = _compress(tool_outputs)
            context_str = self._budget.assemble(
                memory_results=memory_results,
                tool_outputs=_to_assemble,       # pass compressed accumulated results to LLM
                confirmed_facts=state.confirmed_facts,
                working_set=working_set,
                env_context=env_ctx,
            )
            if skill_section:
                context_str = skill_section + "\n\n" + context_str

            # Attach live tool registry to state so _call_llm / llm_engine can
            # inject it into the system prompt (shows uploaded tools to the LLM)
            state._tool_registry = tool_reg  # type: ignore[attr-defined]
            llm_response = await self._call_llm(query, context_str, state)
            state.tokens_consumed += self._budget._estimate_tokens(context_str + llm_response)
            state.record_response(llm_response)
            chunks.append(llm_response)

            # P1: detect SKILL_LOAD directives and expand detail on demand
            import re
            _skill_loads_this_turn_r: set[str] = set()
            for skill_id in re.findall(r"\[SKILL_LOAD:(\w+)\]", llm_response):
                if skill_id in _skill_loads_this_turn_r:
                    continue
                _skill_loads_this_turn_r.add(skill_id)
                called_tools.add(f"SKILL_LOAD:{skill_id}")
                if self._skill_catalog:
                    detail = self._skill_catalog.load_detail(skill_id)
                    if detail:
                        context_str += "\n\n" + detail
                        logger.debug("SkillCatalog: loaded detail for %s", skill_id)

            # Execute tool calls — one per turn only
            _single = self._parse_tool_call(llm_response)
            tool_calls = [_single] if _single else []
            new_tool_calls = [(n, a) for n, a in tool_calls if _call_key(n, a) not in called_tools]
            for tool_name, tool_args in new_tool_calls:
                state.record_tool_call(tool_name)
                called_tools.add(_call_key(tool_name, tool_args))

                # Skill-as-tool guard: only block if the name is a skill AND NOT a real tool.
                # When a name exists in both catalogs (e.g. list_devices is both a skill
                # description AND a callable tool), the tool takes priority.
                _is_skill_only = False
                if self._skill_catalog and tool_name not in tool_reg:
                    try:
                        _is_skill_only = any(
                            s.skill_id == tool_name
                            for s in self._skill_catalog.list_skills()
                        )
                    except Exception:
                        pass
                if _is_skill_only:
                    raw = (
                        f"[ERROR] '{tool_name}' is a SKILL description, not a callable tool. "
                        f"Use [SKILL_LOAD:{tool_name}] to read its steps, "
                        f"then call the individual tools it describes."
                    )
                    logger.warning("run: LLM called skill-only '%s' as tool — injecting error", tool_name)
                else:
                    raw = await self._execute_tool(tool_name, tool_args, tool_reg)
                stored = self._budget.store_tool_result(tool_name, raw)
                tool_outputs[_call_key(tool_name, tool_args)] = stored   # accumulate ALL results
                last_tool_result = raw

                # P1: post-verification after each tool call
                if self._cfg.enable_post_verification and tool_name != "read_stored_result":
                    post = await self.post_verify(tool_name, raw, state.confirmed_facts)
                    if not post.passed:
                        logger.warning("Post-verify failed: %s", post.reason)
                        state.unresolved_points.append(f"Post-verify failed: {post.reason}")

            decision = self._policy.evaluate(state)
            if decision.should_stop:
                final = self._format_final(chunks, decision)
                return LoopResult(
                    outcome=decision.outcome,
                    final_response=final,
                    confirmed_facts=state.confirmed_facts,
                    working_set=getattr(state, "working_set", []),
                    unresolved=state.unresolved_points,
                    tool_summaries=state.tool_summaries,
                    turns_taken=state.turns,
                )

            if self._is_complete(llm_response, new_tool_calls):
                return LoopResult(
                    outcome=StopOutcome.CONTINUE,
                    final_response="\n".join(chunks),
                    confirmed_facts=state.confirmed_facts,
                    working_set=getattr(state, "working_set", []),
                    unresolved=state.unresolved_points,
                    tool_summaries=state.tool_summaries,
                    turns_taken=state.turns,
                )

    # ------------------------------------------------------------------
    # Stream
    # ------------------------------------------------------------------

    async def stream(
        self,
        query:           str,
        session_id:      str,
        env_context:     Optional[dict[str, Any]] = None,
        confirmed_facts: Optional[list[str]] = None,
        working_set:     Optional[list[DeviceRef]] = None,
        tool_registry:   Optional[dict[str, Any]] = None,
        delegation_mode: DelegationMode = DelegationMode.FRESH,
        parent_state:    Optional[LoopState] = None,
        skip_pre_verify: bool = False,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        skip_pre_verify: when True, bypass the PolicyEngine pre-verification
        gate. Post-HITL callers MUST set this to True — operator approval is
        the final authority on whether a destructive action proceeds, and
        running PolicyEngine afterwards causes it to second-guess and BLOCK
        already-approved actions on keywords like "修复" / "restart" / "rollback".
        """
        env_ctx  = env_context or {}
        tool_reg = tool_registry or {}

        if delegation_mode == DelegationMode.FORKED and parent_state is not None:
            fork_ctx = self.build_fork_context(parent_state, self._cfg.default_fork_context)
            confirmed_facts = fork_ctx.get("confirmed_facts", confirmed_facts or [])
            if not working_set:
                working_set = fork_ctx.get("working_set", [])

        state = LoopState()
        state.confirmed_facts = list(confirmed_facts or [])
        setattr(state, "working_set", list(working_set or []))

        # Pre-verification deferred until AFTER memory recall (see below) so
        # that PolicyEngine can resolve references like "this device" against
        # prior turns. Without this, post-HITL callbacks (and any caller that
        # doesn't pre-fill env_context with _fts_context) would run pre_verify
        # with empty context and mis-classify pronoun queries as ambiguous.

        tool_outputs: dict[str, str] = {}   # persists across turns
        called_tools: set[str] = set()       # dedup guard
        _pre_verified = False                # run pre_verify exactly once

        # ── Clarification gate (runs ONCE before the first turn) ─────────
        # When the caller has supplied a complexity decision and its
        # confidence is below the clarification threshold, ask the operator
        # for missing info instead of guessing.
        #
        # Two interaction modes — the choice depends on whether the agent
        # already knows the candidate space:
        #   (a) No closed candidate list  → ask via plain chat turn.
        #       The agent prints its question(s) and returns; the operator's
        #       next turn carries the answer naturally. This is the right
        #       UX for open-ended "which device?" — the operator might type
        #       any of dozens of devices, and a side panel with one
        #       narrow input box doesn't help.
        #   (b) Closed candidate list (e.g. "4 APs match")  → USER_CHOICE
        #       card with clickable options, handled by the skill ambiguity
        #       gate downstream. This gate only handles (a).
        # Soft-capped by max_clarifications to prevent loops.
        _clarification_done = bool(env_ctx.get("_clarification_resolved"))
        _initial_confidence = env_ctx.get("_initial_confidence")
        try:
            _initial_confidence = float(_initial_confidence) if _initial_confidence is not None else None
        except (TypeError, ValueError):
            _initial_confidence = None

        if (
            not _clarification_done
            and not skip_pre_verify
            and _initial_confidence is not None
            and _initial_confidence < self._cfg.stop_policy.clarification_threshold
            and self._cfg.stop_policy.max_clarifications > 0
        ):
            # Skip if prior turn already asked clarification — the current
            # query IS the operator's answer; running the gate again would
            # double-ask. We detect this by sniffing recent recall context
            # for the agent's question marker.
            _prior_recall = env_ctx.get("_fts_context", "") or ""
            if (
                "为了准确处理这个请求，我需要您补充" in _prior_recall
                or "Clarification asked via chat turn" in _prior_recall
            ):
                logger.info(
                    "Clarification gate: prior turn already asked — "
                    "treating this query as the answer, skipping gate",
                )
            else:
                # Derive what's likely missing — use the LLM-extracted gaps the
                # caller has supplied via env_context, otherwise heuristic.
                missing = env_ctx.get("_clarification_fields") or _heuristic_missing_fields(query)
                if missing:
                    logger.info(
                        "Clarification gate: confidence=%.2f < threshold=%.2f — "
                        "asking operator for %s via chat turn",
                        _initial_confidence,
                        self._cfg.stop_policy.clarification_threshold,
                        [f["key"] for f in missing],
                    )
                    # Render a friendly chat-turn message that lists the
                    # questions. The operator's next message in the same
                    # session naturally provides the answers; no card UI
                    # needed. The agent reads recall context next time so
                    # it sees both the question and the answer.
                    _q_lines = []
                    for f in missing:
                        _ph = f.get("placeholder", "")
                        _star = "" if f.get("required", True) else "（可选）"
                        line = f"- **{f['prompt']}**{_star}"
                        if _ph:
                            line += f"  _例: {_ph}_"
                        _q_lines.append(line)
                    _ask_text = (
                        "为了准确处理这个请求，我需要您补充几个关键信息：\n\n"
                        + "\n".join(_q_lines)
                        + "\n\n请直接回复，我会基于您的补充继续。"
                    )
                    # Stream as tokens so the chat UI shows it like any
                    # normal agent reply.
                    for _i in range(0, len(_ask_text), 80):
                        yield {"token": _ask_text[_i:_i+80]}
                    yield {
                        "node":      "clarification_chat",
                        "node_step": "Clarification asked via chat turn",
                        "reason":    "clarification_needed",
                        "confidence": _initial_confidence,
                        # Persist a hint for the next-turn coreference resolver
                        "_clarification_fields_asked": [f["key"] for f in missing],
                        "message": "Clarification asked — awaiting operator's next message",
                    }
                    return

        while True:
            state.turns += 1
            memory_results = await self._retrieve_memory(query, session_id)

            # Run pre_verify on the FIRST turn only, AFTER recall. Inject the
            # recalled context (and any caller-supplied _fts_context) so the
            # policy LLM can see entities mentioned earlier in the session.
            # Skipped when caller has already obtained operator approval (HITL).
            if (not _pre_verified
                    and self._cfg.enable_pre_verification
                    and not skip_pre_verify):
                _pre_env = dict(env_ctx)
                # Prefer caller-supplied context; otherwise use freshly recalled.
                if "_fts_context" not in _pre_env and memory_results:
                    try:
                        _pre_env["_fts_context"] = "\n".join(
                            str(m) for m in memory_results
                        )[:1500]
                    except Exception:
                        pass
                pre = await self.pre_verify(query, state.confirmed_facts, _pre_env)
                _pre_verified = True
                if not pre.passed:
                    yield {"message": f"Pre-verification failed: {pre.reason}", "node": "pre_verify"}
                    return

            skill_section  = ""
            skill_count    = 0
            selected_skills: list = []
            skill_ambiguous = False
            if self._skill_catalog:
                try:
                    sel = self._skill_catalog.select_skills_for_query(query, top_k=5)
                    skill_section   = sel.summary
                    selected_skills = sel.selected          # [(skill_id, score), ...]
                    skill_ambiguous = sel.ambiguous
                    skill_count     = len(selected_skills)
                except AttributeError:
                    # Fallback if select_skills_for_query not available
                    skill_section = self._skill_catalog.format_summary()
                    skill_count   = len(getattr(self._skill_catalog, "_skills", {}))

            # Compress paged results before assembly to prevent context overflow.
            # Without this, accumulated read_stored_result pages send the full
            # paged content back to the LLM every turn — Ollama times out.
            from runtime.context_budget import compress_paged_outputs as _compress
            _to_assemble = _compress(tool_outputs)
            context_str = self._budget.assemble(
                memory_results=memory_results,
                tool_outputs=_to_assemble,       # compressed accumulated results
                confirmed_facts=state.confirmed_facts,
                working_set=working_set,
                env_context=env_ctx,
            )
            if skill_section:
                context_str = skill_section + "\n\n" + context_str
                # Q1: emit named matched skills so Flow tab shows exactly which skills loaded
                skill_names = ", ".join(f"{sid}({sc:.2f})" for sid, sc in selected_skills) \
                              or f"{skill_count} skills"
                yield {
                    "node_step": f"Skills matched: {skill_names}",
                    "node":      "skill_load",
                    "skill_count": skill_count,
                    "selected_skills": [{"id": sid, "score": sc} for sid, sc in selected_skills],
                    "ambiguous": skill_ambiguous,
                }
                # Q4 ambiguity: top skills are too similar — surface them as
                # a USER_CHOICE so the operator picks the intended one
                # instead of the loop guessing or grinding to a generic HITL.
                # Enabled by default; set HITL_SKILL_AMBIGUITY=false to disable.
                import os as _os
                if skill_ambiguous and _os.getenv("HITL_SKILL_AMBIGUITY", "true").lower() != "false":
                    # Build choice options from the top-N candidate skills.
                    # Capped at 5 to keep the UI scannable.
                    _choices = []
                    for sid, score in selected_skills[:5]:
                        summary = self._catalog.get_summary(sid) if hasattr(self, "_catalog") and self._catalog else None
                        if summary is None and self._skill_catalog is not None:
                            try:
                                summary = self._skill_catalog.get_summary(sid)
                            except Exception:
                                summary = None
                        _choices.append({
                            "id":       sid,
                            "label":    (summary.name if summary else sid),
                            "description": (
                                f"{(summary.purpose if summary else '')[:120]}"
                                + (f" · {summary.risk_level}" if summary and summary.risk_level else "")
                            ),
                            "metadata": {
                                "skill_id": sid,
                                "score":    round(float(score), 3),
                                "tags":     (summary.tags[:4] if summary else []),
                            },
                        })
                    yield {
                        "message": (
                            f"Multiple skills match this request — top candidates: "
                            f"{[c['id'] for c in _choices[:3]]}"
                        ),
                        "node":           "hitl_gate",
                        "stop_hitl":      True,
                        "reason":         "skill_ambiguity",
                        "hitl_kind":      "user_choice",
                        "summary":        f"找到多个匹配的 skill，请选择要使用的具体 skill：",
                        "choices":        _choices,
                        # Keep top_skills for backwards compatibility
                        "top_skills":     [c["id"] for c in _choices[:2]],
                    }
                    return

            # ── Type #3 CLARIFICATION gate ────────────────────────────
            # When the agent is about to hallucinate a plan from too little
            # info, ask the operator for missing pieces instead. Triggered by:
            #   - top skill score < clarification_confidence_floor AND
            #   - query mentions an action word but no concrete target (e.g.
            #     "修复" without device_id, "查日志" without time range), AND
            #   - we haven't already asked the operator clarification_max_per_session
            #     times in this session (avoid loops).
            # Skipped on subsequent turns (state.turns > 1) — clarify only at
            # the start of a new request, never mid-execution.
            if state.turns == 1 and self._cfg.clarification_max_per_session > 0:
                # Build recent_context from current turn's recall + caller-supplied
                # _fts_context. _maybe_clarification_fields uses this to detect
                # "prior turn already asked" and skip re-asking.
                _recent_for_clar = env_ctx.get("_fts_context", "") or ""
                if not _recent_for_clar and memory_results:
                    try:
                        _recent_for_clar = "\n".join(str(m) for m in memory_results)[:1500]
                    except Exception:
                        pass
                clar_fields = self._maybe_clarification_fields(
                    query=query,
                    top_skill_score=(selected_skills[0][1] if selected_skills else 0.0),
                    asked_count=self._clarification_counts.get(session_id, 0),
                    recent_context=_recent_for_clar,
                )
                if clar_fields:
                    self._clarification_counts[session_id] = (
                        self._clarification_counts.get(session_id, 0) + 1
                    )
                    # Open-ended clarification — render as a CHAT TURN, not a
                    # HITL card. The operator's next message in the same
                    # session naturally provides the answers. This is the
                    # right UX for "which device do you mean?" — operators
                    # have hundreds of devices to potentially reference and
                    # a one-line input box doesn't help them.
                    # Closed candidate lists (e.g. "4 APs match") still go
                    # through the USER_CHOICE card path — see skill ambiguity.
                    logger.info(
                        "Clarification gate (turn 1): asking operator for %s via chat turn",
                        [getattr(f, "key", f.get("key") if isinstance(f, dict) else "?")
                         for f in clar_fields],
                    )
                    _q_lines = []
                    for f in clar_fields:
                        if hasattr(f, "key"):  # ClarificationField object
                            _key = f.key
                            _prompt = f.prompt
                            _ph = getattr(f, "placeholder", "") or ""
                            _required = getattr(f, "required", True)
                        else:                    # plain dict
                            _key = f.get("key", "")
                            _prompt = f.get("prompt", _key)
                            _ph = f.get("placeholder", "")
                            _required = f.get("required", True)
                        _star = "" if _required else "（可选）"
                        line = f"- **{_prompt}**{_star}"
                        if _ph:
                            line += f"  _例: {_ph}_"
                        _q_lines.append(line)
                    _ask_text = (
                        "为了准确处理这个请求，我需要您补充几个关键信息：\n\n"
                        + "\n".join(_q_lines)
                        + "\n\n请直接回复，我会基于您的补充继续。"
                    )
                    # Stream as chat tokens — no card, no stop_hitl.
                    for _i in range(0, len(_ask_text), 80):
                        yield {"token": _ask_text[_i:_i+80]}
                    yield {
                        "node":      "clarification_chat",
                        "node_step": "Clarification asked via chat turn",
                        "reason":    "clarification_needed",
                        "message":   "Clarification asked — awaiting operator's next message",
                    }
                    return

            yield {"node_step": f"Turn {state.turns}: analysing", "node": "runtime_loop"}

            # Attach live tool registry to state so _call_llm / llm_engine can
            # inject it into the system prompt (shows uploaded tools to the LLM)
            state._tool_registry = tool_reg  # type: ignore[attr-defined]
            llm_response = await self._call_llm(query, context_str, state)
            state.tokens_consumed += self._budget._estimate_tokens(context_str + llm_response)
            state.record_response(llm_response)

            # CAP 6: emit LLM trace so Flow tab shows messages and token usage
            _trace = {}
            if hasattr(state, "_llm_traces") and state._llm_traces:
                _trace = state._llm_traces[-1]
            yield {
                "type":             "llm_trace",
                "turn":             state.turns,
                "model":            _trace.get("model", "mock"),
                "system_chars":     _trace.get("system_chars", len(context_str)),
                "context_chars":    _trace.get("context_chars", len(context_str)),
                "response_chars":   _trace.get("response_chars", len(llm_response)),
                "has_tool_call":    "[TOOL:" in llm_response,
                "system_preview":   _trace.get("system_preview", context_str[:200]),
                "response_preview": _trace.get("response_preview", llm_response[:200]),
            }

            # ── Stream tokens to user — strip [TOOL:...] lines ──────────
            # The raw LLM response may contain [TOOL:name] {...} directives.
            # These are execution instructions, not prose — never show them
            # to the user.  Strip any line that starts with [TOOL: before
            # yielding tokens.
            import re as _re
            _visible_lines = [
                ln for ln in llm_response.splitlines()
                if not _re.match(r'\s*\[TOOL:\w+\]', ln)
            ]
            _visible = "\n".join(_visible_lines).strip()
            if _visible:
                # Stream in 80-char chunks preserving newlines + whitespace.
                # Frontend renders markdown after the stream completes, so block
                # structure (\n\n, table rows, ```fences, ## headers) must survive.
                for _i in range(0, len(_visible), 80):
                    yield {"token": _visible[_i:_i+80]}
                    await asyncio.sleep(0)
            # If the entire response was tool calls (no prose), yield nothing —
            # the tool result will be injected in the next turn's context and
            # the LLM will produce a proper prose answer then.

            import re
            _skill_loads_this_turn: set[str] = set()
            for skill_id in re.findall(r"\[SKILL_LOAD:(\w+)\]", llm_response):
                if skill_id in _skill_loads_this_turn:
                    continue   # deduplicate within a single response
                _skill_loads_this_turn.add(skill_id)
                # Mark as "called" so dedup blocks repeated SKILL_LOAD across turns
                called_tools.add(f"SKILL_LOAD:{skill_id}")
                if self._skill_catalog:
                    detail = self._skill_catalog.load_detail(skill_id)
                    if detail:
                        context_str += "\n\n" + detail   # inject for next turn
                        yield {"node_step": f"Loading skill details: {skill_id}", "node": "skill_load"}

            # ── Single tool call enforcement ──────────────────────────
            # _parse_tool_call() returns only the FIRST [TOOL:] found.
            # Multiple calls in one response are a model error — execute
            # only the first so we feed back real data before the next call.
            _single = self._parse_tool_call(llm_response)
            tool_calls = [_single] if _single else []
            new_tool_calls = [(n, a) for n, a in tool_calls if _call_key(n, a) not in called_tools]
            for tool_name, tool_args in new_tool_calls:
                state.record_tool_call(tool_name)
                called_tools.add(_call_key(tool_name, tool_args))

                # ── Skill-as-tool guard ───────────────────────────────
                # If the LLM called a SKILL name as if it were a tool,
                # inject an error result so the LLM corrects itself on
                # the next turn rather than hitting the HITL gate or
                # getting a "not registered" error with no explanation.
                # Skill-as-tool guard: only block if the name is a skill AND NOT a real tool.
                # Tools and skills can share the same name (e.g. list_devices is both a
                # skill description and a real callable tool). The tool always wins.
                _is_skill_only = False
                if self._skill_catalog and tool_name not in tool_reg:
                    try:
                        _is_skill_only = any(
                            s.skill_id == tool_name
                            for s in self._skill_catalog.list_skills()
                        )
                    except Exception:
                        pass
                if _is_skill_only:
                    # ── Special case: HITL-required skill called as [TOOL:] ──
                    # Rather than injecting a "not a tool" error (which causes
                    # the LLM to loop through SKILL_LOAD indefinitely), detect
                    # the requires_hitl flag and route straight to stop_hitl.
                    # This lets restart_service, rollback_service etc. trigger
                    # the HITL interrupt card exactly like edit_device_config.
                    _skill_requires_hitl = False
                    if self._skill_catalog:
                        try:
                            _skill_requires_hitl = self._skill_catalog.requires_hitl(tool_name)
                        except Exception:
                            pass
                    if _skill_requires_hitl:
                        import json as _json
                        logger.info(
                            "stream: HITL-required skill '%s' called as tool — routing to HITL",
                            tool_name,
                        )
                        yield {
                            "message": (
                                f"stop_hitl: skill '{tool_name}' requires human approval "
                                "before execution. Routing to HITL graph."
                            ),
                            "node":          "hitl_gate",
                            "stop_hitl":     True,
                            "tool_name":     tool_name,
                            "tool_args":     tool_args,
                            "tool_args_json": _json.dumps(tool_args, default=str),
                        }
                        return
                    # Non-HITL skill-only: inject guidance error so LLM learns to SKILL_LOAD
                    _skill_err = (
                        f"[ERROR] '{tool_name}' is a SKILL description, not a callable tool. "
                        f"Use [SKILL_LOAD:{tool_name}] to read its steps, "
                        f"then call the individual tools it describes."
                    )
                    logger.warning("stream: LLM called skill-only '%s' as tool — injecting error", tool_name)
                    tool_outputs[_call_key(tool_name, tool_args)] = _skill_err
                    yield {"node_step": f"Skill-only error: {tool_name}", "node": "runtime_loop"}
                    continue   # skip HITL check and _execute_tool for this name

                # CAP 5: gate tool against HITL watch-list BEFORE execution
                # Only fires for REAL tools, not skill names (guarded above).
                _needs_hitl = tool_name in self._cfg.hitl_tool_names
                if not _needs_hitl and self._skill_catalog:
                    try:
                        # Only check HITL for a name if it is actually a registered tool
                        # (i.e. present in the tool registry), not a stray skill name
                        _is_real_tool = tool_name in tool_reg
                        if _is_real_tool:
                            _needs_hitl = self._skill_catalog.requires_hitl(tool_name)
                    except Exception:
                        pass
                if _needs_hitl:
                    import json as _json
                    # Type #2: if this tool is on the editable list, surface
                    # the keys the operator should be allowed to tweak before
                    # approving. The executor will fire trigger_edit_approval
                    # (showing inline editors) instead of a bare approve panel.
                    _editable_keys = list(
                        self._cfg.editable_hitl_tools.get(tool_name, [])
                    )
                    yield {
                        "message": (
                            f"stop_hitl: tool '{tool_name}' is on the HITL watch-list "
                            "and requires human approval before execution. "
                            "Routing to HITL graph."
                        ),
                        "node":      "hitl_gate",
                        "stop_hitl": True,
                        "tool_name": tool_name,
                        "tool_args": tool_args,          # carry args for post-approval replay
                        "tool_args_json": _json.dumps(tool_args, default=str),
                        # Type #2 multi-mode HITL: signal to the executor that
                        # the operator may edit these specific param keys.
                        "hitl_kind":            "edit" if _editable_keys else None,
                        "editable_param_keys":  _editable_keys,
                    }
                    return

                yield {"node_step": f"Calling tool: {tool_name}", "node": "runtime_loop"}
                logger.info("TOOL▶ %s args=%s", tool_name, tool_args)
                if logger.isEnabledFor(logging.DEBUG):
                    import json as _json
                    logger.debug("TOOL ARGS\n%s\n%s\n%s", "─"*72,
                                 _json.dumps(tool_args, indent=2, default=str), "─"*72)
                raw = await self._execute_tool(tool_name, tool_args, tool_reg)
                stored = self._budget.store_tool_result(tool_name, raw)
                tool_outputs[_call_key(tool_name, tool_args)] = stored   # accumulate ALL results
                # Update count so llm_engine knows how many current-turn results exist
                state._current_tool_outputs_count = len(tool_outputs)  # type: ignore[attr-defined]
                state._tool_output_keys = list(tool_outputs.keys())      # type: ignore[attr-defined]
                # Keep raw results for has_more / paging detection in llm_engine
                if not hasattr(state, "_tool_outputs_raw"):
                    state._tool_outputs_raw = {}  # type: ignore[attr-defined]
                state._tool_outputs_raw[_call_key(tool_name, tool_args)] = raw  # type: ignore[attr-defined]
                # Log when tool returns error/empty — high hallucination risk
                _raw_lower = raw.lower() if isinstance(raw, str) else ""
                if (raw.startswith("[Error]")
                        or "not found" in _raw_lower
                        or raw.strip() in ("", "[]", "{}")
                        or "no devices" in _raw_lower):
                    logger.warning(
                        "tool %r returned error/empty: %s", tool_name, raw[:120]
                    )
                logger.info("TOOL◀ %s result_chars=%d stored=%s",
                            tool_name, len(raw), stored.startswith("[STORED:"))
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("TOOL RESULT %s\n%s\n%s\n%s", tool_name, "─"*72, raw[:2000], "─"*72)
                yield {
                    "node_result": {
                        "tool":   tool_name,
                        "result": stored,      # full stored label (for large) or full raw text (for inline)
                        "raw":    raw,         # always full raw text — used by frontend Results tab
                        "args":   tool_args,   # pass args so frontend can label the card accurately
                    },
                    "node": "runtime_tool_result",
                }

                if self._cfg.enable_post_verification and tool_name != "read_stored_result":
                    post = await self.post_verify(tool_name, raw, state.confirmed_facts)
                    if not post.passed:
                        yield {"node_step": f"Post-verify warning: {post.reason}", "node": "post_verify"}

            decision = self._policy.evaluate(state)
            if decision.should_stop:
                yield {"message": self._format_final([], decision), "node": "runtime_loop"}
                # Write tool execution ledger into confirmed_facts for next-round reuse
                _ledger = _build_tool_ledger(tool_outputs, tool_reg,
                                              getattr(state, "_tool_outputs_raw", {}))
                state.confirmed_facts.extend(_ledger)
                yield {"type": "confirmed_facts", "confirmed_facts": list(state.confirmed_facts)}
                return

            if self._is_complete(llm_response, new_tool_calls):
                # Guard: never exit with empty or trivial response when we have context.
                # Also fires when LLM returns a very short response after tool errors/empty
                # results — it should tell the user something meaningful, not go silent.
                _resp_stripped = llm_response.strip()
                _has_context   = bool(tool_outputs) or bool(state.confirmed_facts)
                _tool_had_result = bool(tool_outputs)
                _resp_too_short  = len(_resp_stripped) < 30 and _tool_had_result
                if (not _resp_stripped or _resp_too_short) and _has_context and state.turns < 3:
                    # Force one more turn with an explicit synthesis instruction
                    logger.warning(
                        "stream: empty response with available context at turn %d — nudging",
                        state.turns,
                    )
                    # Inject a nudge into confirmed_facts as a one-time instruction
                    # Detect if tools returned errors or empty results
                    _tool_vals = list(tool_outputs.values())
                    _has_errors = any(
                        "[Error]" in str(v) or "not found" in str(v).lower()
                        or "No devices" in str(v)
                        for v in _tool_vals
                    )
                    if _has_errors:
                        _nudge_text = (
                            "_NUDGE: Your previous response was empty or too short. "
                            "The tools returned errors or empty results. "
                            "Report this clearly to the user: explain what the tool found (or didn't find), "
                            "and suggest what they could do next. Do NOT fabricate data."
                        )
                    else:
                        _nudge_text = (
                            "_NUDGE: Your previous response was empty or too short. "
                            "Write a complete answer using the available context and tool results."
                        )
                    state.confirmed_facts.append(_nudge_text)
                    # NOTE: do NOT manually increment state.turns here — the
                    # `continue` jumps back to the top of the while loop where
                    # `state.turns += 1` runs as the first statement. Manual
                    # increment caused the < 3 check to allow only 1 retry.
                    continue  # retry the LLM call
                # Remove any lingering nudge entries
                state.confirmed_facts = [
                    f for f in state.confirmed_facts if not f.startswith("_NUDGE:")
                ]
                _ledger = _build_tool_ledger(tool_outputs, tool_reg,
                                              getattr(state, "_tool_outputs_raw", {}))
                state.confirmed_facts.extend(_ledger)
                # Capture final synthesis response (not intermediate page-reading turns)
                _resp_clean = llm_response.strip()
                _is_synthesis = (
                    len(_resp_clean) > 150
                    and "[TOOL:" not in _resp_clean
                    and "[SKILL_LOAD:" not in _resp_clean
                    and state.turns > 1
                )
                if _is_synthesis:
                    summary_line = _resp_clean[:500].replace("\n", " ")
                    state.confirmed_facts.append(f"PREV_ANALYSIS: {summary_line}")
                # NOTE: turn persistence (after_turn) is handled by the backend's
                # post-turn hook in webui/backend.py:600 via dtm.after_turn(). Do NOT
                # write here — it would create duplicate long_term_chunks entries.
                yield {"type": "confirmed_facts", "confirmed_facts": list(state.confirmed_facts)}
                return

    async def _retrieve_memory(self, query: str, session_id: str) -> list[Any]:
        if self._memory is None:
            return []
        try:
            # MemoryAdapter.recall_for_session returns a single string of recalled context
            recalled = await self._memory.recall_for_session(query, session_id)
            return [recalled] if recalled else []
        except Exception as exc:
            logger.warning("RuntimeLoop: memory retrieval failed: %s", exc)
            return []

    async def _call_llm(self, query: str, context: str, state: LoopState) -> str:
        """
        Default no-op stub. Raises if called — production code MUST patch this.

        At startup, integrations.llm_engine.patch_runtime_loop() replaces this
        method with one that dispatches to the real LLM engine (Ollama / OpenAI /
        Anthropic). If you see this exception in logs, patching never ran —
        check that main.py reached `patch_runtime_loop(executor, llm_engine)`.
        """
        raise RuntimeError(
            "AgentRuntimeLoop._call_llm has not been patched. "
            "Call integrations.llm_engine.patch_runtime_loop(loop, engine) at startup."
        )

    @staticmethod
    def _strip_thinking(response: str) -> str:
        """
        Remove <think>...</think> blocks emitted by thinking models
        (qwen3, deepseek-r1, etc.) before tool parsing or display.
        Preserves everything outside the think block.
        """
        import re
        # Remove <think>...</think> blocks (case-insensitive, multiline, non-greedy)
        cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE)
        return cleaned.strip()

    def _parse_tool_calls(self, response: str) -> list[tuple[str, dict[str, Any]]]:
        """
        Parse [TOOL:name] {...} directives from LLM response.

        Handles:
          - Thinking model output: strips <think>...</think> first
          - Nested JSON values (uses brace-depth counter not [^}]* regex)
          - Code fence wrapping: ```[TOOL:name] {...}```
          - Whitespace variants between [TOOL:name] and {
          - Malformed JSON: falls back to empty args dict
          - Multiple tool calls in one response (takes first only if dedup active)
        """
        import re
        import json as _json

        # Step 1: strip thinking block
        text = self._strip_thinking(response)

        # Step 2: also strip markdown code fences around the tool call
        text = re.sub(r"```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```", "", text)

        calls = []
        # Match [TOOL:toolname] then optional whitespace then optional JSON object
        for m in re.finditer(r"\[TOOL:(\w+)\]\s*(\{)?", text):
            tool_name = m.group(1)
            if not m.group(2):
                # No opening brace — no args
                calls.append((tool_name, {}))
                continue

            # Extract balanced JSON object starting at the {
            start = m.start(2)
            depth = 0
            end   = start
            in_str = False
            escape = False
            for i, ch in enumerate(text[start:], start):
                if escape:
                    escape = False
                    continue
                if ch == '\\' and in_str:
                    escape = True
                    continue
                if ch == '"' and not escape:
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break

            raw_json = text[start:end]
            try:
                args = _json.loads(raw_json)
            except Exception:
                # Try to recover partial JSON
                try:
                    args = _json.loads(raw_json + "}")
                except Exception:
                    args = {}
            calls.append((tool_name, args))

        return calls

    def _parse_tool_call(self, response: str) -> tuple[str, dict] | None:
        """
        Parse exactly ONE tool call from an LLM response — the first one found.

        This is the safe entry point used by the stream() loop.  Multiple
        [TOOL:] directives in one response violate the system prompt ("AT MOST
        ONE tool per response") and are almost always caused by the model
        hallucinating a sequence of calls it cannot actually execute in one
        turn.  We honour only the first and discard the rest so the loop can
        execute it, get the result, and give the model a chance to decide
        what to call next with real data.

        Returns (tool_name, args) or None if no tool call found.
        """
        calls = self._parse_tool_calls(response)
        return calls[0] if calls else None

    async def _execute_tool(
        self, tool_name: str, args: dict[str, Any], registry: dict[str, Any]
    ) -> str:
        await asyncio.sleep(0)
        tool_fn = registry.get(tool_name)
        if tool_fn is not None:
            try:
                result = await tool_fn(args)
                return str(result)
            except Exception as exc:
                return f"[Tool error: {exc}]"
        return f"[Tool {tool_name!r} not registered — args={args}]"

    @staticmethod
    def _skill_loads_in(response: str) -> set:
        import re as _re
        return set(_re.findall(r"\[SKILL_LOAD:(\w+)\]", response))

    @staticmethod
    def _is_complete(response: str, tool_calls: list) -> bool:
        # If the LLM emitted a SKILL_LOAD directive, it needs one more turn
        # to read the loaded detail and then call the actual tools.
        # A pure SKILL_LOAD response (no prose, no tool calls) means the model
        # asked for skill detail and is waiting for it — keep the loop running
        # so next turn can read the loaded detail. Only mark complete if the
        # model produced real prose + tool calls alongside the SKILL_LOAD.
        import re as _re
        skill_loads = _re.findall(r"\[SKILL_LOAD:\w+\]", response)
        if skill_loads:
            stripped = _re.sub(r"\[SKILL_LOAD:\w+\]", "", response).strip()
            if len(stripped) == 0 and len(tool_calls) == 0:
                # Pure SKILL_LOAD — keep looping so next turn sees the detail
                return False
            # SKILL_LOAD plus other content — completion follows the tool-call rule
            return len(tool_calls) == 0
        return len(tool_calls) == 0

    @staticmethod
    def _format_final(chunks: list[str], decision: StopDecision) -> str:
        base = "\n".join(chunks) if chunks else ""
        stop = decision.summary or decision.reason
        if stop:
            return f"{base}\n\n---\n{stop}".strip()
        return base.strip()