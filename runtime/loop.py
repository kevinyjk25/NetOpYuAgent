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
    from memory.router import MemoryRouter
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
    "restart", "rollback", "delete", "drain", "failover", "flush",
    "reboot", "terminate", "shutdown", "wipe", "reset",
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


def _call_key(tool_name: str, tool_args: dict) -> str:
    """
    Deduplicate tool calls by name+args fingerprint, not just name.
    This allows calling validate_device_config(ap-01) and validate_device_config(ap-02)
    in the same session without the second being blocked as a duplicate.
    Only blocks genuinely identical calls (same tool, same arguments).
    """
    import json as _json
    try:
        args_sig = _json.dumps(tool_args, sort_keys=True)
    except Exception:
        args_sig = str(tool_args)
    return f"{tool_name}|{args_sig}"

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

    # ------------------------------------------------------------------
    # Classify
    # ------------------------------------------------------------------

    def classify(self, query: str) -> ComplexityDecision:
        q = query.lower()
        if any(kw in q for kw in _DESTRUCTIVE_KEYWORDS):
            return ComplexityDecision(
                complexity=QueryComplexity.COMPLEX,
                reason="Destructive action detected — requires HITL approval",
                confidence=0.95, model_tier="full_model",
            )
        if any(kw in q for kw in _P0P1_KEYWORDS):
            return ComplexityDecision(
                complexity=QueryComplexity.COMPLEX,
                reason="P0/P1 severity — requires full HITL + DAG orchestration",
                confidence=0.90, model_tier="full_model",
            )
        if any(kw in q for kw in _PARALLEL_KEYWORDS):
            return ComplexityDecision(
                complexity=QueryComplexity.COMPLEX,
                reason="Parallel multi-entity analysis — requires TaskPlanner DAG",
                confidence=0.85, model_tier="full_model",
            )
        # P2: fast model hint for simple lookups
        tier = "fast_model" if any(kw in q for kw in _FAST_MODEL_KEYWORDS) else "full_model"
        return ComplexityDecision(
            complexity=QueryComplexity.SIMPLE,
            reason="Single-intent diagnostic query — Runtime Loop sufficient",
            confidence=0.80, model_tier=tier,
        )

    # ------------------------------------------------------------------
    # Pre-verification (P1)
    # ------------------------------------------------------------------

    async def pre_verify(
        self,
        query: str,
        confirmed_facts: list[str],
        env_context: dict[str, Any],
    ) -> VerificationResult:
        """
        Pre-action verification: check the request is safe to proceed with.
        Replace with real checks (policy engine, ACL) in production.
        """
        q = query.lower()

        # Check: destructive action in non-SIMPLE path should never reach here
        if any(kw in q for kw in _DESTRUCTIVE_KEYWORDS):
            return VerificationResult.fail(
                "Destructive action reached pre_verify — escalate to HITL"
            )

        # Check: change window (from env_context)
        if env_context.get("change_window") is False:
            if any(kw in q for kw in {"restart", "rollback", "delete", "flush"}):
                return VerificationResult.fail(
                    "Change window is closed — operation not permitted"
                )

        # Check: production restriction
        if env_context.get("allow_destructive") is False:
            if "production" in q or "prod" in q:
                if any(kw in q for kw in _DESTRUCTIVE_KEYWORDS):
                    return VerificationResult.fail(
                        "Destructive operations on production are restricted"
                    )

        return VerificationResult.ok("Pre-verification passed")

    # ------------------------------------------------------------------
    # Post-verification (P1)
    # ------------------------------------------------------------------

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
            # Stub: should call service_health tool to verify recovery
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

        # P1: pre-verification
        if self._cfg.enable_pre_verification:
            pre = await self.pre_verify(query, state.confirmed_facts, env_ctx)
            if not pre.passed:
                return LoopResult(
                    outcome=StopOutcome.STOP_HITL,
                    final_response=f"Pre-verification failed: {pre.reason}",
                    confirmed_facts=state.confirmed_facts,
                    verification=pre,
                )

        chunks: list[str] = []
        last_tool_result  = ""
        tool_outputs: dict[str, str] = {}   # persists across turns — tool results feed next LLM call
        # Seed called_tools from any prior tool calls visible in memory context.
        # This prevents the LLM from re-calling ap-01 when memory already has
        # its config from a previous stream() invocation.
        # called_tools uses _call_key(name, args) fingerprints — not bare names.
        # This allows validate_device_config(ap-01) and validate_device_config(ap-02)
        # to both execute within one session.
        called_tools: set[str] = set()

        while True:
            state.turns += 1
            memory_results = await self._retrieve_memory(query, session_id)

            # NOTE: We intentionally do NOT seed called_tools from memory context.
            # Memory may mention [TOOL: validate_device_config] from a prior call on
            # device X, but we still need to call it for devices Y and Z.
            # Deduplication is now by _call_key (name+args), so the same tool with
            # different device_id args is allowed. Same call with same args is blocked.

            # P2: skill catalog summary always prepended (cache-stable prefix)
            skill_section = ""
            if self._skill_catalog:
                skill_section = self._skill_catalog.format_summary()

            context_str = self._budget.assemble(
                memory_results=memory_results,
                tool_outputs=tool_outputs,       # pass accumulated results to LLM
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
            for skill_id in re.findall(r"\[SKILL_LOAD:(\w+)\]", llm_response):
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
    ) -> AsyncIterator[dict[str, Any]]:
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

        if self._cfg.enable_pre_verification:
            pre = await self.pre_verify(query, state.confirmed_facts, env_ctx)
            if not pre.passed:
                yield {"message": f"Pre-verification failed: {pre.reason}", "node": "pre_verify"}
                return

        tool_outputs: dict[str, str] = {}   # persists across turns
        called_tools: set[str] = set()       # dedup guard

        while True:
            state.turns += 1
            memory_results = await self._retrieve_memory(query, session_id)

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

            context_str = self._budget.assemble(
                memory_results=memory_results,
                tool_outputs=tool_outputs,       # accumulated tool results feed LLM
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
                # Q4 ambiguity: if top skills are too similar, escalate to HITL
                # Only when explicitly enabled via HITL_SKILL_AMBIGUITY=true env var
                import os as _os
                if skill_ambiguous and _os.getenv("HITL_SKILL_AMBIGUITY", "false").lower() == "true":
                    top2 = [sid for sid, _ in selected_skills[:2]]
                    yield {
                        "message": (
                            f"stop_hitl: ambiguous skill match — top skills {top2} have "
                            "nearly identical scores. Routing to HITL for clarification."
                        ),
                        "node": "hitl_gate",
                        "stop_hitl": True,
                        "reason": "skill_ambiguity",
                        "top_skills": top2,
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
                for word in _visible.split():
                    yield {"token": word + " "}
                    await asyncio.sleep(0)
            # If the entire response was tool calls (no prose), yield nothing —
            # the tool result will be injected in the next turn's context and
            # the LLM will produce a proper prose answer then.

            import re
            for skill_id in re.findall(r"\[SKILL_LOAD:(\w+)\]", llm_response):
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
                    yield {
                        "message": (
                            f"stop_hitl: tool '{tool_name}' is on the HITL watch-list "
                            "and requires human approval before execution. "
                            "Routing to HITL graph."
                        ),
                        "node": "hitl_gate",
                        "stop_hitl": True,
                        "tool_name": tool_name,
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
                return

            if self._is_complete(llm_response, new_tool_calls):
                return

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _retrieve_memory(self, query: str, session_id: str) -> list[Any]:
        if self._memory is None:
            return []
        try:
            from memory.schemas import RetrievalQuery
            rq = RetrievalQuery(
                query_text=query,
                session_id=session_id,
                max_tokens=self._cfg.budget.memory_tokens,
            )
            return await self._memory.retrieve(rq)
        except Exception as exc:
            logger.warning("RuntimeLoop: memory retrieval failed: %s", exc)
            return []

    async def _call_llm(self, query: str, context: str, state: LoopState) -> str:
        """
        Central LLM dispatch for the runtime loop.

        Development stub (no LLM installed):
          Turn 1  → emits one [TOOL:name] call matched by keyword sets
          Turn 2+ → tool results are in context → plain summary, no more calls

        Production: patch_runtime_loop(loop, engine) replaces this method
        entirely. The stub is never called after patching.

        Keyword sets (any()) replace the old if/elif chain — easier to
        extend and easier to test without touching control flow.
        """
        import json as _json
        import re as _re

        await asyncio.sleep(0)
        q = query.lower()

        has_results = (
            state.turns > 1
            or "[STORED:" in context
            or "Tool outputs:" in context
        )

        # ── Turn 2+: summarise from context, no tool call ─────────────
        if has_results:
            ctx = context.lower()
            if any(k in ctx for k in ("syslog", "error log", "log entry")):
                return (
                    "Based on syslog analysis:\n"
                    "Found error patterns in recent logs. "
                    "Recommend reviewing certificate expiry, interface flapping, "
                    "and RADIUS timeout. No immediate P0 condition."
                )
            if any(k in ctx for k in ("bgp", "neighbor", "route table")):
                return (
                    "BGP analysis complete:\n"
                    "Neighbour table retrieved. Sessions established. "
                    "No route flapping detected in last 5 minutes."
                )
            if any(k in ctx for k in ("dns", "resolv", "nxdomain")):
                return "DNS analysis: resolution operational. No NXDOMAIN or timeout errors."
            if any(k in ctx for k in ("device", "firmware", "cpu usage")):
                return "Device status: reachable and operational. CPU/memory within normal ranges."
            if any(k in ctx for k in ("incident", " p1 ", " p0 ")):
                return "Incident summary: 2 open P1 incidents — INC-1291 (RADIUS), INC-1305 (DNS)."
            return (
                "Analysis complete:\n"
                "Reviewed telemetry. No critical anomalies in current state."
            )

        # ── Turn 1: intent → tool mapping ─────────────────────────────
        # Each entry: (keyword_set, tool_name, default_args_dict)
        INTENT_MAP = [
            (
                {"netflow", "traffic", "flow", "bandwidth"},
                "netflow_dump",
                {"site": "site-a", "flows": 500},
            ),
            (
                {"metric", "prometheus", "utilisa", "cpu", "memory"},
                "prometheus_query",
                {"metric": "up", "job": "network_devices", "range_minutes": 60},
            ),
            (
                {"dns", "resolv", "nslookup"},
                "dns_lookup",
                {"hostname": "payments.internal"},
            ),
            (
                {"auth", "radius", "certificate", "cert", "login"},
                "syslog_search",
                {"host": "radius-*", "severity": "error", "lines": 50},
            ),
            (
                {"bgp", "routing", "prefix", "neighbor"},
                "get_bgp_summary",
                {"router": "router-01"},
            ),
            (
                {"incident", "ticket", "open incident"},
                "list_incidents",
                {"severity": "P1", "status": "open"},
            ),
            (
                {"ip address", "ipam", "subnet"},
                "search_ip_addresses",
                {"prefix": "10.0.0.0/8"},
            ),
            # syslog is last so more-specific intents (auth, radius) match first
            (
                {"syslog", "log", "error", "alert"},
                "syslog_search",
                {"host": "ap-*", "severity": "error", "lines": 100},
            ),
        ]

        for keywords, tool_name, default_args in INTENT_MAP:
            if any(k in q for k in keywords):
                args_str = _json.dumps(default_args)
                return (
                    f"Analysing: {query}\n"
                    f"[TOOL:{tool_name}] {args_str}\n"
                    "Retrieving data for analysis."
                )

        # Device / AP / switch queries — extract device ID from query
        if any(k in q for k in ("device", "status", "switch", "interface", "port")):
            m = _re.search(r"(ap-\d+|sw-[\w-]+|router-\d+)", q)
            dev = m.group(1) if m else "ap-01"
            args_str = _json.dumps({"device_id": dev})
            return (
                f"Checking device: {query}\n"
                f"[TOOL:get_device_status] {args_str}\n"
                "Fetching live device metrics."
            )

        # No matching intent — general query, no tool needed
        return (
            f"Analysing: {query}\n"
            "Based on system context, this appears to be a general operational query. "
            "All monitored services are nominal. For targeted diagnostics, "
            "provide a device name, time range, or affected service."
        )



    @staticmethod
    def _strip_thinking(response: str) -> str:
        """
        Remove <think>...</think> blocks emitted by thinking models
        (qwen3, deepseek-r1, etc.) before tool parsing or display.
        Preserves everything outside the think block.
        """
        import re
        # Remove <think>...</think> (may be multiline, non-greedy)
        cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        # Also handle /think variant
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
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
    def _is_complete(response: str, tool_calls: list) -> bool:
        # If the LLM emitted a SKILL_LOAD directive, it needs another turn
        # to read the loaded detail and then call the actual tools.
        # Do NOT terminate on SKILL_LOAD turns even though tool_calls is empty.
        import re as _re
        if _re.search(r"\[SKILL_LOAD:\w+\]", response):
            return False
        return len(tool_calls) == 0

    @staticmethod
    def _format_final(chunks: list[str], decision: StopDecision) -> str:
        base = "\n".join(chunks) if chunks else ""
        stop = decision.summary or decision.reason
        if stop:
            return f"{base}\n\n---\n{stop}".strip()
        return base.strip()