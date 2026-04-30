"""
hitl/a2a_integration.py  [v4 — Dual-Track Memory integrated]
-------------------------------------------------------------
Changes from v3
---------------
  1. DualTrackMemory (DTM) integrated as the primary recall/write path.
       - recall:    dtm.recall() replaces curator.recall_for_session()
       - write:     dtm.after_turn() replaces separate fts + curator calls
       - fallback:  if dtm is None, all v3 Hermes hooks run as before
         (full backward compatibility — no breakage if DTM not wired)
  2. Constructor accepts optional `dtm` parameter (DualTrackMemory instance).
  3. _hermes_post_turn() delegates to DTM when available, else runs
     the original three-task concurrent pattern (fts + curator + user_model).
  4. _recall() is a single entry point for both SIMPLE and COMPLEX paths.
  5. All v3 features preserved: HITL graph, runtime loop dual routing,
     post-action verification, skill evolver, task bridge, memory helpers.

Backward compatible: all new parameters are Optional.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Optional

from a2a.agent_executor import AgentExecutor, A2AEventProcessor, DEFAULT_PROCESSORS
from a2a.event_queue import EventQueue, RequestContext
from a2a.schemas import (
    Artifact, DataPart, Message, MessageEvent,
    TaskArtifactUpdateEvent, TaskState, TaskStatus,
    TaskStatusUpdateEvent, TextPart,
)
from hitl.audit import HitlAuditService
from hitl.decision import HitlDecisionRouter
from hitl.graph import build_hitl_graph, run_with_hitl
from hitl.review import HitlReviewService
from hitl.schemas import (
    AuditEventKind, HitlAuditRecord, HitlPayload,
    HitlInterruptEvent, ProposedAction, RiskLevel, TriggerKind,
)
from hitl.triggers import HitlConfig
from runtime import (
    AgentRuntimeLoop, QueryComplexity, RuntimeConfig,
    StopOutcome, ContextBudgetManager, BudgetConfig,
    DeviceRef,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# A2A event processor for HITL interrupt chunks
# ---------------------------------------------------------------------------

class HitlA2AEventProcessor(A2AEventProcessor):
    async def process(self, chunk, event_queue, task_id, context_id):
        if not chunk.get("hitl_interrupt"):
            return
        review_url = f"http://localhost:3000/hitl/review/{chunk.get('interrupt_id', '')}"
        event_data = {
            "kind":            "hitl_interrupt",
            "interrupt_id":    chunk.get("interrupt_id", ""),
            "trigger_kind":    chunk.get("trigger_kind", ""),
            "risk_level":      chunk.get("risk_level", ""),
            "summary":         chunk.get("summary", ""),
            "proposed_action": chunk.get("proposed_action", {}),
            "review_url":      review_url,
            "tag":             "hitl_interrupt",
        }
        await event_queue.enqueue_event(TaskArtifactUpdateEvent(
            task_id=task_id, context_id=context_id,
            artifact=Artifact(
                name="hitl_interrupt",
                description=f"Human review required: {chunk.get('trigger_kind', '')}",
                parts=[DataPart(data=event_data)],
                metadata={"review_url": review_url},
            ),
        ))
        logger.info("HitlA2AEventProcessor: interrupt_id=%s", chunk.get("interrupt_id"))


def build_hitl_processors() -> list[A2AEventProcessor]:
    from a2a.agent_executor import (
        A2ATokenProcessor, A2ABatchTokenProcessor,
        A2AMessageProcessor, A2ANodeResultProcessor, A2ANodeStepProcessor,
    )
    return [
        A2ATokenProcessor(), A2ABatchTokenProcessor(),
        A2AMessageProcessor(), A2ANodeResultProcessor(),
        A2ANodeStepProcessor(), HitlA2AEventProcessor(),
    ]


# ---------------------------------------------------------------------------
# Main executor
# ---------------------------------------------------------------------------

class ITOpsHitlAgentExecutor(AgentExecutor):
    """
    v4: Routes SIMPLE queries to AgentRuntimeLoop (thin fast path)
        and COMPLEX/destructive queries to HITL graph + TaskPlanner.

    Memory strategy (priority order):
      1. DualTrackMemory (dtm) — if wired, handles ALL recall and write.
         Runs Track A (FTS5 chunks + daily .md files) and Track B
         (curated facts) in parallel, arbitrates with MMR + temporal decay.
      2. Fallback Hermes v3 — if dtm is None, runs the original three
         concurrent hooks: fts_store.write_turn, curator.after_turn,
         user_model.after_turn. All v3 code paths are fully preserved.

    Hermes learning loop modules (all Optional):
      fts_store:      FTS5SessionStore  — raw turn storage (Track A)
      memory_curator: MemoryCurator     — LLM fact extraction (Track B)
      user_model:     UserModelEngine   — behavioral profile
      skill_evolver:  SkillEvolver      — autonomous skill creation
      skill_catalog:  SkillCatalogService
      dtm:            DualTrackMemory   — converged dual-track memory (v4)
    """

    def __init__(
        self,
        hitl_router:    HitlDecisionRouter,
        review_service: HitlReviewService,
        audit_service:  HitlAuditService,
        hitl_config:    Optional[HitlConfig]      = None,
        memory_router:  Optional[Any]             = None,
        task_system:    Optional[Any]             = None,
        runtime_config: Optional[RuntimeConfig]   = None,
        tool_registry:  Optional[dict[str, Any]]  = None,
        # ── Hermes v3 modules (fallback when dtm is None) ─────────────
        fts_store:      Optional[Any]             = None,
        memory_curator: Optional[Any]             = None,
        user_model:     Optional[Any]             = None,
        skill_evolver:  Optional[Any]             = None,
        skill_catalog:  Optional[Any]             = None,
        # ── v4: Dual-Track Memory ──────────────────────────────────────
        dtm:            Optional[Any]             = None,
        # ── LLM engine for post-HITL llm_answer fallback ───────────────
        llm_engine:     Optional[Any]             = None,
    ) -> None:
        self._hitl_router    = hitl_router
        self._review_service = review_service
        self._audit_service  = audit_service
        self._hitl_config    = hitl_config or HitlConfig()
        self._memory         = memory_router
        self._task_system    = task_system
        self._tool_registry  = tool_registry or {}
        self._graph          = build_hitl_graph(self._hitl_config)
        self._processors     = build_hitl_processors()
        self._cancelled:     dict[str, bool] = {}

        # ── Hermes v3 modules ─────────────────────────────────────────
        self._fts_store      = fts_store
        self._curator        = memory_curator
        self._user_model     = user_model
        self._skill_evolver  = skill_evolver
        self._skill_catalog  = skill_catalog

        # ── v4: DTM — primary recall/write path ───────────────────────
        self._dtm            = dtm

        # ── LLM engine for fallback post-HITL answer generation ───────
        self._llm_engine     = llm_engine

        self._runtime = AgentRuntimeLoop(
            memory_router=memory_router,
            config=runtime_config,
        )
        cfg = runtime_config or RuntimeConfig()
        self._budget = ContextBudgetManager(cfg.budget)

        if dtm:
            logger.info("ITOpsHitlAgentExecutor: DualTrackMemory wired (v4 recall path)")
        else:
            logger.info("ITOpsHitlAgentExecutor: Hermes v3 recall path (no DTM)")

    # ------------------------------------------------------------------
    # Top-level execute — routes SIMPLE vs COMPLEX
    # ------------------------------------------------------------------

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id    = context.task_id
        context_id = context.context_id
        query      = context.get_user_input()
        session_id = self._extract_session_id(context)

        logger.info("ITOpsHitlAgentExecutor.execute task_id=%s", task_id)
        self._cancelled[task_id] = False

        await event_queue.enqueue_event(TaskStatusUpdateEvent(
            task_id=task_id, context_id=context_id,
            status=TaskStatus(state=TaskState.WORKING),
        ))

        try:
            # ── Fast path: multi-mode HITL (USER_CHOICE / CLARIFICATION / EDIT) ──
            # When the runtime loop's gates detect ambiguity / low-confidence /
            # editable destructive op, the backend forwards hitl_kind +
            # supporting fields here. We fire the appropriate interrupt
            # via the new helper API and register a callback that re-runs
            # the agent on the operator's resolved input.
            _kind = context.metadata.get("hitl_kind", "")
            if _kind in ("user_choice", "clarification"):
                await self._execute_multi_mode_hitl(
                    kind       = _kind,
                    query      = query,
                    session_id = session_id,
                    context    = context,
                    event_queue= event_queue,
                    task_id    = task_id,
                    context_id = context_id,
                )
                return

            # ── Fast path: force_hitl_tool set by stop_hitl in loop.py ──────────
            # When the runtime loop detects a requires_hitl tool, it emits stop_hitl
            # and the backend re-routes here with force_hitl_tool in context.metadata.
            # Skip complexity classification entirely — go straight to the interrupt.
            if context.metadata.get("force_hitl_tool"):
                await self._execute_complex(
                    query, session_id, context, event_queue, task_id, context_id
                )
                return

            decision = self._runtime.classify(query)
            logger.info(
                "Complexity: %s — %s (task_id=%s)",
                decision.complexity.value, decision.reason, task_id,
            )

            # COMPLEX routing is appropriate for:
            #   - Destructive actions (restart, rollback, delete...)
            #   - True P0/P1 incidents
            #   - Parallel multi-entity DAG tasks
            # It is NOT appropriate for:
            #   - Summary/format requests ("make a table", "translate to Chinese")
            #   - Analysis requests that only need the LLM to synthesize prior results
            # For the latter, _execute_simple handles it correctly via loop.stream().
            # Re-check: if COMPLEX but reason is "P0/P1" or "Parallel" (not destructive),
            # and there is no force_hitl_tool, use _execute_simple as a safe fallback
            # because the HITL graph executor_node has no tool to run and returns empty.
            _use_simple = (
                decision.complexity == QueryComplexity.SIMPLE
                or (
                    decision.complexity == QueryComplexity.COMPLEX
                    and not context.metadata.get("force_hitl_tool")
                    and "Destructive" not in decision.reason
                )
            )
            if _use_simple:
                await self._execute_simple(
                    query, session_id, context, event_queue, task_id, context_id
                )
            else:
                await self._execute_complex(
                    query, session_id, context, event_queue, task_id, context_id
                )
        except Exception as exc:
            logger.exception("ITOpsHitlAgentExecutor error task_id=%s: %s", task_id, exc)
            await event_queue.enqueue_event(TaskStatusUpdateEvent(
                task_id=task_id, context_id=context_id,
                status=TaskStatus(state=TaskState.FAILED, message=str(exc)),
            ))
            await event_queue.enqueue_event(MessageEvent(
                task_id=task_id, context_id=context_id,
                message=Message(role="assistant", parts=[TextPart(text=f"Task failed: {exc}")]),
            ))
        finally:
            self._cancelled.pop(task_id, None)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        self._cancelled[context.task_id] = True
        await event_queue.enqueue_event(TaskStatusUpdateEvent(
            task_id=context.task_id, context_id=context.context_id,
            status=TaskStatus(state=TaskState.CANCELED),
        ))
        await event_queue.enqueue_event(MessageEvent(
            task_id=context.task_id, context_id=context.context_id,
            message=Message(role="assistant", parts=[TextPart(text="Task cancelled.")]),
        ))

    # ------------------------------------------------------------------
    # Path A: Runtime Loop (SIMPLE queries)
    # ------------------------------------------------------------------

    async def _execute_simple(
        self, query, session_id, context, event_queue, task_id, context_id
    ) -> None:
        env_ctx    = context.metadata.get("env_context", {})
        facts      = context.metadata.get("confirmed_facts", [])
        ws_raw     = context.metadata.get("working_set", [])
        working_set = [
            DeviceRef(id=d["id"], label=d.get("label", d["id"]))
            for d in ws_raw
            if isinstance(d, dict) and "id" in d
        ]

        # ── Recall ────────────────────────────────────────────────────
        past_context = await self._recall(query, session_id)
        user_profile = ""
        if self._user_model and not self._dtm:
            # When DTM is wired it folds user profile into prompt_context.
            # When DTM is absent, inject user profile separately (v3 path).
            try:
                user_profile = self._user_model.get_prompt_section(session_id)
            except Exception as exc:
                logger.debug("User profile injection skipped: %s", exc)

        if past_context or user_profile:
            extra = [x for x in [past_context, user_profile] if x]
            env_ctx = {**env_ctx, "_hermes_context": "\n\n".join(extra)}

        response_chunks: list[str] = []

        async for chunk in self._runtime.stream(
            query=query,
            session_id=session_id,
            env_context=env_ctx,
            confirmed_facts=facts,
            working_set=working_set,
            tool_registry=self._tool_registry,
        ):
            if self._cancelled.get(task_id):
                break
            for processor in self._processors:
                await processor.process(chunk, event_queue, task_id, context_id)
            if "token" in chunk:
                response_chunks.append(chunk["token"])
            msg = chunk.get("message", "")
            if chunk.get("stop_hitl") or "stop_hitl" in msg or "HITL" in msg:
                # ── Multi-mode HITL routing ──────────────────────────────
                # Loop emitted a stop_hitl chunk. Inspect hitl_kind to pick
                # the right interaction mode. Falls back to _execute_complex
                # (vanilla approve/reject) for stop_hitl chunks that don't
                # specify a hitl_kind.
                _kind = chunk.get("hitl_kind")
                if _kind == "user_choice":
                    await self._handle_user_choice_chunk(
                        chunk, query, session_id, context, event_queue, task_id
                    )
                elif _kind == "clarification":
                    await self._handle_clarification_chunk(
                        chunk, query, session_id, context, event_queue, task_id
                    )
                elif _kind == "edit":
                    # editable destructive tool — let the executor route
                    # through _execute_complex, which already builds the
                    # HitlPayload, but pass editable_param_keys via
                    # context.metadata so _handle_interrupt_chunk picks
                    # them up.
                    context.metadata["editable_param_keys"] = chunk.get(
                        "editable_param_keys", []
                    )
                    context.metadata["force_hitl_tool"] = chunk.get("tool_name", "")
                    context.metadata["force_hitl_args"] = chunk.get("tool_args", {})
                    await self._execute_complex(
                        query, session_id, context, event_queue, task_id, context_id
                    )
                else:
                    # Default: classic approve/reject HITL via complex graph
                    await self._execute_complex(
                        query, session_id, context, event_queue, task_id, context_id
                    )
                return

        assistant_text = "".join(response_chunks) or "Analysis complete."
        await event_queue.enqueue_event(TaskStatusUpdateEvent(
            task_id=task_id, context_id=context_id,
            status=TaskStatus(state=TaskState.COMPLETED),
        ))
        await event_queue.enqueue_event(MessageEvent(
            task_id=task_id, context_id=context_id,
            message=Message(role="assistant", parts=[TextPart(text=assistant_text)]),
        ))
        await self._write_back_memory(session_id, query, assistant_text)

        tool_calls = self._extract_tool_calls_from_chunks(response_chunks)
        await self._hermes_post_turn(session_id, query, assistant_text, tool_calls)

        # Fire skill evolver for SIMPLE path too — if the turn used at least one
        # tool call it may be worth capturing as a reusable skill recipe.
        if self._skill_evolver and tool_calls:
            try:
                await self._skill_evolver.after_task(
                    task_description = query,
                    solution_summary = assistant_text[:400],
                    tools_used       = [tc.get("tool", "") for tc in tool_calls if tc.get("tool")],
                    solution_steps   = [],
                    key_observations = [],
                    complexity       = 5.0,   # simple path: lower complexity → higher bar for skill creation
                    session_id       = session_id,
                )
            except Exception as exc:
                logger.debug("SkillEvolver.after_task (simple path) skipped: %s", exc)

    # ------------------------------------------------------------------
    # Path B: Full HITL graph (COMPLEX / destructive queries)
    # ------------------------------------------------------------------

    async def _execute_multi_mode_hitl(
        self,
        *,
        kind:       str,                # "user_choice" | "clarification"
        query:      str,
        session_id: str,
        context:    "RequestContext",
        event_queue,
        task_id:    str,
        context_id: str,
    ) -> None:
        """Fire a USER_CHOICE or CLARIFICATION interrupt that the runtime
        loop has surfaced via env metadata. On operator resume, re-run
        the agent loop with the resolved input folded into env_context
        so the second pass actually executes the work.
        """
        meta = context.metadata or {}
        summary = meta.get("hitl_summary") or "Operator input required"

        # Capture the bits we need for the resume callback. Use locals
        # rather than `self.*` so the closure stays tied to THIS request
        # even if a concurrent request mutates the executor.
        #
        # IMPORTANT — two runtime loop attributes exist on the executor:
        #   - self._runtime       : built in __init__ for classify(), NOT
        #                            patched with the LLM engine. Calling
        #                            .stream() on it raises "AgentRuntimeLoop.
        #                            _call_llm has not been patched".
        #   - self._runtime_loop  : injected at startup by main.py, IS the
        #                            patched loop instance from
        #                            services["runtime_loop"] — this is what
        #                            actually has _call_llm wired to the LLM.
        # Always read self._runtime_loop here. Fall back to self._runtime
        # only if main.py didn't run the wiring (defensive — would have
        # failed previously with NameError).
        _runtime      = (
            getattr(self, "_runtime_loop", None)
            or getattr(self, "_runtime", None)
        )
        _llm_engine   = self._llm_engine
        _tool_reg     = self._tool_registry or {}
        _memory       = getattr(self, "_memory", None)

        async def _resume_callback(decision):
            """Re-run the agent loop with the operator's input applied."""
            from hitl.schemas import DecisionKind
            extra_env: dict = {"_clarification_resolved": True}
            resolved_query = query

            if kind == "user_choice":
                if decision.decision != DecisionKind.CHOOSE:
                    return {
                        "tool":  "user_choice_cancelled",
                        "result": f"Operator did not pick a choice (decision={decision.decision}).",
                    }
                # Resolve the picked choice — append a hint to the query so
                # the agent loop knows which option to take. We don't try to
                # be clever about substituting into the query; we annotate
                # so the LLM sees the disambiguation.
                picked = decision.selected_choice_id
                resolved_query = (
                    f"{query}\n\n"
                    f"[OPERATOR DISAMBIGUATION] Use specifically: {picked}"
                )
                logger.info(
                    "Multi-mode HITL: user_choice resolved — picked=%s", picked,
                )

            elif kind == "clarification":
                if decision.decision != DecisionKind.ANSWER:
                    return {
                        "tool":  "clarification_cancelled",
                        "result": f"Operator did not provide answers (decision={decision.decision}).",
                    }
                answers = decision.clarification_answers or {}
                # Append answers as a clear annotation so the agent loop's
                # next pass treats them as authoritative facts.
                ans_lines = "\n".join(f"  - {k}: {v}" for k, v in answers.items() if v)
                resolved_query = (
                    f"{query}\n\n"
                    f"[OPERATOR-PROVIDED CLARIFICATIONS]\n{ans_lines}"
                )
                logger.info(
                    "Multi-mode HITL: clarification answered — keys=%s",
                    list(answers.keys()),
                )

            else:
                return {"tool": "unknown_hitl_kind", "result": kind}

            # Run the agent loop on the resolved query. Pre-recall memory
            # so the second pass has full context (mirrors _post_hitl_agent_callback).
            if _runtime is None:
                return {
                    "tool":  kind,
                    "result": (
                        f"Operator input recorded ({kind}), but no runtime loop "
                        f"available to act on it. Resolved query:\n{resolved_query}"
                    ),
                }

            _fts_ctx = ""
            if _memory is not None:
                try:
                    rec = await _memory.recall(resolved_query, session_id)
                    _fts_ctx = getattr(rec, "prompt_context", "") or ""
                except Exception as exc:
                    logger.debug("Resume recall skipped: %s", exc)

            tokens: list[str] = []
            try:
                async for chunk in _runtime.stream(
                    query           = resolved_query,
                    session_id      = session_id,
                    env_context     = {
                        "_fts_context":             _fts_ctx,
                        "_clarification_resolved":  True,
                    },
                    tool_registry   = _tool_reg,
                    skip_pre_verify = True,    # operator already confirmed
                ):
                    if "token" in chunk:
                        tokens.append(chunk["token"])
                full_text = "".join(tokens).strip()
                if not full_text:
                    full_text = (
                        f"[{kind} resolved but agent produced no output]\n"
                        f"resolved_query={resolved_query[:200]}"
                    )
                # Persist the resolved turn to memory ourselves — backend's
                # _submit_hitl_decision will also write a HITL-tagged turn,
                # but that's about the decision audit, not the work output.
                # The agent's full_text IS the work output and belongs in
                # the conversation history.
                logger.info(
                    "Multi-mode HITL: agent loop produced %d chars after %s resolution",
                    len(full_text), kind,
                )
                return {"tool": "agent_loop", "result": full_text}
            except Exception as exc:
                logger.exception("Multi-mode HITL resume agent loop failed: %s", exc)
                return {"tool": "agent_loop", "error": str(exc)}

        # Fire the right interrupt type
        if kind == "user_choice":
            choices = meta.get("hitl_choices") or []
            if not choices:
                logger.warning("user_choice triggered but no choices supplied — falling back to simple")
                await self._execute_simple(query, session_id, context, event_queue, task_id, context_id)
                return
            await self.trigger_user_choice(
                summary    = summary,
                choices    = choices,
                callback   = _resume_callback,
                context    = context,
                session_id = session_id,
                event_queue= event_queue,
                task_id    = task_id,
            )
        elif kind == "clarification":
            fields = meta.get("hitl_clarification_fields") or []
            if not fields:
                logger.warning("clarification triggered but no fields supplied — falling back to simple")
                await self._execute_simple(query, session_id, context, event_queue, task_id, context_id)
                return
            await self.trigger_clarification(
                summary    = summary,
                fields     = fields,
                callback   = _resume_callback,
                context    = context,
                session_id = session_id,
                event_queue= event_queue,
                task_id    = task_id,
            )

        # Mark task as awaiting input (UI will show ⏸ awaiting_hitl)
        await event_queue.enqueue_event(TaskStatusUpdateEvent(
            task_id=task_id, context_id=context_id,
            status=TaskStatus(state=TaskState.INPUT_REQUIRED),
        ))


    async def _execute_complex(
        self, query, session_id, context, event_queue, task_id, context_id
    ) -> None:
        # ── Recall ────────────────────────────────────────────────────
        past_context   = await self._recall(query, session_id)
        memory_context = await self._build_memory_context(context, session_id)
        if past_context:
            memory_context = past_context + "\n\n" + memory_context

        response_chunks: list[str] = []

        # ── Fast path: force_hitl_tool bypasses the LangGraph graph ──
        # When stop_hitl fires in loop.py for a requires_hitl tool, the backend
        # sets force_hitl_tool in context.metadata. We build the HitlPayload
        # directly here — no LangGraph interrupt() needed, works on all versions.
        force_tool = context.metadata.get("force_hitl_tool", "")
        force_args = context.metadata.get("force_hitl_args") or {}
        if force_tool:
            import uuid as _uuid
            from hitl.schemas import (
                HitlPayload, ProposedAction, TriggerKind, RiskLevel, InterruptState,
            )
            interrupt_id = str(_uuid.uuid4())
            payload = HitlPayload(
                interrupt_id    = interrupt_id,
                thread_id       = context_id,
                context_id      = context_id,
                task_id         = task_id,
                trigger_kind    = TriggerKind.DESTRUCTIVE_OP,
                risk_level      = RiskLevel.HIGH,
                confidence_score= 0.95,
                user_query      = query,
                intent_summary  = (
                    f"Tool '{force_tool}' requires human approval before execution.\n"
                    f"Args: {force_args}"
                ),
                proposed_action = ProposedAction(
                    action_type  = f"tool_call:{force_tool}",
                    target       = force_tool,
                    parameters   = force_args,
                    risk_summary = f"'{force_tool}' is on the HITL watch-list — approval required.",
                    reversible   = False,
                ),
                sla_seconds     = 600,
            )
            # Emit the interrupt chunk — _handle_interrupt_chunk calls
            # register_interrupt which stores it in _payload_store automatically
            # Pull editable_param_keys from context.metadata if the caller
            # routed an "edit" hitl_kind through here. This lets the UI show
            # inline editors for the operator to tweak params before approving.
            _editable_keys = list(context.metadata.get("editable_param_keys") or [])

            # Auto-derive sensible defaults if the caller didn't specify.
            # For destructive config tools, the operator almost always wants
            # to be able to tweak the actual config_lines / args before
            # approving — that's the whole point of edit-before-approve.
            # This makes EDIT-mode the default for tools where it makes
            # sense, without forcing every caller to spell out the keys.
            if not _editable_keys and isinstance(force_args, dict):
                _candidate_keys = []
                # Hand-curated mapping by tool type. Keys we know are
                # safe + sensible for an operator to tweak before approve.
                _by_tool = {
                    "edit_device_config":   ["config_lines", "reason", "section"],
                    "push_config":           ["config_lines", "device_id", "reason"],
                    "restart_service":       ["service", "scope", "reason"],
                    "rollback_deploy":       ["target_version", "reason"],
                    "drain_node":            ["node_id", "drain_timeout_seconds"],
                    "failover":              ["primary", "secondary", "reason"],
                    "delete_resource":       ["resource_id", "reason"],
                }
                for k in _by_tool.get(force_tool, []):
                    if k in force_args:
                        _candidate_keys.append(k)
                # Generic fallback: editable keys = top-level dict params,
                # excluding the device_id which the operator generally
                # shouldn't repoint (that would change scope, not just args).
                if not _candidate_keys:
                    for k, v in force_args.items():
                        if k == "device_id":
                            continue
                        if isinstance(v, (str, list, dict, int, float)):
                            _candidate_keys.append(k)
                _editable_keys = _candidate_keys
                logger.info(
                    "force_hitl_tool: auto-derived editable_param_keys=%s for tool=%s",
                    _editable_keys, force_tool,
                )

            interrupt_chunk = {
                "hitl_interrupt": True,
                "interrupt_id":   interrupt_id,
                "trigger_kind":   payload.trigger_kind.value,
                "risk_level":     payload.risk_level.value,
                "summary":        payload.intent_summary,
                "proposed_action": payload.proposed_action.model_dump(),
                # New: only set when the caller specifically routed an EDIT
                # — empty list => UI shows the classic approve/reject panel.
                "editable_param_keys": _editable_keys,
                "thread_id":      context_id,
                "node":           "hitl",
                "tag":            "hitl_interrupt",
                "kind":           "hitl_interrupt",
            }
            await self._handle_interrupt_chunk(interrupt_chunk, context, session_id)
            for processor in self._processors:
                await processor.process(interrupt_chunk, event_queue, task_id, context_id)

            # Register a direct callback so the decision router can execute the tool
            # after the operator clicks Approve — without needing LangGraph resume.
            _tool_reg = getattr(self, "_tool_registry", {}) or {}
            async def _approved_tool_callback(
                decision=None,
                _tool=force_tool, _args=force_args, _reg=_tool_reg,
                _sid=session_id, _q=query,
            ):
                if _tool not in _reg:
                    return {"error": f"Tool {_tool!r} not in registry after approval"}
                # Merge operator's edits into args if this was an EDIT decision.
                # Without this, the operator's edits in the UI are silently
                # discarded and the tool runs with the original proposed args.
                _final_args = dict(_args)
                if decision is not None and getattr(decision, "parameter_patch", None):
                    _patch = decision.parameter_patch or {}
                    _final_args = {**_final_args, **_patch}
                    logger.info(
                        "force_hitl callback: applying operator edit — "
                        "patch_keys=%s",
                        list(_patch.keys()),
                    )
                try:
                    raw = await _reg[_tool](_final_args)
                    await self._hermes_post_turn(
                        _sid, _q,
                        f"[TOOL:{_tool}] approved and executed. Result: {str(raw)[:300]}", []
                    )
                    return {"tool": _tool, "args": _final_args, "result": str(raw)}
                except Exception as exc:
                    return {"tool": _tool, "error": str(exc)}

            import time as _time_cb
            # Prune stale callbacks (>30 min) before registering
            _cb_ttl = 1800
            _cb_now = _time_cb.monotonic()
            self._hitl_router._direct_callbacks = {
                k: v for k, v in self._hitl_router._direct_callbacks.items()
                if _cb_now - (v[1] if isinstance(v, tuple) else 0) < _cb_ttl
            }
            self._hitl_router._direct_callbacks[interrupt_id] = (_approved_tool_callback, _cb_now)

            await event_queue.enqueue_event(MessageEvent(
                task_id=task_id, context_id=context_id,
                message=Message(
                    role="assistant",
                    parts=[TextPart(text=(
                        f"⚠ HITL interrupt — human approval required.\n"
                        f"Tool: {force_tool}\n"
                        f"Args: {force_args}\n"
                        f"Interrupt ID: {interrupt_id}\n\n"
                        "Approval card is now in the HITL tab. "
                        "Click Approve or Reject to continue."
                    ))],
                ),
            ))
            await self._hermes_post_turn(session_id, query,
                                         f"HITL interrupt raised for {force_tool}", [])
            return

        async for chunk in run_with_hitl(
            query=query,
            thread_id=context_id,
            context_id=context_id,
            task_id=task_id,
            user_metadata={
                **context.metadata,
                "memory_context": memory_context,
                "session_id":     session_id,
            },
            hitl_config=self._hitl_config,
            graph=self._graph,
        ):
            if self._cancelled.get(task_id):
                break

            # HITL interrupt: register BEFORE letting processors emit
            # TaskArtifactUpdateEvent — avoids race where browser polls
            # /hitl/pending before register_interrupt() has run.
            if chunk.get("hitl_interrupt"):
                # Step 1: register → payload is now in _payload_store
                await self._handle_interrupt_chunk(chunk, context, session_id)
                # Step 2: processors emit TaskArtifactUpdateEvent
                for processor in self._processors:
                    await processor.process(chunk, event_queue, task_id, context_id)
                interrupt_id = chunk.get("interrupt_id", "?")
                trigger      = chunk.get("trigger_kind", "destructive_op")
                risk         = chunk.get("risk_level", "high")
                # Step 3: close queue so SSE stream ends cleanly
                await event_queue.enqueue_event(MessageEvent(
                    task_id=task_id, context_id=context_id,
                    message=Message(
                        role="assistant",
                        parts=[TextPart(text=(
                            f"⚠ HITL interrupt — human approval required.\n"
                            f"Trigger: {trigger}  Risk: {risk}\n"
                            f"Interrupt ID: {interrupt_id}\n\n"
                            "Approval card is now in the HITL tab. "
                            "Click Approve or Reject to continue execution."
                        ))],
                    ),
                ))
                return

            # Non-HITL chunks: normal processor pipeline
            for processor in self._processors:
                await processor.process(chunk, event_queue, task_id, context_id)
            if "token" in chunk:
                response_chunks.append(chunk["token"])

        await self._verify_action_result(context, session_id, response_chunks)
        await event_queue.enqueue_event(TaskStatusUpdateEvent(
            task_id=task_id, context_id=context_id,
            status=TaskStatus(state=TaskState.COMPLETED),
        ))
        await event_queue.enqueue_event(MessageEvent(
            task_id=task_id, context_id=context_id,
            message=Message(role="assistant", parts=[TextPart(text="Task completed.")]),
        ))
        assistant_text = "".join(response_chunks) or "Task completed."
        await self._write_back_memory(session_id, query, assistant_text)

        tool_calls = self._extract_tool_calls_from_chunks(response_chunks)
        await self._hermes_post_turn(session_id, query, assistant_text, tool_calls)

    # ------------------------------------------------------------------
    # Recall — DTM v4 or Hermes v3 fallback
    # ------------------------------------------------------------------

    async def _recall(self, query: str, session_id: str) -> str:
        """
        Single recall entry point for both SIMPLE and COMPLEX paths.

        v4 (DTM wired):
            Runs Track A (FTS5 raw turns + daily .md chunks) and
            Track B (curated facts from facts.jsonl) in parallel.
            Arbitrates scores: temporal decay × relevance for Track A,
            confidence × type-boost × track_b_weight for Track B.
            MMR-deduplicates (λ=0.7) and returns combined prompt string.

        v3 fallback (DTM absent):
            curator.recall_for_session() — FTS5 BM25 search only.
        """
        if self._dtm:
            try:
                result = await self._dtm.recall(query, session_id, max_chars=1200)
                logger.debug(
                    "DTM recall: A=%d B=%d winner=%s chars=%d",
                    result.track_a_count, result.track_b_count,
                    result.winner, len(result.prompt_context),
                )
                return result.prompt_context
            except Exception as exc:
                logger.warning("DTM recall failed, falling back to v3: %s", exc)

        # v3 fallback
        if self._curator:
            try:
                return await self._curator.recall_for_session(query, session_id)
            except Exception as exc:
                logger.debug("FTS5 recall skipped: %s", exc)
        return ""

    # ------------------------------------------------------------------
    # Post-turn Hermes hooks — DTM v4 or v3 fallback
    # ------------------------------------------------------------------

    async def _hermes_post_turn(
        self,
        session_id:     str,
        user_text:      str,
        assistant_text: str,
        tool_calls:     list[dict],
    ) -> None:
        """
        Called after every turn (both SIMPLE and COMPLEX paths).

        v4 (DTM wired):
            dtm.after_turn() runs Track A write (FTS5 + daily .md buffer)
            and Track B write (curator.after_turn → facts.jsonl) internally.
            user_model.after_turn() runs separately (outside DTM scope).

        v3 fallback (DTM absent):
            Three concurrent tasks: fts_store.write_turn + curator.after_turn
            + user_model.after_turn (original Hermes v3 behaviour).
        """
        if self._dtm:
            tasks = [
                self._safe(
                    self._dtm.after_turn(
                        session_id     = session_id,
                        user_text      = user_text,
                        assistant_text = assistant_text,
                        tool_calls     = tool_calls,
                        importance     = 0.6,
                    ),
                    "dtm.after_turn",
                )
            ]
            if self._user_model:
                tasks.append(self._safe(
                    self._user_model.after_turn(
                        session_id, user_text, assistant_text, tool_calls
                    ),
                    "user_model.after_turn",
                ))
            await asyncio.gather(*tasks)
            return

        # v3 fallback
        tasks = []
        if self._fts_store:
            tasks.append(self._safe(
                self._fts_store.write_turn(
                    session_id, user_text, assistant_text,
                    tool_calls=tool_calls, importance=0.6,
                ),
                "fts_store.write_turn",
            ))
        if self._curator:
            tasks.append(self._safe(
                self._curator.after_turn(
                    session_id, user_text, assistant_text, tool_calls
                ),
                "curator.after_turn",
            ))
        if self._user_model:
            tasks.append(self._safe(
                self._user_model.after_turn(
                    session_id, user_text, assistant_text, tool_calls
                ),
                "user_model.after_turn",
            ))
        if tasks:
            await asyncio.gather(*tasks)

    # ------------------------------------------------------------------
    # Post-action verification + §05 skill evolution
    # ------------------------------------------------------------------

    async def _verify_action_result(
        self, context: RequestContext, session_id: str, response_chunks: list[str]
    ) -> None:
        query          = context.get_user_input()
        action_type    = context.metadata.get("action_type", "complex_task")
        assistant_text = "".join(response_chunks)
        tools_used     = self._extract_tool_names_from_text(assistant_text)

        logger.info(
            "Post-action verification: action_type=%s tools=%s session=%s",
            action_type, tools_used, session_id,
        )
        # Always fire skill evolver for COMPLEX path if any tools were used
        if self._skill_evolver and (tools_used or assistant_text):
            try:
                await self._skill_evolver.after_task(
                    task_description = query,
                    solution_summary = assistant_text[:400],
                    tools_used       = tools_used,
                    solution_steps   = [],
                    key_observations = [],
                    complexity       = 7.5,   # complex path: higher complexity → lower bar for skill creation
                    session_id       = session_id,
                )
            except Exception as exc:
                logger.debug("SkillEvolver.after_task (complex path) skipped: %s", exc)

    # ------------------------------------------------------------------
    # HITL interrupt registration
    # ------------------------------------------------------------------

    async def _handle_interrupt_chunk(
        self, chunk: dict[str, Any], context: RequestContext, session_id: str
    ) -> None:
        interrupt_id = chunk.get("interrupt_id", str(uuid.uuid4()))
        action_raw   = chunk.get("proposed_action", {})

        # Extended fields for multi-mode HITL — choices for USER_CHOICE,
        # clarification_fields for CLARIFICATION, editable_param_keys for
        # EDIT-flavoured approves. All optional — empty list if not given.
        from hitl.schemas import ChoiceOption, ClarificationField
        _choices_raw = chunk.get("choices") or []
        _choices = [
            c if isinstance(c, ChoiceOption) else ChoiceOption(**c)
            for c in _choices_raw
        ]
        _clarif_raw = chunk.get("clarification_fields") or []
        _clarifs = [
            c if isinstance(c, ClarificationField) else ClarificationField(**c)
            for c in _clarif_raw
        ]
        _editable_keys = list(chunk.get("editable_param_keys") or [])

        # Fallback: when the trigger looks like an action that the operator
        # might want to tweak before approving, expose every non-trivial
        # parameter as editable. We detect this from EITHER:
        #   (a) explicit destructive trigger_kind, OR
        #   (b) low_confidence trigger but the user's query contains a
        #       destructive verb ("修复", "fix", "restart", "rollback"...).
        # This makes EDIT-mode the default UX for actionable HITL — operators
        # almost always want to confirm or adjust the actual config_lines /
        # scope before signing off. device_id is excluded (changing scope ≠
        # tweaking args).
        _trig_kind_str = chunk.get("trigger_kind", "")
        _user_query = (context.get_user_input() or "").lower()
        _has_destructive_verb = any(
            kw in _user_query
            for kw in (
                "修复", "修改", "重启", "重置", "调整", "下发", "推送",
                "fix", "repair", "restart", "reboot", "rollback", "revert",
                "push config", "apply config", "delete", "drain", "failover",
            )
        )
        _is_actionable = (
            _trig_kind_str in ("destructive_op", "destructive_action")
            or chunk.get("force_hitl_tool")
            or (_trig_kind_str == "low_confidence" and _has_destructive_verb)
        )
        _params_for_keys = (action_raw.get("parameters") or {}) if isinstance(
            action_raw, dict
        ) else {}

        # Note: we DON'T add a synthetic "instructions" field for the first-
        # round destructive HITL. The operator's role here is to approve
        # the INTENT ("yes, this user is allowed to fix configs"), not to
        # author the specific change. The actual parameter editing happens
        # in a SECOND HITL round, when the agent has decided which concrete
        # tool to call (e.g. edit_device_config) and the operator can tweak
        # config_lines / reason / etc. That second round is fired via the
        # force_hitl_tool path with real parameters already populated.
        if not _editable_keys and _is_actionable and _params_for_keys:
            _editable_keys = [
                k for k, v in _params_for_keys.items()
                if k != "device_id" and isinstance(v, (str, int, float, list, dict))
            ]
            if _editable_keys:
                logger.info(
                    "_handle_interrupt_chunk: auto-derived editable_param_keys=%s "
                    "(trigger=%s, has_destructive_verb=%s, no explicit keys)",
                    _editable_keys, _trig_kind_str, _has_destructive_verb,
                )

        payload = HitlPayload(
            interrupt_id     = interrupt_id,
            thread_id        = context.context_id,
            context_id       = context.context_id,
            task_id          = context.task_id,
            trigger_kind     = TriggerKind(chunk.get("trigger_kind", "low_confidence")),
            risk_level       = RiskLevel(chunk.get("risk_level", "medium")),
            user_query       = context.get_user_input(),
            intent_summary   = chunk.get("summary", ""),
            confidence_score = context.metadata.get("confidence_score", 0.0),
            proposed_action  = ProposedAction(
                action_type = action_raw.get("action_type", "unknown"),
                target      = action_raw.get("target", "unknown"),
                parameters  = action_raw.get("parameters", {}),
                reversible  = action_raw.get("reversible", True),
            ),
            choices               = _choices,
            clarification_fields  = _clarifs,
            editable_param_keys   = _editable_keys,
        )
        await self._hitl_router.register_interrupt(payload)

        # Fallback: if no tool-specific callback was registered for this interrupt
        # (e.g. low_confidence triggers from COMPLEX classification), register a
        # callback that runs the FULL AGENT LOOP on approval. This gives the
        # post-HITL action access to the tool registry, system prompt, memory
        # recall, and skill catalog — so it can actually call tools (e.g.
        # get_device_config) instead of just answering from memory.
        if interrupt_id not in self._hitl_router._direct_callbacks:
            # Use main.py's injected _runtime_loop (the LLM-patched instance),
            # NOT self._runtime (the unpatched internal loop used for
            # classify only). See comments in _execute_multi_mode_hitl.
            _agent_loop = (
                getattr(self, "_runtime_loop", None)
                or getattr(self, "_runtime", None)
            )
            _llm = self._llm_engine
            _user_query = context.get_user_input() or ""
            _sid_local  = session_id
            _tool_reg_local = self._tool_registry
            # Prefer executor's own memory router (same instance the
            # chat_stream path writes to). Falling back to runtime_loop._memory
            # is fine for some setups but unreliable when the runtime_loop
            # was constructed with a different (or None) memory router. The
            # bug we're fixing: post-HITL callback returned an answer that
            # said "I need a device id" because pre-recall came back empty —
            # but the prior turn had clearly set ap-01 as the focus device.
            _memory_local = (
                self._memory
                or getattr(_agent_loop, "_memory", None)
            )
            logger.info(
                "Post-HITL callback wiring: agent_loop=%s memory=%s tool_reg_size=%d",
                "ok" if _agent_loop is not None else "MISSING",
                "executor" if _memory_local is self._memory
                    else ("runtime_loop" if _memory_local is not None else "MISSING"),
                len(_tool_reg_local or {}),
            )

            # confirmed_facts is structured "TOOL_EXEC: tool|args" history
            # the runtime loop builds during the original (pre-HITL) turn.
            # We capture it for the resume callback so coreference resolution
            # can read it (most reliable signal for "current focus device").
            _confirmed_facts_local = list(context.metadata.get("confirmed_facts") or [])

            async def _post_hitl_agent_callback(
                decision=None,
                _loop=_agent_loop, _llm_ref=_llm, _q=_user_query, _sid=_sid_local,
                _tool_reg=_tool_reg_local, _mem=_memory_local,
                _facts=_confirmed_facts_local,
            ):
                """
                Run the agent on the approved query. Prefer the full runtime
                loop (real tool calls); fall back to single-shot LLM if no loop
                is wired.
                """
                # If the operator submitted an EDIT decision with a
                # parameter_patch, log it so we can see what they tweaked.
                # First-round destructive HITLs typically have no patch
                # (operator just clicked Approve on the intent).
                # Second-round force_hitl_tool ones have a patch like
                # {"config_lines": [...]} which the dedicated
                # _approved_tool_callback (registered separately) merges
                # before invoking the tool.
                _patch = getattr(decision, "parameter_patch", None) if decision else None
                if _patch:
                    logger.info(
                        "_post_hitl_agent_callback: operator parameter_patch "
                        "received but this is the intent-approval round — "
                        "patch_keys=%s will be IGNORED (forwarded to next-round "
                        "tool HITL if applicable)",
                        list(_patch.keys()) if isinstance(_patch, dict) else "?",
                    )
                # Pre-recall: fetch session context BEFORE handing to stream so
                # stream's internal pre_verify (which fires before its own
                # _retrieve_memory) gets the same recall the chat_stream path
                # gets. Without this, the post-HITL turn runs with NO awareness
                # of prior turns — see "After HITL" prompt in user reports.
                _env_ctx: dict = {}
                # Forward confirmed_facts from the original turn so coreference
                # resolution + the runtime loop's pre_verify can both see the
                # structured tool-call history (most reliable signal for
                # "what device was the agent just operating on").
                if _facts:
                    _env_ctx["_confirmed_facts"] = list(_facts)
                if _mem is not None and _q:
                    try:
                        _rec = await _mem.recall(_q, _sid, max_chars=1200)
                        if _rec and getattr(_rec, "prompt_context", ""):
                            _env_ctx["_fts_context"] = _rec.prompt_context
                            logger.info(
                                "Post-HITL callback: pre-recalled context "
                                "for session=%s query=%r → %d chars",
                                _sid, _q[:60], len(_rec.prompt_context),
                            )
                        else:
                            logger.info(
                                "Post-HITL callback: pre-recall returned EMPTY "
                                "for session=%s query=%r — Path B will answer blind",
                                _sid, _q[:60],
                            )
                    except Exception as _re:
                        logger.warning("Post-HITL pre-recall failed: %s", _re)

                # Path A — full agent loop (preferred). Tool registry, memory,
                # skills all available; the LLM can call get_device_config etc.
                if _loop is not None and hasattr(_loop, "stream"):
                    try:
                        full_text = ""
                        last_message = ""

                        # ── Code-level coreference resolution ────────────
                        # When the operator's request doesn't name a device
                        # but recall shows a clear "focus device" from the
                        # most recent tool call, prepend an explicit fact so
                        # the LLM doesn't have to infer this. Without it, the
                        # LLM has been observed to ignore in-prompt context
                        # and reply "缺少目标设备 ID" — see screenshot bug
                        # report. Banner instructions alone don't cover all
                        # the LLM's failure modes; a hard fact does.
                        _focus_device = self._infer_focus_device(_q, _env_ctx)
                        if _focus_device:
                            _q_for_loop = (
                                f"{_q}\n\n"
                                f"[RESOLVED FROM CONTEXT] target_device = {_focus_device}\n"
                                f"(The user did not name a device but the most recent tool\n"
                                f" call in this session was on `{_focus_device}`. Use this\n"
                                f" device unless the operator explicitly says otherwise.)"
                            )
                            logger.info(
                                "Post-HITL callback: code-level coreference — "
                                "query did not name a device, but recall shows "
                                "%s as the focus. Bound query to that device.",
                                _focus_device,
                            )
                        else:
                            _q_for_loop = _q

                        # Wrap the user query with a post-HITL banner so the LLM
                        # knows operator approval has ALREADY been granted for
                        # this request and it should NOT ask the user to
                        # "reply 确认 to trigger HITL approval".
                        #
                        # CRITICAL: the LLM has been observed to ignore softer
                        # instructions and still produce "请回复确认或执行" /
                        # "是否确认执行此修改" pseudo-confirmation text. The
                        # banner below uses very explicit MUST/MUST-NOT framing
                        # plus a forbidden-phrases list to stop this behaviour.
                        _approved_query = (
                            "================================================================\n"
                            "【SYSTEM OVERRIDE — POST-HITL EXECUTION CONTEXT】\n"
                            "================================================================\n"
                            "An operator has ALREADY clicked Approve on this request.\n"
                            "This turn IS the approved execution. There is NO further\n"
                            "user confirmation step. You are NOT in the planning phase.\n"
                            "\n"
                            "COREFERENCE RULES — read these BEFORE deciding what to do:\n"
                            "  • The operator's request may not name a device explicitly.\n"
                            "    If the recent conversation context (above) shows you\n"
                            "    operating on a SPECIFIC device (e.g. you ran\n"
                            "    get_device_config on ap-01 in the last few turns), the\n"
                            "    current request is for THAT SAME DEVICE. Do not ask\n"
                            "    'which device?' — the answer is in the context.\n"
                            "  • If the recent context is genuinely ambiguous (e.g. user\n"
                            "    has browsed the full device list with no specific focus\n"
                            "    device), then list_devices is the answer — call it,\n"
                            "    don't tell the user to re-type a device name.\n"
                            "  • Never reply 'I cannot proceed without a device ID' —\n"
                            "    either resolve it from context, or call list_devices.\n"
                            "\n"
                            "MUST DO (one of):\n"
                            "  (a) Call the relevant config/action tool DIRECTLY in this\n"
                            "      same turn — e.g. [TOOL:edit_device_config] {...}.\n"
                            "      That tool itself is HITL-flagged and will surface a\n"
                            "      fine-grained approval card showing the exact change.\n"
                            "      You do NOT need to ask the user for permission first.\n"
                            "  (b) If the request is purely informational/diagnostic and\n"
                            "      no action tool is needed, give your final analysis as\n"
                            "      a terminal answer (no [TOOL:] line at all).\n"
                            "\n"
                            "MUST NOT, under any circumstance:\n"
                            "  • Output 'Are you sure?' / '是否确认' / '是否执行此修改'\n"
                            "  • Output '请回复 确认 / 执行 / yes / proceed'\n"
                            "  • Output 'reply confirm to trigger HITL approval'\n"
                            "  • Output '缺少目标设备 ID' / 'missing device ID' /\n"
                            "    '请告诉我您想...哪台设备' / '请直接回复设备名称' —\n"
                            "    instead resolve it from the conversation context above\n"
                            "    or call list_devices to enumerate options.\n"
                            "  • Ask the user any clarification, confirmation, or\n"
                            "    permission question. The operator already approved.\n"
                            "  • Produce a 'plan / proposal' phrased as a request for\n"
                            "    user sign-off. If you have a plan, EXECUTE it via the\n"
                            "    tool, do not describe it as pending approval.\n"
                            "\n"
                            "Approved request:\n"
                            f"  {_q_for_loop}\n"
                            "================================================================"
                        )
                        # Use the ORIGINAL session_id so _retrieve_memory can
                        # recall facts from the pre-HITL turns of this same
                        # conversation (e.g. a router-01 config fetched moments
                        # ago). Prefixing with "posthitl__" creates a brand-new
                        # empty session and breaks contextual continuity.
                        async for chunk in _loop.stream(
                            query           = _approved_query,
                            session_id      = _sid,
                            tool_registry   = _tool_reg,
                            env_context     = _env_ctx,
                            skip_pre_verify = True,   # ← operator approval is final
                        ):
                            tok = chunk.get("token") if isinstance(chunk, dict) else None
                            if tok:
                                full_text += tok
                            # Capture stop/error messages so we can log why
                            # Path A produced no text (typical reasons: pre_verify
                            # blocked, stop_policy fired, LLM patch missing).
                            msg = chunk.get("message") if isinstance(chunk, dict) else None
                            if msg:
                                last_message = msg

                            # Nested HITL: the agent's first-pass intent has
                            # already been approved; now the LLM has decided
                            # to call a HITL-gated tool with concrete args.
                            # Fire a SECOND interrupt with the actual
                            # parameters editable, so the operator can review
                            # / tweak the exact change (e.g. config_lines).
                            if (isinstance(chunk, dict)
                                    and chunk.get("stop_hitl")
                                    and chunk.get("tool_name")):
                                _nested_tool = chunk.get("tool_name")
                                _nested_args = chunk.get("tool_args") or {}
                                _nested_editable = list(
                                    chunk.get("editable_param_keys") or []
                                )
                                # Auto-derive editable keys from the args dict
                                # if loop didn't set them.
                                if not _nested_editable and isinstance(_nested_args, dict):
                                    _nested_editable = [
                                        k for k, v in _nested_args.items()
                                        if k != "device_id" and isinstance(
                                            v, (str, int, float, list, dict)
                                        )
                                    ]
                                logger.info(
                                    "Nested HITL detected: agent wants to call "
                                    "%s with args=%s (editable=%s)",
                                    _nested_tool, list(_nested_args.keys()),
                                    _nested_editable,
                                )
                                full_text = (
                                    f"[NESTED HITL] 操作员已批准修复意图，agent 决定调用工具 "
                                    f"`{_nested_tool}`。已发起第二轮审批，请在 HITL 面板确认/编辑"
                                    f"具体参数后批准执行。"
                                )
                                # Register the second-round interrupt directly
                                # with the executor's helper. After the
                                # operator approves the nested card,
                                # _approved_tool_callback will invoke the
                                # actual tool with merged params.
                                try:
                                    await self._fire_nested_tool_hitl(
                                        tool_name      = _nested_tool,
                                        tool_args      = _nested_args,
                                        editable_keys  = _nested_editable,
                                        original_query = _q,
                                        session_id     = _sid,
                                        tool_registry  = _tool_reg or {},
                                    )
                                except Exception as _ne:
                                    logger.exception(
                                        "Nested HITL register failed: %s", _ne,
                                    )
                                    full_text = (
                                        f"[NESTED HITL ERROR] 无法发起二轮审批：{_ne}"
                                    )
                                break
                        if full_text.strip():
                            # Quality check: did the agent actually FINISH the
                            # task, or did it stop mid-plan? Pattern observed:
                            #   "我将对 ap-01 和 ap-02 进行配置修复...
                            #    第一步：修复 ap-01 的 RADIUS 超时配置"
                            #   <stream ends — never says "已修复" or "完成">
                            # When this happens, downstream recall sees a
                            # turn whose body is full of future-tense "will do"
                            # text, so the LLM in the NEXT turn correctly
                            # concludes nothing was actually done. Detect this
                            # case and fall back to Path B for a final summary.
                            _ft_lower = full_text.lower()
                            _has_completion_marker = any(
                                kw in full_text or kw in _ft_lower
                                for kw in (
                                    "已修复", "已完成", "已应用", "修复成功", "修复完成",
                                    "配置已", "已生效", "已优化", "执行完毕",
                                    "completed", "applied", "fixed successfully",
                                    "has been", "configuration updated",
                                )
                            )
                            _has_plan_only = any(
                                kw in full_text
                                for kw in (
                                    "我将立即", "我将对", "我将依次", "首先",
                                    "第一步", "第二步", "## 第", "首先获取",
                                )
                            ) and not _has_completion_marker
                            if _has_plan_only:
                                logger.warning(
                                    "Post-HITL callback: Path A produced a PLAN-ONLY "
                                    "response (no completion marker). Falling through "
                                    "to Path B so the operator gets a final summary "
                                    "with explicit action results. preview=%r",
                                    full_text[:200],
                                )
                                # Don't return; fall through to Path B below
                            else:
                                logger.info(
                                    "Post-HITL callback: Path A (agent loop) "
                                    "succeeded — %d chars",
                                    len(full_text),
                                )
                                return {"tool": "agent_loop", "result": full_text.strip()}
                        else:
                            logger.warning(
                                "Post-HITL callback: Path A (agent loop) produced "
                                "ZERO tokens — falling back to Path B. "
                                "last_message=%r env_ctx_keys=%s "
                                "approved_query_len=%d",
                                last_message[:200],
                                list(_env_ctx.keys()) if _env_ctx else [],
                                len(_approved_query),
                            )
                    except Exception as exc:
                        # FULL traceback so we can see exactly which line in
                        # the agent loop blew up. Previously this was logged at
                        # WARNING with no traceback, hiding the real cause.
                        logger.error(
                            "Post-HITL agent loop FAILED, falling back to Path B: %s",
                            exc, exc_info=True,
                        )

                # Path B — direct LLM call (no tool access)
                if _llm_ref is None:
                    return {
                        "tool":   "llm_answer",
                        "result": "Approved, but no agent loop or LLM is configured. "
                                  "Manual investigation required.",
                    }
                # Reuse the same code-level coreference resolution Path A did.
                # If Path A bound the query to a focus device, Path B should
                # see the same binding so it doesn't ask "which device?" either.
                _q_for_b = (
                    locals().get("_q_for_loop", None)
                    or _q
                )
                try:
                    # Include recalled session context so the LLM can resolve
                    # references like "该设备" / "this device" against prior turns.
                    # Without this, Path B answers blind ("未提供具体设备型号...").
                    _ctx_block = ""
                    _fts = _env_ctx.get("_fts_context") if _env_ctx else None
                    if _fts:
                        _ctx_block = (
                            "Prior conversation context (entities mentioned "
                            "earlier in this session count as known — use this "
                            "to resolve references like '该设备', 'this device', "
                            "'it'):\n"
                            f"-----\n{str(_fts)[:1500]}\n-----\n\n"
                        )
                    prompt = (
                        f"{_ctx_block}"
                        "[POST-HITL EXECUTION CONTEXT — operator approval already granted]\n"
                        "The operator has ALREADY approved this request:\n"
                        f"  {_q_for_b}\n\n"
                        "COREFERENCE RULES:\n"
                        "  • If the request doesn't name a device but the recent\n"
                        "    context clearly shows you operating on a specific device\n"
                        "    (e.g. you ran get_device_config on ap-01 in the last few\n"
                        "    turns), the request is for THAT device. Do not ask\n"
                        "    'which device?'.\n"
                        "  • Never reply with '请告诉我您想...哪台设备' or '缺少目标设备 ID' —\n"
                        "    resolve from context or list all devices.\n\n"
                        "Provide your best answer based on the prior context above. "
                        "Reply in the SAME LANGUAGE as the request, using markdown: "
                        "brief verdict, key points as bullets, and a short next-step "
                        "recommendation. "
                        "IMPORTANT: do NOT ask the user to 'reply 确认' / 'reply 执行' / "
                        "'trigger HITL approval' — there is no such pseudo-command and "
                        "approval was already granted for THIS turn. Give your final "
                        "answer directly."
                    )
                    logger.info(
                        "Post-HITL callback: Path B (direct LLM) — prompt %d chars "
                        "(of which %d chars are recalled context)",
                        len(prompt), len(_ctx_block),
                    )
                    messages = [{"role": "user", "content": prompt}]
                    if hasattr(_llm_ref, "_chat"):
                        text = await _llm_ref._chat(messages)
                    else:
                        text = await _llm_ref.call(prompt, "", state=None)
                    return {"tool": "llm_answer", "result": (text or "").strip()}
                except Exception as exc:
                    return {"tool": "llm_answer", "error": str(exc)}

            import time as _time_fb
            self._hitl_router._direct_callbacks[interrupt_id] = (_post_hitl_agent_callback, _time_fb.monotonic())
        asyncio.create_task(self._review_service.notify(payload))
        if self._task_system and self._task_system.hitl_bridge:
            try:
                session = await self._task_system.session_mgr.get_or_create(
                    context_id=context.context_id
                )
                from task.schemas import TaskDefinition, TaskScope, TaskState as TState
                task_ref = TaskDefinition(
                    task_id     = context.task_id,
                    session_id  = session_id,
                    context_id  = context.context_id,
                    scope       = TaskScope.INTRA,
                    description = context.get_user_input(),
                    state       = TState.WAITING_HITL,
                )
                await self._task_system.hitl_bridge.suspend_for_review(
                    task_ref, payload, session
                )
            except Exception as exc:
                logger.warning("HitlTaskBridge suspend failed: %s", exc)

    # ------------------------------------------------------------------
    # Memory helpers (L1–L4 MemoryRouter — unchanged from v3)
    # ------------------------------------------------------------------

    async def _build_memory_context(self, context: RequestContext, session_id: str) -> str:
        if self._memory is None:
            return ""
        try:
            return await self._memory.recall_for_session(
                context.get_user_input(), session_id,
            )
        except Exception as exc:
            logger.warning("Memory retrieval failed: %s", exc)
            return ""

    async def _write_back_memory(
        self, session_id: str, user_text: str, assistant_text: str
    ) -> None:
        if self._memory is None:
            return
        try:
            await self._memory.ingest_turn(session_id, user_text, assistant_text)
        except Exception as exc:
            logger.warning("Memory write-back failed: %s", exc)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _safe(coro, name: str) -> None:
        """Run a coroutine — log but never propagate exceptions."""
        try:
            await coro
        except Exception as exc:
            logger.warning("Hermes hook %s failed: %s", name, exc)

    @staticmethod
    def _extract_tool_calls_from_chunks(chunks: list[str]) -> list[dict]:
        """Extract [TOOL:name] calls from assembled response text."""
        import re, json
        text  = "".join(chunks)
        calls = []
        for m in re.finditer(r"\[TOOL:(\w+)\]\s*(\{[^}]*\})?", text):
            tool_name = m.group(1)
            try:
                args = json.loads(m.group(2) or "{}")
            except Exception:
                args = {}
            calls.append({"tool": tool_name, "args": args})
        return calls

    @staticmethod
    def _extract_tool_names_from_text(text: str) -> list[str]:
        """Extract unique tool names from response text."""
        import re
        return list(dict.fromkeys(re.findall(r"\[TOOL:(\w+)\]", text)))

    @staticmethod
    def _extract_session_id(context: RequestContext) -> str:
        return context.metadata.get("session_id", context.context_id)

    # ------------------------------------------------------------------
    # Code-level coreference resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_focus_device(query: str, env_ctx: dict) -> Optional[str]:
        """If the query doesn't name a device but recall context shows a
        clear 'focus device' from a recent tool call or analysis, return
        that device id. Otherwise return None.

        Heuristic — order of evidence:
          1. If the query already contains a device id pattern, return None
             (no resolution needed).
          2. Scan the recall context (env_ctx['_fts_context']) for the
             MOST RECENT TOOL_EXEC line on a device-bearing tool. The
             tool args carry device_id explicitly.
          3. Fall back to the most recent device id mentioned anywhere
             in the recall context.

        Returns None when there's no clear single focus, e.g. recent
        context shows several different devices being discussed.
        """
        import re as _re
        # Step 1: query already has a device id → no need to resolve
        _device_pattern = (
            r"(?<![a-z0-9])"
            r"(ap|sw|router|switch)[-_]?[a-z0-9]*[-_]?\d+"
            r"(?![a-z0-9])"
        )
        if _re.search(_device_pattern, (query or "").lower()):
            return None

        ctx = (env_ctx or {}).get("_fts_context", "") or ""
        # confirmed_facts is a list of strings the runtime loop builds up
        # ("TOOL_EXEC: get_device_config|..." entries). The post-HITL
        # callback may not include it in _fts_context, so check it
        # separately. This is the most reliable source for "what device
        # was the agent just operating on" since it's structured.
        _facts = (env_ctx or {}).get("_confirmed_facts", []) or []
        if _facts:
            ctx = ctx + "\n" + "\n".join(str(f) for f in _facts)
        if not ctx.strip():
            return None

        # Step 2: walk lines bottom-up looking for TOOL_EXEC on
        # device-aware tools. Most-recent-first by virtue of recall ordering.
        # Pattern matches: TOOL_EXEC: get_device_config|{"device_id": "ap-01"} ...
        tool_re = _re.compile(
            r'TOOL_EXEC:\s*(get_device_config|validate_device_config|edit_device_config'
            r'|push_config|restart_service|drain_node|failover|delete_resource|'
            r'syslog_search)\s*\|\s*\{[^}]*"device_id"\s*:\s*"([^"]+)"',
            _re.IGNORECASE,
        )
        # Search bottom-up for "most recent". Recall context typically
        # orders turns oldest→newest so we want the LAST match.
        matches = list(tool_re.finditer(ctx))
        if matches:
            return matches[-1].group(2)

        # Step 3: fall back to device-id mentions. Take the LAST one
        # mentioned in the context — proxy for "most recently discussed".
        # We dedupe but keep the last unique mention.
        last_mention = None
        for m in _re.finditer(_device_pattern, ctx.lower()):
            last_mention = m.group(0)
        return last_mention

    # ------------------------------------------------------------------
    # Nested HITL — second-round tool-parameter approval
    # ------------------------------------------------------------------

    async def _fire_nested_tool_hitl(
        self,
        *,
        tool_name:      str,
        tool_args:      dict,
        editable_keys:  list[str],
        original_query: str,
        session_id:     str,
        tool_registry:  dict,
    ) -> str:
        """
        Register a second-round HITL interrupt for an actual tool call,
        with concrete parameters that the operator may edit before
        approving. Returns the new interrupt_id.

        This is invoked from _post_hitl_agent_callback when the agent's
        post-approval pass produces a stop_hitl chunk for a HITL-gated
        tool (e.g. edit_device_config). The first-round HITL approves
        the user's INTENT; this second round approves the SPECIFIC tool
        invocation with its actual args.
        """
        import uuid as _uuid
        import time as _time
        from hitl.schemas import (
            HitlPayload, ProposedAction, TriggerKind, RiskLevel,
        )

        interrupt_id = str(_uuid.uuid4())
        # Build the payload with the actual tool args populated and
        # editable_param_keys set so the UI shows inline editors.
        payload = HitlPayload(
            interrupt_id     = interrupt_id,
            thread_id        = session_id,
            context_id       = session_id,
            task_id          = session_id,   # best-effort; this is fired post-stream
            trigger_kind     = TriggerKind.DESTRUCTIVE_OP,
            risk_level       = RiskLevel.HIGH,
            user_query       = original_query,
            intent_summary   = (
                f"Second-round approval: agent intends to call `{tool_name}` "
                f"with the following args. Review and edit if needed."
            ),
            confidence_score = 0.95,
            proposed_action  = ProposedAction(
                action_type = f"tool_call:{tool_name}",
                target      = tool_name,
                parameters  = dict(tool_args),
                reversible  = False,
            ),
            editable_param_keys = list(editable_keys or []),
            sla_seconds      = 600,
        )
        await self._hitl_router.register_interrupt(payload)

        # Register a callback that invokes the actual tool when approved,
        # merging the operator's parameter_patch into the args.
        async def _nested_approved_callback(
            decision=None,
            _tool=tool_name, _args=tool_args, _reg=tool_registry,
            _sid=session_id, _q=original_query,
        ):
            if _tool not in _reg:
                return {"error": f"Tool {_tool!r} not in registry after approval"}
            _final_args = dict(_args)
            if decision is not None and getattr(decision, "parameter_patch", None):
                _patch = decision.parameter_patch or {}
                _final_args = {**_final_args, **_patch}
                logger.info(
                    "Nested HITL callback: applying operator edit — "
                    "patch_keys=%s",
                    list(_patch.keys()),
                )
            try:
                raw = await _reg[_tool](_final_args)
                # Persist the result via Hermes hook so the conversation
                # records what actually happened.
                await self._hermes_post_turn(
                    _sid, _q,
                    f"[TOOL:{_tool}] approved (round 2) and executed. "
                    f"Result: {str(raw)[:300]}",
                    [{"tool": _tool}],
                )
                return {"tool": _tool, "args": _final_args, "result": str(raw)}
            except Exception as exc:
                logger.exception("Nested tool execution failed: %s", exc)
                return {"tool": _tool, "error": str(exc)}

        # Prune stale callbacks before registering (same TTL as first round)
        _now = _time.monotonic()
        _ttl = 1800
        self._hitl_router._direct_callbacks = {
            k: v for k, v in self._hitl_router._direct_callbacks.items()
            if _now - (v[1] if isinstance(v, tuple) else 0) < _ttl
        }
        self._hitl_router._direct_callbacks[interrupt_id] = (
            _nested_approved_callback, _now,
        )
        logger.info(
            "Nested HITL registered: interrupt_id=%s tool=%s editable=%s",
            interrupt_id[:12], tool_name, editable_keys,
        )
        return interrupt_id


    # ==================================================================
    # SIMPLE-path multi-mode HITL handlers
    # ==================================================================
    # These are invoked by _execute_simple when the runtime loop emits a
    # stop_hitl chunk with a specific hitl_kind. They emit the right
    # interrupt type via the trigger_* helpers and register a callback
    # that resumes the loop with the operator's response folded in.

    async def _handle_user_choice_chunk(
        self, chunk: dict, query: str, session_id: str,
        context: RequestContext, event_queue, task_id: str,
    ) -> None:
        """Operator picks one of N skill candidates → re-run the loop with
        a query reformulation that hard-pins the chosen skill."""
        choices = chunk.get("choices", [])
        summary = chunk.get("summary", "Multiple matches — please pick one.")
        if not choices:
            # No choices? Degrade gracefully to vanilla HITL approve panel.
            logger.warning("_handle_user_choice: empty choices, falling back to _execute_complex")
            await self._execute_complex(
                query, session_id, context, event_queue, task_id, context.context_id
            )
            return

        async def _after_pick(decision):
            """Resume the loop, hard-pinning the operator-selected skill via
            an explicit reformulation so the LLM doesn't re-ambiguate."""
            chosen = getattr(decision, "selected_choice_id", None)
            if not chosen:
                logger.warning("_after_pick: no selected_choice_id in decision")
                return {"tool": "user_choice", "result": "[Error: no selection]"}
            logger.info(
                "Post-HITL user_choice: operator chose %s — resuming agent loop",
                chosen,
            )
            # Resume by running the agent loop with the original query AND
            # an explicit instruction to use the chosen skill.
            return await self._resume_after_simple_hitl(
                original_query = query,
                augmented_query = (
                    f"{query}\n\n[OPERATOR SELECTED SKILL]: {chosen}\n"
                    f"Use this specific skill to fulfill the request."
                ),
                session_id = session_id,
                context    = context,
            )

        await self.trigger_user_choice(
            summary       = summary,
            choices       = choices,
            callback      = _after_pick,
            context       = context,
            session_id    = session_id,
            event_queue   = event_queue,
            task_id       = task_id,
            action_target = "skill_selection",
        )
        # Tell the SIMPLE path to stop streaming further chunks; the
        # interrupt is now in the queue and the operator's decision will
        # drive the rest.
        await event_queue.enqueue_event(TaskStatusUpdateEvent(
            task_id=task_id, context_id=context.context_id,
            status=TaskStatus(state=TaskState.WORKING),
            metadata={"hitl_pending": True, "kind": "user_choice"},
        ))

    async def _handle_clarification_chunk(
        self, chunk: dict, query: str, session_id: str,
        context: RequestContext, event_queue, task_id: str,
    ) -> None:
        """Agent ran low on confidence and asked for missing info → register
        a callback that re-runs the loop with the operator's answers folded
        into the query. Skip pre_verify on resume because the operator's
        answer is the policy."""
        fields  = chunk.get("clarification_fields", [])
        summary = chunk.get("summary", "More information needed to proceed.")
        if not fields:
            logger.warning("_handle_clarification: empty fields, returning unanswered")
            return

        async def _after_answer(decision):
            answers = getattr(decision, "clarification_answers", None) or {}
            if not answers:
                return {"tool": "clarification", "result": "[Error: no answers received]"}
            logger.info(
                "Post-HITL clarification: operator answered %d field(s) — resuming",
                len(answers),
            )
            # Build an augmented query that bakes the answers in so the
            # next loop iteration can act on them directly.
            answer_lines = "\n".join(f"  • {k}: {v}" for k, v in answers.items())
            augmented = (
                f"{query}\n\n[OPERATOR-PROVIDED CLARIFICATIONS]:\n{answer_lines}\n"
                f"Proceed with this additional information."
            )
            return await self._resume_after_simple_hitl(
                original_query  = query,
                augmented_query = augmented,
                session_id      = session_id,
                context         = context,
            )

        await self.trigger_clarification(
            summary     = summary,
            fields      = fields,
            callback    = _after_answer,
            context     = context,
            session_id  = session_id,
            event_queue = event_queue,
            task_id     = task_id,
        )
        await event_queue.enqueue_event(TaskStatusUpdateEvent(
            task_id=task_id, context_id=context.context_id,
            status=TaskStatus(state=TaskState.WORKING),
            metadata={"hitl_pending": True, "kind": "clarification"},
        ))

    async def _resume_after_simple_hitl(
        self, *, original_query: str, augmented_query: str,
        session_id: str, context: RequestContext,
    ) -> dict:
        """Re-run the runtime loop with the operator-augmented query.
        Used by user_choice + clarification callbacks. Skip pre_verify
        because the operator's input is the policy. Capture full text
        and return as a callback result that the decision router will
        feed into _submit_hitl_decision's synthesis step."""
        # Use main.py's injected patched loop, NOT self._runtime (which
        # is unpatched and only meant for classify()).
        _patched_loop = (
            getattr(self, "_runtime_loop", None)
            or getattr(self, "_runtime", None)
        )
        if not _patched_loop:
            return {"tool": "agent_loop", "result": "[Error: runtime loop not wired]"}

        env_ctx = dict(context.metadata.get("env_context", {}) or {})
        # Pre-recall so the loop's pre_verify (if it runs) sees prior turns
        try:
            past = await self._recall(original_query, session_id)
            if past:
                env_ctx["_fts_context"] = past[:1500]
        except Exception:
            pass

        out_chunks: list[str] = []
        try:
            async for ck in _patched_loop.stream(
                query           = augmented_query,
                session_id      = session_id,
                env_context     = env_ctx,
                tool_registry   = self._tool_registry,
                skip_pre_verify = True,    # operator already approved scope
            ):
                if "token" in ck:
                    out_chunks.append(ck["token"])
                if ck.get("stop_hitl"):
                    # Nested HITL — let the loop's own gate handle it.
                    out_chunks.append(
                        f"\n\n[Nested HITL: {ck.get('reason', 'further approval needed')}]"
                    )
                    break
        except Exception as exc:
            logger.exception("_resume_after_simple_hitl: stream failed: %s", exc)
            return {"tool": "agent_loop", "error": str(exc)}

        full_text = "".join(out_chunks).strip() or "[No response generated]"
        return {"tool": "agent_loop", "result": full_text}


    # ==================================================================
    # Multi-mode HITL helpers — public API
    # ==================================================================
    # These let any code path inside the executor (or a tool wrapped by
    # it) raise a HITL interrupt of one of the new types without having
    # to know about HitlPayload, _direct_callbacks, or event queues.
    #
    # Each helper:
    #   1. Builds the HitlPayload with the right trigger_kind
    #   2. Registers it via _handle_interrupt_chunk
    #   3. Registers the user-supplied callback so /hitl/{id}/{decision}
    #      can resume execution after the operator responds
    #
    # Callbacks receive a HitlDecision object so they can read
    # decision.selected_choice_id / parameter_patch / clarification_answers.

    async def _emit_hitl_payload(
        self,
        *,
        trigger_kind: TriggerKind,
        risk_level:   RiskLevel,
        summary:      str,
        proposed_action: dict,
        choices:      Optional[list[dict]]   = None,
        clarification_fields: Optional[list[dict]] = None,
        editable_param_keys:  Optional[list[str]]  = None,
        callback,
        context:      RequestContext,
        session_id:   str,
        event_queue,
        task_id:      str,
        sla_seconds:  int = 600,
    ) -> str:
        """Emit a HITL interrupt of the given type and register the resume
        callback. Returns the new interrupt_id.

        Common machinery used by the three trigger helpers below — also
        directly callable for custom flows that don't fit the canned types.
        """
        import uuid as _uuid
        import time as _time_cb
        interrupt_id = str(_uuid.uuid4())
        chunk = {
            "hitl_interrupt": True,
            "interrupt_id":   interrupt_id,
            "trigger_kind":   trigger_kind.value,
            "risk_level":     risk_level.value,
            "summary":        summary,
            "proposed_action": proposed_action,
            "choices":              choices              or [],
            "clarification_fields": clarification_fields or [],
            "editable_param_keys":  editable_param_keys  or [],
            "thread_id":      context.context_id,
            "node":           "hitl",
            "tag":            "hitl_interrupt",
            "kind":           "hitl_interrupt",
        }
        await self._handle_interrupt_chunk(chunk, context, session_id)
        for processor in self._processors:
            await processor.process(chunk, event_queue, task_id, context.context_id)

        # Register the resume callback. Old callbacks took no args;
        # _resume detects the new (decision) signature via inspect.
        _cb_now = _time_cb.monotonic()
        _cb_ttl = 1800
        self._hitl_router._direct_callbacks = {
            k: v for k, v in self._hitl_router._direct_callbacks.items()
            if _cb_now - (v[1] if isinstance(v, tuple) else 0) < _cb_ttl
        }
        self._hitl_router._direct_callbacks[interrupt_id] = (callback, _cb_now)
        logger.info(
            "HITL interrupt emitted — id=%s kind=%s sla=%ds",
            interrupt_id[:12], trigger_kind.value, sla_seconds,
        )
        return interrupt_id

    async def trigger_user_choice(
        self, *,
        summary:    str,
        choices:    list[dict],          # each: {id, label, description?, metadata?}
        callback,                         # async (decision) → result_dict
        context:    RequestContext,
        session_id: str,
        event_queue,
        task_id:    str,
        action_target: str = "user_choice",
    ) -> str:
        """Type 1 — request the operator to pick one of N options.

        Use when:
          • user's request matched multiple skills/tools and you can't
            pick deterministically (e.g. "查日志" → 3 candidate skills)
          • the action would target many entities and the operator should
            narrow scope (e.g. "查所有 ap 配置" → 4 APs found)

        The callback receives the HitlDecision; read .selected_choice_id.
        """
        return await self._emit_hitl_payload(
            trigger_kind = TriggerKind.USER_CHOICE,
            risk_level   = RiskLevel.LOW,
            summary      = summary,
            proposed_action = {
                "action_type": "user_choice",
                "target":      action_target,
                "parameters":  {"options": [c.get("id") for c in choices]},
                "reversible":  True,
            },
            choices    = choices,
            callback   = callback,
            context    = context,
            session_id = session_id,
            event_queue= event_queue,
            task_id    = task_id,
        )

    async def trigger_clarification(
        self, *,
        summary:     str,
        fields:      list[dict],          # each: {key, prompt, placeholder?, required?}
        callback,                          # async (decision) → result_dict
        context:     RequestContext,
        session_id:  str,
        event_queue,
        task_id:     str,
    ) -> str:
        """Type 3 — agent asks the operator for missing information.

        Use when:
          • LLM confidence is low AND clarification (not approval) is
            what the agent actually needs to proceed
          • the proposed action requires args that the user hasn't
            provided and shouldn't be guessed at

        Cap the number of clarifications per session to avoid loops —
        recommended ≤ 2 from a single ambiguous query.
        """
        return await self._emit_hitl_payload(
            trigger_kind = TriggerKind.CLARIFICATION,
            risk_level   = RiskLevel.LOW,
            summary      = summary,
            proposed_action = {
                "action_type": "clarification",
                "target":      "operator",
                "parameters":  {"fields": [f.get("key") for f in fields]},
                "reversible":  True,
            },
            clarification_fields = fields,
            callback   = callback,
            context    = context,
            session_id = session_id,
            event_queue= event_queue,
            task_id    = task_id,
        )

    async def trigger_edit_approval(
        self, *,
        summary:     str,
        proposed_action: dict,             # action_type/target/parameters
        editable_keys: list[str],          # parameters keys the operator may edit
        callback,                           # async (decision) → result_dict
        context:     RequestContext,
        session_id:  str,
        event_queue,
        task_id:     str,
    ) -> str:
        """Type 2 — destructive op with operator-editable parameters.

        Use when:
          • the action is HITL-gated AND the operator should be able to
            tweak the exact params before approval (e.g. config_lines for
            edit_device_config — operator might want to change the value)

        Callback should:
          • on EDIT decision: read decision.parameter_patch and merge
            into proposed_action.parameters before executing
          • on APPROVE decision: execute as-is
        """
        return await self._emit_hitl_payload(
            trigger_kind        = TriggerKind.DESTRUCTIVE,
            risk_level          = RiskLevel.MEDIUM,
            summary             = summary,
            proposed_action     = proposed_action,
            editable_param_keys = editable_keys,
            callback     = callback,
            context      = context,
            session_id   = session_id,
            event_queue  = event_queue,
            task_id      = task_id,
        )