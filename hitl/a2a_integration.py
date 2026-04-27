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
            if "stop_hitl" in msg or "HITL" in msg:
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
            interrupt_chunk = {
                "hitl_interrupt": True,
                "interrupt_id":   interrupt_id,
                "trigger_kind":   payload.trigger_kind.value,
                "risk_level":     payload.risk_level.value,
                "summary":        payload.intent_summary,
                "proposed_action": payload.proposed_action.model_dump(),
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
                _tool=force_tool, _args=force_args, _reg=_tool_reg,
                _sid=session_id, _q=query,
            ):
                if _tool not in _reg:
                    return {"error": f"Tool {_tool!r} not in registry after approval"}
                try:
                    raw = await _reg[_tool](_args)
                    await self._hermes_post_turn(
                        _sid, _q,
                        f"[TOOL:{_tool}] approved and executed. Result: {str(raw)[:300]}", []
                    )
                    return {"tool": _tool, "args": _args, "result": str(raw)}
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
        )
        await self._hitl_router.register_interrupt(payload)
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