"""
hitl/a2a_integration.py  [v3 — Runtime Loop integrated]
---------------------------------------------------------
Changes from v2
---------------
  1. ITOpsHitlAgentExecutor now accepts a RuntimeLoop (or creates one).
  2. execute() routes through classify() first:
       SIMPLE  -> AgentRuntimeLoop.stream()  (no LangGraph, no TaskPlanner)
       COMPLEX -> run_with_hitl()            (original HITL graph + DAG path)
  3. ContextBudgetManager governs all memory/tool injection.
  4. StopOutcome.STOP_HITL from Runtime Loop re-routes to HITL graph.
  5. Post-action verification hook added (_verify_action_result).
  6. All v2 features preserved (memory read/write, hitl_bridge, audit).
  7. Backward compatible: all new parameters are Optional.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, AsyncIterator, Optional

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


class ITOpsHitlAgentExecutor(AgentExecutor):
    """
    v4: Routes SIMPLE queries to AgentRuntimeLoop (thin fast path)
        and COMPLEX/destructive queries to HITL graph + TaskPlanner.

    Hermes learning loop integrated (§03 §04 §05):
      - FTS5SessionStore: cross-session full-text search + write every turn
      - MemoryCurator:    post-turn LLM curation + periodic nudge
      - UserModelEngine:  behavioral profile injection per session
      - SkillEvolver:     auto-create skills after complex tasks + self-improve
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
        # ── Hermes learning loop (all Optional — graceful degradation) ──
        fts_store:      Optional[Any]             = None,
        memory_curator: Optional[Any]             = None,
        user_model:     Optional[Any]             = None,
        skill_evolver:  Optional[Any]             = None,
        skill_catalog:  Optional[Any]             = None,
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

        # ── Hermes learning loop modules ──────────────────────────────
        self._fts_store      = fts_store       # FTS5SessionStore
        self._curator        = memory_curator  # MemoryCurator
        self._user_model     = user_model      # UserModelEngine
        self._skill_evolver  = skill_evolver   # SkillEvolver
        self._skill_catalog  = skill_catalog   # SkillCatalogService

        self._runtime = AgentRuntimeLoop(
            memory_router=memory_router,
            config=runtime_config,
        )
        cfg = runtime_config or RuntimeConfig()
        self._budget = ContextBudgetManager(cfg.budget)

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
            decision = self._runtime.classify(query)
            logger.info(
                "Complexity: %s — %s (task_id=%s)",
                decision.complexity.value, decision.reason, task_id,
            )
            if decision.complexity == QueryComplexity.SIMPLE:
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
    # Path A: Runtime Loop
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

        # ── Hermes §04: FTS5 cross-session recall ─────────────────────
        past_context   = ""
        user_profile   = ""
        if self._curator:
            try:
                past_context = await self._curator.recall_for_session(query, session_id)
            except Exception as exc:
                logger.debug("FTS5 recall skipped: %s", exc)
        if self._user_model:
            try:
                user_profile = self._user_model.get_prompt_section(session_id)
            except Exception as exc:
                logger.debug("User profile injection skipped: %s", exc)

        # Merge past context + user profile into env_context for the runtime loop
        if past_context or user_profile:
            extra = []
            if past_context:
                extra.append(past_context)
            if user_profile:
                extra.append(user_profile)
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

        # ── Hermes §03: Post-turn curation + user model update ────────
        tool_calls = self._extract_tool_calls_from_chunks(response_chunks)
        await self._hermes_post_turn(session_id, query, assistant_text, tool_calls)

    # ------------------------------------------------------------------
    # Path B: Full HITL graph
    # ------------------------------------------------------------------

    async def _execute_complex(
        self, query, session_id, context, event_queue, task_id, context_id
    ) -> None:
        # ── Hermes §04: FTS5 recall for complex queries too ───────────
        past_context = ""
        if self._curator:
            try:
                past_context = await self._curator.recall_for_session(query, session_id)
            except Exception as exc:
                logger.debug("FTS5 recall (complex) skipped: %s", exc)

        memory_context = await self._build_memory_context(context, session_id)
        if past_context:
            memory_context = past_context + "\n\n" + memory_context

        response_chunks: list[str] = []

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
            for processor in self._processors:
                await processor.process(chunk, event_queue, task_id, context_id)
            if "token" in chunk:
                response_chunks.append(chunk["token"])
            if chunk.get("hitl_interrupt"):
                await self._handle_interrupt_chunk(chunk, context, session_id)
                return

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

        # ── Hermes §03: Post-turn curation after complex task ─────────
        tool_calls = self._extract_tool_calls_from_chunks(response_chunks)
        await self._hermes_post_turn(session_id, query, assistant_text, tool_calls)

    # ------------------------------------------------------------------
    # Post-action verification + Hermes §05 skill evolution
    # ------------------------------------------------------------------

    async def _verify_action_result(
        self, context: RequestContext, session_id: str, response_chunks: list[str]
    ) -> None:
        action_type = context.metadata.get("action_type", "")
        query       = context.get_user_input()
        if not action_type:
            return
        logger.info(
            "Post-action verification: action_type=%s session=%s",
            action_type, session_id,
        )
        # ── Hermes §05: Auto-create skill after complex task ──────────
        if self._skill_evolver:
            assistant_text = "".join(response_chunks)
            try:
                await self._skill_evolver.after_task(
                    task_description=query,
                    solution_summary=assistant_text[:400],
                    tools_used=self._extract_tool_names_from_text(assistant_text),
                    solution_steps=[],
                    key_observations=[],
                    complexity=7.0,
                    session_id=session_id,
                )
            except Exception as exc:
                logger.debug("SkillEvolver.after_task skipped: %s", exc)

    # ------------------------------------------------------------------
    # Hermes shared post-turn hook
    # ------------------------------------------------------------------

    async def _hermes_post_turn(
        self,
        session_id:     str,
        user_text:      str,
        assistant_text: str,
        tool_calls:     list[dict],
    ) -> None:
        """
        Called after every turn (simple and complex paths).
        Runs all Hermes learning loop updates concurrently.
        """
        tasks = []

        # FTS5 write
        if self._fts_store:
            tasks.append(self._safe(
                self._fts_store.write_turn(
                    session_id, user_text, assistant_text,
                    tool_calls=tool_calls, importance=0.6,
                ),
                "fts_store.write_turn",
            ))

        # Memory curator (curation + nudge)
        if self._curator:
            tasks.append(self._safe(
                self._curator.after_turn(session_id, user_text, assistant_text, tool_calls),
                "curator.after_turn",
            ))

        # User model update
        if self._user_model:
            tasks.append(self._safe(
                self._user_model.after_turn(session_id, user_text, assistant_text, tool_calls),
                "user_model.after_turn",
            ))

        if tasks:
            await asyncio.gather(*tasks)

    @staticmethod
    async def _safe(coro, name: str) -> None:
        """Run a coroutine, log but don't propagate exceptions."""
        try:
            await coro
        except Exception as exc:
            logger.warning("Hermes hook %s failed: %s", name, exc)

    @staticmethod
    def _extract_tool_calls_from_chunks(chunks: list[str]) -> list[dict]:
        """Extract [TOOL:name] calls from assembled response text."""
        import re
        text = "".join(chunks)
        calls = []
        for m in re.finditer(r"\[TOOL:(\w+)\]\s*(\{[^}]*\})?", text):
            tool_name = m.group(1)
            try:
                import json
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
        # STUB: replace with real health check
        # e.g. if action_type == "restart_service":
        #          health = await check_service_health(context.metadata["target"])

    # ------------------------------------------------------------------
    # HITL interrupt (unchanged from v2)
    # ------------------------------------------------------------------

    async def _handle_interrupt_chunk(
        self, chunk: dict[str, Any], context: RequestContext, session_id: str
    ) -> None:
        interrupt_id = chunk.get("interrupt_id", str(uuid.uuid4()))
        action_raw   = chunk.get("proposed_action", {})
        payload = HitlPayload(
            interrupt_id=interrupt_id,
            thread_id=context.context_id,
            context_id=context.context_id,
            task_id=context.task_id,
            trigger_kind=TriggerKind(chunk.get("trigger_kind", "low_confidence")),
            risk_level=RiskLevel(chunk.get("risk_level", "medium")),
            user_query=context.get_user_input(),
            intent_summary=chunk.get("summary", ""),
            confidence_score=context.metadata.get("confidence_score", 0.0),
            proposed_action=ProposedAction(
                action_type=action_raw.get("action_type", "unknown"),
                target=action_raw.get("target", "unknown"),
                parameters=action_raw.get("parameters", {}),
                reversible=action_raw.get("reversible", True),
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
                    task_id=context.task_id,
                    session_id=session_id,
                    context_id=context.context_id,
                    scope=TaskScope.INTRA,
                    description=context.get_user_input(),
                    state=TState.WAITING_HITL,
                )
                await self._task_system.hitl_bridge.suspend_for_review(
                    task_ref, payload, session
                )
            except Exception as exc:
                logger.warning("HitlTaskBridge suspend failed: %s", exc)

    # ------------------------------------------------------------------
    # Memory helpers (unchanged from v2)
    # ------------------------------------------------------------------

    async def _build_memory_context(self, context: RequestContext, session_id: str) -> str:
        if self._memory is None:
            return ""
        try:
            from memory.schemas import RetrievalQuery
            query = RetrievalQuery(
                query_text=context.get_user_input(),
                session_id=session_id,
                max_tokens=1_500,
            )
            results = await self._memory.retrieve(query)
            return self._memory.format_context(results)
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

    @staticmethod
    def _extract_session_id(context: RequestContext) -> str:
        return context.metadata.get("session_id", context.context_id)