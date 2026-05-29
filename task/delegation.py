"""
task/delegation.py — Phase 2B capability-based delegation wiring
================================================================

Produces the ``delegate_fn`` that ``AgentRuntimeLoop`` calls when the LLM
emits a ``[DELEGATE:...]`` directive. Keeping this in the task/ package (not
runtime/) preserves module independence: the runtime loop receives an injected
async callable and never imports registry/ or task/ directly.

Contract (what the runtime loop expects):

    async def delegate_fn(
        directive,                # runtime.directive_parser.DelegateDirective
        session_id: str,
        shared_facts: list[str],  # parent confirmed_facts iff directive.forked
    ) -> AsyncIterator[dict]:     # streams chunk dicts back from the peer

Resolution rules (per PHASE_2B_DESIGN §5):
  - explicit agent_id  → registry.get_agent(agent_id) direct lookup
  - *capability        → registry.resolve(capability) (uses _pick strategy)
Failure is graceful: if the peer can't be resolved / isn't healthy, the
generator yields a short note + nothing else; the runtime loop turns that into
a "delegation unavailable, continue locally" context injection. We never raise
into the loop (same lesson as H2 — a hard error would make the LLM retry).

Audit scope (per decision Q4): the ENTRY agent records only the delegation
boundary (DISPATCHED + a terminal result event); the delegated agent audits
its own execution in its own process. Use context_id to join the two sides.
"""
from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Callable, Optional

logger = logging.getLogger(__name__)


def build_delegate_fn(
    registry: Any,                       # registry.AgentRegistry
    dispatcher: Any,                     # task.A2ATaskDispatcher
    task_store: Any,                     # task.TaskStore
    *,
    own_agent_id: str = "",
) -> Callable[..., AsyncIterator[dict]]:
    """Build the delegate_fn closure injected into AgentRuntimeLoop.

    Args:
        registry:   AgentRegistry — resolves agent_id / capability → peer.
        dispatcher: A2ATaskDispatcher — streams the subtask to the peer.
        task_store: TaskStore — persists the TaskDefinition (DISPATCHED, etc).
        own_agent_id: this agent's id, used to exclude self from capability
                      resolution (an agent shouldn't delegate to itself).

    Returns:
        An async generator function matching the runtime loop's contract.
    """
    # Imports kept local so importing this module never drags task/registry
    # schemas into callers that only need the type at call time.
    from task.schemas import TaskDefinition, AgentAssignment

    async def _resolve_assignment(directive) -> Optional["AgentAssignment"]:
        """Resolve a DelegateDirective to a concrete AgentAssignment, or None."""
        # Explicit agent_id → direct lookup.
        if not directive.by_capability:
            agent = await registry.get_agent(directive.agent_id)
            if agent is None:
                logger.warning("delegate: unknown agent_id=%s", directive.agent_id)
                return None
            if not getattr(agent, "is_available", True):
                logger.warning("delegate: agent %s not available", directive.agent_id)
                return None
            # Pick a representative skill_id for the assignment metadata. The
            # peer runs its full agent loop on the task text regardless, so
            # the exact skill_id is informational; use the first advertised.
            skills = list(getattr(agent.card, "skills", []) or [])
            skill_id = skills[0].id if skills else "delegated_task"
            return AgentAssignment(
                agent_id=agent.agent_id,
                agent_url=agent.base_url,
                skill_id=skill_id,
            )
        # *capability → registry.resolve (applies _pick strategy + excludes self).
        excl = [own_agent_id] if own_agent_id else None
        res = await registry.resolve(directive.capability, exclude_agent_ids=excl)
        if res is None:
            logger.warning("delegate: no peer for capability=%s", directive.capability)
            return None
        return AgentAssignment(
            agent_id=res.agent_id,
            agent_url=res.agent_url,
            skill_id=res.skill_id,
        )

    async def delegate_fn(
        directive,
        session_id: str,
        shared_facts: list[str],
        *,
        original_query: str = "",
    ) -> AsyncIterator[dict]:
        assignment = await _resolve_assignment(directive)
        if assignment is None:
            # Resolution failed — yield a single note. The loop turns the
            # absence of a result into a "continue locally" injection.
            yield {
                "node_step": f"Delegation target unresolved: {directive.target}",
                "node": "delegate",
                "source_agent": directive.target,
            }
            return

        # ── SINGLE DELEGATION GATE (A2A Phase 3) ──────────────────────────
        # Identity of a delegated task = (session_id, target_agent): while a
        # delegation to this peer is still in flight (any non-terminal state),
        # the SAME originating request must NOT be delegated to the SAME peer
        # again. This is the one source of truth that replaces the old
        # env_ctx-scoped guards (count / pending-set / resume-flag), which
        # broke because env_ctx is per-execute_query and reset every time the
        # resume driver started a fresh synthesis turn. TaskStore is durable
        # across turns AND streams, and is the same store the UI reads — so
        # the gate and the UI can never disagree.
        #
        # Terminal states (delegation finished, a fresh one is allowed):
        #   COMPLETED, FAILED, CANCELLED.
        # Everything else (RUNNING, AWAITING_PEER_HITL, PENDING, …) is in
        # flight → suppress.
        try:
            from task.schemas import TaskState, TaskScope
            _TERMINAL = {TaskState.COMPLETED, TaskState.FAILED,
                         TaskState.CANCELLED}
            _existing = await task_store.get_by_session(session_id)
            _inflight = next(
                (t for t in _existing
                 if t.assignment is not None
                 and t.assignment.agent_id == assignment.agent_id
                 and t.scope == TaskScope.INTER
                 and t.state not in _TERMINAL),
                None,
            )
            if _inflight is not None:
                logger.warning(
                    "delegate gate: suppressing duplicate delegation to %s "
                    "(session=%s) — existing task %s is in-flight (state=%s)",
                    assignment.agent_id, session_id,
                    _inflight.task_id[:12], _inflight.state.value,
                )
                yield {
                    "node_step": f"Delegation to {assignment.agent_id} "
                                 f"already in progress",
                    "node": "delegate",
                    "source_agent": assignment.agent_id,
                }
                yield {
                    "_inject_context":
                        f"[委派进行中] 你已就当前请求委派 {assignment.agent_id}，"
                        f"该委派尚未完成（状态：{_inflight.state.value}）。"
                        f"⚠ 禁止再次委派 {assignment.agent_id}。"
                        f"结果会在对方完成后自动补充。请基于现有信息给出简短中间答复，"
                        f"或处理其他未委派的子任务，然后结束本轮。",
                    "_delegation_suppressed": True,
                }
                return
        except Exception as _gate_exc:
            # Gate must never break delegation on an internal error; log and
            # fall through to normal dispatch.
            logger.debug("delegate gate check skipped: %s", _gate_exc)

        # Build the subtask. context_id = session-derived join key shared with
        # the peer's audit. parameters carry shared facts only when forked.
        params: dict[str, Any] = {}
        if directive.forked and shared_facts:
            params["parent_confirmed_facts"] = list(shared_facts)
        # metadata carries delegation provenance — the dispatcher embeds
        # this in the A2A request's params.metadata so the peer's
        # agent_executor can read it via context.metadata. Peer-side HITL
        # cards pull source_agent / source_session_id / source_query from
        # here. Keep keys stable; downstream code reads them by name.
        task = TaskDefinition(
            session_id=session_id,
            context_id=session_id,           # 1:1 for single-hop; Phase 3 may derive
            scope=TaskScope.INTER,           # outbound delegation — gate keys on this
            description=directive.task,
            assignment=assignment,
            parameters=params,
            metadata={
                "delegated_by":         own_agent_id,
                "forked":               directive.forked,
                "shared_facts_count":   len(shared_facts) if directive.forked else 0,
                # Provenance for peer-side HITL cards (Phase 2B+, 2026-05).
                # Empty original_query is fine — peer simply skips the
                # "原 query" banner row.
                "source_agent":         own_agent_id,
                "source_session_id":    session_id,
                "source_query":         original_query or "",
            },
        )

        registry.record_task_start(assignment.agent_id)
        try:
            async for chunk in dispatcher.dispatch(task, assignment, task_store):
                yield chunk
        except Exception as exc:
            logger.exception("delegate: dispatch to %s failed: %s",
                             assignment.agent_id, exc)
            yield {
                "node_step": f"Delegation dispatch error: {exc}",
                "node": "delegate",
                "source_agent": assignment.agent_id,
            }
        finally:
            registry.record_task_end(assignment.agent_id)

    return delegate_fn
