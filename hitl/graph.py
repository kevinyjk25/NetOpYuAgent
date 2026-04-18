"""
hitl/graph.py
-------------
Layer 2 — LangGraph StateGraph with HITL interrupt support.

Graph node order
----------------
  intent_classifier
      ↓  (route_after_intent)
  risk_assessor
      ↓  (route_after_risk)
  planner
      ↓  (route_after_plan)   ← triggers evaluated here
  [interrupt checkpoint]       ← graph pauses here if trigger fires
  executor
      ↓
  result_formatter
      ↓
  END

Key LangGraph primitives used
------------------------------
  interrupt()           – pause the graph and surface payload to the caller
  graph.update_state()  – patch state before resuming (operator edits)
  graph.invoke(None)    – resume from the last checkpoint

Checkpoint backend
------------------
  Development : MemorySaver (in-process, lost on restart)
  Production  : Use langgraph-checkpoint-redis or langgraph-checkpoint-postgres
                Drop-in: replace MemorySaver with RedisSaver / PostgresSaver

Usage
-----
    graph = build_hitl_graph()
    config = {"configurable": {"thread_id": "conv-abc-123"}}

    # First run — may pause at interrupt
    result = await graph.ainvoke(initial_state, config)

    # Resume after operator approves
    result = await graph.ainvoke(None, config)

    # Resume after operator edits params
    graph.update_state(config, {"proposed_action": edited_action})
    result = await graph.ainvoke(None, config)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, AsyncIterator, Optional
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from .schemas import (
    AuditEventKind,
    DecisionKind,
    HitlDecision,
    HitlPayload,
    HitlState,
    ProposedAction,
    RiskLevel,
    TriggerKind,
)
from .triggers import HitlConfig, HitlTrigger, build_trigger_chain, evaluate_triggers

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class ITOpsGraphState(TypedDict, total=False):
    """
    Shared state for the IT-Ops HITL graph.

    Using TypedDict (not bare dict) is critical: LangGraph merges each node's
    return dict into the accumulated state ({**old, **new}). With a bare dict
    each node receives ONLY the previous node's return value — earlier keys
    like intent_type are silently dropped before planner_node runs.
    """
    # Input
    query:            str
    thread_id:        str
    context_id:       str
    task_id:          str
    user_metadata:    dict

    # intent_classifier outputs
    intent_type:       str
    intent_confidence: float
    intent_candidates: list
    intent_summary:    str

    # risk_assessor outputs
    is_destructive:   bool
    risk_level:       Any   # RiskLevel enum
    risk_reasons:     list

    # planner outputs
    proposed_action:  dict
    plan_steps:       list

    # hitl_interrupt_node outputs
    hitl_required:    bool
    hitl_payload:     dict
    hitl_decision:    dict

    # executor outputs
    execution_results: list
    error:             Any

    # result_formatter outputs
    emitted_chunks:   list

    # Internal: injected by build_hitl_graph for hitl_interrupt_node
    _hitl_config:     Any


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

async def intent_classifier_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Classify the user's query into an intent type.

    Replace the stub below with your real LangChain / LLM chain call, e.g.:
        result = await intent_chain.ainvoke({"query": state["query"]})
    """
    query: str = state.get("query", "")
    logger.info("intent_classifier: query=%r", query[:80])

    # ── Stub: simple keyword routing ──────────────────────────────────
    query_lower = query.lower()
    if any(k in query_lower for k in ("restart", "rollback", "delete", "drain")):
        intent_type = "destructive_op"
        confidence  = 0.95
        candidates  = [{"intent": "destructive_op", "confidence": 0.95}]
    elif any(k in query_lower for k in ("alert", "alarm", "p0", "p1", "outage")):
        intent_type = "alert_analysis"
        confidence  = 0.88
        candidates  = [{"intent": "alert_analysis", "confidence": 0.88}]
    elif any(k in query_lower for k in ("predict", "forecast", "trend")):
        intent_type = "trend_prediction"
        confidence  = 0.82
        candidates  = [{"intent": "trend_prediction", "confidence": 0.82}]
    else:
        intent_type = "general_query"
        confidence  = 0.65
        candidates  = [
            {"intent": "general_query", "confidence": 0.65},
            {"intent": "alert_analysis", "confidence": 0.55},
        ]

    return {
        "intent_type": intent_type,
        "intent_confidence": confidence,
        "intent_candidates": candidates,
        "intent_summary": f"User intent: {intent_type} (confidence={confidence:.2f})",
    }


async def risk_assessor_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Assess the risk of executing the classified intent.
    Sets is_destructive, risk_level, and risk_reasons.
    """
    intent_type: str = state.get("intent_type", "")
    query: str       = state.get("query", "").lower()

    is_destructive = intent_type == "destructive_op"
    risk_reasons: list[str] = []

    if is_destructive:
        risk_reasons.append("Action type is classified as destructive.")

    recent_alerts: list[dict] = state.get("user_metadata", {}).get("recent_alerts", [])
    has_critical = any(
        a.get("severity", "").upper() in ("P0", "P1") for a in recent_alerts
    )
    if has_critical:
        risk_reasons.append("Active P0/P1 alerts exist in the environment.")

    if "production" in query or "prod" in query:
        risk_reasons.append("Query references production environment.")

    risk_level = (
        RiskLevel.CRITICAL if is_destructive and has_critical
        else RiskLevel.HIGH if is_destructive or has_critical
        else RiskLevel.MEDIUM if risk_reasons
        else RiskLevel.LOW
    )

    logger.info("risk_assessor: risk=%s reasons=%s", risk_level, risk_reasons)
    return {
        "is_destructive": is_destructive,
        "risk_level": risk_level,
        "risk_reasons": risk_reasons,
    }


async def planner_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Build the action plan from the classified intent.

    Replace the stub below with your real planning chain.
    """
    intent_type: str = state.get("intent_type", "")
    query: str       = state.get("query", "")

    # ── Stub: intent → action mapping ────────────────────────────────
    if intent_type == "destructive_op":
        action = {
            "action_type": "restart_service",
            "target": "payments-service / prod",
            "parameters": {"host_count": 3, "rolling": True},
            "estimated_impact": "~2 min downtime per pod; rolling restart",
            "reversible": True,
        }
        plan_steps = [
            "Cordon affected pods",
            "Perform rolling restart (1 pod at a time)",
            "Wait for health checks",
            "Uncordon pods",
        ]
    elif intent_type == "alert_analysis":
        action = {
            "action_type": "query_alert_store",
            "target": "prometheus / alertmanager",
            "parameters": {"time_range": "24h", "severity_filter": "all"},
            "reversible": True,
        }
        plan_steps = ["Query Prometheus", "Aggregate by severity", "Summarise trends"]
    else:
        action = {
            "action_type": "llm_answer",
            "target": "knowledge_base",
            "parameters": {"query": query},
            "reversible": True,
        }
        plan_steps = ["Retrieve context", "Generate answer", "Format response"]

    logger.info("planner: action_type=%s", action["action_type"])
    return {"proposed_action": action, "plan_steps": plan_steps}


async def executor_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Execute the approved action plan.

    In pragmatic mode: dispatches to real tools via _tool_registry injected in state.
    In mock mode: returns stub completions for each step.
    Tool registry is injected by patch_hitl_graph(engine, tool_registry=...) in main.py.
    """
    action        = state.get("proposed_action", {})
    steps         = state.get("plan_steps", [])
    tool_registry = state.get("_tool_registry") or {}
    results       = []

    for step in steps:
        logger.info("executor: step=%r", step)
        result_text = f"Completed: {step}"

        if tool_registry:
            # Best-effort: match step text to a tool name
            step_lower = step.lower().replace(" ", "_").replace("-", "_")
            matched = None
            for tname in tool_registry:
                if tname in step_lower or tname.replace("_", " ") in step.lower():
                    matched = tname
                    break
            if matched:
                try:
                    raw = await tool_registry[matched](action.get("parameters", {}))
                    result_text = f"[{matched}] {str(raw)[:600]}"
                    logger.info("executor: dispatched %s → %d chars", matched, len(str(raw)))
                except Exception as exc:
                    result_text = f"[{matched}] ERROR: {exc}"
                    logger.warning("executor: tool %s failed: %s", matched, exc)

        await asyncio.sleep(0)
        results.append({"step": step, "status": "ok", "output": result_text})

    return {"execution_results": results, "error": None}


async def result_formatter_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Format execution results into chunks for the A2A EventQueue.

    Emits dicts compatible with the 5 A2AEventProcessor classes in a2a/agent_executor.py.
    """
    results  = state.get("execution_results", [])
    action   = state.get("proposed_action", {})
    decision = state.get("hitl_decision")

    chunks: list[dict[str, Any]] = []

    # Opening message
    chunks.append({
        "message": f"Executing: {action.get('action_type', 'operation')} on {action.get('target', 'target')}",
        "node": "result_formatter",
    })

    # Per-step results as node_result
    for r in results:
        chunks.append({"node_step": r["step"], "node": "executor"})
        chunks.append({
            "node_result": {"summary": r["output"], "raw_data": str(r)},
            "node": "executor_result",
        })

    # Operator decision note
    if decision:
        dec = HitlDecision.model_validate(decision)
        note = f"Operator '{dec.operator_id}' {dec.decision.value}d this action."
        if dec.comment:
            note += f" Comment: {dec.comment}"
        chunks.append({"message": note, "node": "result_formatter"})

    # Closing token stream
    summary = f"Completed {len(results)} step(s) for {action.get('action_type', 'operation')}."
    for word in summary.split():
        chunks.append({"token": word + " "})

    return {"emitted_chunks": chunks}


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_intent(state: dict[str, Any]) -> str:
    return "risk_assessor"


def route_after_risk(state: dict[str, Any]) -> str:
    return "planner"


def route_after_plan(
    state: dict[str, Any],
    triggers: list[HitlTrigger],
) -> str:
    """
    Evaluate all triggers against the current state.
    If any fires, route to the interrupt node; otherwise go straight to executor.

    NOTE: Do NOT mutate `state` here. LangGraph conditional edge functions are
    routing-only — their return value is the next node name, and any in-place
    mutations to `state` are NOT persisted to the checkpoint store. The payload
    is written by planner_node into the returned state dict instead.
    """
    hitl_state = HitlState.model_validate(state)
    thread_id  = state.get("thread_id", str(uuid.uuid4()))
    context_id = state.get("context_id", thread_id)
    task_id    = state.get("task_id", thread_id)

    logger.info(
        "route_after_plan: intent_type=%r is_destructive=%r risk_level=%r action_type=%r",
        state.get("intent_type"),
        state.get("is_destructive"),
        getattr(state.get("risk_level"), "value", state.get("risk_level")),
        (state.get("proposed_action") or {}).get("action_type"),
    )

    payload = evaluate_triggers(hitl_state, triggers, thread_id, context_id, task_id)
    if payload:
        return "hitl_interrupt_node"

    return "executor"


def route_after_decision(state: dict[str, Any]) -> str:
    """Route based on the operator's decision."""
    decision_raw = state.get("hitl_decision")
    if not decision_raw:
        # No decision yet — graph should not reach here; safety fallback
        return END

    decision = HitlDecision.model_validate(decision_raw)
    if decision.decision in (DecisionKind.APPROVE, DecisionKind.EDIT):
        return "executor"
    elif decision.decision in (DecisionKind.REJECT, DecisionKind.ESCALATE, DecisionKind.TIMEOUT):
        return END
    else:
        return END


# ---------------------------------------------------------------------------
# HITL interrupt node
# ---------------------------------------------------------------------------

async def hitl_interrupt_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    This node calls LangGraph's interrupt() to pause the graph.

    interrupt() raises an internal exception that LangGraph catches, serialises
    the full state to the checkpoint backend, and surfaces the payload to the
    caller via the graph's stream / invoke return value.

    The graph resumes when:
      1. The decision router POSTs a HitlDecision to /hitl/decisions/{interrupt_id}
      2. The caller calls graph.invoke(None, config={"configurable": {"thread_id": ...}})
         — optionally after graph.update_state() for operator edits

    IMPORTANT: We build the payload HERE (not in route_after_plan) because node
    return values ARE persisted to the checkpoint, but conditional edge function
    mutations are NOT.
    """
    thread_id  = state.get("thread_id",  str(uuid.uuid4()))
    context_id = state.get("context_id", thread_id)
    task_id    = state.get("task_id",    thread_id)

    # Build triggers and evaluate against current state
    # (same logic as route_after_plan, but here the result is persisted)
    config     = state.get("_hitl_config")  # injected by build_hitl_graph if provided
    triggers   = build_trigger_chain(config if isinstance(config, HitlConfig) else None)
    hitl_state = HitlState.model_validate(state)
    payload    = evaluate_triggers(hitl_state, triggers, thread_id, context_id, task_id)

    if payload is None:
        # Construct a minimal destructive-action payload from state if triggers
        # didn't fire (can happen if classify=complex but trigger thresholds differ)
        payload = HitlPayload(
            interrupt_id    = str(uuid.uuid4()),
            thread_id       = thread_id,
            context_id      = context_id,
            task_id         = task_id,
            trigger_kind    = TriggerKind.DESTRUCTIVE_OP,
            risk_level      = RiskLevel.HIGH,
            user_query      = state.get("query", ""),
            intent_summary  = state.get("intent_summary", state.get("query", ""))[:200],
            proposed_action = ProposedAction(
                action_type = state.get("action_type", "unknown"),
                target      = state.get("target",      "unknown"),
                parameters  = state.get("parameters",  {}),
                reversible  = state.get("reversible",  False),
            ),
        )

    logger.info(
        "HITL interrupt — interrupt_id=%s trigger=%s risk=%s",
        payload.interrupt_id,
        payload.trigger_kind.value,
        payload.risk_level.value,
    )

    # ── This call pauses the graph ────────────────────────────────────
    # The dict passed to interrupt() is returned to the caller as the
    # graph's result until the graph is resumed.
    operator_decision: dict[str, Any] = interrupt(payload.model_dump())
    # ─────────────────────────────────────────────────────────────────

    # After resume, operator_decision contains the HitlDecision payload
    logger.info("HITL resumed — decision=%s", operator_decision.get("decision"))
    return {
        "hitl_decision":  operator_decision,
        "hitl_required":  True,
        "hitl_payload":   payload.model_dump(),
    }


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def build_hitl_graph(
    config: Optional[HitlConfig] = None,
    checkpointer=None,
) -> Any:
    """
    Build and compile the HITL StateGraph.

    Parameters
    ----------
    config:
        Trigger thresholds and SLA map. Defaults to HitlConfig().
    checkpointer:
        LangGraph checkpoint backend. Defaults to MemorySaver (dev only).
        Production: pass RedisSaver or PostgresSaver.

    Returns
    -------
    CompiledStateGraph
        Call .ainvoke() / .astream() / .update_state() on this object.

    Example
    -------
        graph = build_hitl_graph()
        thread_cfg = {"configurable": {"thread_id": "conv-001"}}
        result = await graph.ainvoke({"query": "restart payments-service"}, thread_cfg)
    """
    triggers   = build_trigger_chain(config)
    saver      = checkpointer or MemorySaver()
    builder    = StateGraph(ITOpsGraphState)

    # ── Nodes ────────────────────────────────────────────────────────
    builder.add_node("intent_classifier",  intent_classifier_node)
    builder.add_node("risk_assessor",      risk_assessor_node)
    builder.add_node("planner",            planner_node)
    builder.add_node("hitl_interrupt_node", hitl_interrupt_node)
    builder.add_node("executor",           executor_node)
    builder.add_node("result_formatter",   result_formatter_node)

    # ── Edges ────────────────────────────────────────────────────────
    builder.set_entry_point("intent_classifier")
    builder.add_edge("intent_classifier", "risk_assessor")
    builder.add_edge("risk_assessor",     "planner")

    # Conditional: planner → [hitl_interrupt_node | executor]
    builder.add_conditional_edges(
        "planner",
        lambda s: route_after_plan(s, triggers),
        {
            "hitl_interrupt_node": "hitl_interrupt_node",
            "executor":            "executor",
        },
    )

    # Conditional: hitl_interrupt_node → [executor | END]
    builder.add_conditional_edges(
        "hitl_interrupt_node",
        route_after_decision,
        {
            "executor": "executor",
            END:        END,
        },
    )

    builder.add_edge("executor",         "result_formatter")
    builder.add_edge("result_formatter", END)

    return builder.compile(checkpointer=saver)


# ---------------------------------------------------------------------------
# High-level async generator (plug into ITOpsAgentExecutor._run_agent)
# ---------------------------------------------------------------------------

async def run_with_hitl(
    query: str,
    thread_id: str,
    context_id: str,
    task_id: str,
    user_metadata: Optional[dict[str, Any]] = None,
    hitl_config: Optional[HitlConfig] = None,
    graph=None,
) -> AsyncIterator[dict[str, Any]]:
    """
    High-level async generator that runs the HITL graph and yields A2A-compatible
    chunk dicts.  Plug this into ``ITOpsAgentExecutor._run_agent()``.

    Special chunk shapes emitted
    ----------------------------
    All standard chunks (token, tokens, message, node_result, node_step) plus:

        {
            "hitl_interrupt": True,
            "interrupt_id": str,
            "trigger_kind": str,
            "risk_level": str,
            "summary": str,
            "proposed_action": dict,
            "node": "hitl",
        }

    The caller (ITOpsAgentExecutor) should surface this via A2ANodeStepProcessor
    or a dedicated HitlEventProcessor as a TaskArtifactUpdateEvent with
    artifact.name == "hitl_interrupt".

    Parameters
    ----------
    query        : User's natural-language query.
    thread_id    : LangGraph thread ID (= A2A context_id / conversation ID).
    context_id   : A2A context_id.
    task_id      : A2A task_id.
    user_metadata: Optional dict with recent_alerts, user_id, etc.
    hitl_config  : Custom HitlConfig; defaults to HitlConfig().
    graph        : Pre-built compiled graph; built fresh if None.
    """
    g = graph or build_hitl_graph(hitl_config)
    thread_cfg = {"configurable": {"thread_id": thread_id}}

    initial_state: dict[str, Any] = {
        "query":         query,
        "thread_id":     thread_id,
        "context_id":    context_id,
        "task_id":       task_id,
        "user_metadata": user_metadata or {},
    }

    # ── Run the graph — detect interrupt via astream ──────────────────
    # LangGraph's interrupt() mechanism varies by version:
    #   values stream  : __interrupt__ key MAY appear in state snapshot
    #   updates stream : __interrupt__ appears as a standalone update event (≥0.2.30)
    #   GraphInterrupt : raised directly from astream in some builds
    #   aget_state()   : ALWAYS reliable — checkpointer has the state
    #
    # We try all four methods in order, log every step for diagnostics.
    interrupt_payload_raw: Optional[dict[str, Any]] = None
    final_state: dict[str, Any] = {}

    try:
        async for event_type, state_update in g.astream(
            initial_state, thread_cfg, stream_mode=["values", "updates"]
        ):
            if not isinstance(state_update, dict):
                continue

            logger.debug(
                "Graph stream event: type=%s keys=%s",
                event_type,
                list(state_update.keys()),
            )

            if event_type == "values":
                final_state = state_update
                # Method 1: __interrupt__ key in full-state snapshot
                interrupts = state_update.get("__interrupt__") or []
                if interrupts:
                    iv = interrupts[0]
                    interrupt_payload_raw = (
                        iv.value if hasattr(iv, "value")
                        else iv if isinstance(iv, dict) else {}
                    )
                    logger.info("Graph: interrupt via __interrupt__ in values snapshot")
                    break

            elif event_type == "updates":
                # Method 2: __interrupt__ in a per-node update event (≥0.2.30)
                interrupts = state_update.get("__interrupt__") or []
                if interrupts:
                    iv = interrupts[0]
                    interrupt_payload_raw = (
                        iv.value if hasattr(iv, "value")
                        else iv if isinstance(iv, dict) else {}
                    )
                    logger.info("Graph: interrupt via __interrupt__ in updates event")
                    break
                # Merge node outputs into final_state for the hitl_payload fallback
                for node_name, node_output in state_update.items():
                    if node_name != "__interrupt__" and isinstance(node_output, dict):
                        final_state.update(node_output)

    except Exception as stream_exc:
        # Method 3: some LangGraph builds raise GraphInterrupt from astream itself
        exc_name = type(stream_exc).__name__
        logger.info(
            "Graph stream raised %s — treating as interrupt: %s", exc_name, stream_exc
        )
        raw = getattr(stream_exc, "value", None) or (
            stream_exc.args[0] if stream_exc.args else {}
        )
        if isinstance(raw, dict):
            interrupt_payload_raw = raw
        elif hasattr(raw, "__dict__"):
            interrupt_payload_raw = vars(raw)

    # Method 4: hitl_payload in node-persisted state
    # hitl_interrupt_node now writes this into its return dict (a node return IS persisted)
    if interrupt_payload_raw is None:
        raw = final_state.get("hitl_payload")
        if raw and final_state.get("hitl_required"):
            logger.info("Graph: interrupt via hitl_payload in final_state")
            interrupt_payload_raw = raw

    # Method 5: aget_state() — most reliable across all versions.
    # When interrupt() is called, LangGraph checkpoints the state BEFORE raising.
    # aget_state() on the same thread_cfg always returns the paused snapshot.
    if interrupt_payload_raw is None:
        try:
            snapshot = await g.aget_state(thread_cfg)
            logger.debug(
                "aget_state snapshot: has_tasks=%s values_keys=%s",
                bool(snapshot and snapshot.tasks),
                list((snapshot.values if snapshot else {}).keys()),
            )
            if snapshot and snapshot.tasks:
                for task in snapshot.tasks:
                    task_interrupts = getattr(task, "interrupts", [])
                    if task_interrupts:
                        iv = task_interrupts[0]
                        interrupt_payload_raw = (
                            iv.value if hasattr(iv, "value")
                            else iv if isinstance(iv, dict) else {}
                        )
                        logger.info("Graph: interrupt via aget_state().tasks[].interrupts")
                        break
            if interrupt_payload_raw is None and snapshot:
                snap_vals = snapshot.values if hasattr(snapshot, "values") else {}
                raw = snap_vals.get("hitl_payload")
                if raw and snap_vals.get("hitl_required"):
                    logger.info("Graph: interrupt via aget_state().values.hitl_payload")
                    interrupt_payload_raw = raw
        except Exception as snap_exc:
            logger.debug("aget_state() fallback failed: %s", snap_exc)

    logger.info(
        "Graph interrupt detection complete: found=%s",
        interrupt_payload_raw is not None,
    )

    # ── Interrupt detected ────────────────────────────────────────────
    if interrupt_payload_raw:
        try:
            payload = HitlPayload.model_validate(interrupt_payload_raw)
        except Exception:
            # interrupt_payload_raw might already be a HitlPayload dict
            # Build a minimal payload from available keys
            payload = HitlPayload(
                interrupt_id    = interrupt_payload_raw.get("interrupt_id", str(uuid.uuid4())),
                thread_id       = thread_id,
                context_id      = context_id,
                task_id         = task_id,
                trigger_kind    = TriggerKind(interrupt_payload_raw.get("trigger_kind", "destructive_op")),
                risk_level      = RiskLevel(interrupt_payload_raw.get("risk_level", "high")),
                user_query      = query,
                intent_summary  = interrupt_payload_raw.get("intent_summary", query[:120]),
                proposed_action = ProposedAction(
                    action_type = interrupt_payload_raw.get("action_type", "unknown"),
                    target      = interrupt_payload_raw.get("target", "unknown"),
                    parameters  = interrupt_payload_raw.get("parameters", {}),
                    reversible  = interrupt_payload_raw.get("reversible", True),
                ),
            )

        yield {
            "node_step": f"Human review required — {payload.trigger_kind.value}",
            "node": "hitl",
        }
        yield {
            "hitl_interrupt":  True,
            "interrupt_id":    payload.interrupt_id,
            "trigger_kind":    payload.trigger_kind.value,
            "risk_level":      payload.risk_level.value,
            "summary":         payload.intent_summary,
            "proposed_action": payload.proposed_action.model_dump(),
            "thread_id":       thread_id,
            "node":            "hitl",
        }
        return

    # ── No interrupt — yield emitted chunks from result_formatter ────
    for chunk in final_state.get("emitted_chunks", []):
        yield chunk
        await asyncio.sleep(0)