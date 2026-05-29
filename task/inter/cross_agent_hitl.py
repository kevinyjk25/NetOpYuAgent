"""
task/inter/cross_agent_hitl.py
==============================

CrossAgentHitlBridge — correlates a cross-agent HITL interrupt with the
originating agent's resume callback (A2A Phase 3, mode B: the delegated-to
side approves; the result is pushed back to the delegating side).

Two directions of state, kept in two small in-memory maps (persistence is
deferred to P3-d):

  1. DELEGATED-TO side (e.g. dc): when an inbound delegation raises a local
     HITL, record where to call back when the operator resolves it —
     keyed by the local interrupt_id, value = {source_agent, source_session_id,
     correlation_id}. The dc resume hook looks this up to POST the result to
     lan's /hitl_resolved endpoint (URL resolved from source_agent via the
     registry).

  2. DELEGATING side (e.g. lan): when an outbound delegation comes back
     INPUT_REQUIRED, record the awaiting context keyed by (peer_agent,
     peer_interrupt_id) so the inbound /hitl_resolved callback can find the
     local session to resume.

This module holds NO network/business logic — it is pure correlation state.
Thread-safety: all access is from the asyncio event loop (A2A handlers +
HITL resolve callbacks), so plain dict ops are safe without a lock.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PeerHitlRecord:
    """Delegated-to side: how to call back the originator on resolution."""
    interrupt_id:       str
    source_agent:       str            # who delegated to us (e.g. "lan-agent")
    source_session_id:  str            # the originator's session to resume
    correlation_id:     str            # chains this across agents (P3-c uses it)
    source_query:       str = ""


@dataclass
class AwaitingPeerRecord:
    """Delegating side: a local session awaiting a peer's HITL resolution."""
    local_session_id:   str
    peer_agent:         str
    peer_interrupt_id:  str
    correlation_id:     str
    original_query:     str = ""
    outbound_task_id:   str = ""    # the LAN task to transition COMPLETED on terminal callback
    resumed:            bool = False    # guard against double-resume


class CrossAgentHitlBridge:
    """In-memory correlation store for cross-agent HITL (P3-b)."""

    def __init__(self) -> None:
        # delegated-to side: local interrupt_id -> PeerHitlRecord
        self._inbound: dict[str, PeerHitlRecord] = {}
        # delegating side: (peer_agent, peer_interrupt_id) -> AwaitingPeerRecord
        self._awaiting: dict[tuple[str, str], AwaitingPeerRecord] = {}

    # ── delegated-to side (e.g. dc) ───────────────────────────────────
    def record_inbound_hitl(
        self,
        *,
        interrupt_id: str,
        source_agent: str,
        source_session_id: str,
        source_query: str = "",
        correlation_id: Optional[str] = None,
    ) -> str:
        """Called when an inbound delegation raises a local HITL. Returns the
        correlation_id (generated if not supplied)."""
        cid = correlation_id or f"xahitl-{uuid.uuid4().hex[:12]}"
        self._inbound[interrupt_id] = PeerHitlRecord(
            interrupt_id=interrupt_id,
            source_agent=source_agent,
            source_session_id=source_session_id,
            correlation_id=cid,
            source_query=source_query,
        )
        logger.info(
            "CrossAgentHitlBridge: recorded inbound HITL interrupt=%s "
            "source=%s/%s cid=%s",
            interrupt_id[:12], source_agent, source_session_id[:12], cid,
        )
        return cid

    def pop_inbound_hitl(self, interrupt_id: str) -> Optional[PeerHitlRecord]:
        """Called on the delegated-to side when the operator resolves the
        interrupt — returns the callback record (and removes it)."""
        return self._inbound.pop(interrupt_id, None)

    def peek_inbound_hitl(self, interrupt_id: str) -> Optional[PeerHitlRecord]:
        return self._inbound.get(interrupt_id)

    # ── delegating side (e.g. lan) ────────────────────────────────────
    def record_awaiting_peer(
        self,
        *,
        local_session_id: str,
        peer_agent: str,
        peer_interrupt_id: str,
        correlation_id: Optional[str] = None,
        original_query: str = "",
        outbound_task_id: str = "",
    ) -> None:
        """Called when an outbound delegation returns INPUT_REQUIRED, so the
        later /hitl_resolved callback can find the local session to resume
        AND transition the corresponding outbound task to COMPLETED."""
        self._awaiting[(peer_agent, peer_interrupt_id)] = AwaitingPeerRecord(
            local_session_id=local_session_id,
            peer_agent=peer_agent,
            peer_interrupt_id=peer_interrupt_id,
            correlation_id=correlation_id or "",
            original_query=original_query,
            outbound_task_id=outbound_task_id,
        )
        logger.info(
            "CrossAgentHitlBridge: awaiting peer HITL peer=%s interrupt=%s "
            "local_session=%s",
            peer_agent, peer_interrupt_id[:12], local_session_id[:12],
        )

    def resolve_awaiting_peer(
        self, *, peer_agent: str, peer_interrupt_id: str,
        terminal: bool = True,
    ) -> Optional[AwaitingPeerRecord]:
        """Called by the /hitl_resolved callback. Returns the awaiting record
        or None if unknown / already resumed.

        case-3 (HITL + async follow-up) delivers two callbacks per delegation:
          • terminal=False (phase="approval"): intermediate. Returns the record
            WITHOUT marking it resumed, so the later terminal callback can
            still find it.
          • terminal=True (phase="result"): final. Marks resumed so a duplicate
            terminal callback is ignored.
        case-1/2 send only the terminal callback (default terminal=True).
        """
        rec = self._awaiting.get((peer_agent, peer_interrupt_id))
        if rec is None:
            logger.warning(
                "CrossAgentHitlBridge: no awaiting record for peer=%s "
                "interrupt=%s (unknown or already cleaned up)",
                peer_agent, peer_interrupt_id[:12],
            )
            return None
        if rec.resumed:
            logger.warning(
                "CrossAgentHitlBridge: peer=%s interrupt=%s already resumed "
                "— ignoring duplicate callback",
                peer_agent, peer_interrupt_id[:12],
            )
            return None
        if terminal:
            rec.resumed = True
        return rec

    def forget_awaiting(self, *, peer_agent: str, peer_interrupt_id: str) -> None:
        self._awaiting.pop((peer_agent, peer_interrupt_id), None)


# Process-wide singleton (one agent per process). Persistence deferred to P3-d.
_BRIDGE: Optional[CrossAgentHitlBridge] = None


def get_cross_agent_hitl_bridge() -> CrossAgentHitlBridge:
    global _BRIDGE
    if _BRIDGE is None:
        _BRIDGE = CrossAgentHitlBridge()
    return _BRIDGE
