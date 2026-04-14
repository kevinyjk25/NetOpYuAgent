"""
hitl/review.py
--------------
Layer 3 — Review delivery layer.

HitlReviewService fans out a HitlPayload to all configured channels
simultaneously. Each channel is an independent async task so a slow
webhook never blocks the others.

Supported channels
------------------
  A2APushNotificationChannel  – re-uses the A2A push_notifications module
  SlackWebhookChannel         – Slack Incoming Webhook (Block Kit message)
  PagerDutyChannel            – PagerDuty Events API v2
  WebDashboardSSEChannel      – Server-Sent Events for the internal dashboard

Adding a new channel
--------------------
  1. Subclass NotificationChannel
  2. Implement async send(payload) → None
  3. Register in HitlReviewService.__init__

Usage
-----
    service = HitlReviewService.from_env()
    await service.notify(hitl_payload)
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from .schemas import HitlPayload, RiskLevel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Channel config
# ---------------------------------------------------------------------------

@dataclass
class ReviewChannelConfig:
    # A2A push
    a2a_webhook_url: Optional[str] = None

    # Slack
    slack_webhook_url: Optional[str] = None

    # PagerDuty
    pagerduty_routing_key: Optional[str] = None
    pagerduty_events_url: str = "https://events.pagerduty.com/v2/enqueue"

    # Web dashboard SSE  (internal; uses an asyncio.Queue broadcast)
    enable_sse: bool = True

    # WebSocket channel (for external agent systems)
    enable_websocket: bool = True

    # HTTP timeout for outbound webhooks
    http_timeout: float = 10.0

    # Retry attempts per channel
    max_retries: int = 3


# ---------------------------------------------------------------------------
# Abstract channel base
# ---------------------------------------------------------------------------

class NotificationChannel(ABC):
    """Base class for all HITL notification channels."""

    name: str = "base"

    @abstractmethod
    async def send(self, payload: HitlPayload) -> None:
        """Send the HITL payload. Raise on permanent failure."""


# ---------------------------------------------------------------------------
# Channel 1: A2A Push Notification
# ---------------------------------------------------------------------------

class A2APushNotificationChannel(NotificationChannel):
    """
    Sends to the operator's registered A2A webhook URL.
    Reuses the retry logic from a2a/push_notifications.py.
    """
    name = "a2a_push"

    def __init__(self, webhook_url: str, timeout: float = 10.0) -> None:
        self._url = webhook_url
        self._timeout = timeout

    async def send(self, payload: HitlPayload) -> None:
        body = {
            "event": "hitl_interrupt",
            "interrupt_id": payload.interrupt_id,
            "thread_id": payload.thread_id,
            "trigger": payload.trigger_kind.value,
            "risk_level": payload.risk_level.value,
            "summary": payload.intent_summary,
            "proposed_action": payload.proposed_action.model_dump(),
            "sla_seconds": payload.sla_seconds,
            "created_at": payload.created_at,
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                self._url,
                json=body,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
        logger.info("A2APushNotificationChannel: sent interrupt_id=%s", payload.interrupt_id)


# ---------------------------------------------------------------------------
# Channel 2: Slack Incoming Webhook (Block Kit)
# ---------------------------------------------------------------------------

_RISK_EMOJI = {
    RiskLevel.LOW:      ":large_blue_circle:",
    RiskLevel.MEDIUM:   ":large_yellow_circle:",
    RiskLevel.HIGH:     ":orange_circle:",
    RiskLevel.CRITICAL: ":red_circle:",
}


class SlackWebhookChannel(NotificationChannel):
    """Posts a structured Block Kit message to a Slack channel."""
    name = "slack"

    def __init__(self, webhook_url: str, timeout: float = 10.0) -> None:
        self._url = webhook_url
        self._timeout = timeout

    def _build_blocks(self, payload: HitlPayload) -> list[dict]:
        emoji  = _RISK_EMOJI.get(payload.risk_level, ":white_circle:")
        action = payload.proposed_action

        return [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} IT Ops HITL — Human review required",
                },
            },
            {"type": "divider"},
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Trigger:*\n{payload.trigger_kind.value}"},
                    {"type": "mrkdwn", "text": f"*Risk:*\n{payload.risk_level.value.upper()}"},
                    {"type": "mrkdwn", "text": f"*Action:*\n`{action.action_type}`"},
                    {"type": "mrkdwn", "text": f"*Target:*\n{action.target}"},
                    {"type": "mrkdwn", "text": f"*Confidence:*\n{payload.confidence_score:.0%}"},
                    {"type": "mrkdwn", "text": f"*SLA:*\n{payload.sla_seconds // 60} min"},
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Summary:* {payload.intent_summary}",
                },
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Review"},
                        "style": "primary",
                        "url": f"http://localhost:3000/hitl/review/{payload.interrupt_id}",
                    },
                ],
            },
        ]

    async def send(self, payload: HitlPayload) -> None:
        body = {"blocks": self._build_blocks(payload)}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                self._url,
                json=body,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
        logger.info("SlackWebhookChannel: sent interrupt_id=%s", payload.interrupt_id)


# ---------------------------------------------------------------------------
# Channel 3: PagerDuty Events API v2
# ---------------------------------------------------------------------------

_RISK_TO_PD_SEVERITY = {
    RiskLevel.LOW:      "info",
    RiskLevel.MEDIUM:   "warning",
    RiskLevel.HIGH:     "error",
    RiskLevel.CRITICAL: "critical",
}


class PagerDutyChannel(NotificationChannel):
    """Triggers a PagerDuty incident via Events API v2."""
    name = "pagerduty"

    def __init__(
        self,
        routing_key: str,
        events_url: str = "https://events.pagerduty.com/v2/enqueue",
        timeout: float = 10.0,
    ) -> None:
        self._routing_key = routing_key
        self._events_url  = events_url
        self._timeout     = timeout

    async def send(self, payload: HitlPayload) -> None:
        action = payload.proposed_action
        severity = _RISK_TO_PD_SEVERITY.get(payload.risk_level, "error")

        body = {
            "routing_key": self._routing_key,
            "event_action": "trigger",
            "dedup_key": payload.interrupt_id,
            "payload": {
                "summary": (
                    f"[HITL] {payload.trigger_kind.value} — "
                    f"{action.action_type} on {action.target}"
                ),
                "severity": severity,
                "source": "ITOpsAgent",
                "custom_details": {
                    "interrupt_id":   payload.interrupt_id,
                    "thread_id":      payload.thread_id,
                    "confidence":     payload.confidence_score,
                    "sla_seconds":    payload.sla_seconds,
                    "intent_summary": payload.intent_summary,
                    "action_params":  action.parameters,
                },
            },
            "links": [
                {
                    "href": f"http://localhost:3000/hitl/review/{payload.interrupt_id}",
                    "text": "Review in dashboard",
                }
            ],
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                self._events_url,
                json=body,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
        logger.info("PagerDutyChannel: sent interrupt_id=%s", payload.interrupt_id)


# ---------------------------------------------------------------------------
# Channel 4: Web Dashboard SSE (in-process broadcast)
# ---------------------------------------------------------------------------

class WebDashboardSSEChannel(NotificationChannel):
    """
    Broadcasts HITL events to connected web dashboard clients via SSE.

    The FastAPI SSE endpoint (hitl/router.py) subscribes to this channel.
    Multiple browser tabs connect to the same broadcast queue.
    """
    name = "sse"

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue] = []

    def subscribe(self) -> asyncio.Queue:
        """Called by each new SSE client connection."""
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Called when an SSE client disconnects."""
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    async def send(self, payload: HitlPayload) -> None:
        event = json.dumps({
            "event": "hitl_interrupt",
            "data": payload.model_dump(),
        })
        dead: list[asyncio.Queue] = []
        for q in self._subscribers:
            try:
                await asyncio.wait_for(q.put(event), timeout=1.0)
            except asyncio.TimeoutError:
                dead.append(q)
        for q in dead:
            self.unsubscribe(q)
        logger.debug(
            "WebDashboardSSEChannel: broadcast to %d clients interrupt_id=%s",
            len(self._subscribers),
            payload.interrupt_id,
        )


# ---------------------------------------------------------------------------
# Channel 5: WebSocket (for external agent systems)
# ---------------------------------------------------------------------------

class WebSocketHitlManager:
    """
    Manages all active WebSocket connections to /hitl/ws.

    Responsibilities
    ----------------
    - connect() / disconnect()  lifecycle management
    - broadcast_interrupt()     push HitlPayload JSON to every connected client
    - Each connected WebSocket can also SEND a HitlDecision back over the
      same connection, handled by the router endpoint.

    Connection message format (server → client)
    --------------------------------------------
    {
        "type": "hitl_interrupt",
        "interrupt_id": "...",
        "thread_id": "...",
        "trigger_kind": "...",
        "risk_level": "...",
        "intent_summary": "...",
        "proposed_action": {...},
        "sla_seconds": 600,
        "review_url": "..."
    }

    Decision message format (client → server)
    ------------------------------------------
    {
        "type": "hitl_decision",
        "interrupt_id": "...",
        "thread_id": "...",
        "decision": "approve|reject|edit|escalate",
        "operator_id": "external-agent-system",
        "comment": "...",
        "parameter_patch": {...}   // for 'edit' only
    }

    Keepalive (server → client, every 30 s)
    -----------------------------------------
    {"type": "ping"}

    Client acknowledgement (optional, client → server)
    ----------------------------------------------------
    {"type": "pong"}
    """

    def __init__(self) -> None:
        # Use a set of (WebSocket, client_id) tuples
        self._connections: dict[str, Any] = {}   # client_id → WebSocket
        self._lock = asyncio.Lock()

    async def connect(self, websocket: Any, client_id: str) -> None:
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self._connections[client_id] = websocket
        logger.info(
            "WebSocketHitlManager: client connected client_id=%s total=%d",
            client_id, len(self._connections),
        )

    async def disconnect(self, client_id: str) -> None:
        """Remove a WebSocket connection."""
        async with self._lock:
            self._connections.pop(client_id, None)
        logger.info(
            "WebSocketHitlManager: client disconnected client_id=%s total=%d",
            client_id, len(self._connections),
        )

    async def broadcast(self, message: dict) -> None:
        """
        Send a JSON message to all connected clients.
        Dead connections are cleaned up silently.
        """
        payload_str = json.dumps(message)
        dead: list[str] = []

        async with self._lock:
            clients = dict(self._connections)

        for client_id, ws in clients.items():
            try:
                await ws.send_text(payload_str)
            except Exception as exc:
                logger.warning(
                    "WebSocketHitlManager: send failed client_id=%s: %s",
                    client_id, exc,
                )
                dead.append(client_id)

        for cid in dead:
            await self.disconnect(cid)

    async def send_to(self, client_id: str, message: dict) -> bool:
        """Send a message to one specific client. Returns False if not found."""
        ws = self._connections.get(client_id)
        if ws is None:
            return False
        try:
            await ws.send_text(json.dumps(message))
            return True
        except Exception as exc:
            logger.warning(
                "WebSocketHitlManager: targeted send failed client_id=%s: %s",
                client_id, exc,
            )
            await self.disconnect(client_id)
            return False

    @property
    def connection_count(self) -> int:
        return len(self._connections)


class WebSocketHitlChannel(NotificationChannel):
    """
    NotificationChannel implementation that broadcasts HitlPayload events
    to all connected WebSocket clients via WebSocketHitlManager.

    Used by HitlReviewService alongside SSE, Slack, and PagerDuty.
    """
    name = "websocket"

    def __init__(self, manager: "WebSocketHitlManager") -> None:
        self._manager = manager

    async def send(self, payload: HitlPayload) -> None:
        message = {
            "type":           "hitl_interrupt",
            "interrupt_id":   payload.interrupt_id,
            "thread_id":      payload.thread_id,
            "context_id":     payload.context_id,
            "task_id":        payload.task_id,
            "trigger_kind":   payload.trigger_kind.value,
            "risk_level":     payload.risk_level.value,
            "intent_summary": payload.intent_summary,
            "confidence_score": payload.confidence_score,
            "proposed_action": payload.proposed_action.model_dump(),
            "sla_seconds":    payload.sla_seconds,
            "created_at":     payload.created_at,
        }
        await self._manager.broadcast(message)
        logger.debug(
            "WebSocketHitlChannel: broadcast interrupt_id=%s to %d client(s)",
            payload.interrupt_id, self._manager.connection_count,
        )


# ---------------------------------------------------------------------------
# Process-global singletons (instantiated after all classes are defined)
# ---------------------------------------------------------------------------

_GLOBAL_SSE_CHANNEL = WebDashboardSSEChannel()
_GLOBAL_WS_MANAGER  = WebSocketHitlManager()
_GLOBAL_WS_CHANNEL  = WebSocketHitlChannel(_GLOBAL_WS_MANAGER)


class HitlReviewService:
    """
    Fans out a HitlPayload to all enabled channels concurrently.

    Errors in individual channels are logged and swallowed so one broken
    integration never prevents the others from notifying.
    """

    def __init__(
        self,
        channels: list[NotificationChannel],
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
    ) -> None:
        self._channels         = channels
        self._max_retries      = max_retries
        self._retry_base_delay = retry_base_delay

    @classmethod
    def from_config(cls, cfg: ReviewChannelConfig) -> "HitlReviewService":
        """Build a HitlReviewService from a ReviewChannelConfig."""
        channels: list[NotificationChannel] = []

        if cfg.a2a_webhook_url:
            channels.append(
                A2APushNotificationChannel(cfg.a2a_webhook_url, cfg.http_timeout)
            )
        if cfg.slack_webhook_url:
            channels.append(
                SlackWebhookChannel(cfg.slack_webhook_url, cfg.http_timeout)
            )
        if cfg.pagerduty_routing_key:
            channels.append(
                PagerDutyChannel(
                    cfg.pagerduty_routing_key,
                    cfg.pagerduty_events_url,
                    cfg.http_timeout,
                )
            )
        if cfg.enable_sse:
            channels.append(_GLOBAL_SSE_CHANNEL)
        if cfg.enable_websocket:
            channels.append(_GLOBAL_WS_CHANNEL)

        return cls(channels, max_retries=cfg.max_retries)

    @classmethod
    def from_env(cls) -> "HitlReviewService":
        """Build from environment variables (convenience for production)."""
        import os
        return cls.from_config(
            ReviewChannelConfig(
                a2a_webhook_url=os.getenv("HITL_A2A_WEBHOOK_URL"),
                slack_webhook_url=os.getenv("HITL_SLACK_WEBHOOK_URL"),
                pagerduty_routing_key=os.getenv("HITL_PAGERDUTY_ROUTING_KEY"),
                enable_sse=True,
            )
        )

    async def notify(self, payload: HitlPayload) -> None:
        """Fan out to all channels concurrently. Fire-and-forget per channel."""
        tasks = [
            asyncio.create_task(self._send_with_retry(ch, payload))
            for ch in self._channels
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_with_retry(
        self, channel: NotificationChannel, payload: HitlPayload
    ) -> None:
        for attempt in range(1, self._max_retries + 1):
            try:
                await channel.send(payload)
                return
            except Exception as exc:
                delay = self._retry_base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Channel %s attempt %d/%d failed for interrupt_id=%s: %s. "
                    "Retrying in %.1fs…",
                    channel.name, attempt, self._max_retries,
                    payload.interrupt_id, exc, delay,
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(delay)
        logger.error(
            "Channel %s permanently failed for interrupt_id=%s",
            channel.name, payload.interrupt_id,
        )


def get_sse_channel() -> WebDashboardSSEChannel:
    """Return the process-global SSE channel (injected into FastAPI routes)."""
    return _GLOBAL_SSE_CHANNEL


def get_ws_manager() -> WebSocketHitlManager:
    """Return the process-global WebSocket manager (injected into FastAPI routes)."""
    return _GLOBAL_WS_MANAGER