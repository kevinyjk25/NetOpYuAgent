"""
a2a/push_notifications.py
-------------------------
Async push-notification sender.

When a client registers a ``webhook_url`` in the request metadata, the
server calls that webhook with a ``PushNotificationPayload`` whenever the
task state changes.

Usage::

    notifier = PushNotificationService()
    await notifier.notify(webhook_url, payload)

Retry policy: up to 3 attempts with exponential back-off (1 s → 2 s → 4 s).
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from .schemas import PushNotificationPayload

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # seconds


class PushNotificationService:
    """
    Sends push notifications to a client-supplied webhook URL.

    Each call to :meth:`notify` is fire-and-forget from the caller's
    perspective – errors are logged but not re-raised.
    """

    def __init__(self, timeout: float = 10.0) -> None:
        self._timeout = timeout

    async def notify(
        self,
        webhook_url: str,
        payload: PushNotificationPayload,
    ) -> None:
        """
        Send *payload* to *webhook_url* with retries.

        This method swallows all exceptions so it never disrupts the
        main task-execution flow.
        """
        asyncio.create_task(self._send_with_retry(webhook_url, payload))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _send_with_retry(
        self,
        webhook_url: str,
        payload: PushNotificationPayload,
    ) -> None:
        body = payload.model_dump_json()
        headers = {"Content-Type": "application/json"}

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(
                        webhook_url, content=body, headers=headers
                    )
                    response.raise_for_status()
                    logger.info(
                        "PushNotification sent to %s (attempt %d) status=%d",
                        webhook_url,
                        attempt,
                        response.status_code,
                    )
                    return  # success – stop retrying

            except Exception as exc:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "PushNotification attempt %d/%d failed for %s: %s. "
                    "Retrying in %.1f s…",
                    attempt,
                    MAX_RETRIES,
                    webhook_url,
                    exc,
                    delay,
                )
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(delay)

        logger.error(
            "PushNotification permanently failed for %s after %d attempts",
            webhook_url,
            MAX_RETRIES,
        )
