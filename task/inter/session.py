"""
task/inter/session.py
---------------------
SessionManager — create, restore, and update sessions across turns.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from ..schemas import MultiRoundContext, SessionRecord

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages SessionRecord lifecycle.

    Storage: Redis (hot) + PostgreSQL (cold archive).
    Falls back to in-process dict when neither is available.
    """

    def __init__(
        self,
        redis_client: Any | None = None,
        pg_pool:      Any | None = None,
        ttl_seconds:  int = 3_600 * 8,   # 8 h idle timeout
    ) -> None:
        self._redis = redis_client
        self._pg    = pg_pool
        self._ttl   = ttl_seconds
        self._local: dict[str, SessionRecord] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_or_create(
        self,
        context_id: str,
        user_id: Optional[str] = None,
        memory_session_id: Optional[str] = None,
    ) -> SessionRecord:
        """
        Return existing session for context_id, or create a new one.
        Call this on every incoming A2A message.
        """
        session = await self._load(context_id)
        if session is None:
            session = SessionRecord(
                context_id=context_id,
                user_id=user_id,
                memory_session_id=memory_session_id or context_id,
            )
            logger.info("SessionManager: new session context_id=%s", context_id)
        else:
            logger.debug("SessionManager: restored session context_id=%s turn=%d",
                         context_id, session.turn_count)

        return session

    async def increment_turn(self, session: SessionRecord) -> SessionRecord:
        """Bump turn count and update last_active_at. Call after each response."""
        session.turn_count    += 1
        session.last_active_at = datetime.now(timezone.utc).isoformat()
        await self._save(session)
        return session

    async def add_active_task(self, session: SessionRecord, task_id: str) -> None:
        if task_id not in session.active_task_ids:
            session.active_task_ids.append(task_id)
        await self._save(session)

    async def remove_active_task(self, session: SessionRecord, task_id: str) -> None:
        session.active_task_ids = [t for t in session.active_task_ids if t != task_id]
        await self._save(session)

    async def update_multi_round(
        self,
        session: SessionRecord,
        context: MultiRoundContext,
    ) -> None:
        session.multi_round = context
        await self._save(session)

    async def close_session(self, session: SessionRecord) -> None:
        """Mark session inactive; archive to Postgres if available."""
        logger.info("SessionManager: closing session context_id=%s", session.context_id)
        if self._redis:
            await self._redis.delete(self._redis_key(session.context_id))
        if self._pg:
            await self._pg_archive(session)

    # ------------------------------------------------------------------
    # Storage helpers
    # ------------------------------------------------------------------

    def _redis_key(self, context_id: str) -> str:
        return f"session:{context_id}"

    async def _load(self, context_id: str) -> Optional[SessionRecord]:
        if self._redis:
            raw = await self._redis.get(self._redis_key(context_id))
            if raw:
                return SessionRecord.model_validate_json(raw)
        return self._local.get(context_id)

    async def _save(self, session: SessionRecord) -> None:
        if self._redis:
            await self._redis.set(
                self._redis_key(session.context_id),
                session.model_dump_json(),
                ex=self._ttl,
            )
        else:
            self._local[session.context_id] = session

    async def _pg_archive(self, session: SessionRecord) -> None:
        try:
            async with self._pg.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO sessions
                      (session_id, context_id, user_id, turn_count,
                       memory_session_id, created_at, last_active_at, metadata)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb)
                    ON CONFLICT (session_id) DO UPDATE
                      SET turn_count=EXCLUDED.turn_count,
                          last_active_at=EXCLUDED.last_active_at
                    """,
                    session.session_id, session.context_id,
                    session.user_id, session.turn_count,
                    session.memory_session_id,
                    datetime.fromisoformat(session.created_at),
                    datetime.fromisoformat(session.last_active_at),
                    json.dumps(session.metadata),
                )
        except Exception as exc:
            logger.error("SessionManager Postgres archive failed: %s", exc)
