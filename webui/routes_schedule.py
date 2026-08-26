"""webui/routes_schedule.py — Scheduler (SCHEDULE tab) endpoints.

Exposes the in-memory SchedulerService for the WebUI SCHEDULE tab:
  GET  /webui/schedule           -> {jobs: [...], history: [...]}
  POST /webui/schedule/cancel    -> {ok, job_id}

The scheduler is in-memory (prototype scope); results are NOT pushed to the
user — this tab is the place to inspect jobs + recent run history.

Public API:
    register_schedule_routes(app, services)
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def register_schedule_routes(app: FastAPI, services: dict[str, Any]) -> None:
    """Attach /schedule + /schedule/cancel endpoints to `app`."""

    @app.get("/schedule")
    async def get_schedule() -> JSONResponse:
        sched = services.get("scheduler")
        if sched is None:
            return JSONResponse({"jobs": [], "history": [], "enabled": False})
        return JSONResponse({
            "enabled": True,
            "jobs": sched.list_jobs(include_inactive=True),
            "history": sched.history(limit=50),
        })

    @app.post("/schedule/cancel")
    async def cancel_schedule(request: Request) -> JSONResponse:
        sched = services.get("scheduler")
        if sched is None:
            return JSONResponse({"ok": False, "error": "scheduler not available"},
                                status_code=503)
        try:
            body = await request.json()
        except Exception:
            body = {}
        job_id = (body or {}).get("job_id", "")
        ok = sched.cancel_job(job_id) if job_id else False
        return JSONResponse({"ok": ok, "job_id": job_id})

    logger.info("WebUI: schedule routes registered (/schedule, /schedule/cancel)")
