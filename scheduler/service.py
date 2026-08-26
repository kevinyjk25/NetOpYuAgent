"""scheduler/service.py — in-memory periodic task scheduler.

A `SchedulerService` holds jobs in memory and runs a single background tick
loop. Each tick, any job whose `next_run_at <= now` fires, then reschedules
to `now + interval_s` (interval jobs) or is marked done (one-shot `at` jobs).

Two execution modes per job:
  - mode="tool":  call a local tool via the injected `tool_invoker`
                  (contract: async (tool_name, args_dict) -> str).
  - mode="query": send a query to this agent's own loop via the injected
                  `query_runner` (contract: async (query, session_id) -> str).

Run results are appended to a bounded per-service history (NOT pushed to the
user) — the SCHEDULE tab reads them via the service's `history()` accessor.

Design choices (mirroring SkillJournalConsumer):
  - Pure background task — never blocks the runtime stream loop.
  - All execution errors logged + recorded in history, never raised.
  - In-memory only — jobs and history are lost on restart (prototype scope).
  - L0/L1 decoupled — no import of runtime/ or webui/; behaviour comes from
    the two injected callables.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Optional

logger = logging.getLogger(__name__)

ToolInvoker = Callable[[str, dict], Awaitable[str]]
QueryRunner = Callable[[str, str], Awaitable[str]]

JobMode = Literal["tool", "query"]

# Guardrails (prototype-scale, prevent runaway scheduling).
MIN_INTERVAL_S = 5
MAX_JOBS = 100
MAX_HISTORY = 200


@dataclass
class JobRun:
    """One execution of a job."""
    job_id: str
    started_at: float
    ok: bool
    result_preview: str          # first ~500 chars of the result/error
    duration_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "started_at": self.started_at,
            "ok": self.ok,
            "result_preview": self.result_preview,
            "duration_s": round(self.duration_s, 3),
        }


@dataclass
class ScheduledJob:
    """A periodic or one-shot scheduled task (in-memory)."""
    job_id: str
    name: str
    mode: JobMode
    # mode="tool": payload = {"tool_name": str, "args": dict}
    # mode="query": payload = {"query": str}
    payload: dict[str, Any]
    interval_s: Optional[int]    # periodic if set; one-shot if None
    next_run_at: float
    created_at: float = field(default_factory=time.time)
    runs: int = 0
    last_run_at: Optional[float] = None
    last_ok: Optional[bool] = None
    cancelled: bool = False
    done: bool = False           # one-shot job that already fired

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "mode": self.mode,
            "payload": self.payload,
            "interval_s": self.interval_s,
            "next_run_at": self.next_run_at,
            "created_at": self.created_at,
            "runs": self.runs,
            "last_run_at": self.last_run_at,
            "last_ok": self.last_ok,
            "cancelled": self.cancelled,
            "done": self.done,
            "state": self._state_label(),
        }

    def _state_label(self) -> str:
        if self.cancelled:
            return "cancelled"
        if self.done:
            return "done"
        return "active"

    @property
    def active(self) -> bool:
        return not (self.cancelled or self.done)


class SchedulerService:
    """In-memory periodic task scheduler with a single background tick loop."""

    def __init__(
        self,
        *,
        tool_invoker: Optional[ToolInvoker] = None,
        query_runner: Optional[QueryRunner] = None,
        tick_interval_s: float = 2.0,
    ) -> None:
        self._jobs: dict[str, ScheduledJob] = {}
        self._history: list[JobRun] = []
        self._tool_invoker = tool_invoker
        self._query_runner = query_runner
        self._tick = tick_interval_s
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    # ── Injection (wired after construction, like delegate_fn) ──────────
    def set_tool_invoker(self, fn: ToolInvoker) -> None:
        self._tool_invoker = fn

    def set_query_runner(self, fn: QueryRunner) -> None:
        self._query_runner = fn

    # ── Lifecycle ───────────────────────────────────────────────────────
    async def start(self) -> None:
        if self._task is not None:
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop(), name="scheduler_tick")
        logger.info("SchedulerService: tick loop started (interval=%.1fs)", self._tick)

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
            self._task = None

    # ── Job management (used by the agent tools + UI) ───────────────────
    def create_job(
        self,
        *,
        name: str,
        mode: JobMode,
        payload: dict[str, Any],
        interval_s: Optional[int] = None,
        first_delay_s: Optional[int] = None,
    ) -> ScheduledJob:
        active = [j for j in self._jobs.values() if j.active]
        if len(active) >= MAX_JOBS:
            raise ValueError(f"too many active jobs (max {MAX_JOBS})")
        if mode not in ("tool", "query"):
            raise ValueError(f"mode must be 'tool' or 'query', got {mode!r}")
        if interval_s is not None and interval_s < MIN_INTERVAL_S:
            raise ValueError(f"interval_s must be >= {MIN_INTERVAL_S}")
        if mode == "tool" and not payload.get("tool_name"):
            raise ValueError("tool mode requires payload.tool_name")
        if mode == "query" and not payload.get("query"):
            raise ValueError("query mode requires payload.query")

        now = time.time()
        delay = first_delay_s if first_delay_s is not None else (interval_s or 0)
        job = ScheduledJob(
            job_id=uuid.uuid4().hex[:12],
            name=name or f"{mode}-job",
            mode=mode,
            payload=dict(payload),
            interval_s=interval_s,
            next_run_at=now + max(0, delay),
        )
        self._jobs[job.job_id] = job
        logger.info(
            "SchedulerService: created job %s name=%r mode=%s interval=%s",
            job.job_id, job.name, job.mode, job.interval_s,
        )
        return job

    def cancel_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None or not job.active:
            return False
        job.cancelled = True
        logger.info("SchedulerService: cancelled job %s", job_id)
        return True

    def list_jobs(self, *, include_inactive: bool = True) -> list[dict[str, Any]]:
        jobs = self._jobs.values()
        if not include_inactive:
            jobs = [j for j in jobs if j.active]
        return [j.to_dict() for j in sorted(jobs, key=lambda j: j.created_at)]

    def history(self, *, limit: int = 50) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self._history[-limit:][::-1]]

    # ── Tick loop ───────────────────────────────────────────────────────
    async def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self.tick_once()
            except Exception as exc:
                logger.warning("SchedulerService tick failed: %s", exc)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._tick)
                return  # stopped
            except asyncio.TimeoutError:
                pass

    async def tick_once(self) -> int:
        """Fire all due jobs. Returns the number fired. Public for tests."""
        now = time.time()
        due = [
            j for j in self._jobs.values()
            if j.active and j.next_run_at <= now
        ]
        for job in due:
            await self._fire(job)
            # Reschedule or finalize.
            if job.interval_s is not None:
                job.next_run_at = time.time() + job.interval_s
            else:
                job.done = True   # one-shot
        return len(due)

    async def _fire(self, job: ScheduledJob) -> None:
        start = time.time()
        ok = False
        preview = ""
        try:
            if job.mode == "tool":
                if self._tool_invoker is None:
                    raise RuntimeError("tool_invoker not wired")
                result = await self._tool_invoker(
                    job.payload["tool_name"], job.payload.get("args", {})
                )
            else:  # query
                if self._query_runner is None:
                    raise RuntimeError("query_runner not wired")
                # Each fire uses a distinct session so periodic queries don't
                # accumulate into one ever-growing conversation.
                sid = f"sched-{job.job_id}-{job.runs}"
                result = await self._query_runner(job.payload["query"], sid)
            preview = str(result)[:500]
            ok = True
        except Exception as exc:
            preview = f"ERROR: {exc}"
            logger.warning("SchedulerService: job %s fire failed: %s", job.job_id, exc)
        finally:
            job.runs += 1
            job.last_run_at = start
            job.last_ok = ok
            self._history.append(JobRun(
                job_id=job.job_id, started_at=start, ok=ok,
                result_preview=preview, duration_s=time.time() - start,
            ))
            if len(self._history) > MAX_HISTORY:
                self._history = self._history[-MAX_HISTORY:]


# ── Tool metadata (for the retriever corpus, so the LLM can discover these) ──
# Same shape as ToolLoader.build_metadata() entries:
#   {tool_name: {description, parameters, returns, hitl, tags}}
SCHEDULER_TOOL_METADATA: dict[str, dict[str, Any]] = {
    "schedule_create": {
        "description": (
            "Create a periodic or one-shot scheduled task. mode='tool' runs a "
            "local tool on a schedule; mode='query' sends a query to this agent "
            "on a schedule. Set interval_s for a repeating job, or first_delay_s "
            "with no interval_s for a one-shot. Results are recorded in the "
            "scheduler history (SCHEDULE tab), not pushed to the user."
        ),
        "parameters": {
            "name": "str — human label for the job",
            "mode": "str — 'tool' or 'query'",
            "interval_s": "int — repeat every N seconds (omit for one-shot; min 5)",
            "first_delay_s": "int — delay before first/only run (seconds)",
            "tool_name": "str — (mode=tool) the local tool to invoke",
            "tool_args": "dict — (mode=tool) arguments for that tool",
            "query": "str — (mode=query) the query to send to this agent",
        },
        "returns": "Confirmation with the new job id and schedule summary.",
        "hitl": False,
        "tags": ["scheduler", "periodic", "cron", "timer", "automation", "定时", "周期"],
    },
    "schedule_list": {
        "description": "List all scheduled tasks (active, done, cancelled) with "
                       "their mode, schedule, run count, and target.",
        "parameters": {"include_inactive": "bool — include done/cancelled jobs"},
        "returns": "A list of scheduled jobs.",
        "hitl": False,
        "tags": ["scheduler", "list", "定时", "周期", "任务"],
    },
    "schedule_cancel": {
        "description": "Cancel an active scheduled task by its job id.",
        "parameters": {"job_id": "str — the id of the job to cancel"},
        "returns": "Confirmation of cancellation.",
        "hitl": False,
        "tags": ["scheduler", "cancel", "stop", "定时", "取消"],
    },
}


# ── Agent tools (registered via ToolRouter.register_local) ──────────────
def build_scheduler_tools(service: SchedulerService) -> dict[str, Callable]:
    """Return {tool_name: async callable(args_dict) -> str} for registration.

    Mirrors the local-tool contract used by ToolRouter / runtime loop:
    every callable takes a single args dict and returns a string.
    """

    async def schedule_create(args: dict) -> str:
        name = args.get("name") or ""
        mode = args.get("mode") or "tool"
        interval_s = args.get("interval_s")
        first_delay_s = args.get("first_delay_s")
        if interval_s is not None:
            try:
                interval_s = int(interval_s)
            except (TypeError, ValueError):
                return "❌ interval_s must be an integer (seconds)."
        if first_delay_s is not None:
            try:
                first_delay_s = int(first_delay_s)
            except (TypeError, ValueError):
                return "❌ first_delay_s must be an integer (seconds)."

        if mode == "tool":
            payload = {
                "tool_name": args.get("tool_name") or "",
                "args": args.get("tool_args") or {},
            }
        else:
            payload = {"query": args.get("query") or ""}

        try:
            job = service.create_job(
                name=name, mode=mode, payload=payload,
                interval_s=interval_s, first_delay_s=first_delay_s,
            )
        except ValueError as exc:
            return f"❌ Could not create scheduled job: {exc}"

        kind = (f"every {interval_s}s" if interval_s
                else f"once after {first_delay_s or 0}s")
        target = (payload["tool_name"] if mode == "tool"
                  else f"query: {payload['query'][:60]}")
        return (
            f"✅ Scheduled job created.\n"
            f"  id       : {job.job_id}\n"
            f"  name     : {job.name}\n"
            f"  mode     : {mode}\n"
            f"  schedule : {kind}\n"
            f"  target   : {target}"
        )

    async def schedule_list(args: dict) -> str:
        jobs = service.list_jobs(include_inactive=bool(args.get("include_inactive")))
        if not jobs:
            return "No scheduled jobs."
        lines = ["Scheduled jobs:"]
        for j in jobs:
            tgt = (j["payload"].get("tool_name") if j["mode"] == "tool"
                   else (j["payload"].get("query", "")[:50]))
            sched = f"every {j['interval_s']}s" if j["interval_s"] else "one-shot"
            lines.append(
                f"  [{j['state']}] {j['job_id']} · {j['name']} · {j['mode']} "
                f"· {sched} · runs={j['runs']} · → {tgt}"
            )
        return "\n".join(lines)

    async def schedule_cancel(args: dict) -> str:
        job_id = args.get("job_id") or ""
        if not job_id:
            return "❌ schedule_cancel requires a job_id."
        ok = service.cancel_job(job_id)
        return (f"✅ Cancelled job {job_id}." if ok
                else f"❌ No active job with id {job_id}.")

    return {
        "schedule_create": schedule_create,
        "schedule_list": schedule_list,
        "schedule_cancel": schedule_cancel,
    }
