"""tests/test_scheduler.py — in-memory periodic task scheduler (Phase 4)."""
import asyncio
import time
import unittest

from scheduler import SchedulerService, build_scheduler_tools, SCHEDULER_TOOL_METADATA
from scheduler.service import MIN_INTERVAL_S, MAX_JOBS, MAX_HISTORY


class TestSchedulerCore(unittest.TestCase):

    def test_tick_fires_due_tool_job(self):
        async def run():
            calls = []
            async def invoker(name, args):
                calls.append((name, args))
                return f"ran {name}"
            svc = SchedulerService(tool_invoker=invoker)
            svc.create_job(name="j", mode="tool",
                           payload={"tool_name": "get_user_access", "args": {"user_id": "alice"}},
                           interval_s=None, first_delay_s=0)
            fired = await svc.tick_once()
            self.assertEqual(fired, 1)
            self.assertEqual(calls, [("get_user_access", {"user_id": "alice"})])
            # one-shot → done, won't fire again
            self.assertEqual(await svc.tick_once(), 0)
        asyncio.run(run())

    def test_query_mode_distinct_sessions(self):
        async def run():
            sids = []
            async def runner(query, session_id):
                sids.append(session_id)
                return "ok"
            svc = SchedulerService(query_runner=runner)
            job = svc.create_job(name="q", mode="query",
                                 payload={"query": "check alice"},
                                 interval_s=MIN_INTERVAL_S, first_delay_s=0)
            await svc.tick_once()          # run 0
            job.next_run_at = time.time()  # force due again
            await svc.tick_once()          # run 1
            self.assertEqual(len(sids), 2)
            self.assertNotEqual(sids[0], sids[1])  # distinct per fire
        asyncio.run(run())

    def test_interval_reschedules_oneshot_done(self):
        async def run():
            async def invoker(name, args): return "x"
            svc = SchedulerService(tool_invoker=invoker)
            periodic = svc.create_job(name="p", mode="tool",
                                      payload={"tool_name": "t"}, interval_s=MIN_INTERVAL_S,
                                      first_delay_s=0)
            once = svc.create_job(name="o", mode="tool",
                                  payload={"tool_name": "t"}, interval_s=None, first_delay_s=0)
            await svc.tick_once()
            self.assertTrue(periodic.active)      # rescheduled
            self.assertFalse(once.active)         # done
            self.assertTrue(once.done)
            self.assertGreater(periodic.next_run_at, time.time())
        asyncio.run(run())

    def test_cancel(self):
        async def run():
            async def invoker(name, args): return "x"
            svc = SchedulerService(tool_invoker=invoker)
            job = svc.create_job(name="c", mode="tool", payload={"tool_name": "t"},
                                 interval_s=MIN_INTERVAL_S, first_delay_s=0)
            self.assertTrue(svc.cancel_job(job.job_id))
            self.assertFalse(svc.cancel_job(job.job_id))  # already cancelled
            self.assertEqual(await svc.tick_once(), 0)    # cancelled won't fire
        asyncio.run(run())

    def test_guardrails(self):
        async def run():
            svc = SchedulerService()
            with self.assertRaises(ValueError):  # interval too small
                svc.create_job(name="x", mode="tool", payload={"tool_name": "t"},
                               interval_s=1)
            with self.assertRaises(ValueError):  # bad mode
                svc.create_job(name="x", mode="bogus", payload={})
            with self.assertRaises(ValueError):  # tool mode needs tool_name
                svc.create_job(name="x", mode="tool", payload={})
            with self.assertRaises(ValueError):  # query mode needs query
                svc.create_job(name="x", mode="query", payload={})
        asyncio.run(run())

    def test_max_jobs(self):
        svc = SchedulerService()
        for i in range(MAX_JOBS):
            svc.create_job(name=f"j{i}", mode="tool", payload={"tool_name": "t"},
                           interval_s=MIN_INTERVAL_S)
        with self.assertRaises(ValueError):
            svc.create_job(name="overflow", mode="tool", payload={"tool_name": "t"},
                           interval_s=MIN_INTERVAL_S)

    def test_history_ring_buffer_and_error_recording(self):
        async def run():
            async def bad_invoker(name, args):
                raise RuntimeError("boom")
            svc = SchedulerService(tool_invoker=bad_invoker)
            job = svc.create_job(name="e", mode="tool", payload={"tool_name": "t"},
                                 interval_s=MIN_INTERVAL_S, first_delay_s=0)
            await svc.tick_once()
            hist = svc.history()
            self.assertEqual(len(hist), 1)
            self.assertFalse(hist[0]["ok"])
            self.assertIn("boom", hist[0]["result_preview"])
            # ring buffer cap
            for _ in range(MAX_HISTORY + 10):
                job.next_run_at = time.time()
                await svc.tick_once()
            self.assertLessEqual(len(svc._history), MAX_HISTORY)
        asyncio.run(run())


class TestSchedulerTools(unittest.TestCase):

    def test_tools_create_list_cancel(self):
        async def run():
            async def invoker(name, args): return "x"
            svc = SchedulerService(tool_invoker=invoker)
            tools = build_scheduler_tools(svc)
            self.assertEqual(set(tools), {"schedule_create", "schedule_list", "schedule_cancel"})

            out = await tools["schedule_create"]({
                "name": "ping", "mode": "tool", "interval_s": 30,
                "tool_name": "get_user_access", "tool_args": {"user_id": "alice"},
            })
            self.assertIn("Scheduled job created", out)

            listed = await tools["schedule_list"]({})
            self.assertIn("ping", listed)

            job_id = svc.list_jobs()[0]["job_id"]
            cancelled = await tools["schedule_cancel"]({"job_id": job_id})
            self.assertIn("Cancelled", cancelled)

        asyncio.run(run())

    def test_create_validation_returns_message_not_raise(self):
        async def run():
            svc = SchedulerService()
            tools = build_scheduler_tools(svc)
            out = await tools["schedule_create"]({"mode": "tool", "interval_s": 1})
            self.assertIn("❌", out)   # too-small interval → friendly error string
        asyncio.run(run())

    def test_metadata_shape(self):
        self.assertEqual(set(SCHEDULER_TOOL_METADATA),
                         {"schedule_create", "schedule_list", "schedule_cancel"})
        for meta in SCHEDULER_TOOL_METADATA.values():
            self.assertIn("description", meta)
            self.assertIn("parameters", meta)
            self.assertFalse(meta["hitl"])


if __name__ == "__main__":
    unittest.main()
