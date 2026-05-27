"""
tests/test_inbound_delegation_completion.py
===========================================

Locks in the fix for the inbound-delegation-stuck-PENDING bug (2026-05).

When the dc agent serves a peer [DELEGATE:] request and the tool requires
HITL, execute() returns at the interrupt leaving the inbound TaskDefinition
parked in PENDING with metadata['awaiting_hitl_id'] = interrupt_id. The
operator approves LATER (separate event), so execute()'s completion path is
never re-reached. Without _complete_inbound_by_interrupt the task stays
PENDING forever — the delegating agent's view never closes, and (because the
delegation never completes) the originator keeps RE-DELEGATING ("请再次..."),
piling up duplicate inbound tasks.

These tests drive HitlExecutor._complete_inbound_by_interrupt directly with a
real in-memory TaskStore — no LLM, no httpx, no event queue.
"""
import asyncio
import unittest

import pytest
# task.schemas → pydantic; skip the whole module if pydantic isn't installed
# (e.g. a minimal CI runner) rather than erroring at collection time.
pytest.importorskip("pydantic")

from task.schemas import TaskDefinition, TaskState
from task.intra.store import TaskStore
from integrations.adapters.hitl_executor import HitlExecutor


def _executor(store):
    ex = HitlExecutor.__new__(HitlExecutor)   # bypass heavy __init__
    ex._task_store = store
    return ex


def _pending_inbound(task_id, interrupt_id):
    return TaskDefinition(
        task_id=task_id, session_id="s1", context_id="s1",
        description="grant alice crm", state=TaskState.PENDING,
        metadata={"direction": "inbound", "awaiting_hitl_id": interrupt_id},
    )


class TestInboundDelegationCompletion(unittest.TestCase):
    def test_approve_marks_inbound_completed(self):
        async def run():
            store = TaskStore()
            await store.save(_pending_inbound("t1", "iid-1"))
            ex = _executor(store)
            await ex._complete_inbound_by_interrupt(
                interrupt_id="iid-1", decision="approve",
                result_text="access granted",
            )
            t = await store.get("t1")
            self.assertEqual(t.state, TaskState.COMPLETED)
            self.assertEqual(t.result, {"text": "access granted"})
            # awaiting marker cleared so it can't be re-matched
            self.assertNotIn("awaiting_hitl_id", t.metadata)
            self.assertIsNotNone(t.completed_at)
        asyncio.run(run())

    def test_reject_marks_inbound_failed(self):
        async def run():
            store = TaskStore()
            await store.save(_pending_inbound("t2", "iid-2"))
            ex = _executor(store)
            await ex._complete_inbound_by_interrupt(
                interrupt_id="iid-2", decision="reject", result_text="",
            )
            t = await store.get("t2")
            self.assertEqual(t.state, TaskState.FAILED)
            self.assertTrue(t.error)
        asyncio.run(run())

    def test_unknown_interrupt_is_noop(self):
        # A resolution for an interrupt with no matching inbound task must not
        # crash and must not disturb other pending tasks.
        async def run():
            store = TaskStore()
            await store.save(_pending_inbound("t3", "iid-3"))
            ex = _executor(store)
            await ex._complete_inbound_by_interrupt(
                interrupt_id="does-not-exist", decision="approve",
                result_text="x",
            )
            t = await store.get("t3")
            self.assertEqual(t.state, TaskState.PENDING)  # untouched
        asyncio.run(run())

    def test_no_task_store_is_safe(self):
        async def run():
            ex = HitlExecutor.__new__(HitlExecutor)
            ex._task_store = None
            # must simply no-op, not raise
            await ex._complete_inbound_by_interrupt(
                interrupt_id="iid", decision="approve", result_text="x",
            )
        asyncio.run(run())

    def test_only_matching_task_completes(self):
        # Two pending inbound tasks; resolving one interrupt must complete
        # only its own task.
        async def run():
            store = TaskStore()
            await store.save(_pending_inbound("a", "iid-A"))
            await store.save(_pending_inbound("b", "iid-B"))
            ex = _executor(store)
            await ex._complete_inbound_by_interrupt(
                interrupt_id="iid-A", decision="approve", result_text="ok",
            )
            self.assertEqual((await store.get("a")).state, TaskState.COMPLETED)
            self.assertEqual((await store.get("b")).state, TaskState.PENDING)
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main(verbosity=2)
