"""Behavioral delegation dispatcher tests (upgrades source-grep → behavior).

The outbound TaskDefinition state machine (RUNNING → COMPLETED / FAILED /
AWAITING_PEER_HITL) and the token accumulation were previously pinned only
by grepping coordinator.py source. This drives the REAL A2ATaskDispatcher.
dispatch() generator with the network seam (_stream_request) faked, asserting
the actual state transitions a delegating agent's Delegations tab depends on.
"""
import asyncio
import unittest

from task.inter.coordinator import A2ATaskDispatcher
from task.schemas import TaskDefinition, AgentAssignment, TaskState


class _FakeStore:
    """Minimal TaskStore: records saves + audits in memory."""
    def __init__(self):
        self.saved = []
        self.audits = []

    async def save(self, task):
        # snapshot the state at save time (task is mutated in place)
        self.saved.append(task.state)

    async def write_audit(self, record):
        self.audits.append(record)


class _ScriptedDispatcher(A2ATaskDispatcher):
    """Overrides the only network seam with a scripted peer stream."""
    def __init__(self, chunks=None, raise_exc=None):
        super().__init__()
        self._chunks = chunks or []
        self._raise = raise_exc

    async def _stream_request(self, agent_url, body):
        for ch in self._chunks:
            yield ch
        if self._raise:
            raise self._raise


def _task():
    return TaskDefinition(session_id="sess-1", context_id="sess-1",
                          description="diagnose alice access")


def _assignment():
    return AgentAssignment(agent_id="dc-agent",
                           agent_url="http://localhost:8001/api/v1/a2a",
                           skill_id="dc_app_access_diagnose")


async def _drive(dispatcher, task, store):
    """Consume the dispatch generator to completion, returning yielded chunks."""
    out = []
    try:
        async for ch in dispatcher.dispatch(task, _assignment(), store):
            out.append(ch)
    except Exception:
        pass  # dispatch re-raises on stream error; finally still runs
    return out


class TestDispatcherOutboundState(unittest.TestCase):
    def test_success_marks_completed_and_accumulates_result(self):
        async def run():
            disp = _ScriptedDispatcher(chunks=[
                {"token": "alice "}, {"token": "无 crm 权限"},
                {"type": "done"},
            ])
            task, store = _task(), _FakeStore()
            chunks = await _drive(disp, task, store)
            self.assertEqual(task.state, TaskState.COMPLETED)
            # peer tokens accumulated into the outbound result
            self.assertIn("alice", task.result["text"])
            self.assertIn("crm", task.result["text"])
            # state went RUNNING (first save) → COMPLETED (terminal save)
            self.assertEqual(store.saved[0], TaskState.RUNNING)
            self.assertEqual(store.saved[-1], TaskState.COMPLETED)
            # all peer chunks were forwarded to the caller
            self.assertTrue(any(c.get("token") == "alice " for c in chunks))
        asyncio.run(run())

    def test_peer_error_marks_failed(self):
        async def run():
            disp = _ScriptedDispatcher(
                chunks=[{"token": "starting"}],
                raise_exc=RuntimeError("peer connection reset"))
            task, store = _task(), _FakeStore()
            await _drive(disp, task, store)
            self.assertEqual(task.state, TaskState.FAILED)
            self.assertIn("peer connection reset", task.error)
            self.assertEqual(store.saved[-1], TaskState.FAILED)
        asyncio.run(run())

    def test_error_chunk_marks_failed(self):
        async def run():
            disp = _ScriptedDispatcher(chunks=[
                {"token": "ok"}, {"error": "peer LLM backend timeout"},
            ])
            task, store = _task(), _FakeStore()
            await _drive(disp, task, store)
            self.assertEqual(task.state, TaskState.FAILED)
            self.assertIn("timeout", task.error)
        asyncio.run(run())

    def test_peer_hitl_marks_awaiting_peer_hitl(self):
        async def run():
            disp = _ScriptedDispatcher(chunks=[
                {"token": "checking app acl"},
                {"type": "hitl_interrupt", "hitl_interrupt": True,
                 "interrupt_id": "int-xyz"},
            ])
            task, store = _task(), _FakeStore()
            await _drive(disp, task, store)
            # outbound task parks on the peer's operator, not terminal-completed
            self.assertEqual(task.state, TaskState.AWAITING_PEER_HITL)
            self.assertTrue(task.metadata.get("peer_hitl_pending"))
            self.assertEqual(task.metadata.get("peer_interrupt_id"), "int-xyz")
        asyncio.run(run())


class TestDispatcherNoDoubleCount(unittest.TestCase):
    def test_completed_result_is_accumulated_tokens_not_duplicated(self):
        """The result text equals the concatenation of streamed tokens — the
        dispatcher must not also append a separate 'final' copy (the
        double-count bug the source-grep test guarded against)."""
        async def run():
            disp = _ScriptedDispatcher(chunks=[
                {"token": "part-A "}, {"token": "part-B"},
            ])
            task, store = _task(), _FakeStore()
            await _drive(disp, task, store)
            self.assertEqual(task.result["text"], "part-A part-B")
            # exactly the two parts, no tripled/duplicated tail
            self.assertEqual(task.result["text"].count("part-A"), 1)
            self.assertEqual(task.result["text"].count("part-B"), 1)
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main(verbosity=2)
