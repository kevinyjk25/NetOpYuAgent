"""Regression: the single delegation gate (task/delegation.py).

Identity of a delegated task = (session_id, target_agent). While a delegation
to a peer is in a NON-TERMINAL state (RUNNING, AWAITING_PEER_HITL, PENDING…),
the same originating request must NOT delegate to the same peer again — the
gate suppresses it (no new task, no dispatch). Terminal states (COMPLETED,
FAILED, CANCELLED) allow a fresh delegation.

This replaces the old env_ctx-scoped guards (count / pending-set / resume-
flag), which reset every execute_query and let duplicates through across the
resume synthesis turn. The gate reads TaskStore — durable across turns AND
streams, the same store the UI reads, so gate and UI never disagree.
"""
import asyncio
import unittest

from task.delegation import build_delegate_fn
from task.schemas import TaskDefinition, AgentAssignment, TaskState, TaskScope


class _FakeStore:
    def __init__(self, tasks=None):
        self._tasks = list(tasks or [])
        self.saved = []

    async def get_by_session(self, session_id):
        return [t for t in self._tasks if t.session_id == session_id]

    async def save(self, task):
        self.saved.append(task)
        self._tasks.append(task)

    async def write_audit(self, *a, **k):
        pass


class _FakeAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.base_url = "http://peer"
        self.is_available = True
        class _Card:
            skills = []
        self.card = _Card()


class _FakeRegistry:
    def __init__(self, agent_id="dc-agent"):
        self._agent = _FakeAgent(agent_id)
    async def get_agent(self, agent_id):
        return self._agent if agent_id == self._agent.agent_id else None
    def record_task_start(self, *a): pass
    def record_task_end(self, *a): pass


class _FakeDispatcher:
    def __init__(self):
        self.dispatched = []
    async def dispatch(self, task, assignment, store):
        self.dispatched.append(task)
        await store.save(task)
        yield {"token": "ok", "source_agent": assignment.agent_id}


class _Directive:
    def __init__(self, agent_id="dc-agent", task="diagnose"):
        self.by_capability = False
        self.agent_id = agent_id
        self.capability = None
        self.target = agent_id
        self.task = task
        self.forked = False


def _inflight_task(session, agent_id, state):
    return TaskDefinition(
        session_id=session, context_id=session, scope=TaskScope.INTER,
        description="prior", state=state,
        assignment=AgentAssignment(agent_id=agent_id, agent_url="http://peer",
                                   skill_id="s"),
    )


class TestDelegationGate(unittest.TestCase):
    def _run(self, store):
        reg = _FakeRegistry("dc-agent")
        disp = _FakeDispatcher()
        fn = build_delegate_fn(reg, disp, store, own_agent_id="lan-agent")
        async def _collect():
            out = []
            async for ch in fn(_Directive(), "sess-1", []):
                out.append(ch)
            return out
        return asyncio.run(_collect()), disp

    def test_suppresses_when_inflight_awaiting_peer_hitl(self):
        store = _FakeStore([_inflight_task("sess-1", "dc-agent",
                                           TaskState.AWAITING_PEER_HITL)])
        chunks, disp = self._run(store)
        # No dispatch happened.
        self.assertEqual(len(disp.dispatched), 0)
        # A suppression inject chunk was yielded.
        self.assertTrue(any(c.get("_delegation_suppressed") for c in chunks))

    def test_suppresses_when_inflight_running(self):
        store = _FakeStore([_inflight_task("sess-1", "dc-agent",
                                           TaskState.RUNNING)])
        _, disp = self._run(store)
        self.assertEqual(len(disp.dispatched), 0)

    def test_allows_when_prior_completed(self):
        store = _FakeStore([_inflight_task("sess-1", "dc-agent",
                                           TaskState.COMPLETED)])
        _, disp = self._run(store)
        # Terminal prior → fresh delegation allowed.
        self.assertEqual(len(disp.dispatched), 1)

    def test_allows_when_no_prior(self):
        store = _FakeStore([])
        _, disp = self._run(store)
        self.assertEqual(len(disp.dispatched), 1)

    def test_inflight_to_different_peer_does_not_block(self):
        store = _FakeStore([_inflight_task("sess-1", "other-agent",
                                           TaskState.AWAITING_PEER_HITL)])
        _, disp = self._run(store)
        # Different peer in flight → this delegation still allowed.
        self.assertEqual(len(disp.dispatched), 1)

    def test_created_task_is_scoped_inter(self):
        """The gate keys on scope==INTER, so the created task must carry it."""
        store = _FakeStore([])
        _, disp = self._run(store)
        self.assertEqual(disp.dispatched[0].scope, TaskScope.INTER)


if __name__ == "__main__":
    unittest.main(verbosity=2)
