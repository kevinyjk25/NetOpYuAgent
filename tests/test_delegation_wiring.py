"""
tests/test_delegation_wiring.py — Phase 2B delegate_fn integration tests
========================================================================

Tests build_delegate_fn() against mock registry + dispatcher (no httpx, runs
in sandbox + CI). Covers:
  - explicit agent_id resolution → dispatch streams chunks
  - *capability resolution via registry.resolve (excludes self)
  - forked vs fresh: shared facts passed only when forked
  - graceful degradation: unknown agent / unhealthy / no capability match
  - record_task_start/end bracketing
"""
import asyncio
import unittest
from types import SimpleNamespace

# task.delegation → task package __init__ → task.schemas → pydantic. CI's
# lightweight safety-tests job has no pydantic, so guard the import and skip
# the module cleanly instead of failing pytest collection. (The wiring logic
# itself is pure-python; only the import chain needs pydantic.)
try:
    from task.delegation import build_delegate_fn
    from runtime.directive_parser import find_delegate_directives
except ImportError as _exc:  # pragma: no cover - env without pydantic
    raise unittest.SkipTest(f"task.delegation unavailable: {_exc}")


def _directive(text):
    return find_delegate_directives(text)[0]


class _MockSkill:
    def __init__(self, sid):
        self.id = sid


class _MockAgent:
    def __init__(self, agent_id, url, skills, available=True):
        self.agent_id = agent_id
        self.base_url = url
        self.card = SimpleNamespace(name=agent_id, skills=[_MockSkill(s) for s in skills])
        self._available = available

    @property
    def is_available(self):
        return self._available


class _MockRegistry:
    def __init__(self, agents):
        self._agents = {a.agent_id: a for a in agents}
        self.task_starts = []
        self.task_ends = []

    async def get_agent(self, agent_id):
        return self._agents.get(agent_id)

    async def resolve(self, capability, exclude_agent_ids=None):
        excl = set(exclude_agent_ids or [])
        for a in self._agents.values():
            if a.agent_id in excl:
                continue
            if any(s.id == capability for s in a.card.skills) and a.is_available:
                return SimpleNamespace(
                    agent_id=a.agent_id, agent_url=a.base_url,
                    skill_id=capability, skill=None,
                    agent_name=a.agent_id, health="healthy",
                )
        return None

    def record_task_start(self, agent_id):
        self.task_starts.append(agent_id)

    def record_task_end(self, agent_id):
        self.task_ends.append(agent_id)


class _MockDispatcher:
    """Captures the task it was handed + streams canned chunks."""
    def __init__(self, chunks=None, raise_exc=None):
        self._chunks = chunks or [{"token": "spine-1 BGP up"}, {"token": " 3 neighbors"}]
        self._raise = raise_exc
        self.last_task = None
        self.last_assignment = None

    async def dispatch(self, task, assignment, store):
        self.last_task = task
        self.last_assignment = assignment
        if self._raise:
            raise self._raise
        for c in self._chunks:
            yield c


class _MockStore:
    pass


async def _drain(agen):
    return [c async for c in agen]


class TestDelegateExplicit(unittest.TestCase):
    def test_explicit_agent_dispatch(self):
        async def run():
            reg = _MockRegistry([_MockAgent("dc-agent", "http://dc/api/v1/a2a",
                                            ["dc_fabric_diagnose"])])
            disp = _MockDispatcher()
            fn = build_delegate_fn(reg, disp, _MockStore(), own_agent_id="lan-agent")
            d = _directive("[DELEGATE:dc-agent] check BGP on spine-1")
            chunks = await _drain(fn(d, "sess-1", []))
            # all chunks present (source_agent tagging is the runtime loop's
            # job in _handle_delegate, not delegate_fn's — delegate_fn yields
            # raw peer chunks).
            toks = [c.get("token") for c in chunks if c.get("token")]
            self.assertIn("spine-1 BGP up", toks)
            # dispatched with the right task description
            self.assertEqual(disp.last_task.description, "check BGP on spine-1")
            self.assertEqual(disp.last_assignment.agent_id, "dc-agent")
            # task-load bracketed
            self.assertEqual(reg.task_starts, ["dc-agent"])
            self.assertEqual(reg.task_ends, ["dc-agent"])
        asyncio.run(run())

    def test_forked_passes_facts(self):
        async def run():
            reg = _MockRegistry([_MockAgent("dc-agent", "http://dc", ["dc_fabric_diagnose"])])
            disp = _MockDispatcher()
            fn = build_delegate_fn(reg, disp, _MockStore(), own_agent_id="lan-agent")
            d = _directive("[DELEGATE:dc-agent#forked] correlate")
            await _drain(fn(d, "sess-1", ["fact-A", "fact-B"]))
            self.assertEqual(
                disp.last_task.parameters.get("parent_confirmed_facts"),
                ["fact-A", "fact-B"])
            self.assertEqual(disp.last_task.metadata["shared_facts_count"], 2)
        asyncio.run(run())

    def test_fresh_omits_facts(self):
        async def run():
            reg = _MockRegistry([_MockAgent("dc-agent", "http://dc", ["dc_fabric_diagnose"])])
            disp = _MockDispatcher()
            fn = build_delegate_fn(reg, disp, _MockStore(), own_agent_id="lan-agent")
            d = _directive("[DELEGATE:dc-agent] no facts shared")
            await _drain(fn(d, "sess-1", ["fact-A"]))
            self.assertNotIn("parent_confirmed_facts", disp.last_task.parameters)
            self.assertEqual(disp.last_task.metadata["shared_facts_count"], 0)
        asyncio.run(run())


class TestDelegateCapability(unittest.TestCase):
    def test_capability_resolution_excludes_self(self):
        async def run():
            reg = _MockRegistry([
                _MockAgent("lan-agent", "http://lan", ["dc_fabric_diagnose"]),  # self also has it
                _MockAgent("dc-agent",  "http://dc",  ["dc_fabric_diagnose"]),
            ])
            disp = _MockDispatcher()
            fn = build_delegate_fn(reg, disp, _MockStore(), own_agent_id="lan-agent")
            d = _directive("[DELEGATE:*dc_fabric_diagnose] trace path")
            await _drain(fn(d, "sess-1", []))
            # must NOT pick self
            self.assertEqual(disp.last_assignment.agent_id, "dc-agent")
        asyncio.run(run())


class TestDelegateDegradation(unittest.TestCase):
    def test_unknown_agent(self):
        async def run():
            reg = _MockRegistry([])
            disp = _MockDispatcher()
            fn = build_delegate_fn(reg, disp, _MockStore(), own_agent_id="lan-agent")
            d = _directive("[DELEGATE:ghost-agent] do x")
            chunks = await _drain(fn(d, "sess-1", []))
            # one note chunk, no dispatch
            self.assertTrue(any("unresolved" in str(c.get("node_step", "")).lower()
                                for c in chunks))
            self.assertIsNone(disp.last_task)
            self.assertEqual(reg.task_starts, [])   # never started
        asyncio.run(run())

    def test_unhealthy_agent(self):
        async def run():
            reg = _MockRegistry([_MockAgent("dc-agent", "http://dc",
                                            ["dc_fabric_diagnose"], available=False)])
            disp = _MockDispatcher()
            fn = build_delegate_fn(reg, disp, _MockStore(), own_agent_id="lan-agent")
            d = _directive("[DELEGATE:dc-agent] do x")
            chunks = await _drain(fn(d, "sess-1", []))
            self.assertIsNone(disp.last_task)
        asyncio.run(run())

    def test_no_capability_match(self):
        async def run():
            reg = _MockRegistry([_MockAgent("dc-agent", "http://dc", ["dc_fabric_diagnose"])])
            disp = _MockDispatcher()
            fn = build_delegate_fn(reg, disp, _MockStore(), own_agent_id="lan-agent")
            d = _directive("[DELEGATE:*nonexistent_cap] do x")
            chunks = await _drain(fn(d, "sess-1", []))
            self.assertIsNone(disp.last_task)
        asyncio.run(run())

    def test_dispatch_exception_brackets_task_end(self):
        async def run():
            reg = _MockRegistry([_MockAgent("dc-agent", "http://dc", ["dc_fabric_diagnose"])])
            disp = _MockDispatcher(raise_exc=RuntimeError("peer down"))
            fn = build_delegate_fn(reg, disp, _MockStore(), own_agent_id="lan-agent")
            d = _directive("[DELEGATE:dc-agent] do x")
            chunks = await _drain(fn(d, "sess-1", []))
            # error surfaced as a chunk, and task_end still called (finally)
            self.assertTrue(any("error" in str(c.get("node_step", "")).lower()
                                for c in chunks))
            self.assertEqual(reg.task_starts, ["dc-agent"])
            self.assertEqual(reg.task_ends, ["dc-agent"])
        asyncio.run(run())


class TestHandleDelegateLoopSide(unittest.TestCase):
    """Tests AgentRuntimeLoop._handle_delegate: source_agent tagging +
    _inject_context generation, with a mock delegate_fn."""

    def _loop_with(self, delegate_fn):
        from runtime.loop import AgentRuntimeLoop
        return AgentRuntimeLoop(delegate_fn=delegate_fn)

    def test_tags_source_and_injects_result(self):
        async def run():
            async def fake_delegate(directive, session_id, shared_facts):
                yield {"token": "spine-1 ok"}
                yield {"token": " neighbors=3"}
            loop = self._loop_with(fake_delegate)
            from runtime.stop_policy import LoopState
            d = _directive("[DELEGATE:dc-agent] check spine-1")
            out = await _drain(loop._handle_delegate(d, LoopState(), "sess-1"))
            # forwarded chunks tagged source_agent=dc-agent
            tok_chunks = [c for c in out if c.get("token")]
            self.assertTrue(tok_chunks)
            for c in tok_chunks:
                self.assertEqual(c.get("source_agent"), "dc-agent")
            # exactly one _inject_context carrying the merged result
            injects = [c for c in out if c.get("_inject_context")]
            self.assertEqual(len(injects), 1)
            self.assertIn("spine-1 ok", injects[0]["_inject_context"])
            self.assertIn("dc-agent", injects[0]["_inject_context"])
        asyncio.run(run())

    def test_no_delegate_fn_degrades(self):
        async def run():
            loop = self._loop_with(None)   # delegation not wired
            from runtime.stop_policy import LoopState
            d = _directive("[DELEGATE:dc-agent] x")
            out = await _drain(loop._handle_delegate(d, LoopState(), "sess-1"))
            injects = [c for c in out if c.get("_inject_context")]
            self.assertEqual(len(injects), 1)
            self.assertIn("委派", injects[0]["_inject_context"])  # degradation note
        asyncio.run(run())

    def test_peer_hitl_surfaces_hint(self):
        async def run():
            async def fake_delegate(directive, session_id, shared_facts):
                yield {"type": "hitl_interrupt", "token": "needs approval"}
            loop = self._loop_with(fake_delegate)
            from runtime.stop_policy import LoopState
            d = _directive("[DELEGATE:dc-agent] push config")
            out = await _drain(loop._handle_delegate(d, LoopState(), "sess-1"))
            injects = [c for c in out if c.get("_inject_context")]
            self.assertEqual(len(injects), 1)
            self.assertIn("审批", injects[0]["_inject_context"])  # HITL hint
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main(verbosity=2)
