"""
tests/test_clarification_gate.py
================================

Locks in the Item 4 4d extraction of AgentRuntimeLoop._run_clarification_gate.

The gate is an async generator:
  - If clarification is needed (turn 1, vague query), it streams the question
    as chat tokens + a clarification_chat chunk, then yields a terminal
    sentinel {"_clarification_terminal": True} so _stream_impl returns.
  - If not needed, it yields nothing.

We drive the method directly with a stubbed _maybe_clarification_fields and a
minimal _LoopContext, so no LLM / pydantic / httpx is involved.
"""
import asyncio
import unittest


def _ctx(loop, turns=1):
    from runtime.loop_context import _LoopContext
    from runtime.stop_policy import LoopState
    from runtime.loop_types import DelegationMode
    st = LoopState()
    st.turns = turns
    return _LoopContext(
        query="修复设备",                 # action word, no concrete target
        session_id="sess-clar",
        env_ctx={},
        tool_reg={},
        delegation_mode=DelegationMode.FRESH,
        parent_state=None,
        state=st,
    )


def _loop():
    from runtime.loop import AgentRuntimeLoop
    return AgentRuntimeLoop(memory_router=None)


class TestClarificationGate(unittest.TestCase):
    def test_asks_then_emits_terminal_sentinel(self):
        async def run():
            loop = _loop()
            ctx = _ctx(loop, turns=1)

            async def _stub_fields(*, query, top_skill_score, asked_count, recent_context=""):
                return [{"key": "device_id", "prompt": "哪个设备?", "required": True}]
            loop._maybe_clarification_fields = _stub_fields  # type: ignore

            chunks = []
            async for c in loop._run_clarification_gate(ctx, memory_results=[], selected_skills=[]):
                chunks.append(c)

            # Must have streamed at least one token + ended with the sentinel.
            self.assertTrue(any("token" in c for c in chunks))
            self.assertTrue(any(c.get("_clarification_terminal") for c in chunks))
            self.assertEqual(chunks[-1], {"_clarification_terminal": True})
            # And it counted the ask against the session budget.
            self.assertEqual(loop._clarification_counts.get("sess-clar", 0), 1)
        asyncio.run(run())

    def test_no_clarification_yields_nothing(self):
        async def run():
            loop = _loop()
            ctx = _ctx(loop, turns=1)

            async def _stub_none(*, query, top_skill_score, asked_count, recent_context=""):
                return []   # query is specific enough
            loop._maybe_clarification_fields = _stub_none  # type: ignore

            chunks = [c async for c in loop._run_clarification_gate(ctx, [], [])]
            self.assertEqual(chunks, [])
        asyncio.run(run())

    def test_skipped_after_turn_1(self):
        async def run():
            loop = _loop()
            ctx = _ctx(loop, turns=2)   # mid-execution, never clarify

            async def _should_not_be_called(**kw):
                raise AssertionError("clarification must not run after turn 1")
            loop._maybe_clarification_fields = _should_not_be_called  # type: ignore

            chunks = [c async for c in loop._run_clarification_gate(ctx, [], [])]
            self.assertEqual(chunks, [])
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main(verbosity=2)
