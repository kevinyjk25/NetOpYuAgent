"""tests/test_delegation_park_on_peer_hitl.py — A2A Phase 3 case2 park
=====================================================================

When a delegated peer raises an operator-approval HITL (case2), the originating
request must PARK: emit a deterministic interim and end the stream, WITHOUT
busy-looping. Busy-looping races with the async stage-2 result callback — the
instant it flips the outbound task to COMPLETED, the in-flight gate releases and
the next loop turn re-delegates → duplicate inbound task on the peer → storm.

This proves:
  * the stream ends after the peer HITL (no 2nd LLM turn fired),
  * an interim message reached the user,
  * delegate_fn was called exactly once (no re-delegation).
"""
import asyncio
import unittest

from runtime.loop import AgentRuntimeLoop, RuntimeConfig


class _CountingLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
    async def __call__(self, query, context, state):
        i = self.calls
        self.calls += 1
        return self._responses[i] if i < len(self._responses) else "done."


async def _drain(agen):
    return [c async for c in agen]


class TestParkOnPeerHitl(unittest.TestCase):
    def test_parks_and_does_not_reloop(self):
        async def run():
            # The LLM would delegate on turn 1, and (if it got a turn 2) would
            # delegate AGAIN — simulating qwen's disobedience. The park must
            # prevent turn 2 from ever happening.
            llm = _CountingLLM([
                "[DELEGATE:dc-agent] grant alice crm access",
                "[DELEGATE:dc-agent] grant alice crm access AGAIN",
            ])
            delegate_calls = {"n": 0}

            async def delegate_fn(directive, session_id, shared_facts):
                delegate_calls["n"] += 1
                # Peer streams an intermediate token then raises a HITL.
                yield {"token": "alice has no role for crm",
                       "source_agent": "dc-agent"}
                yield {"type": "hitl_interrupt", "source_agent": "dc-agent",
                       "interrupt_id": "x-123"}

            loop = AgentRuntimeLoop(
                memory_router=None, config=RuntimeConfig(),
                llm_fn=llm, delegate_fn=delegate_fn,
            )
            chunks = await _drain(loop.stream(
                query="诊断 alice 访问 crm 失败；DC 权限问题委派 dc-agent",
                session_id="sess-park",
                tool_registry={},
            ))

            # delegate_fn called exactly once — NO re-delegation.
            self.assertEqual(delegate_calls["n"], 1,
                             "peer HITL must park, not re-delegate")
            # The LLM ran exactly once (turn 1). The park ended the stream
            # before any synthesis/second turn could fire another delegate.
            self.assertEqual(llm.calls, 1,
                             "park must end the stream — no 2nd LLM turn")
            # An interim message reached the user.
            user_text = "".join(
                c["token"] for c in chunks
                if c.get("token") and c.get("source_agent") != "dc-agent")
            self.assertIn("操作员审批", user_text)
            # A machine-readable park marker was emitted for the frontend.
            self.assertTrue(any(c.get("type") == "cross_agent_parked" for c in chunks),
                            "park must emit a cross_agent_parked marker chunk")
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestCase1NoReDelegate(unittest.TestCase):
    """A case1 (synchronous, no-HITL) peer return that the LLM reads as
    inconclusive must NOT be re-delegated to the same peer in the same
    request. The per-request block forces synthesis/degradation instead."""

    def test_same_peer_not_re_delegated_in_one_request(self):
        async def run():
            # Turn 1: delegate. Turn 2: LLM tries to delegate the SAME task to
            # the SAME peer again (simulating the storm). Turn 3 safety: prose.
            llm = _CountingLLM([
                "[DELEGATE:dc-agent] diagnose alice crm access",
                "[DELEGATE:dc-agent] diagnose alice crm access",  # must be blocked
                "Final: alice lacks a crm role; grant pending operator action.",
            ])
            calls = {"n": 0}

            async def delegate_fn(directive, session_id, shared_facts):
                calls["n"] += 1
                # case1: peer returns a synchronous (no-HITL) inconclusive answer
                yield {"token": "alice has no crm role; shall I grant?",
                       "source_agent": "dc-agent"}

            loop = AgentRuntimeLoop(
                memory_router=None, config=RuntimeConfig(),
                llm_fn=llm, delegate_fn=delegate_fn,
            )
            chunks = await _drain(loop.stream(
                query="诊断 alice 访问 crm 失败；DC 权限问题委派 dc-agent",
                session_id="sess-case1",
                tool_registry={},
            ))
            # delegate_fn called exactly once — the 2nd attempt was suppressed.
            self.assertEqual(calls["n"], 1,
                             "same (task,peer) must not be re-delegated in one request")
            # No raw [DELEGATE:] directive leaked into the user-visible stream.
            user_text = "".join(c["token"] for c in chunks if c.get("token"))
            self.assertNotIn("[DELEGATE:", user_text,
                             "directive text must be stripped from the visible stream")
        asyncio.run(run())
