"""
tests/test_delegation_e2e.py — Phase 2B end-to-end delegation through stream()
==============================================================================

Drives the REAL AgentRuntimeLoop.stream() with a scripted 2-turn LLM and a
mock delegate_fn, proving the full path:

  turn 1: LLM emits [DELEGATE:dc-agent] <subtask>
          → loop detects directive, calls delegate_fn, streams peer chunks
            (tagged source_agent), injects the peer result into context
  turn 2: LLM (now seeing the injected delegated result in context) produces
          the final synthesized answer

Also verifies the mutual-exclusion rule: when the LLM emits BOTH [TOOL:] and
[DELEGATE:] in one turn, the delegate is suppressed (tool path wins).

No httpx / fastapi — uses injected llm_fn + delegate_fn, memory_router=None.
Runs in sandbox + CI.
"""
import asyncio
import unittest

from runtime.loop import AgentRuntimeLoop, RuntimeConfig


class _ScriptedLLM:
    """Returns a pre-scripted response per turn; records the context it saw."""
    def __init__(self, responses):
        self._responses = list(responses)
        self.contexts_seen = []
        self._i = 0

    async def __call__(self, query, context, state):
        self.contexts_seen.append(context)
        if self._i < len(self._responses):
            r = self._responses[self._i]
            self._i += 1
            return r
        return "done."   # safety: terminate


async def _drain(agen):
    return [c async for c in agen]


class TestDelegationE2E(unittest.TestCase):

    def _loop(self, llm_responses, delegate_chunks):
        async def delegate_fn(directive, session_id, shared_facts):
            for c in delegate_chunks:
                yield c
        return AgentRuntimeLoop(
            memory_router=None,
            config=RuntimeConfig(),
            llm_fn=_ScriptedLLM(llm_responses),
            delegate_fn=delegate_fn,
        ), None

    def test_delegate_then_synthesize(self):
        async def run():
            llm = _ScriptedLLM([
                "[DELEGATE:dc-agent] check BGP EVPN on spine-1",
                "Based on the DC agent: spine-1 BGP EVPN has 3 healthy neighbors. "
                "Your LAN-side issue is unrelated to fabric.",
            ])
            async def delegate_fn(directive, session_id, shared_facts):
                # Simulate the peer streaming back a result.
                self.assertEqual(directive.agent_id, "dc-agent")
                self.assertEqual(directive.task, "check BGP EVPN on spine-1")
                yield {"token": "spine-1: BGP EVPN up, 3 neighbors"}

            loop = AgentRuntimeLoop(
                memory_router=None, config=RuntimeConfig(),
                llm_fn=llm, delegate_fn=delegate_fn,
            )
            chunks = await _drain(loop.stream(
                query="why can't users connect on the LAN?",
                session_id="sess-e2e",
                tool_registry={},
            ))
            # A delegate flow event was emitted, tagged source_agent.
            dlg_steps = [c for c in chunks
                         if c.get("node") == "delegate" and c.get("source_agent")]
            self.assertTrue(dlg_steps, "expected a delegate node_step chunk")
            self.assertEqual(dlg_steps[0]["source_agent"], "dc-agent")

            # The peer token was forwarded (tagged).
            peer_toks = [c for c in chunks
                         if c.get("token") and c.get("source_agent") == "dc-agent"]
            self.assertTrue(peer_toks, "expected forwarded peer token")

            # Turn 2's context contained the injected delegated result, so the
            # LLM could synthesize. Check the 2nd context the LLM saw.
            self.assertGreaterEqual(len(llm.contexts_seen), 2)
            ctx2 = llm.contexts_seen[1]
            self.assertIn("委派结果", ctx2)
            self.assertIn("spine-1", ctx2)

            # Final answer (turn 2 prose) reached the user.
            final_text = "".join(
                c["token"] for c in chunks
                if c.get("token") and c.get("source_agent") != "dc-agent")
            self.assertIn("3 healthy neighbors", final_text)
        asyncio.run(run())

    def test_mutual_exclusion_tool_wins(self):
        async def run():
            # Turn 1 emits BOTH a tool call and a delegate — delegate must be
            # suppressed (tool path wins), and a nudge added to context.
            llm = _ScriptedLLM([
                "[TOOL:list_devices] {}\n[DELEGATE:dc-agent] also check fabric",
                "final answer after tool.",
            ])
            delegate_called = {"n": 0}
            async def delegate_fn(directive, session_id, shared_facts):
                delegate_called["n"] += 1
                yield {"token": "should not be called"}

            # Provide a trivial tool so the tool path has something to run.
            async def list_devices(args):
                return "device-1, device-2"
            loop = AgentRuntimeLoop(
                memory_router=None, config=RuntimeConfig(),
                llm_fn=llm, delegate_fn=delegate_fn,
            )
            await _drain(loop.stream(
                query="list devices and check fabric",
                session_id="sess-mutex",
                tool_registry={"list_devices": list_devices},
            ))
            # delegate_fn must NOT have been called (tool path won).
            self.assertEqual(delegate_called["n"], 0)
        asyncio.run(run())

    def test_unwired_delegate_degrades_gracefully(self):
        async def run():
            # No delegate_fn → loop should inject a "delegation unavailable"
            # note and still produce a final answer (no crash, no loop).
            llm = _ScriptedLLM([
                "[DELEGATE:dc-agent] check fabric",
                "I could not delegate, here is what I can say locally.",
            ])
            loop = AgentRuntimeLoop(
                memory_router=None, config=RuntimeConfig(),
                llm_fn=llm, delegate_fn=None,
            )
            chunks = await _drain(loop.stream(
                query="check fabric",
                session_id="sess-unwired",
                tool_registry={},
            ))
            # Turn 2 saw the degradation note in context.
            self.assertGreaterEqual(len(llm.contexts_seen), 2)
            self.assertIn("委派", llm.contexts_seen[1])
            # Produced a final answer.
            final = "".join(c.get("token", "") for c in chunks)
            self.assertIn("locally", final)
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main(verbosity=2)
