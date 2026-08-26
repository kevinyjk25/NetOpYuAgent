"""Reasoning / interim / conclusion separation.

The UI mixed thinking + conclusion in one bubble. The loop now splits output
into three chunk kinds so the frontend renders process vs answer separately:
  - reasoning : the model's <think>…</think> content
  - interim   : outward prose on a turn that ALSO calls a tool (not the answer)
  - token     : the final turn's prose (the conclusion)
"""
import asyncio
import unittest

from runtime.loop import AgentRuntimeLoop
from runtime.loop_types import RuntimeConfig


def _collect(llm, tool_registry=None):
    async def go():
        loop = AgentRuntimeLoop(config=RuntimeConfig(), llm_fn=llm)
        out = {"reasoning": [], "interim": [], "tokens": []}
        async for ch in loop.stream(query="q", session_id="sep",
                                    tool_registry=tool_registry or {}):
            if not isinstance(ch, dict):
                continue
            if ch.get("type") == "reasoning":
                out["reasoning"].append(ch["reasoning"])
            elif ch.get("type") == "interim":
                out["interim"].append(ch["interim"])
            elif ch.get("token"):
                out["tokens"].append(ch["token"])
        return out
    return asyncio.run(go())


class TestReasoningSeparation(unittest.TestCase):
    def test_think_block_becomes_reasoning_not_conclusion(self):
        async def llm(q, c, s):
            return "<think>推理过程在这里</think>最终结论:一切正常。"
        out = _collect(llm)
        self.assertTrue(out["reasoning"])
        self.assertIn("推理过程", out["reasoning"][0])
        # conclusion bubble must NOT contain the thinking
        joined = "".join(out["tokens"])
        self.assertIn("最终结论", joined)
        self.assertNotIn("推理过程", joined)

    def test_intermediate_prose_is_interim_not_conclusion(self):
        _t = {"n": 0}
        async def llm(q, c, s):
            _t["n"] += 1
            if _t["n"] == 1:
                return "我先查一下设备。\n[TOOL:list_devices] {}"
            return "诊断完成:设备在线。"
        async def list_devices(a):
            return "dev1 up"
        out = _collect(llm, {"list_devices": list_devices})
        # the "let me check" narration is interim (process), not the answer
        self.assertTrue(out["interim"])
        self.assertIn("我先查", out["interim"][0])
        joined = "".join(out["tokens"])
        self.assertIn("诊断完成", joined)
        self.assertNotIn("我先查", joined)

    def test_single_turn_no_tool_goes_to_bubble(self):
        async def llm(q, c, s):
            return "直接结论,无需工具。"
        out = _collect(llm)
        self.assertEqual(out["interim"], [])
        self.assertIn("直接结论", "".join(out["tokens"]))


class TestDelegationSeparation(unittest.TestCase):
    """Delegated peer output must render separately from the origin's
    conclusion: the peer's stream is a 'delegated' chunk (process area), the
    origin's post-delegation synthesis is the conclusion (bubble)."""
    def test_peer_output_is_delegated_not_conclusion(self):
        async def go():
            async def delegate(directive, session_id, shared_facts, *, original_query=""):
                yield {"token": "DC: EVPN 路由正常"}
            _t = {"n": 0}
            async def llm(q, c, s):
                _t["n"] += 1
                if _t["n"] == 1:
                    return "查 DC。\n[DELEGATE:dc-agent] 查路由"
                return "综合结论:一切正常。"
            loop = AgentRuntimeLoop(config=RuntimeConfig(), llm_fn=llm,
                                    delegate_fn=delegate)
            out = {"interim": [], "delegated": [], "tokens": []}
            async for ch in loop.stream(query="q", session_id="ds",
                                        tool_registry={}):
                if not isinstance(ch, dict):
                    continue
                if ch.get("type") == "interim":
                    out["interim"].append(ch["interim"])
                elif ch.get("type") == "delegated":
                    out["delegated"].append(ch["delegated"])
                elif ch.get("token"):
                    out["tokens"].append(ch["token"])
            return out
        out = asyncio.run(go())
        # peer output is in the delegated stream, not the bubble
        self.assertIn("EVPN", "".join(out["delegated"]))
        self.assertNotIn("EVPN", "".join(out["tokens"]))
        # the delegate turn's narration is interim, not the bubble
        self.assertIn("查 DC", "".join(out["interim"]))
        # the bubble holds only the origin's synthesis
        self.assertIn("综合结论", "".join(out["tokens"]))
        self.assertNotIn("查 DC", "".join(out["tokens"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
