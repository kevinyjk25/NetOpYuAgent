"""Behavioral tests for run() = stream() collector (option-a refactor, 2026-06).

Pins the three contract points of the refactor:
  1. Graceful path: tokens/facts/turns collected into LoopResult faithfully.
  2. THE SAFETY FIX: a watch-listed destructive tool proposed by the LLM must
     surface as outcome=STOP_HITL and must NOT execute — the old duplicated
     run() body had NO hitl_tool_names gate, so the non-streaming /chat path
     executed destructive tools without operator approval.
  3. Tool execution still works for non-gated tools through the wrapper.
"""
import asyncio
import unittest

from runtime.loop import AgentRuntimeLoop
from runtime.loop_types import RuntimeConfig
from runtime.stop_policy import StopOutcome


class TestRunWrapper(unittest.TestCase):
    def test_graceful_prose_answer_collected(self):
        async def t():
            async def llm(query, context, state):
                return "诊断完成:alice 的 LAN 准入正常。"
            loop = AgentRuntimeLoop(llm_fn=llm, config=RuntimeConfig())
            res = await loop.run(query="诊断 alice", session_id="s-run-1",
                                 tool_registry={})
            self.assertEqual(res.outcome, StopOutcome.STOP_GRACEFUL)
            self.assertIn("准入正常", res.final_response)
            self.assertGreaterEqual(res.turns_taken, 1)
        asyncio.run(t())

    def test_watchlisted_tool_returns_stop_hitl_not_executed(self):
        """THE SAFETY FIX: old run() executed restart_service without HITL."""
        async def t():
            executed = []
            async def restart_service(args):
                executed.append(args)
                return "restarted"
            async def llm(query, context, state):
                return '执行重启。\n[TOOL:restart_service] {"service": "crm"}'
            cfg = RuntimeConfig()
            cfg.hitl_tool_names = {"restart_service"}
            loop = AgentRuntimeLoop(llm_fn=llm, config=cfg)
            res = await loop.run(query="重启 crm", session_id="s-run-2",
                                 tool_registry={"restart_service": restart_service})
            self.assertEqual(res.outcome, StopOutcome.STOP_HITL,
                             "watch-listed tool must surface as STOP_HITL via run()")
            self.assertEqual(executed, [],
                             "the destructive tool must NOT have executed")
            pend = getattr(res, "pending_interaction", None)
            self.assertIsNotNone(pend, "structured gate card must be attached")
            self.assertEqual(pend.get("tool_name"), "restart_service")
        asyncio.run(t())

    def test_non_gated_tool_executes_through_wrapper(self):
        async def t():
            calls = []
            async def list_users(args):
                calls.append(args)
                return '["alice", "bob"]'
            responses = iter([
                '查询用户列表。\n[TOOL:list_users] {}',
                "查询完成:系统中共有 2 个用户,分别是 alice 和 bob,均为正常状态,无异常账号。",
            ])
            async def llm(query, context, state):
                return next(responses,
                            "综合以上结果:共 2 个用户 alice 和 bob,状态正常。")
            loop = AgentRuntimeLoop(llm_fn=llm, config=RuntimeConfig())
            res = await loop.run(query="列出用户", session_id="s-run-3",
                                 tool_registry={"list_users": list_users})
            self.assertEqual(len(calls), 1, "non-gated tool should execute once")
            self.assertEqual(res.outcome, StopOutcome.STOP_GRACEFUL)
            self.assertIn("alice", res.final_response)
        asyncio.run(t())


if __name__ == "__main__":
    unittest.main(verbosity=2)
