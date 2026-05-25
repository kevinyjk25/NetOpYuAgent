"""
tests/test_handle_tools_phase.py
================================

Locks in the Item 4 4e extraction of AgentRuntimeLoop._handle_tools.

Two contracts that MUST be preserved exactly (HITL safety):
  1. Non-HITL tool  → executes, stores result in ctx.tool_outputs, marks
     ctx.called_tools, streams a tool-result chunk, no terminal sentinel.
  2. HITL-watchlist tool → yields a stop_hitl/hitl_gate chunk and then the
     {"_tools_terminal": True} sentinel (so _stream_impl returns and the
     executor's HITL interrupt fires). The tool must NOT execute.

Driven directly with a stubbed _execute_tool + minimal _LoopContext — no LLM,
no pydantic, no httpx.
"""
import asyncio
import unittest


def _ctx(query="check status"):
    from runtime.loop_context import _LoopContext
    from runtime.stop_policy import LoopState
    from runtime.loop_types import DelegationMode
    st = LoopState()
    st.turns = 1
    st._skill_journal = None   # set dynamically by _stream_impl setup in prod
    return _LoopContext(
        query=query, session_id="sess-tools", env_ctx={},
        tool_reg={"get_status": object(), "edit_device_config": object()},
        delegation_mode=DelegationMode.FRESH, parent_state=None, state=st,
    )


def _loop(hitl_names=frozenset()):
    from runtime.loop import AgentRuntimeLoop, RuntimeConfig
    cfg = RuntimeConfig(hitl_tool_names=hitl_names)
    return AgentRuntimeLoop(memory_router=None, config=cfg)


class TestHandleToolsPhase(unittest.TestCase):
    def test_non_hitl_tool_executes_and_streams(self):
        async def run():
            loop = _loop(hitl_names=frozenset())   # nothing gated
            ctx = _ctx()

            async def _stub_exec(tool_name, args, registry):
                return f"RESULT[{tool_name}]: 3 neighbors up"
            loop._execute_tool = _stub_exec  # type: ignore

            chunks = []
            async for c in loop._handle_tools(ctx, [("get_status", {"device": "spine-1"})],
                                              llm_response="[TOOL:get_status]"):
                chunks.append(c)

            # No HITL terminal sentinel.
            self.assertFalse(any(c.get("_tools_terminal") for c in chunks))
            # A tool-result chunk was streamed.
            self.assertTrue(any("node_result" in c for c in chunks))
            # Result stored + tool marked called in ctx.
            self.assertTrue(any("3 neighbors up" in str(v) for v in ctx.tool_outputs.values()))
            self.assertTrue(len(ctx.called_tools) >= 1)
        asyncio.run(run())

    def test_hitl_tool_stops_and_does_not_execute(self):
        async def run():
            loop = _loop(hitl_names=frozenset({"edit_device_config"}))
            ctx = _ctx(query="fix the vlan on spine-1")

            executed = {"called": False}
            async def _stub_exec(tool_name, args, registry):
                executed["called"] = True
                return "SHOULD NOT RUN"
            loop._execute_tool = _stub_exec  # type: ignore

            chunks = []
            async for c in loop._handle_tools(
                ctx,
                [("edit_device_config", {"config_lines": ["vlan 10"]})],
                llm_response="[TOOL:edit_device_config]",
            ):
                chunks.append(c)

            # Must have raised a HITL gate chunk ...
            self.assertTrue(any(c.get("stop_hitl") for c in chunks),
                            f"expected a stop_hitl chunk, got {chunks}")
            # ... and ended with the terminal sentinel.
            self.assertEqual(chunks[-1], {"_tools_terminal": True})
            # The destructive tool must NOT have executed.
            self.assertFalse(executed["called"], "HITL tool executed without approval!")
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main(verbosity=2)
