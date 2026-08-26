"""S10 — user interrupt (Stop button → abort → partial preserved).

The webui SSE generator catches GeneratorExit/CancelledError when the
operator hits Stop, sets outcome=USER_CANCELLED, and preserves the partial
answer. That backend wrapper relies on a loop-level contract: when the
consumer closes the stream mid-flight, the loop must

  - have already yielded the partial chunks (not buffer-all-then-emit), and
  - run its cleanup / SESSION_END-with-abort path on the way out.

This pins that loop contract (the backend SSE wrapper itself needs the full
app and is covered by manual/integration testing — see matrix S10 note).
"""
import asyncio
import unittest

from runtime.hooks import get_hook_registry, HookEvent
from runtime.loop import AgentRuntimeLoop
from runtime.loop_types import RuntimeConfig


def _loop(llm_fn):
    return AgentRuntimeLoop(memory_router=None, config=RuntimeConfig(), llm_fn=llm_fn)


class TestUserInterrupt(unittest.TestCase):
    def test_abort_midstream_fires_session_end_and_keeps_partial(self):
        async def t():
            # capture SESSION_END payloads
            seen = []
            reg = get_hook_registry()

            async def _on_end(stats):
                seen.append(dict(stats))

            _hname = reg.register(HookEvent.SESSION_END, _on_end)
            try:
                async def llm(query, context, state):
                    # long enough to be a real turn; we abort before turn 2
                    return ("正在执行多步诊断,第一步已完成,正在继续后续检查项,"
                            "这是一段足够长的中间输出用于验证部分保留。")

                loop = _loop(llm)
                collected = []
                agen = loop.stream(query="长诊断", session_id="s10-abort",
                                   tool_registry={})
                # consume ONE chunk then abort (simulate Stop button)
                async for chunk in agen:
                    collected.append(chunk)
                    break
                await agen.aclose()   # consumer closes the iterator → GeneratorExit

                # partial output was emitted before abort (not all-or-nothing)
                self.assertTrue(collected, "expected at least one partial chunk before abort")
                # SESSION_END fired on the way out, marked as a consumer close
                self.assertTrue(seen, "SESSION_END must fire even on abort")
                self.assertEqual(seen[-1].get("outcome"), "consumer_closed")
            finally:
                reg.unregister(HookEvent.SESSION_END, _hname)
        asyncio.run(t())

    def test_normal_completion_fires_session_end_completed(self):
        async def t():
            seen = []
            reg = get_hook_registry()

            async def _on_end(stats):
                seen.append(dict(stats))

            _hname = reg.register(HookEvent.SESSION_END, _on_end)
            try:
                async def llm(query, context, state):
                    return "诊断完成,一切正常,这是一段足够长的最终答复用于正常结束流程验证。"
                loop = _loop(llm)
                async for _ in loop.stream(query="q", session_id="s10-ok",
                                           tool_registry={}):
                    pass
                self.assertTrue(seen)
                self.assertEqual(seen[-1].get("outcome"), "completed")
            finally:
                reg.unregister(HookEvent.SESSION_END, _hname)
        asyncio.run(t())


if __name__ == "__main__":
    unittest.main(verbosity=2)
