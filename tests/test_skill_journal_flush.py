"""tests/test_skill_journal_flush.py — journal must reach the global store.

The per-stream SkillJournal records selection / skill_load / tool_call during
the stream, but those records are only useful if they're flushed into the
process-wide SkillJournalStore that the /skill_journal/* API + JOURNAL tab read.
This proves the stream() wrapper's finally performs that flush.
"""
import asyncio
import unittest

from runtime.loop import AgentRuntimeLoop, RuntimeConfig
from runtime.skill_journal import get_journal_store


class _LLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
    async def __call__(self, query, context, state):
        i = self.calls
        self.calls += 1
        return self._responses[i] if i < len(self._responses) else "done."


async def _drain(agen):
    return [c async for c in agen]


class TestSkillJournalFlush(unittest.TestCase):
    def test_stream_flushes_journal_to_global_store(self):
        async def run():
            store = get_journal_store()
            before = store.stats().get("count", 0)

            # A plain prose answer (no tools/delegates) — one turn, completes.
            llm = _LLM(["alice is admitted on the LAN; no action needed."])
            loop = AgentRuntimeLoop(
                memory_router=None, config=RuntimeConfig(), llm_fn=llm,
            )
            await _drain(loop.stream(
                query="check alice lan admission",
                session_id="sess-journal-flush-1",
                tool_registry={},
            ))

            after = store.stats().get("count", 0)
            self.assertEqual(after, before + 1,
                             "stream end must flush exactly one journal into the global store")

            recent = store.list_recent(limit=5)
            self.assertTrue(any(e.get("session_id") == "sess-journal-flush-1" for e in recent),
                            "the flushed journal must be visible via list_recent (JOURNAL tab)")
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
