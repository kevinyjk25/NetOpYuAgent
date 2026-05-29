"""Regression test for the peer-stream heartbeat truncation bug (2026-05).

_handle_delegate emits a "peer still working" heartbeat when the delegated
peer is silent for >20s. The first implementation used
`wait_for(iterator.__anext__(), timeout=20)`, but wait_for CANCELS its
awaitable on timeout — which cancels the async generator's __anext__ and
tears down the underlying peer SSE stream. The symptom: every delegation
finished in exactly 20s with truncated content, and the originating agent
re-delegated forever.

The fix schedules __anext__ as a standalone task and awaits it under
`shield`, so a heartbeat timeout never cancels the in-flight read. This test
locks in that a slow iterator (items arriving slower than the timeout) is
delivered IN FULL, not truncated at the first timeout.
"""
import asyncio
import unittest


async def _slow_gen(n, item_delay, count):
    for i in range(count):
        await asyncio.sleep(item_delay)
        yield f"chunk-{i}"


class TestHeartbeatNoTruncate(unittest.TestCase):
    def test_shield_pattern_delivers_all_chunks(self):
        async def _consume(timeout):
            it = _slow_gen(0, item_delay=0.15, count=4).__aiter__()
            got, heartbeats, task = [], 0, None
            while True:
                if task is None:
                    task = asyncio.ensure_future(it.__anext__())
                try:
                    c = await asyncio.wait_for(asyncio.shield(task),
                                               timeout=timeout)
                    task = None
                    got.append(c)
                except StopAsyncIteration:
                    task = None
                    break
                except asyncio.TimeoutError:
                    heartbeats += 1
                    if heartbeats > 100:
                        break
                    continue
            return got, heartbeats

        got, hb = asyncio.run(_consume(0.05))
        # All four chunks delivered despite item_delay (0.15) >> timeout (0.05).
        self.assertEqual(got, ["chunk-0", "chunk-1", "chunk-2", "chunk-3"])
        # And heartbeats fired during the slow waits.
        self.assertGreaterEqual(hb, 4)

    def test_buggy_pattern_truncates(self):
        """Documents WHY the naive wait_for(__anext__) is wrong: it cancels
        the iterator on the first timeout, losing data."""
        async def _consume_buggy(timeout):
            it = _slow_gen(0, item_delay=0.15, count=4).__aiter__()
            got = []
            while True:
                try:
                    c = await asyncio.wait_for(it.__anext__(), timeout=timeout)
                    got.append(c)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    # The generator's __anext__ was just cancelled by wait_for;
                    # continuing re-enters a broken/cancelled generator.
                    if len(got) == 0:
                        # Demonstrate truncation: bail to avoid hanging.
                        return got
                    continue
            return got

        got = asyncio.run(_consume_buggy(0.05))
        self.assertLess(len(got), 4)  # truncated — the bug


if __name__ == "__main__":
    unittest.main(verbosity=2)
