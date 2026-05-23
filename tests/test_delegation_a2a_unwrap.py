"""
tests/test_delegation_a2a_unwrap.py
===================================

Regression test for the delegation transport bug found in live two-agent
testing (2026-05): the peer (delegated) agent streams A2A PROTOCOL EVENTS
(TaskArtifactUpdateEvent / MessageEvent / TaskStatusUpdateEvent) over SSE,
where the actual token/message text is nested at
`event.artifact.parts[*].data`. The dispatcher previously yielded the raw
event envelope to the delegating runtime loop, which looks for a top-level
`token` key — so delegation "succeeded" but returned empty content.

A2ATaskDispatcher._unwrap_a2a_event must translate each envelope into the
flat {token,...} / {message,...} chunks the loop understands, and emit nothing
for non-content events (status transitions like WORKING).

Pure dict-shape tests — no httpx / network. Runs in sandbox + CI.
"""
import unittest

from task.inter.coordinator import A2ATaskDispatcher as D


class TestA2AUnwrap(unittest.TestCase):
    def test_token_artifact(self):
        evt = {"artifact": {"name": "llm_token",
                            "parts": [{"data": {"token": "spine-1 up", "type": "token"}}]}}
        self.assertEqual(D._unwrap_a2a_event(evt), [{"token": "spine-1 up"}])

    def test_message_artifact(self):
        evt = {"artifact": {"name": "node_message",
                            "parts": [{"data": {"text": "done", "node": "verify",
                                                "type": "message"}}]}}
        out = D._unwrap_a2a_event(evt)
        self.assertEqual(out, [{"message": "done", "node": "verify"}])

    def test_tokens_batch(self):
        evt = {"artifact": {"parts": [{"data": {"tokens": ["a", "b", "c"],
                                                "type": "tokens_batch"}}]}}
        self.assertEqual(D._unwrap_a2a_event(evt),
                         [{"token": "a"}, {"token": "b"}, {"token": "c"}])

    def test_status_working_emits_keepalive(self):
        # A non-terminal status (working/submitted/running) now emits a brief
        # progress chunk — this keeps the delegating-side SSE alive during the
        # peer's long agent loop and gives the operator a Flow event. (Changed
        # from the original "silent" contract to fix the 300s stall where the
        # parent cancelled before the peer's first token; v13.)
        out = D._unwrap_a2a_event({"status": {"state": "working"}})
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["node"], "delegate")
        self.assertIn("working", out[0]["node_step"])
        # Terminal states stay silent (caller sees MessageEvent / stream end).
        self.assertEqual(D._unwrap_a2a_event({"status": {"state": "completed"}}), [])
        self.assertEqual(D._unwrap_a2a_event({"status": {"state": "canceled"}}), [])

    def test_status_failed_surfaces_error(self):
        out = D._unwrap_a2a_event({"status": {"state": "failed", "message": "boom"}})
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["error"], "boom")
        self.assertEqual(out[0]["node"], "delegate")

    def test_error_envelope(self):
        out = D._unwrap_a2a_event({"error": "peer exploded"})
        self.assertEqual(len(out), 1)
        self.assertIn("peer exploded", out[0]["error"])

    def test_final_assistant_message(self):
        evt = {"message": {"role": "assistant",
                           "parts": [{"text": "Here is the result"}]}}
        self.assertEqual(D._unwrap_a2a_event(evt), [{"token": "Here is the result"}])

    def test_final_message_skips_placeholder(self):
        # "Task completed." is the executor's placeholder, not real content.
        evt = {"message": {"role": "assistant",
                           "parts": [{"text": "Task completed."}]}}
        self.assertEqual(D._unwrap_a2a_event(evt), [])

    def test_non_dict_safe(self):
        self.assertEqual(D._unwrap_a2a_event(None), [])
        self.assertEqual(D._unwrap_a2a_event("oops"), [])

    def test_end_to_end_token_accumulation(self):
        """Simulate a realistic peer event sequence → flat chunks the loop
        would accumulate into a usable result."""
        events = [
            {"status": {"state": "working"}},                 # silent
            {"artifact": {"parts": [{"data": {"token": "BGP EVPN on spine-1: ",
                                              "type": "token"}}]}},
            {"artifact": {"parts": [{"data": {"token": "3 neighbors Established",
                                              "type": "token"}}]}},
            {"status": {"state": "completed"}},               # silent
            {"message": {"role": "assistant", "parts": [{"text": "Task completed."}]}},  # skipped
        ]
        toks = []
        for e in events:
            for c in D._unwrap_a2a_event(e):
                if c.get("token"):
                    toks.append(c["token"])
        self.assertEqual("".join(toks), "BGP EVPN on spine-1: 3 neighbors Established")

    def test_round_trip_matches_executor_event_shape(self):
        """The exact event shape HitlExecutor.execute now emits (a
        TaskArtifactUpdateEvent with a DataPart token) must round-trip through
        model_dump → _unwrap_a2a_event back to {token}. This is the contract
        whose break caused the ReadTimeout: emitter and consumer must agree on
        the artifact.parts[].data.token path."""
        try:
            import importlib.util, types, sys
            if "a2a" not in sys.modules:
                pkg = types.ModuleType("a2a"); pkg.__path__ = ["a2a"]
                sys.modules["a2a"] = pkg
            spec = importlib.util.spec_from_file_location("a2a.schemas", "a2a/schemas.py")
            sch = importlib.util.module_from_spec(spec)
            sys.modules["a2a.schemas"] = sch
            spec.loader.exec_module(sch)
        except Exception as e:
            self.skipTest(f"a2a.schemas unavailable: {e}")
        evt = sch.TaskArtifactUpdateEvent(
            task_id="t1", context_id="c1",
            artifact=sch.Artifact(
                name="llm_token",
                parts=[sch.DataPart(data={"token": "spine-1 up", "type": "token"})]),
        )
        out = D._unwrap_a2a_event(evt.model_dump())
        self.assertEqual(out, [{"token": "spine-1 up"}])
        # MessageEvent (kind=="message") is the queue-finalizing terminal event.
        me = sch.MessageEvent(
            task_id="t1", context_id="c1",
            message=sch.Message(role="assistant", parts=[sch.TextPart(text="done")]))
        self.assertEqual(me.kind, "message")


if __name__ == "__main__":
    unittest.main(verbosity=2)
