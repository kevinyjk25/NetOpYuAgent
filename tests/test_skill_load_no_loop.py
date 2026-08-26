"""Regression: SKILL_LOAD across turns must not loop.

DC was observed re-emitting `[SKILL_LOAD:dc_app_access_diagnose]` on every
turn because context_str is rebuilt per turn (the skill detail appended to
context_str didn't survive), and the cross-turn dedup that called_tools.add
was supposed to enable was never actually checked. This test pins both:

  1. called_tools containing "SKILL_LOAD:X" suppresses re-emission of the
     same directive in a later turn.
  2. The loaded skill detail is persisted via state.record_new_fact so the
     next turn's context_str still contains the skill content (eliminating
     the LLM's reason to re-emit SKILL_LOAD).

DESIGN EVOLUTION (2026-06): these guards used to be asserted in BOTH run()
and stream() (count >= 2) because run() carried its own duplicated turn
loop. run() is now a thin collector over stream() (single execution path —
the duplication was deliberately deleted, also closing a safety gap where
run() executed watch-listed tools without HITL). The guards therefore exist
exactly ONCE, in _stream_impl; a third test pins run() as a wrapper so the
duplicated-loop drift can never silently return.
"""
import unittest
from pathlib import Path


def _src() -> str:
    return Path("runtime/loop.py").read_text(encoding="utf-8")


class TestSkillLoadCrossTurnDedup(unittest.TestCase):
    def test_skill_load_persists_as_confirmed_fact(self):
        src = _src()
        self.assertIn("record_new_fact", src,
                      "expected state.record_new_fact to persist SKILL_LOAD detail")
        self.assertIn("[SKILL LOADED:", src,
                      "expected '[SKILL LOADED:' marker in the persisted fact")
        n = src.count('f"[SKILL LOADED: {skill_id}]\\n{detail}"')
        self.assertGreaterEqual(n, 1,
                                f"expected SKILL LOADED fact persistence in the "
                                f"unified stream path; found {n}")

    def test_skill_load_cross_turn_guard_present(self):
        src = _src()
        self.assertIn('f"SKILL_LOAD:{skill_id}" in called_tools', src,
                      "expected cross-turn SKILL_LOAD dedup against called_tools")
        n = src.count('f"SKILL_LOAD:{skill_id}" in called_tools')
        self.assertGreaterEqual(n, 1,
                                f"expected cross-turn SKILL_LOAD guard in the "
                                f"unified stream path; found {n}")

    def test_run_is_stream_collector_no_duplicate_loop(self):
        """run() must remain a collector over stream() — if a turn loop
        (while True + _call_llm) reappears inside run(), the drift class
        this refactor eliminated has returned."""
        src = _src()
        i = src.find("    async def run(")
        j = src.find("\n    # ------", i + 10)
        body = src[i:j]
        self.assertIn("self.stream(", body,
                      "run() must delegate to stream()")
        self.assertNotIn("while True", body,
                         "run() must NOT contain its own turn loop")
        self.assertNotIn("_call_llm", body,
                         "run() must NOT call the LLM directly")
        # The safety gap that motivated the refactor: HITL gating must apply
        # to run() via stream(); a stop_hitl chunk maps to STOP_HITL.
        self.assertIn("stop_hitl", body,
                      "run() must surface stream's HITL gate as STOP_HITL")






class TestSkillLoadCompletion(unittest.TestCase):
    """is_complete must keep the loop running after an HONORED SKILL_LOAD —
    even one accompanied by prose preamble — so the skill's procedure runs.
    A suppressed re-emit must NOT keep looping forever."""

    def test_honored_skill_load_with_prose_is_not_complete(self):
        from runtime.loop_helpers import is_complete
        # The exact shape that caused the dormant-skill bug: prose + SKILL_LOAD,
        # no tool call. Honored load → must NOT be complete (need next turn).
        resp = ("收到，我将启动新员工 alice 的访问开通流程。首先，加载标准操作指南：\n"
                "[SKILL_LOAD:lan_new_employee_onboarding_access]")
        self.assertFalse(is_complete(resp, [], skill_load_honored=True),
                         "honored SKILL_LOAD+prose must keep the loop running")

    def test_pure_skill_load_is_not_complete(self):
        from runtime.loop_helpers import is_complete
        self.assertFalse(is_complete("[SKILL_LOAD:x]", [], skill_load_honored=True))

    def test_suppressed_reemit_does_not_loop_forever(self):
        from runtime.loop_helpers import is_complete
        # Already-loaded skill re-emitted → not honored → falls through to the
        # tool-call rule → with no tool call, this is "complete" so the loop
        # can terminate instead of spinning.
        resp = "[SKILL_LOAD:already_loaded]"
        self.assertTrue(is_complete(resp, [], skill_load_honored=False),
                        "suppressed re-emit must not keep the loop alive forever")

    def test_normal_prose_answer_is_complete(self):
        from runtime.loop_helpers import is_complete
        self.assertTrue(is_complete("alice is admitted; done.", []))

    def test_tool_call_is_not_complete(self):
        from runtime.loop_helpers import is_complete
        self.assertFalse(is_complete("[TOOL:get_user_access] {}", [("get_user_access", {})]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
