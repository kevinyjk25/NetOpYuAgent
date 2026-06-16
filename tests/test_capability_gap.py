"""Tests for the C capability-gap protocol (2026-06).

When the LLM declares [CAPABILITY_GAP: ...], the loop must:
  - record it to the journal (capability_gap event)
  - add a structured unresolved point
  - emit a capability_gap chunk
  - stop gracefully (no spinning trying alternatives)
  - strip the marker from user-visible prose
"""
import asyncio
import unittest

from runtime.directive_parser import find_capability_gap, strip_capability_gap
from runtime.loop import AgentRuntimeLoop
from runtime.loop_types import RuntimeConfig
from runtime.stop_policy import StopOutcome


class TestCapabilityGapParser(unittest.TestCase):
    def test_parse_detail(self):
        self.assertEqual(
            find_capability_gap("已完成前两步。[CAPABILITY_GAP: 步骤3需重置AD域控密码,无此工具]"),
            "步骤3需重置AD域控密码,无此工具")

    def test_no_marker_returns_none(self):
        self.assertIsNone(find_capability_gap("一切正常,诊断完成。"))

    def test_bare_marker_empty_string(self):
        self.assertEqual(find_capability_gap("[CAPABILITY_GAP:]"), "")

    def test_strip_removes_marker(self):
        self.assertNotIn("CAPABILITY_GAP",
                         strip_capability_gap("答复。[CAPABILITY_GAP: x]"))


class TestCapabilityGapLoop(unittest.TestCase):
    def test_gap_recorded_and_stops_gracefully(self):
        async def t():
            async def llm(query, context, state):
                return ("我已确认用户 alice 的 LAN 准入正常。但本次请求还需重置其 AD 域控密码,"
                        "我没有对应工具,也无可委派的对端。\n"
                        "[CAPABILITY_GAP: 重置 AD 域控密码 — 缺少 AD 管理工具,无法委派]")
            loop = AgentRuntimeLoop(llm_fn=llm, config=RuntimeConfig())
            res = await loop.run(query="诊断 alice 并重置她的 AD 密码",
                                 session_id="s-gap-1", tool_registry={})
            # honest partial answer, not a fake success
            self.assertIn("准入正常", res.final_response)
            # marker stripped from visible prose
            self.assertNotIn("CAPABILITY_GAP", res.final_response)
            # structured unresolved point recorded
            self.assertTrue(any("缺少能力" in u for u in res.unresolved),
                            f"expected gap in unresolved: {res.unresolved}")
            # did not spin many turns trying alternatives
            self.assertLessEqual(res.turns_taken, 2)
        asyncio.run(t())

    def test_gap_event_in_stream(self):
        async def t():
            async def llm(query, context, state):
                return "无法完成。\n[CAPABILITY_GAP: 缺少无线热力图扫描工具]"
            loop = AgentRuntimeLoop(llm_fn=llm, config=RuntimeConfig())
            saw_gap = {"hit": False, "detail": None}
            async for ch in loop.stream(query="扫描WiFi热力图", session_id="s-gap-2",
                                        tool_registry={}):
                if isinstance(ch, dict) and ch.get("type") == "capability_gap":
                    saw_gap["hit"] = True
                    saw_gap["detail"] = ch.get("detail")
            self.assertTrue(saw_gap["hit"], "expected a capability_gap stream event")
            self.assertIn("热力图", saw_gap["detail"] or "")
        asyncio.run(t())


class TestCapabilityGapJournal(unittest.TestCase):
    def test_journal_records_gap(self):
        from runtime.skill_journal import SkillJournal
        j = SkillJournal(session_id="s-gap-3", query="q")
        j.record_capability_gap(turn=1, detail="缺少 AD 管理工具", query="q")
        d = j.to_dict()
        self.assertEqual(d["capability_gaps"], ["缺少 AD 管理工具"])


class TestCapabilityGapLongChain(unittest.TestCase):
    """S7 — long-chain (request needs n tools, system has k<n).

    Structural PRE-FLIGHT detection (plan-level coverage, schema dep-check) is
    P2-1 (B/A, not yet built). C is the runtime SAFETY NET: after doing the
    coverable prefix, the LLM declares the gap for the uncoverable step instead
    of silently dropping it or faking completion. This pins that net.
    """
    def test_partial_chain_does_prefix_then_declares_gap(self):
        async def t():
            executed = []

            async def get_user_access(args):
                executed.append("get_user_access")
                return '{"user": "alice", "admission": "ADMITTED"}'

            # chain: step1 query access (HAVE tool) → step2 reset AD pw (NO tool)
            turns = iter([
                '先确认准入。\n[TOOL:get_user_access] {"user_id": "alice"}',
                ("准入已确认 ADMITTED(第一步完成)。第二步需要重置 alice 的 AD 域控密码,"
                 "但本 agent 没有 AD 管理工具,也没有可委派的对端。\n"
                 "[CAPABILITY_GAP: 重置 AD 域控密码 — 缺少 AD 管理工具]"),
            ])

            async def llm(query, context, state):
                return next(turns, "无法继续,缺少所需能力,已说明缺口。")

            loop = AgentRuntimeLoop(llm_fn=llm, config=RuntimeConfig())
            res = await loop.run(
                query="确认 alice 准入,然后重置她的 AD 密码",
                session_id="s7-chain", tool_registry={"get_user_access": get_user_access})
            # coverable prefix WAS executed (not abandoned wholesale)
            self.assertIn("get_user_access", executed)
            # the uncoverable step surfaced as a structured gap, not faked
            self.assertTrue(any("缺少能力" in u for u in res.unresolved),
                            f"long-chain gap must be recorded: {res.unresolved}")
            # partial success is reported honestly
            self.assertIn("ADMITTED", res.final_response)
            self.assertNotIn("CAPABILITY_GAP", res.final_response)
        asyncio.run(t())


if __name__ == "__main__":
    unittest.main(verbosity=2)
