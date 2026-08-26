"""S8 — skill-ambiguity → HITL choice card (cross-module, was a blind spot).

When the catalog finds multiple comparably-scored skills, the loop must
surface a `stop_hitl / reason=skill_ambiguity / hitl_kind=user_choice` card
listing the candidates + a "use none" option — UNLESS resolution is
suppressed (non_interactive run, or a prior choice already resolved).

Catalog SCORING is covered by test_capability_index/test_skill_preference;
this pins the GATE BEHAVIOR (does ambiguity actually produce the card, and
does suppression actually suppress it) with a stub catalog that forces
ambiguous=True, so the test doesn't depend on scoring tuning.
"""
import asyncio
import unittest

from runtime.loop import AgentRuntimeLoop
from runtime.loop_types import RuntimeConfig


class _AmbiguousSel:
    """Mimics SkillCatalogService.select_skills_for_query's return."""
    def __init__(self):
        self.ambiguous = True
        self.ambiguous_kind = "weak"
        self.top_score = 0.12
        self.second_score = 0.10
        self.selected = [("lan_user_access_diagnose", 0.12),
                         ("app_access_troubleshoot", 0.10)]
        self.summary = ("[RELEVANT SKILLS — top matches]\n"
                        "  lan_user_access_diagnose [low]\n"
                        "  app_access_troubleshoot [low]\n")


class _StubSummary:
    def __init__(self, sid):
        self.name = sid
        self.purpose = f"purpose of {sid}"
        self.risk_level = "low"
        self.tags = ["access"]
        self.requires_hitl = False


class _StubCatalog:
    """Minimal catalog that always reports an ambiguous match."""
    def select_skills_for_query(self, query, top_k=5):
        return _AmbiguousSel()

    def get_summary(self, sid):
        return _StubSummary(sid)


def _loop(llm_fn):
    return AgentRuntimeLoop(
        memory_router=None, config=RuntimeConfig(),
        skill_catalog=_StubCatalog(), llm_fn=llm_fn)


class TestSkillAmbiguityGate(unittest.TestCase):
    def test_ambiguity_emits_choice_card_when_interactive(self):
        async def t():
            async def llm(query, context, state):
                return "我来处理这个用户访问诊断请求,先确认准入状态。"
            loop = _loop(llm)
            card = None
            async for ch in loop.stream(
                query="诊断用户 alice 访问应用失败", session_id="s8-ambig",
                env_context={},          # interactive: choice NOT pre-resolved
                tool_registry={}):
                if isinstance(ch, dict) and ch.get("reason") == "skill_ambiguity":
                    card = ch
                    break
            self.assertIsNotNone(card, "ambiguity must emit a choice card")
            self.assertTrue(card.get("stop_hitl"))
            self.assertEqual(card.get("hitl_kind"), "user_choice")
            # candidates + a "none" escape option present
            ids = [c["id"] for c in card.get("choices", [])]
            self.assertIn("__none__", ids)
            self.assertIn("lan_user_access_diagnose", ids)
            # preference-learning metadata carried for the resolution handler
            self.assertIn("_pref_meta", card)
            self.assertEqual(set(card["_pref_meta"]["candidates"]),
                             {"lan_user_access_diagnose", "app_access_troubleshoot"})
        asyncio.run(t())

    def test_non_interactive_suppresses_choice_card(self):
        """run()'s non_interactive default sets _skill_choice_resolved → the
        LLM proceeds without a card (an API caller can't click one)."""
        async def t():
            async def llm(query, context, state):
                return "按最佳判断处理用户访问诊断,不加载特定技能,直接分析准入链路。"
            loop = _loop(llm)
            saw_card = False
            async for ch in loop.stream(
                query="诊断用户 alice 访问应用失败", session_id="s8-noninter",
                env_context={"_skill_choice_resolved": True},   # suppressed
                tool_registry={}):
                if isinstance(ch, dict) and ch.get("reason") == "skill_ambiguity":
                    saw_card = True
            self.assertFalse(saw_card,
                             "non-interactive must suppress the ambiguity card")
        asyncio.run(t())


if __name__ == "__main__":
    unittest.main(verbosity=2)
