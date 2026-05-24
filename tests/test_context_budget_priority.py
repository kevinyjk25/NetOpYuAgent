"""
tests/test_context_budget_priority.py
======================================

Tests the Tier 2 #3 wiring: cfg.context_budget.strategy = "priority" routes
AgentRuntimeLoop context assembly through the v2 TokenBudget instead of the
legacy ContextBudgetManager.assemble path.

Verifies:
  - _assemble_priority produces a non-empty context containing the rendered
    sections (reusing the legacy formatters → identical section text).
  - Under a tight total_chars cap, the P0 confirmed-facts section survives
    while low-priority sections (env) are trimmed first.
  - The default strategy stays "legacy" (no behaviour change unless opted in).

Pure runtime + asyncio; no pydantic / httpx / LLM.
"""
import unittest


class TestPriorityAssembly(unittest.TestCase):
    def _loop(self):
        from runtime.loop import AgentRuntimeLoop
        return AgentRuntimeLoop(memory_router=None)

    def test_default_strategy_is_legacy(self):
        loop = self._loop()
        # Unless config opts into "priority", stay legacy.
        self.assertIn(loop._ctx_strategy, ("legacy", "priority"))
        # In the sandbox config it should be the default legacy.
        self.assertEqual(loop._ctx_strategy, "legacy")

    def test_assemble_priority_produces_sections(self):
        loop = self._loop()
        ctx = loop._assemble_priority(
            skill_section="## SKILLS\nsome skills",
            memory_results=None,
            tool_outputs={"dc_bgp_evpn_status": "spine-1: 3 neighbors Established"},
            confirmed_facts=["spine-1 is a fabric spine"],
            working_set=None,
            env_context={"site": "dc-east"},
        )
        self.assertTrue(ctx)
        # Skill prefix + the fact + the tool output should all be present.
        self.assertIn("skills", ctx.lower())
        self.assertIn("spine-1", ctx)

    def test_p0_facts_survive_tight_budget(self):
        loop = self._loop()
        # Force a tiny budget so trimming must occur.
        class _CB:
            strategy = "priority"
            total_chars = 200
            section_system_core = 4000
            section_user_profile = 500
            section_recent_turns = 20000
            section_tool_results = 30000
            section_retrieved_memory = 10000
            section_skills = 5000
            section_older_summary = 5000
        loop._ctx_budget_cfg = _CB()
        facts = ["CRITICAL: spine-1 ASN is 65000"]
        big_env = {"site": "x" * 5000}   # huge low-priority section
        ctx = loop._assemble_priority(
            skill_section="",
            memory_results=None,
            tool_outputs=None,
            confirmed_facts=facts,
            working_set=None,
            env_context=big_env,
        )
        # The P0 confirmed fact must survive; the giant P3 env must be trimmed
        # so the whole thing fits the 200-char cap (plus section formatting).
        self.assertIn("65000", ctx)
        self.assertLess(len(ctx), 1500)   # env was trimmed, not included whole


if __name__ == "__main__":
    unittest.main(verbosity=2)
