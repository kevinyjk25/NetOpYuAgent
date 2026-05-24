"""
tests/test_skill_evolve_suggest_only.py
========================================

Regression for the auto_evolve_apply master switch (Item 1, 2026-05):

  SkillEvolver(apply_changes=False) must run the self-improvement loop —
  call the LLM, compute the patch / new skill, record it in version history
  and logs — but MUST NOT mutate the live catalog (no _update_catalog_from_markdown,
  no _catalog.register_all). This lets an operator observe suggestions in
  production before trusting auto-application.

  apply_changes=True (the default) keeps the existing behaviour: apply the
  patch to the catalog (still gated by the A/B compliance bench elsewhere).

These tests use a mock catalog + stub LLM — no pydantic, no real LLM, run in
CI's lightweight env.
"""
import asyncio
import unittest


def _make_evolver(apply_changes: bool):
    from skills.evolver import SkillEvolver

    class _MockCatalog:
        def __init__(self):
            self.details = {"sk-1": "# Skill 1\nOriginal content."}
            self.registered = []          # records register_all calls
            self.updated = []             # records _update_catalog_from_markdown effects

        def load_detail(self, skill_id):
            return self.details.get(skill_id)

        def register_all(self, mapping):
            self.registered.append(dict(mapping))

    cat = _MockCatalog()

    async def _stub_llm(system, user):
        # Return a well-formed feedback patch JSON.
        return (
            '{"updated_content": "# Skill 1\\nIMPROVED content.", '
            '"changes": ["clarified step 2"], "quality_delta": 0.1}'
        )

    ev = SkillEvolver(
        catalog=cat,
        llm_fn=_stub_llm,
        apply_changes=apply_changes,
    )
    # Capture catalog-mutation calls by patching the private updater.
    async def _track_update(skill_id, content):
        cat.updated.append((skill_id, content))
    ev._update_catalog_from_markdown = _track_update  # type: ignore
    return ev, cat


class TestSuggestOnlyMode(unittest.TestCase):
    def test_apply_changes_false_does_not_mutate_catalog(self):
        async def run():
            ev, cat = _make_evolver(apply_changes=False)
            result = await ev.apply_feedback("sk-1", "step 2 is unclear", success=True)
            # The patch WAS computed + a version recorded ...
            self.assertIsNotNone(result)
            self.assertEqual(result.new_version, 1)
            self.assertIn("sk-1", ev._versions)
            self.assertEqual(len(ev._versions["sk-1"]), 1)
            # ... but the live catalog was NOT touched.
            self.assertEqual(cat.updated, [])
        asyncio.run(run())

    def test_apply_changes_true_mutates_catalog(self):
        async def run():
            ev, cat = _make_evolver(apply_changes=True)
            result = await ev.apply_feedback("sk-1", "step 2 is unclear", success=True)
            self.assertIsNotNone(result)
            # Default mode applies the patch to the catalog.
            self.assertEqual(len(cat.updated), 1)
            self.assertEqual(cat.updated[0][0], "sk-1")
            self.assertIn("IMPROVED", cat.updated[0][1])
        asyncio.run(run())

    def test_default_is_apply(self):
        # Constructing without the flag keeps the historical behaviour.
        from skills.evolver import SkillEvolver
        ev = SkillEvolver(catalog=object(), llm_fn=None)
        self.assertTrue(ev._apply_changes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
