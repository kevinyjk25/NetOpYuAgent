"""
tests/test_delegate_directive.py
================================

Unit tests for the [DELEGATE:...] directive parser (Phase 2B).
Pure parser tests — no httpx / fastapi, run in sandbox + CI.
"""
import unittest

from runtime.directive_parser import (
    find_delegate_directives,
    has_delegate_directive,
    strip_delegate_directives,
    strip_all_directives,
)


class TestDelegateParser(unittest.TestCase):
    def test_explicit_agent_id(self):
        d = find_delegate_directives(
            "[DELEGATE:dc-agent] check BGP on spine-1")[0]
        self.assertFalse(d.by_capability)
        self.assertEqual(d.agent_id, "dc-agent")
        self.assertEqual(d.capability, "")
        self.assertFalse(d.forked)
        self.assertEqual(d.task, "check BGP on spine-1")

    def test_forked_modifier(self):
        d = find_delegate_directives(
            "[DELEGATE:dc-agent#forked] correlate facts")[0]
        self.assertTrue(d.forked)
        self.assertEqual(d.agent_id, "dc-agent")
        self.assertEqual(d.task, "correlate facts")

    def test_explicit_fresh_modifier(self):
        d = find_delegate_directives(
            "[DELEGATE:dc-agent#fresh] do x")[0]
        self.assertFalse(d.forked)

    def test_by_capability(self):
        d = find_delegate_directives(
            "[DELEGATE:*dc_fabric_diagnose] trace path to leaf-3")[0]
        self.assertTrue(d.by_capability)
        self.assertEqual(d.capability, "dc_fabric_diagnose")
        self.assertEqual(d.agent_id, "")
        self.assertEqual(d.task, "trace path to leaf-3")

    def test_whitespace_tolerance(self):
        d = find_delegate_directives(
            "[ DELEGATE : dc-agent ] spaced out")[0]
        self.assertEqual(d.agent_id, "dc-agent")
        self.assertEqual(d.task, "spaced out")

    def test_hyphenated_agent_id(self):
        d = find_delegate_directives("[DELEGATE:k8s-edge-agent] x")[0]
        self.assertEqual(d.agent_id, "k8s-edge-agent")

    def test_no_directive(self):
        self.assertEqual(find_delegate_directives("just prose"), [])
        self.assertFalse(has_delegate_directive("just prose"))
        self.assertFalse(has_delegate_directive(""))

    def test_has_delegate(self):
        self.assertTrue(has_delegate_directive("[DELEGATE:dc-agent] x"))

    def test_multiple_directives_source_order(self):
        text = "[DELEGATE:a-agent] first\n[DELEGATE:b-agent] second"
        ds = find_delegate_directives(text)
        self.assertEqual([d.agent_id for d in ds], ["a-agent", "b-agent"])

    def test_strip_delegate(self):
        out = strip_delegate_directives("pre [DELEGATE:dc-agent] task\nkeep")
        self.assertNotIn("DELEGATE", out)
        self.assertIn("keep", out)
        self.assertIn("pre", out)

    def test_strip_all_includes_delegate(self):
        text = "[TOOL:foo] {} and [DELEGATE:dc-agent] bar"
        out = strip_all_directives(text)
        self.assertNotIn("DELEGATE", out)
        self.assertNotIn("TOOL", out)

    def test_empty_task_allowed(self):
        # A delegate with no task text parses (runtime decides if that's valid).
        d = find_delegate_directives("[DELEGATE:dc-agent]")[0]
        self.assertEqual(d.task, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
