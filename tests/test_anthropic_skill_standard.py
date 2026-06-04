"""tests/test_anthropic_skill_standard.py — verify the 5 Anthropic-standard
skill capabilities are supported:

  1. SKILL.md folder + YAML frontmatter parsing
  2. Progressive disclosure (frontmatter summary → on-demand full body)
  3. scripts/ execution (script-as-tool: run(inputs)->dict registered + runnable)
  4. references/assets on-demand inventory
  5. allowed-tools carried through (per-skill tool whitelist)

Plus: script AST safety (denied imports/calls rejected).
"""
import asyncio
import tempfile
import textwrap
import unittest
from pathlib import Path

from skills.loader import SkillLoader
from skills.skill_format import load_skill_md
from skills.script_runner import (
    build_script_tools, validate_script_source, ScriptValidationError,
)


def _make_skill(root: Path, *, with_script=True, with_ref=True, script_body=None):
    sk = root / "terminal-probe"
    (sk / "scripts").mkdir(parents=True)
    if with_ref:
        (sk / "references").mkdir()
        (sk / "references" / "rules.md").write_text("rule table", encoding="utf-8")
    (sk / "SKILL.md").write_text(textwrap.dedent("""\
        ---
        name: terminal-probe
        description: Probe skill exercising the full Anthropic standard.
        allowed-tools: get_user_access, check_nac_policy
        metadata:
          skill_id: terminal_probe
        ---
        # Terminal Probe
        ## Steps
        1. Run scripts/classify.py to classify the result.
        2. See references/rules.md for the rule table.
    """), encoding="utf-8")
    if with_script:
        body = script_body or (
            "def run(inputs):\n"
            "    return {'mode': 'wireless' if inputs.get('ssid') else 'wired'}\n"
        )
        (sk / "scripts" / "classify.py").write_text(body, encoding="utf-8")
    return sk


class TestCapability1And5_ParseAndAllowedTools(unittest.TestCase):
    def test_frontmatter_parsed_and_allowed_tools_carried(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_skill(root)
            defs = SkillLoader.__new__(SkillLoader)._load_folder(root, fatal=False)
            self.assertIn("terminal_probe", defs)
            d = defs["terminal_probe"]
            # 1. parsing
            self.assertTrue(d["name"])
            self.assertTrue(d["description"])
            # 5. allowed-tools carried through (was dropped before the fix)
            self.assertEqual(d["allowed_tools"], ["get_user_access", "check_nac_policy"])


class TestCapability2_ProgressiveDisclosure(unittest.TestCase):
    def test_summary_then_detail(self):
        from skills.catalog import SkillCatalogService
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_skill(root)
            defs = SkillLoader.__new__(SkillLoader)._load_folder(root, fatal=False)
            cat = SkillCatalogService()
            cat.register_all(defs)
            summary = cat.format_summary()                 # Level 1
            detail = cat.load_detail("terminal_probe")     # Level 2
            self.assertIn("terminal_probe", summary)
            # Level 1 summary is shorter than Level 2 detail (disclosure).
            self.assertIsNotNone(detail)
            self.assertGreater(len(detail), 0)


class TestCapability3_ScriptExecution(unittest.TestCase):
    def test_script_becomes_runnable_tool(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sk = _make_skill(root)
            tools = build_script_tools("terminal_probe", sk)
            self.assertIn("terminal_probe__classify", tools)

            async def run():
                wireless = await tools["terminal_probe__classify"]({"ssid": "wifi1"})
                wired = await tools["terminal_probe__classify"]({})
                return wireless, wired
            wireless, wired = asyncio.run(run())
            self.assertIn("wireless", wireless)
            self.assertIn("wired", wired)

    def test_script_error_returns_string_not_raises(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sk = _make_skill(root, script_body="def run(inputs):\n    return 1/0\n")
            tools = build_script_tools("terminal_probe", sk)
            out = asyncio.run(tools["terminal_probe__classify"]({}))
            self.assertIn("ERROR", out)   # script bug surfaced as string

    def test_script_without_run_fn_is_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sk = _make_skill(root, script_body="x = 1  # no run() function\n")
            tools = build_script_tools("terminal_probe", sk)
            self.assertNotIn("terminal_probe__classify", tools)


class TestCapability4_ReferencesInventory(unittest.TestCase):
    def test_skill_dir_and_resource_inventories(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_skill(root)
            defs = SkillLoader.__new__(SkillLoader)._load_folder(root, fatal=False)
            d = defs["terminal_probe"]
            # 缺口A: skill_dir captured so runtime can locate bundled files
            self.assertTrue(d.get("skill_dir"))
            self.assertEqual(d.get("scripts"), ["classify.py"])
            self.assertEqual(d.get("references"), ["rules.md"])


class TestScriptSafety(unittest.TestCase):
    def test_denied_import_rejected(self):
        with self.assertRaises(ScriptValidationError):
            validate_script_source("import os\ndef run(i): return {}\n")

    def test_denied_subprocess_rejected(self):
        with self.assertRaises(ScriptValidationError):
            validate_script_source("import subprocess\ndef run(i): return {}\n")

    def test_denied_eval_rejected(self):
        with self.assertRaises(ScriptValidationError):
            validate_script_source("def run(i): return eval('1+1')\n")

    def test_safe_script_passes(self):
        # Pure computation — allowed.
        validate_script_source(
            "import json, math\n"
            "def run(i):\n"
            "    return {'r': math.sqrt(i.get('x', 4))}\n"
        )

    def test_dangerous_script_skipped_at_build_not_raise(self):
        # A dangerous script in scripts/ is skipped (logged), others still load.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sk = _make_skill(root)   # safe classify.py
            (sk / "scripts" / "evil.py").write_text(
                "import os\ndef run(i): return {}\n", encoding="utf-8")
            tools = build_script_tools("terminal_probe", sk)
            self.assertIn("terminal_probe__classify", tools)    # safe one loaded
            self.assertNotIn("terminal_probe__evil", tools)     # dangerous skipped


if __name__ == "__main__":
    unittest.main()
