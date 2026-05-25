"""
tests/test_hitl_submit_scope.py
===============================

Regression for the H2 async-HITL follow-up NameError (2026-05):

    _submit_hitl_decision: H2 follow-up failed ... name '_message_history'
    is not defined

`_submit_hitl_decision` is a MODULE-LEVEL function (it only receives `services`),
not a closure of `create_webui_app`, so it cannot reference that factory's local
`_message_history`. The fix publishes `_message_history` into `services` and has
the handler read `services["_message_history"]`.

This test guards the class of bug statically (no fastapi/httpx needed): the
function body must not reference `_message_history` (or other known
create_webui_app closure locals) as a free/global name — only via `services`.
"""
import ast
import unittest


def _func_node(name):
    tree = ast.parse(open("webui/backend.py").read())
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n
    raise AssertionError(f"{name} not found")


class TestSubmitHitlScope(unittest.TestCase):
    # Names that live in create_webui_app's closure and are NOT visible to the
    # module-level _submit_hitl_decision. Referencing them there = NameError.
    CLOSURE_LOCALS = {"_message_history"}

    def _free_names(self, fn):
        """Names that are loaded but never bound (assigned/param/comprehension)
        anywhere inside fn — i.e. resolved from an enclosing/global scope."""
        bound, loaded = set(), set()
        for a in fn.args.args + fn.args.kwonlyargs:
            bound.add(a.arg)
        if fn.args.vararg: bound.add(fn.args.vararg.arg)
        if fn.args.kwarg: bound.add(fn.args.kwarg.arg)
        for node in ast.walk(fn):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    bound.add(node.id)
                elif isinstance(node.ctx, ast.Load):
                    loaded.add(node.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                bound.add(node.name)
        return loaded - bound

    def test_submit_hitl_decision_no_closure_leak(self):
        fn = _func_node("_submit_hitl_decision")
        free = self._free_names(fn)
        leaked = self.CLOSURE_LOCALS & free
        self.assertEqual(
            leaked, set(),
            f"_submit_hitl_decision references create_webui_app closure local(s) "
            f"{leaked} — these are NameErrors at runtime. Read them via "
            f"services[...] instead.",
        )

    def test_message_history_published_to_services(self):
        # create_webui_app must publish _message_history into services so
        # module-level handlers can reach it.
        src = open("webui/backend.py").read()
        self.assertIn('services["_message_history"] = _message_history', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
