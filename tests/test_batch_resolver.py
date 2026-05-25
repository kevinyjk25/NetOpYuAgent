"""
tests/test_batch_resolver.py
============================

Locks in L0/L1 Stage B: the multi-target batch HITL decision is injected
business logic, not baked into the L0 loop.

  - L0 contract: with NO batch_resolver_fn injected (default profile), the loop
    raises a single-target HITL (batch_calls is None). Verified via the
    _handle_tools path in test_handle_tools_phase already; here we assert the
    loop stores the injected fn and the network resolver behaves correctly.
  - Network resolver (L1): Path A (multiple same-name destructive [TOOL:] →
    deduped batch) and Path B (one [TOOL:] but prose names 2+ devices →
    fabricated siblings). Default/non-network → resolver is None.
"""
import unittest


class TestProfileResolverSelection(unittest.TestCase):
    def test_default_profile_has_no_resolver(self):
        from profiles import get_batch_resolver_for_profile
        self.assertIsNone(get_batch_resolver_for_profile("default"))

    def test_network_profiles_get_resolver(self):
        from profiles import get_batch_resolver_for_profile
        self.assertIsNotNone(get_batch_resolver_for_profile("lan"))
        self.assertIsNotNone(get_batch_resolver_for_profile("dc"))

    def test_loop_stores_injected_resolver(self):
        from runtime.loop import AgentRuntimeLoop
        marker = lambda **kw: None
        loop = AgentRuntimeLoop(memory_router=None, batch_resolver_fn=marker)
        self.assertIs(loop._batch_resolver_fn, marker)
        # default: None
        loop2 = AgentRuntimeLoop(memory_router=None)
        self.assertIsNone(loop2._batch_resolver_fn)


class TestNetworkBatchResolver(unittest.TestCase):
    def _resolve(self, **kw):
        from profiles.network_batch_resolver import resolve_network_batch
        return resolve_network_batch(**kw)

    def test_path_a_multiple_same_name_calls(self):
        # LLM emitted two edit_device_config calls in one turn → batch of 2.
        hitl = frozenset({"edit_device_config"})
        a1 = {"device_id": "sw-core-01", "config_lines": ["vlan 10"]}
        a2 = {"device_id": "sw-core-02", "config_lines": ["vlan 10"]}
        out = self._resolve(
            tool_name="edit_device_config", tool_args=a1,
            llm_response="apply to both",
            hitl_tool_names=hitl, confirmed_facts=[],
            all_parsed=[("edit_device_config", a1), ("edit_device_config", a2)],
        )
        self.assertIsNotNone(out)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0][1]["device_id"], "sw-core-01")
        self.assertEqual(out[1][1]["device_id"], "sw-core-02")

    def test_path_b_prose_fabrication(self):
        # ONE call but prose names two devices → fabricate the sibling.
        hitl = frozenset({"edit_device_config"})
        a1 = {"device_id": "ap-01", "config_lines": ["radius 1.1.1.1"], "reason": "fix"}
        out = self._resolve(
            tool_name="edit_device_config", tool_args=a1,
            llm_response="我将为 ap-01 和 ap-02 下发修复配置 [TOOL:edit_device_config]",
            hitl_tool_names=hitl, confirmed_facts=[],
            all_parsed=[("edit_device_config", a1)],
        )
        self.assertIsNotNone(out)
        self.assertEqual(len(out), 2)
        devs = {c[1]["device_id"] for c in out}
        self.assertEqual(devs, {"ap-01", "ap-02"})
        # fabricated call is marked auto-derived
        fab = [c for c in out if c[1]["device_id"] == "ap-02"][0]
        self.assertIn("auto-derived", fab[1]["reason"])

    def test_no_batch_when_single_device(self):
        hitl = frozenset({"edit_device_config"})
        a1 = {"device_id": "ap-01", "config_lines": ["x"]}
        out = self._resolve(
            tool_name="edit_device_config", tool_args=a1,
            llm_response="just fix ap-01 [TOOL:edit_device_config]",
            hitl_tool_names=hitl, confirmed_facts=[],
            all_parsed=[("edit_device_config", a1)],
        )
        self.assertIsNone(out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
