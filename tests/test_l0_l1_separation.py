"""
tests/test_l0_l1_separation.py
==============================

Locks in Stage A of the L0/L1 separation (2026-05): the L0 runtime framework
must carry no hardcoded business (network-ops) nouns; business specifics are
injected by the active profile / config (L1).

Covers:
  - ResourceRef is the neutral working-set ref; DeviceRef is a back-compat
    alias; default type is the domain-free "resource".
  - RuntimeConfig.editable_hitl_tools defaults EMPTY in L0 (no edit_device_config
    etc. baked in) and is injectable.
"""
import unittest


class TestResourceRef(unittest.TestCase):
    def test_resourceref_is_neutral(self):
        from runtime.context_budget import ResourceRef
        r = ResourceRef(id="x1", label="X One")
        self.assertEqual(r.type, "resource")        # no domain assumption
        self.assertEqual(str(r), "X One (x1)")

    def test_deviceref_is_backcompat_alias(self):
        from runtime.context_budget import ResourceRef, DeviceRef
        self.assertIs(DeviceRef, ResourceRef)
        # old 3-positional / id+label call still constructs
        d = DeviceRef(id="ap-01", label="AP 1")
        self.assertEqual(d.id, "ap-01")

    def test_runtime_reexports_both(self):
        from runtime import ResourceRef, DeviceRef
        self.assertIs(ResourceRef, DeviceRef)


class TestEditableHitlInjection(unittest.TestCase):
    def test_l0_default_is_empty(self):
        from runtime.loop_types import RuntimeConfig
        # No business tool names baked into the L0 default.
        self.assertEqual(RuntimeConfig().editable_hitl_tools, {})

    def test_injectable(self):
        from runtime.loop_types import RuntimeConfig
        rc = RuntimeConfig(editable_hitl_tools={"edit_device_config": ["config_lines"]})
        self.assertEqual(rc.editable_hitl_tools["edit_device_config"], ["config_lines"])

    def test_no_business_names_in_l0_default_factory(self):
        # The dataclass field default must not reference any concrete tool name.
        from runtime.loop_types import RuntimeConfig
        import dataclasses
        f = {f.name: f for f in dataclasses.fields(RuntimeConfig)}["editable_hitl_tools"]
        produced = f.default_factory()   # type: ignore
        self.assertEqual(produced, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
