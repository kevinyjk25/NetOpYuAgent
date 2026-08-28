from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from network_runtime.contracts import PlanIntegrityError
from network_runtime.l0.cli import main as l0_cli
from network_runtime.l0.expressions import ExpressionError, render_template
from network_runtime.l0.models import CompiledAtomicEffect
from network_runtime.l0.production import BINDINGS, CATALOG, PRODUCTION_DEFINITIONS
from network_runtime.l0.runtime_loader import RESOLVERS, validate_runtime_projection
from network_runtime.l0_skills import REGISTRY, compile_intent
from network_runtime.policies import reviewed_contracts


def _value(name: str, parameter) -> object:
    known = {
        "environment": "staging", "to_transport": "lte",
        "config_lines": ["interface Ethernet1/1"], "changes": {"timeout": 3},
        "force": False, "dry_run": False, "vlan_id": 100,
        "expected_revision": 1, "grace_period_s": 60,
    }
    if name in known:
        return known[name]
    if parameter.enum:
        return parameter.enum[0]
    if parameter.type == "integer":
        return int(max(parameter.minimum or 0, 1))
    if parameter.type == "boolean":
        return False
    if parameter.type == "array":
        return ["value"]
    if parameter.type == "object":
        return {"value": True}
    return {
        "reason": "approved change", "config_text": "ntp server 10.0.0.8",
        "user_id": "alice", "app_id": "crm", "role": "reader",
        "service": "crm", "device_id": "ap-01", "node": "leaf-1",
        "node_id": "node-1", "deploy_id": "deploy-1", "resource_id": "db-1",
        "target": "db-2", "tunnel": "tun-sf-dc", "interface": "eth2",
        "version": "1.2.3", "change_id": "CHG-100", "correlation_id": "corr-1",
        "scope": "service", "section": "ntp", "verification_probe_id": "probe-1",
    }.get(name, "value")


class ProductionL0V2Tests(unittest.TestCase):
    def test_every_reviewed_write_is_activated_as_one_compiled_v2_contract(self) -> None:
        reviewed = reviewed_contracts()
        runtime_contracts = REGISTRY.contracts()
        compiled = CATALOG.contracts()
        self.assertEqual(len(reviewed), 21)
        self.assertEqual(len(runtime_contracts), 21)
        self.assertEqual(len(compiled), 21)
        self.assertEqual(
            {item.tool_name for item in runtime_contracts}, set(reviewed),
        )
        self.assertEqual(
            {item.tool_name for item in PRODUCTION_DEFINITIONS}, set(reviewed),
        )
        self.assertTrue(all(item.schema_version == 2 for item in runtime_contracts))
        self.assertTrue(all(isinstance(item, CompiledAtomicEffect) for item in compiled))

    def test_all_v2_bindings_match_parameters_intent_effect_and_legacy_adapters(self) -> None:
        reviewed = reviewed_contracts()
        for runtime_contract in REGISTRY.contracts():
            with self.subTest(skill=runtime_contract.skill_id):
                runtime_contract.verify_integrity()
                compiled = runtime_contract.compiled_contract
                binding = BINDINGS[(runtime_contract.skill_id, runtime_contract.version)]
                tool_contract = reviewed[runtime_contract.tool_name]
                self.assertEqual(binding.tool_contract_id, tool_contract.contract_id)
                self.assertEqual(binding.verifier_id, tool_contract.verifier)
                self.assertEqual(binding.compensator_id, tool_contract.compensator)
                arguments = {
                    name: _value(name, parameter)
                    for name, parameter in compiled.spec.parameters.items()
                }
                provenance = {name: "user_explicit" for name in arguments}
                targets = tuple(
                    f"{name}:{arguments[name]}" for name in runtime_contract.target_fields
                )
                intent = compile_intent(
                    runtime_contract,
                    profile=runtime_contract.allowed_profiles[0],
                    tool_name=runtime_contract.tool_name,
                    arguments=arguments,
                    provenance=provenance,
                    targets=targets,
                )
                parity = validate_runtime_projection(
                    compiled=compiled,
                    tool_name=runtime_contract.tool_name,
                    tool_contract_id=tool_contract.contract_id,
                    verifier_id=tool_contract.verifier,
                    compensator_id=tool_contract.compensator,
                    profile=runtime_contract.allowed_profiles[0],
                    arguments=arguments,
                    intent=intent.to_dict(),
                )
                self.assertTrue(parity.ok, parity.errors)
                self.assertEqual(parity.effect_arguments, arguments)

    def test_runtime_catalog_cli_reports_complete_v2_authority(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            status = l0_cli(["runtime-validate"])
        self.assertEqual(status, 0)
        self.assertIn('"contracts": 21', output.getvalue())
        self.assertIn('"readable_trajectories": 21', output.getvalue())
        self.assertIn('"promotion_ready": 21', output.getvalue())
        self.assertIn('"exact_round_trips": 21', output.getvalue())
        self.assertIn('"runtime_authority": "l0-v2-compiled"', output.getvalue())

    def test_expression_engine_is_typed_and_non_executable(self) -> None:
        self.assertEqual(
            render_template("${arguments.vlan_id}", {"arguments": {"vlan_id": 100}}),
            100,
        )
        with self.assertRaisesRegex(ExpressionError, "unsupported"):
            render_template("${__import__('os').system('id')}", {"arguments": {}})
        with self.assertRaisesRegex(ExpressionError, "unresolved"):
            render_template("${arguments.missing}", {"arguments": {}})

    def test_unknown_resolver_fails_closed(self) -> None:
        with self.assertRaisesRegex(PlanIntegrityError, "unregistered"):
            RESOLVERS.resolve("model.guess", "alice", {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
