from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from network_runtime.l0 import L0Catalog
from network_runtime.l0.compiler import L0CompileError, compile_documents, load_documents, parse_document
from network_runtime.l0.models import CompiledAtomicEffect, CompiledCompositeEffect
from network_runtime.l0_skills import L0SkillContract, L0SkillRegistry


EXAMPLES = Path(__file__).resolve().parents[1] / "network_runtime" / "l0" / "examples"


class L0V2CompilerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.documents = load_documents(EXAMPLES)
        cls.catalog = L0Catalog(compile_documents(cls.documents))

    def test_example_pack_covers_base_constraint_extension_and_composite(self) -> None:
        contracts = self.catalog.contracts()
        self.assertEqual(len(contracts), 5)
        self.assertEqual(
            {getattr(item, "derivation", "composite") for item in contracts},
            {"base", "constraint", "extension", "composite"},
        )
        self.assertEqual(
            len(self.catalog.for_capability("rest.url1.network-access.grant")), 3,
            "one REST capability must support S1 and multiple semantic S11 contracts",
        )

    def test_legacy_compatible_registry_supports_versions_and_semantic_ambiguity(self) -> None:
        registry = L0SkillRegistry()
        for skill_id, version in (
            ("network.access.grant", "1.0.0"),
            ("network.access.grant", "1.1.0"),
            ("network.guest-access.grant", "1.0.0"),
        ):
            registry.register(L0SkillContract.create(
                skill_id=skill_id,
                version=version,
                tool_name="url1_grant_network_access",
                tool_contract_id="rest-access-grant-v1",
                intent_kind=skill_id.replace(".", "_"),
                target_fields=("user_id",),
                allowed_profiles=("lan",),
                compensatable=True,
            ))
        self.assertEqual(registry.get("network.access.grant").version, "1.1.0")
        self.assertEqual(len(registry.candidates_for_tool("lan", "url1_grant_network_access")), 3)
        self.assertIsNone(
            registry.for_tool("lan", "url1_grant_network_access"),
            "tool name alone must not guess between semantic S1/S11 entrypoints",
        )
        selected = registry.for_tool(
            "lan", "url1_grant_network_access",
            skill_id="network.access.grant", version="1.0.0",
        )
        self.assertEqual(selected.version, "1.0.0")

    def test_constraint_derivation_only_narrows_and_is_fully_flattened(self) -> None:
        parent = self.catalog.require("network.access.grant", "1.0.0")
        child = self.catalog.require("network.guest-access.grant", "1.0.0")
        self.assertIsInstance(parent, CompiledAtomicEffect)
        self.assertIsInstance(child, CompiledAtomicEffect)
        self.assertEqual(child.derivation, "constraint")
        self.assertEqual(child.spec.effect, parent.spec.effect)
        self.assertEqual(child.spec.parameters["vlan_id"].fixed, 300)
        self.assertEqual(child.spec.parameters["duration_minutes"].maximum, 480)
        self.assertEqual(parent.spec.parameters["duration_minutes"].maximum, 10080)
        self.assertEqual(child.spec.approval.risk, "high")
        self.assertEqual(child.lineage[0].id, parent.metadata.id)
        self.assertNotEqual(child.contract_hash, parent.contract_hash)

    def test_extension_adds_inputs_reads_predicates_and_stronger_approval(self) -> None:
        child = self.catalog.require("network.privileged-access.grant", "1.0.0")
        self.assertIsInstance(child, CompiledAtomicEffect)
        self.assertEqual(child.derivation, "extension")
        self.assertIn("change_id", child.spec.parameters)
        self.assertIn("sponsor_id", child.spec.parameters)
        self.assertEqual(len(child.spec.preflight), 3)
        self.assertEqual(len(child.spec.verification.predicates), 4)
        self.assertEqual(child.spec.intent.desired_state["privileged"], True)
        self.assertEqual((child.spec.approval.risk, child.spec.approval.mode), ("critical", "dual"))

    def test_composite_binds_exact_child_versions_hashes_and_projects_to_saga(self) -> None:
        composite = self.catalog.require("employee.application-access.provision", "1.0.0")
        self.assertIsInstance(composite, CompiledCompositeEffect)
        self.assertEqual([item.id for item in composite.steps], ["network-access", "application-access"])
        self.assertTrue(all(item.contract_hash.startswith("sha256:") for item in composite.steps))
        saga = self.catalog.to_saga_definition(
            "employee.application-access.provision", "1.0.0",
        )
        self.assertEqual(saga.steps[1].depends_on, ("network-access",))
        self.assertEqual(
            saga.steps[0].compensation_capability_id, "rest.network-access.restore",
        )
        self.assertTrue(saga.definition_hash.startswith("sha256:"))

    def test_compilation_is_deterministic_and_latest_version_is_resolved(self) -> None:
        first = L0Catalog(compile_documents(self.documents))
        second = L0Catalog(compile_documents(self.documents))
        self.assertEqual(first.to_json(), second.to_json())

        source = yaml.safe_load((EXAMPLES / "s1-network-access-grant.yaml").read_text())
        source["metadata"]["version"] = "1.1.0"
        extended_documents = [*self.documents, parse_document(source, source="s1@1.1.0")]
        catalog = L0Catalog(compile_documents(extended_documents))
        self.assertEqual(catalog.require("network.access.grant").metadata.version, "1.1.0")
        self.assertEqual(
            len(catalog.for_capability("rest.url1.network-access.grant")), 4,
        )

    def _documents_with_mutation(self, file_name: str, mutation) -> list:
        values = []
        for path in sorted(EXAMPLES.glob("*.yaml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            if path.name == file_name:
                mutation(raw)
            values.append(parse_document(raw, source=path.name))
        return values

    def test_derived_contract_cannot_weaken_approval(self) -> None:
        documents = self._documents_with_mutation(
            "s11-guest-access-constraint.yaml",
            lambda raw: raw["spec"].update({
                "approval": {"required": False, "risk": "low", "mode": "single"},
            }),
        )
        with self.assertRaisesRegex(L0CompileError, "approval may only increase"):
            compile_documents(documents)

    def test_derived_contract_cannot_expand_parameter_bounds(self) -> None:
        documents = self._documents_with_mutation(
            "s11-guest-access-constraint.yaml",
            lambda raw: raw["spec"]["constrainParameters"]["duration_minutes"].update({
                "maximum": 20000,
            }),
        )
        with self.assertRaisesRegex(L0CompileError, "maximum cannot be higher"):
            compile_documents(documents)

    def test_derived_contract_cannot_overwrite_parent_desired_state(self) -> None:
        documents = self._documents_with_mutation(
            "s11-privileged-access-extension.yaml",
            lambda raw: raw["spec"]["desiredStateAdditions"].update({"allowed": False}),
        )
        with self.assertRaisesRegex(L0CompileError, "cannot overwrite"):
            compile_documents(documents)

    def test_composite_rejects_missing_child_argument(self) -> None:
        documents = self._documents_with_mutation(
            "s111-network-and-application-composite.yaml",
            lambda raw: raw["spec"]["steps"][1]["arguments"].pop("change_id"),
        )
        with self.assertRaisesRegex(L0CompileError, "misses required child arguments"):
            compile_documents(documents)

    def test_atomic_effect_requires_independent_observation(self) -> None:
        raw = yaml.safe_load((EXAMPLES / "s1-network-access-grant.yaml").read_text())
        raw["spec"]["preflight"] = []
        with self.assertRaisesRegex(L0CompileError, "independent preflight"):
            parse_document(raw, source="write-only-url1")

    def test_atomic_template_rejects_unknown_argument_reference(self) -> None:
        raw = yaml.safe_load((EXAMPLES / "s1-network-access-grant.yaml").read_text())
        raw["spec"]["effect"]["request"]["user_id"] = "${arguments.usr_typo}"
        with self.assertRaisesRegex(L0CompileError, "unknown argument"):
            compile_documents([parse_document(raw, source="typo")])

    def test_constraint_fixed_value_must_satisfy_narrowed_bounds(self) -> None:
        documents = self._documents_with_mutation(
            "s11-guest-access-constraint.yaml",
            lambda raw: raw["spec"]["constrainParameters"]["duration_minutes"].update({
                "fixed": 600,
                "maximum": 480,
            }),
        )
        with self.assertRaisesRegex(L0CompileError, "fixed parameter value is above maximum"):
            compile_documents(documents)

    def test_composite_input_contract_must_be_subset_of_child_parameter(self) -> None:
        documents = self._documents_with_mutation(
            "s111-network-and-application-composite.yaml",
            lambda raw: raw["spec"]["inputs"]["vlan_id"].update({"maximum": 5000}),
        )
        with self.assertRaisesRegex(L0CompileError, "cannot satisfy"):
            compile_documents(documents)

    def test_composite_embedded_input_reference_is_checked(self) -> None:
        documents = self._documents_with_mutation(
            "s111-network-and-application-composite.yaml",
            lambda raw: raw["spec"]["steps"][0]["arguments"].update({
                "reason": "${input.missing_change}: employee application access",
            }),
        )
        with self.assertRaisesRegex(L0CompileError, "unknown input"):
            compile_documents(documents)

    def test_explain_and_graph_make_compiled_behavior_visible(self) -> None:
        explanation = self.catalog.explain("network.privileged-access.grant")
        self.assertIn("Preflight: rest.network-access.get", explanation)
        self.assertIn("Approval: critical/dual", explanation)
        graph = self.catalog.graph("employee.application-access.provision")
        self.assertIn("flowchart TD", graph)
        self.assertIn("network-access --> application-access", graph)

    def test_semantic_diff_exposes_constraint_without_reading_compiled_json(self) -> None:
        difference = self.catalog.diff(
            "network.access.grant", "network.guest-access.grant",
        )
        self.assertEqual(difference["relationship"], "constraint")
        self.assertTrue(difference["same_effect_capability"])
        self.assertEqual(
            difference["parameters"]["changed"]["vlan_id"]["to"]["fixed"], 300,
        )
        self.assertEqual(difference["approval"]["to"]["risk"], "high")


if __name__ == "__main__":
    unittest.main(verbosity=2)
