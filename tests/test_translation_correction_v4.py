"""Regression probes, not a Skill generalization benchmark or model score."""

from dataclasses import replace
from pathlib import Path
import json

import pytest

from evaluation.public_skill_fixture_mcp import FixtureCapability
from evaluation.public_skill_translation_v2 import (
    ParameterEvidence, ReadProposal, TranslationEnvelope, _prompt, bind_parameters, link_catalog,
)
from evaluation.translation_case_authoring import (
    NotApplicableSlot, ParameterDefinition, authoring_output_schema,
    materialize_tool_catalog, validate_anchored_bundle, validate_translation_tool_catalog,
)
from tests.test_public_skill_translation_v2 import _catalog
from tests.test_translation_case_authoring import _read_bundle_with_anchor


def _evidence(prompt: str, value: str, name: str = "resource_id") -> ParameterEvidence:
    start = prompt.index(value)
    return ParameterEvidence(
        name=name, value=value, source_text=value, start=start, end=start + len(value),
    )


@pytest.mark.parametrize("prompt,value,failure", [
    ("resource_id=router-a; resource_id=router-b.", "router-a", "parameter_conflicting"),
    ("resource_id=router-a. Owner is user-b.", "user-b", "parameter_evidence_not_owned"),
    ("Owner is router-a. resource_id=router-a.", "router-a", "parameter_evidence_not_owned"),
    ("resource_id=<target>; resource_id=router-a.", "router-a", "parameter_invalid"),
    ('resource_id="unterminated; resource_id=router-a.', "router-a", "parameter_invalid"),
    ("Owner is router-a.", "router-a", "parameter_unbound"),
])
def test_real_literal_is_not_field_ownership(prompt: str, value: str, failure: str) -> None:
    evidence = _evidence(prompt, value)
    proposal = TranslationEnvelope(decision=ReadProposal(
        kind="read_proposal", operation_intent="inspect", parameter_evidence=(evidence,),
        confidence=1.0, explanation="A deliberately untrusted model proposal.",
    ))
    link = link_catalog(proposal, _catalog(), prompt)
    assert link.status == "unlinked"
    assert link.route != "l0_read"
    assert f"{failure}:resource_id" in link.failures


@pytest.mark.parametrize("separator", ["=", ": ", " is ", " "])
def test_named_scalar_ownership_is_supported(separator: str) -> None:
    prompt = f"Inspect resource_id{separator}router-a."
    values, sources, failures = bind_parameters(prompt, _catalog()[0], (_evidence(prompt, "router-a"),))
    assert values == {"resource_id": "router-a"}
    assert sources["resource_id"]["method"] == "model_span"
    assert not failures


def test_duplicate_same_value_is_not_a_conflict() -> None:
    values, _, failures = bind_parameters("resource_id=a; resource_id=a.", _catalog()[0], ())
    assert values == {"resource_id": "a"}
    assert not failures


def test_invalid_typed_assignment_cannot_hide_behind_valid_assignment() -> None:
    values, _, failures = bind_parameters(
        "resource_id=a desired_state=ready expected_revision=oops; expected_revision=1.",
        _catalog()[1], (),
    )
    assert "expected_revision" not in values
    assert "parameter_invalid:expected_revision" in failures


def _static_capability() -> FixtureCapability:
    return FixtureCapability(
        "health.read", "health_read", "Observe health", "read_only",
        {"type": "object", "additionalProperties": False, "required": [], "properties": {}},
        {"kind": "static", "resultKey": "health"},
    )


def test_zero_argument_and_optional_argument_binding() -> None:
    primary = _static_capability()
    assert bind_parameters("Inspect health.", primary, ()) == ({}, {}, [])
    optional = replace(primary, input_schema={
        **primary.input_schema, "properties": {"region": {"type": "string"}},
    })
    assert bind_parameters("Inspect health.", optional, ()) == ({}, {}, [])
    assert bind_parameters("Inspect region=west.", optional, ())[0] == {"region": "west"}
    assert bind_parameters("region=west; region=east.", optional, ())[2] == ["parameter_conflicting:region"]


def test_translator_prompt_is_invariant_to_scoring_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = "evaluation.public_skill_translation_v2"
    monkeypatch.setattr(f"{module}.inspect_skill_package", lambda _: {"gate": "passed", "packageDigest": "fixed"})
    monkeypatch.setattr(f"{module}.build_skill_disclosure_packet", lambda _: {"files": []})
    (tmp_path / "SKILL.md").write_text("Inspect health without changing state.")
    case = {
        "caseId": "nominal-001", "challenge": "nominal", "language": "en",
        "userPrompt": "Inspect health.", "runtimePackageDigest": "fixed", "packageDigest": "source",
    }
    catalog = {"apiVersion": "test", "assignmentId": "nominal-assignment", "capabilities": []}
    before = _prompt(case, catalog, tmp_path)
    after = _prompt(
        {**case, "caseId": "adversarial-999", "challenge": "failure_or_adversarial", "language": "zh", "expected_behavior": "reject"},
        {**catalog, "assignmentId": "private-gold-group"}, tmp_path,
    )
    assert before == after
    payload = json.loads(before[0])
    assert set(payload["case"]) == {"userPrompt"}
    assert set(payload["toolCatalog"]) == {"apiVersion", "capabilities"}


def _no_required_bundle(*, optional: bool = False, write: bool = False):
    source = "Read health without arguments. A reset operation changes the service without arguments."
    bundle = _read_bundle_with_anchor(source)
    parameters = (ParameterDefinition(
        name="region", value_type="string", description="Optional regional filter",
        example_value="west", required=False,
    ),) if optional else ()
    operation = bundle.operation.model_copy(update={
        "parameters": parameters, "mode": "write" if write else "read",
        "effect_semantics": "irreversible" if write else "none",
    })
    nominal = bundle.tasks[0].model_copy(update={
        "user_prompt": "Reset the service." if write else "Inspect health.",
        "expected_behavior": "l0_write_candidate" if write else "l0_read_candidate",
        "approval_required": write, "max_effect_calls": int(write), "risk": "high" if write else "low",
    })
    return source, bundle.model_copy(update={
        "operation": operation, "tasks": (nominal, bundle.tasks[2]),
        "not_applicable_slots": (NotApplicableSlot(
            slot_id=f"{bundle.assignment_id}-ambiguous", reason="no_required_parameters",
        ),),
    })


@pytest.mark.parametrize("optional,write", [(False, False), (True, False), (False, True)])
def test_author_construct_no_longer_invents_required_inputs(optional: bool, write: bool) -> None:
    source, bundle = _no_required_bundle(optional=optional, write=write)
    result = validate_anchored_bundle(
        {"files": [{"path": "SKILL.md", "content": source}]}, bundle.assignment_id, bundle,
    )
    assert result["passed"], result["failures"]
    assert result["notApplicableSlots"][0]["reason"] == "no_required_parameters"
    assert result["transactionClosureEstablished"] is False
    catalog = result["catalog"]
    capabilities = validate_translation_tool_catalog(catalog)
    assert len(capabilities) == 1
    assert capabilities[0]["inputSchema"]["required"] == []
    assert capabilities[0]["phase"] == ("effect" if write else "observe")


def test_optional_absence_does_not_justify_clarification() -> None:
    source, bundle = _no_required_bundle(optional=True)
    old = _read_bundle_with_anchor(source)
    bundle = bundle.model_copy(update={"tasks": (bundle.tasks[0], old.tasks[1], bundle.tasks[1]), "not_applicable_slots": ()})
    result = validate_anchored_bundle({"files": [{"path": "SKILL.md", "content": source}]}, bundle.assignment_id, bundle)
    assert not result["passed"]
    assert "not_applicable_slot_declaration_missing" in result["failures"]
    assert any("ambiguous_missing_parameter_shape_failed" in failure for failure in result["failures"])


def test_missing_required_parameter_cannot_be_declared_not_applicable() -> None:
    source, bundle = _no_required_bundle(optional=True)
    operation = bundle.operation.model_copy(update={
        "parameters": (bundle.operation.parameters[0].model_copy(update={"required": True}),),
    })
    bundle = bundle.model_copy(update={"operation": operation})
    result = validate_anchored_bundle({"files": [{"path": "SKILL.md", "content": source}]}, bundle.assignment_id, bundle)
    assert "not_applicable_slot_unjustified" in result["failures"]
    assert any("nominal_parameter_closure_failed" in failure for failure in result["failures"])


def test_current_schema_requires_explicit_requiredness() -> None:
    schema = authoring_output_schema()
    assert "required" in schema["$defs"]["ParameterDefinition"]["required"]
    assert "not_applicable_slots" in schema["required"]
    source, bundle = _no_required_bundle()
    catalog = materialize_tool_catalog(bundle.assignment_id, bundle.operation)
    assert catalog["capabilities"][0]["inputSchema"]["properties"] == {}
