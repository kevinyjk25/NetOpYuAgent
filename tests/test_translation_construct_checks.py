from __future__ import annotations

import pytest

from evaluation.translation_construct_checks import inspect_task_parameters
from evaluation.translation_case_authoring import (
    normalize_author_candidate, validate_anchored_bundle,
)
from tests.test_translation_case_authoring import _read_bundle_with_anchor


def _parameters(value_type: str = "string") -> list[dict[str, str]]:
    return [{"name": "service_id", "value_type": value_type, "example_value": "svc-17"}]


@pytest.mark.parametrize("value", ["svc-99", '"Tampering"', "'two words'", "001"])
def test_typed_alternate_values_are_bound_not_missing(value: str) -> None:
    result = inspect_task_parameters(f"Inspect service_id={value}.", _parameters())
    assert result["boundParameters"] == ["service_id"]
    assert result["missingParameters"] == []
    evidence = result["evidence"][0]
    assert evidence["text"] == f"Inspect service_id={value}."[evidence["start"]:evidence["end"]]


@pytest.mark.parametrize("value_type,value", [
    ("integer", "42"), ("integer", "-1"), ("number", "1.25"),
    ("number", ".25"), ("boolean", "false"), ("boolean", "true"),
])
def test_scalar_types_are_checked(value_type: str, value: str) -> None:
    result = inspect_task_parameters(f"Inspect service_id={value}.", _parameters(value_type))
    assert result["boundParameters"] == ["service_id"]


@pytest.mark.parametrize("value_type,value", [
    ("integer", "42wrong"), ("integer", "1.5"), ("number", "NaN"),
    ("boolean", "yes"), ("boolean", "True"), ("string", '""'),
    ("string", '"unterminated'), ("string", ""),
])
def test_invalid_scalar_is_not_a_bound_parameter(value_type: str, value: str) -> None:
    result = inspect_task_parameters(f"Inspect service_id={value};", _parameters(value_type))
    assert result["boundParameters"] == []
    assert result["invalidParameters"] == ["service_id"]


@pytest.mark.parametrize("value", [
    '"$(openssl rand -hex 32)"', '"${TOKEN}"', "<GOAL_ID>", '"{{target}}"',
])
def test_deferred_expressions_stay_inert_and_unbound(value: str) -> None:
    result = inspect_task_parameters(f"service_id={value}.", _parameters())
    assert result["boundParameters"] == []
    assert result["unresolvedParameters"] == ["service_id"]


@pytest.mark.parametrize("prompt", [
    "service_id=Alpha; service_id=alpha.",
    "service id is 'my-runner-job'; service_id=my-runner.",
    "service_id=svc-17; service_id=svc-99.",
])
def test_conflicting_values_are_preserved_with_locations(prompt: str) -> None:
    result = inspect_task_parameters(prompt, _parameters())
    assert result["conflictingParameters"] == ["service_id"]
    assert result["boundParameters"] == []
    assert len(result["evidence"]) == 2
    assert all(prompt[item["start"]:item["end"]] == item["text"] for item in result["evidence"])


def test_repeated_equal_values_are_not_conflicts() -> None:
    result = inspect_task_parameters("service_id=svc-17; service_id=svc-17.", _parameters())
    assert result["boundParameters"] == ["service_id"]
    assert result["conflictingParameters"] == []


def test_other_identifiers_and_values_do_not_count_as_assignments() -> None:
    result = inspect_task_parameters("other_service_id=svc-17; svc-17", _parameters())
    assert result["missingParameters"] == ["service_id"]


def _validate_prompt(prompt: str, *, slot: int = 0, version: str = "v3") -> dict:
    source = "Inspect one service health record. Never mutate it."
    bundle = _read_bundle_with_anchor(source)
    tasks = list(bundle.tasks)
    tasks[slot] = tasks[slot].model_copy(update={"user_prompt": prompt})
    bundle = bundle.model_copy(update={"tasks": tuple(tasks)})
    normalized, events = normalize_author_candidate(bundle)
    assert normalized.tasks[slot].user_prompt == prompt
    assert not any("appended" in event for event in events)
    return validate_anchored_bundle(
        {"files": [{"path": "SKILL.md", "content": source}]},
        bundle.assignment_id, normalized, validation_version=version,
    )


def test_normalizer_does_not_fill_missing_or_override_prose() -> None:
    result = _validate_prompt("Inspect the service called 'svc-99'.")
    assert result["passed"] is False
    assert any("nominal_parameter_closure_failed" in item for item in result["failures"])


def test_alternate_value_cannot_fake_clarification() -> None:
    assert _validate_prompt("Inspect service_id=svc-99.")["passed"] is True
    result = _validate_prompt("Inspect service_id=svc-99.", slot=1)
    assert any("ambiguous_missing_parameter_shape_failed" in item for item in result["failures"])
    # Sealed v1/v2 artifacts keep their original rules for honest reproduction.
    assert _validate_prompt("Inspect service_id=svc-99.", slot=1, version="v2")["passed"] is True
    assert "constructChecks" not in _validate_prompt("Inspect service_id=svc-99.", version="v2")


def test_nominal_conflicts_reject_but_ambiguous_conflicts_are_visible() -> None:
    prompt = "Inspect service_id=svc-17; service_id=svc-99."
    result = _validate_prompt(prompt)
    assert any("conflicting_parameter_values" in item for item in result["failures"])
    assert _validate_prompt(prompt, slot=1)["passed"] is True


@pytest.mark.parametrize("prompt", [
    "Generate a read candidate using service_id=svc-17.",
    "Generate translation-development candidates using service_id=svc-17.",
    "Create l0_read_candidate for service_id=svc-17.",
])
def test_unrelated_evaluation_meta_task_is_rejected(prompt: str) -> None:
    result = _validate_prompt(prompt)
    assert any("unsupported_evaluation_meta_task" in item for item in result["failures"])


def test_legitimate_skill_word_and_unparsed_prose_do_not_imply_semantic_proof() -> None:
    result = _validate_prompt("Use the Skill to inspect service_id=svc-17.")
    assert result["passed"] is True
    checks = result["constructChecks"]
    assert checks["sourceApiSchemaVerified"] is False
    assert checks["semanticAlignmentProven"] is False
    assert checks["tasks"][0]["arbitraryProseConsistencyVerified"] is False
    # A bounded parser cannot certify this negation. Keep this limitation explicit.
    result = _validate_prompt("Do not inspect service_id=svc-17.")
    assert result["constructChecks"]["tasks"][0]["arbitraryProseConsistencyVerified"] is False


def test_unknown_validation_version_cannot_silently_skip_new_checks() -> None:
    with pytest.raises(ValueError, match="unsupported authoring validation version"):
        _validate_prompt("Inspect service_id=svc-17.", version="v99")
