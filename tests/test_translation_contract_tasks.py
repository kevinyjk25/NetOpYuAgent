from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evaluation.translation_contract_tasks import (
    ContractReview, ContractTaskRequest, SourceDocument, TaskText,
    build_contract_task_plan, contract_review_input, task_author_input, validate_task_text,
)
from evaluation.translation_contract_task_runner import inspect_contract_tasks, run_contract_tasks
from evaluation.translation_source_alignment import ClaimAssessment, SourceAssessment


def _source(text: str, origin: str) -> SourceDocument:
    return SourceDocument(origin=origin, text=text, sha256="sha256:" + hashlib.sha256(text.encode()).hexdigest())


def _request(required: tuple[str, ...] = (), *, optional: bool = False, action: str = "read_only") -> ContractTaskRequest:
    properties = {name: {"type": "string"} for name in required}
    if optional:
        properties["region"] = {"type": "string"}
    tool = {"name": "health_snapshot", "description": "Observe service health without mutations.",
            "inputSchema": {"type": "object", "additionalProperties": False,
                            "properties": properties, "required": list(required)}}
    return ContractTaskRequest(
        skill_source=_source("Use health_snapshot to inspect service health. This fixture never writes.", "fixture://skill"),
        tool_source=_source(json.dumps(tool), "fixture://tool-schema"), action_type=action,
        fixture_values={name: f"value-{index}" for index, name in enumerate(required)},
    )


def _review(request: ContractTaskRequest, verdict: str = "supported") -> ContractReview:
    payload = contract_review_input(request)
    return ContractReview(reviewer_id="offline-test-fixture-not-an-independent-reviewer", reviewer_kind="test_fixture",
        assessment=SourceAssessment(input_digest=payload["inputDigest"], claims=tuple(
            ClaimAssessment(claim_id=claim["claimId"], verdict=verdict,
                source_span_ids=tuple("skill-0001" if kind == "skill" else "tool-0001" for kind in claim["requiredEvidenceKinds"]),
                rationale="Mechanical test fixture only; no semantic accuracy claim.",
                suggested_revision="Obtain independent source evidence before authoring.")
            for claim in payload["claims"]
        ), scope_note="One operation in an explicitly synthetic test fixture."))


@pytest.mark.parametrize("required,optional,count", [((), False, 1), ((), True, 1), (("device_id",), True, 2), (("device_id", "site_id"), False, 3)])
def test_applicability_is_derived_from_reviewed_requiredness(required: tuple, optional: bool, count: int) -> None:
    request = _request(required, optional=optional)
    plan = build_contract_task_plan(request, _review(request))
    assert plan["status"] == "ready_for_task_authoring"
    assert len(plan["slots"]) == count
    assert [slot["missingParameter"] for slot in plan["slots"]] == [None, *sorted(required)]
    assert bool(plan["notApplicable"]) == (not required)
    assert not plan["runtimeAuthorityGranted"]
    assert not plan["goldAuthored"]


def test_source_bytes_and_review_digest_cannot_drift() -> None:
    request = _request()
    review = _review(request)
    changed = request.model_copy(update={"action_type": "write"})
    with pytest.raises(ValueError, match="input digest"):
        build_contract_task_plan(changed, review)
    broken = request.model_copy(update={"tool_source": request.tool_source.model_copy(update={"text": "tampered"})})
    with pytest.raises(ValueError, match="source document digest"):
        contract_review_input(broken)


@pytest.mark.parametrize("verdict", ["contradicted", "insufficient_evidence"])
def test_source_disagreement_prevents_task_authoring(verdict: str) -> None:
    request = _request()
    plan = build_contract_task_plan(request, _review(request, verdict))
    assert plan["status"] == "blocked"
    assert plan["slots"] == []


def test_unknown_effect_semantics_cannot_be_approved_into_read_only() -> None:
    request = _request(action="unknown")
    plan = build_contract_task_plan(request, _review(request))
    assert plan["reasons"] == ["effect_semantics_unknown"]


def test_missing_or_circular_review_citation_is_rejected() -> None:
    request = _request()
    review = _review(request)
    claims = list(review.assessment.claims)
    claims[-1] = claims[-1].model_copy(update={"source_span_ids": ("skill-0001",)})
    changed = review.model_copy(update={"assessment": review.assessment.model_copy(update={"claims": tuple(claims)})})
    with pytest.raises(ValueError, match="required source evidence"):
        build_contract_task_plan(request, changed)


def test_structured_or_conditional_schemas_are_not_silently_flattened() -> None:
    request = _request()
    tool = json.loads(request.tool_source.text)
    tool["inputSchema"]["properties"] = {"filters": {"type": "object"}}
    request = request.model_copy(update={"tool_source": _source(json.dumps(tool), "fixture://nested-schema")})
    plan = build_contract_task_plan(request, _review(request))
    assert plan["status"] == "blocked"
    assert "unsupported_parameter_schema:filters" in plan["reasons"]


def test_optional_fixture_does_not_create_a_missing_required_task() -> None:
    request = _request(optional=True).model_copy(update={"fixture_values": {"region": "west"}})
    plan = build_contract_task_plan(request, _review(request))
    assert len(plan["slots"]) == 1
    assert plan["slots"][0]["fixtureValues"] == {"region": "west"}


def test_author_cannot_fill_omitted_parameters_or_change_values() -> None:
    request = _request(("device_id",))
    plan = build_contract_task_plan(request, _review(request))
    nominal, missing = plan["slots"]
    assert validate_task_text(plan, nominal["slotId"], TaskText(user_prompt="Inspect device_id=value-0."))["status"] == "needs_task_semantic_review"
    assert validate_task_text(plan, missing["slotId"], TaskText(user_prompt="Inspect device health."))["status"] == "needs_task_semantic_review"
    for slot, prompt in [(nominal, "Inspect device_id=other."), (missing, "Inspect device_id=value-0.")]:
        assert validate_task_text(plan, slot["slotId"], TaskText(user_prompt=prompt))["status"] == "rejected_text"


def test_text_protocol_cannot_change_slots_contract_or_answers() -> None:
    with pytest.raises(ValueError):
        TaskText.model_validate({"user_prompt": "Inspect health.", "expected_behavior": "read", "parameters": []})
    request = _request()
    plan = build_contract_task_plan(request, _review(request))
    payload = task_author_input(plan, plan["slots"][0]["slotId"])
    assert "intentHypothesis" not in payload
    assert "review" not in payload
    assert "slotId" not in payload
    assert "notApplicable" not in payload


class FakeWriter:
    def __init__(self) -> None:
        self.calls = 0

    def preflight(self) -> dict[str, str]:
        return {"model": "qwen3.5:9b", "modelArtifactDigest": "offline-fixture"}

    def author(self, payload: dict):
        self.calls += 1
        text = "Inspect health " + ", ".join(f"{name}={value}" for name, value in payload["namedInputLiterals"].items()) + "."
        return json.dumps({"user_prompt": text}), {"modelCalls": 1, "latencyMs": 1.0}


def test_runner_is_sealed_and_completed_reentry_never_calls_model(tmp_path: Path) -> None:
    request = _request(("device_id",))
    review = _review(request)
    writer = FakeWriter()
    root = tmp_path / "run"
    report = run_contract_tasks(request, review, root, adapter=writer)
    assert report["statusCounts"] == {"needs_task_semantic_review": 2}
    assert writer.calls == 2
    snapshot = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert run_contract_tasks(request, review, root, adapter=writer) == report
    assert writer.calls == 2
    assert snapshot == {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    path = next((root / "checkpoints").iterdir())
    path.write_text("{}")
    with pytest.raises(ValueError, match="sealed content drift"):
        inspect_contract_tasks(root)


def test_blocked_contract_makes_no_model_call(tmp_path: Path) -> None:
    request = _request(action="unknown")
    writer = FakeWriter()
    report = run_contract_tasks(request, _review(request), tmp_path / "blocked", adapter=writer)
    assert report["status"] == "blocked"
    assert report["modelCalls"] == 0
    assert writer.calls == 0


def test_partial_run_preserves_completed_checkpoint_without_automatic_replay(tmp_path: Path) -> None:
    class InterruptedWriter(FakeWriter):
        def author(self, payload: dict):
            if self.calls:
                raise KeyboardInterrupt()
            return super().author(payload)
    request = _request(("device_id",))
    writer = InterruptedWriter()
    root = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        run_contract_tasks(request, _review(request), root, adapter=writer)
    assert len(list((root / "checkpoints").iterdir())) == 1
    with pytest.raises(ValueError, match="incomplete task run"):
        run_contract_tasks(request, _review(request), root, adapter=writer)
    assert writer.calls == 1
