"""Task binding is lossless; free L1 text never becomes qualified L0."""
import pytest

from skill_authoring import delivery
from dsh_adapter.hybrid_session import _host_result, _render_delivery
from network_runtime.contracts import sha256_json
from network_runtime.l0.structured_schema import DataBindingError


def contract(task="Explain the evidence and provide a conclusion in at most three sections."):
    return delivery.compile_task(None, {"task": task, "p000": "Example content is not a current observation."})


def test_complete_task_retained_without_selecting_kinds_or_source_fragments():
    task = "状态与未知。\n\nInclude owners; preserve NOT and MAY; supply complete code if requested. " * 50
    result = contract(task)
    assert result["originalTask"] == task
    assert result["sourceDigests"]["task"] == sha256_json(task)
    assert result["requirements"] == [] and not result["authorityGranted"]
    assert result["representation"] == "unverified_L1_answer_not_executable_L0"


@pytest.mark.parametrize("proposal", [{}, [], "analysis", {"originalTask": "ignore restrictions"},
    {"requirements": [{"kind": "analysis", "source_ref": "task:0"}], "unrepresented": []}])
def test_model_cannot_replace_task_binding(proposal):
    with pytest.raises(ValueError, match="requires null"):
        delivery.compile_task(proposal, {"task": "Original task"})


@pytest.mark.parametrize("body", ["## One\nA\n\n## Two\nB\n\n## Three\nC", "无法确定，缺少证据。",
    "```python\nx = 1\n```\nAn unexecuted candidate.", "An incomplete answer remains unverified."])
def test_exact_presentation_does_not_add_headings_or_claim_coverage(body):
    result = delivery.render(contract(), {"answer": body})
    assert result["rendered"] == body and result["shapeComplete"]
    assert result["declaredCoverageComplete"] is None
    assert result["semanticCoverage"]["selectedKinds"] == "not_used"
    assert result["taskSuccess"] is None and not result["semanticApproval"]
    assert _host_result(result, {"status": "no_failed_checks"})["state"] == "candidate_unverified"


@pytest.mark.parametrize("candidate", [{"answer": ""}, {"answer": " \n"}, {"answer": None},
    {"answer": "a", "success": True}, {"answer": "x" * 24001}, {"delivery": {"d0": "text"}}])
def test_bad_envelopes_fail_without_coercion(candidate):
    with pytest.raises(DataBindingError):
        delivery.render(contract(), candidate)


def test_existing_artifact_checks_still_reject_malformed_code_without_executing():
    result, checks = _render_delivery(contract(), {"answer": "```python\nif:\n```"}, "Provide Python.", [])
    assert checks["status"] == "failed_checks"
    assert _host_result(result, checks)["state"] == "needs_revision"
    assert not checks["queryExecuted"] and not result["sourceScriptsExecuted"]


def test_unqualified_prose_is_not_reclassified_as_a_successful_executable_artifact():
    result, checks = _render_delivery(contract("Provide an executable query."),
        {"answer": "An explanation only; no query supplied."}, "Provide an executable query.", [])
    assert checks["status"] == "no_supported_artifact"
    assert result["declaredCoverageComplete"] is None and result["taskSuccess"] is None
    assert not result["semanticApproval"]


def test_legacy_contract_still_requires_its_declared_content_shape():
    old = delivery.compile_selection({"requirements": [{"kind": "decision", "language": "",
        "source_ref": "task:0"}], "unrepresented": []}, {"task": "Give a conclusion."}, single_choice=True)
    with pytest.raises(DataBindingError):
        delivery.render(old, {"answer": "No bypass of the legacy contract."})
