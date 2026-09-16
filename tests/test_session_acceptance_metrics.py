"""Evaluator validity must not turn engineering completion into task success."""
from copy import deepcopy

import pytest

from evaluation.hybrid_session_acceptance import audit_expectations, completion_dimensions


def aligned_spec():
    return {"task": "Read the current inventory and report the difference.",
            "expectations": [{"id": "c1", "critical": True, "statement": "Report the difference."}],
            "expectationAudit": {"c1": {"status": "aligned", "taskQuotes": ["report the difference"],
                                         "reason": "Explicit request; values come from observations."}}}


def test_quotes_are_provenance_not_automatic_entailment():
    spec = aligned_spec()
    assert audit_expectations(spec)["calibrated"]
    assert not audit_expectations(spec)["entailmentAutomaticallyProven"]
    assert not audit_expectations({"task": spec["task"]})["calibrated"]
    spec["expectationAudit"]["c1"]["status"] = "ambiguous"
    assert not audit_expectations(spec)["calibrated"]


@pytest.mark.parametrize("fault", ["missing", "duplicate", "false_quote", "empty", "status"])
def test_incomplete_or_invented_preflight_anchors_rejected(fault):
    spec = aligned_spec()
    if fault == "missing":
        spec["expectationAudit"] = {}
    elif fault == "duplicate":
        spec["expectations"] *= 2
    elif fault == "false_quote":
        spec["expectationAudit"]["c1"]["taskQuotes"] = ["report exact loss percentages"]
    elif fault == "empty":
        spec["expectationAudit"]["c1"]["taskQuotes"] = [""]
    else:
        spec["expectationAudit"]["c1"]["status"] = "automatic_approval"
    with pytest.raises(ValueError):
        audit_expectations(spec)


@pytest.mark.parametrize("fault", [None, "authorization", "rejection", "output", "alignment", "criteria", "empty_read", "timeout", "multiple"])
def test_success_requires_authorized_admitted_faithful_and_aligned_delivery(fault):
    values = dict(process_exit=0, completed=True, requests=[{"tool": "read"}], authorized=[True],
                  prepared=1, retained=1, criteria={"c1": {"met": True}}, delivery_admitted=True,
                  faithful_terminal=True, alignment={"calibrated": True})
    original = deepcopy(values)
    if fault == "authorization":
        values["authorized"] = [False]
    elif fault == "rejection":
        values["delivery_admitted"] = False
    elif fault == "output":
        values["faithful_terminal"] = False
    elif fault == "alignment":
        values["alignment"] = {"calibrated": False}
    elif fault == "criteria":
        values["criteria"] = {"c1": {"met": False}}
    elif fault == "empty_read":
        values["requests"] = []
    elif fault == "timeout":
        values["process_exit"] = None
    elif fault == "multiple":
        values["prepared"] = 2
    result = completion_dimensions(**values)
    assert result["taskPassed"] is (fault is None)
    assert bool(result["blockers"]) is (fault is not None)
    if fault in {"authorization", "rejection", "output", "alignment"}:
        assert result["rawFixtureTaskPassed"]  # Previously missing conjunctions.
    assert completion_dimensions(**original)["taskPassed"]
