import pytest

from evaluation.hybrid_artifact_checks import inspect
from skill_authoring.artifact_checks import inspect_candidate
from evaluation.hybrid_draft_review import build_review_input
from tests.test_hybrid_draft_review import inputs


def checked(draft, observation="Window [2028-01-02T01:00:00Z, 2028-01-02T02:00:00Z)."):
    raw = inputs()
    raw["candidate"]["draft"] = draft
    raw["observations"]["n0"]["observations"] = {"read": observation}
    return inspect(build_review_input(raw))


def code(text, language="kql"):
    return f"```{language}\n{text}\n```"


@pytest.mark.parametrize("operator,status", [("<", "partial_checks_only"), ("<=", "failed_checks")])
def test_shared_candidate_api_binds_actual_evidence_not_a_semantic_score(operator, status):
    candidate = {"draft": "Draft only.\n" + code(
        f"Events\n| where t >= datetime(2028-01-02T01:00:00Z) and t {operator} datetime(2028-01-02T02:00:00Z)"),
        "uncertainties": [], "remaining_actions": []}
    result = inspect_candidate(candidate, "Draft a query", [{"text": "Window [2028-01-02T01:00:00Z,2028-01-02T02:00:00Z)."}])
    assert result["status"] == status and not result["semanticApproval"] and not result["queryExecuted"]
    assert any(r["check"] == "kql_complete_language" and r["status"] == "unverified" for r in result["inspection"]["checks"])


def test_shared_candidate_cannot_create_its_own_time_constraint():
    draft = "Window [2028-01-02T01:00:00Z,2028-01-02T02:00:00Z).\n" + code(
        "Events\n| where t >= datetime(2028-01-02T01:00:00Z) and t < datetime(2028-01-02T02:00:00Z)")
    result = inspect_candidate({"draft": draft}, "Draft from real observations only", [])
    assert next(r for r in result["inspection"]["checks"] if r["check"] == "half_open_time_filter")["status"] == "unverified"


def test_no_supported_artifact_never_becomes_approved_answer():
    result = inspect_candidate({"draft": "Pia approved it (unsupported assertion)."}, "Explain approval status", [])
    assert result["status"] == "no_supported_artifact" and not result["semanticApproval"]


def test_unfenced_code_and_unsupported_code_are_not_executed():
    for draft in ["import os; os.abort()", code("import os; os.abort()", "python"), code("exit 1", "bash")]:
        result = inspect_candidate({"draft": draft}, "Inspect", [])
        assert not result["semanticApproval"] and result["inspection"]["sourceScriptsExecuted"] is False


@pytest.mark.parametrize("name", ["EventTime", "x", "timestamp_utc"])
@pytest.mark.parametrize("reversed_order", [False, True])
def test_interval_rules_generalize_identifier_and_order_without_execution(name, reversed_order):
    terms = [f"{name} >= datetime(2028-01-02T01:00:00Z)", f"{name} < datetime(2028-01-02T02:00:00Z)"]
    if reversed_order:
        terms.reverse()
    result = checked(code("Events\n| where " + " and ".join(terms)))
    assert next(r for r in result["checks"] if r["check"] == "half_open_time_filter")["status"] == "pass"
    assert any(r["status"] == "unverified" for r in result["checks"])
    assert not result["completeAnswerApproved"] and not result["sourceScriptsExecuted"]


@pytest.mark.parametrize("upper,operation", [("2028-01-02T02:00:00Z", "<="), ("2028-01-02T03:00:00Z", "<")])
def test_closed_endpoint_and_changed_time_fail(upper, operation):
    result = checked(code(f"T\n| where t >= datetime(2028-01-02T01:00:00Z) and t {operation} datetime({upper})"))
    assert next(r for r in result["checks"] if r["check"] == "half_open_time_filter")["status"] == "fail"


@pytest.mark.parametrize("predicate", [
    "t between (datetime'2028-01-02T01:00:00Z') and (datetime'2028-01-02T02:00:00Z')",
    "t between (1) and (2)",
])
def test_malformed_between_is_local_failure(predicate):
    result = checked(code("T\n| where " + predicate))
    assert any(r["check"] == "kql_between_shape" and r["status"] == "fail" for r in result["checks"])


@pytest.mark.parametrize("predicate", [
    "t >= datetime(2028-01-02T01:00:00Z) or t < datetime(2028-01-02T02:00:00Z)",
    "t >= start and t < end",
])
def test_unsupported_intervals_never_pass_or_get_executed(predicate):
    result = checked(code("T\n| where " + predicate))
    assert next(r for r in result["checks"] if r["check"] == "half_open_time_filter")["status"] == "unverified"


def test_candidate_interval_is_not_promoted_to_source_constraint():
    result = checked("[2028-01-02T01:00:00Z, 2028-01-02T02:00:00Z)\n\n" + code(
        "T\n| where t >= datetime(2028-01-02T01:00:00Z) and t < datetime(2028-01-02T02:00:00Z)"), "No interval supplied.")
    assert next(r for r in result["checks"] if r["check"] == "half_open_time_filter")["status"] == "unverified"


@pytest.mark.parametrize("language,text,status", [
    ("json", '{"x": 1}', "pass"), ("json", '{"x": 1, "x": 2}', "fail"),
    ("json", '{"x": NaN}', "fail"), ("python", "x = (", "fail"),
    ("python", "raise RuntimeError('MUST NEVER EXECUTE')", "pass"),
    ("bash", "touch /tmp/MUST_NEVER_EXECUTE", "unverified"),
])
def test_inert_parsing(language, text, status):
    rows = checked(code(text, language))["checks"]
    assert rows[-1]["status"] == status


@pytest.mark.parametrize("ratio,status", [("100/2000 = 5%", "pass"), ("7/100=7%", "pass"),
                                         ("10/2000=5%", "fail"), ("0/0=0%", "fail")])
def test_arithmetic_is_exact_and_narrow(ratio, status):
    assert checked(ratio)["checks"][0]["status"] == status


def test_unclosed_fence_is_not_parsed():
    rows = checked("```python\nraise RuntimeError()")['checks']
    assert len(rows) == 1 and rows[0]["check"] == "fence_closed" and rows[0]["status"] == "fail"


@pytest.mark.parametrize("expression", [
    "t between (datetime(2028-01-02T01:00:00Z) .. datetime(2028-01-02T02:00:00Z)) and ok == true",
    'message contains "where t between (one) and (two)"',
])
def test_valid_or_unsupported_forms_not_falsely_failed(expression):
    assert not any(r["status"] == "fail" for r in checked(code("T\n| where " + expression))["checks"])


@pytest.mark.parametrize("predicate", [
    "t >= datetime'2028-01-02T01:00:00Z' and t < datetime'2028-01-02T02:00:00Z'",
    "t between (datetime'2028-01-02T01:00:00Z') and (datetime'2028-01-02T02:00:00Z')",
])
def test_malformed_datetime_literal_is_rejected_independently(predicate):
    assert any(r["check"] == "kql_datetime_literal" and r["status"] == "fail" for r in checked(code("T\n| where " + predicate))["checks"])


def test_literal_in_string_or_comment_does_not_become_code_issue():
    rows = checked(code('T\n| where message == "datetime\'2028-01-02T01:00:00Z\'"\n// t >= datetime\'2028-01-02T01:00:00Z\''))["checks"]
    assert not any(r["check"] == "kql_datetime_literal" for r in rows)
