from copy import deepcopy
import pytest

from evaluation.hybrid_artifact_lowering import lower
from evaluation.hybrid_artifact_checks import inspect
from evaluation.hybrid_draft_review import build_review_input
from tests.test_hybrid_draft_review import inputs


def fixture(column="OccurredAt", start="2028-01-02T01:00:00Z", end="2028-01-02T02:00:00Z"):
    raw = inputs()
    raw["candidate"]["draft"] = f"# Draft\n\n```kql\nEvents\n| where {column} >= datetime'{start}' and {column} < datetime'{end}'\n| take 7\n```\n"
    raw["observations"]["n0"]["observations"] = {"read": f"Window [{start},{end})."}
    return raw


@pytest.mark.parametrize("column", ["OccurredAt", "RecordedOn", "event_utc"])
def test_typed_render_preserves_values_operators_columns_notes_and_source(column):
    raw = fixture(column)
    original = deepcopy(raw)
    payload = build_review_input(raw)
    candidate, report = lower(payload, raw["candidate"]["values"], remaining_changes=4)
    assert raw == original and len(report["edits"]) == 2 and report["remainingChangeBudgetAfter"] == 2
    assert candidate["notes"] == raw["candidate"]["notes"] and candidate["values"] == raw["candidate"]["values"]
    for row in report["edits"]:
        assert row["after"] == "datetime(" + row["typedValue"]["value"] + ")"
    checks = inspect(build_review_input({**raw, "candidate": candidate}))["checks"]
    assert next(r for r in checks if r["check"] == "half_open_time_filter")["status"] == "pass"
    assert not report["fullQueryValidityProven"] and not report["queryExecuted"]


def test_over_budget_and_values_drift_atomic_rejection():
    raw = fixture()
    payload = build_review_input(raw)
    with pytest.raises(ValueError, match="remaining shared"):
        lower(payload, raw["candidate"]["values"], remaining_changes=1)
    with pytest.raises(ValueError, match="binding drift"):
        lower(payload, {}, remaining_changes=8)


@pytest.mark.parametrize("variant", ["or", "string", "python", "unbound", "invalid_date", "observed_code"])
def test_unsupported_or_readonly_context_is_not_rewritten(variant):
    raw = fixture()
    if variant == "or":
        raw["candidate"]["draft"] = raw["candidate"]["draft"].replace(" and ", " or ")
    elif variant == "string":
        raw["candidate"]["draft"] = '```kql\nT\n| where message == "t >= datetime\'2028-01-02T01:00:00Z\'"\n```'
    elif variant == "python":
        raw["candidate"]["draft"] = raw["candidate"]["draft"].replace("```kql", "```python")
    elif variant == "unbound":
        raw["observations"]["n0"]["observations"] = {"read": "No timestamp here."}
    elif variant == "invalid_date":
        raw = fixture(start="2028-99-99T01:00:00Z", end="2028-99-99T02:00:00Z")
    else:
        raw["observations"]["n0"]["observations"] = {"read": raw["candidate"]["draft"]}
    candidate, report = lower(build_review_input(raw), raw["candidate"]["values"], remaining_changes=8)
    assert candidate == raw["candidate"] and report["edits"] == []
