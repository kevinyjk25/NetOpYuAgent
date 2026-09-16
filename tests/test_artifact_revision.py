"""Mechanical repair ownership/limits, never evidence of LLM task accuracy."""
import copy
import json

import pytest

from dsh_adapter import hybrid_session as session
from skill_authoring import artifact_checks, artifact_repair, compiler, local_execution
from skill_authoring.kql_checks import column_checks, interval
from skill_authoring.artifacts import read_json
from network_runtime.l0.hybrid_execution import ReasoningReply
from tests import test_hybrid_session as fixtures

host = fixtures.host
isolated_host = fixtures.isolated_host

LOW, HIGH = "2029-03-02T08:00:00Z", "2029-03-02T09:00:00Z"
WINDOW = f"Window [{LOW},{HIGH})."
BAD = f"Events\n| where t between (datetime({LOW}) .. datetime({HIGH}))\n| summarize N=count() by who\n| project who, N"
GOOD = BAD.replace(f"t between (datetime({LOW}) .. datetime({HIGH}))", f"t >= datetime({LOW}) and t < datetime({HIGH})")
ANSWER = "## Draft\n```kql\n" + BAD + "\n```\nNot executed; observations may be incomplete.\n"


def test_interval_constraints_are_conjunctions_not_a_between_keyword_ban():
    windows = {(LOW, HIGH): []}
    assert interval(BAD, windows)[0] == "fail"
    assert interval(GOOD, windows)[0] == "pass"
    between = f"t between (datetime({LOW}) .. datetime({HIGH}))"
    for expr in [between + f" and t < datetime({HIGH})", between + f"\n| where t < datetime({HIGH})"]:
        assert interval("Events | where " + expr, windows)[0] == "pass"
    assert interval("Events | where " + between + " or exempt", windows)[0] == "unverified"
    assert interval("Events | where " + between + " | where t < upper", windows)[0] == "unverified"
    assert interval("Events | where " + between + f" | order by t | where t < datetime({HIGH})", windows)[0] == "unverified"
    assert interval("let x = Events; x | where " + between, windows)[0] == "unverified"
    assert interval(BAD, {(LOW, HIGH): [], (LOW, LOW): []})[0] == "unverified"
    assert interval(BAD.replace("t between", "other between"), {})[0] == "unverified"


@pytest.mark.parametrize("column", ["time_utc", "score", "sensorId"])
def test_column_lineage_is_generic_and_unknown_functions_are_not_assumed(column):
    prefix = "Telemetry | summarize Total=sum(amount) by owner"
    assert column_checks(prefix + f" | project owner, Total, {column}")[0][0] == "fail"
    assert column_checks(prefix + " | project owner, Total")[0][0] == "pass"
    assert column_checks(prefix + f" | invoke AnyFunction() | project owner, {column}")[0][0] == "unverified"
    assert column_checks("Telemetry | summarize any(*) by owner | project " + column)[0][0] == "unverified"
    assert column_checks(prefix + " | extend score=1 | project score")[0][0] == "unverified"


def test_inert_lexer_cannot_treat_quoted_pipelines_as_column_operations():
    assert not column_checks('Events | where message == "| summarize N=count() by who | project other"')
    assert not column_checks("Events // | summarize N=count() by who | project other")


def test_repair_owns_only_failed_code_bodies_preserves_prose_and_other_artifacts():
    text = ANSWER + '\n```json\n{"keep":true}\n```\n'
    checks = artifact_checks.inspect_candidate({"draft": text}, WINDOW, [])
    slots = artifact_repair.regions(text, checks)
    assert list(slots) == ["draft-1"]
    patched = artifact_repair.apply(text, checks, {"replacements": [{"location": "draft-1", "code": GOOD}]})
    assert patched["answer"] == text.replace(BAD, GOOD)
    for patch in [{"answer": "all done"}, {"replacements": []},
                  {"replacements": [{"location": "draft-3", "code": "{}"}]},
                  {"replacements": [{"location": "draft-1", "code": "\n"}]},
                  {"replacements": [{"location": "draft-1", "code": "```kql\n" + GOOD + "\n```"}]},
                  {"replacements": [{"location": "draft-1", "code": GOOD}] * 2}]:
        with pytest.raises(ValueError):
            artifact_repair.apply(text, checks, patch)


def pending(isolated_host, monkeypatch, *, native=False, enabled=True):
    root, profile, request, _, _, _ = isolated_host
    if enabled:
        profile["artifactRepair"] = True
    profile["resources"]["read_export"][1]["text"] = WINDOW
    (root / "host/profile.json").write_text(json.dumps(profile))
    invoked = []
    def generate(req, folder, costs):
        invoked.append(req)
        costs.append({"inputTokens": 10, "outputTokens": 10, "latencyMs": 1})
        return ReasoningReply({"answer": ANSWER}, compiler.MODEL, compiler.CONFIG_DIGEST, 10, 10)
    monkeypatch.setattr(local_execution, "invoke_local", generate)
    if native:
        def reject(*_):
            raise session.isolated_compiler.ResponseRejected("test rejected schema")
        monkeypatch.setattr(session.isolated_compiler, "invoke", reject)
    prepared = session.prepare(request)
    sid = prepared["session_id"]
    if native:
        session.read({"session_id": sid, "tool": "read_export", "arguments": request["arguments"]})
        result = session.deliver({"session_id": sid, "response_json": json.dumps({"answer": ANSWER})})
    else:
        result = session.draft({"session_id": sid})
    return root, sid, result, invoked


@pytest.mark.parametrize("native", [False, True])
def test_one_diagnostic_revision_on_both_routes_preserves_initial_and_has_no_execution(isolated_host, monkeypatch, native):
    root, sid, initial, invoked = pending(isolated_host, monkeypatch, native=native)
    assert initial["revisionAllowed"] and initial["hostResult"]["state"] == "needs_revision"
    patch = {"replacements": [{"location": "draft-1", "code": GOOD}]}
    result = session.deliver({"session_id": sid, "response_json": json.dumps(patch)})
    assert result["hostResult"]["state"] == "candidate_unverified" and not result["revisionAllowed"]
    assert result["task"]["delivery"]["rendered"] == ANSWER.replace(BAD, GOOD)
    assert result["evidenceDigest"] == initial["evidenceDigest"]
    assert result["modelCalls"] == result["providerCalls"] == [] and len(invoked) == (0 if native else 1)
    assert read_json(root / "sessions" / sid / "draft/result/report.json") == initial
    assert session.inspect({"session_id": sid}) == result
    assert session.deliver({"session_id": sid, "response_json": json.dumps(patch)})["existing"] == result
    with pytest.raises(PermissionError):
        session.read({"session_id": sid, "tool": "read_export", "arguments": {"path": "/sandbox/snapshot"}})


@pytest.mark.parametrize("kind", ["invalid", "same_error", "close", "unknown"])
def test_revision_budget_includes_rejection_nonimprovement_and_unknown(isolated_host, monkeypatch, kind):
    root, sid, initial, _ = pending(isolated_host, monkeypatch)
    folder = root / "sessions" / sid
    if kind == "unknown":
        (folder / "revision").mkdir()
        assert session.inspect({"session_id": sid})["phase"] == "revision"
    if kind == "close":
        result = session.draft({"session_id": sid, "close_incomplete": True})
    else:
        wire = '{"answer":"hide the error"}' if kind == "invalid" else json.dumps({"replacements": [{"location": "draft-1", "code": BAD}]})
        result = session.deliver({"session_id": sid, "response_json": wire})
    if kind == "unknown":
        assert result["existing"]["route"] == "pending_or_unknown_no_retry"
    else:
        assert result["hostResult"]["state"] == ("needs_revision" if kind == "same_error" else "rejected")
        assert not result["revisionAllowed"]
    assert session.deliver({"session_id": sid, "response_json": "{}"})["route"] == "already_drafted_or_unknown"
    assert read_json(folder / "draft/result/report.json") == initial


def test_default_no_revision_and_operator_policy_not_model_parameters(isolated_host, monkeypatch):
    _, sid, result, _ = pending(isolated_host, monkeypatch, enabled=False)
    assert "revisionAllowed" not in result
    assert session.deliver({"session_id": sid, "response_json": "{}"})["route"] == "already_drafted_or_unknown"
    root, profile, *_ = isolated_host
    for bad in [False, 1, "true"]:
        revised = copy.deepcopy(profile)
        revised["artifactRepair"] = bad
        (root / "host/profile.json").write_text(json.dumps(revised))
        with pytest.raises(PermissionError):
            session.describe()
