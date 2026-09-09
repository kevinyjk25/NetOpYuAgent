"""Separate language extraction uncertainty from deterministic Boolean logic."""

import copy
import json

import pytest
from jsonschema import ValidationError

from evaluation import flow_checkpoint as checkpoint
from evaluation.flow_condition_expression import author, derive, parse_proposal, request
from tests.test_flow_joint_conditions import fixture


def match(field, value=True):
    return dict(kind="match", field=field, equals=value, source_ids=["s0007"])


def proposal(expr):
    return dict(status="specified", condition=expr, on_match="read", on_mismatch="unsupported",
        match_source_ids=["s0009"], mismatch_source_ids=["s0008"])


def surface(expression):
    return dict(expression=expression, source_ids=["s0007"], read_source_ids=["s0009"],
        stop_source_ids=["s0008"], on_failure="unsupported", unresolved_question=None)


@pytest.mark.parametrize("kind", ["all", "any"])
def test_boolean_combinations_are_computed_not_model_classified(kind):
    _, sources, tree = fixture("invoice")
    result = derive(sources, tree, proposal(dict(kind=kind, source_ids=["s0007"],
        children=[match("settled"), match("waiver_exists")])) )
    assert {key for key, value in result["table"].items() if value["outcome"] == "read"} == (
        {"v11"} if kind == "all" else {"v01", "v10", "v11"})
    assert not result["semanticAlignmentProven"]
    assert result["fullSourceReview"] == "required_not_run"


def test_nested_all_any_and_composite_negation():
    _, sources, tree = fixture("dispatch")
    condition = dict(kind="all", source_ids=["s0007"], children=[match("valid"),
        dict(kind="any", source_ids=["s0007"], children=[match("fast_track"), match("signed_review")])])
    result = derive(sources, tree, proposal(condition))
    assert {k for k, v in result["table"].items() if v["outcome"] == "read"} == {"v011", "v101", "v111"}
    inverted = derive(sources, tree, proposal(dict(kind="not", source_ids=["s0007"], child=condition)))
    assert {k for k, v in inverted["table"].items() if v["outcome"] == "read"} == {"v000", "v001", "v010", "v100", "v110"}


def test_false_requirement_and_unresolved_are_not_true_or_accepted():
    _, sources, tree = fixture("release")
    expr = dict(kind="all", source_ids=["s0007"],
        children=[match("decision_present"), match("stale", False), match("permitted")])
    result = derive(sources, tree, proposal(expr))
    assert {k for k, v in result["table"].items() if v["outcome"] == "read"} == {"v110"}
    result = derive(sources, tree, dict(status="unresolved", source_ids=["s0007"], question="Missing authoritative policy meaning."))
    assert result["status"] == "unresolved_no_revision"
    assert not {"table", "revision"} & result.keys()


@pytest.mark.parametrize("bad", ["field", "literal", "source", "empty", "depth"])
def test_invalid_or_unbounded_proposal_rejected(bad):
    _, sources, tree = fixture("invoice")
    expr = match("settled")
    if bad == "field":
        expr["field"] = "fabricated"
    elif bad == "literal":
        expr["equals"] = 1
    elif bad == "source":
        expr["source_ids"] = ["fake-source"]
    elif bad == "empty":
        expr = dict(kind="all", source_ids=["s0007"], children=[])
    else:
        for _ in range(9):
            expr = dict(kind="not", source_ids=["s0007"], child=expr)
    with pytest.raises((ValueError, ValidationError)):
        derive(sources, tree, proposal(expr))


def test_source_request_hides_tables_branches_and_oracles():
    _, sources, tree = fixture("invoice")
    payload = json.loads(request(sources, tree)["messages"][1]["content"])
    assert not {"assignments", "draft", "tree", "observations", "reference", "suite"} & payload.keys()
    assert "when_equal" not in json.dumps(payload)
    assert payload["targetSkillSpans"]
    assert payload["outputSchema"] == request(sources, tree)["format"]


@pytest.mark.parametrize("formula", ["__import__('os').system('false')", "settled.value", "[settled]",
    "settled == 1", "True", "unknown_flag", "settled + waiver_exists", "lambda: settled"])
def test_expression_is_a_non_executable_allowlisted_language(formula):
    _, sources, tree = fixture("invoice")
    with pytest.raises(ValueError):
        parse_proposal(sources, tree, surface(formula))


def test_readable_syntax_preserves_grouping_and_negation():
    _, sources, tree = fixture("dispatch")
    raw = surface("valid and (fast_track or signed_review)")
    result = derive(sources, tree, parse_proposal(sources, tree, raw))
    assert {k for k, v in result["table"].items() if v["outcome"] == "read"} == {"v011", "v101", "v111"}
    raw["unresolved_question"] = "Unresolved prerequisite is still present."
    with pytest.raises(ValueError, match="hide"):
        parse_proposal(sources, tree, raw)


def test_one_attempt_replay_and_drift_rejection(tmp_path, monkeypatch):
    _, sources, tree = fixture("invoice")
    raw = surface("settled or waiver_exists")
    monkeypatch.setattr(checkpoint.OllamaAnchoredAuthorAdapter, "preflight", lambda self: {"model": "qwen3.5:9b"})
    calls = []

    def send(arm, wire):
        calls.append(copy.deepcopy(wire))
        return dict(httpStatus=200, latencyMs=1, body=json.dumps(dict(model="qwen3.5:9b", done=True,
            done_reason="stop", message=dict(content=json.dumps(raw)), prompt_eval_count=1, eval_count=1)))

    monkeypatch.setattr(checkpoint, "send", send)
    root = tmp_path / "attempt"
    with pytest.raises(ValueError, match="budget"):
        author(sources, tree, root)
    result = author(sources, tree, root, max_new_calls=1)
    assert result == author(sources, tree, root) and len(calls) == 1
    (root / "expression.json").write_text("{}")
    with pytest.raises(ValueError, match="checkpoint"):
        author(sources, tree, root, max_new_calls=1)
    assert len(calls) == 1
