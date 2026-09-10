"""Typed syntax remains inert, exact and separate from source semantics."""
import copy

import pytest
from jsonschema import Draft202012Validator

from evaluation import source_program, source_program_anchors, source_program_lines as lines


CATALOG = {"tools": [{"name": "inspect", "outputSchema": {"type": "boolean"}}]}


def completed():
    return [{"op": "read", "name": "observation", "tool": "inspect"},
            {"op": "end", "outcome": "read_path_completed", "explanation": "Inspection completed.", "duties": []}]


def parsed(rows, catalog=CATALOG, modes=(), bindings=()):
    program = lines.render(rows, catalog, modes, bindings)
    annotated, _ = source_program_anchors.inject(program)
    return source_program.parse(annotated, catalog=catalog)


def test_modes_and_source_operations_require_exact_declared_choices():
    modes = [{"hostTool": "inspect", "modes": [{"id": "status"}, {"id": "snapshot"}]}]
    bindings = [{"hostTool": "inspect", "sourceOperation": "original_status"}]
    rows = completed()
    rows[0].update(operationMode="status", sourceOperation="original_status")
    plan = parsed(rows, modes=modes, bindings=bindings)
    assert plan["steps"][0]["operationMode"] == "status"
    assert plan["steps"][0]["sourceOperation"] == "original_status"
    with pytest.raises(ValueError, match="typed program"):
        lines.render(rows, CATALOG)
    rows[0]["operationMode"] = "invented"
    with pytest.raises(ValueError, match="typed program"):
        lines.render(rows, CATALOG, modes, bindings)


def test_quoted_text_cannot_escape_into_code():
    rows = completed()
    content = '\"); __import__("os").system("not-executed")\n# 中文 "quote"'
    rows[1]["explanation"] = content
    plan = parsed(rows)
    assert len(plan["steps"]) == 1 and plan["exit"]["explanation"] == content


@pytest.mark.parametrize("value", [True, False, None, 0, -2, 1.5, "false", "x\n\"y"])
def test_literals_are_exact_not_truthiness_coercions(value):
    rows = completed()
    end = copy.deepcopy(rows.pop())
    rows += [{"op": "if_equal", "value": {"kind": "field", "source": "observation", "pointer": ""}, "equals": value,
              "when_equal": [end], "otherwise": [end]}]
    # Type equality qualification is done by the original compiler after parsing.
    result = parsed(rows)["steps"][1]["equals"]
    assert type(result) is type(value) and result == value


@pytest.mark.parametrize("count", [0, 1, 17])
def test_cardinality_statement_lowers_to_original_length_without_inventing_guard(count):
    catalog = {"tools": [{"name": "inspect", "outputSchema": {"type": "array", "items": {"type": "string"}}}]}
    read, end = completed()
    rows = [read, {"op": "if_length_equal", "source": "observation", "pointer": "", "equals": count,
                   "when_equal": [end], "otherwise": [end]}]
    original = copy.deepcopy(rows)
    new = parsed(rows, catalog)
    old = copy.deepcopy(rows)
    old[1] = {"op": "if_equal", "value": {"kind": "length", "source": "observation", "pointer": ""},
              "equals": count, "when_equal": [end], "otherwise": [end]}
    assert new == parsed(old, catalog)
    assert rows == original
    normalized = lines.canonicalize(rows)
    assert normalized["statements"] == rows
    assert not normalized["businessConditionsInferred"]
    # Nothing automatically adds cardinality to a program that did not ask for it.
    assert "length(" not in lines.render([read, end], catalog)


@pytest.mark.parametrize("count", [-1, 0.5, "0", True, None])
def test_cardinality_statement_rejects_non_integer_counts(count):
    read, end = completed()
    rows = [read, {"op": "if_length_equal", "source": "observation", "pointer": "", "equals": count,
                   "when_equal": [end], "otherwise": [end]}]
    with pytest.raises(ValueError, match="typed program"):
        parsed(rows)


def test_cardinality_statement_rejects_non_array_data_source():
    read, end = completed()
    rows = [read, {"op": "if_length_equal", "source": "observation", "pointer": "", "equals": 0,
                   "when_equal": [end], "otherwise": [end]}]
    with pytest.raises(ValueError, match="array"):
        parsed(rows)


@pytest.mark.parametrize("pointer,kind,constant,valid", [
    ("/enabled", "if_equal", False, True), ("/enabled", "if_equal", True, True),
    ("/enabled", "if_equal", "false", False), ("/backup/enabled", "if_equal", False, True),
    ("/count", "if_equal", 0, True), ("/count", "if_equal", 17, True), ("/count", "if_equal", "0", False),
    ("/items", "if_equal", "[]", False), ("/items", "if_length_equal", 0, True),
    ("/items", "if_length_equal", 17, True), ("/label", "if_equal", "some label", True),
    ("/label", "if_length_equal", 0, False), ("/nullable", "if_equal", None, True),
    ("/nullable", "if_equal", "a value", True), ("/invented", "if_equal", False, False),
])
def test_model_predicates_use_original_types_not_selected_business_answers(pointer, kind, constant, valid):
    paths = [{"pointer": key, "types": types} for key, types in [
        ("/enabled", ["boolean"]), ("/backup/enabled", ["boolean"]), ("/count", ["integer"]),
        ("/items", ["array"]), ("/label", ["string"]), ("/nullable", ["null", "string"])]]
    schema = lines.schema(CATALOG, model_view=True, value_paths=paths)
    end = {"op": "complete", "explanation": "Explicit completed branch."}
    operand = {"value": {"kind": "field" if kind == "if_equal" else "length", "source": "observation", "pointer": pointer}}
    rows = [{"op": "if_equal", **operand, "equals": constant, "when_equal": [end], "otherwise": [end]}]
    assert Draft202012Validator(schema).is_valid(rows) is valid
    # This schema does not prove an observation exists or choose its source:
    # original scope/type qualification still blocks this unbound program.
    if valid:
        with pytest.raises(ValueError):
            parsed(rows)


@pytest.mark.parametrize("mutation", ["indent", "extra_else", "future", "no_end", "extra_parameters", "script"])
def test_no_silent_structure_repair_or_arbitrary_code(mutation):
    rows = completed()
    if mutation == "indent":
        rows[1]["level"] = 2
    elif mutation == "extra_else":
        rows.insert(1, {"op": "else"})
    elif mutation == "future":
        rows.insert(0, {"op": "define", "name": "value", "value": {"kind": "field", "source": "observation", "pointer": ""}})
    elif mutation == "no_end":
        rows.pop()
    elif mutation == "extra_parameters":
        rows[0]["arguments"] = {"id": "made-up"}
    else:
        rows[0]["op"] = "exec"
    with pytest.raises(ValueError):
        parsed(rows)


def branch(yes, no):
    return {"op": "if_equal", "value": {"kind": "field", "source": "observation", "pointer": ""},
            "equals": False, "when_equal": yes, "otherwise": no}


def test_only_surviving_arm_continuation_is_mechanical_and_source_mapped():
    read, end = completed()
    second = {**read, "name": "second"}
    predicate = branch([end], [end])
    predicate["value"]["source"] = "second"
    rows = [read, branch([end], [second]), predicate]
    original = copy.deepcopy(rows)
    normalized = lines.canonicalize(rows)
    assert rows == original
    assert normalized["moves"] == [{"rule": "append_to_only_fallthrough_arm", "branchOrigin": "/program/1",
        "targetArm": "otherwise", "movedStatementOrigins": ["/program/2"]}]
    assert normalized["statements"][1]["otherwise"] == [second, predicate]
    assert len(normalized["statementOrigins"]) == 7
    assert not normalized["businessConditionsInferred"]
    assert len(parsed(rows)["steps"]) == 2  # the second predicate is now lexically dominated
    assert lines.canonicalize(normalized["statements"])["moves"] == []


@pytest.mark.parametrize("terminated_arm", ["when_equal", "otherwise"])
def test_continuation_passing_preserves_both_branch_traces(terminated_arm):
    read, end = completed()
    children = {"when_equal": [{**read, "name": "other"}], "otherwise": [{**read, "name": "other"}]}
    children[terminated_arm] = [end]
    rows = [read, branch(**{"yes": children["when_equal"], "no": children["otherwise"]}),
            {**read, "name": "last"}, end]
    normalized = lines.canonicalize(rows)["statements"]
    def trace(items, choose_equal, calls):
        for row in items:
            if row["op"] == "read":
                calls.append(row["name"])
            elif row["op"] == "end":
                return True
            elif trace(row["when_equal" if choose_equal else "otherwise"], choose_equal, calls):
                return True
        return False
    for value in (False, True):
        before, after = [], []
        assert trace(rows, value, before) == trace(normalized, value, after)
        assert before == after


def test_two_open_arms_cannot_export_a_possibly_undefined_value():
    read, end = completed()
    second = {**read, "name": "second"}
    after = branch([end], [end])
    after["value"]["source"] = "second"
    rows = [read, branch([], [second]), after]
    # Empty true body is not a valid statement block; use an unrelated read
    # to keep both arms open without defining second on the true path.
    rows[1]["when_equal"] = [{**read, "name": "unrelated"}]
    assert lines.canonicalize(rows)["moves"] == []
    with pytest.raises(ValueError, match="dominate"):
        parsed(rows)


def test_one_redundant_completion_is_retained_as_dead_metadata_not_success():
    read, end = completed()
    handoff = {"op": "handoff", "outcome": "needs_l1", "explanation": "Further source duties remain.",
               "duties": [{"when": "otherwise branch", "requirement": "Explain evidence; never execute changes."}]}
    rows = [read, branch([end], [handoff]), {"op": "complete", "explanation": "Redundant model ending."}]
    report = lines.canonicalize(rows)
    assert report["statements"] == rows[:-1]
    assert report["redundantTerminals"][0]["statement"] == rows[-1]
    assert report["redundantTerminals"][0]["originalPointer"] == "/program/2"
    assert parsed(rows)["steps"][1]["otherwise"]["exit"]["outcome"] == "needs_l1"


@pytest.mark.parametrize("work", [
    {"op": "read", "name": "never_called", "tool": "inspect"},
    {"op": "handoff", "outcome": "needs_l1", "explanation": "Still owed work.",
     "duties": [{"when": "later stage", "requirement": "Preserve this responsibility."}]},
])
def test_unreachable_calls_or_duties_are_never_removed(work):
    read, end = completed()
    with pytest.raises(ValueError, match="unreachable"):
        lines.canonicalize([read, branch([end], [end]), work])


def test_work_and_restrictions_are_both_preserved_without_inferred_meaning():
    read, _ = completed()
    work = [{"when": "after inspection", "requirement": "Explain the selected evidence."}]
    restrictions = [{"when": "when explaining", "requirement": "Never disclose raw payload details."}]
    row = {"op": "handoff", "outcome": "needs_l1", "explanation": "Explanation remains for L1.",
           "duties": work, "restrictions": restrictions}
    actual = parsed([read, row])["exit"]["duties"]
    assert [(d["when"], d["requirement"]) for d in actual] == [
        (d["when"], d["requirement"]) for d in work + restrictions]
    row["duties"] = work * 8
    row["restrictions"] = restrictions * 8
    assert len(parsed([read, row])["exit"]["duties"]) == 16
