"""Generation grammar parity: well-shaped still does not mean semantically right."""

import copy

import pytest
from jsonschema import Draft202012Validator

from evaluation.source_candidate_schema import omit_schema_titles, tighten
from evaluation.structured_authoring import response_schema
from evaluation.structured_flow_demo import fixture
from network_runtime.l0.structured_bindings import compile_binding


def expression_validator():
    bundle, tree, reads, _ = fixture()
    schema = response_schema({"bundle": bundle, "inputSchema": tree.input_schema}, {}, {})
    tighten(schema, list(reads))
    return Draft202012Validator({"$ref": "#/$defs/BindingExpression", "$defs": schema["$defs"]})


@pytest.mark.parametrize("expression,target,sources", [
    ({"kind": "literal", "value": {"x": 1}}, {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"], "additionalProperties": False}, {}),
    ({"kind": "object", "fields": {"x": {"kind": "literal", "value": 1}}}, {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"], "additionalProperties": False}, {}),
    ({"kind": "array", "items": [{"kind": "literal", "value": "a"}]}, {"type": "array", "items": {"type": "string"}}, {}),
    ({"kind": "reference", "source": "input", "pointer": "/x"}, {"type": "integer"}, {"input": {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"], "additionalProperties": False}}),
])
def test_generation_shapes_match_existing_binding_compiler(expression, target, sources):
    assert not list(expression_validator().iter_errors(expression))
    plan = compile_binding(sources, target, expression)
    assert not plan["runtimeAuthorityGranted"] and not plan["semanticAlignmentProven"]


@pytest.mark.parametrize("expression", [
    {"node": {"kind": "literal", "value": "example"}},
    {"kind": "object", "fields": {"x": 123}},
    {"kind": "literal", "value": 1, "coerce": True},
    {"kind": "reference", "source": "input"},
    {"kind": "eval", "code": "arbitrary()"},
    {"kind": "column_rows", "source": "r", "pointer": "", "fields": ["x", "x"], "max_rows": 2, "max_columns": 1},
])
def test_invalid_arguments_rejected_before_generation_completion(expression):
    assert list(expression_validator().iter_errors(expression))


def test_authoring_bounds_do_not_modify_runtime_models_or_invent_target_tools():
    bundle, tree, reads, _ = fixture()
    packet = {"bundle": bundle, "inputSchema": tree.input_schema}
    original = response_schema(packet, {}, {})
    before = copy.deepcopy(original)
    bounded = tighten(copy.deepcopy(original), list(reads))
    assert original == before
    assert original["$defs"]["StructuredFlowTree"]["properties"]["steps"]["maxItems"] == 64
    assert bounded["$defs"]["StructuredFlowTree"]["properties"]["steps"]["maxItems"] == 8
    assert bounded["$defs"]["StructuredTreeRead"]["properties"]["tool"]["enum"] == list(reads)


def test_schema_annotation_compaction_preserves_literal_title_keys_and_validation():
    schema = {"title": "Root annotation", "type": "object", "properties": {
        "title": {"title": "Field annotation", "type": "string"},
        "payload": {"title": "Annotation", "const": {"title": "ACTUAL DATA", "properties": {"title": "not a schema"}}}},
        "required": ["title", "payload"], "additionalProperties": False}
    before = copy.deepcopy(schema)
    compact = omit_schema_titles(schema)
    assert schema == before and "title" in compact["properties"]
    assert compact["properties"]["payload"]["const"] == schema["properties"]["payload"]["const"]
    for value in [{"title": "value", "payload": schema["properties"]["payload"]["const"]}, {}, {"title": 123}]:
        assert Draft202012Validator(schema).is_valid(value) == Draft202012Validator(compact).is_valid(value)


def test_compaction_preserves_binding_grammar_acceptance():
    original = expression_validator().schema
    compact = omit_schema_titles(original)
    for value in [{"kind": "literal", "value": {"title": "unchanged"}}, {"kind": "object", "fields": {}},
                  {"node": {"kind": "literal", "value": "wrong-root"}}]:
        assert Draft202012Validator(original).is_valid(value) == Draft202012Validator(compact).is_valid(value)
