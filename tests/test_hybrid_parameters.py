import pytest
from jsonschema import Draft202012Validator

from evaluation.hybrid_parameters import binding_schema, lower, obj


def test_host_shape_never_selects_business_parameter_for_model():
    caller = obj({"sourcePath": {"type": "string"}, "unrelatedName": {"type": "string"}})
    target = obj({"path": {"type": "string"}, "limit": {"type": "integer"}, "mode": {"type": "string", "const": "get"}})
    raw = {"path": {"caller": "input#/sourcePath"}, "limit": {"literal": 3, "origin": "p0", "quote": "Read 3 records"}, "mode": {"host_const": True}}
    schema = binding_schema(target, caller, ["p0"])
    assert "input#/unrelatedName" in str(schema)
    Draft202012Validator(schema).validate(raw)
    expression, mapping = lower(target, caller, raw, {"p0": {"origin": "skill_source", "text": "Read 3 records"}})
    assert expression["fields"]["path"] == {"kind": "reference", "source": "input", "pointer": "/sourcePath"}
    assert mapping[2]["origin"] == "host_schema_const"


@pytest.mark.parametrize("raw", [{"literal": "input#/sourcePath", "origin": "p0", "quote": "Read 3 records"},
    {"literal": "forged", "origin": "p0", "quote": "forged"}, {"host_const": True}, {"caller": "input#/missing"}])
def test_fake_reference_or_unattributed_value_not_silently_repaired(raw):
    caller = obj({"sourcePath": {"type": "string"}})
    with pytest.raises(ValueError):
        lower({"type": "string"}, caller, raw, {"p0": {"origin": "skill_source", "text": "Read 3 records"}})


def test_nested_array_shape_and_typed_numeric_literal():
    caller = obj({"resource": {"type": "string"}})
    target = obj({"selection": {"type": "array", "items": {"type": "string"}, "minItems": 1}, "after": {"type": "integer"}})
    raw = {"selection": [{"caller": "input#/resource"}], "after": {"literal": -3600, "origin": "p0", "quote": "after=-3600"}}
    Draft202012Validator(binding_schema(target, caller, ["p0"])).validate(raw)
    expression, _ = lower(target, caller, raw, {"p0": {"origin": "skill_source", "text": "after=-3600"}})
    assert expression["fields"]["after"]["value"] == -3600
