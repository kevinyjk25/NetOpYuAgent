"""Boundary probes for lossless finite literal binding, not semantic scores."""

import json
import math

import pytest

from evaluation.public_skill_translation_v2 import ParameterEvidence, bind_schema_parameters
from evaluation.translation_contract_tasks import contract_review_input
from tests.test_translation_contract_tasks import _request, _source


@pytest.mark.parametrize("value", ['a"b', r"a\b", "line\nnext", "路由器甲", "with spaces"])
def test_quoted_json_strings_round_trip_without_changing_bytes(value: str) -> None:
    literal = json.dumps(value, ensure_ascii=False)
    prompt = f"label={literal}"
    schema = {"properties": {"label": {"type": "string"}}, "required": ["label"]}
    evidence = ParameterEvidence(name="label", value=value, source_text=literal, start=6, end=len(prompt))
    for proposals in [(), (evidence,)]:
        values, sources, failures = bind_schema_parameters(prompt, schema, proposals)
        assert values == {"label": value}
        assert not failures
        assert prompt[sources["label"]["start"]:sources["label"]["end"]] == literal


@pytest.mark.parametrize("literal", ['"bad\\q"', r'"\u0024{target}"', '"${target}"'])
def test_invalid_escape_or_decoded_placeholder_cannot_bind(literal: str) -> None:
    values, _, failures = bind_schema_parameters(
        f"label={literal}", {"properties": {"label": {"type": "string"}}, "required": ["label"]},
    )
    assert not values
    assert failures == ["parameter_invalid:label"]


@pytest.mark.parametrize("literal", ["9" * 400, "-" + "9" * 400, "NaN", "Infinity"])
def test_non_finite_numbers_are_never_bound(literal: str) -> None:
    values, _, failures = bind_schema_parameters(
        f"load={literal}", {"properties": {"load": {"type": "number"}}, "required": ["load"]},
    )
    assert not values
    assert failures == ["parameter_invalid:load"]


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_non_finite_fixture_values_rejected_even_after_model_copy(value: float) -> None:
    request = _request().model_copy(update={"fixture_values": {"load": value}})
    with pytest.raises(ValueError):
        contract_review_input(request)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_source_json_constants_are_rejected(constant: str) -> None:
    text = '{"name":"status","inputSchema":{},"example":' + constant + '}'
    request = _request().model_copy(update={"tool_source": _source(text, "fixture://bad-schema")})
    with pytest.raises(ValueError, match="non-finite constant"):
        contract_review_input(request)


@pytest.mark.parametrize("literal", ["1e999", "-1e999"])
def test_source_json_nested_exponent_overflow_is_rejected(literal: str) -> None:
    text = '{"name":"status","inputSchema":{},"metadata":{"examples":[' + literal + ']}}'
    request = _request().model_copy(update={"tool_source": _source(text, "fixture://overflow")})
    with pytest.raises(ValueError):
        contract_review_input(request)
