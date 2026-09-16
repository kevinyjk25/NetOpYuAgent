"""Format/authority invariants, not evidence that model interpretation is correct."""
import copy
import json

import pytest

from skill_authoring import delivery
from skill_authoring.artifact_checks import inspect_candidate
from network_runtime.l0.structured_schema import checked_schema


def contract(kind="artifact", language="kql", quote="Draft a bounded query from the actual inventory."):
    return delivery.compile_contract({"requirements": [{"kind": kind, "language": language,
        "origin": "task", "quote": quote}], "unrepresented": []}, {"task": quote})


def response(content, state="provided", gap=""):
    return {"delivery": {"d0": {"state": state, "content": content, "gap": gap}}, "uncertainties": []}


@pytest.mark.parametrize("kind,language,content", [
    ("artifact", "python", {"body": "print('a harmless draft')"}),
    ("artifact", "json", {"body": '{"configured": false}'}),
    ("analysis", "", {"text": "Observed data is incomplete."}),
    ("decision", "", {"conclusion": "Open, not ready for closure.", "basis": "Required review is absent."}),
    ("next_steps", "", {"items": ["Obtain the missing review record."]}),
])
def test_shapes_reuse_original_schema_validator_and_never_approve_meaning(kind, language, content):
    frozen = contract(kind, language)
    checked_schema(delivery.response_schema(frozen))
    result = delivery.render(frozen, response(content))
    assert result["shapeComplete"] and not result["semanticApproval"] and result["taskSuccess"] is None
    assert not result["sourceScriptsExecuted"]


def test_invented_source_quote_and_wrong_origin_rejected():
    proposal = contract()["proposal"]
    for key, value in [("quote", "Invented requirement not in the source."), ("origin", "oracle")]:
        changed = copy.deepcopy(proposal)
        changed["requirements"][0][key] = value
        with pytest.raises(ValueError):
            delivery.compile_contract(changed, {"task": proposal["requirements"][0]["quote"]})


@pytest.mark.parametrize("body", ["", "   ", "1. Select records\n2. Count them", "```kql\nT\n```", "## Query instructions"])
def test_blank_outline_or_fenced_body_does_not_pass_as_code_artifact(body):
    result = delivery.render(contract(), response({"body": body}))
    assert not result["shapeComplete"] and "Not delivered" in result["rendered"]


def test_valid_shape_with_bad_syntax_is_not_approved_and_remains_inspectable():
    result = delivery.render(contract(language="python"), response({"body": "def broken("}))
    assert result["shapeComplete"] and not result["semanticApproval"]
    checks = inspect_candidate({"draft": result["rendered"]}, "Produce code", [])
    assert checks["status"] == "failed_checks" and not checks["queryExecuted"]


def test_explicit_gap_is_not_silently_deleted_or_called_complete():
    result = delivery.render(contract(), response({"body": ""}, "unresolved", "The inventory does not include column types."))
    assert not result["shapeComplete"] and "column types" in result["rendered"]
    assert result["checks"][0]["status"] == "unresolved"


def test_decision_and_next_steps_need_their_own_fields():
    with pytest.raises(ValueError):
        delivery.render(contract("decision", ""), response({"text": "Review is absent."}))
    result = delivery.render(contract("decision", ""), response({"conclusion": "", "basis": "Review is absent."}))
    assert not result["shapeComplete"]
    assert not delivery.render(contract("next_steps", ""), response({"items": []}))["shapeComplete"]


def test_renderer_does_not_claim_a_wrong_kind_interpretation_is_proven():
    # Membership can pass even if the model chose a poor kind. Explicitly open.
    frozen = contract("analysis", "", "Return runnable code, not an explanation.")
    assert frozen["quoteMembershipChecked"] and "not_semantically_proven" in frozen["coverageAndKindInterpretation"]
    result = delivery.render(frozen, response({"text": "An explanation."}))
    assert result["shapeComplete"] and not result["semanticApproval"]


def test_exact_keys_no_model_supplied_success_or_permission():
    candidate = response({"body": "T | count"})
    candidate["approved"] = True
    with pytest.raises(ValueError):
        delivery.render(contract(), candidate)


def test_markdown_artifact_cannot_break_out_of_its_host_fence():
    result = delivery.render(contract(language="markdown"), response({"body": "# A document\n```python\nprint(1)\n```"}))
    assert "````markdown" in result["rendered"] and result["shapeComplete"]


def test_explicit_json_text_transport_is_lossless_not_field_coercion():
    original = response({"text": "未批准 ≠ 未审阅。"})
    assert delivery.decode_response(json.dumps(original, ensure_ascii=False)) == original
    bad = response({"text": 123})
    decoded = delivery.decode_response(json.dumps(bad))
    assert decoded["delivery"]["d0"]["content"]["text"] == 123
    with pytest.raises(delivery.DataBindingError) as caught:
        delivery.render(contract("analysis", ""), decoded)
    assert caught.value.pointer == "/delivery/d0/content/text"


@pytest.mark.parametrize("key", ["gap", "state", "content"])
def test_delivery_required_field_diagnostic_does_not_insert_or_move_values(key):
    value = response({"items": ["Obtain an additional observation."]})
    misplaced = value["delivery"]["d0"].pop(key)
    value["delivery"][key] = misplaced
    before = copy.deepcopy(value)
    with pytest.raises(delivery.DataBindingError) as caught:
        delivery.render(contract("next_steps", ""), value)
    assert caught.value.pointer == "/delivery/d0/" + key
    assert "required field is missing" in caught.value.detail and value == before


@pytest.mark.parametrize("wire", [
    {}, "[]", "null", '"{}"', "```json\n{}\n```", "a: 1",
    '{"a":1,"a":2}', '{"a":{"b":1,"b":2}}', '{"n":NaN}',
    '{"n":Infinity}', '{"n":1e309}', "{" * 1500,
])
def test_bad_wire_has_no_permissive_decoding_or_source_execution(wire):
    with pytest.raises(delivery.DataBindingError) as caught:
        delivery.decode_response(wire)
    assert caught.value.pointer == "/response_json"


def test_wire_limit_is_bytes_not_unicode_characters(monkeypatch):
    monkeypatch.setattr(delivery, "MAX_WIRE_BYTES", 20)
    with pytest.raises(delivery.DataBindingError, match="wire_budget"):
        delivery.decode_response('{"a":"一二三四五六七"}')


def test_malformed_wire_reports_position_without_echoing_payload():
    with pytest.raises(delivery.DataBindingError) as caught:
        delivery.decode_response('{"secret":"do-not-echo"}}')
    assert "line 1, column" in caught.value.detail and "do-not-echo" not in caught.value.detail


def test_unrepresented_requires_an_exact_source_anchor_not_free_text_read_status():
    proposal = copy.deepcopy(contract()["proposal"])
    origins = {"task": proposal["requirements"][0]["quote"]}
    proposal["unrepresented"] = ["The export has not been read yet."]
    with pytest.raises(ValueError, match="schema mismatch"):
        delivery.compile_contract(proposal, origins)
    proposal["unrepresented"] = [{"origin": "task", "quote": "The export has not been read yet.", "reason": "Need evidence"}]
    with pytest.raises(ValueError, match="not exact"):
        delivery.compile_contract(proposal, origins)


def test_completed_read_state_never_clears_unrepresented_requirements():
    proposal = copy.deepcopy(contract()["proposal"])
    quote = proposal["requirements"][0]["quote"]
    proposal["unrepresented"] = [{"origin": "task", "quote": quote, "reason": "A delivery kind is not represented."}]
    frozen = delivery.compile_contract(proposal, {"task": quote})
    state = {"completedReads": 3, "contentSufficiency": "not_assessed", "authorityGranted": False}
    result = delivery.render(frozen, response({"body": "T | count"}), evidence_state=state)
    assert not result["shapeComplete"] and not result["declaredCoverageComplete"]
    assert result["evidenceState"] == state and quote in result["rendered"]
    assert not result["semanticApproval"] and not frozen["authorityGranted"]


def test_legacy_stale_annotation_is_not_erased_or_silently_upgraded():
    old = {**contract(), "profile": "source-anchored-delivery/v1", "uncovered": ["The export has not been read yet."]}
    result = delivery.render(old, response({"body": "T | count"}))
    assert not result["shapeComplete"] and old["uncovered"][0] in result["rendered"]
