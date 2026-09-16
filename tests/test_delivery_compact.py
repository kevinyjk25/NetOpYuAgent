"""V3 transport/source addressing invariants, never a semantic accuracy score."""
import copy
import json
from json import dumps

import pytest

from skill_authoring import delivery
from network_runtime.l0.structured_schema import DataBindingError, checked_schema


def contract(kind="analysis", language="", task="Explain the observed state. Identify the next evidence."):
    return delivery.compile_selection({"requirements": [{"kind": kind, "language": language,
        "source_ref": "task:0"}], "unrepresented": []}, {"task": task})


@pytest.mark.parametrize("text", ["", "\n", "  ", "你好。\n下一条？ 末尾\n", "A. B!\n\nC?\n",
    "x" * 1903, "```sh\nrm -rf /do-not-execute\n```\n", "same\n\nsame\n\n"])
def test_source_ids_are_lossless_spans_not_semantic_summaries(text):
    refs = delivery.source_references({"task": text})
    assert "".join(row["text"] for row in refs.values()) == text
    for key, row in refs.items():
        assert key == f'task:{row["offset"]}'
        assert text[row["offset"]:row["offset"] + len(row["text"]) ] == row["text"]
        assert len(row["text"]) <= 900


def test_same_text_at_different_offsets_is_not_conflated():
    task = "Repeated sentence. Repeated sentence. "
    refs = delivery.source_references({"task": task})
    keys = list(refs)
    assert len(keys) == 2 and refs[keys[0]]["text"] == refs[keys[1]]["text"]
    frozen = delivery.compile_selection({"requirements": [{"kind": "analysis", "language": "", "source_ref": keys[1]}],
                                        "unrepresented": []}, {"task": task})
    assert frozen["requirements"][0]["offset"] == len(refs[keys[0]]["text"])
    assert frozen["requirements"][0]["source_ref"] == keys[1]


@pytest.mark.parametrize("mutation", ["unknown", "quote", "duplicate", "hidden"])
def test_selection_does_not_allow_forged_quotes_or_unseen_references(mutation):
    proposal = copy.deepcopy(contract()["proposal"])
    if mutation == "unknown":
        proposal["requirements"][0]["source_ref"] = "task:999"
    elif mutation == "hidden":
        proposal["unrepresented"] = [{"source_ref": "hidden:0", "reason": "Do not grant access"}]
    elif mutation == "quote":
        proposal["requirements"][0]["quote"] = "Invented text"
    else:
        proposal["requirements"] *= 2
    with pytest.raises(ValueError):
        delivery.compile_selection(proposal, {"task": "Explain the observed state."})


@pytest.mark.parametrize("kind,language,content", [
    ("analysis", "", {"text": "An observation, not approval."}),
    ("decision", "", {"conclusion": "Not ready", "basis": "Required evidence absent"}),
    ("artifact", "python", {"body": "print('inert text')"}),
    ("next_steps", "", {"items": ["Read the relevant export if authorized."]}),
])
def test_compact_content_maps_to_host_metadata_without_changing_user_content(kind, language, content):
    frozen = contract(kind, language)
    checked_schema(delivery.response_schema(frozen))
    candidate = {"delivery": {"d0": content}, "unresolved": {}, "uncertainties": ["Not independently verified."]}
    before = copy.deepcopy(candidate)
    rendered = delivery.render(frozen, candidate)
    canonical = delivery._canonical_response(frozen, candidate)
    assert canonical["delivery"]["d0"] == {"state": "provided", "content": content, "gap": ""}
    assert candidate == before and rendered["shapeComplete"] and rendered["semanticApproval"] is False
    assert rendered["candidateDigest"] == delivery.sha256_json(candidate)
    assert rendered["canonicalCandidateDigest"] == delivery.sha256_json(canonical)
    assert "Not independently verified" in rendered["rendered"]


@pytest.mark.parametrize("kind,language", [("analysis", ""), ("decision", ""), ("artifact", "sql"), ("next_steps", "")])
def test_null_is_an_explicit_unresolved_choice_with_its_own_reason(kind, language):
    frozen = contract(kind, language)
    candidate = {"delivery": {"d0": None}, "unresolved": {"d0": "Missing the relevant observation."}, "uncertainties": []}
    rendered = delivery.render(frozen, candidate)
    assert not rendered["shapeComplete"] and rendered["checks"][0]["status"] == "unresolved"
    assert "Missing the relevant observation" in rendered["rendered"]


@pytest.mark.parametrize("values,unresolved", [
    ({"d0": None}, {}), ({"d0": None}, {"d0": "  "}),
    ({"d0": {"text": "Provided"}}, {"d0": "Conflicting"}),
    ({"d0": {"text": "Provided"}}, {"other": "Unbound"}),
    ({}, {}), ({"d0": {"text": 9}}, {}),
    ({"d0": {"state": "provided", "content": {"text": "Old wire"}, "gap": ""}}, {}),
])
def test_no_missing_ids_or_silent_legacy_repair(values, unresolved):
    candidate = {"delivery": values, "unresolved": unresolved, "uncertainties": []}
    before = copy.deepcopy(candidate)
    with pytest.raises(DataBindingError):
        delivery.render(contract(), candidate)
    assert candidate == before


def test_blanks_and_wrong_kinds_do_not_become_semantic_success():
    candidate = {"delivery": {"d0": {"text": ""}}, "unresolved": {}, "uncertainties": []}
    assert not delivery.render(contract(), candidate)["shapeComplete"]
    wrong = contract(task="Provide a complete executable query, not an explanation.")
    candidate["delivery"]["d0"]["text"] = "An explanation instead of a query."
    assert delivery.render(wrong, candidate)["shapeComplete"]
    assert not delivery.render(wrong, candidate)["semanticApproval"]


def test_strict_wire_and_legacy_protocol_do_not_autoupgrade():
    candidate = {"delivery": {"d0": {"text": "用户文字"}}, "unresolved": {}, "uncertainties": []}
    assert delivery.decode_response(json.dumps(candidate, ensure_ascii=False)) == candidate
    legacy = delivery.compile_contract({"requirements": [{"kind": "analysis", "language": "", "origin": "task",
        "quote": "Explain the observed state."}], "unrepresented": []}, {"task": "Explain the observed state."})
    with pytest.raises(DataBindingError):
        delivery.render(legacy, candidate)


def test_local_decoder_receives_exact_host_output_schema_and_validation_stays_independent(tmp_path, monkeypatch):
    from skill_authoring import local_execution, compiler
    captured = []
    schema = delivery.response_schema(contract())
    candidate = {"delivery": {"d0": {"text": "Unverified text"}}, "unresolved": {}, "uncertainties": []}
    class Reply:
        def __init__(self, value):
            self.value = value
        def raise_for_status(self):
            pass
        def json(self):
            return self.value
    class Client:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def get(self, url):
            assert url == local_execution.ENDPOINT + "/api/tags"
            return Reply({"models": [{"name": compiler.MODEL, "digest": "test-model-only"}]})
        def post(self, url, *, json):
            assert url == local_execution.ENDPOINT + "/api/chat"
            captured.append(copy.deepcopy(json))
            return Reply({"model": compiler.MODEL, "done": True, "done_reason": "stop",
                "message": {"content": dumps(candidate)}, "prompt_eval_count": 1, "eval_count": 1})
    monkeypatch.setattr(local_execution.httpx, "Client", lambda **_: Client())
    request = {"nodeId": "n7", "instructions": "Return the required candidate, not an approval.", "inputs": {},
        "outputSchema": schema, "evidencePolicy": {}, "observationAgesAtStartMs": {}, "maxOutputTokens": 2048}
    costs = []
    local_execution.invoke_local(request, tmp_path / "valid", costs)
    assert captured[0]["format"] == schema and captured[0]["format"] is not schema
    assert not captured[0]["think"] and len(captured) == 1
    candidate["delivery"]["d0"]["text"] = 17
    with pytest.raises(DataBindingError):
        local_execution.invoke_local(request, tmp_path / "invalid", costs)
    assert len(captured) == 2  # one request per invocation; no fallback or retry
    assert costs[-1]["errorType"] == "DataBindingError"
