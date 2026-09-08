"""Mock transport only; real model results are separate immutable artifacts."""

import json

import pytest

from evaluation import flow_source_duty_pilot as pilot
from evaluation.flow_source_duties import SourceBundle
from evaluation.flow_translation import _write
from tests.test_flow_source_duties import bundle, prose_candidate, review_fixture


def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "implementation", lambda: {"test": "fixed"})
    monkeypatch.setattr(pilot, "environment", lambda: {"test": "fixed"})
    monkeypatch.setattr(pilot.OllamaAnchoredAuthorAdapter, "preflight", lambda self: {"model": "qwen3.5:9b", "modelArtifactDigest": "test"})
    inputs, root = tmp_path / "cases.json", tmp_path / "batch"
    _write(inputs, [dict(id="first", bundle=bundle("Read once; never write.").model_dump(mode="json"))])
    pilot.freeze(inputs, root)
    return root


def completed(root, content=None, done="stop"):
    manifest = pilot.load(root)
    case = manifest["cases"][0]
    docs = SourceBundle.model_validate(case["bundle"])
    reply = dict(model="qwen3.5:9b", done=True, done_reason=done,
        prompt_eval_count=5, eval_count=7, message=dict(content=content or prose_candidate(docs).model_dump_json()))
    envelope = dict(httpStatus=200, body=json.dumps(reply), latencyMs=12.0)
    folder = root / "first"
    folder.mkdir()
    _write(folder / "request.json", dict(wireRequest=case["request"], model=manifest["model"]))
    _write(folder / "response.json", envelope)
    files, status = pilot.derive(docs, envelope)
    for name, value in files.items():
        _write(folder / name, value)
    _write(folder / "status.json", status)
    _write(folder / "receipt.json", pilot.receipt(folder))
    return files, status


def test_complete_replay_has_no_new_model_call_and_full_review_is_separate(tmp_path, monkeypatch):
    root = setup(tmp_path, monkeypatch)
    files, status = completed(root)
    monkeypatch.setattr(pilot.OllamaAnchoredAuthorAdapter, "preflight", lambda self: pytest.fail("no new preflight"))
    pilot.run(root, 0)
    first = pilot.report(root)
    assert first["completed"] and first["structurallyQualified"] == 1 and first["sourceReviewed"] == 0
    reviews = tmp_path / "reviews"
    reviews.mkdir()
    _write(reviews / "first.json", review_fixture(files["review-input.json"]).model_dump(mode="json"))
    reviewed = pilot.report(root, reviews)
    assert reviewed["reviewSupported"] == 1 and reviewed["semanticAccuracy"] is None
    assert reviewed["wholeSkillTranslations"] == 0 and reviewed["runtimeExecutions"] == 0
    assert first == pilot.report(root)


def test_pending_budget_and_uncertain_folder_never_retries(tmp_path, monkeypatch):
    root = setup(tmp_path, monkeypatch)
    assert not pilot.report(root)["completed"]
    with pytest.raises(ValueError, match="budget"):
        pilot.run(root, 0)
    (root / "first").mkdir()
    with pytest.raises(ValueError, match="checkpoint"):
        pilot.run(root, 4)


@pytest.mark.parametrize("content,done", [("not json", "stop"), ('{"rows":{}}', "stop"), ('{}', "length")])
def test_failed_originals_preserved_not_filled_or_retried(tmp_path, monkeypatch, content, done):
    root = setup(tmp_path, monkeypatch)
    files, status = completed(root, content, done)
    assert status["status"] == "blocked"
    assert "compilation.json" not in files
    pilot.run(root, 0)
    result = pilot.report(root)
    assert result["completed"] and result["structurallyQualified"] == 0
    assert result["inputTokens"] == 5 and result["outputTokens"] == 7


def test_raw_or_derived_tamper_detected(tmp_path, monkeypatch):
    root = setup(tmp_path, monkeypatch)
    completed(root)
    with (root / "first/status.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="checkpoint"):
        pilot.report(root)


@pytest.mark.parametrize("case_id", ["../bad", "/bad", "..", "", "a/b"])
def test_invalid_case_path(case_id):
    with pytest.raises(ValueError):
        pilot.cases_at([dict(id=case_id, bundle=bundle("Read.").model_dump(mode="json"))])


def test_duplicate_case_paths_and_invalid_timing():
    case = dict(id="same", bundle=bundle("Read.").model_dump(mode="json"))
    with pytest.raises(ValueError):
        pilot.cases_at([case, case])
    with pytest.raises(ValueError, match="latency"):
        pilot.derive(bundle("Read."), dict(httpStatus=200, body="{}", latencyMs=float("nan")))


@pytest.mark.parametrize("tokens", [True, -1, "12", 0.5])
def test_invalid_model_metrics_not_summed(tokens):
    reply = dict(model="qwen3.5:9b", done=True, done_reason="stop", prompt_eval_count=tokens)
    _, status = pilot.derive(bundle("Read."), dict(httpStatus=200, body=json.dumps(reply), latencyMs=1))
    assert status["status"] == "blocked" and "invalid token" in status["error"]
    assert "inputTokens" not in status
