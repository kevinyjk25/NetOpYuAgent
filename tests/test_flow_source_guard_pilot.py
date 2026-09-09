"""Frozen probe accounting and no-retry behavior; model transport is simulated."""

import json

import pytest

from evaluation import flow_source_guard_pilot as pilot
from evaluation.flow_translation import _write
from tests.test_flow_source_guards import fixture
from tests.test_flow_source_duties import review_fixture


def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "implementation", lambda: {"test": "fixed"})
    monkeypatch.setattr(pilot.previous, "environment", lambda: {"test": "fixed"})
    monkeypatch.setattr(pilot.OllamaAnchoredAuthorAdapter, "preflight", lambda self: {"model": "qwen3.5:9b", "modelArtifactDigest": "fixture"})
    source, candidate = fixture()
    inputs, root = tmp_path / "cases.json", tmp_path / "batch"
    _write(inputs, [dict(id="first", cohort="regression", bundle=source.model_dump(mode="json"))])
    pilot.freeze(inputs, root)
    return root, source, candidate


def completed(root, source, candidate, model="qwen3.5:9b", content=None, done="stop"):
    m = pilot.load(root)
    reply = dict(model=model, done=True, done_reason=done, prompt_eval_count=9, eval_count=8,
        message=dict(content=content or candidate.model_dump_json()))
    envelope = dict(httpStatus=200, body=json.dumps(reply), latencyMs=15.0)
    folder = root / "first"
    folder.mkdir()
    _write(folder / "request.json", dict(wireRequest=m["cases"][0]["request"], model=m["model"]))
    _write(folder / "response.json", envelope)
    files, status = pilot.derive(source, envelope)
    for name, value in files.items():
        _write(folder / name, value)
    _write(folder / "status.json", status)
    _write(folder / "receipt.json", pilot.receipt(folder))
    return files, status


def test_reentry_is_read_only_and_review_not_implicit(tmp_path, monkeypatch):
    root, source, candidate = setup(tmp_path, monkeypatch)
    assert not pilot.report(root)["completed"]
    files, _ = completed(root, source, candidate)
    monkeypatch.setattr(pilot.OllamaAnchoredAuthorAdapter, "preflight", lambda self: pytest.fail("no model call"))
    pilot.run(root, 0)
    plain = pilot.report(root)
    assert plain["cohorts"]["regression"]["sourceReviewed"] == 0
    reviews = tmp_path / "reviews"
    reviews.mkdir()
    _write(reviews / "first.json", review_fixture(files["review-input.json"]).model_dump(mode="json"))
    reviewed = pilot.report(root, reviews)
    assert reviewed["cohorts"]["regression"]["reviewSupported"] == 1
    assert reviewed["wholeSkillTranslations"] == 0 and reviewed["semanticAccuracy"] is None
    assert plain == pilot.report(root)


@pytest.mark.parametrize("content,done", [("not JSON", "stop"), ('{"rows":{}}', "stop"), ("{}", "length")])
def test_bad_response_stays_original_and_cannot_be_repaired(tmp_path, monkeypatch, content, done):
    root, source, candidate = setup(tmp_path, monkeypatch)
    files, status = completed(root, source, candidate, content=content, done=done)
    assert status["status"] == "blocked" and "compilation.json" not in files
    pilot.run(root, 0)
    assert pilot.report(root)["cohorts"]["regression"]["structurallyQualified"] == 0


def test_missing_budget_uncertain_checkpoint_and_tamper(tmp_path, monkeypatch):
    root, source, candidate = setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="budget"):
        pilot.run(root, 0)
    (root / "first").mkdir()
    with pytest.raises(ValueError, match="checkpoint"):
        pilot.run(root, 1)


@pytest.mark.parametrize("body", ["[]", "null", '{}', '{"model":"wrong","done":true,"done_reason":"stop"}',
    '{"model":"qwen3.5:9b","done":true,"done_reason":"stop","prompt_eval_count":true}'])
def test_envelope_errors_are_not_semantic_model_failures(body):
    source, _ = fixture()
    files, status = pilot.derive(source, dict(httpStatus=200, body=body, latencyMs=1))
    assert status["status"] == "blocked" and not status["envelopeValid"] and not files


def test_cohort_is_required_and_not_visible_to_model():
    source, _ = fixture()
    item = dict(id="trial", bundle=source.model_dump(mode="json"))
    with pytest.raises(ValueError, match="cohort"):
        pilot.cases_at([item])
    a = pilot.cases_at([{**item, "cohort": "regression"}])[0]
    b = pilot.cases_at([{**item, "cohort": "new_development"}])[0]
    assert a["request"] == b["request"] and "cohort" not in json.dumps(a["request"])
