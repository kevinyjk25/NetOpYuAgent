"""Batch integrity checks; fake responses are not translation evidence."""

import json

import pytest

from evaluation import flow_translation as translation
from evaluation import flow_translation_batch as batch


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(batch.OllamaAnchoredAuthorAdapter, "preflight", lambda self: {"model": "fake-test-only"})
    root = tmp_path / "batch"
    batch.freeze(root)
    return root


def fake_author(sources, folder, *, expected_model):
    folder.mkdir()
    batch._write(folder / "sources.json", sources.model_dump(mode="json"))
    batch._write(folder / "request.json", {"model": expected_model, "wireRequest": translation.author_request(sources)})
    batch._write(folder / "status.json", {"status": "blocked", "errorType": "TestFixture"})


def test_eight_distinct_known_flows_share_one_auditable_host_tool():
    rows = batch.development_sources()
    assert len(rows) == len({case["feature"] for case, _ in rows}) == 8
    for case, source in rows:
        assert set(source.reads) == {"read_inventory_device"}
        assert not source.effects
        assert "not live device health" in source.source_text
        wire = translation.author_request(source)
        assert case["id"] not in wire["messages"][1]["content"]
        assert "source_path" not in wire["messages"][1]["content"]
        assert wire["model"] == "qwen3.5:9b" and wire["think"] is False


def test_request_refactor_preserves_c3a_saved_wire():
    # No dependency on ignored artifacts; metadata changes cannot affect the prompt.
    sources = translation.local_sources()
    assert translation.author_request(sources) == translation.author_request(sources.model_copy(update={"source_path": "hidden-evaluator-label"}))


def test_freeze_never_overwrites(frozen):
    before = (frozen / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        batch.freeze(frozen)
    assert (frozen / "manifest.json").read_bytes() == before


def test_manifest_mutation_rejected(frozen):
    raw = json.loads((frozen / "manifest.json").read_text())
    raw["attemptsPerCase"] = 2
    (frozen / "manifest.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="digest"):
        batch.load_manifest(frozen)


def test_implementation_drift_blocks_resume_before_call(frozen, monkeypatch):
    monkeypatch.setattr(batch, "fingerprint", lambda: {"changed": "yes"})
    monkeypatch.setattr(batch, "author", lambda *a, **kw: pytest.fail("must not call model"))
    with pytest.raises(ValueError, match="implementation changed"):
        batch.run(frozen)


def test_successful_checkpoint_resume_never_repeats_even_blocked_calls(frozen, monkeypatch):
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[1].name)
        return fake_author(*args, **kwargs)

    monkeypatch.setattr(batch, "author", counted)
    first = batch.run(frozen)
    assert len(calls) == 8
    assert batch.run(frozen) == first and len(calls) == 8
    assert first["semanticAccepted"] is None and first["runtimeExecutions"] == 0


def test_ambiguous_interruption_does_not_retry(frozen, monkeypatch):
    (frozen / "direct-read").mkdir()
    monkeypatch.setattr(batch, "author", lambda *a, **kw: pytest.fail("no retry"))
    assert batch.report(frozen)["rows"][0]["status"] == "interrupted_or_active_no_retry"
    with pytest.raises(ValueError, match="needs inspection"):
        batch.run(frozen)


@pytest.mark.parametrize("file", ["sources.json", "request.json", "status.json"])
def test_completed_checkpoint_tampering_detected(frozen, monkeypatch, file):
    monkeypatch.setattr(batch, "author", fake_author)
    batch.run(frozen)
    (frozen / "direct-read" / file).write_text("{}")
    with pytest.raises(ValueError, match="checkpoint changed"):
        batch.report(frozen)


@pytest.mark.parametrize("failure", ["preflight", "model_drift"])
def test_author_preflight_failure_records_block_without_post(tmp_path, monkeypatch, failure):
    def preflight(self):
        if failure == "preflight":
            raise ValueError("preflight failed")
        return {"model": "other"}

    monkeypatch.setattr(translation.OllamaAnchoredAuthorAdapter, "preflight", preflight)
    monkeypatch.setattr(translation.httpx, "Client", lambda *a, **kw: pytest.fail("must not post"))
    root = tmp_path / "attempt"
    result = translation.author(translation.local_sources(), root, expected_model={"model": "expected"})
    assert result["status"] == "blocked"
    assert json.loads((root / "status.json").read_text()) == result
    assert not (root / "response.json").exists()


def test_no_current_fixture_resampling_on_resume(frozen, monkeypatch):
    monkeypatch.setattr(batch, "development_sources", lambda: pytest.fail("use frozen sources, not current fixtures"))
    monkeypatch.setattr(batch, "author", fake_author)
    assert len(batch.run(frozen)["rows"]) == 8
