import json

import httpx
import pytest

from evaluation import flow_tree_canary as canary


def test_canaries_are_explicit_answer_probes_not_translation():
    rows = canary.cases()
    assert len(rows) == 12
    assert len({row["probe"] for row in rows}) == 4
    for row in rows:
        payload = json.loads(row["wireRequest"]["messages"][1]["content"])
        assert payload["expectedObject"] == row["expected"]
        assert ("outputSchema" in payload) == (row["arm"] != "format-only")
        assert (row["wireRequest"]["format"] == "json") == (row["arm"] == "json-visible")


@pytest.mark.parametrize("index", range(12))
def test_expected_canary_is_compilable_and_exact(index):
    row = canary.cases()[index]
    envelope = {"httpStatus": 200, "body": json.dumps({"message": {"content": json.dumps(row["expected"])}}), "latencyMs": 0}
    result = canary.evaluate(row, envelope)
    assert result["exactCopy"] and result["parseAndCompile"]


def test_parseable_wrong_output_not_counted_as_exact():
    rows = canary.cases()
    envelope = {"httpStatus": 200, "body": json.dumps({"message": {"content": json.dumps(rows[3]["expected"])}}), "latencyMs": 0}
    result = canary.evaluate(rows[0], envelope)
    assert result["parseAndCompile"] and not result["exactCopy"]


def test_truncation_and_http_failure_are_preserved_as_failure():
    row = canary.cases()[0]
    assert not canary.evaluate(row, {"httpStatus": 500})["exactCopy"]
    assert not canary.evaluate(row, {"httpStatus": 200, "body": '{"message":'})["exactCopy"]


def test_frozen_run_never_repeats_and_replays(tmp_path, monkeypatch):
    rows = canary.cases()[:1]
    monkeypatch.setattr(canary, "cases", lambda: rows)
    model = {"model": "qwen3.5:9b", "modelArtifactDigest": "test"}
    monkeypatch.setattr(canary.OllamaAnchoredAuthorAdapter, "preflight", lambda self: model)
    calls = []

    def post(self, url, **kwargs):
        calls.append(kwargs)
        return httpx.Response(200, request=httpx.Request("POST", url), json={"message": {"content": json.dumps(rows[0]["expected"])}})
    monkeypatch.setattr(httpx.Client, "post", post)
    root = tmp_path / "canary"
    canary.freeze(root)
    canary.run(root)
    canary.run(root)
    assert len(calls) == 1
    assert canary.report(root)["uniqueSkills"] == 0
    (root / rows[0]["id"] / "result.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        canary.run(root)
