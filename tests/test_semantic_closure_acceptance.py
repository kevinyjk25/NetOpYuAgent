import copy

import pytest

from evaluation.semantic_closure_acceptance import check_judgment, summarize
from evaluation.flow_tree_authoring import digest_file
from evaluation.structured_authoring import seal
from evaluation.structured_binding_probe import write_artifacts


def rows():
    return [{"case": f"case-{i}", "skill": f"skill-{i % 6}", "repository": f"repo-{i % 4}",
             "domain": f"domain-{i % 3}", "expectations": [{"id": "safety", "critical": True}],
             "judgment": {"substantive": True, "outcome": "scoped_task_fulfilled", "criteria": [{"id": "safety", "result": "met"}],
                          "unsafeCallObserved": False, "falseCompletionObserved": False, "criticalTaskMismatch": False}}
            for i in range(12)]


@pytest.mark.parametrize("failure", ["none", "skills", "critical_unknown", "false_claim", "substantive"])
def test_arithmetic_does_not_equate_safe_stops_cases_or_unknowns_with_completion(failure):
    source = rows()
    if failure == "skills":
        for row in source:
            row["skill"] = "same-skill"
    elif failure == "critical_unknown":
        source[0]["judgment"]["criteria"][0]["result"] = "unknown"
    elif failure == "false_claim":
        source[0]["judgment"]["falseCompletionObserved"] = True
    elif failure == "substantive":
        for row in source:
            row["judgment"]["outcome"] = "correct_boundary"
    result = summarize(source, {"skills": 6, "repositories": 4, "domains": 3, "tasks": 12})
    assert result["prototypeExitMetByDeveloperJudgment"] is (failure == "none")
    assert result["productionSuccessProbability"] is None and not result["independentGold"]


def test_manual_judgment_binds_criteria_input_result_and_real_evidence(tmp_path):
    folder = tmp_path / "case-one"
    write_artifacts(folder / "intake", {"receipt.json": {"source": "frozen"}})
    write_artifacts(folder / "summary", {"report.json": {"status": "actual retained attempt"}})
    artifacts = {name: digest_file(folder / name) for name in ("intake/receipt.json", "summary/report.json")}
    row = copy.deepcopy(rows()[0]["judgment"])
    row.update(case=folder.name, reviewKind="developer_ai_not_independent_gold", reviewedArtifacts=artifacts)
    row["criteria"][0].update(explanation="Explicit synthetic fixture review.", evidence=["summary/report.json"])
    # Normal test-fixture serialization, not a real external judgment.
    import json
    (folder / "judgment.json").write_text(json.dumps(seal(row)))
    criteria = [{"id": "safety", "critical": True}]
    assert check_judgment(folder, criteria)["case"] == folder.name
    with pytest.raises(ValueError, match="every predeclared"):
        check_judgment(folder, [*criteria, {"id": "omitted", "critical": True}])
    (folder / "summary/report.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        check_judgment(folder, criteria)
def test_known_development_cannot_be_counted_as_unseen_exit(tmp_path, monkeypatch):
    from evaluation import semantic_closure_acceptance as acceptance
    monkeypatch.setattr(acceptance, "verify", lambda _: {"sourceSelectionHasOccurred": True})
    import pytest
    with pytest.raises(ValueError, match="known development"):
        acceptance.collect(tmp_path)
