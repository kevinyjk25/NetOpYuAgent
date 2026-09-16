import copy

import pytest

from evaluation.hybrid_historical_candidate import prepare
from evaluation.structured_authoring import seal
from evaluation.structured_binding_probe import write_artifacts
from network_runtime.contracts import sha256_json
from tests.test_hybrid_draft_review import inputs


def fixture(root, mutation=None):
    original = inputs()
    candidate = {**copy.deepcopy(original["candidate"]), "draft": "A historical edited answer; still unverified."}
    app = {"candidateDigest": sha256_json(candidate), "previousCandidateDigest": sha256_json(original["candidate"]),
        "edits": [{"start": 0, "end": len(original["candidate"]["draft"]), "text": original["candidate"]["draft"],
                   "replacement": candidate["draft"], "action": "replace"}]}
    graph = {"graphDigest": sha256_json("historical graph fixture"), "proposal": {"source_digest": sha256_json("source fixture")}}
    frozen = {"sourceInputs": original, "graph": graph}
    if mutation == "edit_location":
        app["edits"][0]["end"] -= 1
    elif mutation == "original_source":
        original["candidate"]["draft"] = "Mutated old candidate."
    elif mutation == "modified_values":
        candidate["values"] = {"invented": True}
        app["candidateDigest"] = sha256_json(candidate)
    elif mutation == "wrong_rendering":
        candidate["draft"] += " Injected extra assertion."
        app["candidateDigest"] = sha256_json(candidate)
    frozen, app = seal(frozen), seal(app)
    summary = seal({"freezeDigest": frozen["reportDigest"], "materializedCandidateDigest": sha256_json(candidate),
        "execution": {"status": "governed_graph_completed", "graphDigest": graph["graphDigest"]}})
    write_artifacts(root / "freeze", {"inputs.json": frozen})
    write_artifacts(root / "summary", {"report.json": summary})
    write_artifacts(root / "materialized", {"candidate.json": candidate, "application.json": app})
    return original, candidate


def test_historical_candidate_is_checked_data_without_old_opinions_or_action_permission(tmp_path):
    original, candidate = fixture(tmp_path / "historical")
    _, receipt, supplied, _ = prepare(tmp_path / "historical")
    assert supplied == {**original, "candidate": candidate}
    assert receipt["kind"] == "historical_repair_candidate_data_import"
    assert not receipt["oldModelJudgmentsImported"] and not receipt["newActionAuthority"] and not receipt["semanticApproval"]
    assert receipt["newUnseenSources"] == receipt["newBusinessReadCalls"] == 0
    assert "review" not in supplied


@pytest.mark.parametrize("mutation", ["edit_location", "original_source", "modified_values", "wrong_rendering"])
def test_resigned_inconsistent_historical_materialization_is_not_a_hash_bypass(tmp_path, mutation):
    fixture(tmp_path / "historical", mutation)
    with pytest.raises(ValueError, match="historical"):
        prepare(tmp_path / "historical")
