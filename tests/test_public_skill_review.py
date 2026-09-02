from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.public_skill_draft_author import run_public_market_drafts
from evaluation.public_skill_review import (
    export_assisted_review_kit,
    export_blind_gold_kit,
    inspect_assisted_review_kit,
    inspect_blind_gold_kit,
)
from tests.test_public_skill_library import _DraftAdapter, _author_kit


def _workspace(tmp_path: Path) -> Path:
    kit = _author_kit(tmp_path / "author-kit")
    drafts = tmp_path / "drafts"
    run_public_market_drafts(kit, drafts, model="offline", adapter=_DraftAdapter())
    review = tmp_path / "review"
    export_assisted_review_kit(kit, drafts, review)
    return review


def _complete_case_review(review: Path) -> None:
    path = review / "reviews/wild-assignment-001.review.json"
    catalog = review / "materials/catalogs/example.json"
    catalog.write_text(json.dumps({
        "apiVersion": "effect-runtime.io/public-skill-tool-catalog/v1",
        "assignmentId": "wild-assignment-001", "capabilities": [],
    }), encoding="utf-8")
    value = json.loads(path.read_text(encoding="utf-8"))
    value["reviewer"].update({
        "authorId": "case-author-01", "independentFromRuntimeTeam": True,
    })
    decisions = ("accept_prompt", "edit_prompt", "reject_slot")
    for slot, decision in zip(value["slots"], decisions, strict=True):
        slot["decision"] = decision
        slot["rationale"] = f"Independent decision: {decision}"
        if decision == "reject_slot":
            slot["task"] = None
            continue
        slot["task"].update({
            "toolCatalogRef": "materials/catalogs/example.json", "authorId": "case-author-01",
            "authoredAt": "2026-09-01T00:00:00Z",
        })
        if decision == "edit_prompt":
            slot["task"]["userPrompt"] += " with independently clarified scope"
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def test_assisted_review_kit_withholds_model_semantics_and_tracks_decisions(tmp_path: Path) -> None:
    review = _workspace(tmp_path)
    initial = inspect_assisted_review_kit(review)
    assert initial["decisionCounts"]["pending"] == 3
    assert initial["reviewComplete"] is False
    assert initial["goldAuthorKitExportEligible"] is False
    source = (review / "source/prompt-candidates.jsonl").read_text(encoding="utf-8")
    assert "Bounded candidate" not in source
    assert '"containsGoldOrOracle": true' not in (review / "workspace.json").read_text(encoding="utf-8")
    html = (review / "review-queue.html").read_text(encoding="utf-8")
    assert "Review nominal" not in html

    _complete_case_review(review)
    completed = inspect_assisted_review_kit(review)
    assert completed["reviewComplete"] is True
    assert completed["goldAuthorKitExportEligible"] is True
    assert completed["materialFileCount"] == 1
    assert completed["decisionCounts"] == {
        "accept_prompt": 1, "edit_prompt": 1, "author_from_scratch": 0,
        "pending": 0, "reject_slot": 1,
    }


def test_assisted_review_kit_rejects_gold_injection_and_source_tampering(tmp_path: Path) -> None:
    review = _workspace(tmp_path)
    path = review / "reviews/wild-assignment-001.review.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["gold"] = {"expectedDisposition": "proposal"}
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden Gold/Oracle"):
        inspect_assisted_review_kit(review)

    review = _workspace(tmp_path / "second")
    candidates = review / "source/prompt-candidates.jsonl"
    candidates.write_text(candidates.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="sealed source digest"):
        inspect_assisted_review_kit(review)


def test_completed_review_requires_existing_safe_materials(tmp_path: Path) -> None:
    review = _workspace(tmp_path)
    path = review / "reviews/wild-assignment-001.review.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["reviewer"].update({"authorId": "case-author-01", "independentFromRuntimeTeam": True})
    for slot in value["slots"]:
        slot["decision"] = "accept_prompt"
        slot["rationale"] = "Reviewed independently"
        slot["task"].update({
            "toolCatalogRef": "materials/catalogs/missing.json",
            "authorId": "case-author-01", "authoredAt": "2026-09-01T00:00:00Z",
        })
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="missing or unsafe"):
        inspect_assisted_review_kit(review)


def test_blind_gold_kit_requires_human_gate_and_withholds_model_semantics(tmp_path: Path) -> None:
    review = _workspace(tmp_path)
    with pytest.raises(ValueError, match="complete independent Case Author review"):
        export_blind_gold_kit(review, tmp_path / "gold-too-early")

    _complete_case_review(review)
    gold_root = tmp_path / "gold"
    manifest = export_blind_gold_kit(review, gold_root)
    inspected = inspect_blind_gold_kit(gold_root)
    assert manifest["containsModelSemanticCandidates"] is False
    assert inspected["decisionCounts"] == {"author_gold": 0, "pending": 2, "reject_task": 0}
    assert inspected["goldAuthoringComplete"] is False
    assert inspected["officialEsP1QualificationEligible"] is False
    source = (gold_root / "source/tasks.jsonl").read_text(encoding="utf-8")
    assert "Bounded candidate" not in source

    for path in sorted((gold_root / "gold").glob("*.gold.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        value.update({"decision": "author_gold", "rationale": "Independently authored Gold"})
        value["goldAuthor"].update({
            "authorId": "gold-author-01", "independentFromCaseAuthor": True,
            "independentFromRuntimeTeam": True, "modelSemanticCandidatesSeen": False,
        })
        value["gold"].update({
            "expectedDisposition": "read_only", "intendedOutcome": "Return verified bounded data",
            "risk": "low", "approvalRequired": False, "maxEffectCalls": 0,
            "authorId": "gold-author-01",
        })
        value["oracle"]["assertions"] = [{
            "path": "outcome", "operator": "equals", "expected": "verified",
        }]
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    complete = inspect_blind_gold_kit(gold_root)
    assert complete["goldAuthoringComplete"] is True
    assert complete["pairedEvaluationAuthoringEligible"] is True
    assert complete["officialEsP1QualificationEligible"] is False


def test_blind_gold_kit_rejects_unexpected_files_and_nonblind_authors(tmp_path: Path) -> None:
    review = _workspace(tmp_path)
    _complete_case_review(review)
    gold_root = tmp_path / "gold"
    export_blind_gold_kit(review, gold_root)

    unexpected = gold_root / "notes.txt"
    unexpected.write_text("unsealed", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected files"):
        inspect_blind_gold_kit(gold_root)
    unexpected.unlink()

    path = next((gold_root / "gold").glob("*.gold.json"))
    value = json.loads(path.read_text(encoding="utf-8"))
    value.update({"decision": "author_gold", "rationale": "Not actually blind"})
    value["goldAuthor"].update({
        "authorId": "gold-author-01", "independentFromCaseAuthor": True,
        "independentFromRuntimeTeam": True, "modelSemanticCandidatesSeen": True,
    })
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="independence disclosure"):
        inspect_blind_gold_kit(gold_root)
