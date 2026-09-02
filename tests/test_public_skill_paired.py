from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.public_skill_paired import (
    export_public_paired_study_kit,
    inspect_public_paired_study_kit,
)
from evaluation.public_skill_review import export_blind_gold_kit
from tests.test_public_skill_library import _DraftAdapter, _author_kit
from evaluation.public_skill_draft_author import run_public_market_drafts
from evaluation.public_skill_review import export_assisted_review_kit


def _case_review(tmp_path: Path) -> tuple[Path, Path]:
    author = _author_kit(tmp_path / "author")
    drafts = tmp_path / "drafts"
    run_public_market_drafts(author, drafts, model="offline", adapter=_DraftAdapter())
    review = tmp_path / "review"
    export_assisted_review_kit(author, drafts, review)
    catalog = review / "materials/catalogs/example.json"
    catalog.write_text(json.dumps({
        "apiVersion": "effect-runtime.io/public-skill-tool-catalog/v2",
        "assignmentId": "wild-assignment-001", "capabilities": [{
            "capabilityId": "directory.user.read", "toolName": "get_user",
            "description": "Read one deterministic user record", "actionType": "read_only",
            "inputSchema": {
                "type": "object", "additionalProperties": False,
                "required": ["user_id"], "properties": {"user_id": {"type": "string"}},
            },
            "operation": {"kind": "read_record", "collection": "users", "keyArgument": "user_id"},
        }],
    }), encoding="utf-8")
    path = review / "reviews/wild-assignment-001.review.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["reviewer"].update({"authorId": "case-author-01", "independentFromRuntimeTeam": True})
    for index, slot in enumerate(value["slots"]):
        if index == 2:
            slot.update({"decision": "reject_slot", "rationale": "Not reproducible", "task": None})
            continue
        slot.update({"decision": "accept_prompt", "rationale": "Independently reviewed"})
        slot["task"].update({
            "toolCatalogRef": "materials/catalogs/example.json", "authorId": "case-author-01",
            "authoredAt": "2026-09-01T00:00:00Z",
            "fixtureRefs": [f"materials/fixtures/{slot['slotId']}.json"],
        })
        fixture = review / slot["task"]["fixtureRefs"][0]
        fixture.write_text(json.dumps({
            "apiVersion": "effect-runtime.io/public-skill-fixture-state/v1",
            "caseId": slot["slotId"], "approval": {"approved": True, "actor": "reviewer-01"},
            "fault": "none", "collections": {
                "users": {"alice": {"role": "viewer", "revision": 1}},
            },
            "staticResults": {}, "verificationMismatchPatch": {},
        }), encoding="utf-8")
    path.write_text(json.dumps(value), encoding="utf-8")
    return author, review


def _gold(tmp_path: Path) -> tuple[Path, Path]:
    author, review = _case_review(tmp_path)
    gold = tmp_path / "gold"
    export_blind_gold_kit(review, gold)
    for path in sorted((gold / "gold").glob("*.gold.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        value.update({"decision": "author_gold", "rationale": "Independent expected result"})
        value["goldAuthor"].update({
            "authorId": "gold-author-01", "independentFromCaseAuthor": True,
            "independentFromRuntimeTeam": True, "modelSemanticCandidatesSeen": False,
        })
        value["gold"].update({
            "expectedDisposition": "read_only", "intendedOutcome": "PRIVATE_SCORING_ONLY",
            "requiredCapabilities": ["directory.user.read"],
            "risk": "low", "approvalRequired": False, "maxEffectCalls": 0,
            "authorId": "gold-author-01",
        })
        value["oracle"]["assertions"] = [{
            "path": "outcome", "operator": "equals", "expected": "verified",
        }]
        path.write_text(json.dumps(value), encoding="utf-8")
    return author, gold


def test_public_paired_study_kit_separates_agent_and_gold(tmp_path: Path) -> None:
    author, gold = _gold(tmp_path)
    output = tmp_path / "paired"
    manifest = export_public_paired_study_kit(gold, author, output)
    inspected = inspect_public_paired_study_kit(output)
    assert manifest["caseCount"] == 2
    assert inspected["model"] == "qwen3.5:9b"
    assert inspected["agentGoldIsolation"] is True
    assert inspected["pairedExecutionCompleted"] is False
    assert inspected["fixtureMcpExecutableCaseCount"] == 2
    assert inspected["fixtureMcpInputEligible"] is True
    assert inspected["translationReportAttached"] is False
    assert inspected["pairedExecutionInputEligible"] is False
    assert inspected["officialEsP1QualificationEligible"] is False
    assert "PRIVATE_SCORING_ONLY" not in (output / "agent/cases.jsonl").read_text(encoding="utf-8")
    assert "PRIVATE_SCORING_ONLY" in (output / "scoring/gold.jsonl").read_text(encoding="utf-8")
    assert json.loads((output / "study-plan.json").read_text(encoding="utf-8"))["repetitions"] == 3


def test_public_paired_study_kit_requires_complete_gold(tmp_path: Path) -> None:
    author, review = _case_review(tmp_path)
    gold = tmp_path / "gold"
    export_blind_gold_kit(review, gold)
    with pytest.raises(ValueError, match="complete independently authored Gold"):
        export_public_paired_study_kit(gold, author, tmp_path / "paired")


def test_public_paired_study_kit_detects_agent_or_authority_tampering(tmp_path: Path) -> None:
    author, gold = _gold(tmp_path)
    output = tmp_path / "paired"
    export_public_paired_study_kit(gold, author, output)
    cases = output / "agent/cases.jsonl"
    cases.write_text(cases.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="sealed file set or digest drift"):
        inspect_public_paired_study_kit(output)

    output = tmp_path / "paired-second"
    export_public_paired_study_kit(gold, author, output)
    workspace = output / "workspace.json"
    value = json.loads(workspace.read_text(encoding="utf-8"))
    value["officialEsP1QualificationEligible"] = True
    from network_runtime.contracts import sha256_json
    value["workspaceDigest"] = sha256_json({
        key: item for key, item in value.items() if key != "workspaceDigest"
    })
    workspace.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="authority boundary"):
        inspect_public_paired_study_kit(output)
