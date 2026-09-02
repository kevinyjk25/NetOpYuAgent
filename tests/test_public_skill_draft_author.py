from __future__ import annotations

import json
from pathlib import Path

from evaluation.public_skill_draft_author import (
    DraftBundle,
    DraftTask,
    inspect_public_market_drafts,
    run_public_market_drafts,
)
from network_runtime.contracts import sha256_json


class FakeAdapter:
    def preflight(self) -> dict[str, str]:
        return {"model": "offline-draft", "modelArtifactDigest": sha256_json({"model": "offline-draft"})}

    def draft(self, prompt: str):  # type: ignore[no-untyped-def]
        payload = json.loads(prompt)
        tasks = []
        for slot in payload["taskSlots"]:
            challenge = slot["challenge"]
            if challenge == "nominal":
                disposition, effects = "proposal", 1
            elif challenge == "ambiguous_or_missing":
                disposition, effects = "safe_stop_clarify", 0
            else:
                disposition, effects = "safe_stop_reject", 0
            tasks.append(DraftTask(
                slot_id=slot["slotId"], challenge=challenge,
                user_prompt=f"Task for {challenge}", intended_outcome="Bounded draft outcome",
                expected_disposition=disposition, required_capabilities=("example.read",),
                forbidden_capabilities=("example.undeclared",), parameters={}, risk="medium",
                approval_required=effects > 0, max_effect_calls=effects,
                preconditions=("scope is explicit",), verification=("read result",),
                recovery=("stop safely",), assumptions=("human review required",),
            ))
        return DraftBundle(assignment_id=payload["assignmentId"], tasks=tuple(tasks)), {
            "modelCalls": 1, "inputTokens": 10, "outputTokens": 20,
            "latencyMs": 1.0, "rawDigest": sha256_json(payload), "error": None,
        }


def test_model_drafts_remain_non_gold_and_resume_bound(tmp_path: Path) -> None:
    kit = tmp_path / "kit"
    package_id = "safe-package"
    package = kit / "packages" / package_id
    package.mkdir(parents=True)
    package.joinpath("SKILL.md").write_text(
        "---\nname: safe-package\ndescription: A safe public package.\n---\nRead state.\n",
        encoding="utf-8",
    )
    assignment = {
        "apiVersion": "effect-runtime.io/public-skill-author-assignment/v1",
        "assignmentId": "wild-assignment-001", "packageId": package_id,
        "packageDigest": "sha256:" + "a" * 64, "skillName": "safe-package",
        "repository": "owner/repo", "commitSha": "b" * 40, "sourcePath": "skills/safe",
        "packageEntry": f"packages/{package_id}/SKILL.md",
        "taskSlots": [
            {"slotId": "wild-assignment-001-01", "challenge": "nominal"},
            {"slotId": "wild-assignment-001-02", "challenge": "ambiguous_or_missing"},
            {"slotId": "wild-assignment-001-03", "challenge": "failure_or_adversarial"},
        ],
    }
    kit.joinpath("assignments.jsonl").write_text(json.dumps(assignment) + "\n", encoding="utf-8")
    kit.joinpath("schemas.json").write_text("{}\n", encoding="utf-8")
    kit.joinpath("README.md").write_text("static-only\n", encoding="utf-8")
    sealed = {
        path.relative_to(kit).as_posix(): "sha256:" + __import__("hashlib").sha256(path.read_bytes()).hexdigest()
        for path in sorted(item for item in kit.rglob("*") if item.is_file())
    }
    body = {
        "apiVersion": "effect-runtime.io/public-skill-independent-author-kit/v1",
        "createdAt": "2026-09-01T00:00:00+00:00",
        "evidenceClass": "public_market_independent_annotation_workspace",
        "executionPolicy": "static_only", "officialEsP1QualificationEligible": False,
        "privateHoldout": False, "sourceSnapshotManifestDigest": "sha256:" + "c" * 64,
        "sourceSnapshotInspection": {"acceptedCount": 1, "runtimePackageGates": {"passed": 1}},
        "selectedPackageCount": 1, "tasksPerSkill": 3, "taskSlotCount": 3,
        "selectedPackageDigests": {package_id: "sha256:" + "a" * 64},
        "sealedFiles": sealed, "containsRuntimeOrEvaluator": False,
        "containsGeneratedGold": False, "thirdPartyExecutionAttempted": False,
        "claimBoundary": "annotation only",
    }
    kit.joinpath("workspace.json").write_text(
        json.dumps({**body, "workspaceDigest": sha256_json(body)}), encoding="utf-8",
    )
    output = tmp_path / "drafts"
    report = run_public_market_drafts(kit, output, model="offline-draft", adapter=FakeAdapter())
    assert report["draftedTaskCount"] == 3
    assert report["containsTrustedGold"] is False
    assert report["humanReviewRequired"] is True
    assert report["officialEsP1QualificationEligible"] is False
    resumed = run_public_market_drafts(kit, output, model="offline-draft", adapter=FakeAdapter())
    assert resumed["draftsDigest"] == report["draftsDigest"]
    assert resumed["reportDigest"] == report["reportDigest"]
    inspected = inspect_public_market_drafts(output, kit)
    assert inspected["verified"] is True
    assert inspected["authority"] == "draft_only_human_review_required"


def test_model_draft_inspection_rejects_tampering(tmp_path: Path) -> None:
    test_model_drafts_remain_non_gold_and_resume_bound(tmp_path)
    report_path = tmp_path / "drafts" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["containsTrustedGold"] = True
    report_path.write_text(json.dumps(report), encoding="utf-8")
    try:
        inspect_public_market_drafts(tmp_path / "drafts", tmp_path / "kit")
    except ValueError as exc:
        assert "report digest mismatch" in str(exc)
    else:
        raise AssertionError("tampered model draft report was accepted")
